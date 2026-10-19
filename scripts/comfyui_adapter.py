"""ComfyUI HTTP adapter: upload a source, submit an API workflow, fetch output.

No model dependencies are bundled. A running ComfyUI installation and its
API-format workflow are supplied by the user, separately from the workbench.
"""
from __future__ import annotations

import hashlib
import json
import mimetypes
import secrets
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any
import workflow_timing


def _validate_workflow(document: Any) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise ValueError("ComfyUI workflow must be an API-format JSON object")
    graph = document.get("workflow", document)
    if not isinstance(graph, dict) or not graph or any(
        not isinstance(node, dict) or "class_type" not in node or not isinstance(node.get("inputs"), dict)
        for node in graph.values()
    ):
        raise ValueError("ComfyUI requires an API-format node map; export 'Save (API Format)', not the UI graph")
    return document


def load_workflow(path: Path) -> dict[str, Any]:
    return _validate_workflow(json.loads(path.read_text(encoding="utf-8-sig")))


def _load_workflow_with_digest(path: Path) -> tuple[dict[str, Any], str]:
    """Parse and identify one immutable workflow read.

    Generation used to parse the JSON and then read the same file again only
    to compute its provenance digest. Keep the bytes used for parsing as the
    digest input so a generation does one filesystem read and one hash pass.
    """
    data = path.read_bytes()
    return _validate_workflow(json.loads(data.decode("utf-8-sig"))), hashlib.sha256(data).hexdigest()


def prepare_workflow(document: dict[str, Any], source: str, prompt: str) -> dict[str, Any]:
    """Bind both actual source and prompt; explicit bindings support plain exports."""
    consumed: set[str] = set()

    def replace(value: Any) -> Any:
        if isinstance(value, str):
            for key, replacement in (("source", source), ("prompt", prompt)):
                if "{{" + key + "}}" in value:
                    consumed.add(key)
                    value = value.replace("{{" + key + "}}", replacement)
            return value
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        return value

    graph = replace(document.get("workflow", document))
    for key, binding in document.get("bindings", {}).items():
        if key not in {"source", "prompt"}:
            raise ValueError(f"unsupported ComfyUI binding: {key}")
        node_id, input_name = str(binding["node"]), str(binding["input"])
        graph[node_id]["inputs"][input_name] = source if key == "source" else prompt
        consumed.add(key)
    if consumed != {"source", "prompt"}:
        raise ValueError("ComfyUI workflow must bind both {{source}} and {{prompt}} or declare source/prompt bindings")
    for encoder_id, encoder in graph.items():
        if encoder['class_type'] != 'CosplayTextEncodeQwenImage21':
            continue
        samplers = [node for node in graph.values() if node['inputs'].get('positive') == [encoder_id, 0]]
        if not samplers or any(node['class_type'] != 'KSampler'
            or node['inputs'].get('negative') != [encoder_id, 1]
            or node['inputs'].get('cfg') != [encoder_id, 3] for node in samplers):
            raise ValueError('Qwen conditioning CFG must feed the same standard KSampler guidance')
        for sampler in samplers:
            model_binding = sampler['inputs'].get('model', [])
            model = (graph.get(str(model_binding[0]), {}) if isinstance(model_binding, list)
                     and len(model_binding) == 2 and model_binding[1] == 0 else {})
            loader_binding = model.get('inputs', {}).get('model', [])
            loader = (graph.get(str(loader_binding[0]), {}) if isinstance(loader_binding, list)
                      and len(loader_binding) == 2 and loader_binding[1] == 0 else {})
            if (model.get('class_type') != 'QwenImage21Cache'
                    or loader.get('class_type') not in {'UNETLoader', 'UnetLoaderGGUF'}):
                raise ValueError('Qwen CFG1 encoding requires the current model path without custom CFG hooks')
    return graph


def prepare_condition_workflow(document: dict[str, Any], names: list[str], prompt: str,
                               *, coordinate_space: str | None = None) -> dict[str, Any]:
    """Bind ordered Qwen 2.1 condition images; no hard-mask claim."""
    if not 1 <= len(names) <= 10:
        raise ValueError('Qwen 2.1 conditions must contain one to ten images')
    graph=prepare_workflow(document,names[0],prompt)
    if coordinate_space == 'source_canvas_roi':
        # Explicit native crops own an exact canvas. The global 2K target
        # must not rescale a local edit whose coordinates are used to paste it.
        for row in graph.values():
            if row['class_type'] in {'TextEncodeQwenImage21', 'CosplayTextEncodeQwenImage21'}:
                row['inputs']['resolution'] = 0
    if len(names)==1:
        return graph
    encoders=[row for row in graph.values() if row['class_type'] in
              {'TextEncodeQwenImage21', 'CosplayTextEncodeQwenImage21'}]
    if len(encoders)!=1:
        raise ValueError('Multiple conditions require exactly one TextEncodeQwenImage21 node')
    inputs=encoders[0]['inputs']
    for key in list(inputs):
        if key.startswith('images.image_'):
            inputs.pop(key)
    next_id=max((int(k) for k in graph if str(k).isdigit()),default=0)+1
    for index,name in enumerate(names,1):
        node=str(next_id);next_id+=1
        graph[node]={'class_type':'LoadImage','inputs':{'image':name}}
        inputs[f'images.image_{index}']=[node,0]
    # Recompute the prefix instead of retaining all references' per-block K/V.
    # The 16 GB desktop's CPU cache exhausted Windows commit during a real
    # three-reference request. "off" keeps the lossless computation semantics.
    for row in graph.values():
        if row['class_type'] == 'QwenImage21Cache':
            row['inputs']['device'] = 'off'
    return graph


def request_generation(endpoint: str, image: Path, workflow_path: Path, prompt: str,
                       timeout: float, events: list[dict[str, Any]],
                       key: str = "", poll_interval: float = 0.5,
                       condition_inputs: list[dict[str, Any]] | None = None) -> tuple[dict[str, Any], bytes, str]:
    document, workflow_digest = _load_workflow_with_digest(workflow_path)
    # Validate template before performing an upload.
    prepare_workflow(document, "pending-source.png", prompt)
    conditions=condition_inputs or [{'role':'source','path':str(image)}]
    if not 1 <= len(conditions) <= 10 or Path(conditions[0]['path']).resolve()!=image.resolve():
        raise ValueError('Condition images require one to ten images with --image first')
    # Do not hash every condition once for validation and then read it again
    # for multipart upload.  The upload pass below reads each immutable input
    # once, computes its digest from those bytes, and validates the declared
    # identity immediately before sending it.  This keeps the integrity check
    # while removing a full extra disk scan from every local generation.
    files=[]
    for row in conditions:
        path=Path(row['path']).resolve()
        stat=path.stat()
        files.append({**row,'path':str(path),'declared_sha256':row.get('sha256'),
                      'bytes':int(stat.st_size),'stat_key':(str(path), int(stat.st_size), int(stat.st_mtime_ns))})
    coordinate_space = conditions[0].get('coordinate_space')
    prepare_condition_workflow(document,['pending-'+str(i)+'.png' for i in range(len(files))],prompt,
                               coordinate_space=coordinate_space)
    started = time.perf_counter()
    deadline = time.monotonic() + timeout
    base = endpoint.rstrip("/")
    headers = {"User-Agent": "cosplay-comfyui-provider/1.0"}
    if key:
        headers["Authorization"] = "Bearer " + key
    # Local inference must not traverse a workstation's HTTP proxy.
    local = urllib.parse.urlsplit(base).hostname in {"127.0.0.1", "localhost", "::1"}
    open_request = (urllib.request.build_opener(urllib.request.ProxyHandler({})).open
                    if local else urllib.request.urlopen)
    timings: dict[str, Any] = {"uploads_seconds": []}
    http_submissions = 0

    def request(path: str, body: bytes | None = None, content_type: str | None = None) -> bytes:
        nonlocal http_submissions
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("ComfyUI generation timed out")
        request_headers = dict(headers)
        if content_type:
            request_headers["Content-Type"] = content_type
        req = urllib.request.Request(base + path, data=body, headers=request_headers)
        http_submissions += 1
        workflow_timing.annotate(http_request=True, http_submissions=http_submissions)
        if path == '/prompt':
            workflow_timing.annotate(model_request=True, model_submissions=None)
        with open_request(req, timeout=remaining) as response:
            return response.read()

    def json_request(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        result = json.loads(request(path, data, "application/json" if data is not None else None))
        if not isinstance(result, dict):
            raise RuntimeError(f"ComfyUI {path} returned a non-object response")
        return result

    uploads=[];names=[]
    upload_cache={}
    bytes_cache={}
    for row in files:
        path=Path(row['path'])
        data=bytes_cache.get(row['stat_key'])
        if data is None:
            data=path.read_bytes()
            bytes_cache[row['stat_key']]=data
        actual=hashlib.sha256(data).hexdigest()
        if row.get('declared_sha256') and row['declared_sha256']!=actual:
            raise ValueError('Condition image changed during upload preparation')
        row['sha256']=actual
        row['bytes']=len(data)
        public_row={key:value for key,value in row.items()
                    if key not in {'declared_sha256','stat_key'}}
        cached_upload=upload_cache.get(actual)
        if cached_upload is not None:
            uploaded=cached_upload
            uploads.append({**public_row,'index':len(uploads)+1,'upload':uploaded,'condition_only':True,
                            'hard_edit_mask_enforced':False,'upload_reused':True})
            names.append("/".join(part for part in (str(uploaded.get('subfolder') or ""),str(uploaded['name'])) if part))
            events.append({'event':'comfyui_upload_reused','index':len(uploads),'input_sha256':actual,
                           'bytes':len(data),'upload':uploaded,'seconds':0.0})
            continue
        boundary = "cosplay-" + secrets.token_hex(16)
        # ComfyUI compares bytes for duplicate names. A content name lets its
        # node cache retain identical conditioning, while changed bytes always
        # select a new LoadImage input. No client-side upload is skipped.
        upload_name = "cosplay-" + row['sha256'] + path.suffix.lower()
        upload_body = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{upload_name}"\r\n'
            f'Content-Type: {mimetypes.guess_type(path.name)[0] or "application/octet-stream"}\r\n\r\n'
        ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
        upload_started = time.perf_counter()
        uploaded = json.loads(request("/upload/image", upload_body, "multipart/form-data; boundary=" + boundary))
        upload_seconds = time.perf_counter() - upload_started
        timings['uploads_seconds'].append(upload_seconds)
        if not uploaded.get("name"):
            raise RuntimeError("ComfyUI upload returned no filename")
        upload_cache[actual]=uploaded
        name="/".join(part for part in (str(uploaded.get("subfolder") or ""),str(uploaded['name'])) if part)
        uploads.append({**public_row,'index':len(uploads)+1,'upload':uploaded,'condition_only':True,
                        'hard_edit_mask_enforced':False})
        names.append(name)
        events.append({'event':'comfyui_uploaded','index':len(uploads),'input_sha256':row['sha256'],
                       'bytes':len(data),'upload':uploaded,'seconds':upload_seconds})
    graph = prepare_condition_workflow(document,names,prompt,coordinate_space=coordinate_space)
    uploaded=uploads[0]['upload']
    if document.get("randomize_seed"):
        seed = secrets.randbits(63)
        for node in graph.values():
            if node["class_type"] in {"KSampler", "KSamplerAdvanced"}:
                node["inputs"]["seed"] = seed
    client_id = "cosplay-" + secrets.token_hex(16)
    submitted_workflow_sha256 = hashlib.sha256(json.dumps(
        graph, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    submit_started = time.perf_counter()
    submitted = json_request("/prompt", {"prompt": graph, "client_id": client_id})
    timings['submission_seconds'] = time.perf_counter() - submit_started
    prompt_id = submitted.get("prompt_id")
    if submitted.get("error") or submitted.get("node_errors") or not prompt_id:
        raise RuntimeError(f"ComfyUI rejected workflow: {json.dumps(submitted, ensure_ascii=False)}")
    prompt_id = str(prompt_id)
    workflow_timing.annotate(model_request=True, model_submissions=1)
    wait_started = time.perf_counter()
    events.append({"event": "comfyui_submitted", "prompt_id": prompt_id, "client_id": client_id,
                   "submitted_workflow_sha256": submitted_workflow_sha256,
                   "request": {"prompt": graph, "client_id": client_id},
                   "seconds":time.perf_counter()-started})
    output_node = str(document["output_node"]) if document.get("output_node") is not None else None
    while True:
        history = json_request("/history/" + urllib.parse.quote(prompt_id, safe=""))
        entry = history.get(prompt_id)
        if entry:
            status = entry.get("status") or {}
            if status.get("status_str") == "error":
                events.append({'event':'comfyui_execution_failed','prompt_id':prompt_id,
                               'status':status,'seconds':time.perf_counter()-started,'retrying':False})
                raise RuntimeError(f"ComfyUI execution failed: {json.dumps(status.get('messages', []), ensure_ascii=False)}")
            outputs = entry.get("outputs") or {}
            nodes = [(output_node, outputs.get(output_node, {}))] if output_node else list(outputs.items())
            images = [(node_id, descriptor) for node_id, output in nodes for descriptor in output.get("images", [])]
            if images:
                node_id, descriptor = images[0]
                query = urllib.parse.urlencode({"filename": descriptor["filename"],
                                                "subfolder": descriptor.get("subfolder", ""),
                                                "type": descriptor.get("type", "output")})
                timings['queue_and_execution_seconds'] = time.perf_counter()-wait_started
                fetch_started = time.perf_counter()
                output_bytes = request("/view?" + query)
                timings['fetch_seconds'] = time.perf_counter()-fetch_started
                timings['total_seconds'] = time.perf_counter()-started
                events.append({"event": "comfyui_completed", "prompt_id": prompt_id, "output_node": node_id,
                               'timings':timings,'status':status})
                return {
                    "id": prompt_id,
                    "comfyui": {
                        "prompt_id": prompt_id, "client_id": client_id,
                        "workflow_sha256": workflow_digest,
                        "submitted_workflow_sha256": submitted_workflow_sha256,
                        "workflow_path": str(workflow_path), "upload": uploaded,
                        "condition_inputs": uploads,
                        "model_name": document.get("model_name"),
                        "output_node": node_id, "output": descriptor,
                        "timings": timings, "execution_status": status,
                        "request": {"prompt": graph, "client_id": client_id},
                    },
                }, output_bytes, "comfyui_view"
            if status.get("completed"):
                raise RuntimeError("ComfyUI completed without an image at the selected output node")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("ComfyUI generation timed out; no candidate registered")
        time.sleep(min(poll_interval, remaining))
