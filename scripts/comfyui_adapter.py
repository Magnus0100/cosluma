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


def load_workflow(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(document, dict):
        raise ValueError("ComfyUI workflow must be an API-format JSON object")
    graph = document.get("workflow", document)
    if not isinstance(graph, dict) or not graph or any(
        not isinstance(node, dict) or "class_type" not in node or not isinstance(node.get("inputs"), dict)
        for node in graph.values()
    ):
        raise ValueError("ComfyUI requires an API-format node map; export 'Save (API Format)', not the UI graph")
    return document


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
    return graph


def prepare_condition_workflow(document: dict[str, Any], names: list[str], prompt: str) -> dict[str, Any]:
    """Bind ordered Qwen 2.1 condition images; no hard-mask claim."""
    if not 1 <= len(names) <= 10:
        raise ValueError('Qwen 2.1 conditions must contain one to ten images')
    graph=prepare_workflow(document,names[0],prompt)
    if len(names)==1:
        return graph
    encoders=[row for row in graph.values() if row['class_type']=='TextEncodeQwenImage21']
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
    document = load_workflow(workflow_path)
    workflow_digest = hashlib.sha256(workflow_path.read_bytes()).hexdigest()
    # Validate template before performing an upload.
    prepare_workflow(document, "pending-source.png", prompt)
    conditions=condition_inputs or [{'role':'source','path':str(image)}]
    if not 1 <= len(conditions) <= 10 or Path(conditions[0]['path']).resolve()!=image.resolve():
        raise ValueError('Condition images require one to ten images with --image first')
    files=[]
    for row in conditions:
        path=Path(row['path']).resolve()
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if row.get('sha256') and row['sha256']!=actual:
            raise ValueError('Condition image bytes changed before upload')
        files.append({**row,'path':str(path),'sha256':actual,'bytes':path.stat().st_size})
    prepare_condition_workflow(document,['pending-'+str(i)+'.png' for i in range(len(files))],prompt)
    deadline = time.monotonic() + timeout
    base = endpoint.rstrip("/")
    headers = {"User-Agent": "cosplay-comfyui-provider/1.0"}
    if key:
        headers["Authorization"] = "Bearer " + key

    def request(path: str, body: bytes | None = None, content_type: str | None = None) -> bytes:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("ComfyUI generation timed out")
        request_headers = dict(headers)
        if content_type:
            request_headers["Content-Type"] = content_type
        req = urllib.request.Request(base + path, data=body, headers=request_headers)
        with urllib.request.urlopen(req, timeout=remaining) as response:
            return response.read()

    def json_request(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        result = json.loads(request(path, data, "application/json" if data is not None else None))
        if not isinstance(result, dict):
            raise RuntimeError(f"ComfyUI {path} returned a non-object response")
        return result

    uploads=[];names=[]
    for row in files:
        path=Path(row['path']); data=path.read_bytes()
        if hashlib.sha256(data).hexdigest()!=row['sha256']:
            raise ValueError('Condition image changed during upload preparation')
        boundary = "cosplay-" + secrets.token_hex(16)
        upload_name = "cosplay-" + secrets.token_hex(12) + path.suffix.lower()
        upload_body = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{upload_name}"\r\n'
            f'Content-Type: {mimetypes.guess_type(path.name)[0] or "application/octet-stream"}\r\n\r\n'
        ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
        uploaded = json.loads(request("/upload/image", upload_body, "multipart/form-data; boundary=" + boundary))
        if not uploaded.get("name"):
            raise RuntimeError("ComfyUI upload returned no filename")
        name="/".join(part for part in (str(uploaded.get("subfolder") or ""),str(uploaded['name'])) if part)
        uploads.append({**row,'index':len(uploads)+1,'upload':uploaded,'condition_only':True,
                        'hard_edit_mask_enforced':False})
        names.append(name)
    graph = prepare_condition_workflow(document,names,prompt)
    uploaded=uploads[0]['upload']
    if document.get("randomize_seed"):
        seed = secrets.randbits(63)
        for node in graph.values():
            if node["class_type"] in {"KSampler", "KSamplerAdvanced"}:
                node["inputs"]["seed"] = seed
    client_id = "cosplay-" + secrets.token_hex(16)
    submitted = json_request("/prompt", {"prompt": graph, "client_id": client_id})
    prompt_id = submitted.get("prompt_id")
    if submitted.get("error") or submitted.get("node_errors") or not prompt_id:
        raise RuntimeError(f"ComfyUI rejected workflow: {json.dumps(submitted, ensure_ascii=False)}")
    prompt_id = str(prompt_id)
    events.append({"event": "comfyui_submitted", "prompt_id": prompt_id, "client_id": client_id,
                   "submitted_workflow_sha256": hashlib.sha256(json.dumps(graph, sort_keys=True,
                       separators=(",", ":")).encode()).hexdigest(),
                   "request": {"prompt": graph, "client_id": client_id}})
    output_node = str(document["output_node"]) if document.get("output_node") is not None else None
    while True:
        history = json_request("/history/" + urllib.parse.quote(prompt_id, safe=""))
        entry = history.get(prompt_id)
        if entry:
            status = entry.get("status") or {}
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI execution failed: {json.dumps(status.get('messages', []), ensure_ascii=False)}")
            outputs = entry.get("outputs") or {}
            nodes = [(output_node, outputs.get(output_node, {}))] if output_node else list(outputs.items())
            images = [(node_id, descriptor) for node_id, output in nodes for descriptor in output.get("images", [])]
            if images:
                node_id, descriptor = images[0]
                query = urllib.parse.urlencode({"filename": descriptor["filename"],
                                                "subfolder": descriptor.get("subfolder", ""),
                                                "type": descriptor.get("type", "output")})
                output_bytes = request("/view?" + query)
                events.append({"event": "comfyui_completed", "prompt_id": prompt_id, "output_node": node_id})
                return {
                    "id": prompt_id,
                    "comfyui": {
                        "prompt_id": prompt_id, "client_id": client_id,
                        "workflow_sha256": workflow_digest,
                        "submitted_workflow_sha256": hashlib.sha256(json.dumps(graph, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                        "workflow_path": str(workflow_path), "upload": uploaded,
                        "condition_inputs": uploads,
                        "model_name": document.get("model_name"),
                        "output_node": node_id, "output": descriptor,
                        "request": {"prompt": graph, "client_id": client_id},
                    },
                }, output_bytes, "comfyui_view"
            if status.get("completed"):
                raise RuntimeError("ComfyUI completed without an image at the selected output node")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("ComfyUI generation timed out; no candidate registered")
        time.sleep(min(poll_interval, remaining))
