"""GPT vision JSON transport, separate from the image-generation adapter."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageCms

import provider_settings
import reference_provider
from raw_support import RGB_DECODE_VERSION, is_raw, open_rgb


TIMEOUT = 180
VISION_GUIDE = "https://developers.openai.com/api/docs/guides/images-vision"


class VisionHTTPError(RuntimeError):
    def __init__(self, status: int, detail: str):
        self.status = status
        self.detail = detail
        super().__init__(f"vision provider HTTP {status}: {detail}")


class VisionResponseError(RuntimeError):
    """An observed response failure; retry authority requires its saved evidence."""
    def __init__(self, failure_kind, message, *, response=None, event_type=None):
        self.failure_kind = failure_kind
        self.failure_evidence = None
        self._response = response
        self._event_type = event_type
        super().__init__(message)


def _sanitized(value, key):
    if isinstance(value, dict):
        return {_sanitized(name, key): '<redacted>' if str(name).lower() in
                {'authorization', 'api_key', 'api-key', 'x-api-key', 'access_token',
                 'b64_json', 'image_base64', 'base64_image'}
                else _sanitized(item, key) for name, item in value.items()}
    if isinstance(value, list):
        return [_sanitized(item, key) for item in value]
    if isinstance(value, str):
        value = value.replace(key, '<redacted>') if key else value
        return re.sub(r'data:image/[^\s"\']+', '<image-data-redacted>', value)
    return value


def _record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': reference_provider.sha256(path), 'bytes': path.stat().st_size}


def _exclusive_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def _final_prompt(prompt):
    return prompt + "\nReturn only one JSON object, without markdown. Follow the requested JSON schema and its uncertainty conventions. Do not guess unsupported identities."


def _terminal_provider_failure(payload, event_type=None):
    if not isinstance(payload, dict):
        return False
    return bool(event_type in {'response.failed', 'error'}
                or payload.get('status') == 'failed' or payload.get('error'))


def _provider_retryable(payload, event_type):
    if not _terminal_provider_failure(payload, event_type):
        return False
    if event_type is not None and payload.get('type') != event_type:
        return False
    if payload.get('type') in {'response.incomplete', 'response.completed'}:
        return False
    if payload.get('status') in {'completed', 'incomplete', 'in_progress', 'queued'}:
        return False
    terminal = payload.get('response') if event_type == 'response.failed' else payload
    if not isinstance(terminal, dict) or terminal.get('status') in {'completed', 'incomplete', 'in_progress', 'queued'}:
        return False
    error = terminal.get('error')
    return bool(isinstance(error, dict) and error.get('type') == 'server_error'
        and error.get('code') in {'upstream_http2_stream_error', 'upstream_error'})


def _response_text(response, protocol):
    if protocol == 'chat/completions':
        choices = response.get('choices')
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return None
        message = choices[0].get('message') or {}
        return message.get('content') if isinstance(message, dict) else None
    return response.get('output_text') or ''.join(
        part.get('text', '') for item in response.get('output', []) if isinstance(item, dict)
        and item.get('type') == 'message' for part in item.get('content', [])
        if isinstance(part, dict) and part.get('type') == 'output_text')


def _completed(response, protocol):
    if response.get('type') in {'error', 'response.failed', 'response.incomplete'}:
        return False
    if protocol == 'responses':
        return response.get('status') == 'completed'
    choices = response.get('choices')
    return bool(isinstance(choices, list) and choices and isinstance(choices[0], dict)
                and choices[0].get('finish_reason') == 'stop')


def _syntax_invalid(text):
    if not isinstance(text, str) or not text.strip():
        return False
    try:
        _profile_json(text)
    except RuntimeError as exc:
        return isinstance(exc.__cause__, json.JSONDecodeError)
    return False


def error_metadata(exc, api_key_env=None):
    """Additional structured metadata; consumers retain type/message and HTTP ABI."""
    if isinstance(exc, VisionHTTPError):
        return {'http_status': exc.status}
    if isinstance(exc, VisionResponseError) and exc.failure_evidence:
        return {'failure_kind': exc.failure_kind, 'failure_evidence': dict(exc.failure_evidence)}
    return {}


def retry_policy_for_error(error):
    if not isinstance(error, dict):
        return None
    status = error.get('http_status')
    if (error.get('type') == 'VisionHTTPError' and type(status) is int
            and (status == 429 or 500 <= status <= 599)
            and str(error.get('message', '')).startswith('vision provider HTTP '+str(status)+': ')):
        return 'one_retry_after_explicit_http_429_or_5xx'
    if error.get('type') != 'VisionResponseError' or not isinstance(error.get('failure_evidence'), dict):
        return None
    try:
        asset = error['failure_evidence']
        if _record(asset['path']) != asset:
            return None
        receipt = json.loads(Path(asset['path']).read_text(encoding='utf-8'))
        payload = receipt['response_payload']
        if _record(payload['path']) != payload or receipt.get('failure_kind') != error.get('failure_kind'):
            return None
        response = json.loads(Path(payload['path']).read_text(encoding='utf-8'))
        if error['failure_kind'] == 'provider_failure' and _provider_retryable(response, receipt.get('event_type')):
            return 'one_retry_after_explicit_provider_upstream_error'
        if (error['failure_kind'] == 'completed_invalid_json' and _completed(response, receipt['protocol'])
                and not _terminal_provider_failure(response)
                and _syntax_invalid(_response_text(response, receipt['protocol']))):
            return 'one_retry_after_completed_response_invalid_json'
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None
    return None


def validate_saved_failure(error, comparison, prompt):
    """Reconstruct qualification and exact request pixels without sending a request.

    Returned closure is inside the historical attempt directory. The original
    comparison is independently checked against the current same-byte board.
    """
    policy = retry_policy_for_error(error)
    if policy is None:
        raise ValueError('Saved vision failure does not establish retry qualification')
    if error.get('type') == 'VisionHTTPError':
        return []
    asset = error['failure_evidence']
    receipt = json.loads(Path(asset['path']).read_text(encoding='utf-8'))
    directory = Path(asset['path']).resolve().parent
    if (Path(asset['path']).name != 'response-failure.json'
            or receipt.get('schema') != 'vision_response_failure.v1'):
        raise ValueError('Unknown saved response failure receipt')
    binding = receipt['request_binding']
    if _record(binding['path']) != binding or Path(binding['path']).resolve() != directory/'request-binding.json':
        raise ValueError('Saved failure request binding changed or belongs to another attempt')
    request = json.loads(Path(binding['path']).read_text(encoding='utf-8'))
    expected_prompt_hash = hashlib.sha256(_final_prompt(prompt).encode('utf-8')).hexdigest()
    if (request.get('schema') != 'vision_request_binding.v1'
            or receipt['protocol'] not in {'responses', 'chat/completions'}
            or receipt['protocol'] not in request['allowed_protocols']
            or request['prompt_sha256'] != expected_prompt_hash):
        raise ValueError('Saved failure protocol or prompt differs from the actual board request')
    original = request['source']
    if (_record(original['path']) != original or reference_provider.sha256(Path(comparison)) != original['sha256']):
        raise ValueError('Saved failure source differs from the actual comparison pixels')
    image = request['api_input']
    image_path = Path(image['path']).resolve()
    if (image_path != directory/'inputs/vision-input.jpg' or image.get('source') !=
            {'path': original['path'], 'sha256': original['sha256']}):
        raise ValueError('Saved failure JPEG belongs to another source or attempt')
    image_record = _record(image_path)
    if any(image_record[key] != image.get(key) for key in ('path', 'sha256', 'bytes')):
        raise ValueError('Saved failure API JPEG bytes changed')
    with tempfile.TemporaryDirectory(prefix='vision-failure-replay-') as tmp:
        replay, metadata = _input_image(Path(original['path']), Path(tmp))
        if reference_provider.sha256(replay) != image['sha256'] or any(
                metadata[k] != image.get(k) for k in ('format', 'size', 'derived', 'decode')):
            raise ValueError('Saved failure JPEG does not depict the bound comparison transport')
    payload = receipt['response_payload']
    if Path(payload['path']).resolve() != directory/'response-failure-payload.json' or _record(payload['path']) != payload:
        raise ValueError('Saved terminal response payload changed or belongs to another attempt')
    proof = [asset, binding, image_record, payload]
    if any(_record(row['path']) != row for row in proof):
        raise ValueError('Saved failure evidence changed during validation')
    return proof


def _save_response_failure(exc, out_dir, request_binding, protocol, key):
    response = _sanitized(exc._response, key)
    payload_path = out_dir/'response-failure-payload.json'
    _exclusive_json(payload_path, response)
    receipt = {'schema': 'vision_response_failure.v1', 'failure_kind': exc.failure_kind,
        'event_type': exc._event_type, 'protocol': protocol,
        'request_binding': request_binding, 'response_payload': _record(payload_path)}
    path = out_dir/'response-failure.json'
    _exclusive_json(path, receipt)
    exc.failure_evidence = _record(path)
    exc.args = (_safe_detail(str(exc), key),)
    exc._response = response


def _safe_detail(value: str, key: str) -> str:
    value = value.replace(key, "<redacted>") if key else value
    value = re.sub(r"data:image/[^\s\"']+", "<image-data-redacted>", value)
    return value[:500]


def _request(endpoint: str, key: str, body: dict | None = None) -> dict:
    request = urllib.request.Request(endpoint,
        data=json.dumps(body).encode("utf-8") if body is not None else None,
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                 "Accept": "application/json, text/event-stream",
                 "User-Agent": "cosplay-vision/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            if response.headers.get_content_type() == "text/event-stream":
                for line in response:
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == b"[DONE]":
                        continue
                    event = json.loads(data)
                    if event.get("type") == "response.completed":
                        return event["response"]
                    if event.get("type") in {"error", "response.failed"}:
                        message = event.get("message") or event.get("response", {}).get("error") or event.get("type")
                        raise VisionResponseError('provider_failure', "vision response failed: " + _safe_detail(str(message), key),
                            response=event, event_type=event.get('type'))
                    if event.get('type') == 'response.incomplete':
                        raise RuntimeError('vision response incomplete')
                raise RuntimeError("vision response stream ended without a completed response")
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = _safe_detail(exc.read(4096).decode("utf-8", errors="replace"), key)
        raise VisionHTTPError(exc.code, detail) from None
    except urllib.error.URLError as exc:
        raise RuntimeError("vision provider connection failed: " + _safe_detail(str(exc.reason), key)) from None
    if not isinstance(payload, dict):
        raise RuntimeError("vision provider returned a non-object JSON response")
    if _terminal_provider_failure(payload):
        raise VisionResponseError('provider_failure', 'vision response failed: '+_safe_detail(str(payload.get('error') or 'failed'), key), response=payload)
    return payload


def _connection(*, provider=None, api_base=None, api_url=None, api_key_env=None):
    config = reference_provider.provider_config(provider=provider, api_base=api_base,
                                                api_url=api_url, api_key_env=api_key_env)
    entry = reference_provider._provider_entry(config["registry_id"], reference_provider.load_provider_registry())
    if config["kind"] != "openai-compatible" and entry.get("vision_provider") == "auto":
        # Image generation and text/vision planning have independent transports.
        # Never send GPT credentials to a local image server's endpoint.
        chosen = os.environ.get("COSPLAY_VISION_PROVIDER")
        if not chosen:
            default = provider_settings.load_settings().get("default_provider")
            if default and reference_provider._provider_entry(default, reference_provider.load_provider_registry()).get("kind") == "openai-compatible":
                chosen = default
            else:
                chosen = "ccswitch_sub2api" if provider_settings.discover_ccswitch()["available"] else "openai-compatible"
        config = reference_provider.provider_config(provider=chosen,
            api_base=os.environ.get("COSPLAY_VISION_API_BASE"),
            api_url=os.environ.get("COSPLAY_VISION_API_URL"),
            api_key_env=os.environ.get("COSPLAY_VISION_API_KEY_ENV"))
    if config["kind"] != "openai-compatible":
        raise ValueError("GPT vision requires an OpenAI-compatible provider")
    endpoint = str(config["endpoint"] or "").rstrip("/")
    selected_endpoint = None
    for suffix in ("/images/edits", "/responses", "/chat/completions", "/models"):
        if endpoint.endswith(suffix):
            if suffix in {"/responses", "/chat/completions"}:
                selected_endpoint = suffix
            endpoint = endpoint[:-len(suffix)]
            break
    key = provider_settings.resolve_credential(config["registry_id"], config["api_key_env"],
                                               config["credential_source"], endpoint)
    if not key:
        raise RuntimeError("vision provider credentials are unavailable")
    return config, endpoint, key, selected_endpoint


def _vision_models(payload: dict) -> list[str]:
    names = [str(item["id"]) for item in payload.get("data", []) if isinstance(item, dict) and item.get("id")]
    return [name for name in names if re.match(r"^gpt-(?:[456](?:[.\-]|$)|4o)", name, re.I)
            and not any(term in name.lower() for term in ("image", "audio", "realtime", "transcribe", "search", "spark"))]


def discover_vision_models(*, provider=None, api_base=None, api_url=None, api_key_env=None) -> dict:
    """Read actual gateway model IDs; never substitute an image model."""
    config, endpoint, key, _ = _connection(provider=provider, api_base=api_base,
                                          api_url=api_url, api_key_env=api_key_env)
    models = _vision_models(_request(endpoint + "/models", key))
    return {"provider": config["registry_id"], "endpoint": endpoint,
            "models": models, "network_checked": True}


def _model_rank(name: str):
    match = re.match(r"gpt-(\d+)(?:\.(\d+))?", name)
    major, minor = (int(match[1]), int(match[2] or 0)) if match else (4, 0)
    return major, minor, -int(any(term in name for term in ("mini", "luna", "nano"))), name


def _input_image(source: Path, out_dir: Path) -> tuple[Path, dict]:
    if is_raw(source):
        return reference_provider.prepare_api_input(source, out_dir, long_edge=1024)
    decoded = open_rgb(source)
    oriented_size = list(decoded.size)
    decoded.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    target = out_dir.resolve() / "inputs" / "vision-input.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    from raw_support import deterministic_srgb_profile
    profile = deterministic_srgb_profile()
    with target.open('xb') as stream:
        decoded.save(stream, "JPEG", quality=93, subsampling=0, icc_profile=profile)
    return target, {"path": str(target), "sha256": reference_provider.sha256(target),
        "bytes": target.stat().st_size, "format": "JPEG", "size": list(decoded.size),
        "derived": True, "source": {"path": str(source.resolve()), "sha256": reference_provider.sha256(source)},
        "decode": {"decode_version": RGB_DECODE_VERSION, "oriented_size": oriented_size,
                   "output_transfer": "IEC_sRGB", "long_edge_limit": 1024}}


def _profile_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("vision model returned invalid profile JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("vision model must return one profile JSON object")
    return payload


def request_vision_json(source: Path, out_dir: Path, prompt: str, *, provider=None,
                        vision_model=None, api_base=None, api_url=None,
                        api_key_env=None) -> tuple[dict, dict]:
    """Send one visual recognition request; errors remain explicit and usable."""
    source, out_dir = Path(source).resolve(), Path(out_dir).resolve()
    model = vision_model or os.environ.get("COSPLAY_VISION_MODEL")
    if "gpt-image" in str(model).lower():
        raise ValueError("gpt-image models cannot perform text vision recognition")
    if out_dir.exists() and any(path.is_file() for path in out_dir.rglob('*')):
        raise FileExistsError('Vision attempt evidence already exists; use a new independent directory')
    config, endpoint, key, selected_endpoint = _connection(provider=provider,
        api_base=api_base, api_url=api_url, api_key_env=api_key_env)
    discovered_models = []
    if not model:
        discovered_models = _vision_models(_request(endpoint + "/models", key))
        if not discovered_models:
            raise RuntimeError("provider /models returned no GPT vision text model; set COSPLAY_VISION_MODEL")
        model = max(discovered_models, key=_model_rank)
    if "gpt-image" in str(model).lower():
        raise ValueError("gpt-image models cannot perform text vision recognition")
    source_before = reference_provider.sha256(source)
    image, image_meta = _input_image(source, out_dir)
    if image_meta['source']['sha256'] != source_before:
        raise ValueError('Vision source changed during input preparation')
    image_uri = "data:image/jpeg;base64," + base64.b64encode(image.read_bytes()).decode("ascii")
    final_prompt = _final_prompt(prompt)
    protocol = "responses" if selected_endpoint != "/chat/completions" else "chat/completions"
    fallback = None
    binding_path = out_dir/'request-binding.json'
    _exclusive_json(binding_path, {'schema': 'vision_request_binding.v1', 'source': _record(source),
        'api_input': image_meta, 'prompt_sha256': hashlib.sha256(final_prompt.encode('utf-8')).hexdigest(),
        'allowed_protocols': ['responses', 'chat/completions'] if protocol == 'responses' else ['chat/completions']})
    binding = _record(binding_path)
    def send(url, body):
        try:
            result = _request(url, key, body)
            # Also inspect injected/non-streamed HTTP200 terminal envelopes.
            if _terminal_provider_failure(result):
                raise VisionResponseError('provider_failure', 'vision response failed: '+_safe_detail(str(result.get('error') or 'failed'), key), response=result)
            return result
        except VisionResponseError as exc:
            _save_response_failure(exc, out_dir, binding, protocol, key)
            raise
    if protocol == "responses":
        body = {"model": model, "store": False, "stream": True,
                "input": [{"role": "user", "content": [
                    {"type": "input_text", "text": final_prompt},
                    {"type": "input_image", "image_url": image_uri}]}]}
        try:
            response = send(endpoint + "/responses", body)
        except VisionHTTPError as exc:
            unsupported = exc.status in {405, 501} or (exc.status == 404
                and "model" not in exc.detail.lower()) or (exc.status in {400, 422}
                and "responses" in exc.detail.lower()
                and any(term in exc.detail.lower() for term in ("unsupported", "not supported", "unknown endpoint")))
            if not unsupported:
                raise
            fallback = {"from": "responses", "http_status": exc.status, "reason": "endpoint_unsupported"}
            protocol = "chat/completions"
    if protocol == "chat/completions":
        body = {"model": model, "messages": [{"role": "user", "content": [
            {"type": "text", "text": final_prompt},
            {"type": "image_url", "image_url": {"url": image_uri}}]}]}
        response = send(endpoint + "/chat/completions", body)
    text = _response_text(response, protocol)
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("vision model returned no profile text")
    try:
        payload = _profile_json(text)
    except RuntimeError as exc:
        if isinstance(exc.__cause__, json.JSONDecodeError) and _completed(response, protocol):
            failure = VisionResponseError('completed_invalid_json', 'completed vision model returned invalid profile JSON', response=response)
            _save_response_failure(failure, out_dir, binding, protocol, key)
            raise failure from None
        raise
    provenance = {"schema_version": "vision_transport.v1", "provider": config["registry_id"],
        "model": model, "response_model": response.get("model"), "response_id": response.get("id"),
        "endpoint": endpoint + "/" + protocol, "protocol": protocol,
        "created_at": datetime.now(timezone.utc).isoformat(), "timeout_seconds": TIMEOUT,
        "source": {"path": str(source), "sha256": reference_provider.sha256(source)},
        "api_input": image_meta, "input_sha256": image_meta["sha256"],
        "prompt_sha256": hashlib.sha256(final_prompt.encode("utf-8")).hexdigest(),
        "usage": response.get("usage"), "discovered_models": discovered_models,
        "fallback": fallback, "network_used": True, "generation_used": False,
        "docs_source": VISION_GUIDE}
    return payload, provenance
