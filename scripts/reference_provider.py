#!/usr/bin/env python3
"""Generate a reference candidate through OpenAI-compatible or ComfyUI APIs.

This adapter owns transport and provider provenance only.  The generated
image is handed to ``register_reference.py`` for normalization and candidate
registration; it never creates a manual final or edits a dataset lock.

Public connection settings live in per-user provider-settings.json. Secrets
stay in the existing environment or read-only CCSwitch/Sub2API store. The
current authorized CCSwitch provider is discovered without shell injection.
Environment overrides remain available:

    $env:COSPLAY_REFERENCE_API_BASE = "https://api.example/v1"
    $env:COSPLAY_REFERENCE_API_KEY = "..."       # never put this in a file
    $env:COSPLAY_REFERENCE_MODEL = "gpt-image-1"

The endpoint can be overridden with ``COSPLAY_REFERENCE_API_URL`` or
``--api-url``.  The request is a multipart ``POST /images/edits`` and accepts
the common ``data[0].b64_json`` or ``data[0].url`` response forms.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import secrets
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
import register_reference  # noqa: E402
import comfyui_adapter  # noqa: E402
import provider_settings  # noqa: E402
from raw_support import RGB_DECODE_VERSION, is_raw, open_rgb, _require_rawpy  # noqa: E402

REGISTRY_PATH = Path(getattr(sys, "_MEIPASS", ROOT)) / "providers" / "registry.json"

# The registry is also the extension point for non-OpenAI engines.  Keeping
# this small contract here lets the UI discover and dry-run an adapter without
# pulling a local model, ComfyUI, or any optional dependency into the core
# package. Local HTTP remains discovery-only; ComfyUI has a concrete adapter.
_ADAPTER_DEFAULTS: dict[str, dict[str, Any]] = {
    "openai-compatible": {"protocol": "openai-compatible", "generation_supported": True},
    "manual_import": {"protocol": "manual-import", "generation_supported": False},
    "local": {"protocol": "local-http", "generation_supported": False},
    "comfyui": {"protocol": "comfyui-api", "generation_supported": True},
}


def adapter_contract(entry: dict[str, Any]) -> dict[str, Any]:
    """Return public adapter metadata with conservative defaults.

    Unknown registry kinds are discoverable but never treated as an
    OpenAI-compatible transport by accident.
    """
    kind = str(entry.get("kind") or "openai-compatible")
    defaults = _ADAPTER_DEFAULTS.get(kind, {"protocol": kind, "generation_supported": False})
    return {
        "adapter": entry.get("adapter") or kind,
        "protocol": entry.get("protocol") or defaults["protocol"],
        "generation_supported": bool(entry.get("generation_supported", defaults["generation_supported"])),
    }


def load_provider_registry(path: Optional[Path] = None) -> list[dict[str, Any]]:
    """Load the public provider catalog (secrets are never stored here)."""
    registry_path = path or REGISTRY_PATH
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return []
    entries = payload.get("providers") if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict) and isinstance(entry.get("id"), str)]


def _provider_entry(name: Optional[str], entries: list[dict[str, Any]]) -> dict[str, Any]:
    requested = (name or "openai-compatible").strip()
    folded = requested.casefold()
    for entry in entries:
        aliases = {str(entry.get("id", "")).casefold(), str(entry.get("display", "")).casefold()}
        if folded in aliases:
            return entry
    # Preserve historical labels and arbitrary user-defined providers even if
    # an old registry file has not yet been updated.
    if folded in {"ccswitch/sub2api", "ccswitch", "sub2api"}:
        for entry in entries:
            if str(entry.get("id", "")).casefold() == "ccswitch_sub2api":
                return entry
    for entry in entries:
        if str(entry.get("kind", "")).casefold() == folded:
            return entry
    return {"id": requested, "kind": "openai-compatible", "model": None,
            "endpoint_env": "COSPLAY_REFERENCE_API_BASE", "key_env": "COSPLAY_REFERENCE_API_KEY",
            "capabilities": [], "timeout": 180, "retry": {}, "enabled": True}


def provider_registry() -> list[dict[str, Any]]:
    """Return registry entries with runtime configuration status, no secrets."""
    result = []
    for raw in load_provider_registry():
        entry = dict(raw)
        config = provider_config(provider=str(entry["id"]))
        contract = adapter_contract(entry)
        entry.update(contract)
        entry["configured"] = bool(config.get("configured"))
        entry["endpoint_configured"] = bool(config.get("endpoint")) if entry.get("kind") != "manual_import" else True
        entry["status"] = str(provider_health(provider=str(entry["id"])).get("status"))
        entry["ready"] = bool(provider_health(provider=str(entry["id"])).get("ready"))
        result.append(entry)
    return result


def provider_health(*, provider: Optional[str] = None, api_base: Optional[str] = None,
                    api_url: Optional[str] = None, model: Optional[str] = None,
                    api_key_env: Optional[str] = None) -> dict[str, Any]:
    """Return a local provider readiness report without making a network call.

    This is deliberately a configuration health check.  It reports only
    booleans, public endpoint/config metadata, and the environment variable
    *name* used for the key; secret values are never returned.
    """
    config = provider_config(provider=provider, api_base=api_base,
                             api_url=api_url, model=model,
                             api_key_env=api_key_env)
    endpoint = config.get("endpoint")
    kind = str(config.get("kind") or "")
    endpoint_ok = bool(endpoint) if kind != "manual_import" else True
    key_ok = bool(config.get("api_key_configured")) if kind != "manual_import" else True
    enabled = bool(config.get("enabled", True))
    supported = bool(config.get("generation_supported"))
    configured = bool(config.get("configured"))
    missing_models: list[str] = []
    if config["registry_id"] in {"qwen_image_2_1", "qwen_image_2_1_uncensored", "qwen_image_2_1_fast", "qwen_image_2_1_uncensored_fast"} and endpoint == "http://127.0.0.1:8188":
        from start_qwen_image import model_files
        missing_models = [entry["role"] for entry in model_files(config["registry_id"]) if not entry["complete"]]
    if not enabled:
        status, reason = "disabled", "provider_disabled"
    elif kind == "manual_import":
        # Manual import is a usable source of a candidate even though it has
        # no generation capability.
        status, reason = "ready", "manual_import"
    elif not supported:
        # A discovery-only adapter can expose configuration without claiming
        # a working generation implementation.
        status = "configured" if configured else "unsupported"
        reason = "adapter_not_implemented"
    elif not endpoint_ok:
        status, reason = "missing_endpoint", "endpoint_not_configured"
    elif kind == "comfyui" and not config.get("workflow_configured"):
        status, reason = "missing_workflow", "workflow_file_not_configured"
    elif missing_models:
        status, reason = "missing_models", "model_files_incomplete"
    elif not key_ok:
        status, reason = "missing_key", "api_key_not_configured"
    else:
        status, reason = "ready", "configured"
    ready = bool(enabled and not missing_models and (kind == "manual_import" or (supported and endpoint_ok and key_ok and
                 (kind != "comfyui" or config.get("workflow_configured")))))
    return {
        "provider": config.get("provider"),
        "registry_id": config.get("registry_id"),
        "kind": kind,
        "model": config.get("model"),
        "endpoint": endpoint,
        "endpoint_configured": bool(config.get("endpoint")) if kind != "manual_import" else True,
        "api_key_env": config.get("api_key_env"),
        "api_key_configured": key_ok,
        "credential_source": config.get("credential_source"),
        "enabled": enabled,
        "timeout": config.get("timeout"),
        "retry": config.get("retry") or {},
        "ready": ready,
        "configured": configured,
        "generation_supported": supported,
        "adapter": config.get("adapter"),
        "protocol": config.get("protocol"),
        "workflow_path": config.get("workflow_path"),
        "workflow_configured": config.get("workflow_configured"),
        "status": status,
        "reason": reason,
        "network_checked": False,
        "missing_models": missing_models,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_condition_inputs(path: Path, image: Path, root: Path,
                             out_dir: Path) -> list[dict[str, Any]]:
    """Validate explicit ordered raster conditions before any network request."""
    from PIL import Image
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    rows = document.get("inputs") if isinstance(document, dict) else document
    if not isinstance(rows, list) or not 1 <= len(rows) <= 10:
        raise ValueError("--inputs-json requires one to ten ordered inputs")
    allowed = {"scene_reference", "source_identity", "edit_mask", "identity_crop", "source"}
    result = []
    roles = set()
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict) or row.get("role") not in allowed:
            raise ValueError("Unknown condition image role")
        role = row["role"]
        if role in roles or not row.get("path") or not row.get("sha256") or not row.get("coordinate_space"):
            raise ValueError("Conditions require unique roles, path, SHA and coordinate_space")
        roles.add(role)
        actual = Path(row["path"]).expanduser()
        actual = actual.resolve() if actual.is_absolute() else (root / actual).resolve()
        if not actual.is_file() or sha256(actual) != row["sha256"]:
            raise ValueError("Condition image bytes differ from declared SHA")
        if is_raw(actual):
            raise ValueError("Explicit conditions require a bound displayed raster derivative, not RAW")
        if index == 1 and actual != image.resolve():
            raise ValueError("First condition must be the actual --image / output canvas")
        _, api_input = prepare_api_input(actual, out_dir)
        if api_input.get("orientation") not in (None, 1):
            raise ValueError("Explicit condition raster must already have orientation applied")
        record = {"role": role, "index": index, "input_sha256": row["sha256"],
                  "coordinate_space": row["coordinate_space"], "api_input": api_input}
        if role == "identity_crop":
            box = row.get("box")
            parent = next((item for item in result if item["role"] == "source_identity"), None)
            if (not isinstance(box, list) or len(box) != 4 or
                    any(type(item) is not int for item in box) or parent is None or
                    row.get("parent_sha256") != parent["input_sha256"]):
                raise ValueError("Identity crop requires bound box and parent_sha256")
            with Image.open(parent["api_input"]["path"]) as original, Image.open(actual) as crop:
                left, top, right, bottom = box
                if not (0 <= left < right <= original.width and 0 <= top < bottom <= original.height):
                    raise ValueError("Identity crop box lies outside its source_identity parent")
                expected = original.convert("RGB").crop(tuple(box))
                if crop.size != expected.size or crop.convert("RGB").tobytes() != expected.tobytes():
                    raise ValueError("Identity crop pixels differ from the declared parent box")
            record.update(box=box, parent_sha256=row["parent_sha256"])
        if role == "edit_mask":
            with Image.open(actual) as opened:
                if opened.mode != "L" or opened.size != tuple(result[0]["api_input"]["size"]):
                    raise ValueError("Edit mask must be L in the first image's exact canvas")
                values = {item[1] for item in opened.getcolors(maxcolors=257) or []}
                if values != {0, 255}:
                    raise ValueError("Edit mask must have nonempty binary allowed and excluded regions")
            if result[0]["role"] != "scene_reference" or row["coordinate_space"] != result[0]["coordinate_space"]:
                raise ValueError("Edit mask must declare the scene_reference coordinate space")
            record.update(white_means="edit_allowed", conditioning_only=True,
                          hard_edit_mask_enforced=False)
        result.append(record)
    if "edit_mask" in roles and "source_identity" not in roles:
        raise ValueError("Material correction requires source_identity as a separate condition")
    if "edit_mask" in roles and [row["role"] for row in result[:3]] != ["scene_reference", "source_identity", "edit_mask"]:
        raise ValueError("Material correction order must be scene_reference, source_identity, edit_mask")
    return result


def prepare_api_input(image: Path, out_dir: Path, long_edge: int = 2048) -> tuple[Path, dict[str, Any]]:
    """Make a camera RAW uploadable without altering or copying the original.

    Raster inputs retain their existing request bytes. RAW decoding mirrors
    the project's camera-WB/no-auto-bright RGB boundary; only the API JPEG is
    resized. The derivative is an input artifact, never a reference candidate.
    """
    from PIL import Image, ImageCms

    original = {"path": str(image.resolve()), "sha256": sha256(image), "bytes": image.stat().st_size}
    if not is_raw(image):
        with Image.open(image) as opened:
            facts = {"format": opened.format, "mode": opened.mode, "size": list(opened.size),
                     "orientation": opened.getexif().get(274)}
        return image, {**original, **facts, "derived": False,
                       "decode": {"decoder": "passthrough", "orientation_applied": False}}

    rawpy = _require_rawpy()

    with rawpy.imread(str(image)) as raw:
        sensor_size = [int(raw.sizes.width), int(raw.sizes.height)]
        flip = int(raw.sizes.flip)
    # Share the workbench's explicit RAW16 linear -> IEC sRGB boundary. The
    # provider must see the same source appearance that is rendered locally.
    decoded = open_rgb(image)
    oriented_size = list(decoded.size)
    decoded.thumbnail((long_edge, long_edge), Image.Resampling.LANCZOS)
    target = out_dir.resolve() / "inputs" / "provider-input.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    srgb_profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    decoded.save(target, format="JPEG", quality=95, subsampling=0, icc_profile=srgb_profile)
    return target, {
        "path": str(target), "sha256": sha256(target), "bytes": target.stat().st_size,
        "format": "JPEG", "mode": "RGB", "bits_per_channel": 8, "size": list(decoded.size),
        "derived": True, "source": original,
        "decode": {"decoder": "rawpy/LibRaw", "white_balance": "camera", "auto_brightness": False,
                   "color_space": "sRGB", "gamma": [1, 1],
                   "source_precision": "raw16_linear", "source_output_bps": 16,
                   "output_transfer": "IEC_sRGB", "decode_version": RGB_DECODE_VERSION,
                   "icc_embedded": "sRGB",
                   "sensor_size": sensor_size, "oriented_size": oriented_size,
                   "libraw_flip": flip, "orientation_applied": flip != 0,
                   "orientation_handled_by": "LibRaw postprocess", "long_edge_limit": long_edge},
    }


def env_or(value: Optional[str], *names: str, default: Optional[str] = None) -> Optional[str]:
    if value:
        return value
    for name in names:
        found = os.environ.get(name)
        if found:
            return found
    return default


def endpoint_for(api_base: Optional[str], api_url: Optional[str]) -> str:
    explicit = env_or(api_url, "COSPLAY_REFERENCE_API_URL")
    if explicit:
        return explicit.rstrip("/")
    base = env_or(api_base, "COSPLAY_REFERENCE_API_BASE", "OPENAI_BASE_URL", default="https://api.openai.com/v1")
    assert base is not None
    return base.rstrip("/") + "/images/edits"


def provider_config(
    *,
    api_base: Optional[str] = None,
    api_url: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    api_key_env: Optional[str] = None,
) -> dict[str, Any]:
    """Resolve transport settings without ever returning the secret itself.

    Both the CLI and localhost backend use this function so a UI dry-run and
    the actual request share one configuration contract.
    """
    settings = provider_settings.load_settings()
    requested_provider = env_or(provider, "COSPLAY_REFERENCE_PROVIDER", default=settings.get("default_provider"))
    discovery = provider_settings.discover_ccswitch()
    resolved_provider = requested_provider or ("ccswitch_sub2api" if discovery["available"] else "openai-compatible")
    entry = _provider_entry(resolved_provider, load_provider_registry())
    registry_id = str(entry.get("id", resolved_provider))
    saved = settings.get("providers", {}).get(registry_id, {})
    endpoint_env_value = entry.get("endpoint_env")
    endpoint_env = str(endpoint_env_value) if endpoint_env_value else None
    key_env = api_key_env or saved.get("secret_env") or str(entry.get("key_env") or "") or None
    # Explicit legacy variables retain precedence over registry defaults.
    kind = str(entry.get("kind") or "openai-compatible")
    model_env = str(entry.get("model_env") or "") or None
    model_names = [name for name in (model_env,) if name]
    if kind not in {"local", "comfyui"}:
        model_names.append("COSPLAY_REFERENCE_MODEL")
    default_model = entry.get("model")
    if kind == "comfyui" and entry.get("workflow_resource"):
        default_model = str(Path(getattr(sys, "_MEIPASS", ROOT)) / entry["workflow_resource"])
    resolved_model = env_or(model, *model_names, default=saved.get("model") or default_model or (None if kind in {"local", "comfyui"} else "gpt-image-1"))
    if api_url:
        resolved_url = endpoint_for(api_base, api_url) if kind == "openai-compatible" else str(api_url).rstrip("/")
    elif api_base:
        resolved_url = endpoint_for(api_base, None) if kind == "openai-compatible" else str(api_base).rstrip("/")
    elif kind == "manual_import":
        resolved_url = None
    elif kind in {"local", "comfyui"}:
        endpoint_value = os.environ.get(endpoint_env, "") if endpoint_env else ""
        resolved_url = (endpoint_value or saved.get("endpoint") or entry.get("endpoint") or "").rstrip("/") or None
    else:
        discovered_base = discovery.get("endpoint") if registry_id == "ccswitch_sub2api" else None
        explicit_url = os.environ.get("COSPLAY_REFERENCE_API_URL")
        base = env_or(None, *(name for name in (endpoint_env, "COSPLAY_REFERENCE_API_BASE", "OPENAI_BASE_URL") if name),
                      default=saved.get("endpoint") or discovered_base or "https://api.openai.com/v1")
        resolved_url = explicit_url or str(base).rstrip("/")
        if not explicit_url and not resolved_url.endswith("/images/edits"):
            resolved_url += "/images/edits"
    contract = adapter_contract(entry)
    credential_source = saved.get("credential_source") or ("auto" if key_env else "none")
    key = provider_settings.resolve_credential(registry_id, key_env, credential_source, resolved_url)
    key_configured = bool(key) if key_env or credential_source in {"ccswitch", "environment"} else True
    # Local remains discoverable by model or endpoint. ComfyUI needs an
    # endpoint and an API-format workflow file for its implemented transport.
    model_configured = bool(resolved_model)
    endpoint_configured = bool(resolved_url)
    input_configured = endpoint_configured or (kind == "local" and model_configured)
    configured = bool(entry.get("enabled", True) and key_configured and
                      (input_configured if kind in {"local", "comfyui"} else True))
    workflow_path = Path(str(resolved_model)).expanduser().resolve() if kind == "comfyui" and resolved_model else None
    workflow_configured = bool(workflow_path and workflow_path.is_file())
    if kind == "comfyui":
        configured = configured and workflow_configured
    return {
        "provider": resolved_provider,
        "registry_id": registry_id,
        "kind": kind,
        "model": resolved_model,
        "endpoint": resolved_url,
        "endpoint_env": endpoint_env,
        "api_key_env": key_env,
        "api_key_configured": key_configured,
        "credential_source": credential_source,
        "capabilities": entry.get("capabilities", []),
        "timeout": entry.get("timeout", 180),
        "retry": {"max_attempts": 1, "backoff_seconds": 0, "retry_statuses": []}
                 if kind == 'openai-compatible' else entry.get("retry", {}),
        "enabled": bool(entry.get("enabled", True)),
        "configured": configured,
        "endpoint_configured": endpoint_configured if kind != "manual_import" else True,
        "workflow_path": str(workflow_path) if workflow_path else None,
        "workflow_configured": workflow_configured if kind == "comfyui" else None,
        **contract,
    }


def multipart_body(image: Path, fields: dict[str, str]) -> tuple[bytes, str]:
    boundary = "----cosplay-" + secrets.token_hex(16)
    chunks: list[bytes] = []
    marker = boundary.encode("ascii")
    for name, value in fields.items():
        chunks += [b"--" + marker + b"\r\n", f'Content-Disposition: form-data; name="{name}"'.encode(), b"\r\n\r\n", value.encode(), b"\r\n"]
    content_type = mimetypes.guess_type(image.name)[0] or "application/octet-stream"
    chunks += [
        b"--" + marker + b"\r\n",
        f'Content-Disposition: form-data; name="image"; filename="{image.name}"'.encode(),
        b"\r\n",
        f"Content-Type: {content_type}".encode(),
        b"\r\n\r\n",
        image.read_bytes(),
        b"\r\n--" + marker + b"--\r\n",
    ]
    return b"".join(chunks), "multipart/form-data; boundary=" + boundary


def request_edit(endpoint: str, key: str, image: Path, model: str, prompt: str,
                 size: Optional[str], timeout: float,
                 retry: Optional[dict[str, Any]] = None,
                 events: Optional[list[dict[str, Any]]] = None) -> tuple[dict[str, Any], bytes, str]:
    fields = {"model": model, "prompt": prompt, "response_format": "b64_json", "n": "1"}
    if size:
        fields["size"] = size
    body, content_type = multipart_body(image, fields)
    request = urllib.request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "Authorization": "Bearer " + key,
            "Content-Type": content_type,
            "Accept": "application/json",
            "User-Agent": "cosplay-reference-provider/1.0",
        },
    )
    # A lost/late image response can still have been billed. Each requested
    # variant gets one submission; further generation is an explicit user retry.
    # Older saved provider registries must not silently raise that spend limit.
    events = events if events is not None else []
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            response_type = response.headers.get_content_type()
        events.append({"event": "provider_attempt", "attempt": 1, "status": "ok"})
    except urllib.error.HTTPError as exc:
        detail = exc.read(1200).decode("utf-8", errors="replace").replace(key, "<redacted>")
        events.append({"event": "provider_attempt", "attempt": 1, "status": "error",
                       "http_status": int(exc.code), "retrying": False})
        raise RuntimeError(f"reference provider HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", None) or str(exc)
        events.append({"event": "provider_attempt", "attempt": 1, "status": "error",
                       "error": "connection_or_timeout", "retrying": False})
        raise RuntimeError(f"reference provider connection failed: {reason}") from exc
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"reference provider returned non-JSON ({response_type})") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("reference provider response is not an object")
    data = payload.get("data")
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        raise RuntimeError("reference provider response has no data[0] image")
    item = data[0]
    if item.get("b64_json"):
        try:
            encoded = str(item["b64_json"])
            if encoded.startswith("data:") and "," in encoded:
                encoded = encoded.split(",", 1)[1]
            return payload, base64.b64decode(encoded, validate=True), "b64_json"
        except (ValueError, base64.binascii.Error) as exc:
            raise RuntimeError("reference provider returned invalid b64_json") from exc
    if item.get("url"):
        url = str(item["url"])
        try:
            # Do not forward the API key to the returned URL; normal providers
            # return a presigned URL and compatible gateways should do the same.
            with urllib.request.urlopen(urllib.request.Request(url, headers={"Accept": "image/*"}), timeout=timeout) as image_response:
                return payload, image_response.read(), "url"
        except urllib.error.URLError as exc:
            raise RuntimeError(f"could not download provider image URL: {exc.reason}") from exc
    raise RuntimeError("reference provider data[0] has neither b64_json nor url")


def write_candidate(image_bytes: bytes) -> Path:
    handle = tempfile.NamedTemporaryFile(prefix="cosplay-reference-", suffix=".png", delete=False)
    path = Path(handle.name)
    try:
        handle.write(image_bytes)
    finally:
        handle.close()
    return path


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # Health/dry-run are configuration-only commands; image/sample/prompt are
    # required only for an actual generation request.
    parser.add_argument("--image", type=Path, help="source image sent to /images/edits")
    parser.add_argument("--inputs-json", type=Path, help="ordered, SHA-bound Qwen condition images")
    parser.add_argument("--preserve-alpha", action="store_true", help="retain actual generated alpha")
    parser.add_argument("--asset-role", choices=("design_reference", "effect_layer"), default="design_reference")
    parser.add_argument("--sample", help="sample ID for the reference candidate")
    parser.add_argument("--prompt", help="exact generation prompt")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out-dir", type=Path, help="candidate output directory; defaults to a new run-local directory")
    parser.add_argument("--source", type=Path, help="source path recorded in provenance; defaults to --image")
    parser.add_argument("--model", help="model name, or ComfyUI API workflow JSON path")
    parser.add_argument("--provider", help="provider label; defaults to COSPLAY_REFERENCE_PROVIDER")
    parser.add_argument("--api-base", help="OpenAI-compatible base URL, e.g. https://host/v1")
    parser.add_argument("--api-url", help="full edits endpoint; overrides --api-base")
    parser.add_argument("--api-key-env", default=None, help="environment variable containing the API key")
    parser.add_argument("--size", default=None, help="optional API image size")
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--generation-id", help="optional external generation/job ID")
    parser.add_argument("--force", action="store_true", help="replace an existing registered candidate")
    parser.add_argument("--dry-run", action="store_true", help="validate config and print endpoint without making a request")
    parser.add_argument("--health", action="store_true", help="report local provider readiness without making a request")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    root = args.root.expanduser().resolve()
    image = None
    if args.image:
        image = args.image.expanduser()
        image = image.resolve() if image.is_absolute() else (root / image).resolve()
    config = provider_config(
        api_base=args.api_base,
        api_url=args.api_url,
        model=args.model,
        provider=args.provider,
        api_key_env=args.api_key_env,
    )
    endpoint = config["endpoint"]
    model = str(config["model"])
    provider = str(config["provider"])
    assert model is not None and provider is not None
    key_env = config.get("api_key_env")
    key = provider_settings.resolve_credential(str(config["registry_id"]), key_env,
                                               str(config["credential_source"]), endpoint)
    if args.health:
        print(json.dumps(provider_health(provider=args.provider, api_base=args.api_base,
                                         api_url=args.api_url, model=args.model,
                                         api_key_env=args.api_key_env), ensure_ascii=False, indent=2))
        return 0
    if args.dry_run:
        print(json.dumps({**config, "health": provider_health(provider=args.provider, api_base=args.api_base,
                         api_url=args.api_url, model=args.model, api_key_env=args.api_key_env),
                         "input": str(image) if image else None, "would_register_reference": bool(image),
                         "would_modify_dataset_lock": False, "would_create_manual_final": False},
                         ensure_ascii=False, indent=2))
        return 0
    if image is None or not image.is_file():
        raise FileNotFoundError(image or "--image is required for generation")
    if not args.sample or not args.prompt:
        raise ValueError("--sample and --prompt are required for generation")
    if not config.get("generation_supported"):
        status = provider_health(provider=args.provider, api_base=args.api_base,
                                 api_url=args.api_url, model=args.model,
                                 api_key_env=args.api_key_env).get("status")
        raise RuntimeError(
            f"provider adapter {config.get('adapter')} for {config.get('provider')} "
            f"does not support generation yet (status={status}); use dry-run/health "
            "or configure an implemented provider"
        )
    if not config.get("enabled"):
        raise RuntimeError("provider_disabled")
    if not config.get("api_key_configured"):
        raise RuntimeError(f"missing API credential: source={config['credential_source']}, environment={key_env or 'none'}")
    timeout = float(args.timeout if args.timeout is not None else (config.get("timeout") or 180.0))
    if timeout <= 0:
        raise ValueError("--timeout must be positive")
    if not endpoint:
        raise RuntimeError(f"provider kind {config.get('kind')} has no image generation endpoint")
    if config.get("kind") == "comfyui" and not config.get("workflow_configured"):
        raise RuntimeError("missing_workflow: set --model or COSPLAY_COMFYUI_WORKFLOW to an API workflow JSON file")
    requested_out_dir = args.out_dir or Path("outputs/runs") / ("provider-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")) / "reference"
    out_dir = requested_out_dir if requested_out_dir.is_absolute() else root / requested_out_dir
    if not args.force and any((out_dir / (args.sample + suffix)).exists()
                              for suffix in (".png", ".provenance.json")):
        raise FileExistsError("Candidate already exists; refusing a new provider submission")
    source = args.source.expanduser() if args.source else image
    source = source.resolve() if source.is_absolute() else (root / source).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    api_image, api_input = prepare_api_input(image, out_dir)
    conditions = None
    if args.inputs_json:
        if config.get("kind") != "comfyui":
            raise ValueError("Explicit multi-image conditions currently require the ComfyUI Qwen adapter")
        manifest = args.inputs_json.expanduser()
        manifest = manifest.resolve() if manifest.is_absolute() else (root / manifest).resolve()
        conditions = prepare_condition_inputs(manifest, image, root, out_dir)
        comfyui_adapter.prepare_condition_workflow(
            comfyui_adapter.load_workflow(Path(config['workflow_path'])),
            ['pending-'+str(index)+'.png' for index in range(len(conditions))], args.prompt)
    if args.asset_role == "effect_layer" and not args.preserve_alpha:
        raise ValueError("effect_layer requires explicit --preserve-alpha")
    transport_events: list[dict[str, Any]] = []
    try:
        if config.get("kind") == "comfyui":
            if config["registry_id"] in {"qwen_image_2_1", "qwen_image_2_1_uncensored", "qwen_image_2_1_fast", "qwen_image_2_1_uncensored_fast"}:
                from start_qwen_image import ensure_running
                ensure_running(config["registry_id"], str(endpoint))
            extra = {"condition_inputs": [{**row, "path": row["api_input"]["path"],
                     "sha256": row["api_input"]["sha256"]} for row in conditions]} if conditions is not None else {}
            payload, output_bytes, response_mode = comfyui_adapter.request_generation(
                str(endpoint), api_image, Path(config["workflow_path"]), args.prompt, timeout, transport_events, key=key, **extra)
        else:
            payload, output_bytes, response_mode = request_edit(
                str(endpoint), key, api_image, model, args.prompt, args.size, timeout,
                retry=config.get("retry"), events=transport_events)
    except Exception as error:
        # Preserve a successful submission even when the service crashes before
        # /view. This diagnostic never grants an image retry or acceptance.
        out_dir.mkdir(parents=True, exist_ok=True)
        failure = {"schema": "reference_transport_failure.v1", "error_type": type(error).__name__,
                   "provider": config['registry_id'], "endpoint": endpoint,
                   "api_input": api_input, "condition_inputs": conditions,
                   "events": transport_events, "automatic_retry": False,
                   "candidate_registered": False,
                   "recorded_at": datetime.now(timezone.utc).isoformat()}
        failure_path = out_dir / (args.sample + '.transport-failure.json')
        with failure_path.open('x', encoding='utf8') as stream:
            json.dump(failure, stream, ensure_ascii=False, indent=2)
        raise
    if not output_bytes:
        raise RuntimeError("reference provider returned an empty image")
    for row in conditions or []:
        if sha256(Path(row["api_input"]["path"])) != row["input_sha256"]:
            raise ValueError("Condition image changed during generation; candidate not registered")
    temporary = write_candidate(output_bytes)
    generated_model = str((payload.get("comfyui") or {}).get("model_name") or model)
    try:
        register_args = [
            "--root", str(root), "--image", str(temporary), "--sample", args.sample,
            "--out-dir", str(out_dir), "--model", generated_model, "--provider", provider,
            "--prompt", args.prompt, "--source", str(source),
        ]
        generation_id = payload.get("id") if config.get("kind") == "comfyui" else (args.generation_id or payload.get("id"))
        if generation_id:
            register_args += ["--generation-id", str(generation_id)]
        if args.force:
            register_args.append("--force")
        if args.preserve_alpha:
            register_args.append("--preserve-alpha")
        register_args += ["--asset-role", args.asset_role]
        code = register_reference.main(register_args)
        provenance_path = out_dir.resolve() / f"{args.sample}.provenance.json"
        if provenance_path.is_file():
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            provenance["provider_contract"] = {
                "schema_version": "1.0",
                "protocol": config.get("protocol") or "openai-compatible",
                "endpoint": endpoint,
                "response_mode": response_mode,
                "request": {
                    "model": generated_model,
                    "prompt": args.prompt,
                    "size": args.size,
                    "input_sha256": api_input["sha256"],
                    "api_input": api_input,
                },
                "response": {
                    "id": payload.get("id"),
                    "created": payload.get("created"),
                    "output_sha256": hashlib.sha256(output_bytes).hexdigest(),
                },
                "transport": {
                    "timeout_seconds": timeout,
                    "retry": config.get("retry") or {},
                    "events": transport_events,
                },
                "policy": {
                    "manual_final_required": True,
                    "dataset_lock_unchanged": True,
                    "human_confirmation_required": True,
                },
                "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            if payload.get("comfyui"):
                provenance["provider_contract"]["comfyui"] = payload["comfyui"]
            if conditions is not None:
                request = provenance["provider_contract"]["request"]
                request["condition_inputs"] = conditions
                request["edit_mask"] = next((row for row in conditions if row["role"] == "edit_mask"), None)
            provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return int(code or 0)
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"reference_provider failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
