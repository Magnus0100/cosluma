#!/usr/bin/env python3
"""Offline smoke for the OpenAI-compatible reference provider adapter."""

from __future__ import annotations

import base64
import io
import json
import os
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from PIL import Image

import reference_provider


def png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), (120, 80, 200)).save(buffer, format="PNG")
    return buffer.getvalue()


def main() -> int:
    previous_key = os.environ.get("COSPLAY_REFERENCE_API_KEY")
    previous_local_base = os.environ.get("COSPLAY_LOCAL_API_BASE")
    registry = reference_provider.provider_registry()
    ids = {entry["id"] for entry in registry}
    assert {"openai-compatible", "ccswitch_sub2api", "manual_import", "local", "comfyui"} <= ids
    local = reference_provider.provider_health(provider="local")
    assert local["status"] == "unsupported"
    assert local["generation_supported"] is False
    os.environ["COSPLAY_LOCAL_API_BASE"] = "http://127.0.0.1:8000"
    configured_local = reference_provider.provider_health(provider="local")
    assert configured_local["status"] == "configured"
    assert configured_local["ready"] is False
    if previous_local_base is None:
        os.environ.pop("COSPLAY_LOCAL_API_BASE", None)
    else:
        os.environ["COSPLAY_LOCAL_API_BASE"] = previous_local_base
    # Registry status exposes only booleans and env-var names, never secret values.
    os.environ["COSPLAY_REFERENCE_API_KEY"] = "smoke-secret"
    assert "smoke-secret" not in json.dumps(registry)
    health = reference_provider.provider_health(provider="CCSwitch/Sub2API")
    assert health["ready"] is True
    assert health["network_checked"] is False
    ccswitch = reference_provider.provider_config(provider="CCSwitch/Sub2API")
    assert ccswitch["registry_id"] == "ccswitch_sub2api"
    assert ccswitch["model"] == "gpt-image-2.5-sunburst"
    output = png_bytes()
    observed: dict[str, str] = {}
    attempts = {"count": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            attempts["count"] += 1
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            observed["authorization"] = self.headers.get("Authorization", "")
            observed["content_type"] = self.headers.get("Content-Type", "")
            observed["body"] = body.decode("latin1")
            if attempts["count"] == 1:
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error":"transient offline smoke"}')
                return
            payload = {"id": "offline-smoke-generation", "created": 1, "data": [{"b64_json": base64.b64encode(output).decode("ascii")}]}
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    os.environ["COSPLAY_REFERENCE_API_KEY"] = "offline-smoke-key"
    try:
        with tempfile.TemporaryDirectory(prefix="cosplay_provider_smoke_") as temp:
            root = Path(temp)
            source = root / "source.png"
            source.write_bytes(output)
            out_dir = root / "data" / "reference"
            endpoint = f"http://127.0.0.1:{server.server_port}/v1/images/edits"
            call = [
                "--root", str(root), "--image", str(source), "--sample", "SMOKE",
                "--prompt", "offline smoke prompt", "--api-url", endpoint,
                "--model", "gpt-image-2.5-sunburst", "--provider", "CCSwitch/Sub2API",
                "--out-dir", str(out_dir), "--generation-id", "offline-smoke-generation",
            ]
            try:
                reference_provider.main(call)
            except RuntimeError as exc:
                assert '503' in str(exc)
            else:
                raise AssertionError('HTTP failure must be returned without automatic paid retry')
            assert attempts['count'] == 1
            # The next call is an explicit new invocation, not transport retry.
            code = reference_provider.main(call)
            provenance = json.loads((out_dir / "SMOKE.provenance.json").read_text(encoding="utf-8"))
            assert code == 0
            assert observed["authorization"] == "Bearer offline-smoke-key"
            assert "multipart/form-data" in observed["content_type"]
            assert "gpt-image-2.5-sunburst" in observed["body"]
            assert attempts["count"] == 2
            assert provenance["provider_contract"]["response"]["id"] == "offline-smoke-generation"
            transport_events = provenance["provider_contract"]["transport"]["events"]
            assert not any(event.get("event") == "provider_retry" for event in transport_events)
            assert sum(event.get('event') == 'provider_attempt' for event in transport_events) == 1
            assert provenance["status"] == "reference_candidate"
            print(json.dumps({"ok": True, "candidate": str(out_dir / "SMOKE.png"), "provenance": str(out_dir / "SMOKE.provenance.json")}, ensure_ascii=False, indent=2))
    finally:
        server.shutdown()
        server.server_close()
        if previous_key is None:
            os.environ.pop("COSPLAY_REFERENCE_API_KEY", None)
        else:
            os.environ["COSPLAY_REFERENCE_API_KEY"] = previous_key
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
