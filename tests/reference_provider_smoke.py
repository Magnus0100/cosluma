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
import provider_settings


def png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), (120, 80, 200)).save(buffer, format="PNG")
    return buffer.getvalue()


def main() -> int:
    previous_key = os.environ.get("COSPLAY_REFERENCE_API_KEY")
    previous_local_base = os.environ.get("COSPLAY_LOCAL_API_BASE")
    previous_settings = os.environ.get("COSPLAY_PROVIDER_SETTINGS")
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
    settings_file = Path(tempfile.mkdtemp(prefix="cosplay_provider_settings_smoke_")) / "provider-settings.json"
    provider_settings.save_settings({"providers": {"ccswitch_sub2api": {
        "model": "gpt-image-2.5-sunburst", "credential_source": "environment",
    }}}, settings_file)
    os.environ["COSPLAY_PROVIDER_SETTINGS"] = str(settings_file)
    os.environ["COSPLAY_REFERENCE_API_KEY"] = "smoke-secret"
    assert "smoke-secret" not in json.dumps(registry)
    health = reference_provider.provider_health(provider="CCSwitch/Sub2API")
    assert health["ready"] is True
    assert health["network_checked"] is False
    ccswitch = reference_provider.provider_config(provider="CCSwitch/Sub2API")
    assert ccswitch["registry_id"] == "ccswitch_sub2api"
    assert ccswitch["model"] == "gpt-image-2.5-sunburst"
    assert ccswitch["timeout"] == 900
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
            if attempts["count"] <= 2:
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error":"transient offline smoke offline-smoke-key"}')
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
                assert 'offline-smoke-key' not in str(exc)
                assert '<redacted>' in str(exc)
            else:
                raise AssertionError('HTTP failure must be returned without automatic paid retry')
            assert attempts['count'] == 1
            failure = json.loads((out_dir / 'SMOKE.transport-failure.json').read_text('utf8'))
            assert failure['error_chain'][0]['type'] == 'RuntimeError'
            assert failure['events'][-1]['http_status'] == 503
            assert failure['events'][-1]['elapsed_seconds'] >= 0
            # A stale failure record must not mask the next real HTTP failure.
            try:
                reference_provider.main(call)
            except RuntimeError as exc:
                assert '503' in str(exc)
            else:
                raise AssertionError('Repeated HTTP failure must remain readable')
            assert attempts['count'] == 2
            assert len(list(out_dir.glob('SMOKE.transport-*.previous-failure.json'))) == 1
            # The next call is an explicit new invocation, not transport retry.
            code = reference_provider.main(call)
            provenance = json.loads((out_dir / "SMOKE.provenance.json").read_text(encoding="utf-8"))
            assert code == 0
            assert observed["authorization"] == "Bearer offline-smoke-key"
            assert "multipart/form-data" in observed["content_type"]
            assert "gpt-image-2.5-sunburst" in observed["body"]
            assert attempts["count"] == 3
            assert provenance["provider_contract"]["response"]["id"] == "offline-smoke-generation"
            transport_events = provenance["provider_contract"]["transport"]["events"]
            assert not any(event.get("event") == "provider_retry" for event in transport_events)
            assert sum(event.get('event') == 'provider_attempt' for event in transport_events) == 1
            request = next(row for row in transport_events if row['event'] == 'provider_request')
            assert request['request_bytes'] == len(observed['body'].encode('latin1'))
            assert request['image_bytes'] == len(output)
            assert request['encoding'] == 'multipart_binary'
            assert len(request['image_sha256']) == 64
            assert not (out_dir / 'SMOKE.transport-failure.json').exists()
            history = [json.loads(path.read_text('utf8')) for path in out_dir.glob('SMOKE.transport-*.json')
                       if not path.name.endswith('.previous-failure.json')]
            assert sorted(row['state'] for row in history) == ['failed', 'failed', 'registered']
            assert 'offline-smoke-key' not in json.dumps(history)
            try:
                reference_provider.main(call)
            except FileExistsError:
                pass
            else:
                raise AssertionError('Registered candidates must still prevent a new submission')
            assert attempts['count'] == 3
            assert provenance["status"] == "reference_candidate"
            print(json.dumps({"ok": True, "candidate": str(out_dir / "SMOKE.png"), "provenance": str(out_dir / "SMOKE.provenance.json")}, ensure_ascii=False, indent=2))
    finally:
        server.shutdown()
        server.server_close()
        if previous_key is None:
            os.environ.pop("COSPLAY_REFERENCE_API_KEY", None)
        else:
            os.environ["COSPLAY_REFERENCE_API_KEY"] = previous_key
        if previous_settings is None:
            os.environ.pop("COSPLAY_PROVIDER_SETTINGS", None)
        else:
            os.environ["COSPLAY_PROVIDER_SETTINGS"] = previous_settings
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
