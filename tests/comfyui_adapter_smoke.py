"""Offline end-to-end ComfyUI adapter smoke with the real HTTP protocol."""
from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import parse_qs, urlsplit

from PIL import Image

import comfyui_adapter
import reference_provider
import single_source_workflow


def main() -> int:
    buffer = io.BytesIO()
    Image.new("RGB", (96, 128), (128, 60, 180)).save(buffer, format="PNG")
    output = buffer.getvalue()
    state = {"mode": "success", "polls": 0, "calls": [], "submitted": None}

    class Handler(BaseHTTPRequestHandler):
        def send(self, body: bytes, status: int = 200, content_type: str = "application/json") -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            state["calls"].append(self.path)
            if self.path == "/upload/image":
                assert "multipart/form-data" in self.headers["Content-Type"]
                assert output in body
                assert self.headers.get("Authorization") is None
                self.send(b'{"name":"uploaded.png","subfolder":"sources","type":"input"}')
            elif self.path == "/prompt":
                state["submitted"] = json.loads(body)
                self.send(b'{"prompt_id":"actual-job-id","node_errors":{}}')
            else:
                self.send(b"{}", 404)

        def do_GET(self) -> None:  # noqa: N802
            state["calls"].append(self.path)
            if self.path == "/history/actual-job-id":
                state["polls"] += 1
                if state["mode"] == "timeout" or state["polls"] == 1:
                    self.send(b"{}")
                elif state["mode"] == "error":
                    self.send(b'{"actual-job-id":{"status":{"status_str":"error","messages":[["execution_error",{"exception_message":"missing checkpoint"}]]}}}')
                else:
                    self.send(b'{"actual-job-id":{"status":{"completed":true,"status_str":"success"},"outputs":{"3":{"images":[{"filename":"result name.png","subfolder":"demo","type":"output"}]}}}}')
            elif self.path.startswith("/view?"):
                query = parse_qs(urlsplit(self.path).query)
                assert query == {"filename": ["result name.png"], "subfolder": ["demo"], "type": ["output"]}
                self.send(output, content_type="image/png")
            else:
                self.send(b"{}", 404)

        def log_message(self, _format: str, *_args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    Thread(target=server.serve_forever, daemon=True).start()
    endpoint = f"http://127.0.0.1:{server.server_port}"
    previous = {name: os.environ.pop(name, None) for name in ("COSPLAY_COMFYUI_API_BASE", "COSPLAY_COMFYUI_WORKFLOW")}
    try:
        with tempfile.TemporaryDirectory(prefix="cosplay_comfyui_smoke_") as temp:
            root = Path(temp)
            image = root / "source.png"
            image.write_bytes(output)
            workflow = root / "workflow.json"
            graph = {"1": {"class_type": "LoadImage", "inputs": {"image": "{{source}}"}},
                     "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "{{prompt}}"}},
                     "3": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}}}
            workflow.write_text(json.dumps({"workflow": graph, "output_node": "3"}), encoding="utf-8")
            assert reference_provider.provider_health(provider="comfyui")["status"] == "missing_endpoint"
            assert reference_provider.provider_health(provider="comfyui", api_base=endpoint)["status"] == "missing_workflow"
            health = reference_provider.provider_health(provider="comfyui", api_base=endpoint, model=str(workflow))
            assert health["ready"] and health["generation_supported"] and not health["network_checked"]
            prompt = 'soft violet light and preserve source pose "test"'
            args = ["--root", str(root), "--image", str(image), "--sample", "COMFY",
                    "--prompt", prompt, "--provider", "comfyui", "--api-base", endpoint,
                    "--model", str(workflow), "--timeout", "3"]
            assert reference_provider.main(args) == 0
            candidates = list(root.glob("outputs/runs/*/reference/COMFY.png"))
            assert len(candidates) == 1 and not (root / "data/reference").exists()
            provenance = json.loads(candidates[0].with_suffix(".provenance.json").read_text(encoding="utf-8"))
            contract = provenance["provider_contract"]
            assert provenance["generation_id"] == "actual-job-id"
            assert contract["response_mode"] == "comfyui_view"
            assert contract["response"]["output_sha256"] == hashlib.sha256(output).hexdigest()
            assert contract["comfyui"]["workflow_sha256"] == reference_provider.sha256(workflow)
            assert contract["comfyui"]["prompt_id"] == "actual-job-id"
            submitted = state["submitted"]["prompt"]
            assert submitted["1"]["inputs"]["image"] == "sources/uploaded.png"
            assert submitted["2"]["inputs"]["text"] == prompt
            assert state["calls"][:2] == ["/upload/image", "/prompt"]
            assert any(path.startswith("/view?") for path in state["calls"])
            # Plain API exports can instead use explicit node/input bindings.
            wrapper = {"workflow": graph, "bindings": {"source": {"node": "1", "input": "image"},
                                                       "prompt": {"node": "2", "input": "text"}}}
            assert comfyui_adapter.prepare_workflow(wrapper, "bound.png", "bound prompt")["1"]["inputs"]["image"] == "bound.png"
            # The low-level worker must reject unchecked generation before
            # model preparation or adapter transport. Provider protocol coverage
            # above and below uses only this isolated HTTP fixture.
            calls_before = list(state['calls'])
            try:
                single_source_workflow.create_run(single_source_workflow.parse_args([
                    "--root", str(root), "--source", str(image), "--provider", "comfyui",
                    "--generate", "--run-id", "comfy-source",
                ]))
            except ValueError as exc:
                assert 'checked scoped task' in str(exc)
            else:
                raise AssertionError('Unchecked generation entered production')
            assert state['calls'] == calls_before
            assert not (root/'outputs/runs/comfy-source').exists()
            for mode in ("error", "timeout"):
                state.update(mode=mode, polls=0)
                failed_out = root / "outputs/runs" / mode / "reference"
                try:
                    reference_provider.main(args[:-2] + ["--timeout", "0.08" if mode == "timeout" else "3",
                                                       "--out-dir", str(failed_out)])
                except (RuntimeError, TimeoutError) as exc:
                    assert "ComfyUI" in str(exc)
                else:
                    raise AssertionError(f"{mode} must not register a candidate")
                assert not (failed_out/'COMFY.png').exists()
            print(json.dumps({"ok": True, "protocol": "upload/prompt/history/view", "generation_id": "actual-job-id",
                              "run_local": True, "unchecked_generation_rejected": True,
                              "failure_and_timeout_no_candidate": True}, indent=2))
    finally:
        server.shutdown()
        server.server_close()
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
