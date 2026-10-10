"""Offline HTTP integration checks for local Qwen quality/speed profiles."""
from __future__ import annotations

import io
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
from threading import Thread
from unittest.mock import patch

from PIL import Image

import reference_provider
import vision_transport
import start_qwen_image  # Initialize the local HTTP opener before the isolated environment.


def main() -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 96), "steelblue").save(buffer, "PNG")
    output = buffer.getvalue()
    graphs = []

    class Handler(BaseHTTPRequestHandler):
        def send(self, body: bytes) -> None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            assert self.headers.get("Authorization") is None
            if self.path == "/upload/image":
                assert output in body
                self.send(b'{"name":"actual-source.png","type":"input"}')
            elif self.path == "/prompt":
                graphs.append(json.loads(body)["prompt"])
                self.send(json.dumps({"prompt_id": f"job-{len(graphs)}", "node_errors": {}}).encode())
            else:
                raise AssertionError(self.path)

        def do_GET(self):
            if self.path.startswith("/history/"):
                job_id = self.path.rsplit("/", 1)[-1]
                self.send(json.dumps({job_id: {"status": {"completed": True},
                    "outputs": {"8": {"images": [{"filename": "result.png", "type": "output"}]}}}}).encode())
            elif self.path.startswith("/view?"):
                self.send(output)
            else:
                raise AssertionError(self.path)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    Thread(target=server.serve_forever, daemon=True).start()
    endpoint = f"http://127.0.0.1:{server.server_port}"
    try:
        with tempfile.TemporaryDirectory(prefix="cosplay_qwen_smoke_") as temp:
            root = Path(temp)
            source = root / "source.png"
            source.write_bytes(output)
            environment = {"COSPLAY_PROVIDER_SETTINGS": str(root / "settings.json"), "USERPROFILE": str(root),
                           "COSPLAY_REFERENCE_MODEL": "must-not-leak-gpt-model"}
            with patch.dict(os.environ, environment, clear=True):
                for provider, loader, model in (
                    ("qwen_image_2_1", "UNETLoader", "Qwen-Image-2.1 BF16"),
                    ("qwen_image_2_1_uncensored", "UnetLoaderGGUF", "abenzerps/Qwen-Image-2.1-Uncensored-GGUF BF16"),
                    ("qwen_image_2_1_fast", "UNETLoader", "Qwen-Image-2.1 INT8 ConvRot"),
                    ("qwen_image_2_1_uncensored_fast", "UnetLoaderGGUF", "abenzerps/Qwen-Image-2.1-Uncensored-GGUF Q6_K")):
                    config = reference_provider.provider_config(provider=provider)
                    assert config["endpoint"] == "http://127.0.0.1:8188"
                    assert config["workflow_configured"] and config["credential_source"] == "none"
                    assert "must-not-leak" not in config["model"]
                    destination = root / provider
                    reference_provider.main(["--root", str(root), "--image", str(source),
                        "--sample", "PROBE", "--provider", provider, "--api-base", endpoint,
                        "--prompt", "cool blue light", "--out-dir", str(destination)])
                    provenance = json.loads((destination / "PROBE.provenance.json").read_text(encoding="utf-8"))
                    assert provenance["provider"] == provider and provenance["model"] == model
                    assert provenance["generation_id"] == f"job-{len(graphs)}"
                    contract = provenance["provider_contract"]
                    assert contract["comfyui"]["model_name"] == model
                    assert contract["request"]["model"] == model
                    graph = graphs[-1]
                    assert graph["1"]["class_type"] == loader
                    expected_precision = "bf16" if not provider.endswith("_fast") else "int8_convrot"
                    assert expected_precision in graph["2"]["inputs"]["clip_name"]
                    required = {e["role"] for e in start_qwen_image.model_files(provider)}
                    assert len(required) == 3 and "vae" in required
                    assert ("fast_text_encoder" in required) == provider.endswith("_fast")
                    assert graph["4"]["inputs"]["image"] == "actual-source.png"
                    assert graph["5"]["inputs"]["prompt"] == "cool blue light"
                    assert graph["5"]["inputs"]["images.image_1"] == ["4", 0]
                    assert graph["6"]["inputs"]["seed"] != 42
                assert graphs[0]["6"]["inputs"]["seed"] != graphs[1]["6"]["inputs"]["seed"]
                # A local image choice must still route textual planning to
                # its own configured vision host, including explicit overrides.
                with patch.dict(os.environ, {"COSPLAY_VISION_PROVIDER": "openai-compatible",
                        "COSPLAY_VISION_API_BASE": "https://vision.example/v1", "OPENAI_API_KEY": "test-key"}):
                    config, base, key, _ = vision_transport._connection(provider="qwen_image_2_1_uncensored", api_base=endpoint)
                    assert config["kind"] == "openai-compatible" and base == "https://vision.example/v1"
                    assert key == "test-key" and endpoint not in base
            print(json.dumps({"ok": True, "providers": 4, "source_bound": True,
                "model_provenance": True, "random_seeds": True, "separate_vision_transport": True}))
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
