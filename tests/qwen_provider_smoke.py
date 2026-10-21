"""Offline HTTP integration checks for local Qwen quality/speed profiles."""
from __future__ import annotations

import io
import json
import os
import re
from contextlib import nullcontext
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
from threading import Thread
from unittest.mock import patch

from PIL import Image
import numpy as np

import reference_provider
import vision_transport
import start_qwen_image  # Initialize the local HTTP opener before the isolated environment.


def main() -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 96), "steelblue").save(buffer, "PNG")
    output = buffer.getvalue()
    graphs = []
    uploads = []
    failures = {'enabled':False}

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
                assert source.read_bytes() in body
                uploads.append(re.search(br'filename="([^"]+)"',body).group(1).decode())
                self.send(b'{"name":"actual-source.png","type":"input"}')
            elif self.path == "/prompt":
                graphs.append(json.loads(body)["prompt"])
                self.send(json.dumps({"prompt_id": f"job-{len(graphs)}", "node_errors": {}}).encode())
            else:
                raise AssertionError(self.path)

        def do_GET(self):
            if self.path.startswith("/history/"):
                job_id = self.path.rsplit("/", 1)[-1]
                if failures['enabled']:
                    self.send(json.dumps({job_id:{'status':{'status_str':'error','messages':[
                        ['execution_error',{'exception_message':'hostbuf_file_reader_read failed'}]]}}}).encode())
                    return
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
            # RAW derivatives must keep canonical ICC metadata; its creation
            # timestamp must not invalidate identical model conditioning.
            camera=SimpleNamespace(sizes=SimpleNamespace(width=64,height=96,flip=0))
            raw=SimpleNamespace(imread=lambda _:nullcontext(camera))
            with (patch.object(reference_provider,'is_raw',return_value=True),
                  patch.object(reference_provider,'_require_rawpy',return_value=raw),
                  patch.object(reference_provider,'open_rgb',side_effect=lambda _:Image.open(source).convert('RGB'))):
                first,first_info=reference_provider.prepare_api_input(source,root/'raw-a')
                second,second_info=reference_provider.prepare_api_input(source,root/'raw-b')
                assert first_info['sha256']==second_info['sha256'] and first.read_bytes()==second.read_bytes()
                with Image.open(first) as uploaded:
                    assert uploaded.info['icc_profile'][24:36]==bytes.fromhex('07d000010001000000000000')
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
                    submitted_prompt = ("\n  Create a complete cinematic cosplay photograph; " +
                        ("preserve the existing faint blue haze." if provider.endswith('_fast')
                         else "use the character's amber palette in the existing station.") + "\n")
                    reference_provider.main(["--root", str(root), "--image", str(source),
                        "--sample", "PROBE", "--provider", provider, "--api-base", endpoint,
                        "--prompt", submitted_prompt, "--out-dir", str(destination)])
                    provenance = json.loads((destination / "PROBE.provenance.json").read_text(encoding="utf-8"))
                    assert provenance["provider"] == provider and provenance["model"] == model
                    assert provenance["generation_id"] == f"job-{len(graphs)}"
                    contract = provenance["provider_contract"]
                    assert Path(contract['quality']['metrics']['path']).parent == destination.resolve()
                    assert not list(destination.glob('cosplay-reference-*.png'))
                    timings=contract['comfyui']['timings']
                    assert len(timings['uploads_seconds'])==1 and timings['total_seconds']>0
                    assert contract["comfyui"]["model_name"] == model
                    assert contract["request"]["model"] == model
                    assert contract["request"]["prompt"] == submitted_prompt
                    graph = graphs[-1]
                    assert graph["1"]["class_type"] == loader
                    expected_precision = "bf16" if not provider.endswith("_fast") else "int8_convrot"
                    assert expected_precision in graph["2"]["inputs"]["clip_name"]
                    required = {e["role"] for e in start_qwen_image.model_files(provider)}
                    assert len(required) == 3 and "vae" in required
                    assert ("fast_text_encoder" in required) == provider.endswith("_fast")
                    assert graph["4"]["inputs"]["image"] == "actual-source.png"
                    assert graph["5"]["inputs"]["prompt"] == submitted_prompt
                    assert graph['5']['class_type'] == 'CosplayTextEncodeQwenImage21'
                    assert graph['5']['inputs']['cfg'] == 3.0
                    assert graph['5']['inputs']['negative_prompt'] == ''
                    assert graph['5']['inputs']['resolution'] == 2048
                    assert graph['6']['inputs']['steps'] == 40
                    assert graph['6']['inputs']['sampler_name'] == 'euler'
                    assert graph['6']['inputs']['scheduler'] == 'simple'
                    assert graph['6']['inputs']['cfg'] == ['5',3]
                    assert graph["5"]["inputs"]["images.image_1"] == ["4", 0]
                    assert graph["6"]["inputs"]["seed"] != 42
                    # The encoder returns an empty target latent; the source
                    # enters reference conditioning. Partial denoise on that
                    # empty target is not source-preserving image-to-image.
                    assert graph["6"]["inputs"]["latent_image"] == ["5", 2]
                    assert graph["6"]["inputs"]["denoise"] == 1.0
                    # Preserve raw model pixels; no experimental image filter
                    # may silently rewrite a clean or finely textured result.
                    assert graph["7"]["class_type"] == "VAEDecode"
                    assert graph['3']['inputs']['vae_name'] == 'texture_fix_vae_for_qwen_image_2.1_bf16.safetensors'
                    assert not any(row['class_type'] == 'VAEDeGrid' for row in graph.values())
                    assert graph["8"]["class_type"] == "SaveImage"
                    assert graph["8"]["inputs"]["images"] == ["7", 0]
                assert graphs[0]["6"]["inputs"]["seed"] != graphs[1]["6"]["inputs"]["seed"]
                assert len(set(uploads))==1, 'Identical bytes must retain the same upload name'
                # Same filesystem path with changed content must invalidate the
                # input identity, and a new prompt must reach the actual graph.
                original_name=uploads[0]
                Image.new('RGB',(64,96),'red').save(source,'PNG')
                reference_provider.main(['--root',str(root),'--image',str(source),'--sample','CHANGED',
                    '--provider','qwen_image_2_1_fast','--api-base',endpoint,'--prompt','changed lighting',
                    '--out-dir',str(root/'changed')])
                assert uploads[-1]!=original_name and graphs[-1]['5']['inputs']['prompt']=='changed lighting'
                # A high input grid score can be ordinary fabric or scene
                # texture. Preserve its diagnostic without blocking a request.
                yy, xx = np.indices((192, 192))
                lattice = ((xx + yy) % 8 < 3).astype(np.uint8) * 255
                Image.fromarray(np.repeat(lattice[..., None], 3, axis=2), 'RGB').save(source, 'PNG')
                diagnostic_dir = root / 'textured-input'
                submitted_before = len(graphs)
                reference_provider.main(['--root', str(root), '--image', str(source),
                    '--sample', 'LATTICE', '--provider', 'qwen_image_2_1_fast',
                    '--api-base', endpoint, '--prompt', 'repair the background haze',
                    '--out-dir', str(diagnostic_dir)])
                assert len(graphs) == submitted_before + 1
                assert (diagnostic_dir / 'LATTICE.provenance.json').is_file()
                assert not (diagnostic_dir / 'LATTICE.input-quality-rejection.json').exists()
                attempts = list(diagnostic_dir.glob('LATTICE.transport-*.json'))
                diagnostic = json.loads(attempts[0].read_text('utf8'))['input_quality']
                assert diagnostic['passed'] and diagnostic['grid_detected']
                assert diagnostic['grid_policy'] == 'diagnostic_only'
                # Output quality rejection happens after a real submission to
                # this fake server; it must never use the zero-submission type.
                output = source.read_bytes()
                Image.new('RGB', (64, 96), 'red').save(source, 'PNG')
                output_rejected = root / 'rejected-output'
                submitted_before = len(graphs)
                try:
                    reference_provider.main(['--root', str(root), '--image', str(source),
                        '--sample', 'OUTPUT_GRID', '--provider', 'qwen_image_2_1_fast',
                        '--api-base', endpoint, '--prompt', 'edit the lighting',
                        '--out-dir', str(output_rejected)])
                except RuntimeError as error:
                    assert not isinstance(error, reference_provider.QwenInputQualityError)
                    assert 'output rejected before reference registration' in str(error)
                else:
                    raise AssertionError('Generated diagonal lattice reached candidate registration')
                assert len(graphs) == submitted_before + 1
                rejection = json.loads((output_rejected / 'OUTPUT_GRID.quality-rejection.json').read_text('utf8'))
                assert rejection['image_generation_submissions'] == 1
                assert rejection['quality']['context'] == 'generated_output'
                assert not (output_rejected / 'OUTPUT_GRID.provenance.json').exists()
                output = buffer.getvalue()
                # Restore a clean fixture before exercising the independent
                # ComfyUI execution-error receipt below.
                Image.new('RGB', (64, 96), 'red').save(source, 'PNG')
                import comfyui_adapter
                submissions=len(graphs);events=[]
                try:
                    comfyui_adapter.request_generation(endpoint,source,
                        Path(reference_provider.provider_config(provider='qwen_image_2_1_fast')['workflow_path']),
                        'wrong source',3,events,condition_inputs=[{'path':str(source),'sha256':'wrong'}])
                except ValueError:pass
                else:raise AssertionError('A changed source identity reached the server')
                assert len(graphs)==submissions and not events
                failures['enabled']=True
                failed=root/'failed'
                try:
                    reference_provider.main(['--root',str(root),'--image',str(source),'--sample','FAILED',
                        '--provider','qwen_image_2_1_fast','--api-base',endpoint,'--prompt','failure probe',
                        '--out-dir',str(failed)])
                except RuntimeError as error:assert 'hostbuf_file_reader_read failed' in str(error)
                else:raise AssertionError('Local execution failure registered a candidate')
                assert len(graphs)==submissions+1 and not (failed/'FAILED.png').exists()
                diagnostic=json.loads((failed/'FAILED.transport-failure.json').read_text('utf8'))
                assert diagnostic['events'][-1]['event']=='comfyui_execution_failed'
                assert not diagnostic['automatic_retry'] and diagnostic['candidate_registered'] is False
                failures['enabled']=False
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
