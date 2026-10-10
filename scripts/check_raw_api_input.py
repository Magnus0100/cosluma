"""Local RAW API-input boundary smoke; no network or generation credits used."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import reference_provider


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True, help="Explicit local RAW fixture; never distributed")
    args = parser.parse_args()
    raw = args.raw.resolve()
    original_hash = reference_provider.sha256(raw)
    seen = {}

    def local_response(endpoint, key, image, model, prompt, size, timeout, **kwargs):
        assert image != raw and image.name == "provider-input.jpg"
        with Image.open(image) as opened:
            assert opened.format == "JPEG" and opened.mode == "RGB"
            assert max(opened.size) == 2048 and opened.info.get("icc_profile")
        seen["api_path"] = str(image)
        return {"id": "local-boundary-response-id"}, image.read_bytes(), "b64_json"

    with tempfile.TemporaryDirectory(prefix="cosplay_raw_provider_smoke_") as temp:
        root = Path(temp)
        run_dir = root / "outputs/runs/raw-boundary"
        with patch.object(reference_provider, "open_rgb", wraps=reference_provider.open_rgb) as source_decode, \
             patch.dict(os.environ, {"COSPLAY_REFERENCE_API_KEY": "local-only-smoke-key"}), \
             patch.object(reference_provider, "request_edit", side_effect=local_response):
            assert reference_provider.main([
                "--root", str(root), "--image", str(raw), "--sample", "RAWSMOKE",
                "--provider", "openai-compatible", "--prompt", "source-preserving lighting",
                "--api-url", "http://127.0.0.1:9/not-called", "--out-dir", str(run_dir),
            ]) == 0
            source_decode.assert_called_once_with(raw)
        provenance = json.loads((run_dir / "RAWSMOKE.provenance.json").read_text(encoding="utf-8"))
        request = provenance["provider_contract"]["request"]
        api_input = request["api_input"]
        assert provenance["source"]["path"] == str(raw) and provenance["source"]["sha256"] == original_hash
        assert provenance["generation_id"] == "local-boundary-response-id"
        assert api_input["source"]["sha256"] == original_hash
        assert api_input["derived"] and api_input["path"] == seen["api_path"]
        assert api_input["decode"]["gamma"] == [1, 1]
        assert api_input["decode"]["source_precision"] == "raw16_linear"
        assert api_input["decode"]["source_output_bps"] == 16
        assert api_input["decode"]["output_transfer"] == "IEC_sRGB"
        assert api_input["decode"]["decode_version"] == reference_provider.RGB_DECODE_VERSION
        assert api_input["sha256"] == request["input_sha256"]
        assert request["input_sha256"] == reference_provider.sha256(Path(api_input["path"]))
        assert request["input_sha256"] != original_hash
        assert reference_provider.sha256(raw) == original_hash and not list(root.rglob("*.ARW"))
        print(json.dumps({"ok": True, "network_used": False, "raw_unchanged": True,
                          "api_input_size": api_input["size"], "decode": api_input["decode"],
                          "response_id_fallback": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
