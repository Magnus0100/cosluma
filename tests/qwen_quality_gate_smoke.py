"""Input texture is diagnostic; generated grid and unreadable bytes are rejected."""
from __future__ import annotations

import tempfile
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import reference_provider
import start_qwen_image
from reference_provider import QwenInputQualityError, qwen_quality_gate


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="cos-qwen-gate-") as directory:
        root = Path(directory)
        clean = root / "clean.png"
        Image.new("RGB", (96, 96), "steelblue").save(clean)
        assert qwen_quality_gate(clean)["passed"]

        yy, xx = np.indices((192, 192))
        lattice = ((xx + yy) % 8 < 3).astype(np.uint8) * 255
        Image.fromarray(np.repeat(lattice[..., None], 3, axis=2), "RGB").save(root / "lattice.png")
        rejected = qwen_quality_gate(root / "lattice.png")
        assert not rejected["passed"]
        assert rejected["diag_peak_lag_4_16"] > rejected["threshold"]
        source_diagnostic = qwen_quality_gate(root / "lattice.png", context="source_input")
        assert source_diagnostic["passed"] and source_diagnostic["decodable"]
        assert source_diagnostic["grid_detected"]
        assert source_diagnostic["grid_policy"] == "diagnostic_only"
        assert source_diagnostic["reason"] is None
        assert source_diagnostic["diag_peak_lag_4_16"] == rejected["diag_peak_lag_4_16"]
        # Broad cloth folds or architectural bands correlate at nearby lags
        # without forming repeated local peaks. A high scalar is human-review
        # evidence, not an automatic rejection of ordinary material detail.
        natural = np.clip(128 + 60 * np.sin((xx + yy) / 24.0), 0, 255).astype(np.uint8)
        Image.fromarray(np.repeat(natural[..., None], 3, axis=2), "RGB").save(root / "natural.png")
        natural_quality = qwen_quality_gate(root / "natural.png")
        assert natural_quality["threshold_exceeded"] and natural_quality["review_recommended"]
        assert natural_quality["passed"] and not natural_quality["grid_detected"]
        assert rejected["periodicity"]["corroborated"]
        assert all(pair["corroborated"] for pair in rejected["periodicity"]["tested_pairs"])

        # A truncated provider response is a quality rejection, not an
        # uncaught decoder exception.  The caller can persist this result and
        # leave the failed bytes inspectable without registering a candidate.
        unreadable = root / "truncated.png"
        unreadable.write_bytes(b"not-a-raster")
        malformed = qwen_quality_gate(unreadable)
        assert not malformed["passed"]
        assert malformed["reason"] == "unreadable_image"
        assert malformed["diag_peak_lag_4_16"] is None
        assert malformed["metrics"]["error_type"]
        unreadable_input = qwen_quality_gate(unreadable, context="source_input")
        assert not unreadable_input["passed"] and not unreadable_input["decodable"]
        # The production pre-submit branch must expose trustworthy zero usage,
        # preserve a readable receipt and never start a service or HTTP request.
        destination = root / "bad-input"
        with (patch.dict(os.environ, {"COSPLAY_PROVIDER_SETTINGS": str(root / "settings.json")}),
              patch.object(reference_provider, "qwen_quality_gate", return_value=unreadable_input),
              patch.object(start_qwen_image, "ensure_running") as service,
              patch.object(reference_provider.comfyui_adapter, "request_generation") as transport):
            try:
                reference_provider.main(["--root", str(root), "--image", str(clean),
                    "--sample", "BAD", "--provider", "qwen_image_2_1_fast",
                    "--api-base", "http://127.0.0.1:1", "--prompt", "edit the lighting",
                    "--out-dir", str(destination)])
            except QwenInputQualityError as error:
                assert error.request_submitted is False
                assert error.model_submissions == error.image_generation_submissions == error.http_submissions == 0
                receipt = json.loads(Path(error.receipt_path).read_text("utf8"))
                assert receipt["request_submitted"] is False
                assert receipt["model_submissions"] == receipt["image_generation_submissions"] == receipt["http_submissions"] == 0
                assert receipt["reason"] == "unreadable_image"
            else:
                raise AssertionError("Unreadable input reached provider transport")
            try:
                reference_provider.main(["--root", str(root), "--image", str(clean),
                    "--sample", "EMPTY", "--provider", "qwen_image_2_1_fast",
                    "--api-base", "http://127.0.0.1:1", "--prompt", " \n ",
                    "--out-dir", str(root / "empty-prompt")])
            except QwenInputQualityError as error:
                assert error.request_submitted is False and error.model_submissions == 0
                receipt = json.loads(Path(error.receipt_path).read_text("utf8"))
                assert receipt["reason"] == "empty_edit_prompt"
                assert receipt["model_submissions"] == receipt["http_submissions"] == 0
            else:
                raise AssertionError("Empty prompt reached provider transport")
            service.assert_not_called()
            transport.assert_not_called()
        minimal = QwenInputQualityError("unreadable input")
        assert minimal.model_submissions == 0 and minimal.receipt_path is None
    print({"ok": True, "clean_passes": True, "generated_grid_rejected": True,
           "input_texture_diagnostic_nonblocking": True, "unreadable_input_zero_submissions": True})


if __name__ == "__main__":
    main()
