#!/usr/bin/env python3
"""Current BiRefNet errors must surface; content-bound masks may be cached."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import source_alpha


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="cosplay_source_alpha_") as temp:
        root = Path(temp); source = root / "source.png"
        image = Image.new("RGB", (96, 64), (75, 105, 145)); image.save(source)
        with patch.dict(source_alpha.os.environ, {"COSPLAY_ALPHA_MODEL": str(root/".cache/models/birefnet_lite_fp16.onnx")}):
            try:
                source_alpha.source_silhouette(root, source, image)
            except RuntimeError as error:
                assert 'BiRefNet model is unavailable' in str(error)
            else:
                raise AssertionError('Missing BiRefNet silently produced a mask')
            for backend in ('ellipse', 'polygon', 'u2net'):
                try: source_alpha.source_silhouette(root, source, image, backend=backend)
                except ValueError: pass
                else: raise AssertionError('Removed silhouette route was accepted')
            model = root / ".cache/models/birefnet_lite_fp16.onnx"
            model.parent.mkdir(parents=True); model.write_bytes(b"cache-contract-fixture")
            with patch.object(source_alpha, "_infer", return_value=(Image.new("L", (32, 32), 255), {"preprocess": {"color": "RGB"}})) as infer:
                _, first = source_alpha.source_silhouette(root, source, image)
                _, second = source_alpha.source_silhouette(root, source, image)
                assert infer.call_count == 1 and not first["cache_hit"] and second["cache_hit"]
                model.write_bytes(b"new-model-fixture")
                _, third = source_alpha.source_silhouette(root, source, image)
                assert third["cache_key"] != first["cache_key"] and infer.call_count == 2
            # Runtime failures must surface their real cause.
            model.write_bytes(b"failure-fixture")
            with patch.object(source_alpha, "_infer", side_effect=RuntimeError("runtime fixture")):
                try: source_alpha.source_silhouette(root, source, image)
                except RuntimeError as error: assert 'runtime fixture' in str(error)
                else: raise AssertionError('Inference failure silently produced a mask')
            model.unlink()
    print("source alpha smoke ok: current route only, explicit inference errors, content-bound model cache")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
