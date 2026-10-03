#!/usr/bin/env python3
"""Optional U2Net human segmentation alpha backend.

This is intentionally separate from ``make_mask.py``: the normal workflow has
no model dependency, while production jobs can opt in after installing
``onnxruntime`` and downloading ``u2net_human_seg.onnx``.  The output is still
passed through the same editable confidence profile, so segmentation only
defines the subject silhouette; it does not decide which GPT structures should
be overwritten.

Example (from the repository root)::

    $py -m pip install --target .cache/onnxruntime onnxruntime
    $env:PYTHONPATH = (Resolve-Path .cache/onnxruntime).Path
    $py scripts/u2net_mask.py --sample DSC02634 \
        --model .cache/u2net_human_seg.onnx --out-dir outputs/u2net
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageOps

# Reuse the editable prior and source-size conventions from the lightweight
# backend.  Importing this module does not import onnxruntime.
from make_mask import POLYGONS, apply_confidence_profile, source_size


def segment(path: Path, model_path: Path, input_size: int = 320) -> Image.Image:
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError(
            "U2Net backend requires onnxruntime; install it into an isolated "
            "target and set PYTHONPATH (see scripts/u2net_mask.py)"
        ) from exc
    with Image.open(path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
    size = (int(input_size), int(input_size))
    arr = np.asarray(image.resize(size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    tensor = np.transpose(arr, (2, 0, 1))[None].astype(np.float32)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    output = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    score = np.asarray(output[0, 0], dtype=np.float32)
    # U2Net emits a per-image saliency map.  Min/max normalization is stable
    # across the two portrait samples and preserves its soft boundary.
    score = (score - float(score.min())) / max(1e-6, float(score.max() - score.min()))
    return Image.fromarray(np.uint8(np.clip(score, 0.0, 1.0) * 255.0 + 0.5), mode="L").resize(
        image.size, Image.Resampling.LANCZOS
    )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", action="append", help="sample id; repeat or omit for all polygon samples")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source-dir", type=Path,
                        help="directory containing true source JPGs; defaults to root/data")
    parser.add_argument("--model", type=Path, required=True, help="path to u2net_human_seg.onnx")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--input-size", type=int, default=320)
    parser.add_argument("--head-alpha", type=int, default=0)
    parser.add_argument("--body-alpha", type=int, default=184)
    parser.add_argument("--head-end", type=float, default=0.43)
    parser.add_argument("--body-start", type=float, default=0.52)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    source_dir = args.source_dir.resolve() if args.source_dir else root / "data"
    model = args.model.resolve()
    if not model.exists():
        raise FileNotFoundError(model)
    out_root = (args.out_dir or root / "outputs" / "u2net").resolve()
    samples = args.sample or sorted(POLYGONS)
    rows = []
    for sample in samples:
        candidates = [source_dir / f"{sample}.jpg", source_dir / f"{sample}.JPG", source_dir / f"{sample}.jpeg", source_dir / f"{sample}.JPEG"]
        src = next((p for p in candidates if p.exists()), candidates[0])
        if not src.exists():
            raise FileNotFoundError(src)
        soft = segment(src, model, args.input_size)
        alpha = apply_confidence_profile(
            soft, args.head_alpha, args.body_alpha, args.head_end, args.body_start
        )
        path = out_root / sample / "alpha.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        alpha.save(path, format="PNG", compress_level=1)
        rows.append(
            {
                "sample": sample,
                "size": list(source_size(root, sample)),
                "path": str(path),
                "backend": "u2net_human_seg",
                "model": str(model),
                "input_size": args.input_size,
                "confidence_profile": {
                    "head_alpha": args.head_alpha,
                    "body_alpha": args.body_alpha,
                    "head_end": args.head_end,
                    "body_start": args.body_start,
                },
            }
        )
    print(json.dumps({"outputs": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
