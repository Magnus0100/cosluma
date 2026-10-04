#!/usr/bin/env python3
"""Estimate local edge displacements after global scale/translation alignment.

This is a diagnostic only.  It never changes a production image.  For each
proxy grid cell it searches a small reference displacement and writes a JSON
map plus an arrow overlay.  Cells with low texture/correlation are marked
unreliable so they cannot be copied into a mesh warp configuration blindly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_demo  # noqa: E402


def corr(a: np.ndarray, b: np.ndarray) -> float:
    x = a.reshape(-1).astype(np.float32)
    y = b.reshape(-1).astype(np.float32)
    x -= x.mean()
    y -= y.mean()
    den = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float((x @ y) / (den + 1e-6)) if den else -1.0


def local_cell(es: np.ndarray, er: np.ndarray, x0: int, x1: int,
               y0: int, y1: int, max_shift: int):
    """Return best (dx,dy,score), where ref[x+dx,y+dy] matches source[x,y]."""
    h, w = es.shape
    best = (0, 0, -1.0)
    # Ignore tiny border truncation by comparing only the overlap in the cell.
    for dy in range(-max_shift, max_shift + 1):
        sy0, sy1 = max(y0, y0 - dy), min(y1, y1 - dy)
        ry0, ry1 = sy0 + dy, sy1 + dy
        if sy1 - sy0 < 8 or ry0 < 0 or ry1 > h:
            continue
        for dx in range(-max_shift, max_shift + 1):
            sx0, sx1 = max(x0, x0 - dx), min(x1, x1 - dx)
            rx0, rx1 = sx0 + dx, sx1 + dx
            if sx1 - sx0 < 8 or rx0 < 0 or rx1 > w:
                continue
            score = corr(es[sy0:sy1, sx0:sx1], er[ry0:ry1, rx0:rx1])
            if score > best[2]:
                best = (dx, dy, score)
    texture = float(np.percentile(es[y0:y1, x0:x1], 90))
    return best, texture


def run(root: Path, sample: str, source_dir: Path | None, out_dir: Path,
        grid_x: int, grid_y: int, max_shift: int, proxy_side: int):
    source_path, reference_path, _ = run_demo.find_assets(root, sample, source_dir)
    source = run_demo.rgb_image(source_path)
    reference = run_demo.rgb_image(reference_path)
    ps, pr, size = run_demo.proxy_arrays(source, reference, proxy_side)
    scale, gx, gy, global_score = run_demo.estimate_scale_translation(ps, pr, max_shift=24)
    if abs(scale - 1.0) > 1e-6:
        reference = run_demo.scale_center_image(reference, scale)
    if gx or gy:
        reference = run_demo.translate_image(reference, gx * reference.width / size[0], gy * reference.height / size[1])
    ps, pr, size = run_demo.proxy_arrays(source, reference, proxy_side)
    h, w = ps.shape[:2]
    edge_s, edge_r = run_demo._edge_luma(ps), run_demo._edge_luma(pr)
    cells = []
    for iy in range(grid_y):
        y0, y1 = round(iy * h / grid_y), round((iy + 1) * h / grid_y)
        for ix in range(grid_x):
            x0, x1 = round(ix * w / grid_x), round((ix + 1) * w / grid_x)
            (dx, dy, score), texture = local_cell(edge_s, edge_r, x0, x1, y0, y1, max_shift)
            cells.append({
                "ix": ix, "iy": iy, "box_proxy": [x0, y0, x1, y1],
                "dx_proxy": int(dx), "dy_proxy": int(dy),
                "score": round(float(score), 6), "texture_p90": round(texture, 6),
                "reliable": bool(score >= 0.35 and texture >= 0.015),
            })
    # A compact overlay makes discontinuities obvious during review.
    base = Image.fromarray(np.uint8(np.clip(pr, 0, 1) * 255 + 0.5), mode="RGB")
    draw = ImageDraw.Draw(base)
    for c in cells:
        x0, y0, x1, y1 = c["box_proxy"]
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        dx, dy = c["dx_proxy"], c["dy_proxy"]
        color = "#41ff6b" if c["reliable"] else "#ffb52e"
        draw.line((cx, cy, cx + dx * 3, cy + dy * 3), fill=color, width=2)
        draw.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill=color)
    out_sample = out_dir / sample
    out_sample.mkdir(parents=True, exist_ok=True)
    overlay = out_sample / "local_shift_overlay.png"
    base.save(overlay, format="PNG", compress_level=1)
    result = {
        "sample": sample,
        "inputs": {"source": str(source_path), "reference": str(reference_path)},
        "proxy_size": [int(w), int(h)], "grid": [grid_x, grid_y],
        "global_alignment": {"scale": round(float(scale), 6), "dx_proxy": int(gx), "dy_proxy": int(gy), "score": round(float(global_score), 6)},
        "max_shift_proxy": int(max_shift), "cells": cells,
        "overlay": str(overlay),
        "direction": "reference sample at (x+dx,y+dy) compared with source at (x,y)",
    }
    path = out_sample / "local_shift.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"sample": sample, "json": str(path), "overlay": str(overlay)}, ensure_ascii=False))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--source-dir", type=Path)
    p.add_argument("--sample", action="append", required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--grid-x", type=int, default=6)
    p.add_argument("--grid-y", type=int, default=8)
    p.add_argument("--max-shift", type=int, default=8)
    p.add_argument("--proxy-side", type=int, default=768)
    args = p.parse_args(argv)
    root = args.root.resolve()
    out = (args.out_dir or root / "outputs" / "local_shift_diagnostic").resolve()
    source_dir = args.source_dir.resolve() if args.source_dir else None
    for sample in args.sample:
        run(root, sample, source_dir, out, args.grid_x, args.grid_y, args.max_shift, args.proxy_side)


if __name__ == "__main__":
    main()
