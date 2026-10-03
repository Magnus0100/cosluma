#!/usr/bin/env python3
"""Minimal, deterministic appearance-transfer demo for the cosplay dataset.

The script intentionally uses only Pillow and NumPy.  It estimates an RGB affine
operator globally and on a coarse grid at proxy resolution, then applies the
smoothly interpolated coefficients to the full-resolution original in tiles.
The generated alpha is a replaceable soft subject trust mask; a user supplied
mask can be passed with ``--alpha``.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageOps


RESAMPLE = Image.Resampling


def rgb_image(path: Path) -> Image.Image:
    """Open an image as RGB while respecting camera orientation metadata."""
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def find_assets(root: Path, sample: str) -> Tuple[Path, Path, Path]:
    data = root / "data"
    original = data / f"{sample}.jpg"
    reference = data / f"{sample}参考.png"
    manual = data / f"{sample}.png"
    missing = [str(p) for p in (original, reference, manual) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing sample asset(s): " + ", ".join(missing))
    return original, reference, manual


def proxy_arrays(source: Image.Image, reference: Image.Image, max_side: int = 768):
    scale = min(1.0, float(max_side) / max(source.size))
    size = (max(16, round(source.width * scale)), max(16, round(source.height * scale)))
    s = np.asarray(source.resize(size, RESAMPLE.BILINEAR), dtype=np.float32) / 255.0
    r = np.asarray(reference.resize(size, RESAMPLE.BILINEAR), dtype=np.float32) / 255.0
    return s, r, size


def affine_fit(x: np.ndarray, y: np.ndarray, ridge: float = 0.03) -> np.ndarray:
    """Fit a stable per-channel affine ``y = scale*x + bias``.

    A full cross-channel fit is ill-conditioned for portrait RGB data and can
    turn a strong edit into a gray veil.  Independent, clipped channel fits
    remain an interpretable affine appearance operator while preserving local
    contrast and texture.
    """
    src, dst = x[:, :3], y[:, :3]
    sm, dm = src.mean(axis=0), dst.mean(axis=0)
    # Moment matching is more robust than covariance regression when the edit
    # adds bright particles or changes only a part of the frame.
    scale = np.clip(dst.std(axis=0) / np.maximum(src.std(axis=0), 1e-5), 0.75, 1.8)
    # A small identity prior avoids overreacting to nearly flat cells.
    scale = (1.0 - ridge) * scale + ridge
    coef = np.zeros((4, 3), dtype=np.float32)
    coef[:3, :3] = np.diag(scale.astype(np.float32))
    coef[3, :] = (dm - scale * sm).astype(np.float32)
    return coef


def fit_operator(src: np.ndarray, ref: np.ndarray, grid: Tuple[int, int], seed: int = 17, trust: np.ndarray | None = None):
    h, w, _ = src.shape
    rng = np.random.default_rng(seed)
    n = h * w
    trusted_ids = np.flatnonzero(trust.reshape(-1) > 0.5) if trust is not None else np.arange(n)
    if len(trusted_ids) < 1000:
        trusted_ids = np.arange(n)
    take = min(n, 450_000)
    ids = rng.choice(trusted_ids, size=min(len(trusted_ids), take), replace=False)
    sf = src.reshape(-1, 3)[ids]
    rf = ref.reshape(-1, 3)[ids]
    x = np.concatenate([sf, np.ones((len(sf), 1), dtype=np.float32)], axis=1)
    global_coef = affine_fit(x, rf)

    gx, gy = grid
    local = np.empty((gy, gx, 4, 3), dtype=np.float32)
    counts = np.zeros((gy, gx), dtype=np.int32)
    # A local fit is shrunk towards the global fit, which keeps sparse/flat cells stable.
    for iy in range(gy):
        y0, y1 = round(iy * h / gy), round((iy + 1) * h / gy)
        for ix in range(gx):
            x0, x1 = round(ix * w / gx), round((ix + 1) * w / gx)
            cs = src[y0:y1, x0:x1].reshape(-1, 3)
            cr = ref[y0:y1, x0:x1].reshape(-1, 3)
            if trust is not None:
                keep = trust[y0:y1, x0:x1].reshape(-1) > 0.5
                cs, cr = cs[keep], cr[keep]
            stride = max(1, len(cs) // 20_000)
            cs, cr = cs[::stride], cr[::stride]
            counts[iy, ix] = len(cs)
            if len(cs) < 80:
                local[iy, ix] = global_coef
                continue
            xx = np.concatenate([cs, np.ones((len(cs), 1), dtype=np.float32)], axis=1)
            fit = affine_fit(xx, cr, ridge=0.05)
            weight = min(1.0, len(cs) / 12_000.0)
            local[iy, ix] = (1.0 - weight) * global_coef + weight * fit
    return global_coef, local, counts


def bilinear_coefficients(grid: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Interpolate a (gy,gx,4,3) coefficient grid at normalized grid coords."""
    gy, gx = grid.shape[:2]
    y = np.clip(ys, 0, gy - 1)
    x = np.clip(xs, 0, gx - 1)
    y0, x0 = np.floor(y).astype(np.int32), np.floor(x).astype(np.int32)
    y1, x1 = np.minimum(y0 + 1, gy - 1), np.minimum(x0 + 1, gx - 1)
    wy, wx = (y - y0).astype(np.float32), (x - x0).astype(np.float32)
    wy, wx = wy[..., None, None], wx[..., None, None]
    return (
        grid[y0, x0] * (1 - wy) * (1 - wx)
        + grid[y1, x0] * wy * (1 - wx)
        + grid[y0, x1] * (1 - wy) * wx
        + grid[y1, x1] * wy * wx
    )


def apply_operator(source: Image.Image, local: np.ndarray, tile: int = 768) -> Image.Image:
    w, h = source.size
    gx, gy = local.shape[1], local.shape[0]
    result = Image.new("RGB", (w, h))
    # Coordinate arrays are generated per tile so 63MP inputs stay within modest RAM.
    for top in range(0, h, tile):
        bottom = min(h, top + tile)
        yy = np.arange(top, bottom, dtype=np.float32)
        # Mapping to the coefficient grid, with endpoints included.
        gyc = yy[:, None] / max(1, h - 1) * (gy - 1)
        for left in range(0, w, tile):
            right = min(w, left + tile)
            xx = np.arange(left, right, dtype=np.float32)
            ys = np.broadcast_to(gyc, (bottom - top, right - left))
            xs = np.broadcast_to(xx[None, :] / max(1, w - 1) * (gx - 1), ys.shape)
            coeff = bilinear_coefficients(local, ys, xs)
            arr = np.asarray(source.crop((left, top, right, bottom)), dtype=np.float32) / 255.0
            # coeff[..., channel, input+intercept].
            out = np.empty_like(arr)
            aug = np.empty((arr.shape[0], arr.shape[1], 4), dtype=np.float32)
            aug[..., :3] = arr
            aug[..., 3] = 1.0
            out[:] = np.einsum("hwc,hwck->hwk", aug, coeff)
            result.paste(Image.fromarray(np.uint8(np.clip(out, 0, 1) * 255.0 + 0.5)), (left, top))
    return result


def make_alpha(size: Tuple[int, int], center=(0.50, 0.53), radii=(0.26, 0.47), softness=0.16) -> Image.Image:
    w, h = size
    # Build in horizontal strips to avoid a second full-size float image.
    out = Image.new("L", size)
    cx, cy = center
    rx, ry = radii
    for top in range(0, h, 1024):
        bottom = min(h, top + 1024)
        x = (np.arange(w, dtype=np.float32)[None, :] / max(1, w - 1) - cx) / rx
        y = (np.arange(top, bottom, dtype=np.float32)[:, None] / max(1, h - 1) - cy) / ry
        d = np.sqrt(x * x + y * y)
        # Smooth transition around the ellipse boundary.
        t = np.clip((1.0 + softness - d) / (2.0 * softness), 0.0, 1.0)
        a = t * t * (3.0 - 2.0 * t)
        out.paste(Image.fromarray(np.uint8(a * 255.0 + 0.5), mode="L"), (0, top))
    return out


def blend_tiles(styled: Image.Image, reference: Image.Image, alpha: Image.Image, tile=768) -> Image.Image:
    w, h = styled.size
    result = Image.new("RGB", (w, h))
    for top in range(0, h, tile):
        for left in range(0, w, tile):
            box = (left, top, min(w, left + tile), min(h, top + tile))
            a = np.asarray(alpha.crop(box), dtype=np.float32)[..., None] / 255.0
            s = np.asarray(styled.crop(box), dtype=np.float32)
            r = np.asarray(reference.crop(box), dtype=np.float32)
            result.paste(Image.fromarray(np.uint8(np.clip(a * s + (1 - a) * r, 0, 255) + 0.5)), (left, top))
    return result


def save_png(im: Image.Image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, format="PNG", compress_level=1)


def comparison_sheet(images: Iterable[Tuple[str, Image.Image]], path: Path):
    thumb_w, thumb_h = 420, 630
    items = list(images)
    sheet = Image.new("RGB", (thumb_w * 2, (thumb_h + 34) * 2), "#202020")
    draw = ImageDraw.Draw(sheet)
    for i, (label, im) in enumerate(items[:4]):
        thumb = ImageOps.contain(im, (thumb_w, thumb_h), RESAMPLE.BILINEAR)
        x, y = (i % 2) * thumb_w, (i // 2) * (thumb_h + 34)
        sheet.paste(thumb, (x + (thumb_w - thumb.width) // 2, y))
        draw.text((x + 8, y + thumb_h + 8), label, fill="white")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=92)


def run_sample(root: Path, sample: str, out_root: Path, args) -> Dict:
    original_path, reference_path, manual_path = find_assets(root, sample)
    original, reference, manual = rgb_image(original_path), rgb_image(reference_path), rgb_image(manual_path)
    if original.size != manual.size:
        manual = manual.resize(original.size, RESAMPLE.LANCZOS)
    proxy_s, proxy_r, proxy_size = proxy_arrays(original, reference, args.proxy_side)
    fit_trust = None
    if args.fit_alpha:
        fit_mask = ImageOps.exif_transpose(Image.open(args.fit_alpha)).convert("L").resize(proxy_size, RESAMPLE.BILINEAR)
        fit_trust = np.asarray(fit_mask, dtype=np.float32) / 255.0
    global_coef, local_coef, counts = fit_operator(proxy_s, proxy_r, (args.grid_x, args.grid_y), trust=fit_trust)
    styled = apply_operator(original, local_coef, args.tile)
    upscaled = reference.resize(original.size, RESAMPLE.LANCZOS)
    if args.alpha:
        alpha = ImageOps.exif_transpose(Image.open(args.alpha)).convert("L").resize(original.size, RESAMPLE.LANCZOS)
        alpha_mode = "external"
    else:
        alpha = make_alpha(original.size, args.alpha_center, args.alpha_radii, args.alpha_softness)
        alpha_mode = "soft_ellipse"
    final = blend_tiles(styled, upscaled, alpha, args.tile)
    out = out_root / sample
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "styled_original": out / "styled_original.png",
        "upscaled_reference": out / "upscaled_reference.png",
        "alpha": out / "alpha.png",
        "final": out / "final.png",
        "comparison": out / "comparison.jpg",
        "metadata": out / "metadata.json",
    }
    save_png(styled, files["styled_original"])
    save_png(upscaled, files["upscaled_reference"])
    save_png(alpha, files["alpha"])
    save_png(final, files["final"])
    comparison_sheet((("original", original), ("reference", upscaled), ("manual final", manual), ("demo final", final)), files["comparison"])
    metadata = {
        "sample": sample,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {"original": str(original_path), "reference": str(reference_path), "manual_final": str(manual_path)},
        "input_size": list(original.size), "proxy_size": list(proxy_size), "grid": [args.grid_x, args.grid_y],
        "operator": "stable per-channel RGB affine y=[r,g,b,1]@coef; local fits shrunk to global; bilinear coefficient interpolation",
        "global_coef": np.round(global_coef, 7).tolist(),
        "local_cell_counts": counts.tolist(),
        # Keep the exact mask source in the run record.  A generated alpha has
        # no input path, while an edited mask should be reproducible from the
        # path shown here and the command recorded by the caller.
        "alpha": {
            "mode": alpha_mode,
            "path": str(args.alpha) if args.alpha else None,
            "center": list(args.alpha_center),
            "radii": list(args.alpha_radii),
            "softness": args.alpha_softness,
        },
        "fit_trust": str(args.fit_alpha) if args.fit_alpha else None,
        "parameters": {"proxy_side": args.proxy_side, "tile": args.tile},
        "outputs": {k: str(v) for k, v in files.items()},
    }
    files["metadata"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def parse_pair(value: str) -> Tuple[float, float]:
    vals = [float(x.strip()) for x in value.split(",")]
    if len(vals) != 2:
        raise argparse.ArgumentTypeError("expected x,y")
    return vals[0], vals[1]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample", action="append", help="sample id; repeat, or omit for all available samples")
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--proxy-side", type=int, default=768)
    p.add_argument("--grid-x", type=int, default=8)
    p.add_argument("--grid-y", type=int, default=12)
    p.add_argument("--tile", type=int, default=768)
    p.add_argument("--alpha", type=Path, help="optional grayscale mask; resized to original dimensions")
    p.add_argument("--fit-alpha", type=Path, help="optional grayscale mask; fit the appearance operator only on pixels above 50%")
    p.add_argument("--alpha-center", type=parse_pair, default=(0.50, 0.53))
    p.add_argument("--alpha-radii", type=parse_pair, default=(0.26, 0.47))
    p.add_argument("--alpha-softness", type=float, default=0.16)
    args = p.parse_args(argv)
    root = args.root.resolve()
    out_root = (args.out_dir or root / "outputs").resolve()
    samples = args.sample or sorted(p.stem for p in (root / "data").glob("*.jpg"))
    all_metadata = [run_sample(root, sample, out_root, args) for sample in samples]
    print(json.dumps({"samples": samples, "out_dir": str(out_root), "metadata": all_metadata}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
