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
from PIL import Image, ImageDraw, ImageFilter, ImageOps


RESAMPLE = Image.Resampling


def rgb_image(path: Path) -> Image.Image:
    """Open an image as RGB while respecting camera orientation metadata."""
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def find_assets(root: Path, sample: str, source_dir: Path | None = None) -> Tuple[Path, Path, Path]:
    data = root / "data"
    source_root = source_dir.resolve() if source_dir else data
    matches = sorted(
        (p for p in source_root.iterdir() if p.is_file()
         and p.stem.casefold() == sample.casefold()
         and p.suffix.casefold() in {".jpg", ".jpeg"}),
        key=lambda p: (p.suffix.casefold() != ".jpg", p.name),
    ) if source_root.exists() else []
    original = matches[0] if matches else source_root / f"{sample}.jpg"
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


def _edge_luma(rgb: np.ndarray) -> np.ndarray:
    gray = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    return np.hypot(gx, gy)


def structure_disagreement(source: np.ndarray, reference: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """Estimate where the generated reference has a different structure.

    Colour and exposure changes are deliberately removed from this diagnostic:
    it compares normalized luma edge strength in the already aligned proxy
    images.  A displaced face contour, hair strand, hand or garment edge leaves
    a high score while a broad colour wash with the same geometry stays low.
    The returned score is proxy-sized and in ``[0, 1]`` so it can be inspected
    without allocating a second full-resolution image.
    """
    es = _edge_luma(source)
    er = _edge_luma(reference)
    # Per-image robust scaling makes the comparison insensitive to different
    # contrast/exposure in the source and generated reference.
    ps = float(np.percentile(es, 99.0))
    pr = float(np.percentile(er, 99.0))
    es = np.clip(es / max(ps, 1e-5), 0.0, 1.0)
    er = np.clip(er / max(pr, 1e-5), 0.0, 1.0)
    score = np.clip(np.abs(es - er), 0.0, 1.0).astype(np.float32)

    # Band means make the result actionable when deciding where alpha should
    # be lowered.  They are normalized coordinates, independent of image size.
    by, bx = 8, 6
    bands = np.zeros((by, bx), dtype=np.float32)
    h, w = score.shape
    for iy in range(by):
        y0, y1 = round(iy * h / by), round((iy + 1) * h / by)
        for ix in range(bx):
            x0, x1 = round(ix * w / bx), round((ix + 1) * w / bx)
            bands[iy, ix] = float(score[y0:y1, x0:x1].mean())
    stats = {
        "proxy_size": [int(w), int(h)],
        "edge_scale_p99": {"source": round(ps, 6), "reference": round(pr, 6)},
        "mean": round(float(score.mean()), 6),
        "p50": round(float(np.percentile(score, 50)), 6),
        "p90": round(float(np.percentile(score, 90)), 6),
        "p95": round(float(np.percentile(score, 95)), 6),
        "p99": round(float(np.percentile(score, 99)), 6),
        "max": round(float(score.max()), 6),
        "fraction_above": {
            str(t): round(float((score >= t).mean()), 6)
            for t in (0.15, 0.25, 0.35, 0.50)
        },
        "band_means_y8_x6": np.round(bands, 6).tolist(),
        "interpretation": "high scores indicate edge/shape disagreement; lower alpha there to preserve GPT structure",
    }
    return score, stats


def structure_overlay(reference: np.ndarray, score: np.ndarray) -> Image.Image:
    """Render a compact red/yellow disagreement heatmap over the reference."""
    base = np.uint8(np.clip(reference, 0.0, 1.0) * 255.0 + 0.5)
    # Accent only meaningful disagreements; this keeps the image readable when
    # most of the frame is structurally stable.
    heat = np.clip((score - 0.05) / 0.35, 0.0, 1.0)
    tint = np.empty_like(base)
    tint[..., 0] = 255
    tint[..., 1] = np.uint8((1.0 - heat) * 110.0)
    tint[..., 2] = np.uint8((1.0 - heat) * 20.0)
    out = np.uint8((base.astype(np.float32) * (1.0 - 0.58 * heat[..., None]) + tint.astype(np.float32) * (0.58 * heat[..., None])) + 0.5)
    return Image.fromarray(out, mode="RGB")


def structure_safe_alpha(
    alpha: Image.Image,
    score: np.ndarray,
    threshold: float = 0.28,
    strength: float = 0.78,
    blur: float = 1.0,
    floor: float = 0.08,
) -> Image.Image:
    """Lower write-back confidence where the generated image changed edges.

    ``alpha`` remains the authoritative subject region.  The edge disagreement
    map only modulates confidence *inside* that region; it can therefore never
    grow a mask into the background.  The score is blurred at proxy resolution
    before it is enlarged, which turns one-pixel edge noise into a narrow soft
    transition at the full image size.  This is deliberately a conservative
    rule: a score below ``threshold`` leaves the hand-authored confidence map
    unchanged, while a score of one still retains ``floor`` of the original
    confidence for a feathered seam.
    """
    if score.ndim != 2:
        raise ValueError("structure disagreement score must be a 2-D array")
    # Blur in the small diagnostic domain, where the radius is meaningful and
    # memory use is bounded even for the 60MP source photos.
    score_img = Image.fromarray(np.uint8(np.clip(score, 0.0, 1.0) * 255.0 + 0.5), mode="L")
    if blur > 0:
        score_img = score_img.filter(ImageFilter.GaussianBlur(float(blur)))
    score_full = np.asarray(score_img.resize(alpha.size, RESAMPLE.BILINEAR), dtype=np.float32) / 255.0
    t = float(np.clip(threshold, 0.0, 0.99))
    s = float(np.clip(strength, 0.0, 1.0))
    lo = float(np.clip(floor, 0.0, 1.0))
    disagreement = np.clip((score_full - t) / max(1e-6, 1.0 - t), 0.0, 1.0)
    factor = np.maximum(lo, 1.0 - s * disagreement)
    base = np.asarray(alpha, dtype=np.float32)
    # Multiplication preserves the polygon boundary and all manually painted
    # zeroes.  Round only once, after modulation, to avoid banding.
    return Image.fromarray(np.uint8(np.clip(base * factor, 0.0, 255.0) + 0.5), mode="L")


def estimate_translation(source: np.ndarray, reference: np.ndarray, max_shift: int = 24):
    """Find a small proxy translation using edge correlation.

    The two images have different appearance, but their bridge and costume
    edges are shared.  A small translation search is enough for this dataset
    and avoids blending two slightly offset faces into a ghost.
    """
    a, b = _edge_luma(source), _edge_luma(reference)
    best_score, best_shift = -1.0, (0, 0)
    h, w = a.shape
    for dy in range(-max_shift, max_shift + 1):
        ay0, ay1 = max(0, dy), min(h, h + dy)
        by0, by1 = max(0, -dy), max(0, -dy) + (ay1 - ay0)
        for dx in range(-max_shift, max_shift + 1):
            ax0, ax1 = max(0, dx), min(w, w + dx)
            bx0, bx1 = max(0, -dx), max(0, -dx) + (ax1 - ax0)
            x = a[ay0:ay1, ax0:ax1].reshape(-1)
            y = b[by0:by1, bx0:bx1].reshape(-1)
            x = x - x.mean()
            y = y - y.mean()
            score = float((x @ y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-6))
            if score > best_score:
                best_score, best_shift = score, (dx, dy)
    return best_shift[0], best_shift[1], best_score


def translate_image(image: Image.Image, shift_x: float, shift_y: float) -> Image.Image:
    """Translate an RGB image while replicating pixels at the exposed edge."""
    arr = np.asarray(image, dtype=np.uint8)
    h, w = arr.shape[:2]
    yy = np.clip(np.arange(h) - int(round(shift_y)), 0, h - 1)
    xx = np.clip(np.arange(w) - int(round(shift_x)), 0, w - 1)
    return Image.fromarray(arr[yy[:, None], xx[None, :]], mode="RGB")


def scale_center_image(image: Image.Image, scale: float) -> Image.Image:
    """Scale an image around its centre while keeping the original canvas.

    The alignment search works on a proxy, but the winning transform must be
    applied to the full-resolution reference before the appearance operator is
    fitted.  Cropping (or edge-padding) keeps all downstream image sizes and
    interfaces unchanged.
    """
    if abs(float(scale) - 1.0) < 1e-6:
        return image
    w, h = image.size
    nw = max(1, int(round(w * float(scale))))
    nh = max(1, int(round(h * float(scale))))
    resized = image.resize((nw, nh), RESAMPLE.BICUBIC)
    arr = np.asarray(resized, dtype=np.uint8)
    if nw >= w:
        x0 = (nw - w) // 2
        arr = arr[:, x0:x0 + w]
    else:
        pad_l = (w - nw) // 2
        pad_r = w - nw - pad_l
        arr = np.pad(arr, ((0, 0), (pad_l, pad_r), (0, 0)), mode="edge")
    if nh >= h:
        y0 = (nh - h) // 2
        arr = arr[y0:y0 + h]
    else:
        pad_t = (h - nh) // 2
        pad_b = h - nh - pad_t
        arr = np.pad(arr, ((pad_t, pad_b), (0, 0), (0, 0)), mode="edge")
    return Image.fromarray(arr, mode="RGB")


def scale_center_proxy(rgb: np.ndarray, scale: float) -> np.ndarray:
    """Centre-scale a proxy RGB array and preserve its shape."""
    if abs(float(scale) - 1.0) < 1e-6:
        return rgb
    h, w = rgb.shape[:2]
    image = Image.fromarray(np.uint8(np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5), mode="RGB")
    scaled = scale_center_image(image, scale)
    return np.asarray(scaled, dtype=np.float32) / 255.0


def alignment_scales(scale_min: float, scale_max: float, scale_step: float):
    """Return a small, deterministic scale grid and always include 1.0."""
    if scale_step <= 0:
        raise ValueError("alignment scale step must be positive")
    lo, hi = sorted((float(scale_min), float(scale_max)))
    values = [lo]
    while values[-1] + scale_step < hi - 1e-8:
        values.append(values[-1] + scale_step)
    if hi > values[-1] + 1e-8:
        values.append(hi)
    if lo <= 1.0 <= hi and not any(abs(v - 1.0) < 1e-6 for v in values):
        values.append(1.0)
    return sorted(set(round(v, 6) for v in values))


def estimate_scale_translation(
    source: np.ndarray,
    reference: np.ndarray,
    scale_min: float = 0.97,
    scale_max: float = 1.03,
    scale_step: float = 0.01,
    max_shift: int = 24,
):
    """Search a narrow centre-scale plus translation transform on the proxy.

    Each candidate is scored with the existing edge correlation routine.  The
    search is intentionally small (the default is seven scales) so it remains
    practical for a one-pass demo while correcting the global size mismatch
    that a pure translation cannot address.
    """
    best = (1.0, 0, 0, -1.0)
    for scale in alignment_scales(scale_min, scale_max, scale_step):
        scaled_ref = scale_center_proxy(reference, scale)
        dx, dy, score = estimate_translation(source, scaled_ref, max_shift=max_shift)
        if score > best[3]:
            best = (float(scale), int(dx), int(dy), float(score))
    return best


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
    original_path, reference_path, manual_path = find_assets(root, sample, getattr(args, "source_dir", None))
    original, reference, manual = rgb_image(original_path), rgb_image(reference_path), rgb_image(manual_path)
    if original.size != manual.size:
        manual = manual.resize(original.size, RESAMPLE.LANCZOS)
    proxy_s, proxy_r, proxy_size = proxy_arrays(original, reference, args.proxy_side)
    align_scale_min = getattr(args, "align_scale_min", 0.97)
    align_scale_max = getattr(args, "align_scale_max", 1.03)
    align_scale_step = getattr(args, "align_scale_step", 0.01)
    align_max_shift = getattr(args, "align_max_shift", 24)
    alignment_override = getattr(args, "alignment_override", None)
    if alignment_override:
        align_scale = float(alignment_override.get("scale", 1.0))
        shift_x = int(round(float(alignment_override.get("dx", 0))))
        shift_y = int(round(float(alignment_override.get("dy", 0))))
        align_score = None
        alignment_method = "manual proxy scale+translation override"
    else:
        align_scale, shift_x, shift_y, align_score = estimate_scale_translation(
            proxy_s,
            proxy_r,
            scale_min=align_scale_min,
            scale_max=align_scale_max,
            scale_step=align_scale_step,
            max_shift=align_max_shift,
        )
        alignment_method = "proxy edge scale+translation search"
    # Apply the winning proxy transform at full resolution before fitting the
    # RGB operator.  This keeps the existing output contract while avoiding a
    # second interpolation of the generated reference image.
    if abs(align_scale - 1.0) > 1e-6:
        reference = scale_center_image(reference, align_scale)
    if shift_x or shift_y:
        reference = translate_image(
            reference,
            shift_x * reference.width / proxy_size[0],
            shift_y * reference.height / proxy_size[1],
        )
    if abs(align_scale - 1.0) > 1e-6 or shift_x or shift_y:
        proxy_s, proxy_r, proxy_size = proxy_arrays(original, reference, args.proxy_side)
    # Keep this diagnostic at proxy resolution.  It is intended to show where
    # GPT changed geometry (face, hair, hands, garment edges), not to become a
    # second multi-gigabyte full-resolution render.
    mismatch_score, mismatch_stats = structure_disagreement(proxy_s, proxy_r)
    # An external output mask is also the best trust region for fitting the
    # appearance operator.  Requiring a second, identical path made the
    # common one-pass workflow easy to misuse, so ``--alpha`` now supplies
    # fit trust by default.  ``--fit-alpha`` still wins when a deliberately
    # different fitting mask is useful.
    fit_trust = None
    fit_alpha_path = args.fit_alpha or (args.alpha if not args.fit_full else None)
    fit_trust_mode = None
    if fit_alpha_path:
        fit_mask = ImageOps.exif_transpose(Image.open(fit_alpha_path)).convert("L").resize(proxy_size, RESAMPLE.BILINEAR)
        fit_trust = np.asarray(fit_mask, dtype=np.float32) / 255.0
        fit_trust_mode = "explicit" if args.fit_alpha else "alpha_default"
    elif args.fit_full:
        fit_trust_mode = "full_frame"
    global_coef, local_coef, counts = fit_operator(proxy_s, proxy_r, (args.grid_x, args.grid_y), trust=fit_trust)
    styled = apply_operator(original, local_coef, args.tile)
    upscaled = reference.resize(original.size, RESAMPLE.LANCZOS)
    if args.alpha:
        alpha = ImageOps.exif_transpose(Image.open(args.alpha)).convert("L").resize(original.size, RESAMPLE.LANCZOS)
        alpha_mode = "external"
    else:
        alpha = make_alpha(original.size, args.alpha_center, args.alpha_radii, args.alpha_softness)
        alpha_mode = "soft_ellipse"
    structure_safe = bool(getattr(args, "structure_safe", False))
    if structure_safe:
        alpha = structure_safe_alpha(
            alpha,
            mismatch_score,
            threshold=args.structure_threshold,
            strength=args.structure_strength,
            blur=args.structure_blur,
            floor=args.structure_floor,
        )
        alpha_mode += "+structure_safe"
    final = blend_tiles(styled, upscaled, alpha, args.tile)
    out = out_root / sample
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "styled_original": out / "styled_original.png",
        "upscaled_reference": out / "upscaled_reference.png",
        "alpha": out / "alpha.png",
        "final": out / "final.png",
        "comparison": out / "comparison.jpg",
        "structure_mismatch": out / "structure_mismatch.png",
        "metadata": out / "metadata.json",
    }
    save_png(styled, files["styled_original"])
    save_png(upscaled, files["upscaled_reference"])
    save_png(alpha, files["alpha"])
    save_png(final, files["final"])
    save_png(structure_overlay(proxy_r, mismatch_score), files["structure_mismatch"])
    comparison_sheet((("original", original), ("reference", upscaled), ("manual final", manual), ("demo final", final)), files["comparison"])
    metadata = {
        "sample": sample,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {"original": str(original_path), "reference": str(reference_path), "manual_final": str(manual_path)},
        "input_size": list(original.size), "proxy_size": list(proxy_size), "grid": [args.grid_x, args.grid_y],
        "alignment": {
            "method": alignment_method,
            "proxy_scale": round(float(align_scale), 6),
            "proxy_shift_xy": [shift_x, shift_y],
            "score": round(float(align_score), 6) if align_score is not None else None,
            "override": alignment_override,
            "scale_range": [align_scale_min, align_scale_max, align_scale_step],
            "max_shift": align_max_shift,
        },
        "structure_diagnostic": {
            "method": "proxy normalized luma-edge disagreement after scale+translation alignment",
            **mismatch_stats,
            "path": str(files["structure_mismatch"]),
        },
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
            "structure_safe": {
                "enabled": structure_safe,
                "threshold": args.structure_threshold,
                "strength": args.structure_strength,
                "blur_proxy_px": args.structure_blur,
                "floor": args.structure_floor,
            },
        },
        "fit_trust": {
            "path": str(fit_alpha_path) if fit_alpha_path else None,
            "mode": fit_trust_mode,
        },
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
    p.add_argument("--source-dir", type=Path, help="directory containing the true source JPGs; references remain under root/data")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--proxy-side", type=int, default=768)
    p.add_argument("--align-scale-min", type=float, default=0.97, help="minimum centre scale searched on the proxy")
    p.add_argument("--align-scale-max", type=float, default=1.03, help="maximum centre scale searched on the proxy")
    p.add_argument("--align-scale-step", type=float, default=0.01, help="scale search increment")
    p.add_argument("--align-max-shift", type=int, default=24, help="maximum proxy translation searched in each axis")
    p.add_argument("--grid-x", type=int, default=8)
    p.add_argument("--grid-y", type=int, default=12)
    p.add_argument("--tile", type=int, default=768)
    p.add_argument("--alpha", type=Path, help="optional grayscale mask; resized to original dimensions")
    p.add_argument("--fit-alpha", type=Path, help="optional grayscale mask; fit the appearance operator only on pixels above 50%")
    p.add_argument("--fit-full", action="store_true", help="fit on the full aligned frame even when --alpha is supplied")
    p.add_argument("--alpha-center", type=parse_pair, default=(0.50, 0.53))
    p.add_argument("--alpha-radii", type=parse_pair, default=(0.26, 0.47))
    p.add_argument("--alpha-softness", type=float, default=0.16)
    p.add_argument(
        "--structure-safe",
        action="store_true",
        help="modulate alpha down at proxy edge/shape disagreements (preserves GPT structure)",
    )
    p.add_argument("--structure-threshold", type=float, default=0.28,
                   help="disagreement score below this value leaves alpha unchanged")
    p.add_argument("--structure-strength", type=float, default=0.78,
                   help="maximum alpha reduction at a score of one, 0..1")
    p.add_argument("--structure-blur", type=float, default=1.0,
                   help="Gaussian blur radius in proxy pixels before upsampling")
    p.add_argument("--structure-floor", type=float, default=0.08,
                   help="minimum fraction of base alpha retained in high-disagreement pixels")
    args = p.parse_args(argv)
    root = args.root.resolve()
    out_root = (args.out_dir or root / "outputs").resolve()
    samples = args.sample or sorted(p.stem for p in (root / "data").glob("*.jpg"))
    all_metadata = [run_sample(root, sample, out_root, args) for sample in samples]
    print(json.dumps({"samples": samples, "out_dir": str(out_root), "metadata": all_metadata}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
