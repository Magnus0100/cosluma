#!/usr/bin/env python3
"""Generate lightweight, editable subject alpha masks for the cosplay samples.

This backend deliberately has no model dependency.  It uses a small set of
normalized polygon control points (one outline per sample), rasterizes at
proxy resolution, and applies a narrow Gaussian feather.  The points are
easy to adjust when a new pose is added; the generated PNG can also be
painted over and passed to ``run_demo.py --alpha``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFilter, ImageOps


RESAMPLE = Image.Resampling

# Normalized x,y outlines, clockwise.  They intentionally include the outer
# hands/arms and crop at the lower frame edge, matching the source photos.
# Keep this table human-editable: adding a pose is just adding another list.
POLYGONS: Dict[str, List[Tuple[float, float]]] = {
    "DSC02631": [
        (0.435, 0.245), (0.490, 0.225), (0.545, 0.235), (0.595, 0.285),
        (0.615, 0.345), (0.670, 0.390), (0.705, 0.490), (0.715, 0.610),
        (0.690, 0.680), (0.655, 0.690), (0.660, 0.815), (0.675, 1.000),
        (0.545, 1.000), (0.515, 0.855), (0.485, 0.775), (0.455, 0.855),
        (0.425, 1.000), (0.295, 1.000), (0.315, 0.820), (0.340, 0.715),
        (0.300, 0.650), (0.275, 0.570), (0.295, 0.475), (0.350, 0.400),
        (0.395, 0.355), (0.410, 0.285),
    ],
    "DSC02634": [
        (0.350, 0.250), (0.450, 0.220), (0.560, 0.230), (0.650, 0.280),
        (0.665, 0.340), (0.670, 0.385), (0.700, 0.345), (0.735, 0.405),
        (0.665, 0.480), (0.645, 0.515), (0.640, 0.640), (0.650, 0.790),
        (0.665, 1.000), (0.535, 1.000), (0.515, 0.855), (0.480, 0.760),
        (0.445, 0.855), (0.425, 1.000), (0.290, 1.000), (0.305, 0.820),
        (0.325, 0.700), (0.285, 0.630), (0.270, 0.555), (0.305, 0.470),
        (0.360, 0.410), (0.400, 0.350),
    ],
}


def source_size(root: Path, sample: str) -> Tuple[int, int]:
    path = root / "data" / f"{sample}.jpg"
    if not path.exists():
        raise FileNotFoundError(path)
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).size


def load_points(path: Path | None) -> Dict[str, List[Tuple[float, float]]]:
    if path is None:
        return POLYGONS
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: [tuple(map(float, p)) for p in v] for k, v in raw.items()}


def polygon_alpha(
    size: Tuple[int, int],
    points: Sequence[Tuple[float, float]],
    proxy_side: int = 1024,
    feather: float = 10.0,
    interior: int = 245,
) -> Image.Image:
    """Rasterize a normalized outline and return an L alpha at ``size``."""
    w, h = size
    scale = min(1.0, float(proxy_side) / max(size))
    small = (max(32, round(w * scale)), max(32, round(h * scale)))
    poly = [(round(x * (small[0] - 1)), round(y * (small[1] - 1))) for x, y in points]
    mask = Image.new("L", small, 0)
    ImageDraw.Draw(mask).polygon(poly, fill=int(interior))
    # Feather in proxy pixels so the final edge remains narrow at 6k input.
    radius = max(0.5, float(feather) * scale)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=radius))
    return mask.resize(size, RESAMPLE.LANCZOS)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", action="append", help="sample id; repeat or omit for both")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--proxy-side", type=int, default=1024)
    parser.add_argument("--feather", type=float, default=32.0, help="final-image feather width in pixels")
    parser.add_argument("--interior", type=int, default=245, help="interior opacity, 0..255")
    parser.add_argument("--points-file", type=Path, help="JSON mapping sample -> [[x,y], ...] normalized points")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    out_root = (args.out_dir or root / "outputs" / "mask_backend").resolve()
    points = load_points(args.points_file)
    samples = args.sample or sorted(points)
    rows = []
    for sample in samples:
        if sample not in points:
            raise KeyError(f"No polygon points for {sample}; use --points-file")
        size = source_size(root, sample)
        alpha = polygon_alpha(size, points[sample], args.proxy_side, args.feather, args.interior)
        path = out_root / sample / "alpha.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        alpha.save(path, format="PNG", compress_level=1)
        rows.append({"sample": sample, "size": list(size), "path": str(path),
                     "backend": "editable_polygon", "points": points[sample],
                     "feather_px": args.feather, "interior": args.interior})
    print(json.dumps({"outputs": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

