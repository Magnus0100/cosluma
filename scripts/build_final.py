#!/usr/bin/env python3
"""Build the production image set for every source/reference pair.

``run_demo.py`` remains the image engine; this entry point supplies the bits
that are easy to get wrong when moving from an experiment to a deliverable:
each sample gets its own mask, alignment is run independently, full-resolution
renders are written, and one manifest records the exact inputs and settings.
No model or network dependency is introduced.

Typical use::

    python scripts/build_final.py --root . --out-dir outputs/final

Masks are selected in this order: ``--alpha-map``, ``--alpha-dir``, the
previously edited mask directories, then the editable polygon backend.  Pass
``--no-auto-mask`` when an explicit mask is required for every sample.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

from PIL import Image, ImageOps

# Import sibling modules when called as a file from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_mask  # noqa: E402
import run_demo  # noqa: E402


def image_storage_info(path: Path) -> Dict:
    """Return storage facts without converting the image to working RGB."""
    with Image.open(path) as im:
        mode = im.mode
        bits = 16 if ("16" in mode or mode in {"I", "I;16", "I;16B", "I;16L"}) else 8
        return {
            "path": str(path),
            "mode": mode,
            "bits_per_channel": bits,
            "icc_profile": bool(im.info.get("icc_profile")),
            "size": list(ImageOps.exif_transpose(im).size),
        }


def embed_source_profile(paths: Iterable[Path], source_path: Path) -> bool:
    """Carry the source camera profile onto RGB deliverables when available.

    The operator works on the source RGB values and does not perform a colour
    space transform, so retaining the input profile is the honest operation.
    Masks remain untagged grayscale images.
    """
    with Image.open(source_path) as source:
        profile = source.info.get("icc_profile")
    if not profile:
        return False
    for path in paths:
        if not path.exists() or path.suffix.lower() != ".png":
            continue
        with Image.open(path) as image:
            if image.mode not in ("RGB", "RGBA"):
                continue
            image.save(path, format="PNG", compress_level=1, icc_profile=profile)
    return True


def load_alpha_map(path: Optional[Path]) -> Dict[str, Path]:
    if path is None:
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("--alpha-map must contain a JSON object mapping sample to mask path")
    return {str(k): Path(str(v)) for k, v in raw.items()}


def find_alpha(root: Path, sample: str, alpha_dir: Optional[Path], alpha_map: Dict[str, Path]) -> Optional[Path]:
    """Resolve a sample-specific mask, accepting either a file or a directory."""
    if sample in alpha_map:
        p = alpha_map[sample]
        return p if p.is_absolute() else (root / p)
    if alpha_dir:
        candidates = [alpha_dir / sample / "alpha.png", alpha_dir / f"{sample}.png"]
        for p in candidates:
            if p.exists():
                return p
    # Edited masks win over generated masks.  These are intentionally explicit
    # conventions so a production run is reproducible from the manifest.
    for d in ("mask_review", "mask_editor", "mask_backend"):
        p = root / "outputs" / d / sample / "alpha.png"
        if p.exists():
            return p
    return None


def auto_polygon_alpha(root: Path, sample: str, out_root: Path) -> Optional[Path]:
    points = make_mask.POLYGONS.get(sample)
    if not points:
        return None
    size = make_mask.source_size(root, sample)
    alpha = make_mask.polygon_alpha(size, points, proxy_side=1024, feather=32.0, interior=245)
    alpha = make_mask.apply_confidence_profile(alpha)
    path = out_root / sample / "_auto_alpha.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    alpha.save(path, format="PNG", compress_level=1)
    return path


def build_args(base: argparse.Namespace, alpha: Optional[Path]) -> argparse.Namespace:
    """Create the small argument object expected by ``run_demo.run_sample``."""
    values = vars(base).copy()
    values["alpha"] = alpha
    values["fit_alpha"] = None
    values["fit_full"] = True
    # run_demo only needs these fields; keeping this explicit prevents a CLI
    # option added to this wrapper from accidentally changing image behavior.
    return argparse.Namespace(**values)


def main(argv: Optional[Iterable[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--out-dir", type=Path, default=None, help="production output directory")
    p.add_argument("--sample", action="append", help="sample id; repeat, or omit for all JPGs")
    p.add_argument("--alpha-dir", type=Path, help="directory containing <sample>/alpha.png")
    p.add_argument("--alpha-map", type=Path, help="JSON object mapping sample id to a mask path")
    p.add_argument("--no-auto-mask", action="store_true", help="fail when no sample-specific mask is found")
    p.add_argument("--proxy-side", type=int, default=768)
    p.add_argument("--align-scale-min", type=float, default=0.97)
    p.add_argument("--align-scale-max", type=float, default=1.03)
    p.add_argument("--align-scale-step", type=float, default=0.01)
    p.add_argument("--align-max-shift", type=int, default=24)
    p.add_argument("--grid-x", type=int, default=8)
    p.add_argument("--grid-y", type=int, default=12)
    p.add_argument("--tile", type=int, default=768)
    p.add_argument("--alpha-center", type=run_demo.parse_pair, default=(0.50, 0.53))
    p.add_argument("--alpha-radii", type=run_demo.parse_pair, default=(0.26, 0.47))
    p.add_argument("--alpha-softness", type=float, default=0.16)
    p.add_argument("--no-structure-safe", dest="structure_safe", action="store_false",
                   help="disable automatic alpha reduction in structural disagreement regions")
    p.set_defaults(structure_safe=True, structure_threshold=0.28, structure_strength=0.78,
                   structure_blur=1.0, structure_floor=0.08)
    args = p.parse_args(argv)
    root = args.root.resolve()
    out_root = (args.out_dir or root / "outputs" / "final").resolve()
    alpha_dir = args.alpha_dir.resolve() if args.alpha_dir else None
    alpha_map = load_alpha_map(args.alpha_map.resolve() if args.alpha_map else None)
    samples = args.sample or sorted(p.stem for p in (root / "data").glob("*.jpg"))
    rows = []
    for sample in samples:
        original, reference, manual = run_demo.find_assets(root, sample)
        alpha = find_alpha(root, sample, alpha_dir, alpha_map)
        alpha_source = "explicit"
        if alpha is None and not args.no_auto_mask:
            alpha = auto_polygon_alpha(root, sample, out_root)
            alpha_source = "auto_polygon" if alpha else "soft_ellipse"
        if alpha is None and args.no_auto_mask:
            raise FileNotFoundError(f"No alpha mask for {sample}; pass --alpha-dir/--alpha-map or omit --no-auto-mask")
        if alpha is not None and not alpha.exists():
            raise FileNotFoundError(f"Mask for {sample} does not exist: {alpha}")

        run = run_demo.run_sample(root, sample, out_root, build_args(args, alpha))
        icc_embedded = embed_source_profile(
            [Path(run["outputs"][k]) for k in ("styled_original", "upscaled_reference", "final")],
            original,
        )
        run["production"] = {
            "entrypoint": "scripts/build_final.py",
            "mask_source": alpha_source,
            "mask_path": str(alpha) if alpha else None,
            "full_resolution": True,
            "working_space": "sRGB 8-bit (Pillow/NumPy)",
            "icc_profile_embedded": icc_embedded,
            "source_storage": image_storage_info(original),
            "reference_storage": image_storage_info(reference),
            "manual_storage": image_storage_info(manual),
            "note": "Source/reference are converted to RGB for the affine operator; 16-bit inputs are reported but are not silently claimed as 16-bit output.",
        }
        Path(run["outputs"]["metadata"]).write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append(run)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "entrypoint": "scripts/build_final.py",
        "samples": samples,
        "out_dir": str(out_root),
        "render": {"full_resolution": True, "alignment": "independent proxy scale+translation per sample", "operator": "run_demo RGB affine"},
        "items": [{"sample": r["sample"], "final": r["outputs"]["final"], "metadata": r["outputs"]["metadata"], "alpha": r["alpha"]} for r in rows],
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
