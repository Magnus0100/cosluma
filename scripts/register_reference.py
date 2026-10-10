#!/usr/bin/env python3
"""Register a generated image as a traceable GPT reference candidate.

This command intentionally stops at the reference stage.  It copies one
generated image into the repository's reference directory, writes provenance
next to it, and never creates a manual final or changes a dataset lock.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageOps


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_info(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        oriented = ImageOps.exif_transpose(image)
        return {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "format": image.format,
            "mode": image.mode,
            "raw_size": [image.width, image.height],
            "oriented_size": [oriented.width, oriented.height],
            "icc_present": bool(image.info.get("icc_profile")),
        }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="generated image to register")
    parser.add_argument("--sample", required=True, help="target sample id, e.g. DSC06657")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out-dir", type=Path, default=Path("data/reference"))
    parser.add_argument("--model", default="gpt-image-2.5-sunburst")
    parser.add_argument("--provider", default="CCSwitch/Sub2API")
    parser.add_argument("--prompt", default="", help="exact generation prompt")
    parser.add_argument("--source", type=Path, help="source image used as generation input, if any")
    parser.add_argument("--generation-id", default=None)
    parser.add_argument("--preserve-alpha", action="store_true", help="preserve real alpha instead of normal RGB registration")
    parser.add_argument("--asset-role", choices=("design_reference", "effect_layer"), default="design_reference")
    parser.add_argument("--force", action="store_true", help="replace an existing candidate and provenance")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    image = args.image if args.image.is_absolute() else (Path.cwd() / args.image)
    image = image.resolve()
    if not image.is_file():
        raise FileNotFoundError(image)
    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{args.sample}.png"
    provenance_path = out_dir / f"{args.sample}.provenance.json"
    if (target.exists() or provenance_path.exists()) and not args.force:
        raise FileExistsError(f"reference candidate exists; use --force to replace: {target}")
    # Normalize the candidate to PNG so the registration convention is stable.
    with Image.open(image) as opened:
        oriented = ImageOps.exif_transpose(opened)
        has_alpha = "A" in oriented.getbands() or "transparency" in oriented.info
        if args.asset_role == "effect_layer":
            if not args.preserve_alpha or not has_alpha:
                raise ValueError("Effect layer requires actual alpha and explicit preservation")
            alpha = oriented.convert("RGBA").getchannel("A")
            minimum, maximum = alpha.getextrema()
            if minimum == 255 or maximum == 0:
                raise ValueError("Effect layer alpha must contain visible and transparent regions")
        normalized = oriented.convert("RGBA" if args.preserve_alpha and has_alpha else "RGB")
        normalized.save(target, format="PNG", compress_level=1)
    source_info = None
    if args.source:
        source = args.source if args.source.is_absolute() else (Path.cwd() / args.source)
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        source_info = {"path": str(source), "sha256": sha256(source), "bytes": source.stat().st_size}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    provenance = {
        "schema_version": "1.0",
        "status": "reference_candidate",
        "sample": args.sample,
        "created_at": now,
        "provider": args.provider,
        "model": args.model,
        "generation_id": args.generation_id,
        "prompt": args.prompt,
        "source": source_info,
        "asset": image_info(target),
        "asset_role": args.asset_role,
        "alpha": {"preservation_requested": args.preserve_alpha,
                  "source_has_alpha": has_alpha, "preserved": normalized.mode == "RGBA"},
        "policy": {
            "manual_final_required": True,
            "review_required": True,
            "does_not_change_dataset_lock": True,
            "unrelated_images_must_not_be_registered": True,
        },
    }
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"reference": str(target), "provenance": str(provenance_path), "status": "reference_candidate"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
