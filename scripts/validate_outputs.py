#!/usr/bin/env python3
"""Validate source assets and generated demo outputs.

The validator deliberately samples large images after a small thumbnail resize,
so validating a 60 MP original does not require loading the complete raster into
memory.  It writes a JSON report and a short Markdown report and is safe to run
before or after the demo.

Examples (from the repository root)::

    python scripts/validate_outputs.py
    python scripts/validate_outputs.py --strict
    python scripts/validate_outputs.py --outputs outputs --json outputs/validation.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import numpy as np
    from PIL import Image, UnidentifiedImageError
except ImportError as exc:  # pragma: no cover - gives a useful CLI error
    print(
        "Missing dependency: Pillow and NumPy are required to run this script "
        f"({exc}).",
        file=sys.stderr,
    )
    raise SystemExit(2)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
SOURCE_ASSETS = (
    ("DSC02631", "original", "data/DSC02631.jpg"),
    ("DSC02631", "manual_final_candidate", "data/DSC02631.png"),
    ("DSC02631", "gpt_reference_candidate", "data/DSC02631参考.png"),
    ("DSC02634", "original", "data/DSC02634.jpg"),
    ("DSC02634", "manual_final_candidate", "data/DSC02634.png"),
    ("DSC02634", "gpt_reference_candidate", "data/DSC02634参考.png"),
)


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _ratio(width: int, height: int) -> Optional[float]:
    return width / height if height else None


def _finite(value: float) -> Optional[float]:
    return float(value) if math.isfinite(float(value)) else None


def inspect_image(path: Path, root: Path, sample_size: int) -> Dict[str, Any]:
    """Read metadata and thumbnail statistics for one image."""

    record: Dict[str, Any] = {
        "path": path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else 0,
        "readable": False,
        "errors": [],
    }
    if not path.is_file():
        record["errors"].append("missing file")
        return record

    try:
        # verify() catches truncated JPEG/PNG streams without decoding a full image.
        with Image.open(path) as image:
            record.update(
                {
                    "format": image.format,
                    "mode": image.mode,
                    "width": image.width,
                    "height": image.height,
                    "ratio": _ratio(image.width, image.height),
                    "icc_present": bool(image.info.get("icc_profile")),
                    "has_transparency": "transparency" in image.info or "A" in image.getbands(),
                }
            )
            image.verify()

        # Reopen after verify(); verify() invalidates the image object.
        with Image.open(path) as image:
            thumb = image.convert("RGBA")
            thumb.thumbnail((sample_size, sample_size), Image.Resampling.LANCZOS)
            array = np.asarray(thumb, dtype=np.uint8)
            rgb = array[..., :3]
            alpha = array[..., 3]
            record["sample"] = {
                "width": int(thumb.width),
                "height": int(thumb.height),
                "rgb_min": int(rgb.min()),
                "rgb_max": int(rgb.max()),
                "rgb_mean": [round(float(v), 3) for v in rgb.mean(axis=(0, 1))],
                "luma_mean": round(float((0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).mean()), 3),
                "alpha_min": int(alpha.min()),
                "alpha_max": int(alpha.max()),
                "alpha_mean": round(float(alpha.mean()), 3),
                "alpha_nonzero_fraction": round(float(np.count_nonzero(alpha) / alpha.size), 6),
                "alpha_opaque_fraction": round(float(np.count_nonzero(alpha == 255) / alpha.size), 6),
            }
        record["sha256"] = _sha256(path)
        record["readable"] = True
    except (OSError, ValueError, UnidentifiedImageError, SyntaxError) as exc:
        record["errors"].append(f"{type(exc).__name__}: {exc}")
    return record


def _output_kind(path: Path) -> str:
    name = path.stem.lower()
    if any(token in name for token in ("mask", "alpha", "matte")):
        return "mask"
    if any(token in name for token in ("compare", "comparison", "contact", "grid", "sheet")):
        return "comparison"
    if any(token in name for token in ("reference", "ref", "gpt")):
        return "reference"
    if any(token in name for token in ("composite", "merge", "fused", "result", "styled", "demo", "output")):
        return "composite"
    return "image"


def inspect_outputs(outputs_dir: Path, root: Path, sample_size: int) -> Dict[str, Any]:
    images: List[Dict[str, Any]] = []
    if outputs_dir.is_dir():
        for path in sorted(outputs_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                record = inspect_image(path, root, sample_size)
                record["kind"] = _output_kind(path)
                record["group"] = next((g for g in ("DSC02631", "DSC02634") if g.lower() in path.stem.lower()), None)
                images.append(record)

    unreadable = [item["path"] for item in images if not item["readable"]]
    by_kind: Dict[str, int] = {}
    by_group: Dict[str, int] = {}
    for item in images:
        by_kind[item["kind"]] = by_kind.get(item["kind"], 0) + 1
        if item.get("group"):
            by_group[item["group"]] = by_group.get(item["group"], 0) + 1
    return {
        "directory": outputs_dir.relative_to(root).as_posix() if outputs_dir.is_relative_to(root) else str(outputs_dir),
        "exists": outputs_dir.is_dir(),
        "image_count": len(images),
        "complete": bool(images) and not unreadable,
        "unreadable": unreadable,
        "by_kind": by_kind,
        "by_group": by_group,
        "images": images,
        "note": "No generated images yet; this is expected before run_demo.py." if not images else None,
    }


def build_report(root: Path, outputs_dir: Path, sample_size: int) -> Dict[str, Any]:
    sources: List[Dict[str, Any]] = []
    for group, role, relative in SOURCE_ASSETS:
        item = inspect_image(root / relative, root, sample_size)
        item.update({"group": group, "role": role})
        sources.append(item)

    source_errors = [item["path"] for item in sources if not item["readable"]]
    ratios = [item.get("ratio") for item in sources if item.get("ratio")]
    ratio_spread = (max(ratios) - min(ratios)) if ratios else None
    outputs = inspect_outputs(outputs_dir, root, sample_size)
    warnings: List[str] = []
    if source_errors:
        warnings.append(f"Unreadable or missing source assets: {', '.join(source_errors)}")
    if ratio_spread is not None and ratio_spread > 1e-6:
        warnings.append(f"Source aspect ratios differ (spread={ratio_spread:.8g}); check registration/cropping.")
    if outputs["unreadable"]:
        warnings.append(f"Unreadable generated images: {', '.join(outputs['unreadable'])}")
    if not outputs["image_count"]:
        warnings.append("No generated image outputs found; run the demo before judging output completeness.")

    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "root": str(root),
        "sample_size": sample_size,
        "source_summary": {
            "expected_count": len(SOURCE_ASSETS),
            "found_count": sum(1 for item in sources if item["exists"]),
            "readable_count": sum(1 for item in sources if item["readable"]),
            "all_readable": not source_errors,
            "aspect_ratio_spread": ratio_spread,
        },
        "sources": sources,
        "outputs": outputs,
        "warnings": warnings,
        # A clean pre-demo checkout has zero generated images; surface that as
        # CHECK so the report cannot be mistaken for a successful demo run.
        "ok": not source_errors and not outputs["unreadable"] and bool(outputs["image_count"]),
    }


def markdown_report(report: Dict[str, Any]) -> str:
    lines = [
        "# Demo 输出验证报告",
        "",
        f"生成时间（UTC）：`{report['generated_at']}`",
        "",
        f"源素材：`{report['source_summary']['readable_count']}/{report['source_summary']['expected_count']}` 可读；",
        f"输出图片：`{report['outputs']['image_count']}` 个；整体状态：`{'PASS' if report['ok'] else 'CHECK'}`。",
        "",
        "## 源素材",
        "",
        "| 组别 | 角色 | 文件 | 尺寸 | 比例 | 模式 | 可读 | RGB 均值 | alpha 均值 |",
        "|---|---|---|---:|---:|---|---|---:|---:|",
    ]
    for item in report["sources"]:
        size = f"{item.get('width', '?')} × {item.get('height', '?')}"
        sample = item.get("sample") or {}
        rgb_mean = ", ".join(f"{v:.1f}" for v in sample.get("rgb_mean", [])) or "-"
        alpha_mean = f"{sample['alpha_mean']:.1f}" if "alpha_mean" in sample else "-"
        lines.append(
            f"| {item.get('group', '-')} | {item.get('role', '-')} | `{item['path']}` | {size} | "
            f"{item.get('ratio', 0):.6f} | {item.get('mode', '-')} | "
            f"{'是' if item['readable'] else '否'} | {rgb_mean} | {alpha_mean} |"
        )

    lines += ["", "## 输出文件", "", "| 文件 | 类型 | 组别 | 尺寸 | 可读 | alpha 均值 |", "|---|---|---|---:|---|---:|"]
    for item in report["outputs"]["images"]:
        sample = item.get("sample") or {}
        alpha_mean = f"{sample['alpha_mean']:.1f}" if "alpha_mean" in sample else "-"
        lines.append(
            f"| `{item['path']}` | {item.get('kind', '-')} | {item.get('group') or '-'} | "
            f"{item.get('width', '?')} × {item.get('height', '?')} | "
            f"{'是' if item['readable'] else '否'} | {alpha_mean} |"
        )
    if report["warnings"]:
        lines += ["", "## 检查提示", ""]
        lines.extend(f"- {warning}" for warning in report["warnings"])
    return "\n".join(lines) + "\n"


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Repository root")
    parser.add_argument("--outputs", type=Path, default=None, help="Generated output directory (default: ROOT/outputs)")
    parser.add_argument("--json", dest="json_path", type=Path, default=None, help="JSON report path (default: outputs/validation_report.json)")
    parser.add_argument("--markdown", dest="markdown_path", type=Path, default=None, help="Markdown report path (default: outputs/validation_report.md)")
    parser.add_argument("--sample-size", type=int, default=512, help="Maximum thumbnail edge used for statistics")
    parser.add_argument("--strict", action="store_true", help="Return non-zero when a source/output image is unreadable")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    outputs_dir = (args.outputs or root / "outputs").resolve()
    if args.sample_size < 16:
        raise SystemExit("--sample-size must be at least 16")
    report = build_report(root, outputs_dir, args.sample_size)
    json_path = (args.json_path or outputs_dir / "validation_report.json").resolve()
    markdown_path = (args.markdown_path or outputs_dir / "validation_report.md").resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    print(f"Sources readable: {report['source_summary']['readable_count']}/{report['source_summary']['expected_count']}; outputs: {report['outputs']['image_count']}")
    return 0 if (report["ok"] or not args.strict) else 1


if __name__ == "__main__":
    raise SystemExit(main())
