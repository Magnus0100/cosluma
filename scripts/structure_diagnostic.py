#!/usr/bin/env python3
"""Write a compact structural-mismatch heatmap and statistics JSON.

The two inputs should already be in the same framing.  This is intentionally a
small, independent entry point for inspecting an existing aligned pair; it
does not alter either image or run the appearance-transfer operator.

Example::

    python scripts/structure_diagnostic.py \
        --source data/DSC02634.jpg \
        --reference data/DSC02634参考.png \
        --out-dir outputs/structure_diagnostic/DSC02634
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Reuse the deterministic Pillow/NumPy implementation from the demo without
# adding another image-processing dependency.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_demo import proxy_arrays, rgb_image, structure_disagreement, structure_overlay  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="aligned source/original image")
    parser.add_argument("--reference", type=Path, required=True, help="aligned generated reference image")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--proxy-side", type=int, default=768)
    args = parser.parse_args(argv)

    source = rgb_image(args.source)
    reference = rgb_image(args.reference)
    proxy_s, proxy_r, proxy_size = proxy_arrays(source, reference, args.proxy_side)
    score, stats = structure_disagreement(proxy_s, proxy_r)
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    heatmap_path = out / "structure_mismatch.png"
    stats_path = out / "structure_mismatch.json"
    structure_overlay(proxy_r, score).save(heatmap_path, format="PNG", compress_level=1)
    record = {
        "source": str(args.source.resolve()),
        "reference": str(args.reference.resolve()),
        "method": "proxy normalized luma-edge disagreement",
        "proxy_size": list(proxy_size),
        "outputs": {"heatmap": str(heatmap_path), "statistics": str(stats_path)},
        **stats,
    }
    stats_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
