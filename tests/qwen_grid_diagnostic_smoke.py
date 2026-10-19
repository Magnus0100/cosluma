"""Offline smoke for the read-only Qwen grid diagnostic."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from qwen_grid_diagnostic import grid_metrics, periodic_grid_evidence


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="cosplay_qwen_grid_") as temp:
        root = Path(temp)
        height, width = 128, 96
        yy, xx = np.mgrid[:height, :width]
        smooth = np.full((height, width), 96, dtype=np.uint8)
        diagonal = (((xx + yy) % 8) < 4).astype(np.uint8) * 96
        smooth_path = root / "smooth.png"
        diagonal_path = root / "diagonal.png"
        Image.fromarray(smooth, mode="L").save(smooth_path)
        Image.fromarray(diagonal, mode="L").save(diagonal_path)

        smooth_metrics = grid_metrics(smooth_path)
        diagonal_metrics = grid_metrics(diagonal_path)
        assert smooth_metrics["size"] == [width, height]
        assert len(smooth_metrics["sha256"]) == 64
        assert diagonal_metrics["diag_peak_lag_4_16"] > smooth_metrics["diag_peak_lag_4_16"]
        assert smooth_metrics["path"] == str(smooth_path.resolve())
        assert not periodic_grid_evidence(smooth_metrics, threshold=0.02)["corroborated"]
        assert periodic_grid_evidence(diagonal_metrics, threshold=0.02)["corroborated"]
        # Missing neighbor data cannot turn a legacy single-lag score into a
        # newly corroborated periodic failure.
        assert not periodic_grid_evidence({"diag_lag": {"4": 1., "8": 1., "16": 1.}},
                                          threshold=0.02)["corroborated"]
    print(json.dumps({"ok": True, "read_only": True, "synthetic_diagonal_detected": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
