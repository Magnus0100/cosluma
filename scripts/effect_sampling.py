"""Sample frozen RGBA on global pixel centres, matching full bilinear resize.

Local Pillow box resize recomputes its scale from rounded box endpoints. A
noninteger native enlargement then differs from the full master. This uses
the full canvas ratio for every ROI and preserves Pillow's float32 horizontal
intermediate before the vertical pass. Downscaled review rasters retain the
existing whole-canvas Pillow antialias filter and exact reviewed pixels.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

SAMPLING_ABI = 'global-pixel-centres-bilinear-float32-horizontal-v1'


def sample_frozen_effect(asset, key, box, canvas_size):
    from design_reconstruction import _planes
    path = Path(asset['path'])
    channels = _planes(str(path), path.stat().st_mtime_ns)[key]
    l, t, r, b = map(int, box)
    w, h = map(int, canvas_size)
    if not (0 <= l < r <= w and 0 <= t < b <= h):
        raise ValueError('Effect ROI must remain inside its global canvas')
    sw, sh = channels[0].size
    result = []
    for plane in channels:
        if w < sw or h < sh:
            # Review sizes are small; the whole-image filter is the canonical
            # filter. Slice it rather than introduce local-box phase changes.
            result.append(np.asarray(plane.resize((w, h), Image.Resampling.BILINEAR), np.float32)[t:b, l:r].copy())
            continue
        source = np.asarray(plane, np.float32)
        x = (np.arange(l, r, dtype=np.float64)+.5)*(sw/w)-.5
        y = (np.arange(t, b, dtype=np.float64)+.5)*(sh/h)-.5
        x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
        fx, fy = x-x0, y-y0
        fx[(x0 < 0) | (x0+1 >= sw)] = 0
        fy[(y0 < 0) | (y0+1 >= sh)] = 0
        x1, y1 = np.clip(x0+1, 0, sw-1), np.clip(y0+1, 0, sh-1)
        x0, y0 = np.clip(x0, 0, sw-1), np.clip(y0, 0, sh-1)
        # Bound intermediates by ROI height, including for full-frame replay.
        rows = []
        for start in range(0, b-t, 128):
            stop = min(start+128, b-t)
            def horizontal(indices):
                return (source[indices[:, None], x0[None, :]].astype(np.float64)*(1-fx[None, :])
                    +source[indices[:, None], x1[None, :]].astype(np.float64)*fx[None, :]).astype(np.float32)
            a, c = horizontal(y0[start:stop]), horizontal(y1[start:stop])
            rows.append((a.astype(np.float64)*(1-fy[start:stop, None])
                +c.astype(np.float64)*fy[start:stop, None]).astype(np.float32))
        result.append(np.concatenate(rows))
    return result[0] if len(result) == 1 else np.stack(result, axis=-1)
