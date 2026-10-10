"""Pure display-linear sRGB/OKLab transforms for current appearance modules.

These functions change only the colour representation. They do not fit image
pairs, create masks, own caches, or apply a saved colour-matching contract.
"""
from __future__ import annotations

import numpy as np

M1 = np.array([[.4122214708, .5363325363, .0514459929],
               [.2119034982, .6806995451, .1073969566],
               [.0883024619, .2817188376, .6299787005]], np.float32)
M2 = np.array([[.2104542553, .7936177850, -.0040720468],
               [1.9779984951, -2.4285922050, .4505937099],
               [.0259040371, .7827717662, -.8086757660]], np.float32)
IM1 = np.linalg.inv(M1).astype(np.float32)
IM2 = np.linalg.inv(M2).astype(np.float32)


def to_lab(rgb):
    """Convert display-linear sRGB to OKLab, preserving the current arithmetic."""
    return np.cbrt(np.maximum(np.asarray(rgb, np.float32) @ M1.T, 0)) @ M2.T


def from_lab(lab):
    """Convert OKLab to display-linear sRGB; the caller owns gamut policy."""
    return ((lab @ IM2.T) ** 3) @ IM1.T
