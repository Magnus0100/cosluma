"""Non-destructive, tiled display-HSL colour edits on scene-linear RGB.

Hue ranges are selected from the incoming colour once. Complementary smooth
weights prevent discontinuities between the eight editable colour families.
HDR intensity and negative linear residuals stay outside the bounded HSL
operation; associated foreground is unassociated and reassociated per tile.
"""
from __future__ import annotations

import numpy as np

SCOPES = ('foreground', 'background', 'composite')
FAMILIES = ('all', 'red', 'orange', 'yellow', 'green', 'aqua', 'blue', 'purple', 'magenta')
CENTRES = np.array([0, 30, 60, 120, 180, 240, 270, 300, 360], np.float32)


def empty_hsl_adjust():
    return {'schema_version': 'hsl_adjust.v1', 'enabled': True, 'scopes': {
        scope: {'enabled': True, 'colours': {
            name: {'hue': 0., 'saturation': 0., 'lightness': 0.} for name in FAMILIES}}
        for scope in SCOPES}}


def _encode(rgb):
    return np.where(rgb <= .0031308, 12.92 * rgb,
                    1.055 * np.power(np.maximum(rgb, 0), 1 / 2.4) - .055)


def _decode(rgb):
    return np.where(rgb <= .04045, rgb / 12.92,
                    np.power(np.maximum((rgb + .055) / 1.055, 0), 2.4))


def _rgb_hsl(rgb):
    hi = rgb.max(axis=-1); lo = rgb.min(axis=-1); delta = hi - lo
    lightness = (hi + lo) * .5
    saturation = np.divide(delta, 1 - np.abs(2 * lightness - 1),
                           out=np.zeros_like(delta), where=delta > 1e-7)
    hue = np.zeros_like(delta); nonzero = delta > 1e-7
    r, g, b = np.moveaxis(rgb, -1, 0)
    red = nonzero & (hi == r); green = nonzero & ~red & (hi == g)
    blue = nonzero & ~red & ~green
    hue[red] = ((g[red] - b[red]) / delta[red]) % 6
    hue[green] = (b[green] - r[green]) / delta[green] + 2
    hue[blue] = (r[blue] - g[blue]) / delta[blue] + 4
    return hue * 60, saturation, lightness


def _hsl_rgb(hue, saturation, lightness):
    chroma = (1 - np.abs(2 * lightness - 1)) * saturation
    sector = (hue % 360) / 60
    x = chroma * (1 - np.abs(sector % 2 - 1))
    zero = np.zeros_like(chroma)
    rgb = np.empty((*chroma.shape, 3), np.float32)
    for index, channels in enumerate(((chroma, x, zero), (x, chroma, zero),
            (zero, chroma, x), (zero, x, chroma), (x, zero, chroma), (chroma, zero, x))):
        mask = np.floor(sector) == index
        for channel, values in enumerate(channels):
            rgb[..., channel][mask] = values[mask]
    return rgb + (lightness - chroma * .5)[..., None]


def _adjust_straight(straight, colours):
    positive = np.maximum(straight, 0)
    negative = np.minimum(straight, 0)
    hdr_scale = np.maximum(positive.max(axis=-1, keepdims=True), 1)
    hue, saturation, lightness = _rgb_hsl(_encode(positive / hdr_scale))
    master = colours.get('all', {})
    dh = np.full_like(hue, float(master.get('hue', 0)))
    ds = np.full_like(hue, float(master.get('saturation', 0)))
    dl = np.full_like(hue, float(master.get('lightness', 0)))
    segment = np.searchsorted(CENTRES, hue, side='right') - 1
    fraction = (hue - CENTRES[segment]) / (CENTRES[segment + 1] - CENTRES[segment])
    fraction = fraction * fraction * (3 - 2 * fraction)
    # Near-neutral pixels have no stable colour family. Master adjustments
    # remain usable, while chromatic family edits fade smoothly to zero.
    chroma_weight = np.minimum(saturation / .08, 1)
    for index, name in enumerate(FAMILIES[1:]):
        settings = colours.get(name, {})
        weight = (np.where(segment == index, 1 - fraction, 0) +
                  np.where((segment + 1) % 8 == index, fraction, 0)) * chroma_weight
        dh += weight * float(settings.get('hue', 0))
        ds += weight * float(settings.get('saturation', 0))
        dl += weight * float(settings.get('lightness', 0))
    saturation = np.clip(saturation * np.maximum(1 + ds / 100, 0), 0, 1)
    amount = np.clip(dl / 100, -1, 1)
    lightness = np.where(amount >= 0, lightness + (1 - lightness) * amount,
                         lightness * (1 + amount))
    return _decode(_hsl_rgb(hue + dh, saturation, lightness)) * hdr_scale + negative


def apply_hsl_adjust(values, layer, *, branch, subject_alpha, premultiplied_alpha=None):
    if not layer.get('enabled', True):
        return values
    result = np.asarray(values, np.float32)
    height = result.shape[0]
    for scope in SCOPES:
        if branch is not None and branch != scope:
            continue
        settings = layer.get('scopes', {}).get(scope, {})
        colours = settings.get('colours', {})
        if not settings.get('enabled', True) or not any(
                value != 0 for colour in colours.values() for value in colour.values()):
            continue
        for top in range(0, height, 128):
            bottom = min(top + 128, height); tile = result[top:bottom]
            alpha = None; straight = tile
            if scope == 'foreground' and premultiplied_alpha is not None:
                alpha = np.asarray(premultiplied_alpha)[top:bottom, :, None]
                straight = np.divide(tile, alpha, out=np.zeros_like(tile), where=alpha > 0)
            adjusted = _adjust_straight(straight, colours)
            if alpha is not None:
                adjusted *= alpha
            if branch is None and scope != 'composite':
                mask = np.asarray(subject_alpha)[top:bottom, :, None]
                if scope == 'background':
                    mask = 1 - mask
                tile[:] += (adjusted - tile) * mask
            else:
                tile[:] = adjusted
    return result
