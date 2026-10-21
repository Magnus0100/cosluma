"""Objective acceptance measurements for one face contour shaping mapping.

Current retouching work scores an edit by target attainment, by whether
untouched regions really stay untouched, and by whether resampling destroys
texture; the aesthetic judgement itself stays human.

Strain is reported under named quantities, never as one ambiguous number:
'forward_strain' is the checked bound quantity (the reciprocal area ratio
1/det(ds/dx)), 'principal_stretch' is the local principal stretch of the
forward deformation (central-difference estimate), and
'principal_stretch_envelope' is the conservative one-sided over-estimate kept
for audit. 'unlimited_*' records what the request would have produced without
the amplitude limiter, and 'strain_limited_fraction' records the fraction that
was actually delivered. Those measurements are
computed here from a checked mapping and the base/result pixels. Nothing here
loads a model, writes a run, or claims artistic or human acceptance.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
from scipy.ndimage import gaussian_filter

SCHEMA = 'cosluma_shaping_quality.v1'

PARAMETERS = {
    'texture_sigma': 1.6,               # high-pass scale of resampled skin and hair detail
    'changed_tolerance': 1 / 255.,      # one display step of an eight-bit result
    'support_margin': 1,                # bilinear taps may straddle the declared support
    'texture_retention_min': .80,       # in-band high-frequency energy kept after resampling
    'delta_e_outside_tolerance': 1e-6,  # pixels outside the support must be untouched
    'seam_gradient_ratio_max': 1.6,     # boundary gradient against the untouched ring
    'attainment_min': .90,              # delivered narrowing against the requested target
    'row_change_max': 1.,               # one side control change between adjacent mesh rows over the mesh pitch
}
_ITEMS = ('target_attainment', 'untouched_regions', 'texture_retention', 'tonal_neutrality',
    'boundary_seam', 'geometric_naturalness')


def _rgb(values):
    array = np.asarray(values)
    if array.dtype != np.float32 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError('Shaping quality needs float32 RGB pixels')
    if not np.isfinite(array).all():
        raise ValueError('Shaping quality pixels must be finite')
    return array


def _luma(rgb):
    return rgb @ np.array([.2126, .7152, .0722], np.float32)


def _highpass(grey):
    return grey - gaussian_filter(grey, PARAMETERS['texture_sigma'], mode='nearest')


def _lab(rgb):
    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    matrix = np.array([[.4124, .3576, .1805], [.2126, .7152, .0722], [.0193, .1192, .9505]], np.float32)
    xyz = linear @ matrix.T
    white = np.array([.9505, 1., 1.089], np.float32)
    ratio = xyz / white
    delta = 6 / 29.
    f = np.where(ratio > delta ** 3, np.cbrt(ratio), ratio / (3 * delta ** 2) + 4 / 29.)
    return np.stack((116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])), -1)


def _dilate(mask):
    out = mask.copy()
    for axis in (0, 1):
        out |= np.roll(mask, 1, axis) | np.roll(mask, -1, axis)
    return out


def _erode(mask):
    out = mask.copy()
    for axis in (0, 1):
        out &= np.roll(mask, 1, axis) & np.roll(mask, -1, axis)
    return out


def _gradient(grey):
    vertical = np.zeros_like(grey)
    horizontal = np.zeros_like(grey)
    vertical[1:-1] = .5 * (grey[2:] - grey[:-2])
    horizontal[:, 1:-1] = .5 * (grey[:, 2:] - grey[:, :-2])
    return np.hypot(vertical, horizontal)


def _slice_box(box):
    return (slice(box[1], box[3]), slice(box[0], box[2]))


def _landmark_width(mapping):
    points = np.asarray(mapping['canonical_landmarks'], np.float64)
    return float(points[:, 0].max() - points[:, 0].min()) if points.size else 1.


def measure(mapping, base, shaped):
    """Measure one mapping against the pixels it produced.

    ``mapping`` is a checked ``face_contour_deform`` mapping, ``base`` the
    current unshaped pixels and ``shaped`` the same canvas after sampling.
    Returns a sealed report with per-item values, hard verdicts and separate
    artistic/human acceptance flags that stay false.
    """
    if not isinstance(mapping, dict) or not str(mapping.get('schema', '')).startswith('cosluma_face_contour'):
        raise ValueError('Shaping quality needs a checked face contour mapping')
    stats = mapping.get('stats')
    if not isinstance(stats, dict):
        raise ValueError('Shaping quality needs the mapping statistics')
    base = _rgb(base)
    shaped = _rgb(shaped)
    if base.shape != shaped.shape:
        raise ValueError('Shaping quality needs base and result of one canvas')
    box = mapping['box']
    left, top, right, bottom = box
    support = mapping['edit_support']
    if support.shape != (bottom - top, right - left):
        raise ValueError('Shaping quality support does not match its mapping box')
    support_native = np.zeros(base.shape[:2], bool)
    support_native[top:bottom, left:right] = support
    amount = float(mapping['amount'])
    report = {'schema': SCHEMA, 'amount': amount, 'box': list(box),
        'canvas_size': [int(base.shape[1]), int(base.shape[0])],
        'source_detections_id': mapping['source_detections_id'],
        'mapping_id': mapping['mapping_id'], 'parameters': dict(PARAMETERS),
        'artistic_acceptance': False, 'human_acceptance': False, 'model_inferences': 0}

    difference = np.abs(shaped - base).max(axis=2)
    changed = difference > PARAMETERS['changed_tolerance']
    margin = _dilate(support_native) if PARAMETERS['support_margin'] else support_native
    outside = int(np.count_nonzero(changed & ~margin))
    changed_total = int(np.count_nonzero(changed))
    untouched = {'changed_pixels': changed_total, 'changed_outside_support': outside,
        'changed_fraction_of_canvas': float(changed_total / changed.size),
        'changed_inside_support_fraction': float(changed[support_native].mean()) if support_native.any() else 0.,
        'ok': bool(outside == 0 and changed_total == 0) if amount == 0 else bool(outside == 0 and changed_total > 0)}

    requested = float(stats['target_delta'])
    delivered = float(stats['maximum_control_delta'])
    attainment = {'requested_narrowing': requested, 'delivered_narrowing': delivered,
        'delivered_fraction_of_landmark_width': float(delivered / max(1e-9, _landmark_width(mapping))),
        'delivered_to_requested': float(delivered / requested) if requested else 1.,
        'ok': bool(amount == 0 or (requested > 0 and delivered / requested >= PARAMETERS['attainment_min']))}

    window = _slice_box(box)
    band = support_native[window]
    ring = _dilate(band) & ~band
    base_grey = _luma(base)[window]
    shaped_grey = _luma(shaped)[window]
    base_high = _highpass(base_grey)
    shaped_high = _highpass(shaped_grey)
    energy_base = float(np.abs(base_high[band]).mean()) if band.any() else 0.
    energy_shaped = float(np.abs(shaped_high[band]).mean()) if band.any() else 0.
    retention = energy_shaped / energy_base if energy_base > 1e-6 else 1.
    ring_base = float(np.abs(base_high[ring]).mean()) if ring.any() else 0.
    ring_shaped = float(np.abs(shaped_high[ring]).mean()) if ring.any() else 0.
    texture = {'in_band_high_frequency_ratio': float(retention),
        'untouched_ring_high_frequency_ratio': float(ring_shaped / ring_base) if ring_base > 1e-6 else 1.,
        'ok': bool(amount == 0) if changed_total == 0 else bool(retention >= PARAMETERS['texture_retention_min'])}

    luma_difference = np.abs(_luma(shaped) - _luma(base))
    delta_e = np.abs(_lab(shaped) - _lab(base)).max(axis=2)
    outside_delta_e = float(delta_e[~margin].max(initial=0.)) if (~margin).any() else 0.
    # A pure geometric resample keeps every untouched pixel byte-identical, so
    # any measurable colour difference outside the support is a real defect. The
    # in-band luma change is motion, not a tone shift, and stays informational.
    tonal = {'maximum_luma_shift_in_edit': float(luma_difference[changed].max(initial=0.)) if changed_total else 0.,
        'mean_luma_shift_in_edit': float(luma_difference[changed].mean()) if changed_total else 0.,
        'mean_delta_e_in_edit': float(delta_e[changed].mean()) if changed_total else 0.,
        'maximum_delta_e_outside_support': outside_delta_e,
        'ok': bool(outside_delta_e <= PARAMETERS['delta_e_outside_tolerance'])}

    boundary = band & ~_erode(band)
    if boundary.any() and ring.any():
        inside_gradient = float(_gradient(shaped_grey)[boundary].mean())
        outside_gradient = float(_gradient(shaped_grey)[ring].mean())
        ratio = inside_gradient / outside_gradient if outside_gradient > 1e-6 else 1.
    else:
        ratio = 1.
    seam = {'boundary_to_ring_gradient_ratio': float(ratio),
        'ok': bool(ratio <= PARAMETERS['seam_gradient_ratio_max'])}

    strain_ok = bool(stats['natural_strain_bound_met'])
    # The gated quantity is the control change between adjacent mesh rows of one
    # side, i.e. the row series the deformation actually follows. The same
    # difference read inside one lattice column is kept for audit: it also moves
    # when a handle node shifts by a cell, so it reports placement, not jitter
    # (measured 1.4891 and 1.4884 against this bound before the handle search
    # kept every control on its own row). The delivered field's own adjacent-row
    # change is reported beside them as the quantity a viewer can see.
    row_change = float(stats['maximum_adjacent_face_row_change'])
    # 'forward_strain' is the checked bound quantity: the reciprocal area ratio
    # 1/det(ds/dx). It is not a directional stretch, so the same field is also
    # reported by its principal stretches (central-difference estimate), by the
    # conservative one-sided envelope, and by the values the request would have
    # produced without the amplitude limiter. Reading any of them as the bound
    # would be a different claim, so every one of them states its own quantity.
    naturalness = {'forward_strain': [float(stats['forward_strain_min']), float(stats['forward_strain_max'])],
        'sampling_jacobian': [float(stats['jacobian_min']), float(stats['jacobian_max'])],
        'principal_stretch': [float(stats['principal_stretch_min']), float(stats['principal_stretch_max'])],
        'principal_stretch_envelope': [float(stats['principal_stretch_envelope_min']),
            float(stats['principal_stretch_envelope_max'])],
        'unlimited_forward_strain': [float(stats['unlimited_forward_strain_min']),
            float(stats['unlimited_forward_strain_max'])],
        'unlimited_principal_stretch': [float(stats['unlimited_principal_stretch_min']),
            float(stats['unlimited_principal_stretch_max'])],
        'strain_limited_fraction': float(stats['strain_limited_fraction']),
        'delivered_fraction_of_request': float(stats['delivered_fraction_of_request']),
        'strain_metric': dict(stats['strain_metric']),
        'strain_bound': list(stats['strain_bound']),
        'maximum_adjacent_row_change': row_change,
        'maximum_adjacent_row_change_per_column': float(stats['maximum_adjacent_face_column_change']),
        'maximum_adjacent_field_row_change': float(stats['maximum_adjacent_field_row_change']),
        'row_change_max': float(PARAMETERS['row_change_max']),
        'row_change_metric': dict(stats['row_change_metric']),
        'bound_met': strain_ok,
        'row_change_ok': bool(row_change <= PARAMETERS['row_change_max']),
        'ok': bool(strain_ok and row_change <= PARAMETERS['row_change_max'])}

    items = {'target_attainment': attainment, 'untouched_regions': untouched,
        'texture_retention': texture, 'tonal_neutrality': tonal,
        'boundary_seam': seam, 'geometric_naturalness': naturalness}
    report['items'] = items
    report['failed_items'] = [name for name in _ITEMS if not items[name]['ok']]
    report['verdict'] = 'pass' if not report['failed_items'] else 'fail'
    report['evidence_note'] = ('Measured from one checked mapping and its own pixels. '
        'A pass is not an artistic or human acceptance.')
    report['report_sha256'] = _digest(report)
    return report


def _digest(report):
    body = {key: value for key, value in report.items() if key != 'report_sha256'}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


_FIELDS = frozenset({
    'schema', 'amount', 'box', 'canvas_size', 'source_detections_id', 'mapping_id', 'parameters',
    'artistic_acceptance', 'human_acceptance', 'model_inferences', 'items', 'failed_items',
    'verdict', 'evidence_note', 'report_sha256'})


def validate(report):
    """Re-check one report before anything consumes it."""
    if not isinstance(report, dict) or report.get('schema') != SCHEMA:
        raise ValueError('Complete current shaping quality report required')
    if set(report) != _FIELDS:
        raise ValueError('Shaping quality report lost its checked field set')
    items = report['items']
    if not isinstance(items, dict) or set(items) != set(_ITEMS):
        raise ValueError('Shaping quality report needs all measured items')
    if report['artistic_acceptance'] or report['human_acceptance']:
        raise ValueError('Shaping quality must not claim artistic or human acceptance')
    failed = [name for name in _ITEMS if not items[name]['ok']]
    if failed != list(report['failed_items']) or report['verdict'] != ('fail' if failed else 'pass'):
        raise ValueError('Shaping quality verdict differs from its measured items')
    if _digest(report) != report['report_sha256']:
        raise ValueError('Shaping quality report evidence changed')
    return report
