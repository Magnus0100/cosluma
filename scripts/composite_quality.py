"""Objective acceptance measurements for one finished composite.

Retouching, partial compositing and full compositing all end at the same
question: does the delivered photo still read as one photograph? These
measurements answer the part that can be computed from the parent canvas, the
delivered canvas and the composite's own edit coverage: how hard the join
reads, whether it steps in luminance or colour temperature, whether anything
outside the declared target region moved, whether the generated area kept
photographic high-frequency energy and grain, and how wide a luminance halo
the join leaves behind. The aesthetic judgement stays human; nothing here
loads a model, writes a run, or claims artistic or human acceptance.

The measured pixels are display-encoded sRGB (the delivered preview), because
that is what a reviewer and the export actually see. ``coverage`` is the
composite's per-pixel weight of newly generated content over its immediate
parent: zero outside the write, one where the delivered pixel is entirely
generated, fractional inside a feathered join.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion, convolve, distance_transform_edt, gaussian_filter

SCHEMA = 'cosluma_composite_quality.v1'
WORK_TYPES = ('retouch', 'partial_composite', 'full_composite')

PARAMETERS = {
    'changed_tolerance': 1 / 255.,       # one display step of an eight-bit delivery
    'region_margin': 1,                  # bilinear taps may straddle the declared region
    'highpass_sigma': 1.6,               # scale of skin, cloth and scene micro-detail
    'seam_gradient_floor': 1 / 255.,     # a join below one display step is not a seam
    'seam_gradient_ratio_max': 1.6,      # join gradient against the parent's own continuation
    'tonality_profile_depth': 6,         # shells compared on each side of the join
    'tonality_step_max': 2 / 255.,       # excess per-pixel luminance step across the join
    'warmth_step_max': .9,               # excess per-pixel CIE b* step across the join
    'generated_alpha_min': .9,           # delivered pixels this composite generated
    'original_alpha_max': .02,           # delivered pixels this composite left to its parent
    'original_fraction_min': .02,        # too few untouched pixels -> compare with the parent
    'detail_ratio_min': .55,             # generated detail against original detail
    'detail_ratio_max': 2.2,
    'noise_ratio_max': 1.8,              # generated grain against original grain
    'halo_smoothing': 2.,                # low-frequency tone of the join
    'halo_visible': 2 / 255.,            # a rim changes the join's tone by this much
    'halo_search': 24,                   # deepest ring still treated as part of the join
    'halo_width_max': 2,                 # visible rim thickness in pixels
}
_ITEMS = ('untouched_regions', 'boundary_seam', 'cross_boundary_tonality',
    'generated_detail', 'boundary_halo')
_FIELDS = frozenset({'schema', 'work_type', 'label', 'canvas_size', 'region', 'coverage',
    'inputs', 'parameters', 'items', 'not_applicable_items', 'failed_items', 'verdict',
    'artistic_acceptance', 'human_acceptance', 'model_inferences', 'evidence_note',
    'report_sha256'})


def _rgb(values, name):
    array = np.asarray(values)
    if array.dtype == np.uint8 or array.dtype == np.uint16:
        array = array.astype(np.float32) / np.float32(255. if array.dtype == np.uint8 else 65535.)
    if array.dtype != np.float32 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f'Composite quality needs decoded {name} pixels as float32 RGB')
    if not np.isfinite(array).all():
        raise ValueError(f'Composite quality {name} pixels must be finite')
    if array.min() < 0. or array.max() > 1.:
        raise ValueError(f'Composite quality {name} pixels must be display-encoded sRGB in 0..1')
    return np.ascontiguousarray(array)


def _luma(rgb):
    return rgb @ np.array([.2126, .7152, .0722], np.float32)


def _lab(rgb):
    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    matrix = np.array([[.4124, .3576, .1805], [.2126, .7152, .0722], [.0193, .1192, .9505]], np.float32)
    xyz = linear @ matrix.T
    white = np.array([.9505, 1., 1.089], np.float32)
    ratio = xyz / white
    delta = 6 / 29.
    f = np.where(ratio > delta ** 3, np.cbrt(ratio), ratio / (3 * delta ** 2) + 4 / 29.)
    return np.stack((116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])), -1)


def _gradient(grey):
    vertical = np.zeros_like(grey)
    horizontal = np.zeros_like(grey)
    vertical[1:-1] = .5 * (grey[2:] - grey[:-2])
    horizontal[:, 1:-1] = .5 * (grey[:, 2:] - grey[:, :-2])
    return np.hypot(vertical, horizontal)


def _dilate(mask):
    out = mask.copy()
    for axis in (0, 1):
        out |= np.roll(mask, 1, axis) | np.roll(mask, -1, axis)
    return out


def _noise(grey, mask):
    """Immerkaer grain estimate: one Laplacian response, robust to content."""
    kernel = np.array([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], np.float32)
    response = np.abs(convolve(grey, kernel, mode='reflect'))
    return float(np.sqrt(np.pi / 2.) / 6. * response[mask].mean())


def _shell_median(field, mask):
    return float(np.median(field[mask])) if np.any(mask) else None


def _shells(field, covered, inside, outside, depth):
    """Median tone in unit-width shells walking away from the join on both sides."""
    inward = []
    for step in range(1, depth + 1):
        inward.append(_shell_median(field, covered & (inside >= step) & (inside < step + 1)))
    outward = []
    for step in range(depth, 0, -1):
        outward.append(_shell_median(field, (~covered) & (outside >= step) & (outside < step + 1)))
    return outward, inward


def _join_step(field, covered, inside, outside, depth):
    """Per-pixel step across the join against the content's own per-pixel slope.

    A continuous fade changes the tone smoothly through the feather, so the
    difference between the first shells on each side matches the surrounding
    slopes. A visible jump exceeds them.
    """
    if depth < 2:
        return None
    outward, inward = _shells(field, covered, inside, outside, depth)
    if inward[0] is None or outward[-1] is None:
        return None
    slopes = [abs(right - left) for left, right in zip(inward, inward[1:])
              if left is not None and right is not None]
    slopes += [abs(right - left) for left, right in zip(outward, outward[1:])
               if left is not None and right is not None]
    if not slopes:
        return None
    step = abs(inward[0] - outward[-1])
    return step, float(np.median(slopes))


def _halo(result_grey, parent_grey, covered, inside, outside, deepest):
    """Thickness of a luminance rim that no continuous join could produce.

    The smoothed delivered tone has to travel from the untouched background
    level to the level of the generated interior, so every shell of a
    continuous join lies inside that interval. A halo is a shell that leaves
    the interval by a visible amount; its width is how deep into the generated
    area the excursion still reaches.
    """
    offset = gaussian_filter(result_grey - parent_grey, PARAMETERS['halo_smoothing'], mode='nearest')
    if deepest < 3:
        return None
    interior = covered & (inside >= deepest)
    if np.count_nonzero(interior) < 16:
        interior = covered & (inside >= max(2, deepest // 2))
    if np.count_nonzero(interior) < 16:
        interior = covered & (inside >= 1)
    level = float(np.median(offset[interior]))
    outside_level = _shell_median(offset, (~covered) & (outside >= 1) & (outside < 2))
    outside_level = 0. if outside_level is None else outside_level
    high, low = max(level, outside_level), min(level, outside_level)
    width, visible = 0, 0.
    for step in range(1, deepest + 1):
        value = _shell_median(offset, covered & (inside >= step) & (inside < step + 1))
        if value is None:
            continue
        excess = max(0., value - high, low - value)
        visible = max(visible, excess)
        if excess > PARAMETERS['halo_visible']:
            width = step
    return level, float(visible), int(width)


def _digest(report):
    body = {key: value for key, value in report.items() if key != 'report_sha256'}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode('utf-8')).hexdigest()


def _pixel_digest(array):
    body = np.ascontiguousarray(array, np.float32)
    hasher = hashlib.sha256()
    hasher.update(f'{body.shape[0]}x{body.shape[1]}x{body.shape[2]}:'.encode())
    hasher.update(body.tobytes())
    return hasher.hexdigest()


def _coverage(values, shape):
    if values is None:
        return np.ones(shape, np.float32)
    if isinstance(values, dict) and set(values) == {'uniform'} and values['uniform'] in (0, 1):
        values = values['uniform']
    if type(values) is int and values in (0, 1):
        return np.full(shape, values, np.float32)
    array = np.asarray(values)
    if array.dtype not in (np.float16, np.float32, np.float64) or array.shape != shape:
        raise ValueError('Composite coverage must be one float weight per delivered pixel')
    array = array.astype(np.float32)
    if not np.isfinite(array).all() or array.min() < 0. or array.max() > 1.:
        raise ValueError('Composite coverage weights must be finite and inside 0..1')
    return array


def _region(values, shape):
    if values is None:
        return None
    if (not isinstance(values, (list, tuple)) or len(values) != 4
            or any(type(value) is not int for value in values)):
        raise ValueError('Composite region needs four integer native pixel coordinates')
    left, top, right, bottom = values
    height, width = shape
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError('Composite region lies outside or reverses the delivered canvas')
    if (right - left) < 8 or (bottom - top) < 8:
        raise ValueError('Composite region is too small to carry a join')
    return [left, top, right, bottom]


def measure(parent, result, *, work_type, coverage=None, region=None, label=''):
    """Measure one finished composite against the canvas it was built from.

    ``parent`` is the immediate parent canvas the composite started from,
    ``result`` the delivered canvas, ``coverage`` this composite's per-pixel
    generated weight and ``region`` the authorized target box in native
    pixels. Returns a sealed report with per-item values and hard verdicts;
    artistic and human acceptance stay false because they are not measured.
    """
    if work_type not in WORK_TYPES:
        raise ValueError('Composite quality needs a known editing work type')
    if not isinstance(label, str) or len(label) > 200:
        raise ValueError('Composite quality label must be short text')
    parent = _rgb(parent, 'parent')
    result = _rgb(result, 'result')
    if parent.shape != result.shape:
        raise ValueError('Composite quality needs parent and result of one canvas')
    height, width = parent.shape[:2]
    region = _region(region, (height, width))
    coverage = _coverage(coverage, (height, width))
    if work_type == 'full_composite' and region is not None and list(region) != [0, 0, width, height]:
        raise ValueError('A full composite writes the whole canvas, not a bounded region')
    if work_type != 'full_composite' and (region is None or list(region) == [0, 0, width, height]):
        raise ValueError('A retouch or partial composite needs a bounded target region')
    if float(coverage.max()) <= 0.:
        raise ValueError('Composite quality needs generated pixels inside its coverage')
    region_mask = np.ones((height, width), bool) if region is None else np.zeros((height, width), bool)
    if region is not None:
        region_mask[region[1]:region[3], region[0]:region[2]] = True
    permitted = _dilate(region_mask) if PARAMETERS['region_margin'] else region_mask
    if float(coverage[~permitted].max(initial=0.)) > 1e-6:
        raise ValueError('Composite coverage reaches outside its declared target region')
    if np.array_equal(parent, result):
        raise ValueError('Composite quality needs delivered pixel changes')

    difference = np.abs(result - parent).max(axis=2)
    changed = difference > PARAMETERS['changed_tolerance']
    outside = int(np.count_nonzero(changed & ~permitted))
    written = _dilate(coverage > 0.) if PARAMETERS['region_margin'] else coverage > 0.
    under_zero = int(np.count_nonzero(changed & ~written))
    untouched = {'changed_pixels': int(np.count_nonzero(changed)),
        'changed_outside_region': outside,
        'changed_under_zero_coverage': under_zero,
        'changed_fraction_of_canvas': float(np.count_nonzero(changed) / changed.size),
        'region_fraction_of_canvas': float(region_mask.mean()),
        'scope_covers_canvas': bool(region is None or region == [0, 0, width, height]),
        'ok': bool(outside == 0 and under_zero == 0)}

    covered = coverage >= .5
    seam_available = bool(covered.any() and not covered.all())
    parent_grey = _luma(parent)
    result_grey = _luma(result)
    if seam_available:
        eight = np.ones((3, 3), bool)
        seam_in = covered & ~binary_erosion(covered, eight, border_value=1)
        seam_out = (~covered) & ~binary_erosion(~covered, eight, border_value=1)
        band = seam_in | seam_out
        join_gradient = float(_gradient(result_grey)[band].mean())
        parent_gradient = float(_gradient(parent_grey)[band].mean())
        ratio = join_gradient / max(parent_gradient, PARAMETERS['seam_gradient_floor'])
        seam = {'applicable': True, 'join_gradient': join_gradient,
            'parent_gradient': parent_gradient, 'gradient_ratio': float(ratio),
            'ok': bool(ratio <= PARAMETERS['seam_gradient_ratio_max'])}
        inside = distance_transform_edt(covered)
        outside_distance = distance_transform_edt(~covered)
        depth = int(min(PARAMETERS['tonality_profile_depth'], inside.max(), outside_distance.max()))
        luma_step = _join_step(result_grey, covered, inside, outside_distance, depth)
        warmth = _lab(result)[..., 2]
        warmth_step = _join_step(warmth, covered, inside, outside_distance, depth)
        tonality = {'applicable': bool(luma_step is not None and warmth_step is not None)}
        if tonality['applicable']:
            tonality.update(luma_step=float(luma_step[0]),
                luma_content_step=float(luma_step[1]),
                luma_excess=float(luma_step[0] - luma_step[1]),
                warmth_step=float(warmth_step[0]),
                warmth_content_step=float(warmth_step[1]),
                warmth_excess=float(warmth_step[0] - warmth_step[1]))
            tonality['ok'] = bool(abs(tonality['luma_excess']) <= PARAMETERS['tonality_step_max']
                and abs(tonality['warmth_excess']) <= PARAMETERS['warmth_step_max'])
        else:
            tonality.update(luma_step=0., luma_content_step=0., luma_excess=0.,
                warmth_step=0., warmth_content_step=0., warmth_excess=0., ok=True)
        halo_depth = int(min(PARAMETERS['halo_search'], inside.max()))
        halo = _halo(result_grey, parent_grey, covered, inside, outside_distance, halo_depth)
        if halo is None:
            boundary_halo = {'applicable': False, 'deep_tone_offset': 0.,
                'visible_excess': 0., 'halo_width': 0, 'ok': True}
        else:
            boundary_halo = {'applicable': True, 'deep_tone_offset': float(halo[0]),
                'visible_excess': float(halo[1]), 'halo_width': int(halo[2]),
                'ok': bool(halo[2] <= PARAMETERS['halo_width_max'])}
    else:
        seam = {'applicable': False, 'join_gradient': 0., 'parent_gradient': 0.,
            'gradient_ratio': 1., 'ok': True}
        tonality = {'applicable': False, 'luma_step': 0., 'luma_content_step': 0.,
            'luma_excess': 0., 'warmth_step': 0., 'warmth_content_step': 0.,
            'warmth_excess': 0., 'ok': True}
        boundary_halo = {'applicable': False, 'deep_tone_offset': 0.,
            'visible_excess': 0., 'halo_width': 0, 'ok': True}

    generated = coverage >= PARAMETERS['generated_alpha_min']
    original = coverage <= PARAMETERS['original_alpha_max']
    if int(np.count_nonzero(generated)) < 16:
        raise ValueError('Composite quality needs a generated area to measure detail')
    highpass = result_grey - gaussian_filter(result_grey, PARAMETERS['highpass_sigma'], mode='nearest')
    highpass_parent = parent_grey - gaussian_filter(parent_grey, PARAMETERS['highpass_sigma'], mode='nearest')
    original_minimum = max(2000, int(PARAMETERS['original_fraction_min'] * changed.size))
    if int(np.count_nonzero(original)) >= original_minimum:
        basis = 'untouched_original_pixels'
        generated_detail = float(np.abs(highpass[generated]).mean())
        original_detail = float(np.abs(highpass[original]).mean())
        generated_noise = _noise(result_grey, generated)
        original_noise = _noise(result_grey, original)
    else:
        basis = 'immediate_parent_pixels'
        generated_detail = float(np.abs(highpass[generated]).mean())
        original_detail = float(np.abs(highpass_parent[generated]).mean())
        generated_noise = _noise(result_grey, generated)
        original_noise = _noise(parent_grey, generated)
    detail_ratio = generated_detail / original_detail if original_detail > 1e-9 else 1.
    noise_ratio = generated_noise / original_noise if original_noise > 1e-9 else 1.
    # The photograph's own detail in exactly this region, recorded beside the
    # absolute ratio so a content-limited region is visible as a measurement
    # instead of a claim. It does not relax the threshold.
    parent_region_detail = float(np.abs(highpass_parent[generated]).mean())
    detail = {'basis': basis,
        'generated_fraction_of_canvas': float(generated.mean()),
        'original_fraction_of_canvas': float(original.mean()),
        'generated_high_frequency': generated_detail,
        'original_high_frequency': original_detail,
        'high_frequency_ratio': float(detail_ratio),
        'generated_noise': generated_noise, 'original_noise': original_noise,
        'noise_ratio': float(noise_ratio),
        'parent_region_high_frequency': parent_region_detail,
        'parent_self_similarity': float(parent_region_detail / original_detail
            if original_detail > 1e-9 else 1.),
        'observed_over_parent_region': float(generated_detail / parent_region_detail
            if parent_region_detail > 1e-9 else 1.),
        'ok': bool(PARAMETERS['detail_ratio_min'] <= detail_ratio <= PARAMETERS['detail_ratio_max']
            and noise_ratio <= PARAMETERS['noise_ratio_max'])}

    items = {'untouched_regions': untouched, 'boundary_seam': seam,
        'cross_boundary_tonality': tonality, 'generated_detail': detail,
        'boundary_halo': boundary_halo}
    report = {'schema': SCHEMA, 'work_type': work_type, 'label': label,
        'canvas_size': [int(width), int(height)],
        'region': None if region is None else [int(value) for value in region],
        'coverage': _coverage_record(coverage),
        'inputs': {'parent_sha256': _pixel_digest(parent),
            'result_sha256': _pixel_digest(result)},
        'parameters': dict(PARAMETERS), 'items': items,
        'not_applicable_items': [name for name in _ITEMS if not items[name].get('applicable', True)],
        'failed_items': [name for name in _ITEMS if not items[name]['ok']],
        'artistic_acceptance': False, 'human_acceptance': False, 'model_inferences': 0,
        'evidence_note': ('Measured from one parent canvas, one delivered canvas and the '
            "composite's own edit coverage. A pass is not an artistic or human acceptance.")}
    report['verdict'] = 'fail' if report['failed_items'] else 'pass'
    report['report_sha256'] = _digest(report)
    return report


def _coverage_record(coverage):
    if float(coverage.min()) == 1. and float(coverage.max()) == 1.:
        return {'uniform': 1}
    if float(coverage.min()) == 0. and float(coverage.max()) == 0.:
        return {'uniform': 0}
    return {'sha256': _pixel_digest(coverage[..., None]),
        'mean': float(coverage.mean()),
        'generated_fraction': float((coverage >= PARAMETERS['generated_alpha_min']).mean()),
        'original_fraction': float((coverage <= PARAMETERS['original_alpha_max']).mean())}


def _expected_ok(name, item):
    if name == 'untouched_regions':
        return bool(item['changed_outside_region'] == 0 and item['changed_under_zero_coverage'] == 0)
    if name == 'boundary_halo':
        return True if not item['applicable'] else bool(item['halo_width'] <= PARAMETERS['halo_width_max'])
    if name == 'boundary_seam':
        return True if not item['applicable'] else bool(
            item['gradient_ratio'] <= PARAMETERS['seam_gradient_ratio_max'])
    if name == 'cross_boundary_tonality':
        return True if not item['applicable'] else bool(
            abs(item['luma_excess']) <= PARAMETERS['tonality_step_max']
            and abs(item['warmth_excess']) <= PARAMETERS['warmth_step_max'])
    return bool(PARAMETERS['detail_ratio_min'] <= item['high_frequency_ratio']
        <= PARAMETERS['detail_ratio_max'] and item['noise_ratio'] <= PARAMETERS['noise_ratio_max'])


def validate(report):
    """Re-check one report before anything consumes it."""
    if not isinstance(report, dict) or report.get('schema') != SCHEMA:
        raise ValueError('Complete current composite quality report required')
    if set(report) != _FIELDS:
        raise ValueError('Composite quality report lost its checked field set')
    if report['work_type'] not in WORK_TYPES or not isinstance(report['label'], str):
        raise ValueError('Composite quality report needs a known work type and label')
    if report['artistic_acceptance'] or report['human_acceptance']:
        raise ValueError('Composite quality must not claim artistic or human acceptance')
    if report['model_inferences'] != 0:
        raise ValueError('Composite quality report must record zero model inferences')
    if report['parameters'] != PARAMETERS:
        raise ValueError('Composite quality report used different measured thresholds')
    items = report['items']
    if not isinstance(items, dict) or set(items) != set(_ITEMS):
        raise ValueError('Composite quality report needs all measured items')
    for name in _ITEMS:
        item = items[name]
        if not isinstance(item, dict) or type(item.get('applicable', True)) is not bool:
            raise ValueError('Composite quality item lacks its applicability')
        try:
            expected = _expected_ok(name, item)
        except (KeyError, TypeError):
            raise ValueError('Composite quality item lacks its measured values') from None
        if item.get('ok') is not expected:
            raise ValueError('Composite quality item verdict differs from its measurement')
    # The parent-region basis is recomputed from the numbers the report itself
    # carries, so a content-limited region is a measurement, not a claim.
    detail_item = items['generated_detail']
    try:
        generated = float(detail_item['generated_high_frequency'])
        reference = float(detail_item['original_high_frequency'])
        retained = float(detail_item['parent_region_high_frequency'])
        recorded = (float(detail_item['high_frequency_ratio']),
            float(detail_item['parent_self_similarity']),
            float(detail_item['observed_over_parent_region']))
    except (KeyError, TypeError, ValueError):
        raise ValueError('Composite quality detail item lacks its parent basis') from None
    recomputed = (generated / reference if reference > 1e-9 else 1.,
        retained / reference if reference > 1e-9 else 1.,
        generated / retained if retained > 1e-9 else 1.)
    for value, expected in zip(recorded, recomputed):
        if abs(value - expected) > 1e-9 * max(1., abs(expected)):
            raise ValueError('Composite quality detail ratio differs from its own measurements')
    failed = [name for name in _ITEMS if not items[name]['ok']]
    if failed != list(report['failed_items']) or report['verdict'] != ('fail' if failed else 'pass'):
        raise ValueError('Composite quality verdict differs from its measured items')
    absent = [name for name in _ITEMS if not items[name].get('applicable', True)]
    if absent != list(report['not_applicable_items']):
        raise ValueError('Composite quality applicability differs from its measured items')
    if _digest(report) != report['report_sha256']:
        raise ValueError('Composite quality report evidence changed')
    return report


def load_canvas(path):
    """Decode one delivered or parent raster exactly as production decodes it."""
    from raw_support import open_rgb
    with open_rgb(Path(path)) as image:
        return _rgb(np.asarray(image, np.float32) / np.float32(255.), Path(path).name)


def load_coverage(path):
    """Read one stored edit coverage as float weights in display order."""
    path = Path(path)
    if path.suffix.casefold() == '.npy':
        values = np.load(path, allow_pickle=False)
    else:
        from raw_support import open_rgb
        with open_rgb(path) as image:
            values = np.asarray(image, np.float32).mean(axis=2) / np.float32(255.)
    return np.asarray(values, np.float32)


def _region_argument(text):
    parts = text.split(',')
    if len(parts) != 4:
        raise argparse.ArgumentTypeError('Region needs x0,y0,x1,y1 in native pixels')
    try:
        values = [int(part) for part in parts]
    except ValueError as error:
        raise argparse.ArgumentTypeError('Region needs four integers') from error
    return values


def main(argv=None):
    """Measure one finished composite from real files; 0 pass, 1 fail, 2 invalid."""
    parser = argparse.ArgumentParser(description='Objective composite quality measurements')
    parser.add_argument('--parent', type=Path, required=True, help='Canvas before this composite')
    parser.add_argument('--result', type=Path, required=True, help='Delivered canvas after it')
    parser.add_argument('--coverage', type=Path,
        help='Stored edit coverage (.npy or grayscale raster); omit for whole-canvas replacement')
    parser.add_argument('--work-type', choices=WORK_TYPES, required=True)
    parser.add_argument('--region', type=_region_argument, help='Authorized x0,y0,x1,y1 native pixels')
    parser.add_argument('--label', default='', help='Short identity of the measured attempt')
    parser.add_argument('--out', type=Path, help='Write the sealed report here')
    args = parser.parse_args(argv)
    try:
        report = measure(load_canvas(args.parent), load_canvas(args.result),
            work_type=args.work_type,
            coverage=None if args.coverage is None else load_coverage(args.coverage),
            region=args.region, label=args.label)
    except (ValueError, OSError) as error:
        print(json.dumps({'ok': False, 'error': str(error)}, ensure_ascii=False))
        return 2
    text = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + '\n', encoding='utf-8')
    print(text)
    return 0 if report['verdict'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
