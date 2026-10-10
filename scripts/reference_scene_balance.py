"""Bounded appearance balance after native colour matching and colour fusion.

The layer stores preweighted EV/ab, not RGB or geometry from the reference.
Preparation uses the current 16-bit native composite only as a frozen opaque
appearance baseline. Rendering applies this layer to straight foreground once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from runtime_resources import code_path

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from reference_colour_fit import to_lab, from_lab, REGIONS

SCHEMA = 'reference_scene_balance.v1'
LIMITS = {'minimum_ev': -.15, 'maximum_ev': .4, 'maximum_ab': .025}
PREPARATION_POLICY = 'reference_supported_macro_balance.v2'
PREPARATION_LIMITS = {'minimum_ev': -.25, 'maximum_ev': .75, 'maximum_ab': .025}
REGION_LIMITS = {'hair': [-.15, .75], 'clothing': [-.20, .35], 'skin': [-.25, .25]}
SDR_AREA_ROUNDOFF = 4 * np.finfo(np.float32).eps


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def _smooth(value):
    value = np.clip(value, 0, 1)
    return value*value*(3-2*value)


def _low(values, mask, sigma):
    support = gaussian_filter(mask.astype(np.float32), sigma)
    shape = (sigma, sigma) + (0,)*(values.ndim-2)
    filtered = gaussian_filter(values*mask[(...,)+(None,)*(values.ndim-2)], shape)
    divisor = np.maximum(support, 1e-5)[(...,)+(None,)*(values.ndim-2)]
    return filtered/divisor, support


def _receipt_limits(receipt):
    """Accept only the current persisted preparation policy."""
    if (receipt.get('preparation_policy') == PREPARATION_POLICY
            and receipt.get('limits') == PREPARATION_LIMITS
            and receipt.get('region_limits') == REGION_LIMITS):
        return PREPARATION_LIMITS
    raise ValueError('Unknown or modified scene preparation limits')


def read_skin_protection(path, size):
    """Read both existing mask layouts without hiding invalid mask data."""
    import cv2
    with np.load(path, allow_pickle=False) as resource:
        name = 'protection' if 'protection' in resource else 'values'
        if name not in resource:
            raise ValueError('Scene skin protection has no supported mask array')
        values = np.asarray(resource[name])
    if (values.ndim != 2 or not values.size or not np.isfinite(values).all()
            or values.min() < 0 or values.max() > 1
            or len(size) != 2 or any(not isinstance(x, (int,np.integer)) or isinstance(x,bool) or x <= 0 for x in size)):
        raise ValueError('Invalid scene skin lightness protection shape or range')
    protection = cv2.resize(values.astype(np.float32), tuple(map(int,size)), interpolation=cv2.INTER_LINEAR)
    if (protection.shape != (size[1],size[0]) or not np.isfinite(protection).all()
            or protection.min() < 0 or protection.max() > 1):
        raise ValueError('Invalid resized scene skin lightness protection')
    return protection


def macro_ev(name, current_lightness, desired_ev, *, skin_protection=None):
    """Only the broad EV field changes; source detail and coordinates remain."""
    if name == 'head_hair':
        low, high = np.quantile(current_lightness, [.4, .85])
        highlight = _smooth((current_lightness-low)/max(high-low, .02))
        return np.clip(desired_ev, -.15, 0)+np.clip(desired_ev, 0, .75)*(.25+.75*highlight)
    if name == 'clothing':
        low, high = np.quantile(current_lightness, [.25, .65])
        dark = 1-_smooth((current_lightness-low)/max(high-low, .02))
        return np.clip(desired_ev, -.20, 0)+np.clip(desired_ev, 0, .35)*dark
    ev = np.clip(desired_ev, -.25, .25)
    if skin_protection is not None:
        ev = ev*(1-.85*np.clip(skin_protection, 0, 1))
    return ev


def excess_warm_suppression(predicted_ab, target_ab, source_high, mask):
    """Warm target highlights are valid; suppress only excess relative to them."""
    excess = predicted_ab-target_ab
    orange = _smooth((excess[..., 1]-.003)/.014)
    red = _smooth((excess[..., 0]-.018)/.014)*_smooth((excess[..., 1]+.004)/.014)
    return .8*gaussian_filter(np.maximum(orange, red)*source_high*mask, 1.)*mask


def _limited_lightness(lightness, ev):
    """Exact midtone gain, with a continuous monotone positive highlight dose."""
    gain = np.exp2(ev/3)
    knee = np.minimum(.7, (2.8-gain)/(1.8*gain))
    knee = np.where(gain > 1, knee, .7)
    t = np.clip((lightness-knee)/(1-knee), 0, 1)
    y0 = gain*knee
    secant = (1-y0)/(1-knee)
    m0, m1 = gain, secant*np.maximum(.4, 1-5*np.abs(gain-1))
    curve = (2*t**3-3*t**2+1)*y0 + (t**3-2*t**2+t)*(1-knee)*m0
    curve += (-2*t**3+3*t**2) + (t**3-t**2)*(1-knee)*m1
    return np.where((ev > 0) & (lightness > knee), curve, lightness*gain)


def apply_scene_balance(values, field, strength=1):
    """Pixelwise operation; field coverage has already been consumed at prepare.

    Gamut conflicts reduce the shared EV/ab dose continuously by bisection.
    No RGB clipping or chroma compression is used, and inactive pixels are exact.
    """
    values, field = np.asarray(values), np.asarray(field)
    if (not 0 <= strength <= 1 or values.ndim != 3 or values.shape[-1] != 3
            or field.shape != (*values.shape[:2], 4)
            or not np.isfinite(values).all() or not np.isfinite(field).all()
            or values.min() < 0 or values.max() > 1):
        raise ValueError('Invalid scene balance input, field or strength')
    result = values.copy()
    active = np.any(field[..., :3] != 0, axis=-1)
    if strength == 0 or not active.any():
        return result
    original = values[active]
    lab = to_lab(original)
    selected = field[active]
    ev, ab = selected[:, 0], selected[:, 1:3]

    def candidate(dose):
        target = lab.copy()
        target[:, 0] = _limited_lightness(lab[:, 0], ev*dose)
        target[:, 1:] += ab*dose[:, None]
        return from_lab(target)

    dose = np.full(len(lab), strength, np.float32)
    proposed = candidate(dose)
    bad = (proposed.min(-1) < 0) | (proposed.max(-1) > 1) | ~np.isfinite(proposed).all(-1)
    if bad.any():
        # Fixed iterations give the same dose in any ROI/full-image execution.
        low, high = dose.copy(), dose.copy()
        low[bad] = 0
        for _ in range(20):
            middle = (low+high)*.5
            trial = candidate(middle)
            valid = np.isfinite(trial).all(-1) & (trial.min(-1) >= 0) & (trial.max(-1) <= 1)
            low = np.where(bad & valid, middle, low)
            high = np.where(bad & ~valid, middle, high)
        dose[bad] = low[bad]
        proposed = candidate(dose)
    fallback = ((dose < 1e-6) | ~np.isfinite(proposed).all(-1)
                | (proposed.min(-1) < 0) | (proposed.max(-1) > 1))
    proposed[fallback] = original[fallback]
    result[active] = proposed
    return result


def _background(plan):
    return next(layer for layer in plan['layers'] if layer['kind'] == 'background_design')['assets']


def _asset_binding(asset):
    return {key: asset.get(key) for key in ('sha256', 'size', 'coordinate_space', 'representation')}


def _binding(parameters, colour_path, plan):
    colour = _read(colour_path)
    assets = _background(plan)
    return {'core_parameters_sha256': _json_sha(parameters),
            'source': parameters['binding']['source'], 'reference': parameters['binding']['reference'],
            'alpha': _asset_binding(plan['source_baseline']['assets']['foreground_alpha']),
            'source_protection': parameters['binding']['source_protection'],
            'mapping': parameters['binding']['mapping'],
            'colour_receipt_sha256': _sha(colour_path),
            'colour_asset_sha256': colour['asset']['sha256'],
            'background': _asset_binding(assets['background']),
            'background_validity': _asset_binding(assets['validity'])}


def validate_scene_layer(receipt_path, parameters, colour_receipt_path, plan,
                         native_size, proxy_size):
    receipt = _read(receipt_path)
    if (receipt.get('schema') != SCHEMA or receipt.get('working_domain') != 'display_linear_sRGB'
            or receipt.get('native_size') != list(native_size)
            or receipt.get('proxy_size') != list(proxy_size)):
        raise ValueError('Scene layer domain, schema or coordinate mismatch')
    limits = _receipt_limits(receipt)
    if receipt['binding'] != _binding(parameters, colour_receipt_path, plan):
        raise ValueError('Scene layer dependency binding changed')
    if receipt['algorithm_sha256'] != _sha(code_path('reference_scene_balance.py')):
        raise ValueError('Scene balance module changed; prepare the layer again')
    colour = _read(colour_receipt_path)
    if _sha(colour['asset']['path']) != colour['asset']['sha256']:
        raise ValueError('Scene balance colour asset changed')
    if _sha(receipt['asset']['path']) != receipt['asset']['sha256']:
        raise ValueError('Scene balance field asset changed')
    provenance = receipt['baseline_provenance']
    if _sha(provenance['receipt_copy_path']) != provenance['receipt_sha256']:
        raise ValueError('Frozen baseline receipt changed')
    # The old master is intentionally not reopened: rendering overwrites it.
    with np.load(receipt['asset']['path'], allow_pickle=False) as resource:
        field, baseline = resource['values'], resource['baseline_rgb']
    shape = (proxy_size[1], proxy_size[0])
    if (field.shape != (*shape, 4) or baseline.shape != (*shape, 3)
            or not np.isfinite(field).all() or not np.isfinite(baseline).all()
            or baseline.min() < 0 or baseline.max() > 1
            or field[..., 0].min() < limits['minimum_ev']-1e-6
            or field[..., 0].max() > limits['maximum_ev']+1e-6
            or np.linalg.norm(field[..., 1:3], axis=-1).max() > limits['maximum_ab']+1e-6
            or field[..., 3].min() < 0 or field[..., 3].max() > 1
            or np.any(field[field[..., 3] == 0, :3] != 0)):
        raise ValueError('Invalid scene balance field or baseline')
    return field, receipt


def _labels(root, identity, size, expected):
    import cv2
    for path in (root/'outputs/cache/local-analysis'/identity[:16]).glob('*/semantics/labels.npz'):
        with np.load(path, allow_pickle=False) as resource:
            labels = cv2.resize(resource['labels'], size, interpolation=cv2.INTER_NEAREST)
        if hashlib.sha256(labels.tobytes()).hexdigest() == expected:
            return labels
    raise ValueError('No cached semantic labels match the current colour receipt')


def _master(path, native):
    import cv2
    from float_appearance import srgb_to_linear
    encoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if encoded is None or encoded.dtype != np.uint16 or encoded.shape != (native[1], native[0], 3):
        raise ValueError('Baseline requires the native uint16 sRGB appearance TIFF')
    linear = encoded[..., ::-1].astype(np.float32)/65535
    del encoded
    for top in range(0, len(linear), 128):
        linear[top:top+128] = srgb_to_linear(linear[top:top+128])
    return linear


def resize_sdr_baseline(values, size):
    """Area-average a proven SDR baseline and normalize endpoint roundoff.

    OpenCV's float32 area weights may sum one ULP above one on a saturated
    white patch. The source must be strictly finite SDR; only tiny excursions
    created by this convex resampling step may be corrected. The pixel
    operator itself retains its strict SDR validation and gamut algorithm.
    """
    import cv2
    values = np.asarray(values)
    if (values.ndim != 3 or values.shape[-1] != 3 or not np.isfinite(values).all()
            or values.min() < 0 or values.max() > 1):
        raise ValueError('Scene baseline resampling requires finite SDR RGB input')
    result = cv2.resize(values, size, interpolation=cv2.INTER_AREA)
    if not np.isfinite(result).all():
        raise ValueError('Scene baseline area resampling produced nonfinite RGB')
    minimum, maximum = float(result.min()), float(result.max())
    if minimum < -SDR_AREA_ROUNDOFF or maximum > 1 + SDR_AREA_ROUNDOFF:
        raise ValueError('Scene baseline area resampling exceeded SDR roundoff tolerance')
    corrected = int(np.count_nonzero((result < 0) | (result > 1)))
    if corrected:
        np.clip(result, 0, 1, out=result)
    return result, {'method': 'INTER_AREA_then_normalize_SDR_endpoint_roundoff',
        'float32_roundoff_tolerance': float(SDR_AREA_ROUNDOFF),
        'before_min': minimum, 'before_max': maximum,
        'corrected_channels': corrected, 'maximum_excursion': max(0., -minimum, maximum - 1),
        'after_min': float(result.min()), 'after_max': float(result.max()),
        'source_required_strict_SDR': True, 'pixel_operator_validation_unchanged': True}


def main():
    import cv2
    from raw_support import open_rgb, linear_preview
    from float_appearance import srgb_to_linear
    from evaluate_reference_colour import sheet, optional_review_boxes
    from render_reference_colour_composite import sample_maps

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--match-dir', type=Path, default=Path('outputs/inspection/reference-match-v2'))
    parser.add_argument('--warm-colour', action='store_true', help='Also correct warm residuals on hair/legs only')
    args = parser.parse_args()
    root = args.root.resolve()
    directory = args.match_dir.resolve()
    run, parameters = _read(args.run), _read(directory/'parameters.json')
    plan = run['options']['appearance_revision']['design_reconstruction_plan']
    native, size = tuple(plan['canvas']['source_oriented_size']), tuple(plan['canvas']['source_proxy_size'])
    colour_path = directory/'reference-colour-layer.json'
    from reference_colour_blend import validate_layer
    _, colour = validate_layer(colour_path, parameters, native, size)
    baseline_receipt_path = directory/'composite-receipt.json'
    baseline_receipt = _read(baseline_receipt_path)
    if (baseline_receipt.get('scene_balance_layer') or baseline_receipt.get('reference_scene_balance')
            or baseline_receipt.get('scene_balance') or baseline_receipt.get('reference_spatial_lightness_used')):
        raise ValueError('Prepare from the pre-scene T+B baseline, not a recursive scene result')
    master = directory/'appearance-composite.tif'
    if (baseline_receipt['output_sha256'] != _sha(master)
            or baseline_receipt['size'] != list(native)
            or baseline_receipt['parameters_sha256'] != _sha(directory/'parameters.json')
            or baseline_receipt['reference_colour_layer']['receipt_sha256'] != _sha(colour_path)):
        raise ValueError('Current native baseline is not bound to current T+B parameters')
    for role in ('source', 'reference'):
        if parameters['binding'][role] != run[role]['sha256'] or _sha(run[role]['path']) != run[role]['sha256']:
            raise ValueError('Source/reference binding mismatch')
    for asset in (plan['source_baseline']['assets']['foreground_alpha'],
                  _background(plan)['background'], _background(plan)['validity']):
        if _sha(asset['path']) != asset['sha256']:
            raise ValueError('Scene baseline alpha/background asset bytes changed')
    frozen_receipt = directory/'scene-balance-baseline-receipt.json'
    frozen_receipt.write_bytes(baseline_receipt_path.read_bytes())
    print('Reading native 16-bit T+B appearance baseline', flush=True)
    native_baseline = _master(master, native)
    baseline, baseline_resampling = resize_sdr_baseline(native_baseline, size)
    print(json.dumps({'baseline_resampling': baseline_resampling}), flush=True)
    with np.load(plan['source_baseline']['assets']['foreground_alpha']['path'], allow_pickle=False) as resource:
        native_alpha = resource['values']
    if (native_alpha.ndim != 2 or not np.isfinite(native_alpha).all()
            or native_alpha.min() < 0 or native_alpha.max() > 1):
        raise ValueError('Invalid source coverage for scene balance preparation')
    alpha = cv2.resize(native_alpha, size, interpolation=cv2.INTER_AREA)
    protection_asset = plan['mask_roles']['identity_protection']
    if (_sha(protection_asset['path']) != parameters['binding']['source_protection']
            or protection_asset['sha256'] != parameters['binding']['source_protection']):
        raise ValueError('Scene skin protection differs from current colour fitting')
    protection = read_skin_protection(protection_asset['path'], size)
    image = open_rgb(Path(run['reference']['path']))
    ref_rgb = srgb_to_linear(np.asarray(image, np.float32)/255)
    sl = _labels(root, run['source']['sha256'], size, colour['binding']['source_labels_sha256'])
    rl = _labels(root, run['reference']['sha256'], image.size, colour['binding']['reference_labels_sha256'])
    H = np.asarray(parameters['binding']['mapping'])
    face_H = np.asarray(colour['face_reference_mapping'])
    aligned = cv2.warpPerspective(ref_rgb, H, size)
    ref_labels = cv2.warpPerspective(rl, H, size, flags=cv2.INTER_NEAREST)
    facial = cv2.warpPerspective(ref_rgb, face_H, size)
    facial_labels = cv2.warpPerspective(rl, face_H, size, flags=cv2.INTER_NEAREST)
    baseline_lab, ref_lab, face_lab = to_lab(baseline), to_lab(aligned), to_lab(facial)
    from reference_automatic_layers import load_maps,scene_field,POLICY
    local_record = colour.get('local_correspondence')
    if not local_record:
        raise ValueError('Automatic scene preparation requires independently bound local correspondence')
    maps = load_maps(local_record,sl.shape,rl.shape,parameters,root=root,
        source_labels=sl,reference_labels=rl)
    print('Solving automatic local lightness and background relationship',flush=True)
    field,reports,automatic_background = scene_field(baseline_lab,to_lab(ref_rgb),sl,rl,
        alpha,protection,maps,local_record)
    # Preview applies only where the composite is effectively opaque; no alpha is multiplied into the field again.
    print(json.dumps({'scene_input': {'baseline_shape':list(baseline.shape),
        'field_shape':list(field.shape),'baseline_finite':bool(np.isfinite(baseline).all()),
        'field_finite':bool(np.isfinite(field).all()),'baseline_min':float(baseline.min()),
        'baseline_max':float(baseline.max())}}), flush=True)
    proposed = apply_scene_balance(baseline, field)
    proposed[alpha < .995] = baseline[alpha < .995]
    person_distance = distance_transform_edt(alpha < .01)
    exterior = (alpha < .01) & (person_distance >= 12) & (person_distance <= 80)
    all_person_ids = sorted({x for ids in REGIONS.values() for x in ids} | set(range(24, 29)) | {2})
    exterior &= ~np.isin(ref_labels, all_person_ids)
    reference_valid = cv2.warpPerspective(np.ones(ref_rgb.shape[:2], np.float32), H, size)
    exterior &= reference_valid >= .999
    bg_checks = {'automatic_relationship': automatic_background, 'background_curve': 'identity', 'nearby_background_pixels': int(exterior.sum()),
                 'physical_light_direction_inferred': False}
    if exterior.any():
        bg_delta = 3*np.log2(np.maximum(ref_lab[..., 0][exterior], .03)/np.maximum(baseline_lab[..., 0][exterior], .03))
        bg_checks['nearby_background_exposure_residual_ev_q10_50_90'] = np.quantile(bg_delta, [.1,.5,.9]).tolist()
        bg_checks['nearby_background_relative_L_ratio_median'] = float(np.median(ref_lab[..., 0][exterior]/np.maximum(baseline_lab[..., 0][exterior], .03)))
    asset_path = directory/'scene-balance-layer.npz'
    owners = sl.astype(np.int32).copy()
    owners[protection >= .5] += 1000
    np.savez_compressed(asset_path, values=field, baseline_rgb=baseline, owners=owners)
    receipt = {'schema': SCHEMA, 'working_domain': 'display_linear_sRGB', 'native_size': list(native),
               'support_policy': 'reference_support.v1',
               'automatic_policy': POLICY, 'local_correspondence_asset_sha256': local_record['asset']['sha256'],
               'proxy_size': list(size), 'binding': _binding(parameters, colour_path, plan),
               'algorithm_sha256': _sha(code_path('reference_scene_balance.py')),
               'asset': {'path': str(asset_path), 'sha256': _sha(asset_path),
                         'representation': 'preweighted_EV_delta_a_delta_b_plus_QA_coverage; coverage_not_reapplied',
                         'baseline_rgb': 'native_uint16_sRGB_T+B_composite_decoded_linear_then_AREA; opaque_donors_alpha>=.995'},
               'baseline_provenance': {'master_path': str(master), 'master_sha256': baseline_receipt['output_sha256'],
                                       'receipt_copy_path': str(frozen_receipt), 'receipt_sha256': _sha(frozen_receipt),
                                       'master_is_provenance_only_after_prepare': True, 'precision': 'uint16_sRGB',
                                       'resampling': baseline_resampling},
               'regions': reports, 'limits': PREPARATION_LIMITS, 'region_limits': REGION_LIMITS,
               'preparation_policy': PREPARATION_POLICY, 'background_checks': bg_checks,
               'protected_skin_lightness_dose': .15, 'skin_chroma_protection_changed': False, 'alpha_modified': False,
               'reference_geometry_used_for_output': False, 'reference_high_frequency_used': False,
               'gamut_policy': 'continuous_monotone_highlight_dose; shared_EV_ab_bisection; no_RGB_clip',
               'application': 'after_T_and_colour_fusion; before_foreground_premultiplication; one_application',
               'preview_opaque_only': True, 'visual_acceptance': False}
    receipt_path = directory/'scene-balance-layer.json'
    receipt_path.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf8')
    validate_scene_layer(receipt_path, parameters, colour_path, plan, native, size)
    sheet([linear_preview(baseline), linear_preview(aligned), linear_preview(proposed)],
          ['Native T+B baseline', 'Registered reference', 'Bounded scene balance (opaque preview)'], 2400).save(
              directory/'scene-balance-review.jpg', quality=97)
    validation_path = directory/'validation.json'
    validation = _read(validation_path) if validation_path.is_file() else {}
    face_box = optional_review_boxes(validation, native).get('face')
    boxes = {'face': face_box} if face_box is not None else {}
    if face_box is None:
        print('Optional native face review skipped: source face crop is unavailable', flush=True)
    yy, xx = np.nonzero(np.isin(sl, REGIONS['clothing']))
    if len(xx):
        pbox = (max(0, xx.min()-10), max(0, yy.min()-10), min(size[0], xx.max()+11), min(size[1], yy.min()+410))
        boxes['clothing'] = [round(v*s) for v,s in zip(pbox,(native[0]/size[0],native[1]/size[1])*2)]
    for name, box in boxes.items():
        x0,y0,x1,y1 = box
        current = native_baseline[y0:y1,x0:x1]
        adjusted = apply_scene_balance(current, sample_maps(field, box, size, native, owners=owners))
        adjusted[native_alpha[y0:y1,x0:x1] < .995] = current[native_alpha[y0:y1,x0:x1] < .995]
        linear_preview(adjusted).save(directory/f'scene-balance-native-{name}.png')
        sheet([linear_preview(current), linear_preview(adjusted)], ['Native baseline', 'Scene balance opaque preview'], 1800).save(
            directory/f'scene-balance-{name}-review.jpg', quality=97)
    print(json.dumps({'receipt': str(receipt_path), 'regions': reports, 'background_checks': bg_checks,
                      'visual_acceptance': False}), flush=True)


if __name__ == '__main__':
    main()
