"""Derive independent controls from owned native premultiplied appearance.

OKLab(alpha * RGB) = cbrt(alpha) * OKLab(RGB).  Both photographs share
coverage, so exchanging its L or ab requires neither unpremultiplication nor
models.  Only native corner preparation is nonlinear; replay is a convex RGB
corner bank sampled identically for full, ROI and reduced previews.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

SCHEMA = 'reference_appearance_stack.v2'
CONTROL_RECIPE = 'premultiplied_oklab_four_corner.v1'
RECEIPT_SCHEMA = 'reference_appearance_controls.v1'
CONTROL_ROLES = ('colour_delta', 'lightness_delta')
M1 = np.array([[.4122214708, .5363325363, .0514459929],
               [.2119034982, .6806995451, .1073969566],
               [.0883024619, .2817188376, .6299787005]], np.float64)
M2 = np.array([[.2104542553, .7936177850, -.0040720468],
               [1.9779984951, -2.4285922050, .4505937099],
               [.0259040371, .7827717662, -.8086757660]], np.float64)
IM1, IM2 = np.linalg.inv(M1), np.linalg.inv(M2)


def to_lab(values):
    return np.cbrt(np.asarray(values, np.float64) @ M1.T) @ M2.T


def from_lab(values):
    return ((np.asarray(values, np.float64) @ IM2.T) ** 3) @ IM1.T


def _legal(values, alpha):
    return np.isfinite(values).all(-1) & (values >= 0).all(-1) & (values <= alpha[..., None]).all(-1)


def corner_targets(source, target, alpha):
    """Bound the two mixed Lab corners, preserving source/full RGB endpoints.

    If an existing endpoint is outside SDR coverage gamut, the two new corners
    retain source there.  The existing full target is never silently repaired.
    Along valid pixels only the requested L or ab dose is reduced toward source.
    """
    source, target, alpha = np.asarray(source, np.float32), np.asarray(target, np.float32), np.asarray(alpha, np.float32)
    if source.shape != target.shape or source.shape != (*alpha.shape, 3):
        raise ValueError('Native control source/target/alpha shapes differ')
    if (not np.isfinite(source).all() or not np.isfinite(target).all()
            or not np.isfinite(alpha).all() or np.any(alpha < 0) or np.any(alpha > 1)):
        raise ValueError('Native control inputs must be finite with alpha between zero and one')
    sl, tl = to_lab(source), to_lab(target)
    supported = _legal(source, alpha) & _legal(target, alpha) & (alpha > 0)
    unchanged = np.all(source == target, axis=-1)
    outputs, doses = {}, {}
    for role, channels in (('colour_delta', slice(1, 3)), ('lightness_delta', slice(0, 1))):
        desired = sl.copy(); desired[..., channels] = tl[..., channels]
        rgb = from_lab(desired)
        dose = np.ones(alpha.shape, np.float64)
        bad = supported & ~_legal(rgb, alpha)
        if np.any(bad):
            base = sl[bad]; shift = desired[bad] - base; bound = alpha[bad]
            low, high = np.zeros(len(base)), np.ones(len(base))
            for _ in range(24):
                middle = (low + high) * .5
                valid = _legal(from_lab(base + shift * middle[:, None]), bound)
                low = np.where(valid, middle, low); high = np.where(valid, high, middle)
            dose[bad] = low
            rgb[bad] = from_lab(base + shift * low[:, None])
        # Only roundoff at the native RGB boundary is clamped; unsupported
        # source/full endpoints remain unchanged and separately reported.
        rgb[supported] = np.minimum(np.maximum(rgb[supported], 0), alpha[supported, None])
        rgb[~supported] = source[~supported]
        rgb[unchanged] = source[unchanged]
        dose[~supported] = 0
        outputs[role] = rgb.astype(np.float32)
        doses[role] = dose
    return outputs, doses, supported


def blend_delta(full, colour, lightness, colour_strength, lightness_strength):
    """Bilinear four-corner RGB replay; 00/11 are explicit exact endpoints."""
    c, l = float(colour_strength), float(lightness_strength)
    if not np.isfinite((c, l)).all() or not (0 <= c <= 1 and 0 <= l <= 1):
        raise ValueError('Colour and lightness controls must be between zero and one')
    if c == 1 and l == 1:
        return full
    if c == 0 and l == 0:
        return np.zeros_like(full)
    return (np.asarray(colour, np.float32) * np.float32(c * (1-l))
            + np.asarray(lightness, np.float32) * np.float32((1-c) * l)
            + np.asarray(full, np.float32) * np.float32(c * l))


def coupling_measurements(source, target, colour, lightness, alpha):
    """Quantify RGB interpolation versus requested Lab interpolation.

    Opaque samples are reported in ordinary OKLab units. All samples also use
    premultiplied Lab units; no low-alpha RGB division is used even for QA.
    """
    sl, tl = to_lab(source), to_lab(target)
    full, cd, ld = target-source, colour-source, lightness-source
    opaque = alpha >= .995
    rows = []
    for c in (0., .25, .5, .75, 1.):
        for l in (0., .25, .5, .75, 1.):
            actual = to_lab(source + blend_delta(full, cd, ld, c, l))
            requested = sl.copy()
            requested[..., 0] += l * (tl[..., 0]-sl[..., 0])
            requested[..., 1:] += c * (tl[..., 1:]-sl[..., 1:])
            le = np.abs(actual[..., 0]-requested[..., 0])
            ce = np.linalg.norm(actual[..., 1:]-requested[..., 1:], axis=-1)
            row = {'colour_strength': c, 'lightness_strength': l,
                   'premultiplied_L_error_max': float(le.max(initial=0)),
                   'premultiplied_ab_error_max': float(ce.max(initial=0))}
            if opaque.any():
                row.update(opaque_L_error_q50_q95_max=np.quantile(le[opaque], [.5, .95, 1]).tolist(),
                           opaque_ab_error_q50_q95_max=np.quantile(ce[opaque], [.5, .95, 1]).tolist())
            rows.append(row)
    return {'interpretation': 'Bilinear_RGB_doses_are_independent_controls_not_strict_Lab_channel_identity',
            'native_samples': int(alpha.size), 'opaque_samples': int(opaque.sum()), 'dose_grid': rows}


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _write(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    temporary.replace(path)


def receipt_identity(value):
    """JSON semantics survive the portable writer's newline normalization."""
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',', ':'),
        ensure_ascii=False,allow_nan=False).encode('utf8')).hexdigest()


def _read_native(asset):
    from reference_appearance_stack import verify_asset
    path = Path(asset['path']); verify_asset(asset)
    with np.load(path, allow_pickle=False) as resource:
        if set(resource.files) != {'values'}:
            raise ValueError('Native control asset requires a single values array')
        return np.asarray(resource['values'], np.float32)


def prepare_owned_controls(plan, output_dir, *, run=None, cancel=None, progress=None):
    """Prepare current v2 controls solely from owned Fp/alpha/full delta.

    Does not change input plan, assets, saved run or pointer. No models, RAW,
    reference decoding or T/B/S refitting. progress(stage,message,fraction) and
    cancel() have the same callback shapes as reference_pipeline.prepare.
    """
    from reference_appearance_stack import KIND, validate_controls, validate_preparation
    started = time.perf_counter()
    result = copy.deepcopy(plan)
    layer = next(row for row in result['layers'] if row['kind'] == KIND)
    validate_preparation(layer, result, run)
    if any(role in layer.get('assets', {}) for role in (*CONTROL_ROLES, 'control_receipt')):
        validate_controls(layer, result, run)
        return result
    source_schema = layer['params']['schema']
    def report(fraction, message):
        if cancel and cancel():
            raise InterruptedError('Native appearance control preparation cancelled; saved versions are unchanged')
        if progress:
            progress('appearance_controls', message, fraction)
    baseline = result['source_baseline']['assets']
    identity = {'recipe': CONTROL_RECIPE, 'source_fp_sha256': baseline['foreground_premultiplied']['sha256'],
                'alpha_sha256': baseline['foreground_alpha']['sha256'],
                'automatic_delta_sha256': layer['assets']['automatic_delta']['sha256'],
                'producer_sha256': _sha(__file__)}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    output = Path(output_dir).resolve()/key; output.mkdir(parents=True, exist_ok=True)
    report(0, '从当前版本原生人物基底准备颜色与明暗控制，不重新拟合。')
    source = _read_native(baseline['foreground_premultiplied'])
    alpha = _read_native(baseline['foreground_alpha'])
    full = _read_native(layer['assets']['automatic_delta'])
    native = tuple(layer['params']['canvas_size'])
    if source.shape != (native[1], native[0], 3) or source.shape != full.shape or alpha.shape != source.shape[:2]:
        raise ValueError('Control derivation requires the full original native canvas')
    temporary_paths = []
    arrays = {}; samples = {name: [] for name in ('source', 'target', 'colour', 'lightness', 'alpha')}
    measures = {role: {'supported_pixels': 0, 'active_pixels': 0, 'reduced_pixels': 0,
                      'dose_sum': 0., 'active_dose_min': 1.} for role in CONTROL_ROLES}
    exceptions = {'source_above_alpha_pixels': 0, 'target_above_alpha_pixels': 0,
                  'source_above_alpha_max': 0., 'target_above_alpha_max': 0.,
                  'source_negative_pixels': 0, 'target_negative_pixels': 0,
                  'unsupported_endpoint_pixels': 0, 'zero_alpha_nonzero_endpoint_pixels': 0}
    try:
        for role in CONTROL_ROLES:
            path = output/(role+'.tmp.npy'); temporary_paths.append(path)
            arrays[role] = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=source.shape)
        for top in range(0, len(source), 128):
            report(.8*top/len(source), '在原生覆盖率域分离颜色和明暗并限制独立角点色域。')
            end = min(len(source), top+128)
            base, a = source[top:end], alpha[top:end]
            target = base + full[top:end]
            corners, doses, supported = corner_targets(base, target, a)
            for role in CONTROL_ROLES:
                arrays[role][top:end] = corners[role]-base
                active = supported & (np.max(np.abs(target-base), axis=-1) > 0)
                values = doses[role][active]
                item = measures[role]; item['supported_pixels'] += int(supported.sum())
                item['active_pixels'] += len(values); item['reduced_pixels'] += int(np.count_nonzero(values < .99999))
                item['dose_sum'] += float(values.sum())
                if len(values): item['active_dose_min'] = min(item['active_dose_min'], float(values.min()))
            exceptions['unsupported_endpoint_pixels'] += int(np.count_nonzero((a > 0) & ~supported))
            exceptions['zero_alpha_nonzero_endpoint_pixels'] += int(np.count_nonzero((a == 0) & (np.any(base != 0, -1) | np.any(target != 0, -1))))
            for name, values in (('source', base), ('target', target)):
                above = np.max(values-a[..., None], axis=-1)
                exceptions[name+'_above_alpha_pixels'] += int(np.count_nonzero(above > 2e-6))
                exceptions[name+'_above_alpha_max'] = max(exceptions[name+'_above_alpha_max'], float(above.max(initial=0)))
                exceptions[name+'_negative_pixels'] += int(np.count_nonzero(np.any(values < 0, -1)))
            # Fixed native-coordinate samples, not rescaled RGB inputs.
            sample = (slice((-top) % 32, None, 32), slice(None, None, 32))
            for name, values in (('source', base), ('target', target), ('colour', corners['colour_delta']),
                                 ('lightness', corners['lightness_delta']), ('alpha', a)):
                samples[name].append(values[sample].reshape((-1, 3) if name != 'alpha' else (-1,)).copy())
        assets = {}
        for index, role in enumerate(CONTROL_ROLES):
            report(.8+.08*index, '保存原生独立剂量资产。')
            arrays[role].flush()
            path = output/('reference-appearance-'+role.replace('_', '-')+'.npz')
            tmp = path.with_suffix('.tmp.npz'); np.savez_compressed(tmp, values=arrays[role]); tmp.replace(path)
            assets[role] = {'path': str(path), 'sha256': _sha(path), 'size': list(native),
                'coordinate_space': 'source_canvas', 'working_space': 'display_linear_sRGB',
                'representation': 'premultiplied_delta', 'sampling': 'bilinear'}
        for role, item in measures.items():
            item['active_dose_mean'] = item.pop('dose_sum')/max(item['active_pixels'], 1)
            item['reduced_active_fraction'] = item['reduced_pixels']/max(item['active_pixels'], 1)
        qa = {name: np.concatenate(values) for name, values in samples.items()}
        receipt = {'schema': RECEIPT_SCHEMA, 'recipe': CONTROL_RECIPE,
            'binding': {**identity, 'source_sha256': layer['params']['source_sha256'],
                'reference_sha256': layer['params']['reference_sha256'], 'canvas_size': list(native),
                'working_domain': 'display_linear_sRGB',
                'control_assets_sha256': {role: assets[role]['sha256'] for role in CONTROL_ROLES}},
            'method': 'native_premultiplied_OKLab_homogeneity_no_alpha_division',
            'replay': 'c*(1-l)*colour_delta + (1-c)*l*lightness_delta + c*l*automatic_delta',
            'gamut_policy': 'Reduce_single_channel_dose_for_valid_endpoints; preserve_source_at_invalid_independent_corners; preserve_original_00_and_11',
            'original_endpoint_exceptions': exceptions, 'gamut_dose': measures,
            'intermediate_coupling': coupling_measurements(**qa),
            'endpoint_contract': {'source_00_unchanged': True, 'full_11_uses_original_delta_bytes': True,
                'zero_alpha_no_new_values': exceptions['zero_alpha_nonzero_endpoint_pixels'] == 0,
                'TBS_refitted': False, 'RAW_or_models_used': False},
            'elapsed_seconds': round(time.perf_counter()-started, 2), 'visual_acceptance': False}
        receipt_path = output/'control-receipt.json'; _write(receipt_path, receipt)
        assets['control_receipt'] = {'path': str(receipt_path), 'sha256': _sha(receipt_path),
                                     'representation': 'path_independent_native_control_provenance'}
        layer['assets'].update(assets)
        layer.update(colour_strength=1., lightness_strength=1.)
        layer['params'].update(schema=SCHEMA, control_recipe=CONTROL_RECIPE,
            control_source_delta_sha256=identity['automatic_delta_sha256'],
            control_assets_sha256={role: assets[role]['sha256'] for role in CONTROL_ROLES},
            control_receipt_identity_sha256=receipt_identity(receipt),
            control_binding_sha256=hashlib.sha256(json.dumps(receipt['binding'], sort_keys=True).encode()).hexdigest())
        result.setdefault('provenance', {})['appearance_controls'] = {
            'recipe': CONTROL_RECIPE, 'source_schema': source_schema,
            'original_automatic_delta_sha256': identity['automatic_delta_sha256'],
            'control_receipt_sha256': assets['control_receipt']['sha256'], 'TBS_refitted': False}
        validate_controls(layer, result, run)
        report(1, '颜色与明暗独立控制已准备；原完整目标和已保存父版本不变。')
        return result
    finally:
        for values in arrays.values():
            values.flush()
            values._mmap.close()
        for path in temporary_paths:
            if path.exists(): path.unlink()
