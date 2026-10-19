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
import zipfile

import numpy as np
from asset_identity import sha256_file
from native_array_storage import NativeArrayRowWriter
from runtime_resources import ordered_native_map

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
    # The finite check is implied by the closed SDR bounds: NaN and either
    # infinity fail one of the two comparisons because alpha is validated as
    # finite before this helper is called.  Combining the comparisons also
    # reduces the per-iteration reduction during gamut bisection.
    return ((values[..., 0] >= 0) & (values[..., 0] <= alpha)
            & (values[..., 1] >= 0) & (values[..., 1] <= alpha)
            & (values[..., 2] >= 0) & (values[..., 2] <= alpha))


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
    supported = _legal(source, alpha) & _legal(target, alpha) & (alpha > 0)
    unchanged = np.all(source == target, axis=-1)
    from native_cuda_fields import eligible, control_corners
    if eligible(source.shape):
        return control_corners(source, target, alpha, supported, unchanged)
    # Unsupported endpoints already select the exact source RGB corner below.
    # Do not convert zero-coverage/background pixels to Lab or run their gamut
    # search. Equal supported endpoints still participate: at a gamut boundary
    # their round-trip dose is part of the existing QA accounting.
    outputs = {role: source.copy() for role in CONTROL_ROLES}
    doses = {role: supported.astype(np.float64) for role in CONTROL_ROLES}
    if not np.any(supported):
        return outputs, doses, supported
    supported_source = source[supported]
    sl, tl = to_lab(supported_source), to_lab(target[supported])
    bound = alpha[supported]
    equal = unchanged[supported]
    for role, channels in (('colour_delta', slice(1, 3)), ('lightness_delta', slice(0, 1))):
        desired = sl.copy(); desired[..., channels] = tl[..., channels]
        rgb = from_lab(desired)
        dose = np.ones(bound.shape, np.float64)
        bad = ~_legal(rgb, bound)
        if np.any(bad):
            base = sl[bad]; shift = desired[bad] - base; bad_bound = bound[bad]
            low, high = np.zeros(len(base)), np.ones(len(base))
            for _ in range(24):
                middle = (low + high) * .5
                candidate = from_lab(base + shift * middle[:, None])
                valid = _legal(candidate, bad_bound)
                low = np.where(valid, middle, low); high = np.where(valid, high, middle)
            dose[bad] = low
            # ``candidate`` is the exact from_lab result for the final middle
            # values.  Reuse it where that final step was valid; only rows that
            # stayed invalid need the old final evaluation at their unchanged
            # low bound.  This removes one full matrix transform without
            # changing any arithmetic or float bits in the resulting corners.
            rgb_bad = candidate.copy()
            if np.any(~valid):
                rgb_bad[~valid] = from_lab(base[~valid] + shift[~valid] * low[~valid, None])
            rgb[bad] = rgb_bad
        # Only roundoff at the native RGB boundary is clamped; unsupported
        # source/full endpoints remain unchanged and separately reported.
        rgb = np.minimum(np.maximum(rgb, 0), bound[:, None])
        rgb[equal] = supported_source[equal]
        outputs[role][supported] = rgb.astype(np.float32)
        doses[role][supported] = dose
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
    return sha256_file(path)


def _write(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    temporary.replace(path)


def receipt_identity(value):
    """JSON semantics survive the portable writer's newline normalization."""
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',', ':'),
        ensure_ascii=False,allow_nan=False).encode('utf8')).hexdigest()


def _reuse_cached_controls(output, layer, result, run, identity, native):
    """Attach a completely published control bank from an interrupted retry.

    The control bank is published as two independent NPZ files followed by a
    receipt.  A process can therefore die after the expensive corner derivation
    but before the prepared plan is published.  Reusing it is safe only when
    every binding and asset digest still validates through the normal producer
    contract; a partial or stale directory simply falls through to recompute.
    """
    output = Path(output)
    receipt_path = output/'control-receipt.json'
    paths = {role: output/('reference-appearance-'+role.replace('_','-')+'.npz')
             for role in CONTROL_ROLES}
    if not receipt_path.is_file() or any(not path.is_file() for path in paths.values()):
        return None
    try:
        receipt = json.loads(receipt_path.read_text(encoding='utf8'))
        if receipt.get('schema') != RECEIPT_SCHEMA or receipt.get('recipe') != CONTROL_RECIPE:
            return None
        params = layer['params']
        original_schema = params.get('schema')
        expected_binding = {**identity, 'source_sha256': params['source_sha256'],
            'reference_sha256': params['reference_sha256'], 'canvas_size': list(native),
            'working_domain': 'display_linear_sRGB'}
        binding = receipt.get('binding') or {}
        if any(binding.get(key) != value for key, value in expected_binding.items()):
            return None
        expected = (binding.get('control_assets_sha256') or {})
        # ``validate_controls`` below performs the authoritative SHA and
        # ndarray contract checks.  Reuse the receipt's declared digests here
        # so a retry does not hash each 395 MB bank twice; a changed/partial
        # file still fails that validator and falls through to recomputation.
        digests = {role: expected.get(role) for role in CONTROL_ROLES}
        if any(not isinstance(digest, str) or len(digest) != 64 for digest in digests.values()):
            return None
        receipt_digest = _sha(receipt_path)
        assets = {role: {'path': str(paths[role]), 'sha256': digests[role],
                         'size': list(native), 'coordinate_space': 'source_canvas',
                         'working_space': 'display_linear_sRGB',
                         'representation': 'premultiplied_delta', 'sampling': 'bilinear'}
                  for role in CONTROL_ROLES}
        assets['control_receipt'] = {'path': str(receipt_path), 'sha256': receipt_digest,
                                     'representation': 'path_independent_native_control_provenance'}
        layer['assets'].update(assets)
        layer.update(colour_strength=1., lightness_strength=1.)
        layer['params'].update(schema=SCHEMA, control_recipe=CONTROL_RECIPE,
            control_source_delta_sha256=identity['automatic_delta_sha256'],
            control_assets_sha256={role: assets[role]['sha256'] for role in CONTROL_ROLES},
            control_receipt_identity_sha256=receipt_identity(receipt),
            control_binding_sha256=hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest())
        result.setdefault('provenance', {})['appearance_controls'] = {
            'recipe': CONTROL_RECIPE, 'source_schema': original_schema,
            'original_automatic_delta_sha256': identity['automatic_delta_sha256'],
            'control_receipt_sha256': assets['control_receipt']['sha256'], 'TBS_refitted': False,
            'cache_hit': True}
        from reference_appearance_stack import validate_controls
        validate_controls(layer, result, run)
        return result
    except (OSError, ValueError, KeyError, TypeError, EOFError,
            zipfile.BadZipFile, json.JSONDecodeError):
        return None


def _read_native(asset):
    from reference_appearance_stack import verify_asset
    path = Path(asset['path']); verify_asset(asset)
    with np.load(path, allow_pickle=False) as resource:
        if set(resource.files) != {'values'}:
            raise ValueError('Native control asset requires a single values array')
        return np.asarray(resource['values'], np.float32)


def _corner_bands(source, alpha, full, *, workers=2):
    """Compute unchanged 128-row corners with a bounded, ordered queue."""
    def compute(top):
        end = min(len(source), top+128)
        base, a = source[top:end], alpha[top:end]
        target = base + full[top:end]
        corners, doses, supported = corner_targets(base, target, a)
        return top, base, a, target, corners, doses, supported
    yield from ordered_native_map(compute, range(0, len(source), 128), workers=workers)


def prepare_owned_controls(plan, output_dir, *, run=None, cancel=None, progress=None,
                           native_inputs=None):
    """Prepare current v2 controls solely from owned Fp/alpha/full delta.

    Does not change input plan, assets, saved run or pointer. No models, RAW,
    reference decoding or T/B/S refitting. progress(stage,message,fraction) and
    cancel() have the same callback shapes as reference_pipeline.prepare.

    The native freeze can supply its live working_foreground, alpha and
    automatic_delta, together with the already completed beauty_execution.
    These are the same producer's arrays, not alternate asset locators. They
    are consumed read-only without another decompression or Beauty replay.
    """
    from reference_appearance_stack import KIND, validate_controls, validate_preparation
    started = time.perf_counter()
    result = copy.deepcopy(plan)
    layer = next(row for row in result['layers'] if row['kind'] == KIND)
    live = None
    if native_inputs is not None:
        if (not isinstance(native_inputs, dict) or set(native_inputs) !=
                {'working_foreground', 'alpha', 'automatic_delta', 'beauty_execution'}):
            raise ValueError('Native control inputs require the current working foreground, alpha and delta')
        live = {key: native_inputs[key] for key in ('alpha', 'automatic_delta')}
    validate_preparation(layer, result, run, native_inputs=live)
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
    from runtime_resources import code_path
    from native_cuda_fields import execution
    identity['cuda_fields_sha256'] = _sha(code_path('native_cuda_fields.py'))
    identity['field_backend'] = execution()['backend']
    if run is not None:
        from portrait_beauty_layer import identity_for_run
        identity['portrait_beauty'] = identity_for_run(run)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    output = Path(output_dir).resolve()/key; output.mkdir(parents=True, exist_ok=True)
    report(0, '从当前版本原生人物基底准备颜色与明暗控制，不重新拟合。')
    # A previous attempt may have published both control arrays but died
    # before attaching them to the prepared plan.  Revalidate and attach that
    # immutable bank before reading native arrays or replaying Beauty.
    cached = _reuse_cached_controls(output, layer, result, run, identity,
                                    tuple(layer['params']['canvas_size']))
    if cached is not None:
        report(1, '复用已验证的原生颜色与明暗控制资产。')
        return cached
    if native_inputs is None:
        source = _read_native(baseline['foreground_premultiplied'])
        alpha = _read_native(baseline['foreground_alpha'])
        beauty_execution = {'enabled': False, 'applied': False}
        if run is not None:
            from portrait_beauty_layer import apply_for_run
            source, beauty_execution = apply_for_run(source, run, premultiplied_alpha=alpha)
        full = _read_native(layer['assets']['automatic_delta'])
    else:
        source = native_inputs['working_foreground']
        alpha = native_inputs['alpha']
        full = native_inputs['automatic_delta']
        if any(not isinstance(values, np.ndarray) or values.dtype != np.float32
               for values in (source, alpha, full)):
            raise ValueError('Native control inputs must be live float32 producer arrays')
        if np.shares_memory(source, full):
            raise ValueError('Native control source and delta must have independent storage')
        beauty_execution = native_inputs['beauty_execution']
        beauty = identity.get('portrait_beauty')
        expected_applied = bool(beauty and beauty['strength']
            and len(beauty['disabled_faces']) < len(beauty['faces']))
        if (not isinstance(beauty_execution, dict)
                or beauty_execution.get('enabled') is not bool(beauty)
                or beauty_execution.get('applied') is not expected_applied):
            raise ValueError('Native control foreground must include this version\'s Beauty replay')
    native = tuple(layer['params']['canvas_size'])
    if source.shape != (native[1], native[0], 3) or source.shape != full.shape or alpha.shape != source.shape[:2]:
        raise ValueError('Control derivation requires the full original native canvas')
    arrays = {}; samples = {name: [] for name in ('source', 'target', 'colour', 'lightness', 'alpha')}
    measures = {role: {'supported_pixels': 0, 'active_pixels': 0, 'reduced_pixels': 0,
                      'dose_sum': 0., 'active_dose_min': 1.} for role in CONTROL_ROLES}
    exceptions = {'source_above_alpha_pixels': 0, 'target_above_alpha_pixels': 0,
                  'source_above_alpha_max': 0., 'target_above_alpha_max': 0.,
                  'source_negative_pixels': 0, 'target_negative_pixels': 0,
                  'unsupported_endpoint_pixels': 0, 'zero_alpha_nonzero_endpoint_pixels': 0}
    try:
        for role in CONTROL_ROLES:
            path = output/('reference-appearance-'+role.replace('_', '-')+'.npz')
            arrays[role] = NativeArrayRowWriter(path, source.shape)
        for top, base, a, target, corners, doses, supported in _corner_bands(source, alpha, full):
            report(.8*top/len(source), '在原生覆盖率域分离颜色和明暗并限制独立角点色域。')
            active = supported & np.any(target != base, axis=-1)
            supported_pixels = int(supported.sum())
            for role in CONTROL_ROLES:
                arrays[role].write(corners[role]-base)
                values = doses[role][active]
                item = measures[role]; item['supported_pixels'] += supported_pixels
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
            # Blocks already have their unchanged float bits in the stored
            # member. Close its ZIP and atomically publish in the same role
            # order, without writing/rereading a full scratch NPY copy.
            arrays[role].publish()
            path = arrays[role].path
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
                'source_00_basis': 'beauty_working_foreground' if beauty_execution['enabled'] else 'original_foreground',
                'beauty_applied_before_corner_derivation': beauty_execution['applied'],
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
        validate_controls(layer, result, run, native_controls=arrays, native_inputs=live)
        report(1, '颜色与明暗独立控制已准备；原完整目标和已保存父版本不变。')
        return result
    finally:
        for values in arrays.values():
            values.close()
