"""Same-photo finish as a source-carried local affine colour operator.

Only the preparation subprocess imports OpenCV. Frozen replay samples owned
coefficients at global native pixel centres; reference RGB is never composited.
"""
from __future__ import annotations

import argparse
import copy
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from asset_identity import sha256_file
from colour_space import M1, M2, IM1, IM2

KIND = 'registered_reference_style'
SCHEMA = 'registered_reference_style.v1'
SCHEMA_V2 = 'registered_reference_style.v2'
PROFILE = {
    'id': 'same_photo_source_guided_affine_v1',
    'registration_long_edge': 1600, 'coefficient_long_edge': 512,
    'sift_features': 14000, 'match_ratio': .75, 'ransac_pixels': 3.,
    'minimum_inliers': 100, 'minimum_inlier_ratio': .55,
    'maximum_inlier_q90_pixels': 2.75,
    'minimum_inlier_hull_fraction': .12, 'minimum_inlier_axis_span': .40,
    'minimum_inlier_grid_cells': 6, 'minimum_reference_coverage': .50,
    'minimum_H_singular': .30, 'maximum_H_singular': 3.,
    'maximum_H_condition': 4.,
    'window_sigma_at_1600': 12., 'observation_sigma_at_1600': 2.,
    'ridge_diagonal': [.0004, .000025, .000025],
    'huber_OKLab': .025, 'robust_iterations': 2,
    'minimum_J_singular': .10, 'maximum_J_singular': 3.,
    'maximum_J_condition': 30.,
    'coefficient_sampling': 'global_pixel_centres_clamped_bilinear_float32.v1',
    'gamut': 'source_to_mapped_common_RGB_dose_smooth_min.v1',
    'gamut_knee': .10, 'row_block': 128,
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode('utf8')).hexdigest()


def _write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    temporary.replace(path)


def _asset(path, **metadata):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha256_file(path), **metadata}


def _verified(asset):
    if not isinstance(asset, dict) or sha256_file(asset['path']) != asset.get('sha256'):
        raise ValueError('Registered reference style asset identity differs')
    return Path(asset['path'])


def checked_contract(run, root=None):
    """Require the one checked reference-role descriptor, never infer permission."""
    from creative_direction import validate_reference_contract
    plan = run.get('task_plan') or {}
    intent = run.get('task_intent') or plan.get('intent') or {}
    if plan.get('ready') is not True or plan.get('intent') != intent:
        raise ValueError('Same-photo style requires a checked matching task intent')
    inputs = plan.get('inputs') or {}
    role = (intent.get('constraints') or {}).get('reference_role')
    for key in ('source', 'reference'):
        checked_sha = (inputs.get(key) or {}).get('sha256')
        actual = run.get(key) or {}
        required_checked = key == 'source' or role != 'source_conditioned_design' or intent.get('mode') != 'create_reference'
        if ((required_checked or checked_sha is not None) and checked_sha != actual.get('sha256')
                or sha256_file(actual['path']) != actual.get('sha256')):
            raise ValueError('Same-photo style requires checked source/reference file identities')
    expected = validate_reference_contract(plan.get('reference_contract'), intent, inputs)
    if not expected:
        raise ValueError('Explicit checked same-photo reference contract required')
    if role == 'source_conditioned_design':
        if root is None:
            raise ValueError('Source-conditioned style requires its workspace for real provider provenance')
        from creative_direction import bind_source_conditioned_reference
        expected = bind_source_conditioned_reference(Path(root), plan, run['reference'])
    return copy.deepcopy(expected)


def _master_info(path):
    import tifffile
    with tifffile.TiffFile(path) as file:
        page = file.pages[0]
        metadata = (file.shaped_metadata or ({},))[0] or {}
        if (metadata.get('working_domain') != 'display_linear_sRGB' or page.dtype != np.float32
                or len(page.shape) != 3 or page.shape[-1] != 3 or page.is_tiled or page.planarconfig != 1):
            raise ValueError('Same-photo baseline requires a display-linear float32 RGB strip master')
        return [int(page.shape[1]), int(page.shape[0])]


def prepare(root, run, baseline_master, output):
    """Prepare one owned frozen layer; the baseline is read, never overwritten."""
    from cache_store import exclusive_file_lock
    from runtime_resources import code_path, isolated_python_environment
    from local_photo_analysis import local_runtime_paths
    contract = checked_contract(run, root)
    master = Path(baseline_master['path'] if isinstance(baseline_master, dict) else baseline_master).resolve()
    native = _master_info(master)
    from reference_colour_fit import source_appearance_policy_for_run
    appearance_policy = source_appearance_policy_for_run(run)
    single_colour_owner = appearance_policy.get('colour_owner') == 'registered_reference_affine.v1'
    from skin_illumination import checked_context
    material = None if single_colour_owner else checked_context(run, native)
    master_sha = sha256_file(master)
    if material is not None:
        material['binding'].update(reference_sha256=contract['reference_sha256'],
            baseline_master_sha256=master_sha, producer_code=producer_code())
    if isinstance(baseline_master, dict) and baseline_master.get('sha256') != master_sha:
        raise ValueError('Same-photo baseline master SHA differs')
    identity = {'reference_contract': contract, 'baseline_master_sha256': master_sha,
        'native_size': native, 'profile': profile_for(material is not None), 'producer_code': producer_code(),
        'colour_ownership': ('single_registered_affine.v1' if single_colour_owner
                             else 'registered_affine_plus_checked_material.v1'),
        'source_appearance_policy': appearance_policy}
    if material is not None:
        identity['material_context'] = material['binding']
    output = Path(output).resolve() / _digest(identity)[:24]
    output.mkdir(parents=True, exist_ok=True)
    result = output / 'layer.json'
    with exclusive_file_lock(output / 'prepare.lock'):
        if result.is_file():
            layer = json.loads(result.read_text(encoding='utf8'))
            saved = validate_layer(layer)
            if saved['identity'] != identity:
                raise ValueError('Same-photo style cache binding differs')
            return layer
        request = {'identity': identity, 'source': run['source'], 'reference': run['reference'],
            'baseline_master': str(master), 'output': str(output), 'root': str(Path(root).resolve()),
            'checked_task_plan': run['task_plan']}
        if material is not None:
            request['material_context'] = material
        request_path = output / 'request.json'
        _write(request_path, request)
        selected = run.get('options', {}).get('model_registry')
        paths = local_runtime_paths(Path(root), registry_path=selected)
        command = [str(paths['python']), '-X', 'utf8', str(code_path('registered_reference_style.py')),
                   '--worker', str(request_path)]
        log_path = output / 'preparation.log'
        with log_path.open('w', encoding='utf8') as log:
            completed = subprocess.run(command, stdout=log, stderr=log, timeout=900,
                env=isolated_python_environment(),
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        if completed.returncode or not result.is_file():
            raise RuntimeError('Same-photo style preparation failed; see ' + str(log_path))
        layer = json.loads(result.read_text(encoding='utf8'))
        if validate_layer(layer)['identity'] != identity:
            raise ValueError('Same-photo style worker returned another baseline binding')
        # Detect a file replacement while the isolated producer was running.
        if sha256_file(master) != master_sha:
            raise ValueError('Same-photo baseline changed during preparation')
        checked_contract(run, root)
        current_material = None if single_colour_owner else checked_context(run, native)
        if current_material is not None:
            current_material['binding'].update(reference_sha256=contract['reference_sha256'],
                baseline_master_sha256=master_sha, producer_code=producer_code())
        if current_material != material:
            raise ValueError('Skin illumination source masks or goal changed during preparation')
        return layer


def producer_code():
    from runtime_resources import code_path
    from preset_store import DEFAULT_PATH
    result = {name: sha256_file(code_path(name)) for name in
        ('registered_reference_style.py', 'colour_space.py', 'raw_support.py',
         'display_transform.py', 'float_appearance.py', 'native_detail.py', 'creative_direction.py',
         'skin_illumination.py', 'preset_store.py')}
    result['presets/presets.json'] = sha256_file(DEFAULT_PATH)
    return result


def profile_for(material=False):
    if not material:
        return PROFILE
    from skin_illumination import PROFILE as skin_profile
    return {**PROFILE, 'id': 'source_guided_affine_with_skin_illumination_v2',
        'skin_illumination': skin_profile,
        'skin_receipt_binding': 'owned_hash_bound_JSON_asset_record.v1'}


@lru_cache(maxsize=4)
def _coefficients(path, expected, modification_ns, size):
    if sha256_file(path) != expected:
        raise ValueError('Frozen style coefficients differ from their SHA')
    with np.load(path, allow_pickle=False) as resource:
        values = resource['values'].copy()
    if (values.dtype != np.float32 or values.ndim != 3 or values.shape[-1] != 12
            or min(values.shape[:2]) < 2 or max(values.shape[:2]) > 512
            or not np.isfinite(values).all()):
        raise ValueError('Invalid frozen registered-style coefficient grid')
    J = np.eye(3, dtype=np.float32)+values[..., :9].reshape((*values.shape[:2], 3, 3))
    sv = np.linalg.svd(J, compute_uv=False)
    symmetric = (J+J.swapaxes(-1, -2))*.5
    if (np.linalg.eigvalsh(symmetric).min() < .1-3e-6 or sv.max() > 3.+3e-6):
        raise ValueError('Frozen style Jacobians exceed their convex gain/orientation bounds')
    values.flags.writeable = False
    return values


def validate_layer(layer):
    """Read historical owned evidence without requiring original temp inputs."""
    schema = layer.get('params', {}).get('schema')
    if (layer.get('kind') != KIND or schema not in (SCHEMA, SCHEMA_V2)):
        raise ValueError('Unsupported registered reference style layer')
    assets = layer.get('assets') or {}
    path = _verified(assets['contract'])
    receipt = json.loads(path.read_text(encoding='utf8'))
    if (receipt.get('schema') != schema or receipt.get('profile') != profile_for(schema == SCHEMA_V2)
            or receipt.get('reference_RGB_composited') is not False
            or receipt.get('model_submissions') != 0
            or receipt.get('coefficients_sha256') != assets['coefficients'].get('sha256')
            or layer['params'].get('identity_sha256') != _digest(receipt['identity'])
            or layer['params'].get('native_size') != receipt['identity']['native_size']):
        raise ValueError('Frozen registered-style contract differs')
    cpath = _verified(assets['coefficients']); stat = cpath.stat()
    _coefficients(str(cpath), assets['coefficients']['sha256'], stat.st_mtime_ns, stat.st_size)
    if schema == SCHEMA_V2:
        _skin_fields(layer, receipt)
    return receipt


def _skin_fields(layer, receipt):
    import skin_illumination as skin
    assets = layer['assets']
    path = _verified(assets['skin_receipt'])
    evidence = json.loads(path.read_text(encoding='utf8'))
    parent = receipt.get('identity') or {}
    binding = evidence.get('binding') or {}
    contract = parent.get('reference_contract') or {}
    nested = receipt.get('material_illumination', {}).get('receipt_asset')
    if not isinstance(nested, dict):
        raise ValueError('Registered skin receipt requires an owned hash-bound asset record')
    # Generic materialization and portable import rewrite JSON byte encoding
    # and then rebind standard {path,sha256} records in dependency order. A bare
    # embedded leaf-file SHA cannot follow that relocation/serialization.
    _verified(nested)
    if (nested.get('sha256') != assets['skin_receipt']['sha256']
            or parent.get('material_context') != binding
            or binding.get('source_sha256') != contract.get('source_sha256')
            or binding.get('reference_sha256') != contract.get('reference_sha256')
            or binding.get('baseline_master_sha256') != parent.get('baseline_master_sha256')
            or binding.get('producer_code') != parent.get('producer_code')
            or binding.get('native_size') != parent.get('native_size')):
        raise ValueError('Registered skin receipt identity differs')
    return skin.validate(assets, evidence, receipt['identity']['material_context'])


def _linear3(v, matrix):
    # Avoid shape-dependent BLAS kernels: full, thin crop and row block agree.
    return np.stack([((v[..., 0]*matrix[j, 0] + v[..., 1]*matrix[j, 1])
                      + v[..., 2]*matrix[j, 2]) for j in range(3)], axis=-1)


def _to_lab(v):
    return _linear3(np.cbrt(np.maximum(_linear3(v, M1), np.float32(0))), M2)


def _from_lab(v):
    lms = _linear3(v, IM2)
    return _linear3((lms*lms)*lms, IM1)


def _affine_delta(lab, A, b):
    first = lab[..., 0, None]*A[..., 0, :] + lab[..., 1, None]*A[..., 1, :]
    return (first + lab[..., 2, None]*A[..., 2, :]) + b


def _sample(coefficients, box, native_size):
    left, top, right, bottom = box
    h, w = coefficients.shape[:2]
    xx = np.clip((np.arange(left, right, dtype=np.float64)+.5)*w/native_size[0]-.5, 0, w-1)
    yy = np.clip((np.arange(top, bottom, dtype=np.float64)+.5)*h/native_size[1]-.5, 0, h-1)
    x0, y0 = np.floor(xx).astype(np.intp), np.floor(yy).astype(np.intp)
    x1, y1 = np.minimum(x0+1, w-1), np.minimum(y0+1, h-1)
    wx, wy = (xx-x0).astype(np.float32)[None, :, None], (yy-y0).astype(np.float32)[:, None, None]
    a = coefficients[y0[:, None], x0[None, :]]
    b = coefficients[y0[:, None], x1[None, :]]
    c = coefficients[y1[:, None], x0[None, :]]
    d = coefficients[y1[:, None], x1[None, :]]
    upper = a + wx*(b-a)
    lower = c + wx*(d-c)
    return upper + wy*(lower-upper)


def _ray_gamut(source, mapped, strength):
    # Common RGB dose along a line from valid source; no channel hard clipping.
    src = source.astype(np.float64)
    delta = (mapped.astype(np.float64)-src)*float(strength)
    capacity = np.full(delta.shape, np.inf, np.float64)
    np.divide(1-src, delta, out=capacity, where=delta > 0)
    np.divide(-src, delta, out=capacity, where=delta < 0)
    upper = np.maximum(0, capacity.min(-1))
    knee = PROFILE['gamut_knee']
    dose = np.minimum(1, upper) - np.maximum(knee-np.abs(1-upper), 0)**2/(4*knee)
    dose = np.maximum(0, dose)
    dose *= np.where(dose < 1, 1-2e-7, 1)
    result = (src+delta*dose[..., None]).astype(np.float32)
    if not np.isfinite(result).all() or result.min() < 0 or result.max() > 1:
        raise ValueError('Registered-style common gamut dose left the SDR domain')
    return result, dose


def apply_with_report(values, layer, box, native_size, *, strength=None, preview=False):
    values = np.asarray(values)
    if (values.dtype != np.float32 or values.shape != (box[3]-box[1], box[2]-box[0], 3)
            or not np.isfinite(values).all() or values.min() < 0 or values.max() > 1
            or any(type(v) not in (int, np.int32, np.int64) for v in (*box, *native_size))
            or not (0 <= box[0] < box[2] <= native_size[0] and 0 <= box[1] < box[3] <= native_size[1])):
        raise ValueError('Registered-style application requires a native display-linear RGB box')
    frozen_native = layer.get('params', {}).get('native_size')
    if (layer.get('kind') != KIND or layer.get('params', {}).get('schema') not in (SCHEMA, SCHEMA_V2)
            or not isinstance(frozen_native, list) or len(frozen_native) != 2):
        raise ValueError('Registered-style application requires its frozen canvas metadata')
    if preview:
        if (native_size[0] > frozen_native[0] or native_size[1] > frozen_native[1]
                or abs(native_size[0]/native_size[1] - frozen_native[0]/frozen_native[1]) > 2/min(native_size)):
            raise ValueError('Style preview requires a smaller full-source canvas with the same aspect ratio')
    elif list(native_size) != frozen_native:
        raise ValueError('Registered-style field belongs to another native canvas')
    dose = layer.get('strength', 1.) if strength is None else strength
    if isinstance(dose, bool) or not np.isfinite(dose) or not 0 <= dose <= 1:
        raise ValueError('Registered-style strength must be finite in [0,1]')
    if dose == 0 or layer.get('enabled') is False:
        return values.copy(), {'applied': False, 'pixels': values.shape[0]*values.shape[1], 'strength': float(dose)}
    receipt = validate_layer(layer)
    skin_fields = _skin_fields(layer, receipt) if layer['params']['schema'] == SCHEMA_V2 else None
    skin_summary = {'active_pixels': 0, 'gamut_attenuated_pixels': 0, 'gamut_dose_minimum': 1.}
    asset = layer['assets']['coefficients']; path = Path(asset['path']); stat = path.stat()
    coefficients = _coefficients(str(path), asset['sha256'], stat.st_mtime_ns, stat.st_size)
    result = np.empty_like(values)
    count = 0; dose_sum = 0.; dose_min = 1.; affected = 0
    delta_sum = np.zeros(3, np.float64); delta_max = 0.
    gradient = np.zeros(6, np.float64)
    previous_source = previous_output = None
    def record_gradients(a, b):
        a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
        gradient[:] += (a.size, a.sum(), b.sum(), a@a, b@b, a@b)
    for top in range(0, len(values), PROFILE['row_block']):
        bottom = min(len(values), top+PROFILE['row_block'])
        source = values[top:bottom]
        lab = _to_lab(source)
        c = _sample(coefficients, (box[0], box[1]+top, box[2], box[1]+bottom), native_size)
        A = c[..., :9].reshape((*c.shape[:2], 3, 3))
        delta = _affine_delta(lab, A, c[..., 9:])
        mapped = _from_lab(lab+delta)
        output, gamut = _ray_gamut(source, mapped, dose)
        # An exactly zero saved operator must not add a colour round trip.
        inactive = np.all(c == 0, axis=-1)
        output[inactive] = source[inactive]
        gamut[inactive] = 1
        if skin_fields is not None:
            from skin_illumination import apply as apply_skin
            output, skin_report = apply_skin(output, skin_fields,
                (box[0], box[1]+top, box[2], box[1]+bottom), native_size, dose)
            skin_summary['active_pixels'] += skin_report.get('active_pixels', 0)
            skin_summary['gamut_attenuated_pixels'] += skin_report.get('gamut_attenuated_pixels', 0)
            skin_summary['gamut_dose_minimum'] = min(skin_summary['gamut_dose_minimum'], skin_report.get('gamut_dose_minimum', 1.))
        result[top:bottom] = output
        count += gamut.size; dose_sum += float(gamut.sum()); dose_min = min(dose_min, float(gamut.min()))
        affected += int(np.count_nonzero(gamut < 1))
        difference = abs(output.astype(np.float64)-source)
        delta_sum += difference.sum((0, 1)); delta_max = max(delta_max, float(difference.max()))
        out_L = _to_lab(output)[..., 0]
        src_L = lab[..., 0]
        record_gradients(np.diff(src_L, axis=1), np.diff(out_L, axis=1))
        record_gradients(np.diff(src_L, axis=0), np.diff(out_L, axis=0))
        if previous_source is not None:
            record_gradients(src_L[0]-previous_source, out_L[0]-previous_output)
        previous_source, previous_output = src_L[-1].copy(), out_L[-1].copy()
    n, source_mean, output_mean, ss, oo, so = gradient
    source_var = max(0., ss-source_mean*source_mean/max(n, 1))
    output_var = max(0., oo-output_mean*output_mean/max(n, 1))
    covariance = so-source_mean*output_mean/max(n, 1)
    return result, {'applied': True, 'pixels': count, 'strength': float(dose),
        **({'skin_illumination': skin_summary} if skin_fields is not None else {}),
        'gamut_dose_minimum': dose_min, 'gamut_dose_mean': dose_sum/max(count, 1),
        'gamut_attenuated_pixels': affected, 'mean_absolute_RGB_change': (delta_sum/max(count, 1)).tolist(),
        'maximum_absolute_RGB_change': delta_max, 'source_geometry_changed': False,
        'source_texture_bitexact': False, 'reference_RGB_composited': False,
        'sampling_canvas': list(native_size), 'sampling_scope': 'source_preview' if preview else 'native_source_canvas',
        'native_adjacent_L_gradient_correlation': float(covariance/np.sqrt(source_var*output_var)) if source_var*output_var > 1e-20 else None,
        'native_adjacent_L_gradient_std_ratio': float(np.sqrt(output_var/source_var)) if source_var > 1e-20 else None,
        'gradient_measurement': 'actual source/output native adjacent L; includes edges and illumination, not a geometry or texture identity proof'}


def apply(values, layer, box, native_size, *, strength=None):
    return apply_with_report(values, layer, box, native_size, strength=strength)[0]




def registration_quality(source_points, reference_points, inliers, H, source_size, reference_size, *, coverage):
    """Numeric gate shared by the worker and CPU adversarial fixtures."""
    s, r = np.asarray(source_points, np.float64), np.asarray(reference_points, np.float64)
    good = np.asarray(inliers, bool).reshape(-1)
    H = np.asarray(H, np.float64)
    if (s.ndim != 2 or s.shape != r.shape or s.shape[1:] != (2,) or good.shape != (len(s),)
            or H.shape != (3, 3) or not np.isfinite(H).all() or abs(H[2, 2]) < 1e-10):
        raise ValueError('Invalid same-photo registration evidence')
    H = H/H[2, 2]
    if not np.isfinite(s).all() or not np.isfinite(r).all():
        raise ValueError('Nonfinite same-photo registration observations')
    count = int(good.sum()); ratio = count/max(len(s), 1)
    if count < PROFILE['minimum_inliers'] or ratio < PROFILE['minimum_inlier_ratio']:
        raise ValueError('Reference is not a qualified same photograph: insufficient matching inliers')
    homogeneous = np.c_[s[good], np.ones(count)] @ H.T
    if np.any(homogeneous[:, 2] <= 0): raise ValueError('Unhealthy same-photo projective horizon')
    errors = np.linalg.norm(homogeneous[:, :2]/homogeneous[:, 2:] - r[good], axis=-1)
    q90 = float(np.quantile(errors, .9))
    if q90 > PROFILE['maximum_inlier_q90_pixels']:
        raise ValueError('Same-photo registration residual is too large')
    from scipy.spatial import ConvexHull, QhullError
    spatial = {}
    for role, points, size in (('source', s[good], source_size), ('reference', r[good], reference_size)):
        try: hull = float(ConvexHull(points).volume)/(size[0]*size[1])
        except QhullError as exc: raise ValueError('Collinear same-photo evidence') from exc
        span = np.ptp(points, axis=0)/np.asarray(size)
        cells = np.clip((points/np.asarray(size)*4).astype(int), 0, 3)
        occupied = len(np.unique(cells[:, 1]*4+cells[:, 0]))
        if (hull < PROFILE['minimum_inlier_hull_fraction'] or span.min() < PROFILE['minimum_inlier_axis_span']
                or occupied < PROFILE['minimum_inlier_grid_cells']):
            raise ValueError('Same-photo matches do not cover the scene spatially')
        spatial[role] = {'hull_fraction': hull, 'axis_span': span.tolist(), 'grid_cells': occupied}
    extrema = []
    for x, y in ((0, 0), (source_size[0], 0), (0, source_size[1]), tuple(source_size),
                 (source_size[0]/2, source_size[1]/2)):
        p = H @ np.array([x, y, 1.])
        if p[2] <= .1: raise ValueError('Same-photo H crosses a projective horizon')
        jacobian = (H[:2, :2]*p[2] - p[:2, None]*H[2:3, :2])/(p[2]*p[2])
        sv = np.linalg.svd(jacobian, compute_uv=False)
        if (np.linalg.det(jacobian) <= 0 or sv.min() < PROFILE['minimum_H_singular']
                or sv.max() > PROFILE['maximum_H_singular'] or sv.max()/sv.min() > PROFILE['maximum_H_condition']):
            raise ValueError('Same-photo H has unhealthy local scale, reflection or condition')
        extrema.append(sv.tolist())
    if not np.isfinite(coverage) or not PROFILE['minimum_reference_coverage'] <= coverage <= 1:
        raise ValueError('Same-photo reference crop covers too little of the original')
    return {'supported': True, 'good_matches': len(s), 'inliers': count, 'inlier_ratio': ratio,
        'inlier_error_median_q90': [float(np.median(errors)), q90],
        'spatial': spatial, 'H_local_singular_values': extrema, 'reference_crop_coverage': float(coverage)}


def _gaussian(values, sigma):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(values, (sigma, sigma)+(0,)*(values.ndim-2), mode='nearest')


def _normal(values, weight, sigma):
    den = _gaussian(weight, sigma)
    expand = (...,)+(None,)*(values.ndim-2)
    out = np.divide(_gaussian(values*weight[expand], sigma), den[expand],
        out=np.zeros_like(values), where=den[expand] > 1e-12)
    return out, den


def _extend(values, weight, sigma):
    expand = (...,)+(None,)*(values.ndim-2)
    field = np.broadcast_to((values*weight[expand]).sum((0, 1))/max(float(weight.sum()), 1e-9), values.shape).copy()
    for scale in (8, 4, 2, 1):
        estimate, den = _normal(values, weight, sigma*scale)
        gate = np.clip(den/.2, 0, 1); gate = gate*gate*(3-2*gate)
        field = estimate*gate[expand] + field*(1-gate)[expand]
    return field


def bound_jacobians(coefficients):
    """Reduce delta dose toward identity, preserving orientation and conditioning."""
    shape = coefficients.shape
    if shape[-1] != 12 or not np.isfinite(coefficients).all():
        raise ValueError('Invalid style coefficient Jacobians')
    c = coefficients.copy().reshape((-1, 12))
    A = c[:, :9].reshape((-1, 3, 3)); eye = np.eye(3, dtype=np.float32)
    # Two convex constraints survive bilinear interpolation: sym(J)>=.1 I
    # and ||J||2<=3. They imply positive determinant, smin>=.1 and cond<=30.
    # Reduce the whole delta towards identity; do not invent a new rotation.
    symmetric = (A+A.transpose(0, 2, 1))*.5
    minimum = np.linalg.eigvalsh(symmetric)[:, 0]
    dose = np.minimum(1., .89999/np.maximum(-minimum, 1e-12)).astype(np.float32)
    needs = np.linalg.svd(eye+A*dose[:, None, None], compute_uv=False)[:, 0] > 2.99999
    if needs.any():
        low = np.zeros(int(needs.sum()), np.float32); high = dose[needs].copy()
        subset = A[needs]
        for _ in range(20):
            trial = (low+high)*.5
            largest = np.linalg.svd(eye+subset*trial[:, None, None], compute_uv=False)[:, 0]
            ok = largest <= 2.99999
            low = np.where(ok, trial, low); high = np.where(ok, high, trial)
        dose[needs] = low
    c *= dose[:, None]
    J = eye + c[:, :9].reshape((-1, 3, 3))
    sv = np.linalg.svd(J, compute_uv=False); det = np.linalg.det(J)
    if (sv.min() < PROFILE['minimum_J_singular']-2e-6 or sv.max() > PROFILE['maximum_J_singular']+2e-6
            or np.any(det <= 0) or np.any(sv[:, 0]/sv[:, -1] > PROFILE['maximum_J_condition'])):
        raise ValueError('Bounded style Jacobian verification failed')
    return c.reshape(shape), {'policy': 'positive_symmetric_part_and_spectral_norm; delta_dose_toward_identity',
        'dose_minimum': float(dose.min()), 'dose_mean': float(dose.mean()),
        'attenuated_nodes': int(np.count_nonzero(dose < 1)), 'minimum_determinant': float(det.min()),
        'singular_range': [float(sv.min()), float(sv.max())],
        'maximum_condition': float(np.max(sv[:, 0]/sv[:, -1])),
        'bilinear_bound': 'convex sym(J)>=0.1I and spectral_norm(J)<=3; holds between nodes'}


def fit_coefficients(baseline, target, validity):
    """Fit low-frequency observations; only source colours guide sharp responses."""
    guide, target, validity = map(lambda v: np.asarray(v, np.float32), (baseline, target, validity))
    if (guide.shape != target.shape or guide.shape[:2] != validity.shape or guide.shape[-1] != 3
            or not all(np.isfinite(v).all() for v in (guide, target, validity))
            or max(guide.shape[:2]) > 512 or min(guide.shape[:2]) < 2
            or np.any((guide < 0)|(guide > 1)) or np.any((target < 0)|(target > 1))
            or validity.min() < 0 or validity.max() > 1 or validity.sum() < 64):
        raise ValueError('Style fitting requires supported finite SDR proxy observations')
    I, T = _to_lab(guide), _to_lab(target)
    scale = max(guide.shape[:2])/1600
    sigma = PROFILE['window_sigma_at_1600']*scale
    low_sigma = PROFILE['observation_sigma_at_1600']*scale
    # Normalize the paired delta under one validity measure. Independently
    # filtering target and full source would create a fictitious correction
    # at a crop boundary even when the two observed photographs are identical.
    p, _ = _normal(T-I, validity, low_sigma)
    weight = validity.copy(); residual = np.zeros_like(validity)
    for _ in range(PROFILE['robust_iterations']):
        mi, _ = _normal(I, weight, sigma); mp, _ = _normal(p, weight, sigma)
        covariance = np.empty((*I.shape[:2], 3, 3), np.float32)
        cross = np.empty_like(covariance)
        for j in range(3):
            for k in range(3):
                second, _ = _normal(I[..., j]*I[..., k], weight, sigma)
                covariance[..., j, k] = second-mi[..., j]*mi[..., k]
                second, _ = _normal(I[..., j]*p[..., k], weight, sigma)
                cross[..., j, k] = second-mi[..., j]*mp[..., k]
        for j, ridge in enumerate(PROFILE['ridge_diagonal']): covariance[..., j, j] += ridge
        A = np.linalg.solve(covariance, cross)
        b = mp-_affine_delta(mi, A, np.zeros_like(mp))
        residual = np.linalg.norm(_affine_delta(I, A, b)-p, axis=-1)
        weight = validity*np.minimum(1., PROFILE['huber_OKLab']/np.maximum(residual, 1e-6))
    coefficients = _extend(np.concatenate([A.reshape((*I.shape[:2], 9)), b], axis=-1), weight, sigma)
    coefficients, bound = bound_jacobians(coefficients)
    return coefficients, {'window_sigma_proxy': sigma, 'observation_sigma_proxy': low_sigma,
        'fit_residual_median_q90': np.quantile(residual[validity > .01], [.5, .9]).tolist(),
        'jacobian': bound, 'semantic_material_isolation': False,
        'observed_pixels': int(np.count_nonzero(validity)),
        'unobserved_extrapolated_pixels': int(np.count_nonzero(validity == 0)),
        'reference_coverage_policy': 'evidence_only; no hard application hull',
        'reference_high_frequency_RGB_used_as_output': False}


def _proxy_size(size, edge):
    factor = min(1., edge/max(size))
    return tuple(max(2, round(x*factor)) for x in size)


def _master_proxy(path, native, size, cv2):
    """Area reduce native strips; never materialize another full native master."""
    from native_detail import read_scene_crop
    w, h = native; pw, ph = size
    proxy = np.zeros((ph, pw, 3), np.float32)
    for top in range(0, h, 128):
        bottom = min(h, top+128)
        values, _ = read_scene_crop(path, (0, top, w, bottom))
        if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
            raise ValueError('Baseline master has invalid display-linear RGB')
        rows = cv2.resize(values, (pw, bottom-top), interpolation=cv2.INTER_AREA)
        for j, row in enumerate(rows):
            a, b = (top+j)*ph/h, (top+j+1)*ph/h
            first, last = int(np.floor(a)), min(ph-1, int(np.ceil(b)-1))
            for y in range(first, last+1): proxy[y] += row*np.float32(min(b, y+1)-max(a, y))
    return proxy


def _worker(request_path):
    import cv2  # Deliberately worker-only: desktop runtime has no OpenCV.
    import scipy
    from PIL import Image
    from scipy.ndimage import binary_erosion
    from raw_support import open_linear, open_rgb, RGB_DECODE_VERSION
    from display_transform import source_transform_for_precision, display_linear
    from float_appearance import srgb_to_linear, linear_to_srgb
    started = time.perf_counter()
    request = json.loads(Path(request_path).read_text(encoding='utf8'))
    identity = request['identity']; directory = Path(request['output'])
    material = request.get('material_context')
    for role in ('source', 'reference'): _verified(request[role])
    from creative_direction import reference_contract_for
    role = identity.get('reference_contract', {}).get('role')
    if role == 'source_conditioned_design':
        from creative_direction import bind_source_conditioned_reference
        expected = bind_source_conditioned_reference(Path(request['root']), request['checked_task_plan'], request['reference'])
    else:
        expected = reference_contract_for({'mode': 'existing_reference', 'constraints': {
            'background_mode': 'enhance_source', 'reference_role': 'same_photo_finish'}}, request)
    if (identity.get('reference_contract') != expected or identity.get('profile') != profile_for(material is not None)
            or identity.get('producer_code') != producer_code()):
        raise ValueError('Same-photo worker request differs from checked source/reference or producer')
    if sha256_file(request['baseline_master']) != identity['baseline_master_sha256']:
        raise ValueError('Worker baseline master differs from checked identity')
    native = _master_info(request['baseline_master'])
    if material is not None:
        from creative_direction import skin_cast_contract_for
        import skin_illumination as skin
        task = request['checked_task_plan']
        goal = (task.get('creative_direction') or {}).get('material_refinement')
        if (goal != skin_cast_contract_for(task.get('intent') or {}) or goal != material.get('goal')
                or expected.get('role') != 'source_conditioned_design'
                or identity.get('material_context') != material.get('binding')
                or material.get('source_sha256') != request['source']['sha256']
                or material.get('native_size') != native
                or material.get('binding', {}).get('reference_sha256') != expected.get('reference_sha256')
                or material.get('binding', {}).get('baseline_master_sha256') != identity.get('baseline_master_sha256')
                or material.get('binding', {}).get('producer_code') != identity.get('producer_code')
                or material.get('binding', {}).get('goal_sha256') != skin._digest(goal)):
            raise ValueError('Worker skin material permission/source binding differs')
    raw, precision = open_linear(Path(request['source']['path']))
    if [raw.shape[1], raw.shape[0]] != native: raise ValueError('Baseline and original source canvases differ')
    source_transform = source_transform_for_precision(precision['source_precision'])
    for top in range(0, len(raw), 128): raw[top:top+128] = display_linear(raw[top:top+128], source_transform)
    size = _proxy_size(native, PROFILE['registration_long_edge'])
    original = cv2.resize(raw, size, interpolation=cv2.INTER_AREA); del raw
    ref_image = open_rgb(Path(request['reference']['path']))
    ref_native = list(ref_image.size); ref_size = _proxy_size(ref_native, PROFILE['registration_long_edge'])
    reference = srgb_to_linear(np.asarray(ref_image.resize(ref_size, Image.Resampling.LANCZOS), np.float32)/255)
    clahe = cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8))
    def gray(v):
        encoded = np.uint8(np.clip(linear_to_srgb(v), 0, 1)*255+.5)
        return clahe.apply(cv2.cvtColor(encoded, cv2.COLOR_RGB2GRAY))
    sift = cv2.SIFT_create(nfeatures=PROFILE['sift_features'])
    ks, ds = sift.detectAndCompute(gray(original), None)
    kr, dr = sift.detectAndCompute(gray(reference), None)
    if ds is None or dr is None: raise ValueError('No same-photo SIFT evidence')
    matches = cv2.BFMatcher().knnMatch(ds, dr, k=2)
    good = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < PROFILE['match_ratio']*pair[1].distance]
    if len(good) < PROFILE['minimum_inliers']: raise ValueError('Too few same-photo feature matches')
    sp = np.float32([ks[m.queryIdx].pt for m in good]); rp = np.float32([kr[m.trainIdx].pt for m in good])
    H, inliers = cv2.findHomography(sp, rp, cv2.RANSAC, PROFILE['ransac_pixels'], maxIters=10000, confidence=.999)
    if H is None: raise ValueError('No healthy same-photo registration')
    coverage = cv2.warpPerspective(np.ones(reference.shape[:2], np.float32), np.linalg.inv(H), size)
    quality = registration_quality(sp, rp, inliers, H, size, ref_size, coverage=float((coverage > .999).mean()))
    grid = _proxy_size(native, PROFILE['coefficient_long_edge'])
    baseline = _master_proxy(request['baseline_master'], native, grid, cv2)
    # Transform reference proxy pixel centres into the source coefficient grid.
    def resize_H(old, new):
        x, y = new[0]/old[0], new[1]/old[1]
        return np.array([[x, 0, (x-1)/2], [0, y, (y-1)/2], [0, 0, 1.]])
    source_to_grid = resize_H(size, grid)
    ref_to_grid = source_to_grid @ np.linalg.inv(H)
    aligned = cv2.warpPerspective(reference, ref_to_grid, grid, flags=cv2.INTER_LINEAR)
    validity = cv2.warpPerspective(np.ones(reference.shape[:2], np.float32), ref_to_grid, grid)
    validity = binary_erosion(validity > .999, iterations=1).astype(np.float32)
    coefficients, fit = fit_coefficients(baseline, aligned, validity)
    coefficient_path = directory/'coefficients.npz'
    observation_confidence = _gaussian(validity, fit['window_sigma_proxy'])
    np.savez_compressed(coefficient_path, values=coefficients,
        observation_validity=validity, observation_confidence=observation_confidence)
    coefficient_asset = _asset(coefficient_path, size=list(grid), native_size=native,
        coordinate_space='source_canvas', representation='source_OKLab_to_delta_affine_9A_3b',
        sampling=PROFILE['coefficient_sampling'])
    schema = SCHEMA_V2 if material is not None else SCHEMA
    receipt = {'schema': schema, 'profile': profile_for(material is not None), 'identity': identity,
        'reference_contract': identity['reference_contract'], 'registration': quality,
        'source_to_reference_proxy_H': H.tolist(), 'source_registration_size': list(size),
        'source_native_to_reference_native_H': (np.linalg.inv(resize_H(ref_native, ref_size))@H@resize_H(native, size)).tolist(),
        'reference_registration_size': list(ref_size), 'reference_native_size': ref_native,
        'source_display_transform': source_transform, 'decode_version': RGB_DECODE_VERSION,
        'reference_color_management': ref_image.info.get('color_management'), 'fit': fit,
        'runtime_versions': {'numpy': np.__version__, 'scipy': scipy.__version__, 'opencv': cv2.__version__},
        'coefficients_sha256': coefficient_asset['sha256'],
        'source_geometry_changed': False, 'source_texture_bitexact': False,
        'reference_RGB_composited': False, 'model_submissions': 0,
        'reference_crop_application': 'smooth normalized coefficient extrapolation; no validity hull dose',
        'colour_ownership': identity.get('colour_ownership'),
        'observation_evidence': {'asset': 'coefficients', 'validity_member': 'observation_validity',
            'smooth_confidence_member': 'observation_confidence',
            'confidence_is_application_dose': False},
        'quality_status': 'needs_visual_review', 'elapsed_seconds': time.perf_counter()-started,
        'limitations': ['Source-edge guided only; semantic/material separation is not inferred.',
            'Reference crop extrapolation is an appearance estimate, not new reference observations.',
            'Low-frequency style does not reconstruct fog assets, leaves or reference hair strands.',
            'Same-photo registration does not certify local retouch warps or photographic relighting.']}
    receipt_path = directory/'contract.json'; _write(receipt_path, receipt)
    layer = {'id': KIND, 'kind': KIND, 'enabled': True, 'strength': 1.,
        'params': {'schema': schema, 'identity_sha256': _digest(identity), 'native_size': native},
        'assets': {'coefficients': coefficient_asset, 'contract': _asset(receipt_path)}}
    if material is not None:
        import skin_illumination as skin
        fine_size = tuple(material['source_assets']['source_semantics']['size'])
        fine = _master_proxy(request['baseline_master'], native, fine_size, cv2)
        fine_c = _sample(coefficients, (0, 0, *fine_size), fine_size)
        fine_lab = _to_lab(fine)
        fine_delta = _affine_delta(fine_lab, fine_c[..., :9].reshape((*fine_c.shape[:2], 3, 3)), fine_c[..., 9:])
        styled, _ = _ray_gamut(fine, _from_lab(fine_lab+fine_delta), 1.)
        styled[np.all(fine_c == 0, axis=-1)] = fine[np.all(fine_c == 0, axis=-1)]
        skin_assets, skin_evidence = skin.prepare(styled, material, native, directory)
        layer['assets'].update(skin_assets)
        receipt['material_illumination'] = {'schema': skin.SCHEMA,
            'receipt_asset': copy.deepcopy(skin_assets['skin_receipt']),
            'processed_domains': skin_evidence['processed_domains'],
            'reference_RGB_sampled': False, 'nail_RGB_bitexact': False}
        receipt['limitations'][0] = 'Reference field is source-edge guided; optional explicitly permitted source-owned skin illumination is separate.'
        del fine, fine_c, fine_lab, fine_delta, styled
    # Proxy diagnostics describe coefficient-grid response, not native proof.
    c = coefficients[..., :9].reshape((*grid[::-1], 3, 3))
    lab = _to_lab(baseline)
    delta = _affine_delta(lab, c, coefficients[..., 9:])
    candidate, gdose = _ray_gamut(baseline, _from_lab(lab+delta), 1.)
    receipt['proxy_application'] = {'gamut_dose_minimum': float(gdose.min()), 'gamut_dose_mean': float(gdose.mean()),
        'gamut_attenuated_fraction': float((gdose < 1).mean()),
        'OKLab_error_before_median_q90': np.quantile(np.linalg.norm(_to_lab(baseline)-_to_lab(aligned), axis=-1)[validity > .99], [.5, .9]).tolist(),
        'OKLab_error_after_median_q90': np.quantile(np.linalg.norm(_to_lab(candidate)-_to_lab(aligned), axis=-1)[validity > .99], [.5, .9]).tolist()}
    if material is not None:
        receipt['proxy_application']['scope'] = 'base_reference_field_before_explicit_material_illumination'
        frozen_skin = skin.validate(layer['assets'], skin_evidence, material['binding'])
        candidate, material_report = skin.apply(candidate, frozen_skin, (0, 0, *grid), grid)
        receipt['material_proxy_application'] = material_report
    for name, image in (('baseline-preview.png', baseline), ('candidate-preview.png', candidate), ('reference-registered-preview.png', aligned)):
        Image.fromarray(np.uint8(np.clip(linear_to_srgb(image), 0, 1)*255+.5)).save(directory/name)
    receipt['elapsed_seconds'] = time.perf_counter()-started
    _write(receipt_path, receipt); layer['assets']['contract'] = _asset(receipt_path)
    _write(directory/'layer.json', layer)
    print(json.dumps({'layer': str(directory/'layer.json'), 'elapsed_seconds': receipt['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--worker', required=True)
    _worker(parser.parse_args().worker)
