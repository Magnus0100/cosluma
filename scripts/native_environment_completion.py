"""Register the completed environment plate without silhouette donor injection.

Topological source windows are gap *proposals*, never authority to transplant
design-window appearance. The existing cleaned/SR plate owns environment RGB.
Its paired P/V stays immutable; candidate person RGB and source RGB are not read.
Unqualified old window deltas are removed rather than added to correct P again.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes, map_coordinates

SCHEMA = 'native_environment_completion.v1'
ORIGIN_CODES = {'unknown_or_outside_mapping': 0, 'design_background_observed_resampled': 1,
                'existing_LaMa_completion_and_background_SR': 2}


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _bound(asset):
    if not isinstance(asset, dict) or not asset.get('path') or not asset.get('sha256'):
        raise ValueError('environment_asset_binding_missing')
    path = Path(asset['path']).resolve()
    if not path.is_file() or _sha(path) != asset['sha256']:
        raise ValueError('environment_asset_sha256_mismatch:' + path.name)
    return path


def _mask_view(asset, output_size):
    path = _bound(asset)
    if path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {'values'}:
                raise ValueError('environment_coverage_array_keys_wrong')
            values = data['values']
        if (values.ndim != 2 or not np.isfinite(values).all() or
                np.any(values < 0) or np.any(values > 1)):
            raise ValueError('environment_coverage_array_invalid')
        if tuple(asset.get('size', ())) != (values.shape[1], values.shape[0]):
            raise ValueError('environment_coverage_array_size_mismatch')
        image = Image.fromarray(values.astype(np.float32))
    else:
        with Image.open(path) as source:
            image = Image.fromarray(np.asarray(source.convert('L'), np.float32) / 255)
    return np.asarray(image.resize(output_size, Image.Resampling.BILINEAR), np.float32)


def _origin_map(plan, native, proxy, validity, candidate_size):
    removal_asset = (plan.get('mask_roles') or {}).get('candidate_removal')
    mapping = (plan.get('canvas') or {}).get('candidate_mapping') or {}
    matrix = mapping.get('candidate_original_to_final_original')
    origin = np.zeros(validity.shape, np.uint8)
    if not removal_asset or matrix is None:
        return origin, {'state': 'unknown', 'reason': 'removal_or_registration_observation_missing'}
    path = _bound(removal_asset)
    with Image.open(path) as image:
        removed = np.asarray(image.convert('L'), np.float32) / 255
    inverse = np.linalg.inv(np.asarray(matrix, np.float64))
    if inverse.shape != (3, 3) or not np.isfinite(inverse).all():
        raise ValueError('environment_registration_matrix_invalid')
    x = (np.arange(proxy[0], dtype=np.float64)+.5)*native[0]/proxy[0]-.5
    y = (np.arange(proxy[1], dtype=np.float64)+.5)*native[1]/proxy[1]-.5
    xx, yy = np.meshgrid(x, y)
    denominator = inverse[2, 0]*xx + inverse[2, 1]*yy + inverse[2, 2]
    if np.any(np.abs(denominator) < 1e-12):
        raise ValueError('environment_registration_at_infinity')
    tx = (inverse[0, 0]*xx + inverse[0, 1]*yy + inverse[0, 2])/denominator
    ty = (inverse[1, 0]*xx + inverse[1, 1]*yy + inverse[1, 2])/denominator
    valid = (validity > 0) & (tx >= -.5) & (ty >= -.5) & (tx < candidate_size[0]-.5) & (ty < candidate_size[1]-.5)
    mx = (tx+.5)*removed.shape[1]/candidate_size[0]-.5
    my = (ty+.5)*removed.shape[0]/candidate_size[1]-.5
    removal = map_coordinates(removed, [my, mx], order=0, mode='constant', cval=1, prefilter=False)
    origin[valid & (removal <= .5)] = ORIGIN_CODES['design_background_observed_resampled']
    origin[valid & (removal > .5)] = ORIGIN_CODES['existing_LaMa_completion_and_background_SR']
    return origin, {'state': 'registered', 'removal_sha256': removal_asset['sha256'],
        'selected_reference_header_size': list(candidate_size),
        'meaning': 'existing candidate cleaning provenance; not geometric confidence or RAW scene truth',
        'registration_sha256': hashlib.sha256(json.dumps(matrix, sort_keys=True).encode()).hexdigest()}


def prepare_native_environment_completion(run, plan, output_dir):
    """Register correct existing P/V, gap proposals and origin; no new inference.

    A missing/visually broken cleaned plate remains incomplete and requires an
    explicitly bounded environment-only completion, not a silhouette RGB donor.
    This operation cannot certify all scene defects from a topology mask.
    """
    directory = Path(output_dir).resolve(); directory.mkdir(parents=True, exist_ok=True)
    background = next(layer for layer in plan['layers'] if layer['kind'] == 'background_design')
    assets = background['assets']; native = tuple(plan['canvas']['source_oriented_size'])
    proxy = tuple(plan['canvas']['source_proxy_size'])
    p, v = assets['background'], assets['validity']
    if (p.get('representation') != 'sampling_validity_associated_RGB' or p.get('validity_associated') is not True or
            p.get('coordinate_space') != 'source_canvas' or v.get('coordinate_space') != 'source_canvas' or
            tuple(p.get('size', ())) != native or tuple(v.get('size', ())) != native):
        raise ValueError('environment_requires_registered_native_associated_PV')
    _bound(p); _bound(v)
    alpha_asset = plan['mask_roles']['foreground_alpha']
    alpha = _mask_view(alpha_asset, proxy)
    validity = _mask_view(v, proxy)
    gap = (binary_fill_holes(alpha > .5) & (alpha <= .005)).astype(np.float32)
    reference_path = _bound(run['reference'])
    # Header/grid only: no generated person or candidate RGB enters this
    # registration. Do not infer an absent original grid from proxy masks.
    with Image.open(reference_path) as reference_header:
        candidate_size = reference_header.size
    origin, origin_registration = _origin_map(plan, native, proxy, validity, candidate_size)
    gap_path = directory/'environment-gap-proposals.npz'
    origin_path = directory/'environment-origin.npz'
    np.savez_compressed(gap_path, values=gap)
    np.savez_compressed(origin_path, values=origin)
    def mask_asset(path, role, categorical=False):
        return {'path': str(path), 'sha256': _sha(path), 'size': list(proxy),
            'native_size': list(native), 'coordinate_space': 'source_proxy', 'role': role,
            'representation': 'categorical_origin' if categorical else 'coverage_zero_topological_window_proposal',
            'sampling': 'nearest' if categorical else 'bilinear', 'foreground_alpha': False,
            'geometric_applicability_established': False, 'render_weight': False}
    output_assets = {'background': copy.deepcopy(p), 'validity': copy.deepcopy(v),
        'gap_mask': mask_asset(gap_path, 'environment_internal_gap_proposal'),
        'origin_map': mask_asset(origin_path, 'registered_design_environment_origin', True)}
    old_delta = assets.get('observed_window_delta')
    binding = {'source_sha256': run['source']['sha256'], 'reference_sha256': run['reference']['sha256'],
        'background_sha256': p['sha256'], 'validity_sha256': v['sha256'], 'source_canvas_size': list(native)}
    report = {'schema': SCHEMA, 'source_binding': binding,
        'action': 'retain_completed_PV_remove_unqualified_silhouette_donor_reinjection',
        'prior_observed_window_sha256': old_delta.get('sha256') if old_delta else None,
        'prior_observed_window_reason': 'source/design empty-alpha topology does not establish environment appearance donor qualification',
        'source_alpha_proposal_sha256': alpha_asset['sha256'],
        'source_geometry_or_alpha_changed': False, 'background_PV_bytes_changed': False,
        'candidate_or_source_person_RGB_read': False, 'gap_proposals_proxy_pixels': int((gap > 0).sum()),
        'origin_codes': ORIGIN_CODES, 'origin_registration': origin_registration,
        'origin_counts': {name: int((origin == code).sum()) for name, code in ORIGIN_CODES.items()},
        'asset_sha256': {name: asset['sha256'] for name, asset in output_assets.items()},
        'environment_completion_globally_certified': False,
        'new_LaMa_RAW_GPT_model_calls': 0, 'runtime_research_models_required': False,
        'producer_code_sha256': _sha(__file__)}
    report_path = directory/'environment-completion.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', 'utf8')
    return {**report, 'assets': output_assets, 'descriptor_asset': {
        'path': str(report_path), 'sha256': _sha(report_path), 'role': 'native_environment_completion_execution'}}


def apply_environment_completion(plan, completion):
    """Return the sole plan with registered P/V and no rejected window branch.

    Renderer arithmetic remains current*(1-strength*V)+strength*P, then final
    over once. No additional RGB operation, darkening, blur or alpha edit.
    """
    if completion.get('schema') != SCHEMA:
        raise ValueError('environment_completion_schema_wrong')
    result = copy.deepcopy(plan)
    layer = next(layer for layer in result['layers'] if layer['kind'] == 'background_design')
    original = layer['assets']; binding = completion['source_binding']
    if (binding['source_sha256'] != result['source_sha256'] or
            binding['reference_sha256'] != result['candidate_sha256'] or
            binding['background_sha256'] != original['background']['sha256'] or
            binding['validity_sha256'] != original['validity']['sha256'] or
            binding['source_canvas_size'] != result['canvas']['source_oriented_size']):
        raise ValueError('environment_completion_registered_pair_mismatch')
    for name, asset in completion['assets'].items():
        _bound(asset)
        key = {'gap_mask': 'environment_gap_mask', 'origin_map': 'environment_origin'}.get(name, name)
        original[key] = copy.deepcopy(asset)
    _bound(completion['descriptor_asset'])
    original['environment_completion'] = copy.deepcopy(completion['descriptor_asset'])
    original.pop('observed_window_delta', None)
    provenance = result.setdefault('provenance', {})
    provenance.pop('observed_background_windows', None)
    provenance['environment_completion'] = {key: copy.deepcopy(value) for key, value in completion.items()
                                           if key not in {'assets', 'descriptor_asset'}}
    return result
