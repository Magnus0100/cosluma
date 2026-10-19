"""Preserve observed native receiver detail inside verified paired reflections.

This CPU producer transfers luminance bands, not a source floor RGB plate.
Candidate low-frequency colour remains the basis. Original source pixels keep
their native coordinates; foreground coverage and contact attenuation are not
consumed again. Missing qualification returns an unchanged plan and unknown.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates
from scipy.spatial import ConvexHull
from asset_identity import sha256_file

SCHEMA = 'native_receiver_preservation.v1'
RECIPE = 'qualified_native_receiver_luminance_bands_v1'
DOMAIN = 'display_linear_sRGB'
CONFIG = {'lowpass_sigma_candidate_pixels': 4., 'inward_feather_candidate_pixels': 1.,
          'max_native_roi_pixels': 2_000_000, 'minimum_joint_receiver': .5,
          'minimum_sampling_validity': .999, 'foreground_alpha_required': 0.,
          'relative_contrast_gain_bounds': [.5, 2.]}
LUMA = np.array([.2126, .7152, .0722], np.float32)


def sha(path):
    return sha256_file(path)


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def array(asset):
    if sha(asset['path']) != asset['sha256']:
        raise ValueError('Native receiver asset bytes changed')
    path = Path(asset['path'])
    if path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as resource:
            values = np.asarray(resource['values'], np.float32)
    else:
        from PIL import Image
        values = np.asarray(Image.open(path).convert('L'), np.float32)/255
    if list(values.shape[:2][::-1]) != asset['size'] or not np.isfinite(values).all():
        raise ValueError('Native receiver asset grid or finite values changed')
    return values


def sample_mask(values, native_to_input, box):
    """Nearest categorical support in native coordinates, without RGB warping."""
    left, top, right, bottom = map(int, box)
    y, x = np.mgrid[top:bottom, left:right].astype(np.float64)
    matrix = np.asarray(native_to_input, np.float64)
    denominator = matrix[2, 0]*x+matrix[2, 1]*y+matrix[2, 2]
    usable = np.abs(denominator) > 1e-10
    denominator = np.where(usable, denominator, 1)
    sx = (matrix[0, 0]*x+matrix[0, 1]*y+matrix[0, 2])/denominator
    sy = (matrix[1, 0]*x+matrix[1, 1]*y+matrix[1, 2])/denominator
    usable &= np.isfinite(sx) & np.isfinite(sy)
    sx = np.where(usable, sx, -1).astype(np.float32)
    sy = np.where(usable, sy, -1).astype(np.float32)
    return map_coordinates(np.asarray(values, np.float32), [sy, sx], order=0,
        mode='constant', cval=0, prefilter=False)


def _scale(a, b):
    return np.diag([b[0]/a[0], b[1]/a[1], 1.]).astype(np.float64)


def qualified_support(plan, run):
    """Verify frozen observation bytes, distributed inliers and joint receiver."""
    from reference_appearance_prepare import preserve_contact_layers
    from local_floor_pollution import PARAMS
    native = plan['canvas']['source_oriented_size']
    proxy = plan['canvas']['source_proxy_size']
    mapping = plan['canvas'].get('candidate_mapping') or {}
    contract = (plan.get('provenance', {}).get('candidate_floor_contract') or {}).get('reflection_ownership') or {}
    roles = plan.get('mask_roles') or {}
    protection = roles.get('candidate_reflection_protection')
    source_fragment = roles.get('source_observed_floor_fragments')
    unknown = {'state': 'unknown', 'qualified': False, 'reason': 'paired_reflection_observation_unqualified'}
    if (contract.get('qualified') is not True or contract.get('mapping_supported') is not True or
            mapping.get('scene_registration_supported') is not True or not protection or not source_fragment):
        return None, unknown
    binding = contract.get('binding') or {}
    if (contract.get('schema') != 'observed_floor_reflection_ownership.v1' or
            binding.get('source_sha256') != run['source']['sha256'] or
            binding.get('candidate_sha256') != run['reference']['sha256'] or
            binding.get('protection_sha256') != protection['sha256'] or
            binding.get('source_detected_fragment_sha256') != source_fragment['sha256'] or
            protection.get('coordinate_space') != 'candidate_scene_proxy' or
            protection.get('role') != 'paired_observed_floor_reflection_protection' or
            source_fragment.get('coordinate_space') != 'source_scene_proxy'):
        raise ValueError('Native receiver paired observation binding changed')
    parameters = contract.get('observation_bindings', {}).get('qualification_params')
    if parameters != PARAMS:
        raise ValueError('Native receiver reflection qualification parameters changed')
    matrix = np.asarray(mapping.get('candidate_original_to_final_original'), np.float64)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all() or np.linalg.det(matrix) <= 0):
        raise ValueError('Native receiver supported mapping changed')
    original_candidate = protection.get('original_canvas_size', protection['size'])
    candidate_to_native = matrix @ _scale(protection['size'], original_candidate)
    native_to_candidate = np.linalg.inv(candidate_to_native)
    protected = array(protection)
    observed = array(source_fragment)
    if np.any((protected < 0) | (protected > 1)) or np.any((observed < 0) | (observed > 1)):
        raise ValueError('Native receiver observation masks outside unit interval')
    if int((protected > .5).sum()) != contract.get('protected_pixels'):
        raise ValueError('Native receiver protected observation count changed')
    verified = []
    for part in contract.get('components', []):
        if part.get('qualified') is not True:
            continue
        sp, cp = np.asarray(part.get('source_inlier_points'), float), np.asarray(part.get('candidate_inlier_points'), float)
        if (sp.ndim != 2 or sp.shape[1:] != (2,) or sp.shape != cp.shape or
                not np.isfinite(sp).all() or not np.isfinite(cp).all()):
            raise ValueError('Native receiver paired photographic inliers missing')
        unique = np.unique(np.round(cp, 3), axis=0)
        try:
            fraction = ConvexHull(unique).volume/max(1, part['pixels'])
        except Exception:
            fraction = 0.
        source_native = sp * (np.asarray(native)/np.asarray(source_fragment['size']))
        projected = np.column_stack([source_native, np.ones(len(sp))]) @ native_to_candidate.T
        error = np.linalg.norm(projected[:, :2]/projected[:, 2:] - cp, axis=1)
        if (len(unique) < PARAMS['minimum_reflection_inliers'] or
                len(unique) != part.get('verified_unique_inliers') or
                fraction < PARAMS['minimum_inlier_hull_fraction'] or
                part.get('candidate_overlap', 0) < PARAMS['minimum_candidate_overlap'] or
                part.get('source_overlap', 0) < PARAMS['minimum_source_overlap'] or
                np.max(error) > PARAMS['maximum_local_reprojection_pixels']):
            raise ValueError('Native receiver distributed paired geometry qualification changed')
        for points, mask in ((sp, observed), (cp, protected)):
            xy = np.rint(points).astype(int)
            if ((xy < 0).any() or (xy[:, 0] >= mask.shape[1]).any() or
                    (xy[:, 1] >= mask.shape[0]).any() or not (mask[xy[:, 1], xy[:, 0]] > .5).all()):
                raise ValueError('Native receiver photographic inliers outside bound observed masks')
        verified.append({'candidate_component': part['candidate_component'], 'unique_inliers': len(unique),
            'inlier_hull_fraction': float(fraction), 'maximum_reprojection_pixels': float(error.max())})
    if not verified:
        return None, unknown
    contacts, _ = preserve_contact_layers(plan, run)
    contact = next((row for row in contacts if row['params']['receiver_qualification']['qualified_for_automatic_contact']), None)
    if not contact:
        return None, {**unknown, 'reason': 'current_joint_receiver_unqualified'}
    assets = contact['assets']
    joint, source_floor = array(assets['receiver_support']), array(assets['source_support_surface'])
    if joint.shape != tuple(proxy[::-1]) or source_floor.shape != joint.shape:
        raise ValueError('Native receiver joint support sampling grid changed')
    native_to_proxy = _scale(native, proxy)
    # The small proxy only finds the bounded native ROI. Final permission is
    # intersected again directly from all original masks on the native grid.
    native_box = (0, 0, *proxy)
    source_small = sample_mask(observed, _scale(proxy, source_fragment['size']), native_box)
    proxy_to_candidate = native_to_candidate @ _scale(proxy, native)
    candidate_small = sample_mask(protected, proxy_to_candidate, native_box)
    small = (source_small > .5) & (candidate_small > .5) & (joint > .5) & (source_floor > .5)
    ys, xs = np.nonzero(small)
    if not len(xs):
        return None, {**unknown, 'reason': 'qualified_observation_has_no_current_native_receiver_support'}
    # Bandwidth follows donor sampling pitch, never image-specific coordinates.
    pitch = max(np.linalg.norm(candidate_to_native[:2, 0]), np.linalg.norm(candidate_to_native[:2, 1]))
    sigma = max(2., CONFIG['lowpass_sigma_candidate_pixels'] * pitch)
    margin = int(np.ceil(4*sigma+pitch+2))
    box = [max(0, int(xs.min()*native[0]/proxy[0])-margin),
           max(0, int(ys.min()*native[1]/proxy[1])-margin),
           min(native[0], int(np.ceil((xs.max()+1)*native[0]/proxy[0]))+margin),
           min(native[1], int(np.ceil((ys.max()+1)*native[1]/proxy[1]))+margin)]
    if (box[2]-box[0])*(box[3]-box[1]) > CONFIG['max_native_roi_pixels']:
        return None, {**unknown, 'reason': 'observed_receiver_exceeds_bounded_native_roi_budget', 'native_box': box}
    masks = {'reflection': (sample_mask(observed, _scale(native, source_fragment['size']), box) > .5) &
                         (sample_mask(protected, native_to_candidate, box) > .5),
             'receiver': (sample_mask(joint, native_to_proxy, box) > .5) &
                         (sample_mask(source_floor, native_to_proxy, box) > .5)}
    report = {'state': 'qualified_observed_receiver', 'qualified': True, 'native_box': box,
        'native_roi_pixels': (box[2]-box[0])*(box[3]-box[1]), 'candidate_sampling_pitch_native': float(pitch),
        'sigma_native_pixels': float(sigma), 'qualification': verified, 'contract_sha256': identity(contract),
        'observation_assets': {key: copy.deepcopy(value) for key, value in
            (('source_fragment', source_fragment), ('candidate_protection', protection),
             ('joint_receiver', assets['receiver_support']), ('source_receiver', assets['source_support_surface']))},
        'source_rgb_warped': False, 'unqualified_components_consumed': False,
        'complete_reflection_segmentation': False}
    return (masks, box, sigma, pitch), report


def transfer_receiver_bands(source, candidate, permitted, receiver, *, sigma, feather):
    """Transfer only normalized luminance detail; zero permission is exact."""
    source, candidate = np.asarray(source, np.float32), np.asarray(candidate, np.float32)
    permitted, receiver = np.asarray(permitted, bool), np.asarray(receiver, bool)
    if (source.shape != candidate.shape or source.shape[:2] != permitted.shape or
            receiver.shape != permitted.shape or source.shape[2:] != (3,) or
            not np.isfinite(source[receiver]).all() or not np.isfinite(candidate).all() or
            (source[receiver] < 0).any() or (source[receiver] > 1).any() or
            (candidate < 0).any() or (candidate > 1).any() or not np.isfinite([sigma, feather]).all() or
            sigma <= 0 or feather <= 0):
        raise ValueError('Native receiver transfer requires finite unit native RGB and bounded masks')
    support = permitted & receiver
    weight = np.minimum(distance_transform_edt(support)/feather, 1).astype(np.float32)
    weight[~support] = 0
    result = candidate.copy()
    if not support.any():
        return result, weight, {'state': 'unknown', 'changed_pixels': 0, 'outside_support_exact': True}

    def smooth(value):
        return gaussian_filter(value, sigma, mode='reflect', truncate=4.)

    denominator = smooth(receiver.astype(np.float32))
    normalizer = np.maximum(denominator, 1e-8)
    source_y = np.where(receiver, source @ LUMA, 0)
    candidate_y = candidate @ LUMA
    low_source = smooth(source_y)/normalizer
    low_candidate = smooth(np.where(receiver, candidate_y, 0))/normalizer
    low_rgb = np.stack([smooth(np.where(receiver, candidate[..., channel], 0))/normalizer
                        for channel in range(3)], axis=-1)
    gain = np.clip(low_candidate/np.maximum(low_source, 1e-6), *CONFIG['relative_contrast_gain_bounds'])
    detail = (source_y-low_source)*gain - (candidate_y-low_candidate)
    chroma = low_rgb/np.maximum(low_candidate[..., None], 1e-6)
    delta = detail[..., None]*chroma*weight[..., None]
    delta[~support] = 0
    # Reduce the added difference along one ray; never clip the underlying plate.
    dose = np.ones(permitted.shape, np.float32)
    for channel in range(3):
        d, c = delta[..., channel], candidate[..., channel]
        np.minimum(dose, np.divide(1-c, d, out=np.ones_like(d), where=d > 0), out=dose)
        np.minimum(dose, np.divide(-c, d, out=np.ones_like(d), where=d < 0), out=dose)
    dose = np.clip(dose, 0, 1)
    dose = np.where(dose < 1, dose*(1-4*np.finfo(np.float32).eps), dose)
    result[support] = candidate[support]+delta[support]*dose[support, None]
    if not np.isfinite(result).all() or result.min() < 0 or result.max() > 1:
        raise ValueError('Native receiver gamut reduction did not retain the unit domain')
    return result, weight, {'state': 'prepared', 'changed_pixels': int(np.any(result != candidate, axis=-1).sum()),
        'outside_support_exact': bool(np.array_equal(result[~support], candidate[~support])),
        'source_high_frequency_luminance_consumed': True, 'source_low_frequency_RGB_copied': False,
        'native_contour_coordinates_changed': False, 'gamut_reduced_pixels': int((support & (dose < 1)).sum()),
        'relative_contrast_gain_range': [float(gain[support].min()), float(gain[support].max())],
        'low_frequency_colour_basis': 'candidate_receiver_normalized_gaussian',
        'support_inward_feather': True, 'permitted_pixels': int(support.sum())}


def prepare_native_receiver_preservation(run, plan, output):
    """Create a separate frozen native background for fresh prepare or CPU child."""
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    result = copy.deepcopy(plan)
    support, report = qualified_support(plan, run)
    receipt = {'schema': SCHEMA, 'recipe': RECIPE, 'config': CONFIG,
        'producer_sha256': sha(__file__), 'source_sha256': run['source']['sha256'],
        'reference_sha256': run['reference']['sha256'], 'qualification': report,
        'state': 'unknown', 'visual_acceptance': False, 'whole_floor_replaced': False,
        'synthetic_reflection_created': False, 'foreground_alpha_changed': False,
        'source_or_reference_file_changed': False, 'old_plan_changed': False}
    if support is None:
        write(output/'receiver-preservation.json', receipt)
        return {'ok': True, 'design_reconstruction_plan': result, 'receiver_preservation': receipt, 'cache_hit': False}
    native = plan['canvas']['source_oriented_size']
    baseline = plan.get('source_baseline') or {}
    bg_layer = next(row for row in result['layers'] if row['kind'] == 'background_design')
    bg_asset, validity = bg_layer['assets']['background'], bg_layer['assets']['validity']
    old_asset = baseline.get('assets', {}).get('old_environment')
    observed_alpha_asset = baseline.get('assets', {}).get('foreground_alpha')
    alpha_asset = plan['mask_roles']['foreground_alpha']
    if (baseline.get('working_domain') != DOMAIN or not old_asset or
            not observed_alpha_asset or observed_alpha_asset.get('size') != native or
            observed_alpha_asset.get('coordinate_space') != 'source_canvas' or
            old_asset.get('size') != native or old_asset.get('coordinate_space') != 'source_canvas' or
            alpha_asset.get('size') != native or bg_asset.get('size') != native or
            bg_asset.get('coordinate_space') != 'source_canvas' or validity.get('size') != native or
            bg_asset.get('representation') != 'sampling_validity_associated_RGB'):
        receipt['qualification'].update(state='unknown', qualified=False, reason='native_paired_receiver_baseline_unavailable')
        write(output/'receiver-preservation.json', receipt)
        return {'ok': True, 'design_reconstruction_plan': result, 'receiver_preservation': receipt, 'cache_hit': False}
    previous = bg_layer['assets'].get('native_receiver_preservation')
    if previous:
        if sha(previous['path']) != previous['sha256']:
            raise ValueError('Frozen native receiver preservation receipt changed')
        saved = json.loads(Path(previous['path']).read_text(encoding='utf8'))
        bindings = saved.get('binding') or {}
        if (saved.get('recipe') != RECIPE or saved.get('producer_sha256') != receipt['producer_sha256'] or
                saved.get('config') != CONFIG or saved.get('qualification', {}).get('contract_sha256') != report['contract_sha256'] or
                bindings.get('source_environment_sha256') != old_asset['sha256'] or
                bindings.get('alpha_sha256') != alpha_asset['sha256'] or
                bindings.get('observed_source_alpha_sha256') != observed_alpha_asset['sha256'] or
                bindings.get('validity_sha256') != validity['sha256'] or
                saved.get('output_background', {}).get('sha256') != bg_asset['sha256'] or sha(bg_asset['path']) != bg_asset['sha256']):
            raise ValueError('Receiver preparation changed; reprepare from the unmodified parent plate')
        return {'ok': True, 'design_reconstruction_plan': result, 'receiver_preservation': saved, 'cache_hit': True}
    masks, box, sigma, pitch = support
    left, top, right, bottom = box
    sl = np.s_[top:bottom, left:right]
    alpha = array(alpha_asset)[sl].copy()
    observed_alpha = (alpha if observed_alpha_asset['sha256'] == alpha_asset['sha256']
                      else array(observed_alpha_asset)[sl].copy())
    coverage = array(validity)[sl].copy()
    if not ((alpha >= 0) & (alpha <= 1) & (observed_alpha >= 0) & (observed_alpha <= 1) &
            (coverage >= 0) & (coverage <= 1)).all():
        raise ValueError('Native receiver coverage outside unit interval')
    source = array(old_asset)[sl].copy()
    background = array(bg_asset)
    original = background[sl].copy()
    straight = np.divide(original, coverage[..., None], out=np.zeros_like(original), where=coverage[..., None] > 1e-8)
    receiver = masks['receiver'] & (alpha == 0) & (observed_alpha == 0) & (coverage >= CONFIG['minimum_sampling_validity'])
    permitted = masks['reflection'] & receiver
    after, weight, checks = transfer_receiver_bands(source, straight, permitted, receiver,
        sigma=sigma, feather=max(1., pitch*CONFIG['inward_feather_candidate_pixels']))
    changed = weight > 0
    patch = original.copy()
    patch[changed] = after[changed]*coverage[changed, None]
    if not np.array_equal(patch[~changed], original[~changed]) or np.any(changed & (alpha > 0)):
        raise ValueError('Native receiver support or foreground invariance failed')
    background[sl] = patch
    target = output/'native-receiver-background.npz'
    np.savez_compressed(target, values=background)
    del background
    new_asset = {**bg_asset, 'path': str(target), 'sha256': sha(target),
        'native_receiver_detail_origin': 'qualified_source_native_high_frequency_luminance_only'}
    support_path = output/'native-receiver-support.npz'
    np.savez_compressed(support_path, values=weight)
    receipt.update(state='prepared' if changed.any() else 'unknown', binding={
        'source_environment_sha256': old_asset['sha256'], 'alpha_sha256': alpha_asset['sha256'],
        'observed_source_alpha_sha256': observed_alpha_asset['sha256'],
        'input_background_sha256': bg_asset['sha256'], 'validity_sha256': validity['sha256'],
        'receiver_contract_sha256': report['contract_sha256']},
        source_environment=copy.deepcopy(old_asset), input_background=copy.deepcopy(bg_asset),
        output_background=copy.deepcopy(new_asset), support_asset={
            'path': str(support_path), 'sha256': sha(support_path), 'size': list(weight.shape[::-1]),
            'coordinate_space': 'source_canvas_roi', 'native_box': box, 'original_canvas_size': native},
        checks={**checks, 'outside_native_box_exact_by_construction': True,
            'foreground_and_partial_alpha_exact': True, 'sampling_validity_changed': False,
            'only_original_source_zero_alpha_receiver_consumed': True,
            'receiver_attenuation_changed': False},
        limitations=['Only mutually observed qualified reflection fragments have permission.',
            'Low-frequency reflection radiance and unobserved floor remain unknown.',
            'Gaussian band replacement needs independent native visual review.',
            'Compressed native input arrays are decoded in full; filtering uses a bounded ROI.'])
    path = output/'receiver-preservation.json'; write(path, receipt)
    bg_layer['assets']['background'] = new_asset
    bg_layer['assets']['native_receiver_preservation'] = {'path': str(path), 'sha256': sha(path),
        'representation': 'qualified_native_receiver_preservation_provenance'}
    result.setdefault('provenance', {})['native_receiver_preservation'] = {
        'recipe': RECIPE, 'state': receipt['state'], 'receipt_sha256': sha(path),
        'producer_sha256': receipt['producer_sha256'], 'visual_acceptance': False}
    return {'ok': True, 'design_reconstruction_plan': result, 'receiver_preservation': receipt, 'cache_hit': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run = json.loads(args.run.read_text(encoding='utf8'))
    plan = run['options']['appearance_revision']['design_reconstruction_plan']
    result = prepare_native_receiver_preservation(run, plan, args.output)
    write(args.output/'prepared-plan.json', result)
    print(json.dumps({'ok': result['ok'], 'state': result['receiver_preservation']['state'],
        'checks': result['receiver_preservation'].get('checks'),
        'prepared_plan': str((args.output/'prepared-plan.json').resolve())}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
