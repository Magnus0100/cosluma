"""Selected spatial reference colour blending after source-only matching.

The reference contributes low-frequency chroma. Source geometry, lightness and
texture stay in the native source canvas; no reference person RGB is pasted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from runtime_resources import code_path
from asset_identity import sha256_file

import numpy as np
from native_array_storage import save_proxy_arrays

from reference_colour_fit import to_lab, from_lab

SCHEMA = 'reference_spatial_colour.v2'
FEATURE_CHROMA_POLICY = 'beauty_feature_chroma_and_lip_material_domains.v3'




def apply_colour_layer(values, field, *, strength=1.):
    """Consume a preweighted source-coordinate chroma field exactly once."""
    if not 0 <= strength <= 1 or field.shape != (*values.shape[:2], 3):
        raise ValueError('Invalid spatial colour blend dimensions or strength')
    if not np.isfinite(values).all() or not np.isfinite(field).all():
        raise ValueError('Nonfinite source or spatial colour field')
    if field[..., 2].min() < 0 or field[..., 2].max() > 1:
        raise ValueError('Invalid spatial colour coverage')
    if strength == 0:
        return values.copy()
    # A zero chroma field restores the original RGB exactly below. Exclude
    # those pixels before the nonlinear transform and gamut search, including
    # source-visible face exclusions and background outside material support.
    active = np.any(field[..., :2] != 0, axis=-1)
    result = np.asarray(values, np.float32).copy()
    if not np.any(active):
        return result
    active_values = values[active]
    source_lab = to_lab(active_values)
    delta = strength*field[active, :2]
    lab = source_lab.copy();lab[...,1:] += delta
    rgb = from_lab(lab)
    # An absolute RGB tolerance becomes a large relative error in deep shadows.
    # Clip only after a strict gamut search, rather than admitting negative
    # dark channels and then changing their OKLab lightness by clipping.
    bad = (rgb.min(-1) < 0) | (rgb.max(-1) > 1)
    if np.any(bad):
        base = source_lab[bad];shift = delta[bad]
        lower = np.zeros(len(base),np.float32);upper = np.ones(len(base),np.float32)
        # Reduce only the requested correction. Scaling all ab would also
        # change the original costume/eye colour, and clipping flattens white.
        for _ in range(18):
            dose = (lower+upper)*.5
            proposal = base.copy();proposal[:,1:] += dose[:,None]*shift
            candidate = from_lab(proposal)
            safe = (candidate.min(-1) >= 0)&(candidate.max(-1) <= 1)
            lower = np.where(safe,dose,lower);upper = np.where(safe,upper,dose)
        proposal = base.copy();proposal[:,1:] += lower[:,None]*shift
        rgb[bad] = from_lab(proposal)
        rgb[bad] = np.where((lower == 0)[:, None], active_values[bad], rgb[bad])
    result[active] = np.clip(rgb, 0, 1).astype(np.float32)
    return result


def fuse_reference_colour(values, field, *, source_values, strength=1.):
    """Apply B, then bound cumulative T+B eye/lip colour against Beauty.

    The last three channels contain ambient ab per Beauty L and a soft feature
    mask. Only chroma changes; final lightness, Beauty detail and geometry stay.
    No original photograph RGB is restored and no reference texture is copied.
    """
    if (field.shape != (*values.shape[:2], 6) or source_values.shape != values.shape
            or not np.isfinite(field[..., 3:]).all()
            or field[..., 5].min() < 0 or field[..., 5].max() > 1
            or np.linalg.norm(field[..., 3:5], axis=-1).max() > .080001):
        raise ValueError('Current reference colour requires bound ambient feature channels')
    result = apply_colour_layer(values, field[..., :3], strength=strength)
    active = field[..., 5] > 0
    if strength == 0 or not active.any():
        return result
    baseline = source_values[active]
    if not np.isfinite(baseline).all():
        raise ValueError('Feature colour requires finite current Beauty pixels')
    original = to_lab(baseline)
    current = to_lab(result[active])
    ambient = original[:, :1]*field[active, 3:5]
    correction = (original[:, 1:]+ambient-current[:, 1:])
    correction *= (strength*field[active, 5])[:, None]
    packed = np.empty((len(correction), 1, 3), np.float32)
    packed[:, 0, :2] = correction; packed[:, 0, 2] = 1
    # Reuse the same strict gamut search. It shrinks the new correction only,
    # preserving incoming lightness even at black/white gamut boundaries.
    result[active] = apply_colour_layer(result[active, None, :], packed)[:, 0]
    return result


def validate_layer(receipt_path, parameters, native_size, proxy_size=None):
    receipt_path = Path(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding='utf8'))
    if (receipt.get('schema') != SCHEMA or receipt.get('working_domain') != 'display_linear_sRGB'
            or receipt.get('feature_chroma_policy') != FEATURE_CHROMA_POLICY):
        raise ValueError('Reference colour layer requires the explicit appearance domain')
    for role in ('source', 'reference', 'alpha', 'source_protection'):
        if receipt['binding'][role] != parameters['binding'][role]:
            raise ValueError(f'Reference colour layer {role} binding mismatch')
    if receipt['binding']['mapping'] != parameters['binding']['mapping'] or receipt['native_size'] != list(native_size):
        raise ValueError('Reference colour layer coordinate binding mismatch')
    if proxy_size is not None and receipt['proxy_size'] != list(proxy_size):
        raise ValueError('Reference colour layer proxy sampling dimensions mismatch')
    if receipt['binding']['parameters_sha256'] != hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest():
        raise ValueError('Reference colour layer was prepared for different matching parameters')
    if receipt['algorithm_sha256'] != sha256_file(code_path('reference_colour_blend.py')):
        raise ValueError('Reference colour blend implementation changed; prepare the layer again')
    path = Path(receipt['asset']['path'])
    if sha256_file(path) != receipt['asset']['sha256']:
        raise ValueError('Reference colour layer asset changed')
    with np.load(path, allow_pickle=False) as resource:
        field = resource['values']
    if field.shape != (receipt['proxy_size'][1], receipt['proxy_size'][0], 6) or not np.isfinite(field).all():
        raise ValueError('Invalid source-coordinate reference colour field')
    if field[..., 2].min() < 0 or field[..., 2].max() > 1 or np.linalg.norm(field[..., :2], axis=-1).max() > .040001:
        raise ValueError('Unbounded spatial colour layer')
    if (field[..., 5].min() < 0 or field[..., 5].max() > 1
            or np.linalg.norm(field[..., 3:5], axis=-1).max() > .080001):
        raise ValueError('Unbounded ambient feature colour layer')
    return field, receipt


def main():
    import cv2
    from raw_support import open_rgb, linear_preview
    from float_appearance import srgb_to_linear
    from local_photo_analysis import prepare_local_appearance
    from local_face_protection import (prepare_face_protection_pair, prepare_feature_chroma_guard,
                                       prepare_strict_lip_support, prepare_feature_materials)
    from reference_colour_fit import apply, REGIONS
    from evaluate_reference_colour import sheet
    from render_reference_colour_composite import sample_reference_field
    from portrait_beauty_layer import hair_correction_guard_for_run, sample_visible_face_support

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--match-dir', type=Path, default=Path('outputs/inspection/reference-match-v2'))
    args = parser.parse_args()
    root = args.root.resolve()
    directory = args.match_dir.resolve()
    run = json.loads(args.run.read_text(encoding='utf8'))
    plan = run['options']['appearance_revision']['design_reconstruction_plan']
    parameters = json.loads((directory/'parameters.json').read_text(encoding='utf8'))
    size = tuple(plan['canvas']['source_proxy_size']); native = tuple(plan['canvas']['source_oriented_size'])
    hair_guard = hair_correction_guard_for_run(run, root=root)
    face_support = sample_visible_face_support(hair_guard,(0,0,size[0],size[1]),size) if hair_guard else None
    H = np.asarray(parameters['binding']['mapping'])
    analyses = [prepare_local_appearance(root, Path(run[role]['path']), source_hash=run[role]['sha256'],
        role='source' if role == 'source' else 'candidate',
        registry_path=run.get('options', {}).get('model_registry')) for role in ('source', 'reference')]
    labels = []
    for item in analyses:
        with np.load(item['semantic_result']['labels_path'], allow_pickle=False) as resource:
            labels.append(resource['labels'])
    faces = list(prepare_face_protection_pair(root,
        Path(analyses[0]['analysis_directory'])/'source-srgb.png', analyses[0]['semantic_result'],
        Path(analyses[1]['analysis_directory'])/'source-srgb.png', analyses[1]['semantic_result']))
    sl = cv2.resize(labels[0], size, interpolation=cv2.INTER_NEAREST)
    reference = open_rgb(Path(run['reference']['path']))
    ref_rgb = srgb_to_linear(np.asarray(reference, np.float32)/255)
    rl = cv2.resize(labels[1], reference.size, interpolation=cv2.INTER_NEAREST)
    semantic_source_sha = hashlib.sha256(sl.tobytes()).hexdigest()
    semantic_reference_sha = hashlib.sha256(rl.tobytes()).hexdigest()
    feature_domains = parameters['feature_material_domains']
    if feature_domains['schema'] != 'feature_material_domains.v1':
        raise ValueError('Colour preparation requires current feature material domains')
    rl, _, reference_material_report = prepare_feature_materials(
        rl, np.zeros(rl.shape, np.float32), feature_domains['reference_regions'])
    # The cached detector coordinates refer to each analysis image, not native.
    source_points = np.asarray(faces[0]['landmarks'], np.float32).reshape(-1, 2)*np.asarray(size)/np.asarray(faces[0]['size'])
    ref_points = np.asarray(faces[1]['landmarks'], np.float32).reshape(-1, 2)*np.asarray(reference.size)/np.asarray(faces[1]['size'])
    facial_H = H
    alignment = {'method': 'background_H_only', 'mouth_max_error_proxy_px': None}
    if len(source_points) >= 455 and len(ref_points) >= 455:
        train = np.array([33, 263, 1, 234, 454])
        A, inliers = cv2.estimateAffinePartial2D(ref_points[train], source_points[train], method=cv2.LMEDS)
        if A is not None:
            proposed = np.vstack([A, [0, 0, 1.]])
            predicted = cv2.perspectiveTransform(ref_points[[61, 291, 152]][None], proposed)[0]
            errors = np.linalg.norm(predicted-source_points[[61, 291, 152]], axis=1)
            if errors.max() <= 3:
                facial_H = proposed
                alignment = {'method': 'reference_only_similarity_from_source_reference_landmarks',
                    'fit_indices': train.tolist(), 'check_indices': [61, 291, 152],
                    'check_errors_proxy_px': errors.tolist(), 'mouth_max_error_proxy_px': float(errors[:2].max())}
    print('Reference-only face colour alignment', alignment, flush=True)
    aligned = cv2.warpPerspective(ref_rgb, H, size)
    ref_labels = cv2.warpPerspective(rl, H, size, flags=cv2.INTER_NEAREST)
    facial_ref = cv2.warpPerspective(ref_rgb, facial_H, size)
    facial_labels = cv2.warpPerspective(rl, facial_H, size, flags=cv2.INTER_NEAREST)
    feature_regions = feature_domains['source_regions']
    with np.load(faces[0]['field_path'], allow_pickle=False) as resource:
        protection = cv2.resize(resource['protection'], size, interpolation=cv2.INTER_LINEAR)
    # Preserve the wider detector exclusion for clean ambient donors only.
    # T/B/S material ownership and lip application use the canonical domains.
    sl, _, source_material_report = prepare_feature_materials(sl, protection, feature_regions)
    eye_protection = np.zeros((size[1], size[0]), np.uint8)
    for polygon in feature_regions:
        if polygon['region'] not in {'left_eye', 'right_eye', 'lips'}:
            continue
        pts = np.asarray(polygon['polygon'])
        mask = np.zeros_like(eye_protection)
        cv2.fillConvexPoly(mask, np.round(pts).astype(np.int32), 1)
        if polygon['region'] != 'lips':
            # Core excludes unsafe B observations; the existing expanded soft
            # eye/lip mask below also limits cumulative T+B colour at replay.
            eye_protection |= mask
    # Teeth/tongue never participate in the explicitly selected lip colour.
    eye_protection[np.isin(sl, [2, 26, 27, 28])] = 1
    strict_exclusion = np.isin(sl, [2, 26, 27, 28])
    with np.load(plan['source_baseline']['assets']['foreground_alpha']['path'], allow_pickle=False) as resource:
        native_alpha = resource['values']
        if (native_alpha.ndim != 2 or not np.isfinite(native_alpha).all()
                or native_alpha.min() < 0 or native_alpha.max() > 1):
            raise ValueError('Invalid source coverage for spatial colour preparation')
        alpha = cv2.resize(native_alpha, size)
    lip_support, skin_reject_collar, lip_support_report = prepare_strict_lip_support(
        sl.shape, feature_regions, sl, alpha)
    lip_owned = np.isin(sl, [24, 25])
    # A semantic lip class can extend past the visible lip line. Reject those
    # observations before solving, and keep the tiny adjacent skin collar out
    # of local pigment transfer. Neither mask paints or reconstructs skin.
    strict_exclusion |= (lip_owned & (lip_support == 0)) | skin_reject_collar
    with np.load(directory/'source-weights.npz', allow_pickle=False) as resource:
        weights = resource['values']
        chroma_weights = resource['chroma_values'] if 'chroma_values' in resource else None
    from native_source_baseline import source_for_reference_prepare
    source, _, _ = source_for_reference_prepare(
        run, root=root, binding=parameters['binding'], apply_beauty=True, writable=False)
    proxy = cv2.resize(source, size)
    base = apply(proxy, weights, parameters, canvas_size=size, chroma_weights=chroma_weights)
    base_lab = to_lab(base)
    ref_lab, facial_lab = to_lab(aligned), to_lab(facial_ref)
    from reference_local_correspondence import prepare_local_correspondence
    from reference_automatic_layers import (pack_correspondence,load_maps,colour_field,POLICY)
    print('Preparing independent local person correspondence',flush=True)
    local = prepare_local_correspondence(root,run,size,sl,rl,
        source_faces=faces[0],reference_faces=faces[1])
    local_record = pack_correspondence(local,run,size,root=root,source_labels=sl,
        reference_labels=rl,source_faces=faces[0],reference_faces=faces[1])
    maps = load_maps(local_record,sl.shape,rl.shape,parameters,root=root,source_labels=sl,
        reference_labels=rl,source_faces=faces[0],reference_faces=faces[1])
    print('Solving automatic material colour fields',flush=True)
    task_intent = run.get('task_intent') or (run.get('options') or {}).get('task_intent') or {}
    constraints = task_intent.get('constraints') or {}
    allow_relaxed_continuation = bool(
        task_intent.get('mode') in {'create_reference', 'existing_reference'}
        and constraints.get('reference_role') != 'source_conditioned_design'
        and constraints.get('background_mode') != 'enhance_source')
    field,reports = colour_field(base_lab,to_lab(ref_rgb),sl,rl,alpha,maps,local_record,
        excluded=strict_exclusion | eye_protection.astype(bool),visible_face_support=face_support,
        preserve_hair=bool(parameters.get('source_hair_policy')),
        allow_relaxed_continuation=allow_relaxed_continuation)
    field[lip_owned, :2] *= lip_support[lip_owned, None]
    field[~np.any(field[..., :2] != 0, axis=-1), 2] = 0
    reports['lips']['lip_colour_support'] = lip_support_report
    reports['lips']['applied_pixels_after_lip_support'] = int(
        np.count_nonzero(np.any(field[lip_owned, :2] != 0, axis=-1)))
    reports['lips']['maximum_applied_chroma_shift_after_lip_support'] = float(
        np.linalg.norm(field[lip_owned, :2], axis=-1).max(initial=0))
    assert not field[strict_exclusion, :2].any()
    unguarded = apply_colour_layer(base, field)
    if faces[0].get('protected_regions'):
        feature_guard, feature_report = prepare_feature_chroma_guard(
            to_lab(proxy), to_lab(unguarded), sl, alpha, protection,
            feature_regions=feature_regions)
    else:
        feature_guard = np.zeros_like(field)
        feature_report = {'status': 'unsupported', 'reason': 'no_source_feature_landmarks',
                          'source_texture_restored': False}
    field = np.concatenate((field, feature_guard), axis=-1)
    path = directory/'reference-colour-layer.npz'
    owners = sl.astype(np.int32).copy()
    owners[strict_exclusion | eye_protection.astype(bool)] += 1000
    save_proxy_arrays(path, {'values': field, 'owners': owners})
    receipt = {'schema': SCHEMA, 'working_domain': 'display_linear_sRGB',
        'feature_chroma_policy': FEATURE_CHROMA_POLICY, 'feature_chroma_guard': feature_report,
        'feature_material_domains': {'source': source_material_report,
                                     'reference': reference_material_report},
        'lip_colour_support': lip_support_report,
        'region_metric_scope': 'B_solver_proposal_before_strict_lip_support_and_cumulative_feature_guard; explicit_lip_after_support_metrics',
        'support_policy': 'reference_support.v1',
        'automatic_policy': POLICY, 'local_correspondence': local_record,
        'proxy_size': list(size), 'native_size': list(native),
        'hair_correction_guard': hair_guard['binding'] if hair_guard else None,
        'binding': {**{key: parameters['binding'][key] for key in ('source', 'reference', 'alpha', 'source_protection', 'mapping')},
            'parameters_sha256': hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest(),
            'source_labels_sha256': semantic_source_sha,
            'reference_labels_sha256': semantic_reference_sha,
            'source_face_landmarks': faces[0]['identity'], 'reference_face_landmarks': faces[1]['identity']},
        'algorithm_sha256': sha256_file(code_path('reference_colour_blend.py')),
        'asset': {'path': str(path), 'sha256': sha256_file(path),
            'representation': 'B_delta_ab_QA_coverage_ambient_ab_per_Beauty_L_soft_feature_mask; no_lightness_delta'},
        'face_reference_mapping': facial_H.tolist(), 'face_alignment_validation': alignment,
        'lowpass_baseline': 'T(native_display_source_reduced_to_proxy); proxy_approximation_of_native_matching',
        'regions': reports, 'reference_spatial_colour_used': True,
        'reference_geometry_used_for_output': False, 'reference_high_frequency_texture_used': False,
        'eye_mouth_protection': 'Beauty_feature_ab_plus_supported_smooth_ambient; lip_colour_inside_source_lip_hull_only; inward_feather; cumulative_T_B_guard; final_L_retained; no_original_RGB_restoration',
        'alpha_modified': False, 'visual_reviewed': False}
    (directory/'reference-colour-layer.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf8')
    # Native intermediates are reconstructed by the final saved-stage review.
    # Preparation only needs a small proxy of its field, never a second native
    # face render and warp whose pixels are not consumed by downstream stages.
    sampled = sample_reference_field(field, (0, 0, size[0], size[1]), size, size,
        owners=owners, guard=hair_guard)
    proposed = fuse_reference_colour(base, sampled, source_values=proxy)
    sheet([linear_preview(base), linear_preview(facial_ref), linear_preview(proposed)],
        ['Matched source proxy', 'Aligned reference proxy', 'Colour fusion proxy'], 1800).save(
            directory/'fusion-colour-proxy-review.jpg', quality=95)
    print(json.dumps({'regions': reports, 'output': str(directory/'reference-colour-layer.json')}), flush=True)


if __name__ == '__main__':
    main()
