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

import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt

from reference_colour_fit import to_lab, from_lab


def normalized_colour(colour, mask, sigma):
    support = gaussian_filter(mask.astype(np.float32), sigma)
    filtered = gaussian_filter(colour*mask[..., None], (sigma, sigma, 0))
    return filtered/np.maximum(support[..., None], 1e-5), support


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
    source_lab = to_lab(values)
    delta = strength*field[..., :2]
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
        rgb[bad] = np.where((lower == 0)[:, None], values[bad], rgb[bad])
    result = np.clip(rgb, 0, 1).astype(np.float32)
    result[np.all(field[..., :2] == 0, axis=-1)] = values[np.all(field[..., :2] == 0, axis=-1)]
    return result


def validate_layer(receipt_path, parameters, native_size, proxy_size=None):
    receipt_path = Path(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding='utf8'))
    if receipt.get('schema') != 'reference_spatial_colour.v1' or receipt.get('working_domain') != 'display_linear_sRGB':
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
    if receipt['algorithm_sha256'] != hashlib.sha256(code_path('reference_colour_blend.py').read_bytes()).hexdigest():
        raise ValueError('Reference colour blend implementation changed; prepare the layer again')
    path = Path(receipt['asset']['path'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != receipt['asset']['sha256']:
        raise ValueError('Reference colour layer asset changed')
    with np.load(path, allow_pickle=False) as resource:
        field = resource['values']
    if field.shape != (receipt['proxy_size'][1], receipt['proxy_size'][0], 3) or not np.isfinite(field).all():
        raise ValueError('Invalid source-coordinate reference colour field')
    if field[..., 2].min() < 0 or field[..., 2].max() > 1 or np.linalg.norm(field[..., :2], axis=-1).max() > .040001:
        raise ValueError('Unbounded spatial colour layer')
    return field, receipt


def main():
    import cv2
    from PIL import Image
    from display_transform import display_linear
    from raw_support import open_linear, open_rgb, linear_preview
    from float_appearance import srgb_to_linear
    from local_photo_analysis import prepare_local_appearance
    from local_face_protection import prepare_face_protection
    from reference_colour_fit import apply, REGIONS
    from evaluate_reference_colour import sheet, optional_review_boxes
    from render_reference_colour_composite import sample_maps

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
    H = np.asarray(parameters['binding']['mapping'])
    analyses = [prepare_local_appearance(root, Path(run[role]['path']), source_hash=run[role]['sha256'],
        role='source' if role == 'source' else 'candidate',
        registry_path=run.get('options', {}).get('model_registry')) for role in ('source', 'reference')]
    labels = []
    faces = []
    for item in analyses:
        with np.load(item['semantic_result']['labels_path'], allow_pickle=False) as resource:
            labels.append(resource['labels'])
        faces.append(prepare_face_protection(root, Path(item['analysis_directory'])/'source-srgb.png', item['semantic_result']))
    sl = cv2.resize(labels[0], size, interpolation=cv2.INTER_NEAREST)
    reference = open_rgb(Path(run['reference']['path']))
    ref_rgb = srgb_to_linear(np.asarray(reference, np.float32)/255)
    rl = cv2.resize(labels[1], reference.size, interpolation=cv2.INTER_NEAREST)
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
    eye_protection = np.zeros((size[1], size[0]), np.uint8)
    mouth_core = np.zeros_like(eye_protection)
    source_eye_masks = {}
    for polygon in faces[0].get('protected_regions', []):
        if polygon['region'] not in {'left_eye', 'right_eye', 'lips'}:
            continue
        pts = np.asarray(polygon['polygon'])*np.asarray(size)/np.asarray(faces[0]['size'])
        mask = np.zeros_like(eye_protection)
        cv2.fillConvexPoly(mask, np.round(pts).astype(np.int32), 1)
        if polygon['region'] == 'lips':
            mouth_core |= mask
        else:
            # Protect the eye opening, while allowing surrounding skin colour
            # to adapt. The previous expanded hard mask froze warm eye rings.
            eye_protection |= mask
            source_eye_masks[polygon['region']] = mask.astype(bool)
    # Teeth/tongue never participate in the explicitly selected lip colour.
    eye_protection[np.isin(sl, [2, 26, 27, 28])] = 1
    strict_exclusion = np.isin(sl, [2, 26, 27, 28])
    eye_distance = distance_transform_edt(eye_protection == 0)
    eye_transition = np.clip(eye_distance/3., 0, 1)
    eye_transition = eye_transition**2*(3-2*eye_transition)
    with np.load(plan['source_baseline']['assets']['foreground_alpha']['path'], allow_pickle=False) as resource:
        native_alpha = resource['values']
        if (native_alpha.ndim != 2 or not np.isfinite(native_alpha).all()
                or native_alpha.min() < 0 or native_alpha.max() > 1):
            raise ValueError('Invalid source coverage for spatial colour preparation')
        alpha = cv2.resize(native_alpha, size)
    with np.load(directory/'source-weights.npz', allow_pickle=False) as resource:
        weights = resource['values']
        chroma_weights = resource['chroma_values'] if 'chroma_values' in resource else None
    from native_source_baseline import source_for_reference_prepare
    source, _, _ = source_for_reference_prepare(run, root=root, binding=parameters['binding'])
    proxy = cv2.resize(source, size)
    base = apply(proxy, weights, parameters, canvas_size=size, chroma_weights=chroma_weights)
    base_lab = to_lab(base)
    ref_lab, facial_lab = to_lab(aligned), to_lab(facial_ref)
    field = np.zeros((*base.shape[:2], 3), np.float32)
    # Lip/skin is an internal boundary between two adapted materials. Feather
    # the outer facial boundary, rather than restoring source colour on both
    # sides of the lip line and leaving a warm/green seam.
    source_face_materials = np.isin(sl, [3, 24, 25]) & (alpha >= .995) & (eye_protection == 0)
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
    field,reports = colour_field(base_lab,to_lab(ref_rgb),sl,rl,alpha,maps,local_record,
        excluded=strict_exclusion | eye_protection.astype(bool))
    assert not field[strict_exclusion, :2].any()
    path = directory/'reference-colour-layer.npz'
    owners = sl.astype(np.int32).copy()
    owners[strict_exclusion | eye_protection.astype(bool)] += 1000
    np.savez_compressed(path, values=field, owners=owners)
    receipt = {'schema': 'reference_spatial_colour.v1', 'working_domain': 'display_linear_sRGB',
        'support_policy': 'reference_support.v1',
        'automatic_policy': POLICY, 'local_correspondence': local_record,
        'proxy_size': list(size), 'native_size': list(native),
        'binding': {**{key: parameters['binding'][key] for key in ('source', 'reference', 'alpha', 'source_protection', 'mapping')},
            'parameters_sha256': hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest(),
            'source_labels_sha256': hashlib.sha256(sl.tobytes()).hexdigest(),
            'reference_labels_sha256': hashlib.sha256(rl.tobytes()).hexdigest(),
            'source_face_landmarks': faces[0]['identity'], 'reference_face_landmarks': faces[1]['identity']},
        'algorithm_sha256': hashlib.sha256(code_path('reference_colour_blend.py').read_bytes()).hexdigest(),
        'asset': {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'representation': 'preweighted_OKLab_delta_ab_plus_coverage; no_lightness_delta'},
        'face_reference_mapping': facial_H.tolist(), 'face_alignment_validation': alignment,
        'lowpass_baseline': 'T(native_display_source_reduced_to_proxy); proxy_approximation_of_native_matching',
        'regions': reports, 'reference_spatial_colour_used': True,
        'reference_geometry_used_for_output': False, 'reference_high_frequency_texture_used': False,
        'eye_mouth_protection': 'source_L_geometry_and_native_texture_retained; T_feature_chroma_adaptation_allowed; B_eye_core_teeth_tongue_eyeglass_excluded',
        'alpha_modified': False, 'visual_reviewed': False}
    (directory/'reference-colour-layer.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf8')
    validation_path = directory/'validation.json'
    validation = json.loads(validation_path.read_text(encoding='utf8')) if validation_path.is_file() else {}
    box = optional_review_boxes(validation, native).get('face')
    if box is not None:
        x0, y0, x1, y1 = box
        w = sample_maps(weights, box, size, native)
        cw = sample_maps(chroma_weights, box, size, native) if chroma_weights is not None else None
        current = apply(source[y0:y1, x0:x1], w, parameters, origin=(x0, y0), canvas_size=native, chroma_weights=cw)
        proposed = apply_colour_layer(current, sample_maps(field, box, size, native, owners=owners))
        mapping = np.diag([native[0]/size[0], native[1]/size[1], 1.]) @ facial_H
        mapping = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1.]]) @ mapping
        ref_crop = cv2.warpPerspective(ref_rgb, mapping, (x1-x0, y1-y0))
        sheet([linear_preview(current), linear_preview(ref_crop), linear_preview(proposed)],
            ['Current source person', 'Reference colour', 'Selected low-frequency blend'], 2200).save(directory/'fusion-colour-review.jpg', quality=97)
        if len(source_points) > 291:
            feature_points = source_points[[33, 263, 61, 291]]*np.asarray(native)/np.asarray(size)
            px0, py0 = np.floor(feature_points.min(0)-[80, 65]).astype(int)
            px1, py1 = np.ceil(feature_points.max(0)+[80, 65]).astype(int)
            crop_box = (px0-x0, py0-y0, px1-x0, py1-y0)
            sheet([linear_preview(current).crop(crop_box), linear_preview(proposed).crop(crop_box)],
                ['Before: frozen eye/mouth colour', 'After: colour adaptation, native detail'], 1800).save(directory/'eyes-mouth-comparison.jpg', quality=98)
    else:
        print('Optional native face review skipped: source face crop is unavailable', flush=True)
    print(json.dumps({'regions': reports, 'output': str(directory/'reference-colour-layer.json')}), flush=True)


if __name__ == '__main__':
    main()
