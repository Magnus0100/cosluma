"""Profile-bound DINO mutual matches and local reference-coordinate evidence."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import Delaunay, cKDTree
from scipy.spatial import QhullError

from reference_local_correspondence import (SCHEMA, PREPROCESS, MATCHING, TOKEN_LAYOUT,
    DINO3_L_PROFILE, algorithm_configuration, file_sha, array_sha)

REGIONS = {'face': [3], 'left_arm': [7, 11], 'right_arm': [16, 20],
    'left_hand': [6], 'right_hand': [15], 'legs': [5, 8, 12, 14, 17, 21, 22],
    'hair': [4], 'clothing': [1, 10, 13, 19, 23], 'shoes': [9, 18], 'bag': [200],
    'lips': [24, 25], 'left_eye': [3], 'right_eye': [3]}
DINO_LABELS = {label for name, ids in REGIONS.items() if name not in ('face', 'lips', 'left_eye', 'right_eye') for label in ids}


def token_materials(labels, model_size, *, patch=16, minimum_fraction=.65):
    resized = np.asarray(Image.fromarray(labels.astype(np.int32), mode='I').resize(model_size, Image.Resampling.NEAREST))
    width, height = model_size
    blocks = resized.reshape(height//patch, patch, width//patch, patch).transpose(0, 2, 1, 3).reshape(-1, patch*patch)
    dominant = np.zeros(len(blocks), np.int32)
    fractions = np.zeros(len(blocks), np.float32)
    for i, values in enumerate(blocks):
        ids, counts = np.unique(values, return_counts=True)
        index = counts.argmax()
        dominant[i] = ids[index]
        fractions[i] = counts[index]/len(values)
    dominant[fractions < minimum_fraction] = 0
    return dominant, fractions


def patch_tokens(hidden, model_size, *, patch_size, token_layout):
    """Remove explicit CLS/register tokens; never treat them as image patches."""
    if (hidden.ndim != 3 or hidden.shape[0] != 1 or len(model_size) != 2
            or any(n <= 0 or n % patch_size for n in model_size)):
        raise ValueError('Invalid patch-token grid or model batch')
    count = (model_size[0]//patch_size)*(model_size[1]//patch_size)
    prefix = token_layout['class_tokens']+token_layout['register_tokens']
    if hidden.shape[1:] != (prefix+count, token_layout['embedding_dimension']):
        raise ValueError('Model tokens differ from the bound CLS/register/patch layout')
    return hidden[0, prefix:]


def extract(model, image, labels, device, *, maximum_edge=None, preprocess=None,
            settings=None, token_layout=None):
    import torch
    cfg, matching, layout = preprocess or PREPROCESS, settings or MATCHING, token_layout or TOKEN_LAYOUT
    patch = cfg['patch_size']
    factor = (maximum_edge or cfg['maximum_edge'])/max(image.size)
    size = tuple(max(patch, int(round(n*factor/patch))*patch) for n in image.size)
    resized = image.resize(size, Image.Resampling.BICUBIC)
    rgb = np.asarray(resized, np.float32)/255.
    rgb = (rgb-np.asarray(cfg['mean'], np.float32))/np.asarray(cfg['std'], np.float32)
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy())[None].to(device)
    with torch.inference_mode():
        features = patch_tokens(model(pixel_values=tensor).last_hidden_state, size,
                                patch_size=patch, token_layout=layout)
        features = torch.nn.functional.normalize(features.float(), dim=-1).cpu().numpy()
    if not np.isfinite(features).all():
        raise ValueError('Model patch features contain nonfinite values')
    width, height = size
    yy, xx = np.mgrid[:height//patch, :width//patch]
    points = np.stack([(xx+.5)*patch-.5, (yy+.5)*patch-.5], axis=-1).reshape(-1, 2)
    classes, fraction = token_materials(labels, size, patch=patch,
        minimum_fraction=matching['minimum_token_material_fraction'])
    if len(features) != len(points):
        raise ValueError('DINO token grid does not match explicit patch preprocessing')
    return {'features': features, 'points': points, 'labels': classes, 'fraction': fraction, 'size': size,
            'patch_size': patch}


def mutual_matches(source, reference, *, settings=None):
    """Exact semantic label, mutual best feature, similarity and ambiguity gates."""
    cfg = {**MATCHING, **(settings or {})}
    rows, reports = [], {}
    for label in sorted(DINO_LABELS):
        si, ri = np.flatnonzero(source['labels'] == label), np.flatnonzero(reference['labels'] == label)
        report = {'source_tokens': int(len(si)), 'reference_tokens': int(len(ri)), 'mutual_unambiguous': 0}
        reports[str(label)] = report
        if min(len(si), len(ri)) < 2:
            report['rejected'] = 'insufficient_same_semantic_tokens'
            continue
        similarity = source['features'][si] @ reference['features'][ri].T
        top = np.argsort(similarity, axis=1)[:, -2:]
        reverse = similarity.argmax(axis=0)
        reverse_top = np.sort(similarity, axis=0)[-2:]
        best = top[:, 1]
        score = similarity[np.arange(len(si)), best]
        margin = score-similarity[np.arange(len(si)), top[:, 0]]
        reverse_margin = reverse_top[1, best]-reverse_top[0, best]
        keep = ((reverse[best] == np.arange(len(si))) & (score >= cfg['minimum_cosine'])
                & (margin >= cfg['minimum_top2_margin']) & (reverse_margin >= cfg['minimum_top2_margin']))
        for local in np.flatnonzero(keep):
            confidence = float(np.clip((score[local]-cfg['minimum_cosine'])/(1-cfg['minimum_cosine']), 0, 1)
                * np.clip(min(margin[local], reverse_margin[local])/.10, 0, 1))
            rows.append({'source_index': int(si[local]), 'reference_index': int(ri[best[local]]),
                'label': label, 'cosine': float(score[local]), 'margin': float(min(margin[local], reverse_margin[local])),
                'confidence': confidence})
        report['mutual_unambiguous'] = int(keep.sum())
        report['rejected_similarity_or_ambiguity_or_mutual'] = int(len(si)-keep.sum())
    return rows, reports


def _affine(source, target):
    design = np.c_[source, np.ones(len(source))]
    if np.linalg.matrix_rank(design) < 3:
        return None
    return np.linalg.lstsq(design, target, rcond=None)[0]


def _affine_valid(matrix, cfg, face=False):
    jacobian = matrix[:2].T
    if np.linalg.det(jacobian) <= 0:
        return False
    singular = np.linalg.svd(jacobian, compute_uv=False)
    lower = cfg['minimum_face_triangle_scale'] if face else cfg['minimum_affine_scale']
    upper = cfg['maximum_face_triangle_scale'] if face else cfg['maximum_affine_scale']
    condition = cfg['maximum_face_triangle_condition'] if face else cfg['maximum_affine_condition']
    return bool(singular.min() >= lower and singular.max() <= upper and singular.max()/singular.min() <= condition)


def geometric_matches(rows, source, reference, *, settings=None):
    """Leave each query out: neighbouring same-label pairs predict its target."""
    cfg = {**MATCHING, **(settings or {})}
    accepted, rejected = [], {}
    patch = source.get('patch_size', PREPROCESS['patch_size'])
    if reference.get('patch_size', PREPROCESS['patch_size']) != patch:
        raise ValueError('Source/reference patch-coordinate scales differ')
    for label in sorted({row['label'] for row in rows}):
        subset = [row for row in rows if row['label'] == label]
        source_points = source['points'][[row['source_index'] for row in subset]]
        ref_points = reference['points'][[row['reference_index'] for row in subset]]
        tree = cKDTree(source_points)
        count = 0
        for i, row in enumerate(subset):
            neighbours = [j for j in tree.query_ball_point(source_points[i], cfg['neighbour_radius_patches']*patch) if j != i]
            if len(neighbours) < cfg['minimum_neighbours']:
                count += 1
                continue
            # A robust fit rejects locally inconsistent matches without a
            # global scene transform or forcing one pose onto another.
            x, y = source_points[neighbours], ref_points[neighbours]
            matrix = _affine(x, y)
            if matrix is None:
                count += 1
                continue
            errors = np.linalg.norm(np.c_[x, np.ones(len(x))] @ matrix-y, axis=1)
            inliers = errors <= cfg['maximum_leave_one_out_error_patches']*patch
            if inliers.sum() < cfg['minimum_neighbours']:
                count += 1
                continue
            matrix = _affine(x[inliers], y[inliers])
            if matrix is None or not _affine_valid(matrix, cfg):
                count += 1
                continue
            error = float(np.linalg.norm(np.r_[source_points[i], 1.] @ matrix-ref_points[i])/patch)
            if error > cfg['maximum_leave_one_out_error_patches']:
                count += 1
                continue
            confidence = row['confidence']*max(0., 1-error/cfg['maximum_leave_one_out_error_patches'])
            if confidence < cfg['minimum_confidence']:
                count += 1
                continue
            accepted.append({**row, 'confidence': confidence, 'leave_one_out_error_patches': error})
        rejected[str(label)] = count
    return accepted, rejected


def _rasterize_triangles(source_points, target_points, confidences, source_labels, reference_labels,
                         eligible_labels, maps, *, max_edge, settings=None, face=False):
    """No extrapolation: only positive, bounded local triangles of accepted pairs."""
    cfg = {**MATCHING, **(settings or {})}
    if len(source_points) < 3:
        return {'accepted_triangles': 0, 'rejected_triangles': 0}
    try:
        triangles = Delaunay(source_points).simplices
    except QhullError:
        return {'accepted_triangles': 0, 'rejected_triangles': 1}
    height, width = source_labels.shape
    accepted = rejected = 0
    for indices in triangles:
        src, ref = source_points[indices], target_points[indices]
        matrix = _affine(src, ref)
        source_edges = np.linalg.norm(src-np.roll(src, 1, axis=0), axis=1)
        # Scale checks use normalized full canvases, since reference and source
        # may have different pixel counts. No interpolation crosses a fold.
        normalized = _affine(src/np.array([width, height]),
                             ref/np.array([reference_labels.shape[1], reference_labels.shape[0]]))
        area = abs(float(np.linalg.det(np.stack([src[1]-src[0], src[2]-src[0]]))))/2
        minimum_area = .5 if face else cfg['minimum_triangle_area_patches_squared']*(
            max_edge/cfg['maximum_triangle_edge_patches'])**2
        if (matrix is None or normalized is None or not _affine_valid(normalized, cfg, face=face)
                or source_edges.max() > max_edge or area < minimum_area):
            rejected += 1
            continue
        left, top = np.maximum(np.floor(src.min(0)).astype(int), 0)
        right, bottom = np.minimum(np.ceil(src.max(0)).astype(int)+1, [width, height])
        yy, xx = np.mgrid[top:bottom, left:right]
        xy = np.stack([xx, yy], axis=-1)
        inverse = np.linalg.inv(np.c_[src, np.ones(3)])
        barycentric = np.c_[xy.reshape(-1, 2), np.ones(xy.size//2)] @ inverse
        inside = np.all(barycentric >= -1e-5, axis=1)
        target = np.c_[xy.reshape(-1, 2), np.ones(xy.size//2)] @ matrix
        rx, ry = np.rint(target).astype(int).T
        bounded = ((target[:, 0] >= 0) & (target[:, 0] <= reference_labels.shape[1]-1)
                   & (target[:, 1] >= 0) & (target[:, 1] <= reference_labels.shape[0]-1)
                   & (rx >= 0) & (rx < reference_labels.shape[1]) & (ry >= 0) & (ry < reference_labels.shape[0]))
        source_class = source_labels[top:bottom, left:right].reshape(-1)
        semantic = np.zeros(len(target), bool)
        semantic[bounded] = np.isin(source_class[bounded], eligible_labels) & (
            reference_labels[ry[bounded], rx[bounded]] == source_class[bounded])
        confidence = barycentric @ np.asarray(confidences)[indices]
        valid = inside & bounded & semantic & (confidence >= cfg['minimum_confidence'])
        view_conf = maps['confidence'][top:bottom, left:right].reshape(-1).copy()
        valid &= confidence > view_conf
        if not valid.any():
            rejected += 1
            continue
        for key, values in (('ref_x', target[:, 0]), ('ref_y', target[:, 1]), ('confidence', confidence), ('valid', valid)):
            view = maps[key][top:bottom, left:right].copy().reshape(-1)
            view[valid] = values[valid]
            maps[key][top:bottom, left:right] = view.reshape(bottom-top, right-left)
        accepted += 1
    return {'accepted_triangles': accepted, 'rejected_triangles': rejected}


def rasterize_dino(rows, source, reference, source_labels, reference_labels, maps, *, settings=None):
    cfg = {**MATCHING, **(settings or {})}
    patch = source.get('patch_size', PREPROCESS['patch_size'])
    reports = {}
    size = np.array([source_labels.shape[1], source_labels.shape[0]])
    ref_size = np.array([reference_labels.shape[1], reference_labels.shape[0]])
    for label in sorted({row['label'] for row in rows}):
        subset = [row for row in rows if row['label'] == label]
        source_scale = size/np.array(source['size'])
        ref_scale = ref_size/np.array(reference['size'])
        source_offset = ref_offset = np.zeros(2)
        if source.get('crop_box') is not None:
            box = np.asarray(source['crop_box'])
            source_scale = (box[2:]-box[:2])/np.array(source['size'])*size/np.array(source['full_image_size'])
            source_offset = box[:2]*size/np.array(source['full_image_size'])
        if reference.get('crop_box') is not None:
            box = np.asarray(reference['crop_box'])
            ref_scale = (box[2:]-box[:2])/np.array(reference['size'])*ref_size/np.array(reference['full_image_size'])
            ref_offset = box[:2]*ref_size/np.array(reference['full_image_size'])
        src = (source['points'][[row['source_index'] for row in subset]]+.5)*source_scale+source_offset-.5
        ref = (reference['points'][[row['reference_index'] for row in subset]]+.5)*ref_scale+ref_offset-.5
        reports[str(label)] = _rasterize_triangles(src, ref, [row['confidence'] for row in subset],
            source_labels, reference_labels, [label], maps,
            max_edge=cfg['maximum_triangle_edge_patches']*patch*max(source_scale), settings=cfg)
    return reports


def face_correspondence(source_face, reference_face, source_labels, reference_labels, maps, *, settings=None):
    """Canonical landmarks form locally checked triangles; eye/lip require these."""
    cfg = {**MATCHING, **(settings or {})}
    result = {'method': 'mediapipe_verified_piecewise_affine', 'accepted_triangles': 0,
              'rejected_triangles': 0, 'valid_landmarks': 0}
    if not source_face or not reference_face:
        return {**result, 'rejected': 'face_landmark_observations_not_supplied'}
    source = np.asarray(source_face.get('landmarks', []), float)
    reference = np.asarray(reference_face.get('landmarks', []), float)
    if (source.ndim != 2 or reference.ndim != 2 or source.shape[1:] != (2,)
            or reference.shape[1:] != (2,) or min(len(source), len(reference)) < 455
            or not np.isfinite(source).all() or not np.isfinite(reference).all()):
        return {**result, 'rejected': 'missing_or_nonfinite_face_landmarks'}
    src_size = np.array([source_labels.shape[1], source_labels.shape[0]])
    ref_size = np.array([reference_labels.shape[1], reference_labels.shape[0]])
    source = source[:468]*src_size/np.asarray(source_face['size'])
    reference = reference[:468]*ref_size/np.asarray(reference_face['size'])
    # Landmarks inside the predicted face/eye/lip semantic support only. Teeth
    # and tongue never supply a lip target. Validate each landmark using only
    # its neighbours; no self-fitted error or global background alignment.
    eligible = [3, 24, 25]
    src_xy, ref_xy = np.rint(source).astype(int), np.rint(reference).astype(int)
    bounded = ((src_xy >= 0).all(1) & (src_xy < src_size).all(1)
               & (ref_xy >= 0).all(1) & (ref_xy < ref_size).all(1))
    semantic = np.zeros(len(source), bool)
    semantic[bounded] = np.isin(source_labels[src_xy[bounded, 1], src_xy[bounded, 0]], eligible) & np.isin(
        reference_labels[ref_xy[bounded, 1], ref_xy[bounded, 0]], eligible)
    ids = np.flatnonzero(semantic)
    if len(ids) < 12:
        return {**result, 'rejected': 'insufficient_face_material_landmarks'}
    src, ref = source[ids], reference[ids]
    span = float(max(np.ptp(src, axis=0)))
    ref_span = float(max(np.ptp(ref, axis=0)))
    if min(span, ref_span) <= 1:
        return {**result, 'rejected': 'degenerate_face_landmark_geometry'}
    tree = cKDTree(src)
    confidences = np.zeros(len(src))
    for i in range(len(src)):
        _, indices = tree.query(src[i], k=min(9, len(src)))
        neighbours = np.asarray(indices)[1:]
        matrix = _affine(src[neighbours]/span, ref[neighbours]/ref_span)
        if matrix is None or not _affine_valid(matrix, cfg, face=True):
            continue
        error = float(np.linalg.norm(np.r_[src[i]/span, 1.] @ matrix-ref[i]/ref_span))
        if error <= cfg['maximum_face_heldout_error_fraction']:
            confidences[i] = .95*(1-error/cfg['maximum_face_heldout_error_fraction'])
    accepted = confidences >= cfg['minimum_confidence']
    result['valid_landmarks'] = int(accepted.sum())
    if accepted.sum() < 12:
        return {**result, 'rejected': 'face_local_heldout_geometry_failed'}
    result.update(_rasterize_triangles(src[accepted], ref[accepted], confidences[accepted],
        source_labels, reference_labels, eligible, maps, max_edge=.35*span, face=True, settings=cfg))
    return result


def _eye_report_mask(name, face, labels):
    mask = np.zeros(labels.shape, bool)
    if not face or not face.get('size'):
        return mask
    scale = np.array([labels.shape[1], labels.shape[0]])/np.asarray(face['size'])
    for region in face.get('protected_regions') or []:
        if region.get('region') != name:
            continue
        points = np.asarray(region['polygon'], float)*scale
        canvas = Image.new('L', (labels.shape[1], labels.shape[0]))
        ImageDraw.Draw(canvas).polygon([tuple(point) for point in points], fill=1)
        mask |= np.asarray(canvas, bool)
    # Sapiens class 2 is eyeglass, not an eye. Iris/sclera are merged into
    # Face_Neck (3); their own material support comes from landmark polygons.
    return mask & (labels == 3)


def build_maps(source, reference, source_labels, reference_labels, source_faces=None, reference_faces=None,
               zooms=None, *, settings=None, algorithm_profile=DINO3_L_PROFILE):
    rows, token_report = mutual_matches(source, reference, settings=settings)
    accepted, geometry_rejected = geometric_matches(rows, source, reference, settings=settings)
    maps = {'ref_x': np.full(source_labels.shape, -1., np.float32),
            'ref_y': np.full(source_labels.shape, -1., np.float32),
            'confidence': np.zeros(source_labels.shape, np.float32),
            'valid': np.zeros(source_labels.shape, bool)}
    triangle_report = rasterize_dino(accepted, source, reference, source_labels, reference_labels, maps, settings=settings)
    zoom_reports = []
    for source_zoom, reference_zoom, label in (zooms or []):
        matches, tokens = mutual_matches(source_zoom, reference_zoom, settings=settings)
        verified, rejected = geometric_matches(matches, source_zoom, reference_zoom, settings=settings)
        triangles = rasterize_dino(verified, source_zoom, reference_zoom, source_labels, reference_labels, maps, settings=settings)
        zoom_reports.append({'label': label, 'token_matching': tokens.get(str(label)),
            'verified_match_count': len(verified), 'geometry_rejected_tokens': rejected,
            'triangles': triangles, 'source_crop': source_zoom['crop_box'],
            'reference_crop': reference_zoom['crop_box']})
    # Semantic face interiors do not identify eyes. DINO maps may only support
    # skin outside these polygons; feature pixels require canonical landmarks.
    eyes = _eye_report_mask('left_eye', source_faces, source_labels) | _eye_report_mask('right_eye', source_faces, source_labels)
    for key in ('ref_x', 'ref_y'):
        maps[key][eyes] = -1
    maps['confidence'][eyes] = 0
    maps['valid'][eyes] = False
    face_report = face_correspondence(source_faces, reference_faces, source_labels, reference_labels, maps, settings=settings)
    for name in ('left_eye', 'right_eye'):
        sm = _eye_report_mask(name, source_faces, source_labels)
        rm = _eye_report_mask(name, reference_faces, reference_labels)
        ys, xs = np.nonzero(sm & maps['valid'])
        if len(xs):
            rx, ry = np.rint(maps['ref_x'][ys, xs]).astype(int), np.rint(maps['ref_y'][ys, xs]).astype(int)
            ys, xs = ys[~rm[ry, rx]], xs[~rm[ry, rx]]
            maps['ref_x'][ys, xs] = maps['ref_y'][ys, xs] = -1
            maps['confidence'][ys, xs] = 0
            maps['valid'][ys, xs] = False
    regions = {}
    for name, ids in REGIONS.items():
        mask = (_eye_report_mask(name, source_faces, source_labels) if name in ('left_eye', 'right_eye')
                else np.isin(source_labels, ids))
        valid = mask & maps['valid']
        count, supported = int(mask.sum()), int(valid.sum())
        regions[name] = {'source_pixels': count, 'eligible_pixels': supported, 'valid_pixels': supported,
            'source_support_fraction': supported/max(count, 1), 'status': 'partial' if supported else 'unsupported',
            'confidence_q10_50_90': np.quantile(maps['confidence'][valid], [.1, .5, .9]).tolist() if supported else [],
            'methods': ([] if name in ('face', 'lips', 'left_eye', 'right_eye') else
                        ['dinov3_mutual_semantic_neighbours_local_geometry'])+(
                ['mediapipe_verified_piecewise_affine'] if name in ('face', 'lips', 'left_eye', 'right_eye') and face_report['accepted_triangles'] else []),
            'reason': 'locally_verified_partial_support' if supported else 'no_valid_local_geometry_or_unambiguous_features',
            'same_semantic_material_only': True, 'background_H_used': False, 'full_region_correspondence': False}
    maps['material_labels'] = source_labels.astype(np.int32)
    return maps, {'regions': regions, 'token_matching': token_report,
        'geometry_rejected_tokens': geometry_rejected, 'triangles': triangle_report,
        'material_zooms': zoom_reports,
        'face': face_report, 'mutual_match_count': len(rows), 'verified_match_count': len(accepted)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True, type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding='utf8'))
    profile = request['identity'].get('algorithm_profile')
    adapter = algorithm_configuration(profile)
    preprocess, settings, layout = adapter['preprocess'], adapter['matching'], adapter['token_layout']
    if (request.get('schema') != SCHEMA or request['identity']['preprocess'] != preprocess
            or request['identity']['matching'] != settings
            or request['identity']['model']['repository'] != adapter['repository']
            or request['identity'].get('token_layout') != layout):
        raise ValueError('Local correspondence worker recipe mismatch')
    if request['identity']['worker_sha256'] != file_sha(__file__):
        raise ValueError('Local correspondence worker changed during preparation')
    model_path = Path(request['model_directory'])
    for name, sha in request['identity']['model']['files_sha256'].items():
        if file_sha(model_path/name) != sha:
            raise ValueError('DINO model content changed: '+name)
    # Model failure is an explicit worker error; there is no hand-authored,
    # background-H or identity-model substitute masquerading as success.
    import torch
    from transformers import AutoModel
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    started = time.perf_counter()
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    model = AutoModel.from_pretrained(model_path, local_files_only=True).eval().to(device)
    if (getattr(model.config, 'model_type', None) != layout['model_type']
            or getattr(model.config, 'patch_size', None) != preprocess['patch_size']
            or getattr(model.config, 'num_register_tokens', 0) != layout['register_tokens']
            or getattr(model.config, 'hidden_size', None) != layout['embedding_dimension']):
        raise ValueError('Loaded DINO config differs from bound token/grid layout')
    if any(getattr(model.config, key, None) != value
           for key, value in adapter.get('expected_config', {}).items()):
        raise ValueError('Loaded DINO architecture differs from the explicit profile')
    observations, labels, original, images, raw_semantics = {}, {}, {}, {}, {}
    for role in ('source', 'reference'):
        record = request['images'][role]
        if file_sha(record['path']) != record['sha256'] or file_sha(record['labels_path']) != record['labels_sha256']:
            raise ValueError('Worker image/semantic input bytes changed: '+role)
        with np.load(record['labels_path'], allow_pickle=False) as data:
            raw_labels = data['labels']
        if array_sha(raw_labels) != request['binding'][role+'_labels_sha256']:
            raise ValueError('Worker semantic content binding mismatch: '+role)
        image = Image.open(record['path']).convert('RGB')
        images[role] = image
        raw_semantics[role] = np.asarray(Image.fromarray(raw_labels.astype(np.int32), mode='I').resize(image.size, Image.Resampling.NEAREST))
        size = request['binding']['proxy_size'] if role == 'source' else record['original_size']
        labels[role] = np.asarray(Image.fromarray(raw_labels.astype(np.int32), mode='I').resize(tuple(size), Image.Resampling.NEAREST))
        original[role] = record['original_size']
        observations[role] = extract(model, image, raw_labels, device, preprocess=preprocess,
                                    settings=settings, token_layout=layout)
    zooms = []
    for label in sorted(DINO_LABELS):
        if min(np.count_nonzero(observations[role]['labels'] == label) for role in ('source', 'reference')) >= preprocess['material_zoom_if_tokens_below']:
            continue
        if min(np.count_nonzero(raw_semantics[role] == label) for role in ('source', 'reference')) < preprocess['material_crop_minimum_pixels']:
            continue
        pair = []
        for role in ('source', 'reference'):
            yy, xx = np.nonzero(raw_semantics[role] == label)
            margin = int(np.ceil(max(np.ptp(xx), np.ptp(yy))*preprocess['material_crop_context_fraction']))
            box = (max(0,int(xx.min())-margin), max(0,int(yy.min())-margin),
                   min(images[role].width,int(xx.max())+margin+1), min(images[role].height,int(yy.max())+margin+1))
            selected = raw_semantics[role][box[1]:box[3], box[0]:box[2]].copy()
            selected[selected != label] = 0
            obs = extract(model, images[role].crop(box), selected, device,
                          maximum_edge=preprocess['material_zoom_maximum_edge'], preprocess=preprocess,
                          settings=settings, token_layout=layout)
            obs.update(crop_box=list(box), full_image_size=list(images[role].size))
            pair.append(obs)
        zooms.append((*pair, label))
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    maps, diagnostics = build_maps(observations['source'], observations['reference'], labels['source'], labels['reference'],
        request['faces']['source'], request['faces']['reference'], zooms=zooms,
        settings=settings, algorithm_profile=profile)
    directory = Path(request['output_directory'])
    asset_path = directory/'local-correspondence.npz'
    temporary_asset = directory/('local-correspondence-'+str(os.getpid())+'.npz')
    np.savez_compressed(temporary_asset, **maps)
    temporary_asset.replace(asset_path)
    receipt = {'schema': SCHEMA, 'binding': request['binding'], 'identity': request['identity'],
        'asset': {'path': str(asset_path), 'sha256': file_sha(asset_path),
                  'representation': 'source_proxy_to_original_reference_pixel_centres_ref_x_ref_y_confidence_valid'},
        'source_size': original['source'], 'reference_size': original['reference'],
        'proxy_size': request['binding']['proxy_size'], 'coordinate_space': 'source_proxy',
        'reference_coordinate_space': 'original_oriented_reference_pixels',
        'invalid_coordinate': -1., 'confidence_is_spatial_evidence_not_effect_dose': True,
        'person_geometry_warped': False, 'reference_texture_imported': False, 'background_H_used': False,
        'visual_acceptance': False, 'diagnostics': {k:v for k,v in diagnostics.items() if k != 'regions'},
        'regions': diagnostics['regions'],
        'feature_observations': {role: {'model_size': list(obs['size']),
            'patch_count': len(obs['features']), 'embedding_dimension': obs['features'].shape[1],
            'excluded_class_tokens': layout['class_tokens'], 'excluded_register_tokens': layout['register_tokens'],
            'patch_size': obs['patch_size'], 'features_finite': bool(np.isfinite(obs['features']).all())}
            for role, obs in observations.items()},
        'runtime': {'device': device,
            'torch': torch.__version__, 'transformers': importlib.metadata.version('transformers'),
            'elapsed_seconds': round(time.perf_counter()-started, 3),
            'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else 0,
            'peak_gpu_reserved_bytes': torch.cuda.max_memory_reserved() if device == 'cuda' else 0}}
    temporary_receipt = directory/('receipt-'+str(os.getpid())+'.json')
    temporary_receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    temporary_receipt.replace(directory/'receipt.json')
    print(json.dumps({'asset': str(asset_path), 'regions': receipt['regions'], 'runtime': receipt['runtime']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
