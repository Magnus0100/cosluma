"""Cached CPU face landmarks; semantic crop and source-derived eye/lip masks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from local_photo_analysis import _worker, file_identity
from runtime_resources import code_path


LIP_OUTER_INDICES = (61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
                     375, 321, 405, 314, 17, 84, 181, 91, 146)
LIP_INNER_INDICES = (78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
                     324, 318, 402, 317, 14, 87, 178, 88, 95)


def source_feature_regions(face, size):
    """Scale cached source features while retaining the ordered lip topology.

    The outer contour contains the Cupid notch; the inner contour is the mouth
    opening. Neither can be recovered from the worker's convex protection hull.
    Missing full landmarks do not fall back to that hull or invoke a detector.
    """
    import numpy as np

    size, source_size = np.asarray(size, np.float32), np.asarray(face.get('size', []), np.float32)
    if (size.shape != (2,) or source_size.shape != (2,) or not np.isfinite(size).all()
            or not np.isfinite(source_size).all() or np.any(size <= 0) or np.any(source_size <= 0)):
        raise ValueError('Source feature regions require finite source and target canvas sizes')
    scale = size/source_size
    regions = []
    for region in face.get('protected_regions', []):
        if region.get('region') in {'left_eye', 'right_eye'}:
            points = _feature_polygon(region.get('polygon'), 'eye')
            regions.append({**region, 'polygon': (points*scale).tolist()})
    points = np.asarray(face.get('landmarks', []), np.float32)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) <= max(LIP_INNER_INDICES)
            or not np.isfinite(points).all()):
        return regions
    outer = _feature_polygon(points[list(LIP_OUTER_INDICES)]*scale, 'outer lips')
    inner = _feature_polygon(points[list(LIP_INNER_INDICES)]*scale, 'inner mouth', allow_flat=True)
    regions.append({'region': 'lips', 'topology': 'mediapipe_canonical_ordered_lips',
                    'inner_mouth_state': 'open' if abs(_polygon_area(inner)) >= 1e-6 else 'closed',
                    'landmark_indices': list(LIP_OUTER_INDICES),
                    'inner_landmark_indices': list(LIP_INNER_INDICES),
                    'polygon': outer.tolist(), 'inner_polygon': inner.tolist()})
    return regions


def _polygon_area(points):
    import numpy as np

    return float(np.sum(points[:, 0]*np.roll(points[:, 1], -1)-points[:, 1]*np.roll(points[:, 0], -1)))


def _feature_polygon(values, name, *, allow_flat=False):
    import numpy as np

    points = np.asarray(values, np.float32)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3
            or not np.isfinite(points).all()):
        raise ValueError(f'Source {name} requires a finite ordered polygon')
    if np.array_equal(points[0], points[-1]):
        points = points[:-1]
    if len(points) < 3 or (not allow_flat and abs(_polygon_area(points)) < 1e-6):
        raise ValueError(f'Source {name} requires a nondegenerate ordered polygon')
    return points


def _polygon_distance(points, xy):
    """Exact Euclidean segment distance and even-odd membership, without hulls."""
    import numpy as np

    nearest = np.full(xy.shape[:2], np.inf, np.float32)
    inside = np.zeros(xy.shape[:2], bool)
    for start, end in zip(points, np.roll(points, -1, axis=0)):
        edge = end-start
        length_squared = float(edge @ edge)
        if length_squared == 0:
            continue
        offset = xy-start
        fraction = np.clip(np.sum(offset*edge, axis=-1)/length_squared, 0, 1)
        residual = offset-fraction[..., None]*edge
        nearest = np.minimum(nearest, np.sqrt(np.sum(residual*residual, axis=-1)))
        if edge[1] != 0:
            crossing = ((start[1] > xy[..., 1]) != (end[1] > xy[..., 1]))
            intersection_x = start[0]+(xy[..., 1]-start[1])*edge[0]/edge[1]
            inside ^= crossing & (xy[..., 0] < intersection_x)
    return inside | (nearest <= 1e-6), nearest


def _lip_geometry(shape, region, padding=0.):
    import numpy as np

    outer = _feature_polygon(region.get('polygon'), 'outer lips')
    inner_values = region.get('inner_polygon')
    inner = _feature_polygon(inner_values, 'inner mouth', allow_flat=True) if inner_values is not None else None
    # A closed mouth has a legitimate zero-area inner contour. It supplies no
    # opening to subtract and must not invalidate its ordered outer lip shape.
    if inner is not None and abs(_polygon_area(inner)) < 1e-6:
        inner = None
    span = float(np.ptp(outer, axis=0).max())
    if span < 1:
        raise ValueError('Source lip topology requires a nondegenerate source contour')
    left, top = np.maximum(np.floor(outer.min(axis=0)-padding-1), 0).astype(int)
    right, bottom = np.minimum(np.ceil(outer.max(axis=0)+padding+2), shape[::-1]).astype(int)
    if left >= right or top >= bottom:
        raise ValueError('Source lip contour does not intersect the current proxy canvas')
    roi = np.s_[top:bottom, left:right]
    yy, xx = np.indices((bottom-top, right-left), dtype=np.float32)
    xy = np.stack((xx+left, yy+top), axis=-1)
    in_outer, outer_distance = _polygon_distance(outer, xy)
    in_inner = np.zeros(in_outer.shape, bool)
    nearest = outer_distance
    if inner is not None:
        in_inner, inner_distance = _polygon_distance(inner, xy)
        in_inner &= in_outer
        nearest = np.minimum(nearest, inner_distance)
    material = in_outer & ~in_inner
    signed_distance = np.where(material, nearest, -nearest)
    return roi, span, in_outer, in_inner, outer_distance, signed_distance


def prepare_feature_materials(labels, protection, feature_regions):
    """Refine appearance material ownership without changing source semantics.

    Skin mislabeled as a lip receives the normal face adaptation. The actual
    opening is oral material; only actual mouth features retain L protection.
    T, B and S consume these same derived labels and this bounded protection.
    """
    import numpy as np

    labels, protection = np.asarray(labels), np.asarray(protection)
    if (labels.ndim != 2 or protection.shape != labels.shape or not isinstance(feature_regions, list)
            or not np.isfinite(protection).all() or protection.min() < 0 or protection.max() > 1):
        raise ValueError('Feature material refinement requires matching bounded masks')
    appearance, bounded = labels.copy(), np.zeros(labels.shape, np.float32)
    lips = [region for region in feature_regions if region.get('region') == 'lips']
    report = {'schema': 'feature_material_policy.v1', 'status': 'unsupported',
              'reason': 'no_source_ordered_lip_topology', 'raw_semantics_modified': False,
              'pixel_count_space': 'source_proxy',
              'raw_semantics_policy': 'input_array_and_source_semantic_assets_retained; appearance_labels_only',
              'appearance_semantic_lip_to_face_pixels': 0, 'inner_mouth_to_oral_pixels': 0,
              'lip_outer_pixels': 0, 'inner_mouth_pixels': 0,
              'inner_mouth_states': [],
              'outer_landmark_indices': list(LIP_OUTER_INDICES),
              'inner_landmark_indices': list(LIP_INNER_INDICES),
              'protection_policy': 'existing_eye_field_and_inward_outer_mouth_only',
              'geometry_modified': False, 'person_coverage_modified': False}
    if not lips or any(region.get('inner_polygon') is None for region in lips):
        return appearance, protection.astype(np.float32, copy=True), report
    raw_lips = np.isin(labels, [24, 25])
    appearance[raw_lips] = 3
    for region in feature_regions:
        if region.get('region') not in {'left_eye', 'right_eye'}:
            continue
        points = _feature_polygon(region.get('polygon'), 'eye')
        span = float(np.ptp(points, axis=0).max())
        padding = max(2., .3*span)
        left, top = np.maximum(np.floor(points.min(axis=0)-padding), 0).astype(int)
        right, bottom = np.minimum(np.ceil(points.max(axis=0)+padding+1), labels.shape[::-1]).astype(int)
        roi = np.s_[top:bottom, left:right]
        bounded[roi] = np.maximum(bounded[roi], protection[roi])
    for region in lips:
        roi, span, in_outer, in_inner, outer_distance, _ = _lip_geometry(labels.shape, region)
        local_labels, local_appearance = labels[roi], appearance[roi]
        local_lips = raw_lips[roi] & in_outer
        local_appearance[local_lips] = local_labels[local_lips]
        oral = in_inner & np.isin(local_labels, [3, 24, 25, 26, 27, 28])
        local_appearance[oral] = 26
        dose = np.clip(outer_distance/max(1., .02*span), 0, 1)
        dose = (dose*dose*(3-2*dose)).astype(np.float32)
        dose *= in_outer & np.isin(local_labels, [3, 24, 25, 26, 27, 28])
        bounded[roi] = np.maximum(bounded[roi], dose)
        report['lip_outer_pixels'] += int(in_outer.sum())
        report['inner_mouth_pixels'] += int(in_inner.sum())
        report['inner_mouth_states'].append(region.get('inner_mouth_state',
            'open' if abs(_polygon_area(_feature_polygon(region['inner_polygon'], 'inner mouth', allow_flat=True))) >= 1e-6
            else 'closed'))
    report.update(status='prepared', reason='source_ordered_outer_and_inner_material_ownership',
                  appearance_semantic_lip_to_face_pixels=int(np.count_nonzero(raw_lips & (appearance == 3))),
                  inner_mouth_to_oral_pixels=int(np.count_nonzero((labels != 26) & (appearance == 26))),
                  bounded_protection_pixels=int(np.count_nonzero(bounded)))
    return appearance, bounded, report


def mouth_lightness_domain(shape, feature_regions, labels):
    """The ordered mouth receives face illumination, independently of pigment.

    Skin, lip and oral appearance labels stay unchanged. This temporary domain
    never enlarges the lip colour support or supplies reference mouth pixels.
    """
    import numpy as np

    labels = np.asarray(labels)
    if labels.shape != tuple(shape) or not isinstance(feature_regions, list):
        raise ValueError('Mouth illumination requires matching source feature domains')
    domain = np.zeros(shape, bool)
    for region in feature_regions:
        if region.get('region') != 'lips':
            continue
        roi, _, outer, _, _, _ = _lip_geometry(shape, region)
        domain[roi] |= outer & np.isin(labels[roi], [3, 24, 25, 26, 27, 28])
    return domain


def prepare_strict_lip_support(shape, feature_regions, labels, alpha):
    """Inward-only canonical lip confidence and a narrow skin rejection collar.

    The collar is an exclusion signal, never a colour or feature application dose.
    Landmark polygons must already use the current source proxy coordinates.
    """
    import numpy as np
    shape = tuple(shape)
    labels, alpha = map(np.asarray, (labels, alpha))
    if (len(shape) != 2 or any(n <= 0 for n in shape) or labels.shape != shape or alpha.shape != shape
            or not isinstance(feature_regions, list) or not np.isfinite(alpha).all()
            or alpha.min() < 0 or alpha.max() > 1):
        raise ValueError('Strict lip support requires current source regions and matching bounded masks')
    support = np.zeros(shape, np.float32)
    collar = np.zeros(shape, bool)
    regions = []
    inward_safety = 2.13
    for region in feature_regions:
        if region.get('region') != 'lips':
            continue
        points = _feature_polygon(region.get('polygon'), 'outer lips')
        span = float(np.ptp(points, axis=0).max())
        if span < 1:
            raise ValueError('Strict lip support requires a nondegenerate source lip contour')
        feather, collar_width = max(1., .02*span), max(1., .025*span)
        roi, _, in_outer, in_inner, outer_distance, inward_distance = _lip_geometry(
            shape, region, padding=collar_width)
        inside = in_outer & ~in_inner
        # Bilinear corners can be sqrt(2) proxy pixels away, plus at most
        # sqrt(2)/2 for native pixel-centre offsets. The 2.13 inward band covers
        # this bound for every native/proxy ratio >= 1, without growing colour.
        dose = np.clip((inward_distance-inward_safety)/feather, 0, 1)
        dose = (dose*dose*(3-2*dose)).astype(np.float32)
        opaque = alpha[roi] >= .995
        dose *= inside & np.isin(labels[roi], [3, 24, 25]) & opaque
        support[roi] = np.maximum(support[roi], dose)
        collar[roi] |= (~in_outer) & (outer_distance <= collar_width) & (labels[roi] == 3) & opaque
        regions.append({'bounds_proxy': [int(roi[1].start), int(roi[0].start), int(roi[1].stop), int(roi[0].stop)],
                        'lip_span_proxy_px': span, 'inward_safety_proxy_px': inward_safety, 'inward_feather_proxy_px': feather,
                        'boundary_distance': 'exact_ordered_outer_and_inner_polygon_segments',
                        'inner_mouth_excluded': bool(np.any(in_inner)),
                        'skin_reject_collar_proxy_px': collar_width})
    report = {'schema': 'strict_source_lip_support.v2',
              'status': 'prepared' if np.any(support) else 'unsupported',
              'reason': 'source_ordered_lip_ring_inward_feather' if regions else 'no_source_lip_landmarks',
              'regions': regions, 'supported_pixels': int(np.count_nonzero(support)),
              'skin_reject_collar_pixels': int(collar.sum()),
              'source_lip_labels_outside_support': int(np.count_nonzero(np.isin(labels, [24, 25]) & (support == 0))),
              'outside_outer_colour_support': False, 'inner_mouth_colour_support': False,
              'collar_used_for_colour_dose': False,
              'output_lightness_modified': False}
    return support, collar, report


def prepare_feature_chroma_guard(beauty_lab, fused_lab, labels, alpha, protection, *, feature_regions):
    """Infer smooth ambient ab/L from clean skin for the existing eye/lip mask.

    Current source landmark polygons are already scaled to this proxy canvas.
    The consumer uses native Beauty L to turn this ratio into an ab shift.
    Neither generated detail nor output lightness is reconstructed here.
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter

    beauty = np.asarray(beauty_lab, np.float32)
    fused = np.asarray(fused_lab, np.float32)
    labels, alpha, protection = map(np.asarray, (labels, alpha, protection))
    shape = beauty.shape[:2]
    if (beauty.ndim != 3 or beauty.shape[-1] != 3 or fused.shape != beauty.shape
            or any(value.shape != shape for value in (labels, alpha, protection))
            or any(not np.isfinite(value).all() for value in (beauty, fused, alpha, protection))
            or alpha.min() < 0 or alpha.max() > 1 or protection.min() < 0 or protection.max() > 1):
        raise ValueError('Feature chroma guard requires finite matching proxy canvases and bounded masks')
    if not isinstance(feature_regions, list):
        raise ValueError('Feature chroma guard requires current source landmark regions')
    owned = np.isin(labels, [2, 3, 24, 25, 26, 27, 28]) & (alpha >= .995)
    guard = np.zeros((*shape, 3), np.float32)
    lip_support, _, lip_report = prepare_strict_lip_support(shape, feature_regions, labels, alpha)
    guard[..., 2] = lip_support
    for region in feature_regions:
        if region.get('region') not in {'left_eye', 'right_eye'}:
            continue
        points = np.asarray(region.get('polygon'), np.float32)
        if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3
                or not np.isfinite(points).all()):
            raise ValueError('Eye protection requires current finite source eye polygons')
        span = float(np.ptp(points, axis=0).max())
        # This only selects the existing soft eye field; it creates no new eye
        # dose and keeps the worker's expanded lip field out of colour replay.
        padding = max(2., .3*span)
        left, top = np.maximum(np.floor(points.min(axis=0)-padding), 0).astype(int)
        right, bottom = np.minimum(np.ceil(points.max(axis=0)+padding+1), shape[::-1]).astype(int)
        eye_roi = np.s_[top:bottom, left:right]
        guard[eye_roi][..., 2] = np.maximum(guard[eye_roi][..., 2], protection[eye_roi]*owned[eye_roi])
    report = {'schema': 'feature_chroma_guard.v1', 'status': 'rejected',
              'reason': None, 'protected_pixels': int(np.count_nonzero(guard[..., 2])),
              'donor_pixels': 0, 'accepted_donor_pixels': 0, 'rejected_donor_pixels': 0,
              'maximum_ambient_ab_per_beauty_L': .08,
              'fallback': 'existing_feature_mask_with_zero_ambient; Beauty_feature_chroma_retained',
              'output_lightness_modified': False, 'source_texture_used': False,
              'lip_support': lip_report}
    if not report['protected_pixels']:
        report.update(status='identity', reason='no_owned_opaque_protected_features' if feature_regions
                      else 'no_source_feature_landmarks', roi=None)
        return guard, report
    yy, xx = np.nonzero(owned)
    left, top, right, bottom = int(xx.min()), int(yy.min()), int(xx.max())+1, int(yy.max())+1
    roi = np.s_[top:bottom, left:right]
    sigma = max(1., .06 * max(right-left, bottom-top))
    report.update(roi=[left, top, right, bottom], sigma_proxy_px=sigma)
    donor = (labels[roi] == 3) & (protection[roi] < .05) & (guard[roi][..., 2] < .05) & (alpha[roi] >= .995)
    report['donor_pixels'] = int(donor.sum())
    if report['donor_pixels'] < 24:
        report['reason'] = 'insufficient_clean_opaque_skin_donors'
        return guard, report
    ratio = (fused[roi][..., 1:] - beauty[roi][..., 1:]) / np.maximum(beauty[roi][..., :1], .05)
    observed = ratio[donor]
    center = np.median(observed, axis=0)
    radius = np.linalg.norm(observed-center, axis=-1)
    median_radius = float(np.median(radius))
    mad = float(np.median(np.abs(radius-median_radius)))
    cutoff = max(.015, median_radius + 3 * 1.4826 * mad)
    keep = radius <= cutoff
    donor[donor] = keep
    report.update(accepted_donor_pixels=int(keep.sum()), rejected_donor_pixels=int((~keep).sum()),
                  donor_outlier_cutoff_ab_per_L=cutoff)
    if report['accepted_donor_pixels'] < 24:
        report['reason'] = 'insufficient_robust_skin_donors'
        return guard, report
    support = gaussian_filter(donor.astype(np.float32), sigma)
    numerator = gaussian_filter(ratio * donor[..., None], (sigma, sigma, 0))
    ambient = numerator / np.maximum(support[..., None], 1e-5)
    ambient[support < 1e-5] = 0
    norm = np.linalg.norm(ambient, axis=-1)
    ambient *= np.minimum(1., .08 / np.maximum(norm, 1e-8))[..., None]
    guarded = guard[roi][..., 2] > 0
    guard[roi][..., :2] = np.where(guarded[..., None], ambient, 0)
    missing = int(np.count_nonzero(guarded & (support < 1e-5)))
    report.update(status='partial' if missing else 'prepared',
                  reason='feature_pixels_without_local_donor_support' if missing else 'robust_local_skin_ambient',
                  unsupported_feature_pixels=missing,
                  actual_maximum_ambient_ab_per_L=float(np.linalg.norm(guard[roi][..., :2], axis=-1).max()))
    return guard, report


def prepare_face_protection(root, input_path, semantic_result, *, registry_path=None):
    return _prepare_face_protections(root, [(input_path, semantic_result)], registry_path=registry_path)[0]


def prepare_face_protection_pair(root, source_path, source_semantics, reference_path,
                                 reference_semantics, *, registry_path=None):
    """Resolve both caches before starting one CPU detector for only the misses."""
    return tuple(_prepare_face_protections(root, [(source_path, source_semantics),
        (reference_path, reference_semantics)], registry_path=registry_path))


def _face_result(directory, identity, input_path, model, *, staged=False):
    """A result marker is useful only after every per-image artifact is complete."""
    import numpy as np
    from PIL import Image
    from zipfile import BadZipFile
    try:
        result = json.loads((directory/'result.json').read_text(encoding='utf8'))
        if (result.get('schema_version') != 'source_face_protection.v1'
                or (not staged and result.get('identity') != identity)
                or Path(result['input_path']).resolve() != input_path.resolve()
                or result.get('model_sha256') != identity['model']):
            return None
        field = directory/'face-protection.npz'
        if Path(result['field_path']).resolve() != field.resolve():
            return None
        with Image.open(input_path) as opened:
            size = list(opened.size)
        if result.get('size') != size:
            return None
        with np.load(field, allow_pickle=False) as resource:
            values = resource['protection']
            if (values.shape != (size[1], size[0]) or values.dtype != np.float32
                    or not np.isfinite(values).all() or values.min() < 0 or values.max() > 1):
                return None
        points = np.asarray(result['landmarks'], dtype=np.float64)
        if not np.isfinite(points).all():
            return None
        if points.size:
            if (points.ndim != 2 or points.shape[1] != 2
                    or result.get('method') != 'mediapipe_source_eyes_lips'
                    or not isinstance(result.get('protected_regions'), list)
                    or len(result['protected_regions']) != 3):
                return None
        elif (result.get('method') != 'no_landmarks_protect_entire_face_neck'
              or result.get('protected_regions') != []):
            return None
        for name in ('face-protection.png', 'face-protection-overlay.jpg'):
            with Image.open(directory/name) as opened:
                if list(opened.size) != size:
                    return None
                opened.verify()
        return result
    except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile):
        return None


def _face_write(path, value):
    import os
    import uuid
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('w', encoding='utf8') as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _prepare_face_protections(root, items, *, registry_path=None):
    import os
    import tempfile
    from contextlib import ExitStack
    from cache_store import exclusive_file_lock
    root = Path(root)
    from current_model_registry import load_registry
    registry = load_registry(root, registry_path=registry_path)
    worker = code_path('local_face_protection_worker.py')
    model = Path(registry['face_model_path'])
    python = Path(registry['face_python'])
    contexts = []
    for input_path, semantic_result in items:
        input_path = Path(input_path)
        labels = Path(semantic_result['labels_path'])
        identity = {'input': file_identity(input_path), 'labels': file_identity(labels),
            'model': file_identity(model), 'worker': file_identity(worker),
            'parameters': {'num_faces': 1, 'semantic_crop': 'Face_Neck',
                'detection_confidence': .3, 'presence_confidence': .3}}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
        directory = root/'outputs/cache/face-protection'/key
        directory.mkdir(parents=True, exist_ok=True)
        contexts.append({'input': input_path, 'semantic': semantic_result,
            'identity': identity, 'directory': directory})
    # Stable ordering and deduplication prevent pair/single and reversed-pair
    # processes from racing cache publication or deadlocking on the same image.
    unique = {str(row['directory'].resolve()): row for row in contexts}
    with ExitStack() as locks:
        for key in sorted(unique):
            directory = unique[key]['directory']
            locks.enter_context(exclusive_file_lock(directory.with_name(directory.name+'.single-flight.lock')))
        missing = []
        for row in unique.values():
            result = _face_result(row['directory'], row['identity'], row['input'], model)
            if result is None:
                row['staging'] = Path(locks.enter_context(tempfile.TemporaryDirectory(
                    prefix='.prepare-', dir=row['directory'])))
                _face_write(row['directory']/'semantic-input.json', row['semantic'])
                missing.append(row)
            else:
                row['result'] = {**result, 'cache_hit': True}
        if missing:
            batch = missing[0]['staging']/'batch.json'
            _face_write(batch, [{'input': str(row['input']),
                'semantics': str(row['directory']/'semantic-input.json'),
                'output_dir': str(row['staging'])} for row in missing])
            _worker([python, '-X', 'utf8', worker, '--model', model, '--batch', batch],
                missing[0]['directory']/'worker.log')
            for row in missing:
                result = _face_result(row['staging'], row['identity'], row['input'], model, staged=True)
                if (result is None or file_identity(row['input']) != row['identity']['input']
                        or file_identity(Path(row['semantic']['labels_path'])) != row['identity']['labels']):
                    raise ValueError('Face protection worker returned incomplete per-image assets')
                row['result'] = {**result, 'identity': row['identity'], 'cache_hit': False,
                    'field_path': str((row['directory']/'face-protection.npz').resolve())}
            for row in missing:
                for name in ('face-protection.npz', 'face-protection.png', 'face-protection-overlay.jpg'):
                    os.replace(row['staging']/name, row['directory']/name)
                # Publish the complete contract last. A failed worker never
                # exposes a partial result.json, and readers share this lock.
                _face_write(row['directory']/'result.json', row['result'])
        return [unique[str(row['directory'].resolve())]['result'] for row in contexts]
