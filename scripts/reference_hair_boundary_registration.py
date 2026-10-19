"""Strict, source-bound fallback registration for a hair boundary.

The normal local correspondence producer deliberately requires same-material
DINO anchors.  A generated reference can lose all hair token matches even when
the photographed face and hair silhouette remain aligned.  This producer is a
separate observation: it derives one affine from independently verified face
landmarks, then admits it for hair only after checking the *actual* source and
reference alpha/owner contours.  It never relaxes the DINO receipt or imports
reference pixels.
"""
from __future__ import annotations

import hashlib
import numpy as np
from native_cuda_ndimage import distance_transform_edt

SCHEMA = 'source_hair_boundary_registration.v1'
PROFILE = {
    'maximum_face_q90_reference_pixels': 8.0,
    'maximum_registration_error_reference_pixels': 8.0,
    'minimum_registered_contour_fraction': .7,
    'maximum_affine_condition': 3.0,
    'minimum_face_anchors': 12,
    'owner': 4,
}


def array_sha(values):
    values = np.ascontiguousarray(values)
    return hashlib.sha256(str((values.shape, values.dtype.str)).encode() + values.tobytes()).hexdigest()


def _affine(source, target):
    design = np.c_[source, np.ones(len(source))]
    if np.linalg.matrix_rank(design) < 3:
        return None
    return np.linalg.lstsq(design, target, rcond=None)[0]


def _points(face, shape):
    if not isinstance(face, dict):
        return None
    points = np.asarray(face.get('landmarks', []), float)
    size = np.asarray(face.get('size', []), float)
    if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 468:
        return None
    if size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0):
        return None
    points = points[:468] * np.asarray([shape[1], shape[0]], float) / size
    return points if np.isfinite(points).all() else None


def _semantic_face_anchors(source, reference, source_labels, reference_labels):
    si, ri = np.rint(source).astype(int), np.rint(reference).astype(int)
    source_size = np.asarray([source_labels.shape[1], source_labels.shape[0]])
    reference_size = np.asarray([reference_labels.shape[1], reference_labels.shape[0]])
    bounded = ((si >= 0).all(1) & (si < source_size).all(1)
               & (ri >= 0).all(1) & (ri < reference_size).all(1))
    semantic = np.zeros(len(source), bool)
    semantic[bounded] = (np.isin(source_labels[si[bounded, 1], si[bounded, 0]], (3, 24, 25))
                         & np.isin(reference_labels[ri[bounded, 1], ri[bounded, 0]], (3, 24, 25)))
    return np.flatnonzero(semantic)


def _fit_face_affine(source, reference, ids):
    """Robust raw-pixel fit; raw source and reference grids may differ in size."""
    if len(ids) < PROFILE['minimum_face_anchors']:
        return None, {'state': 'rejected', 'reason': 'insufficient_face_landmark_anchors',
                      'face_anchor_count': int(len(ids))}
    X = np.c_[source[ids], np.ones(len(ids))]
    Y = reference[ids]
    affine = _affine(source[ids], reference[ids])
    if affine is None:
        return None, {'state': 'rejected', 'reason': 'collinear_face_landmark_anchors',
                      'face_anchor_count': int(len(ids))}
    for _ in range(8):
        residual = np.linalg.norm(X @ affine - Y, axis=1)
        weight = np.minimum(1., PROFILE['maximum_face_q90_reference_pixels'] /
                            np.maximum(residual, 1e-4))
        affine = np.linalg.lstsq(X * weight[:, None], Y * weight[:, None], rcond=None)[0]
    residual = np.linalg.norm(X @ affine - Y, axis=1)
    singular = np.linalg.svd(affine[:2], compute_uv=False)
    q90 = float(np.quantile(residual, .9))
    condition = float(singular.max() / max(singular.min(), 1e-8))
    ok = (np.linalg.det(affine[:2]) > 0 and np.isfinite(condition)
          and condition <= PROFILE['maximum_affine_condition']
          and q90 <= PROFILE['maximum_face_q90_reference_pixels'])
    row = {'state': 'qualified' if ok else 'rejected', 'face_anchor_count': int(len(ids)),
           'face_q90_reference_pixels': q90, 'face_max_reference_pixels': float(residual.max()),
           'affine_condition': condition, 'affine': affine.tolist()}
    return (affine if ok else None), row


def prepare(source_alpha, reference_alpha, source_labels, reference_labels, faces,
            *, source_box=(0, 0), source_sha256=None, reference_sha256=None,
            source_alpha_sha256=None, reference_alpha_sha256=None,
            source_labels_sha256=None, reference_labels_sha256=None,
            face_producer_identity_sha256=None):
    """Return a strict local native map and an independently bound receipt.

    ``source_labels`` and ``source_alpha`` are the current native crop.  Face
    landmarks are transformed from their own observation sizes to the full
    native source grid, then to this crop.  The returned map is valid only on
    source hair/alpha pixels and is never used for other semantic owners.
    """
    a = np.asarray(source_alpha, np.float32)
    ra = np.asarray(reference_alpha, np.float32)
    sl = np.asarray(source_labels)
    rl = np.asarray(reference_labels)
    if a.ndim != 2 or ra.ndim != 2 or sl.shape != a.shape or rl.shape != ra.shape:
        raise ValueError('hair fallback alpha/owner shapes differ')
    if sl.dtype.kind not in 'iu' or rl.dtype.kind not in 'iu':
        raise ValueError('hair fallback owners must be integer labels')
    source_face = _points((faces or {}).get('source'), faces.get('source_native_size')
                          if isinstance(faces, dict) and faces.get('source_native_size') else a.shape)
    reference_face = _points((faces or {}).get('reference'), ra.shape)
    # Source face points are supplied in proxy coordinates; caller supplies the
    # native full size separately in the face record for an unambiguous scale.
    if source_face is None or reference_face is None:
        return _empty('missing_or_nonfinite_face_landmarks')
    # _points already scales to source_native_size.  Apply the full-image crop
    # translation before fitting against local source pixels.
    bx, by = map(float, source_box)
    source_face = source_face - np.asarray([bx, by])
    ids = _semantic_face_anchors(source_face, reference_face, sl, rl)
    affine, registration = _fit_face_affine(source_face, reference_face, ids)
    binding = {
        'schema': SCHEMA, 'source_sha256': source_sha256, 'reference_sha256': reference_sha256,
        'source_alpha_sha256': source_alpha_sha256, 'reference_alpha_sha256': reference_alpha_sha256,
        'source_labels_sha256': source_labels_sha256 or array_sha(sl),
        'reference_labels_sha256': reference_labels_sha256 or array_sha(rl),
        'source_box_native': [int(source_box[0]), int(source_box[1])],
        'source_native_crop_size': [int(a.shape[1]), int(a.shape[0])],
        'reference_native_size': [int(ra.shape[1]), int(ra.shape[0])],
        'face_producer_identity_sha256': face_producer_identity_sha256,
    }
    receipt = {'schema': SCHEMA, 'profile': dict(PROFILE), 'binding': binding,
               'registration': registration, 'state': 'rejected', 'changed_pixels': 0,
               'source_geometry_changed': False, 'source_alpha_changed': False,
               'reference_texture_imported': False, 'background_is_light_evidence': False}
    if affine is None:
        receipt['reason'] = registration['reason']
        return {'maps': None, 'receipt': receipt}
    source_inside = a >= .5
    hair = source_inside & (sl == PROFILE['owner'])
    if not hair.any():
        receipt['reason'] = 'source_hair_owner_alpha_empty'
        return {'maps': None, 'receipt': receipt}
    # Use an alpha/owner boundary band, not a semantic interior, for the hair
    # qualification.  The depth is measured in native source pixels from the
    # observed reference-space budget.
    # Measure the band from the hair owner itself.  Measuring distance to the
    # whole person would retain only the outermost silhouette and can miss the
    # hair/face or hair/cloth contour where the same owner is actually tested.
    from boundary_light_geometry_context import ExactMaskDistance
    source_hair_depth = ExactMaskDistance.compute(hair)
    source_depth = source_hair_depth.for_mask(hair)
    scale = float(np.sqrt(abs(np.linalg.det(affine[:2]))))
    max_depth = int(np.ceil(PROFILE['maximum_registration_error_reference_pixels'] /
                            max(scale, 1e-6)))
    band = hair & (source_depth <= max_depth)
    y, x = np.indices(a.shape, dtype=np.float32)
    rx = x * affine[0, 0] + y * affine[1, 0] + affine[2, 0]
    ry = x * affine[0, 1] + y * affine[1, 1] + affine[2, 1]
    valid = ((rx >= 0) & (ry >= 0) & (rx < rl.shape[1]) & (ry < rl.shape[0]))
    xi = np.rint(np.clip(rx, 0, rl.shape[1] - 1)).astype(int)
    yi = np.rint(np.clip(ry, 0, rl.shape[0] - 1)).astype(int)
    target_hair = (rl == PROFILE['owner']) & (ra >= .5)
    same = valid & target_hair[yi, xi]
    distance = distance_transform_edt(~target_hair)[yi, xi]
    measured = band & valid
    fraction = float(same[measured].mean()) if measured.any() else 0.
    q90 = float(np.quantile(distance[measured], .9)) if measured.any() else float('inf')
    registration.update({'hair_anchor_band_pixels': int(measured.sum()),
                         'hair_registered_contour_fraction': fraction,
                         'hair_q90_reference_pixels': q90,
                         'hair_anchor_band_native_depth_pixels': max_depth})
    if fraction < PROFILE['minimum_registered_contour_fraction']:
        receipt['reason'] = 'hair_owner_contour_fraction_below_threshold'
        return {'maps': None, 'receipt': receipt}
    if q90 > PROFILE['maximum_registration_error_reference_pixels']:
        receipt['reason'] = 'hair_contour_q90_above_threshold'
        return {'maps': None, 'receipt': receipt}
    maps = {
        'ref_x': np.full(a.shape, -1., np.float32),
        'ref_y': np.full(a.shape, -1., np.float32),
        'confidence': np.zeros(a.shape, np.float32),
        'valid': np.zeros(a.shape, bool),
    }
    eligible = hair & valid
    maps['ref_x'][eligible] = rx[eligible]
    maps['ref_y'][eligible] = ry[eligible]
    maps['confidence'][eligible] = np.float32(max(.2, min(.95, 1. - q90 / 8.)))
    maps['valid'][eligible] = True
    receipt.update(state='qualified', changed_pixels=0,
                   fallback_policy='verified_face_affine_plus_alpha_owner_contour',
                   fallback_owner=PROFILE['owner'])
    return {'maps': maps, 'receipt': receipt, 'source_hair_depth': source_hair_depth}


def _empty(reason):
    return {'maps': None, 'receipt': {'schema': SCHEMA, 'profile': dict(PROFILE),
        'binding': {}, 'registration': {'state': 'rejected', 'reason': reason},
        'state': 'rejected', 'changed_pixels': 0, 'source_geometry_changed': False,
        'source_alpha_changed': False, 'reference_texture_imported': False,
        'background_is_light_evidence': False, 'reason': reason}}


def validate(receipt, *, binding):
    if receipt.get('schema') != SCHEMA or receipt.get('binding') != binding:
        raise ValueError('hair fallback registration binding mismatch')
    if receipt.get('state') != 'qualified':
        raise ValueError('hair fallback registration is not qualified')
    row = receipt.get('registration', {})
    if row.get('hair_registered_contour_fraction', 0.) < PROFILE['minimum_registered_contour_fraction']:
        raise ValueError('hair fallback contour qualification missing')
    if row.get('hair_q90_reference_pixels', float('inf')) > PROFILE['maximum_registration_error_reference_pixels']:
        raise ValueError('hair fallback contour error exceeds profile')
    return True
