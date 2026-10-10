"""Reference-observed hair boundary response, with unchanged source coverage.

No contour snapping, bright-pixel extension or smoothed reference ribbons.
Clothing is excluded because its artistic control failed. This is local
appearance transfer, not physical relighting or recovered source microtexture.
"""
from __future__ import annotations
import hashlib
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates, median_filter

PROFILE = {'id': 'reference_hair_boundary_observed_response_v2', 'owners': (4,),
    'opaque_alpha': .95, 'partial_reference_alpha': .2, 'partial_reference_support': .6,
    'core_depth_reference_pixels': 18., 'core_sigma_reference_pixels': 20.,
    'maximum_depth_reference_pixels': 18., 'depth_taper_reference_pixels': 6.,
    'minimum_core_mass': .004, 'maximum_registration_error_reference_pixels': 8.,
    'minimum_registered_contour_fraction': .7, 'source_response_extension_reference_pixels': 4.,
    'response_noise_floor': .003, 'response_transition': .012,
    'source_detail_sigma_reference_pixels': 1.6, 'source_detail_exponent': 1.5, 'source_detail_native_median_window': 3}
LUMA = np.asarray([.2126, .7152, .0722], np.float32)


def _unit(value, shape, name):
    value = np.asarray(value, np.float32)
    if value.shape != shape or not np.isfinite(value).all() or np.any((value < 0)|(value > 1)):
        raise ValueError(name+' must be finite [0,1] with the expected shape')
    return value


def _sha(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str((value.shape, value.dtype.str)).encode()+value.tobytes()).hexdigest()


def _sample(values, x, y, *, nearest=False):
    if values.ndim == 3:
        return np.stack([_sample(values[..., c], x, y, nearest=nearest) for c in range(values.shape[-1])], -1)
    return map_coordinates(values, [y, x], order=0 if nearest else 1, mode='constant', cval=0, prefilter=False)


def _contour(alpha, sigma=1.):
    inside = alpha >= .5
    depth = (distance_transform_edt(inside)-distance_transform_edt(~inside)).astype(np.float32)
    ny, nx = np.gradient(gaussian_filter(depth, sigma)); norm = np.maximum(np.hypot(nx, ny), 1e-5)
    return depth, nx/norm, ny/norm


def _low(values, domain, sigma):
    weight = gaussian_filter(domain.astype(np.float32), sigma)
    values = gaussian_filter(np.where(domain[..., None], values, 0), (sigma, sigma, 0))
    return values/np.maximum(weight[..., None], 1e-6)


def _affine(maps, labels, owner=4):
    valid = (labels == owner)&np.asarray(maps['valid'], bool)&(maps['confidence'] >= .2)
    y, x = np.where(valid)
    if len(x) < 48:
        return None, {'state': 'rejected', 'reason': 'insufficient_interior_registration_anchors'}
    take = np.arange(0, len(x), max(1, len(x)//4096))
    X = np.column_stack((x[take], y[take], np.ones(len(take))))
    Y = np.column_stack((maps['ref_x'][valid][take], maps['ref_y'][valid][take]))
    if np.linalg.matrix_rank(X) < 3:
        return None, {'state': 'rejected', 'reason': 'collinear_registration_anchors'}
    affine = np.linalg.lstsq(X, Y, rcond=None)[0]
    for _ in range(5):
        residual = np.linalg.norm(X@affine-Y, axis=1)
        weight = np.minimum(1, 3/np.maximum(residual, 1e-4))
        affine = np.linalg.lstsq(X*weight[:, None], Y*weight[:, None], rcond=None)[0]
    singular = np.linalg.svd(affine[:2], compute_uv=False)
    q90 = float(np.quantile(np.linalg.norm(X@affine-Y, axis=1), .9))
    ok = np.linalg.det(affine[:2]) > 0 and singular.min() > 0 and singular.max()/singular.min() < 2 and q90 <= 3
    return (affine if ok else None), {'state': 'qualified' if ok else 'rejected',
        'anchor_pixels': int(len(x)), 'q90_anchor_fit_error_reference_pixels': q90,
        'affine': affine.tolist(), 'boundary_alignment_accuracy_claimed': False}


def _observe(rgb, alpha, labels, scale, support=None):
    depth, _, _ = _contour(alpha, scale)
    opaque = (alpha >= .95)&(labels == 4)
    domain = opaque if support is None else ((labels == 4)&(alpha >= .2)&(opaque|(support >= .6)))
    core = opaque&(depth >= 18*scale)
    if not core.any():
        return np.zeros_like(rgb), np.zeros_like(alpha), domain
    mass = gaussian_filter(core.astype(np.float32), 20*scale)
    baseline = _low(rgb, core, 20*scale)
    response = np.maximum(rgb-baseline, 0)
    taper = np.clip((18*scale-depth)/(6*scale), 0, 1)
    valid = domain&(mass >= .004)&(depth <= 18*scale)
    response *= (taper*valid)[..., None]
    return response, valid.astype(np.float32), domain


def prepare_boundary_light(current_p, alpha, reference_rgb, reference_alpha,
                           source_labels, reference_labels, maps, *, validity=None,
                           identity_protection=None, reference_fp=None, reference_fb_support=None):
    p = np.asarray(current_p, np.float32)
    if p.ndim != 3 or p.shape[-1] != 3 or min(p.shape[:2]) < 4:
        raise ValueError('current_p must be HWC associated RGB')
    p = _unit(p, p.shape, 'current_p'); shape = p.shape[:2]; a = _unit(alpha, shape, 'alpha')
    if np.any(p > a[..., None]+2e-5): raise ValueError('current_p exceeds source alpha')
    ref = np.asarray(reference_rgb, np.float32)
    if ref.ndim != 3 or ref.shape[-1] != 3 or min(ref.shape[:2]) < 4: raise ValueError('reference must be HWC RGB')
    ref = _unit(ref, ref.shape, 'reference'); ra = _unit(reference_alpha, ref.shape[:2], 'reference_alpha')
    sl, rl = np.asarray(source_labels), np.asarray(reference_labels)
    if sl.shape != shape or rl.shape != ra.shape or sl.dtype.kind not in 'iu' or rl.dtype.kind not in 'iu':
        raise ValueError('independent semantic labels must match their own image')
    for key in ('ref_x', 'ref_y', 'confidence', 'valid'):
        if key not in maps or np.asarray(maps[key]).shape != shape or not np.isfinite(maps[key]).all():
            raise ValueError('invalid registration map: '+key)
    v = np.ones(shape, np.float32) if validity is None else _unit(validity, shape, 'validity')
    protection = np.zeros(shape, np.float32) if identity_protection is None else _unit(identity_protection, shape, 'identity_protection')
    if (reference_fp is None) != (reference_fb_support is None):
        raise ValueError('partial reference observations require both Fp and F/B support')
    support = None
    if reference_fp is not None:
        fp = _unit(reference_fp, ref.shape, 'reference_fp')
        support = _unit(reference_fb_support, ra.shape, 'reference_fb_support').copy()
        bounded = np.all(fp <= ra[..., None]+2e-6, axis=-1); support[~bounded] = 0
        ref = np.divide(fp, ra[..., None], out=np.zeros_like(fp), where=(ra[..., None] >= .2)&bounded[..., None])
    source = np.divide(p, a[..., None], out=np.zeros_like(p), where=a[..., None] >= .95)
    delta = np.zeros_like(p); observed = np.zeros_like(ref)
    affine, registration = _affine(maps, sl); row = {'owner': 4, 'registration': registration, 'changed_pixels': 0}
    if affine is not None:
        scale = float(np.sqrt(np.linalg.det(affine[:2])))
        y, x = np.indices(shape, dtype=np.float32)
        rx = x*affine[0, 0]+y*affine[1, 0]+affine[2, 0]; ry = x*affine[0, 1]+y*affine[1, 1]+affine[2, 1]
        sd, snx, sny = _contour(a); rd, rnx, rny = _contour(ra)
        error = _sample(rd, rx, ry)-sd*scale; same_owner = _sample(rl, rx, ry, nearest=True) == 4
        contour = (sl == 4)&(a >= .5)&(sd <= 2/scale)
        agrees = same_owner&(np.abs(error) <= 8)
        fraction = float(agrees[contour].mean()) if contour.any() else 0.; row['registered_contour_fraction'] = fraction
        if fraction >= .7:
            target, qualified, _ = _observe(ref, ra, rl, 1., support)
            current, _, opaque = _observe(source, a, sl, 1/scale); observed[:] = target
            if opaque.any():
                gap, ind = distance_transform_edt(~opaque, return_indices=True)
                current = current[ind[0], ind[1]]; current[gap > 4/scale] = 0
                missing = np.maximum(_sample(target, rx, ry)-current, 0)
                strength = np.clip((missing@LUMA-.003)/.012, 0, 1)
                nx, ny = _sample(rnx, rx, ry), _sample(rny, rx, ry)
                sx = snx*affine[0, 0]+sny*affine[1, 0]; sy = snx*affine[0, 1]+sny*affine[1, 1]
                cosine = (sx*nx+sy*ny)/np.maximum(np.hypot(sx, sy), 1e-5)
                confidence = _sample(qualified, rx, ry)*np.clip((8-np.abs(error))/4, 0, 1)
                confidence *= np.clip((cosine-.25)/.5, 0, 1)*same_owner
                allowed = (sl == 4)&(a > 0)&(v >= .8)&(protection < .5)
                luminance = p@LUMA; sigma = 1.6/scale
                observed_luminance = np.where(a >= .95, median_filter(luminance, size=3), luminance)
                denominator = gaussian_filter(luminance, sigma); coverage = gaussian_filter(a, sigma)
                detail = np.clip(np.divide(observed_luminance*coverage, np.maximum(a*denominator, 1e-5)), 0, 2)**1.5
                dose = a[..., None]*missing*detail[..., None]*(strength*confidence*allowed)[..., None]
                delta = np.minimum(dose, np.maximum(a[..., None]-p, 0)).astype(np.float32)
        else: row['reason'] = 'coarse_registration_does_not_match_hair_contour'
    changed = np.any(delta != 0, axis=-1); row['changed_pixels'] = int(changed.sum())
    receipt = {'schema': 'reference_boundary_light_transfer.v2', 'profile': dict(PROFILE),
        'state': 'prepared' if changed.any() else 'evidence_limited', 'components': [row],
        'changed_pixels': int(changed.sum()), 'clothing_enabled': False,
        'reference_person_is_light_evidence': True, 'background_is_light_evidence': False,
        'reference_RGB_appearance_is_donor': True, 'source_alpha_changed': False, 'source_geometry_changed': False,
        'source_microtexture_preserved_in_changed_region': False, 'partial_alpha_physical_response_certified': False,
        'physical_relighting_claimed': False, 'working_domain': 'display_linear_sRGB', 'visual_acceptance': False,
        'binding': {'current_p': _sha(p), 'alpha': _sha(a), 'reference': _sha(ref),
            'reference_alpha': _sha(ra), 'source_labels': _sha(sl), 'reference_labels': _sha(rl)}}
    return {'delta_p': delta, 'after_p': p+delta, 'alpha': a.copy(), 'support': changed,
        'reference_response': observed, 'receipt': receipt}
