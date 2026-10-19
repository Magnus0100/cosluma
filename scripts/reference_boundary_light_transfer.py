"""Bounded low-frequency hair illumination with unchanged source coverage.

Reference RGB supplies a normalized lighting envelope, never individual hair
strands. Observation confidence is sampled with its associated response; it
does not become a second nearest-neighbour application boundary.
"""
from __future__ import annotations
import hashlib
import time
import numpy as np
from scipy.ndimage import map_coordinates
from native_cuda_ndimage import distance_transform_edt
from native_cuda_filters import gaussian_filter, cuda_filter_eligible, filter_execution

PROFILE = {'id': 'reference_hair_boundary_continuous_illumination_v3', 'owners': (4,),
    'opaque_alpha': .95, 'partial_reference_alpha': .2, 'partial_reference_support': .6,
    'core_depth_reference_pixels': 18., 'core_sigma_reference_pixels': 20.,
    'maximum_depth_reference_pixels': 18., 'depth_taper_reference_pixels': 6.,
    'minimum_core_mass': .004, 'maximum_registration_error_reference_pixels': 8.,
    'minimum_registered_contour_fraction': .7,
    'response_noise_floor': .003, 'response_transition': .012,
    'illumination_sigma_reference_pixels': 3., 'core_transition_reference_pixels': 3.,
    'source_owner_uncertainty_proxy_pixels': 2., 'source_partial_alpha_minimum': .2,
    'validity_zero': .8, 'identity_protection_zero': .5,
    'maximum_local_exposure_EV': 1.5,
    'gamut_headroom_regularizer': 1e-6,
    'observation_policy': 'associated_normalized_low_frequency_RGB.v1',
    'source_detail_policy': 'source_pixels_only_no_reference_strands_no_detail_boost',
    'dose_policy': 'luminance_tanh_then_common_RGB_soft_headroom.v1'}
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


def _contour(alpha, sigma=1., *, inside_depth=None):
    inside = alpha >= .5
    if inside_depth is None:
        interior = distance_transform_edt(inside)
    else:
        from boundary_light_geometry_context import ExactMaskDistance
        if not isinstance(inside_depth, ExactMaskDistance):
            raise TypeError('Boundary contour requires a live exact mask distance')
        interior = inside_depth.for_mask(inside)
    depth = (interior-distance_transform_edt(~inside)).astype(np.float32)
    ny, nx = np.gradient(gaussian_filter(depth, sigma)); norm = np.maximum(np.hypot(nx, ny), 1e-5)
    return depth, nx/norm, ny/norm


def _low(values, domain, sigma, *, weight=None):
    if weight is None:
        weight = gaussian_filter(domain.astype(np.float32), sigma)
    values = np.where(domain[..., None], values, 0)
    if cuda_filter_eligible(values, (sigma, sigma, 0)):
        values = gaussian_filter(values, (sigma, sigma, 0))
    elif domain.size >= 1_000_000:
        # The colour axis has zero sigma: its channels are independent.
        # Keep SciPy's same spatial pass order and float32 intermediates;
        # bound large native filters with the existing two-worker executor.
        from runtime_resources import ordered_native_map
        filtered = np.empty_like(values)
        def channel(index):
            return index, gaussian_filter(values[..., index], sigma)
        for index, result in ordered_native_map(channel, range(values.shape[-1])):
            filtered[..., index] = result
        values = filtered
    else:
        values = gaussian_filter(values, (sigma, sigma, 0))
    return values/np.maximum(weight[..., None], 1e-6)


def _affine(maps, labels, owner=4, anchor_mask=None):
    valid = (labels == owner)&np.asarray(maps['valid'], bool)&(maps['confidence'] >= .2)
    if anchor_mask is not None:
        anchor_mask = np.asarray(anchor_mask, bool)
        if anchor_mask.shape != valid.shape:
            raise ValueError('registration anchor mask shape mismatch')
        # Registration is certified from the observed source boundary.  A
        # broad semantic interior can contain non-rigid correspondence
        # residuals that are irrelevant to the boundary light being measured.
        valid &= anchor_mask
    y, x = np.where(valid)
    if len(x) < 48:
        return None, {'state': 'rejected', 'reason': 'insufficient_interior_registration_anchors'}
    take = np.arange(0, len(x), max(1, len(x)//4096))
    X = np.column_stack((x[take], y[take], np.ones(len(take))))
    Y = np.column_stack((maps['ref_x'][valid][take], maps['ref_y'][valid][take]))
    if np.linalg.matrix_rank(X) < 3:
        return None, {'state': 'rejected', 'reason': 'collinear_registration_anchors'}
    max_error = float(PROFILE['maximum_registration_error_reference_pixels'])
    affine = np.linalg.lstsq(X, Y, rcond=None)[0]
    for _ in range(5):
        residual = np.linalg.norm(X@affine-Y, axis=1)
        # Use the declared registration budget for robust fitting.  The old
        # literal 3 biased the fit toward a small inlier core and could push
        # the measured q90 above the same profile's 8-pixel acceptance limit.
        weight = np.minimum(1, max_error/np.maximum(residual, 1e-4))
        affine = np.linalg.lstsq(X*weight[:, None], Y*weight[:, None], rcond=None)[0]
    singular = np.linalg.svd(affine[:2], compute_uv=False)
    q90 = float(np.quantile(np.linalg.norm(X@affine-Y, axis=1), .9))
    # Keep the acceptance rule in one place: PROFILE is part of the frozen
    # producer receipt and declares the reference-pixel registration budget.
    # The old literal ``3`` silently overrode that budget, rejecting valid
    # source/reference pairs whose measured error was inside the documented
    # 8-pixel safety limit.
    ok = (np.linalg.det(affine[:2]) > 0 and singular.min() > 0
          and singular.max()/singular.min() < 2 and q90 <= max_error)
    return (affine if ok else None), {'state': 'qualified' if ok else 'rejected',
        'anchor_pixels': int(len(x)), 'q90_anchor_fit_error_reference_pixels': q90,
        'anchor_policy': 'source_alpha_boundary' if anchor_mask is not None else 'owner_semantic_region',
        'affine': affine.tolist(), 'boundary_alignment_accuracy_claimed': False}


def _smooth(value):
    value = np.clip(value, 0, 1)
    return value*value*(3-2*value)


def _weighted_low(rgb, weight, sigma):
    mass = gaussian_filter(weight, sigma)
    return _low(rgb*weight[..., None], weight > 0, sigma, weight=mass), mass


def _observe(rgb, alpha, labels, scale, support=None, *, depth=None, associated=True):
    """Return associated illumination, confidence and the observed domain.

    Filter RGB before taking its positive residual. Rectifying individual
    reference strands before filtering would retain a checkerboard's positive
    bias and transplant texture as light. Only the correction field is filtered.
    """
    if depth is None:
        depth, _, _ = _contour(alpha, scale)
    hair = labels == 4
    opaque_alpha = PROFILE['opaque_alpha']
    partial_alpha = PROFILE['partial_reference_alpha']
    partial_support = PROFILE['partial_reference_support']
    opaque = _smooth((alpha-(opaque_alpha-.05))/.05)*hair
    evidence = opaque
    if support is not None:
        partial = (_smooth((alpha-partial_alpha)/(opaque_alpha-partial_alpha))
                   *_smooth((support-partial_support)/(1-partial_support))*hair)
        evidence = opaque+(1-opaque)*partial
    core = opaque*_smooth((depth-PROFILE['core_depth_reference_pixels']*scale)
                         /(PROFILE['core_transition_reference_pixels']*scale))
    if not core.any():
        return np.zeros_like(rgb), np.zeros_like(alpha), evidence > 0
    baseline, mass = _weighted_low(rgb, core, PROFILE['core_sigma_reference_pixels']*scale)
    lighting, _ = _weighted_low(rgb, evidence, PROFILE['illumination_sigma_reference_pixels']*scale)
    response = np.maximum(lighting-baseline, 0)
    taper = _smooth((PROFILE['maximum_depth_reference_pixels']*scale-depth)
                    /(PROFILE['depth_taper_reference_pixels']*scale))
    minimum_mass = PROFILE['minimum_core_mass']
    confidence = evidence*_smooth((mass-minimum_mass)/minimum_mass)*taper
    # Source subtraction measures its existing light even where the *reference
    # boundary application's* depth confidence reaches zero. Dividing an
    # associated source observation and then special-casing confidence==0
    # would reintroduce a jump at that unrelated support edge.
    signal = response*confidence[..., None] if associated else response
    return signal.astype(np.float32), confidence.astype(np.float32), evidence > 0


def _sample_observation(associated, confidence, x, y):
    weight = _sample(confidence, x, y)
    signal = _sample(associated, x, y)
    estimate = np.divide(signal, weight[..., None], out=np.zeros_like(signal), where=weight[..., None] > 0)
    return estimate, weight


def _source_support(labels, alpha, validity, protection, uncertainty_pixels):
    if not np.isfinite(uncertainty_pixels) or uncertainty_pixels <= 0:
        raise ValueError('Source owner uncertainty must be finite and positive')
    # The coarse semantic edge is an uncertain inter-material boundary, not a
    # measured native hair silhouette. Fade exclusively inside its owner and
    # leave every garment/identity pixel exactly untouched. One-pixel islands
    # have no certain interior and deliberately receive no increment.
    hair = labels == 4
    distance = distance_transform_edt(np.pad(hair, 1))[1:-1, 1:-1].astype(np.float32)
    owner = _smooth((distance-1)/uncertainty_pixels)*hair
    minimum_alpha = PROFILE['source_partial_alpha_minimum']
    minimum_validity = PROFILE['validity_zero']
    maximum_protection = PROFILE['identity_protection_zero']
    return (owner*_smooth((alpha-minimum_alpha)/(PROFILE['opaque_alpha']-minimum_alpha))
            *_smooth((validity-minimum_validity)/(1-minimum_validity))
            *_smooth((maximum_protection-protection)/maximum_protection)).astype(np.float32)


def _bounded_dose(p, alpha, dose):
    """One RGB scale for local EV and gamut, with no channel clipping.

    1.5 EV is a conservative artistic *additional* hair-light budget after
    T/B/S, not a recovered radiometric exposure. Tanh is identity to first
    order for weak evidence and asymptotes to this cap. A common RGB soft
    headroom scale preserves the evidence direction as any channel saturates.
    """
    luminance, increment = p@LUMA, dose@LUMA
    budget = luminance*np.float32(np.exp2(PROFILE['maximum_local_exposure_EV'])-1)
    ratio = np.divide(increment, budget, out=np.zeros_like(increment), where=budget > 0)
    limited = budget*np.tanh(ratio)
    exposure_scale = np.divide(limited, increment, out=np.zeros_like(increment), where=increment > 0)
    bounded = dose*exposure_scale[..., None]
    headroom = np.maximum(alpha[..., None]-p, 0)
    # Include zero-dose channels as well. Skipping them would be discontinuous
    # at an exactly saturated channel when its requested dose moves 0 -> eps.
    denominator = headroom+bounded+alpha[..., None]*PROFILE['gamut_headroom_regularizer']
    channel_scale = np.divide(headroom, denominator, out=np.zeros_like(bounded), where=denominator > 0)
    return (bounded*np.min(channel_scale, axis=-1)[..., None]).astype(np.float32)


def prepare_boundary_light(current_p, alpha, reference_rgb, reference_alpha,
                           source_labels, reference_labels, maps, *, validity=None,
                           identity_protection=None, reference_fp=None, reference_fb_support=None,
                           registration_anchor_mask=None, contour_qualification_mask=None,
                           source_inside_depth=None, source_owner_uncertainty_pixels=1.):
    started = time.perf_counter()
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
    if contour_qualification_mask is not None:
        contour_qualification_mask = np.asarray(contour_qualification_mask, bool)
        if contour_qualification_mask.shape != shape:
            raise ValueError('contour qualification mask shape mismatch')
    if (reference_fp is None) != (reference_fb_support is None):
        raise ValueError('partial reference observations require both Fp and F/B support')
    support = None
    if reference_fp is not None:
        fp = _unit(reference_fp, ref.shape, 'reference_fp')
        support = _unit(reference_fb_support, ra.shape, 'reference_fb_support').copy()
        bounded = np.all(fp <= ra[..., None]+2e-6, axis=-1); support[~bounded] = 0
        ref = np.divide(fp, ra[..., None], out=np.zeros_like(fp), where=(ra[..., None] >= .2)&bounded[..., None])
    delta = np.zeros_like(p); observed = np.zeros_like(ref)
    affine, registration = _affine(maps, sl, anchor_mask=registration_anchor_mask)
    row = {'owner': 4, 'registration': registration, 'changed_pixels': 0}
    if affine is not None:
        scale = float(np.sqrt(np.linalg.det(affine[:2])))
        y, x = np.indices(shape, dtype=np.float32)
        rx = x*affine[0, 0]+y*affine[1, 0]+affine[2, 0]; ry = x*affine[0, 1]+y*affine[1, 1]+affine[2, 1]
        sd, snx, sny = _contour(a, inside_depth=source_inside_depth); rd, rnx, rny = _contour(ra)
        same_owner = _sample(rl, rx, ry, nearest=True) == 4
        if contour_qualification_mask is None:
            error = _sample(rd, rx, ry)-sd*scale
        else:
            # A verified fallback owns an explicit hair contour band.  The
            # full-person alpha contour is a different geometry and can move
            # in a generated reference; measure the same owner on both sides
            # while retaining the same 8 px error and 0.7 coverage gates.
            target_hair = (rl == 4) & (ra >= .5)
            error = _sample(distance_transform_edt(~target_hair), rx, ry)
        contour = ((sl == 4)&(a >= .5)&(sd <= 2/scale)
                   if contour_qualification_mask is None else contour_qualification_mask & (sl == 4) & (a >= .5))
        agrees = same_owner&(np.abs(error) <= 8)
        fraction = float(agrees[contour].mean()) if contour.any() else 0.; row['registered_contour_fraction'] = fraction
        if fraction >= .7:
            target, qualified, _ = _observe(ref, ra, rl, 1., support, depth=rd)
            observed[:] = target
            # Both responses are nonnegative. A zero reference response can
            # never produce a missing highlight, so do not filter the native
            # source (potentially millions of pixels at a large sigma).
            if np.any(target != 0):
                source = np.divide(p, a[..., None], out=np.zeros_like(p), where=a[..., None] >= .2)
                current, _, opaque = _observe(source, a, sl, 1/scale,
                    support=np.ones_like(a), depth=sd, associated=False)
            else:
                opaque = None
            if opaque is not None and opaque.any():
                target_estimate, observation_confidence = _sample_observation(target, qualified, rx, ry)
                missing = np.maximum(target_estimate-current, 0)
                strength = _smooth((missing@LUMA-PROFILE['response_noise_floor'])/PROFILE['response_transition'])
                nx, ny = _sample(rnx, rx, ry), _sample(rny, rx, ry)
                sx = snx*affine[0, 0]+sny*affine[1, 0]; sy = snx*affine[0, 1]+sny*affine[1, 1]
                cosine = (sx*nx+sy*ny)/np.maximum(np.hypot(sx, sy), 1e-5)
                confidence = observation_confidence*_smooth((8-np.abs(error))/4)
                confidence *= _smooth((cosine-.25)/.5)
                allowed = _source_support(sl, a, v, protection, source_owner_uncertainty_pixels)
                dose = a[..., None]*missing*(strength*confidence*allowed)[..., None]
                delta = _bounded_dose(p, a, dose)
        else: row['reason'] = 'coarse_registration_does_not_match_hair_contour'
    changed = np.any(delta != 0, axis=-1); row['changed_pixels'] = int(changed.sum())
    before_luma, after_luma = p@LUMA, (p+delta)@LUMA
    ev = np.log2(np.divide(after_luma, before_luma, out=np.ones_like(before_luma), where=before_luma > 0))
    receipt = {'schema': 'reference_boundary_light_transfer.v3', 'profile': dict(PROFILE),
        'elapsed_seconds': time.perf_counter()-started,
        'source_owner_uncertainty_native_pixels': float(source_owner_uncertainty_pixels),
        'maximum_applied_exposure_EV': float(np.max(ev)),
        'maximum_associated_RGB_excess': float(np.maximum(p+delta-a[..., None], 0).max()),
        'spatial_filter_execution': filter_execution(),
        'state': 'prepared' if changed.any() else 'evidence_limited', 'components': [row],
        'changed_pixels': int(changed.sum()), 'clothing_enabled': False,
        'reference_person_is_light_evidence': True, 'background_is_light_evidence': False,
        'reference_RGB_appearance_is_donor': False, 'reference_low_frequency_RGB_is_light_evidence': True,
        'reference_texture_imported': False, 'source_texture_is_output_owner': True,
        'source_alpha_changed': False, 'source_geometry_changed': False,
        'source_microtexture_preserved_in_changed_region': False, 'partial_alpha_physical_response_certified': False,
        'physical_relighting_claimed': False, 'working_domain': 'display_linear_sRGB', 'visual_acceptance': False,
        'binding': {'current_p': _sha(p), 'alpha': _sha(a), 'reference': _sha(ref),
            'reference_alpha': _sha(ra), 'source_labels': _sha(sl), 'reference_labels': _sha(rl)}}
    return {'delta_p': delta, 'after_p': p+delta, 'alpha': a.copy(), 'support': changed,
        'reference_response': observed, 'receipt': receipt}
