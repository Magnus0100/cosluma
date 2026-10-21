"""Bounded background-to-subject light response for native composition.

The generated/reference background owns the local light field. This module
transfers only low-frequency illumination and colour spill to the original
foreground RGB; it never copies candidate subject pixels or changes alpha.

The response is two-way and joint. A final plate that is darker than the
photograph it replaces has to darken the subject exactly as a brighter one
lifts it: the reviewed response could only brighten, which is the opposite of
the direction a scene balance applies, so the subject followed the environment
in one direction only and the two drifted apart. When the photographed
environment is supplied the solve measures the environment's own change and
transfers that, which keeps the subject and its surroundings solved together
instead of normalising the change away; without it the local field under the
subject is used, in both directions.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

SCHEMA = 'source_environment_light.v1'
CONFIG = {
    'analysis_max_edge': 512,
    'sigma_fraction': .02,
    'minimum_support': .04,
    'minimum_luma': .005,
    'dose': .65,
    # Symmetric bound. A floor of exactly one forced every response to
    # brighten, which cannot follow a scene balance that darkens; the same
    # +-0.678 EV (1.6x) magnitude is the reviewed bound in both directions.
    'minimum_gain': 1/1.6,
    'maximum_gain': 1.60,
    'gain_ev_bounds': [-.678071905112638, .678071905112638],
    'edge_alpha_floor': .02,
}
LUMA = np.asarray([.2126, .7152, .0722], np.float32)


def _resize(value, size, *, nearest=False):
    import cv2
    return cv2.resize(value, tuple(size), interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)


def _field(plate, valid, sigma, *, mask=None):
    """A validity-weighted low-frequency field, optionally measured on a subset."""
    weight = valid if mask is None else (valid*np.asarray(mask, np.float32))
    support = gaussian_filter(weight, sigma, mode="nearest")
    numerator = gaussian_filter(plate*weight[..., None], (sigma, sigma, 0), mode="nearest")
    return numerator/np.maximum(support[..., None], 1e-5), support


def estimate(background, validity, alpha, *, source_environment=None):
    """Estimate a two-way source-canvas gain field from the final background plate.

    ``source_environment`` is the photographed environment the final plate
    replaces. It is optional because the composition stage may not carry it;
    when it is present the gain is the environment's own measured change, so a
    darker environment darkens the subject and a brighter one lifts it. Without
    it the transfer stays a local field around the subject, still solved in
    both directions.
    """
    background, validity, alpha = map(lambda value: np.asarray(value, np.float32),
                                      (background, validity, alpha))
    plates = [background] if source_environment is None else [background, np.asarray(source_environment, np.float32)]
    if (any(plate.ndim != 3 or plate.shape[-1] != 3 or plate.shape != background.shape for plate in plates)
            or validity.shape != background.shape[:2] or alpha.shape != validity.shape
            or not all(np.isfinite(value).all() for value in (*plates, validity, alpha))
            or any(plate.min() < 0 or plate.max() > 1 for plate in plates)
            or validity.min() < 0 or validity.max() > 1
            or alpha.min() < 0 or alpha.max() > 1):
        raise ValueError('Environment light needs finite bounded background, validity and alpha')
    h, w = validity.shape
    factor = min(1., CONFIG['analysis_max_edge']/max(h, w))
    proxy_size = (max(1, round(w*factor)), max(1, round(h*factor)))
    bg = _resize(background, proxy_size)
    valid = _resize(validity, proxy_size)
    sigma = max(2., max(proxy_size)*CONFIG['sigma_fraction'])
    environment, support = _field(bg, valid, sigma)
    luma = environment@LUMA
    # The environment written back around the subject, never the subject's own
    # footprint: the design stage fills that, so a response measured there would
    # be solved against an infill instead of against the surroundings.
    sky = _resize((alpha < CONFIG['edge_alpha_floor']).astype(np.float32), proxy_size, nearest=True) > .5
    eligible = (support >= CONFIG['minimum_support']) & (luma >= CONFIG['minimum_luma']) & sky
    limit = np.float32(np.log2(CONFIG['maximum_gain']))
    report = {'schema': SCHEMA, 'config': dict(CONFIG), 'proxy_size': list(proxy_size),
              'source_size': [w, h], 'sigma': float(sigma), 'eligible_pixels': int(eligible.sum()),
              'applied': False, 'visual_acceptance': False, 'directions': 'brighten_and_darken',
              'candidate_subject_RGB_used': False, 'alpha_modified': False, 'bypass_reason': None}
    log_gain = None
    if source_environment is None:
        if not eligible.any():
            report['bypass_reason'] = 'no_eligible_environment'
            return np.ones((h, w, 3), np.float32), report
        baseline = np.median(environment[eligible], axis=0).astype(np.float32)
        baseline_luma = max(float(baseline@LUMA), CONFIG['minimum_luma'])
        exposure = np.clip(np.log2(luma/baseline_luma), -limit, limit)
        base_chroma = baseline/max(baseline_luma, .02)
        chroma = np.log2(np.clip((environment/np.maximum(luma[..., None], .02))/base_chroma, .5, 2.))
        # The colour spill is dosed by the size of the change, whichever way it goes.
        chroma_dose = np.clip(np.abs(exposure)/.18, 0, 1)[..., None]
        log_gain = exposure[..., None] + .3*chroma*chroma_dose
        report.update(solve='local_environment_field_two_way', field_source='validity_weighted_final_background_low_frequency',
                      baseline_linear_rgb=baseline.tolist())
    else:
        old = _resize(plates[1], proxy_size)
        old_environment, old_support = _field(old, valid, sigma)
        old_luma = old_environment@LUMA
        both = eligible & (old_support >= np.float32(CONFIG['minimum_support'])) & (old_luma >= np.float32(CONFIG['minimum_luma']))
        if not both.any():
            report['bypass_reason'] = 'no_eligible_environment_on_both_plates'
            return np.ones((h, w, 3), np.float32), report
        change = np.log2(np.maximum(environment, 1e-6)/np.maximum(old_environment, 1e-6))
        # Under the subject the photographed environment is an infill of its own,
        # so the change there is taken from the surroundings instead.
        around, around_support = _field(change, valid, sigma, mask=sky.astype(np.float32))
        # The relation D1 measures is the subject's against the environment as a
        # whole, so the transfer is anchored on the environment's own measured
        # change and keeps only the local variation around the subject on top of
        # it. Without the anchor the solve answers very locally (the pixels right
        # next to the subject), which cannot carry a whole-scene change; without
        # the local term it would stop carrying the light the subject actually
        # stands in.
        global_change = np.median(change[both], axis=0).astype(np.float32)
        bias = global_change-np.median(around[both], axis=0).astype(np.float32)
        change = np.where(sky[..., None], change, around+bias)
        # The measured change is a relation, not a stylistic dose: it is applied
        # once, at full strength, so the subject keeps the light relation it had
        # with the photographed environment. The local-field branch keeps the
        # reviewed dose because there the response is an estimate, not a
        # measurement.
        log_gain = np.clip(change, -limit, limit)
        report['joint_dose'] = 1.
        report['clamped_pixels'] = int(np.count_nonzero(np.any(np.abs(change) > limit, axis=-1)))
        report.update(environment_change_ev_global=[round(float(v), 4) for v in global_change],
                      environment_change_ev_around=[round(float(v), 4) for v in np.median(around[both], axis=0)],
                      joint_bias_ev=[round(float(v), 4) for v in bias],
                      solve='joint_photographed_environment_change_two_way',
                      field_source='photographed_to_final_environment_change',
                      environment_change_ev_median=[round(float(v), 4) for v in np.median(change[both], axis=0)],
                      environment_change_ev_range=[[round(float(v), 4) for v in np.min(change[both], axis=0)],
                                                   [round(float(v), 4) for v in np.max(change[both], axis=0)]],
                      eligible_pixels=int(both.sum()))
    if source_environment is None:
        log_gain = np.clip(log_gain, -limit, limit)*np.float32(CONFIG['dose'])
        support_dose = np.clip((support-CONFIG['minimum_support'])/.12, 0, 1)
    else:
        log_gain = np.clip(log_gain, -limit, limit)
        # The change is a whole-environment measurement, so its confidence is the
        # environment's own coverage and not the support of the pixels the
        # subject occupies: a local support dies inside a large subject and would
        # silently switch the response off exactly where it is needed.
        confidence = float(np.clip((float(eligible.mean())-CONFIG['minimum_support'])/.12, 0., 1.))
        support_dose = np.float32(confidence)
        report['environment_confidence'] = round(confidence, 4)
    log_gain = log_gain*support_dose[..., None]
    gain_proxy = np.exp2(log_gain).astype(np.float32)
    gain = _resize(gain_proxy, (w, h))
    gain = np.clip(gain, CONFIG['minimum_gain'], CONFIG['maximum_gain']).astype(np.float32)
    # A uniform background should be an exact no-op.  Without this snap the
    # float32 resize/median path can produce a one-ulp gain above one and
    # invalidate native exact-replay contracts on otherwise unchanged scenes.
    gain[np.abs(gain-1) <= 1e-6] = 1.
    active = alpha >= CONFIG['edge_alpha_floor']
    active_gain = gain[active]
    report.update({'gain_q10_50_90': np.quantile(active_gain, [.1, .5, .9], axis=0).tolist() if active_gain.size else [[1., 1., 1.]]*3,
                   'gain_min_max': [active_gain.min(axis=0).tolist(), active_gain.max(axis=0).tolist()] if active_gain.size else [[1., 1., 1.], [1., 1., 1.]],
                   'changed_pixels': int(np.count_nonzero(np.any(np.abs(gain-1) > 1e-6, axis=-1) & active)),
                   'darkened_pixels': int(np.count_nonzero(np.any(gain < 1.-np.float32(1e-6), axis=-1) & active)),
                   'brightened_pixels': int(np.count_nonzero(np.any(gain > 1.+np.float32(1e-6), axis=-1) & active)),
                   'response_scope': 'source_foreground_only',
                   'applied': bool(np.any(np.abs(gain-1) > 1e-6))})
    return gain, report


def apply(foreground, gain, alpha, *, strength=1.):
    """Apply the gain to premultiplied source RGB without changing coverage."""
    foreground, gain, alpha = map(lambda value: np.asarray(value, np.float32),
                                   (foreground, gain, alpha))
    if (foreground.ndim != 3 or foreground.shape[-1] != 3
            or gain.shape != foreground.shape or alpha.shape != foreground.shape[:2]
            or not np.isfinite(foreground).all() or not np.isfinite(gain).all()
            or not np.isfinite(alpha).all() or not 0 <= strength <= 1
            or foreground.min() < 0 or gain.min() < 0
            or alpha.min() < 0 or alpha.max() > 1):
        raise ValueError('Environment light apply inputs are invalid')
    dose = np.clip(alpha, 0, 1)[..., None]*np.float32(strength)
    result = foreground*(1+dose*(gain-1))
    # Keep the premultiplied RGB contract exact at the foreground boundary.
    result = np.minimum(np.clip(result, 0, 1), alpha[..., None])
    return result.astype(np.float32)
