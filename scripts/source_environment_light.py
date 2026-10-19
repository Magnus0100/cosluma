"""Bounded background-to-subject light response for native composition.

The generated/reference background owns the local light field. This module
transfers only low-frequency illumination and colour spill to the original
foreground RGB; it never copies candidate subject pixels or changes alpha.
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
    'minimum_gain': 1.0,
    'maximum_gain': 1.60,
    'edge_alpha_floor': .02,
}
LUMA = np.asarray([.2126, .7152, .0722], np.float32)


def _resize(value, size, *, nearest=False):
    import cv2
    return cv2.resize(value, tuple(size), interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)


def estimate(background, validity, alpha):
    """Estimate a source-canvas gain field from the final background plate."""
    background, validity, alpha = map(lambda value: np.asarray(value, np.float32),
                                      (background, validity, alpha))
    if (background.ndim != 3 or background.shape[-1] != 3
            or validity.shape != background.shape[:2] or alpha.shape != validity.shape
            or not all(np.isfinite(value).all() for value in (background, validity, alpha))
            or background.min() < 0 or background.max() > 1
            or validity.min() < 0 or validity.max() > 1
            or alpha.min() < 0 or alpha.max() > 1):
        raise ValueError('Environment light needs finite bounded background, validity and alpha')
    h, w = validity.shape
    factor = min(1., CONFIG['analysis_max_edge']/max(h, w))
    proxy_size = (max(1, round(w*factor)), max(1, round(h*factor)))
    bg = _resize(background, proxy_size)
    valid = _resize(validity, proxy_size)
    sigma = max(2., max(proxy_size)*CONFIG['sigma_fraction'])
    support = gaussian_filter(valid, sigma, mode='nearest')
    numerator = gaussian_filter(bg*valid[..., None], (sigma, sigma, 0), mode='nearest')
    environment = numerator/np.maximum(support[..., None], 1e-5)
    luma = environment@LUMA
    eligible = ((support >= CONFIG['minimum_support'])
                & (luma >= CONFIG['minimum_luma']))
    report = {'schema': SCHEMA, 'config': dict(CONFIG), 'proxy_size': list(proxy_size),
              'source_size': [w, h], 'sigma': float(sigma), 'eligible_pixels': int(eligible.sum()),
              'applied': False, 'visual_acceptance': False,
              'candidate_subject_RGB_used': False, 'alpha_modified': False}
    if not eligible.any():
        return np.ones((h, w, 3), np.float32), report
    baseline = np.median(environment[eligible], axis=0).astype(np.float32)
    baseline_luma = max(float(baseline@LUMA), CONFIG['minimum_luma'])
    exposure = np.maximum(np.log2(luma/ baseline_luma), 0)
    exposure = np.clip(exposure, 0, np.log2(CONFIG['maximum_gain']))
    base_chroma = baseline/ max(baseline_luma, .02)
    chroma = np.log2(np.clip((environment/np.maximum(luma[..., None], .02))/base_chroma, .5, 2.))
    chroma_dose = np.clip(exposure/.18, 0, 1)[..., None]
    log_gain = exposure[..., None] + .3*chroma*chroma_dose
    log_gain = np.maximum(log_gain, 0)*np.float32(CONFIG['dose'])
    support_dose = np.clip((support-CONFIG['minimum_support'])/.12, 0, 1)
    log_gain *= support_dose[..., None]
    gain_proxy = np.exp2(log_gain).astype(np.float32)
    gain = _resize(gain_proxy, (w, h))
    gain = np.clip(gain, CONFIG['minimum_gain'], CONFIG['maximum_gain']).astype(np.float32)
    # A uniform background should be an exact no-op.  Without this snap the
    # float32 resize/median path can produce a one-ulp gain above one and
    # invalidate native exact-replay contracts on otherwise unchanged scenes.
    gain[np.abs(gain-1) <= 1e-6] = 1.
    active = alpha >= CONFIG['edge_alpha_floor']
    active_gain = gain[active]
    report.update({'baseline_linear_rgb': baseline.tolist(),
                   'gain_q10_50_90': np.quantile(active_gain, [.1, .5, .9], axis=0).tolist() if active_gain.size else [[1., 1., 1.]]*3,
                   'gain_min_max': [active_gain.min(axis=0).tolist(), active_gain.max(axis=0).tolist()] if active_gain.size else [[1., 1., 1.], [1., 1., 1.]],
                   'changed_pixels': int(np.count_nonzero(np.any(np.abs(gain-1) > 1e-6, axis=-1) & active)),
                   'field_source': 'validity_weighted_final_background_low_frequency',
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
