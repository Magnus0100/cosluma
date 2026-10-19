"""Extract low-frequency relighting evidence from a generated candidate.

The reference image is allowed to describe how the generated scene lights a
person, but its subject pixels are never used as output texture or geometry.
This module turns only a robust, spatially smooth appearance residual into a
bounded source-coordinate gain field.  It is deliberately a preparation
signal for the source-owned foreground, not a second foreground compositor.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, gaussian_filter

from float_appearance import srgb_to_linear
from asset_identity import sha256_file

SCHEMA = 'candidate_subject_light.v1'
LUMA = np.asarray([.2126, .7152, .0722], np.float32)
CONFIG = {
    'analysis_max_edge': 768,
    'lowpass_sigma_fraction': .018,
    'propagation_sigma_fraction': .035,
    'minimum_support': .08,
    'minimum_observation_pixels': 96,
    'minimum_gain': .72,
    'maximum_gain': 1.75,
    'maximum_ev': .65,
    'minimum_ev': -.35,
    'maximum_chroma_log': .14,
    'dose': .85,
    'edge_erosion_pixels': 2,
}


def _as_float(value, name, shape=None):
    result = np.asarray(value, np.float32)
    if shape is not None and result.shape != shape:
        raise ValueError(f'{name} shape differs from the source proxy')
    if not np.isfinite(result).all():
        raise ValueError(f'{name} contains nonfinite values')
    if result.min(initial=0) < 0 or result.max(initial=0) > 1:
        raise ValueError(f'{name} must be bounded to [0,1]')
    return result


def _normalised_blur(values, weight, sigma):
    sigma = max(1.5, float(sigma))
    support = gaussian_filter(weight.astype(np.float32), sigma, mode='nearest')
    if values.ndim == 2:
        numerator = gaussian_filter(values * weight, sigma, mode='nearest')
        return numerator / np.maximum(support, 1e-5), support
    numerator = gaussian_filter(values * weight[..., None], (sigma, sigma, 0), mode='nearest')
    return numerator / np.maximum(support[..., None], 1e-5), support


def _weighted_median(values, weights):
    values = np.asarray(values, np.float32).reshape(-1)
    weights = np.asarray(weights, np.float32).reshape(-1)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return 0.0
    values, weights = values[valid], weights[valid]
    order = np.argsort(values, kind='mergesort')
    values, weights = values[order], weights[order]
    return float(values[np.searchsorted(np.cumsum(weights), weights.sum()*.5, side='left')])


def _validate_mapping(mapping):
    if mapping is None:
        return np.eye(3, dtype=np.float64)
    matrix = np.asarray(mapping, np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-10:
        raise ValueError('Candidate subject light mapping must be an invertible 3x3 matrix')
    return matrix


def _warp(values, matrix, size, *, nearest=False):
    width, height = map(int, size)
    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    border = 0 if values.ndim == 2 else (0, 0, 0)
    return cv2.warpPerspective(np.asarray(values, np.float32), matrix, (width, height),
                               flags=interpolation, borderMode=cv2.BORDER_CONSTANT,
                               borderValue=border)


def load_reference_proxy(reference_path, candidate_mask, target_size, mapping=None):
    """Load candidate RGB and occupancy into the source proxy coordinates.

    The candidate is resized only to the occupancy observation canvas before
    the frozen scene registration is applied.  The returned RGB is evidence
    only; callers must never composite it into the final foreground.
    """
    reference_path = Path(reference_path).resolve()
    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)
    mask = _as_float(candidate_mask, 'candidate_mask')
    mask_h, mask_w = mask.shape
    with Image.open(reference_path) as image:
        rgb = np.asarray(image.convert('RGB').resize((mask_w, mask_h), Image.Resampling.LANCZOS), np.float32)/255
    rgb = srgb_to_linear(rgb)
    matrix = _validate_mapping(mapping)
    warped_rgb = _warp(rgb, matrix, target_size)
    warped_mask = _warp(mask, matrix, target_size, nearest=True)
    validity = _warp(np.ones(mask.shape, np.float32), matrix, target_size, nearest=True)
    return warped_rgb, np.clip(warped_mask, 0, 1), np.clip(validity, 0, 1)


def _read_mask_asset(asset):
    if not isinstance(asset, dict) or not asset.get('path') or not asset.get('sha256'):
        raise ValueError('Candidate occupancy asset identity is required')
    path = Path(asset['path']).resolve()
    if not path.is_file() or sha256_file(path) != asset['sha256']:
        raise ValueError('Candidate occupancy asset bytes changed')
    if path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as resource:
            key = 'values' if 'values' in resource else 'labels' if 'labels' in resource else None
            if key is None:
                raise ValueError('Candidate occupancy NPZ has no values plane')
            mask = np.asarray(resource[key], np.float32)
    else:
        mask = np.asarray(Image.open(path).convert('L'), np.float32)/255
    return _as_float(mask, 'candidate_occupancy')


def estimate_from_reference(reference_path, candidate_mask_asset, source_foreground,
                            source_alpha, *, mapping=None, source_proxy_size=None,
                            expected_reference_sha256=None):
    """Prepare and estimate a candidate-only light field for a native source.

    The reference file and occupancy mask are content-bound inputs.  The
    returned gain is native source-canvas RGB, while the candidate image is
    retained only in the local evidence calculation.
    """
    reference_path = Path(reference_path).resolve()
    if expected_reference_sha256 and sha256_file(reference_path) != expected_reference_sha256:
        raise ValueError('Candidate reference bytes differ from the bound run')
    foreground = np.asarray(source_foreground, np.float32)
    alpha = np.asarray(source_alpha, np.float32)
    if foreground.ndim != 3 or foreground.shape[-1] != 3 or alpha.shape != foreground.shape[:2]:
        raise ValueError('Native source foreground/alpha shapes differ')
    if source_proxy_size is None:
        source_proxy_size = (min(768, foreground.shape[1]), min(768, foreground.shape[0]))
    proxy_w, proxy_h = map(int, source_proxy_size)
    if proxy_w <= 0 or proxy_h <= 0:
        raise ValueError('Source proxy size must be positive')
    mask = _read_mask_asset(candidate_mask_asset)
    candidate_rgb, candidate_mask, mapping_valid = load_reference_proxy(
        reference_path, mask, (proxy_w, proxy_h), mapping=mapping)
    source_alpha_proxy = cv2.resize(alpha, (proxy_w, proxy_h), interpolation=cv2.INTER_AREA)
    source_straight = np.divide(foreground, alpha[..., None], out=np.zeros_like(foreground),
                                where=alpha[..., None] > 1e-5)
    source_straight_proxy = cv2.resize(source_straight, (proxy_w, proxy_h), interpolation=cv2.INTER_AREA)
    candidate_mask = np.minimum(candidate_mask, mapping_valid)
    proxy_gain, report = estimate(candidate_rgb, source_straight_proxy, candidate_mask,
                                  source_alpha_proxy)
    native_gain = cv2.resize(proxy_gain, (foreground.shape[1], foreground.shape[0]),
                             interpolation=cv2.INTER_LINEAR)
    report.update({'reference_sha256': expected_reference_sha256 or sha256_file(reference_path),
                   'candidate_occupancy_sha256': candidate_mask_asset['sha256'],
                   'mapping_used': mapping is not None,
                   'native_size': [foreground.shape[1], foreground.shape[0]],
                   'native_gain_q10_50_90': np.quantile(native_gain.reshape(-1, 3), [.1, .5, .9], axis=0).tolist()})
    return np.clip(native_gain, CONFIG['minimum_gain'], CONFIG['maximum_gain']).astype(np.float32), report


def estimate(candidate_rgb, source_rgb, candidate_mask, source_alpha):
    """Estimate a source-proxy gain field from candidate relighting evidence.

    ``source_rgb`` is the current source-owned appearance endpoint, while
    ``candidate_rgb`` is consumed only through low-frequency residuals inside
    the registered subject overlap.  High-frequency texture, candidate edges,
    and candidate geometry are excluded before the residual is solved.
    """
    candidate_rgb = _as_float(candidate_rgb, 'candidate_rgb')
    source_rgb = _as_float(source_rgb, 'source_rgb', candidate_rgb.shape)
    shape = candidate_rgb.shape[:2]
    candidate_mask = _as_float(candidate_mask, 'candidate_mask', shape)
    source_alpha = _as_float(source_alpha, 'source_alpha', shape)
    if candidate_rgb.ndim != 3 or candidate_rgb.shape[-1] != 3:
        raise ValueError('candidate_rgb must be HWC RGB')
    if source_rgb.ndim != 3 or source_rgb.shape[-1] != 3:
        raise ValueError('source_rgb must be HWC RGB')
    h, w = shape
    factor = min(1., CONFIG['analysis_max_edge']/max(h, w))
    proxy_size = (max(1, round(w*factor)), max(1, round(h*factor)))
    candidate = cv2.resize(candidate_rgb, proxy_size, interpolation=cv2.INTER_AREA)
    source = cv2.resize(source_rgb, proxy_size, interpolation=cv2.INTER_AREA)
    cmask = cv2.resize(candidate_mask, proxy_size, interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(source_alpha, proxy_size, interpolation=cv2.INTER_AREA)
    observed = (cmask >= .7) & (alpha >= .95)
    erosion = max(0, int(round(CONFIG['edge_erosion_pixels']*factor)))
    if erosion:
        observed = binary_erosion(observed, iterations=erosion, border_value=0)
    report = {
        'schema': SCHEMA, 'config': dict(CONFIG), 'source_proxy_size': [w, h],
        'analysis_size': [proxy_size[0], proxy_size[1]], 'candidate_texture_imported': False,
        'candidate_geometry_imported': False, 'alpha_modified': False, 'applied': False,
        'visual_acceptance': False,
    }
    count = int(observed.sum())
    report['observed_overlap_pixels'] = count
    if count < CONFIG['minimum_observation_pixels']:
        report['bypass_reason'] = 'insufficient_registered_subject_overlap'
        return np.ones((h, w, 3), np.float32), report

    sigma = max(2., max(proxy_size)*CONFIG['lowpass_sigma_fraction'])
    propagation = max(sigma, max(proxy_size)*CONFIG['propagation_sigma_fraction'])
    eps = np.float32(.006)
    candidate_luma = np.maximum(candidate @ LUMA, eps)
    source_luma = np.maximum(source @ LUMA, eps)
    candidate_log_luma = np.log2(candidate_luma)
    source_log_luma = np.log2(source_luma)
    candidate_low, support = _normalised_blur(candidate_log_luma, observed.astype(np.float32), sigma)
    source_low, source_support = _normalised_blur(source_log_luma, observed.astype(np.float32), sigma)
    residual_ev = candidate_low - source_low
    baseline = _weighted_median(residual_ev[observed], support[observed])
    residual_ev = residual_ev - baseline

    candidate_chroma = np.log2(np.maximum(candidate, eps)) - candidate_log_luma[..., None]
    source_chroma = np.log2(np.maximum(source, eps)) - source_log_luma[..., None]
    candidate_chroma_low, _ = _normalised_blur(candidate_chroma, observed.astype(np.float32), sigma)
    source_chroma_low, _ = _normalised_blur(source_chroma, observed.astype(np.float32), sigma)
    residual_chroma = candidate_chroma_low - source_chroma_low
    chroma_baseline = np.asarray([_weighted_median(residual_chroma[..., channel][observed], support[observed])
                                  for channel in range(3)], np.float32)
    residual_chroma -= chroma_baseline
    residual_chroma -= residual_chroma.mean(-1, keepdims=True)

    # Propagate only the smooth residual.  The output is a source-coordinate
    # field; no candidate RGB or edge detail survives this operation.
    field_weight = np.clip(support / max(CONFIG['minimum_support'], 1e-5), 0, 1)
    ev_field, propagation_support = _normalised_blur(
        residual_ev*field_weight, field_weight, propagation)
    chroma_field, _ = _normalised_blur(
        residual_chroma*field_weight[..., None], field_weight, propagation)
    source_domain = alpha >= .5
    supported = source_domain & (propagation_support >= CONFIG['minimum_support'])
    ev_field = np.clip(ev_field, CONFIG['minimum_ev'], CONFIG['maximum_ev'])
    chroma_field = np.clip(chroma_field, -CONFIG['maximum_chroma_log'], CONFIG['maximum_chroma_log'])
    chroma_field -= chroma_field.mean(-1, keepdims=True)
    log_gain = (ev_field[..., None] + chroma_field)*np.float32(CONFIG['dose'])
    log_gain[~supported] = 0
    gain_small = np.exp2(log_gain).astype(np.float32)
    gain = cv2.resize(gain_small, (w, h), interpolation=cv2.INTER_LINEAR)
    gain = np.clip(gain, CONFIG['minimum_gain'], CONFIG['maximum_gain']).astype(np.float32)
    gain[np.abs(gain-1) <= 1e-6] = 1.
    active_pixels = cv2.resize(supported.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST) >= .5
    active_gain = gain[active_pixels]
    report.update({
        'baseline_ev_removed': float(baseline),
        'baseline_chroma_removed': chroma_baseline.tolist(),
        'supported_pixels': int(supported.sum()),
        'gain_q10_50_90': np.quantile(active_gain, [.1, .5, .9], axis=0).tolist() if active_gain.size else [[1., 1., 1.]]*3,
        'gain_min_max': [active_gain.min(axis=0).tolist(), active_gain.max(axis=0).tolist()] if active_gain.size else [[1., 1., 1.], [1., 1., 1.]],
        'changed_pixels': int(np.count_nonzero(np.any(np.abs(gain-1) > 1e-6, axis=-1) & active_pixels)),
        'field_source': 'registered_candidate_subject_low_frequency_light_evidence',
        'response_scope': 'source_foreground_only', 'candidate_RGB_used_as': 'light_evidence_only',
        'applied': bool(np.any(np.abs(gain-1) > 1e-5)),
    })
    return gain, report


def apply(foreground, gain, alpha, *, strength=1.):
    foreground = np.asarray(foreground, np.float32)
    gain = np.asarray(gain, np.float32)
    alpha = np.asarray(alpha, np.float32)
    if (foreground.ndim != 3 or foreground.shape[-1] != 3 or gain.shape != foreground.shape
            or alpha.shape != foreground.shape[:2] or not np.isfinite(foreground).all()
            or not np.isfinite(gain).all() or not np.isfinite(alpha).all()
            or not 0 <= strength <= 1 or foreground.min(initial=0) < 0
            or gain.min(initial=0) < 0 or alpha.min(initial=0) < 0 or alpha.max(initial=0) > 1):
        raise ValueError('Invalid candidate subject light application inputs')
    dose = np.clip(alpha, 0, 1)[..., None]*np.float32(strength)
    result = foreground*(1+dose*(gain-1))
    result = np.minimum(np.clip(result, 0, 1), alpha[..., None])
    return result.astype(np.float32)
