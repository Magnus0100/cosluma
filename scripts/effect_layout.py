"""Place RGBA material and bound its dose with deterministic source-canvas rules.

The image model supplies material only. Its suggested pixel position and
opacity are not trusted as the scene layout or photographic exposure.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

SCHEMA = 'independent_effect_consumer_layout.v2'
PARAMETERS = {'support_alpha_threshold': .01, 'extent_inset_fraction': .04,
    'local_luminance_quantile': .75, 'effect_statistic_quantile': .95,
    'appearance_local_gain': 3., 'appearance_frame_key_fraction': .8,
    'appearance_frame_highlight_cap': .6, 'maximum_alpha_normalization_gain': 4.,
    'density_presets': {'scattering_smoke': [.24, .45], 'atmospheric_haze': [.14, .3],
        'dust_particles': [.3, .65], 'luminous_particles': [.5, .9], 'light_beam': [.3, .65]},
    'measurement_max_side': 512}
_LUMA = np.array([.2126, .7152, .0722], np.float32)


def layout_transform(alpha, extent, size):
    if (len(extent) != 4 or any(not np.isfinite(n) or not 0 <= n <= 1 for n in extent)
            or not extent[0] < extent[2] or not extent[1] < extent[3]):
        raise ValueError('Effect layout requires the frozen nonempty normalized extent')
    ys, xs = np.nonzero(alpha > PARAMETERS['support_alpha_threshold'])
    if not len(xs):
        raise ValueError('Effect material has no alpha support')
    box = [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)]
    w, h = size
    target = np.array([extent[0]*w, extent[1]*h, extent[2]*w, extent[3]*h], np.float64)
    scale = min((target[2]-target[0])/(box[2]-box[0]), (target[3]-target[1])/(box[3]-box[1]))
    scale *= 1-2*PARAMETERS['extent_inset_fraction']
    centre = (target[:2]+target[2:])/2
    original_centre = (np.array(box[:2])+np.array(box[2:]))/2
    matrix = np.eye(3, dtype=np.float64)
    matrix[0, 0] = matrix[1, 1] = scale
    matrix[:2, 2] = centre-scale*original_centre
    yy, xx = np.mgrid[:h, :w]
    visibility = ((xx+.5 >= target[0]) & (xx+.5 < target[2])
        & (yy+.5 >= target[1]) & (yy+.5 < target[3])).astype(np.float32)
    return matrix, visibility, box


def measure_before_luminance(run, plan, size):
    from design_reconstruction import apply_plan, geometry_for_plan
    from native_source_baseline import source_for_plan
    ratio = min(1., PARAMETERS['measurement_max_side']/max(size))
    probe_size = tuple(max(1, round(n*ratio)) for n in size)
    source = source_for_plan(plan, probe_size)
    values, _ = apply_plan(source, source.copy(), plan, geometry_for_plan(plan, probe_size),
        run=run, include_optics=False)
    luma = np.maximum(values @ _LUMA, 0)
    if not np.isfinite(luma).all():
        raise ValueError('Effect dose requires finite actual before-composite luminance')
    resized = np.asarray(Image.fromarray(luma).resize(tuple(size), Image.Resampling.BILINEAR), np.float32)
    return resized, luma


def bound_dose(mapped, before_luma, extent_visibility, material='scattering_smoke'):
    if mapped.shape[:2] != before_luma.shape or mapped.shape[:2] != extent_visibility.shape:
        raise ValueError('Effect dose and before-composite must share the same camera grid')
    alpha = mapped[..., 3]
    support = (alpha > PARAMETERS['support_alpha_threshold']) & (extent_visibility > 0)
    local = before_luma[extent_visibility > 0]
    if not np.any(support) or not len(local):
        raise ValueError('Effect dose has no visible material or valid local measurement')
    if material not in PARAMETERS['density_presets']:
        raise ValueError('Unknown effect material cannot choose density or optical appearance')
    q = PARAMETERS['effect_statistic_quantile']
    alpha_q = float(np.quantile(alpha[support], q))
    straight = np.divide(mapped[..., :3], alpha[..., None], out=np.zeros_like(mapped[..., :3]),
        where=alpha[..., None] > 1e-8)
    straight_q = float(np.quantile((straight @ _LUMA)[support], q))
    local_q = float(np.quantile(local, PARAMETERS['local_luminance_quantile']))
    frame_key, frame_high = map(float, np.quantile(before_luma, [.9, .95]))
    target = min(frame_high*PARAMETERS['appearance_frame_highlight_cap'],
        max(local_q*PARAMETERS['appearance_local_gain'], frame_key*PARAMETERS['appearance_frame_key_fraction']))
    if target <= 0 or straight_q <= 0:
        raise ValueError('Unlit frame or zero RGB material has no automatic optical appearance')
    if material in {'luminous_particles', 'light_beam'}:
        target = max(target, frame_high)
    rgb_gain = target/max(straight_q, 1e-12)
    density_q, peak = PARAMETERS['density_presets'][material]
    alpha_gain = min(PARAMETERS['maximum_alpha_normalization_gain'], density_q/max(alpha_q, 1e-12))
    result_alpha = np.minimum(alpha*np.float32(alpha_gain), peak)
    # RGB appearance and optical transmittance are separate. Associate the
    # adjusted straight material with the final alpha once, before consumption.
    result_rgb = np.clip(straight*np.float32(rgb_gain), 0, 1)*result_alpha[..., None]
    result = np.dstack([result_rgb, result_alpha]).astype(np.float32)
    return result, {'RGB_appearance_multiplier': rgb_gain, 'alpha_density_multiplier': alpha_gain,
        'source_alpha_q95': alpha_q, 'source_straight_luminance_q95': straight_q,
        'local_before_luminance_q75': local_q, 'frame_before_luminance_q90': frame_key,
        'frame_before_luminance_q95': frame_high, 'target_material_luminance_q95': target,
        'actual_material_luminance_q95': float(np.quantile((np.clip(straight*rgb_gain, 0, 1) @ _LUMA)[support], q)),
        'actual_effect_alpha_q95': float(np.quantile(result_alpha[support], q)),
        'actual_effect_alpha_max': float(result_alpha.max()),
        'rules': 'RGB appearance follows measured light; alpha independently follows fixed material transparency preset'}
