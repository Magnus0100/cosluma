"""Compile effect-only model text from a closed numeric/enumerated schema.

Free-form role/scene prose is a QA requirement, never model conditioning.
Explicit typed decisions select material/placement; background measurements
provide ambient colour. No prose guessing or per-photo prompt search is used.
"""
from __future__ import annotations

import copy
import math

import numpy as np

SCHEMA = 'independent_effect_condition.v1'
METHOD = 'typed_effect_whitelist_64_channel_canvas_v4'
MATERIALS = {'scattering_smoke', 'atmospheric_haze', 'dust_particles', 'luminous_particles', 'light_beam'}


def validate_condition(value):
    if not isinstance(value, dict) or set(value) != {'schema', 'material', 'colour_rgb8', 'extent_2d', 'ambient', 'density'}:
        raise ValueError('Effect condition is a closed effect-only schema; unknown prose/keys are refused')
    if (value['schema'] != SCHEMA or not isinstance(value['material'], str)
            or value['material'] not in MATERIALS or value['density'] != 'sparse'):
        raise ValueError('Only whitelisted effect materials and sparse density can condition the model')
    def colour(item):
        if not isinstance(item, list) or len(item) != 3 or any(type(n) is not int or not 0 <= n <= 255 for n in item):
            raise ValueError('Effect/ambient colour must be three 8-bit numeric values')
    colour(value['colour_rgb8'])
    rect = value['extent_2d']
    if (not isinstance(rect, list) or len(rect) != 4 or any(type(n) not in {int, float} or not math.isfinite(n) or not 0 <= n <= 1 for n in rect)
            or not rect[0] < rect[2] or not rect[1] < rect[3]):
        raise ValueError('Effect extent must be a nonempty normalized two-dimensional rectangle')
    ambient = value['ambient']
    if not isinstance(ambient, dict) or set(ambient) != {'colour_rgb8', 'linear_luminance', 'illumination'}:
        raise ValueError('Ambient condition must contain only measured colour/luminance and fixed illumination')
    colour(ambient['colour_rgb8'])
    if (type(ambient['linear_luminance']) not in {int, float} or not math.isfinite(ambient['linear_luminance'])
            or not 0 <= ambient['linear_luminance'] <= 1 or ambient['illumination'] != 'soft_diffuse'):
        raise ValueError('Invalid numeric/enumerated ambient condition')
    return copy.deepcopy(value)


def _ambient(plan):
    from design_reconstruction import sample_asset
    background = next(row for row in plan['layers'] if row['kind'] == 'background_design')['assets']
    size, box = (96, 144), (0, 0, 96, 144)
    rgb = sample_asset(background['background'], 'values', box, size)
    valid = sample_asset(background['validity'], 'values', box, size)
    alpha = sample_asset(plan['mask_roles']['foreground_alpha'], 'values', box, size)
    selected = (valid > .5) & (alpha < .05) & np.isfinite(rgb).all(axis=-1)
    values = np.clip(rgb[selected], 0, 1)
    if not len(values):
        raise ValueError('Effect ambient colour requires actual visible background pixels')
    luma = values @ np.array([.2126, .7152, .0722], np.float32)
    lo, hi = np.quantile(luma, [.25, .9])
    central = values[(luma >= lo) & (luma <= hi)]
    median = np.median(central if len(central) else values, axis=0)
    srgb = np.where(median <= .0031308, median*12.92, 1.055*median**(1/2.4)-.055)
    return {'colour_rgb8': np.uint8(np.clip(srgb*255+.5, 0, 255)).tolist(),
        'linear_luminance': round(float(median @ np.array([.2126, .7152, .0722])), 4),
        'illumination': 'soft_diffuse'}


def compile_condition(spec, plan):
    if spec.get('effect_condition') is not None:
        condition = validate_condition(spec['effect_condition'])
    else:
        design = validate_effect_design(spec.get('effect_design'))
        condition = validate_condition({'schema': SCHEMA, **design, 'ambient': _ambient(plan)})
    scattering = condition['material'] in {'scattering_smoke', 'atmospheric_haze', 'dust_particles'}
    if scattering != (spec['operation'] == 'over'):
        raise ValueError('Effect material differs from scattering-over versus emitted-radiance operation')
    return condition


def validate_effect_design(value):
    """Typed creative decisions; actual plate supplies ambient measurements."""
    if not isinstance(value, dict) or set(value) != {'material', 'colour_rgb8', 'extent_2d', 'density'}:
        raise ValueError('Independent effects require explicit typed material/colour/extent/density; prose cannot authorize generation')
    checked = validate_condition({'schema': SCHEMA, **value, 'ambient': {
        'colour_rgb8': [0, 0, 0], 'linear_luminance': 0., 'illumination': 'soft_diffuse'}})
    return {key: checked[key] for key in value}


def prompt_from_condition(condition):
    value = validate_condition(condition)
    material = {'scattering_smoke': 'one thin soft swirling smoke wisp',
        'atmospheric_haze': 'a thin sparse translucent haze', 'dust_particles': 'sparse tiny suspended dust motes',
        'luminous_particles': 'sparse tiny luminous particles', 'light_beam': 'one narrow soft luminous beam'}[value['material']]
    colour = '#'+''.join(f'{n:02X}' for n in value['colour_rgb8'])
    ambient = '#'+''.join(f'{n:02X}' for n in value['ambient']['colour_rgb8'])
    l, t, r, b = value['extent_2d']
    return (f'This is an RGBA image with transparency. A standalone transparent visual-effects asset: {material}. '
        f'Sparse effect colour {colour}, softly illuminated by diffuse ambient colour {ambient}. '
        f'The entire canvas is transparent except this effect. Empty transparent space occupies most of the canvas. '
        f'The effect occupies only the rectangle from x={l:.3f} to {r:.3f}, y={t:.3f} to {b:.3f}, '
        'with x measured from the left edge and y from the top edge of the full canvas. '
        'Extremely soft feathered fading edges and low fractional alpha inside the effect; alpha exactly zero outside it. '
        'The background is transparent. '+('Preserve natural fine wisp detail. '
            if value['material'] in {'scattering_smoke', 'atmospheric_haze'} else 'Preserve fine effect detail. '))
