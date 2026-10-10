"""Versioned relative creative environment, supplied by per-photo art intent.

Directions are camera-relative art coordinates, not recovered physical lamps.
The generated design owns this intent; no image filename or fixed color preset
chooses it. The resulting environment is a continuous light constraint only.
"""
from pathlib import Path
import hashlib
import json

import numpy as np

FRAME = 'art_camera_x_right_y_up_z_viewer'


def normalize_environment_intent(value):
    if not value:
        return None
    ambient = np.asarray(value['ambient_linear_rgb'], np.float64)
    if ambient.shape != (3,) or not np.isfinite(ambient).all() or np.any(ambient < 0):
        raise ValueError('Creative ambient light requires three nonnegative finite RGB values')
    lights = []
    for item in value['lights']:
        direction = np.asarray(item['direction'], np.float64)
        radiance = np.asarray(item['peak_linear_rgb'], np.float64)
        angle = float(item['angular_fwhm_degrees'])
        if (direction.shape != (3,) or not np.isfinite(direction).all() or
            np.linalg.norm(direction) == 0 or radiance.shape != (3,) or
            not np.isfinite(radiance).all() or np.any(radiance < 0) or not 0 < angle < 180):
            raise ValueError('Creative lights need a nonzero direction, positive FWHM and nonnegative RGB radiance')
        lights.append({'role': str(item.get('role', 'creative light')),
            'direction': (direction/np.linalg.norm(direction)).tolist(),
            'angular_fwhm_degrees': angle, 'peak_linear_rgb': radiance.tolist()})
    return {'schema': 'photographic_environment_intent.v1', 'frame': FRAME,
        'ambient_linear_rgb': ambient.tolist(), 'lights': lights,
        'physical_lamps_recovered': False, 'radiance_units': 'relative creative linear RGB'}


def environment_from_intent(intent, height=256, width=512):
    theta = (np.arange(height, dtype=np.float64)+.5)*np.pi/height
    phi = ((np.arange(width, dtype=np.float64)+.5)/width-.5)*2*np.pi
    st = np.sin(theta)[:, None]
    directions = np.stack(np.broadcast_arrays(st*np.sin(phi)[None, :],
        np.cos(theta)[:, None], -st*np.cos(phi)[None, :]), -1)
    rgb = np.broadcast_to(np.asarray(intent['ambient_linear_rgb'], np.float64), (height, width, 3)).copy()
    for light in intent['lights']:
        axis = np.asarray(light['direction'], np.float64); axis /= np.linalg.norm(axis)
        half_angle = np.deg2rad(float(light['angular_fwhm_degrees'])/2)
        concentration = np.log(2)/(1-np.cos(half_angle))
        lobe = np.exp(concentration*(directions @ axis-1))
        rgb += lobe[..., None]*np.asarray(light['peak_linear_rgb'], np.float64)
    return rgb.astype(np.float32), directions.astype(np.float32)


def prepare_environment(intent, run, output_dir):
    normalized = normalize_environment_intent(intent)
    if normalized is None:
        return None
    directory = Path(output_dir).resolve(); directory.mkdir(parents=True, exist_ok=True)
    rgb, directions = environment_from_intent(normalized)
    path = directory/'creative-environment.npz'
    np.savez_compressed(path, values=rgb, directions=directions)
    asset = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'array_key': 'values', 'candidate_sha256': run['reference']['sha256'],
        'source_sha256': run['source']['sha256'], 'frame': FRAME,
        'intent': normalized, 'radiance_units': 'relative declared creative linear RGB'}
    (directory/'environment.json').write_text(json.dumps(asset, ensure_ascii=False, indent=2), encoding='utf-8')
    return asset
