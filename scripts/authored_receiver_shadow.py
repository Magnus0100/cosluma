"""Foot-anchored 2D artistic silhouette projection on an authored receiver.

This CPU producer reads only source coverage/footwear and environment validity.
It never recovers physical cast transport, copies RGB, modifies the matte, or
uses a captured shadow/reflection donor. Declaration execution stays false;
an independently enabled user shadow plan requests this artistic operation.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from scene_receiver_intent import (GROUNDED_FEET, make_receiver_binding,
                                   validate_bound_receiver_intent)

PLAN_SCHEMA = 'authored_receiver_shadow_plan.v1'
FIELD_SCHEMA = 'source_bound_authored_receiver_shadow.v1'
RECIPE = 'source-alpha-foot-anchored-2d-artistic-projection-v1'
_PLAN_KEYS = {'schema_version', 'enabled', 'direction_to_light_xy',
              'length', 'opacity', 'softness'}
_UV_SIZE = 1024
_FIELD_LONG_EDGE = 1024
_FOOT_ANCHOR_DISTANCE_FRACTION = .02


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _content(value):
    if isinstance(value, dict):
        return {key: _content(item) for key, item in value.items()
                if key not in {'path', 'identity'}}
    if isinstance(value, list):
        return [_content(item) for item in value]
    return value


def _code_sha():
    from runtime_resources import code_path
    return _sha(code_path('authored_receiver_shadow.py'))


def normalize_receiver_shadow_plan(value=None):
    """Canonical saved artist settings; unknown direction and off are defaults."""
    value = {} if value is None else value
    if not isinstance(value, dict) or set(value)-_PLAN_KEYS:
        raise ValueError('Authored receiver shadow has unsupported settings')
    if value.get('schema_version', PLAN_SCHEMA) != PLAN_SCHEMA:
        raise ValueError('Unsupported authored receiver shadow plan schema')
    enabled = value.get('enabled', False)
    if type(enabled) is not bool:
        raise ValueError('Authored receiver shadow enabled must be boolean')

    def number(name, default, low, high):
        number = value.get(name, default)
        if (isinstance(number, bool) or not isinstance(number, (int, float)) or
                not math.isfinite(number) or not low <= number <= high):
            raise ValueError('Authored receiver shadow '+name+' is outside its finite bounds')
        return float(number)

    direction = value.get('direction_to_light_xy')
    if direction is not None:
        if (not isinstance(direction, (list, tuple)) or len(direction) != 2 or
                any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                    not math.isfinite(v) or not -1 <= v <= 1 for v in direction) or
                math.hypot(*direction) <= 1e-8):
            raise ValueError('Artist light direction must be a finite nonzero screen XY pair')
        direction = [float(v) for v in direction]
    return {'schema_version': PLAN_SCHEMA, 'enabled': enabled,
        'direction_to_light_xy': direction, 'length': number('length', .7, .01, 4.),
        'opacity': number('opacity', .4, 0., 1.),
        'softness': number('softness', .025, 0., .25)}


def _context(run, plan):
    """Build a fresh binding exactly from the current native asset closure."""
    background = next((layer for layer in plan['layers'] if layer['kind'] == 'background_design'), None)
    if background is None:
        raise ValueError('Authored shadow requires a bound native background')
    assets, masks = background['assets'], plan['mask_roles']
    size = list(plan['canvas']['source_oriented_size'])
    baseline = plan['source_baseline']
    environment = {'background': assets['background'], 'validity': assets['validity'],
                   'candidate_removal': masks['candidate_removal']}
    observations = {'foreground_alpha': masks['foreground_alpha']}
    contact = next((layer for layer in plan['layers'] if layer['kind'] == 'contact_shadow'), {})
    contact_assets = contact.get('assets', {})
    if contact_assets.get('source_component_owner'):
        observations['foot_owner'] = contact_assets['source_component_owner']
    if contact_assets.get('receiver_support'):
        observations['receiver_observation'] = contact_assets['receiver_support']
    current_binding = make_receiver_binding(source=run['source'], reference=run['reference'],
        source_canvas_size=size, decode_version=baseline['decode_version'],
        environment_assets=environment, environment_mapping=plan['canvas'].get('candidate_mapping'),
        observation_assets=observations)
    intent = validate_bound_receiver_intent(
        plan['scene_design']['authored_receiver_intent'], expected_binding=current_binding)
    if (plan.get('source_sha256') != run['source']['sha256'] or
            plan.get('candidate_sha256') != run['reference']['sha256'] or
            baseline['decode_version'] != run['source']['decode_version']):
        raise ValueError('Authored shadow source/reference/decode differs from its native plan')
    footwear = contact_assets.get('source_footwear')
    if not footwear or not observations.get('foot_owner'):
        raise ValueError('Source footwear observation and owner are required for artist foot anchors')
    if (footwear.get('coordinate_space') not in {'final_source_proxy', 'source_canvas'} or
            footwear.get('original_canvas_size', size) != size or
            masks['foreground_alpha'].get('coordinate_space') != 'source_canvas'):
        raise ValueError('Source footwear/alpha does not span the declared source canvas')
    binding = {'receiver_binding': _content(current_binding),
        'receiver_intent_identity': intent['identity'],
        'receiver_declaration_sha256': intent['declaration_sha256'],
        'footwear_sha256': footwear['sha256'],
        'footwear_metadata_sha256': _digest(_content(footwear)),
        'producer_sha256': _code_sha(), 'recipe': RECIPE}
    inputs = [run['source'], run['reference'], *environment.values(),
              *observations.values(), footwear]
    return intent, binding, inputs, masks['foreground_alpha'], footwear, assets['validity']


def _verify_files(assets):
    seen = set()
    for asset in assets:
        identity = (asset['path'], asset['sha256'])
        if identity not in seen and _sha(asset['path']) != asset['sha256']:
            raise ValueError('Authored shadow input content hash differs: '+str(asset['path']))
        seen.add(identity)


def _load_scalar(asset):
    with np.load(asset['path'], allow_pickle=False) as data:
        values = np.asarray(data[asset.get('array_key', 'values')], np.float32)
    if (values.ndim != 2 or not np.isfinite(values).all() or
            values.min() < 0 or values.max() > 1.000001):
        raise ValueError('Authored shadow requires finite scalar coverage in [0,1]')
    return np.clip(values, 0, 1)


def _sample(values, x, y, canvas_size):
    """Native integer pixel centers to the scalar asset's full-canvas grid."""
    width, height = canvas_size
    sx = (x+.5)*values.shape[1]/width-.5
    sy = (y+.5)*values.shape[0]/height-.5
    return map_coordinates(values, [sy, sx], order=1, mode='grid-constant',
                           cval=0., prefilter=False).astype(np.float32)


def _homography(plane):
    """Declared ordered corners map unit-square UV to native screen centers."""
    rows, target = [], []
    for (u, v), (x, y) in zip(((0, 0), (1, 0), (1, 1), (0, 1)), plane):
        rows.extend(([u, v, 1, 0, 0, 0, -u*x, -v*x],
                     [0, 0, 0, u, v, 1, -u*y, -v*y]))
        target.extend((x, y))
    matrix = np.append(np.linalg.solve(np.asarray(rows, np.float64), target), 1.).reshape(3, 3)
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) <= 1e-12:
        raise ValueError('Authored projection plane is singular')
    return matrix


def _transform(matrix, x, y):
    denominator = matrix[2, 0]*x+matrix[2, 1]*y+matrix[2, 2]
    valid = np.isfinite(denominator) & (np.abs(denominator) > 1e-10)
    safe = np.where(valid, denominator, 1.)
    tx = (matrix[0, 0]*x+matrix[0, 1]*y+matrix[0, 2])/safe
    ty = (matrix[1, 0]*x+matrix[1, 1]*y+matrix[1, 2])/safe
    return tx, ty, valid & np.isfinite(tx) & np.isfinite(ty)


def _jacobian(matrix, point):
    x, y = point
    u, v, valid = _transform(matrix, x, y)
    if not valid:
        raise ValueError('Anchor is outside finite projective coordinates')
    denominator = matrix[2, 0]*x+matrix[2, 1]*y+matrix[2, 2]
    return np.array([[(matrix[0, 0]-u*matrix[2, 0])/denominator,
                      (matrix[0, 1]-u*matrix[2, 1])/denominator],
                     [(matrix[1, 0]-v*matrix[2, 0])/denominator,
                      (matrix[1, 1]-v*matrix[2, 1])/denominator]], np.float64)


def _inside(x, y, polygon):
    result = np.zeros(np.broadcast(x, y).shape, bool)
    for index, a in enumerate(polygon):
        b = polygon[(index+1) % len(polygon)]
        if a[1] != b[1]:
            crossing = ((a[1] > y) != (b[1] > y)) & (
                x < (b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0])
            result ^= crossing
    return result


def _anchor_receipt(alpha, footwear, anchors, size):
    """Artist contact points must be near actual source footwear and coverage."""
    h, w = footwear.shape
    y = (np.arange(h)+.5)*size[1]/h-.5
    x = (np.arange(w)+.5)*size[0]/w-.5
    yy, xx = np.broadcast_arrays(y[:, None], x[None, :])
    source_alpha = _sample(alpha, xx, yy, size)
    trusted = (footwear > .5) & (source_alpha > .5)
    if not trusted.any():
        raise ValueError('Source footwear has no observed foreground coverage')
    foot_x, foot_y = xx[trusted], yy[trusted]
    maximum = _FOOT_ANCHOR_DISTANCE_FRACTION*max(size)
    distances = {}
    for name, (ax, ay) in anchors.items():
        nearest = float(np.hypot(foot_x-ax, foot_y-ay).min())
        distances[name] = nearest
        if nearest > maximum:
            raise ValueError('Authored '+name+' foot anchor is too far from source footwear/alpha')
    return {'distance_native_pixels': distances, 'maximum_native_pixels': maximum,
            'source_footwear_and_alpha_checked': True,
            'anatomical_label_inferred': False}


def _project_mask(alpha, validity, size, declaration, config):
    """One invertible artistic projection, UV blur, then source receiver clips."""
    plane, region = declaration['receiver_plane'], declaration['receiver_region']
    anchors = declaration['foot_anchors']
    points = np.array(list(anchors.values()), np.float64)
    pivot = points.mean(axis=0)
    plane_to_screen = _homography(plane)
    screen_to_plane = np.linalg.inv(plane_to_screen)
    pu, pv, valid_pivot = _transform(screen_to_plane, *pivot)
    if not valid_pivot:
        raise ValueError('Foot pivot has no finite plane UV')
    pivot_uv = np.array([pu, pv])
    jacobian = _jacobian(screen_to_plane, pivot)
    if len(points) == 2:
        dx = points[1, 0]-points[0, 0]
        if abs(dx) <= 1e-6:
            raise ValueError('Dual-foot screen ground line is vertical/degenerate')
        slope = (points[1, 1]-points[0, 1])/dx
        u, v, valid = _transform(screen_to_plane, points[:, 0], points[:, 1])
        if not valid.all():
            raise ValueError('Foot anchor plane coordinates are not finite')
        # Use the two UV anchor midpoint as the receiver pivot. Projective
        # mapping does not preserve a screen midpoint; mixing the two centers
        # would offset both sole attachments even though their slope is right.
        pivot_uv = np.array([u.mean(), v.mean()])
        x_axis_uv = (np.array([u[1], v[1]])-np.array([u[0], v[0]]))/dx
    else:
        slope = 0.
        x_axis_uv = jacobian[:, 0]
    # Artists provide light XY with y up. The opposite screen direction is
    # [-light_x,+light_y] because native image y increases downward.
    lx, ly = config['direction_to_light_xy']
    shadow_uv = jacobian@np.array([-lx, ly], np.float64)
    direction_norm = np.linalg.norm(shadow_uv)
    if direction_norm <= 1e-12:
        raise ValueError('Artist shadow direction degenerates in the authored plane')
    shadow_uv /= direction_norm
    ah, aw = alpha.shape
    yy, xx = np.nonzero(alpha > .02)
    if not len(xx):
        raise ValueError('Source alpha has no projection silhouette')
    native_x = (xx+.5)*size[0]/aw-.5
    native_y = (yy+.5)*size[1]/ah-.5
    source_height = float(np.max(pivot[1]+slope*(native_x-pivot[0])-native_y))
    if source_height <= 1e-6:
        raise ValueError('Source silhouette has no positive height above the authored ground line')
    vertical_uv = shadow_uv*config['length']/source_height
    source_to_uv = np.eye(3, dtype=np.float64)
    source_to_uv[:2, 0] = x_axis_uv+vertical_uv*slope
    source_to_uv[:2, 1] = -vertical_uv
    source_to_uv[:2, 2] = pivot_uv-x_axis_uv*pivot[0]+vertical_uv*(pivot[1]-slope*pivot[0])
    if abs(np.linalg.det(source_to_uv)) <= 1e-12:
        raise ValueError('Artist light direction is parallel to the foot ground line')
    inverse = np.linalg.inv(source_to_uv)
    uv = (np.arange(_UV_SIZE, dtype=np.float64)+.5)/_UV_SIZE
    vv, uu = np.broadcast_arrays(uv[:, None], uv[None, :])
    sx, sy, finite = _transform(inverse, uu, vv)
    above_line = sy <= pivot[1]+slope*(sx-pivot[0])
    projected = _sample(alpha, sx, sy, size)*finite*above_line
    if config['softness']:
        projected = gaussian_filter(projected, config['softness']*_UV_SIZE,
                                    mode='constant', cval=0., truncate=4.)
    width = max(1, round(size[0]*min(1., _FIELD_LONG_EDGE/max(size))))
    height = max(1, round(size[1]*min(1., _FIELD_LONG_EDGE/max(size))))
    x = (np.arange(width, dtype=np.float64)+.5)*size[0]/width-.5
    y = (np.arange(height, dtype=np.float64)+.5)*size[1]/height-.5
    yy, xx = np.broadcast_arrays(y[:, None], x[None, :])
    u, v, finite = _transform(screen_to_plane, xx, yy)
    support = map_coordinates(projected, [v*_UV_SIZE-.5, u*_UV_SIZE-.5],
        order=1, mode='grid-constant', cval=0., prefilter=False)
    domain = finite & _inside(xx, yy, region) & _inside(xx, yy, plane)
    environment_validity = _sample(validity, xx, yy, size)
    # No (1-alpha): foreground coverage belongs to the final over operation.
    # Exclude only opaque current foreground, and retain fractional-edge role.
    opaque = _sample(alpha, xx, yy, size) >= .98
    support = np.clip(support, 0, 1)*domain*environment_validity*(~opaque)
    support = np.asarray(support, np.float32)
    receipt = {'plane_uv_to_source_canvas': plane_to_screen.tolist(),
        'source_canvas_to_plane_uv': screen_to_plane.tolist(),
        'source_silhouette_to_plane_uv': source_to_uv.tolist(),
        'pivot_source_canvas': pivot.tolist(), 'pivot_plane_uv': pivot_uv.tolist(),
        'ground_line_slope_source_pixels': float(slope),
        'source_positive_height_native_pixels': source_height,
        'source_x_axis_plane_uv_per_native_pixel': x_axis_uv.tolist(),
        'shadow_direction_plane_uv': shadow_uv.tolist(),
        'uv_grid_size': [_UV_SIZE, _UV_SIZE], 'field_grid_size': [width, height],
        'positive_samples': int((support > 0).sum()),
        'receiver_plane_clip_exact': bool(np.all(support[~domain] == 0)),
        'opaque_foreground_clip_exact': bool(np.all(support[opaque] == 0)),
        'source_coverage_multiplied_into_receiver': False,
        'artistic_2d_projection': True,
        'physical_cast_shadow': False}
    return support, receipt


def _state(plan, status, reason, **facts):
    plan.setdefault('execution', {})['authored_receiver_shadow'] = {
        'status': status, 'reason': reason, 'recipe': RECIPE,
        'physical_cast_shadow': False, **facts}
    return plan


def prepare_receiver_shadow(root, run, plan):
    """Return an owned plan copy; incomplete declarations skip ordinary saves."""
    result = copy.deepcopy(plan)
    result['layers'] = [layer for layer in result.get('layers', [])
                        if layer.get('kind') != 'authored_receiver_shadow']
    revision = (run.get('options') or {}).get('appearance_revision') or {}
    config = normalize_receiver_shadow_plan(revision.get('receiver_shadow_plan'))
    if not config['enabled']:
        return _state(result, 'disabled', 'artist_shadow_plan_disabled')
    if config['direction_to_light_xy'] is None:
        return _state(result, 'partial', 'artist_light_direction_unspecified')
    try:
        intent, binding, inputs, alpha_asset, footwear_asset, validity_asset = _context(run, result)
        declaration = intent['declaration']
        feet = GROUNDED_FEET[declaration['foot_stance']]
        if (not feet or declaration['ground_type'] == 'unknown' or
                not declaration['receiver_region'] or not declaration['receiver_plane'] or
                set(declaration['foot_anchors']) != set(feet)):
            return _state(result, 'partial', 'grounded_receiver_plane_and_foot_anchors_incomplete')
        binding['artist_plan_sha256'] = _digest(config)
        identity = _digest(binding)
        _verify_files(inputs)
        directory = Path(root)/'outputs/cache/authored-receiver-shadow'/identity[:24]
        field_path, manifest_path = directory/'attenuation.npz', directory/'manifest.json'
        if field_path.is_file() and manifest_path.is_file():
            cached = json.loads(manifest_path.read_text(encoding='utf-8'))
            if cached['binding'] != binding or _sha(field_path) != cached['field_sha256']:
                raise ValueError('Cached authored shadow binding or content differs')
            receipt = cached['projection']
        else:
            alpha, footwear, validity = map(_load_scalar, (alpha_asset, footwear_asset, validity_asset))
            size = binding['receiver_binding']['source_canvas_size']
            anchor_check = _anchor_receipt(alpha, footwear, declaration['foot_anchors'], size)
            support, receipt = _project_mask(alpha, validity, size, declaration, config)
            receipt['source_anchor_observation_check'] = anchor_check
            directory.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(field_path, values=support, uncertainty=(1-support).astype(np.float32))
            cached = {'schema_version': FIELD_SCHEMA, 'binding': binding,
                'field_sha256': _sha(field_path), 'projection': receipt}
            manifest_path.write_text(json.dumps(cached, ensure_ascii=False, indent=2,
                allow_nan=False)+'\n', encoding='utf-8')
        size = binding['receiver_binding']['source_canvas_size']
        asset = {'path': str(field_path.resolve()), 'sha256': cached['field_sha256'],
            'size': receipt['field_grid_size'], 'field_grid_size': receipt['field_grid_size'],
            'native_size': size, 'original_canvas_size': size,
            'coordinate_space': 'source_canvas', 'array_key': 'values',
            'representation': 'scalar', 'dtype': 'float32',
            'color_space': 'unitless_linear_mask', 'sampling': 'bilinear',
            'role': 'authored_2d_artistic_receiver_shadow_attenuation',
            'contract_schema': FIELD_SCHEMA, 'source_binding': binding,
            'uncertainty_role': 'one_minus_artistic_attenuation_support_not_calibrated_probability'}
        layer = {'id': 'authored_receiver_shadow', 'kind': 'authored_receiver_shadow',
            'enabled': True, 'strength': 1., 'assets': {'attenuation_support': asset},
            'params': {'opacity': config['opacity'], 'artist_plan': config,
                'application': 'background *= 1-opacity*attenuation_support; final over owns alpha',
                'physical_cast_shadow': False, 'foreground_exclusion': 'opaque_binary_only'},
            'provenance': {'schema_version': FIELD_SCHEMA, 'recipe': RECIPE,
                'binding': binding, 'projection': receipt,
                'receiver_intent_executes_effects': False,
                'artistic_2d_projection': True,
                'captured_source_donor_used': False, 'candidate_person_RGB_used': False,
                'source_alpha_changed': False, 'physical_cast_shadow': False}}
        result['layers'].append(layer)
        validate_receiver_shadow_layer(layer, run, result)
        return _state(result, 'ready', 'explicit_2d_artist_projection',
                      identity=identity, positive_samples=receipt['positive_samples'])
    except (ValueError, KeyError, TypeError, OSError, np.linalg.LinAlgError) as exc:
        result['layers'] = [layer for layer in result['layers'] if layer.get('kind') != 'authored_receiver_shadow']
        return _state(result, 'partial', 'receiver_shadow_inputs_unqualified', detail=str(exc))


def validate_receiver_shadow_layer(layer, run, plan):
    """Strict fresh binding/file/array checks before native full or ROI use."""
    if not isinstance(layer, dict) or layer.get('kind') != 'authored_receiver_shadow':
        raise ValueError('Expected authored_receiver_shadow layer')
    intent, binding, inputs, _, _, _ = _context(run, plan)
    revision = (run.get('options') or {}).get('appearance_revision') or {}
    config = normalize_receiver_shadow_plan(revision.get('receiver_shadow_plan'))
    binding['artist_plan_sha256'] = _digest(config)
    asset = layer['assets']['attenuation_support']
    if (not config['enabled'] or config['direction_to_light_xy'] is None or
            layer.get('enabled') is not True or layer.get('strength') != 1. or
            layer['params'].get('artist_plan') != config or
            layer['params'].get('opacity') != config['opacity'] or
            asset.get('source_binding') != binding or
            layer.get('provenance', {}).get('binding') != binding or
            asset.get('contract_schema') != FIELD_SCHEMA or
            asset.get('coordinate_space') != 'source_canvas' or
            asset.get('role') != 'authored_2d_artistic_receiver_shadow_attenuation' or
            asset.get('native_size') != binding['receiver_binding']['source_canvas_size'] or
            any(intent['execution'].values())):
        raise ValueError('Authored shadow layer identity/settings/authority changed')
    _verify_files(inputs)
    if _sha(asset['path']) != asset['sha256']:
        raise ValueError('Authored shadow scalar field content hash differs')
    with np.load(asset['path'], allow_pickle=False) as data:
        if set(data.files) != {'values', 'uncertainty'}:
            raise ValueError('Authored shadow only transports scalar attenuation and uncertainty')
        values, uncertainty = np.asarray(data['values']), np.asarray(data['uncertainty'])
    if (values.ndim != 2 or values.dtype != np.dtype('float32') or
            uncertainty.dtype != np.dtype('float32') or uncertainty.shape != values.shape or
            list(values.shape[1::-1]) != asset['field_grid_size'] or
            not np.isfinite(values).all() or not np.isfinite(uncertainty).all() or
            values.min() < 0 or values.max() > 1 or
            not np.allclose(uncertainty, 1-values, rtol=0, atol=1e-6)):
        raise ValueError('Authored shadow finite scalar/uncertainty contract failed')
    return {'schema_version': FIELD_SCHEMA, 'valid': True,
        'field_sha256': asset['sha256'], 'physical_cast_shadow': False,
        'artistic_2d_projection': True,
        'candidate_person_RGB_used': False, 'source_alpha_changed': False,
        'uncertainty_direct_weight': False, 'foreground_exclusion': 'opaque_binary_only',
        'binding_identity': _digest(binding)}
