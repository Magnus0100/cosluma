"""Source-bound authored landing/receiver intent, without effect execution.

The small declaration is editable before an environment exists. Binding it to
actual scene assets is a separate responsibility, after environment preparation.
Canvas points are top-left image pixels; anatomical left/right never mean screen
left/right. A drawn plane is a projection declaration, not measured 3D geometry.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re

SCHEMA = 'authored_receiver_intent.v1'
BOUND_SCHEMA = 'source_bound_authored_receiver_intent.v1'
RECIPE = 'source-bound-authored-receiver-intent-v1'
FOOT_STANCES = {
    'unknown': '落脚关系未指定', 'both_grounded': '双脚着地',
    'left_grounded': '人物左脚着地', 'right_grounded': '人物右脚着地',
    'airborne': '双脚离地',
}
GROUND_TYPES = {
    'unknown': '地面未指定', 'dry_floor': '干燥硬质地面',
    'soft_ground': '泥土或软地面', 'wet_floor': '湿地面',
    'water': '水面', 'mirror': '镜面',
}
ORIGINS = frozenset(('unknown', 'user_authored', 'saved_authored_brief'))
COORDINATE_SPACE = 'source_canvas_pixels_top_left_x_right_y_down'
DECLARATION_KEYS = frozenset(('schema_version', 'foot_stance', 'ground_type',
    'receiver_region', 'receiver_plane', 'foot_anchors', 'origin', 'text'))
GROUNDED_FEET = {'unknown': (), 'both_grounded': ('left', 'right'),
    'left_grounded': ('left',), 'right_grounded': ('right',), 'airborne': ()}
EXECUTION = {'cast_shadow': False, 'wet_reflection': False,
    'mirror_reflection': False, 'physical_ground_measured': False,
    'contact_height_measured': False, 'plane_is_metric_geometry': False,
    'candidate_person_RGB_used': False, 'human_normal_used': False,
    'old_contact_darkening_used': False, 'enables_automatic_contact': False}


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode('utf-8')).hexdigest()


def _content(value):
    if isinstance(value, dict):
        return {key: _content(item) for key, item in value.items()
                if key not in {'path', 'identity'}}
    if isinstance(value, list):
        return [_content(item) for item in value]
    return value


def _size(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 2 or
            any(type(v) is not int or v <= 0 for v in value)):
        raise ValueError('Source canvas size must be two positive integers')
    return list(value)


def _point(value, size):
    if (not isinstance(value, (list, tuple)) or len(value) != 2 or
            any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                not math.isfinite(v) for v in value)):
        raise ValueError('Receiver points must be finite source-canvas pixel pairs')
    point = [float(v) for v in value]
    if not (0 <= point[0] <= size[0]-1 and 0 <= point[1] <= size[1]-1):
        raise ValueError('Receiver point lies outside the oriented source canvas')
    return point


def _cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def _on_segment(p, a, b):
    return abs(_cross(a, b, p)) <= 1e-8 and all(
        min(a[i], b[i])-1e-8 <= p[i] <= max(a[i], b[i])+1e-8 for i in (0, 1))


def _intersects(a, b, c, d):
    products = (_cross(a, b, c), _cross(a, b, d),
                _cross(c, d, a), _cross(c, d, b))
    if products[0]*products[1] < 0 and products[2]*products[3] < 0:
        return True
    return any(abs(v) <= 1e-8 and _on_segment(p, u, w) for v, p, u, w in (
        (products[0], c, a, b), (products[1], d, a, b),
        (products[2], a, c, d), (products[3], b, c, d)))


def _polygon(value, size, *, plane=False):
    if value is None:
        return None
    if (not isinstance(value, (list, tuple)) or
            not (len(value) == 4 if plane else 3 <= len(value) <= 64)):
        raise ValueError('Draw an ordered four-corner plane or a 3–64 point receiver region')
    points = [_point(p, size) for p in value]
    if len({tuple(p) for p in points}) != len(points):
        raise ValueError('Receiver polygon has repeated points')
    count = len(points)
    area = sum(points[i][0]*points[(i+1) % count][1] -
               points[(i+1) % count][0]*points[i][1] for i in range(count)) / 2
    if abs(area) <= 1e-6:
        raise ValueError('Receiver polygon is degenerate')
    for i in range(count):
        for j in range(i+1, count):
            if j == i+1 or (i == 0 and j == count-1):
                continue
            if _intersects(points[i], points[(i+1) % count],
                           points[j], points[(j+1) % count]):
                raise ValueError('Receiver polygon intersects itself')
    if plane:
        turns = [_cross(points[i], points[(i+1) % count],
                        points[(i+2) % count]) for i in range(count)]
        if not (all(v > 1e-8 for v in turns) or all(v < -1e-8 for v in turns)):
            raise ValueError('Authored plane corners must form a convex ordered quadrilateral')
    return points


def _inside(point, polygon):
    inside = False
    for i, a in enumerate(polygon):
        b = polygon[(i+1) % len(polygon)]
        if _on_segment(point, a, b):
            return True
        if (a[1] > point[1]) != (b[1] > point[1]):
            x = a[0] + (point[1]-a[1])*(b[0]-a[0])/(b[1]-a[1])
            if point[0] < x:
                inside = not inside
    return inside


def _segment_inside(a, b, polygon):
    # Split at every boundary intersection; a concave region can contain both
    # endpoints while excluding part of the segment between them.
    direction = [b[i]-a[i] for i in (0, 1)]
    cuts = [0., 1.]
    for i, c in enumerate(polygon):
        d = polygon[(i+1) % len(polygon)]
        edge = [d[j]-c[j] for j in (0, 1)]
        delta = [c[j]-a[j] for j in (0, 1)]
        denominator = direction[0]*edge[1]-direction[1]*edge[0]
        if abs(denominator) > 1e-8:
            t = (delta[0]*edge[1]-delta[1]*edge[0])/denominator
            u = (delta[0]*direction[1]-delta[1]*direction[0])/denominator
            if 0 <= t <= 1 and 0 <= u <= 1:
                cuts.append(t)
        else:
            axis = 0 if abs(direction[0]) >= abs(direction[1]) else 1
            for point in (c, d):
                if _on_segment(point, a, b):
                    cuts.append((point[axis]-a[axis])/direction[axis])
    cuts = sorted(set(cuts))
    return all(_inside([a[j]+((x+y)/2)*direction[j] for j in (0, 1)], polygon)
        for x, y in zip(cuts, cuts[1:]))


def normalize_receiver_intent(value=None, *, source_canvas_size=None):
    """Normalize one explicit declaration; never parse text into geometry.

    The two photographic choices can be saved without a source or drawn points.
    Only supplied geometry requires canvas bounds. Missing information is unknown,
    not a validation error or a reason to block ordinary native editing.
    """
    value = {} if value is None else value
    if not isinstance(value, dict) or set(value)-DECLARATION_KEYS:
        raise ValueError('Receiver intent contains unsupported fields')
    if value.get('schema_version', SCHEMA) != SCHEMA:
        raise ValueError('Unsupported receiver intent schema')
    stance, ground = value.get('foot_stance', 'unknown'), value.get('ground_type', 'unknown')
    origin, text = value.get('origin', 'unknown'), value.get('text', '')
    if (not all(isinstance(v, str) for v in (stance, ground, origin)) or
            stance not in FOOT_STANCES or ground not in GROUND_TYPES or origin not in ORIGINS):
        raise ValueError('Unsupported photographic receiver choice or declaration origin')
    if not isinstance(text, str) or len(text.encode('utf-8')) > 8192:
        raise ValueError('Receiver description must be text of at most 8 KiB')
    anchors = value.get('foot_anchors', {})
    if not isinstance(anchors, dict) or set(anchors)-{'left', 'right'}:
        raise ValueError('Anchors use the person’s anatomical left/right feet')
    has_geometry = bool(anchors) or any(value.get(k) is not None
        for k in ('receiver_region', 'receiver_plane'))
    size = _size(source_canvas_size) if source_canvas_size is not None else None
    if has_geometry and size is None:
        raise ValueError('Drawn receiver geometry requires the source canvas size')
    region = _polygon(value.get('receiver_region'), size)
    plane = _polygon(value.get('receiver_plane'), size, plane=True)
    anchors = {foot: _point(anchors[foot], size) for foot in sorted(anchors)}
    if set(anchors)-set(GROUNDED_FEET[stance]):
        raise ValueError('Contact anchors conflict with the declared landing relationship')
    if region and plane and (not all(_inside(p, region) for p in plane) or
            not all(_segment_inside(a, plane[(i+1) % 4], region) for i, a in enumerate(plane))):
        raise ValueError('Authored receiver plane conflicts with the receiver region')
    for point in anchors.values():
        if any(polygon and not _inside(point, polygon) for polygon in (region, plane)):
            raise ValueError('Contact anchor lies outside the declared receiver region or plane')
    declared = stance != 'unknown' or ground != 'unknown' or has_geometry or bool(text)
    if declared and origin == 'unknown':
        raise ValueError('Declared receiver intent needs an explicit authored origin')
    return {'schema_version': SCHEMA, 'foot_stance': stance, 'ground_type': ground,
        'receiver_region': region, 'receiver_plane': plane, 'foot_anchors': anchors,
        'origin': origin, 'text': text}


def validate_receiver_intent(value, *, source_canvas_size=None):
    """Validate saved canonical declarations, without requiring a plane or anchors."""
    expected = normalize_receiver_intent(value, source_canvas_size=source_canvas_size)
    if not isinstance(value, dict) or value != expected:
        raise ValueError('Saved receiver declaration is not canonical')
    return copy.deepcopy(expected)


def receiver_intent_identity(value=None, *, source_canvas_size=None):
    return _digest(normalize_receiver_intent(value, source_canvas_size=source_canvas_size))


def _asset(value):
    if (not isinstance(value, dict) or not isinstance(value.get('path'), str) or
            not value['path'] or not isinstance(value.get('sha256'), str) or
            not re.fullmatch('[0-9a-f]{64}', value['sha256'])):
        raise ValueError('Receiver binding requires an actual asset path and SHA256')
    return {'path': value['path'], 'sha256': value['sha256'],
            'metadata_sha256': _digest(_content(value))}


def make_receiver_binding(*, source, reference, source_canvas_size, decode_version,
                          environment_assets, environment_mapping=None, observation_assets=None):
    """Bind supplied actual descriptors; file verification remains the asset owner’s job.

    No image/model is opened here. Callers must supply their current verified asset
    closure, not reinterpret a historical declaration as a current measurement.
    """
    size = _size(source_canvas_size)
    if not isinstance(decode_version, str) or not decode_version:
        raise ValueError('Receiver binding needs the source decode identity')
    if not isinstance(source, dict) or not isinstance(reference, dict):
        raise ValueError('Receiver binding needs source and reference descriptors')
    if source.get('oriented_size') is not None and _size(source['oriented_size']) != size:
        raise ValueError('Receiver canvas differs from the oriented source')
    if (not isinstance(environment_assets, dict) or
            not {'background', 'validity', 'candidate_removal'} <= set(environment_assets)):
        raise ValueError('Bind the actual background, validity and candidate-removal assets')
    observations = observation_assets or {}
    if not isinstance(observations, dict) or set(observations)-{'foot_owner', 'receiver_observation', 'foreground_alpha'}:
        raise ValueError('Unsupported receiver observation binding')
    if environment_mapping is not None and not isinstance(environment_mapping, dict):
        raise ValueError('Environment mapping must be its current descriptor or unknown')
    return {'source_raw': _asset(source), 'reference': _asset(reference),
        'source_canvas_size': size, 'decode_version': decode_version,
        'environment_assets': {name: _asset(asset) for name, asset in sorted(environment_assets.items())},
        'environment_mapping_sha256': _digest(_content(environment_mapping)) if environment_mapping is not None else None,
        'observation_assets': {name: _asset(asset) for name, asset in sorted(observations.items())}}


def _binding_valid(binding):
    required = {'source_raw', 'reference', 'source_canvas_size', 'decode_version',
                'environment_assets', 'environment_mapping_sha256', 'observation_assets'}
    if not isinstance(binding, dict) or set(binding) != required:
        raise ValueError('Receiver binding is incomplete')
    _size(binding['source_canvas_size'])
    if not isinstance(binding['decode_version'], str) or not binding['decode_version']:
        raise ValueError('Receiver decode identity is missing')
    environments, observations = binding['environment_assets'], binding['observation_assets']
    if (not isinstance(environments, dict) or
            not {'background', 'validity', 'candidate_removal'} <= set(environments) or
            not isinstance(observations, dict) or
            set(observations)-{'foot_owner', 'receiver_observation', 'foreground_alpha'}):
        raise ValueError('Receiver asset binding is incomplete')
    for asset in [binding['source_raw'], binding['reference'], *environments.values(), *observations.values()]:
        if not isinstance(asset, dict) or set(asset) != {'path', 'sha256', 'metadata_sha256'}:
            raise ValueError('Malformed receiver bound asset')
        _asset(asset)
        if not isinstance(asset['metadata_sha256'], str) or not re.fullmatch('[0-9a-f]{64}', asset['metadata_sha256']):
            raise ValueError('Bound receiver asset metadata identity is missing')
    mapping = binding['environment_mapping_sha256']
    if mapping is not None and (not isinstance(mapping, str) or not re.fullmatch('[0-9a-f]{64}', mapping)):
        raise ValueError('Receiver mapping identity is malformed')


def _bound(declaration, binding):
    grounded = GROUNDED_FEET[declaration['foot_stance']]
    complete = bool(grounded) and declaration['ground_type'] != 'unknown' and bool(
        declaration['receiver_region']) and set(declaration['foot_anchors']) == set(grounded)
    sources = {key: declaration['origin'] if present else 'unknown' for key, present in (
        ('foot_stance', declaration['foot_stance'] != 'unknown'),
        ('ground_type', declaration['ground_type'] != 'unknown'),
        ('receiver_region', declaration['receiver_region'] is not None),
        ('receiver_plane', declaration['receiver_plane'] is not None))}
    sources['foot_anchors'] = {foot: declaration['origin'] for foot in declaration['foot_anchors']}
    result = {'schema_version': BOUND_SCHEMA, 'producer': RECIPE,
        'binding': copy.deepcopy(binding), 'declaration': declaration,
        'declaration_sha256': _digest(declaration), 'coordinate_space': COORDINATE_SPACE,
        'geometry_source': 'authored_art_constraint_not_measurement', 'declaration_sources': sources,
        'readiness': {'authored_contact_layout_declared': complete,
            'foot_observation_bound': 'foot_owner' in binding['observation_assets'],
            'receiver_observation_bound': 'receiver_observation' in binding['observation_assets'],
            'anchors_sampled_against_observations': False,
            'relative_contact_height': 'declared_zero_at_grounded_anchors_only' if declaration['foot_anchors'] else 'unknown',
            'plane_interpretation': 'authored_2d_planar_projection_no_metric_scale' if declaration['receiver_plane'] else 'unknown'},
        'execution': copy.deepcopy(EXECUTION)}
    result['identity'] = _digest(_content(result))
    return result


def bind_receiver_intent(value, *, source, reference, source_canvas_size, decode_version,
                         environment_assets, environment_mapping=None, observation_assets=None):
    binding = make_receiver_binding(source=source, reference=reference,
        source_canvas_size=source_canvas_size, decode_version=decode_version,
        environment_assets=environment_assets, environment_mapping=environment_mapping,
        observation_assets=observation_assets)
    declaration = normalize_receiver_intent(value, source_canvas_size=source_canvas_size)
    return _bound(declaration, binding)


def validate_bound_receiver_intent(value, *, expected_binding=None):
    """Validate a bound document; consumers should supply a fresh current binding.

    Omission allows inspection of a standalone saved document only. It is never
    evidence that the declared assets still belong to the active scene.
    """
    if not isinstance(value, dict) or value.get('schema_version') != BOUND_SCHEMA:
        raise ValueError('Unsupported bound receiver intent schema')
    binding = value.get('binding')
    _binding_valid(binding)
    if expected_binding is not None:
        _binding_valid(expected_binding)
        if _content(binding) != _content(expected_binding):
            raise ValueError('Receiver intent belongs to different source/reference/canvas/environment assets')
    declaration = validate_receiver_intent(value.get('declaration'),
        source_canvas_size=binding['source_canvas_size'])
    expected = _bound(declaration, binding)
    if value != expected:
        raise ValueError('Bound receiver intent identity, sources or execution boundary changed')
    return copy.deepcopy(expected)
