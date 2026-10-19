"""Replay one frozen cloth-light delta after current hair and before finishing.

The current v2 hair endpoint and asset contract remain unchanged. Cloth owns an
independent delta whose qualification includes the preceding hair replay state.
"""
from __future__ import annotations

import json
import numpy as np

import boundary_light_runtime as hair
from native_subject_detail_runtime import _read, _verify, _patches, _native_channels, _box, array_sha
from subject_environment_runtime import _digest

KIND = 'source_cloth_boundary_light'
SCHEMA = 'source_cloth_boundary_light.v1'
RECIPE = 'reference_cloth_boundary_material_response_v1'
OWNER_CLASSES = (23,)
OPAQUE_ALPHA_MIN = .95


def preceding_hair_identity(plan, run=None):
    layers = [row for row in plan['layers'] if row['kind'] == hair.KIND]
    if len(layers) > 1:
        raise ValueError('Cloth light requires at most one preceding hair layer')
    if not layers:
        return None
    layer = layers[0]
    frozen, _, _ = hair._load(layer)
    state = hair.validate_layer(layer, plan, run)
    return {'schema': frozen['schema'], 'recipe': frozen['recipe'],
        'binding': frozen['binding'], 'delta_sha256': frozen['delta']['sha256'],
        'patches': frozen['patches'], 'enabled': bool(layer.get('enabled')),
        'strength': float(layer.get('strength', 1)), 'bypass_reason': state['bypass_reason']}


def endpoint_binding(plan, run=None):
    result = hair.endpoint_binding(plan, run)
    result['preceding_hair_digest'] = _digest(preceding_hair_identity(plan, run))
    return result


def endpoint_binding_for_reuse(plan, run=None):
    """Build the cloth endpoint identity without replaying the hair ROI.

    ``endpoint_binding`` is also used by full layer validation, where
    ``preceding_hair_identity`` deliberately calls ``hair.validate_layer`` to
    compare every native target patch.  A retry fast path has already loaded
    the frozen cloth assets and only needs to establish that the preceding
    hair layer identity is unchanged; replaying a 3k x 6k hair patch here
    repeated the expensive validation before the cache decision.  The frozen
    hair binding and controls digest are still compared exactly, and the
    normal validator remains the owner of the full pixel contract whenever a
    layer is newly prepared or applied.
    """
    result = hair.endpoint_binding(plan, run)
    layers = [row for row in plan['layers'] if row['kind'] == hair.KIND]
    if len(layers) > 1:
        raise ValueError('Cloth light requires at most one preceding hair layer')
    if not layers:
        identity = None
    else:
        layer = layers[0]
        frozen, _ = hair._read(layer['assets']['frozen_delta'])
        receipt, _ = hair._read(layer['assets']['qualification_receipt'])
        if (layer.get('params', {}).get('binding') != frozen.get('binding')
                or receipt.get('state') not in ('prepared', 'no_change')):
            raise ValueError('Preceding hair layer is not an intact frozen identity')
        identity = {'schema': frozen['schema'], 'recipe': frozen['recipe'],
            'binding': frozen['binding'], 'delta_sha256': frozen['delta']['sha256'],
            'patches': frozen['patches'], 'enabled': bool(layer.get('enabled')),
            'strength': float(layer.get('strength', 1)),
            'bypass_reason': ('boundary_light_disabled'
                              if not layer.get('enabled') or not layer.get('strength', 1)
                              else None)}
    result['preceding_hair_digest'] = _digest(identity)
    return result


def endpoint(plan, run, box, size):
    values = hair.endpoint(plan, run, box, size)
    layers = [row for row in plan['layers'] if row['kind'] == hair.KIND]
    if len(layers) > 1:
        raise ValueError('Cloth light requires at most one preceding hair layer')
    for layer in layers:
        if layer.get('enabled') and layer.get('strength', 1):
            values, state = hair.apply_boundary_light(values, layer, plan, run, box=box, canvas_size=size)
            if not state['applied']:
                raise ValueError('Current active hair endpoint requires new preparation before cloth light')
    return values


def _load(layer):
    frozen, path = _read(layer['assets']['frozen_delta'])
    receipt, _ = _read(layer['assets']['qualification_receipt'])
    if (layer.get('kind') != KIND or any(item.get('schema') != SCHEMA or item.get('recipe') != RECIPE
            for item in (frozen, receipt, layer.get('params', {})))
            or layer['params'].get('binding') != frozen.get('binding') or receipt.get('state') != 'prepared'):
        raise ValueError('Cloth light requires the current qualified native recipe')
    binding = frozen['binding']
    if binding.get('working_domain') != 'display_linear_sRGB' or 'preceding_hair_digest' not in binding:
        raise ValueError('Cloth light needs its complete display-linear hair endpoint binding')
    native = tuple(binding['native_size']); delta = _verify(frozen['delta'], path.parent); stat = delta.stat()
    patches = _patches(str(delta), stat.st_mtime_ns, stat.st_size, frozen['delta']['sha256'],
        json.dumps(frozen['patches'], sort_keys=True), native)
    if sum(row['changed_pixels'] for row, _, _ in patches) != receipt.get('changed_pixels'):
        raise ValueError('Cloth light receipt differs from frozen pixels')
    for asset in frozen.get('observation_assets', {}).values():
        _verify(asset, path.parent)
    return frozen, path, patches


def validate_layer(layer, plan, run=None):
    from design_reconstruction import sample_asset
    frozen, _, patches = _load(layer); binding = frozen['binding']; native = tuple(binding['native_size'])
    strength = float(layer.get('strength', 1))
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('Cloth light strength must be [0,1]')
    if (plan['source_sha256'] != binding['source_sha256'] or plan['candidate_sha256'] != binding['reference_sha256']
            or tuple(plan['canvas']['source_oriented_size']) != native or plan['working_domain'] != 'display_linear_sRGB'
            or (run and (run['source']['sha256'] != binding['source_sha256'] or run['reference']['sha256'] != binding['reference_sha256']))):
        raise ValueError('Cloth light belongs to a different source/reference/native canvas')
    kinds = [row['kind'] for row in plan['layers']]
    if kinds.count(KIND) != 1 or kinds.count('reference_appearance_stack') != 1 or kinds.count(hair.KIND) > 1:
        raise ValueError('Cloth light needs one cloth response and a unique preceding appearance/hair endpoint')
    for kind in ('reference_appearance_stack', 'source_subject_detail', 'source_environment_response', hair.KIND):
        if kind in kinds and kinds.index(kind) > kinds.index(KIND):
            raise ValueError('Cloth light must follow appearance, detail, environment and hair')
    record = {'kind': KIND, 'applied': False, 'visual_acceptance': False,
        'source_alpha_changed': False, 'source_geometry_changed': False, 'hair_changed': False,
        'owner_classes': list(OWNER_CLASSES), 'delta_already_premultiplied': True,
        'minimum_source_alpha': OPAQUE_ALPHA_MIN, 'partial_alpha_response_supported': False,
        'bypass_reason': None, 'execution_stage': 'source_cloth_boundary_light_after_hair_before_manual_foreground'}
    if endpoint_binding(plan, run) != binding or plan['mask_roles'].get('edited_edge_light_applicability'):
        record['bypass_reason'] = 'cloth_boundary_endpoint_changed_requires_new_preparation'
        return record
    for row, box, delta in patches:
        a = sample_asset(plan['mask_roles']['foreground_alpha'], 'values', box, native)
        labels = hair.source_owners(plan, box); active = np.any(delta != 0, -1)
        if (not np.isfinite(a).all() or np.any((a < 0)|(a > 1)) or np.any(delta < 0)
                or np.any(active & ((a < OPAQUE_ALPHA_MIN)|~np.isin(labels, OWNER_CLASSES)))):
            raise ValueError('Cloth light darkens pixels or crosses opaque source outer-clothing ownership')
        protect = plan['mask_roles'].get('identity_protection')
        if protect:
            protection = sample_asset(protect, 'protection', box, native)
            if not np.isfinite(protection).all() or np.any(active & (protection >= .5)):
                raise ValueError('Cloth light crosses protected identity')
        before = endpoint(plan, run, box, native)
        if (array_sha(before) != row['target_patch_sha256'] or not np.isfinite(before).all()
                or np.any(before+delta < -2e-5)
                or np.any(np.any(delta != 0, axis=-1)[..., None] & (before+delta > a[..., None]+2e-5))):
            raise ValueError('Cloth light native endpoint or associated RGB range changed')
    if not layer.get('enabled') or not strength:
        record['bypass_reason'] = 'cloth_boundary_light_disabled'
    return record


def apply_cloth_boundary_light(foreground, layer, plan, run=None, *, box=None, canvas_size=None):
    values = np.asarray(foreground, np.float32)
    if values.ndim != 3 or values.shape[-1] != 3 or not np.isfinite(values).all():
        raise ValueError('Invalid cloth boundary foreground')
    h, w = values.shape[:2]; size = tuple(canvas_size or (w, h))
    target = _box(list(box or (0, 0, w, h)), size)
    if (target[2]-target[0], target[3]-target[1]) != (w, h):
        raise ValueError('Cloth boundary ROI size differs')
    record = validate_layer(layer, plan, run)
    if record['bypass_reason']:
        return foreground, record
    frozen, path, patches = _load(layer); native = tuple(frozen['binding']['native_size']); result = values.copy()
    strength = np.float32(layer.get('strength', 1))
    if size == native:
        for _, (l, t, r, b), delta in patches:
            x, y, rr, bb = max(l, target[0]), max(t, target[1]), min(r, target[2]), min(b, target[3])
            if x < rr and y < bb:
                result[y-target[1]:bb-target[1], x-target[0]:rr-target[0]] += delta[y-t:bb-t, x-l:rr-l]*strength
    else:
        if size[0] > native[0] or size[1] > native[1] or abs(size[0]*native[1]-size[1]*native[0]) > max(native):
            raise ValueError('Cloth preview must proportionally reduce the native canvas')
        from design_reconstruction import _sample_channels
        delta_path = _verify(frozen['delta'], path.parent); stat = delta_path.stat()
        channels = _native_channels(str(delta_path), stat.st_mtime_ns, stat.st_size, frozen['delta']['sha256'],
            json.dumps(frozen['patches'], sort_keys=True), native)
        result += _sample_channels(channels, target, size)*strength
    record.update(applied=True, replay_canvas_size=list(size), source_coordinate_box=list(target))
    return result, record
