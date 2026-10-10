"""Replay one bound native hair-light delta before manual foreground finishing."""
from __future__ import annotations
import json
import numpy as np
from native_subject_detail_runtime import _read, _verify, _patches, _native_channels, _box, _labels, array_sha
from subject_environment_runtime import (_digest, foreground_controls_digest,
    preceding_detail_digest, scene_context_digest, _approved_retouch_identity)

KIND = 'source_boundary_light'
SCHEMA = 'source_boundary_light.v1'
RECIPE = 'reference_hair_boundary_observed_response_v2'


def source_owners(plan, box):
    """Global pixel-centre categorical sampling, invariant to ROI crop size."""
    record = plan['mask_roles']['source_semantics']; path = _verify(record); stat = path.stat()
    proxy = tuple(plan['canvas']['source_proxy_size']); native = tuple(plan['canvas']['source_oriented_size'])
    labels = _labels(str(path), stat.st_mtime_ns, stat.st_size, record['sha256'], proxy)
    l, t, r, b = box
    x = np.clip(((np.arange(l, r)+.5)*proxy[0]/native[0]).astype(int), 0, proxy[0]-1)
    y = np.clip(((np.arange(t, b)+.5)*proxy[1]/native[1]).astype(int), 0, proxy[1]-1)
    return labels[np.ix_(y, x)]


def endpoint(plan, run, box, size):
    from subject_environment_worker import endpoint as appearance_endpoint
    from subject_environment_runtime import apply_environment
    values = appearance_endpoint(plan, run, box, size)
    for layer in plan['layers']:
        if layer['kind'] == 'source_environment_response' and layer.get('enabled') and layer.get('strength', 1):
            values, _ = apply_environment(values, layer, plan, run, box=box, canvas_size=size)
    return values


def endpoint_binding(plan, run):
    from design_reconstruction import reference_stack_execution_weights
    stack = next(row for row in plan['layers'] if row['kind'] == 'reference_appearance_stack')
    weights = reference_stack_execution_weights(stack)['bank_weights']
    environment = []
    for layer in plan['layers']:
        if layer['kind'] == 'source_environment_response':
            frozen, _ = _read(layer['assets']['frozen_delta'])
            environment.append({'binding': frozen['binding'], 'delta_sha256': frozen['delta']['sha256'],
                'patches': frozen['patches'], 'enabled': layer.get('enabled'), 'strength': layer.get('strength', 1)})
    baseline = plan['source_baseline']; native = plan['canvas']['source_oriented_size']
    protection = plan['mask_roles'].get('identity_protection')
    foreground = next(row for row in plan['layers'] if row['kind'] == 'background_design')['assets']['foreground_premultiplied']
    return {'source_sha256': plan['source_sha256'], 'reference_sha256': plan['candidate_sha256'],
        'native_size': native, 'working_domain': plan['working_domain'],
        'foreground_sha256': foreground['sha256'], 'alpha_sha256': plan['mask_roles']['foreground_alpha']['sha256'],
        'semantic_sha256': plan['mask_roles']['source_semantics']['sha256'],
        'identity_protection_sha256': protection['sha256'] if protection else None,
        'appearance_weights': weights, 'appearance_banks': {k: stack['assets'][k]['sha256'] for k, v in weights.items() if v},
        'preceding_detail_digest': preceding_detail_digest(plan), 'environment_digest': _digest(environment),
        'scene_context_digest': scene_context_digest(plan), 'foreground_controls_digest': foreground_controls_digest(run),
        'approved_retouch': _approved_retouch_identity(baseline.get('native_source_retouch'),
            source_sha256=plan['source_sha256'], native_size=native)}


def _load(layer):
    frozen, path = _read(layer['assets']['frozen_delta'])
    receipt, _ = _read(layer['assets']['qualification_receipt'])
    if (layer.get('kind') != KIND or any(item.get('schema') != SCHEMA or item.get('recipe') != RECIPE
            for item in (frozen, receipt, layer.get('params', {})))
            or layer['params'].get('binding') != frozen.get('binding') or receipt.get('state') != 'prepared'):
        raise ValueError('Boundary light requires the current qualified native recipe')
    native = tuple(frozen['binding']['native_size']); delta = _verify(frozen['delta'], path.parent); stat = delta.stat()
    patches = _patches(str(delta), stat.st_mtime_ns, stat.st_size, frozen['delta']['sha256'],
        json.dumps(frozen['patches'], sort_keys=True), native)
    if sum(row['changed_pixels'] for row, _, _ in patches) != receipt.get('changed_pixels'):
        raise ValueError('Boundary light receipt differs from frozen pixels')
    for asset in frozen.get('observation_assets', {}).values(): _verify(asset, path.parent)
    return frozen, path, patches


def validate_layer(layer, plan, run=None):
    from design_reconstruction import sample_asset
    frozen, path, patches = _load(layer); binding = frozen['binding']; native = tuple(binding['native_size'])
    strength = float(layer.get('strength', 1))
    if not np.isfinite(strength) or not 0 <= strength <= 1: raise ValueError('Boundary light strength must be [0,1]')
    if (plan['source_sha256'] != binding['source_sha256'] or plan['candidate_sha256'] != binding['reference_sha256']
            or tuple(plan['canvas']['source_oriented_size']) != native or plan['working_domain'] != 'display_linear_sRGB'
            or (run and (run['source']['sha256'] != binding['source_sha256'] or run['reference']['sha256'] != binding['reference_sha256']))):
        raise ValueError('Boundary light belongs to a different source/reference/native canvas')
    if sum(row['kind'] == KIND for row in plan['layers']) != 1: raise ValueError('Boundary light cannot be stacked repeatedly')
    record = {'kind': KIND, 'applied': False, 'visual_acceptance': False,
        'source_alpha_changed': False, 'source_geometry_changed': False, 'clothing_enabled': False,
        'reference_appearance_donor': True, 'bypass_reason': None}
    if endpoint_binding(plan, run) != binding or plan['mask_roles'].get('edited_edge_light_applicability'):
        record['bypass_reason'] = 'boundary_light_endpoint_changed_requires_new_preparation'
        return record
    for row, box, delta in patches:
        a = sample_asset(plan['mask_roles']['foreground_alpha'], 'values', box, native)
        labels = source_owners(plan, box)
        active = np.any(delta != 0, -1)
        if np.any(delta < 0) or np.any(active & ((a <= 0)|(labels != 4))):
            raise ValueError('Boundary light darkens pixels or crosses source hair ownership')
        protect = plan['mask_roles'].get('identity_protection')
        if protect and np.any(active & (sample_asset(protect, 'protection', box, native) >= .5)):
            raise ValueError('Boundary light crosses protected identity')
        before = endpoint(plan, run, box, native)
        if array_sha(before) != row['target_patch_sha256'] or np.any(before+delta > a[..., None]+2e-5):
            raise ValueError('Boundary light native endpoint or associated RGB range changed')
    if not layer.get('enabled') or not strength: record['bypass_reason'] = 'boundary_light_disabled'
    return record


def apply_boundary_light(foreground, layer, plan, run=None, *, box=None, canvas_size=None):
    values = np.asarray(foreground, np.float32); h, w = values.shape[:2]
    if values.ndim != 3 or values.shape[-1] != 3 or not np.isfinite(values).all(): raise ValueError('Invalid boundary foreground')
    size = tuple(canvas_size or (w, h)); target = _box(list(box or (0, 0, w, h)), size)
    if (target[2]-target[0], target[3]-target[1]) != (w, h): raise ValueError('Boundary light ROI size differs')
    record = validate_layer(layer, plan, run)
    if record['bypass_reason']: return foreground, record
    frozen, path, patches = _load(layer); native = tuple(frozen['binding']['native_size']); result = values.copy()
    strength = np.float32(layer.get('strength', 1))
    if size == native:
        for _, (l, t, r, b), delta in patches:
            x, y, rr, bb = max(l, target[0]), max(t, target[1]), min(r, target[2]), min(b, target[3])
            if x < rr and y < bb:
                result[y-target[1]:bb-target[1], x-target[0]:rr-target[0]] += delta[y-t:bb-t, x-l:rr-l]*strength
    else:
        if size[0] > native[0] or size[1] > native[1] or abs(size[0]*native[1]-size[1]*native[0]) > max(native):
            raise ValueError('Boundary preview must proportionally reduce the native canvas')
        from design_reconstruction import _sample_channels
        delta_path = _verify(frozen['delta'], path.parent); stat = delta_path.stat()
        channels = _native_channels(str(delta_path), stat.st_mtime_ns, stat.st_size, frozen['delta']['sha256'],
            json.dumps(frozen['patches'], sort_keys=True), native)
        result += _sample_channels(channels, target, size)*strength
    record.update(applied=True, replay_canvas_size=list(size), source_coordinate_box=list(target))
    return result, record
