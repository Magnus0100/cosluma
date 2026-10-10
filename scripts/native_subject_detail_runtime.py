"""Replay measured subject detail on its exact frozen source-coordinate support.

This consumer never decodes an original donor, applies source retouch again,
or runs a detail filter. Preview reduction follows formation of the native
associated delta; a sparse patch is never resized to the whole canvas.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

KIND = 'source_subject_detail'
SCHEMA = 'native_subject_detail.v1'
RECIPE = 'source_owned_geometry_scaled_log_luminance_bands_v2'
DOMAIN = 'display_linear_sRGB'
REGION_OWNER_CLASSES = {'left_hand': (6,), 'right_hand': (15,),
    'left_arm': (7, 11), 'right_arm': (16, 20),
    'upper_clothing': (1, 23), 'lower_clothing': (13,), 'hair': (4,)}
OPAQUE_ALPHA = .995


def array_sha(values):
    values = np.asarray(values, np.float32)
    digest = hashlib.sha256(json.dumps(list(values.shape), separators=(',', ':')).encode())
    for top in range(0, values.shape[0], 256):
        digest.update(np.ascontiguousarray(values[top:top+256]).tobytes())
    return digest.hexdigest()


def _verify(asset, parent=None):
    from reference_appearance_stack import verify_asset
    bound = dict(asset)
    path = Path(bound.get('path', ''))
    if not path.is_absolute() and parent is not None:
        path = Path(parent)/path
    bound['path'] = str(path.resolve())
    verify_asset(bound)
    return Path(bound['path'])


@lru_cache(maxsize=16)
def _json(path, modified, size, digest):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _read(asset, parent=None):
    path = _verify(asset, parent); stat = path.stat()
    return _json(str(path), stat.st_mtime_ns, stat.st_size, asset['sha256']), path


def _box(value, size):
    if (not isinstance(value, (list, tuple)) or len(value) != 4
            or any(type(item) is not int for item in value)
            or not (0 <= value[0] < value[2] <= size[0]
                    and 0 <= value[1] < value[3] <= size[1])):
        raise ValueError('Subject detail patch must use integer native source coordinates')
    return tuple(value)


def _retouch_identity(binding, parent=None):
    """Compare approved source pixels across JSON/path ownership rewrites.

    The retouch difference is never consumed here. Owner IDs and JSON locator
    hashes can change on relocation; source, domain, approved patch identities
    and the immutable delta must remain identical.
    """
    if binding is None:
        return None
    if not isinstance(binding, dict) or binding.get('schema') != 'automatic_native_retouch_binding.v1':
        raise ValueError('Subject detail approved source retouch binding is invalid')
    layer, path = _read(binding.get('layer') or {}, parent)
    _verify(layer.get('delta') or {}, path.parent)
    return {'binding': {key:value for key,value in binding.items()
            if key not in {'layer','owner_run_id','path','bytes','dependencies'}},
        'source_sha256': (layer.get('source') or {}).get('sha256'),
        'schema_version':layer.get('schema_version'),'recipe':layer.get('recipe'),
        'enabled':layer.get('enabled'),'native_size':layer.get('native_size'),
        'delta_sha256':layer['delta']['sha256'],'patches':layer.get('patches'),
        'clone_retouch':layer.get('clone_retouch'),
        'approval_candidate_ids':[row.get('candidate_id') for row in layer.get('approvals') or []],
        'source_rgb_donor_only':layer.get('source_rgb_donor_only'),
        'reference_rgb_used':layer.get('reference_rgb_used'),'generation_used':layer.get('generation_used')}


@lru_cache(maxsize=8)
def _labels(path, modified, size, digest, proxy):
    with np.load(path, allow_pickle=False) as data:
        labels = data['labels']
    if (labels.shape != (proxy[1],proxy[0]) or not np.issubdtype(labels.dtype,np.integer)
            or labels.min() < 0 or labels.max() >= 29):
        raise ValueError('Subject detail needs exact frozen source semantic labels')
    labels.setflags(write=False)
    return labels


@lru_cache(maxsize=8)
def _patches(path, modified, size, digest, descriptors, native):
    patches = json.loads(descriptors); boxes = []; result = []
    with np.load(path, allow_pickle=False) as data:
        keys = [row.get('array_key') for row in patches]
        if len(set(keys)) != len(keys) or set(data.files) != set(keys) or not keys:
            raise ValueError('Subject detail archive must contain exactly its distinct sparse patch arrays')
        for row in patches:
            box = _box(row.get('native_box'), native)
            if any(box[0] < other[2] and other[0] < box[2]
                    and box[1] < other[3] and other[1] < box[3] for other in boxes):
                raise ValueError('Subject detail sparse patches cannot overlap')
            boxes.append(box)
            values = data[row['array_key']]
            if (values.dtype != np.dtype('float32')
                    or values.shape != (box[3]-box[1], box[2]-box[0], 3)
                    or not np.isfinite(values).all()
                    or array_sha(values) != row.get('delta_array_sha256')):
                raise ValueError('Subject detail patch values differ from the frozen finite native RGB array')
            count = int(np.any(values != 0, axis=-1).sum())
            if type(row.get('changed_pixels')) is not int or count <= 0 or count != row['changed_pixels']:
                raise ValueError('Subject detail changed-pixel count differs from its actual patch')
            if not isinstance(row.get('target_patch_sha256'), str) or len(row['target_patch_sha256']) != 64:
                raise ValueError('Subject detail patch needs its exact measured foreground endpoint')
            values = values.copy(); values.setflags(write=False)
            result.append((row, box, values))
    return tuple(result)


def _load(layer):
    frozen, path = _read((layer.get('assets') or {}).get('frozen_delta', {}))
    receipt, _ = _read((layer.get('assets') or {}).get('qualification_receipt', {}))
    params = layer.get('params') or {}
    if (layer.get('kind') != KIND or any(item.get('schema') != SCHEMA
            or item.get('recipe') != RECIPE for item in (params, frozen, receipt))
            or params.get('binding') != frozen.get('binding')):
        raise ValueError('Subject detail layer must bind the current frozen measured-detail recipe')
    binding = frozen.get('binding') or {}; native = binding.get('native_size')
    if (not isinstance(native, list) or len(native) != 2
            or any(type(item) is not int or item <= 0 for item in native)
            or binding.get('working_domain') != DOMAIN):
        raise ValueError('Subject detail needs a finite native display-linear source canvas')
    delta_path = _verify(frozen.get('delta', {}), path.parent); stat = delta_path.stat()
    patches = _patches(str(delta_path), stat.st_mtime_ns, stat.st_size, frozen['delta']['sha256'],
        json.dumps(frozen.get('patches'), sort_keys=True), tuple(native))
    count = sum(row['changed_pixels'] for row, _, _ in patches)
    qualification = receipt.get('qualification') or {}; checks = receipt.get('checks') or {}
    if (receipt.get('state') != 'prepared' or qualification.get('qualified') is not True
            or qualification.get('actual_paired_QA_verified') is not True
            or checks.get('changed_pixels') != count
            or checks.get('alpha_changed') is not False
            or checks.get('source_geometry_changed') is not False
            or checks.get('approved_source_retouch_reapplied') is not False
            or any(receipt.get(key) != binding.get(key) for key in ('source_sha256', 'reference_sha256'))):
        raise ValueError('Subject detail requires an intact independently qualified measured-detail receipt')
    return frozen, receipt, patches, path


def validate_layer(layer, plan, run=None):
    """Changed coverage or appearance doses bypass; damaged frozen data fail."""
    from design_reconstruction import reference_stack_execution_weights, sample_asset
    frozen, receipt, patches, frozen_path = _load(layer)
    binding = frozen['binding']; native = tuple(binding['native_size'])
    strength = float(layer.get('strength', 1))
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('Subject detail strength must be between zero and one')
    if (plan.get('working_domain') != DOMAIN or tuple(plan.get('canvas', {}).get('source_oriented_size', ())) != native
            or plan.get('source_sha256') != binding['source_sha256']
            or plan.get('candidate_sha256') != binding['reference_sha256']
            or (run and (run.get('source', {}).get('sha256') != binding['source_sha256']
                or run.get('reference', {}).get('sha256') != binding['reference_sha256']))):
        raise ValueError('Subject detail source/reference/native canvas differs from its frozen qualification')
    if sum(row.get('kind') == KIND for row in plan.get('layers', [])) != 1:
        raise ValueError('Exactly one frozen subject detail layer may recover the measured endpoint')
    source_baseline = plan.get('source_baseline') or {}
    if source_baseline.get('schema') != 'appearance_source_baseline.v1' or source_baseline.get('working_domain') != DOMAIN:
        raise ValueError('Subject detail requires the current approved display-linear source baseline')
    if _retouch_identity(source_baseline.get('native_source_retouch')) != _retouch_identity(frozen.get('approved_retouch_binding'), frozen_path.parent):
        raise ValueError('Subject detail approved retouch differs from its frozen donor baseline')
    baseline = source_baseline.get('assets') or {}
    donors = frozen.get('donor_assets') or {}
    for role, key in [('foreground_premultiplied', 'source_fp_sha256'), ('foreground_alpha', 'source_alpha_sha256')]:
        if baseline.get(role, {}).get('sha256') != binding.get(key) or donors.get(role, {}).get('sha256') != binding.get(key):
            raise ValueError('Subject detail donor differs from the approved frozen source baseline')
        if tuple(baseline[role].get('size', ())) != native or tuple(donors[role].get('size', ())) != native:
            raise ValueError('Subject detail donors must share the exact native canvas')
        _verify(baseline[role]); _verify(donors[role], frozen_path.parent)
    semantics = plan.get('mask_roles', {}).get('source_semantics') or {}
    owner = frozen.get('owner_asset') or {}
    if semantics.get('sha256') != binding.get('semantic_sha256') or owner.get('sha256') != binding.get('semantic_sha256'):
        raise ValueError('Subject detail owner differs from its independently bound source semantics')
    semantic_path = _verify(semantics); _verify(owner, frozen_path.parent)
    proxy = tuple(plan.get('canvas', {}).get('source_proxy_size', ()))
    if (len(proxy) != 2 or tuple(semantics.get('size', ())) != proxy
            or tuple(semantics.get('native_size', ())) != native
            or semantics.get('source_sha256') != binding['source_sha256']):
        raise ValueError('Subject detail owner label source/proxy canvas differs from qualification')
    stat = semantic_path.stat(); labels = _labels(str(semantic_path),stat.st_mtime_ns,stat.st_size,semantics['sha256'],proxy)
    windows = receipt.get('reviewed_windows')
    if not isinstance(windows,list) or not windows:
        raise ValueError('Subject detail receipt must retain its actually reviewed owner windows')
    owner_windows = []
    for window in windows:
        region = window.get('region'); owners = window.get('owner_class_ids')
        if (region not in REGION_OWNER_CLASSES or not isinstance(owners,list) or not owners
                or any(type(value) is not int for value in owners)
                or not set(owners) <= set(REGION_OWNER_CLASSES[region])):
            raise ValueError('Subject detail reviewed window has no independent supported source owner')
        owner_windows.append((_box(window.get('native_box'),native),tuple(owners)))
    protection = plan.get('mask_roles',{}).get('identity_protection')
    if protection:_verify(protection)
    for _, box, delta in patches:
        donor_alpha = sample_asset(baseline['foreground_alpha'], 'values', box, native)
        if not np.isfinite(donor_alpha).all() or np.any((donor_alpha < 0) | (donor_alpha > 1)):
            raise ValueError('Subject detail approved donor coverage must be finite between zero and one')
        support = np.any(delta != 0,axis=-1)
        if np.any(support & (donor_alpha < OPAQUE_ALPHA)):
            raise ValueError('Subject detail delta leaks outside approved opaque source ownership')
        l,t,r,b = box
        xx=np.clip(((np.arange(l,r)+.5)*proxy[0]/native[0]).astype(int),0,proxy[0]-1)
        yy=np.clip(((np.arange(t,b)+.5)*proxy[1]/native[1]).astype(int),0,proxy[1]-1)
        patch_labels=labels[np.ix_(yy,xx)]; allowed=np.zeros(support.shape,bool)
        for (wl,wt,wr,wb), owners in owner_windows:
            x,y,rr,bb=max(l,wl),max(t,wt),min(r,wr),min(b,wb)
            if x < rr and y < bb:
                allowed[y-t:bb-t,x-l:rr-l] |= np.isin(patch_labels[y-t:bb-t,x-l:rr-l],owners)
        if np.any(support & ~allowed):
            raise ValueError('Subject detail delta leaks outside reviewed independent source owner windows')
        if protection:
            protect = sample_asset(protection,'protection',box,native)
            if not np.isfinite(protect).all() or np.any(support & (protect >= .01)):
                raise ValueError('Subject detail delta invades independently protected identity features')
    alpha = plan.get('mask_roles', {}).get('foreground_alpha') or {}
    _verify(alpha)
    stacks = [row for row in plan.get('layers', []) if row.get('kind') == 'reference_appearance_stack']
    if len(stacks) != 1:
        raise ValueError('Subject detail requires its unique frozen appearance endpoint')
    stack = stacks[0]; weights = reference_stack_execution_weights(stack)['bank_weights']
    record = {'schema': SCHEMA, 'recipe': RECIPE, 'working_domain': DOMAIN,
        'applied': False, 'bypass_reason': None, 'native_size': list(native),
        'source_geometry_changed': False, 'alpha_changed': False, 'source_retouch_reapplied': False,
        'delta_already_premultiplied': True, 'preview_support_formed_before_reduction': True,
        'changed_native_pixels': sum(row['changed_pixels'] for row, _, _ in patches),
        'execution_stage': 'source_subject_detail_after_appearance_before_manual_foreground',
        'visual_acceptance': False}
    if not layer.get('enabled') or not strength:
        record['bypass_reason'] = 'subject_detail_disabled'
    elif alpha.get('sha256') != binding.get('alpha_sha256') or plan.get('mask_roles', {}).get('edited_edge_light_applicability'):
        record['bypass_reason'] = 'subject_detail_current_alpha_changed_requires_new_qualification'
    elif weights != binding.get('appearance_weights'):
        record['bypass_reason'] = 'subject_detail_appearance_doses_changed_requires_new_qualification'
    elif (run or {}).get('options', {}).get('denoise_strength', 0):
        record['bypass_reason'] = 'subject_detail_user_denoise_changed_requires_new_qualification'
    expected_banks = binding.get('appearance_banks') or {}
    if set(expected_banks) != {role for role, value in binding.get('appearance_weights', {}).items() if value}:
        raise ValueError('Subject detail frozen bank binding omits or adds an active bank')
    for role, digest in expected_banks.items():
        asset = stack.get('assets', {}).get(role) or {}; _verify(asset)
        if asset.get('sha256') != digest:
            raise ValueError('Subject detail appearance bank differs from the measured endpoint')
    if record['bypass_reason']:
        return record
    from reference_appearance_stack import apply_stack
    foreground = next(row for row in plan['layers'] if row['kind'] == 'background_design')['assets']['foreground_premultiplied']
    _verify(foreground)
    for row, box, delta in patches:
        coverage = sample_asset(alpha, 'values', box, native)
        if not np.isfinite(coverage).all() or np.any((coverage < 0) | (coverage > 1)):
            raise ValueError('Subject detail coverage must be finite between zero and one')
        if np.any(np.any(delta != 0, axis=-1) & (coverage < OPAQUE_ALPHA)):
            raise ValueError('Subject detail delta leaks outside current opaque foreground ownership')
        target, _ = apply_stack(sample_asset(foreground, 'values', box, native), stack, plan, run,
            box=box, canvas_size=native)
        if array_sha(target) != row['target_patch_sha256']:
            raise ValueError('Subject detail measured foreground patch changed before frozen recovery')
    return record


@lru_cache(maxsize=2)
def _native_channels(path, modified, size, digest, descriptors, native):
    patches = _patches(path, modified, size, digest, descriptors, native)
    channels = []
    for channel in range(3):
        plane = np.zeros((native[1], native[0]), np.float32)
        for _, (l, t, r, b), values in patches:
            plane[t:b, l:r] = values[..., channel]
        channels.append(Image.fromarray(plane))
    return tuple(channels)


def apply_detail(foreground, layer, plan, run=None, *, box=None, canvas_size=None):
    """Return foreground and truthful execution receipt for full/ROI/preview."""
    values = np.asarray(foreground, np.float32)
    if values.ndim != 3 or values.shape[-1] != 3 or not np.isfinite(values).all():
        raise ValueError('Subject detail foreground must be finite HWC RGB')
    height, width = values.shape[:2]; size = tuple(canvas_size or (width, height))
    target = _box(tuple(box or (0, 0, width, height)), size)
    if target[2]-target[0] != width or target[3]-target[1] != height:
        raise ValueError('Subject detail source-coordinate ROI must match its RGB buffer')
    record = validate_layer(layer, plan, run)
    if record['bypass_reason']:
        return foreground, record
    frozen, _, patches, frozen_path = _load(layer); native = tuple(frozen['binding']['native_size'])
    if (size[0] > native[0] or size[1] > native[1]
            or abs(size[0]*native[1]-size[1]*native[0]) > max(native)):
        raise ValueError('Subject detail preview must be a proportional reduction of its source canvas')
    strength = np.float32(layer.get('strength', 1)); result = values.copy()
    if size == native:
        for _, (l, t, r, b), delta in patches:
            x, y, rr, bb = max(l, target[0]), max(t, target[1]), min(r, target[2]), min(b, target[3])
            if x < rr and y < bb:
                result[y-target[1]:bb-target[1], x-target[0]:rr-target[0]] += delta[y-t:bb-t, x-l:rr-l]*strength
    else:
        from design_reconstruction import _sample_channels
        path = _verify(frozen['delta'], frozen_path.parent); stat = path.stat()
        channels = _native_channels(str(path), stat.st_mtime_ns, stat.st_size, frozen['delta']['sha256'],
            json.dumps(frozen['patches'], sort_keys=True), native)
        result += _sample_channels(channels, target, size)*strength
    record.update(applied=True, replay_canvas_size=list(size), source_coordinate_box=list(target))
    return result, record
