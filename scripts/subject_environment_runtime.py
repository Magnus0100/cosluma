"""Replay qualified low-frequency environment response in source coordinates.

The frozen RGB delta is already associated with foreground coverage. No source
retouch, geometric transform, lighting inference or donor filtering runs here.
Native support is formed before preview reduction.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from native_subject_detail_runtime import (array_sha, _verify, _read, _box, _labels,
    _patches, _native_channels, _retouch_identity)

KIND = 'source_environment_response'
SCHEMA = 'subject_environment_response.v1'
RECIPE = 'final_subject_low_frequency_residual_v1'
DOMAIN = 'display_linear_sRGB'
OWNER_CLASSES = (1, 4, 13, 23)
OPAQUE_ALPHA = .995


def _digest(value):
    def identity(item):
        if isinstance(item, dict):
            if isinstance(item.get('sha256'), str) and item.get('path'):
                return {key: identity(value) for key, value in item.items() if key not in {'path', 'bytes'}}
            return {key: identity(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [identity(value) for value in item]
        return item
    return hashlib.sha256(json.dumps(identity(value), sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf8')).hexdigest()


def _approved_retouch_identity(binding, *, source_sha256, native_size, parent=None):
    """Absent retouch equals only a verified, disabled, pixel-empty operation.

    Child ownership can make a formerly implicit no-op explicit. Counts alone
    cannot prove that identity: the layer, approval list, clone strokes and
    immutable numeric delta are checked before canonicalizing it to absence.
    Actual retouch remains subject to the unchanged frozen identity contract.
    """
    if binding is None:
        return None
    if (not isinstance(binding, dict) or binding.get('schema') != 'automatic_native_retouch_binding.v1'
            or binding.get('source_sha256') != source_sha256
            or tuple(binding.get('native_size', ())) != tuple(native_size)
            or binding.get('working_domain') not in {'scene_linear_sRGB', 'display_linear_sRGB'}):
        raise ValueError('Environment response approved retouch source/native/domain binding is invalid')
    identity = _retouch_identity(binding, parent)
    layer, path = _read(binding.get('layer') or {}, parent)
    if (identity.get('source_sha256') != source_sha256 or tuple(identity.get('native_size', ())) != tuple(native_size)
            or layer.get('schema_version') != 'native_retouch_delta.v1'):
        raise ValueError('Environment response approved retouch layer source/native identity is invalid')
    # Retouch owns the decoded source before the display transform. Its RAW
    # domain is independently established by that frozen decoder, never by the
    # display-linear appearance delta applied later in this consumer.
    from raw_support import RGB_DECODE_VERSION
    decode = layer.get('source', {}).get('decode') or {}
    layer_domain = {'raw16_linear': 'scene_linear_sRGB',
                    'raster8_srgb': 'display_linear_sRGB'}.get(decode.get('source_precision'))
    if (layer.get('recipe') != 'source-native-compact-outlier-two-crop-approval-v1'
            or decode.get('decode_version') != RGB_DECODE_VERSION
            or layer_domain is None or binding['working_domain'] != layer_domain
            or ('working_domain' in layer and layer['working_domain'] != layer_domain)):
        raise ValueError('Environment response approved retouch domain differs from its actual frozen source decode')
    clone = layer.get('clone_retouch') or {}
    proven_empty = (type(binding.get('approved_patches')) is int and binding['approved_patches'] == 0
        and layer.get('enabled') is False and layer.get('patches') == [] and layer.get('approvals') == []
        and clone.get('strokes') == [] and layer.get('source_rgb_donor_only') is True
        and layer.get('reference_rgb_used') is False and layer.get('generation_used') is False)
    if not proven_empty:
        return identity
    delta_path = _verify(layer.get('delta') or {}, path.parent)
    with np.load(delta_path, allow_pickle=False) as arrays:
        for key in arrays.files:
            values = arrays[key]
            if (not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all()
                    or np.any(values != 0)):
                return identity
    return None


def foreground_controls_digest(run):
    """Bind operative controls while ignoring child ownership and JSON paths."""
    from design_reconstruction import _manual_light_plans
    from source_color import source_color_plan_for_run
    options = (run or {}).get('options') or {}; revision = options.get('appearance_revision') or {}
    colour = source_color_plan_for_run(run or {})
    if revision.get('source_color_plan_origin') not in {
            'manual', 'model_with_user_edits', 'local_model_with_user_edits', 'local_automatic_with_user_edits', 'user'}:
        colour = {'enabled': False}
    return _digest({'denoise_strength': options.get('denoise_strength', 0),
        'manual_layers': options.get('manual_layers') or {},
        'appearance_params': revision.get('params') or {},
        'appearance_regions': revision.get('regions'),
        'manual_light_plans': _manual_light_plans(run or {}), 'manual_colour_plan': colour})


def preceding_detail_digest(plan):
    """Identity of the exact preceding detail response, stable on relocation."""
    layers = [row for row in plan.get('layers', []) if row.get('kind') == 'source_subject_detail']
    if len(layers) > 1:
        raise ValueError('Environment response requires at most one preceding source detail response')
    if not layers:
        return None
    from native_subject_detail_runtime import _load as load_detail
    layer = layers[0]; frozen, _, _, path = load_detail(layer)
    return _digest({'schema': frozen['schema'], 'recipe': frozen['recipe'],
        'binding': frozen['binding'], 'delta_sha256': frozen['delta']['sha256'],
        'patches': frozen['patches'], 'enabled': bool(layer.get('enabled')),
        'strength': float(layer.get('strength', 1)),
        'approved_retouch': _retouch_identity(frozen.get('approved_retouch_binding'), path.parent)})


def scene_context_digest(plan):
    """Bind visible background/effects without binding JSON locator hashes.

    Numeric arrays retain immutable byte identities. JSON ownership changes
    are followed to their semantic contents, so materialization and portable
    relocation alone cannot invalidate a photographic qualification.
    """
    ignored = {'path', 'bytes', 'owner_run_id', 'target_run_id', 'run_id',
        'dependencies', 'provenance', 'producer', 'created_at', 'updated_at', 'prepared_at'}
    def content(item, parent=None):
        if isinstance(item, dict):
            if isinstance(item.get('sha256'), str) and item.get('path'):
                path = _verify(item, parent)
                if path.suffix.lower() == '.json':
                    value, _ = _read(item, parent)
                    return {'json_contents': content(value, path.parent)}
                return {key: content(value, parent) for key, value in item.items() if key not in ignored}
            return {key: content(value, parent) for key, value in item.items() if key not in ignored}
        if isinstance(item, (list, tuple)):
            return [content(value, parent) for value in item]
        return item
    result = []
    for layer in plan.get('layers', []):
        kind = layer.get('kind'); strength = float(layer.get('strength', 1))
        if not np.isfinite(strength):
            raise ValueError('Environment context strength must be finite')
        if kind == 'background_design':
            assets = layer.get('assets') or {}
            result.append({'kind': kind, 'enabled': bool(layer.get('enabled')), 'strength': strength,
                'assets': {role: content(assets[role]) for role in ('background', 'validity', 'old_environment') if role in assets}})
        elif kind in {'contact_shadow', 'authored_receiver_shadow', 'rim_light', 'atmosphere', 'particles', 'bloom', 'independent_effect'}:
            if not layer.get('enabled') or not strength:
                continue
            result.append({'kind': kind, 'strength': strength, 'params': content(layer.get('params') or {}),
                'assets': {role: content(value) for role, value in (layer.get('assets') or {}).items()
                    if role not in {'qualification_receipt', 'review_receipt'}}})
    return _digest(result)


def _load(layer):
    frozen, path = _read((layer.get('assets') or {}).get('frozen_delta', {}))
    receipt, _ = _read((layer.get('assets') or {}).get('qualification_receipt', {}))
    params = layer.get('params') or {}
    if (layer.get('kind') != KIND or any(item.get('schema') != SCHEMA or item.get('recipe') != RECIPE
            for item in (params, frozen, receipt)) or params.get('binding') != frozen.get('binding')):
        raise ValueError('Environment response must bind its current qualified frozen recipe')
    binding = frozen.get('binding') or {}; native = binding.get('native_size')
    if (not isinstance(native, list) or len(native) != 2
            or any(type(value) is not int or value <= 0 for value in native)
            or binding.get('working_domain') != DOMAIN
            or not isinstance(binding.get('foreground_controls_digest'), str)
            or len(binding['foreground_controls_digest']) != 64
            or not isinstance(binding.get('scene_context_digest'), str) or len(binding['scene_context_digest']) != 64
            or 'preceding_detail_digest' not in binding or 'identity_protection_sha256' not in binding):
        raise ValueError('Environment response needs its complete native display-linear endpoint binding')
    delta_path = _verify(frozen.get('delta', {}), path.parent); stat = delta_path.stat()
    patches = _patches(str(delta_path), stat.st_mtime_ns, stat.st_size, frozen['delta']['sha256'],
        json.dumps(frozen.get('patches'), sort_keys=True), tuple(native))
    count = sum(row['changed_pixels'] for row, _, _ in patches)
    qualification = receipt.get('qualification') or {}; checks = receipt.get('checks') or {}
    if (receipt.get('state') != 'prepared'
            or any(qualification.get(key) is not True for key in
                ('qualified', 'actual_saved_evidence_verified', 'holdout_improved'))
            or checks.get('changed_pixels') != count
            or any(checks.get(key) is not False for key in
                ('alpha_changed', 'source_geometry_changed', 'approved_source_retouch_reapplied',
                 'unknown_support_changed', 'reference_texture_imported'))
            or any(receipt.get(key) != binding.get(key) for key in ('source_sha256', 'reference_sha256'))):
        raise ValueError('Environment response needs intact saved-evidence and holdout qualification')
    return frozen, receipt, patches, path


def validate_layer(layer, plan, run=None):
    """Control changes bypass response; damaged pixels and ownership reject."""
    from design_reconstruction import reference_stack_execution_weights, sample_asset
    frozen, _, patches, frozen_path = _load(layer)
    binding = frozen['binding']; native = tuple(binding['native_size'])
    strength = float(layer.get('strength', 1))
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('Environment response strength must be between zero and one')
    if (plan.get('working_domain') != DOMAIN
            or tuple(plan.get('canvas', {}).get('source_oriented_size', ())) != native
            or plan.get('source_sha256') != binding['source_sha256']
            or plan.get('candidate_sha256') != binding['reference_sha256']
            or (run and (run.get('source', {}).get('sha256') != binding['source_sha256']
                or run.get('reference', {}).get('sha256') != binding['reference_sha256']))):
        raise ValueError('Environment response source/reference/native coordinates differ from qualification')
    kinds = [row.get('kind') for row in plan.get('layers', [])]
    if kinds.count(KIND) != 1 or kinds.count('reference_appearance_stack') != 1:
        raise ValueError('Environment response requires one response and its unique preceding appearance stack')
    if kinds.index('reference_appearance_stack') > kinds.index(KIND) or (
            'source_subject_detail' in kinds and kinds.index('source_subject_detail') > kinds.index(KIND)):
        raise ValueError('Environment response must follow frozen appearance and source detail')
    baseline = plan.get('source_baseline') or {}
    if baseline.get('schema') != 'appearance_source_baseline.v1' or baseline.get('working_domain') != DOMAIN:
        raise ValueError('Environment response requires the approved display-linear source baseline')
    if _approved_retouch_identity(baseline.get('native_source_retouch'),
            source_sha256=binding['source_sha256'], native_size=native) != _approved_retouch_identity(
            frozen.get('approved_retouch_binding'), source_sha256=binding['source_sha256'],
            native_size=native, parent=frozen_path.parent):
        raise ValueError('Environment response approved source retouch binding changed')
    assets = baseline.get('assets') or {}; donors = frozen.get('donor_assets') or {}
    for role, key in [('foreground_premultiplied', 'source_fp_sha256'), ('foreground_alpha', 'source_alpha_sha256')]:
        if assets.get(role, {}).get('sha256') != binding.get(key) or donors.get(role, {}).get('sha256') != binding.get(key):
            raise ValueError('Environment response approved source baseline differs from qualification')
        if tuple(assets[role].get('size', ())) != native or tuple(donors[role].get('size', ())) != native:
            raise ValueError('Environment response source donors must use its native canvas')
        _verify(assets[role]); _verify(donors[role], frozen_path.parent)
    foreground = next(row for row in plan['layers'] if row['kind'] == 'background_design')['assets']['foreground_premultiplied']
    _verify(foreground)
    if foreground.get('sha256') != binding['source_fp_sha256']:
        raise ValueError('Environment response current foreground differs from its approved source baseline')
    semantics = plan.get('mask_roles', {}).get('source_semantics') or {}; owner = frozen.get('owner_asset') or {}
    if semantics.get('sha256') != binding.get('semantic_sha256') or owner.get('sha256') != binding.get('semantic_sha256'):
        raise ValueError('Environment response owner differs from frozen independent source semantics')
    semantic_path = _verify(semantics); _verify(owner, frozen_path.parent)
    proxy = tuple(plan.get('canvas', {}).get('source_proxy_size', ()))
    if (len(proxy) != 2 or tuple(semantics.get('size', ())) != proxy
            or tuple(semantics.get('native_size', ())) != native
            or semantics.get('source_sha256') != binding['source_sha256']):
        raise ValueError('Environment response semantic source/proxy binding changed')
    stat = semantic_path.stat()
    labels = _labels(str(semantic_path), stat.st_mtime_ns, stat.st_size, semantics['sha256'], proxy)
    protection = plan.get('mask_roles', {}).get('identity_protection')
    if protection:
        _verify(protection)
    # Frozen ownership is checked even when changed controls bypass rendering.
    for row, box, delta in patches:
        support = np.any(delta != 0, axis=-1)
        donor_alpha = sample_asset(assets['foreground_alpha'], 'values', box, native)
        if (not np.isfinite(donor_alpha).all() or np.any((donor_alpha < 0) | (donor_alpha > 1))
                or np.any(support & (donor_alpha < OPAQUE_ALPHA))):
            raise ValueError('Environment response leaks outside approved opaque source ownership')
        l, t, r, b = box
        xx = np.clip(((np.arange(l, r)+.5)*proxy[0]/native[0]).astype(int), 0, proxy[0]-1)
        yy = np.clip(((np.arange(t, b)+.5)*proxy[1]/native[1]).astype(int), 0, proxy[1]-1)
        patch_labels = labels[np.ix_(yy, xx)]
        owner_class = row.get('owner_class_id')
        if owner_class is not None:
            if type(owner_class) is not int or owner_class not in OWNER_CLASSES:
                raise ValueError('Environment response patch has an unsupported independent material owner')
            allowed = patch_labels == owner_class
        else:
            allowed = np.isin(patch_labels, OWNER_CLASSES)
        if np.any(support & ~allowed):
            raise ValueError('Environment response invades skin, unknown ownership or background')
        if protection:
            protect = sample_asset(protection, 'protection', box, native)
            if not np.isfinite(protect).all() or np.any(support & (protect >= .01)):
                raise ValueError('Environment response invades protected identity features')
    alpha = plan.get('mask_roles', {}).get('foreground_alpha') or {}; _verify(alpha)
    stack = next(row for row in plan['layers'] if row['kind'] == 'reference_appearance_stack')
    weights = reference_stack_execution_weights(stack)['bank_weights']
    expected_banks = binding.get('appearance_banks') or {}
    if set(expected_banks) != {role for role, value in binding.get('appearance_weights', {}).items() if value}:
        raise ValueError('Environment response active appearance-bank binding is incomplete')
    for role, digest in expected_banks.items():
        asset = stack.get('assets', {}).get(role) or {}; _verify(asset)
        if asset.get('sha256') != digest:
            raise ValueError('Environment response appearance bank changed since qualification')
    record = {'schema': SCHEMA, 'recipe': RECIPE, 'working_domain': DOMAIN, 'applied': False,
        'bypass_reason': None, 'native_size': list(native), 'source_geometry_changed': False,
        'alpha_changed': False, 'source_retouch_reapplied': False, 'delta_already_premultiplied': True,
        'preview_support_formed_before_reduction': True, 'visual_acceptance': False,
        'changed_native_pixels': sum(row['changed_pixels'] for row, _, _ in patches),
        'execution_stage': 'source_environment_response_after_appearance_and_detail_before_manual_foreground'}
    if not layer.get('enabled') or not strength:
        record['bypass_reason'] = 'environment_response_disabled'
    elif alpha.get('sha256') != binding.get('alpha_sha256') or plan.get('mask_roles', {}).get('edited_edge_light_applicability'):
        record['bypass_reason'] = 'environment_response_current_alpha_changed_requires_new_qualification'
    elif ((protection and protection.get('sha256') != binding.get('identity_protection_sha256')) or
            (protection is None and binding.get('identity_protection_sha256') is not None)):
        record['bypass_reason'] = 'environment_response_identity_protection_changed_requires_new_qualification'
    elif weights != binding.get('appearance_weights'):
        record['bypass_reason'] = 'environment_response_appearance_doses_changed_requires_new_qualification'
    elif (run or {}).get('options', {}).get('denoise_strength', 0):
        record['bypass_reason'] = 'environment_response_user_denoise_changed_requires_new_qualification'
    elif preceding_detail_digest(plan) != binding.get('preceding_detail_digest'):
        record['bypass_reason'] = 'environment_response_preceding_detail_changed_requires_new_qualification'
    elif foreground_controls_digest(run) != binding.get('foreground_controls_digest'):
        record['bypass_reason'] = 'environment_response_foreground_controls_changed_requires_new_qualification'
    elif scene_context_digest(plan) != binding.get('scene_context_digest'):
        record['bypass_reason'] = 'environment_response_scene_context_changed_requires_new_qualification'
    if record['bypass_reason']:
        return record
    from reference_appearance_stack import apply_stack
    from native_subject_detail_runtime import apply_detail
    details = [row for row in plan['layers'] if row['kind'] == 'source_subject_detail']
    for row, box, delta in patches:
        coverage = sample_asset(alpha, 'values', box, native)
        if (not np.isfinite(coverage).all() or np.any((coverage < 0) | (coverage > 1))
                or np.any(np.any(delta != 0, axis=-1) & (coverage < OPAQUE_ALPHA))):
            raise ValueError('Environment response leaks outside current opaque foreground ownership')
        target, _ = apply_stack(sample_asset(foreground, 'values', box, native), stack, plan, run, box=box, canvas_size=native)
        if details:
            target, _ = apply_detail(target, details[0], plan, run, box=box, canvas_size=native)
        if array_sha(target) != row['target_patch_sha256']:
            raise ValueError('Environment response measured foreground endpoint changed')
        active = np.any(delta != 0, axis=-1)
        corrected = (target[active]+delta[active])/coverage[active, None]
        if (not np.isfinite(corrected).all() or np.any(corrected < -1e-6)
                or np.any(corrected > 1+1e-6)):
            raise ValueError('Environment response frozen correction violates its qualified display RGB range')
    return record


def apply_environment(foreground, layer, plan, run=None, *, box=None, canvas_size=None):
    values = np.asarray(foreground, np.float32)
    if values.ndim != 3 or values.shape[-1] != 3 or not np.isfinite(values).all():
        raise ValueError('Environment response foreground must be finite HWC RGB')
    height, width = values.shape[:2]; size = tuple(canvas_size or (width, height))
    target = _box(tuple(box or (0, 0, width, height)), size)
    if target[2]-target[0] != width or target[3]-target[1] != height:
        raise ValueError('Environment response source ROI must match its RGB buffer')
    record = validate_layer(layer, plan, run)
    if record['bypass_reason']:
        return foreground, record
    frozen, _, patches, frozen_path = _load(layer); native = tuple(frozen['binding']['native_size'])
    if (size[0] > native[0] or size[1] > native[1]
            or abs(size[0]*native[1]-size[1]*native[0]) > max(native)):
        raise ValueError('Environment response preview must proportionally reduce its native source canvas')
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
    if not np.isfinite(result).all():
        raise ValueError('Environment response produced nonfinite foreground')
    record.update(applied=True, replay_canvas_size=list(size), source_coordinate_box=list(target))
    return result, record
