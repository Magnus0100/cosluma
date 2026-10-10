"""Replay a frozen native original/reference appearance delta.

Preparation owns colour matching and blending. Replay consumes only immutable
float assets in display-linear sRGB, without models or producer dependencies.
The delta is already associated with original coverage; never multiply alpha
again. Matte edits may retain its original support before preview reduction.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path

import numpy as np

KIND = 'reference_appearance_stack'
SCHEMA = 'reference_appearance_stack.v2'
CONTROL_RECIPE = 'premultiplied_oklab_four_corner.v1'
WORKING_DOMAIN = 'display_linear_sRGB'
PREPARED_STAGES = ('reference_colour_match', 'reference_colour_blend',
                   'reference_scene_balance', 'edge_donor_extension')


@lru_cache(maxsize=64)
def _file_hash(path, modified, size, inode):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_asset(asset):
    """Content identity survives materialization and portable relocation."""
    path = Path(asset.get('path', ''))
    if not path.is_file() or not asset.get('sha256'):
        raise ValueError('Frozen appearance asset is missing; restore the saved asset or prepare a new version')
    stat = path.stat()
    digest = _file_hash(str(path.resolve()), stat.st_mtime_ns, stat.st_size, stat.st_ino)
    if digest != asset['sha256']:
        raise ValueError('Frozen appearance asset bytes changed; restore the saved asset or prepare a new version')
    return digest


@lru_cache(maxsize=8)
def _validate_delta(path, modified, byte_size, digest, native_size):
    from design_reconstruction import _planes
    planes = _planes(path, modified).get('values', ())
    if len(planes) != 3 or any(plane.size != native_size for plane in planes):
        raise ValueError('Appearance delta must be native HWC RGB on the source canvas')
    for plane in planes:
        for top in range(0, native_size[1], 256):
            if not np.isfinite(np.asarray(plane.crop((0, top, native_size[0],
                    min(native_size[1], top + 256))), np.float32)).all():
                raise ValueError('Appearance delta contains nonfinite values')


@lru_cache(maxsize=16)
def _validate_mask(path, modified, digest, native_size):
    from design_reconstruction import _planes
    planes = _planes(path, modified).get('values', ())
    if len(planes) != 1 or planes[0].size != native_size:
        raise ValueError('Appearance coverage/support must be scalar on the native source canvas')
    for top in range(0, native_size[1], 256):
        values = np.asarray(planes[0].crop((0, top, native_size[0],
            min(native_size[1], top + 256))), np.float32)
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            raise ValueError('Appearance coverage/support must contain finite values between zero and one')


def _validate_owned_base(layer, plan, run=None, *, require_controls=True):
    """Validate executable frozen assets, independent of producer code/paths."""
    params = layer.get('params') or {}
    native = tuple(plan.get('canvas', {}).get('source_oriented_size', ()))
    baseline = (plan.get('source_baseline') or {}).get('assets') or {}
    schema = params.get('schema')
    if (layer.get('kind') != KIND or schema != SCHEMA
            or plan.get('working_domain') != WORKING_DOMAIN
            or params.get('working_domain') != WORKING_DOMAIN
            or params.get('source_sha256') != plan.get('source_sha256')
            or params.get('reference_sha256') != plan.get('candidate_sha256')
            or tuple(params.get('canvas_size', ())) != native
            or len(native) != 2 or any(int(x) <= 0 for x in native)
            or params.get('source_fp_sha256') != baseline.get('foreground_premultiplied', {}).get('sha256')
            or params.get('alpha_sha256') != baseline.get('foreground_alpha', {}).get('sha256')):
        raise ValueError('Appearance stack must bind the saved original/reference appearance pair and source canvas')
    if run and (params['source_sha256'] != (run.get('source') or {}).get('sha256')
            or params['reference_sha256'] != (run.get('reference') or {}).get('sha256')):
        raise ValueError('Appearance stack source/reference differs from this version')
    strength = float(layer.get('strength', 1))
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('Appearance stack strength must be between zero and one')
    controls = {name: float(layer.get(name, 1.)) for name in ('colour_strength', 'lightness_strength')}
    if any(not np.isfinite(value) or not 0 <= value <= 1 for value in controls.values()):
        raise ValueError('Colour and lightness controls must be between zero and one')
    if tuple(params.get('prepared_stages', ())) != PREPARED_STAGES:
        raise ValueError('Appearance stack must freeze the complete ordered matching/blending stages')
    if params.get('freeze_mode') != 'native_premultiplied_delta':
        raise ValueError('Appearance stack requires a native premultiplied delta')
    if any(row.get('kind') == 'source_reference_colour_match' for row in plan.get('layers', [])):
        raise ValueError('A frozen appearance stack cannot execute with a second automatic colour matcher')
    if sum(row.get('kind') == KIND for row in plan.get('layers', [])) != 1:
        raise ValueError('Exactly one frozen appearance stack owns automatic person matching/blending')
    for name in ('foreground_premultiplied', 'foreground_alpha', 'old_environment'):
        asset = baseline.get(name, {})
        if tuple(asset.get('size', ())) != native:
            raise ValueError('Appearance baseline assets must share the native canvas')
        verify_asset(asset)
    background = next((row.get('assets', {}) for row in plan.get('layers', [])
                       if row.get('kind') == 'background_design'), {})
    verify_asset(background.get('foreground_premultiplied', {}))
    verify_asset(background.get('old_environment', {}))
    current_alpha = plan.get('mask_roles', {}).get('foreground_alpha', {})
    applicability = plan.get('mask_roles', {}).get('edited_edge_light_applicability')
    if current_alpha.get('sha256') != params['alpha_sha256'] and applicability is None:
        raise ValueError('Changed coverage requires explicit appearance reprepare or retained-original applicability')
    verify_asset(current_alpha)
    alpha_path = Path(current_alpha['path']).resolve()
    _validate_mask(str(alpha_path),alpha_path.stat().st_mtime_ns,current_alpha['sha256'],native)
    if applicability is not None:
        if tuple(applicability.get('size', ())) != native:
            raise ValueError('Retained appearance applicability must share the native canvas')
        verify_asset(applicability)
        support_path = Path(applicability['path']).resolve()
        _validate_mask(str(support_path),support_path.stat().st_mtime_ns,applicability['sha256'],native)
    delta = layer.get('assets', {}).get('automatic_delta', {})
    if (tuple(delta.get('size', ())) != native or delta.get('working_space') != WORKING_DOMAIN
            or delta.get('representation') != 'premultiplied_delta'
            or delta.get('coordinate_space') != 'source_canvas'):
        raise ValueError('Automatic delta must be associated display-linear RGB on the native source canvas')
    digest = verify_asset(delta)
    path = Path(delta['path']).resolve()
    stat = path.stat()
    _validate_delta(str(path), stat.st_mtime_ns, stat.st_size, digest, native)
    control_assets = {}
    if require_controls:
        if (params.get('control_recipe') != CONTROL_RECIPE
                or params.get('control_source_delta_sha256') != digest):
            raise ValueError('Native independent controls must bind the original complete appearance delta')
        expected = params.get('control_assets_sha256') or {}
        for role in ('colour_delta', 'lightness_delta', 'control_receipt'):
            asset = layer.get('assets', {}).get(role, {})
            control_assets[role] = verify_asset(asset)
            if role != 'control_receipt' and expected.get(role) != control_assets[role]:
                raise ValueError('Native control asset differs from the bound control bank')
            if role == 'control_receipt':
                continue
            if (tuple(asset.get('size', ())) != native or asset.get('working_space') != WORKING_DOMAIN
                    or asset.get('representation') != 'premultiplied_delta'
                    or asset.get('coordinate_space') != 'source_canvas'):
                raise ValueError('Independent corner deltas must be native associated RGB on the source canvas')
            asset_path = Path(asset['path']).resolve(); asset_stat = asset_path.stat()
            _validate_delta(str(asset_path),asset_stat.st_mtime_ns,asset_stat.st_size,control_assets[role],native)
        import json
        receipt = json.loads(Path(layer['assets']['control_receipt']['path']).read_text(encoding='utf8'))
        from reference_appearance_controls import receipt_identity
        semantic_identity = params.get('control_receipt_identity_sha256')
        if not semantic_identity or semantic_identity != receipt_identity(receipt):
            raise ValueError('Current native controls require an exact canonical receipt identity')
        binding = receipt.get('binding') or {}
        expected_binding = {'source_sha256': params['source_sha256'], 'reference_sha256': params['reference_sha256'],
            'source_fp_sha256': params['source_fp_sha256'], 'alpha_sha256': params['alpha_sha256'],
            'automatic_delta_sha256': digest, 'canvas_size': list(native), 'working_domain': WORKING_DOMAIN,
            'control_assets_sha256': {role: control_assets[role] for role in ('colour_delta','lightness_delta')}}
        if (receipt.get('schema') != 'reference_appearance_controls.v1' or receipt.get('recipe') != CONTROL_RECIPE
                or any(binding.get(key) != value for key,value in expected_binding.items())
                or params.get('control_binding_sha256') != hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest()):
            raise ValueError('Independent control provenance belongs to a different native appearance pair')
    return {'schema': schema, 'working_domain': WORKING_DOMAIN,
            'prepared_stages': list(PREPARED_STAGES), 'freeze_mode': params['freeze_mode'],
            'consumed_asset_sha256': digest, 'source_fp_sha256': params['source_fp_sha256'],
            'source_sha256':params['source_sha256'],'reference_sha256':params['reference_sha256'],
            'alpha_sha256': params['alpha_sha256'], 'canvas_size': list(native),
            'provenance_identity_sha256':params.get('provenance_identity_sha256'), **controls,
            'control_recipe': params.get('control_recipe'), 'consumed_control_assets_sha256':control_assets,
            'independent_control_space': 'native_premultiplied_OKLab_corners_bilinear_RGB',
            'strict_Lab_channel_identity': False}


def validate_layer(layer, plan, run=None):
    """Only complete current v2 stacks with bound support are executable."""
    checked = _validate_owned_base(layer, plan, run)
    import json
    from reference_support import BINDING_KEYS, report_state, digest, _material_projection
    params, assets = layer['params'], layer['assets']
    binding = {key:params[key] for key in BINDING_KEYS if key != 'automatic_delta_sha256'}
    binding['automatic_delta_sha256'] = assets['automatic_delta']['sha256']
    report = params.get('reference_support')
    if not report_state(report, binding)['valid']:
        raise ValueError('Current appearance requires intact current support; saved input cannot be repaired')
    verify_asset(assets.get('reference_support', {}))
    actual = json.loads(Path(assets['reference_support']['path']).read_text(encoding='utf-8-sig'))
    if digest(actual) != digest(report):
        raise ValueError('Current support file differs from embedded evidence')
    transition = report.get('material_transition')
    if transition:
        verify_asset(assets.get('material_finish_receipt', {}))
        material = json.loads(Path(assets['material_finish_receipt']['path']).read_text(encoding='utf-8-sig'))
        projection = _material_projection(material)
        if (transition.get('parent_binding') != projection['parent_binding']
                or transition.get('material_invariants_sha256') != digest(projection)):
            raise ValueError('Material support transition differs from its actual current residual')
        verify_asset(material['field'])
    return checked


def validate_controls(layer, plan, run=None):
    """Producer-stage control check before its support report is published."""
    return _validate_owned_base(layer, plan, run)


def validate_preparation(layer, plan, run=None):
    """Validate a fresh v2 producer input before deriving its control bank."""
    return _validate_owned_base(layer, plan, run, require_controls=False)


def apply_stack(foreground, layer, plan, run=None, *, box=None, canvas_size=None):
    """Return the foreground and its single validated execution report.

    Full, native ROI and reduced previews use the same sampler. Where coverage
    was edited, the native delta and retained fraction are multiplied before
    reduction, so newly added coverage never inherits an old observation.
    """
    from design_reconstruction import sample_asset, sample_supported_part
    report = validate_layer(layer, plan, run)
    values = np.asarray(foreground, np.float32)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError('Appearance foreground must be HWC RGB')
    height, width = values.shape[:2]
    size = tuple(canvas_size or (width, height))
    target = tuple(box or (0, 0, width, height))
    if target[2] - target[0] != width or target[3] - target[1] != height:
        raise ValueError('Appearance ROI extent must match its RGB buffer')
    strength = float(layer.get('strength', 1))
    if not layer.get('enabled') or strength == 0:
        return foreground, report
    c, l = float(layer.get('colour_strength', 1)), float(layer.get('lightness_strength', 1))
    if c == 0 and l == 0:
        return foreground, report
    delta = layer['assets']['automatic_delta']
    applicability = plan.get('mask_roles', {}).get('edited_edge_light_applicability')
    result = np.empty_like(values)
    for row in range(0, height, 256):
        end = min(height, row + 256)
        region = (target[0], target[1] + row, target[2], target[1] + end)
        def sample(asset):
            return (sample_supported_part(asset, applicability, region, size) if applicability is not None
                    else sample_asset(asset, 'values', region, size))
        change = sample(delta)
        if c != 1 or l != 1:
            from reference_appearance_controls import blend_delta
            colour = sample(layer['assets']['colour_delta']) if c*(1-l) else np.zeros_like(change)
            lightness = sample(layer['assets']['lightness_delta']) if (1-c)*l else np.zeros_like(change)
            change = blend_delta(change,colour,lightness,c,l)
        result[row:end] = values[row:end] + change * np.float32(strength)
    return result, report
