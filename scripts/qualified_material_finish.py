"""Freeze qualified material colour residuals on a saved complete appearance.

The original matching and scene balance are replayed once. This producer changes
only low-frequency chroma of the resulting foreground, then replaces the child
delta. Reopening and exporting consume native assets without another fit.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from runtime_resources import code_path
import time
import importlib

import cv2
import numpy as np
from material_solver_profiles import DEFAULT_PROFILE, solver_configuration

DOMAIN = 'display_linear_sRGB'
SCHEMA = 'qualified_material_finish.v1'


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def _record(path, **kwargs):
    return {'path': str(Path(path).resolve()), 'sha256': _sha(path), **kwargs}


def _array(asset):
    from reference_appearance_stack import verify_asset
    verify_asset(asset)
    with np.load(asset['path'], allow_pickle=False) as resource:
        return np.asarray(resource['values'], np.float32)


def finish_owned_appearance(root, run, inputs_path, output, *, algorithm_profile=DEFAULT_PROFILE,
                            cancel=None, progress=None):
    """Return a separate child plan; preserve all parent pixels and assets.

    inputs_path is a real material review receipt, never a Boolean approval.
    Lightness is retained from the complete parent appearance, so its existing
    scene EV remains applicable. Repeated use of the same qualification is exact.
    """
    from material_anchor_review import load_inputs
    from reference_colour_blend import apply_colour_layer
    from render_reference_colour_composite import sample_maps
    from reference_appearance_stack import validate_layer, verify_asset
    from reference_appearance_prepare import provenance_only

    started = time.perf_counter()
    root, output = Path(root).resolve(), Path(output).resolve()
    source, reference, qualification, diagnosis = load_inputs(Path(inputs_path))
    adapter = solver_configuration(algorithm_profile)
    solver = importlib.import_module(adapter['module'])
    fit_qualified_skin_colour = solver.fit_qualified_skin_colour
    original = run['options']['appearance_revision']['design_reconstruction_plan']
    plan = copy.deepcopy(original)
    layer = next(row for row in plan['layers'] if row['kind'] == 'reference_appearance_stack')
    if diagnosis.get('target') == 'final':
        endpoint = diagnosis.get('appearance_endpoint') or {}
        actual = {'run_id':run['run_id'], 'revision':run['revision'],
            'source_sha256':run['source']['sha256'], 'reference_sha256':run['reference']['sha256'],
            'final_sha256':_sha(Path(run['artifacts']['auto_draft'])),
            'alpha_sha256':plan['mask_roles']['foreground_alpha']['sha256'],
            'automatic_delta_sha256':layer['assets']['automatic_delta']['sha256']}
        if endpoint != actual:
            raise ValueError('Final material targets belong to another complete appearance endpoint')
    validate_layer(layer, plan, run)
    from reference_support import BINDING_KEYS, report_state, material_child_support, digest
    parent_support = layer['params'].get('reference_support')
    parent_binding = {key: layer['params'][key] for key in BINDING_KEYS if key != 'automatic_delta_sha256'}
    parent_binding['automatic_delta_sha256'] = layer['assets']['automatic_delta']['sha256']
    if not report_state(parent_support, parent_binding)['valid']:
        raise ValueError('Material production requires current bound support; saved input cannot be repaired')
    verify_asset(layer['assets']['reference_support'])
    if digest(json.loads(Path(layer['assets']['reference_support']['path']).read_text(encoding='utf-8-sig'))) != digest(parent_support):
        raise ValueError('Embedded support differs from its actual current receipt')
    if (not layer.get('enabled') or float(layer.get('strength', 1)) != 1
            or any(float(layer.get(key, 1)) != 1 for key in ('colour_strength', 'lightness_strength'))):
        raise ValueError('Material preparation requires the complete enabled appearance endpoint')
    native = tuple(plan['canvas']['source_oriented_size'])
    proxy = tuple(source['rgb'].shape[1::-1])
    root_id = source['binding'].get('identity_root', source['binding']['image'])
    reference_root = reference['binding'].get('identity_root', reference['binding']['image'])
    if (root_id['sha256'] != run['source']['sha256']
            or reference_root['sha256'] != run['reference']['sha256']
            or proxy != tuple(plan['canvas']['source_proxy_size'])):
        raise ValueError('Material qualification does not belong to this source/reference canvas')
    baseline = plan['source_baseline']['assets']
    if plan.get('mask_roles', {}).get('foreground_alpha', {}).get('sha256') != baseline['foreground_alpha']['sha256']:
        raise ValueError('Changed coverage requires preparation before material finishing')
    inputs_sha256 = _sha(inputs_path)
    solver_path = code_path(adapter['module']+'.py')
    solver_sha256 = _sha(solver_path)
    solver_dependencies = {name: _sha(code_path(name)) for name in adapter['dependencies']}
    application_dependencies = {name: _sha(code_path(name)) for name in
        ('reference_colour_blend.py', 'render_reference_colour_composite.py')}
    producer_sha256 = _sha(__file__)
    previous = layer['params'].get('material_finish') or {}
    if previous:
        if (previous.get('qualification_sha256') == qualification['qualification_sha256']
                and previous.get('inputs_sha256') == inputs_sha256
                and previous.get('solver_sha256') == solver_sha256
                and previous.get('algorithm_profile') == algorithm_profile
                and previous.get('solver_dependencies_sha256') == solver_dependencies
                and previous.get('application_dependencies_sha256', {}) == application_dependencies
                and previous.get('producer_sha256') == producer_sha256):
            return {'ok': True, 'design_reconstruction_plan': plan, 'cache_hit': True,
                    'material_finish': previous, 'new_rendered_revision': False}
        raise ValueError('Material qualification or implementation changed; reprepare from the parent before material finishing')
    output.mkdir(parents=True, exist_ok=True)

    def report(fraction, message):
        if cancel and cancel():
            raise InterruptedError('Qualified material preparation cancelled')
        if progress:
            progress('material_colour', message, fraction)

    report(0, '读取本版本完整原生人物外观。')
    fp, alpha, delta = (_array(baseline['foreground_premultiplied']),
                        _array(baseline['foreground_alpha']), _array(layer['assets']['automatic_delta']))
    if fp.shape != (*native[::-1], 3) or delta.shape != fp.shape or alpha.shape != native[::-1]:
        raise ValueError('Material preparation requires exact native foreground/coverage/delta grids')
    target = fp + delta
    # Area-average associated quantities before unassociation. Empty coverage
    # falls back to the original observation, never a generated person's RGB.
    associated = cv2.resize(target, proxy, interpolation=cv2.INTER_AREA)
    proxy_alpha = cv2.resize(alpha, proxy, interpolation=cv2.INTER_AREA)
    base = source['rgb'].copy()
    np.divide(associated, proxy_alpha[..., None], out=base, where=proxy_alpha[..., None] > 1e-6)
    if not np.isfinite(base).all() or base.min() < -2e-6 or base.max() > 1+2e-6:
        raise ValueError('Complete appearance is outside the supported SDR material domain')
    base = np.clip(base, 0, 1).astype(np.float32)
    base_path = output/'current-appearance-proxy.npz'
    np.savez_compressed(base_path, values=base)
    base_recipe = {'schema': SCHEMA, 'domain': DOMAIN, 'basis': 'complete_frozen_parent_foreground',
        'source': copy.deepcopy(run['source']), 'reference': copy.deepcopy(run['reference']),
        'foreground': copy.deepcopy(baseline['foreground_premultiplied']),
        'alpha': copy.deepcopy(baseline['foreground_alpha']),
        'delta': copy.deepcopy(layer['assets']['automatic_delta']),
        'sampling': 'AREA_associated_RGB_and_alpha_then_unassociate; zero_alpha_original_observation',
        'parent_source_display_sha256': source['binding']['image']['sha256'],
        'proxy_size': list(proxy), 'native_size': list(native)}
    recipe_path = output/'current-appearance-proxy-recipe.json'; _write(recipe_path, base_recipe)
    report(.15, '按已审核的材质目标求低频偏色修正和眼口环境色衔接。')
    solved = fit_qualified_skin_colour(source, reference, qualification, diagnosis,
        source_base={'rgb': base, 'binding': {'asset': _record(base_path), 'recipe': _record(recipe_path),
                    'parent_source_sha256': source['binding']['image']['sha256']}})
    field = solved['source_ab_field']
    # New targets are colour evidence; opaque native skin/feature material is
    # the only automatic application domain. No new edge donor is invented.
    field[proxy_alpha < .995] = 0
    owners = source['labels'].astype(np.int32).copy()
    for index, (name, values) in enumerate(sorted(source.get('feature_masks', {}).items())):
        owners[values > .5] = 1000+index
    field_path = output/'qualified-material-field.npz'
    np.savez_compressed(field_path, values=field, owners=owners)
    native_changed = 0; l_error = 0.; outside_exact = True
    from reference_colour_fit import to_lab
    for top in range(0, native[1], 128):
        report(.2+.55*top/native[1], '在原尺寸保留人物明暗和纹理，仅应用颜色残差。')
        end = min(native[1], top+128); box = (0, top, native[0], end)
        sampled = sample_maps(field, box, proxy, native, owners=owners)
        active = (alpha[top:end] >= .995) & np.any(sampled[..., :2] != 0, axis=-1)
        if not active.any():
            continue
        unchanged = delta[top:end][~active].copy()
        current = target[top:end][active]/alpha[top:end][active, None]
        adjusted = apply_colour_layer(current[None], sampled[active][None])[0]
        l_error = max(l_error, float(np.max(np.abs(to_lab(current)[:, 0]-to_lab(adjusted)[:, 0]))))
        shift = (adjusted-current)*alpha[top:end][active, None]
        delta[top:end][active] += shift
        outside_exact &= bool(np.array_equal(delta[top:end][~active], unchanged))
        native_changed += int(np.count_nonzero(np.any(shift != 0, axis=-1)))
    report(.8, '冻结本版本颜色修正、审核依据和原生差值。')
    delta_path = output/'qualified-appearance-delta.npz'; np.savez_compressed(delta_path, values=delta)
    if (inputs_sha256 != _sha(inputs_path) or producer_sha256 != _sha(__file__)
                or any(digest != _sha(code_path(name)) for name, digest in solver_dependencies.items())):
        raise ValueError('Material inputs/solver changed while preparing; preserve this attempt and retry from stable inputs')
    if any(digest != _sha(code_path(name)) for name, digest in application_dependencies.items()):
        raise ValueError('Material application code changed while preparing')
    receipt = {'schema': SCHEMA, 'working_domain': DOMAIN,
        'algorithm_profile': algorithm_profile, 'solver_dependencies_sha256': solver_dependencies,
        'application_dependencies_sha256': application_dependencies,
        'input': _record(inputs_path), 'base_recipe': provenance_only(base_recipe),
        'solver': provenance_only(solved['receipt']), 'field': _record(field_path),
        'field_after_opaque_coverage_gate': True,
        'applied_to': 'complete_parent_T_B_S_foreground_before_single_composition',
        'original_T_B_S_reapplied': False, 'scene_EV_refitted': False,
        'scene_EV_reuse_reason': 'material_residual_preserves_parent_lightness',
        'application': 'native_opaque_skin_and_independent_face_features; owner_aware_sampling',
        'native_changed_pixels': native_changed, 'maximum_OKLab_L_error': l_error,
        'outside_field_exact': outside_exact, 'coverage_modified': False,
        'reference_geometry_or_RGB_pasted': False, 'alpha_consumption': 'associated_delta_once',
        'visual_acceptance': False, 'elapsed_seconds': round(time.perf_counter()-started, 2)}
    if l_error > 4e-6 or not outside_exact or np.any(delta[alpha == 0] != 0):
        _write(output/'rejected-material-finish.json', receipt)
        raise ValueError('Native material finish violated lightness/empty-coverage invariants')
    receipt_path = output/'material-finish-receipt.json'; _write(receipt_path, receipt)
    snapshot = json.loads(Path(layer['assets']['preparation_snapshot']['path']).read_text(encoding='utf8'))
    snapshot['material_finish'] = provenance_only(receipt)
    snapshot['material_finish']['qualification_sha256'] = qualification['qualification_sha256']
    snapshot['producer_code']['qualified_material_finish.py'] = _sha(__file__)
    snapshot['producer_code'].update(solver_dependencies)
    snapshot['producer_code'].update(application_dependencies)
    snapshot_path = output/'preparation-snapshot.json'; _write(snapshot_path, snapshot)
    layer['assets']['automatic_delta'] = _record(delta_path, size=list(native),
        coordinate_space='source_canvas', working_space=DOMAIN,
        representation='premultiplied_delta', sampling='bilinear')
    layer['assets']['preparation_snapshot'] = _record(snapshot_path,
        representation='path_independent_preparation_provenance')
    layer['assets']['material_finish_receipt'] = _record(receipt_path,
        representation='qualified_material_finish_provenance')
    layer['params']['provenance_identity_sha256'] = _sha(snapshot_path)
    layer['params']['material_finish'] = {'schema': SCHEMA,
        'qualification_sha256': qualification['qualification_sha256'],
        'receipt_sha256': _sha(receipt_path), 'native_changed_pixels': native_changed,
        'inputs_sha256': inputs_sha256, 'solver_sha256': solver_sha256,
        'algorithm_profile': algorithm_profile, 'solver_dependencies_sha256': solver_dependencies,
        'application_dependencies_sha256': application_dependencies,
        'producer_sha256': producer_sha256}
    # Independent control corners belong to their complete delta. Derive fresh
    # corners instead of carrying obsolete colour/lightness controls forward.
    for name in ('colour_delta', 'lightness_delta', 'control_receipt'):
        layer['assets'].pop(name, None)
    for name in ('control_recipe', 'control_source_delta_sha256', 'control_assets_sha256',
                 'control_receipt_identity_sha256', 'control_binding_sha256'):
        layer['params'].pop(name, None)
    plan.setdefault('provenance', {})['qualified_material_finish'] = provenance_only(receipt)
    del fp, alpha, delta, target
    from reference_appearance_controls import prepare_owned_controls
    plan = prepare_owned_controls(plan, output/'controls', run=run, cancel=cancel)
    finished_layer = next(row for row in plan['layers'] if row['kind'] == 'reference_appearance_stack')
    current_binding = {key: finished_layer['params'][key] for key in BINDING_KEYS if key != 'automatic_delta_sha256'}
    current_binding['automatic_delta_sha256'] = finished_layer['assets']['automatic_delta']['sha256']
    support = material_child_support(parent_support, current_binding, receipt)
    support_path = output/'reference-support.json'
    _write(support_path, support)
    finished_layer['params']['reference_support'] = support
    finished_layer['assets']['reference_support'] = _record(support_path,
        representation='path_independent_reference_support')
    validate_layer(finished_layer, plan, run)
    _write(output/'prepared-plan.json', {'ok': True, 'design_reconstruction_plan': plan,
        'material_finish': receipt, 'cache_hit': False, 'new_rendered_revision': False})
    report(1, '原生材质颜色修正已冻结，可重开、编辑和导出。')
    return {'ok': True, 'design_reconstruction_plan': plan, 'material_finish': receipt,
            'cache_hit': False, 'new_rendered_revision': False}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--profile', choices=(DEFAULT_PROFILE,), default=DEFAULT_PROFILE)
    args = parser.parse_args()
    run = json.loads(args.run.read_text(encoding='utf8'))
    last = {'percent': -1}

    def progress(stage, message, fraction):
        percent = int(100*fraction)
        if percent//10 != last['percent']//10:
            print(json.dumps({'stage': stage, 'percent': percent, 'message': message}, ensure_ascii=False), flush=True)
            last['percent'] = percent

    result = finish_owned_appearance(args.root, run, args.inputs, args.output,
        algorithm_profile=args.profile, progress=progress)
    _write(args.output/'prepared-plan.json', result)
    print(json.dumps({'ok': result['ok'], 'prepared_plan': str(args.output/'prepared-plan.json'),
        'cache_hit': result['cache_hit'], 'native_changed_pixels': result['material_finish'].get('native_changed_pixels')},
        ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
