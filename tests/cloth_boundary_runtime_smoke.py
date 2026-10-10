"""Real cloth/hair consumers on synthetic frozen fields; no algorithm inference."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch

import numpy as np
from PIL import Image

import boundary_light_runtime as hair
import cloth_boundary_runtime as cloth
import design_reconstruction as design
from boundary_light_runtime_smoke import fixture as hair_fixture, _sample
from native_roi_render import render_native_roi
from native_subject_detail_runtime import array_sha
from native_subject_detail_runtime_smoke import closure, record, write, sha
from portable_project import export_project, import_project
from subject_environment_runtime import preceding_detail_digest


def cloth_layer(plan):
    return next(row for row in plan['layers'] if row['kind'] == cloth.KIND)


def fixture(root, *, with_hair=True, with_detail=False, fractional=False):
    run, _, _, alpha, background, labels, protection = hair_fixture(root,
        with_detail=with_detail, fractional=fractional)
    plan = design.plan_for_run(run)
    if fractional:
        # Freeze actual endpoint fields at both sides of the opaque threshold.
        changed_alpha = alpha.copy(); changed_alpha[18, 12] = .95; changed_alpha[18, 13] = .9499
        ratio = np.divide(changed_alpha, alpha, out=np.zeros_like(alpha), where=alpha > 0)
        baseline = plan['source_baseline']['assets']
        background_layer = next(row for row in plan['layers'] if row['kind'] == 'background_design')
        for role in ('foreground_alpha', 'foreground_premultiplied'):
            with np.load(baseline[role]['path'], allow_pickle=False) as data: values = data['values'].copy()
            values = changed_alpha if role == 'foreground_alpha' else values*ratio[..., None]
            path = root/('cloth-opaque-'+role+'.npz'); np.savez_compressed(path, values=values)
            asset = {**baseline[role], **record(path)}
            baseline[role] = asset; background_layer['assets'][role] = asset
            if role == 'foreground_alpha': plan['mask_roles'][role] = asset
        stack = next(row for row in plan['layers'] if row['kind'] == 'reference_appearance_stack')
        for role, asset in list(stack['assets'].items()):
            with np.load(asset['path'], allow_pickle=False) as data: values = data['values'].copy()
            path = root/('cloth-opaque-'+role+'.npz'); np.savez_compressed(path, values=values*ratio[..., None])
            stack['assets'][role] = {**asset, **record(path)}
        alpha = changed_alpha
    labels[4:10, 8:16] = 1; labels[10:17, 8:16] = 13; labels[17:28, 8:16] = 23
    labels[12, 9], labels[13, 9], labels[14, 9] = 6, 4, 0
    labels[7, 15] = 23  # Protected opaque outer clothing must still reject a delta.
    semantic_path = root/'cloth-owners.npz'; np.savez_compressed(semantic_path, labels=labels)
    semantic = {**plan['mask_roles']['source_semantics'], **record(semantic_path)}
    plan['mask_roles']['source_semantics'] = semantic
    for layer in plan['layers']:
        if layer['kind'] not in {'source_subject_detail', 'source_environment_response'}: continue
        frozen = json.loads(Path(layer['assets']['frozen_delta']['path']).read_text('utf8'))
        frozen['owner_asset'] = semantic; frozen['binding']['semantic_sha256'] = semantic['sha256']
        if layer['kind'] == 'source_environment_response': frozen['binding']['preceding_detail_digest'] = preceding_detail_digest(plan)
        layer['params']['binding'] = copy.deepcopy(frozen['binding'])
        layer['assets']['frozen_delta'] = write(root/(layer['kind']+'-cloth-fixture.json'), frozen)
    hair_layer = next(row for row in plan['layers'] if row['kind'] == hair.KIND)
    if with_hair:
        frozen = json.loads(Path(hair_layer['assets']['frozen_delta']['path']).read_text('utf8'))
        qa = json.loads(Path(hair_layer['assets']['qualification_receipt']['path']).read_text('utf8'))
        with np.load(frozen['delta']['path'], allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        for row in frozen['patches']:
            l, t, r, b = row['native_box']; delta = arrays[row['array_key']]
            delta[labels[t:b, l:r] != 4] = 0
            row['delta_array_sha256'] = array_sha(delta); row['changed_pixels'] = int(np.any(delta != 0, -1).sum())
            row['target_patch_sha256'] = array_sha(hair.endpoint(plan, run, row['native_box'], (48, 32)))
        delta_path = root/'current-hair-delta.npz'; np.savez_compressed(delta_path, **arrays)
        frozen['delta'] = record(delta_path); frozen['binding'] = hair.endpoint_binding(plan, run)
        qa['changed_pixels'] = sum(row['changed_pixels'] for row in frozen['patches'])
        hair_layer['params']['binding'] = copy.deepcopy(frozen['binding'])
        hair_layer['assets']['frozen_delta'] = write(root/'current-hair-frozen.json', frozen)
        hair_layer['assets']['qualification_receipt'] = write(root/'current-hair-qualification.json', qa)
    else:
        plan['layers'].remove(hair_layer)
    before = cloth.endpoint(plan, run, (0, 0, 48, 32), (48, 32))
    full_delta = np.zeros_like(before); box = (6, 2, 17, 29); l, t, r, b = box
    allowed = np.isin(labels, cloth.OWNER_CLASSES)&(alpha >= cloth.OPAQUE_ALPHA_MIN)&(protection < .5)
    full_delta[t:b, l:r] = np.float32(.002)*alpha[t:b, l:r, None]*allowed[t:b, l:r, None]
    delta = full_delta[t:b, l:r].copy(); delta_path = root/'cloth-delta.npz'
    np.savez_compressed(delta_path, patch_0=delta)
    observation_path = root/'cloth-reference-observation.npz'
    np.savez_compressed(observation_path, values=np.full((8, 12), .8, np.float32))
    binding = cloth.endpoint_binding(plan, run)
    frozen = {'schema': cloth.SCHEMA, 'recipe': cloth.RECIPE, 'binding': binding, 'delta': record(delta_path),
        'patches': [{'native_box': list(box), 'array_key': 'patch_0', 'delta_array_sha256': array_sha(delta),
            'target_patch_sha256': array_sha(cloth.endpoint(plan, run, box, (48, 32))),
            'changed_pixels': int(np.any(delta != 0, -1).sum())}], 'observation_assets': {'reference': record(observation_path)}}
    receipt = {'schema': cloth.SCHEMA, 'recipe': cloth.RECIPE, 'state': 'prepared',
        'changed_pixels': frozen['patches'][0]['changed_pixels']}
    plan['layers'].append({'id': cloth.KIND, 'kind': cloth.KIND, 'label': '参考肩袖受光', 'enabled': True, 'strength': 1.,
        'params': {'schema': cloth.SCHEMA, 'recipe': cloth.RECIPE, 'binding': copy.deepcopy(binding)},
        'assets': {'frozen_delta': write(root/'cloth-frozen.json', frozen),
            'qualification_receipt': write(root/'cloth-qualification.json', receipt)}})
    return run, before, full_delta, alpha, background, labels, protection


def run_smoke():
    checks = 0
    def require(value):
        nonlocal checks
        assert value; checks += 1
    def rejects(call, message):
        nonlocal checks
        try: call()
        except ValueError: checks += 1
        else: raise AssertionError('Invalid cloth accepted: '+message)
    with tempfile.TemporaryDirectory(prefix='cos-cloth-runtime-') as temporary:
        base = Path(temporary)
        with patch('reference_appearance_stack.validate_layer', return_value={}):
            for mode in ('no-hair', 'hair', 'detail-environment-hair', 'fractional-hair'):
                origin = base/('origin-'+mode)
                run, before, delta, alpha, background, labels, protection = fixture(origin,
                    with_hair=mode != 'no-hair', with_detail=mode == 'detail-environment-hair', fractional=mode == 'fractional-hair')
                plan = design.plan_for_run(run); layer = cloth_layer(plan)
                current_hair = next((row for row in plan['layers'] if row['kind'] == hair.KIND), None)
                hair_files = {}
                if current_hair:
                    frozen, _, _ = hair._load(current_hair)
                    paths = [Path(asset['path']) for asset in current_hair['assets'].values()]+[Path(frozen['delta']['path'])]
                    hair_files = {str(path): sha(path) for path in paths}
                    require(hair.validate_layer(current_hair, plan, run)['bypass_reason'] is None)
                    base_endpoint = hair.endpoint(plan, run, (0, 0, 48, 32), (48, 32))
                    hair_values, _ = hair.apply_boundary_light(base_endpoint, current_hair, plan, run)
                    require(np.array_equal(before, hair_values) and np.any(before != base_endpoint))
                result, state = cloth.apply_cloth_boundary_light(before, layer, plan, run)
                require(np.array_equal(result, before+delta) and state['applied'])
                require(state['source_alpha_changed'] is False and state['hair_changed'] is False)
                require(state['owner_classes'] == [23] and state['minimum_source_alpha'] == .95
                    and state['partial_alpha_response_supported'] is False)
                for mask in (alpha < .95, ~np.isin(labels, cloth.OWNER_CLASSES), protection >= .5):
                    require(np.array_equal(result[mask], before[mask]))
                for owner in cloth.OWNER_CLASSES: require(np.any(delta[labels == owner] > 0))
                if mode == 'fractional-hair':
                    require(not np.any(delta[(alpha > 0)&(alpha < .95)]))
                    require(np.any(delta[alpha == np.float32(.95)] > 0))
                roi = (9, 7, 20, 19)
                roi_values, _ = cloth.apply_cloth_boundary_light(before[7:19, 9:20], layer, plan, run, box=roi, canvas_size=(48, 32))
                require(np.array_equal(roi_values, result[7:19, 9:20]))
                small = _sample(before, (0, 0, 24, 16), (24, 16)); small_delta = _sample(delta, (0, 0, 24, 16), (24, 16))
                preview, _ = cloth.apply_cloth_boundary_light(small, layer, plan, run)
                require(np.array_equal(preview, small+small_delta))
                geometry = Image.fromarray(np.uint8(alpha*255)); zeros = np.zeros_like(before)
                whole, execution = design.apply_plan(zeros, zeros, plan, geometry, run=run, include_optics=False)
                require(np.array_equal(whole, before+delta+(1-alpha[..., None])*background))
                require(next(row for row in execution['layers'] if row['kind'] == cloth.KIND)['applied'])
                full_roi, _ = render_native_roi(run, roi); require(np.array_equal(full_roi, whole[7:19, 9:20]))
                seen = []
                def manual(values, *args, **kwargs):
                    if kwargs.get('branch') == 'foreground': seen.append(values.copy())
                    return values
                with patch('float_appearance.apply_manual_layers_float', side_effect=manual):
                    design.apply_plan(zeros, zeros, plan, geometry, run=run, include_optics=False)
                require(len(seen) == 1 and np.array_equal(seen[0], before+delta))
                for enabled, strength in ((False, 1.), (True, 0.), (True, .5)):
                    layer.update(enabled=enabled, strength=strength)
                    values, status = cloth.apply_cloth_boundary_light(before, layer, plan, run)
                    require(np.array_equal(values, before+delta*np.float32(strength if enabled else 0)))
                    require(status['applied'] == bool(enabled and strength))
                layer.update(enabled=True, strength=1.)
                for mutation in ('appearance', 'foreground-control', 'edited-alpha', 'hair-dose', 'hair-disabled', 'hair-removed', 'environment-dose'):
                    if mutation.startswith('hair-') and not current_hair: continue
                    if mutation == 'environment-dose' and mode == 'fractional-hair': continue
                    changed = copy.deepcopy(run); changed_plan = design.plan_for_run(changed)
                    if mutation == 'appearance': next(row for row in changed_plan['layers'] if row['kind'] == 'reference_appearance_stack')['strength'] = .5
                    elif mutation == 'foreground-control': changed['options']['appearance_revision']['params'] = {'subject': {'exposure': .1}}
                    elif mutation == 'edited-alpha': changed_plan['mask_roles']['edited_edge_light_applicability'] = {'path': 'test-marker'}
                    elif mutation == 'environment-dose': next(row for row in changed_plan['layers'] if row['kind'] == 'source_environment_response')['strength'] = .5
                    else:
                        changed_hair = next(row for row in changed_plan['layers'] if row['kind'] == hair.KIND)
                        if mutation == 'hair-dose': changed_hair['strength'] = .5
                        elif mutation == 'hair-disabled': changed_hair['enabled'] = False
                        else: changed_plan['layers'].remove(changed_hair)
                    values, status = cloth.apply_cloth_boundary_light(before, cloth_layer(changed_plan), changed_plan, changed)
                    require(np.array_equal(values, before) and status['bypass_reason'] == 'cloth_boundary_endpoint_changed_requires_new_preparation')
                for mutation in ('negative', 'alpha-zero', 'owner-1', 'owner-13', 'hair-owner', 'skin-owner', 'unknown-owner', 'protected', 'overflow',
                        'partial-alpha', 'below-opaque-threshold',
                        'target-hash', 'array-hash', 'delta-hash', 'frozen-hash', 'observation-hash', 'count', 'source', 'duplicate', 'wrong-order', 'hair-corruption'):
                    if mutation == 'hair-corruption' and not current_hair: continue
                    if mutation in ('partial-alpha', 'below-opaque-threshold') and mode != 'fractional-hair': continue
                    changed = copy.deepcopy(run); changed_plan = design.plan_for_run(changed); bad_layer = cloth_layer(changed_plan)
                    frozen = json.loads(Path(layer['assets']['frozen_delta']['path']).read_text('utf8'))
                    qa = json.loads(Path(layer['assets']['qualification_receipt']['path']).read_text('utf8')); row = frozen['patches'][0]
                    if mutation in ('negative', 'alpha-zero', 'owner-1', 'owner-13', 'hair-owner', 'skin-owner', 'unknown-owner', 'protected', 'overflow', 'partial-alpha', 'below-opaque-threshold'):
                        with np.load(frozen['delta']['path'], allow_pickle=False) as archive: values = archive['patch_0'].copy()
                        positions = {'alpha-zero': (2, 6), 'owner-1': (6, 10), 'owner-13': (15, 10),
                            'hair-owner': (13, 9), 'skin-owner': (12, 9), 'unknown-owner': (14, 9),
                            'protected': (7, 15), 'partial-alpha': (18, 9), 'below-opaque-threshold': (18, 13)}
                        l, t, _, _ = row['native_box']
                        if mutation in positions: y, x = positions[mutation]; values[y-t, x-l] = .001
                        else: values[20-t, 14-l, 0] = -.001 if mutation == 'negative' else 2.
                        row['delta_array_sha256'] = array_sha(values); row['changed_pixels'] = int(np.any(values != 0, -1).sum())
                        qa['changed_pixels'] = row['changed_pixels']; path = origin/('bad-'+mutation+'.npz')
                        np.savez_compressed(path, patch_0=values); frozen['delta'] = record(path)
                    elif mutation == 'target-hash': row['target_patch_sha256'] = '0'*64
                    elif mutation == 'array-hash': row['delta_array_sha256'] = '0'*64
                    elif mutation == 'delta-hash': frozen['delta']['sha256'] = '0'*64
                    elif mutation == 'observation-hash': frozen['observation_assets']['reference']['sha256'] = '0'*64
                    elif mutation == 'count': qa['changed_pixels'] += 1
                    elif mutation == 'source': changed['source']['sha256'] = '0'*64
                    elif mutation == 'duplicate': changed_plan['layers'].append(copy.deepcopy(bad_layer))
                    elif mutation == 'wrong-order': changed_plan['layers'].remove(bad_layer); changed_plan['layers'].insert(0, bad_layer)
                    elif mutation == 'hair-corruption': next(row for row in changed_plan['layers'] if row['kind'] == hair.KIND)['assets']['frozen_delta']['sha256'] = '0'*64
                    bad_layer['assets']['frozen_delta'] = write(origin/('bad-'+mutation+'.json'), frozen)
                    bad_layer['assets']['qualification_receipt'] = write(origin/('bad-'+mutation+'-qa.json'), qa)
                    if mutation == 'frozen-hash': bad_layer['assets']['frozen_delta']['sha256'] = '0'*64
                    rejects(lambda: cloth.apply_cloth_boundary_light(before, bad_layer, changed_plan, changed), mutation)
                if current_hair:
                    require(all(sha(path) == digest for path, digest in hair_files.items()))
                    require(hair.validate_layer(current_hair, plan, run)['bypass_reason'] is None)
                    changed = copy.deepcopy(run); changed_plan = design.plan_for_run(changed)
                    changed['options']['appearance_revision']['params'] = {'subject': {'exposure': .1}}
                    rejects(lambda: cloth.endpoint(changed_plan, changed, roi, (48, 32)), 'stale-active-hair-endpoint')
                saved = base/('saved-'+mode); owned_run = copy.deepcopy(run)
                binding = cloth.endpoint_binding(plan, run)
                owned_plan = design.materialize_plan(saved, owned_run, plan)
                owned_run['root'] = str(saved); owned_run['options']['appearance_revision']['design_reconstruction_plan'] = owned_plan
                require(cloth.endpoint_binding(owned_plan, owned_run) == binding)
                owned_records = closure(owned_plan); require(owned_records > 12)
                write(saved/'outputs/runs/fixture/manifest.json', owned_run)
                archive = base/(mode+'.cosplay'); export_project(saved, 'fixture', archive)
                moved = import_project(base/('portable-'+mode), archive, run_id='relocated')['run']
                require(cloth.endpoint_binding(design.plan_for_run(moved), moved) == binding)
                require(origin.resolve().is_relative_to(base.resolve()) and origin.name.startswith('origin-'))
                shutil.rmtree(origin)
                for relocated in (owned_run, moved):
                    relocated_plan = design.plan_for_run(relocated); relocated_layer = cloth_layer(relocated_plan)
                    require(closure(relocated_plan) >= owned_records)
                    values, status = cloth.apply_cloth_boundary_light(before, relocated_layer, relocated_plan, relocated)
                    require(np.array_equal(values, before+delta) and status['applied'])
                    full, _ = design.apply_plan(zeros, zeros, relocated_plan, geometry, run=relocated, include_optics=False)
                    crop, _ = render_native_roi(relocated, roi)
                    require(np.array_equal(full, whole) and np.array_equal(crop, whole[7:19, 9:20]))
                    small_zeros = np.zeros((16, 24, 3), np.float32)
                    preview, _ = design.apply_plan(small_zeros, small_zeros, relocated_plan, geometry.resize((24, 16)), run=relocated, include_optics=False)
                    relocated_layer['enabled'] = False
                    base_preview, _ = design.apply_plan(small_zeros, small_zeros, relocated_plan, geometry.resize((24, 16)), run=relocated, include_optics=False)
                    relocated_layer['enabled'] = True
                    require(np.allclose(preview, base_preview+small_delta, rtol=0, atol=3e-8))
    return {'ok': True, 'checks': checks, 'actual_cloth_and_hair_runtime_endpoints': True,
        'appearance_validator_mocked': True, 'native_full_roi_preview_shared_consumers': True,
        'current_hair_v2_assets_unchanged_and_valid': True, 'upstream_hair_controls_bound': True,
        'cloth_owner_23_only_opaque_alpha_at_least_095_and_protection': True,
        'owners_1_13_partial_alpha_and_09499_rejected_095_replays': True,
        'negative_corruption_overflow_duplicate_and_order_rejected': True,
        'materialization_portable_semantic_binding_after_originals_removed': True,
        'model_inferences': 0, 'visual_acceptance': False,
        'scope': 'Synthetic frozen replay/ownership, not cloth algorithm or real-image qualification.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path); args = parser.parse_args()
    report = run_smoke()
    if args.output: write(args.output, report)
    print(json.dumps(report, ensure_ascii=False))
