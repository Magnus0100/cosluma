"""Synthetic native/ROI/preview and portable proof for frozen hair light.

The real boundary runtime, endpoint, delta archives and renderer consumers run.
Only the qualification validator for synthetic appearance banks is mocked; this
test does not establish reference observation quality or real-image acceptance.
"""
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

import boundary_light_runtime as boundary
import design_reconstruction as design
from native_roi_render import render_native_roi
from native_subject_detail_runtime import array_sha
from native_subject_detail_runtime_smoke import closure, record, write
from portable_project import export_project, import_project
from subject_environment_runtime import preceding_detail_digest
from subject_environment_runtime_smoke import fixture as environment_fixture


def _layer(plan):
    return next(row for row in plan['layers'] if row['kind'] == boundary.KIND)


def _sample(values, box, size):
    return design._sample_channels(tuple(Image.fromarray(values[..., c])
        for c in range(values.shape[-1])), box, size)


def fixture(root, *, with_detail=False, fractional=False):
    run, _, _, alpha, background = environment_fixture(root, with_detail=with_detail)
    plan = design.plan_for_run(run)
    if fractional:
        # A separate appearance-only endpoint exercises genuinely fractional
        # source ownership without broadening the opaque environment layer.
        plan['layers'] = [row for row in plan['layers']
            if row['kind'] not in {'source_environment_response', 'source_subject_detail'}]
        changed_alpha = alpha.copy(); changed_alpha[4:28, 8:12] = .25
        ratio = np.divide(changed_alpha, alpha, out=np.zeros_like(alpha), where=alpha > 0)
        baseline = plan['source_baseline']['assets']
        background_layer = next(row for row in plan['layers'] if row['kind'] == 'background_design')
        for role in ('foreground_alpha', 'foreground_premultiplied'):
            with np.load(baseline[role]['path'], allow_pickle=False) as data:
                values = data['values'].copy()
            values = changed_alpha if role == 'foreground_alpha' else values*ratio[..., None]
            path = root/('fractional-'+role+'.npz'); np.savez_compressed(path, values=values)
            asset = {**baseline[role], **record(path)}
            baseline[role] = asset; background_layer['assets'][role] = asset
            if role == 'foreground_alpha': plan['mask_roles'][role] = asset
        stack = next(row for row in plan['layers'] if row['kind'] == 'reference_appearance_stack')
        for role, asset in list(stack['assets'].items()):
            with np.load(asset['path'], allow_pickle=False) as data: values = data['values'].copy()
            path = root/('fractional-'+role+'.npz')
            np.savez_compressed(path, values=values*ratio[..., None])
            stack['assets'][role] = {**asset, **record(path)}
        alpha = changed_alpha
    semantic = plan['mask_roles']['source_semantics']
    with np.load(semantic['path'], allow_pickle=False) as data: labels = data['labels'].copy()
    # Keep unsupported owner pixels outside the preceding detail/environment
    # patches, but inside the new boundary patch's rectangular storage.
    labels[10, 9], labels[11, 9], labels[12, 9] = 23, 6, 0
    path = root/'boundary-owners.npz'; np.savez_compressed(path, labels=labels)
    semantic = {**semantic, **record(path)}; plan['mask_roles']['source_semantics'] = semantic
    for layer in plan['layers']:
        if layer['kind'] not in {'source_subject_detail', 'source_environment_response'}: continue
        frozen = json.loads(Path(layer['assets']['frozen_delta']['path']).read_text('utf8'))
        frozen['owner_asset'] = semantic; frozen['binding']['semantic_sha256'] = semantic['sha256']
        if layer['kind'] == 'source_environment_response':
            frozen['binding']['preceding_detail_digest'] = preceding_detail_digest(plan)
        layer['params']['binding'] = copy.deepcopy(frozen['binding'])
        layer['assets']['frozen_delta'] = write(root/(layer['kind']+'-boundary-fixture.json'), frozen)
    before = boundary.endpoint(plan, run, (0, 0, 48, 32), (48, 32))
    with np.load(plan['mask_roles']['identity_protection']['path'], allow_pickle=False) as data:
        protection = data['protection'].copy()
    box = (6, 2, 44, 28); l, t, r, b = box
    allowed = (alpha > 0) & (labels == 4) & (protection < .5)
    full_delta = np.zeros_like(before)
    full_delta[t:b, l:r] = np.float32(.003)*alpha[t:b, l:r, None]*allowed[t:b, l:r, None]
    delta = full_delta[t:b, l:r].copy(); delta_path = root/'boundary-delta.npz'
    np.savez_compressed(delta_path, patch_0=delta)
    observation_path = root/'reference-observation.npz'
    np.savez_compressed(observation_path, alpha=np.full((8, 12), .7, np.float32))
    observation_json = write(root/'reference-observation.json', {
        'schema': 'synthetic_reference_observation.v1', 'reference_sha256': run['reference']['sha256'],
        'matte': record(observation_path), 'model_inferences': 0, 'truth_claimed': False})
    binding = boundary.endpoint_binding(plan, run)
    frozen = {'schema': boundary.SCHEMA, 'recipe': boundary.RECIPE, 'binding': binding,
        'delta': record(delta_path), 'patches': [{'native_box': list(box), 'array_key': 'patch_0',
            'delta_array_sha256': array_sha(delta), 'target_patch_sha256': array_sha(before[t:b, l:r]),
            'changed_pixels': int(np.any(delta != 0, axis=-1).sum())}],
        'observation_assets': {'matte': record(observation_path), 'receipt': observation_json}}
    receipt = {'schema': boundary.SCHEMA, 'recipe': boundary.RECIPE, 'state': 'prepared',
        'changed_pixels': frozen['patches'][0]['changed_pixels'], 'visual_acceptance': False,
        'source_alpha_changed': False, 'source_geometry_changed': False}
    layer = {'id': boundary.KIND, 'kind': boundary.KIND, 'enabled': True, 'strength': 1.,
        'params': {'schema': boundary.SCHEMA, 'recipe': boundary.RECIPE, 'binding': copy.deepcopy(binding)},
        'assets': {'frozen_delta': write(root/'boundary-frozen.json', frozen),
            'qualification_receipt': write(root/'boundary-qualification.json', receipt)}}
    plan['layers'].append(layer)
    return run, before, full_delta, alpha, background, labels, protection


def run_smoke():
    checks = 0
    def require(value):
        nonlocal checks
        assert value; checks += 1
    def rejects(call, description):
        nonlocal checks
        try: call()
        except ValueError: checks += 1
        else: raise AssertionError('Invalid boundary response accepted: '+description)
    with tempfile.TemporaryDirectory(prefix='cos-boundary-runtime-') as temporary:
        base = Path(temporary)
        with patch('reference_appearance_stack.validate_layer', return_value={}):
            # A categorical semantic proxy has a noninteger scale in both
            # axes. Cropping a broad producer ROI into a tighter frozen patch
            # must retain each source pixel's owner, including class borders.
            owner_root = base/'owner-roi-counterexample'
            owner_run, owner_before, _, owner_alpha, _, _, owner_protection = fixture(owner_root, fractional=True)
            owner_plan = design.plan_for_run(owner_run); owner_layer = _layer(owner_plan)
            yy, xx = np.indices((11, 17))
            proxy_labels = np.where((3*xx+5*yy) % 7 < 3, 4, 23).astype(np.uint8)
            owner_path = owner_root/'small-semantic-proxy.npz'; np.savez_compressed(owner_path, labels=proxy_labels)
            owner_plan['canvas']['source_proxy_size'] = [17, 11]
            owner_plan['mask_roles']['source_semantics'] = {
                **owner_plan['mask_roles']['source_semantics'], **record(owner_path), 'size': [17, 11]}
            full_owners = boundary.source_owners(owner_plan, (0, 0, 48, 32))
            wide_box, tight_box = (3, 2, 45, 29), (9, 7, 36, 24)
            wide_owners = boundary.source_owners(owner_plan, wide_box)
            tight_owners = boundary.source_owners(owner_plan, tight_box)
            for box in (wide_box, tight_box, (1, 1, 47, 31), (12, 4, 25, 27)):
                l, t, r, b = box
                require(np.array_equal(boundary.source_owners(owner_plan, box), full_owners[t:b, l:r]))
            require(np.array_equal(tight_owners, wide_owners[5:22, 6:33]))
            # The float-crop rounding defect needs a realistic proxy/native
            # ratio; small source grids need not land on its rounding ties.
            large_plan = copy.deepcopy(owner_plan)
            ly, lx = np.indices((1024, 1536))
            large_labels = np.where((3*lx+5*ly) % 7 < 3, 4, 23).astype(np.uint8)
            large_path = owner_root/'large-semantic-proxy.npz'; np.savez_compressed(large_path, labels=large_labels)
            large_plan['canvas'].update(source_proxy_size=[1536, 1024], source_oriented_size=[5216, 3472])
            large_plan['mask_roles']['source_semantics'] = {
                **owner_plan['mask_roles']['source_semantics'], **record(large_path),
                'size': [1536, 1024], 'native_size': [5216, 3472]}
            large_wide, large_tight = (90, 90, 700, 900), (100, 100, 600, 800)
            require(np.array_equal(boundary.source_owners(large_plan, large_tight),
                boundary.source_owners(large_plan, large_wide)[10:710, 10:510]))
            legacy_wide = design.sample_asset(large_plan['mask_roles']['source_semantics'], 'labels',
                large_wide, (5216, 3472), sampling='nearest')
            legacy_tight = design.sample_asset(large_plan['mask_roles']['source_semantics'], 'labels',
                large_tight, (5216, 3472), sampling='nearest')
            legacy_owner_changes = int(np.count_nonzero(legacy_tight != legacy_wide[10:710, 10:510]))
            require(legacy_owner_changes > 0)
            l, t, r, b = tight_box
            tight_delta = np.float32(.002)*owner_alpha[t:b, l:r, None]*(
                (tight_owners == 4) & (owner_protection[t:b, l:r] < .5))[..., None]
            tight_delta = tight_delta*np.ones((1, 1, 3), np.float32)
            delta_path = owner_root/'tight-native-owner-delta.npz'; np.savez_compressed(delta_path, patch_0=tight_delta)
            frozen = json.loads(Path(owner_layer['assets']['frozen_delta']['path']).read_text('utf8'))
            frozen['binding'] = boundary.endpoint_binding(owner_plan, owner_run)
            frozen['delta'] = record(delta_path)
            frozen['patches'] = [{'native_box': list(tight_box), 'array_key': 'patch_0',
                'delta_array_sha256': array_sha(tight_delta), 'target_patch_sha256': array_sha(owner_before[t:b, l:r]),
                'changed_pixels': int(np.any(tight_delta != 0, -1).sum())}]
            qa = json.loads(Path(owner_layer['assets']['qualification_receipt']['path']).read_text('utf8'))
            qa['changed_pixels'] = frozen['patches'][0]['changed_pixels']
            owner_layer['params']['binding'] = copy.deepcopy(frozen['binding'])
            owner_layer['assets']['frozen_delta'] = write(owner_root/'tight-native-owner-frozen.json', frozen)
            owner_layer['assets']['qualification_receipt'] = write(owner_root/'tight-native-owner-qualification.json', qa)
            owner_delta = np.zeros_like(owner_before); owner_delta[t:b, l:r] = tight_delta
            owner_values, owner_state = boundary.apply_boundary_light(owner_before, owner_layer, owner_plan, owner_run)
            require(np.array_equal(owner_values, owner_before+owner_delta) and owner_state['applied'])
            owner_crop, _ = boundary.apply_boundary_light(owner_before[t:b, l:r], owner_layer, owner_plan, owner_run,
                box=tight_box, canvas_size=(48, 32))
            require(np.array_equal(owner_crop, owner_values[t:b, l:r]))
            require(np.array_equal(owner_values[full_owners != 4], owner_before[full_owners != 4]))
            for mode in ('appearance-environment', 'appearance-detail-environment', 'fractional-appearance'):
                origin = base/('origin-'+mode)
                run, before, delta, alpha, background, labels, protection = fixture(origin,
                    with_detail=mode == 'appearance-detail-environment', fractional=mode == 'fractional-appearance')
                plan = design.plan_for_run(run); layer = _layer(plan)
                result, state = boundary.apply_boundary_light(before, layer, plan, run)
                require(np.array_equal(result, before+delta) and state['applied'])
                require(state['source_alpha_changed'] is False and state['source_geometry_changed'] is False
                    and state['clothing_enabled'] is False and state['visual_acceptance'] is False)
                for excluded in (alpha == 0, labels != 4, protection >= .5):
                    require(np.array_equal(result[excluded], before[excluded]))
                if mode == 'fractional-appearance':
                    partial = (alpha > 0) & (alpha < .95) & (labels == 4)
                    require(partial.any() and np.any(delta[partial] > 0))
                roi = (9, 7, 23, 18)
                clipped, _ = boundary.apply_boundary_light(before[7:18, 9:23], layer, plan, run,
                    box=roi, canvas_size=(48, 32))
                require(np.array_equal(clipped, result[7:18, 9:23]))
                outside, _ = boundary.apply_boundary_light(before[:2, :4], layer, plan, run,
                    box=(0, 0, 4, 2), canvas_size=(48, 32))
                require(np.array_equal(outside, before[:2, :4]))
                preview_before = _sample(before, (0, 0, 24, 16), (24, 16))
                preview_delta = _sample(delta, (0, 0, 24, 16), (24, 16))
                reduced, _ = boundary.apply_boundary_light(preview_before, layer, plan, run)
                require(np.array_equal(reduced, preview_before+preview_delta))
                geometry = Image.fromarray(np.uint8(alpha*255))
                zeros = np.zeros_like(before)
                whole, execution = design.apply_plan(zeros, zeros, plan, geometry, run=run, include_optics=False)
                require(np.array_equal(whole, before+delta+(1-alpha[..., None])*background))
                require(next(row for row in execution['layers'] if row['kind'] == boundary.KIND)['applied'])
                native_roi, roi_execution = render_native_roi(run, roi)
                require(np.array_equal(native_roi, whole[7:18, 9:23]))
                require(next(row for row in roi_execution['layers'] if row['kind'] == boundary.KIND)['applied'])
                small = np.zeros((16, 24, 3), np.float32)
                layer['enabled'] = False
                preview_base, _ = design.apply_plan(small, small, plan, geometry.resize((24, 16)), run=run, include_optics=False)
                layer['enabled'] = True
                preview, _ = design.apply_plan(small, small, plan, geometry.resize((24, 16)), run=run, include_optics=False)
                require(np.allclose(preview, preview_base+preview_delta, rtol=0, atol=3e-8))
                for enabled, strength in ((False, 1.), (True, 0.), (True, .5)):
                    layer.update(enabled=enabled, strength=strength)
                    expected = before+delta*np.float32(strength if enabled else 0)
                    replay, replay_state = boundary.apply_boundary_light(before, layer, plan, run)
                    require(np.array_equal(replay, expected))
                    require(replay_state['applied'] == bool(enabled and strength))
                    full, _ = design.apply_plan(zeros, zeros, plan, geometry, run=run, include_optics=False)
                    crop, _ = render_native_roi(run, roi)
                    require(np.array_equal(full, expected+(1-alpha[..., None])*background)
                        and np.array_equal(crop, full[7:18, 9:23]))
                layer.update(enabled=True, strength=1.)
                for mutation in ('appearance-dose', 'foreground-control', 'edited-matte', 'environment-dose'):
                    if mutation == 'environment-dose' and mode == 'fractional-appearance': continue
                    changed = copy.deepcopy(run); changed_plan = design.plan_for_run(changed)
                    if mutation == 'appearance-dose':
                        next(row for row in changed_plan['layers'] if row['kind'] == 'reference_appearance_stack')['strength'] = .5
                    elif mutation == 'foreground-control':
                        changed['options']['appearance_revision']['params'] = {'subject': {'exposure': .1}}
                    elif mutation == 'edited-matte': changed_plan['mask_roles']['edited_edge_light_applicability'] = {'path': 'unread-test-marker'}
                    else: next(row for row in changed_plan['layers'] if row['kind'] == 'source_environment_response')['strength'] = .5
                    unchanged, bypass = boundary.apply_boundary_light(before, _layer(changed_plan), changed_plan, changed)
                    require(np.array_equal(unchanged, before) and bypass['bypass_reason'] ==
                        'boundary_light_endpoint_changed_requires_new_preparation')
                for mutation in ('negative', 'zero-alpha', 'clothing', 'skin', 'unknown', 'protection', 'overflow',
                        'nonfinite', 'float64', 'target-hash', 'array-hash', 'delta-file-hash', 'frozen-file-hash',
                        'observation-file-hash', 'count', 'duplicate', 'source', 'strength'):
                    changed = copy.deepcopy(run); changed_plan = design.plan_for_run(changed); changed_layer = _layer(changed_plan)
                    frozen = json.loads(Path(layer['assets']['frozen_delta']['path']).read_text('utf8'))
                    qa = json.loads(Path(layer['assets']['qualification_receipt']['path']).read_text('utf8'))
                    patch_row = frozen['patches'][0]
                    if mutation in ('negative', 'zero-alpha', 'clothing', 'skin', 'unknown', 'protection', 'overflow', 'nonfinite', 'float64'):
                        with np.load(frozen['delta']['path'], allow_pickle=False) as data:
                            arrays = {key: data[key].copy() for key in data.files}
                        key = patch_row['array_key']; l, t, _, _ = patch_row['native_box']
                        positions = {'zero-alpha': (2, 6), 'clothing': (10, 9), 'skin': (11, 9),
                            'unknown': (12, 9), 'protection': (7, 15)}
                        if mutation in positions:
                            y, x = positions[mutation]; arrays[key][y-t, x-l] = .001
                        elif mutation == 'float64': arrays[key] = arrays[key].astype(np.float64)
                        else: arrays[key][8-t, 16-l, 0] = {'negative': -.001, 'overflow': 2., 'nonfinite': np.nan}[mutation]
                        patch_row['delta_array_sha256'] = array_sha(arrays[key])
                        patch_row['changed_pixels'] = int(np.any(arrays[key] != 0, -1).sum())
                        qa['changed_pixels'] = patch_row['changed_pixels']
                        path = origin/('invalid-'+mutation+'.npz'); np.savez_compressed(path, **arrays)
                        frozen['delta'] = record(path)
                    elif mutation == 'target-hash': patch_row['target_patch_sha256'] = '0'*64
                    elif mutation == 'array-hash': patch_row['delta_array_sha256'] = '0'*64
                    elif mutation == 'delta-file-hash': frozen['delta']['sha256'] = '0'*64
                    elif mutation == 'observation-file-hash': frozen['observation_assets']['receipt']['sha256'] = '0'*64
                    elif mutation == 'count': qa['changed_pixels'] += 1
                    elif mutation == 'duplicate': changed_plan['layers'].append(copy.deepcopy(changed_layer))
                    elif mutation == 'source': changed['source']['sha256'] = '0'*64
                    elif mutation == 'strength': changed_layer['strength'] = 1.1
                    changed_layer['assets']['frozen_delta'] = write(origin/('invalid-'+mutation+'.json'), frozen)
                    changed_layer['assets']['qualification_receipt'] = write(origin/('invalid-'+mutation+'-qa.json'), qa)
                    if mutation == 'frozen-file-hash': changed_layer['assets']['frozen_delta']['sha256'] = '0'*64
                    rejects(lambda: boundary.apply_boundary_light(before, changed_layer, changed_plan, changed), mutation)
                rejects(lambda: boundary.apply_boundary_light(np.zeros((16, 21, 3), np.float32), layer, plan, run), 'nonproportional-preview')
                rejects(lambda: boundary.apply_boundary_light(before[:3, :3], layer, plan, run,
                    box=(0, 0, 4, 3), canvas_size=(48, 32)), 'roi-buffer-size')
                original_binding = boundary.endpoint_binding(plan, run)
                original_frozen_sha = layer['assets']['frozen_delta']['sha256']
                saved = base/('saved-'+mode); owned_run = copy.deepcopy(run)
                owned_plan = design.materialize_plan(saved, owned_run, plan)
                owned_run['root'] = str(saved); owned_run['options']['appearance_revision']['design_reconstruction_plan'] = owned_plan
                require(boundary.endpoint_binding(owned_plan, owned_run) == original_binding)
                require(_layer(owned_plan)['assets']['frozen_delta']['sha256'] != original_frozen_sha)
                owned_records = closure(owned_plan); require(owned_records > 12)
                write(saved/'outputs/runs/fixture/manifest.json', owned_run)
                archive = base/(mode+'.cosplay'); export_project(saved, 'fixture', archive)
                moved = import_project(base/('portable-'+mode), archive, run_id='relocated')['run']
                moved_plan = design.plan_for_run(moved)
                require(boundary.endpoint_binding(moved_plan, moved) == original_binding)
                require(closure(moved_plan) >= owned_records)
                require(origin.resolve().is_relative_to(base.resolve()) and origin.name.startswith('origin-'))
                shutil.rmtree(origin); require(not origin.exists())
                for relocated in (owned_run, moved):
                    relocated_plan = design.plan_for_run(relocated)
                    replay, replay_state = boundary.apply_boundary_light(before, _layer(relocated_plan), relocated_plan, relocated)
                    require(np.array_equal(replay, before+delta) and replay_state['applied'])
                    replay_full, _ = design.apply_plan(zeros, zeros, relocated_plan, geometry, run=relocated, include_optics=False)
                    require(np.array_equal(replay_full, whole))
                    replay_roi, _ = render_native_roi(relocated, roi)
                    require(np.array_equal(replay_roi, whole[7:18, 9:23]))
                    replay_preview, _ = design.apply_plan(small, small, relocated_plan, geometry.resize((24, 16)), run=relocated, include_optics=False)
                    # Materialization records current paired source-preview
                    # responsibility; compare boundary on/off on that endpoint.
                    relocated_layer = _layer(relocated_plan); relocated_layer['enabled'] = False
                    base_preview, _ = design.apply_plan(small, small, relocated_plan, geometry.resize((24, 16)), run=relocated, include_optics=False)
                    relocated_layer['enabled'] = True
                    require(np.allclose(replay_preview, base_preview+preview_delta, rtol=0, atol=3e-8))
    return {'ok': True, 'checks': checks, 'actual_boundary_runtime_and_endpoint': True,
        'appearance_validator_mocked': True, 'native_roi_preview_real_consumers': True,
        'fractional_hair_alpha_supported': True, 'zero_alpha_nonhair_and_identity_excluded': True,
        'negative_overflow_and_corrupt_delta_rejected': True, 'endpoint_changes_bypass': True,
        'disabled_and_zero_dose_bypass_half_dose_exact': True,
        'materialization_and_portable_bindings_stable_after_json_locator_changes': True,
        'portable_replay_after_original_assets_removed': True, 'model_inferences': 0,
        'noninteger_semantic_scale_and_unaligned_roi_owner_invariance': True,
        'legacy_pil_nearest_crop_counterexample_reproduced': True,
        'legacy_counterexample_changed_owner_pixels': legacy_owner_changes,
        'scope': 'Synthetic runtime replay and ownership proof; no real-image or reference-observation qualification asserted.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path); args = parser.parse_args()
    report = run_smoke()
    if args.output: write(args.output, report)
    print(json.dumps(report, ensure_ascii=False))
