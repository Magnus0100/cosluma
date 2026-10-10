"""Real cloth worker/frozen consumers with synthetic observations and producer.

No model runs or real-image qualification is claimed by this test.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
from unittest.mock import patch

import numpy as np

import cloth_boundary_runtime as cloth
import design_reconstruction as design
from cloth_boundary_runtime_smoke import fixture
from native_subject_detail_runtime import array_sha
from native_subject_detail_runtime_smoke import record, write, sha, closure
from portable_project import export_project, import_project


def run_smoke():
    checks = 0
    def require(value):
        nonlocal checks
        assert value; checks += 1
    def rejects(call, exception=ValueError):
        nonlocal checks
        try: call()
        except exception: checks += 1
        else: raise AssertionError('Invalid worker result accepted')

    producer_module = types.ModuleType('reference_cloth_boundary_transfer')
    producer_module.prepare_cloth_boundary_light = lambda *a, **kw: None
    with patch.dict(sys.modules, {'reference_cloth_boundary_transfer': producer_module}):
        worker = importlib.import_module('cloth_boundary_worker')
    with tempfile.TemporaryDirectory(prefix='cos-cloth-worker-') as temporary:
        base = Path(temporary); root = base/'project'
        with patch('reference_appearance_stack.validate_layer', return_value={}):
            run, _, _, alpha, _, labels, protection = fixture(root/'inputs', with_hair=True)
            run['root'] = str(root); plan = design.plan_for_run(run)
            plan['layers'] = [row for row in plan['layers'] if row['kind'] != cloth.KIND]
            plan['mask_roles']['candidate_occupancy'] = plan['mask_roles']['foreground_alpha']
            run['auto_draft'] = {'path': run['source']['path']}
            run['options']['appearance_revision']['params'] = {}
            run['options']['manual_layers'] = {}
            parent_path = root/'outputs/runs/fixture/manifest.json'; write(parent_path, run)
            pointer = root/'project.json'; write(pointer, {'kind': 'project', 'state': {'last_source_run': 'fixture'}})
            retained = (parent_path.read_bytes(), pointer.read_bytes())
            hair_layer = next(row for row in plan['layers'] if row['kind'] == 'source_boundary_light')
            hair_frozen = json.loads(Path(hair_layer['assets']['frozen_delta']['path']).read_text('utf8'))
            hair_files = {value['path']: sha(value['path']) for value in hair_layer['assets'].values()}
            hair_files[hair_frozen['delta']['path']] = sha(hair_frozen['delta']['path'])
            cache = root/'mock-reference-observations'; cache.mkdir()
            matte = {'cache_hit': True, 'tiles': {'alpha_tiles': 0}}
            for key, values in [('alpha_path', np.ones((32, 48), np.float32)),
                    ('premultiplied_foreground_path', np.full((32, 48, 3), .12, np.float32)),
                    ('unknown_edge_support_path', np.ones((32, 48), np.float32))]:
                path = cache/(key+'.npz'); np.savez_compressed(path, values=values); matte[key] = str(path)
            diag = cache/'diagnostics.npz'; np.savez_compressed(diag, values=np.ones((32, 48), np.float32))
            matte['fb_diagnostics_path'] = str(diag)
            matte['result_path'] = str(cache/'result.json'); write(Path(matte['result_path']), matte)
            yy, xx = np.indices((32, 48), dtype=np.float32)
            maps = {'ref_x': (xx+.5)*.5-.5, 'ref_y': (yy+.5)*.5-.5,
                'valid': np.ones((32, 48), np.float32)}
            observations = (labels, np.full((16, 24), 23, np.uint8),
                np.zeros((16, 24, 3), np.float32), maps,
                {'reference_labels': plan['mask_roles']['source_semantics']})
            calls = []
            def producer(p, a, reference, ra, sl, rl, native_maps, **kw):
                require(p.dtype == np.float32 and p.shape == (32, 48, 3))
                require(a.shape == sl.shape == kw['validity'].shape == kw['identity_protection'].shape == (32, 48))
                require(reference.shape == kw['reference_fp'].shape == (32, 48, 3))
                require(ra.shape == rl.shape == kw['reference_fb_support'].shape == (32, 48))
                require(np.array_equal(sl, labels) and np.array_equal(kw['identity_protection'], protection))
                require(np.allclose(native_maps['ref_x'], xx, atol=1e-6) and np.allclose(native_maps['ref_y'], yy, atol=1e-6))
                fp = design.sample_asset(plan['source_baseline']['assets']['foreground_premultiplied'], 'values', (0, 0, 48, 32), (48, 32))
                source_assets = plan['source_baseline']['assets']
                original_alpha = design.sample_asset(source_assets['foreground_alpha'], 'values', (0, 0, 48, 32), (48, 32))
                old_environment = design.sample_asset(source_assets['old_environment'], 'values', (0, 0, 48, 32), (48, 32))
                expected_source = fp+(1-original_alpha[..., None])*old_environment
                require(np.array_equal(kw['source_rgb'], expected_source))
                require(np.any(p != fp))
                delta = np.zeros_like(p); allowed = (sl == 23)&(a > 0)&(kw['identity_protection'] < .5)
                delta[allowed] = np.float32(.002)*a[allowed, None]
                calls.append(copy.deepcopy(kw))
                return {'delta_p': delta, 'receipt': {'state': 'prepared', 'mock_producer': True}}
            with patch.object(worker, 'observations', return_value=observations), \
                    patch('source_foreground_cache.prepare_source_foreground', return_value=matte), \
                    patch.object(worker, 'prepare_cloth_boundary_light', side_effect=producer):
                result = worker.prepare(root, run, root/'prepared')
            require(result['changed'] and result['receipt']['changed_pixels'] > 0)
            require(result['receipt']['model_tile_inferences_this_attempt'] == 0 and result['receipt']['hair_changed'] is False)
            result_plan = result['design_reconstruction_plan']; layer = result_plan['layers'][-1]
            require(cloth.validate_layer(layer, result_plan, run)['bypass_reason'] is None)
            require(layer['params']['binding'] == cloth.endpoint_binding(plan, run))
            require((parent_path.read_bytes(), pointer.read_bytes()) == retained)
            require(all(sha(path) == digest for path, digest in hair_files.items()))
            before = cloth.endpoint(plan, run, (0, 0, 48, 32), (48, 32))
            after, status = cloth.apply_cloth_boundary_light(before, layer, result_plan, run)
            require(status['applied'] and np.any(after != before))
            require(np.array_equal(after[labels != 23], before[labels != 23]))
            for mutation in ('negative', 'float64', 'wrong-state'):
                def bad(*args, **kwargs):
                    value = producer(*args, **kwargs)
                    if mutation == 'negative': value['delta_p'] *= -1
                    elif mutation == 'float64': value['delta_p'] = value['delta_p'].astype(np.float64)
                    else: value['receipt']['state'] = 'rejected'
                    return value
                with patch.object(worker, 'observations', return_value=observations), \
                        patch('source_foreground_cache.prepare_source_foreground', return_value=matte), \
                        patch.object(worker, 'prepare_cloth_boundary_light', side_effect=bad):
                    rejects(lambda: worker.prepare(root, run, root/('bad-'+mutation)))
            with patch.object(worker, 'observations', return_value=observations), \
                    patch('source_foreground_cache.prepare_source_foreground', return_value=matte), \
                    patch.object(worker, 'prepare_cloth_boundary_light', return_value={
                        'delta_p': np.zeros((32, 48, 3), np.float32), 'receipt': {'state': 'no_change'}}):
                unchanged = worker.prepare(root, run, root/'unchanged')
            require(not unchanged['changed'] and unchanged['receipt']['changed_pixels'] == 0)
            require(unchanged['design_reconstruction_plan'] == plan and not (root/'unchanged/frozen.json').exists())
            def prepare_again(current, name, callback=producer):
                with patch.object(worker, 'observations', return_value=observations), \
                        patch('source_foreground_cache.prepare_source_foreground', return_value=matte), \
                        patch.object(worker, 'prepare_cloth_boundary_light', side_effect=callback):
                    return worker.prepare(root, current, root/name)
            existing = copy.deepcopy(run)
            existing['options']['appearance_revision']['design_reconstruction_plan'] = copy.deepcopy(result_plan)
            existing_layer = design.plan_for_run(existing)['layers'][-1]
            existing_layer.update(id='user-cloth-layer', enabled=False, strength=.25)
            old_frozen = json.loads(Path(existing_layer['assets']['frozen_delta']['path']).read_text('utf8'))
            with np.load(old_frozen['delta']['path'], allow_pickle=False) as data:
                old_arrays = {key: data[key].copy() for key in data.files}
            different_transport = root/'same-arrays-uncompressed.npz'
            np.savez(different_transport, **old_arrays)
            require(sha(different_transport) != old_frozen['delta']['sha256'])
            old_frozen['delta'] = record(different_transport)
            existing_layer['assets']['frozen_delta'] = write(root/'same-arrays-frozen.json', old_frozen)
            identical = prepare_again(existing, 'identical')
            require(not identical['changed'] and identical['receipt']['reason'] == 'identical_frozen_pixels')
            require(identical['design_reconstruction_plan'] == design.plan_for_run(existing))
            existing_snapshot = root/'existing-input.json'; write(existing_snapshot, existing)
            run_dirs = sorted(path.name for path in (root/'outputs/runs').iterdir())
            with patch.object(worker, 'prepare', return_value=identical), patch.object(sys, 'argv', [
                    'cloth_boundary_worker.py', '--root', str(root), '--run-json', str(existing_snapshot),
                    '--output', str(root/'identical'), '--save-child']):
                worker.main()
            require(sorted(path.name for path in (root/'outputs/runs').iterdir()) == run_dirs)
            require('child_run_id' not in json.loads((root/'identical/result.json').read_text('utf8')))
            def changed_producer(*args, **kwargs):
                changed = producer(*args, **kwargs); changed['delta_p'] *= 2
                return changed
            for enabled in (True, False):
                existing_layer['enabled'] = enabled
                before_existing = copy.deepcopy(existing)
                replacement = prepare_again(existing, 'replaced-'+str(enabled), changed_producer)
                replacement_plan = replacement['design_reconstruction_plan']
                replacement_layer = next(row for row in replacement_plan['layers'] if row['kind'] == cloth.KIND)
                require(replacement['changed'] and replacement_layer['id'] == 'user-cloth-layer')
                require(replacement_layer['enabled'] is enabled and replacement_layer['strength'] == .25)
                require(sum(row['kind'] == cloth.KIND for row in replacement_plan['layers']) == 1)
                require(existing == before_existing)
                require([row for row in replacement_plan['layers'] if row['kind'] != cloth.KIND] == plan['layers'])
            duplicate = copy.deepcopy(existing); duplicate_plan = design.plan_for_run(duplicate)
            duplicate_plan['layers'].append(copy.deepcopy(duplicate_plan['layers'][-1]))
            with patch.object(worker, 'observations', side_effect=AssertionError('Duplicate entered observations')):
                rejects(lambda: worker.prepare(root, duplicate, root/'duplicate'))
            with patch.object(worker, 'prepare', return_value=unchanged), patch.object(sys, 'argv', [
                    'cloth_boundary_worker.py', '--root', str(root), '--run-id', 'fixture',
                    '--output', str(root/'unchanged'), '--save-child']):
                worker.main()
            require([path.name for path in (root/'outputs/runs').iterdir()] == ['fixture'])
            with patch.object(worker, 'prepare', return_value=result), patch.object(sys, 'argv', [
                    'cloth_boundary_worker.py', '--root', str(root), '--run-id', 'fixture',
                    '--output', str(root/'prepared'), '--save-child']):
                worker.main()
            saved_result = json.loads((root/'prepared/result.json').read_text('utf8'))
            child = json.loads(Path(saved_result['child_manifest']).read_text('utf8'))
            child_plan = design.plan_for_run(child)
            require(child['parent_run_id'] == 'fixture' and child['revision'] == 2)
            require(child['options']['preserve_project_selection'] is True and 'auto_draft' not in child)
            require(child['cosplay_post_review']['visual_acceptance'] is False)
            require(cloth.validate_layer(child_plan['layers'][-1], child_plan, child)['bypass_reason'] is None)
            require(closure(child_plan) > 20 and (parent_path.read_bytes(), pointer.read_bytes()) == retained)
            # --run-json must save the operative edit snapshot, rather than
            # rebuilding its settings from the unchanged saved parent.
            saved_no_hair = copy.deepcopy(run); saved_no_hair['run_id'] = 'edited-parent'
            no_hair_plan = design.plan_for_run(saved_no_hair)
            no_hair_plan['layers'] = [row for row in no_hair_plan['layers'] if row['kind'] != 'source_boundary_light']
            write(root/'outputs/runs/edited-parent/manifest.json', saved_no_hair)
            edited = copy.deepcopy(saved_no_hair)
            edited['options']['appearance_revision']['params'] = {'background': {'exposure': .15}}
            edited_plan = design.plan_for_run(edited); edited_layer = copy.deepcopy(layer)
            frozen = json.loads(Path(layer['assets']['frozen_delta']['path']).read_text('utf8'))
            frozen['binding'] = cloth.endpoint_binding(edited_plan, edited)
            for row in frozen['patches']:
                row['target_patch_sha256'] = array_sha(cloth.endpoint(edited_plan, edited, tuple(row['native_box']), (48, 32)))
            edited_layer['params']['binding'] = copy.deepcopy(frozen['binding'])
            edited_layer['assets']['frozen_delta'] = write(root/'edited-prepared/frozen.json', frozen)
            edited_plan['layers'].append(edited_layer)
            require(cloth.validate_layer(edited_layer, edited_plan, edited)['bypass_reason'] is None)
            snapshot = root/'edited-input.json'; write(snapshot, edited)
            edited_result = {**result, 'design_reconstruction_plan': edited_plan}
            with patch.object(worker, 'prepare', return_value=edited_result), patch.object(sys, 'argv', [
                    'cloth_boundary_worker.py', '--root', str(root), '--run-json', str(snapshot),
                    '--output', str(root/'edited-prepared'), '--save-child']):
                worker.main()
            edited_saved = json.loads((root/'edited-prepared/result.json').read_text('utf8'))
            edited_child = json.loads(Path(edited_saved['child_manifest']).read_text('utf8'))
            require(edited_child['options']['appearance_revision']['params'] == edited['options']['appearance_revision']['params'])
            saved_plan = design.plan_for_run(edited_child)
            require(cloth.validate_layer(saved_plan['layers'][-1], saved_plan, edited_child)['bypass_reason'] is None)
            require(Path(saved_plan['layers'][-1]['assets']['frozen_delta']['path']).is_relative_to(
                root/'outputs/runs'/edited_child['run_id']))
            require(Path(saved_plan['layers'][-1]['assets']['frozen_delta']['path']) != Path(edited_layer['assets']['frozen_delta']['path']))
            archive = base/'worker-child.cosplay'; export_project(root, child['run_id'], archive)
            moved = import_project(base/'portable', archive, run_id='relocated')['run']
            require(cloth.endpoint_binding(design.plan_for_run(moved), moved) == layer['params']['binding'])
            shutil.rmtree(root/'inputs'); shutil.rmtree(cache); shutil.rmtree(root/'prepared')
            for replay in (child, moved):
                replay_plan = design.plan_for_run(replay)
                values, state = cloth.apply_cloth_boundary_light(before, replay_plan['layers'][-1], replay_plan, replay)
                require(np.array_equal(values, after) and state['applied'])
            # Wrapper cancellation and no-change routing exercise the real wrapper.
            cancelled = base/'cancelled'
            rejects(lambda: worker.prepare_for_workbench(base, run, cancelled, cancel=lambda: True), InterruptedError)
            require(not cancelled.with_name('cancelled-input.json').exists())
            class Process:
                pid = 123456789
                returncode = None
                def poll(self): return self.returncode
                def terminate(self): self.returncode = -15
                def wait(self, timeout=None): self.returncode = -15; return -15
                def kill(self): self.returncode = -9
            process = Process(); cancellations = iter((False, True))
            def kill_tree(*args, **kwargs): process.returncode = -15
            with patch('current_model_registry.load_registry', return_value={'python': sys.executable}), \
                    patch.object(worker.subprocess, 'Popen', return_value=process), \
                    patch.object(worker.subprocess, 'run', side_effect=kill_tree):
                rejects(lambda: worker.prepare_for_workbench(base, run, cancelled,
                    cancel=lambda: next(cancellations)), InterruptedError)
            require(process.returncode == -15)
            require((parent_path.read_bytes(), pointer.read_bytes()) == retained)
    return {'ok': True, 'checks': checks, 'actual_worker_sampling_and_frozen_runtime': True,
        'mock_producer_and_observations': True, 'sampling_keys_shapes_source_rgb_and_native_maps_checked': True,
        'producer_contract_rejected_before_success': True, 'save_child_no_change_cancel_and_portable': True,
        'original_hair_assets_unchanged': True, 'model_inferences': 0, 'image_generation_submissions': 0,
        'upsert_preserves_id_enabled_and_strength': True, 'same_array_different_npz_transport_preserves_original_plan_without_child': True,
        'duplicate_rejected_before_observations': True, 'cli_edited_background_and_child_owned_plan_preserved': True,
        'visual_acceptance': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path)
    args = parser.parse_args(); result = run_smoke()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', 'utf8')
    print(json.dumps(result, ensure_ascii=False))
