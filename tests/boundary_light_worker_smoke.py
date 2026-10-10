"""Actual hair worker lifecycle with mocked observations/producer; no art gate."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np

import boundary_light_runtime as hair
import boundary_light_worker as worker
import cloth_boundary_runtime as cloth
import design_reconstruction as design
from cloth_boundary_runtime_smoke import fixture
from native_subject_detail_runtime_smoke import record, write, sha, closure


def run_smoke():
    checks = 0
    def require(value):
        nonlocal checks
        assert value; checks += 1
    def rejects(call):
        nonlocal checks
        try: call()
        except ValueError: checks += 1
        else: raise AssertionError('Invalid hair preparation accepted')
    with tempfile.TemporaryDirectory(prefix='cos-hair-worker-') as temporary:
        root = Path(temporary)
        with patch('reference_appearance_stack.validate_layer', return_value={}):
            run, _, _, alpha, _, labels, protection = fixture(root/'inputs', with_hair=True)
            run['root'] = str(root); plan = design.plan_for_run(run)
            plan['mask_roles']['candidate_occupancy'] = plan['mask_roles']['foreground_alpha']
            run['options']['appearance_revision']['params'] = {}; run['options']['manual_layers'] = {}
            run['auto_draft'] = {'path': run['source']['path']}
            old_hair = next(row for row in plan['layers'] if row['kind'] == hair.KIND)
            old_hair['id'] = 'user-hair-layer'
            old_cloth = next(row for row in plan['layers'] if row['kind'] == cloth.KIND)
            require(cloth.validate_layer(old_cloth, plan, run)['bypass_reason'] is None)
            parent = root/'outputs/runs/fixture/manifest.json'; write(parent, run)
            pointer = root/'project.json'; write(pointer, {'kind': 'project', 'state': {'last_source_run': 'fixture'}})
            retained = (parent.read_bytes(), pointer.read_bytes())
            old_frozen = json.loads(Path(old_hair['assets']['frozen_delta']['path']).read_text('utf8'))
            unchanged_files = {value['path']: sha(value['path']) for value in old_hair['assets'].values()}
            unchanged_files[old_frozen['delta']['path']] = sha(old_frozen['delta']['path'])
            cache = root/'mock-reference'; cache.mkdir(); matte = {'cache_hit': True, 'tiles': {'alpha_tiles': 0}}
            for key, values in [('alpha_path', np.ones((32, 48), np.float32)),
                    ('premultiplied_foreground_path', np.full((32, 48, 3), .12, np.float32)),
                    ('unknown_edge_support_path', np.ones((32, 48), np.float32))]:
                path = cache/(key+'.npz'); np.savez_compressed(path, values=values); matte[key] = str(path)
            path = cache/'diagnostics.npz'; np.savez_compressed(path, values=alpha); matte['fb_diagnostics_path'] = str(path)
            matte['result_path'] = str(cache/'result.json'); write(Path(matte['result_path']), matte)
            yy, xx = np.indices((32, 48), dtype=np.float32)
            maps = {'ref_x': (xx+.5)*.5-.5, 'ref_y': (yy+.5)*.5-.5, 'valid': np.ones((32, 48), np.float32)}
            observed = (labels, np.full((16, 24), 4, np.uint8), np.zeros((16, 24, 3), np.float32),
                maps, {'reference_labels': plan['mask_roles']['source_semantics']})
            def producer(p, a, reference, ra, sl, rl, native_maps, **kw):
                require(p.shape == reference.shape == kw['reference_fp'].shape == (32, 48, 3))
                require(a.shape == ra.shape == sl.shape == rl.shape == kw['reference_fb_support'].shape == (32, 48))
                require(np.array_equal(sl, labels) and np.array_equal(kw['identity_protection'], protection))
                require(np.allclose(native_maps['ref_x'], xx) and np.allclose(native_maps['ref_y'], yy))
                expected = hair.endpoint(plan, run, (0, 0, 48, 32), (48, 32))
                # The old target layer and all later cloth must be excluded.
                if run['options']['appearance_revision']['params'] == {}:
                    require(np.array_equal(p, expected))
                delta = np.zeros_like(p); allowed = (sl == 4)&(a > 0)&(kw['identity_protection'] < .5)
                delta[allowed] = np.float32(.006)*a[allowed, None]
                return {'delta_p': delta, 'receipt': {'state': 'prepared', 'mock_producer': True}}
            def prepare(current, name, callback=producer):
                with patch.object(worker, 'observations', return_value=observed), \
                        patch('source_foreground_cache.prepare_source_foreground', return_value=matte), \
                        patch.object(worker, 'prepare_boundary_light', side_effect=callback):
                    return worker.prepare(root, current, root/name)
            result = prepare(run, 'prepared'); result_plan = result['design_reconstruction_plan']
            new_hair = next(row for row in result_plan['layers'] if row['kind'] == hair.KIND)
            require(result['changed'] and new_hair['id'] == old_hair['id'])
            require(new_hair['enabled'] is True and new_hair['strength'] == 1)
            require(result_plan['layers'].index(new_hair) < result_plan['layers'].index(old_cloth))
            require(next(row for row in result_plan['layers'] if row['kind'] == cloth.KIND) == old_cloth)
            require(hair.validate_layer(new_hair, result_plan, run)['bypass_reason'] is None)
            require(cloth.validate_layer(old_cloth, result_plan, run)['bypass_reason'] ==
                'cloth_boundary_endpoint_changed_requires_new_preparation')
            before = hair.endpoint(plan, run, (0, 0, 48, 32), (48, 32))
            after, state = hair.apply_boundary_light(before, new_hair, result_plan, run)
            require(state['applied'] and np.any(after != before))
            cloth_after, cloth_state = cloth.apply_cloth_boundary_light(after, old_cloth, result_plan, run)
            require(np.array_equal(cloth_after, after) and not cloth_state['applied'])
            require(all(sha(path) == digest for path, digest in unchanged_files.items()))
            for enabled in (True, False):
                current = copy.deepcopy(run); current_hair = next(row for row in design.plan_for_run(current)['layers'] if row['kind'] == hair.KIND)
                current_hair.update(enabled=enabled, strength=.35)
                old_current = copy.deepcopy(current)
                replaced = prepare(current, 'upsert-'+str(enabled))
                layers = replaced['design_reconstruction_plan']['layers']; replacement = next(row for row in layers if row['kind'] == hair.KIND)
                require(replaced['changed'] and replacement['enabled'] is enabled and replacement['strength'] == .35)
                require(replacement['id'] == 'user-hair-layer' and sum(row['kind'] == hair.KIND for row in layers) == 1)
                require(current == old_current)
            identical_run = copy.deepcopy(run)
            identical_run['options']['appearance_revision']['design_reconstruction_plan'] = copy.deepcopy(result_plan)
            identical_layer = next(row for row in design.plan_for_run(identical_run)['layers'] if row['kind'] == hair.KIND)
            identical_layer.update(enabled=False, strength=.2)
            frozen = json.loads(Path(identical_layer['assets']['frozen_delta']['path']).read_text('utf8'))
            with np.load(frozen['delta']['path'], allow_pickle=False) as data: arrays = {key: data[key].copy() for key in data.files}
            different_transport = root/'same-arrays-uncompressed.npz'; np.savez(different_transport, **arrays)
            require(sha(different_transport) != frozen['delta']['sha256'])
            frozen['delta'] = record(different_transport)
            identical_layer['assets']['frozen_delta'] = write(root/'same-arrays-frozen.json', frozen)
            identical = prepare(identical_run, 'identical')
            require(not identical['changed'] and identical['receipt']['reason'] == 'identical_frozen_pixels')
            require(identical['design_reconstruction_plan'] == design.plan_for_run(identical_run))
            snapshot = root/'identical-input.json'; write(snapshot, identical_run)
            run_dirs = sorted(path.name for path in (root/'outputs/runs').iterdir())
            with patch.object(worker, 'prepare', return_value=identical), patch.object(sys, 'argv', [
                    'boundary_light_worker.py', '--root', str(root), '--run-json', str(snapshot),
                    '--output', str(root/'identical'), '--save-child']): worker.main()
            require(sorted(path.name for path in (root/'outputs/runs').iterdir()) == run_dirs)
            duplicate = copy.deepcopy(run); design.plan_for_run(duplicate)['layers'].append(copy.deepcopy(old_hair))
            with patch.object(worker, 'observations', side_effect=AssertionError('Duplicate entered observations')):
                rejects(lambda: worker.prepare(root, duplicate, root/'duplicate'))
            unchanged = prepare(run, 'unchanged', lambda *a, **kw: {'delta_p': np.zeros((32, 48, 3), np.float32), 'receipt': {'state': 'no_change'}})
            require(not unchanged['changed'] and unchanged['design_reconstruction_plan'] == plan)
            for mutation in ('negative', 'float64', 'wrong-state'):
                def bad(*args, **kwargs):
                    value = producer(*args, **kwargs)
                    if mutation == 'negative': value['delta_p'] *= -1
                    elif mutation == 'float64': value['delta_p'] = value['delta_p'].astype(np.float64)
                    else: value['receipt']['state'] = 'rejected'
                    return value
                rejects(lambda: prepare(run, 'bad-'+mutation, bad))
            edited = copy.deepcopy(run); edited['options']['appearance_revision']['params'] = {'background': {'exposure': .17}}
            # The producer remains synthetic, while its real worker binds the
            # changed options and real native endpoint into the new freeze.
            run['options']['appearance_revision']['params'] = edited['options']['appearance_revision']['params']
            edited_result = prepare(edited, 'edited-prepared')
            run['options']['appearance_revision']['params'] = {}
            snapshot = root/'edited-input.json'; write(snapshot, edited)
            with patch.object(worker, 'prepare', return_value=edited_result), patch.object(sys, 'argv', [
                    'boundary_light_worker.py', '--root', str(root), '--run-json', str(snapshot),
                    '--output', str(root/'edited-prepared'), '--save-child']): worker.main()
            saved = json.loads((root/'edited-prepared/result.json').read_text('utf8'))
            child = json.loads(Path(saved['child_manifest']).read_text('utf8')); child_plan = design.plan_for_run(child)
            saved_hair = next(row for row in child_plan['layers'] if row['kind'] == hair.KIND)
            require(child['options']['appearance_revision']['params'] == edited['options']['appearance_revision']['params'])
            require(Path(saved_hair['assets']['frozen_delta']['path']).is_relative_to(root/'outputs/runs'/child['run_id']))
            require(hair.validate_layer(saved_hair, child_plan, child)['bypass_reason'] is None)
            require(child['options']['preserve_project_selection'] is True and closure(child_plan) > 20)
            require((parent.read_bytes(), pointer.read_bytes()) == retained)
    return {'ok': True, 'checks': checks, 'actual_worker_endpoint_and_frozen_runtime': True,
        'producer_observations_and_matte_mocked': True, 'upsert_preserves_id_enabled_strength_and_cloth_order': True,
        'new_hair_retains_old_cloth_with_explicit_stale_bypass': True, 'same_array_different_npz_no_child': True,
        'duplicate_and_corrupt_results_rejected': True, 'cli_edited_background_and_child_owned_plan': True,
        'parent_and_original_hair_assets_retained': True, 'model_inferences': 0,
        'image_generation_submissions': 0, 'visual_acceptance': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path)
    args = parser.parse_args(); report = run_smoke()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n', 'utf8')
    print(json.dumps(report))
