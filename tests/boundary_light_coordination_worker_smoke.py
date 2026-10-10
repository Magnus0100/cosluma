"""Atomic two-stage lifecycle on actual frozen consumers with synthetic observations."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import boundary_light_coordination_worker as coordinator
import boundary_light_worker as hair_worker
import cloth_boundary_worker as cloth_worker
import boundary_light_runtime as hair
import cloth_boundary_runtime as cloth
import design_reconstruction as design
from cloth_boundary_runtime_smoke import fixture
from native_subject_detail_runtime_smoke import record, write, sha, closure


def run_smoke():
    checks = 0
    def require(value):
        nonlocal checks
        assert value; checks += 1
    def rejects(call, exception=ValueError):
        nonlocal checks
        try: call()
        except exception: checks += 1
        else: raise AssertionError('Invalid atomic coordination accepted')
    with tempfile.TemporaryDirectory(prefix='boundary-coordination-') as temporary:
        root = Path(temporary)
        with patch('reference_appearance_stack.validate_layer', return_value={}):
            run, _, _, alpha, _, labels, protection = fixture(root/'inputs', with_hair=True)
            run['root'] = str(root); plan = design.plan_for_run(run)
            plan['mask_roles']['candidate_occupancy'] = plan['mask_roles']['foreground_alpha']
            run['options']['appearance_revision']['params'] = {'background': {'exposure': .15}}
            run['options']['manual_layers'] = {}; run['auto_draft'] = {'path': run['source']['path']}
            for kind, identifier in ((hair.KIND, 'my-hair'), (cloth.KIND, 'my-cloth')):
                next(row for row in plan['layers'] if row['kind'] == kind)['id'] = identifier
            parent = root/'outputs/runs/fixture/manifest.json'; write(parent, run)
            pointer = root/'project.json'; write(pointer, {'kind': 'project', 'state': {'last_source_run': 'fixture'}})
            retained = (parent.read_bytes(), pointer.read_bytes(), copy.deepcopy(run))
            original_assets = {}
            for runtime in (hair, cloth):
                layer = next(row for row in plan['layers'] if row['kind'] == runtime.KIND)
                frozen, _, _ = runtime._load(layer)
                for a in [*layer['assets'].values(), frozen['delta']]: original_assets[a['path']] = sha(a['path'])
            cache = root/'observations'; cache.mkdir()
            matte = {'cache_hit': False, 'tiles': {'alpha_tiles': 3}}
            for key, values in [('alpha_path', np.ones((32, 48), np.float32)),
                    ('premultiplied_foreground_path', np.full((32, 48, 3), .12, np.float32)),
                    ('unknown_edge_support_path', np.ones((32, 48), np.float32)),
                    ('fb_diagnostics_path', np.ones((32, 48), np.float32))]:
                path = cache/(key+'.npz'); np.savez_compressed(path, values=values); matte[key] = str(path)
            matte['result_path'] = str(cache/'result.json'); write(Path(matte['result_path']), matte)
            yy, xx = np.indices((32, 48), dtype=np.float32)
            observed = (labels, np.full((16, 24), 23, np.uint8), np.zeros((16, 24, 3), np.float32),
                {'ref_x': (xx+.5)*.5-.5, 'ref_y': (yy+.5)*.5-.5, 'valid': np.ones((32, 48), np.float32)},
                {'reference_labels': plan['mask_roles']['source_semantics']})
            calls = []; matte_calls = []
            def cached_observation(*args, **kw):
                matte_calls.append((args, kw)); return {**matte, 'cache_hit': len(matte_calls) > 1}
            def hair_producer(p, a, ref, ra, sl, rl, maps, **kw):
                calls.append('hair'); d = np.zeros_like(p)
                allowed = (sl == 4)&(a > 0)&(kw['identity_protection'] < .5)
                d[allowed] = np.float32(.006)*a[allowed, None]
                return {'delta_p': d, 'receipt': {'state': 'prepared', 'synthetic_producer': True}}
            def cloth_producer(p, a, ref, ra, sl, rl, maps, **kw):
                calls.append('cloth'); d = np.zeros_like(p)
                allowed = (sl == 23)&(a >= .95)&(kw['identity_protection'] < .5)
                d[allowed] = np.float32(.004)*a[allowed, None]
                return {'delta_p': d, 'receipt': {'state': 'prepared', 'synthetic_producer': True}}
            def prepare(current, name, *, hair_callback=hair_producer, cloth_callback=cloth_producer, cancel=None, progress=None):
                calls.clear(); matte_calls.clear()
                with patch.object(hair_worker, 'observations', return_value=observed), \
                     patch.object(cloth_worker, 'observations', return_value=observed), \
                     patch('source_foreground_cache.prepare_source_foreground', side_effect=cached_observation), \
                     patch.object(hair_worker, 'prepare_boundary_light', side_effect=hair_callback), \
                     patch.object(cloth_worker, 'prepare_cloth_boundary_light', side_effect=cloth_callback):
                    return coordinator.prepare(root, current, root/name, cancel=cancel, progress=progress)
            events = []
            result = prepare(run, 'prepared', progress=lambda *event: events.append(event))
            prepared = result['design_reconstruction_plan']; candidate = copy.deepcopy(run)
            candidate['options']['appearance_revision']['design_reconstruction_plan'] = prepared
            require(result['changed'] and not result['no_change'] and calls == ['hair', 'cloth'])
            require([e[0] for e in events] == list(coordinator.STAGES) and events[0][2] < events[1][2])
            require(result['receipt']['atomic_commit'] and result['receipt']['reference_observation_cache_reused'])
            require(result['receipt']['model_tile_inferences_this_attempt'] == 3 and len(matte_calls) == 2)
            require(matte_calls[0] == matte_calls[1])
            require(coordinator.validate_prepared_pair(prepared, candidate))
            for kind in coordinator.KINDS:
                old = next(row for row in plan['layers'] if row['kind'] == kind)
                new = next(row for row in prepared['layers'] if row['kind'] == kind)
                require(all(old[k] == new[k] for k in ('id', 'enabled', 'strength')))
                require(sum(row['kind'] == kind for row in prepared['layers']) == 1)
            cloth_layer = next(row for row in prepared['layers'] if row['kind'] == cloth.KIND)
            require(cloth_layer['params']['binding']['preceding_hair_digest'] == cloth.endpoint_binding(prepared, candidate)['preceding_hair_digest'])
            require((parent.read_bytes(), pointer.read_bytes(), run) == retained)
            require(all(sha(path) == digest for path, digest in original_assets.items()))
            # Prepare the same actual arrays again: exact old plan, no transport churn.
            identical = prepare(candidate, 'identical')
            require(not identical['changed'] and identical['design_reconstruction_plan'] == prepared)
            require(all(not s['changed'] for s in identical['receipt']['stages']))
            # An over-eager stage that reports changed after ZIP repackaging
            # still cannot alter the hair digest or create a child.
            def repack(project_root, current, output):
                kind = hair.KIND if Path(output).name == coordinator.STAGES[0] else cloth.KIND
                after = copy.deepcopy(design.plan_for_run(current)); layer = next(r for r in after['layers'] if r['kind'] == kind)
                frozen = json.loads(Path(layer['assets']['frozen_delta']['path']).read_text('utf8'))
                Path(output).mkdir()
                with np.load(frozen['delta']['path'], allow_pickle=False) as archive: arrays = {k: archive[k].copy() for k in archive.files}
                delta = Path(output)/'repacked.npz'; np.savez(delta, **arrays); frozen['delta'] = record(delta)
                layer['assets']['frozen_delta'] = write(Path(output)/'frozen.json', frozen)
                return {'changed': True, 'design_reconstruction_plan': after,
                        'receipt': {'state': 'prepared', 'reference_matte_cache_hit': True}}
            with patch.object(hair_worker, 'prepare', side_effect=repack), patch.object(cloth_worker, 'prepare', side_effect=repack):
                transport_only = coordinator.prepare(root, candidate, root/'transport-only')
            require(not transport_only['changed'] and transport_only['design_reconstruction_plan'] == prepared)
            for enabled, dose in ((False, .35), (True, 0.), (True, .4)):
                edited = copy.deepcopy(candidate); edited_plan = design.plan_for_run(edited)
                for layer in edited_plan['layers']:
                    if layer['kind'] in coordinator.KINDS: layer.update(enabled=enabled, strength=dose)
                preserved = copy.deepcopy(edited)
                updated = prepare(edited, 'switch-'+str(enabled)+'-'+str(dose))
                updated_run = copy.deepcopy(edited); updated_run['options']['appearance_revision']['design_reconstruction_plan'] = updated['design_reconstruction_plan']
                require(coordinator.validate_prepared_pair(design.plan_for_run(updated_run), updated_run))
                require(edited == preserved)
                for layer in design.plan_for_run(updated_run)['layers']:
                    if layer['kind'] in coordinator.KINDS: require(layer['enabled'] is enabled and layer['strength'] == dose)
            def failed_cloth(*a, **kw): raise ValueError('synthetic cloth stage failed')
            rejects(lambda: prepare(run, 'cloth-failure', cloth_callback=failed_cloth))
            def zero_cloth(p, *a, **kw): return {'delta_p': np.zeros_like(p), 'receipt': {'state': 'no_change'}}
            rejects(lambda: prepare(run, 'stale-retention', cloth_callback=zero_cloth))
            flag = {'cancel': False}
            def stage_cancel(stage, *_):
                if stage == coordinator.STAGES[1]: flag['cancel'] = True
            rejects(lambda: prepare(run, 'between-cancel', cancel=lambda: flag['cancel'], progress=stage_cancel), InterruptedError)
            require(calls == ['hair'] and len(matte_calls) == 1)
            require((parent.read_bytes(), pointer.read_bytes(), run) == retained)
            require(all(sha(path) == digest for path, digest in original_assets.items()))
            duplicate = copy.deepcopy(run); duplicate_plan = design.plan_for_run(duplicate)
            duplicate_plan['layers'].append(copy.deepcopy(duplicate_plan['layers'][-1]))
            with patch.object(hair_worker, 'prepare', side_effect=AssertionError('duplicate entered producer')):
                rejects(lambda: coordinator.prepare(root, duplicate, root/'duplicate'))
            # Save only the complete pair, and preserve current edited options.
            snapshot = root/'input.json'; write(snapshot, run)
            argv = ['coordinator', '--root', str(root), '--run-json', str(snapshot), '--output', str(root/'prepared'), '--save-child']
            with patch.object(coordinator, 'prepare', return_value=result), patch.object(sys, 'argv', argv): coordinator.main()
            saved = json.loads((root/'prepared/result.json').read_text('utf8'))
            child = json.loads(Path(saved['child_manifest']).read_text('utf8')); child_plan = design.plan_for_run(child)
            require(child['parent_run_id'] == run['run_id'] and child['revision'] == 2)
            require(child['options']['appearance_revision']['params'] == run['options']['appearance_revision']['params'])
            require(child['options']['preserve_project_selection'] and coordinator.validate_prepared_pair(child_plan, child))
            require(closure(child_plan) > 20)
            require(all(Path(next(row for row in child_plan['layers'] if row['kind'] == kind)['assets']['frozen_delta']['path']).is_relative_to(
                Path(saved['child_manifest']).parent) for kind in coordinator.KINDS))
            count = len(list((root/'outputs/runs').iterdir()))
            argv[argv.index('--output')+1] = str(root/'identical')
            with patch.object(coordinator, 'prepare', return_value=identical), patch.object(sys, 'argv', argv): coordinator.main()
            require(len(list((root/'outputs/runs').iterdir())) == count)
            require('child_run_id' not in json.loads((root/'identical/result.json').read_text('utf8')))
            require((parent.read_bytes(), pointer.read_bytes(), run) == retained)
            # Real wrapper: one process, both stages, and no --save-child flag.
            events.clear(); wrapped = root/'wrapped'
            class Process:
                pid = 123456789; returncode = None
                def poll(self): return self.returncode
                def terminate(self): self.returncode = -15
                def wait(self, timeout=None): return self.returncode
                def kill(self): self.returncode = -9
            process = Process()
            def completed(command, **kw):
                require(Path(command[4]).name == 'boundary_light_coordination_worker.py' and '--save-child' not in command)
                wrapped.mkdir(); write(wrapped/'result.json', identical)
                for index, stage in enumerate(coordinator.STAGES):
                    kw['stdout'].write((json.dumps({'type': 'boundary_coordination_progress', 'stage': stage,
                        'message': stage, 'progress': .1+.45*index})+'\n').encode('utf8'))
                kw['stdout'].flush(); process.returncode = 0; return process
            with patch('current_model_registry.load_registry', return_value={'python': sys.executable}), \
                 patch.object(coordinator.subprocess, 'Popen', side_effect=completed):
                require(coordinator.prepare_for_workbench(root, run, wrapped, progress=lambda *e: events.append(e))['no_change'])
            require([e[0] for e in events][-2:] == list(coordinator.STAGES))
            require(json.loads(wrapped.with_name('wrapped-input.json').read_text('utf8')) == run)
            process = Process(); cancel_calls = iter((False, True))
            def kill_tree(*a, **kw): process.returncode = -15
            with patch('current_model_registry.load_registry', return_value={'python': sys.executable}), \
                 patch.object(coordinator.subprocess, 'Popen', return_value=process), \
                 patch.object(coordinator.subprocess, 'run', side_effect=kill_tree):
                rejects(lambda: coordinator.prepare_for_workbench(root, run, root/'wrapper-cancel', cancel=lambda: next(cancel_calls)), InterruptedError)
            require(process.returncode == -15 and (parent.read_bytes(), pointer.read_bytes(), run) == retained)
    return {'ok': True, 'checks': checks, 'actual_two_workers_and_frozen_consumers': True,
        'hair_then_cloth_current_endpoint_bound': True, 'atomic_failure_cancel_and_stale_pair_rejection': True,
        'preserves_ids_switches_doses_and_other_edits': True,
        'identical_pixels_and_repacked_transport_no_change_without_child': True,
        'one_isolated_process_two_progress_stages_observation_cache_reused': True,
        'save_once_child_owned_frozen_assets_parent_pointer_unchanged': True,
        'synthetic_observations_and_producers': True, 'model_inferences': 0,
        'image_generation_submissions': 0, 'visual_acceptance': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path)
    args = parser.parse_args(); result = run_smoke()
    if args.output: write(args.output, result)
    print(json.dumps(result, ensure_ascii=False))
