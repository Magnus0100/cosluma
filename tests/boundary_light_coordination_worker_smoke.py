"""Atomic two-stage lifecycle on actual frozen consumers with synthetic observations."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
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
            # The actual consumers use the explicit candidate plan for all
            # endpoint layers and geometry; the run supplies source/control
            # bindings. An original run locator cannot accidentally select its
            # old hair layer when validating the newly prepared cloth endpoint.
            require(design.plan_for_run(run) != prepared)
            require(coordinator.validate_prepared_pair(prepared, run))
            import single_source_workflow as workflow
            def first_task_boundary(response=None, *, error=None):
                submitted = copy.deepcopy(run)
                submitted.update(run_id='first-task-boundary',
                    task_intent={'mode':'create_reference'},
                    task_plan={'execution_strategy':'layered_reference_transfer'})
                before = copy.deepcopy(design.plan_for_run(submitted))
                def forbidden_gpu(*args, **kwargs):
                    raise AssertionError('Frozen boundary integration reopened local generation')
                with (patch.object(workflow, '_refresh_design_plan') as refresh,
                      patch.object(coordinator, 'prepare_for_workbench',
                          side_effect=error, return_value=copy.deepcopy(response)) as prepare_once,
                      patch.object(design, 'materialize_plan', side_effect=lambda root, run, plan: copy.deepcopy(plan)) as own,
                      patch('start_qwen_image.release_for_run', side_effect=forbidden_gpu)):
                    workflow._prepare_task_boundary_light(root, submitted, SimpleNamespace(cancel_request_path=None))
                    cancel = refresh.call_args.kwargs['cancel']
                    assert callable(cancel) and not cancel()
                require(refresh.call_count == prepare_once.call_count == 1)
                require((parent.read_bytes(), pointer.read_bytes(), run) == retained)
                return submitted, before, own.call_count
            complete, before, owned = first_task_boundary(result)
            require(complete['boundary_light_preparation']['status'] == 'prepared' and owned == 1)
            require(design.plan_for_run(complete) == prepared)
            no_change = {'ok':True, 'no_change':True, 'receipt':{'schema':coordinator.CONTRACT},
                'design_reconstruction_plan':copy.deepcopy(plan)}
            unchanged, before, owned = first_task_boundary(no_change)
            require(unchanged['boundary_light_preparation']['status'] == 'no_change' and owned == 0)
            require(design.plan_for_run(unchanged) == before)
            bad_no_change = copy.deepcopy(no_change)
            bad_no_change['design_reconstruction_plan']['layers'][0]['strength'] = .17
            incomplete = copy.deepcopy(result)
            incomplete['design_reconstruction_plan']['layers'] = [row for row in prepared['layers']
                if row['kind'] != cloth.KIND]
            for response, error in ((bad_no_change,None), (incomplete,None),
                                    (None,ValueError('synthetic coordinated worker failure'))):
                rejected_run, before, owned = first_task_boundary(response, error=error)
                failure = rejected_run['boundary_light_preparation']
                require(failure['status'] == 'failed_submitted_plan_retained' and owned == 0)
                require(failure['error'] and failure['partial_plan_discarded'])
                require(design.plan_for_run(rejected_run) == before)
                require(not failure['source_alpha_changed'] and not failure['source_geometry_changed'])
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
            failure_receipt = json.loads((root/'cloth-failure'/'qualification.json').read_text('utf8'))
            require(failure_receipt['schema'] == coordinator.CONTRACT)
            require(failure_receipt['state'] == 'failed' and failure_receipt['failed_stage'] == coordinator.STAGES[1])
            require(failure_receipt['partial_plan_discarded'] and not failure_receipt['atomic_commit'])
            require(failure_receipt['stages'][-1]['status'] == 'failed')
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
            argv = ['coordinator', '--root', str(root), '--run-json', str(snapshot), '--output', str(root/'prepared')]
            with patch.object(coordinator, 'prepare', return_value=result), patch.object(sys, 'argv', argv): coordinator.main()
            saved = json.loads((root/'prepared/result.json').read_text('utf8'))
            prepared_plan = saved['design_reconstruction_plan']
            candidate = copy.deepcopy(run)
            candidate['options']['appearance_revision']['design_reconstruction_plan'] = prepared_plan
            require('child_manifest' not in saved and coordinator.validate_prepared_pair(prepared_plan, candidate))
            require(closure(prepared_plan) > 20)
            require(all(Path(next(row for row in prepared_plan['layers'] if row['kind'] == kind)['assets']['frozen_delta']['path']).is_file()
                for kind in coordinator.KINDS))
            count = len(list((root/'outputs/runs').iterdir()))
            argv[argv.index('--output')+1] = str(root/'identical')
            with patch.object(coordinator, 'prepare', return_value=identical), patch.object(sys, 'argv', argv): coordinator.main()
            require(len(list((root/'outputs/runs').iterdir())) == count)
            require('child_run_id' not in json.loads((root/'identical/result.json').read_text('utf8')))
            require((parent.read_bytes(), pointer.read_bytes(), run) == retained)
            # The wrapper submits both stages once to the shared native worker.
            events.clear(); wrapped = root/'wrapped'
            def completed(python, project_root, command, directory, cancel, **kw):
                require(python == sys.executable and project_root == root and directory == root)
                require(Path(command[0]).name == 'boundary_light_coordination_worker.py' and '--save-child' not in command)
                require(kw['stage'] == 'boundary_coordination' and kw['log_path'] == root/'wrapped-worker.log')
                wrapped.mkdir(); write(wrapped/'result.json', identical)
                for index, stage in enumerate(coordinator.STAGES):
                    kw['progress'](stage, stage, .1+.45*index)
                return {'worker_pid': 123456789, 'wall_seconds': .1}
            with patch('current_model_registry.load_registry', return_value={'python': sys.executable}), \
                 patch('reference_stage_worker.execute_stage', side_effect=completed):
                require(coordinator.prepare_for_workbench(root, run, wrapped, progress=lambda *e: events.append(e))['no_change'])
            require([e[0] for e in events][-2:] == list(coordinator.STAGES))
            require(json.loads(wrapped.with_name('wrapped-input.json').read_text('utf8')) == run)
            with patch('current_model_registry.load_registry', return_value={'python': sys.executable}), \
                 patch('reference_stage_worker.execute_stage', side_effect=InterruptedError('owned worker stopped')):
                rejects(lambda: coordinator.prepare_for_workbench(root, run, root/'wrapper-cancel'), InterruptedError)
            require(json.loads((root/'wrapper-cancel/qualification.json').read_text('utf8'))['state'] == 'cancelled')
            require((parent.read_bytes(), pointer.read_bytes(), run) == retained)
            # An impossible relaxed registration avoids both native RGB stages
            # and retains the exact submitted plan and source assets.
            with patch.object(cloth_worker, 'registration_upper_bound', return_value={
                    'rejected': True, 'reason': 'insufficient_opaque_same_owner_registration_anchors',
                    'reference_matte_cache_hit': True, 'model_tile_inferences_this_attempt': 0}), \
                 patch.object(hair_worker, 'prepare', side_effect=AssertionError('unqualified pair computed RGB')), \
                 patch.object(cloth_worker, 'prepare', side_effect=AssertionError('unqualified pair computed RGB')):
                declined = coordinator.prepare(root, run, root/'registration-declined')
            require(declined['no_change'] and declined['design_reconstruction_plan'] == plan)
            require(declined['receipt']['atomic_commit'] and not declined['receipt']['stages'])
            require((parent.read_bytes(), pointer.read_bytes(), run) == retained)
    return {'ok': True, 'checks': checks, 'actual_two_workers_and_frozen_consumers': True,
        'hair_then_cloth_current_endpoint_bound': True, 'atomic_failure_cancel_and_stale_pair_rejection': True,
        'preserves_ids_switches_doses_and_other_edits': True,
        'identical_pixels_and_repacked_transport_no_change_without_child': True,
        'one_isolated_process_two_progress_stages_observation_cache_reused': True,
        'preparation_only_frozen_assets_parent_pointer_unchanged': True,
        'first_task_boundary_success_and_local_failure_preserve_TBS': True,
        'actual_pair_consumers_use_explicit_prepared_plan_with_original_run_bindings': True,
        'synthetic_observations_and_producers': True, 'model_inferences': 0,
        'image_generation_submissions': 0, 'visual_acceptance': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path)
    args = parser.parse_args(); result = run_smoke()
    if args.output: write(args.output, result)
    print(json.dumps(result, ensure_ascii=False))
