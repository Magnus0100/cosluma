"""Real SourceFG handoff cancellation reaches the workflow without a plan/save."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

import creative_task
from creative_task_smoke import fixture, intent
import design_reconstruction_pipeline as scene
import reference_pipeline as reference
import reference_stage_worker as stages
import single_source_workflow as workflow
import source_foreground_cache as foreground
from job_store import process_stamp
from task_model_preflight_smoke import mocked_task_capability
import workflow_timing


def exercise(root, mode, caller, native):
    root.mkdir()
    registry, sources, candidate = fixture(root)
    source, candidate = Path(sources[0]), Path(candidate)
    original = {path: workflow.sha256(path) for path in (source, candidate)}
    marker, checkpoint = root/'cancel.requested', root/'worker-started.json'
    model = root/'foreground-model'; model.mkdir()
    for name in ('model.safetensors', 'config.json', 'preprocessor_config.json'):
        (model/name).write_bytes(b'identity-only fixture; never loaded')
    proxy = root/'semantic-proxy'; proxy.mkdir()
    Image.open(source).save(proxy/'source-srgb.png')
    labels = proxy/'labels.npz'
    np.savez(labels, labels=np.full((20, 25), 23, np.uint8))
    observation = {'cache_hit': True, 'analysis_directory': str(proxy),
        'semantic_result': {'labels_path': str(labels)}}
    payload = root/'foreground-cpu-payload.py'
    payload.write_text('import os,json,time\nfrom pathlib import Path\n'
        f'Path({str(checkpoint)!r}).write_text(json.dumps({{"pid":os.getpid(),"parent_pid":os.getppid(),"model_calls":0}}))\n'
        'while True: time.sleep(.01)\n', encoding='utf8')
    runtime = {'python': Path(sys.executable), 'vitmatte_model_root': model,
        'matting_profile': 'vitmatte_small_distinctions', 'registry_path': registry}
    workers, submissions, identities = [], [], []
    real_execute, session_type = stages.execute_stage, stages.NativeStageSession
    stop = threading.Event()
    def forbidden(*args, **kwargs):
        raise AssertionError('Cancelled source preparation reached a model, renderer or boundary producer')
    def silhouette(_root, path, image, **kwargs):
        if mode == 'before' and Path(path) == candidate:
            marker.write_text('cancel before SourceFG', encoding='utf8')
        return Image.new('L', image.size, 128), {'cache_hit': True}
    def capture_session(*args, **kwargs):
        worker = session_type(*args, **kwargs)
        workers.append(worker)
        return worker
    def completed_cpu_stage(python, selected_root, command, directory, cancel, **kwargs):
        assert selected_root == root and Path(command[0]).name == 'source_foreground.py'
        assert kwargs['stage'] == 'source_foreground' and not cancel()
        submissions.append(kwargs['stage'])
        checkpoint.write_text(json.dumps({'pid': None, 'model_calls': 0}), encoding='utf8')
        marker.write_text('cancel on completed CPU stage', encoding='utf8')
        # The real SourceFG wrapper must check cancellation before reading or
        # publishing the absent result, even when the executor just returned.
        return {}
    def execute_cpu_payload(python, selected_root, command, directory, cancel, **kwargs):
        assert selected_root == root and Path(command[0]).name == 'source_foreground.py'
        assert kwargs['stage'] == 'source_foreground' and not cancel()
        submissions.append(kwargs['stage'])
        # Preserve the real SourceFG stage, cancellation callback, selected
        # interpreter and session ownership. Only the numerical producer is a
        # blocking CPU payload so this control test never performs inference.
        return real_execute(python, selected_root, ['-c', payload.read_text('utf8')],
            directory, cancel, **kwargs)
    def cancel_owned_worker():
        deadline = time.monotonic()+10
        while not stop.is_set() and not checkpoint.is_file() and time.monotonic() < deadline:
            time.sleep(.005)
        if checkpoint.is_file():
            observed = json.loads(checkpoint.read_text('utf8'))
            assert len(workers) == 1
            launcher = workers[0].process.pid
            assert observed['parent_pid'] == launcher or observed['pid'] == launcher
            stamp = process_stamp(observed['pid'])
            assert stamp is not None
            identities.append({'worker_pid': observed['pid'], 'launcher_pid': launcher,
                'parent_pid': observed['parent_pid'], 'worker_process_stamp': stamp})
            marker.write_text('cancel owned in-flight SourceFG', encoding='utf8')
    def beauty_completed(_root, run, args, **kwargs):
        if mode == 'preflight':
            marker.write_text('cancel before native refresh', encoding='utf8')
        run['portrait_beauty_preparation'] = {'status': 'no_change', 'generation_inferences_this_attempt': 0}
    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {
            'COSPLAY_MODEL_REGISTRY': str(registry), 'COSPLAY_CACHE_ROOT': str(root/'outputs/cache'),
            'COSPLAY_PROVIDER_SETTINGS': str(root/'provider-settings.json'),
            'COSPLAY_MODEL_SETTINGS': str(root/'model-settings.json')}))
        stack.enter_context(patch('task_model_preflight.inspect_task_models', side_effect=mocked_task_capability))
        stack.enter_context(patch('runtime_preflight.preflight_for_run', return_value={'status': 'ready'}))
        stack.enter_context(patch('runtime_memory.native_analysis_cpu_bytes', return_value=0))
        idle = stack.enter_context(patch('beautygrpo_prepare.release_idle_for_stage', return_value={}))
        stack.enter_context(patch('start_qwen_image.release_for_run', return_value={'state': 'not_managed_qwen'}))
        stack.enter_context(patch.object(scene, 'ready', return_value=True))
        stack.enter_context(patch.object(scene, 'local_runtime_paths', return_value=runtime))
        stack.enter_context(patch.object(scene, '_model_registry_receipt', return_value={'sha256': 'fixture'}))
        stack.enter_context(patch.object(scene, 'identity_for_run', side_effect=lambda _root, run:
            {'source': run['source']['sha256'], 'candidate': run['reference']['sha256']}))
        semantics = stack.enter_context(patch.object(scene, 'prepare_local_appearance_pair',
            return_value=(observation, observation)))
        stack.enter_context(patch('local_scene_analysis.prepare_local_scene', return_value={'cache_hit': True}))
        stack.enter_context(patch.object(scene, 'source_silhouette', side_effect=silhouette))
        stack.enter_context(patch.object(scene, '_prepare_background_stage', side_effect=forbidden))
        stack.enter_context(patch.object(reference, '_registry_identity', return_value={'sha256': 'fixture',
            'semantics_profile': 'sapiens2_seg_0.4b', 'matting_profile': 'vitmatte_small_distinctions'}))
        stack.enter_context(patch.object(foreground, 'local_runtime_paths', return_value=runtime))
        stack.enter_context(patch.object(workflow, '_prepare_task_portrait_beauty', side_effect=beauty_completed))
        stack.enter_context(patch('reference_provider.request_edit', side_effect=forbidden))
        stack.enter_context(patch.object(workflow, 'render_candidate', side_effect=forbidden))
        stack.enter_context(patch('boundary_light_coordination_worker.prepare_for_workbench', side_effect=forbidden))
        stack.enter_context(patch('urllib.request.urlopen', side_effect=forbidden))
        if native:
            stack.enter_context(patch.object(stages, 'NativeStageSession', side_effect=capture_session))
            stack.enter_context(patch.object(stages, 'execute_stage', side_effect=execute_cpu_payload))
        else:
            stack.enter_context(patch.object(stages, 'execute_stage', side_effect=completed_cpu_stage))
        planned = creative_task.plan_task(root, {'source': str(source), 'reference': str(candidate),
            'model_registry': str(registry), 'provider': 'openai-compatible',
            'run_id': 'cancel-create', 'sample': 's',
            'creative_task': intent('existing_reference')})
        assert planned['plan']['ready'], planned['plan']['blocking_reasons']
        watch = None
        if native and mode == 'in_flight':
            watch = threading.Thread(target=cancel_owned_worker, daemon=True)
            watch.start()
        try:
            if caller == 'boundary':
                run = {'run_id': 'cancel-direct', 'source': {'path': str(source), 'sha256': original[source]},
                    'reference': {'path': str(candidate), 'sha256': original[candidate]},
                    'task_intent': planned['plan']['intent'], 'task_plan': planned['plan'],
                    'options': {'render_mode': 'native_source_design', 'model_registry': str(registry),
                                'appearance_revision': {}}}
                before = copy.deepcopy(run['options']['appearance_revision'])
                if mode == 'preflight': marker.write_text('cancel before native refresh', encoding='utf8')
                try:
                    with workflow_timing.request(root, 'source_foreground_cancel') as ledger:
                        workflow._prepare_task_boundary_light(root, run,
                            SimpleNamespace(cancel_request_path=str(marker)))
                except workflow.WorkflowCancelled as error:
                    if mode != 'preflight': assert isinstance(error.__cause__, InterruptedError)
                else:
                    raise AssertionError('SourceFG cancellation was accepted as a prepared plan')
                assert ledger.root_span.status == 'cancelled'
                assert run['options']['appearance_revision'] == before
            else:
                args = workflow.parse_args(['--root', str(root), '--source', str(source),
                    '--reference', str(candidate), '--model-registry', str(registry), '--run-id', 'cancel-create'])
                args.task_intent, args.task_plan = planned['plan']['intent'], planned['plan']
                args.cancel_request_path = str(marker)
                run = workflow.create_run(args)
                assert run['status'] == 'cancelled' and run['next_action'] == 'retry_when_ready'
                saved = workflow.read_json(root/run['manifest'])
                assert saved['status'] == 'cancelled' and 'error' not in saved
                assert workflow.read_json(root/'project.json')['state']['status'] == 'cancelled'
                assert run['generation_budget']['image_submissions'] == 0
                assert not run.get('auto_draft') and not run.get('artifacts')
                assert not (run['options'].get('appearance_revision') or {}).get('design_reconstruction_plan')
                report = workflow.read_json(Path(run['workflow_timing']['path']))
                assert report['status'] == 'cancelled'
            assert 'boundary_light_preparation' not in run
            assert not list((root/'outputs/cache/source-foreground').rglob('result.json'))
            assert not list((root/'outputs/cache/source-foreground').rglob('*.npy'))
            assert not list((root/'outputs/cache/source-foreground').rglob('*.npz'))
            assert all(workflow.sha256(path) == digest for path, digest in original.items())
            assert all(worker.process.poll() is not None for worker in workers)
            if mode == 'preflight': assert idle.call_count == semantics.call_count == 0
            if mode != 'in_flight': assert not workers and not submissions and not checkpoint.exists()
            elif native:
                observed = json.loads(checkpoint.read_text('utf8'))
                assert len(workers) == len(identities) == 1 and observed['model_calls'] == 0
                deadline = time.monotonic()+5
                while process_stamp(observed['pid']) is not None and time.monotonic() < deadline:
                    time.sleep(.005)
                assert process_stamp(observed['pid']) is None, identities
                assert process_stamp(workers[0].process.pid) is None, identities
            else: assert submissions == ['source_foreground']
        finally:
            stop.set()
            if watch:
                watch.join(2)
                assert not watch.is_alive()
    return {'caller': caller, 'mode': mode, 'no_foreground_plan_or_render_published': True,
        'cancelled': True, 'owned_process_identities': identities, 'no_orphan': True}


def forced_close(root, native):
    root.mkdir()
    checkpoint = root/'started.json'
    code = ('import os,json,time\nfrom pathlib import Path\n'
        f'Path({str(checkpoint)!r}).write_text(json.dumps({{"pid":os.getpid(),"parent_pid":os.getppid()}}))\n'
        'while True: time.sleep(.01)\n')
    worker = stages.NativeStageSession(sys.executable, root, root)
    errors = []
    def run_blocking():
        try:
            worker.execute(['-c', code], root, None,
                stage='source_foreground' if native else 'fixture', source_cache=None, profile_path=None)
        except RuntimeError as error: errors.append(str(error))
    execution = threading.Thread(target=run_blocking, daemon=True)
    execution.start()
    try:
        deadline = time.monotonic()+10
        while not checkpoint.is_file() and time.monotonic() < deadline:
            time.sleep(.005)
        assert checkpoint.is_file(), errors
        observed = json.loads(checkpoint.read_text('utf8'))
        assert observed['parent_pid'] == worker.process.pid or observed['pid'] == worker.process.pid
        stamp = process_stamp(observed['pid'])
        assert stamp is not None
        # EOF cannot finish an in-flight payload; the bounded close must use
        # the existing tree-stop helper after its normal graceful wait.
        started = time.perf_counter()
        worker.close()
        elapsed = time.perf_counter()-started
        execution.join(2)
        assert not execution.is_alive() and len(errors) == 1
        assert worker.process.poll() is not None and process_stamp(worker.process.pid) is None
        deadline = time.monotonic()+5
        while process_stamp(observed['pid']) is not None and time.monotonic() < deadline:
            time.sleep(.005)
        assert process_stamp(observed['pid']) is None
        assert not list(root.glob('.stage-*.json'))
        return {'caller': 'forced_close', 'worker_pid': observed['pid'],
            'launcher_pid': worker.process.pid, 'parent_pid': observed['parent_pid'],
            'worker_process_stamp': stamp, 'close_seconds': elapsed, 'no_orphan': True}
    finally:
        worker.close()
        execution.join(2)


def cleanup_exception_scope():
    cleanup_error = OSError('controlled owned worker cleanup failure')
    def fail_close(): raise cleanup_error
    @stages.stage_session
    def call(primary):
        stages._SESSION.get()['worker'] = SimpleNamespace(close=fail_close)
        if primary is not None: raise primary
        return 'complete'
    primary = InterruptedError('controlled original stage cancellation')
    for original in (primary, None):
        try: call(original)
        except BaseException as error:
            assert error is (original or cleanup_error)
            if original is not None:
                assert any(str(cleanup_error) in note for note in error.__notes__)
        else: raise AssertionError('Controlled cleanup failure was hidden')
        assert stages._SESSION.get() is None
    return {'caller': 'cleanup_failure', 'primary_error_preserved': True,
        'successful_call_cleanup_failure_explicit': True, 'session_reset_always': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', action='store_true', help='Use the selected native D runtime and real owned worker')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    results = []
    with tempfile.TemporaryDirectory(prefix='sfc-') as temporary:
        root = Path(temporary)
        for caller in ('boundary', 'create'):
            for mode in ('preflight', 'before', 'in_flight'):
                results.append(exercise(root/(caller+'-'+mode), mode, caller, args.native))
        results.append(forced_close(root/'forced-close', args.native))
        results.append(cleanup_exception_scope())
    assert 'torch' not in sys.modules and 'cupy' not in sys.modules
    record = {'ok': True, 'cases': 8, 'results': results,
        'runtime_python': sys.executable,
        'production_code_sha256': {name: workflow.sha256(Path(__file__).parents[1]/'scripts'/name)
            for name in ('single_source_workflow.py', 'reference_pipeline.py',
                         'design_reconstruction_pipeline.py', 'source_foreground_cache.py',
                         'reference_stage_worker.py', 'source_foreground.py')},
        'fixture_code_sha256': workflow.sha256(Path(__file__)),
        'real_SourceFG_cache_scene_reference_refresh_call_chain': True,
        'caller_sees_WorkflowCancelled_and_saved_workflow_status_cancelled': True,
        'cancel_before_SourceFG_starts_no_worker': True,
        'no_plan_foreground_or_render_output_published': True,
        'actual_owned_native_worker_terminated': args.native,
        'forced_close_and_exact_owned_launcher_payload_no_orphan': True,
        'cleanup_failure_preserves_primary_and_resets_session': True,
        'RAW_GPU_model_HTTP_calls': 0}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2)+'\n', encoding='utf8')
    print(json.dumps(record))


if __name__ == '__main__': main()
