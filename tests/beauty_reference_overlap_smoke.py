"""Current imported Beauty ownership and retired creation rejection; CPU only."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from PIL import Image

import beautygrpo_prepare as beauty
from backend_server import Backend
import creative_task
from creative_task_smoke import fixture, intent
from job_store import process_stamp
import single_source_workflow as workflow
from task_model_preflight_smoke import mocked_task_capability


REAL_POPEN = subprocess.Popen
CPU_SERVICE = r'''
import json, pathlib, subprocess, sys, time
for line in sys.stdin:
    request = json.loads(line)
    events = pathlib.Path(request['event_log'])
    def emit(event):
        with events.open('a', encoding='utf8') as stream:
            stream.write(json.dumps({'request_id':request['request_id'], **event}, ensure_ascii=False)+'\n')
            stream.flush()
    root = pathlib.Path(request['root'])
    with (root/'fixture-requests.jsonl').open('a', encoding='utf8') as stream:
        stream.write(json.dumps({'kind':request.get('kind','prepare'),'request_id':request['request_id']})+'\n')
    if request.get('kind') == 'prewarm':
        emit({'type':'beautygrpo_result','ok':True,'kind':'prewarm','pipeline_loaded':True})
        continue
    output = pathlib.Path(request['output'])
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    mode = (pathlib.Path(request['root'])/'fixture-mode.txt').read_text()
    if mode == 'wait_child':
        child = subprocess.Popen([sys.executable, '-B', '-c', 'import time; time.sleep(60)'])
        (output/'child.pid').write_text(str(child.pid))
    emit({'type':'beautygrpo_service_started'})
    if mode in ('wait', 'wait_child'):
        emit({'type':'beautygrpo_progress','stage':'beauty_face_generation','message':'人脸 1/1：1/28。','progress':.4})
        release = output.with_name(output.name+'-release')
        while not release.exists():
            time.sleep(.005)
    if mode == 'fail':
        emit({'type':'beautygrpo_result','ok':False,'error':'synthetic source worker failure'})
    else:
        result = {'ok':True,'no_change':True,'result':None,'layer':None,
                  'generation_inferences_this_attempt':0,'cache_hit':False,
                  'seconds':time.perf_counter()-started,'CPU_fixture':True}
        if mode in ('frozen','cancel_frozen'):
            result = json.loads((root/'fixture-result.json').read_text('utf8'))
        if mode == 'cancel_frozen':
            (root/'cancel.signal').write_text('cancel')
        (output/'result.json').write_text(json.dumps(result), encoding='utf8')
        emit({'type':'beautygrpo_result','ok':True})
'''


def wait_until(predicate, timeout=5):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.005)
    raise AssertionError('CPU fixture did not reach its expected checkpoint')


@contextmanager
def cpu_service(root, mode='auto'):
    config_path = root/'beauty-installation.json'
    config_path.write_text('{}')
    (root/'fixture-mode.txt').write_text(mode)
    config = {'python':sys.executable, 'config_path':str(config_path),
              'config_sha256':'CPU-fixture', 'execution_offload':'nunchaku_cuda'}
    processes = []
    def launch(_command, **kwargs):
        if '--service' not in _command:
            return REAL_POPEN(_command, **kwargs)
        process = REAL_POPEN([sys.executable, '-B', '-u', '-c', CPU_SERVICE], **kwargs)
        processes.append(process)
        return process
    beauty._stop_beauty_service()
    with ExitStack() as stack:
        stack.enter_context(patch.object(beauty, 'installation_path', return_value=config_path))
        stack.enter_context(patch.object(beauty, 'load_installation', return_value=config))
        stack.enter_context(patch.object(beauty.subprocess, 'Popen', side_effect=launch))
        stack.enter_context(patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES':''}))
        try:
            yield processes
        finally:
            beauty._stop_beauty_service()
            for process in processes:
                if process.poll() is None:
                    beauty._stop_process(process)
            assert all(process.poll() is not None for process in processes)


class OverlapContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='beauty-overlap-contract-')
        self.root = Path(self.temporary.name)
        registry, sources, reference = fixture(self.root)
        self.source, self.reference = Path(sources[0]), Path(reference)
        self.common = {'source':str(self.source),'model_registry':str(registry),
            'provider':'openai-compatible','api_base':'https://example.invalid/v1',
            'api_key_env':'CPU_TEST_KEY','variant_count':1}
        self.stack = ExitStack()
        self.stack.enter_context(patch.dict(os.environ, {'CPU_TEST_KEY':'fixture-only',
            'COSPLAY_MODEL_REGISTRY':str(registry),
            'COSPLAY_PROVIDER_SETTINGS':str(self.root/'provider-settings.json'),
            'COSPLAY_MODEL_SETTINGS':str(self.root/'model-selection.json')}))
        self.stack.enter_context(patch('task_model_preflight.inspect_task_models', side_effect=mocked_task_capability))
        self.stack.enter_context(patch('runtime_preflight.preflight_for_run', return_value={'status':'ready'}))
        # These contracts exercise the isolated service on a true cache miss;
        # complete CPU cache ownership is covered by beauty_cache_handoff_smoke.
        self.stack.enter_context(patch.object(beauty, 'prepare_cached_for_workbench', return_value=None))
        self.stack.enter_context(patch('single_source_workflow._prepare_task_boundary_light'))
        self.stack.enter_context(patch('single_source_workflow.diagnose_cosplay_post'))
        self.stack.enter_context(patch('single_source_workflow.render_candidate', side_effect=self.render))
        self.stack.enter_context(patch('start_qwen_image.release_for_run', return_value={'state':'not_managed_qwen'}))
        self.stack.enter_context(patch('urllib.request.urlopen', side_effect=AssertionError('CPU contract attempted HTTP')))
        self.calls = []

    def tearDown(self):
        beauty._stop_beauty_service()
        self.stack.close()
        self.temporary.cleanup()

    def execution(self):
        body = {**self.common,'creative_task':intent('existing_reference'),'run_id':'owned-overlap',
                'defer_reference_selection':False}
        body['reference'] = str(self.reference)
        prepared = creative_task.plan_task(self.root, body)
        execution = creative_task.execution_request(self.root,
            {**prepared['request'], 'plan_id':prepared['plan']['plan_id']})
        execution['defer_reference_selection'] = False
        return execution

    def args(self, execution=None):
        execution = execution or self.execution()
        argv = Backend(self.root)._one_click_args(execution, str(self.source))
        args = workflow.parse_args(['--root',str(self.root),
            *[value for value in argv[1:] if value != '--no-generate']])
        if args.options_json:
            workflow.apply_workflow_settings(args, json.loads(args.options_json))
        return args

    def render(self, _root, run, _args):
        self.assertEqual(run['portrait_beauty_preparation']['status'], 'no_change')
        self.calls.append('render')
        return {'outputs':{'auto_draft':str(self.source)},'metadata':{'CPU_fixture':True},'review':{}}

    def release(self, pending):
        wait_until(lambda: pending.event_log.is_file())
        pending.output.with_name(pending.output.name+'-release').write_text('release')
        wait_until(lambda: (pending.output/'result.json').is_file()
            and 'beautygrpo_result' in pending.event_log.read_text('utf8'))

    def test_retired_unchecked_creation_cannot_start_beauty_or_generation(self):
        args = workflow.parse_args(['--root',str(self.root),'--source',str(self.source),'--generate'])
        with patch.object(beauty,'prepare_for_workbench') as prepare, \
                patch('photo_edit_finish.initial_candidates') as generate:
            self.assertRaisesRegex(ValueError,'checked scoped task',workflow.create_run,args)
        prepare.assert_not_called()
        generate.assert_not_called()

    def test_existing_reference_still_prepares_explicit_beauty_synchronously(self):
        with cpu_service(self.root) as processes, \
                patch('photo_edit_finish.initial_candidates',side_effect=AssertionError('Imported task generated')):
            run=workflow.create_run(self.args(self.execution()))
        self.assertEqual(run['status'],'needs_visual_review')
        self.assertEqual(run['portrait_beauty_preparation']['status'],'no_change')
        self.assertEqual(len(processes),1)

    def test_other_run_wait_can_cancel_without_taking_ownership(self):
        marker = self.root/'waiting-run-cancel.requested'
        errors = []
        entered = threading.Event()
        with cpu_service(self.root, 'wait') as processes:
            owner = beauty.begin_for_workbench({'run_id':'first-owner'}, self.root, self.root/'first')
            wait_until(lambda: owner.event_log.is_file())
            original_snapshot = owner.snapshot.read_bytes()
            def waiting():
                entered.set()
                try:
                    beauty.prepare_for_workbench({'run_id':'waiting-run'}, self.root, self.root/'second',
                        cancel=lambda: marker.is_file())
                except Exception as exc:
                    errors.append(exc)
            thread = threading.Thread(target=waiting)
            thread.start()
            try:
                self.assertTrue(entered.wait(2))
                self.assertTrue(thread.is_alive())
                self.assertIs(beauty._BEAUTY_SERVICE['pending_request'], owner)
                self.assertEqual(owner.snapshot.read_bytes(), original_snapshot)
                self.assertEqual(len(processes), 1)
                marker.write_text('cancel waiting run')
                thread.join(3)
                self.assertFalse(thread.is_alive())
                self.assertEqual(len(errors), 1)
                self.assertIsInstance(errors[0], InterruptedError)
                self.assertIsNone(owner.process.poll())
                self.assertFalse((self.root/'second-input.json').exists())
            finally:
                marker.write_text('cancel waiting run')
                thread.join(3)
                receipt = beauty.cleanup_for_workbench(owner, reason='CPU_owner_exit')
            self.assertTrue(receipt['process_exit_confirmed'])

    def test_synchronous_different_run_queues_then_reuses_same_service(self):
        entered, finished = threading.Event(), threading.Event()
        results, errors = [], []
        with cpu_service(self.root, 'wait') as processes:
            owner = beauty.begin_for_workbench({'run_id':'first-owner'}, self.root, self.root/'first')
            wait_until(lambda: owner.event_log.is_file())
            def waiting():
                entered.set()
                try:
                    results.append(beauty.prepare_for_workbench({'run_id':'second-owner'}, self.root, self.root/'second'))
                except Exception as exc:
                    errors.append(exc)
                finally:
                    finished.set()
            thread = threading.Thread(target=waiting)
            thread.start()
            self.assertTrue(entered.wait(2))
            self.assertFalse(finished.wait(.05))
            self.assertIs(beauty._BEAUTY_SERVICE['pending_request'], owner)
            (self.root/'fixture-mode.txt').write_text('auto')
            self.release(owner)
            beauty.join_for_workbench(owner)
            thread.join(5)
            self.assertFalse(thread.is_alive())
            self.assertFalse(errors, errors)
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0]['runtime']['service_reused'])
            self.assertEqual(len(processes), 1)
            self.assertEqual(workflow.read_json(self.root/'second-input.json')['run']['run_id'], 'second-owner')

    def test_external_cli_cancel_stops_descendant_process_tree(self):
        shim = self.root/'CPU-cli-shim.py'
        shim.write_text("""import pathlib, subprocess, sys, time
root = pathlib.Path(sys.argv[sys.argv.index('--root')+1])
child = subprocess.Popen([sys.executable, '-B', '-c', 'import time; time.sleep(60)'])
temporary = root/'external-child.pid.tmp'
temporary.write_text(str(child.pid))
temporary.replace(root/'external-child.pid')
time.sleep(60)
""", encoding='utf8')
        backend = Backend(self.root, embedded=False)
        backend.cli = shim
        # Exercise the actual external worker/monitor with a CPU-only CLI shim.
        # Public plan validation is covered by the embedded integration above.
        body = {'source':str(self.source),'local_only':True,'generate':False,'run_id':'external-tree'}
        job_id = backend.jobs.create({'kind':'one_click', **body})['job_id']
        worker = threading.Thread(target=backend._run_one_click_job, args=(job_id, body))
        backend._job_workers[job_id] = worker
        worker.start()
        try:
            wait_until(lambda: (self.root/'external-child.pid').is_file())
            child_pid = int((self.root/'external-child.pid').read_text())
        finally:
            backend.cancel_job(job_id)
        wait_until(lambda: backend.jobs.get(job_id)['status']=='cancelled')
        worker = backend._job_workers.get(job_id)
        if worker:
            worker.join(5)
            self.assertFalse(worker.is_alive())
        self.assertIsNone(process_stamp(child_pid))

    def test_synchronous_cancel_after_terminal_stops_only_its_completed_service(self):
        cancelled = threading.Event()
        finish = beauty._finish_request
        def finish_then_cancel(pending):
            finish(pending)
            cancelled.set()
        with cpu_service(self.root) as processes, patch.object(beauty, '_finish_request', side_effect=finish_then_cancel):
            self.assertRaises(InterruptedError, beauty.prepare_for_workbench,
                {'run_id':'terminal-cancel-owner'}, self.root, self.root/'sync-cancel', cancel=cancelled.is_set)
            self.assertTrue(all(process.poll() is not None for process in processes))
            self.assertIsNone(beauty._BEAUTY_SERVICE)
            self.assertTrue((self.root/'sync-cancel/result.json').is_file())


if __name__ == '__main__':
    unittest.main()
