"""CPU subprocess contracts for the production HTTP/Beauty overlap; no models or HTTP."""
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
import workflow_timing


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

    def execution(self, mode='create_reference'):
        body = {**self.common,'creative_task':intent(mode),'run_id':'owned-overlap',
                'defer_reference_selection':False}
        if mode == 'existing_reference':
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

    def local_execution(self, *, count=1, defer=False):
        body={**self.common,'provider':'qwen_image_2_1_fast','creative_task':intent('create_reference'),
            'variant_count':count,'defer_reference_selection':defer,'run_id':'local-source-first'}
        with patch('reference_provider.provider_health',return_value={'ready':True,'kind':'comfyui'}):
            prepared=creative_task.plan_task(self.root,body)
            return creative_task.execution_request(self.root,{**prepared['request'],'plan_id':prepared['plan']['plan_id']})

    def render(self, _root, run, _args):
        self.assertEqual(run['portrait_beauty_preparation']['status'], 'no_change')
        self.calls.append('render')
        return {'outputs':{'auto_draft':str(self.source)},'metadata':{'CPU_fixture':True},'review':{}}

    def candidate(self, root, run, args):
        self.calls.append('provider')
        # The submission checkpoint must be durable before entering transport.
        saved = workflow.read_json(root/'outputs/runs'/run['run_id']/'manifest.json')
        self.assertEqual(saved['generation_budget']['image_submissions'], 1)
        self.assertEqual(saved['generation_budget']['submissions'][0]['status'], 'submitted_outcome_unknown')
        args.candidate_out_dir.mkdir(parents=True, exist_ok=True)
        candidate = args.candidate_out_dir/(run['sample']+'.png')
        Image.new('RGB',(25,20),'orange').save(candidate)
        workflow.write_json(candidate.with_suffix('.provenance.json'), {
            'generation_id':'CPU-fixture','execution_strategy':'layered_reference_transfer'})
        return candidate

    def release(self, pending):
        wait_until(lambda: pending.event_log.is_file())
        pending.output.with_name(pending.output.name+'-release').write_text('release')
        wait_until(lambda: (pending.output/'result.json').is_file()
            and 'beautygrpo_result' in pending.event_log.read_text('utf8'))

    def test_remote_overlaps_existing_service_and_wall_reconciles(self):
        args = self.args()
        @workflow_timing.timed('generation', model_request=True, http_request=True)
        def provider(root, run, variant_args):
            pending = beauty._BEAUTY_SERVICE['pending_request']
            self.assertFalse(pending.closed)
            self.assertIsNone(pending.process.poll())
            snapshot = workflow.read_json(pending.snapshot)
            self.assertIsNone(snapshot['run']['reference'])
            self.assertEqual(snapshot['run']['source'], run['source'])
            with workflow_timing.span('provider_transport', model_submissions=1, http_submissions=1):
                self.release(pending)
                return self.candidate(root, run, variant_args)
        with cpu_service(self.root, 'wait') as processes, patch.object(workflow, 'generate_candidate', side_effect=provider):
            run = workflow.create_run(args)
            self.assertEqual(len(processes), 1)
            self.assertFalse(beauty._BEAUTY_SERVICE.get('busy_request_id'))
        self.assertEqual(run['status'], 'needs_visual_review')
        self.assertEqual(self.calls, ['provider','render'])
        self.assertEqual(run['generation_budget']['image_submissions'], 1)
        report = workflow.read_json(Path(run['workflow_timing']['path']))
        self.assertLess(abs(report['reconciliation']['difference_seconds']), 1e-8)
        worker = next(row for row in report['measured_workers'] if row['name']=='portrait_beauty_async_request')
        self.assertTrue(worker['metadata']['interval_overlap'])
        self.assertGreater(worker['execution_seconds'], worker['metadata']['join_wait_seconds'])
        join = next(row for row in report['spans'] if row['name']=='portrait_beauty')
        self.assertEqual(join['metadata']['duration_scope'], 'join_and_materialize_only')

    def test_provider_timeout_stops_owned_tree_without_refunding_paid_budget(self):
        args = self.args()
        child_pid = []
        def failed_provider(*_args):
            pending = beauty._BEAUTY_SERVICE['pending_request']
            wait_until(lambda: (pending.output/'child.pid').is_file() and pending.event_log.is_file())
            child_pid.append(int((pending.output/'child.pid').read_text()))
            raise TimeoutError('paid response outcome unknown')
        with cpu_service(self.root, 'wait_child') as processes, patch.object(workflow, 'generate_candidate', side_effect=failed_provider) as provider:
            run = workflow.create_run(args)
            self.assertTrue(all(process.poll() is not None for process in processes))
            self.assertIsNone(beauty._BEAUTY_SERVICE)
        self.assertIsNone(process_stamp(child_pid[0]))
        self.assertEqual(provider.call_count, 1)
        self.assertEqual(run['generation_budget']['image_submissions'], 1)
        self.assertEqual(run['generation_budget']['submissions'][0]['status'], 'failed_or_outcome_unknown')
        record = run['portrait_beauty_preparation']
        self.assertTrue(record['process_exit_confirmed'])
        self.assertIsNone(record['model_submissions'])
        self.assertEqual(record['observed_model_inferences_minimum'], 1)
        self.assertFalse(run['error']['automatic_image_retry'])

    def test_beauty_failure_retains_paid_candidate_and_stops_worker(self):
        with cpu_service(self.root, 'fail') as processes, patch.object(workflow, 'generate_candidate', side_effect=self.candidate):
            run = workflow.create_run(self.args())
            self.assertTrue(all(process.poll() is not None for process in processes))
        self.assertEqual(run['status'], 'blocked')
        self.assertEqual(run['next_action'], 'retry_local_processing')
        self.assertEqual(len(run['variants']), 1)
        self.assertTrue(Path(run['reference']['provenance']).is_file())
        self.assertEqual(run['generation_budget']['image_submissions'], 1)
        self.assertEqual(run['generation_budget']['submissions'][0]['status'], 'succeeded')
        self.assertTrue(run['error']['candidate_retained'])
        self.assertNotIn('render', self.calls)

    def test_completed_beauty_is_retained_after_http_failure_without_rerunning(self):
        def failed_provider(*_args):
            self.release(beauty._BEAUTY_SERVICE['pending_request'])
            raise TimeoutError('paid response lost after source completion')
        with cpu_service(self.root, 'wait') as processes, patch.object(workflow, 'generate_candidate', side_effect=failed_provider):
            run = workflow.create_run(self.args())
            self.assertTrue(all(process.poll() is not None for process in processes))
            self.assertEqual(run['portrait_beauty_preparation']['status'], 'no_change')
            workflow._prepare_task_portrait_beauty(self.root, run, self.args())
            self.assertEqual(len(processes), 1)
        self.assertEqual(run['generation_budget']['image_submissions'], 1)

    def test_embedded_cancel_during_blocked_provider_is_owner_scoped_and_stops_tree(self):
        execution = self.execution()
        entered, release_provider = threading.Event(), threading.Event()
        child_pid = []
        def provider(root, run, variant_args):
            pending = beauty._BEAUTY_SERVICE['pending_request']
            wait_until(lambda: (pending.output/'child.pid').is_file() and pending.event_log.is_file())
            child_pid.append(int((pending.output/'child.pid').read_text()))
            # A different job marker/run cannot cancel this busy request.
            wrong = self.root/'wrong-cancel.requested'; wrong.write_text('cancel')
            self.assertEqual(beauty.cancel_pending_for_run(self.root, 'other-run',
                cancel_request_path=str(wrong))['state'], 'no_owned_matching_request')
            self.assertIsNone(pending.process.poll())
            entered.set()
            self.assertTrue(release_provider.wait(5))
            return self.candidate(root, run, variant_args)
        backend = Backend(self.root, embedded=True)
        with cpu_service(self.root, 'wait_child') as processes, patch.object(workflow, 'generate_candidate', side_effect=provider):
            started = backend.one_click(execution)
            job_id = started['job_id']
            try:
                self.assertTrue(entered.wait(5))
                backend.cancel_job(job_id)
                wait_until(lambda: processes and all(process.poll() is not None for process in processes))
                self.assertIsNone(process_stamp(child_pid[0]))
            finally:
                release_provider.set()
            wait_until(lambda: backend.jobs.get(job_id)['status']=='cancelled')
            worker = backend._job_workers.get(job_id)
            if worker:
                worker.join(5)
                self.assertFalse(worker.is_alive())
            self.assertIsNone(beauty._BEAUTY_SERVICE)
        saved = workflow.read_json(self.root/'outputs/runs/owned-overlap/manifest.json')
        self.assertEqual(saved['status'], 'cancelled')
        self.assertEqual(saved['generation_budget']['image_submissions'], 1)
        self.assertEqual(saved['generation_budget']['submissions'][0]['status'], 'succeeded')
        self.assertEqual(len(saved['variants']), 1)
        self.assertNotIn('render', self.calls)
        self.assertTrue(saved['portrait_beauty_preparation']['process_exit_confirmed'])

    def test_qwen_and_existing_reference_use_synchronous_path_only(self):
        for mode, kind in [('create_reference','comfyui'),('existing_reference','openai-compatible')]:
            with self.subTest(mode=mode):
                execution = self.execution(mode)
                execution['run_id'] = mode+'-serial'
                args = self.args(execution)
                actual_config = {'kind':kind,'provider':'openai-compatible','model':'CPU-fixture'}
                def provider(root, run, variant_args):
                    self.assertIsNotNone(beauty._BEAUTY_SERVICE)
                    self.assertEqual(run['portrait_beauty_preparation']['status'],'no_change')
                    return self.candidate(root, run, variant_args)
                with cpu_service(self.root) as processes, patch('reference_provider.provider_config', return_value=actual_config), \
                        patch.object(workflow, '_begin_task_portrait_beauty', side_effect=AssertionError('serial route dispatched early')), \
                        patch.object(workflow, 'generate_candidate', side_effect=provider) as generate:
                    run = workflow.create_run(args)
                    self.assertEqual(len(processes), 1)
                    self.assertEqual(run['status'], 'needs_visual_review')
                    self.assertEqual(generate.call_count, int(mode=='create_reference'))
                    # Reopening the same source result never starts another request.
                    workflow._prepare_task_portrait_beauty(self.root, run, args)
                    self.assertEqual(len(processes), 1)

    def test_local_warm_source_beauty_frozen_before_first_qwen_without_second_prepare(self):
        from portrait_beauty_backend_smoke import project, prepared
        from production_coordinator import execute_workflow
        from portrait_beauty_layer import load_result
        parent,_=project(self.root/'source-project')
        self.source=Path(parent['source']['path']);self.common['source']=str(self.source)
        frozen=prepared(parent,self.root/'fixture-layer')
        workflow.write_json(self.root/'fixture-result.json',frozen)
        execution=self.local_execution()
        supplied=self.args(execution)
        config={'kind':'comfyui','provider':'qwen_image_2_1_fast','model':'CPU-fixture'}
        original_source=self.source.read_bytes();events=[]
        def provider_main(arguments):
            events.append('qwen_transfer')
            self.assertEqual(Path(arguments[arguments.index('--image')+1]),self.source)
            self.assertEqual(Path(arguments[arguments.index('--source')+1]),self.source)
            self.assertIsNone(beauty._BEAUTY_SERVICE, 'Completed Beauty was not parked before Qwen')
            directory=Path(arguments[arguments.index('--out-dir')+1]);directory.mkdir(parents=True,exist_ok=True)
            saved=workflow.read_json(self.root/'outputs/runs'/supplied.run_id/'manifest.json')
            control=saved['options']['appearance_revision']['portrait_beauty']
            layer,_=load_result(control['result'])
            self.assertEqual(layer['source_sha256'],parent['source']['sha256'])
            self.assertTrue(Path(control['result']).is_relative_to(self.root/'outputs/runs'/supplied.run_id))
            self.assertEqual(saved['generation_budget']['image_submissions'],1)
            Image.new('RGB',(32,24),'orange').save(directory/(saved['sample']+'.png'))
            return 0
        def render(_root,run,_args):
            events.append('native_render')
            self.assertEqual(run['portrait_beauty_preparation']['status'],'frozen_layer_reused')
            return {'outputs':{'auto_draft':str(self.source)},'metadata':{'CPU_fixture':True},'review':{}}
        with cpu_service(self.root,'frozen') as processes, patch('reference_provider.provider_config',return_value=config), \
                patch('reference_provider.main',side_effect=provider_main), \
                patch.object(workflow,'render_candidate',side_effect=render), \
                patch.object(workflow,'_begin_task_portrait_beauty',side_effect=AssertionError('Local path entered remote async dispatch')):
            warm=beauty.prewarm_for_workbench(self.root)
            self.assertEqual(warm['state'],'pipeline_warm')
            warm_pid=processes[0].pid
            result=execute_workflow(self.root,supplied)
            self.assertEqual(result['status'],'needs_visual_review',result.get('error'))
            self.assertEqual(result['result_publication']['state'],'committed')
            self.assertEqual(len(processes),1)
            self.assertEqual(result['options']['portrait_generation_memory_transition']['service_pid'],warm_pid)
            self.assertEqual(result['options']['portrait_generation_memory_transition']['state'],'idle_owned_service_stopped')
            requests=[json.loads(line) for line in (self.root/'fixture-requests.jsonl').read_text('utf8').splitlines()]
            self.assertEqual([row['kind'] for row in requests],['prewarm','prepare'])
            self.assertEqual(events,['qwen_transfer','native_render'])
            self.assertEqual(self.source.read_bytes(),original_source)

    def test_local_pre_reference_failure_or_cancel_cannot_spend_a_qwen_submission(self):
        for mode in ('fail','cancel_frozen'):
            with self.subTest(mode=mode):
                if mode=='cancel_frozen':
                    workflow.write_json(self.root/'fixture-result.json',{'ok':True,'no_change':True,'result':None,
                        'layer':None,'generation_inferences_this_attempt':0,'cache_hit':False})
                execution=self.local_execution();execution['run_id']='local-before-reference-'+mode
                supplied=self.args(execution)
                if mode=='cancel_frozen':supplied.cancel_request_path=str(self.root/'cancel.signal')
                config={'kind':'comfyui','provider':'qwen_image_2_1_fast','model':'CPU-fixture'}
                with cpu_service(self.root,mode),patch('reference_provider.provider_config',return_value=config), \
                        patch('reference_provider.main',side_effect=AssertionError('Qwen image transmitted before Beauty completed')):
                    result=workflow.create_run(supplied)
                self.assertEqual(result['generation_budget']['image_submissions'],0)
                self.assertEqual(result['generation_budget']['submissions'],[])
                self.assertIsNone(result.get('reference'))
                self.assertEqual(result.get('variants'),[])
                self.assertEqual(result['status'],'cancelled' if mode=='cancel_frozen' else 'blocked')
                if result.get('error'):self.assertFalse(result['error']['candidate_retained'])
                with self.assertRaisesRegex(ValueError,'check a new creative task'):
                    Backend(self.root).retry_source_run(result['run_id'],{})

    def test_local_single_candidate_plan_order_matches_source_first_dispatch(self):
        label='原片人像美化并冻结（本地生图前）'
        plan=self.local_execution()['task_plan']
        labels=[row['label'] for row in plan['stages']]
        self.assertLess(labels.index(label),labels.index('创作风格参考'))
        self.assertIn('原生人物基线（复用已冻结美化）',labels)
        for execution in (self.local_execution(count=2),self.local_execution(defer=True),
                          self.execution(),self.execution('existing_reference')):
            self.assertNotIn(label,[row['label'] for row in execution['task_plan']['stages']])
    def test_deferred_selection_and_failed_preflight_do_not_dispatch(self):
        execution = self.execution()
        execution['defer_reference_selection'] = True
        with patch.object(workflow, '_begin_task_portrait_beauty', side_effect=AssertionError('deferred request ran Beauty')), \
                patch.object(workflow, 'generate_candidate', side_effect=self.candidate):
            run = workflow.create_run(self.args(execution))
        self.assertEqual(run['status'], 'awaiting_reference')
        execution['run_id'] = 'preflight-rejected'
        execution['defer_reference_selection'] = False
        with patch('runtime_preflight.preflight_for_run', side_effect=RuntimeError('CPU fixture unavailable runtime')), \
                patch.object(workflow, '_begin_task_portrait_beauty') as begin, \
                patch.object(workflow, 'generate_candidate') as generate:
            self.assertRaisesRegex(RuntimeError, 'unavailable runtime', workflow.create_run, self.args(execution))
        begin.assert_not_called()
        generate.assert_not_called()

    def test_other_run_wait_can_cancel_and_remote_dispatch_defers_without_taking_ownership(self):
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
                contender = {'run_id':'remote-contender','source':{'sha256':'CPU-source'},
                    'task_intent':{'mode':'create_reference'},
                    'task_plan':{'execution_strategy':'layered_reference_transfer'},'options':{}}
                self.assertIsNone(workflow._begin_task_portrait_beauty(self.root, contender, self.args()))
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
