"""Unknown manual image calls survive a new child ID, without locking other goals.

Real public task planning/start, source checks, sessions, tools, saves and budget
guards execute. Only model transports are synthetic. Every asserted duplicate
boundary counts actual request_edit callbacks, never just a plan's fields.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from asset_identity import sha256_file
from creative_task_smoke import snapshot
import user_feedback_finish_smoke as feedback_fixture


class ManualUnknownGuardTests(unittest.TestCase):
    def setUp(self):
        self.h = feedback_fixture.UserFeedbackFinishTests(methodName='runTest')
        self.h.setUp(); self.addCleanup(self.h.doCleanups)
        self.root = self.h.root; self.backend = self.h.fx.backend
        self.transport_calls = []; self.mode = 'ordinary'; self.cpu_failure = False
        self.block_transport = False; self.transport_entered = threading.Event(); self.release_transport = threading.Event()
        self.addCleanup(self.release_transport.set)
        self.threads = []
        image_patch = patch('reference_provider.request_edit', side_effect=self.image_transport)
        vision_patch = patch('vision_transport.request_vision_json', side_effect=self.vision_transport)
        image_patch.start(); vision_patch.start()
        self.addCleanup(image_patch.stop); self.addCleanup(vision_patch.stop)
        self.addCleanup(self.join_threads)

    def join_threads(self):
        self.release_transport.set()
        for thread in self.threads:thread.join(10)

    def image_transport(self, *args, **kwargs):
        self.transport_calls.append({'image_sha256': sha256_file(args[2]),
            'prompt_sha256': hashlib.sha256(args[4].encode('utf-8')).hexdigest(),
            'prompt': args[4], 'model': args[3], 'path': str(args[2])})
        self.transport_entered.set()
        if self.block_transport and not self.release_transport.wait(10):
            raise AssertionError('The test did not release its actual transport boundary')
        return self.h.generate(*args, **kwargs)

    def vision_transport(self, source, output, prompt, **kwargs):
        source = Path(source)
        if self.mode in {'seed_working', 'focus'} and 'Select the single most useful available tool' in prompt:
            self.h.fx.vision_requests.append({'source': str(source), 'prompt': prompt})
            result = {'action': 'tool', 'tool': 'edit_region' if self.mode=='seed_working' else 'integrate_generated_focus',
                'operations': ['detail_repair'], 'instruction': self.h.feedback,
                'reason': 'The actual reviewed local working material requires photographic integration.'}
            if self.mode=='seed_working':result['region']=[.25,.25,.75,.75]
            else:
                result['parameters']={'focus_sigma':.01}
                if self.cpu_failure:
                    blocker=self.root/'outputs/runs/cpu-failed/editing-tools'
                    self.assertFalse(blocker.exists())
                    blocker.write_text('Real filesystem obstruction for an image-free tool.',encoding='utf-8')
            return result, {'source': {'path': str(source.resolve()), 'sha256': sha256_file(source)},
                'network_used': True, 'generation_used': False, 'model': 'synthetic-guard-vision'}
        result, provenance = self.h.vision(source, output, prompt, **kwargs)
        if self.mode=='seed_working' and source.name=='review-board.png':
            result.update(credibility={'status':'fail','findings':['Synthetic generated material has a visible focus mismatch.']},
                creative_benefit={'status':'pass','findings':['Synthetic local scene intervention is visibly useful.']},
                comparison='better',primary_issue='Integrate the generated local material with photographic focus.')
        return result, provenance

    def request(self, run_id, *, parent=None):
        request=self.h.request();request['run_id']=run_id
        if parent is not None:
            request.update(parent_run_id=parent['run_id'],reference=parent['artifacts']['auto_draft'])
        return request

    def project_snapshot(self):
        return snapshot(self.root),sorted(str(path.relative_to(self.root))
            for path in self.root.rglob('*') if path.is_dir())

    def checked(self, request):
        before=self.project_snapshot();counts=(len(self.transport_calls),len(self.h.fx.vision_requests))
        checked=self.backend.task_plan(request)
        self.assertTrue(checked['plan']['ready'],checked['plan'].get('blockers'))
        self.assertEqual(self.project_snapshot(),before,'Read-only planning wrote a request reservation or production file/directory')
        self.assertEqual(counts,(len(self.transport_calls),len(self.h.fx.vision_requests)))
        return checked

    def execute(self, checked):
        return self.backend.start_task({**checked['request'],'plan_id':checked['plan']['plan_id'],'async':False})

    def succeeded(self, checked):
        response=self.execute(checked)
        self.assertTrue(response.get('ok'),str(response.get('stderr') or response)[-2000:])
        return response['run']

    def assert_parent_preserved(self):
        self.assertEqual(self.h.parent_path.read_bytes(),self.h.parent_bytes)
        self.assertEqual(sha256_file(self.h.parent_png),self.h.parent_sha)

    def blocked_without_model(self, request, *, already_checked=None):
        before=self.project_snapshot();counts=(len(self.transport_calls),len(self.h.fx.vision_requests))
        if already_checked is None:
            try:checked=self.backend.task_plan(request)
            except (ValueError,RuntimeError):checked=None
            self.assertEqual(self.project_snapshot(),before,'Blocked planning mutated the persistent guard')
        else:checked=already_checked
        if checked is not None and checked['plan'].get('ready'):
            try:response=self.execute(checked)
            except (ValueError,FileExistsError,RuntimeError):response=None
            if response is not None and response.get('ok'):
                summary=(response.get('run') or {}).get('photo_edit_session') or {}
                self.assertFalse(summary.get('automated_target_met',False),'An unresolved call became feedback acceptance')
                self.assertFalse(summary.get('human_approved',False))
        self.assertEqual(counts,(len(self.transport_calls),len(self.h.fx.vision_requests)),
            'An exact-goal unknown retry reached another actual model transport')

    def timeout(self, run_id='unknown-first'):
        self.h.timeout=True
        checked=self.checked(self.request(run_id));child=self.succeeded(checked)
        state=json.loads(Path(child['photo_edit_session']['path']).read_text('utf-8'))
        self.assertEqual(state['attempts'][-1]['status'],'unknown')
        self.assertEqual(child['editing_iteration_result']['failure']['type'],'TimeoutError')
        self.assertEqual(len(self.transport_calls),1)
        self.assertIn(self.h.feedback,self.transport_calls[0]['prompt'])
        self.assert_parent_preserved()
        return checked,child

    def test_new_child_same_exact_goal_and_same_old_run_cannot_resubmit_unknown(self):
        first,_=self.timeout()
        second=self.request('unknown-new-child')
        self.assertEqual(second['parent_run_id'],first['request']['parent_run_id'])
        self.assertEqual(second['editing_iteration']['user_feedback'],first['request']['editing_iteration']['user_feedback'])
        self.blocked_without_model(second)
        self.blocked_without_model(first['request'],already_checked=first)
        self.assertEqual(len(self.transport_calls),1)
        self.assert_parent_preserved()

    def test_other_feedback_is_a_new_target_and_not_locked_by_unknown(self):
        first,_=self.timeout();first_feedback=self.h.feedback
        self.h.feedback='Repair a different visible background distraction while preserving the established portrait light.'
        checked=self.checked(self.request('different-goal'))
        first_inputs=first['plan']['inputs'];second_inputs=checked['plan']['inputs']
        for role in ('source','reference'):self.assertEqual(first_inputs[role]['sha256'],second_inputs[role]['sha256'])
        self.assertNotEqual(first_feedback,checked['plan']['editing_iteration']['user_feedback'])
        self.succeeded(checked)
        self.assertEqual(len(self.transport_calls),2)
        self.assertEqual(self.transport_calls[0]['image_sha256'],self.transport_calls[1]['image_sha256'])
        self.assertNotEqual(self.transport_calls[0]['prompt_sha256'],self.transport_calls[1]['prompt_sha256'])
        self.assertIn(self.h.feedback,self.transport_calls[1]['prompt'])
        self.assert_parent_preserved()

    def test_stale_prechecked_new_child_is_revalidated_at_start_before_another_model(self):
        self.h.timeout=True
        first=self.checked(self.request('prechecked-first'))
        stale=self.checked(self.request('prechecked-second'))
        child=self.succeeded(first)
        self.assertEqual(child['editing_iteration_result']['failure']['type'],'TimeoutError')
        self.assertEqual(len(self.transport_calls),1)
        self.blocked_without_model(stale['request'],already_checked=stale)
        self.assert_parent_preserved()

    def test_known_success_does_not_permanently_lock_an_explicit_new_task(self):
        self.succeeded(self.checked(self.request('known-first')))
        self.succeeded(self.checked(self.request('known-new-child')))
        self.assertEqual(len(self.transport_calls),2)
        self.assertEqual(self.transport_calls[0]['image_sha256'],self.transport_calls[1]['image_sha256'])
        self.assertEqual(self.transport_calls[0]['prompt_sha256'],self.transport_calls[1]['prompt_sha256'])
        self.assert_parent_preserved()

    def test_known_image_response_after_late_cancel_is_not_an_unknown_resubmit_lock(self):
        self.h.cancel_after_image=True
        checked=self.checked(self.request('late-cancelled'))
        started=self.backend.start_task({**checked['request'],'plan_id':checked['plan']['plan_id'],'async':True})
        self.h.cancel_job_id=started['job_id'];self.h.job_available.set()
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            job=self.backend.jobs.get(self.h.cancel_job_id)
            if job['status'] in {'cancelled','failed','blocked','succeeded'}:break
            time.sleep(.01)
        self.assertEqual(job['status'],'cancelled')
        worker=self.backend._job_workers.get(self.h.cancel_job_id)
        if worker is not None:worker.join(5);self.assertFalse(worker.is_alive())
        attempts=list((self.root/'outputs/runs/late-cancelled/editing-tools').glob('*/tool-attempt.json'))
        self.assertEqual(len(attempts),1)
        attempt=json.loads(attempts[0].read_text('utf-8'))
        self.assertEqual(attempt['state'],'interrupted')
        self.assertEqual(sha256_file(attempt['generated']['path']),attempt['generated']['sha256'])
        self.assertEqual(sha256_file(attempt['provider_receipt']['path']),attempt['provider_receipt']['sha256'])
        self.h.cancel_after_image=False
        self.succeeded(self.checked(self.request('known-cancel-response-new-child')))
        self.assertEqual(len(self.transport_calls),2)
        self.assertEqual(self.transport_calls[0]['image_sha256'],self.transport_calls[1]['image_sha256'])
        self.assertEqual(self.transport_calls[0]['prompt_sha256'],self.transport_calls[1]['prompt_sha256'])
        self.assert_parent_preserved()

    def test_real_native_save_failure_after_known_image_does_not_poison_target(self):
        self.h.fail_save=True
        checked=self.checked(self.request('feedback-child'))
        try:
            response=self.execute(checked)
        except (FileExistsError,NotADirectoryError,RuntimeError):
            response=None
        if response is not None:self.assertFalse(response.get('ok'))
        state=json.loads((self.root/'outputs/runs/feedback-child/editing-session/session.json').read_text('utf-8'))
        self.assertTrue(state['automated_target_met'],'The fixture did not reach the actual reviewed image before save failure')
        tool=next(row for row in state['attempts'] if row['kind']=='tool')
        self.assertEqual(tool['status'],'completed')
        self.assertEqual(tool['costs']['image_submissions'],1)
        self.assertEqual(len(self.transport_calls),1)
        self.h.fail_save=False
        self.succeeded(self.checked(self.request('after-known-native-save-failure')))
        self.assertEqual(len(self.transport_calls),2)
        self.assertEqual(self.transport_calls[0]['image_sha256'],self.transport_calls[1]['image_sha256'])
        self.assertEqual(self.transport_calls[0]['prompt_sha256'],self.transport_calls[1]['prompt_sha256'])
        self.assert_parent_preserved()

    def test_image_free_CPU_failure_does_not_poison_new_task_on_same_goal(self):
        self.mode='seed_working';self.h.feedback='Refine the visible local material seam within the supplied scene.'
        parent=self.succeeded(self.checked(self.request('local-working-parent')))
        parent_path=self.root/'outputs/runs/local-working-parent/manifest.json';parent_bytes=parent_path.read_bytes()
        self.mode='focus';self.h.feedback='Integrate the generated material focus with the surrounding scene.'
        request=self.request('cpu-failed',parent=parent)
        request['editing_iteration']['maximum_image_submissions']=0
        checked=self.checked(request);before_images=len(self.transport_calls);before_vision=len(self.h.fx.vision_requests)
        self.assertIsNotNone(checked['plan']['editing_refinement_authorization']['working_candidate'])
        self.cpu_failure=True
        failed=self.succeeded(checked)
        self.assertIsNotNone(failed['editing_iteration_result']['failure'])
        self.assertIn(failed['editing_iteration_result']['failure']['type'],{'FileExistsError','NotADirectoryError'})
        failed_state=json.loads(Path(failed['photo_edit_session']['path']).read_text('utf-8'))
        self.assertEqual(failed_state['attempts'][-1]['kind'],'tool')
        self.assertEqual(failed_state['attempts'][-1]['costs']['image_submissions'],0)
        self.assertEqual(failed['generation_budget']['image_submissions'],0)
        self.assertEqual(len(self.transport_calls),before_images)
        self.assertEqual(len(self.h.fx.vision_requests)-before_vision,1)
        self.cpu_failure=False;request['run_id']='cpu-new-child'
        completed=self.succeeded(self.checked(request))
        self.assertIsNone(completed['editing_iteration_result']['failure'])
        self.assertEqual(completed['generation_budget']['image_submissions'],0)
        self.assertEqual(len(self.transport_calls),before_images)
        self.assertEqual(len(self.h.fx.vision_requests)-before_vision,3)
        self.assertEqual(parent_path.read_bytes(),parent_bytes)
        self.assert_parent_preserved()

    def test_two_prechecked_concurrent_exact_goals_reach_only_one_image_transport(self):
        self.h.timeout=True;self.block_transport=True
        first=self.checked(self.request('concurrent-first'));second=self.checked(self.request('concurrent-second'))
        vision_before=len(self.h.fx.vision_requests);results=[];errors=[];second_entered=threading.Event()
        def invoke(checked,entered=None):
            if entered is not None:entered.set()
            try:results.append(self.execute(checked))
            except Exception as error:errors.append(error)
        one=threading.Thread(target=invoke,args=(first,));self.threads.append(one);one.start()
        self.assertTrue(self.transport_entered.wait(5),'First task never reached its actual reserved transport')
        two=threading.Thread(target=invoke,args=(second,second_entered));self.threads.append(two);two.start()
        self.assertTrue(second_entered.wait(5))
        two.join(.25)
        self.assertEqual(len(self.transport_calls),1,'A second same-goal request entered transport while the first was still unresolved')
        self.release_transport.set()
        for thread in self.threads:thread.join(10);self.assertFalse(thread.is_alive())
        self.assertEqual(len(self.transport_calls),1,'Concurrent child IDs duplicated the unresolved image request')
        self.assertEqual(len(self.h.fx.vision_requests)-vision_before,1,'The second exact goal paid another decision before being blocked')
        self.assertTrue(errors or any(not result.get('ok') for result in results))
        self.assert_parent_preserved()


class QualityRejectedOwnerEvidenceTests(unittest.TestCase):
    """A returned rejected image is known only with its entire owned evidence."""

    def setUp(self):
        import photo_edit_session as session
        import photo_request_budget as budget
        from PIL import Image
        self.budget=budget
        temporary=tempfile.TemporaryDirectory(prefix='quality-rejected-owner-');self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name).resolve();self.run_id='quality-rejected-child'
        self.enterContext(patch.dict('os.environ',{'COSPLAY_ASSET_IDENTITY_CACHE':str(self.root/'identity-cache')}))
        self.enterContext(patch('urllib.request.urlopen',side_effect=AssertionError('Accounting must not use HTTP')))
        self.run=self.root/'outputs/runs'/self.run_id
        self.output=self.run/'editing-tools/paid-image-attempt';self.provider=self.output/'provider'
        self.provider.mkdir(parents=True);(self.run/'editing-session').mkdir()
        self.original=self.root/'original.ARW';self.original.write_bytes(b'opaque original camera bytes')
        self.source=self.root/'source-preview.png';self.base=self.root/'baseline-U.png'
        Image.new('RGB',(24,32),'gray').save(self.source);Image.new('RGB',(24,32),'blue').save(self.base)
        self.input=self.output/'current-best-input.png';Image.new('RGB',(18,24),'green').save(self.input)
        self.rejected=self.provider/'edited.quality-rejected-fixture.png';Image.new('RGB',(18,24),'magenta').save(self.rejected)
        record=lambda path:{'path':str(path),'sha256':sha256_file(path)}
        self.original_record=record(self.original);base=record(self.base)
        self.target=budget.manual_target_key(source_sha256=self.original_record['sha256'],base_sha256=base['sha256'],
            scope_id='3'*64,design_id='4'*64,user_feedback='An explicit checked local correction',provider_target={'fixture':'qwen'})
        binding={'source':self.original_record,'initial_usage':dict.fromkeys(session.COUNTERS,0)}
        context={'session_id':session.identity(binding),'source_original':self.original_record,
            'images':[{'role':'source',**record(self.source)},{'role':'best',**base}],
            'decision':{'action':'tool','tool':'edit_region','operations':['relight']}}
        costs=dict.fromkeys(session.COUNTERS,0);costs.update(image_submissions=1,tool_calls=1,local_repairs=1,improvement_steps=1)
        self.attempt={'attempt_id':'paid-image-attempt','kind':'tool','status':'unknown','costs':costs,'context':context}
        self.state={'schema':session.SCHEMA,'session_id':context['session_id'],'binding':binding,
            'scope':{'scope_id':self.target['scope_id']},'design':{'design_id':self.target['design_id']},
            'user_intent':{'user_feedback':self.target['user_feedback']},'candidates':[{'image':base}],
            'used':costs,'status':'needs_attention','attempts':[self.attempt]}
        self.session_path=self.run/'editing-session/session.json';self.write(self.session_path,self.state)
        prompt='Preserve the actual checked U and execute only this local correction.'
        self.tool={'schema':'cosluma_photo_edit_tool.v1','state':'needs_attention',
            'binding':{'session_id':context['session_id'],'attempt_id':self.attempt['attempt_id'],
                'context_sha256':budget.identity(context),'source':self.original_record,'current_best':context['images'][1]},
            'generator_input':record(self.input),'compiled_prompt':{'prompt':prompt,'sha256':budget.identity(prompt)},
            'automatic_retry':False,'human_approved':False}
        self.tool_path=self.output/'tool-attempt.json';self.write(self.tool_path,self.tool)
        generated=record(self.rejected);size=self.rejected.stat().st_size
        self.quality={'schema':'qwen_grid_quality_gate.v1','context':'generated_output',
            'grid_policy':'reject_corroborated_periodic_grid','decodable':True,'passed':False,
            'grid_detected':True,'threshold_exceeded':True,'threshold':.02,'diag_peak_lag_4_16':.08,
            'periodicity':{'corroborated':True,'threshold':.02},
            'metrics':{'sha256':generated['sha256'],'bytes':size,'size':[18,24]}}
        self.rejection={'schema':'reference_quality_rejection.v1','state':'rejected_quality','provider':'qwen_image_2_1',
            'model':'Qwen fixture','source':self.original_record,'output_sha256':generated['sha256'],
            'rejected_path':str(self.rejected),'quality':copy.deepcopy(self.quality),'candidate_registered':False,
            'image_generation_submissions':1,'reason':'A diagnostic label is not retry authority'}
        self.rejection_path=self.provider/'edited.quality-rejection.json';self.write(self.rejection_path,self.rejection)
        from comfyui_adapter import prepare_condition_workflow
        document=json.loads((Path(__file__).resolve().parents[1]/'providers/workflows/qwen-image-2.1-edit.json').read_text('utf8'))
        document['model_name']='Qwen fixture'
        self.workflow=self.root/'bound-qwen-workflow.json';self.write(self.workflow,document)
        self.mask=self.output/'allowed-edit-mask.png'
        mask_image=Image.new('L',(18,24),0);mask_image.paste(255,(4,4,12,20));mask_image.save(self.mask)
        graph=prepare_condition_workflow(document,['actual-owned-upload.png','actual-owned-mask.png'],prompt,
            coordinate_space='checked_current_best_crop')
        self.conditions=[{'role':role,'coordinate_space':'checked_current_best_crop',
            'input_sha256':sha256_file(path),'api_input':{**record(path),'bytes':path.stat().st_size}}
            for role,path in (('source',self.input),('edit_mask',self.mask))]
        self.transport={'schema':'reference_transport_attempt.v1','state':'rejected_quality','provider':'qwen_image_2_1',
            'source':self.original_record,'api_input':{**record(self.input),'bytes':self.input.stat().st_size,'derived':False},
            'request':{'method':'POST','requested_prompt':prompt,'prompt':prompt,'model':str(self.workflow)},'automatic_retry':False,
            'response':{'id':'owned-prompt','mode':'comfyui_view','image_bytes':size,'output_sha256':generated['sha256'],
                'generated_model':'Qwen fixture'},'candidate_registered':False,'quality':copy.deepcopy(self.quality),
            'quality_rejection_path':str(self.rejection_path),'quality_rejection_sha256':sha256_file(self.rejection_path),
            'rejected_path':str(self.rejected),'condition_inputs':copy.deepcopy(self.conditions),
            'events':[{'event':'comfyui_uploaded','index':1,'input_sha256':sha256_file(self.input),'bytes':self.input.stat().st_size,
                'upload':{'name':'actual-owned-upload.png','subfolder':'','type':'input'}},
                {'event':'comfyui_uploaded','index':2,'input_sha256':sha256_file(self.mask),'bytes':self.mask.stat().st_size,
                'upload':{'name':'actual-owned-mask.png','subfolder':'','type':'input'}},
                {'event':'comfyui_submitted','prompt_id':'owned-prompt',
                    'submitted_workflow_sha256':hashlib.sha256(json.dumps(graph,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                    'request':{'prompt':graph}},
                {'event':'comfyui_completed','prompt_id':'owned-prompt','output_node':'8','status':{'completed':True,'status_str':'success',
                    'messages':[['execution_success',{'prompt_id':'owned-prompt'}]]}}]}
        self.transport_path=self.provider/'edited.transport-fixture.json';self.write(self.transport_path,self.transport)
        self.guard=budget._manual_path(self.root,self.target);self.guard.parent.mkdir(parents=True)
        self.write(self.guard,{'schema':budget.MANUAL_GUARD_SCHEMA,'binding':self.target,'target_id':budget.identity(self.target),
            'attempts':[{'run_id':self.run_id,'status':'unknown','token':'existing-owner','automatic_resubmission':False}]})

    @staticmethod
    def write(path,value):
        path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')

    def classify(self):
        return self.budget.classify_manual_outcome(self.root,self.target,run_id=self.run_id,require_session=True)

    def test_complete_bound_rejection_is_known_without_changing_guard_usage_or_candidate(self):
        before={path:path.read_bytes() for path in self.run.rglob('*') if path.is_file()}
        guard=self.guard.read_bytes()
        result=self.classify();self.assertEqual(result['status'],'known',result)
        self.assertIn(str(self.rejection_path),[row['path'] for row in result['evidence']])
        self.assertIn(str(self.rejected),[row['path'] for row in result['evidence']])
        self.assertEqual(self.guard.read_bytes(),guard)
        self.assertEqual({path:path.read_bytes() for path in before},before)
        self.assertEqual(self.state['used']['image_submissions'],1)
        self.assertFalse((self.provider/'edited.png').exists());self.assertFalse((self.provider/'edited.provenance.json').exists())
        with self.assertRaises(self.budget.ManualTargetBlocked):self.budget.check_manual_target(self.root,self.target)
        # The unused original source loader 4 and active alias 10 have the same
        # actual bytes; either may serve the live source port, never the mask.
        alias=copy.deepcopy(self.transport)
        submitted=next(row for row in alias['events'] if row['event']=='comfyui_submitted')
        submitted['request']['prompt']['5']['inputs']['images.image_1']=['4',0]
        submitted['submitted_workflow_sha256']=hashlib.sha256(json.dumps(
            submitted['request']['prompt'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        self.write(self.transport_path,alias)
        self.assertEqual(self.classify()['status'],'known')
        self.assertEqual(self.guard.read_bytes(),guard)
        # Native single-image Qwen conditioning follows the original source
        # loader and the same sampled output chain, without a mask condition.
        from comfyui_adapter import prepare_condition_workflow
        single=copy.deepcopy(self.transport);single['condition_inputs']=None
        single['events']=[row for row in single['events'] if row.get('event')!='comfyui_uploaded' or row.get('index')==1]
        submission=next(row for row in single['events'] if row['event']=='comfyui_submitted')
        submission['request']['prompt']=prepare_condition_workflow(json.loads(self.workflow.read_text('utf8')),
            ['actual-owned-upload.png'],self.tool['compiled_prompt']['prompt'])
        submission['submitted_workflow_sha256']=hashlib.sha256(json.dumps(
            submission['request']['prompt'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        self.write(self.transport_path,single);self.assertEqual(self.classify()['status'],'known')
        self.assertEqual(self.guard.read_bytes(),guard)

    def test_changed_or_incomplete_rejection_evidence_stays_unknown(self):
        def changed_workflow(value,*,prompt=False):
            submission=next(row for row in value['events'] if row['event']=='comfyui_submitted');graph=submission['request']['prompt']
            if prompt:graph['5']['inputs']['prompt']='A coherently rehashed wrong prompt'
            else:graph['10']['inputs']['image']='A different unbound upload.png'
            submission['submitted_workflow_sha256']=hashlib.sha256(json.dumps(graph,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        cases=[
            ('fake-state','transport',lambda v:v.update(state='response_received')),
            ('fake-registration','transport',lambda v:v.update(state='registered',candidate_registered=True)),
            ('wrong-schema','transport',lambda v:v.update(schema='other.v1')),
            ('missing-response','transport',lambda v:v.pop('response')),
            ('missing-response-id','transport',lambda v:v['response'].pop('id')),
            ('wrong-response-id','transport',lambda v:v['response'].update(id='other-owner')),
            ('wrong-response-mode','transport',lambda v:v['response'].update(mode='incomplete')),
            ('empty-return','transport',lambda v:v['response'].update(image_bytes=0)),
            ('wrong-return-size','transport',lambda v:v['response'].update(image_bytes=1)),
            ('wrong-return-sha','transport',lambda v:v['response'].update(output_sha256='f'*64)),
            ('wrong-original-source','transport',lambda v:v['source'].update(sha256='f'*64)),
            ('wrong-requested-prompt','transport',lambda v:v['request'].update(requested_prompt='Different operation')),
            ('wrong-effective-prompt','transport',lambda v:v['request'].update(prompt='Different effective operation')),
            ('missing-input','transport',lambda v:v.pop('api_input')),
            ('missing-workflow-binding','transport',lambda v:v['request'].pop('model')),
            ('non-json-workflow-binding','transport',lambda v:v['request'].update(model=str(self.source))),
            ('relative-workflow-binding','transport',lambda v:v['request'].update(model='bound-qwen-workflow.json')),
            ('wrong-input-sha','transport',lambda v:v['api_input'].update(sha256='e'*64)),
            ('wrong-input-path','transport',lambda v:v['api_input'].update(path=str(self.base))),
            ('wrong-upload','transport',lambda v:v['events'][0].update(input_sha256='e'*64)),
            ('no-completion','transport',lambda v:v['events'].pop()),
            ('no-submission','transport',lambda v:v['events'].pop(2)),
            ('wrong-completion-owner','transport',lambda v:v['events'][-1].update(prompt_id='other-owner')),
            ('incomplete','transport',lambda v:v['events'][-1]['status'].update(completed=False)),
            ('failed-completion','transport',lambda v:v['events'][-1]['status'].update(status_str='error')),
            ('missing-success-message','transport',lambda v:v['events'][-1]['status'].update(messages=[])),
            ('wrong-success-owner','transport',lambda v:v['events'][-1]['status']['messages'][0][1].update(prompt_id='other-owner')),
            ('wrong-workflow-sha','transport',lambda v:v['events'][2].update(submitted_workflow_sha256='e'*64)),
            ('rehashed-wrong-workflow-input','transport',changed_workflow),
            ('rehashed-wrong-workflow-prompt','transport',lambda v:changed_workflow(v,prompt=True)),
            ('contradictory-failure-event','transport',lambda v:v['events'].append({'event':'comfyui_execution_failed','prompt_id':'owned-prompt'})),
            ('missing-rejection-locator','transport',lambda v:v.pop('quality_rejection_path')),
            ('wrong-rejection-sha','transport',lambda v:v.update(quality_rejection_sha256='f'*64)),
            ('wrong-rejected-locator','transport',lambda v:v.update(rejected_path=str(self.base))),
            ('registered-transport','transport',lambda v:v.update(candidate_registered=True)),
            ('wrong-rejection-schema','rejection',lambda v:v.update(schema='other.v1')),
            ('wrong-rejection-state','rejection',lambda v:v.update(state='registered')),
            ('wrong-rejection-source','rejection',lambda v:v['source'].update(sha256='e'*64)),
            ('wrong-rejection-output','rejection',lambda v:v.update(output_sha256='e'*64)),
            ('wrong-rejection-model','rejection',lambda v:v.update(model='Other returned model')),
            ('wrong-rejection-provider','rejection',lambda v:v.update(provider='other_provider')),
            ('registered-rejection','rejection',lambda v:v.update(candidate_registered=True)),
            ('wrong-image-count','rejection',lambda v:v.update(image_generation_submissions=0)),
            ('wrong-rejection-input','quality',lambda v:v['quality'].update(context='source_input')),
            ('wrong-quality-schema','quality',lambda v:v['quality'].update(schema='other.v1')),
            ('passed-quality','quality',lambda v:v['quality'].update(passed=True)),
            ('uncorroborated-quality','quality',lambda v:v['quality']['periodicity'].update(corroborated=False)),
            ('wrong-quality-image','quality',lambda v:v['quality']['metrics'].update(sha256='f'*64)),
            ('wrong-quality-byte-count','quality',lambda v:v['quality']['metrics'].update(bytes=1)),
            ('wrong-tool-owner','tool',lambda v:v['binding'].update(session_id='e'*64)),
            ('wrong-tool-attempt','tool',lambda v:v['binding'].update(attempt_id='other-attempt')),
            ('wrong-tool-context','tool',lambda v:v['binding'].update(context_sha256='e'*64)),
            ('wrong-tool-source','tool',lambda v:v['binding']['source'].update(sha256='e'*64)),
            ('wrong-tool-best','tool',lambda v:v['binding']['current_best'].update(sha256='e'*64)),
            ('wrong-generator-input','tool',lambda v:v['generator_input'].update(sha256='e'*64)),
            ('wrong-compiled-prompt-sha','tool',lambda v:v['compiled_prompt'].update(sha256='e'*64)),
        ]
        def live_graph(value,change):
            submission=next(row for row in value['events'] if row['event']=='comfyui_submitted')
            change(submission['request']['prompt'])
            submission['submitted_workflow_sha256']=hashlib.sha256(json.dumps(
                submission['request']['prompt'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        def disconnected_prompt(graph):
            graph['5']['inputs']['prompt']='A different actually encoded operation'
            graph['99']={'class_type':'UnusedNote','inputs':{'prompt':self.tool['compiled_prompt']['prompt']}}
        def other_encoder(graph,port):
            graph['55']=copy.deepcopy(graph['5']);graph['6']['inputs'][port]=['55',{'positive':0,'negative':1,'latent_image':2,'cfg':3}[port]]
        wiring=[
            ('source-mask-swapped',lambda g:g['5']['inputs'].update({'images.image_1':['11',0],'images.image_2':['10',0]})),
            ('all-images-mask',lambda g:g['5']['inputs'].update({'images.image_1':['11',0],'images.image_2':['11',0]})),
            ('disconnected-expected-prompt',disconnected_prompt),
            ('save-direct-uploaded-mask',lambda g:g['8']['inputs'].update(images=['11',0])),
            ('save-direct-source',lambda g:g['8']['inputs'].update(images=['4',0])),
            ('decode-direct-mask',lambda g:g['7']['inputs'].update(samples=['11',0])),
            ('decode-wrong-sampler-port',lambda g:g['7']['inputs'].update(samples=['6',1])),
            ('save-wrong-decoder-port',lambda g:g['8']['inputs'].update(images=['7',1])),
            ('source-wrong-output-port',lambda g:g['5']['inputs'].update({'images.image_1':['10',1]})),
            ('source-float-output-port',lambda g:g['5']['inputs'].update({'images.image_1':['10',0.]})),
            ('positive-other-encoder',lambda g:other_encoder(g,'positive')),
            ('negative-other-encoder',lambda g:other_encoder(g,'negative')),
            ('latent-other-encoder',lambda g:other_encoder(g,'latent_image')),
            ('cfg-other-encoder',lambda g:other_encoder(g,'cfg')),
            ('positive-wrong-output-port',lambda g:g['6']['inputs'].update(positive=['5',1])),
            ('negative-wrong-output-port',lambda g:g['6']['inputs'].update(negative=['5',0])),
            ('latent-wrong-output-port',lambda g:g['6']['inputs'].update(latent_image=['5',0])),
            ('cfg-wrong-output-port',lambda g:g['6']['inputs'].update(cfg=['5',0])),
            ('positive-bool-output-port',lambda g:g['6']['inputs'].update(positive=['5',False])),
            ('negative-bool-output-port',lambda g:g['6']['inputs'].update(negative=['5',True])),
            ('vae-wrong-output-port',lambda g:g['7']['inputs'].update(vae=['3',1])),
            ('missing-live-decoder',lambda g:g.pop('7')),
            ('missing-live-encoder',lambda g:g.pop('5')),
            ('missing-live-image',lambda g:g.pop('10')),
            ('save-cycle',lambda g:g['8']['inputs'].update(images=['8',0])),
            ('decode-cycle',lambda g:g['7']['inputs'].update(samples=['7',0])),
            ('model-cache-cycle',lambda g:g['9']['inputs'].update(model=['9',0])),
            ('unrecognized-live-encoder',lambda g:g['5'].update(class_type='CustomUnknownEncoder')),
            ('unrecognized-live-sampler',lambda g:g['6'].update(class_type='KSamplerAdvanced')),
            ('unrecognized-live-decoder',lambda g:g['7'].update(class_type='CustomUnknownDecoder')),
            ('wrong-model-weight-name',lambda g:g['1']['inputs'].update(unet_name='foreign-model.safetensors')),
            ('wrong-clip-weight-name',lambda g:g['2']['inputs'].update(clip_name='foreign-clip.safetensors')),
            ('wrong-vae-weight-name',lambda g:g['3']['inputs'].update(vae_name='foreign-vae.safetensors')),
            ('wrong-bound-sampling-parameters',lambda g:g['6']['inputs'].update(denoise=0.)),
        ]
        cases.extend((name,'transport',lambda value,change=change:live_graph(value,change)) for name,change in wiring)
        cases.extend((name,'transport',lambda value,output=output:value['events'][-1].update(output_node=output))
            for name,output in (('missing-completed-output',None),('wrong-completed-output','11'),('absent-completed-output','999')))
        paths={'transport':self.transport_path,'rejection':self.rejection_path,'quality':self.rejection_path,'tool':self.tool_path}
        original={path:path.read_bytes() for path in set(paths.values())};guard=self.guard.read_bytes()
        for name,kind,change in cases:
            with self.subTest(case=name):
                for path,data in original.items():path.write_bytes(data)
                value=json.loads(paths[kind].read_text('utf8'));change(value);self.write(paths[kind],value)
                if kind in {'rejection','quality'}:
                    transport=copy.deepcopy(self.transport);transport['quality_rejection_sha256']=sha256_file(self.rejection_path)
                    if kind=='quality':transport['quality']=copy.deepcopy(value['quality'])
                    self.write(self.transport_path,transport)
                self.assertEqual(self.classify()['status'],'unknown',name)
                self.assertEqual(self.guard.read_bytes(),guard)
        for path,data in original.items():path.write_bytes(data)

    def test_missing_or_corrupt_owned_files_and_reason_only_never_become_known(self):
        files=(self.transport_path,self.rejection_path,self.rejected,self.input,self.mask,self.original,self.base,self.workflow)
        original={path:path.read_bytes() for path in files}
        for path in files:
            for mode in ('missing','changed'):
                with self.subTest(file=path.name,mode=mode):
                    if mode=='missing':path.unlink()
                    else:path.write_bytes(b'changed or incomplete bytes')
                    self.assertEqual(self.classify()['status'],'unknown')
                    path.write_bytes(original[path])
        for value in ({'schema':'reference_transport_attempt.v1','state':'rejected_quality'},
                      {**self.transport,'response':None,'events':[{'event':'provider_attempt','http_status':200}],
                       'reason':'qwen_diagonal_grid_artifact'}):
            self.write(self.transport_path,value);self.assertEqual(self.classify()['status'],'unknown')
        self.transport_path.write_bytes(original[self.transport_path])
        for name in ('edited.png','edited.provenance.json'):
            path=self.provider/name;path.write_bytes(b'Not an allowed registered candidate')
            self.assertEqual(self.classify()['status'],'unknown');path.unlink()

    def test_actual_unreadable_return_is_known_without_reclassifying_its_pixels(self):
        self.rejected.write_bytes(b'A complete provider response that is not a readable image')
        digest=sha256_file(self.rejected);size=self.rejected.stat().st_size
        quality={'schema':'qwen_grid_quality_gate.v1','context':'generated_output',
            'grid_policy':'reject_corroborated_periodic_grid','decodable':False,'passed':False,
            'metrics':{'bytes':size,'error_type':'UnidentifiedImageError'},'reason':'unreadable_image'}
        rejection=copy.deepcopy(self.rejection);rejection.update(output_sha256=digest,quality=quality)
        self.write(self.rejection_path,rejection)
        transport=copy.deepcopy(self.transport);transport['response'].update(image_bytes=size,output_sha256=digest)
        transport.update(quality=quality,quality_rejection_sha256=sha256_file(self.rejection_path))
        self.write(self.transport_path,transport)
        self.assertEqual(self.classify()['status'],'known')
        self.assertFalse((self.provider/'edited.png').exists())


class CPUManualTargetIdentityTests(unittest.TestCase):
    def setUp(self):
        import photo_request_budget as budget
        self.budget=budget
        temporary=tempfile.TemporaryDirectory(prefix='cpu-manual-target-');self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name).resolve()
        self.options={'provider':'unused-image-A','model':'unused-model-A','vision_model':'actual-vision-A'}

    def key(self,target):
        return self.budget.manual_target_key(source_sha256='1'*64,base_sha256='2'*64,scope_id='3'*64,
            design_id='4'*64,provider_target=target,shaping_request={'action':'apply','parts':{'face':.3}})

    def test_unknown_CPU_target_cannot_reset_by_changing_unused_model_or_provider(self):
        with patch('reference_provider.provider_config',side_effect=AssertionError('CPU must not parse an unused image target')) as image,\
                patch('vision_transport._connection',return_value=({'kind':'openai-compatible'},
                    'https://actual-vision.invalid/v1','not-persisted',None)) as vision:
            original=self.budget.manual_provider_target(self.options,image_generation=False)
            changed=self.budget.manual_provider_target({**self.options,'provider':'unused-image-B','model':'unused-model-B'},image_generation=False)
        self.assertFalse(image.called);self.assertEqual(vision.call_count,2)
        self.assertEqual(vision.call_args_list[0].kwargs['provider'],'unused-image-A')
        self.assertEqual(vision.call_args_list[1].kwargs['provider'],'unused-image-B')
        self.assertEqual(original['image'],{'kind':'not_requested'});self.assertEqual(original,changed)
        binding=self.key(original);guard=self.budget._manual_path(self.root,binding);guard.parent.mkdir(parents=True)
        guard.write_text(json.dumps({'schema':self.budget.MANUAL_GUARD_SCHEMA,'binding':binding,
            'target_id':self.budget.identity(binding),'attempts':[{'run_id':'actual-unknown-owner','status':'unknown'}]}),encoding='utf8')
        before=guard.read_bytes()
        with self.assertRaises(self.budget.ManualTargetBlocked):self.budget.check_manual_target(self.root,self.key(changed))
        self.assertEqual(guard.read_bytes(),before)

    def test_actual_resolved_vision_model_or_endpoint_changes_remain_distinct(self):
        def connection(**values):
            endpoint=values.get('api_base') or ('https://vision-b.invalid/v1' if values.get('provider')=='vision-B' else 'https://vision-a.invalid/v1')
            return {'kind':'openai-compatible'},endpoint,'never-store-a-key',None
        with patch('vision_transport._connection',side_effect=connection):
            original=self.budget.manual_provider_target(self.options,image_generation=False)
            model=self.budget.manual_provider_target({**self.options,'vision_model':'actual-vision-B'},image_generation=False)
            endpoint=self.budget.manual_provider_target({**self.options,'api_base':'https://another-vision.invalid/v1'},image_generation=False)
            provider=self.budget.manual_provider_target({**self.options,'provider':'vision-B'},image_generation=False)
        for target in (model,endpoint,provider):
            self.assertEqual(target['image'],original['image']);self.assertNotEqual(self.key(target),self.key(original))

    def test_actual_image_request_identity_and_configuration_errors_remain_unchanged(self):
        def image_config(**values):
            return {'kind':'comfyui','registry_id':'qwen-fixture','model':values['model'],'endpoint':'http://image.invalid:8188'}
        with patch('reference_provider.provider_config',side_effect=image_config) as image,\
                patch('vision_transport._connection',return_value=({'kind':'openai-compatible'},'https://vision.invalid/v1','private',None)):
            first=self.budget.manual_provider_target(self.options,image_generation=True)
            second=self.budget.manual_provider_target({**self.options,'model':'another-actual-image-model'},image_generation=True)
        self.assertEqual(image.call_count,2);self.assertEqual(first['image']['kind'],'comfyui')
        self.assertNotEqual(first['image'],second['image'])
        with patch('reference_provider.provider_config',side_effect=ValueError('Image configuration unavailable')):
            with self.assertRaises(ValueError):self.budget.manual_provider_target(self.options,image_generation=True)


if __name__=='__main__':
    unittest.main()
