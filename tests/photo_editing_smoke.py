"""Editing permission is independent of visual style and image-model dialect."""
import copy
import io
import json
from email.message import Message
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import vision_transport as vision

from asset_identity import sha256_file
import photo_art_direction as direction
import photo_editing as editing


def proposal(operation='relight'):
    return {'summary':'人物优先的写实角色照片',
            'observation':{'setting':'indoor','description':'室内人物与木质布景，面部受光偏弱'},
            'changes':[{'operation':operation,'target':'人物和相邻衣料',
                        'instruction':'改善人物主光的方向和明暗层次，并让相邻衣料接受相同照明。',
                        'purpose':'人物成为视觉中心'}],
            'lighting':'让脸、手、衣服和环境的光线方向与软硬一致。',
            'keep':['原有室内空间关系'], 'uncertain':[]}


class EditingScopeTests(unittest.TestCase):
    def test_style_does_not_expand_scene_or_geometry_permission(self):
        retouch=editing.editing_scope({'work_type':'retouch','style':'dramatic'})
        composite=editing.editing_scope({'work_type':'full_composite','style':'natural'})
        self.assertEqual(retouch['scene_policy'],'preserve')
        self.assertFalse(retouch['effects_allowed'])
        self.assertEqual(composite['scene_policy'],'rebuild')
        self.assertEqual(retouch['geometry_policy'],composite['geometry_policy'])
        self.assertRaises(ValueError,editing.validate_observed_design,proposal('scene_composite'),retouch)
        half=editing.editing_scope({'work_type':'partial_composite'})
        self.assertRaises(ValueError,editing.validate_observed_design,proposal('scene_composite'),half)
        self.assertRaises(ValueError,editing.editing_scope,{'work_type':'partial_composite','target_setting':'outdoor'})
        self.assertEqual(editing.editing_scope({'work_type':'full_composite','target_setting':'outdoor'})['target_setting'],'outdoor')

    def test_explicit_veto_and_scope_identity_cannot_be_overridden_by_model(self):
        scope=editing.editing_scope(no_effects=True)
        self.assertRaises(ValueError,editing.validate_observed_design,proposal('effect'),scope)
        changed=copy.deepcopy(scope);changed['allowed_operations'].append('scene_composite')
        self.assertRaises(ValueError,editing.validate_scope,changed)
        changed['scope_id']=editing.identity({k:v for k,v in changed.items() if k!='scope_id'})
        self.assertRaises(ValueError,editing.validate_scope,changed)

    def test_model_dialects_preserve_the_same_decisions_and_critical_invariants(self):
        scope=editing.editing_scope()
        design=editing.validate_observed_design(proposal(),scope)
        gpt=editing.compile_edit_prompt(design,scope,'gpt_image')
        qwen=editing.compile_edit_prompt(design,scope,'qwen_image_2_1')
        self.assertEqual(gpt['design_id'],qwen['design_id'])
        self.assertNotEqual(gpt['prompt'],qwen['prompt'])
        self.assertNotIn('\n',qwen['prompt'])
        self.assertIn('\n',gpt['prompt'])
        for compiled in (gpt,qwen):
            self.assertIn(design['changes'][0]['instruction'],compiled['prompt'])
            self.assertIn('体型',compiled['prompt'])
            self.assertIn('室内外属性',compiled['prompt'])
            self.assertNotIn('scope_id',compiled['prompt'])
        design['changes'][0]['instruction']='改变了已经冻结的指令'
        self.assertRaises(ValueError,editing.compile_edit_prompt,design,scope,'qwen_image_2_1')


class ArtDirectionExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.source=self.root/'source.png';self.source.write_bytes(b'input fixture bytes')
        self.output=self.root/'attempt'
        self.scope=editing.editing_scope()

    def transport(self,source,output,prompt,**kwargs):
        return proposal(),{'source':{'path':str(source),'sha256':sha256_file(source)},
                           'network_used':True,'generation_used':False,'model':'vision-fixture'}

    def reserve_completed_response(self, output=None):
        """Simulate process loss after HTTP completion, before local ready save."""
        output=output or self.output
        Image.new('RGB',(32,48),(60,80,100)).save(self.source)
        response=io.BytesIO(json.dumps({'id':'offline-completed','status':'completed',
            'model':'gpt-6.1-sol','output_text':json.dumps(proposal())}).encode())
        response.headers=Message();response.headers['Content-Type']='application/json'
        with patch.object(vision,'_connection',return_value=(
                {'registry_id':'offline','kind':'openai-compatible'},'http://localhost/v1','fixture-secret',None)), \
             patch.object(vision.urllib.request,'urlopen',return_value=response) as http, \
             patch.object(direction,'atomic_json',side_effect=KeyboardInterrupt('process lost before save')):
            self.assertRaises(KeyboardInterrupt,direction.prepare,self.source,output,self.scope,
                              vision_model='gpt-6.1-sol')
            self.assertEqual(http.call_count,1)
        self.assertEqual(json.loads((output/'art-direction.json').read_text('utf-8'))['status'],'reserved')
        return output

    def test_completed_transport_finishes_reserved_design_without_discovery_http_or_budget_reset(self):
        self.reserve_completed_response()
        observation=self.output/'observation'
        before={str(p):p.read_bytes() for p in observation.rglob('*') if p.is_file()}
        with patch.object(vision,'_connection',side_effect=AssertionError('No connection discovery')), \
             patch.object(vision.urllib.request,'urlopen',side_effect=AssertionError('No HTTP')):
            resumed=direction.prepare(self.source,self.output,self.scope,vision_model='gpt-6.1-sol')
            ready=direction.prepare(self.source,self.output,self.scope,vision_model='gpt-6.1-sol')
        self.assertEqual(resumed['status'],'ready');self.assertTrue(resumed['reused'])
        self.assertEqual(resumed['model_requests_attempted'],1)
        self.assertEqual(resumed['image_submissions'],0)
        self.assertFalse(resumed['automatic_visual_acceptance'])
        self.assertEqual(resumed['provenance']['response_id'],'offline-completed')
        self.assertEqual(resumed['provenance']['response_replay']['new_model_requests'],0)
        self.assertEqual(ready['design'],resumed['design'])
        self.assertEqual(before,{str(p):p.read_bytes() for p in observation.rglob('*') if p.is_file()})

    def test_partial_transport_or_changed_binding_cannot_replay_or_rewrite_reserved_design(self):
        self.reserve_completed_response()
        destination=self.output/'art-direction.json';before=destination.read_bytes()
        with patch.object(vision.urllib.request,'urlopen',side_effect=AssertionError('No retry')):
            for changed in ({'brief':'changed intent'},{'vision_model':'different-model'},{'api_base':'http://different'}):
                kwargs={'vision_model':'gpt-6.1-sol',**changed}
                self.assertRaises(ValueError,direction.prepare,self.source,self.output,self.scope,**kwargs)
            response=self.output/'observation/http-attempts/attempt-0001/response.json'
            response.unlink()
            self.assertRaises(RuntimeError,direction.prepare,self.source,self.output,self.scope,
                              vision_model='gpt-6.1-sol')
        self.assertEqual(destination.read_bytes(),before)

    def test_cancelled_local_replay_does_not_publish_ready_design(self):
        self.reserve_completed_response()
        destination=self.output/'art-direction.json';before=destination.read_bytes()
        checks=iter((False,True))
        with patch.object(vision.urllib.request,'urlopen',side_effect=AssertionError('No retry')):
            self.assertRaises(InterruptedError,direction.prepare,self.source,self.output,self.scope,
                              vision_model='gpt-6.1-sol',cancel=lambda:next(checks))
        self.assertEqual(destination.read_bytes(),before)

    def test_interrupted_response_replay_clears_error_and_checks_cancellation_after_compilation(self):
        self.reserve_completed_response()
        destination=self.output/'art-direction.json'
        record=json.loads(destination.read_text('utf-8'))
        record.update(status='interrupted',error={'type':'InterruptedError','message':'cancelled after response'})
        destination.write_text(json.dumps(record),encoding='utf-8');before=destination.read_bytes()
        checks=iter((False,False,True))
        with patch.object(vision.urllib.request,'urlopen',side_effect=AssertionError('No retry')):
            self.assertRaises(InterruptedError,direction.prepare,self.source,self.output,self.scope,
                              vision_model='gpt-6.1-sol',cancel=lambda:next(checks))
            self.assertEqual(destination.read_bytes(),before)
            ready=direction.prepare(self.source,self.output,self.scope,vision_model='gpt-6.1-sol')
        self.assertEqual(ready['status'],'ready');self.assertNotIn('error',ready)

    def test_cancel_during_final_design_verification_keeps_reservation(self):
        self.reserve_completed_response()
        destination=self.output/'art-direction.json';before=destination.read_bytes()
        compile_prompt=direction.compile_edit_prompt;compiles=[];cancelled=[False]
        def compile_then_cancel(*args,**kwargs):
            result=compile_prompt(*args,**kwargs);compiles.append(result)
            if len(compiles)==3:cancelled[0]=True
            return result
        with patch.object(direction,'compile_edit_prompt',side_effect=compile_then_cancel), \
             patch.object(vision.urllib.request,'urlopen',side_effect=AssertionError('No retry')):
            self.assertRaises(InterruptedError,direction.prepare,self.source,self.output,self.scope,
                              vision_model='gpt-6.1-sol',cancel=lambda:cancelled[0])
        self.assertEqual(len(compiles),4);self.assertEqual(destination.read_bytes(),before)

    def test_one_real_attempt_boundary_reuses_only_matching_source_and_code(self):
        with patch('vision_transport.request_vision_json',side_effect=self.transport) as call:
            initial=direction.prepare(self.source,self.output,self.scope,brief='角色后期')
            reused=direction.prepare(self.source,self.output,self.scope,brief='角色后期')
            self.assertEqual(call.call_count,1)
            self.assertFalse(initial['reused']);self.assertTrue(reused['reused'])
            self.assertEqual(initial['image_submissions'],0)
            self.source.write_bytes(b'changed input fixture bytes')
            self.assertRaises(ValueError,direction.prepare,self.source,self.output,self.scope,brief='角色后期')
            self.assertEqual(call.call_count,1)

    def test_failed_or_forbidden_response_does_not_grant_another_model_call(self):
        with patch('vision_transport.request_vision_json',side_effect=TimeoutError('fixture outcome unknown')) as call:
            self.assertRaises(TimeoutError,direction.prepare,self.source,self.output,self.scope)
            self.assertRaises(RuntimeError,direction.prepare,self.source,self.output,self.scope)
            self.assertEqual(call.call_count,1)
        record=json.loads((self.output/'art-direction.json').read_text(encoding='utf-8'))
        self.assertEqual(record['status'],'failed')
        self.assertFalse(record['automatic_resubmission'])
        retouch=editing.editing_scope({'work_type':'retouch'})
        with patch('vision_transport.request_vision_json',return_value=(proposal('scene_composite'),{
                'source':{'path':str(self.source),'sha256':sha256_file(self.source)},
                'network_used':True,'generation_used':False,'model':'vision-fixture'})) as call:
            output=self.root/'forbidden'
            self.assertRaises(ValueError,direction.prepare,self.source,output,retouch)
            self.assertRaises(RuntimeError,direction.prepare,self.source,output,retouch)
            self.assertEqual(call.call_count,1)
            saved=json.loads((output/'art-direction.json').read_text(encoding='utf-8'))
            self.assertIn('proposed_design',saved)

    def test_cancellation_preserves_returned_evidence_and_never_compiles_a_ready_plan(self):
        with patch('vision_transport.request_vision_json',side_effect=self.transport) as call:
            checks=iter((False,True))
            self.assertRaises(InterruptedError,direction.prepare,self.source,self.output,self.scope,cancel=lambda:next(checks))
            self.assertEqual(call.call_count,1)
        saved=json.loads((self.output/'art-direction.json').read_text(encoding='utf-8'))
        self.assertEqual(saved['status'],'interrupted')
        self.assertIn('provenance',saved)
        self.assertNotIn('compiled_prompts',saved)


if __name__=='__main__':unittest.main()
