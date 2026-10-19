"""Known returned local pixels can be recomposed with zero new model allowance."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import creative_task as task
from creative_direction_smoke import enhanced
from creative_task_smoke import fixture, snapshot
from task_model_preflight_smoke import mocked_task_capability


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf8')


def asset(path):
    return {'path': str(path), 'sha256': task.sha(path)}


def pixels_sha(values):
    values = np.ascontiguousarray(values)
    return hashlib.sha256(str((values.shape, values.dtype.str)).encode()+values.tobytes()).hexdigest()


class CandidateReuseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='regional-known-candidate-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        registry, sources, reference = fixture(self.root)
        self.common = {'source': sources[0], 'reference': reference, 'model_registry': str(registry),
            'provider': 'qwen_image_2_1_uncensored_fast', 'character_name': '祝小红',
            'character_notes': '蝴蝶为情感意象，不是技能', 'input_prompt': 'Keep scene and person'}
        self.enterContext(patch('task_model_preflight.inspect_task_models', side_effect=mocked_task_capability))
        self.enterContext(patch('reference_provider.provider_health', return_value={'ready': True}))
        self.enterContext(patch('urllib.request.urlopen', side_effect=AssertionError('No network permitted')))
        original = task.plan_task(self.root, {**self.common, 'creative_task': enhanced('existing_reference', regional=True)})
        self.parent = {'run_id': 'failed-known', 'status': 'blocked', 'task_intent': original['plan']['intent'],
            'task_plan': original['plan'], 'source': original['plan']['inputs']['source'],
            'reference': original['plan']['inputs']['reference'], 'options': original['request'],
            'generation_budget': {'origin_run_id': 'failed-known'}}
        self.manifest = self.root/'outputs/runs/failed-known/manifest.json';write(self.manifest, self.parent)
        self.directory = self.root/'outputs/cache/one-photo'; self.directory.mkdir(parents=True)
        context = np.full((1024,1024,3), .1, np.float32)
        core = [1,2,12,16];box = [-506,-503,518,521]
        permission = np.zeros((1024,1024), bool)
        permission[core[1]-box[1]:core[3]-box[1],core[0]-box[0]:core[2]-box[0]] = True
        np.savez(self.directory/'input-context.npz', values=context)
        np.savez(self.directory/'reliable-halo.npz', values=np.ones((1024,1024), np.float32))
        Image.fromarray(permission.astype('uint8')*255).save(self.directory/'permission.png')
        Image.new('RGB',(1024,1024),'#444444').save(self.directory/'roi.png')
        conditions = {'inputs': [{'role':'source',**asset(self.directory/'roi.png'),'coordinate_space':'source_canvas_roi'}],
            'permission_mask': {**asset(self.directory/'permission.png'),'transmitted_to_model':False}}
        write(self.directory/'conditions.json', conditions)
        self.raw = self.directory/'provider/regional.png';self.raw.parent.mkdir()
        Image.new('RGB',(1024,1024),'#222222').save(self.raw)
        graph = {'4':{'class_type':'LoadImage','inputs':{'image':'roi-upload.png'}},
            '5':{'class_type':'CosplayTextEncodeQwenImage21','inputs':{'images.image_1':['4',0],'prompt':'ONE local edit'}}}
        self.provenance = {'generation_id':'one-real-id','prompt':'ONE local edit','asset':asset(self.raw),
            'source': self.parent['source'], 'provider_contract': {'response': {'id':'one-real-id'},
            'transport': {'events':[{'event':'comfyui_submitted','prompt_id':'one-real-id','request':{'prompt':graph}}]},
            'comfyui': {'prompt_id':'one-real-id','condition_inputs':[{**asset(self.directory/'roi.png'),'upload':{'name':'roi-upload.png'}}],
                'request':{'prompt':graph}}}}
        self.provenance_path=self.raw.with_name('regional.provenance.json');write(self.provenance_path,self.provenance)
        write(self.directory/'request.json', {'run':self.parent,'plan':{'canvas':{'source_oriented_size':[25,20]}}})
        self.ledger_value = {'schema':'regional_scene_refinement.v1','state':'generated_rejected',
            'provider_returned':True,'model_submissions':1,'prompt_id':'one-real-id','request_started':True,
            'raw_candidate':asset(self.raw),'raw_provenance':asset(self.provenance_path),
            'binding':{'source':self.parent['source']['sha256'],'reference':self.parent['reference']['sha256'],
                'core_box':core,'context_box':box,'native_size':[25,20],'checked_plan_id':original['plan']['plan_id'],
                'direction_id':original['plan']['creative_direction']['direction_id'],'prompt':'ONE local edit',
                'context_pixels_sha256':pixels_sha(context),'permission_pixels_sha256':pixels_sha(permission),
                'base_core_pixels_sha256':pixels_sha(context[505:519,507:518])}}
        self.ledger=self.root/'outputs/regional-reservations/failed-known/regional-refinement.json'
        write(self.ledger,self.ledger_value)
        intent=copy.deepcopy(self.parent['task_intent'])
        intent['regional_refinement'].update(maximum_submissions=0,reuse_generated_from_run_id='failed-known')
        self.body={**self.common,'creative_task':intent}

    def test_zero_generation_checked_proof_roundtrip_and_readonly(self):
        before=snapshot(self.root);checked=task.plan_task(self.root,self.body)
        self.assertEqual(before,snapshot(self.root))
        self.assertEqual(sum(checked['plan']['image_budget'].values()),0)
        self.assertFalse(checked['request']['generate'])
        self.assertFalse(checked['plan']['services']['image_generation'])
        proof=checked['plan']['regional_candidate_reuse_authorization']
        self.assertEqual(proof,checked['request']['regional_candidate_reuse_authorization'])
        self.assertEqual(len(proof['assets']),8)
        self.assertEqual(proof['original_core_box'],[1,2,12,16])
        self.assertEqual(proof['requested_core_box'],[1,2,12,16])
        self.assertEqual(proof,task.revalidate_regional_candidate_reuse(self.root,checked['plan']))
        self.assertEqual(checked['plan']['plan_id'],task.plan_task(self.root,checked['request'])['plan']['plan_id'])
        execute=task.execution_request(self.root,{**checked['request'],'plan_id':checked['plan']['plan_id']})
        task.validate_task_plan(execute['task_plan'],execute['task_intent'])
        from qt_workbench import creative_task_plan_text
        self.assertIn('新增生图 0 次',creative_task_plan_text(checked['plan']))
        bad=copy.deepcopy(checked['plan']);bad.pop('regional_candidate_reuse_authorization')
        with self.assertRaises(ValueError):task.validate_task_plan(bad,bad['intent'])
        bad=copy.deepcopy(checked['plan']);bad['image_budget']['regional_refinement']=1
        with self.assertRaises(ValueError):task.validate_task_plan(bad,bad['intent'])

    def test_backend_retry_preserves_known_pixels_and_zero_budget(self):
        from backend_server import Backend
        from production_coordinator import route_contract
        from argparse import Namespace
        import single_source_workflow as workflow
        backend=Backend(self.root,embedded=True)
        self.parent['generation_budget']['regional_refinement_outcome']='generated_rejected'
        request={**self.common,'input_prompt':self.common['input_prompt']}
        previous_bytes=self.ledger.read_bytes()
        with patch.object(backend,'one_click',side_effect=lambda body:{'submitted':body}):
            result=backend._continue_checked_task(self.parent,request,
                resume_unsubmitted_region=True,regional_reuse_core_box=[2,3,12,16])
        submitted=result['submitted'];plan=submitted['task_plan']
        self.assertEqual(sum(plan['image_budget'].values()),0)
        self.assertFalse(submitted['generate'])
        self.assertEqual(plan['regional_candidate_reuse_authorization']['requested_core_box'],[2,3,12,16])
        self.assertEqual(self.ledger.read_bytes(),previous_bytes)
        argv=backend._one_click_args(submitted,submitted['source'])
        settings=json.loads(argv[argv.index('--options-json')+1])
        proof=plan['regional_candidate_reuse_authorization']
        self.assertEqual(settings['regional_candidate_reuse_authorization'],proof)
        payload=backend._one_click_payload(submitted,submitted['source'])
        self.assertEqual(payload['regional_candidate_reuse_authorization'],proof)
        self.assertIn('regional_candidate_reuse_authorization_sha256',route_contract(payload))
        self.assertEqual(workflow.task_image_budget({'task_plan':plan,'task_intent':plan['intent']},
            Namespace(),'regional_refinement'),0)

    def test_unknown_missing_raw_changed_assets_and_multi_photo_refused(self):
        for change in ({'state':'unknown'},{'state':'submitted'},{'state':'completed'},
                       {'model_submissions':0},{'model_submissions':True},{'provider_returned':False},{'prompt_id':None}):
            write(self.ledger,{**self.ledger_value,**change})
            with self.subTest(change=change),self.assertRaises(ValueError):task.plan_task(self.root,self.body)
        write(self.ledger,self.ledger_value)
        checked=task.plan_task(self.root,self.body)
        self.raw.write_bytes(b'changed')
        with self.assertRaises(ValueError):task.revalidate_regional_candidate_reuse(self.root,checked['plan'])
        Image.new('RGB',(1024,1024),'#222222').save(self.raw)
        for role in ('input_context','reliable_halo','condition_manifest','permission_mask','request','input_image'):
            path=Path(checked['plan']['regional_candidate_reuse_authorization']['assets'][role]['path']);raw=path.read_bytes()
            path.write_bytes(raw+b'\n')
            with self.subTest(role=role),self.assertRaises((ValueError,EOFError)):task.revalidate_regional_candidate_reuse(self.root,checked['plan'])
            path.write_bytes(raw)
        broken=copy.deepcopy(self.provenance)
        broken['provider_contract']['comfyui']['condition_inputs'].append(copy.deepcopy(broken['provider_contract']['comfyui']['condition_inputs'][0]))
        write(self.provenance_path,broken)
        write(self.ledger,{**self.ledger_value,'raw_provenance':asset(self.provenance_path)})
        with self.assertRaises(ValueError):task.plan_task(self.root,self.body)

    def test_source_reference_direction_core_and_permission_cannot_change(self):
        for mutate in (lambda b:b.update(character_name='Other role'),lambda b:b.update(character_notes='Other notes'),
                       lambda b:b.update(input_prompt='Changed base'),lambda b:b['creative_task'].update(brief='Changed story'),
                       lambda b:b['creative_task']['regional_refinement']['regions'][0].update(instruction='Changed local effect')):
            body=copy.deepcopy(self.body);mutate(body)
            with self.assertRaises(ValueError):task.plan_task(self.root,body)
        for extra in ({'final_repair_only':True},{'generate':True},{'provider':'openai-compatible'}):
            with self.assertRaises(ValueError):task.plan_task(self.root,{**self.body,**extra})
        for status in ('running','needs_visual_review','accepted'):
            write(self.manifest,{**self.parent,'status':status})
            with self.assertRaises(ValueError):task.plan_task(self.root,self.body)
        write(self.manifest,self.parent)
        for role in ('source','reference'):
            parent=copy.deepcopy(self.parent);parent[role]['sha256']='f'*64;write(self.manifest,parent)
            with self.assertRaises(ValueError):task.plan_task(self.root,self.body)
        write(self.manifest,self.parent)
        Image.new('L',(1024,1024),255).save(self.directory/'permission.png')
        conditions=json.loads((self.directory/'conditions.json').read_text('utf8'))
        conditions['permission_mask'].update(asset(self.directory/'permission.png'));write(self.directory/'conditions.json',conditions)
        with self.assertRaises(ValueError):task.plan_task(self.root,self.body)

    def test_explicit_cpu_core_can_move_only_inside_the_original_raw_context(self):
        body=copy.deepcopy(self.body)
        body['creative_task']['regional_refinement']['regions'][0]['core_box']=[0,0,15,19]
        before=snapshot(self.root)
        checked=task.plan_task(self.root,body)
        proof=checked['plan']['regional_candidate_reuse_authorization']
        self.assertEqual(proof['original_core_box'],[1,2,12,16])
        self.assertEqual(proof['requested_core_box'],[0,0,15,19])
        self.assertEqual(proof,task.revalidate_regional_candidate_reuse(self.root,checked['plan']))
        self.assertEqual(before,snapshot(self.root))
        self.assertEqual(sum(checked['plan']['image_budget'].values()),0)
        for box in ([0,0,26,20],[0,0,769,20],[0,0,True,20]):
            invalid=copy.deepcopy(body);invalid['creative_task']['regional_refinement']['regions'][0]['core_box']=box
            with self.assertRaises(ValueError):task.plan_task(self.root,invalid)
        bad=copy.deepcopy(checked['plan']);bad['regional_candidate_reuse_authorization']['requested_core_box']=[1,2,12,16]
        with self.assertRaises(ValueError):task.validate_task_plan(bad,bad['intent'])

    def test_reuse_intent_does_not_allow_new_or_implicit_generation(self):
        for mutate in (lambda i:i.update(mode='create_reference'),lambda i:i.update(mode='local_portrait'),
                       lambda i:i['regional_refinement'].update(maximum_submissions=1),
                       lambda i:i['regional_refinement'].update(resume_from_run_id='other'),
                       lambda i:i['regional_refinement'].update(reuse_generated_from_run_id='../escape'),
                       lambda i:i['regional_refinement'].update(regions=[])):
            intent=copy.deepcopy(self.body['creative_task']);mutate(intent)
            with self.assertRaises(ValueError):task.normalize_task_intent(intent)
        active=self.root/'outputs/jobs/active/job.json'
        write(active,{'status':'running','parent_run_id':'failed-known'})
        with self.assertRaises(ValueError):task.plan_task(self.root,self.body)


if __name__ == '__main__':
    unittest.main()
