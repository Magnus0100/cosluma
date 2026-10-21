"""Pure generation checks only used host/provider capabilities; no actual models."""
from contextlib import ExitStack
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

import creative_task
from current_model_registry import SCHEMA
from photo_editing import editing_scope
import runtime_preflight
import task_model_preflight
import scoped_production_smoke as fixture

ACTUAL_INSPECT=task_model_preflight.inspect_task_models


class ScopedModelPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='scoped-model-preflight-')
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ,{
            'COSPLAY_CONFIG_ROOT':str(self.root/'config'),'COSPLAY_PROVIDER_SETTINGS':str(self.root/'provider-settings.json'),
            'COSPLAY_ASSET_IDENTITY_CACHE':str(self.root/'identities'),'PREFLIGHT_FIXTURE_KEY':'unused-fixture-key',
            'COSPLAY_VISION_PROVIDER':'openai-compatible','COSPLAY_VISION_API_BASE':'https://example.invalid/v1',
            'COSPLAY_VISION_API_KEY_ENV':'PREFLIGHT_FIXTURE_KEY'}))
        self.stack.enter_context(patch('urllib.request.urlopen',side_effect=AssertionError('Preflight sent network')))
        self.stack.enter_context(patch('runtime_preflight._probe_runtime',side_effect=AssertionError('Unused model runtime probe')))
        for method in ('_sapiens_and_matting','_dinov3','_sam31','_background_superresolution'):
            self.stack.enter_context(patch('task_model_preflight.'+method,side_effect=AssertionError('Unused native model inspector')))
        self.stack.enter_context(patch('model_setup.beauty_status',side_effect=AssertionError('Unused Beauty probe')))
        self.source=self.root/'source.png';Image.new('RGB',(64,96),'gray').save(self.source)
        self.scope=editing_scope()

    def request(self,**extra):
        return {'source':str(self.source),'provider':'openai-compatible','api_base':'https://example.invalid/v1',
            'api_key_env':'PREFLIGHT_FIXTURE_KEY','model':'gpt-image-2.5-sunburst',
            'editing_request':{'work_type':'partial_composite'},
            'creative_task':{'schema_version':creative_task.INTENT_SCHEMA,'mode':'create_reference',
                'services':{'text_vision':True,'visual_review':False}},**extra}

    def test_no_registry_and_disabled_unused_registry_are_not_native_readiness_claims(self):
        with patch.dict(os.environ,{},clear=False):
            os.environ.pop('COSPLAY_MODEL_REGISTRY',None)
            result=creative_task.plan_task(self.root,self.request())
        plan=result['plan'];self.assertTrue(plan['ready'],plan['blocking_reasons'])
        capability=plan['capabilities']
        self.assertFalse(capability['native_models_required'])
        self.assertFalse(capability['configuration_verified'])
        self.assertEqual(capability['model_preflight']['required_capabilities'],[])
        self.assertEqual(capability['model_preflight']['model_inferences'],0)
        self.assertFalse(capability['model_preflight']['inference_verified'])
        self.assertIsNone(capability['registry_identity']['path'])
        registry=self.root/'disabled-native.json'
        registry.write_text(json.dumps({'schema':SCHEMA,'ready':False,'python':str(self.root/'missing-python.exe'),
            'scene_model_root':str(self.root/'missing-sam'),'dinov3_model_root':str(self.root/'missing-dino')}),'utf8')
        plan=creative_task.plan_task(self.root,self.request(model_registry=str(registry)))['plan']
        self.assertTrue(plan['ready'],plan['blocking_reasons'])
        self.assertFalse(plan['capabilities']['configuration_verified'])
        self.assertTrue(plan['capabilities']['native_registry_metadata_valid'])
        self.assertEqual(plan['capabilities']['models'],[])
        self.assertFalse(plan['capabilities']['model_preflight']['weight_content_verified'])

    def test_explicit_invalid_registries_remain_blocked(self):
        cases={'missing':None,'json':'{bad','schema':json.dumps({'schema':'old','ready':False}),
            'unsafe':json.dumps({'schema':SCHEMA,'ready':False,'python':'https://invalid/python.exe'}),
            'drive-relative':json.dumps({'schema':SCHEMA,'ready':False,'python':'C:relative/python.exe'})}
        for name,text in cases.items():
            with self.subTest(name=name):
                path=self.root/(name+'.json')
                if text is not None:path.write_text(text,'utf8')
                plan=creative_task.plan_task(self.root,self.request(model_registry=str(path)))['plan']
                self.assertFalse(plan['ready'])
                self.assertTrue(plan['blocking_reasons'])
                self.assertEqual(plan['capabilities']['model_preflight']['model_inferences'],0)

    def test_ai_retouch_needs_no_unused_native_models_but_explicit_local_keeps_its_contract(self):
        request=self.request(editing_request={'work_type':'retouch'})
        plan=creative_task.plan_task(self.root,request)['plan']
        self.assertTrue(plan['ready'],plan['blocking_reasons'])
        self.assertEqual(plan['capabilities']['model_preflight']['required_capabilities'],[])
        self.assertEqual(plan['mode'],'create_reference')
        self.assertTrue(plan['services']['text_vision'])
        self.assertEqual(task_model_preflight.required_capabilities('local_portrait',editing_scope=editing_scope(local=True)),
            frozenset({'source_silhouette'}))

    def test_real_qwen_weight_and_workflow_health_still_gate_generation(self):
        import start_qwen_image
        deploy=self.root/'qwen-deployment';deploy.mkdir()
        catalog=self.root/'qwen-catalog.json'
        rows=[]
        for role in ('official_transformer','text_encoder','vae'):
            path=deploy/(role+'.bin');path.write_bytes(b'tiny fixture, never loaded')
            rows.append({'role':role,'destination':path.name,'size':path.stat().st_size})
        catalog.write_text(json.dumps({'deployment_root':str(deploy),'models':rows,
            'provider_roles':{'qwen_image_2_1':[row['role'] for row in rows]}}),'utf8')
        with patch.object(start_qwen_image,'resource_path',return_value=catalog), \
             patch.dict(os.environ,{'COSPLAY_QWEN_DEPLOY_ROOT':str(deploy)}):
            request=self.request(provider='qwen_image_2_1',api_base='http://127.0.0.1:8188',model=None)
            healthy=creative_task.plan_task(self.root,request)['plan']
            self.assertTrue(healthy['ready'],healthy['blocking_reasons'])
            self.assertTrue(healthy['services']['generation_configured'])
            (deploy/'vae.bin').unlink()
            blocked=creative_task.plan_task(self.root,request)['plan']
            self.assertFalse(blocked['ready'])
            self.assertTrue(any('生图服务未就绪' in reason for reason in blocked['blocking_reasons']))
            missing_workflow=creative_task.plan_task(self.root,{**request,'model':str(self.root/'absent-workflow.json')})['plan']
            self.assertFalse(missing_workflow['ready'])

    def test_actual_scoped_start_without_native_installation_uses_no_native_runtime(self):
        fx=fixture.ScopedProductionTests(methodName='runTest');fx.setUp();self.addCleanup(fx.doCleanups)
        fx.registry.write_text(json.dumps({'schema':SCHEMA,'ready':False,
            'python':str(fx.root/'absent-native-python.exe'),'scene_model_root':str(fx.root/'absent-sam')}),'utf8')
        # Replace this shared fixture's model-installation mock with the real
        # route-aware inspector; only actual image/vision transports stay fake.
        with patch('task_model_preflight.inspect_task_models',side_effect=ACTUAL_INSPECT), \
             patch('runtime_preflight._probe_runtime',side_effect=AssertionError('Unused native worker')):
            checked,run=fx.start(fx.request(fx.sources[0],run_id='without-native-analysis'))
        self.assertTrue(checked['plan']['ready'])
        self.assertEqual(run['options']['runtime_preflight']['route'],'scoped_generated_appearance')
        self.assertIsNone(run['options']['runtime_preflight']['model_python'])
        self.assertEqual(run['options']['runtime_preflight']['required_model_imports'],[])
        self.assertEqual(run['result_publication']['state'],'committed')
        self.assertEqual(len(fx.image_requests),1)

    def test_actual_iteration_plan_and_start_ignore_disabled_native_installation(self):
        import photo_edit_iteration_smoke as iteration_fixture
        fx=iteration_fixture.EditingIterationTests(methodName='runTest');fx.setUp();self.addCleanup(fx.doCleanups)
        fx.fx.registry.write_text(json.dumps({'schema':SCHEMA,'ready':False,
            'python':str(fx.root/'missing-native.exe'),'vitmatte_model_root':str(fx.root/'missing-vitmatte')}),'utf8')
        with patch('task_model_preflight.inspect_task_models',side_effect=ACTUAL_INSPECT), \
             patch('runtime_preflight._probe_runtime',side_effect=AssertionError('Unused native runtime')):
            checked=fx.plan()
            self.assertFalse(checked['plan']['capabilities']['native_models_required'])
            self.assertEqual(checked['plan']['capabilities']['model_preflight']['required_capabilities'],[])
            response=fx.start(checked);fx.assert_started(response)
        self.assertEqual(response['run']['result_publication']['state'],'committed')
        self.assertEqual(fx.parent_path.read_bytes(),fx.parent_bytes)

    def test_scoped_cpu_dependencies_still_block_and_legacy_routes_still_need_models(self):
        with patch('runtime_preflight._missing_imports',return_value=['tifffile']):
            plan=creative_task.plan_task(self.root,self.request())['plan']
            self.assertFalse(plan['ready'])
            self.assertTrue(any('tifffile' in row for row in plan['blocking_reasons']))
        self.assertEqual(task_model_preflight.required_capabilities('create_reference',editing_scope=self.scope),frozenset())
        self.assertEqual(task_model_preflight.required_capabilities('existing_reference',editing_scope=self.scope),frozenset())
        self.assertIn('source_geometry',task_model_preflight.required_capabilities('existing_reference'))
        self.assertIn('source_silhouette',task_model_preflight.required_capabilities('local_portrait',
            editing_scope=editing_scope({'work_type':'retouch'},local=True)))


if __name__=='__main__':unittest.main()
