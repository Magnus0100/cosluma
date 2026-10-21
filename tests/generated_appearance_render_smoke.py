"""Synthetic pixel integration for scoped save, frozen replay and export.

Constructed observation metadata is a CPU fixture, not evidence of a model call
or artistic acceptance. No provider is invoked by these checks.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw
import tifffile

from asset_identity import sha256_file
from current_model_registry import atomic_json
from generated_appearance_render import render_candidate
from photo_appearance import create_source_baseline,freeze_candidate,load_asset
from raw_support import open_linear
from photo_editing import editing_scope, validate_observed_design, compile_edit_prompt


class ScopedRenderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='cosplay_generated_render_')
        self.root = Path(self.temporary.name)
        self.cache = patch.dict(os.environ, {'COSPLAY_ASSET_IDENTITY_CACHE':str(self.root/'identity-cache')})
        self.cache.start()
        self.args = argparse.Namespace(cancel_request_path=None)

    def tearDown(self):
        self.cache.stop()
        self.temporary.cleanup()

    def fixture(self, name='portrait', size=(72,96), work='partial_composite'):
        inputs = self.root/'inputs'/name
        inputs.mkdir(parents=True)
        source_path, candidate_path = inputs/'source.png', inputs/'edit.png'
        source = Image.new('RGB',size,(75,80,90))
        draw = ImageDraw.Draw(source)
        draw.ellipse((size[0]//4,size[1]//4,size[0]*3//4,size[1]*3//4),fill=(150,110,95))
        source.save(source_path)
        edited = Image.new('RGB',size,(40,60,90))
        draw = ImageDraw.Draw(edited)
        draw.ellipse((size[0]//4,size[1]//4,size[0]*3//4,size[1]*3//4),fill=(195,170,150))
        edited.save(candidate_path)
        scope = editing_scope({'work_type':work})
        design = validate_observed_design({'summary':'A coherent photographic portrait',
            'observation':{'setting':'indoor','description':'An indoor portrait with existing walls'},
            'changes':[{'operation':'relight','target':'portrait and room','instruction':'Soften directional key light on the face and room',
                        'purpose':'Unify portrait illumination'},
                       {'operation':'scene_edit','target':'wall','instruction':'Reduce wall distractions while retaining its structure',
                        'purpose':'Make the subject prominent'}],
            'lighting':'A coherent soft directional key across person and room',
            'keep':['Identity, pose and main props'], 'uncertain':[]},scope)
        source = {'path':str(source_path),'sha256':sha256_file(source_path)}
        compiled = compile_edit_prompt(design,scope,'qwen_image_2_1')
        direction = {'schema':'cosluma_photo_art_direction.v1','status':'ready',
            'binding':{'source':source,'scope_id':scope['scope_id']},'design':design,
            'compiled_prompts':{family:compile_edit_prompt(design,scope,family) for family in ('gpt_image','qwen_image_2_1')},
            'provenance':{'source':source,'network_used':True,'generation_used':False,'model':'synthetic-fixture'},
            'model_requests_attempted':1,'image_submissions':0}
        generation_design = {'schema_version':'cosluma_scoped_generation_design.v1','status':'ready',
            'source_sha256':source['sha256'],'scope_id':scope['scope_id'],
            'observed_design':design,'art_direction':direction,'compiled_prompt':compiled,
            'effective_prompt':compiled['prompt'],'text_vision_submissions':1}
        baseline=create_source_baseline(inputs/'baseline',source_path=source_path,source_sha256=source['sha256'],
            scope=scope,design_id=design['design_id'])
        provenance=inputs/'edit.provenance.json'
        provenance.write_text(json.dumps({'status':'reference_candidate','source':source,'asset':{
            'path':str(candidate_path),'sha256':sha256_file(candidate_path),'oriented_size':list(size)},
            'provider':'fixture','model':'synthetic-boundary'}),encoding='utf8')
        values,_=open_linear(candidate_path)
        current=freeze_candidate(inputs/'current',kind='generated_full',master=values,parent=baseline,
            source_sha256=source['sha256'],scope=scope,design_id=design['design_id'],generation={
                'image':{'path':str(candidate_path),'sha256':sha256_file(candidate_path)},
                'provider_receipt':{'path':str(provenance),'sha256':sha256_file(provenance)},
                'context_box':None,'core_box':None})
        preview=load_asset(current,source_sha256=source['sha256'],scope=scope,design_id=design['design_id'])['preview_path']
        return {'kind':'single_input_run','run_id':name,'sample':name,'revision':1,
            'status':'candidate_ready','source':source,
            'reference':{'path':str(preview),'sha256':sha256_file(preview)},
            'initial_source_appearance':baseline,
            'generation_design':generation_design,
            'task_plan':{'ready':True,'inputs':{'source':source},'editing_scope':scope,
                         'editing_initialization':{'schema':'cosluma_source_native_initialization.v1','kind':'source_native'},
                         'preparation_policy':'replay_frozen'},
            'options':{'photo_appearance':current,'preparation_policy':'replay_frozen'}}

    def save(self, run, result):
        run.update(auto_draft=result, render=result['metadata'], review=result['review'],
                   artifacts=result['outputs'],status='needs_visual_review')
        atomic_json(self.root/'outputs/runs'/run['run_id']/'manifest.json',run)

    def test_freeze_native_save_and_replay_with_two_different_images(self):
        from export_artifact import _master_header
        for name,size,work in (('portrait',(72,96),'partial_composite'),
                               ('wide',(120,80),'full_composite')):
            with self.subTest(name=name):
                run = self.fixture(name,size,work)
                result = render_candidate(self.root,run,self.args)
                self.save(run,result)
                frozen = copy.deepcopy(run['options']['photo_appearance'])
                receipt_path = Path(frozen['path'])/'receipt.json'
                receipt_before = receipt_path.read_bytes()
                paths = result['outputs']
                with Image.open(paths['auto_draft']) as actual, Image.open(run['reference']['path']) as expected:
                    np.testing.assert_array_equal(np.asarray(actual),np.asarray(expected))
                    self.assertTrue(actual.info.get('icc_profile'))
                    self.assertEqual(actual.size,size)
                _master_header(Path(paths['render_master']),floating=False,size=size)
                _master_header(Path(paths['render_scene_master']),floating=True,size=size)
                self.assertEqual(result['metadata']['alpha']['role'],'cumulative_generated_coverage')
                self.assertFalse(result['metadata']['alpha']['is_source_foreground'])
                self.assertEqual(result['review']['status'],'needs_review')
                self.assertIsNone(result['review']['reviewer'])
                # A reopened task cannot silently call preparation again.
                reopened = json.loads((self.root/'outputs/runs'/name/'manifest.json').read_text('utf8'))
                with patch('raw_support.open_linear',side_effect=AssertionError('Source must not be decoded on render')):
                    replay = render_candidate(self.root,reopened,self.args)
                np.testing.assert_array_equal(tifffile.imread(paths['render_scene_master']),
                                             tifffile.imread(replay['outputs']['render_scene_master']))
                self.assertEqual(reopened['options']['photo_appearance'],frozen)
                self.assertEqual(receipt_before,receipt_path.read_bytes())
                self.assertEqual(sha256_file(paths['auto_draft']),sha256_file(replay['outputs']['auto_draft']))

    def test_replay_rejects_missing_asset_without_repreparation(self):
        run = self.fixture()
        result = render_candidate(self.root,run,self.args)
        self.save(run,result)
        run['options'].pop('photo_appearance')
        with patch('raw_support.open_linear',side_effect=AssertionError('Source must not be decoded on render')):
            with self.assertRaisesRegex(ValueError,'missing'):
                render_candidate(self.root,run,self.args)

    def test_source_native_sub_rgb8_float_is_saved_without_raw_or_preview_reconstruction(self):
        run=self.fixture('native',size=(72,96))
        values=np.full((96,72,3),.125001,np.float32)
        values[...,0]+=np.arange(72,dtype=np.float32)[None,:]*.000001
        scope=run['task_plan']['editing_scope'];design=run['generation_design']['observed_design']['design_id']
        with patch('raw_support.open_linear',return_value=(values.copy(),{
                'source_precision':'raw16_linear','decode_version':'oriented-srgb-icc-linearraw-v3'})):
            baseline=create_source_baseline(self.root/'native-source',source_path=run['source']['path'],
                source_sha256=run['source']['sha256'],scope=scope,design_id=design)
        loaded=load_asset(baseline,source_sha256=run['source']['sha256'],scope=scope,design_id=design)
        run['initial_source_appearance']=run['options']['photo_appearance']=baseline
        run['reference']={'path':str(loaded['preview_path']),'sha256':sha256_file(loaded['preview_path'])}
        with patch('raw_support.open_linear',side_effect=AssertionError('No source redecode')):
            result=render_candidate(self.root,run,self.args)
        np.testing.assert_array_equal(tifffile.imread(result['outputs']['render_scene_master']),values)
        self.assertEqual(sha256_file(result['outputs']['auto_draft']),run['reference']['sha256'])
        self.assertEqual(result['metadata']['render_precision']['source_precision'],'raw16_linear')
        for key,digest in result['metadata']['saved_artifact_sha256'].items():
            self.assertEqual(sha256_file(result['outputs'][key]),digest)

    def test_frozen_typed_render_without_original_uses_only_owned_source_baseline(self):
        run=self.fixture('portable')
        first=render_candidate(self.root,run,self.args)
        Path(run['source']['path']).unlink()
        with patch('raw_support.open_linear',side_effect=AssertionError('Frozen replay must not decode')):
            second=render_candidate(self.root,run,self.args)
        self.assertEqual(sha256_file(first['outputs']['auto_draft']),sha256_file(second['outputs']['auto_draft']))
        np.testing.assert_array_equal(tifffile.imread(first['outputs']['render_scene_master']),
            tifffile.imread(second['outputs']['render_scene_master']))


    def test_cancellation_before_freeze_does_not_create_a_candidate(self):
        from single_source_workflow import WorkflowCancelled
        run=self.fixture()
        marker=self.root/'cancel.request'
        marker.write_text('cancel',encoding='utf8')
        self.args.cancel_request_path=str(marker)
        with self.assertRaises(WorkflowCancelled):
            render_candidate(self.root,run,self.args)
        self.assertIn('photo_appearance',run['options'])
        self.assertFalse((self.root/'outputs/runs'/run['run_id']).exists())

    def test_source_candidate_prompt_scope_and_frozen_design_tampering_fail(self):
        run = self.fixture()
        render_candidate(self.root,run,self.args)
        variants=[]
        item=copy.deepcopy(run);item['source']['sha256']='a'*64;variants.append(item)
        item=copy.deepcopy(run);item['reference']['sha256']='b'*64;variants.append(item)
        item=copy.deepcopy(run);item['generation_design']['effective_prompt']='another edit';variants.append(item)
        item=copy.deepcopy(run);item['options']['photo_appearance']['receipt_sha256']='c'*64;variants.append(item)
        item=copy.deepcopy(run);item['task_plan']['editing_scope']=editing_scope({'work_type':'retouch'});variants.append(item)
        item=copy.deepcopy(run);item['options']['manual_layers']={'strokes':[{'x':.5}]};variants.append(item)
        item=copy.deepcopy(run);item['options']['appearance_revision']={'params':{'subject':{'exposure':1}}};variants.append(item)
        for index,item in enumerate(variants):
            with self.subTest(index=index), self.assertRaises(ValueError):
                render_candidate(self.root,item,self.args)

    def test_save_failure_preserves_old_artifacts_review_and_frozen_asset(self):
        run = self.fixture()
        result = render_candidate(self.root,run,self.args)
        self.save(run,result)
        artifacts_before={key:sha256_file(path) for key,path in result['outputs'].items() if Path(path).is_file()}
        review_before=Path(result['review_path']).read_bytes()
        asset_before=copy.deepcopy(run['options']['photo_appearance'])
        with patch('raw_support.save_scene_master',side_effect=OSError('disk unavailable')):
            with self.assertRaises(OSError):
                render_candidate(self.root,run,self.args)
        self.assertEqual(review_before,Path(result['review_path']).read_bytes())
        self.assertEqual(asset_before,run['options']['photo_appearance'])
        for key,digest in artifacts_before.items():
            self.assertEqual(sha256_file(result['outputs'][key]),digest)
        self.assertFalse(list((self.root/'outputs/runs'/run['run_id']/'render').rglob('*.partial')))

    def test_saved_export_copies_actual_generated_masters(self):
        from export_artifact import export_source_run
        run = self.fixture()
        result = render_candidate(self.root,run,self.args)
        self.save(run,result)
        for format,key in (('PNG','auto_draft'),('TIFF16','render_master'),('TIFF32_LINEAR','render_scene_master')):
            with self.subTest(format=format):
                exported=export_source_run(self.root,run['run_id'],directory=str(self.root/'exports'),format=format)
                self.assertTrue(exported['ok'])
                self.assertEqual(sha256_file(exported['path']),sha256_file(result['outputs'][key]))
                self.assertFalse(exported['metadata']['approved'])
                self.assertFalse(exported['metadata']['export_rerendered'])


if __name__ == '__main__':
    unittest.main()
