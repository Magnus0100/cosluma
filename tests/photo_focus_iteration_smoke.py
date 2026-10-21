"""Checked continuation integrates a reviewed working edit with no new image call."""
from pathlib import Path
import json
import unittest

import numpy as np
from PIL import Image

from asset_identity import sha256_file
from photo_appearance import validate_asset,load_asset
from photo_edit_tools import _close_loaded
import photo_edit_iteration_smoke as iteration_fixture


class FocusIterationTests(unittest.TestCase):
    def setUp(self):
        self.fx=iteration_fixture.EditingIterationTests(methodName='runTest')
        self.fx.setUp();self.addCleanup(self.fx.doCleanups)
        self.root=self.fx.root
        self.feedback='Integrate a restrained generated local material with the surrounding photographic focus.'
        self.fx.repair_outcome='better'
        fixed=self.fx.start(self.fx.plan(self.fx.request(run_id='focus-known-good-parent')))
        self.fx.assert_started(fixed)
        known=fixed['run']
        self.assertTrue(known['photo_edit_session']['automated_target_met'])
        self.fx.repair_outcome='worse'
        actual_vision=self.fx.vision
        def useful_working(source,output,prompt,**kwargs):
            payload,provenance=actual_vision(source,output,prompt,**kwargs)
            if Path(source).name=='review-board.png' and 'Select the single most useful available tool' not in prompt:
                payload.update(comparison='better',primary_issue='The useful generated material is too sharp for its local focus plane.')
            return payload,provenance
        self.fx.fx.vision.side_effect=useful_working
        body=self.fx.request(parent=known,run_id='focus-working-parent')
        body['editing_iteration']['user_feedback']=self.feedback
        response=self.fx.start(self.fx.plan(body));self.fx.assert_started(response)
        self.parent=response['run']
        self.parent_path=self.root/'outputs/runs'/self.parent['run_id']/'manifest.json'
        self.parent_bytes=self.parent_path.read_bytes()
        self.parent_sha=sha256_file(self.parent['artifacts']['auto_draft'])
        self.old_state=json.loads(Path(self.parent['photo_edit_session']['path']).read_text('utf-8'))
        self.working=self.old_state['candidates'][-1]
        self.assertNotEqual(self.working['candidate_id'],self.old_state['best_candidate_id'])
        self.assertEqual(self.working['candidate_id'],self.old_state['working_candidate_id'])
        self.assertEqual(self.working['appearance_kind'],'mixed_regional')
        self.assertEqual(self.working['review']['creative_benefit']['status'],'pass')
        self.initial_images=len(self.fx.fx.image_requests);self.initial_vision=len(self.fx.fx.vision_requests)
        self.fx.repair_outcome='better'
        original=self.fx.vision
        def observe(source,output,prompt,**kwargs):
            if 'Select the single most useful available tool' not in prompt:
                return original(source,output,prompt,**kwargs)
            self.fx.fx.vision_requests.append({'source':str(source),'sha256':sha256_file(source),'prompt':prompt})
            mapping=json.loads((Path(source).parent/'review-board.json').read_text('utf-8'))
            self.assertEqual([row['role'] for row in mapping['images']],['source','best','working'])
            self.assertEqual(mapping['images'][2]['sha256'],self.working['image']['sha256'])
            return {'action':'tool','tool':'integrate_generated_focus','operations':['detail_repair'],
                'instruction':'Match the generated region to the surrounding focus plane while preserving the original scene.',
                'reason':'The reviewed working edit has a visible focus discontinuity.',
                'parameters':{'focus_sigma':.012}}, {
                'source':{'path':str(source),'sha256':sha256_file(source)},
                'network_used':True,'generation_used':False,'model':'fixture-focus-decision'}
        self.fx.fx.vision.side_effect=observe
        self.fx.fx.generation.side_effect=AssertionError('A photographic integration must not submit another image request')

    def request(self,**kwargs):
        body=self.fx.request(**kwargs);body['editing_iteration']['user_feedback']=self.feedback
        return body

    def test_public_plan_start_preserves_best_and_integrates_real_working_with_zero_images(self):
        body=self.request(parent=self.parent,run_id='focus-child')
        body['editing_iteration']['maximum_image_submissions']=0
        checked=self.fx.plan(body)
        self.assertEqual(sum(checked['plan']['image_budget'].values()),0)
        self.assertFalse(checked['plan']['services']['image_generation'])
        proof=checked['plan']['editing_refinement_authorization']['working_candidate']
        self.assertEqual(proof['image_sha256'],self.working['image']['sha256'])
        response=self.fx.start(checked);self.fx.assert_started(response);run=response['run']
        self.assertEqual(run['result_publication']['state'],'committed')
        self.assertEqual(len(self.fx.fx.image_requests),self.initial_images)
        self.assertEqual(len(self.fx.fx.vision_requests)-self.initial_vision,2)
        self.assertEqual(run['editing_iteration_result']['new_requests']['image_submissions'],0)
        self.assertTrue(run['photo_edit_session']['automated_target_met'])
        self.assertFalse(run['photo_edit_session']['human_approved'])
        self.assertEqual(run['reference']['provider'],'local_photographic_processing')
        self.assertIsNone(run['reference']['generation_id'])
        self.assertEqual(run['reference']['processing']['image_generation_submissions'],0)
        state=json.loads(Path(run['photo_edit_session']['path']).read_text('utf-8'))
        selected=next(row for row in state['candidates'] if row['candidate_id']==state['best_candidate_id'])
        self.assertEqual(sha256_file(run['artifacts']['auto_draft']),selected['image']['sha256'])
        binding={'source_sha256':self.parent['source']['sha256'],'scope':self.parent['task_plan']['editing_scope'],
            'design_id':self.parent['generation_design']['observed_design']['design_id']}
        receipt=validate_asset(self.working['appearance_asset'],**binding)
        core=receipt['generation']['core_box'];x0,y0,x1,y1=core
        with Image.open(self.parent['artifacts']['auto_draft']) as before,Image.open(run['artifacts']['auto_draft']) as after:
            changed=np.any(np.asarray(before)!=np.asarray(after),axis=2)
        outside=changed.copy();outside[y0:y1,x0:x1]=False
        self.assertFalse(outside.any())
        parent_float=load_asset(self.parent['options']['photo_appearance'],**binding)
        child_float=load_asset(run['options']['photo_appearance'],**binding)
        mask=np.ones(parent_float['master'].shape[:2],bool);mask[y0:y1,x0:x1]=False
        try:np.testing.assert_array_equal(parent_float['master'][mask],child_float['master'][mask])
        finally:_close_loaded(parent_float,child_float)
        self.assertEqual(self.parent_path.read_bytes(),self.parent_bytes)
        self.assertEqual(sha256_file(self.parent['artifacts']['auto_draft']),self.parent_sha)

    def test_zero_image_budget_without_working_advertises_development_but_not_focus(self):
        body=self.request(run_id='focus-no-working')
        body['editing_iteration']['maximum_image_submissions']=0
        counts=len(self.fx.fx.image_requests),len(self.fx.fx.vision_requests)
        def stop(source,output,prompt,**kwargs):
            current=json.JSONDecoder().raw_decode(prompt.split('CONTEXT\n',1)[1])[0]
            self.assertNotIn('integrate_generated_focus',current['tools'])
            self.assertIn('integrate_generated_focus',current['unavailable_tools'])
            self.assertIn('develop_photograph',current['tools'])
            self.fx.fx.vision_requests.append({'source':str(source),'sha256':sha256_file(source),'prompt':prompt})
            return {'action':'stop','reason':'No useful zero-image adjustment is needed for this synthetic photograph.'}, {
                'source':{'path':str(source),'sha256':sha256_file(source)},'network_used':True,
                'generation_used':False,'model':'fixture-stop-without-working'}
        self.fx.fx.vision.side_effect=stop
        checked=self.fx.plan(body)
        self.assertIsNone(checked['plan']['editing_refinement_authorization']['working_candidate'])
        self.assertEqual(counts,(len(self.fx.fx.image_requests),len(self.fx.fx.vision_requests)))
        response=self.fx.start(checked);self.fx.assert_started(response)
        self.assertEqual(len(self.fx.fx.image_requests),counts[0])
        self.assertEqual(len(self.fx.fx.vision_requests),counts[1]+1)
        self.assertEqual(response['run']['editing_iteration_result']['new_requests']['image_submissions'],0)

    def test_invalid_explicit_image_budgets_are_rejected(self):
        for value in (-1,2,True,1.5,'0'):
            body=self.request(parent=self.parent,run_id='focus-invalid-budget')
            body['editing_iteration']['maximum_image_submissions']=value
            self.fx.rejected_plan(body)

    def test_changed_working_pixels_after_plan_are_rejected_before_any_model_request(self):
        checked=self.fx.plan(self.request(parent=self.parent,run_id='focus-stale'))
        working=Path(self.working['image']['path'])
        with Image.open(working) as image:
            changed=image.copy()
        changed.putpixel((0,0),(255,0,0));changed.save(working)
        counts=len(self.fx.fx.image_requests),len(self.fx.fx.vision_requests)
        with self.assertRaises(ValueError):self.fx.start(checked)
        self.assertEqual(counts,(len(self.fx.fx.image_requests),len(self.fx.fx.vision_requests)))
        self.assertFalse((self.root/'outputs/runs/focus-stale/manifest.json').exists())
        self.assertEqual(self.parent_path.read_bytes(),self.parent_bytes)


if __name__=='__main__':unittest.main()
