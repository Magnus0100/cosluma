"""Real checked photography-tool sessions; only external model transports fake.

Synthetic review scores verify bookkeeping and objective guards, never artistic
acceptance. There is no GPU/API request, fake render or alternate production path.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from asset_identity import sha256_file
from creative_task_smoke import snapshot
import photo_edit_session as session
from photo_development import PARAMETERS
import scoped_production_smoke as production_fixture

VALID={'adjustments':[{'region':[0,0,1,1],'exposure_ev':.5,'warmth_ev':.1,'tint_ev':-.1,'feather':.08}]}


class ParameterTests(unittest.TestCase):
    def test_exact_typed_parameters_accept_global_local_and_ordered_overrides(self):
        self.assertEqual(session.validate_parameter_schema(PARAMETERS),PARAMETERS)
        self.assertEqual(session.validate_parameters(VALID,PARAMETERS),VALID)
        local=copy.deepcopy(VALID);local['adjustments'][0]['region']=[.1,.2,.8,.9]
        local['adjustments']*=4
        self.assertEqual(session.validate_parameters(local,PARAMETERS),local)

    def test_untrusted_parameters_cannot_expand_the_checked_photographic_operation(self):
        cases=[{'adjustments':[]},{'adjustments':VALID['adjustments']*5},
            {'adjustments':VALID['adjustments'],'region':[0,0,1,1]}]
        for field,value in [('exposure_ev',True),('exposure_ev',float('nan')),('exposure_ev',2.1),
            ('warmth_ev',-1.1),('tint_ev',float('inf')),('feather',.51),('feather',False),
            ('region',[0,0,1,True]),('region',[.5,0,.2,1]),('region',[-.1,0,1,1])]:
            row=copy.deepcopy(VALID);row['adjustments'][0][field]=value;cases.append(row)
        extra=copy.deepcopy(VALID);extra['adjustments'][0]['skin_mask']='automatic';cases.append(extra)
        for case in cases:
            with self.subTest(case=case):
                with self.assertRaises(ValueError):session.validate_parameters(case,PARAMETERS)

    def test_coordinator_schema_is_exact_not_a_generic_executor(self):
        for change in ('max_items','exposure_max','unit','boolean_count'):
            bad=copy.deepcopy(PARAMETERS)
            if change=='max_items':bad['adjustments']['max_items']=8
            elif change=='exposure_max':bad['adjustments']['fields']['exposure_ev']['maximum']=20.
            elif change=='boolean_count':bad['adjustments']['min_items']=True
            else:bad['adjustments']['fields']['feather']['unit']='pixels'
            with self.assertRaises(ValueError):session.validate_parameter_schema(bad)


class DevelopmentSessionTests(unittest.TestCase):
    def setUp(self):
        self.fx=production_fixture.ScopedProductionTests(methodName='runTest');self.fx.setUp();self.addCleanup(self.fx.doCleanups)
        self.root=self.fx.root;self.decisions=0;self.params=copy.deepcopy(VALID)
        self.initial_tool='develop_photograph';self.followup='develop_photograph'
        self.fx.vision.side_effect=self.vision
        self.fx.generation.side_effect=AssertionError('Photographic development submitted an image request')

    def vision(self,source,output,prompt,**kwargs):
        source=Path(source)
        if source.name!='review-board.png':return self.fx.vision_request(source,output,prompt,**kwargs)
        self.fx.vision_requests.append({'source':str(source),'sha256':sha256_file(source),'prompt':prompt})
        provenance={'source':{'path':str(source),'sha256':sha256_file(source)},'network_used':True,
            'generation_used':False,'model':'actual-synthetic-development-vision',
            'input_profile':kwargs.get('input_profile','standard')}
        mapping=json.loads((source.parent/'review-board.json').read_text(encoding='utf8'))
        if 'Select the single most useful available tool' in prompt:
            self.decisions+=1
            tool=self.initial_tool if self.decisions==1 else self.followup
            if tool=='stop':return {'action':'stop','reason':'The bounded synthetic model elects to stop without claiming the composite finished.'},provenance
            decision={'action':'tool','tool':tool,'operations':['relight'],
                'instruction':'Balance the supported exposure and colour of the current photograph.',
                'reason':'The supplied photographs show a synthetic exposure and colour difference.'}
            if tool=='develop_photograph':decision['parameters']=copy.deepcopy(self.params)
            elif tool=='edit_region':decision.update(region=[.2,.2,.8,.8],operations=['scene_edit'])
            return decision,provenance
        return {'credibility':{'status':'pass','findings':['Synthetic fixture credibility, not real artistic approval.']},
            'creative_benefit':{'status':'pass','findings':['Deliberate synthetic pass verifies that operation eligibility is independent.']},
            'comparison':'better' if len(mapping['images'])==3 else 'first_candidate','primary_issue':'none'},provenance

    def request(self,*,source=None,work='partial_composite',review=True,k=0,run_id='development-initial'):
        body=self.fx.request(source or self.fx.sources[0],run_id=run_id,review=review,
            editing_autofinish={'maximum_improvements':k if review else 0})
        body['editing_request'].update(work_type=work,style='natural' if work=='retouch' else 'expressive')
        if work=='retouch':body['creative_task']['constraints']={'preserve_background':True,'no_effects':True}
        elif work=='full_composite':body['editing_request'].update(target_setting='indoor',scene_target='A restrained photographed indoor setting')
        return body

    def start(self,body):
        before=snapshot(self.root);images=len(self.fx.image_requests);vision=len(self.fx.vision_requests)
        checked=self.fx.backend.task_plan(body)
        self.assertTrue(checked['plan']['ready'],checked['plan'].get('blocking_reasons'))
        self.assertEqual(snapshot(self.root),before)
        self.assertEqual((len(self.fx.image_requests),len(self.fx.vision_requests)),(images,vision))
        result=self.fx.backend.start_task({**checked['request'],'plan_id':checked['plan']['plan_id'],'async':False})
        self.assertTrue(result.get('ok'),result.get('stderr'))
        return checked,result['run']

    def state(self,run):return session.snapshot(Path(run['photo_edit_session']['path']).parent)

    def assert_photographic_provenance(self,run):
        self.assertEqual(run['reference']['provider'],'local_photographic_development')
        self.assertEqual(run['reference']['model'],'develop_photograph')
        self.assertIsNone(run['reference'].get('generation_id'))
        processing=run['reference']['processing'];receipt=processing['operation_receipt']
        self.assertEqual(sha256_file(receipt['path']),receipt['sha256'])
        self.assertEqual(processing['image_generation_submissions'],0)
        self.assertFalse(processing['operator']['physical_relighting'])
        self.assertNotIn('focus_operator',processing);self.assertNotIn('original_generated_image',processing)
        self.assertEqual(processing['operator']['parameters'],self.params)
        self.assertEqual(run['options']['photo_appearance']['asset_id'],run['render']['photo_appearance']['asset_id'])

    def test_two_originals_cpu_initial_edit_spends_no_images_or_conditional_rounds(self):
        self.params['adjustments'][0]['focus_sigma']=.01
        for index,source in enumerate(self.fx.sources):
            with self.subTest(original=index):
                self.decisions=0
                _,run=self.start(self.request(source=source,run_id='development-'+str(index)))
                self.assertEqual(run['result_publication']['state'],'committed')
                self.assert_photographic_provenance(run)
                state=self.state(run)
                self.assertEqual(state['initial_operation']['status'],'completed')
                self.assertEqual(state['used']['improvement_steps'],0)
                self.assertEqual(state['used']['image_submissions'],0)
                self.assertEqual(state['used']['tool_calls'],1)
                self.assertFalse(run['photo_edit_session']['automated_target_met'])
                self.assertFalse(run['photo_edit_session']['creative_delivery_eligibility']['eligible'])
                self.assertTrue(all(run['photo_edit_session']['independent_evaluation'][axis]['status']=='pass' for axis in session.AXES))
        self.assertEqual(len(self.fx.image_requests),0);self.assertEqual(len(self.fx.vision_requests),8)

    def test_full_composite_cannot_be_satisfied_by_a_photographic_only_chain(self):
        _,run=self.start(self.request(work='full_composite'))
        self.assertFalse(run['photo_edit_session']['automated_target_met'])
        self.assertFalse(run['photo_edit_session']['creative_delivery_eligibility']['eligible'])
        self.assert_photographic_provenance(run)

    def test_review_off_performs_actual_cpu_edit_and_saves_null_review_draft(self):
        checked,run=self.start(self.request(review=False))
        self.assertEqual(checked['plan']['services']['text_vision_requests_upper_bound'],2)
        state=self.state(run);best=session.selected_output(Path(run['photo_edit_session']['path']).parent)['candidate']
        self.assertIsNone(best['review']);self.assertFalse(run['photo_edit_session']['automated_target_met'])
        self.assertEqual(state['used']['vision_requests'],1)
        self.assertEqual(len(self.fx.vision_requests),2);self.assertEqual(len(self.fx.image_requests),0)
        self.assert_photographic_provenance(run)

    def test_tiny_review_off_edit_preserves_changed_float_master_even_with_identical_preview(self):
        from photo_appearance import load_asset,validate_asset
        from photo_edit_tools import _close_loaded
        from photo_edit_finish import save_selected_candidate,reviewed_candidate_record
        import single_source_workflow as workflow
        import numpy as np
        import tifffile
        self.params={'adjustments':[{'region':[0,0,1,1],'exposure_ev':.000001,
            'warmth_ev':0.,'tint_ev':0.,'feather':0.}]}
        checked,run=self.start(self.request(review=False,run_id='tiny-photographic-draft'))
        state=self.state(run);directory=Path(run['photo_edit_session']['path']).parent
        selected=session.selected_output(directory)['candidate']
        baseline=next(row for row in state['candidates'] if row['candidate_id']==state['baseline_candidate_id'])
        binding={'source_sha256':run['source']['sha256'],'scope':run['task_plan']['editing_scope'],
            'design_id':run['generation_design']['observed_design']['design_id']}
        source_receipt=validate_asset(run['initial_source_appearance'],**binding)
        result_receipt=validate_asset(selected['appearance_asset'],**binding)
        self.assertEqual(result_receipt['files']['preview']['sha256'],source_receipt['files']['preview']['sha256'])
        self.assertNotEqual(result_receipt['files']['master']['sha256'],source_receipt['files']['master']['sha256'])
        # Exercise the shared saver against an actually rendered old master,
        # not a fake PNG-only current state. The preview happens to be equal.
        staged=copy.deepcopy(run);staged['options']['photo_appearance']=copy.deepcopy(run['initial_source_appearance'])
        staged['reference']=reviewed_candidate_record(self.root,baseline,staged['options'],evidence_state=state)
        args=workflow.parse_args(['--root',str(self.root),'--source',run['source']['path']])
        for key,value in checked['request'].items():setattr(args,key,value)
        rendered=workflow.render_candidate(self.root,staged,args)
        staged.update(auto_draft=rendered,render=rendered['metadata'],artifacts=rendered['outputs'],review=rendered['review'])
        saved=save_selected_candidate(self.root,staged,args,selected=selected,session_directory=directory)
        self.assertTrue(saved['changed'])
        self.assertEqual(staged['options']['photo_appearance']['asset_id'],selected['appearance_asset']['asset_id'])
        loaded=None
        try:
            loaded=load_asset(selected['appearance_asset'],**binding)
            np.testing.assert_array_equal(tifffile.imread(staged['artifacts']['render_scene_master']),loaded['master'])
        finally:_close_loaded(loaded)
        self.assertIsNone(selected['review']);self.assertFalse(staged['photo_edit_session']['automated_target_met'])
        self.assertEqual(len(self.fx.image_requests),0);self.assertEqual(len(self.fx.vision_requests),2)

    def test_retouch_cpu_initial_and_explicit_zero_image_manual_feedback_use_one_pipeline(self):
        _,parent=self.start(self.request(work='retouch'))
        self.assertTrue(parent['photo_edit_session']['automated_target_met'])
        self.assert_photographic_provenance(parent)
        parent_path=self.root/'outputs/runs'/parent['run_id']/'manifest.json';before=parent_path.read_bytes()
        body=self.request(work='retouch',run_id='retouch-development-feedback');body.pop('editing_autofinish')
        body['creative_task']['mode']='existing_reference'
        body.update(parent_run_id=parent['run_id'],reference=parent['artifacts']['auto_draft'],
            editing_iteration={'maximum_improvements':1,'maximum_image_submissions':0,
                'user_feedback':'Gently improve the existing photographic balance.'})
        before_calls=len(self.fx.vision_requests)
        checked,child=self.start(body)
        self.assertEqual(sum(checked['plan']['image_budget'].values()),0)
        self.assertIn('develop_photograph',checked['plan']['editing_refinement_authorization']['image_free_tools'])
        self.assertEqual(len(self.fx.vision_requests)-before_calls,2)
        self.assertEqual(len(self.fx.image_requests),0)
        self.assertTrue(child['photo_edit_session']['automated_target_met'])
        self.assertEqual(self.state(child)['used']['improvement_steps'],1)
        self.assertFalse(child['photo_edit_session']['human_approved'])
        self.assertEqual(parent_path.read_bytes(),before)
        self.assert_photographic_provenance(child)

    def test_model_can_follow_cpu_initial_with_real_local_creative_edit_using_shared_k(self):
        self.followup='edit_region';self.fx.generation.side_effect=self.fx.image_request
        checked,run=self.start(self.request(k=1))
        self.assertEqual(sum(checked['plan']['image_budget'].values()),2)
        self.assertEqual(len(self.fx.image_requests),1);self.assertEqual(len(self.fx.vision_requests),6)
        state=self.state(run)
        self.assertEqual(state['used']['tool_calls'],2);self.assertEqual(state['used']['improvement_steps'],1)
        self.assertEqual(state['used']['image_submissions'],1)
        self.assertTrue(run['photo_edit_session']['automated_target_met'])
        self.assertTrue(run['photo_edit_session']['creative_delivery_eligibility']['eligible'])
        report=run['photo_edit_session']['origin_request_budget']
        self.assertEqual(report['initial_operations_reserved'],1);self.assertEqual(report['improvement_rounds_reserved'],1)
        self.assertEqual(report['initial_operation_image_submissions'],0);self.assertEqual(report['improvement_image_submissions'],1)

    def test_zero_image_manual_development_of_a_real_creative_parent_keeps_pixel_origin_and_can_meet_feedback(self):
        from photo_appearance import load_asset
        from photo_edit_tools import _close_loaded
        import numpy as np
        self.initial_tool='edit_region';self.fx.generation.side_effect=self.fx.image_request
        _,parent=self.start(self.request(run_id='creative-parent'))
        self.assertTrue(parent['photo_edit_session']['automated_target_met'])
        self.assertEqual(len(self.fx.image_requests),1)
        counts=len(self.fx.image_requests),len(self.fx.vision_requests)
        self.followup='develop_photograph'
        body=self.request(run_id='photographic-feedback');body.pop('editing_autofinish')
        body['creative_task']['mode']='existing_reference'
        body.update(parent_run_id=parent['run_id'],reference=parent['artifacts']['auto_draft'],
            editing_iteration={'maximum_improvements':1,'maximum_image_submissions':0,
                'user_feedback':'Improve the current photographic colour balance while keeping the established scene.'})
        _,child=self.start(body)
        self.assertEqual((len(self.fx.image_requests)-counts[0],len(self.fx.vision_requests)-counts[1]),(0,2))
        self.assertTrue(child['photo_edit_session']['automated_target_met'])
        self.assert_photographic_provenance(child)
        binding={'source_sha256':child['source']['sha256'],'scope':child['task_plan']['editing_scope'],
            'design_id':child['generation_design']['observed_design']['design_id']}
        before=after=None
        try:
            before=load_asset(parent['options']['photo_appearance'],**binding)
            after=load_asset(child['options']['photo_appearance'],**binding)
            self.assertFalse(after['receipt']['photographic_operation']['source_only_photographic_lineage'])
            np.testing.assert_array_equal(before['generated_coverage'],after['generated_coverage'])
        finally:_close_loaded(before,after)

    def test_identical_cpu_no_op_does_not_turn_source_input_into_an_edited_success(self):
        self.params={'adjustments':[{'region':[0,0,1,1],'exposure_ev':0.,'warmth_ev':0.,'tint_ev':0.,'feather':0.}]}
        _,run=self.start(self.request(work='retouch'))
        self.assertFalse(run['photo_edit_session']['automated_target_met'])
        self.assertTrue(run['photo_edit_session']['selected_is_source_baseline'])
        self.assertEqual(run['variants'][0]['status'],'no_edited_candidate')
        self.assertEqual(len(self.fx.image_requests),0)


if __name__=='__main__':unittest.main()
