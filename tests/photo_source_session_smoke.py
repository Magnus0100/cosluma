"""Source float -> actual bounded session/transport adapters; model boundaries fake."""
import copy
import json
from io import BytesIO
from pathlib import Path
import unittest
import numpy as np
from PIL import Image

from asset_identity import sha256_file
import photo_art_direction
import photo_edit_finish as finish
import photo_edit_session as session
import photo_request_budget as budget
import scoped_production_smoke as fixtures
from photo_editing_smoke import proposal


class NativeFixture:
    def __init__(self,owner,*,count=1,review=True,k=2,f=0):
        self.fx=fixtures.ScopedProductionTests(methodName='runTest');self.fx.setUp()
        owner.addCleanup(self.fx.doCleanups)
        self.owner=owner;self.root=self.fx.root;self.calls=[];self.tool='joint_edit'
        self.vision_failure=None;self.image_failure=None
        self.review_failure=None
        self.reject_edits=False;self.identical_images=False
        self.fx.vision.side_effect=self.vision
        original=self.fx.image_request
        def image(*args,**kwargs):
            if self.image_failure:raise self.image_failure
            payload,pixels,mode=original(*args,**kwargs)
            if self.identical_images:return payload,pixels,mode
            values=np.asarray(Image.open(BytesIO(pixels)).convert('RGB')).copy()
            values[:,:,1]=20+len(self.fx.image_requests)*20
            stream=BytesIO();Image.fromarray(values).save(stream,format='PNG')
            return payload,stream.getvalue(),mode
        self.fx.generation.side_effect=image
        body=self.fx.request(self.fx.sources[0],run_id='source-native',variant_count=count,
            defer_reference_selection=count>1,review=review,editing_autofinish={'maximum_improvements':k if review else 0,
                'maximum_zero_image_finishes':f})
        self.checked=self.fx.backend.task_plan(body)
        owner.assertTrue(self.checked['plan']['ready'],self.checked['plan']['blocking_reasons'])
        self.plan=self.checked['plan'];self.directory=self.root/'outputs/runs/source-native'
        self.directory.mkdir(parents=True)
        scope=self.plan['editing_scope'];source=self.plan['inputs']['source']
        record=photo_art_direction.prepare(Path(source['path']),self.directory/'design',scope,
            brief=body['creative_task']['brief'],provider=body['provider'],vision_model=body['vision_model'],
            api_base=body['api_base'],api_key_env=body['api_key_env'])
        generation=photo_art_direction.generation_design(record,scope,family='gpt_image',source_sha256=source['sha256'])
        self.run={'run_id':'source-native','root':str(self.root),'source':copy.deepcopy(source),
            'task_plan':copy.deepcopy(self.plan),'task_intent':copy.deepcopy(self.plan['intent']),
            'generation_design':generation,'options':{**self.checked['request'],'task_plan':copy.deepcopy(self.plan)},
            'generation_budget':{'image_submissions':0,'submissions':[]},'variants':[],
            'artifacts':{},'sample':'SOURCE_NATIVE','status':'preparing'}
        import single_source_workflow as workflow
        self.args=workflow.parse_args(['--root',str(self.root),'--source',source['path']])
        for key,value in self.checked['request'].items():setattr(self.args,key,value)
        self.args.root=self.root

    def vision(self,source,output,prompt,**kwargs):
        source=Path(source);self.calls.append({'source':source,'prompt':prompt,'output':str(output)})
        provenance={'source':{'path':str(source),'sha256':sha256_file(source)},'network_used':True,
            'generation_used':False,'model':'synthetic-actual-transport-model',
            'input_profile':kwargs.get('input_profile','standard')}
        if source.name!='review-board.png':return proposal(),provenance
        mapping=json.loads((source.parent/'review-board.json').read_text(encoding='utf8'))
        if 'decision' in str(output):
            if self.vision_failure:raise self.vision_failure
            result={'action':'tool','tool':self.tool,'operations':['relight'],
                'instruction':'Improve the observed subject and scene lighting.','reason':'Synthetic model-boundary decision.'}
            if self.tool=='edit_region':result['region']=[.2,.2,.8,.8]
            return result,provenance
        if len(mapping['images'])==3 and self.review_failure:raise self.review_failure
        return {'credibility':{'status':'pass','findings':['Synthetic credibility observation']},
            'creative_benefit':{'status':'fail' if self.reject_edits and len(mapping['images'])==3 else 'pass',
                'findings':['Synthetic creative observation']},
            'comparison':'better' if len(mapping['images'])==3 else 'first_candidate',
            'primary_issue':'none'},provenance

    def start(self):return finish.initial_candidates(self.root,self.run,self.args)


class SourceNativeSessionTests(unittest.TestCase):
    def test_source_qa_pass_is_not_an_edit_first_operation_does_not_spend_k(self):
        fx=NativeFixture(self);variants=fx.start();self.assertEqual(len(variants),1)
        item=variants[0];state=session.snapshot(Path(item['photo_edit_session']['path']).parent)
        self.assertEqual(state['status'],'waiting_selection')
        self.assertEqual(state['initial_operation']['status'],'completed')
        self.assertEqual(state['used']['improvement_steps'],0)
        self.assertEqual(state['used']['image_submissions'],1)
        self.assertEqual(state['used']['vision_requests'],3)
        self.assertEqual(len(fx.fx.image_requests),1)
        self.assertEqual(item['status'],'complete')
        self.assertNotEqual(item['photo_appearance']['asset_id'],fx.run['initial_source_appearance']['asset_id'])
        baseline=next(c for c in state['candidates'] if c['candidate_id']==state['baseline_candidate_id'])
        self.assertEqual(baseline['review']['credibility']['status'],'pass')
        attempts=state['attempts']
        self.assertEqual([a['operation_stage'] for a in attempts],['baseline','initial','initial','initial'])
        context=attempts[1]['context']
        self.assertEqual(context['source_original']['sha256'],fx.run['source']['sha256'])
        self.assertEqual(context['appearance_assets']['best'],fx.run['initial_source_appearance'])
        hooks=budget.make_hooks(fx.root,fx.run,finish._connection(fx.run,fx.args))
        report=hooks.report()
        self.assertEqual(report['reserved_image_requests'],1)
        self.assertEqual(report['historical_initial_image_requests'],0)
        self.assertEqual(report['reserved_vision_requests'],4)
        self.assertEqual(report['initial_operations_reserved'],1)
        self.assertEqual(report['improvement_rounds_reserved'],0)

    def test_review_disabled_still_decides_and_executes_without_fake_feedback(self):
        fx=NativeFixture(self,review=False,k=0);variants=fx.start();item=variants[0]
        state=session.snapshot(Path(item['photo_edit_session']['path']).parent)
        self.assertEqual([a['kind'] for a in state['attempts']],['decide','tool'])
        self.assertEqual(state['attempts'][0]['context']['feedback'],[])
        self.assertIsNone(session.selected_output(Path(item['photo_edit_session']['path']).parent)['candidate']['review'])
        self.assertFalse(state['automated_target_met'])
        self.assertEqual(len(fx.fx.image_requests),1)
        self.assertEqual(len(fx.calls),2)

    def test_two_variants_share_source_qa_and_design_but_have_distinct_initial_leases(self):
        fx=NativeFixture(self,count=2);variants=fx.start()
        self.assertEqual([v['variant_id'] for v in variants],['v1','v2'])
        self.assertEqual(len(fx.fx.image_requests),2)
        report=budget.make_hooks(fx.root,fx.run,finish._connection(fx.run,fx.args)).report()
        self.assertEqual(report['reserved_vision_requests'],6)
        self.assertEqual(report['initial_operations_reserved'],2)
        self.assertEqual(report['improvement_rounds_reserved'],0)
        second=session.snapshot(Path(variants[1]['photo_edit_session']['path']).parent)
        self.assertEqual(second['attempts'][0]['kind'],'adopted_evaluation')
        self.assertEqual(second['used']['vision_requests'],2)
        before=len(fx.calls);fx.start();self.assertEqual(len(fx.calls),before)

    def test_unknown_decision_does_not_try_another_variant_or_resubmit(self):
        fx=NativeFixture(self,count=2);fx.vision_failure=TimeoutError('Synthetic external decision timeout')
        self.assertRaises(RuntimeError,fx.start)
        self.assertEqual(len(fx.fx.image_requests),0)
        before=len(fx.calls);self.assertRaises(RuntimeError,fx.start)
        self.assertEqual(len(fx.calls),before)
        report=budget.make_hooks(fx.root,fx.run,finish._connection(fx.run,fx.args)).report()
        self.assertEqual(report['initial_operations_reserved'],1)

    def test_selected_variant_uses_only_shared_k_and_actual_three_images_eight_vision(self):
        fx=NativeFixture(self);fx.reject_edits=True;variants=fx.start()
        fx.run['selected_variant_id']='v1'
        fx.run['reference']=copy.deepcopy(variants[0]['reference'])
        fx.run['options']['photo_appearance']=copy.deepcopy(variants[0]['photo_appearance'])
        result=finish.finish_saved_run(fx.root,fx.run,fx.args)
        report=result['editing_autofinish_result']['origin_budget']
        self.assertEqual(report['reserved_image_requests'],3)
        self.assertEqual(report['reserved_vision_requests'],8)
        self.assertEqual(report['initial_operations_reserved'],1)
        self.assertEqual(report['improvement_rounds_reserved'],2)
        self.assertFalse(result['photo_edit_session']['automated_target_met'])
        before=len(fx.calls);finish.finish_saved_run(fx.root,result,fx.args)
        self.assertEqual(len(fx.calls),before)

    def test_equivalent_attempt_is_visible_to_next_decision_without_becoming_tool_input(self):
        fx=NativeFixture(self,k=2,f=1);fx.reject_edits=True
        original_vision=fx.vision
        def initial_vision(source,output,prompt,**kwargs):
            payload,provenance=original_vision(source,output,prompt,**kwargs)
            if 'decision' not in str(output):payload['creative_benefit']['status']='fail'
            return payload,provenance
        fx.fx.vision.side_effect=initial_vision;item=fx.start()[0]
        directory=Path(item['photo_edit_session']['path']).parent
        initial=session.snapshot(directory);incumbent=initial['best_candidate_id']
        incumbent_sha=next(c['image']['sha256'] for c in initial['candidates'] if c['candidate_id']==incumbent)
        fx.run['selected_variant_id']='v1';fx.run['reference']=copy.deepcopy(item['reference'])
        fx.run['options']['photo_appearance']=copy.deepcopy(item['photo_appearance'])
        decisions=[];observed_attempts=[]
        def vision(source,output,prompt,**kwargs):
            payload,provenance=original_vision(source,output,prompt,**kwargs)
            mapping=json.loads((Path(source).parent/'review-board.json').read_text(encoding='utf8'))
            if 'decision' in str(output):
                decisions.append(mapping)
                if len(decisions)==1:
                    self.assertEqual([r['role'] for r in mapping['images']],['source','best'])
                else:
                    self.assertEqual([r['role'] for r in mapping['images']],['source','best','previous_attempt'])
                    previous=mapping['images'][-1]
                    self.assertNotEqual(previous['sha256'],incumbent_sha)
                    self.assertIn('"candidate_sha256": "'+previous['sha256']+'"',prompt)
                    observed_attempts.append(previous['sha256'])
                if len(decisions)==3:
                    payload={'action':'stop','reason':'Checked finish cannot create the missing scene treatment.'}
            else:
                payload['credibility']['status']='pass';payload['creative_benefit']['status']='pass'
                payload['comparison']='equivalent'
            return payload,provenance
        fx.fx.vision.side_effect=vision
        result=finish.finish_saved_run(fx.root,fx.run,fx.args)
        self.assertIsNone(result['editing_autofinish_result']['failure'])
        state=session.snapshot(directory)
        self.assertEqual(state['best_candidate_id'],incumbent)
        self.assertIsNone(state['working_candidate_id'])
        self.assertFalse(state['automated_target_met']);self.assertFalse(state['human_approved'])
        alternatives=[c for c in state['candidates'] if c['candidate_id']!=incumbent
                      and c['candidate_id']!=state['baseline_candidate_id']]
        self.assertEqual(observed_attempts,[c['image']['sha256'] for c in alternatives])
        for attempt in state['attempts']:
            if attempt['kind']=='tool':
                self.assertEqual([r['role'] for r in attempt['context']['images']],['source','best'])
                self.assertNotIn('previous_attempt',attempt['context']['appearance_assets'])
                self.assertEqual(attempt['context']['images'][1]['sha256'],
                                 initial['candidates'][0]['image']['sha256'] if attempt.get('operation_stage')=='initial' else incumbent_sha)
        report=result['editing_autofinish_result']['origin_budget']
        self.assertEqual(report['reserved_image_requests'],3)
        self.assertEqual(report['reserved_vision_requests'],9)
        self.assertEqual(report['improvement_rounds_reserved'],3)
        self.assertEqual(report['zero_image_finish_rounds_reserved'],1)
        self.assertEqual(report['unknown_operation_reservations'],0)
        self.assertEqual(sha256_file(result['artifacts']['auto_draft']),incumbent_sha)

    def test_eight_candidates_still_have_one_design_one_source_qa_and_shared_k(self):
        fx=NativeFixture(self,count=8);variants=fx.start()
        self.assertEqual(len(variants),8)
        report=budget.make_hooks(fx.root,fx.run,finish._connection(fx.run,fx.args)).report()
        self.assertEqual(report['reserved_image_requests'],8)
        self.assertEqual(report['reserved_vision_requests'],18)
        self.assertEqual(report['checked_image_limit'],10)
        self.assertEqual(report['checked_vision_limit'],22)
        self.assertEqual(report['improvement_rounds_reserved'],0)

    def test_master_tamper_cannot_be_hidden_by_an_unchanged_preview(self):
        fx=NativeFixture(self);item=fx.start()[0]
        master=Path(item['photo_appearance']['path'])/'master.npy'
        data=bytearray(master.read_bytes());data[-1]^=1;master.write_bytes(data)
        self.assertRaises(ValueError,session.snapshot,Path(item['photo_edit_session']['path']).parent)

    def test_unknown_initial_qa_keeps_actual_unreviewed_draft_without_promoting_best(self):
        fx=NativeFixture(self);fx.review_failure=TimeoutError('Synthetic QA timeout')
        item=fx.start()[0]
        self.assertEqual(item['status'],'needs_attention');self.assertEqual(item['pixel_status'],'complete')
        directory=Path(item['photo_edit_session']['path']).parent
        state=session.snapshot(directory);output=session.selected_output(directory)
        self.assertTrue(output['unreviewed_initial_draft'])
        self.assertNotEqual(output['candidate']['candidate_id'],state['best_candidate_id'])
        self.assertIsNone(output['candidate']['review']);self.assertFalse(state['automated_target_met'])
        before=len(fx.calls);fx.start();self.assertEqual(len(fx.calls),before)
        fx.run['selected_variant_id']='v1';fx.run['reference']=item['reference']
        fx.run['options']['photo_appearance']=item['photo_appearance']
        result=finish.finish_saved_run(fx.root,fx.run,fx.args)
        self.assertEqual(result['status'],'needs_attention')
        self.assertEqual(sha256_file(result['artifacts']['auto_draft']),item['candidate_sha256'])
        self.assertFalse(result['photo_edit_session']['objective_reviewed'])
        self.assertEqual(len(fx.calls),before)


if __name__=='__main__':unittest.main()
