"""One session driver, deterministic stops and atomic reviewed-master save."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from asset_identity import sha256_file
import photo_edit_session as session
from photo_edit_finish import drive_session,save_selected_candidate
from photo_edit_session_smoke import design,evidence
from photo_editing import editing_scope
from raw_support import deterministic_srgb_profile
import photo_source_session_smoke as native_fixture


class FinishDriverTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix='generic-session-driver-');self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)
        self.source=self.root/'source.png';Image.new('RGB',(72,96),'gray').save(self.source)
        self.base=self.root/'initial.png';Image.new('RGB',(72,96),'navy').save(self.base,icc_profile=deterministic_srgb_profile())
        self.edited=self.root/'edit.png';Image.new('RGB',(72,96),'teal').save(self.edited,icc_profile=deterministic_srgb_profile())
        self.scope=editing_scope({'language':'en'});self.directory=self.root/'session';self.calls=[]
        session.open_session(self.directory,source=self.source,scope=self.scope,design=design(self.scope),
            execution_identity='checked-fixture',producer_identity='driver-fixture',
            budget={'image_submissions':1,'vision_requests':3,'local_repairs':1,'tool_calls':1,'improvement_steps':1})
        session.adopt_candidate(self.directory,self.base,provenance={'fixture':'already charged initial image'})
        def execute(context):
            self.calls.append('tool');return {'image':str(self.edited),'provenance':{'fixture':'synthetic edited pixels'}}
        self.tool=session.Tool('joint_edit',('relight',),{'image_submissions':1},'fixture',execute)


    def decision(self,context):
        self.calls.append('decide')
        return {'action':'tool','tool':'joint_edit','operations':['relight'],
            'instruction':'Correct the main observed light problem.','reason':'The observed defect remains.',**evidence(context)}

    def review(self,context,*,first_pass=False,worse=False):
        self.calls.append('evaluate')
        later=any(row['role']=='previous_best' for row in context['images'])
        return {'credibility':{'status':'pass' if first_pass or later and not worse else 'fail','findings':['Synthetic observed evidence']},
            'creative_benefit':{'status':'pass','findings':['Synthetic visible benefit']},
            'comparison':('worse' if worse else 'better') if later else 'first_candidate',
            'primary_issue':'none' if first_pass or later and not worse else 'Fix the main light issue.',**evidence(context)}

    def drive(self,**extra):
        return drive_session(self.directory,tools=[self.tool],decide=self.decision,evaluate=self.review,
            model_identity='fixture',**extra)

    def test_initial_pass_never_calls_a_decision_or_image_tool(self):
        result=drive_session(self.directory,tools=[self.tool],decide=self.decision,
            evaluate=lambda c:self.review(c,first_pass=True),model_identity='fixture')
        self.assertEqual(self.calls,['evaluate'])
        self.assertTrue(result['state']['automated_target_met'])
        self.assertEqual(result['state']['used']['image_submissions'],0)
        self.assertEqual(result['state']['binding']['initial_usage']['image_submissions'],0)
        self.assertFalse(result['selected']['human_approved'])

    def test_fail_tool_worse_retains_initial_best_with_one_shared_loop(self):
        result=drive_session(self.directory,tools=[self.tool],decide=self.decision,
            evaluate=lambda c:self.review(c,worse=True),model_identity='fixture')
        self.assertEqual(self.calls,['evaluate','decide','tool','evaluate'])
        self.assertEqual(result['state']['stop_reason'],'no_verified_net_improvement')
        self.assertEqual(result['selected']['image']['sha256'],sha256_file(self.base))
        self.assertEqual(result['state']['used']['image_submissions'],1)
        self.assertEqual(result['state']['used']['vision_requests'],3)

    def test_unknown_tool_outcome_stops_and_cannot_be_deterministically_cleared(self):
        def timeout(context):self.calls.append('tool');raise TimeoutError('outcome unknown')
        self.tool=session.Tool('joint_edit',('relight',),{'image_submissions':1},'fixture',timeout)
        first=self.drive();calls=list(self.calls)
        self.assertEqual(first['state']['status'],'needs_attention')
        self.assertEqual(first['selected']['image']['sha256'],sha256_file(self.base))
        second=self.drive();self.assertEqual(self.calls,calls)
        self.assertEqual(second['state']['status'],'needs_attention')
        with self.assertRaisesRegex(RuntimeError,'unresolved'):
            session.stop(self.directory,'origin_improvement_budget_exhausted')

    def test_origin_limit_stops_before_another_decision_without_fabricating_a_response(self):
        class Hooks:
            def remaining(self):return {'improvement_rounds':0}
        result=self.drive(hooks=Hooks())
        self.assertEqual(self.calls,['evaluate'])
        self.assertEqual(result['state']['stop_reason'],'origin_improvement_budget_exhausted')
        self.assertEqual([row['kind'] for row in result['state']['attempts']],['evaluate'])
        self.assertEqual(result['state']['used']['image_submissions'],0)
        with self.assertRaises(ValueError):session.stop(self.directory,'model_says_perfect')


class ReviewedSaveTests(unittest.TestCase):
    def setUp(self):
        fx=native_fixture.NativeFixture(self,k=0);variant=fx.start()[0]
        self.root=fx.root;self.run=fx.run;self.args=fx.args
        self.directory=Path(variant['photo_edit_session']['path']).parent
        state=session.snapshot(self.directory)
        self.selected=session.selected_output(self.directory)['candidate']
        baseline=next(row for row in state['candidates'] if row['candidate_id']==state['baseline_candidate_id'])
        self.run['selected_variant_id']='v1'
        self.run['options']['photo_appearance']=copy.deepcopy(self.run['initial_source_appearance'])
        from photo_edit_finish import reviewed_candidate_record
        self.run['reference']=reviewed_candidate_record(self.root,baseline,self.run['options'])
        from generated_appearance_render import render_candidate
        self.initial=render_candidate(self.root,self.run,self.args)
        self.run.update(auto_draft=self.initial,render=self.initial['metadata'],review=self.initial['review'],
            artifacts=self.initial['outputs'])
        self.base=Path(self.initial['outputs']['auto_draft'])

    def test_common_save_preserves_reviewed_png_and_original_asset(self):
        old=copy.deepcopy(self.run['options']['photo_appearance']);old_bytes=(Path(old['path'])/'receipt.json').read_bytes()
        identity_before=id(self.run)
        result=save_selected_candidate(self.root,self.run,self.args,selected=self.selected,session_directory=self.directory)
        self.assertTrue(result['changed']);self.assertEqual(id(self.run),identity_before)
        self.assertEqual(sha256_file(self.run['artifacts']['auto_draft']),self.selected['image']['sha256'])
        self.assertNotEqual(self.run['options']['photo_appearance']['asset_id'],old['asset_id'])
        self.assertEqual((Path(old['path'])/'receipt.json').read_bytes(),old_bytes)
        self.assertTrue(self.run['render']['reviewed_preview']['master_RGB8_and_ICC_equal'])
        receipt=json.loads((Path(self.run['options']['photo_appearance']['path'])/'receipt.json').read_text(encoding='utf8'))
        self.assertFalse(receipt['native_generated_detail'])

    def test_unchanged_best_uses_saved_bytes_without_rerender(self):
        import photo_edit_session as session
        state=session.snapshot(self.directory)
        selected=session.candidate_for_io(state,next(row for row in state['candidates']
            if row['candidate_id']==state['baseline_candidate_id']))
        before=copy.deepcopy(self.run)
        with patch('single_source_workflow.render_candidate',side_effect=AssertionError('rerendered')):
            result=save_selected_candidate(self.root,self.run,self.args,selected=selected,session_directory=self.directory)
        self.assertFalse(result['changed']);self.assertEqual(self.run,before)

    def test_post_render_copy_failure_restores_prior_run_review_and_complete_draft(self):
        import shutil
        before=copy.deepcopy(self.run)
        review_path=Path(self.initial['review_path']);review_before=review_path.read_bytes()
        draft_sha=sha256_file(self.base);copyfile=shutil.copyfile
        def failed_copy(source,destination,*args,**kwargs):
            if '.reviewed-' in Path(destination).name:raise OSError('Synthetic late save failure')
            return copyfile(source,destination,*args,**kwargs)
        with patch('photo_edit_finish.shutil.copyfile',side_effect=failed_copy):
            with self.assertRaisesRegex(OSError,'late save'):
                save_selected_candidate(self.root,self.run,self.args,selected=self.selected,session_directory=self.directory)
        self.assertEqual(self.run,before)
        self.assertEqual(review_path.read_bytes(),review_before)
        self.assertEqual(sha256_file(self.base),draft_sha)

    def test_late_cancellation_after_native_render_keeps_original_active_asset(self):
        import single_source_workflow as workflow
        before=copy.deepcopy(self.run);review=Path(self.initial['review_path']);review_before=review.read_bytes()
        native=workflow.render_candidate;cancelled=False
        def late(*args,**kwargs):
            nonlocal cancelled
            result=native(*args,**kwargs);cancelled=True;return result
        with patch('single_source_workflow.render_candidate',side_effect=late):
            with self.assertRaises(InterruptedError):
                save_selected_candidate(self.root,self.run,self.args,selected=self.selected,
                    session_directory=self.directory,cancel=lambda:cancelled)
        self.assertEqual(self.run,before)
        self.assertEqual(review.read_bytes(),review_before)


if __name__=='__main__':unittest.main()
