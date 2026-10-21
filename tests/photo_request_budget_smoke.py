"""Real cross-revision reviews share checked origin quota; model boundaries fake."""
from concurrent.futures import ThreadPoolExecutor
import copy
from io import BytesIO
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from asset_identity import sha256_file
import photo_request_budget as budget
import photo_source_session_smoke as native_fixture
import photo_edit_session as session
import photo_edit_finish as finish
from creative_task_smoke import snapshot


class SourceOriginBudgetTests(unittest.TestCase):
    def fixture(self,**kwargs):return native_fixture.NativeFixture(self,**kwargs)

    def hooks(self,fx):
        if not fx.run.get('initial_source_appearance'):
            from photo_appearance import create_source_baseline
            fx.run['initial_source_appearance']=create_source_baseline(fx.directory/'appearance/source',
                source_path=fx.run['source']['path'],source_sha256=fx.run['source']['sha256'],
                scope=fx.plan['editing_scope'],design_id=fx.run['generation_design']['observed_design']['design_id'])
        return budget.make_hooks(fx.root,fx.run,finish._connection(fx.run,fx.args))

    def context(self,variant,*,stage='improvement',nonce='probe'):
        state=session.snapshot(Path(variant['photo_edit_session']['path']).parent)
        context=session._context(state);context.update(operation_stage=stage,attempt_id=nonce)
        context['tools']=copy.deepcopy(state['runtime']['tools'])
        return context

    def test_checked_native_limits_are_real_and_replanning_is_readonly(self):
        for count in (1,2,8):
            with self.subTest(count=count):
                fx=self.fixture(count=count)
                before=snapshot(fx.root);calls=len(fx.calls)
                checked=fx.fx.backend.task_plan(fx.checked['request'])
                self.assertEqual(snapshot(fx.root),before);self.assertEqual(len(fx.calls),calls)
                self.assertEqual(checked['plan']['generation_submissions_upper_bound'],count+2)
                self.assertEqual(checked['plan']['services']['text_vision_requests_upper_bound'],2+2*count+4)

    def test_source_baseline_initial_slot_is_reserved_once_under_concurrency(self):
        fx=self.fixture(count=2);hooks=self.hooks(fx);binding=budget._read(Path(hooks.locator['path']))['binding']
        key=budget.endpoint(source_sha256=fx.run['source']['sha256'],candidate_sha256=binding['source_preview_sha256'],
            source_preview_sha256=binding['source_preview_sha256'],scope_id=binding['scope_id'],design_id=binding['design_id'],
            prompt='Synthetic source QA endpoint',connection_identity='connection',decoder_identity='decoder')
        def reserve(_):
            try:return budget.reserve(hooks.locator,key,run_id='origin')
            except budget.ReviewPending:return 'pending'
        with ThreadPoolExecutor(max_workers=4) as pool:outcomes=list(pool.map(reserve,range(4)))
        owners=[row for row in outcomes if isinstance(row,dict) and row['owner']]
        self.assertEqual(len(owners),1);self.assertEqual(hooks.report()['reserved_vision_requests'],2)
        budget.finish(owners[0])
        changed={**key,'connection_identity':'another-model'}
        self.assertRaises(budget.ReviewPending,budget.reserve,hooks.locator,changed,run_id='another-child')
        self.assertEqual(hooks.report()['reserved_vision_requests'],2)

    def test_initial_operation_is_not_an_improvement_or_phantom_initial_image(self):
        fx=self.fixture();variant=fx.start()[0];hooks=self.hooks(fx)
        report=hooks.report()
        self.assertEqual(report['initial_operations_reserved'],1)
        self.assertEqual(report['historical_initial_image_requests'],0)
        self.assertEqual(report['reserved_image_requests'],1)
        self.assertEqual(report['improvement_rounds_reserved'],0)
        self.assertRaises(budget.BudgetExhausted,budget.reserve_operation,hooks.locator,'decision',
            self.context(variant,stage='initial',nonce='a-new-child-does-not-reset-initial'))

    def test_zero_image_finish_requires_review_and_positive_normal_rounds(self):
        from creative_task import autofinish_policy
        fx=self.fixture(k=1)
        self.assertEqual(fx.plan['editing_autofinish'],{'maximum_improvements':1,'maximum_zero_image_finishes':0})
        for count,review,finishes in ((0,True,1),(1,False,1),(1,True,2),(1,True,True)):
            intent=copy.deepcopy(fx.plan['intent']);intent['services']['visual_review']=review
            self.assertRaises(ValueError,autofinish_policy,
                {'maximum_improvements':count,'maximum_zero_image_finishes':finishes},intent,fx.plan['editing_scope'],supplied=True)

    def test_two_selected_revision_contexts_share_atomic_conditional_rounds(self):
        fx=self.fixture(count=2);variants=fx.start();hooks=self.hooks(fx)
        contexts=[self.context(v,nonce='different-variant-'+str(i)) for i,v in enumerate(variants)]
        with ThreadPoolExecutor(max_workers=2) as pool:
            tickets=list(pool.map(lambda context:budget.reserve_operation(hooks.locator,'decision',context),contexts))
        self.assertTrue(all(row['owner'] for row in tickets))
        self.assertEqual(hooks.report()['improvement_rounds_reserved'],2)
        self.assertEqual(hooks.report()['initial_operations_reserved'],2)
        self.assertRaises(budget.BudgetExhausted,budget.reserve_operation,hooks.locator,'decision',
            self.context(variants[0],nonce='third-round'))

    def test_postreview_and_tool_require_an_actual_decision_lease(self):
        fx=self.fixture();hooks=self.hooks(fx)
        source=fx.run['source'];asset=fx.run['initial_source_appearance']
        from photo_appearance import load_asset
        loaded=load_asset(asset,source_sha256=source['sha256'],scope=fx.plan['editing_scope'],
            design_id=fx.run['generation_design']['observed_design']['design_id'])
        image={'path':str(loaded['preview_path']),'sha256':sha256_file(loaded['preview_path'])}
        context={'session_id':'not-yet-a-decision','attempt_id':'unleased','scope':fx.plan['editing_scope'],
            'design':fx.run['generation_design']['observed_design'],'source_original':source,
            'variant_id':'v1','operation_stage':'initial','visual_review':True,
            'images':[{'role':'source',**image},{'role':'best',**image}],
            'appearance_assets':{'source':asset,'best':asset}}
        for kind in ('tool','post_review'):
            self.assertRaises(budget.BudgetExhausted,budget.reserve_operation,hooks.locator,kind,context)
        self.assertEqual(hooks.report()['reserved_vision_requests'],1)

    def finish_contexts(self):
        fx=self.fixture(count=2,k=1,f=1);variants=fx.start();hooks=self.hooks(fx)
        first=self.context(variants[0],nonce='normal-round')
        budget.reserve_operation(hooks.locator,'decision',first)
        last=self.context(variants[1],nonce='last-round')
        last['zero_image_only']=True
        for name,tool in last['tools'].items():tool['inputs_available']=name=='develop_photograph'
        return fx,hooks,last

    def test_last_slot_is_atomic_and_stale_normal_ordinal_cannot_become_finish(self):
        _fx,hooks,last=self.finish_contexts();before=hooks.report()['reserved_vision_requests']
        stale=copy.deepcopy(last);stale['zero_image_only']=False
        self.assertRaises(budget.BudgetExhausted,budget.reserve_operation,hooks.locator,'decision',stale)
        self.assertEqual(hooks.report()['reserved_vision_requests'],before)
        def reserve(index):
            context=copy.deepcopy(last);context['attempt_id']='last-race-'+str(index)
            try:return budget.reserve_operation(hooks.locator,'decision',context)
            except (budget.BudgetExhausted,budget.ReviewPending):return None
        with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(reserve,range(4)))
        self.assertEqual(sum(bool(row and row['owner']) for row in results),1)
        self.assertEqual(hooks.report()['zero_image_finish_rounds_reserved'],1)
        self.assertEqual(hooks.report()['reserved_vision_requests'],before+1)

    def test_last_slot_rejects_offered_image_tool_and_fake_zero_image_descriptor(self):
        _fx,hooks,last=self.finish_contexts()
        for name,tool in last['tools'].items():tool['inputs_available']=name=='joint_edit'
        before=hooks.report()['reserved_vision_requests']
        self.assertRaises(ValueError,budget.reserve_operation,hooks.locator,'decision',last)
        self.assertEqual(hooks.report()['reserved_vision_requests'],before)
        last['tools']['joint_edit']['costs']['image_submissions']=0
        ticket=budget.reserve_operation(hooks.locator,'decision',last)
        image={key:last['images'][0][key] for key in ('path','sha256')}
        payload={'action':'tool','tool':'joint_edit','operations':['relight'],
            'instruction':'Synthetic attempted generation.','reason':'Exercise actual execution costs.'}
        result={**payload,'observed_images':[{'role':row['role'],'sha256':row['sha256']} for row in last['images']],
            'provenance':{'model':'synthetic-transport','network_used':True,'generation_used':False,'source':image,
                'decision_image_mapping':{'images':last['images'],'transport_image':image}}}
        budget.finish_operation(ticket,result=result)
        context={**last,'attempt_id':'actual-image-tool','decision':payload}
        self.assertRaises(ValueError,budget.reserve_operation,hooks.locator,'tool',context,tool_costs={'image_submissions':1})
        self.assertEqual(hooks.report()['reserved_image_requests'],2)

    def test_review_off_has_initial_decisions_but_no_baseline_or_postqa_slots(self):
        fx=self.fixture(count=2,review=False);fx.start();report=self.hooks(fx).report()
        self.assertEqual(report['checked_vision_limit'],3)
        self.assertEqual(report['reserved_vision_requests'],3)
        self.assertEqual(report['review_reservations'],0)
        self.assertEqual(report['reserved_image_requests'],2)
        self.assertEqual(report['improvement_rounds_reserved'],0)

    def test_readonly_attempt_cannot_become_a_leased_tool_input_or_replace_best(self):
        source={'role':'source','path':'source.png','sha256':'1'*64}
        best={'role':'best','path':'best.png','sha256':'2'*64}
        attempt={'role':'previous_attempt','path':'attempt.png','sha256':'3'*64}
        decision={'images':[source,best,attempt]}
        descriptor={'requires_working':False}
        self.assertTrue(budget._tool_images_match(decision,{'images':[source,best]},descriptor))
        self.assertFalse(budget._tool_images_match(decision,decision,descriptor))
        self.assertFalse(budget._tool_images_match(decision,
            {'images':[source,{**attempt,'role':'best'}]},descriptor))
        self.assertFalse(budget._tool_images_match(decision,
            {'images':[source,best,{**attempt,'role':'working'}]},descriptor))
        working={'role':'working','path':'working.png','sha256':'4'*64}
        qualified={'images':[source,best,working]}
        self.assertTrue(budget._tool_images_match(qualified,qualified,{'requires_working':True}))
        self.assertFalse(budget._tool_images_match(qualified,{'images':[source,best]},
                                                 {'requires_working':True}))

    def test_missing_ledger_and_changed_checked_limit_never_replenish_history(self):
        fx=self.fixture();fx.start();hooks=self.hooks(fx);path=Path(hooks.locator['path'])
        original=path.read_bytes();changed=copy.deepcopy(fx.run)
        changed['task_plan']['services']['text_vision_requests_upper_bound']+=1
        self.assertRaises(ValueError,budget.prepare,fx.root,changed)
        self.assertEqual(path.read_bytes(),original)
        changed=copy.deepcopy(fx.run)
        changed['task_plan']['editing_autofinish']['maximum_zero_image_finishes']=1
        changed['task_plan']['services']['text_vision_requests_upper_bound']+=2
        self.assertRaises(ValueError,budget.prepare,fx.root,changed)
        self.assertEqual(path.read_bytes(),original,'A new finishing policy replenished an existing origin')
        path.unlink()
        self.assertRaises(budget.ReviewPending,budget.prepare,fx.root,fx.run)
        self.assertFalse(path.exists())

    def test_identical_postreview_cache_has_zero_actual_new_vision_fee(self):
        fx=self.fixture(count=2);fx.identical_images=True;variants=fx.start();report=self.hooks(fx).report()
        self.assertEqual(report['reserved_vision_requests'],5)
        self.assertEqual(len(fx.calls),5)
        state=session.snapshot(Path(variants[1]['photo_edit_session']['path']).parent)
        self.assertEqual(state['used']['vision_requests'],2)  # Conservative session reservations, not actual calls.
        self.assertEqual(report['reserved_image_requests'],2)


if __name__ == '__main__':
    unittest.main()
