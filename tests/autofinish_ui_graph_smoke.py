"""Default one-click finishing reaches the real checked budget and saved graph.

Only image/vision transports and installation probes are synthetic boundaries;
plans, sessions, saved pixels, graphs and offscreen Qt are real CPU execution.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import qt_workbench as qt
import creative_task
import photo_edit_autofinish_smoke as fixture_module
from design_reconstruction_ui_smoke import Window,Pool
from creative_task_ui_smoke import service_fixture
from cos_node_registry import compile_workflow
from photo_editing import editing_scope
from cos_workflow_graph import build_graph
from pipeline_stage_report import read_stage_report,format_stage_report


class AutofinishUIGraphTests(unittest.TestCase):
    def fixture(self):
        fx=fixture_module.DefaultAutofinishTests(methodName='runTest');fx.setUp();self.addCleanup(fx.doCleanups)
        return fx

    def test_checkbox_budget_start_template_and_review_off_are_real_payloads(self):
        app=qt.QApplication.instance() or qt.QApplication([])
        fx=self.fixture();window=Window(fx.root,'http://127.0.0.1:1');window.pool=Pool()
        with patch.object(creative_task,'_services',side_effect=service_fixture), \
             patch('reference_provider.provider_health',return_value={'ready':True,'kind':'openai-compatible'}):
            try:
                window.provider.setCurrentText('openai-compatible')
                window.set_simple_source(str(fx.fx.sources[0]))
                controls=window.editing_scope_panels[-1]
                self.assertTrue(controls['autofinish'].isChecked())
                self.assertIn('GPU',controls['autofinish'].toolTip());self.assertIn('费用',controls['autofinish'].toolTip())
                window.check_creative_task_plan();pending=window.pool.tasks[-1]
                self.assertEqual(pending.body['editing_autofinish'],{'maximum_improvements':2,'maximum_zero_image_finishes':1})
                checked=fx.fx.backend.task_plan(pending.body)
                self.assertTrue(checked['plan']['ready'],checked['plan']['blocking_reasons'])
                self.assertEqual(sum(checked['plan']['image_budget'].values()),3)
                self.assertEqual(checked['plan']['services']['text_vision_requests_upper_bound'],10)
                pending.signals.result.emit(checked)
                self.assertIn('条件自动纠偏',window.creative_plan_summary.toPlainText())
                window.start_planned_creative_task();started=window.pool.tasks[-1]
                self.assertEqual(started.path,'/api/tasks/start')
                self.assertEqual(creative_task.execution_request(fx.root,started.body)['editing_autofinish'],{'maximum_improvements':2,'maximum_zero_image_finishes':1})
                started.signals.error.emit('Captured checked start only')
                self.assertFalse(fx.fx.image_requests);self.assertFalse(fx.fx.vision_requests)
                controls['autofinish'].setChecked(False)
                request=window.creative_task_request_options('create_reference')
                self.assertEqual(request['editing_autofinish'],{'maximum_improvements':0,'maximum_zero_image_finishes':0})
                saved=copy.deepcopy(window._template_payload())
                controls['autofinish'].setChecked(True)
                window.template_selector.addItem('Auto preference',{'parameters':saved})
                window.template_selector.setCurrentIndex(window.template_selector.count()-1);window.load_workbench_template()
                self.assertFalse(controls['autofinish'].isChecked())
                controls['autofinish'].setChecked(True);controls['review'].setChecked(False)
                self.assertEqual(window.editing_autofinish_payload(),{'maximum_improvements':0,'maximum_zero_image_finishes':0})
                self.assertFalse(controls['autofinish'].isEnabled());self.assertFalse(controls['autofinish'].isChecked())
                controls['review'].setChecked(True)
                self.assertEqual(window.editing_autofinish_payload(),{'maximum_improvements':2,'maximum_zero_image_finishes':0})
                # Explicit K1 is a user preference, not a retired default to upgrade.
                window.apply_editing_autofinish_options({'maximum_improvements':1})
                saved_one=copy.deepcopy(window._template_payload())
                window.apply_editing_autofinish_options({'maximum_improvements':2})
                window.template_selector.addItem('Explicit K1',{'parameters':saved_one})
                window.template_selector.setCurrentIndex(window.template_selector.count()-1);window.load_workbench_template()
                self.assertEqual(window.editing_autofinish_payload(),{'maximum_improvements':1,'maximum_zero_image_finishes':0})
                self.assertIn('1 轮',controls['autofinish'].text())
                one=fx.fx.backend.task_plan({**window.creative_task_request_options('create_reference'),
                    'source':str(fx.fx.sources[0])})
                self.assertEqual(sum(one['plan']['image_budget'].values()),2)
                self.assertEqual(one['plan']['services']['text_vision_requests_upper_bound'],6)
                window.creative_task_mode.setCurrentIndex(window.creative_task_mode.findData('local_portrait'))
                self.assertNotIn('editing_autofinish',window.creative_task_request_options('local_portrait'))
                window.creative_task_mode.setCurrentIndex(window.creative_task_mode.findData('existing_reference'))
                self.assertNotIn('editing_autofinish',window.creative_task_request_options('existing_reference'))
            finally:window.session_timer.stop();window.close();app.processEvents()


    def test_default_and_explicit_k1_share_one_bounded_source_native_graph(self):
        from scoped_workflow import SOURCE_NATIVE_INITIALIZATION
        for n in (1,3,8):
            for k in (0,1,2):
                graph=compile_workflow('create_reference',editing_scope=editing_scope(),visual_review=True,
                    editing_autofinish={'maximum_improvements':k},editing_initialization=SOURCE_NATIVE_INITIALIZATION,
                    initial_variant_count=n)
                self.assertEqual(graph['request_limits'],{'image_submissions':n+k,'vision_requests':2+2*n+2*k})
                self.assertEqual(graph['bounded_dynamic_loop']['initial_operations_per_candidate'],1)
                self.assertEqual(graph['bounded_dynamic_loop']['maximum_improvements'],k)
                self.assertTrue(graph['bounded_dynamic_loop']['improvements_shared_across_variants'])
                self.assertNotIn('generate_edit',graph['order']);self.assertNotIn('save_best',graph['order'])
                self.assertEqual(graph['order'].count('selected_edit_tool'),1)
        graph=compile_workflow('create_reference',editing_scope=editing_scope(),visual_review=False,
            editing_autofinish={'maximum_improvements':0},editing_initialization=SOURCE_NATIVE_INITIALIZATION,
            initial_variant_count=3)
        self.assertEqual(graph['request_limits'],{'image_submissions':3,'vision_requests':4})
        self.assertNotIn('baseline_review',graph['order']);self.assertNotIn('independent_recheck',graph['order'])
        self.assertIn('decide_edit',graph['order']);self.assertIn('selected_edit_tool',graph['order'])

    def test_public_plan_is_pure_for_serial_candidates_and_policy_cannot_be_forged(self):
        fx=self.fixture()
        before=len(fx.fx.image_requests),len(fx.fx.vision_requests)
        for n in (1,3,8):
            for review in (False,True):
                request=fx.fx.request(fx.fx.sources[0],review=review,variant_count=n)
                request.pop('editing_autofinish')
                checked=fx.fx.backend.task_plan(request)
                plan=checked['plan'];k=2 if review else 0
                self.assertTrue(plan['ready'],plan['blocking_reasons'])
                self.assertEqual(sum(plan['image_budget'].values()),n+k)
                self.assertEqual(plan['services']['text_vision_requests_upper_bound'],2+2*n+2*(k+1) if review else 1+n)
                changed=copy.deepcopy(plan);changed.pop('editing_initialization')
                with self.assertRaisesRegex(ValueError,'initialization'):
                    creative_task.validate_task_plan(changed,changed['intent'])
                self.assertEqual(plan['layer_workflow']['initial_candidates']['count'],n)
        self.assertEqual((len(fx.fx.image_requests),len(fx.fx.vision_requests)),before)

    def test_old_whole_generation_record_is_rejected_without_policy_backfill(self):
        from scoped_workflow_smoke import ScopedWorkflowTests
        fixture=ScopedWorkflowTests(methodName='runTest');fixture.setUp();self.addCleanup(fixture.tearDown)
        run=fixture.run_fixture(True);run['task_plan'].pop('editing_initialization')
        run['task_plan']['layer_workflow']={'graph_id':'old-initial-whole-signature'}
        before=copy.deepcopy(run)
        with self.assertRaisesRegex(ValueError,'Unsupported retired creation'):build_graph(run)
        self.assertEqual(run,before)


if __name__=='__main__':unittest.main()
