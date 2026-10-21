"""Current public CPU photography -> local generation across native layouts.

Only actual image/vision transports are synthetic. Pixel ownership, task
dispatch, review, native save, graph and timing observers execute normally.
"""
from __future__ import annotations

import copy
from pathlib import Path
import unittest

import numpy as np
from PySide6.QtWidgets import QApplication

from photo_appearance import load_asset
from photo_edit_tools import _close_loaded
from pipeline_stage_report import read_stage_report
from cos_workflow_graph import build_graph
import photo_development_session_smoke as public_fixture
from design_reconstruction_ui_smoke import Window,Pool


class DevelopmentProductTests(unittest.TestCase):
    def fixture(self):
        fx=public_fixture.DevelopmentSessionTests(methodName='runTest')
        self.addCleanup(fx.doCleanups);fx.setUp()
        return fx

    def test_two_native_layouts_cpu_then_local_generation_preserves_adjusted_outer_floats(self):
        fx=self.fixture();fx.followup='edit_region'
        fx.fx.generation.side_effect=fx.fx.image_request
        for index,source in enumerate(fx.fx.sources):
            with self.subTest(layout=index):
                fx.decisions=0
                before_images=len(fx.fx.image_requests);before_vision=len(fx.fx.vision_requests)
                checked,run=fx.start(fx.request(source=source,k=2,run_id='photographic-local-'+str(index)))
                self.assertEqual(checked['plan']['generation_submissions_upper_bound'],3)
                self.assertEqual(checked['plan']['services']['text_vision_requests_upper_bound'],8)
                self.assertEqual(len(fx.fx.image_requests)-before_images,1)
                self.assertEqual(len(fx.fx.vision_requests)-before_vision,6)
                state=fx.state(run)
                tools=[row for row in state['attempts'] if row['kind']=='tool']
                self.assertEqual([row['context']['decision']['tool'] for row in tools],['develop_photograph','edit_region'])
                self.assertEqual([row['operation_stage'] for row in tools],['initial','improvement'])
                self.assertEqual([row['response']['resources']['image_submissions'] for row in tools],[0,1])
                self.assertEqual(state['used']['improvement_steps'],1)
                photographic=next(row for row in state['candidates'] if row.get('attempt_id')==tools[0]['attempt_id'])
                binding={'source_sha256':run['source']['sha256'],'scope':run['task_plan']['editing_scope'],
                    'design_id':run['generation_design']['observed_design']['design_id']}
                developed=final=None
                try:
                    developed=load_asset(photographic['appearance_asset'],**binding)
                    final=load_asset(run['options']['photo_appearance'],**binding)
                    self.assertEqual(developed['receipt']['kind'],'photographic_adjustment')
                    self.assertEqual(final['receipt']['kind'],'mixed_regional')
                    self.assertEqual(developed['master'].dtype,np.float32)
                    self.assertEqual(developed['master'].shape,final['master'].shape)
                    outside=final['generated_coverage']==0
                    self.assertTrue(np.any(outside));self.assertTrue(np.any(~outside))
                    np.testing.assert_array_equal(developed['master'][outside],final['master'][outside])
                    self.assertTrue(final['receipt']['source_origin'])
                finally:_close_loaded(developed,final)
                graph=build_graph(run)
                self.assertTrue(graph['edited_output'])
                selected=next(row for row in graph['nodes'] if row['id']=='selected_edit_tool')
                self.assertEqual(selected['status'],'complete')
                self.assertEqual(selected['execution']['zero_image_tool_calls'],1)
                self.assertEqual(selected['execution']['actual_image_submissions'],1)
                report=read_stage_report(fx.root,run)
                self.assertEqual(report['whole_work']['execution_state'],'complete')
                self.assertTrue(report['whole_work']['automated_target_met'])
                self.assertTrue(report['whole_work']['creative_delivery_eligibility']['eligible'])
                self.assertTrue(any(row['name']=='photographic_development' and row['seconds']>=0 for row in report['timing']['modules']))

    def test_cpu_only_composite_is_saved_viewable_but_not_a_completed_creative_delivery(self):
        fx=self.fixture();_,run=fx.start(fx.request())
        before=copy.deepcopy(run)
        report=read_stage_report(fx.root,run)
        self.assertEqual(report['whole_work']['execution_state'],'complete')
        self.assertEqual(report['whole_work']['result_state'],'needs_creative_edit')
        self.assertTrue(report['whole_work']['saved_current_revision'])
        self.assertFalse(report['whole_work']['automated_target_met'])
        self.assertFalse(report['whole_work']['creative_delivery_eligibility']['eligible'])
        self.assertTrue(all(row['status']=='pass' for row in report['cosplay_post_diagnosis']['checks']))
        self.assertEqual(run,before,'Product observers cannot rewrite the actual model review')
        stage=next(row for row in report['stages'] if row['id']=='selected_edit_tool')
        self.assertEqual(stage['status'],'complete')
        self.assertEqual(stage['execution']['actual_image_submissions'],0)
        self.assertEqual(stage['execution']['zero_image_tool_calls'],1)
        self.assertGreaterEqual(stage['measured_seconds'],0)
        app=QApplication.instance() or QApplication([])
        window=Window(fx.root,'http://127.0.0.1:1');window.pool=Pool()
        try:
            window.load_source_run(run,stage_report=report);app.processEvents()
            self.assertTrue(window.source_generated_appearance)
            self.assertEqual(window.previews['final'].path,Path(run['artifacts']['auto_draft']))
            state=window.editing_improvement_state()
            self.assertTrue(state['visible']);self.assertTrue(state['enabled'])
            self.assertFalse(state['review_passed'])
            self.assertEqual(window.workflow_canvas.node_items['selected_edit_tool'].node['status'],'complete')
            self.assertEqual(len(fx.fx.image_requests),0)
            self.assertEqual(len(fx.fx.vision_requests),4)
        finally:window.session_timer.stop();window.close();app.processEvents()


if __name__=='__main__':unittest.main()
