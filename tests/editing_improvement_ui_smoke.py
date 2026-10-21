"""A reviewed saved draft uses the checked task plan/start path for improvement.

Real CPU planning, publication, assets and review receipts are produced by the
shared fixture. External image/vision boundaries are mocked; this is not an
artistic acceptance or a live-service check.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import patch

import qt_workbench as qt
import creative_task
from asset_identity import sha256_file
from design_reconstruction_ui_smoke import Window, Pool
from scoped_production_smoke import ScopedProductionTests
from single_source_workflow import copy_source_run


def main():
    app = qt.QApplication.instance() or qt.QApplication([])
    fixture = ScopedProductionTests()
    fixture.setUp()
    try:
        _, parent = fixture.start(fixture.request(fixture.sources[0],
            character_name='Fixture character', character_notes='Original known role facts'))
        def passed_review(*args,**kwargs):
            payload,provenance=fixture.vision_request(*args,**kwargs)
            if Path(args[0]).name=='review-board.png' and 'credibility' in payload:
                payload['credibility']={'status':'pass','findings':['Synthetic baseline credibility fixture passed.']}
            return payload,provenance
        fixture.vision.side_effect=passed_review
        passed_checked=fixture.backend.task_plan(fixture.request(fixture.sources[0],run_id='ui-passed-parent'))
        passed_result=fixture.backend.start_task({**passed_checked['request'],
            'plan_id':passed_checked['plan']['plan_id'],'async':False})
        assert passed_result.get('ok'),passed_result
        passed_parent=passed_result['run']
        fixture.vision.side_effect=fixture.vision_request
        passed_manifest=fixture.root/'outputs/runs'/passed_parent['run_id']/'manifest.json'
        passed_bytes=passed_manifest.read_bytes()
        manifest = fixture.root/'outputs/runs'/parent['run_id']/'manifest.json'
        original_manifest = manifest.read_bytes()
        draft = Path(parent['artifacts']['auto_draft']); original_pixels = sha256_file(draft)
        boundary_counts = (len(fixture.image_requests), len(fixture.vision_requests))
        with patch('sys.excepthook') as unhandled, patch.object(qt.QInputDialog,'getMultiLineText',return_value=('',True)) as feedback_dialog:
            window = Window(fixture.root, 'http://127.0.0.1:1'); window.pool = Pool()
            window.load_source_run(parent); window.show(); app.processEvents()
            button = window.editing_improve_buttons[-1]
            assert button.isEnabled() and not button.isHidden()
            assert '原片' in button.toolTip() and '费用' in button.toolTip()
            assert not window.portrait_beauty_prepare.isEnabled()
            assert window.creative_task_mode.currentData() == 'create_reference'
            button.click(); pending = window.pool.tasks[-1]
            assert pending.path == '/api/tasks/plan'
            body = pending.body
            assert body['source'] == parent['source']['path']
            assert body['reference'] == parent['artifacts']['auto_draft']
            assert body['parent_run_id'] == parent['run_id']
            assert body['editing_iteration'] == {'maximum_improvements': 1, 'reason': '根据独立审片纠正主要缺陷'}
            assert body['creative_task']['mode'] == 'existing_reference'
            assert body['creative_task']['services'] == {'text_vision': True, 'visual_review': True}
            assert not {'reference_role', 'refinement_profile'} & body['creative_task']['constraints'].keys()
            assert 'regional_refinement' not in body['creative_task']
            assert body['provider'] == parent['options']['provider']
            assert body['model'] == parent['options']['model']
            assert body['model_registry'] == parent['options']['model_registry']
            assert body['character_name'] == 'Fixture character'
            assert body['character_notes'] == 'Original known role facts'
            assert window.product_run['run_id'] == parent['run_id']
            assert not button.isEnabled()
            count = len(window.pool.tasks); window.start_planned_creative_task()
            assert len(window.pool.tasks) == count
            checked = fixture.backend.task_plan(body)
            assert checked['plan']['ready'], checked['plan']['blocking_reasons']
            assert sum(checked['plan']['image_budget'].values()) == 1
            assert checked['plan']['services']['text_vision_requests_upper_bound'] == 2
            pending.signals.result.emit(checked)
            assert window.creative_start_button.isEnabled()
            assert '按独立审片改进' in window.creative_plan_summary.toPlainText()
            assert '总上限：1' in window.creative_plan_summary.toPlainText()
            assert '请求上限：2' in window.creative_plan_summary.toPlainText()
            assert (len(fixture.image_requests), len(fixture.vision_requests)) == boundary_counts
            window.creative_start_button.click(); started = window.pool.tasks[-1]
            assert started.path == '/api/tasks/start' and started.body['async'] is True
            assert started.body['parent_run_id'] == parent['run_id']
            assert started.body['sample'] == parent['sample']
            execution = creative_task.execution_request(fixture.root, started.body)
            assert execution['editing_iteration']['maximum_improvements'] == 1
            started.signals.error.emit('Captured start; no actual production submission')
            assert manifest.read_bytes() == original_manifest and sha256_file(draft) == original_pixels

            # A different selected version invalidates a pending parent-specific plan.
            window.load_source_run(parent)
            button.click(); late = window.pool.tasks[-1]
            late_checked = fixture.backend.task_plan(late.body)
            child = copy_source_run(fixture.root, parent['run_id'])
            window.load_source_run(child, selection_explicit=True)
            late.signals.result.emit(late_checked)
            assert window.creative_task_plan is None and not window.creative_start_button.isEnabled()
            assert window.product_run['run_id'] == child['run_id']

            # A changed review cannot use a checked request that belonged to older evidence.
            window.load_source_run(parent)
            button.click(); pending = window.pool.tasks[-1]
            checked = fixture.backend.task_plan(pending.body); pending.signals.result.emit(checked)
            window.product_run['photo_edit_session']['independent_evaluation']['primary_issue'] = 'Changed evidence'
            count = len(window.pool.tasks); window.start_planned_creative_task()
            assert len(window.pool.tasks) == count and window.creative_task_plan is None

            window.load_source_run(parent)
            window._session_dirty = True; window.update_editing_improvement_actions()
            assert not button.isEnabled()
            count = len(window.pool.tasks); window.plan_editing_improvement()
            assert len(window.pool.tasks) == count and window._session_dirty
            window._session_dirty = False
            window.render_pending = True; window.update_editing_improvement_actions()
            assert not button.isEnabled()
            window.render_pending = False
            prior_status = window.product_run['photo_edit_session']['status']
            window.product_run['photo_edit_session']['status'] = 'needs_attention'
            window.update_editing_improvement_actions()
            assert not button.isEnabled() and '核查' in button.toolTip()
            window.product_run['photo_edit_session']['status'] = prior_status
            window.product_run['editing_iteration_result'] = {'failure': {'type': 'TimeoutError'}}
            window.update_editing_improvement_actions()
            assert not button.isEnabled() and '不会重复提交' in button.toolTip()
            window.product_run.pop('editing_iteration_result')
            window.product_run['result_publication']['state'] = 'pending'
            window.update_editing_improvement_actions(); assert not button.isEnabled()
            window.product_run['result_publication']['state'] = 'committed'
            window.load_source_run(passed_parent)
            window.update_editing_improvement_actions()
            assert button.isEnabled() and not button.isHidden() and '我的反馈' in button.text()
            count=len(window.pool.tasks);button.click()
            assert len(window.pool.tasks)==count and '填写' in window.operation_state.text()
            feedback_dialog.return_value=('This feedback is cancelled.',False)
            button.click();assert len(window.pool.tasks)==count
            feedback='保留原来的神态，改善人物与环境的光色联系。'
            feedback_dialog.return_value=('  '+feedback+'  ',True)
            button.click();pending=window.pool.tasks[-1]
            assert pending.path=='/api/tasks/plan'
            assert pending.body['editing_iteration']['user_feedback']==feedback
            checked=fixture.backend.task_plan(pending.body)
            assert checked['plan']['ready']
            assert checked['plan']['image_budget']['editing_refinement']==1
            assert checked['plan']['services']['text_vision_requests_upper_bound']==2
            pending.signals.result.emit(checked)
            assert feedback in window.creative_plan_summary.toPlainText()
            assert '历史基线' in window.creative_plan_summary.toPlainText()
            assert '复用父版本已冻结' in window.creative_plan_summary.toPlainText()
            window.creative_start_button.click();started=window.pool.tasks[-1]
            assert started.path=='/api/tasks/start'
            execution=creative_task.execution_request(fixture.root,started.body)
            assert execution['editing_iteration']['user_feedback']==feedback
            started.signals.error.emit('Captured feedback start; no actual production submission')
            assert passed_manifest.read_bytes()==passed_bytes
            assert (len(fixture.image_requests),len(fixture.vision_requests))==boundary_counts
            window.load_source_run(parent)
            feedback_dialog.return_value=('',True)
            window.product_run['photo_edit_session'].pop('independent_evaluation')
            window.update_editing_improvement_actions(); assert button.isHidden()
            saved_improvement = copy.deepcopy(parent)
            saved_improvement['task_intent']['mode'] = 'existing_reference'
            window.load_source_run(saved_improvement)
            controls = window.editing_scope_panels[-1]
            assert controls['work_type'].currentData() == parent['task_plan']['editing_scope']['work_type']
            assert not controls['work_type'].isEnabled()
            assert '保存版本' in controls['note'].text() and '暂不支持' not in controls['note'].text()
            assert manifest.read_bytes() == original_manifest and sha256_file(draft) == original_pixels
            assert (len(fixture.image_requests), len(fixture.vision_requests)) == boundary_counts
            unhandled.assert_not_called()
            window.session_timer.stop(); window.close(); app.processEvents()
    finally:
        fixture.doCleanups()
    print(json.dumps({'ok': True, 'real_checked_parent_and_review_fixture': True,
        'explicit_plan_then_start': True, 'one_image_two_vision_budget': True,
        'stale_parent_or_review_cannot_start': True, 'busy_unsaved_uncommitted_gated': True,
        'passed_parent_requires_explicit_feedback': True, 'cancelled_feedback_has_no_request': True,
        'parent_pixels_and_manifest_preserved': True, 'actual_network_requests': 0,
        'actual_model_inferences': 0}))


if __name__ == '__main__': main()
