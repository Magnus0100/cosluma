"""Default workspace, shared drafts, and automatic preparation chain (CPU transport)."""
from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import qt_workbench as qt
from cos_workflow_graph import preset
from design_reconstruction_ui_smoke import Window, Pool, last
from portrait_beauty_workbench_smoke import run_fixture
from session_draft import digest


def plan_ready(window, pool):
    context = copy.deepcopy(window.creative_task_plan_request)
    assert context is not None
    request = last(pool, context['label'])
    request.signals.result.emit({'ok': True, 'plan': {'ready': True, 'plan_id': 'cpu-plan',
                                 'mode': context['request']['creative_task']['mode']},
                                 'request': context['request']})
    started = last(pool, context['callback'])
    assert started.path == '/api/tasks/start' and started.body['plan_id'] == 'cpu-plan'
    return started


def beauty_ready(window, pool, config, faces, identifier):
    request = last(pool, 'portrait-beauty-job')
    request.signals.result.emit({'async': True, 'job_id': identifier})
    window.result('job-status', {'job': {'kind': 'prepare_portrait_beauty',
        'job_id': identifier, 'status': 'succeeded'}, 'portrait_beauty_result': {
        'run_id': window.source_run_id, 'source_sha256': window.source_run_source_sha256,
        'input_sha256': digest(request.body), 'preparation_contract': 'portrait_beauty_layer.v1',
        'appearance_revision': {'portrait_beauty': config}, 'faces': faces}})


def main():
    assert '未通过' in qt.post_completion_message({'system_post_completion': {'status': 'completed_with_findings'}})
    assert '未新增独立特效' in qt.post_completion_message({'system_post_completion': {'stages': [{'id': 'independent_effects', 'status': 'skipped'}]}})
    assert '未完成' in qt.post_completion_message({'status': 'blocked'})
    app = qt.QApplication.instance() or qt.QApplication([])
    with tempfile.TemporaryDirectory(prefix='cosplay-simple-integration-') as temporary:
        root = Path(temporary)
        run, config, faces, final, preview = run_fixture(root/'local')
        window = Window(root/'local', 'http://127.0.0.1:1'); pool = Pool(); window.pool = pool
        window.show(); app.processEvents()
        assert window.tabs.currentIndex() == window.simple_index and not window.professional_mode
        assert window.navigation.isHidden()
        window.set_simple_source(run['source']['path'])
        window.simple_workspace.beauty_button.click()
        assert window.creative_task_plan_request['callback'] == 'local-only'
        assert window.simple_workspace.beauty_button.isEnabled() is False
        started = plan_ready(window, pool)
        started.signals.result.emit({'run': run})
        assert window.simple_creation['stage'] == 'beauty'
        assert last(pool, 'portrait-beauty-job').body['appearance_revision']['workflow_graph']['name'] == '仅人像美化'
        beauty_ready(window, pool, config, faces, 'beauty-local')
        assert window.portrait_beauty['strength'] == 1. and window.simple_creation is None
        assert window.tabs.currentIndex() == window.simple_index
        window.refresh_cos_workspace()
        window.simple_workspace.strength_slider.setValue(63)
        assert window.portrait_beauty['strength'] == .63
        assert window.workflow_graph['nodes']['beauty']['strength'] == .63
        before = copy.deepcopy(window._source_edit_payload())
        window.simple_workspace.professional_button.click()
        assert window.tabs.currentIndex() == window.workflow_index and window.professional_mode
        assert window._source_edit_payload() == before
        template = window.workflow_canvas.template()
        template['nodes']['beauty'] = {'enabled': True, 'strength': .4}
        template['layout']['beauty'] = [301, 25]
        window.workflow_template_changed(template)
        window.set_professional_mode(False)
        assert window.simple_workspace.strength_slider.value() == 40
        assert window._source_edit_payload()['appearance_revision']['workflow_graph']['layout']['beauty'] == [301, 25]
        assert not window.simple_workspace.export_button.isEnabled()
        window.simple_workspace.compare_button.click()
        assert window.simple_comparing
        window.simple_workspace.compare_button.click()
        assert not window.simple_comparing
        # Both previews keep the same view when inspecting facial detail.
        window.simple_workspace.source_image.set_view(2., .1, -.2)
        window.simple_workspace.source_image.viewChanged.emit(2., .1, -.2)
        assert window.simple_workspace.result_image.zoom == 2.
        assert window.simple_workspace.result_image.offset.y() == -.2
        window._checkpoint_session(); window.source_light_timer.stop()
        reopened = Window(root/'local', 'http://127.0.0.1:1'); reopened.pool = Pool()
        reopened.load_source_run(run)
        assert reopened.workflow_graph['layout']['beauty'] == [301, 25]
        assert reopened.portrait_beauty['strength'] == .4
        assert reopened.tabs.currentIndex() == reopened.simple_index

        # A professional plan remains an explicit plan; only quick-mode intent auto-starts.
        # Generated candidates have no editable native baseline yet. Loading
        # that real state must still reach the selection gate, even though
        # session binding cannot be created before reconstruction.
        candidate = copy.deepcopy(run)
        candidate.update(run_id='pending-candidates', reference=None, reference_role='candidate_pending_selection',
            status='awaiting_reference', artifacts={}, variants=[{'variant_id': 'v1', 'candidate_path': str(preview)}])
        candidate.pop('auto_draft', None); candidate.pop('render', None)
        candidate['options'].pop('source_only_plan', None)
        awaiting = Window(root/'local', 'http://127.0.0.1:1'); awaiting.pool = Pool()
        awaiting.simple_creation = {'source': str(Path(candidate['source']['path']).resolve()), 'mode': 'create_cos', 'stage': 'base'}
        awaiting.load_source_run(candidate)
        awaiting.refresh_cos_workspace()
        assert awaiting.simple_creation['stage'] == 'awaiting_variant'
        assert awaiting.simple_workspace.candidate_button.isEnabled()
        assert awaiting.simple_workspace.progress.value() == 40
        assert not awaiting.render_pending

        expert = Window(root/'local', 'http://127.0.0.1:1'); expert.pool = Pool()
        expert.request_creative_task_plan('local_portrait', source=run['source']['path'])
        ctx = expert.creative_task_plan_request
        expert.result(ctx['label'], {'ok': True, 'plan': {'ready': True, 'plan_id': 'expert-plan'}, 'request': ctx['request']})
        assert not expert.creative_task_start_pending

        paired, paired_config, paired_faces, _, _ = run_fixture(root/'scene', source_only=False)
        # A real first launch starts without project.json; fixture-created
        # identity previously hid a loss of prepared BeautyGRPO edits.
        (root/'scene'/'project.json').unlink()
        from backend_server import Backend
        Backend(root/'scene', embedded=True)
        identity = json.loads((root/'scene'/'project.json').read_text('utf8'))
        assert identity['kind'] == 'cosplay_project' and identity['workflow_mode'] == 'single_source'
        scene = Window(root/'scene', 'http://127.0.0.1:1'); scene.pool = Pool()
        scene.set_simple_source(paired['source']['path'])
        scope_controls = scene.editing_scope_panels[-1]
        scope_controls['work_type'].setCurrentIndex(scope_controls['work_type'].findData('full_composite'))
        scene.variant_count.setValue(2)  # Explicit candidate selection keeps its local editing chain.
        scene.simple_workspace.brief.setText('暖色逆光与魔法光效')
        scene.simple_workspace.create_cos_button.click()
        assert scene.creative_task_plan_request['request']['creative_task']['brief'] == '暖色逆光与魔法光效'
        plan_ready(scene, scene.pool).signals.result.emit({'run': paired})
        beauty_ready(scene, scene.pool, paired_config, paired_faces, 'beauty-scene')
        assert scene.simple_creation['stage'] == 'colour'
        refit = last(scene.pool, 'design-reconstruct-job')
        assert refit.body['appearance_revision']['portrait_beauty']['strength'] == 1.
        refit.signals.result.emit({'async': True, 'job_id': 'refit-scene'})
        scene.result('job-status', {'job': {'kind': 'native_design_reconstruction',
            'job_id': 'refit-scene', 'status': 'succeeded'}, 'reconstruction_result': {
            'run_id': scene.source_run_id, 'source_sha256': scene.source_run_source_sha256,
            'candidate_sha256': scene.source_run_candidate_sha256,
            'input_sha256': digest(refit.body), 'design_reconstruction_plan': paired['options']['appearance_revision']['design_reconstruction_plan']}})
        assert scene.simple_creation is None and not scene.render_pending
        assert scene.tabs.currentIndex() == scene.simple_index and scene._session_dirty
        assert scene._source_edit_payload()['appearance_revision']['portrait_beauty'] == paired_config
        # Fusion preparation exposes the explicit hand-off to independent
        # effects instead of silently ending the quick route.
        assert not scene.simple_workspace.post_button.isHidden()
        assert scene.simple_workspace.post_button.text() == '保存并按当前算法重制'
        scene.simple_workspace.post_button.click()
        assert scene.simple_creation['stage'] == 'post_save_wait'
        assert last(scene.pool, 'source-adjust').path.endswith('/adjust')
        assert last(scene.pool, 'source-adjust').body['appearance_revision']['portrait_beauty'] == paired_config
        # The complete-post endpoint is normally asynchronous.  Loading its
        # terminal run must close the quick route; otherwise the ordinary UI
        # stays busy forever even though the final revision is available.
        scene.simple_creation['stage'] = 'post'
        scene.result('job-run-loaded', {'run': paired})
        assert scene.simple_creation is None and not scene.render_pending

        # Default creation owns the full production in one backend job. Loading
        # its saved result must not launch the staged Beauty/TBS chain again.
        completed = copy.deepcopy(paired)
        completed['options']['appearance_revision']['portrait_beauty'] = paired_config
        single = Window(root/'scene', 'http://127.0.0.1:1'); single.pool = Pool()
        single.set_simple_source(paired['source']['path']); single.simple_action('create_cos')
        assert '检查制作计划' in single.simple_workspace.progress_detail.text()
        assert '尚未确认' in single.simple_workspace.progress.toolTip()
        request = single.creative_task_plan_request['request']
        assert request['variant_count'] == 1 and request['defer_reference_selection'] is False
        assert 'variant_styles' not in request
        submission = plan_ready(single, single.pool)
        assert single.simple_creation['stage'] == 'production'
        assert single.active_checked_task['progress_stages']==single.simple_creation['progress_stages']
        assert single._quick_progress_stages()==single.simple_creation['progress_stages']
        assert '等待任务提交' in single.simple_workspace.progress_detail.text()
        assert single._quick_progress(.5, 'workflow', 'one_click')[0] is None
        assert single._quick_progress(.4, 'workflow', 'one_click')[1] == 'workflow'
        before = len(single.pool.tasks)
        submission.signals.result.emit({'run': completed})
        single.result('one-click', {'run': completed})  # Duplicate terminal delivery stays read-only.
        assert single.simple_creation is None and not single.render_pending
        assert not any(task.label in {'portrait-beauty-job', 'design-reconstruct-job', 'source-adjust', 'one-click'}
                       for task in single.pool.tasks[before:])
        assert single.portrait_beauty == paired_config

        # The existing-reference quick route uses the same complete backend job;
        # its asynchronous saved callback must also stop without re-preparing.
        existing = Window(root/'scene', 'http://127.0.0.1:1'); existing.pool = Pool()
        existing.set_simple_source(paired['source']['path'])
        with patch.object(qt.QFileDialog, 'getOpenFileName', return_value=(paired['reference']['path'], '')):
            existing.simple_action('existing_reference')
        plan_ready(existing, existing.pool).signals.result.emit({'async': True, 'job_id': 'complete-existing'})
        assert existing.simple_creation['stage'] == 'production'
        before = len(existing.pool.tasks)
        existing.result('job-run-loaded', {'run': completed})
        existing.result('job-run-loaded', {'run': completed})
        assert existing.simple_creation is None and not existing.render_pending
        assert not any(task.label in {'portrait-beauty-job', 'design-reconstruct-job', 'source-adjust', 'one-click'}
                       for task in existing.pool.tasks[before:])

        blocked = Window(root/'scene', 'http://127.0.0.1:1'); blocked.pool = Pool()
        blocked.set_simple_source(paired['source']['path']); blocked.simple_action('beauty')
        ctx = blocked.creative_task_plan_request
        blocked.result(ctx['label'], {'ok': True, 'plan': {'ready': False, 'plan_id': 'blocked',
            'blocking_reasons': ['缺少模型']}, 'request': ctx['request']})
        assert blocked.simple_creation is None and '缺少模型' in blocked.operation_state.text()
        assert blocked.simple_workspace.beauty_button.isEnabled()

        # Editing while a quick plan is in flight must release the busy state.
        changed = Window(root/'local', 'http://127.0.0.1:1'); changed.pool = Pool()
        changed.set_simple_source(run['source']['path']); changed.simple_action('beauty')
        old_context = copy.deepcopy(changed.creative_task_plan_request)
        changed.creative_task_brief.setPlainText('改变制作要求')
        assert changed.simple_creation is None and changed.creative_task_plan_request is None
        changed.result(old_context['label'], {'ok': True, 'plan': {'ready': True, 'plan_id': 'stale'}, 'request': old_context['request']})
        assert not changed.creative_task_start_pending and changed.simple_workspace.beauty_button.isEnabled()

        # Cancelling a plan prevents its delayed response from starting production.
        changed.simple_action('beauty'); old_context = copy.deepcopy(changed.creative_task_plan_request)
        assert changed.simple_workspace.cancel_button.isEnabled()
        changed.simple_workspace.cancel_button.click()
        changed.result(old_context['label'], {'ok': True, 'plan': {'ready': True, 'plan_id': 'cancelled'}, 'request': old_context['request']})
        assert changed.simple_creation is None and not changed.creative_task_start_pending

        # A success arriving just after cancel may show the result, but never advances to colour.
        cancelled = Window(root/'scene', 'http://127.0.0.1:1'); cancelled.pool = Pool()
        scope_controls = cancelled.editing_scope_panels[-1]
        scope_controls['work_type'].setCurrentIndex(scope_controls['work_type'].findData('full_composite'))
        cancelled.variant_count.setValue(2)
        cancelled.set_simple_source(paired['source']['path']); cancelled.simple_action('create_cos')
        plan_ready(cancelled, cancelled.pool).signals.result.emit({'run': paired})
        beauty_job = last(cancelled.pool, 'portrait-beauty-job')
        beauty_job.signals.result.emit({'async': True, 'job_id': 'cancel-race'})
        cancelled.simple_action('cancel')
        assert cancelled.simple_creation is None and last(cancelled.pool, 'job-cancel').path.endswith('/cancel-race/cancel')
        cancelled.result('job-status', {'job': {'kind': 'prepare_portrait_beauty', 'job_id': 'cancel-race', 'status': 'succeeded'},
            'portrait_beauty_result': {'run_id': cancelled.source_run_id, 'source_sha256': cancelled.source_run_source_sha256,
                'input_sha256': digest(beauty_job.body), 'preparation_contract': 'portrait_beauty_layer.v1',
                'appearance_revision': {'portrait_beauty': paired_config}, 'faces': paired_faces}})
        assert not any(getattr(task, 'label', None) == 'design-reconstruct-job' for task in cancelled.pool.tasks)
        assert cancelled.simple_creation is None and not cancelled.render_pending
        pending_cancel = Window(root/'local', 'http://127.0.0.1:1'); pending_cancel.pool = Pool()
        pending_cancel.set_simple_source(run['source']['path']); pending_cancel.simple_action('beauty')
        plan_ready(pending_cancel, pending_cancel.pool).signals.result.emit({'run': run})
        pending_beauty = last(pending_cancel.pool, 'portrait-beauty-job')
        assert pending_cancel.portrait_beauty_request == 'submitting'
        pending_cancel.simple_action('cancel')
        pending_beauty.signals.result.emit({'async': True, 'job_id': 'delayed-start-cancel'})
        assert last(pending_cancel.pool, 'job-cancel').path.endswith('/delayed-start-cancel/cancel')
        assert pending_cancel.simple_creation is None and not pending_cancel.simple_cancel_submissions
        pending_cancel.result('job-status', {'job': {'kind': 'prepare_portrait_beauty',
            'job_id': 'delayed-start-cancel', 'status': 'cancelled'}})
        assert not pending_cancel.render_pending and pending_cancel.portrait_beauty_request is None
        for item in (window, reopened, expert, scene, single, existing, blocked, changed, cancelled, pending_cancel):
            item.session_timer.stop(); item.native_matte_timer.stop(); item.source_light_timer.stop(); item.close()
    print(json.dumps({'ok': True, 'default_simple_mode': True, 'shared_edit_and_reopen': True,
        'quick_plan_auto_start_only': True, 'beauty_then_reference_refit': True,
        'single_backend_production_without_duplicate_preparation': True,
        'explicit_multi_candidate_selection_retained': True,
        'synchronised_preview_zoom': True, 'cpu_captured_transport': True}))


if __name__ == '__main__': main()
