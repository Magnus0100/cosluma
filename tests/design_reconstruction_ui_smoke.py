#!/usr/bin/env python3
"""Async reference-stack UI regression with a temporary native project.

Captured tasks simulate backend responses; no model/render/network runs.
"""
from __future__ import annotations
import copy, hashlib, json, os, tempfile
from pathlib import Path
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PIL import Image
import numpy as np
import qt_workbench as qt
from session_draft import SessionDraftStore, digest
from source_appearance_responsibility import REFERENCE_STACK_STAGES

class Pool:
    def __init__(self): self.tasks = []
    def start(self, task): self.tasks.append(task)

class Window(qt.Workbench):
    def refresh(self): pass

def asset(path, size):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'size': size}

def write_run(root, run):
    run.setdefault('execution_strategy', 'layered_reference_transfer')
    run.setdefault('task_intent', {'mode':'existing_reference' if run.get('reference') else 'local_portrait'})
    path = root / 'outputs' / 'runs' / run['run_id'] / 'manifest.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run), encoding='utf-8')

def last(pool, label):
    return next(task for task in reversed(pool.tasks) if getattr(task, 'label', '') == label)

def main():
    assert qt.QT_AVAILABLE
    app = qt.QApplication.instance() or qt.QApplication([])
    with tempfile.TemporaryDirectory(prefix='cosplay-design-ui-') as temporary:
        root = Path(temporary)
        (root / 'project.json').write_text(json.dumps({'kind': 'project', 'workflow_mode': 'single_source'}), encoding='utf-8')
        from current_stack_fixture import prepare_fixture
        fixture = prepare_fixture(root/'current-assets', native=(32, 48))
        plan = copy.deepcopy(fixture['options']['appearance_revision']['design_reconstruction_plan'])
        src, ref = copy.deepcopy(fixture['source']), copy.deepcopy(fixture['reference'])
        source, candidate = Path(src['path']), Path(ref['path'])
        size = [32, 48]
        final, preview = root/'final.png', root/'preview.jpg'
        for path in (final, preview): Image.new('RGB', size, (110, 100, 90)).save(path)
        alpha = root / 'alpha.png'; Image.new('L', size, 255).save(alpha)
        fp = Path(plan['source_baseline']['assets']['foreground_premultiplied']['path'])
        plan['summary'] = 'Fixture stack'
        plan['future_field'] = {'preserve': True}
        plan['layers'][0].update(label='整体仿色融合', strength=.65, blend_mode='opaque-backend-mode')
        params = {'global': {'exposure': .1}}
        run = {'kind': 'single_input_run', 'input_mode': 'single_source', 'sample': 'fixture',
            'run_id': 'design-parent', 'revision': 1, 'status': 'needs_visual_review', 'source': src, 'reference': ref,
            'artifacts': {'reference_candidate': str(candidate), 'auto_draft': str(final),
                'alpha': str(alpha), 'render_scene_master': str(fp)},
            'options': {'render_mode': 'native_source_design', 'manual_layers': {},
                'appearance_revision': {'params': params, 'design_reconstruction_plan': plan}}}
        write_run(root, run)
        window = Window(root, 'http://127.0.0.1:1'); pool = Pool(); window.pool = pool
        window.load_source_run(run)
        assert window.source_run_id == run['run_id']
        window.show(); app.processEvents()
        assert window._session_binding and not window._session_checkpoint_failed
        assert window.local_editor_tabs.currentIndex() == 2 and window.design_reconstruct_button.isEnabled()
        submitted = copy.deepcopy(window._source_edit_payload())
        window.design_reconstruct_button.click(); request = last(pool, 'design-reconstruct-job')
        assert request.path.endswith('/reconstruct-design') and request.method == 'POST'
        assert request.body == submitted and set(request.body) == {'appearance_revision', 'manual_layers'}
        assert not window.design_reconstruct_button.isEnabled()
        job_id = 'fixture-local-prepare'
        request.signals.result.emit({'ok': True, 'async': True, 'job_id': job_id}); app.processEvents()
        assert window.active_job_id == job_id and window.native_reconstruction_job_id == job_id
        prepared = copy.deepcopy(plan); prepared['summary'] = 'New local preparation'; prepared['layers'][0]['strength'] = 1.
        immutable = copy.deepcopy(prepared)
        receipt = {'run_id': run['run_id'], 'input_sha256': digest(submitted),
            'source_sha256': src['sha256'], 'candidate_sha256': ref['sha256'], 'design_reconstruction_plan': prepared}
        job = {'job_id': job_id, 'kind': 'native_design_reconstruction', 'status': 'succeeded'}
        window.result('job-status', {'job': job, 'reconstruction_result': receipt}); app.processEvents()
        assert window.design_stack_list.count() == 2 and window.design_reconstruct_button.isEnabled()
        assert window.design_reconstruction_prepared and window._session_dirty and not window.export_button.isEnabled()
        assert (root / 'outputs' / 'session-drafts').is_dir()
        preview_request = next(task for task in reversed(pool.tasks) if task.label.startswith('source-light-preview:'))
        assert preview_request.path.endswith('/preview')
        assert preview_request.body['appearance_revision']['design_reconstruction_plan'] == prepared
        window.design_stack_list.setCurrentRow(0); window.design_layer_strength.setValue(.4); window.source_light_timer.stop()
        window.design_layer_enabled.setChecked(False); window.source_light_timer.stop()
        assert prepared == immutable, 'Do not mutate the backend response'
        preview_request.signals.result.emit({'run_id': run['run_id'], 'request_id': preview_request.body['request_id'], 'preview_path': str(preview)})
        assert window.previews['final'].path == final, 'Reject preview made before edits'
        window.preview_source_lights(); preview_request = pool.tasks[-1]
        edited = copy.deepcopy(prepared); edited['layers'][0].update(strength=.4, enabled=False)
        assert preview_request.body['appearance_revision']['design_reconstruction_plan'] == edited
        preview_request.signals.result.emit({'run_id': run['run_id'], 'request_id': preview_request.body['request_id'], 'preview_path': str(preview)})
        assert window.previews['final'].path == preview
        window.save_source_lights(); save = last(pool, 'source-adjust'); revision = save.body['appearance_revision']
        assert save.path.endswith('/adjust'), save.path
        assert revision['params'] == submitted['appearance_revision']['params'], revision['params']
        assert revision['params']['global']['exposure'] == params['global']['exposure']
        assert revision['design_reconstruction_plan'] == edited, revision['design_reconstruction_plan']
        assert set(save.body) == {'appearance_revision', 'manual_layers'}
        assert revision['native_session_submission']['parent_run_id'] == run['run_id']
        parent_binding = copy.deepcopy(window._session_binding)
        child = copy.deepcopy(run); child.update(run_id='design-child', revision=2, parent_run_id=run['run_id'])
        child['options']['appearance_revision'] = copy.deepcopy(revision)
        child['options']['manual_layers'] = copy.deepcopy(save.body['manual_layers'])
        child['render'] = {'run_id': child['run_id'], 'revision': 2, 'operator': {'render_mode': 'native_source_design'}}
        write_run(root, child)
        save.signals.result.emit({'ok': True, 'run': child}); app.processEvents()
        assert window.design_reconstruction_plan == edited and window.source_light_preview_path is None
        assert not window._session_dirty
        assert not SessionDraftStore(root).path(parent_binding).exists(), 'Consume only a verified saved child'
        window.close()

        reopened = Window(root, 'http://127.0.0.1:1'); reopened_pool = Pool(); reopened.pool = reopened_pool
        reopened.load_source_run(child)
        assert reopened.source_run_id == child['run_id']
        reopened.design_stack_list.setCurrentRow(0)
        assert reopened.design_layer_strength.value() == .4 and not reopened.design_layer_enabled.isChecked()
        assert reopened.design_reconstruction_plan == edited and not reopened._session_dirty
        reopened.reconstruct_design(); request = last(reopened_pool, 'design-reconstruct-job')
        request.signals.result.emit({'ok': True, 'async': True, 'job_id': 'fixture-stale-prepare'})
        previous = copy.deepcopy(reopened.design_reconstruction_plan)
        stale_job = {**job, 'job_id': 'fixture-stale-prepare'}
        stale_receipt = {**receipt, 'run_id': child['run_id'], 'input_sha256': 'different-edit-snapshot'}
        reopened.result('job-status', {'job': stale_job, 'reconstruction_result': stale_receipt})
        assert reopened.design_reconstruction_plan == previous and '精修已有变化' in reopened.operation_state.text()
        reopened.reconstruct_design(); failed = last(reopened_pool, 'design-reconstruct-job'); before = len(reopened_pool.tasks)
        failed.signals.error.emit('local worker unavailable')
        assert reopened.design_reconstruct_button.isEnabled() and len(reopened_pool.tasks) == before
        assert reopened.design_reconstruction_plan == previous and '当前编辑保留' in reopened.source_light_state.text()
        for captured_pool in (pool, reopened_pool):
            assert not any(getattr(task, 'path', '').endswith(('/plan-color', '/plan-light', '/auto-color', '/retry', '/prepare', '/recognize'))
                or getattr(task, 'path', '') == '/api/one-click' for task in captured_pool.tasks)
        assert reopened.root_path == root.resolve()
        reopened.close(); app.processEvents()
    print(json.dumps({'ok': True, 'async_prepare_submits_current_edit_snapshot': True,
        'prepared_result_bound_and_unsaved': True, 'opaque_assets_preserved': True,
        'stale_preview_and_prepare_rejected': True, 'verified_child_consumes_session': True,
        'saved_layers_reopen': True, 'submission_failure_preserves_edits': True,
        'temporary_project_only': True, 'no_generation_or_automatic_retry': True,
        'backend_render_and_visual_acceptance_simulated': True}))

if __name__ == '__main__': main()

