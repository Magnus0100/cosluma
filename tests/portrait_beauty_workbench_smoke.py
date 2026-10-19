#!/usr/bin/env python3
"""Beauty preparation/edit/save desktop lifecycle, using CPU assets and captured API tasks."""
from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PIL import Image

import qt_workbench as qt
from current_stack_fixture import prepare_fixture
from design_reconstruction_ui_smoke import Pool, Window, last, write_run
from display_transform import source_transform_for_precision, source_transform_for_run
from hsl_adjust import empty_hsl_adjust
from portrait_beauty_layer import freeze_prepared, sha256
from raw_support import RGB_DECODE_VERSION
from session_draft import SessionDraftStore, digest


def run_fixture(root, *, source_only=True):
    root.mkdir(parents=True, exist_ok=True)
    (root/'project.json').write_text(json.dumps({'kind': 'project', 'workflow_mode': 'single_source'}), encoding='utf-8')
    run = prepare_fixture(root/'assets', native=(32, 48))
    run.update(run_id='beauty-parent', status='needs_visual_review')
    run['task_intent'] = {'mode': 'local_portrait' if source_only else 'existing_reference'}
    run['source']['oriented_size'] = [32, 48]
    hsl = empty_hsl_adjust(); hsl['enabled'] = False
    run['options']['manual_layers'] = {'hsl_adjust': hsl}
    run['options']['appearance_revision']['params'] = {'global': {'exposure': .1}}
    baseline = run['options']['appearance_revision']['design_reconstruction_plan']['source_baseline']
    baseline.update(source_sha256=run['source']['sha256'], native_size=[32, 48],
                    source_display=source_transform_for_precision('raster8_srgb'))
    if source_only:
        run.pop('reference')
        run['options']['appearance_revision'].pop('design_reconstruction_plan')
        run['options']['source_only_plan'] = {
            'schema': 'source_only_display_plan.v1', 'working_domain': 'display_linear_sRGB',
            'source_sha256': run['source']['sha256'], 'native_size': [32, 48],
            'source_precision': 'raster8_srgb', 'decode_version': RGB_DECODE_VERSION,
            'source_display': source_transform_for_precision('raster8_srgb'),
            'source_display_applied_once': True, 'reference_assets_used': False}
    final, preview = root/'final.png', root/'preview.png'
    Image.new('RGB', (32, 48), '#697585').save(final)
    Image.new('RGB', (32, 48), '#b58a80').save(preview)
    run['artifacts'] = {'auto_draft': str(final), 'render_scene_master': str(root/'assets'/'fp.npz')}
    if not source_only:
        run['artifacts']['reference_candidate'] = run['reference']['path']
    write_run(root, run)
    pixels, weights = root/'generated.png', root/'weights.npy'
    Image.new('RGB', (10, 12), '#bf8a72').save(pixels)
    np.save(weights, np.ones((12, 10), np.float32))
    prepared = {'schema': 'beautygrpo_layer_prepare.v1', 'working_domain': 'display_linear_sRGB',
        'source_path': run['source']['path'], 'source_sha256': run['source']['sha256'],
        'native_size': [32, 48], 'source_display': source_transform_for_run(run),
        'source_precision': {'source_precision': 'raster8_srgb', 'decode_version': RGB_DECODE_VERSION},
        'model_identity': {'name': 'BeautyGRPO', 'lora_active': True},
        'faces': [{'id': identifier, 'native_box': box,
            'generated_native_roi': str(pixels), 'generated_native_roi_sha256': sha256(pixels),
            'composition_weight': str(weights), 'composition_weight_sha256': sha256(weights)}
            for identifier, box in [('face-01', [5, 6, 15, 18]), ('face-02', [17, 22, 27, 34])]]}
    frozen_path, frozen = freeze_prepared(prepared, root/'frozen')
    config = {'result': str(frozen_path), 'strength': 1., 'disabled_faces': []}
    return run, config, frozen['faces'], final, preview


def main():
    assert qt.QT_AVAILABLE
    app = qt.QApplication.instance() or qt.QApplication([])
    with tempfile.TemporaryDirectory(prefix='cosplay-beauty-ui-') as temporary:
        root = Path(temporary)
        run, config, faces, final, preview = run_fixture(root/'source-only')
        project = root/'source-only'
        window = Window(project, 'http://127.0.0.1:1'); pool = Pool(); window.pool = pool
        window.load_source_run(run); window.show(); app.processEvents()
        assert window._session_binding and not window._session_checkpoint_failed
        assert window.portrait_beauty_prepare.isEnabled() and not window.source_run_has_reference
        assert window.portrait_beauty_strength.value() == 100
        assert not window.portrait_beauty_strength.isEnabled() and window.portrait_beauty is None

        def previews():
            return [task for task in pool.tasks if getattr(task, 'label', '').startswith('source-light-preview:')]

        def submit(identifier):
            window.source_light_timer.stop()
            snapshot = copy.deepcopy(window._source_edit_payload())
            window.portrait_beauty_prepare.click()
            request = last(pool, 'portrait-beauty-job')
            assert request.path == f"/api/source-runs/{window.source_run_id}/prepare-portrait-beauty"
            assert request.method == 'POST' and request.body == snapshot
            assert set(snapshot) == {'appearance_revision', 'manual_layers'}
            assert not window.portrait_beauty_prepare.isEnabled()
            request.signals.result.emit({'ok': True, 'async': True, 'job_id': identifier})
            assert window.active_job_id == identifier
            return snapshot

        def finish(identifier, snapshot, *, status='succeeded', change=False, override=None):
            receipt = {'schema': 'portrait_beauty_job_result.v1', 'preparation_contract': 'portrait_beauty_layer.v1',
                'run_id': window.source_run_id, 'job_id': identifier, 'source_sha256': run['source']['sha256'],
                'input_sha256': digest(snapshot), 'appearance_revision': {'portrait_beauty': copy.deepcopy(config)},
                'faces': copy.deepcopy(faces)}
            if change:
                receipt.update(no_change=True, reason='no_visible_faces_available', appearance_revision={}, faces=[])
            receipt.update(override or {})
            value = {'job': {'kind': 'prepare_portrait_beauty', 'job_id': identifier, 'status': status},
                'portrait_beauty_result': receipt}
            window.result('job-status', value)
            return value

        initial = copy.deepcopy(window._source_edit_payload())
        before = len(previews()); version = window.source_light_version
        snapshot = submit('no-face'); finish('no-face', snapshot, change=True)
        assert window.portrait_beauty is None and not window._session_dirty
        assert len(previews()) == before and window.source_light_version == version
        for status in ('cancelled', 'failed'):
            snapshot = submit(status)
            window.result('job-status', {'job': {'kind': 'prepare_portrait_beauty', 'job_id': status, 'status': 'running'}})
            assert not window.portrait_beauty_strength.isEnabled()
            finish(status, snapshot, status=status)
            assert window._source_edit_payload() == initial and window.portrait_beauty_prepare.isEnabled()
            assert '当前编辑保留' in window.operation_state.text()
        for identifier, override in [('stale-edit', {'input_sha256': 'other-edit'}),
                                     ('stale-source', {'source_sha256': 'other-source'}),
                                     ('stale-run', {'run_id': 'other-run'})]:
            snapshot = submit(identifier); finish(identifier, snapshot, override=override)
            assert window.portrait_beauty is None and not window._session_dirty
            assert len(previews()) == before
        window.portrait_beauty_prepare.click(); failure = last(pool, 'portrait-beauty-job')
        failure.signals.error.emit('local worker unavailable')
        assert window.portrait_beauty is None and window.portrait_beauty_prepare.isEnabled()
        assert '当前编辑保留' in window.portrait_beauty_state.text()

        snapshot = submit('prepared'); immutable = copy.deepcopy(config)
        receipt = finish('prepared', snapshot)
        assert config == immutable and window.portrait_beauty == config
        assert window.source_run_id == run['run_id'] and window._session_dirty
        assert window.portrait_beauty_strength.value() == 100
        assert window.portrait_beauty_face.count() == 2 and not window.portrait_beauty_restore_face.isChecked()
        assert '整张人脸恢复，不是画笔' in window.portrait_beauty_scope.text()
        assert not window.export_project_button.isEnabled() and not any(getattr(task, 'label', '') == 'source-adjust' for task in pool.tasks)
        assert len(previews()) == before + 1
        preview_request = previews()[-1]
        assert preview_request.body['appearance_revision']['portrait_beauty'] == config
        assert preview_request.body['manual_layers'] == snapshot['manual_layers']
        assert preview_request.body['appearance_revision']['params'] == snapshot['appearance_revision']['params']
        window.result('job-status', receipt)
        assert len(previews()) == before + 1, 'Repeated receipt must not prepare another preview'
        preview_request.signals.result.emit({'run_id': run['run_id'], 'request_id': preview_request.body['request_id'], 'preview_path': str(preview)})
        assert window.previews['final'].path == preview
        view = window.previews['final']; view.zoom_slider.setMaximum(5000); view.zoom_slider.setValue(5000)
        app.processEvents(); view.scroll.horizontalScrollBar().setValue(17); view.scroll.verticalScrollBar().setValue(23)
        viewport = (view.zoom, view.scroll.horizontalScrollBar().value(), view.scroll.verticalScrollBar().value())
        window.portrait_beauty_compare.setChecked(True)
        assert window.source_light_compare.isChecked() and view.path == final
        assert (view.zoom, view.scroll.horizontalScrollBar().value(), view.scroll.verticalScrollBar().value()) == viewport
        window.source_light_compare.setChecked(False)
        assert not window.portrait_beauty_compare.isChecked() and view.path == preview
        assert (view.zoom, view.scroll.horizontalScrollBar().value(), view.scroll.verticalScrollBar().value()) == viewport
        for value in (0, 47, 100):
            window.portrait_beauty_strength.setValue(value); window.source_light_timer.stop()
            assert window._source_edit_payload()['appearance_revision']['portrait_beauty']['strength'] == value/100
        window.portrait_beauty_face.setCurrentIndex(1); window.portrait_beauty_restore_face.setChecked(True)
        window.source_light_timer.stop()
        assert window.portrait_beauty['disabled_faces'] == ['face-02']
        window.portrait_beauty_face.setCurrentIndex(0)
        assert not window.portrait_beauty_restore_face.isChecked()
        edited = copy.deepcopy(window.portrait_beauty)
        before = len(previews()); version = window.source_light_version
        snapshot = submit('existing-no-face'); finish('existing-no-face', snapshot, change=True)
        assert window.portrait_beauty == edited and window._session_dirty
        assert len(previews()) == before and window.source_light_version == version
        # Unsaved close/reopen restores the component without invoking the model.
        binding = copy.deepcopy(window._session_binding); window.close(); app.processEvents()
        assert SessionDraftStore(project).path(binding).exists()
        reopened = Window(project, 'http://127.0.0.1:1'); reopened.pool = Pool(); reopened.load_source_run(run)
        assert reopened.portrait_beauty == edited and reopened._session_dirty
        assert reopened.portrait_beauty_face.count() == 2
        reopened.source_light_timer.stop()
        reopened.portrait_beauty_strength.setValue(47); reopened.source_light_timer.stop()
        edited = copy.deepcopy(reopened.portrait_beauty)
        reopened.save_source_lights(); saved = last(reopened.pool, 'source-adjust')
        assert saved.path.endswith('/adjust') and saved.body['appearance_revision']['portrait_beauty'] == edited
        assert saved.body['manual_layers'] == initial['manual_layers']
        child = copy.deepcopy(run); child.update(run_id='beauty-child', revision=2, parent_run_id=run['run_id'])
        child['options']['appearance_revision'] = copy.deepcopy(saved.body['appearance_revision'])
        child['options']['manual_layers'] = copy.deepcopy(saved.body['manual_layers'])
        child['render'] = {'run_id': child['run_id'], 'revision': 2, 'operator': {'render_mode': 'native_source_design'}}
        write_run(project, child); saved.signals.result.emit({'ok': True, 'run': child})
        assert reopened.source_run_id == child['run_id'] and not reopened._session_dirty
        assert not SessionDraftStore(project).path(binding).exists()
        reopened.close()
        saved_window = Window(project, 'http://127.0.0.1:1'); saved_window.pool = Pool(); saved_window.load_source_run(child)
        assert saved_window.portrait_beauty == edited and saved_window.portrait_beauty_strength.value() == 47
        assert not saved_window._session_dirty and not saved_window._session_checkpoint_failed
        saved_window.portrait_beauty_face.setCurrentIndex(1)
        assert saved_window.portrait_beauty_restore_face.isChecked()
        captured = pool.tasks + reopened.pool.tasks + saved_window.pool.tasks
        assert not any(getattr(task, 'path', '') == '/api/one-click' or getattr(task, 'path', '').endswith(('/retry', '/prepare', '/recognize')) for task in captured)
        assert sum(getattr(task, 'label', '') == 'source-adjust' for task in captured) == 1
        saved_window.close()

        # The same component also loads on an imported/generated-reference native run.
        paired_root = root/'paired'
        paired, paired_config, _, _, _ = run_fixture(paired_root, source_only=False)
        paired['options']['appearance_revision']['portrait_beauty'] = paired_config
        write_run(paired_root, paired)
        paired_window = Window(paired_root, 'http://127.0.0.1:1'); paired_window.pool = Pool(); paired_window.load_source_run(paired)
        assert paired_window.source_run_has_reference and paired_window.portrait_beauty == paired_config
        assert paired_window._session_binding and not paired_window._session_checkpoint_failed
        paired_window.close(); app.processEvents()
    print(json.dumps({'ok': True, 'source_only_and_reference_routes': True,
        'model_original_appearance_default_100': True, 'draft_only_prepare_then_explicit_save': True,
        'stale_failure_no_face_preserve_edits': True, 'strength_and_per_face_restore': True,
        'same_view_before_after_preserves_viewport': True, 'saved_and_unsaved_reopen': True,
        'cpu_synthetic_assets_captured_transport': True, 'artistic_acceptance_tested': False}))


if __name__ == '__main__':
    main()
