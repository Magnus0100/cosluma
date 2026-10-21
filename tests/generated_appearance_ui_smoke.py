"""Saved generated canvases stay readable without source-space editing or inference."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import qt_workbench as qt
from design_reconstruction_ui_smoke import Window, Pool
from generated_appearance_render_smoke import ScopedRenderTests
from generated_appearance_render import render_candidate
from native_detail import saved_native_detail
from export_artifact import export_source_run
from asset_identity import sha256_file


def main():
    app = qt.QApplication.instance() or qt.QApplication([])
    fixture = ScopedRenderTests()
    fixture.setUp()
    try:
        run = fixture.fixture()
        scope = run['task_plan']['editing_scope']
        request = {key: scope[key] for key in ('work_type', 'style', 'target_setting', 'scene_target', 'language')}
        run.update(root=str(fixture.root), input_mode='single_source', execution_strategy='layered_reference_transfer',
            task_intent={'schema_version': 'cosplay_task_intent.v1', 'mode': 'create_reference', 'brief': 'Improve original scene',
                'constraints': {'background_mode': 'enhance_source', 'preserve_background': False, 'no_effects': False},
                'services': {'text_vision': True, 'visual_review': False}})
        run['options'].update(editing_request=request, execution_strategy='layered_reference_transfer')
        from scoped_workflow import SOURCE_NATIVE_INITIALIZATION
        from cos_node_registry import compile_workflow
        import photo_appearance
        run['task_plan'].update(editing_initialization=copy.deepcopy(SOURCE_NATIVE_INITIALIZATION),
            editing_request=request,editing_autofinish={'maximum_improvements':0},image_budget={'initial_reference':1},
            layer_workflow=compile_workflow('create_reference',editing_scope=scope,
                editing_initialization=SOURCE_NATIVE_INITIALIZATION,editing_autofinish={'maximum_improvements':0}))
        baseline=photo_appearance.create_source_baseline(fixture.root/'baseline',source_path=run['source']['path'],
            source_sha256=run['source']['sha256'],scope=scope,design_id=run['generation_design']['observed_design']['design_id'])
        receipt=photo_appearance.validate_asset(baseline,source_sha256=run['source']['sha256'],scope=scope,
            design_id=run['generation_design']['observed_design']['design_id'])
        run['initial_source_appearance']=baseline;run['options']['photo_appearance']=baseline
        run['reference']={'path':str(Path(baseline['path'])/'preview.png'),'sha256':receipt['files']['preview']['sha256']}
        result = render_candidate(fixture.root, run, fixture.args)
        fixture.save(run, result)
        before = {key: sha256_file(path) for key, path in result['outputs'].items() if Path(path).is_file()}
        frozen = copy.deepcopy(run['options']['photo_appearance'])
        with patch('sys.excepthook') as unhandled, \
             patch('float_appearance.render_source_float', side_effect=AssertionError('old source renderer')), \
             patch('generated_appearance_render.render_candidate', side_effect=AssertionError('re-render')):
            window = Window(fixture.root, 'http://127.0.0.1:1'); window.pool = Pool()
            window.simple_creation = {'source': str(Path(run['source']['path']).resolve()),
                                      'mode': 'create_cos', 'stage': 'selecting_variant'}
            window.load_source_run(run); window.show(); app.processEvents()
            assert window.source_generated_appearance
            assert window.simple_creation is None
            assert not any(getattr(task, 'label', '') in {'portrait-beauty-job', 'design-reconstruct-job'} for task in window.pool.tasks)
            assert window.previews['final'].path == Path(result['outputs']['auto_draft'])
            assert window.source_light_draft_path == Path(result['outputs']['auto_draft'])
            assert window._session_binding is None and not window._session_dirty and not window._session_checkpoint_failed
            assert not window.local_editor_tabs.isEnabled() and window.workflow_canvas.isEnabled()
            assert window.workflow_canvas.graph['run_id'] == run['run_id']
            assert window.workflow_canvas.graph['readonly']
            assert 'source_native_baseline' in window.workflow_canvas.node_items
            assert 'generate_edit' not in window.workflow_canvas.node_items
            assert 'beauty' not in window.workflow_canvas.node_items
            assert window.appearance.isReadOnly() and window.manual_layers.isReadOnly()
            assert not any(slider.isEnabled() for slider in window.appearance_sliders.values())
            assert not window.portrait_beauty_prepare.isEnabled()
            assert not window.simple_workspace.save_button.isEnabled()
            assert not window.simple_workspace.beauty_button.isEnabled()
            assert window.simple_workspace.export_button.isEnabled()
            assert '已保存' in window.simple_workspace.preparation_note.text()
            assert run['generation_design']['observed_design']['summary'] in window.simple_workspace.direction_summary.text()
            count = len(window.pool.tasks)
            window.preview_source_lights(); window.save_source_lights(); window.adjust_source_run()
            window.task('POST', f'/api/source-runs/{run["run_id"]}/preview', {}, 'source-light-preview:blocked')
            assert len(window.pool.tasks) == count
            window._checkpoint_session(); assert not window._session_dirty and not window._session_checkpoint_failed
            window.export_directory.setText(str(fixture.root/'exports'))
            window.export_format.setCurrentText('PNG')
            window.simple_workspace.export_button.click()
            export_task = window.pool.tasks[-1]
            assert export_task.path.endswith('/export')
            exported = export_source_run(fixture.root, run['run_id'], **export_task.body)
            assert sha256_file(exported['path']) == before['auto_draft']
            window.request_native_detail([.5, .5])
            detail_task = window.pool.tasks[-1]
            assert detail_task.path.endswith('/native-detail')
            detail = saved_native_detail(fixture.root, run['run_id'], detail_task.body)
            with Image.open(detail['preview_path']) as crop:
                assert crop.size == (72, 96)
            assert detail['saved_version_only'] and not detail['unsaved_edits_included']
            assert run['options']['photo_appearance'] == frozen
            for key, sha in before.items(): assert sha256_file(result['outputs'][key]) == sha

            # Records lacking a checked route must not crash a Qt callback or
            # acquire a route from the next-task default.
            incomplete = copy.deepcopy(run)
            incomplete.pop('execution_strategy'); incomplete.pop('task_intent')
            incomplete['options'].pop('execution_strategy')
            window.result('job-run-loaded', {'run': incomplete})
            app.processEvents()
            assert not window.workflow_canvas.isEnabled()
            assert window.previews['final'].path == Path(result['outputs']['auto_draft'])
            from portrait_beauty_workbench_smoke import run_fixture
            legacy, _, _, _, _ = run_fixture(fixture.root/'legacy')
            window.load_source_run(legacy)
            assert not window.source_generated_appearance
            assert window.local_editor_tabs.isEnabled()
            assert window.portrait_beauty_prepare.isEnabled()
            assert not window.appearance.isReadOnly()
            assert window.simple_workspace.strength_group.isEnabled()
            unhandled.assert_not_called()
            window.session_timer.stop(); window.close(); app.processEvents()
    finally:
        fixture.tearDown()
    print(json.dumps({'ok': True, 'typed_native_pixels_reopen': True,
        'source_space_tools_disabled_before_edit': True, 'saved_detail_and_exact_export': True,
        'missing_canvas_route_readable_without_invention': True, 'uncaught_qt_exceptions': 0,
        'network_requests': 0, 'model_inferences': 0}))


if __name__ == '__main__': main()
