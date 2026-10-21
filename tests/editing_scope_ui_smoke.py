"""Scope controls reach checked task start; no network, models or real photos."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from PIL import Image
import creative_task as contract
import qt_workbench as qt
from creative_task_ui_smoke import (Window, Pool, model_registry_fixture,
                                    model_capability_fixture, service_fixture)


def main():
    app = qt.QApplication.instance() or qt.QApplication([])
    with tempfile.TemporaryDirectory(prefix='cosluma-scope-ui-') as folder:
        root = Path(folder)
        source = root / 'original.png'; Image.new('RGB', (64, 96), '#485e79').save(source)
        reference = root / 'reference.png'; Image.new('RGB', (64, 96), '#748da1').save(reference)
        with patch('current_model_registry.registry_status', return_value=model_registry_fixture()), \
             patch('task_model_preflight.inspect_task_models', side_effect=model_capability_fixture), \
             patch.object(contract, '_services', side_effect=service_fixture), \
             patch('reference_provider.provider_config', return_value={'kind': 'comfyui', 'registry_id': 'qwen_image_2_1'}), \
             patch('reference_provider.provider_health', return_value={'ready': True}):
            window = Window(root, 'http://127.0.0.1:1'); window.pool = Pool()
            assert window.creative_task_mode.currentData() == 'create_reference'
            assert len(window.editing_scope_panels) == 2
            pro, quick = window.editing_scope_panels
            scope = window.editing_request_payload()
            assert scope == {'work_type': 'partial_composite', 'style': 'expressive',
                             'target_setting': 'same', 'scene_target': '', 'language': 'zh'}
            assert window.creative_text_vision.isChecked() and not window.creative_text_vision.isEnabled()
            assert window.creative_visual_review.isChecked() and quick['review'].isChecked()
            count = len(window.pool.tasks)
            window.recognize_character()
            assert not hasattr(window,'prepare_generation_design') and not hasattr(window,'prepare_design_button')
            assert len(window.pool.tasks) == count and '未提交额外规划' in window.operation_state.text()
            assert not window.creative_no_effects.isChecked()
            assert 'reference_role' not in window.creative_task_payload()['constraints']
            assert not quick['target'].isVisible()
            assert quick['target_setting'].currentData() == 'same'
            window.set_simple_source(str(source))
            window.simple_workspace.check_plan_button.click()
            planned = window.pool.tasks[-1]
            assert planned.path == '/api/tasks/plan' and planned.body['editing_request'] == scope
            assert planned.body['creative_task']['services']['text_vision'] is True
            assert planned.body['creative_task']['services']['visual_review'] is True
            checked = contract.plan_task(root, planned.body)
            assert checked['plan']['ready'] is True
            assert checked['plan']['editing_scope']['scene_policy'] == 'local_edit'
            planned.signals.result.emit(checked)
            text = window.creative_plan_summary.toPlainText()
            assert '半合成' in text and '鲜明' in text and '保持原片' in text
            assert '不直接复制参考 RGB' not in text
            window.simple_workspace.start_plan_button.click()
            started = window.pool.tasks[-1]
            assert started.path == '/api/tasks/start'
            assert started.body['editing_request'] == checked['request']['editing_request']
            assert contract.execution_request(root, started.body)['editing_request'] == scope
            started.signals.error.emit('Captured; no network submission')

            # Style changes cannot enlarge scene permissions, and stale plans cannot start.
            quick['style'].setCurrentIndex(quick['style'].findData('dramatic'))
            assert window.creative_task_plan is None
            assert pro['style'].currentData() == 'dramatic'
            assert window.editing_request_payload()['target_setting'] == 'same'
            assert window.creative_task_payload()['constraints']['background_mode'] == 'enhance_source'
            quick['review'].setChecked(True)
            assert window.creative_task_payload()['services']['visual_review'] is True
            quick['review'].setChecked(False)
            assert not pro['review'].isChecked()
            assert window.creative_task_request_options('create_reference')['creative_task']['services']['visual_review'] is False
            quick['work_type'].setCurrentIndex(quick['work_type'].findData('full_composite'))
            assert pro['work_type'].currentData() == 'full_composite'
            assert window.editing_request_payload()['target_setting'] == 'same'
            quick['target_setting'].setCurrentIndex(quick['target_setting'].findData('indoor'))
            quick['scene_target'].setText('旧戏台室内，保留现场透视')
            full = window.editing_request_payload()
            assert full['target_setting'] == 'indoor' and full['scene_target'].startswith('旧戏台')
            full_request = window.creative_task_request_options('create_reference', source=str(source))
            full_checked = contract.plan_task(root, {**full_request, 'source': str(source)})
            assert full_checked['plan']['editing_scope']['scene_policy'] == 'rebuild'
            saved = copy.deepcopy(window._template_payload())
            quick['work_type'].setCurrentIndex(quick['work_type'].findData('partial_composite'))
            assert window.editing_request_payload()['target_setting'] == 'same'
            assert window.editing_request_payload()['scene_target'] == ''
            window.template_selector.addItem('范围与审片偏好', {'parameters': saved})
            window.template_selector.setCurrentIndex(window.template_selector.count()-1)
            window.load_workbench_template()
            assert window.editing_request_payload() == full
            assert not window.creative_visual_review.isChecked()
            from generated_appearance_render_smoke import ScopedRenderTests
            from generated_appearance_render import render_candidate
            from photo_editing import validate_observed_design, compile_edit_prompt
            from argparse import Namespace
            reopened_root = root / 'saved-work'
            fixture=ScopedRenderTests();fixture.root=reopened_root
            saved_run=fixture.fixture(work='full_composite')
            current=contract.plan_task(reopened_root,{**full_request,'source':saved_run['source']['path']})
            saved_run.update(root=str(reopened_root),execution_strategy='layered_reference_transfer',
                task_intent=current['plan']['intent'],task_plan=current['plan'])
            saved_run['options'].update(editing_request=current['request']['editing_request'],
                task_intent=current['plan']['intent'],execution_strategy='layered_reference_transfer')
            scope=current['plan']['editing_scope']
            old=saved_run['generation_design']['observed_design']
            observed=validate_observed_design({key:old[key] for key in
                ('summary','observation','changes','lighting','keep','uncertain')},scope)
            compiled=compile_edit_prompt(observed,scope,'qwen_image_2_1')
            saved_run['generation_design'].update(scope_id=scope['scope_id'],observed_design=observed,
                compiled_prompt=compiled,effective_prompt=compiled['prompt'])
            saved_run['generation_design']['art_direction'].update(design=observed,
                compiled_prompts={family:compile_edit_prompt(observed,scope,family) for family in ('gpt_image','qwen_image_2_1')})
            saved_run['generation_design']['art_direction']['binding']['scope_id']=scope['scope_id']
            import photo_appearance
            baseline=photo_appearance.create_source_baseline(reopened_root/'source-baseline',
                source_path=saved_run['source']['path'],source_sha256=saved_run['source']['sha256'],
                scope=scope,design_id=observed['design_id'])
            saved_run['initial_source_appearance']=baseline
            saved_run['options']['photo_appearance']=baseline
            baseline_receipt=photo_appearance.validate_asset(baseline,source_sha256=saved_run['source']['sha256'],
                scope=scope,design_id=observed['design_id'])
            saved_run['reference']={'path':str(Path(baseline['path'])/'preview.png'),
                'sha256':baseline_receipt['files']['preview']['sha256']}
            rendered=render_candidate(reopened_root,saved_run,Namespace(cancel_request_path=None))
            assert rendered['metadata'].get('photo_appearance')==baseline
            from asset_identity import sha256_file
            assert sha256_file(rendered['outputs']['auto_draft'])==baseline_receipt['files']['preview']['sha256']
            fixture.save(saved_run,rendered)
            reopened = Window(reopened_root, 'http://127.0.0.1:1'); reopened.pool = Pool()
            reopened.load_source_run(saved_run)
            assert reopened.editing_request_payload() == full
            assert not reopened.creative_visual_review.isChecked()
            assert reopened.editing_scope_panels[-1]['scene_target'].text() == full['scene_target']
            assert reopened.source_generated_appearance and reopened.workflow_canvas.graph['readonly']
            assert 'source_native_baseline' in reopened.workflow_canvas.node_items
            assert 'generate_edit' not in reopened.workflow_canvas.node_items
            assert 'beauty' not in reopened.workflow_canvas.node_items
            assert reopened.previews['final'].path == Path(rendered['outputs']['auto_draft'])
            assert reopened.final_prompt.isReadOnly()
            assert observed['summary'] in reopened.design_summary.toPlainText()
            assert observed['lighting'] in reopened.design_summary.toPlainText()
            assert all(change['instruction'] in reopened.design_summary.toPlainText() for change in observed['changes'])
            fresh=reopened.generation_request_options()
            assert 'generation_design' not in fresh and 'character_profile' not in fresh
            assert fresh['effective_prompt']==reopened.generation_input_prompt()
            assert not hasattr(reopened,'prepared_design')
            unsupported=copy.deepcopy(saved_run);unsupported['task_plan'].pop('editing_initialization')
            previous=reopened.source_run_id
            assert reopened.load_source_run(unsupported) is False
            assert reopened.source_run_id==previous and '不支持' in reopened.operation_state.text()
            assert reopened.previews['final'].path==Path(rendered['outputs']['auto_draft'])
            reopened.session_timer.stop(); reopened.close()

            quick['work_type'].setCurrentIndex(quick['work_type'].findData('retouch'))
            assert window.creative_task_mode.currentData() == 'create_reference'
            ai_retouch=window.creative_task_request_options('create_reference',source=str(source))
            assert ai_retouch['editing_request']['work_type']=='retouch'
            assert ai_retouch['creative_task']['services']['text_vision'] is True
            checked_retouch=contract.plan_task(root,{**ai_retouch,'source':str(source)})
            assert checked_retouch['plan']['ready'],checked_retouch['plan']['blocking_reasons']
            tools=next(node['controls']['available_tools'] for node in checked_retouch['plan']['layer_workflow']['nodes'] if node['id']=='selected_edit_tool')
            assert 'develop_photograph' in tools and 'edit_region' in tools and 'joint_edit' not in tools
            assert checked_retouch['plan']['editing_autofinish']['maximum_improvements']==0
            quick['review'].setChecked(True)
            quick['autofinish'].setChecked(True)
            assert window.creative_task_request_options('create_reference')['editing_autofinish']['maximum_improvements']==2
            window.creative_task_mode.setCurrentIndex(window.creative_task_mode.findData('local_portrait'))
            local = window.creative_task_request_options('local_portrait', source=str(source))
            assert local['editing_request']['work_type'] == 'retouch'
            assert local['creative_task']['constraints']['no_effects'] is True
            assert quick['work_type'].isEnabled()
            local_checked = contract.plan_task(root, {**local, 'source': str(source)})
            assert not any(local_checked['plan']['image_budget'].values())
            assert local_checked['plan']['services']['text_vision'] is False
            assert local_checked['plan']['services']['visual_review'] is False and not quick['review'].isEnabled()

            # Imported references stay on their distinct zero-new-image route.
            window.creative_task_mode.setCurrentIndex(window.creative_task_mode.findData('existing_reference'))
            window.creative_task_reference.setText(str(reference))
            imported = window.creative_task_request_options('existing_reference', source=str(source))
            assert 'editing_request' not in imported
            assert not quick['work_type'].isEnabled() and '暂不支持' in quick['note'].text()
            existing = contract.plan_task(root, {**imported, 'source': str(source), 'reference': str(reference)})
            assert not any(existing['plan']['image_budget'].values())
            window.session_timer.stop(); window.close(); app.processEvents()
    print(json.dumps({'ok': True, 'scope_and_style_independent': True,
        'default_partial_expressive_original_setting': True, 'checked_plan_and_start_payload': True,
        'scene_change_explicit': True, 'imported_reference_limitation_visible': True,
        'network_requests': 0, 'model_inferences': 0}))


if __name__ == '__main__': main()
