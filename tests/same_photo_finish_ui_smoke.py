"""Shared same-photo opt-in, zero-budget transport and honest UI summaries."""
from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PIL import Image
import qt_workbench as qt
import creative_task as contract
from creative_task_ui_smoke import model_registry_fixture, model_capability_fixture, service_fixture
from design_reconstruction_ui_smoke import Window, Pool


def main():
    app = qt.QApplication.instance() or qt.QApplication([])
    with tempfile.TemporaryDirectory(prefix='cosplay-same-photo-ui-') as folder:
        root = Path(folder)
        source, reference, other = (root/name for name in ('source.png', 'finish.png', 'other.png'))
        for path in (source, reference, other): Image.new('RGB', (64, 96), '#476d8a').save(path)
        window = Window(root, 'http://127.0.0.1:1'); window.pool = Pool()
        window.set_simple_source(str(source)); window.project_dimensions['source'] = [64, 96]
        quick = window.simple_workspace
        assert not window.same_photo_finish_enabled and not quick.same_photo_finish.isEnabled()
        assert 'reference_role' not in window.creative_task_payload()['constraints']
        # Source + requirements can create its own design; a manual finish is not an input requirement.
        quick.task_mode.setCurrentIndex(quick.task_mode.findData('create_reference'))
        quick.brief.setText('冷蓝环境、明净肤色和清晰织纹；保留原片人物与场景结构')
        assert not window.creative_task_reference.text()
        assert window.creative_task_payload()['constraints']['reference_role'] == 'source_conditioned_design'
        assert 'regional_refinement' not in window.creative_task_payload()
        with patch('current_model_registry.registry_status', return_value=model_registry_fixture()), \
             patch('task_model_preflight.inspect_task_models', side_effect=model_capability_fixture), \
             patch.object(contract, '_services', side_effect=service_fixture), \
             patch('reference_provider.provider_config', return_value={'kind': 'comfyui', 'registry_id': 'qwen_image_2_1_uncensored_fast'}), \
             patch('reference_provider.provider_health', return_value={'ready': True}):
            quick.check_plan_button.click(); source_task = window.pool.tasks[-1]
            assert source_task.path == '/api/tasks/plan' and not source_task.body.get('reference')
            source_checked = contract.plan_task(root, source_task.body)
            plan = source_checked['plan']
            assert plan['ready'] is True and set(plan['inputs']) == {'source'}
            assert plan['intent']['constraints']['reference_role'] == 'source_conditioned_design'
            assert plan['image_budget']['initial_reference'] == 1
            assert all(value == 0 for key, value in plan['image_budget'].items() if key != 'initial_reference')
            assert plan['reference_contract']['reference_binding'] == 'after_generation'
            assert plan['reference_contract']['reference_sha256'] is None
            assert plan['reference_contract']['texture_owner'] == 'source_native'
            assert plan['reference_contract']['registration_status'] == 'not_checked'
            source_task.signals.result.emit(source_checked)
            text = window.creative_plan_summary.toPlainText()
            assert '不需要人工精修图作为输入' in text and '总上限：1' in text
            assert '新增生图 0 次' not in text and '同照精修匹配意图' not in text
            quick.start_plan_button.click(); source_start = window.pool.tasks[-1]
            assert source_start.path == '/api/tasks/start' and not source_start.body.get('reference')
            assert source_start.body['creative_task']['constraints']['reference_role'] == 'source_conditioned_design'
            source_start.signals.error.emit('Captured source-only submission; no model or network run')
        quick.task_mode.setCurrentIndex(quick.task_mode.findData('existing_reference'))
        window.creative_task_reference.setText(str(reference))
        quick.brief.setText('按同照精修标尺调整光色，保留原片身份与纹理')
        quick.no_effects.setChecked(False)
        assert quick.same_photo_finish.isEnabled() and window.creative_same_photo_finish.isEnabled()
        assert quick.same_photo_finish.accessibleName() == '对照同一张照片的精修成片'
        assert '执行时实际验证' in quick.same_photo_finish.toolTip()
        quick.regional.enabled.setChecked(True); window.set_regional_box([0, 0, 32, 32])
        assert window.creative_task_payload()['regional_refinement']['maximum_submissions'] == 1
        # Opt-in revokes a pending region and its checked plan, not the user's brief/effect intent.
        window.creative_task_plan = {'plan': {'ready': True}}
        quick.same_photo_finish.setChecked(True)
        intent = window.creative_task_payload()
        assert window.creative_task_plan is None
        assert intent['constraints']['reference_role'] == 'same_photo_finish'
        assert 'regional_refinement' not in intent
        assert not window.regional_enabled and window.regional_core_box is None
        assert not quick.regional.enabled.isEnabled() and not window.regional_controls.geometry.isEnabled()
        assert window.creative_same_photo_finish.isChecked()
        assert intent['brief'] == quick.brief.text() and not intent['constraints']['no_effects']
        assert '尚未实际配准' in quick.direction_summary.text()
        # Professional edits update quick mode, and saved valid intent restores the same opt-in.
        saved = copy.deepcopy(intent)
        window.creative_same_photo_finish.setChecked(False)
        assert not quick.same_photo_finish.isChecked()
        window.apply_creative_task_options(saved)
        assert quick.same_photo_finish.isChecked() and window.creative_task_payload() == saved
        with patch('current_model_registry.registry_status', return_value=model_registry_fixture()), \
             patch('task_model_preflight.inspect_task_models', side_effect=model_capability_fixture), \
             patch.object(contract, '_services', side_effect=service_fixture):
            quick.check_plan_button.click(); pending = window.pool.tasks[-1]
            assert pending.path == '/api/tasks/plan'
            assert pending.body['creative_task'] == saved
            assert 'regional_candidate_reuse_authorization' not in pending.body
            assert 'regional_resume_authorization' not in pending.body
            checked = contract.plan_task(root, pending.body)
            assert all(value == 0 for value in checked['plan']['image_budget'].values())
            assert checked['plan']['reference_contract']['registration_status'] == 'not_checked'
            pending.signals.result.emit(checked)
            text = window.creative_plan_summary.toPlainText()
            assert '尚未实际配准' in text and '原片拥有几何和原生纹理' in text
            assert '总上限：0' in text
            quick.start_plan_button.click(); started = window.pool.tasks[-1]
            assert started.path == '/api/tasks/start'
            assert started.body['creative_task']['constraints']['reference_role'] == 'same_photo_finish'
            assert 'regional_refinement' not in started.body['creative_task']
            started.signals.result.emit({'ok': True, 'async': True, 'job_id': 'same-photo-captured'})
            window.product_run = {'run_id': 'current-generic', 'options': {
                'execution_strategy': 'layered_reference_transfer',
                'task_intent': {'mode': 'existing_reference'}}}
            window.refresh_cos_workspace()
            assert '匹配意图已冻结' in quick.direction_summary.text()
            assert '新增生图 0 次' in quick.direction_summary.text()
            assert not quick.same_photo_finish.isEnabled() and quick.same_photo_finish.isChecked()
            window.result('job-status', {'job': {'job_id': 'same-photo-captured', 'kind': 'one_click',
                'status': 'failed', 'message': 'Captured test; no production started'}})
        window.creative_task_start_pending = False; window.render_pending = False
        window.apply_creative_task_options(saved)
        window.set_background_mode('replace')
        assert not window.same_photo_finish_enabled and not quick.same_photo_finish.isEnabled()
        window.set_background_mode('enhance_source')
        assert not quick.same_photo_finish.isChecked()
        quick.same_photo_finish.setChecked(True)
        window.creative_task_mode.setCurrentIndex(window.creative_task_mode.findData('create_reference'))
        assert not window.same_photo_finish_enabled
        window.apply_creative_task_options(saved)
        window.source_edit.setText(str(other))
        assert not window.same_photo_finish_enabled and not quick.same_photo_finish.isChecked()
        assert 'reference_role' not in window.creative_task_payload()['constraints']
        window.apply_creative_task_options({'schema_version': 'cosplay_task_intent.v1', 'mode': 'existing_reference',
            'brief': 'legacy', 'constraints': {'preserve_background': False, 'no_effects': True},
            'services': {'text_vision': False, 'visual_review': False}})
        assert 'reference_role' not in window.creative_task_payload()['constraints']
        # Restored generated-reference ownership belongs only to its original source.
        source_saved = copy.deepcopy(saved)
        source_saved['constraints']['reference_role'] = 'source_conditioned_design'
        window.apply_creative_task_options(source_saved)
        assert window.source_conditioned_reference and not quick.same_photo_finish.isChecked()
        assert window.creative_task_payload()['constraints']['reference_role'] == 'source_conditioned_design'
        window.source_edit.setText(str(source))
        assert not window.source_conditioned_reference
        assert 'reference_role' not in window.creative_task_payload()['constraints']
        window.session_timer.stop(); window.close(); app.processEvents()
    print(json.dumps({'ok': True, 'shared_opt_in': True, 'zero_budget_captured_transport': True,
                     'source_owner_and_unchecked_registration_visible': True,
                     'default_source_only_generation_budget_one': True, 'generated_reference_bound_after_generation': True}))


if __name__ == '__main__': main()
