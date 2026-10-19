"""Exercise real Qt task planning against the current contract, without services."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PIL import Image
import creative_task as contract
import qt_workbench as qt


class Window(qt.Workbench):
    def refresh(self): pass


class Pool:
    def __init__(self): self.tasks = []
    def start(self, task): self.tasks.append(task)


def service_fixture(request, intent):
    enabled = intent['services']['text_vision']
    visual_review = intent['services']['visual_review']
    return {'configured': True, 'text_vision': enabled, 'visual_review': visual_review, 'provider': 'fixture-service',
        'endpoint': 'https://fixture.invalid/v1', 'network_probed': False,
        'sent_content': (['原片观察图', '参考', '创作要求'] +
                         (['匹配原尺寸窗口'] if visual_review else [])) if enabled else [],
        'text_vision_requests_upper_bound': None if enabled else 0, 'price': '价格未知',
        'offline': not enabled and intent['mode'] != 'create_reference',
        'image_generation': intent['mode'] == 'create_reference'}


def model_registry_fixture():
    return {'configured': True, 'registry_path': None, 'config': {}, 'paths': [
        {'key': key, 'label': label, 'kind': 'file', 'available': True, 'status': 'available'}
        for key, label in (('python', '分析运行时'), ('semantics_checkpoint', '人物语义权重'),
                          ('semantics_repo_dir', '人物语义源码'), ('vitmatte_model_root', '原生 alpha'))]}


def model_capability_fixture(config, *, mode='existing_reference', **kwargs):
    """Model metadata only; planning, input identities and Qt remain real."""
    return {'schema_version': 'cosplay_task_model_preflight.v1', 'mode': mode, 'ready': True,
        'blocking_reasons': [], 'checks': [], 'weight_checks': [],
        'file_structure_verified': False, 'checked_identity_verified': False,
        'weight_content_verified': False, 'inference_verified': False,
        'test_model_capability_mocked': True, 'verification_scope': 'test_model_capability_mock',
        'hash_bytes_read': 0, 'large_weight_bytes_hashed': 0,
        'network_requests': 0, 'model_inferences': 0,
        'limitations': ['Synthetic model capability fixture; no installation or inference verified.']}


def main():
    assert qt.QT_AVAILABLE
    app = qt.QApplication.instance() or qt.QApplication([])
    with tempfile.TemporaryDirectory(prefix='cosplay-task-ui-') as folder:
        root = Path(folder)
        sources = []
        for index, colour in enumerate(('red', 'green', 'blue')):
            path = root / ('source-' + str(index) + '.png'); Image.new('RGB', (12, 18), colour).save(path); sources.append(str(path))
        reference = root / 'reference.png'; Image.new('RGB', (12, 18), 'purple').save(reference)
        with patch('current_model_registry.registry_status', return_value=model_registry_fixture()), \
             patch('task_model_preflight.inspect_task_models', side_effect=model_capability_fixture), \
             patch.object(contract, '_services', side_effect=service_fixture), \
             patch('reference_provider.provider_config', return_value={'kind':'comfyui','registry_id':'qwen_image_2_1_uncensored_fast'}), \
             patch('reference_provider.provider_health', return_value={'ready': True}):
            window = Window(root, 'http://127.0.0.1:1'); pool = Pool(); window.pool = pool
            assert window.variant_count.value() == 1
            assert window.creative_task_mode.currentData() == 'local_portrait'
            assert not window.creative_text_vision.isChecked() and not window.creative_text_vision.isEnabled()
            assert not window.creative_visual_review.isChecked() and not window.creative_visual_review.isEnabled()
            assert window.creative_plan_summary.isReadOnly() and not window.creative_start_button.isEnabled()
            assert not window.character_recognize.isEnabled() and not window.prepare_design_button.isEnabled()
            count = len(pool.tasks); window.recognize_character(); window.prepare_generation_design()
            assert len(pool.tasks) == count and '未提交' in window.operation_state.text()
            assert not hasattr(window, 'resume_btn') and not hasattr(window, 'promote') and not hasattr(window, 'samples')
            window.source_edit.setText(sources[0]); window.creative_task_brief.setPlainText('保留眼线和金色细边，柔和夜景。')
            window.run_local_source(); pending = pool.tasks[-1]
            assert pending.path == '/api/tasks/plan' and not window.render_pending
            assert pending.body['creative_task']['services']['text_vision'] is False
            before = len(pool.tasks); window.start_planned_creative_task(); assert len(pool.tasks) == before
            files_before = sorted(str(path.relative_to(root)) for path in root.rglob('*'))
            local = contract.plan_task(root, pending.body)
            capability = local['plan']['capabilities']
            assert capability['model_preflight']['test_model_capability_mocked'] is True
            assert capability['inference_verified'] is False and capability['model_preflight']['inference_verified'] is False
            assert local['plan']['inputs']['source']['sha256'] == contract.sha(Path(sources[0]))
            assert files_before == sorted(str(path.relative_to(root)) for path in root.rglob('*'))
            pending.signals.result.emit(local)
            assert window.creative_start_button.isEnabled() and '不使用' in window.creative_plan_summary.toPlainText()
            assert '局部通过不认证全图' in window.creative_plan_summary.toPlainText()
            assert '模型文件/小文件身份与配置检查；权重内容和推理在执行验证' in window.creative_plan_summary.toPlainText()
            window.creative_start_button.click(); started = pool.tasks[-1]
            assert (started.path, started.label) == ('/api/tasks/start', 'local-only')
            assert started.body['plan_id'] == local['plan']['plan_id']
            assert contract.execution_request(root, started.body)['task_intent']['mode'] == 'local_portrait'
            count = len(pool.tasks); window.start_planned_creative_task(); assert len(pool.tasks) == count
            started.signals.error.emit('mocked submission failure')
            assert not window.render_pending and not window.creative_task_start_pending

            with patch.object(qt.QFileDialog, 'getOpenFileName', return_value=(str(reference), '')):
                window.use_existing_design_reference()
            pending = pool.tasks[-1]; supplied = contract.plan_task(root, pending.body)
            assert pending.body['creative_task']['mode'] == 'existing_reference'
            assert pending.body['reference'] == str(reference.resolve()) and sum(supplied['plan']['image_budget'].values()) == 0
            assert supplied['plan']['inputs']['reference']['sha256'] == contract.sha(reference)
            pending.signals.result.emit(supplied)
            text = window.creative_plan_summary.toPlainText()
            assert '不使用' in text and '参考纠错 0' in text
            assert '自动投影' in text and not window.creative_text_vision.isChecked()
            window.creative_start_button.click(); started = pool.tasks[-1]
            assert (started.path, started.label) == ('/api/tasks/start', 'existing-design')
            execution = contract.execution_request(root, started.body)
            assert execution['generate'] is False and execution['complete_post'] is True
            assert execution['task_intent']['services']['visual_review'] is False
            assert '未启用，保存后由人工审片' in text
            started.signals.error.emit('mocked submission failure')

            assert not window.creative_visual_review.isEnabled()
            window.set_background_mode('replace')
            window.creative_text_vision.setChecked(True)
            assert window.creative_visual_review.isEnabled()
            window.creative_visual_review.setChecked(True)
            assert window.creative_task_plan is None
            assert window.creative_task_payload()['services']['visual_review'] is True
            window.creative_text_vision.setChecked(False); window.check_creative_task_plan()
            assert not window.creative_visual_review.isChecked() and not window.creative_visual_review.isEnabled()
            assert window.creative_task_payload()['services']['visual_review'] is False
            count = len(pool.tasks); window.recognize_character(); window.prepare_generation_design()
            assert len(pool.tasks) == count and not window.character_recognize.isEnabled()
            pending = pool.tasks[-1]; unplanned = contract.plan_task(root, pending.body); pending.signals.result.emit(unplanned)
            assert window.creative_start_button.isEnabled() and '不使用' in window.creative_plan_summary.toPlainText()
            assert unplanned['request']['complete_post'] and not unplanned['plan']['services']['text_vision']
            window.creative_text_vision.setChecked(True); window.check_creative_task_plan(); stale = pool.tasks[-1]
            stale_value = contract.plan_task(root, stale.body)
            window.creative_task_brief.setPlainText('新的要求，只保留已有主效。'); stale.signals.result.emit(stale_value)
            assert window.creative_task_plan is None and not window.creative_start_button.isEnabled()
            window.check_creative_task_plan(); latest = pool.tasks[-1]; checked = contract.plan_task(root, latest.body)
            latest.signals.result.emit(checked); assert window.creative_start_button.isEnabled()
            window.source_edit.setText(sources[1]); assert not window.creative_start_button.isEnabled()

            window.run_source_only(); pending = pool.tasks[-1]
            single = contract.plan_task(root, pending.body); pending.signals.result.emit(single)
            assert single['plan']['image_budget']['initial_reference'] == 1
            assert single['plan']['candidate_selection']['required'] is False
            assert pending.body['defer_reference_selection'] is False and 'variant_styles' not in pending.body
            window.creative_start_button.click(); started = pool.tasks[-1]
            assert contract.execution_request(root, started.body)['complete_post'] is True
            started.signals.error.emit('mocked submission failure')

            window.set_background_mode('replace')
            window.variant_count.setValue(2); window.run_source_only(); pending = pool.tasks[-1]
            generated = contract.plan_task(root, pending.body); pending.signals.result.emit(generated)
            assert generated['plan']['mode'] == 'create_reference' and generated['plan']['image_budget']['initial_reference'] == 2
            assert generated['plan']['candidate_selection']['required'] is True
            assert pending.body['defer_reference_selection'] is True and len(pending.body['variant_styles']) == 2
            authored_settings = window.generation_request_options()
            authored_settings['variant_styles'] = ['night_scene', 'stage_light']
            with patch.object(window, 'generation_request_options', return_value=authored_settings):
                authored = window.creative_task_request_options('create_reference')
            assert contract.plan_task(root, {**authored, 'source': sources[1]})['request']['variant_styles'] == ['night_scene', 'stage_light']
            window.creative_start_button.click(); started = pool.tasks[-1]
            assert started.label == 'one-click' and contract.execution_request(root, started.body)['generate'] is True
            started.signals.error.emit('mocked submission failure')

            window.batch_sources = sources[:2]; window.run_batch(); pending = pool.tasks[-1]
            assert pending.path == '/api/tasks/batch/plan' and pending.body['sources'] == sources[:2]
            assert not {'appearance_revision', 'manual_layers', 'generation_design', 'character_profile'} & pending.body.keys()
            batch = contract.plan_batch(root, pending.body); pending.signals.result.emit(batch)
            window.creative_start_button.click(); started = pool.tasks[-1]
            assert (started.path, started.label) == ('/api/tasks/batch', 'batch-started')
            assert started.body['plan_id'] == batch['batch_plan']['plan_id']
            started.signals.result.emit({'batch': {'kind': 'creative_task_batch', 'shared_task': {'mode': 'create_reference'},
                'batch_id': 'create-batch', 'items': [
                    {'index': 0, 'source': sources[0], 'reference': None, 'job': {'status': 'failed'}},
                    {'index': 1, 'source': sources[1], 'reference': None, 'job': {'status': 'succeeded'}}]}})
            assert window.local_batch_id == 'create-batch' and window.local_batch_tree.topLevelItem(0).text(1) == '本次创作'
            window.local_batch_tree.setCurrentItem(window.local_batch_tree.topLevelItem(0)); window.retry_local_batch_item()
            retry = pool.tasks[-1]; count = len(pool.tasks); window.retry_local_batch_item()
            assert len(pool.tasks) == count and retry.path == '/api/tasks/batch/create-batch/retry-item'
            retry.signals.error.emit('retry outcome unknown')
            assert pool.tasks[-1].method == 'GET' and pool.tasks[-1].path == '/api/batch/create-batch'
            assert not window.local_batch_retry_button.isEnabled()

            window.local_batch_pairs = [{'source': source, 'reference': str(reference)} for source in sources[:2]]
            window.run_local_reference_batch(); pending = pool.tasks[-1]
            assert 'items' in pending.body and 'sources' not in pending.body
            batch = contract.plan_batch(root, pending.body); pending.signals.result.emit(batch)
            assert sum(batch['batch_plan']['image_budget_total'].values()) == 0
            assert '各图独立' in window.creative_plan_summary.toPlainText()
            window.creative_start_button.click(); started = pool.tasks[-1]
            assert (started.path, started.label) == ('/api/tasks/batch', 'local-batch-started')
            started.signals.result.emit({'batch': {'kind': 'creative_task_batch', 'shared_task': {'mode': 'existing_reference'},
                'batch_id': 'task-batch', 'items': [{'index': 0, 'source': sources[0], 'reference': None, 'job': {'status': 'failed'}}]}})
            window.local_batch_tree.setCurrentItem(window.local_batch_tree.topLevelItem(0)); window.retry_local_batch_item()
            assert pool.tasks[-1].path == '/api/tasks/batch/task-batch/retry-item' and pool.tasks[-1].body == {'index': 0}
            saved = window._template_payload()['creative_task']
            window.creative_task_brief.clear(); window.apply_creative_task_options(saved)
            assert window.creative_task_payload() == saved
            assert not any(task.path in {'/api/run', '/api/resume', '/api/review', '/api/promote', '/api/one-click', '/api/batch/local-reference'} for task in pool.tasks)
            window.close(); app.processEvents()
    print(json.dumps({'ok': True, 'real_Qt_controls': True, 'current_task_contract': True,
        'plan_then_explicit_start': True, 'stale_and_blocked_plans_prevent_start': True,
        'zero_image_existing_reference': True, 'shared_brief_without_geometry': True,
        'isolated_retry': True, 'retired_routes_absent': True,
        'generation_batch_tree_and_retry': True, 'retry_unknown_queries_without_resubmission': True,
        'advanced_service_actions_obey_policy': True,
        'real_input_hashes': True, 'test_model_capability_mocked': True, 'inference_verified': False,
        'network_requests': 0, 'model_inferences': 0}, ensure_ascii=False))


if __name__ == '__main__': main()
