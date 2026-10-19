"""Exercise quick model selection, real configuration routing and saved defaults offline."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PIL import Image
import qt_workbench as qt
import provider_settings
import reference_provider


class Window(qt.Workbench):
    def refresh(self): pass


class Pool:
    def __init__(self): self.tasks = []
    def start(self, task): self.tasks.append(task)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path)
    args = parser.parse_args()
    assert qt.QT_AVAILABLE
    app = qt.QApplication.instance() or qt.QApplication([])
    with tempfile.TemporaryDirectory(prefix='cosplay-model-selection-') as temporary:
        root = Path(temporary)
        settings_path = root/'provider-settings.json'
        source = root/'source.png'
        Image.new('RGB', (400, 600), '#476d8a').save(source)
        environment = {'COSPLAY_PROVIDER_SETTINGS':str(settings_path),
                       'COSPLAY_REFERENCE_API_KEY':'offline-test-key',
                       'QT_QPA_PLATFORM':'offscreen'}
        with patch.dict(os.environ, environment, clear=True), \
                patch.object(provider_settings, '_ccswitch_record', return_value={'status':'not_found'}):
            # Use the real catalog and resolver, without service probes or model loading.
            rows = []
            for entry in reference_provider.load_provider_registry():
                config = reference_provider.provider_config(provider=entry['id'])
                rows.append({**entry, **reference_provider.adapter_contract(entry),
                             'model':config['model'], 'endpoint':config['endpoint'],
                             'ready':config['configured'], 'status':'ready' if config['configured'] else 'missing_workflow'})
            window = Window(root, 'http://127.0.0.1:1'); pool = Pool(); window.pool = pool
            window.resize(1280, 920); window.show(); app.processEvents()
            window.result('providers', {'providers':rows})
            selector = window.simple_workspace.generation_model
            available = [selector.itemData(i) for i in range(selector.count())]
            assert set(available) == {entry['id'] for entry in rows if entry['generation_supported']}
            assert 'manual_import' not in available and 'local' not in available
            assert 'INT8' in selector.itemText(selector.findData('qwen_image_2_1_fast'))
            assert 'gpt-image-2.5-sunburst' in selector.itemText(selector.findData('ccswitch_sub2api'))
            assert not pool.tasks, 'Catalog synchronization must not submit requests'

            # With no persisted choice, prefer the full-precision local
            # workflow when it is ready.  This is a selector default only; it
            # must not submit a generation request.
            window.result('provider-settings', {'default_provider':None, 'providers':{}})
            assert selector.currentData() == 'qwen_image_2_1'
            assert window.creative_task_payload('create_reference')['constraints'].get('reference_role') == 'source_conditioned_design'
            assert not pool.tasks

            window.set_simple_source(str(source))
            before_choice = len(pool.tasks)
            window.creative_task_plan = {'plan':{'ready':True, 'plan_id':'old-plan'}}
            window.prepared_design = {'effective_prompt':'old design'}
            selector.setCurrentIndex(selector.findData('qwen_image_2_1_fast'))
            assert window.provider.currentText() == 'qwen_image_2_1_fast'
            assert window.creative_task_plan is None and window.prepared_design is None
            request = window.creative_task_request_options('create_reference', source=str(source))
            resolved = reference_provider.provider_config(provider=request['provider'], model=request['model'])
            workflow = json.loads(Path(resolved['workflow_path']).read_text('utf8'))
            assert workflow['model_name'] == 'Qwen-Image-2.1 INT8 ConvRot'
            assert window.generation_design_context()[5:7] == (request['provider'], request['model'])
            assert len(pool.tasks) == before_choice, 'Choosing a model must not generate images'

            # A late initial preferences response cannot undo a user's explicit choice.
            window.result('provider-settings', {'default_provider':'ccswitch_sub2api', 'providers':{}})
            assert window.provider.currentText() == 'qwen_image_2_1_fast'
            window.generation_model_save_timer.stop(); window.persist_generation_model()
            local_save = pool.tasks[-1]
            assert local_save.path == '/api/provider-settings' and local_save.body == {'default_provider':'qwen_image_2_1_fast'}
            # Serialize rapid changes: the first save acknowledgement must not reset the next choice.
            selector.setCurrentIndex(selector.findData('ccswitch_sub2api'))
            window.generation_model_save_timer.stop(); window.persist_generation_model()
            assert pool.tasks[-1] is local_save
            saved = provider_settings.save_settings(local_save.body, settings_path)
            local_save.signals.result.emit(saved)
            assert window.provider.currentText() == 'ccswitch_sub2api'
            window.generation_model_save_timer.stop(); window.persist_generation_model()
            remote_save = pool.tasks[-1]
            assert remote_save is not local_save and remote_save.body == {'default_provider':'ccswitch_sub2api'}
            remote_save.signals.result.emit(provider_settings.save_settings(remote_save.body, settings_path))

            # The quick button routes the selected API model into the actual task request.
            window.simple_workspace.create_cos_button.click()
            plan_request = pool.tasks[-1]
            assert 'reference_role' not in plan_request.body['creative_task']['constraints']
            assert plan_request.path == '/api/tasks/plan'
            assert plan_request.body['provider'] == 'ccswitch_sub2api'
            assert plan_request.body['model'] == 'gpt-image-2.5-sunburst'
            body, _ = reference_provider.multipart_body(source, {'model':plan_request.body['model'], 'prompt':'offline probe'})
            assert b'gpt-image-2.5-sunburst' in body
            assert not selector.isEnabled(), 'An active quick task must keep its selected model'
            window.simple_creation = None; window.creative_task_start_pending = False
            window.creative_task_plan_request = None; window.refresh_cos_workspace()

            # Advanced configuration and catalog refresh must stay synchronized with quick mode.
            window.simple_workspace.generation_settings_button.click()
            assert window.tabs.currentIndex() == 1 and window.provider_panel_toggle.isChecked()
            window.model.setText('custom-image-v2')
            assert 'custom-image-v2' in selector.currentText()
            window.result('providers', {'providers':rows})
            assert window.model.text() == 'custom-image-v2'
            window.save_provider_settings()
            custom_save = pool.tasks[-1]
            custom_save.signals.result.emit(provider_settings.save_settings(custom_save.body, settings_path))
            assert provider_settings.load_settings(settings_path)['providers']['ccswitch_sub2api']['model'] == 'custom-image-v2'

            # Both startup response orders restore the saved provider and custom model.
            for providers_first in (False, True):
                reopened = Window(root, 'http://127.0.0.1:1'); reopened.pool = Pool()
                saved = provider_settings.load_settings(settings_path)
                if providers_first: reopened.result('providers', {'providers':rows})
                reopened.result('provider-settings', saved)
                if not providers_first: reopened.result('providers', {'providers':rows})
                assert reopened.provider.currentText() == 'ccswitch_sub2api'
                assert reopened.model.text() == 'custom-image-v2'
                assert reopened.simple_workspace.generation_model.currentData() == 'ccswitch_sub2api'
                assert not reopened.pool.tasks
                reopened.close()

            # A failed preference write leaves the selected model intact and does not retry generation.
            window.set_professional_mode(False)
            selector.setCurrentIndex(selector.findData('qwen_image_2_1_fast'))
            window.generation_model_save_timer.stop(); window.persist_generation_model()
            failed_save = pool.tasks[-1]; count = len(pool.tasks)
            failed_save.signals.error.emit('offline preference failure')
            assert window.provider.currentText() == 'qwen_image_2_1_fast'
            assert window.generation_model_save_active is None and len(pool.tasks) == count
            assert '默认未保存' in window.simple_workspace.generation_model_note.text()
            assert all(getattr(task, 'path', '') != '/api/tasks/start' for task in pool.tasks)
            window.render_pending = True; window.refresh_cos_workspace()
            assert not selector.isEnabled() and not window.simple_workspace.generation_settings_button.isEnabled()
            window.render_pending = False; window.refresh_cos_workspace()
            # Explicit connection saves share the same queue as default selection writes.
            window.save_provider_settings()
            recovery_save = pool.tasks[-1]
            assert recovery_save.path == '/api/provider-settings'
            selector.setCurrentIndex(selector.findData('ccswitch_sub2api'))
            window.model.setText('custom-image-v3'); window.save_provider_settings()
            assert pool.tasks[-1] is recovery_save, 'Public settings writes must not overlap'
            selector.setCurrentIndex(selector.findData('qwen_image_2_1_fast'))
            recovery_save.signals.result.emit(provider_settings.save_settings(recovery_save.body, settings_path))
            window.generation_model_save_timer.stop(); window.persist_generation_model()
            queued_save = pool.tasks[-1]
            assert queued_save.body['default_provider'] == 'qwen_image_2_1_fast'
            assert queued_save.body['providers']['ccswitch_sub2api']['model'] == 'custom-image-v3'
            queued_save.signals.result.emit(provider_settings.save_settings(queued_save.body, settings_path))
            assert provider_settings.load_settings(settings_path)['default_provider'] == 'qwen_image_2_1_fast'
            assert not window.generation_model_save_error
            assert window.simple_workspace.generation_model.currentData() == 'qwen_image_2_1_fast'
            window.operation_state.setText('请选择生成模型，再开始制作。')
            if args.snapshot:
                args.snapshot.parent.mkdir(parents=True, exist_ok=True)
                app.processEvents(); assert window.grab().save(str(args.snapshot))
            window.close(); app.processEvents()
    print(json.dumps({'ok':True, 'quick_local_and_api_selection':True, 'real_model_routing':True,
                      'saved_default_and_custom_model':True, 'late_responses_preserve_choice':True,
                      'busy_task_locked':True, 'image_generation_requests':0}))


if __name__ == '__main__':
    main()
