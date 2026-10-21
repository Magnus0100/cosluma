"""Startup receipts, configuration-only Qwen status and checked quick actions."""
from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import qt_workbench as qt
from design_reconstruction_ui_smoke import Window, Pool, last
from portrait_beauty_workbench_smoke import run_fixture


def main():
    assert qt.QT_AVAILABLE
    app = qt.QApplication.instance() or qt.QApplication([])
    qwen = {'id': 'qwen_image_2_1_uncensored_fast', 'ready': True}
    text = qt.generation_readiness_text(qwen)
    assert '配置已检查' in text and '服务未检测' in text and '模型加载未验证' in text
    assert '服务可连接' in qt.generation_readiness_text({**qwen, 'network_checked': True, 'service_ready': True})
    assert '服务未连接' in qt.generation_readiness_text({**qwen, 'network_checked': True, 'service_ready': False})
    with tempfile.TemporaryDirectory(prefix='cosplay-startup-readiness-') as folder:
        root = Path(folder)
        run, _, _, _, _ = run_fixture(root)
        windows = []
        from start_qwen_image import ENDPOINT
        qwen={'id':'qwen_image_2_1','kind':'comfyui','ready':True,'generation_supported':True,
            'endpoint':ENDPOINT,'model':'fixture-quality-workflow'}
        with patch('beautygrpo_prepare.prewarm_for_workbench') as beauty_warm, \
                patch('start_qwen_image.ensure_running',return_value={'managed':True,'endpoint':ENDPOINT}) as service:
            window=Window(root,'http://127.0.0.1:1');window.pool=Pool();windows.append(window)
            window.set_simple_source(run['source']['path']);window.set_startup_readiness('initializing')
            window.prewarm_startup()
            assert window.startup_readiness=='ready' and window.simple_workspace.create_cos_button.isEnabled()
            assert not window.startup_service_timer.isActive(), 'Do not select a transient provider before settings restore'
            window.result('providers',{'providers':[qwen]})
            assert not window.startup_service_timer.isActive()
            window.result('provider-settings',{'providers':{},'default_provider':'qwen_image_2_1'})
            assert not window.startup_service_timer.isActive(), 'Saved task selection may still need restoring'
            window.result('recent',{'recent':[]})
            assert window.startup_service_timer.isActive()
            window.startup_service_timer.stop();window.start_selected_service()
            task=window.pool.tasks[-1]
            assert isinstance(task,qt.StartupServiceTask) and not service.called
            assert window.simple_workspace.source_button.isEnabled() and window.simple_workspace.create_cos_button.isEnabled()
            task.run();app.processEvents()
            service.assert_called_once()
            assert service.call_args.args==('qwen_image_2_1',ENDPOINT)
            assert callable(service.call_args.kwargs['cancel'])
            assert '模型加载未验证' in window.operation_state.text()
            assert '模型加载未验证' in window.simple_workspace.generation_model_note.text()
            record=next(iter(window._startup_service_records.values()))
            assert record['model_load_verified'] is False and record['model_warming_performed'] is False
            assert record['image_submissions']==0
            count=len(window.pool.tasks);window.schedule_selected_service();window.start_selected_service()
            assert len(window.pool.tasks)==count and service.call_count==1
            beauty_warm.assert_not_called()

            # Only the current selected service may update the UI. Mode switch,
            # user cancel and window exit invalidate scheduled/queued starts.
            window._startup_service_records.clear();window.start_selected_service();stale=window.pool.tasks[-1]
            window.creative_task_mode.setCurrentIndex(window.creative_task_mode.findData('local_portrait'))
            assert stale.cancel.is_set()
            before=service.call_count;stale.run();app.processEvents();assert service.call_count==before
            window.schedule_selected_service();window.start_selected_service()
            assert window.creative_task_mode.currentData()=='local_portrait'
            assert service.call_count==before;beauty_warm.assert_not_called()
            window.creative_task_mode.setCurrentIndex(window.creative_task_mode.findData('create_reference'))
            window.startup_service_timer.stop();window.start_selected_service();cancelled=window.pool.tasks[-1]
            window.simple_action('cancel');assert cancelled.cancel.is_set()
            cancelled.run();assert service.call_count==before
            window.schedule_selected_service();assert not window.startup_service_timer.isActive()
            window._startup_service_suppressed=False;window.start_selected_service();closing=window.pool.tasks[-1]
            window.close();app.processEvents();assert closing.cancel.is_set()
            closing.run();assert service.call_count==before

            window=Window(root,'http://127.0.0.1:1');window.pool=Pool();windows.append(window)
            window.result('providers',{'providers':[qwen,{'id':'openai-compatible','kind':'openai-compatible','ready':True,
                'generation_supported':True,'endpoint':'https://example.invalid/v1','model':'fixture-model'}]})
            window.result('provider-settings',{'providers':{},'default_provider':'openai-compatible'})
            window.result('recent',{'recent':[]})
            window.prewarm_startup();window.start_selected_service()
            assert not window.startup_service_timer.isActive() and service.call_count==before
            window.set_simple_source(run['source']['path'])
            window.set_startup_readiness('degraded')
            window.render_pending=True;window.operation_state.setText('正在渲染当前作品')
            window.set_startup_status('所选服务已连接，模型加载未验证')
            assert window.operation_state.text()=='正在渲染当前作品'
            window.render_pending=False

            failed=Window(root,'http://127.0.0.1:1');failed.pool=Pool();windows.append(failed)
            failed.set_simple_source(run['source']['path'])
            failed.result('providers',{'providers':[qwen]})
            failed.result('provider-settings',{'providers':{},'default_provider':'qwen_image_2_1'})
            failed.result('recent',{'recent':[]})
            failed.prewarm_startup();failed.startup_service_timer.stop();failed.start_selected_service()
            service.side_effect=RuntimeError('Synthetic managed service unavailable')
            failed.pool.tasks[-1].run();app.processEvents()
            assert failed.startup_readiness=='ready' and failed.simple_workspace.create_cos_button.isEnabled()
            assert next(iter(failed._startup_service_records.values()))['state']=='service_unavailable'
            assert '模型加载未验证' in failed.simple_workspace.generation_model_note.text()
            count=len(failed.pool.tasks);failed.schedule_selected_service();failed.start_selected_service()
            assert len(failed.pool.tasks)==count,'Startup failure must not spin up an implicit retry loop'
            beauty_warm.assert_not_called()

        # A degraded startup may inspect a usable route, but cannot bypass a
        # blocked task plan. The same rule applies to Beauty on an open run.
        window = windows[-2]
        window.load_source_run(run)
        before = len(window.pool.tasks)
        window.simple_action('beauty')
        context = copy.deepcopy(window.creative_task_plan_request)
        assert context['prepare_existing_beauty'] is True
        assert last(window.pool, context['label']).path == '/api/tasks/plan'
        window.result(context['label'], {'ok': True, 'plan': {'plan_id': 'blocked', 'ready': False,
            'blocking_reasons': ['缺少人像运行时']}, 'request': context['request']})
        assert window.simple_creation is None and not window.creative_start_button.isEnabled()
        assert not any(task.path.endswith(('/start', '/prepare-portrait-beauty')) for task in window.pool.tasks[before:])
        assert '缺少人像运行时' in window.operation_state.text()
        window.simple_action('beauty')
        context = copy.deepcopy(window.creative_task_plan_request)
        before = len(window.pool.tasks)
        window.result(context['label'], {'ok': True, 'plan': {'plan_id': 'ready', 'ready': True},
                                        'request': context['request']})
        assert last(window.pool, 'portrait-beauty-job').path.endswith('/prepare-portrait-beauty')
        assert not any(task.path == '/api/tasks/start' for task in window.pool.tasks[before:])
        for window in windows:
            window.session_timer.stop(); window.native_matte_timer.stop(); window.source_light_timer.stop()
            window._session_checkpoint_failed = False
            window.close()
    print(json.dumps({'ok': True, 'selected_service_only':True,'no_unused_beauty_load':True,
        'startup_does_not_block_checked_creation':True,'no_implicit_image_inference':True,
        'selection_cancel_and_exit_discard_stale_start':True,
        'qwen_service_not_model_readiness': True, 'open_revision_beauty_plan_gate': True}))


if __name__ == '__main__': main()
