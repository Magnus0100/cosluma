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


class ImmediateThread:
    def __init__(self, *, target, **kwargs): self.target = target
    def start(self): self.target()


def main():
    assert qt.QT_AVAILABLE
    app = qt.QApplication.instance() or qt.QApplication([])
    qwen = {'id': 'qwen_image_2_1_uncensored_fast', 'ready': True}
    text = qt.generation_readiness_text(qwen)
    assert '配置已检查' in text and '服务未检测' in text and '模型加载未验证' in text
    assert '服务可连接' in qt.generation_readiness_text({**qwen, 'network_checked': True, 'service_ready': True})
    assert '服务未连接' in qt.generation_readiness_text({**qwen, 'network_checked': True, 'service_ready': False})
    assert 'ready' == qt.beauty_startup_outcome({'state': 'pipeline_warm'})[0]
    assert 'ready' == qt.beauty_startup_outcome({'state': 'pipeline_already_warm'})[0]
    for result in ({'state': 'busy_service_untouched'}, {'state': 'pipeline_warm', 'ok': False}, {}, None):
        assert qt.beauty_startup_outcome(result)[0] == 'degraded'
    with tempfile.TemporaryDirectory(prefix='cosplay-startup-readiness-') as folder:
        root = Path(folder)
        run, _, _, _, _ = run_fixture(root)
        windows = []
        for result in ({'state': 'pipeline_warm'}, {'state': 'busy_service_untouched'}, RuntimeError('runtime unavailable')):
            window = Window(root, 'http://127.0.0.1:1'); window.pool = Pool(); windows.append(window)
            window.set_simple_source(run['source']['path'])
            window.set_startup_readiness('initializing')
            assert not window.simple_workspace.beauty_button.isEnabled()
            with patch.object(qt.threading, 'Thread', ImmediateThread), \
                    patch('beautygrpo_prepare.prewarm_for_workbench',
                          **({'side_effect': result} if isinstance(result, Exception) else {'return_value': result})):
                window.prewarm_startup()
            app.processEvents()
            assert window.startup_readiness == ('degraded' if isinstance(result, Exception) else qt.beauty_startup_outcome(result)[0])
            assert '无需冷启动' not in window.operation_state.text()
            assert '常用模型已就绪' not in window.operation_state.text()
            window.render_pending = True
            window.operation_state.setText('正在渲染当前作品')
            window.set_startup_status('人像美化预热完成')
            assert window.operation_state.text() == '正在渲染当前作品'
            window.render_pending = False

        # A degraded startup may inspect a usable route, but cannot bypass a
        # blocked task plan. The same rule applies to Beauty on an open run.
        window = windows[-1]
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
    print(json.dumps({'ok': True, 'warm_receipt_required': True,
        'qwen_configuration_not_runtime_readiness': True, 'open_revision_beauty_plan_gate': True}))


if __name__ == '__main__': main()
