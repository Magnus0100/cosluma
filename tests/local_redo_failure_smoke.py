"""Desktop local remakes retain actual HTTP failure detail and saved pixels."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import qt_workbench as qt
from design_reconstruction_ui_smoke import Window, Pool, last
from portrait_beauty_workbench_smoke import run_fixture


class RejectedLocalRequest(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get('Content-Length', 0)))
        payload = json.dumps({'ok': False, 'error': '本地模型配置尚未完成'}, ensure_ascii=False).encode('utf8')
        self.send_response(400); self.send_header('Content-Length', str(len(payload)))
        self.end_headers(); self.wfile.write(payload)

    def log_message(self, *args):
        pass


def main():
    server = ThreadingHTTPServer(('127.0.0.1', 0), RejectedLocalRequest)
    worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
    try:
        try:
            qt.api_call(f'http://127.0.0.1:{server.server_port}', 'POST', '/api/source-runs/saved/complete-post', {})
        except RuntimeError as error:
            assert 'HTTP 400' in str(error) and '本地模型配置尚未完成' in str(error)
        else:
            raise AssertionError('Local HTTP rejection was lost')
    finally:
        server.shutdown(); server.server_close(); worker.join(timeout=2)
    app = qt.QApplication.instance() or qt.QApplication([])
    with tempfile.TemporaryDirectory(prefix='desktop-local-redo-') as temporary:
        root = Path(temporary)
        run, _, _, final, _ = run_fixture(root, source_only=False)
        before = final.read_bytes()
        window = Window(root, 'http://127.0.0.1:1'); window.pool = Pool()
        window.load_source_run(run)
        with patch.object(window, 'current_model_options', return_value={'algorithm_profile': 'dinov3_vitl16'}):
            window.complete_current_post()
        submitted = last(window.pool, 'complete-post')
        assert submitted.path == '/api/source-runs/beauty-parent/complete-post'
        assert window.render_pending
        window.simple_creation = {'stage': 'post'}
        window.error('HTTP 400: 本地模型配置尚未完成', 'complete-post')
        assert not window.render_pending and window.simple_creation is None
        assert '本地重新制作' in window.operation_state.text()
        assert '本地模型配置尚未完成' in window.operation_state.text()
        assert '生成请求未完成' not in window.operation_state.text()
        assert window.source_run_id == run['run_id'] and final.read_bytes() == before
        window.result('complete-post', {'ok': True, 'async': True, 'job_id': 'job-local-remake'})
        assert window.active_job_id == 'job-local-remake'
        for response_label in ('job-status', 'job-stream'):
            window.render_pending = True
            window.simple_creation = {'stage': 'post'}
            window.pool.tasks.clear()
            failed = {'job_id': 'job-local-remake', 'status': 'failed',
                      'run_id': 'uncommitted-child', 'error': 'contract write failed'}
            window.result(response_label, {'job': failed} if response_label == 'job-status'
                          else {'data': failed})
            assert not window.render_pending and window.simple_creation is None
            assert not any(getattr(task, 'label', '') == 'job-run-loaded'
                           for task in window.pool.tasks), response_label
            assert window.source_run_id == run['run_id']
            assert final.read_bytes() == before
        window.operation_state.setText('本次处理执行结束；正在读取作品缺项、证据与批准状态。')
        window.result('job-run-loaded', {'run': run})
        assert window.operation_state.text() == qt.post_completion_message(run)
        assert '正在读取' not in window.operation_state.text()
        window.job_poll_timer.stop(); window.session_timer.stop()
        window.native_matte_timer.stop(); window.source_light_timer.stop()
        window.render_pending = False; window._session_checkpoint_failed = False; window.close()
        app.processEvents()
    print(json.dumps({'ok': True, 'real_HTTP_error_body_retained': True,
        'local_remake_not_reported_as_generation_failure': True,
        'failure_retains_saved_pixels_and_unlocks_UI': True, 'async_job_observed': True,
        'model_inferences': 0}))


if __name__ == '__main__':
    main()
