"""Real HTTP task dispatch queues; CLI's explicit synchronous route stays synchronous."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import threading
import urllib.request
from unittest.mock import patch

from backend_server import Backend, LocalHTTPServer, make_handler
from creative_task_smoke import fixture, intent
from task_model_preflight_smoke import mocked_task_capability


def main():
    with tempfile.TemporaryDirectory(prefix='task-start-dispatch-') as directory:
        root = Path(directory)
        registry, sources, _ = fixture(root)
        backend = Backend(root, embedded=True)
        server = LocalHTTPServer(('127.0.0.1', 0), make_handler(backend))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        dispatched = threading.Event()

        def post(path, body):
            request = urllib.request.Request(f'http://127.0.0.1:{server.server_port}'+path,
                data=json.dumps(body).encode('utf8'), headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request, timeout=10) as response:
                return json.load(response)

        try:
            with (patch.dict(os.environ, {'COSPLAY_MODEL_REGISTRY':str(registry)}),
                  patch('task_model_preflight.inspect_task_models', side_effect=mocked_task_capability),
                  patch.object(backend, '_run_one_click_job', side_effect=lambda *a: dispatched.set()) as worker,
                  patch.object(backend, '_one_click_sync', side_effect=AssertionError('HTTP entered synchronous renderer'))):
                checked = post('/api/tasks/plan', {'source':sources[0],
                    'model_registry':str(registry), 'creative_task':intent('local_portrait')})
                assert checked['plan']['ready']
                body = {**checked['request'], 'plan_id':checked['plan']['plan_id']}
                started = post('/api/tasks/start', body)
                assert started['ok'] and started['async'] and started['job_id']
                assert dispatched.wait(2)
                assert worker.call_count == 1
                backend.jobs.update(started['job_id'], status='cancelled', stage='test_complete')
            with (patch.dict(os.environ, {'COSPLAY_MODEL_REGISTRY':str(registry)}),
                  patch('task_model_preflight.inspect_task_models', side_effect=mocked_task_capability),
                  patch.object(backend, '_one_click_sync', return_value={'ok':True, 'async':False}) as sync):
                result = backend.start_task({**body, 'async':False})
                assert result['ok'] and result['async'] is False
                sync.assert_called_once()
        finally:
            server.shutdown(); server.server_close(); thread.join(5)
    print(json.dumps({'ok':True, 'real_HTTP_queued':True, 'explicit_CLI_sync':True,
                      'model_submissions':0}))


if __name__ == '__main__':
    main()
