"""Real Qt/loopback ownership regression; no images, models or remote requests."""
from __future__ import annotations

import errno
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import urllib.request
from urllib.parse import urlparse
from unittest.mock import patch

sys.dont_write_bytecode = True
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import backend_server
import provider_settings
import qt_workbench as qt


def main():
    app = qt.QApplication.instance() or qt.QApplication([])
    app.setQuitOnLastWindowClosed(False)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    requests, windows = [], []
    original_api = qt.api_call

    def api(base, method, path, body=None):
        assert method == 'GET' and urlparse(base).hostname in {'127.0.0.1', 'localhost'}
        requests.append(base)
        return original_api(base, method, path, body)

    def status(base):
        import json
        with opener.open(base + '/api/status', timeout=2) as response:
            return json.load(response)['report']['root']

    def closed(base):
        try:
            status(base)
        except OSError:
            return True
        return False

    def pump(seconds=.6):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)

    with tempfile.TemporaryDirectory(prefix='cosplay-owned-backend-') as folder:
        root = Path(folder).resolve()
        env = {'LOCALAPPDATA': str(root / 'profile'), 'COSPLAY_PROVIDER_SETTINGS': str(root / 'providers.json'),
               'COSPLAY_MODEL_SETTINGS': str(root / 'selection.json'), 'NO_PROXY': '127.0.0.1,localhost',
               'COSPLAY_MODEL_REGISTRY': str(Path(__file__).resolve().parents[1] / 'providers/local-models.example.json')}
        with patch.dict(os.environ, env), patch.object(provider_settings, '_ccswitch_record', return_value={'status': 'not_found'}), \
                patch.object(provider_settings, 'resolve_credential', return_value=''), patch.object(qt, 'api_call', side_effect=api):
            old = backend_server.build_server(root / 'old listener', port=0, embedded=True)
            thread = threading.Thread(target=old.serve_forever, daemon=True)
            thread.start()
            requested = f'http://127.0.0.1:{old.server_port}'
            try:
                first = qt.Workbench(root / 'A', requested, embedded_backend=True)
                second = qt.Workbench(root / 'B', requested, embedded_backend=True)
                windows.extend((first, second))
                assert not requests, 'Constructors must not query an unowned listener'
                first.start_backend(); second.start_backend(); pump()
                a, b = first.url, second.url
                assert a != b and a != requested and b != requested, 'Windows listeners must bind exclusively'
                assert Path(status(a)).resolve() == root / 'A' and Path(status(b)).resolve() == root / 'B'
                assert requested not in requests, 'Port collisions must never reuse another workspace'
                assert first.root_edit.isReadOnly() and first.url_edit.isReadOnly()
                first.root_edit.setText(str(root / 'wrong root')); first.url_edit.setText(requested)
                first.start_backend(); first.refresh(); pump()
                assert first.root_path == root / 'A' and first.url == a and requested not in requests
                relay = qt.TaskRelay(first, 'recent', first.job_observation)
                first.close(); pump(.1)
                assert closed(a) and Path(status(b)).resolve() == root / 'B'
                assert Path(status(requested)).resolve() == root / 'old listener'
                count = first.recent_projects.count()
                relay.result({'recent': [{'sample': 'stale', 'run_id': 'stale'}]})
                assert not relay.current() and first.recent_projects.count() == count
                before = len(requests)
                with patch.object(backend_server, 'build_server', side_effect=OSError(errno.EADDRINUSE, 'forced bind failure')):
                    failed = qt.Workbench(root / 'failure', requested, embedded_backend=True)
                    windows.append(failed)
                    failed.start_backend(); failed.refresh(); pump(.1)
                    assert failed.embedded_server is None and len(requests) == before
                    failed.close()
                second.close(); pump(.1)
                assert closed(b) and Path(status(requested)).resolve() == root / 'old listener'
            finally:
                for window in windows:
                    window.close()
                pump(.2)
                qt.QThreadPool.globalInstance().waitForDone(10000)
                app.processEvents()
                old.shutdown(); old.server_close(); thread.join(timeout=3)
    print('OK: exclusive loopback ports, isolated roots, owned routing, stale callbacks and cleanup')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
