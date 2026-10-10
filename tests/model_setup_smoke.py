"""Exercise persistent model setup and HTTP/UI without executing model code."""
from __future__ import annotations
import json
import os
from pathlib import Path
import tempfile
import threading
import urllib.request
from unittest.mock import patch

import current_model_registry as models
from backend_server import Backend, build_server
import qt_workbench as qt


def main():
    clean = {key: value for key, value in os.environ.items() if not key.startswith('COSPLAY_')}
    with tempfile.TemporaryDirectory(prefix='cosplay-model-setup-') as folder, patch.dict(os.environ, clean, clear=True):
        root = Path(folder).resolve(); profile = root/'profile'
        os.environ.update({'LOCALAPPDATA': str(profile), 'QT_QPA_PLATFORM': 'offscreen',
            'COSPLAY_PROVIDER_SETTINGS': str(profile/'providers.json')})
        example = json.loads((Path(__file__).resolve().parent.parent/'providers/local-models.example.json').read_text('utf8'))
        with patch.object(models, 'resource_path', return_value=root/'example.json'):
            (root/'example.json').write_text(json.dumps(example), encoding='utf8')
            assert not models.registry_status()['configured']
            config = {'schema': models.SCHEMA, 'ready': True, **models.PROFILES}
            for key in models.PATH_KEYS:
                path = root/'installed'/key
                if key in models.DIRECTORY_KEYS:
                    path.mkdir(parents=True)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'synthetic-not-executed')
                config[key] = str(path)
            original = root/'existing.json'; original.write_text(json.dumps(config), encoding='utf-8-sig')
            preserved = original.read_bytes()
            setup = models.select_registry(original)
            assert setup['paths_available'] and not setup['inference_verified']
            os.environ.pop('COSPLAY_MODEL_REGISTRY')
            assert models.load_registry()['registry_path'] == original
            assert models.registry_status()['origin'] == 'user'
            assert original.read_bytes() == preserved
            busy = Backend(root/'busy-workspace')
            job = busy.jobs.create({'kind': 'synthetic-only'})
            selected_before = models.selection_path().read_bytes()
            try:
                busy.model_setup({'registry_path': str(original)})
            except ValueError as error:
                assert '任务' in str(error)
            else:
                raise AssertionError('Active jobs must prevent a model switch')
            assert models.selection_path().read_bytes() == selected_before
            busy.jobs.update(job['job_id'], status='cancelled')
            started, created = threading.Event(), threading.Event()
            def enqueue():
                started.set(); busy.jobs.create({'kind': 'concurrency-fixture'}); created.set()
            with busy.jobs.configuration_change():
                pending = threading.Thread(target=enqueue); pending.start()
                assert started.wait(2) and not created.wait(.1)
            pending.join(timeout=2); assert created.is_set()
            config['api_key'] = 'should-not-persist'
            setup = models.save_user_config(config)
            assert setup['paths_available']
            assert 'should-not-persist' not in Path(setup['registry_path']).read_text('utf8')
            previous = Path(setup['registry_path']); frozen = previous.read_bytes()
            alternative = dict(config); alternative['python'] = alternative['face_python']
            changed = models.save_user_config(alternative)
            assert Path(changed['registry_path']) != previous and previous.read_bytes() == frozen
            models.select_registry(previous)
            Path(config['lama_checkpoint']).unlink()
            setup = models.registry_status()
            assert setup['configured'] and not setup['paths_available']
            assert next(row for row in setup['paths'] if row['key']=='lama_checkpoint')['status'] == 'missing'
            server = build_server(root/'workspace', '127.0.0.1', 0, embedded=True)
            thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
            url = f'http://127.0.0.1:{server.server_address[1]}'
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            def call(method, body=None):
                request = urllib.request.Request(url+'/api/model-setup',
                    data=json.dumps(body).encode() if body is not None else None, method=method,
                    headers={'Content-Type': 'application/json'})
                with opener.open(request, timeout=5) as response: return json.load(response)
            try:
                assert not call('GET')['setup']['paths_available']
                assert call('POST', {'registry_path': str(original)})['setup']['registry_path'] == str(original)
                app = qt.QApplication.instance() or qt.QApplication([])
                window = qt.Workbench(root/'ui-workspace', url, embedded_backend=True)
                with patch.object(window, 'task'):
                    window.open_model_setup()
                    assert len(window.model_path_edits) == len(models.PATH_KEYS)
                    window.model_path_edits['python'].setText('unsaved user edit')
                    window.model_path_edits['python'].setModified(True)
                    window.show_model_setup_status(models.registry_status())
                    assert window.model_path_edits['python'].text() == 'unsaved user edit'
                    payloads = []
                    with patch.object(window, 'embedded_backend', False), patch.object(window, 'task', side_effect=lambda *args: payloads.append(args)):
                        window.save_model_paths()
                    assert payloads[0][1] == '/api/model-setup' and payloads[0][2]['config']['ready']
                    window.model_path_edits['python'].setText('new edit after save submission')
                    window.model_path_edits['python'].setModified(True)
                    window.result('model-setup-saved', {'setup': setup})
                    assert window.model_path_edits['python'].text() == 'new edit after save submission'
                    assert window.model_path_edits['python'].isModified()
                    with patch.dict(os.environ, {'COSPLAY_MODEL_REGISTRY': 'stale-gui-env.json'}):
                        assert window.current_model_options()['model_registry'] == setup['registry_path']
                    window.model_setup_dialog.close(); window.close()
                assert original.read_bytes() == preserved
            finally:
                server.shutdown(); server.server_close(); thread.join(timeout=5)
    print(json.dumps({'ok': True, 'persistent_selection': True, 'missing_paths_reported': True,
        'http_and_gui_form': True, 'original_config_unchanged': True, 'inference_calls': 0}))


if __name__ == '__main__':
    main()
