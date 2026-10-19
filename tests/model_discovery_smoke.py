"""Recovery from empty/frozen and redirected settings, without weight inference."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

from backend_server import Backend
import current_model_registry as models
import model_setup as setup
import beautygrpo_prepare as beauty
from design_reconstruction_ui_smoke import Window, Pool, last
import qt_workbench as qt


def installed_fixture(root):
    config = {'schema': models.SCHEMA, 'ready': True, **models.PROFILES}
    for key in models.PATH_KEYS:
        target = root/'models'/key
        if key in models.DIRECTORY_KEYS:
            target.mkdir(parents=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(b'synthetic-only')
        config[key] = str(target)
    return config


def main():
    clean = {key: value for key, value in os.environ.items() if not key.startswith('COSPLAY_')}
    app = qt.QApplication.instance() or qt.QApplication([])
    with tempfile.TemporaryDirectory(prefix='cosplay-model-discovery-') as temporary, patch.dict(os.environ, clean, clear=True):
        root = Path(temporary).resolve(); home = root/'home'
        os.environ['LOCALAPPDATA'] = str(home/'AppData/Local')
        os.environ['COSPLAY_MODELS_ROOT'] = str(root/'model')
        example = root/'example.json'
        example.write_text(json.dumps({'schema': models.SCHEMA, 'ready': False, **models.PROFILES}), encoding='utf8')
        config = installed_fixture(root)
        original = root/'installed-project/providers/local-models.current.json'
        original.parent.mkdir(parents=True); original.write_text(json.dumps(config), encoding='utf8')
        preserved = original.read_bytes()
        redirected = home/'AppData/Local/Packages/OpenAI.Codex_test/LocalCache/Local/CosplayWorkbench/config'
        models.atomic_json(redirected/'model-selection.json', {'schema': 'cosplay_model_selection.v1', 'registry_path': str(original)})
        with patch.object(Path, 'home', return_value=home), patch.object(models, 'resource_path', return_value=example):
            assert not setup.setup_status(root)['paths_available']
            backend = Backend(root/'workspace')
            recovered = backend.model_setup({'action': 'discover'})['setup']
            assert recovered['paths_available'] and all(group['available'] for group in recovered['groups'])
            assert recovered['registry_path'] != str(original)
            os.environ.pop('COSPLAY_MODEL_REGISTRY')
            original.unlink()
            assert Backend(root/'restarted').model_setup()['setup']['paths_available']
            # A later discovery must not overwrite a valid user-selected installation.
            selected_bytes = models.selection_path().read_bytes()
            backend.model_setup({'action': 'discover', 'directory': str(root/'models')})
            assert models.selection_path().read_bytes() == selected_bytes

            # Broken saved selection is recoverable using a single chosen directory.
            models.atomic_json(models.selection_path(), {'schema': 'broken'})
            original.write_bytes(preserved)
            repaired = backend.model_setup({'action': 'discover', 'directory': str(original.parent.parent)})['setup']
            assert repaired['paths_available'] and original.read_bytes() == preserved
            active = backend.jobs.create({'kind': 'synthetic'})
            try:
                backend.model_setup({'action': 'discover'})
            except ValueError as error:
                assert '任务' in str(error)
            else:
                raise AssertionError('Discovery must not switch models under an active job')
            backend.jobs.update(active['job_id'], status='cancelled')

            # Portrait configuration is separately visible and survives restart.
            regions = root/'portrait/segface'; (regions/'repo').mkdir(parents=True)
            checkpoint = regions/'checkpoints/convnext_celeba_512/model_299.pt'
            checkpoint.parent.mkdir(parents=True); checkpoint.write_bytes(b'synthetic-parser')
            portrait = root/'portrait/installation.json'
            portrait_config = {'schema': 'beautygrpo_models.v1', 'ready': True,
                'face_registry': repaired['registry_path'], 'regions_model_root': str(regions),
                'verification_receipt': 'installation.verification.json'}
            portrait.write_text(json.dumps(portrait_config), encoding='utf8')
            portrait_preserved = portrait.read_bytes()
            receipt_path = portrait.with_suffix('.verification.json')
            receipt = {'schema': 'beautygrpo_verification.v1', 'ready': True,
                'config_sha256': hashlib.sha256(portrait_preserved).hexdigest()}
            receipt_path.write_text(json.dumps(receipt), encoding='utf8')
            receipt_preserved = receipt_path.read_bytes()
            loaded_portrait = {**portrait_config, 'config_path': str(portrait),
                'config_sha256': receipt['config_sha256'], 'verification_receipt': receipt,
                'verification_receipt_path': str(receipt_path)}
            def load_portrait(path):
                if Path(path).resolve() != portrait:
                    raise ValueError('Synthetic verification receipt is bound to the original installation')
                assert portrait.read_bytes() == portrait_preserved
                assert receipt_path.read_bytes() == receipt_preserved
                return loaded_portrait
            with patch('beautygrpo_worker.load_installation', side_effect=load_portrait):
                saved = setup.save_beauty_selection(portrait)
                assert saved == {'available': True, 'config_path': str(portrait), 'error': None}
                assert json.loads(beauty.selection_path().read_text('utf8')) == {
                    'schema': beauty.SELECTION_SCHEMA, 'model_config': str(portrait)}
                assert not list(models.settings_directory().glob('beautygrpo.*.json'))
                configured = backend.model_setup({'beauty_config_path': str(portrait)})['setup']
                assert configured['beauty']['available']
                os.environ.pop('COSPLAY_BEAUTYGRPO_CONFIG')
                portable = models.model_directory()/'beautygrpo.json'
                models.atomic_json(portable, portrait_config)
                assert beauty.installation_path() == portrait
                assert setup.setup_status()['beauty']['available']
                selected_bytes = beauty.selection_path().read_bytes()
                with patch('beautygrpo_worker.load_installation', side_effect=ValueError('invalid verification receipt')):
                    try:
                        setup.save_beauty_selection(portable)
                    except ValueError as error:
                        assert 'verification receipt' in str(error)
                    else:
                        raise AssertionError('Invalid installation must not replace the saved selection')
                assert beauty.selection_path().read_bytes() == selected_bytes
                assert 'COSPLAY_BEAUTYGRPO_CONFIG' not in os.environ
                # Discovery preserves the installation's verified face registry even after analysis selection changes.
                alternative = {**config, 'python': config['face_python']}
                switched = models.save_user_config(alternative)
                assert switched['registry_path'] != portrait_config['face_registry']
                beauty.selection_path().unlink()
                rediscovered = backend.model_setup({'action': 'discover', 'directory': str(portrait.parent)})['setup']
                assert rediscovered['beauty']['available'] and beauty.installation_path() == portrait
                assert portrait.read_bytes() == portrait_preserved and receipt_path.read_bytes() == receipt_preserved
                window = Window(root/'ui-workspace', 'http://127.0.0.1:1'); window.pool = Pool()
                window.model_auto_discovery_attempted = True
                window.show_model_setup_status(configured); window.open_model_setup()
                assert window.models_button.isVisibleTo(window) and len(window.model_group_states) == len(setup.GROUPS)
                assert '已找到' in window.beauty_model_state.text()
                window.discover_models(str(root/'models'))
                assert last(window.pool, 'model-setup-saved').body == {'action': 'discover', 'directory': str(root/'models')}
                window.result('model-setup-saved', {'setup': configured})
                assert window.model_setup_submission is None
                window.model_setup_dialog.close(); window.close()
            # The conventional folder works without hand-written JSON and remains portable.
            conventional = Path(os.environ['COSPLAY_MODELS_ROOT'])
            for key, relative in setup.LAYOUT.items():
                target = conventional/relative
                if key in models.DIRECTORY_KEYS:
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(b'synthetic-only')
            os.environ.pop('COSPLAY_MODEL_REGISTRY', None)
            models.selection_path().unlink()
            result = backend.model_setup({'action': 'discover'})['setup']
            assert result['paths_available']
            descriptor = json.loads((conventional/'local-models.json').read_text('utf8'))
            assert all(not Path(descriptor[key]).is_absolute() for key in models.PATH_KEYS)
            moved = root/'moved-installation/model'; moved.parent.mkdir()
            conventional.rename(moved)
            os.environ.pop('COSPLAY_MODEL_REGISTRY', None)
            os.environ['COSPLAY_MODELS_ROOT'] = str(moved)
            after_move = models.registry_status()
            assert after_move['paths_available'] and after_move['origin'] == 'model_folder'
            assert all(Path(row['path']).is_relative_to(moved) for row in after_move['paths'])
            manual = models.save_user_config(config)
            os.environ.pop('COSPLAY_MODEL_REGISTRY', None)
            assert models.registry_status()['registry_path'] == manual['registry_path']
            assert models.registry_status()['origin'] == 'user'
            binding = beauty._cache_binding({'source_sha256': 'synthetic', 'source_display': {},
                'source_precision': {}, 'native_size': [1, 1], 'native_source_retouch': None, 'model_input_sha256': 'synthetic'},
                {'face_registry': str(moved/'local-models.json'), 'base_manifest_sha256': 'synthetic',
                 'offload': 'sequential_cpu_offload'}, {'max_faces': 8})
            from beautygrpo_worker import sha
            assert binding['face_landmarker_sha256'] == sha(moved/setup.LAYOUT['face_model_path'])
    print(json.dumps({'ok': True, 'redirected_selection_recovered': True, 'restart_without_source_config': True,
        'corrupt_selection_repaired': True, 'beauty_config_persistent': True,
        'model_management_ui': True, 'inference_calls': 0, 'beauty_installation_metadata_mocked': True}))


if __name__ == '__main__': main()
