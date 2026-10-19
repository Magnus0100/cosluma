"""Verify public setup boundaries without models, network or user settings."""
from pathlib import Path
import os
import sys
import tempfile
from unittest.mock import patch

import current_model_registry as registry
from installed_resources import resource_path
import project_cli
import qt_workbench
import start_qwen_image as qwen


def main():
    example = resource_path('providers/local-models.example.json')
    with patch.dict(os.environ, {}, clear=True):
        try:
            registry.load_registry(registry_path=example)
        except RuntimeError as error:
            assert 'COSPLAY_MODEL_REGISTRY' in str(error)
        else:
            raise AssertionError('Unconfigured registry must stop before model path access')
        catalog = qwen.model_catalog()
        catalog['deployment_root'] = ''
        with patch.object(qwen, 'model_catalog', return_value=catalog):
            assert qwen.deployment_root(required=False) is None
            files = qwen.model_files('qwen_image_2_1')
            assert files and all(e['path'] is None and not e['complete'] for e in files)
            try:
                qwen.deployment_root()
            except RuntimeError as error:
                assert 'COSPLAY_QWEN_DEPLOY_ROOT' in str(error)
            else:
                raise AssertionError('Empty deployment must never resolve to cwd')
            assert not qwen.ensure_running(endpoint='https://service.invalid')['managed']
        with tempfile.TemporaryDirectory(prefix='cosplay-public-start-') as folder:
            profile = Path(folder).resolve()
            with patch.object(sys, 'frozen', True, create=True), patch.object(Path, 'home', return_value=profile), patch.dict(os.environ, {'LOCALAPPDATA': str(profile)}):
                assert qt_workbench._default_repo_root() == profile/'CosplayWorkbench/workspace'
            chosen = profile/'chosen workspace'
            with patch.dict(os.environ, {'COSPLAY_WORKSPACE': str(chosen)}):
                assert qt_workbench._default_repo_root() == chosen
            assert project_cli.native_project_record(chosen)['default_sample'] == 'source'
    print('OK: public model setup errors, remote Qwen ownership, writable default and workspace override')


if __name__ == '__main__':
    main()
