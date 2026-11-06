"""Read configuration; load a credential only for the model call using it."""
from __future__ import annotations

import json
import os
from pathlib import Path


def default_settings_path():
    return Path(os.environ.get('COSLUMA_CONFIG', Path.home() / 'CosplayWorkbench/config/cosluma.json'))


def load_settings(path=None):
    path = Path(path or default_settings_path()).expanduser().resolve()
    config = json.loads(path.read_text(encoding='utf-8-sig'))
    def resolve(filename):
        value = Path(filename).expanduser()
        return str((value if value.is_absolute() else path.parent / value).resolve())
    for key in ('prompt_library',):
        if key in config:
            config[key] = resolve(config[key])
    for service in ('llm', 'image'):
        provider = config.get(service, {})
        if 'api_key_file' in provider:
            provider['api_key_file'] = resolve(provider['api_key_file'])
    return config


def credential(config):
    if 'api_key_file' in config:
        return Path(config['api_key_file']).read_text(encoding='utf-8-sig').strip()
    name = config.get('api_key_env')
    return os.environ[name] if name else None
