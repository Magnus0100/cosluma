"""Read configuration; load a credential only for the model call using it."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys


def resource_path(relative):
    root = Path(sys.executable).parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent.parent
    return root / relative


def default_settings_path():
    return Path(os.environ.get('COSLUMA_CONFIG', Path.home() / 'CosplayWorkbench/config/cosluma.json'))


def load_settings(path=None):
    path = Path(path or default_settings_path()).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'找不到配置文件：{path}。请在桌面“服务设置”中保存配置，或使用 --config 指定已有配置。')
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


def editable_settings(path):
    path = Path(path).expanduser().resolve()
    if path.is_file():
        return load_settings(path)
    config = json.loads(resource_path('providers/settings.example.json').read_text(encoding='utf-8-sig'))
    config['prompt_library'] = str(path.parent / 'prompts.json')
    if getattr(sys, 'frozen', False):
        config.pop('python', None)
    else:
        config['python'] = sys.executable
    return config


def save_settings(path, config):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    library = Path(config['prompt_library']).expanduser()
    library = library if library.is_absolute() else path.parent / library
    if library.resolve() == path.parent / 'prompts.json' and not library.exists():
        # Copy the supplied catalogue only when explicitly saving first-time settings.
        # An existing user catalogue is never replaced.
        with library.open('xb') as stream:
            stream.write(resource_path('providers/prompts.example.json').read_bytes())
    temporary = path.with_name(path.name + '.tmp')
    try:
        temporary.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def credential(config):
    if 'api_key_file' in config:
        return Path(config['api_key_file']).read_text(encoding='utf-8-sig').strip()
    name = config.get('api_key_env')
    return os.environ[name] if name else None
