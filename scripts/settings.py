"""Read configuration; load a credential only for the model call using it."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from copy import deepcopy
import uuid

from prompt_library import load_library


def resource_path(relative):
    root = Path(sys.executable).parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent.parent
    return root / relative


def default_settings_path():
    return Path(os.environ.get('COSLUMA_CONFIG', Path.home() / 'CosplayWorkbench/config/cosluma.json'))


def load_settings(path=None):
    path = Path(path or default_settings_path()).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'找不到配置文件：{path}。请在桌面“连接设置”中保存配置，或使用 --config 指定已有配置。')
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


def save_settings(path, config, keys=None):
    path = Path(path).expanduser().resolve()
    config = deepcopy(config)
    library = Path(config['prompt_library']).expanduser()
    library = (library if library.is_absolute() else path.parent / library).resolve()
    supplied = library == path.parent / 'prompts.json' and not library.exists()
    if not supplied and not library.is_file():
        raise ValueError(f'提示词库不存在：{library}。请选择已有 JSON 文件，或使用默认通用库。')
    try:
        entries = load_library(resource_path('providers/prompts.example.json') if supplied else library)
    except (ValueError, UnicodeError) as error:
        raise ValueError('提示词库不是有效的 UTF-8 JSON 文件，请修正或选择通用库。') from error
    if not isinstance(entries, list) or not entries or any(
            not isinstance(item, dict) or not all(isinstance(item.get(key), str) and item[key].strip()
                for key in ('title', 'text', 'source')) for item in entries):
        raise ValueError('提示词库须为非空 JSON 数组，每条包含 title、text、source 文本。')
    path.parent.mkdir(parents=True, exist_ok=True)
    if supplied:
        content = resource_path('providers/prompts.example.json').read_bytes()
        with library.open('xb') as stream:
            stream.write(content)
    config['prompt_library'] = str(library)
    created = []
    previous = []
    temporary = path.with_name(path.name + '.tmp')
    try:
        for service, secret in (keys or {}).items():
            if service not in ('llm', 'image') or not secret.strip():
                raise ValueError('请填写有效的服务密钥')
            provider = config[service]
            old = provider.get('api_key_file')
            if old:
                old = Path(old).expanduser()
                old = (old if old.is_absolute() else path.parent / old).resolve()
                prefix = f'.{path.stem}-{service}-'
                identifier = old.stem[len(prefix):] if old.name.startswith(prefix) else ''
                if old.parent == path.parent and old.suffix == '.key' and len(identifier) == 32 and all(
                        character in '0123456789abcdef' for character in identifier):
                    previous.append(old)
            target = path.parent / f'.{path.stem}-{service}-{uuid.uuid4().hex}.key'
            with os.fdopen(os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w', encoding='utf-8') as stream:
                created.append(target)
                stream.write(secret.strip())
            provider.pop('api_key_env', None)
            provider['api_key_file'] = str(target)
        temporary.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        os.replace(temporary, path)
    except Exception:
        for target in created:
            target.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)
    for old in previous:
        referenced = []
        for service in ('llm', 'image'):
            value = config.get(service, {}).get('api_key_file')
            if value:
                value = Path(value).expanduser()
                referenced.append((value if value.is_absolute() else path.parent / value).resolve())
        if old in referenced:
            continue
        try:
            old.unlink(missing_ok=True)
        except OSError:
            # The new configuration is already committed; a locked obsolete key
            # must not turn a successful save into a false failure.
            pass
    return config


def credential(config):
    if 'api_key_file' in config:
        return Path(config['api_key_file']).read_text(encoding='utf-8-sig').strip()
    name = config.get('api_key_env')
    return os.environ[name] if name else None
