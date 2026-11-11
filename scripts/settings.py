"""Read configuration; load a credential only for the model call using it."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
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
    library = config.get('prompt_library')
    config['prompt_library'] = resolve(library) if library is not None else None
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
    if getattr(sys, 'frozen', False):
        config.pop('python', None)
    else:
        config['python'] = sys.executable
    return config


def owned_key_file(path, service, value):
    """The pasted-key file this configuration wrote for one service, if that is what the value names."""
    if not value:
        return None
    candidate = Path(value).expanduser()
    candidate = (candidate if candidate.is_absolute() else path.parent / candidate).resolve()
    prefix = f'.{path.stem}-{service}-'
    if candidate.parent != path.parent or candidate.suffix != '.key' or not candidate.name.startswith(prefix):
        return None
    identifier = candidate.stem[len(prefix):]
    return candidate if len(identifier) == 32 and all(
        character in '0123456789abcdef' for character in identifier) else None


def previous_key_files(path):
    """Keys the persisted configuration referenced before this save; an unreadable file only skips this cleanup."""
    try:
        previous = load_settings(path)
        values = [previous.get(service, {}).get('api_key_file') for service in ('llm', 'image')]
    except (OSError, ValueError, AttributeError, TypeError):
        return []
    files = []
    for service, value in zip(('llm', 'image'), values):
        owned = owned_key_file(path, service, value)
        if owned is not None:
            files.append(owned)
    return files


def save_settings(path, config, keys=None):
    path = Path(path).expanduser().resolve()
    config = deepcopy(config)
    library = config.get('prompt_library')
    if library is not None:
        library = Path(library).expanduser()
        library = (library if library.is_absolute() else path.parent / library).resolve()
        if not library.is_file():
            raise ValueError(f'自定义参考词库不存在：{library}。请修正路径，或在连接设置中选择内置参考库。')
    try:
        entries = load_library(library)
    except (ValueError, UnicodeError) as error:
        raise ValueError('参考词库不是有效的 UTF-8 JSON 文件，请修正文件。') from error
    if not isinstance(entries, list) or not entries or any(
            not isinstance(item, dict) or not all(isinstance(item.get(key), str) and item[key].strip()
                for key in ('title', 'text')) or not isinstance(item.get('source'), (str, dict))
                or not item['source'] for item in entries):
        raise ValueError('参考词库须为非空 JSON 数组，每条包含 title、text 文本与 source 来源。')
    path.parent.mkdir(parents=True, exist_ok=True)
    config['prompt_library'] = str(library) if library is not None else None
    obsolete = previous_key_files(path)
    created = []
    temporary = None
    try:
        for service, secret in (keys or {}).items():
            if service not in ('llm', 'image') or not secret.strip():
                raise ValueError('请填写有效的服务密钥')
            provider = config[service]
            target = path.parent / f'.{path.stem}-{service}-{uuid.uuid4().hex}.key'
            with os.fdopen(os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w', encoding='utf-8') as stream:
                created.append(target)
                stream.write(secret.strip())
            provider.pop('api_key_env', None)
            provider['api_key_file'] = str(target)
        descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.', suffix='.tmp')
        temporary = Path(name)
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            stream.write(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
        os.replace(temporary, path)
        temporary = None
    except Exception:
        for target in created:
            target.unlink(missing_ok=True)
        raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    referenced = set()
    for service in ('llm', 'image'):
        value = config.get(service, {}).get('api_key_file')
        if value:
            value = Path(value).expanduser()
            referenced.add((value if value.is_absolute() else path.parent / value).resolve())
    for value in obsolete:
        if value in referenced:
            continue
        try:
            value.unlink(missing_ok=True)
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
