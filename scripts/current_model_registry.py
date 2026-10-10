"""One explicit installed-model registry for all current analysis routes."""
from __future__ import annotations

import json
import hashlib
import os
import uuid
from pathlib import Path

from installed_resources import resource_path

SCHEMA = 'cosplay_local_models.current.v1'
PROFILES = {
    'semantics_profile': 'sapiens2_seg_0.4b',
    'matting_profile': 'vitmatte_small_distinctions',
    'dinov3_profile': 'dinov3_vitl16',
    'scene_profile': 'sam31_static_detector_v1',
}
PATH_KEYS = ('python', 'semantics_checkpoint', 'semantics_repo_dir',
             'vitmatte_model_root', 'dinov3_model_root', 'scene_python', 'scene_model_root',
             'birefnet_model_path', 'lama_checkpoint', 'background_sr_checkpoint',
             'face_model_path', 'face_python', 'onnx_runtime_root')
DIRECTORY_KEYS = {'semantics_repo_dir', 'vitmatte_model_root', 'dinov3_model_root',
                  'scene_model_root', 'onnx_runtime_root'}
PATH_LABELS = {
    'python': '人体语义与抠图 Python', 'semantics_checkpoint': 'Sapiens2 0.4B 权重',
    'semantics_repo_dir': 'Sapiens2 源码目录', 'vitmatte_model_root': 'ViTMatte 模型目录',
    'dinov3_model_root': 'DINOv3 ViT-L/16 模型目录', 'scene_python': 'SAM3.1 Python',
    'scene_model_root': 'SAM3.1 安装目录', 'birefnet_model_path': 'BiRefNet ONNX 权重',
    'lama_checkpoint': 'LaMa 权重', 'background_sr_checkpoint': '背景超分权重',
    'face_model_path': '五官定位模型', 'face_python': '五官定位 Python',
    'onnx_runtime_root': 'ONNX Runtime 安装目录',
}


def settings_directory():
    base = Path(os.environ.get('LOCALAPPDATA') or Path.home() / '.local/share')
    return base / 'CosplayWorkbench/config'


def selection_path():
    return Path(os.environ.get('COSPLAY_MODEL_SETTINGS') or settings_directory() / 'model-selection.json')


def registry_location(root=None, *, registry_path=None):
    selected = registry_path or os.environ.get('COSPLAY_MODEL_REGISTRY')
    origin = 'explicit' if registry_path else 'environment' if selected else None
    if not selected and selection_path().is_file():
        saved = json.loads(selection_path().read_text(encoding='utf-8-sig'))
        if not isinstance(saved, dict) or saved.get('schema') != 'cosplay_model_selection.v1' or not isinstance(saved.get('registry_path'), str):
            raise ValueError('Saved model selection is invalid; choose a model configuration again')
        selected, origin = saved['registry_path'], 'user'
        if not selected or not Path(selected).is_absolute():
            raise ValueError('Saved model selection requires an absolute registry path')
    if selected:
        path = Path(selected).expanduser()
    else:
        try:
            path = resource_path('providers/local-models.current.json')
            origin = 'source'
        except FileNotFoundError:
            path = resource_path('providers/local-models.example.json')
            origin = 'template'
    if not path.is_absolute():
        if root is None:
            raise ValueError('Explicit model registry requires an absolute path')
        path = Path(root)/path
    return path.resolve(), origin


def validate_config(config):
    if not isinstance(config, dict) or config.get('schema') != SCHEMA or not isinstance(config.get('ready'), bool):
        raise ValueError('Current installed-model registry schema/readiness required')
    for key, expected in PROFILES.items():
        if config.get(key) != expected:
            raise ValueError('Unsupported installed-model profile: '+key+'='+str(config.get(key)))
    for key in PATH_KEYS:
        value = config.get(key)
        if not isinstance(value, str) or not value or not Path(value).is_absolute():
            raise ValueError('Installed-model registry requires an absolute path: '+PATH_LABELS[key])
    return config


def load_registry(root=None, *, registry_path=None):
    path, _origin = registry_location(root, registry_path=registry_path)
    if not path.is_file():
        raise RuntimeError('Current installed-model registry is unavailable: '+str(path))
    config = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(config, dict) or config.get('schema') != SCHEMA or not isinstance(config.get('ready'), bool):
        raise ValueError('Current installed-model registry schema/readiness required')
    if not config['ready']:
        raise RuntimeError('Local models are not configured. Copy providers/local-models.example.json, '
                           'set installed model/runtime paths and ready=true, then set '
                           'COSPLAY_MODEL_REGISTRY to that file\'s absolute path.')
    validate_config(config)
    return {**config, 'registry_path': path}


def registry_status(root=None, *, registry_path=None):
    """Inspect only config and filesystem metadata; never load weights or runtimes."""
    path, origin = None, None
    config, rows, error = {}, [], None
    try:
        path, origin = registry_location(root, registry_path=registry_path)
        config = json.loads(path.read_text(encoding='utf-8-sig'))
        load_registry(root, registry_path=path)
    except (OSError, ValueError, RuntimeError) as exc:
        error = str(exc)
    if not isinstance(config, dict):
        config = {}
    for key in PATH_KEYS:
        value = config.get(key)
        target = Path(value) if isinstance(value, str) and value else None
        directory = key in DIRECTORY_KEYS
        absolute = bool(target and target.is_absolute())
        try:
            available = bool(absolute and (target.is_dir() if directory else target.is_file()))
        except OSError:
            available = False
        rows.append({'key': key, 'label': PATH_LABELS[key], 'path': str(target) if target else '',
                     'kind': 'directory' if directory else 'file', 'available': available,
                     'status': 'available' if available else 'not_configured' if not target else 'missing'})
    return {'schema': 'cosplay_model_setup.v1', 'registry_path': str(path) if path else None,
            'origin': origin, 'config': {key: config[key] for key in ('schema', 'ready', *PROFILES, *PATH_KEYS) if key in config}, 'configured': error is None,
            'paths_available': error is None and all(row['available'] for row in rows),
            'error': error, 'paths': rows, 'verification_scope': 'configuration_and_path_presence',
            'inference_verified': False}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('x', encoding='utf8', newline='\n') as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def select_registry(path):
    path = Path(path).expanduser().resolve()
    load_registry(registry_path=path)
    atomic_json(selection_path(), {'schema': 'cosplay_model_selection.v1', 'registry_path': str(path)})
    # Apply to this process and its future workers; explicit per-job paths remain authoritative.
    os.environ['COSPLAY_MODEL_REGISTRY'] = str(path)
    return registry_status()


def save_user_config(config):
    validate_config(config)
    if not config['ready']:
        raise ValueError('Model configuration must enable ready after paths are filled')
    values = {key: config[key] for key in ('schema', 'ready', *PROFILES, *PATH_KEYS)}
    identity = hashlib.sha256(json.dumps(values, sort_keys=True, ensure_ascii=False).encode('utf8')).hexdigest()
    # A second window must not overwrite a registry already bound to another job.
    path = settings_directory() / ('local-models.' + identity + '.json')
    atomic_json(path, values)
    return select_registry(path)
