"""One explicit installed-model registry for all current analysis routes."""
from __future__ import annotations

import json
import hashlib
import os
import uuid
import sys
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

# A registry describes the installable components, while a task only consumes
# the capabilities it actually routes through.  Keep this mapping in one
# place so the preflight, model manager and task plan cannot drift apart.
CAPABILITY_KEYS = {
    'source_silhouette': frozenset(('birefnet_model_path', 'onnx_runtime_root')),
    'source_geometry': frozenset((
        'python', 'semantics_checkpoint', 'semantics_repo_dir',
        'vitmatte_model_root', 'birefnet_model_path', 'onnx_runtime_root')),
    'reference_correspondence': frozenset(('dinov3_model_root',)),
    'scene_analysis': frozenset(('scene_python', 'scene_model_root')),
    'portrait_regions': frozenset(('face_model_path', 'face_python')),
    'background_completion': frozenset(('lama_checkpoint',)),
    'background_superresolution': frozenset(('background_sr_checkpoint',)),
}
CAPABILITY_LABELS = {
    'source_silhouette': 'BiRefNet 人物轮廓与 ONNX 运行时',
    'source_geometry': '人物语义、alpha 与发丝边缘',
    'reference_correspondence': '原片与参考图对应',
    'scene_analysis': '场景实例与遮挡分析',
    'portrait_regions': '五官与人脸区域',
    'background_completion': '背景补全',
    'background_superresolution': '背景高清重建',
}
REQUIRED_REGISTRY_KEYS = frozenset(CAPABILITY_KEYS['source_geometry'])

# Registry selection and validation are queried at nearly every preparation
# boundary.  Keep the parsed JSON process-local and invalidate it on the full
# stat identity (size + mtime + ctime).  Validation still runs on every
# ``load_registry`` call; this cache only avoids repeated disk reads/parsing.
_JSON_CACHE_LIMIT = 8
_REGISTRY_JSON_CACHE: dict[tuple[str, int, int, int], dict] = {}
_SELECTION_JSON_CACHE: dict[tuple[str, int, int, int], dict] = {}


def _cached_json(path, cache):
    path = Path(path).resolve()
    stat = path.stat()
    key = (str(path), int(stat.st_size), int(stat.st_mtime_ns), int(stat.st_ctime_ns))
    value = cache.get(key)
    if value is not None:
        return dict(value)
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(value, dict):
        return value
    if len(cache) >= _JSON_CACHE_LIMIT:
        cache.pop(next(iter(cache)))
    cache[key] = dict(value)
    return dict(value)


def settings_directory():
    configured = os.environ.get('COSPLAY_CONFIG_ROOT')
    if configured:
        return Path(configured).expanduser().resolve()
    base = Path(os.environ.get('LOCALAPPDATA') or Path.home() / '.local/share')
    return base / 'CosplayWorkbench/config'


def selection_path():
    return Path(os.environ.get('COSPLAY_MODEL_SETTINGS') or settings_directory() / 'model-selection.json')


def model_directory():
    """The user-visible model folder beside the EXE or source launcher."""
    override = os.environ.get('COSPLAY_MODELS_ROOT')
    base = Path(sys.executable).parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent.parent
    return Path(override).expanduser().resolve() if override else base/'model'


def resolve_config_paths(config, directory):
    if not isinstance(config, dict):
        return config
    config = dict(config)
    for key in PATH_KEYS:
        value = config.get(key)
        if isinstance(value, str) and value:
            package_root = os.environ.get('COSPLAY_PACKAGE_ROOT')
            if package_root is None and getattr(sys, 'frozen', False):
                package_root = str(Path(sys.executable).resolve().parent)
            if package_root:
                value = value.replace('${PACKAGE_ROOT}', str(Path(package_root).resolve()))
            path = Path(value).expanduser()
            # Keep an explicitly absolute path lexical.  Calling ``resolve``
            # here follows Windows junctions, which previously turned the
            # declared D: model layout into a stale C:\\...\\.cache target
            # without making the environment mix visible to the caller.  A
            # relative portable registry still resolves against its own
            # directory, while an absolute path remains bound to the drive
            # selected by the installer and fails clearly if that asset is
            # missing there.
            config[key] = str(path if path.is_absolute() else (Path(directory)/path).resolve())
    return config


def registry_location(root=None, *, registry_path=None):
    selected = registry_path or os.environ.get('COSPLAY_MODEL_REGISTRY')
    origin = 'explicit' if registry_path else 'environment' if selected else None
    portable = model_directory()/'local-models.json'
    if not selected and selection_path().is_file():
        saved = _cached_json(selection_path(), _SELECTION_JSON_CACHE)
        if not isinstance(saved, dict) or saved.get('schema') != 'cosplay_model_selection.v1' or not isinstance(saved.get('registry_path'), str):
            raise ValueError('Saved model selection is invalid; choose a model configuration again')
        if not saved['registry_path'] or not Path(saved['registry_path']).is_absolute():
            raise ValueError('Saved model selection requires an absolute registry path')
        if saved.get('prefer_user') is True or not portable.is_file():
            selected, origin = saved['registry_path'], 'user'
    if not selected and portable.is_file():
        selected, origin = str(portable), 'model_folder'
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


def validate_config(config, *, required_keys=None, required_profiles=None):
    if not isinstance(config, dict) or config.get('schema') != SCHEMA or not isinstance(config.get('ready'), bool):
        raise ValueError('Current installed-model registry schema/readiness required')
    profiles = PROFILES if required_profiles is None else {
        key: PROFILES[key] for key in required_profiles if key in PROFILES}
    unknown_profiles = set(required_profiles or ()) - set(PROFILES) if required_profiles is not None else set()
    if unknown_profiles:
        raise ValueError('Unknown installed-model profiles: ' + ', '.join(sorted(unknown_profiles)))
    for key, expected in profiles.items():
        if config.get(key) != expected:
            raise ValueError('Unsupported installed-model profile: '+key+'='+str(config.get(key)))
    # Optional route components may be omitted from a current registry.  They
    # are rejected by the task preflight only when that route requests their
    # capability.  This lets a local portrait task run without stale scene or
    # correspondence installations while retaining strict path validation for
    # the source geometry stack needed by every route.
    keys = REQUIRED_REGISTRY_KEYS if required_keys is None else frozenset(required_keys)
    unknown = keys - set(PATH_KEYS)
    if unknown:
        raise ValueError('Unknown installed-model registry keys: ' + ', '.join(sorted(unknown)))
    for key in keys:
        value = config.get(key)
        if not isinstance(value, str) or not value or not Path(value).is_absolute():
            raise ValueError('Installed-model registry requires an absolute path: '+PATH_LABELS[key])
    return config


def load_registry(root=None, *, registry_path=None, required_keys=None, required_profiles=None):
    path, _origin = registry_location(root, registry_path=registry_path)
    if not path.is_file():
        raise RuntimeError('Current installed-model registry is unavailable: '+str(path))
    config = resolve_config_paths(_cached_json(path, _REGISTRY_JSON_CACHE), path.parent)
    if not isinstance(config, dict) or config.get('schema') != SCHEMA or not isinstance(config.get('ready'), bool):
        raise ValueError('Current installed-model registry schema/readiness required')
    if not config['ready']:
        raise RuntimeError('Local models are not configured. Copy providers/local-models.example.json, '
                           'set installed model/runtime paths and ready=true, then set '
                           'COSPLAY_MODEL_REGISTRY to that file\'s absolute path.')
    validate_config(config, required_keys=required_keys, required_profiles=required_profiles)
    return {**config, 'registry_path': path}


def registry_status(root=None, *, registry_path=None, required_keys=None, required_profiles=None):
    """Inspect only config and filesystem metadata; never load weights or runtimes."""
    path, origin = None, None
    config, rows, error = {}, [], None
    try:
        path, origin = registry_location(root, registry_path=registry_path)
        config = resolve_config_paths(_cached_json(path, _REGISTRY_JSON_CACHE), path.parent)
        load_registry(root, registry_path=path, required_keys=required_keys,
                      required_profiles=required_profiles)
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


def select_registry(path, *, prefer_user=True):
    path = Path(path).expanduser().resolve()
    load_registry(registry_path=path)
    atomic_json(selection_path(), {'schema': 'cosplay_model_selection.v1', 'registry_path': str(path), 'prefer_user': prefer_user})
    # Apply to this process and its future workers; explicit per-job paths remain authoritative.
    os.environ['COSPLAY_MODEL_REGISTRY'] = str(path)
    return registry_status()


def save_user_config(config, *, prefer_user=True):
    validate_config(config)
    if not config['ready']:
        raise ValueError('Model configuration must enable ready after paths are filled')
    values = {key: config[key] for key in ('schema', 'ready', *PROFILES, *PATH_KEYS)}
    identity = hashlib.sha256(json.dumps(values, sort_keys=True, ensure_ascii=False).encode('utf8')).hexdigest()
    # A second window must not overwrite a registry already bound to another job.
    path = settings_directory() / ('local-models.' + identity + '.json')
    atomic_json(path, values)
    return select_registry(path, prefer_user=prefer_user)
