"""Discover local installation metadata without loading weights or model runtimes."""
from __future__ import annotations

from collections import deque
import copy
import json
import os
from pathlib import Path

import current_model_registry as models

GROUPS = (
    ('人体分区 · Sapiens2', '识别人物与服装区域', ('python', 'semantics_checkpoint', 'semantics_repo_dir')),
    ('人物抠图 · ViTMatte / BiRefNet', '保留人物和发丝边缘', ('vitmatte_model_root', 'birefnet_model_path', 'onnx_runtime_root')),
    ('参考图对应 · DINOv3', '找到原片与参考图的对应区域', ('dinov3_model_root',)),
    ('场景识别 · SAM3.1', '识别背景、地面和遮挡', ('scene_python', 'scene_model_root')),
    ('背景修复 · LaMa', '补全背景空缺', ('lama_checkpoint',)),
    ('背景清晰度 · RealESRGAN', '提升背景分辨率', ('background_sr_checkpoint',)),
    ('人脸定位', '定位每个人的五官', ('face_model_path', 'face_python')),
)
SKIP = {'.git', 'node_modules', '__pycache__', 'runtime', 'site-packages', 'base', 'lora',
        'checkpoints', '_internal', 'dist', 'build', 'outputs', 'data'}
LAYOUT = {
    'python': 'sapiens2/runtime/Scripts/python.exe',
    'semantics_checkpoint': 'sapiens2/checkpoints/sapiens2_0.4b_seg.safetensors',
    'semantics_repo_dir': 'sapiens2/repo', 'vitmatte_model_root': 'vitmatte',
    'dinov3_model_root': 'dinov3', 'scene_python': 'sam3.1/runtime/Scripts/python.exe',
    'scene_model_root': 'sam3.1', 'birefnet_model_path': 'birefnet/birefnet_lite_fp16.onnx',
    'lama_checkpoint': 'lama/big-lama.pt', 'background_sr_checkpoint': 'realesrgan/realesr-general-x4v3.pth',
    'face_model_path': 'face/face_landmarker.task', 'face_python': 'face/runtime/Scripts/python.exe',
    'onnx_runtime_root': 'onnxruntime',
}


def layout_config(directory):
    return {'schema': models.SCHEMA, 'ready': True, **models.PROFILES,
            **{key: str(Path(directory)/relative) for key, relative in LAYOUT.items()}}


def read_metadata(path):
    try:
        if path.stat().st_size > 262144:
            return None
        value = json.loads(path.read_text(encoding='utf-8-sig'))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def beauty_status(path=None):
    from beautygrpo_prepare import installation_path
    from beautygrpo_worker import load_installation
    target = None
    try:
        target = Path(path).resolve() if path else installation_path()
        config = load_installation(target)
        # The portrait parser and landmark runtime are needed as well as FLUX.
        regions = Path(config['regions_model_root'])
        required = [regions/'repo', regions/'checkpoints/convnext_celeba_512/model_299.pt']
        if not required[0].is_dir() or not required[1].is_file():
            raise ValueError('人脸分区模型 SegFace 的源码或权重缺失')
        registry = models.registry_status(registry_path=config['face_registry'])
        if not all(row['available'] for row in registry['paths'] if row['key'] in {'face_model_path', 'face_python'}):
            raise ValueError('BeautyGRPO 所需的人脸定位模型或运行环境缺失')
        return {'available': True, 'config_path': str(target), 'error': None}
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, ImportError) as error:
        return {'available': False, 'config_path': str(target) if target else None, 'error': str(error)}


def setup_status(root=None):
    setup = models.registry_status(root)
    rows = {row['key']: row for row in setup['paths']}
    setup['groups'] = [{'name': name, 'purpose': purpose,
                        'available': setup['configured'] and all(rows[key]['available'] for key in keys),
                        'missing': [rows[key]['label'] for key in keys if not rows[key]['available']]}
                       for name, purpose, keys in GROUPS]
    setup['beauty'] = beauty_status()
    setup['model_directory'] = str(models.model_directory())
    return setup


def search_roots(root=None):
    """Bounded user locations, remembered folders and existing model locations."""
    home = Path.home()
    roots = [models.model_directory(), models.settings_directory(), home/'AppData/Local/CosplayWorkbench/config']
    # MSIX launchers can redirect AppData. Recover that metadata on Explorer launches too.
    packages = home/'AppData/Local/Packages'
    try:
        roots += [item/'LocalCache/Local/CosplayWorkbench/config' for item in packages.glob('OpenAI.Codex_*')]
    except OSError:
        pass
    saved = read_metadata(models.settings_directory()/'model-search.json') or {}
    roots += [Path(item) for item in saved.get('directories', []) if isinstance(item, str) and Path(item).is_absolute()]
    if os.environ.get('COSPLAY_MODELS_ROOT'):
        roots.append(Path(os.environ['COSPLAY_MODELS_ROOT']))
    if root:
        roots.append(Path(root))
    roots += [home/'CosplayWorkbench/models', home/'Desktop', home/'Downloads']
    setup = models.registry_status(root)
    roots += related_roots(setup['config'])
    return roots


def related_roots(config):
    roots = []
    for key in models.PATH_KEYS:
        value = config.get(key)
        if isinstance(value, str) and Path(value).is_absolute():
            target = Path(value)
            if target.exists():
                roots.append(target.parent if key in models.DIRECTORY_KEYS else target.parent.parent)
    return roots


def metadata_candidates(roots):
    """Never walk weight/runtime contents or execute discovered Python files."""
    pending = deque((Path(root), 0) for root in roots)
    visited, candidates = set(), []
    while pending and len(visited) < 1500:
        directory, depth = pending.popleft()
        try:
            resolved = directory.resolve()
            if resolved in visited or not directory.is_dir():
                continue
            visited.add(resolved)
            entries = sorted(directory.iterdir(), key=lambda item: item.name.casefold())
            for path in entries:
                if path.is_file() and path.suffix.lower() == '.json' and (
                        path.name.startswith(('local-models', 'beautygrpo'))
                        or path.name in {'model-selection.json', 'installation.json'}):
                    candidates.append(path)
                elif depth < 4 and path.is_dir() and path.name not in SKIP and not path.is_symlink() and not path.is_junction():
                    pending.append((path, depth+1))
        except OSError:
            continue
    return candidates


def save_beauty_selection(path):
    """Remember the verified installation without changing its receipt-bound metadata."""
    from beautygrpo_prepare import SELECTION_SCHEMA, selection_path
    status = beauty_status(path)
    if not status['available']:
        raise ValueError(status['error'])
    target = status['config_path']
    models.atomic_json(selection_path(), {'schema': SELECTION_SCHEMA, 'model_config': target})
    os.environ['COSPLAY_BEAUTYGRPO_CONFIG'] = target
    return status


def discover_and_configure(root=None, directory=None):
    if directory is not None and (not isinstance(directory, str) or not Path(directory).is_absolute() or not Path(directory).is_dir()):
        raise ValueError('请选择存在的模型文件夹')
    before = setup_status(root)
    roots = [Path(directory)] if directory else search_roots(root)
    primary = Path(directory) if directory else models.model_directory()
    roots.append(primary/'beautygrpo')
    conventional = layout_config(primary)
    # Standard layout needs no JSON editing. Resolve it into a portable local descriptor.
    presence = {}
    for key, value in conventional.items():
        if key not in models.PATH_KEYS:
            continue
        target = Path(value)
        presence[key] = target.is_dir() if key in models.DIRECTORY_KEYS else target.is_file()
    if all(presence.values()) and not (primary/'local-models.json').exists():
        models.atomic_json(primary/'local-models.json', {'schema': models.SCHEMA, 'ready': True, **models.PROFILES, **LAYOUT})
    candidates = metadata_candidates(roots)
    analyses, beauties = [], []
    inspected = set()

    def inspect(paths):
        for path in paths:
            if path in inspected:
                continue
            inspected.add(path)
            config = read_metadata(path)
            if not config:
                continue
            if config.get('schema') == 'cosplay_model_selection.v1':
                selected = config.get('registry_path')
                if isinstance(selected, str) and Path(selected).is_absolute():
                    inspect([Path(selected)])
            elif config.get('schema') == 'beautygrpo_model_selection.v1':
                selected = config.get('model_config')
                if isinstance(selected, str) and Path(selected).is_absolute():
                    inspect([Path(selected)])
            elif config.get('schema') == models.SCHEMA:
                analyses.append(models.registry_status(registry_path=path))
            elif config.get('schema') == 'beautygrpo_models.v1' and beauty_status(path)['available']:
                beauties.append(path)

    inspect(candidates)
    analyses.sort(key=lambda setup: (setup['paths_available'], sum(row['available'] for row in setup['paths'])), reverse=True)
    if analyses:
        # Metadata may point to models on another drive; inspect only their nearby folders.
        inspect(metadata_candidates(related_roots(analyses[0]['config'])))
    if (directory is not None or not before['paths_available']) and analyses and analyses[0]['paths_available']:
        models.save_user_config(analyses[0]['config'], prefer_user=directory is not None)
    if not before['beauty']['available'] and beauties:
        save_beauty_selection(beauties[0])
    if directory:
        remembered = read_metadata(models.settings_directory()/'model-search.json') or {}
        folders = [directory]+[item for item in remembered.get('directories', []) if item != directory and isinstance(item, str)]
        models.atomic_json(models.settings_directory()/'model-search.json', {'directories': folders[:8]})
    setup = setup_status(root)
    setup['discovery'] = {'inspected_configurations': len(inspected),
                          'analysis_found': bool(analyses and analyses[0]['paths_available']),
                          'beauty_found': bool(beauties),
                          'paths': copy.deepcopy(analyses[0]['paths']) if analyses and not setup['paths_available'] else []}
    if not setup['paths_available'] and not setup['discovery']['paths']:
        setup['discovery']['paths'] = [{'key': key, 'path': conventional[key], 'available': presence[key]}
                                      for key in models.PATH_KEYS]
    return setup
