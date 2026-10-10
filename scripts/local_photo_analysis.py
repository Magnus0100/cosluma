"""Local model orchestration. GPU dependencies stay in an external worker runtime."""
from __future__ import annotations

import hashlib
from functools import lru_cache
import json
import os
from pathlib import Path
import subprocess
import time
from runtime_resources import code_path

import numpy as np
from PIL import Image, ImageCms


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def encode_mask(mask):
    """Portable row-major binary RLE, first count always describes zeros."""
    values = np.asarray(mask, dtype=np.uint8).reshape(-1)
    changes = np.flatnonzero(values[1:] != values[:-1]) + 1
    lengths = np.diff(np.r_[0, changes, len(values)]).astype(int).tolist()
    if len(values) and values[0]:
        lengths.insert(0, 0)
    return {'size': [mask.shape[1], mask.shape[0]], 'counts': lengths}


def local_runtime_paths(root: Path, *, registry_path=None):
    from current_model_registry import load_registry
    config = load_registry(root, registry_path=registry_path)
    return {'python': Path(config['python']), 'checkpoint': Path(config['semantics_checkpoint']),
        'semantics_repo': Path(config['semantics_repo_dir']),
        'registry_path': config['registry_path'],
        'semantics_profile': config['semantics_profile'],
        'matting_profile': config['matting_profile'],
        'vitmatte_model_root': Path(config['vitmatte_model_root']),
        'model_registry': config,
        'semantics_worker': code_path('local_semantics_worker.py')}


def local_models_ready(root: Path, *, role='source', registry_path=None):
    if role not in ('source', 'candidate'):
        raise ValueError('Analysis role must be source or candidate')
    paths = local_runtime_paths(root, registry_path=registry_path)
    registry = paths['registry_path']
    required = ('python', 'checkpoint', 'semantics_worker', 'semantics_repo')
    return registry.is_file() and _json(registry).get('ready', False) and all(paths[key].exists() for key in required)


@lru_cache(maxsize=64)
def _content_digest(path, size, modified):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def file_identity(path):
    path = Path(path)
    stat = path.stat()
    return _content_digest(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def analysis_identity(root, *, role='source', registry_path=None):
    """Identify actual decode, weights, worker code and inference parameters."""
    if role not in ('source', 'candidate'):
        raise ValueError('Analysis role must be source or candidate')
    from raw_support import RGB_DECODE_VERSION
    paths = local_runtime_paths(root, registry_path=registry_path)
    identity = {'decode_version': RGB_DECODE_VERSION, 'analysis_role': role,
        'semantics_checkpoint': file_identity(paths['checkpoint']),
        'workers': {'semantics_worker': file_identity(paths['semantics_worker'])},
        'producer': file_identity(Path(__file__)),
        'input_proxy': 'aspect_preserving_2048_lanczos_srgb',
        'recipe': role+'-semantics-only-v2'}
    from sapiens_model_profiles import profile
    specification = profile(paths['semantics_profile'], 'segmentation')
    if identity['semantics_checkpoint'] != specification['checkpoint_sha256']:
        raise ValueError('Installed Sapiens2 checkpoint differs from the current pinned profile')
    identity.update(model_profile=paths['semantics_profile'], model_specification=specification,
        model_config_sha256=file_identity(paths['semantics_repo']/specification['config']),
        model_profiles_sha256=file_identity(code_path('sapiens_model_profiles.py')),
        recipe=role+'-semantics-current-profile-v4')
    return identity






def _worker(argv, log_path):
    with Path(log_path).open('w', encoding='utf-8') as log:
        completed = subprocess.run([str(item) for item in argv], stdout=log, stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
            timeout=900, env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
    if completed.returncode:
        raise RuntimeError(f'Local model worker failed; see {log_path}')


def prepare_local_appearance(root: Path, source: Path, *, source_hash=None, role='source', registry_path=None):
    """Prepare source or reference semantic regions for matching and manual colour edits."""
    root, source = Path(root), Path(source).resolve()
    paths = local_runtime_paths(root, registry_path=registry_path)
    if not local_models_ready(root, role=role, registry_path=registry_path):
        raise RuntimeError('Installed local analysis runtime is required for role: '+role)
    started = time.perf_counter()
    source_hash = source_hash or hashlib.sha256(source.read_bytes()).hexdigest()
    model_identity = analysis_identity(root, role=role, registry_path=registry_path)
    identity = hashlib.sha256(json.dumps(model_identity, sort_keys=True).encode()).hexdigest()[:12]
    directory = root / 'outputs/cache/local-analysis' / source_hash[:16] / identity
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / 'analysis.json'
    if result_path.is_file():
        return {**_json(result_path), 'cache_hit': True, 'elapsed_seconds': round(time.perf_counter()-started, 3)}
    from raw_support import open_rgb
    image = open_rgb(source)
    original_size = list(image.size)
    image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
    input_path = directory / 'source-srgb.png'
    image.save(input_path, 'PNG', icc_profile=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes())
    semantic_dir = directory / 'semantics'
    semantic_dir.mkdir(exist_ok=True)
    if not (semantic_dir / 'result.json').exists():
        _worker([paths['python'], '-X', 'utf8', paths['semantics_worker'], '--input', input_path,
            '--output-dir', semantic_dir, '--checkpoint', paths['checkpoint'],
            '--repo-dir', paths['semantics_repo'], '--model-profile', paths['semantics_profile']], directory / 'semantics-worker.log')
    semantic = _json(semantic_dir / 'result.json')
    with np.load(semantic['labels_path']) as resource:
        labels = resource['labels']
    source_rgb = np.asarray(image.resize((labels.shape[1], labels.shape[0]), Image.Resampling.BILINEAR), dtype=np.float32) / 255
    groups = {'face_neck': ('脸与颈部', [3]), 'left_arm': ('左臂与手', [6, 7, 11]),
        'right_arm': ('右臂与手', [15, 16, 20]), 'exposed_legs': ('裸露腿足', [5, 8, 12, 14, 17, 21]),
        'exposed_torso': ('裸露躯干', [22])}
    regions = []
    skin = np.zeros(labels.shape, dtype=bool)
    for key, (label, class_ids) in groups.items():
        selected = np.isin(labels, class_ids)
        if not np.any(selected):
            continue
        skin |= selected
        source_median = np.median(source_rgb[selected], axis=0).tolist()
        regions.append({'id': key, 'label': label, 'enabled': True,
            'selection': {'base': 'mask', 'mask': encode_mask(selected), 'feather': .002, 'strokes': []},
            'correction': {'target_rgb': source_median, 'strength': 0, 'exposure_ev': 0},
            'target_origin': 'source_region_median_inactive_until_user_edit',
            'color_guard': {'enabled': False, 'reference_rgb': source_median, 'tolerance': .25}})
    color_plan = {'schema_version': 'source_color_plan.v1', 'enabled': True,
        'summary': 'Sapiens2语义选区；目标色调节默认关闭，用户可独立修改。',
        'regions': regions}
    result = {'ok': True, 'analysis_role': role, 'analysis_identity': model_identity,
        'source': {'path': str(source), 'sha256': source_hash, 'oriented_size': original_size},
        'source_color_plan': color_plan,
        'analysis_directory': str(directory.resolve()), 'cache_hit': False,
        'elapsed_seconds': round(time.perf_counter()-started, 3), 'skin_found': bool(np.any(skin)), 'semantic_result': semantic,
        'gpt_used': False, 'image_generation_used': False}
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return result
