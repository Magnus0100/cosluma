"""Local model orchestration. GPU dependencies stay in an external worker runtime."""
from __future__ import annotations

from runtime_resources import external_python_run

import hashlib
import ast
import json
import os
from pathlib import Path
import subprocess
import time
from functools import lru_cache
from runtime_resources import code_path, isolated_python_environment
from asset_identity import sha256_file
from cache_roots import cache_dir

import numpy as np
from PIL import Image, ImageCms


SEMANTIC_INPUT_RECIPE = {'input_mode': 'RGB', 'color_space': 'sRGB',
    'resize': 'aspect_preserving_thumbnail', 'max_edge': 2048,
    'resampling': 'LANCZOS', 'format': 'PNG', 'icc_profile': 'sRGB'}


@lru_cache(maxsize=16)
def _module_math_functions(path, modification_ns, size):
    """Parse small source once per stat-bound file revision in this process."""
    tree = ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    functions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for name, node in functions.items():
        if (node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str)):
            node.body = node.body[1:]
        functions[name] = ast.dump(node, include_attributes=False)
    return functions


def _function_math_identity(path, names):
    """Fingerprint selected computations without coordinator/comments/line numbers."""
    path = Path(path).resolve()
    stat = path.stat()
    functions = _module_math_functions(str(path), stat.st_mtime_ns, stat.st_size)
    payload = []
    for name in names:
        if name not in functions:
            raise ValueError('Semantic identity function is unavailable: '+name)
        payload.append(functions[name])
    return hashlib.sha256('\n'.join(payload).encode('utf-8')).hexdigest()


def _semantic_input_proxy(image, input_path):
    """Write the exact semantic worker input independently of its scheduling."""
    recipe = SEMANTIC_INPUT_RECIPE
    image.thumbnail((recipe['max_edge'], recipe['max_edge']), Image.Resampling[recipe['resampling']])
    image.save(input_path, recipe['format'],
        icc_profile=ImageCms.ImageCmsProfile(ImageCms.createProfile(recipe['icc_profile'])).tobytes())


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


def file_identity(path):
    return sha256_file(path)


def analysis_identity(root, *, role='source', registry_path=None):
    """Identify actual decode, weights, worker code and inference parameters."""
    if role not in ('source', 'candidate'):
        raise ValueError('Analysis role must be source or candidate')
    from raw_support import RGB_DECODE_VERSION
    paths = local_runtime_paths(root, registry_path=registry_path)
    identity = {'decode_version': RGB_DECODE_VERSION, 'analysis_role': role,
        'semantics_checkpoint': file_identity(paths['checkpoint']),
        'workers': {'semantics_worker': file_identity(paths['semantics_worker'])},
        'input_proxy': ('aspect_preserving_'+str(SEMANTIC_INPUT_RECIPE['max_edge'])
            +'_'+SEMANTIC_INPUT_RECIPE['resampling'].lower()+'_'+SEMANTIC_INPUT_RECIPE['color_space'].lower()),
        'input_recipe': dict(SEMANTIC_INPUT_RECIPE),
        'producer_math': {
            'input': _function_math_identity(__file__, ('_semantic_input_proxy',)),
            'derived_analysis': _function_math_identity(__file__, ('encode_mask', '_semantic_colour_plan'))}}
    from sapiens_model_profiles import profile
    specification = profile(paths['semantics_profile'], 'segmentation')
    if identity['semantics_checkpoint'] != specification['checkpoint_sha256']:
        raise ValueError('Installed Sapiens2 checkpoint differs from the current pinned profile')
    identity.update(model_profile=paths['semantics_profile'], model_specification=specification,
        model_config_sha256=file_identity(paths['semantics_repo']/specification['config']),
        model_profile_math=_function_math_identity(code_path('sapiens_model_profiles.py'),
            ('profile', 'verified_checkpoint')),
        recipe=role+'-semantics-current-profile-v5-math-identity')
    return identity






def _worker(argv, log_path):
    with Path(log_path).open('w', encoding='utf-8') as log:
        completed = external_python_run([str(item) for item in argv], stdout=log, stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
            timeout=900, env=isolated_python_environment())
    if completed.returncode:
        raise RuntimeError(f'Local model worker failed; see {log_path}')


def _analysis_context(root: Path, source: Path, *, source_hash, role, paths, model_identity,
                      decoded_images=None):
    """Prepare one image's cache/input paths without starting the GPU worker."""
    source = Path(source).resolve()
    source_hash = source_hash or sha256_file(source)
    identity = hashlib.sha256(json.dumps(model_identity, sort_keys=True).encode()).hexdigest()[:12]
    cache_base = cache_dir(root, 'local-analysis')
    directory = cache_base / source_hash[:16] / identity
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / 'analysis.json'
    started = time.perf_counter()
    if result_path.is_file():
        return {'source': source, 'source_hash': source_hash, 'role': role,
                'paths': paths, 'model_identity': model_identity, 'directory': directory,
                'cache_base': cache_base, 'result_path': result_path,
                'cached_result': _json(result_path), 'started': started}
    from raw_support import open_rgb
    image = open_rgb(source)
    original_size = list(image.size)
    if decoded_images is not None:
        # The scene coordinator consumes this same full RGB frame once. Keep
        # its 1536 proxy independent of the semantic 2048 resize below.
        decoded_images[role] = {'path': str(source), 'sha256': source_hash,
                                'image': image}
        image = image.copy()
    input_path = directory / 'source-srgb.png'
    _semantic_input_proxy(image, input_path)
    semantic_dir = directory / 'semantics'
    semantic_dir.mkdir(exist_ok=True)
    return {'source': source, 'source_hash': source_hash, 'role': role,
            'paths': paths, 'model_identity': model_identity, 'directory': directory,
            'cache_base': cache_base, 'result_path': result_path, 'image': image,
            'original_size': original_size, 'input_path': input_path,
            'semantic_dir': semantic_dir, 'started': started}


def _semantic_colour_plan(image, labels):
    """Derive selections and inactive median targets from the worker labels."""
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
    return color_plan, bool(np.any(skin))


def _finish_analysis(context):
    """Build and persist the existing per-image semantic analysis contract."""
    started = context['started']
    if context.get('cached_result') is not None:
        return {**context['cached_result'], 'cache_hit': True,
                'cache_root': str(context['cache_base'].resolve()),
                'elapsed_seconds': round(time.perf_counter()-started, 3)}
    source = context['source']; source_hash = context['source_hash']
    directory = context['directory']; result_path = context['result_path']
    image = context['image']; model_identity = context['model_identity']
    semantic = _json(context['semantic_dir'] / 'result.json')
    with np.load(semantic['labels_path']) as resource:
        labels = resource['labels']
    color_plan, skin_found = _semantic_colour_plan(image, labels)
    result = {'ok': True, 'analysis_role': context['role'], 'analysis_identity': model_identity,
        'source': {'path': str(source), 'sha256': source_hash, 'oriented_size': context['original_size']},
        'source_color_plan': color_plan,
        'analysis_directory': str(directory.resolve()), 'cache_root': str(context['cache_base'].resolve()), 'cache_hit': False,
        'elapsed_seconds': round(time.perf_counter()-started, 3), 'skin_found': skin_found, 'semantic_result': semantic,
        'gpt_used': False, 'image_generation_used': False}
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return result


def prepare_local_appearance(root: Path, source: Path, *, source_hash=None, role='source', registry_path=None):
    """Prepare one source/reference semantic analysis using the current worker contract."""
    root = Path(root)
    paths = local_runtime_paths(root, registry_path=registry_path)
    if not local_models_ready(root, role=role, registry_path=registry_path):
        raise RuntimeError('Installed local analysis runtime is required for role: '+role)
    context = _analysis_context(root, source, source_hash=source_hash, role=role, paths=paths,
                                model_identity=analysis_identity(root, role=role, registry_path=registry_path))
    if context.get('cached_result') is None and not (context['semantic_dir'] / 'result.json').exists():
        _worker([paths['python'], '-X', 'utf8', paths['semantics_worker'], '--input', context['input_path'],
            '--output-dir', context['semantic_dir'], '--checkpoint', paths['checkpoint'],
            '--repo-dir', paths['semantics_repo'], '--model-profile', paths['semantics_profile']],
            context['directory'] / 'semantics-worker.log')
    return _finish_analysis(context)


def prepare_local_appearance_pair(root: Path, source: Path, candidate: Path, *, source_hash=None,
                                  candidate_hash=None, registry_path=None, decoded_images=None):
    """Prepare source and candidate semantics in one model process.

    Each image keeps its own content-addressed cache and result contract.  A
    single Sapiens2 worker loads the pinned checkpoint once, then evaluates
    uncached images in order.  This removes duplicate model import/CUDA setup
    from the normal scene-preparation path without changing labels or assets.
    """
    root = Path(root)
    paths = local_runtime_paths(root, registry_path=registry_path)
    if not local_models_ready(root, role='source', registry_path=registry_path):
        raise RuntimeError('Installed local analysis runtime is required for role: source')
    source_identity = analysis_identity(root, role='source', registry_path=registry_path)
    candidate_identity = analysis_identity(root, role='candidate', registry_path=registry_path)
    contexts = [
        _analysis_context(root, source, source_hash=source_hash, role='source', paths=paths,
                          model_identity=source_identity, decoded_images=decoded_images),
        _analysis_context(root, candidate, source_hash=candidate_hash, role='candidate', paths=paths,
                          model_identity=candidate_identity, decoded_images=decoded_images),
    ]
    pending = [context for context in contexts
               if context.get('cached_result') is None and not (context['semantic_dir'] / 'result.json').exists()]
    if pending:
        argv = [paths['python'], '-X', 'utf8', paths['semantics_worker']]
        for context in pending:
            argv.extend(['--input', str(context['input_path']), '--output-dir', str(context['semantic_dir'])])
        argv.extend(['--checkpoint', paths['checkpoint'], '--repo-dir', paths['semantics_repo'],
                     '--model-profile', paths['semantics_profile']])
        log = contexts[0]['directory'].parent / 'pair-semantics-worker.log'
        _worker(argv, log)
    return tuple(_finish_analysis(context) for context in contexts)
