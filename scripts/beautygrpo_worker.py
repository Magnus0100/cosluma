"""Offline BeautyGRPO inference for complete generated face appearance.

One pipeline handles every detected face in a request. The original generated
RGB is retained; composition weights describe seams and visible face ownership,
never skin-only residuals, colour protection or source texture restoration.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from functools import wraps
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

import numpy as np

from asset_identity import sha256_file
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import binary_propagation, distance_transform_edt

from portrait_regions_worker import FACE_OVAL

CONFIG_SCHEMA = 'beautygrpo_models.v1'
VERIFICATION_SCHEMA = 'beautygrpo_verification.v1'
SCHEMA = 'beautygrpo_model_output.v1'
ROI_POLICY = 'complete_face_oval_support_union_square_1p5.v1'
COMPOSITION_POLICY = 'visible_complete_face_upward_forehead_inward_feather.v2'
EXPECTED_LORA_SHA = '23105cbd94fcd9f7b14d224dd28ce84cb5fa0f9bd5ba077cc0de27428e46c8a1'
BASE_ID = 'black-forest-labs/FLUX.1-Kontext-dev'
BASE_REVISION = '24e9dedc4ef646698dc8eb4e18ae2cec3c9fea0d'
BASE_WEIGHT_HASHES = {
    'text_encoder/model.safetensors': '893d67a23f4693ed42cdab4cbad7fe3e727cf59609c40da28a46b5470f9ed082',
    'text_encoder_2/model-00001-of-00002.safetensors': 'ec87bffd1923e8b2774a6d240c922a41f6143081d52cf83b8fe39e9d838c893e',
    'text_encoder_2/model-00002-of-00002.safetensors': 'a5640855b301fcdbceddfa90ae8066cd9414aff020552a201a255ecf2059da00',
    'transformer/diffusion_pytorch_model-00001-of-00003.safetensors': '5414f9ba3d3945512769b1b5ecd41122c8bcfebb7ec906a3ac60daf371d38946',
    'transformer/diffusion_pytorch_model-00002-of-00003.safetensors': 'a6fa67a12833c30040f794365a187cac8eef2e467152b84319034ae2828d9d03',
    'transformer/diffusion_pytorch_model-00003-of-00003.safetensors': '464f5343b6b06a2337d8c34b72ebec38219ebbfb7aafecbc9b342dc031417e7a',
    'vae/diffusion_pytorch_model.safetensors': 'f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3',
}
RECIPE = {
    'prompt': "Beautify this person's face while maintaining a natural and realistic appearance.",
    'size': 1024, 'steps': 28, 'guidance': 2.5, 'seed': 42, 'lora_scale': 1.0,
}
# Include eyes, brows and lips. Hair, clothing, jewellery and background do not
# become facial ownership merely because they are inside a landmark rectangle.
FACE_MATERIAL_LABELS = (2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15)
_PIPELINE_CACHE = {}
_WINDOWS_MEMORY_API = None
OFFLOAD_ENV = 'COSPLAY_BEAUTYGRPO_OFFLOAD'
OFFLOAD_POLICIES = ('nunchaku_cuda',)
DEFAULT_OFFLOAD = 'nunchaku_cuda'
PROMPT_CACHE_SCHEMA = 'beautygrpo_prompt_embeddings.v1'
PROMPT_TENSORS = {'prompt_embeds': [1, 512, 4096], 'pooled_prompt_embeds': [1, 768]}
PROMPT_ENCODING = {'prompt': RECIPE['prompt'], 'prompt_2': RECIPE['prompt'],
                   'max_sequence_length': 512, 'num_images_per_prompt': 1,
                   'dtype': 'bfloat16', 'encoder_lora': None,
                   'encoder_offload': 'accelerate_cpu_offload',
                   'preload_module_classes': {'text_encoder': ['CLIPEncoderLayer'],
                                               'text_encoder_2': ['T5Block']}}


def selected_offload(config=None):
    """Select execution independently of the immutable installation audit."""
    config = config or {}
    if 'execution_offload' in config and config['execution_offload'] not in OFFLOAD_POLICIES:
        raise ValueError('BeautyGRPO execution_offload must be one of: '+', '.join(OFFLOAD_POLICIES))
    if '_execution_offload' in config and config['_execution_offload'] not in OFFLOAD_POLICIES:
        raise ValueError('BeautyGRPO selected execution policy must be one of: '+', '.join(OFFLOAD_POLICIES))
    # The old offload field belongs to the installation audit. It never
    # selects current execution. The environment has the highest priority.
    policy = os.environ.get(OFFLOAD_ENV) or config.get('_execution_offload') or config.get('execution_offload', DEFAULT_OFFLOAD)
    if policy not in OFFLOAD_POLICIES:
        raise ValueError(OFFLOAD_ENV+' must be one of: '+', '.join(OFFLOAD_POLICIES))
    return policy


def execution_profile(config):
    """One pinned GPU backend participates in every service and pixel cache."""
    from runtime_resources import code_path
    from beautygrpo_nunchaku import MODEL_REPO, MODEL_REVISION, MODEL_SHA, PACKAGE_VERSION
    return {'schema': 'beautygrpo_execution.v4', 'offload': selected_offload(config),
            'backend': 'nunchaku_svdquant_int4_cuda',
            'onload_device': f'cuda:{config.get("gpu_id", 0)}', 'offload_device': None,
            'dtype': 'bfloat16', 'torch_cpu_threads': config.get('torch_cpu_threads', 8),
            'disk_offload': False, 'vae_slicing': True, 'vae_tiling': True,
            'transformer': {'quantization': 'SVDQuant_INT4_W4A4_rank32', 'cpu_weight_offload': False,
                'repo': MODEL_REPO, 'revision': MODEL_REVISION, 'sha256': MODEL_SHA,
                'package_version': PACKAGE_VERSION, 'beauty_lora_rank': 32, 'lora_strength': 1.,
                'producer_sha256': sha(code_path('beautygrpo_nunchaku.py'))},
            'prompt_conditioning': {'schema': PROMPT_CACHE_SCHEMA,
                'source': 'installed_exact_embeddings', 'encoding': dict(PROMPT_ENCODING),
                'receipt_sha256': config.get('_prompt_embedding_identity'),
                'omitted_components': ['text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2']},
            'gpu_handoff': 'terminate_owned_idle_service_then_reload_on_demand'}



def _windows_memory_api():
    """Read-only Windows process counters."""
    global _WINDOWS_MEMORY_API
    if os.name != 'nt':
        return None
    if _WINDOWS_MEMORY_API is not None:
        return _WINDOWS_MEMORY_API or None
    import ctypes
    class ProcessMemory(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_uint32), ('page_faults', ctypes.c_uint32)] + [
            (name, ctypes.c_size_t) for name in ('peak_working_set_bytes', 'working_set_bytes',
                'peak_paged_pool_bytes', 'paged_pool_bytes', 'peak_nonpaged_pool_bytes',
                'nonpaged_pool_bytes', 'pagefile_bytes', 'peak_pagefile_bytes', 'private_bytes')]
    class ProcessIO(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in (
            'read_operations', 'write_operations', 'other_operations',
            'io_read_bytes', 'io_write_bytes', 'io_other_bytes')]
    try:
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.GetProcessIoCounters.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessIO)]
        kernel.GetProcessIoCounters.restype = ctypes.c_int
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessMemory), ctypes.c_uint32]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        _WINDOWS_MEMORY_API = {'kernel': kernel, 'psapi': psapi,
            'ProcessMemory': ProcessMemory, 'ProcessIO': ProcessIO}
    except (OSError, AttributeError):
        _WINDOWS_MEMORY_API = False
    return _WINDOWS_MEMORY_API or None


def _runtime_counters():
    result = {'process_cpu_seconds': time.process_time()}
    api = _windows_memory_api()
    if api is not None:
        import ctypes
        handle = api['kernel'].GetCurrentProcess()
        memory = api['ProcessMemory'](); memory.cb = ctypes.sizeof(memory)
        if api['psapi'].GetProcessMemoryInfo(handle, ctypes.byref(memory), memory.cb):
            result.update({key: int(getattr(memory, key)) for key in (
                'page_faults', 'working_set_bytes', 'private_bytes', 'peak_working_set_bytes')})
        counters = api['ProcessIO']()
        if api['kernel'].GetProcessIoCounters(handle, ctypes.byref(counters)):
            result.update({key: int(getattr(counters, key)) for key, _ in counters._fields_})
    # Query already initialized CUDA only. Diagnostics must never initialize it.
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():
        stats = getattr(torch.cuda, 'host_memory_stats', None)
        if stats is not None:
            try:
                result['host_allocator'] = {key: value for key, value in stats().items()
                    if isinstance(value, (int, float)) and math.isfinite(value)}
            except (RuntimeError, AttributeError):
                pass
    return result


def _timing_record(stage, started, before):
    after = _runtime_counters()
    cumulative = ('process_cpu_seconds', 'page_faults', 'read_operations', 'write_operations',
                  'other_operations', 'io_read_bytes', 'io_write_bytes', 'io_other_bytes')
    finished = time.perf_counter()
    finished_unix = time.time()
    return {'stage': stage, 'seconds': finished - started,
            'started_unix_seconds': finished_unix - (finished - started),
            'finished_unix_seconds': finished_unix, 'before': before, 'after': after,
            'counter_delta': {key: after[key] - before[key] for key in cumulative
                              if key in before and key in after}}


@contextmanager
def _measure_inference(pipe):
    """Passive host timings; no CUDA barriers or changes to tensor values."""
    records, restores, handles = [], [], []
    def finish(stage, started, before):
        records.append(_timing_record(stage, started, before))
    def instrument(owner, name, stage):
        original = getattr(owner, name, None)
        if not callable(original):
            return
        local = name in vars(owner)
        @wraps(original)
        def measured(*args, **kwargs):
            before, started = _runtime_counters(), time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                finish(stage, started, before)
        setattr(owner, name, measured)
        restores.append((owner, name, original, local))
    try:
        instrument(pipe, 'encode_prompt', 'prompt_encoding')
        instrument(pipe, 'prepare_latents', 'latent_preparation')
        vae = getattr(pipe, 'vae', None)
        if vae is not None:
            instrument(vae, 'encode', 'vae_encode')
            instrument(vae, 'decode', 'vae_decode')
        transformer = getattr(pipe, 'transformer', None)
        if transformer is not None and hasattr(transformer, 'register_forward_pre_hook'):
            pending = []
            def before_forward(_module, _args):
                pending.append((_runtime_counters(), time.perf_counter()))
            def after_forward(_module, _args, _output):
                if pending:
                    before, started = pending.pop()
                    finish('transformer_forward', started, before)
            # PyTorch's outer call hooks survive Diffusers' first-forward
            # replacement of its internal lazy-prefetch forward wrapper.
            handles.append(transformer.register_forward_pre_hook(before_forward))
            handles.append(transformer.register_forward_hook(after_forward, always_call=True))
        yield {'schema': 'beautygrpo_inference_profile.v1', 'stages': records,
               'timing': 'host_wall_no_extra_cuda_synchronization',
               'counter_limits': 'Page faults include soft faults; process IO may omit mapped page-ins; host allocator timing retains native units.'}
    finally:
        for handle in handles:
            handle.remove()
        for owner, name, original, local in reversed(restores):
            if local:
                setattr(owner, name, original)
            else:
                delattr(owner, name)


def sha(path):
    return sha256_file(path)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def _absolute_path(config, key, *, directory=False):
    value = config.get(key)
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ValueError('BeautyGRPO installation requires an absolute path: '+key)
    path = Path(value).resolve()
    if not (path.is_dir() if directory else path.is_file()):
        raise FileNotFoundError('BeautyGRPO installed resource unavailable: '+key+'='+str(path))
    return path


def _read_installation_config(path):
    """Read and structurally validate a config without hashing model weights."""
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(config, dict) or config.get('schema') != CONFIG_SCHEMA or config.get('ready') is not True:
        raise ValueError('A ready beautygrpo_models.v1 local installation is required')
    for key in ('python', 'lora_path', 'base_manifest', 'regions_python', 'face_registry', 'base_model_root', 'regions_model_root', 'vae_model_root'):
        value = config.get(key)
        if isinstance(value, str) and value:
            package_root = os.environ.get('COSPLAY_PACKAGE_ROOT')
            if package_root is None and getattr(sys, 'frozen', False):
                package_root = str(Path(sys.executable).resolve().parent)
            if package_root:
                value = value.replace('${PACKAGE_ROOT}', str(Path(package_root).resolve()))
            target = Path(value).expanduser()
            config[key] = str((target if target.is_absolute() else path.parent/target).resolve())
    for key in ('python', 'lora_path', 'base_manifest', 'regions_python', 'face_registry'):
        _absolute_path(config, key)
    for key in ('base_model_root', 'regions_model_root'):
        _absolute_path(config, key, directory=True)
    if config.get('vae_model_root'):
        _absolute_path(config, 'vae_model_root', directory=True)
    if config.get('offload') != 'sequential_cpu_offload':
        raise ValueError('BeautyGRPO installation audit requires its original sequential_cpu_offload declaration')
    selected_offload(config)
    if config.get('recipe', RECIPE) != RECIPE:
        raise ValueError('This BeautyGRPO installation requires the fixed evaluated recipe')
    if type(config.get('gpu_id', 0)) is not int or config.get('gpu_id', 0) < 0:
        raise ValueError('BeautyGRPO GPU index must be a nonnegative integer')
    if type(config.get('torch_cpu_threads', 8)) is not int or not 1 <= config.get('torch_cpu_threads', 8) <= 32:
        raise ValueError('BeautyGRPO CPU thread count must lie in [1,32]')
    manifest = Path(config['base_manifest'])
    digest = sha(manifest)
    if config.get('base_manifest_sha256') != digest:
        raise ValueError('Pinned BeautyGRPO base file manifest changed')
    catalog = json.loads(manifest.read_text(encoding='utf8'))
    if (catalog.get('status') != 'ready_for_local_inference' or catalog.get('hf_repo') != BASE_ID
            or catalog.get('hf_revision') != BASE_REVISION):
        raise ValueError('Verified official local FLUX.1-Kontext-dev manifest required')
    rows = catalog.get('completed_files')
    if not isinstance(rows, list) or len(rows) != 25 or len({row.get('path') for row in rows}) != 25:
        raise ValueError('The complete 25-file local BeautyGRPO base inventory is required')
    base = Path(config['base_model_root']).resolve()
    names = set()
    for row in rows:
        name = row.get('path', '')
        digest = row.get('local_sha256')
        if (not name or Path(name).is_absolute()
                or not isinstance(digest, str) or len(digest) != 64 or row.get('verified') is not True
                or type(row.get('bytes')) is not int or row['bytes'] < 0):
            raise ValueError('Incomplete or invalid installed base file: '+str(name))
        if name in BASE_WEIGHT_HASHES and digest != BASE_WEIGHT_HASHES[name]:
            raise ValueError('Official FLUX base weight identity differs: '+name)
        if row.get('expected_sha256') and row['expected_sha256'] != digest:
            raise ValueError('Official base hash differs from its download evidence: '+name)
        names.add(name)
    if not set(BASE_WEIGHT_HASHES).issubset(names) or 'model_index.json' not in names:
        raise ValueError('Complete official FLUX base weights and configuration required')
    return {**config, 'recipe': dict(RECIPE), 'config_path': str(path), 'config_sha256': sha(path),
            '_manifest_catalog': catalog}


def _verification_path(config, config_path):
    value = config.get('verification_receipt')
    if value:
        target = Path(value).expanduser()
        return (target if target.is_absolute() else Path(config_path).parent/target).resolve()
    return Path(config_path).with_suffix('.verification.json').resolve()


def _canonical_sha(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                         allow_nan=False).encode('utf8')
    return hashlib.sha256(encoded).hexdigest()


def _runtime_identity(config):
    return {'worker_sha256': sha(Path(__file__)), 'base_model_id': BASE_ID,
            'base_revision': BASE_REVISION, 'recipe': dict(RECIPE),
            'python_path': str(Path(config['python']).resolve()),
            'offload': config['offload'], 'gpu_id': config.get('gpu_id', 0),
            'torch_cpu_threads': config.get('torch_cpu_threads', 8)}


def _vae_component(config, *, verify_hash=False):
    """Bind a separately stored VAE to the same verified official weights."""
    root = Path(config.get('vae_model_root') or Path(config['base_model_root'])/'vae').resolve()
    files = []
    for row in config['_manifest_catalog']['completed_files']:
        if not row['path'].startswith('vae/'):
            continue
        name = row['path'][4:]
        target = (root/name).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise ValueError('BeautyGRPO VAE component file is unavailable: '+name)
        stat = target.stat()
        if stat.st_size != row['bytes'] or (verify_hash and sha(target) != row['local_sha256']):
            raise ValueError('BeautyGRPO VAE component differs from the official manifest: '+name)
        files.append({'path':name, 'bytes':stat.st_size, 'sha256':row['local_sha256'], 'mtime_ns':stat.st_mtime_ns})
    if not files or not any(row['path'].endswith('.safetensors') for row in files):
        raise ValueError('BeautyGRPO VAE component has no verified weight inventory')
    return {'root':str(root), 'files':files}


def _validate_verification_receipt(config, receipt, *, receipt_path):
    """Validate the installation receipt without reading model weight bytes."""
    if not isinstance(receipt, dict) or receipt.get('schema') != VERIFICATION_SCHEMA or receipt.get('ready') is not True:
        raise ValueError('A ready beautygrpo_verification.v1 receipt is required')
    config_path = Path(config['config_path']).resolve()
    if receipt.get('config_sha256') != config['config_sha256']:
        raise ValueError('BeautyGRPO installation config differs from its verification receipt')
    if config.get('vae_model_root') and receipt.get('vae_component') != _vae_component(config):
        raise ValueError('BeautyGRPO VAE component differs from its verification receipt')
    manifest = Path(config['base_manifest']).resolve()
    manifest_record = receipt.get('manifest') or {}
    if (manifest_record.get('path') != str(manifest)
            or manifest_record.get('sha256') != config['base_manifest_sha256']
            or manifest_record.get('repository') != BASE_ID
            or manifest_record.get('revision') != BASE_REVISION):
        raise ValueError('BeautyGRPO base manifest identity differs from its verification receipt')
    catalog = config['_manifest_catalog']
    expected_files = [{'path': row['path'], 'bytes': row['bytes'], 'sha256': row['local_sha256'],
                       'verified': row.get('verified') is True}
                      for row in catalog['completed_files']]
    receipt_files = manifest_record.get('files') or []
    if len(receipt_files) != len(expected_files):
        raise ValueError('BeautyGRPO verification receipt has an incomplete base inventory')
    # Runtime validation deliberately performs cheap filesystem identity
    # checks only.  The receipt's full content hashes are produced by the
    # installation command; re-reading 31 GB on every portrait would defeat
    # the purpose of an installed, immutable model.  A size/mtime change
    # invalidates the receipt and requires explicit re-verification.
    base_root = Path(config['base_model_root']).resolve()
    for expected, recorded in zip(expected_files, receipt_files):
        if (recorded.get('path') != expected['path'] or recorded.get('bytes') != expected['bytes']
                or recorded.get('sha256') != expected['sha256'] or recorded.get('verified') is not True):
            raise ValueError('BeautyGRPO base file identity differs from its verification receipt: '+expected['path'])
        target = (base_root / expected['path']).resolve()
        if (not target.is_file() or not target.is_relative_to(base_root)
                or target.stat().st_size != expected['bytes']
                or recorded.get('mtime_ns') != target.stat().st_mtime_ns):
            raise ValueError('BeautyGRPO base file changed since installation verification: '+expected['path'])
    if [{key: row[key] for key in ('path', 'bytes', 'sha256', 'verified')} for row in receipt_files] != expected_files:
        raise ValueError('BeautyGRPO base file inventory differs from its verification receipt')
    base_record = receipt.get('base') or {}
    if (base_record.get('root') != str(Path(config['base_model_root']).resolve())
            or base_record.get('file_count') != 25
            or [{key: row[key] for key in ('path', 'bytes', 'sha256', 'verified')}
                for row in base_record.get('files', [])] != expected_files):
        raise ValueError('BeautyGRPO base installation identity differs from its verification receipt')
    lora = Path(config['lora_path']).resolve()
    lora_record = receipt.get('lora') or {}
    if (lora_record.get('path') != str(lora) or lora_record.get('sha256') != EXPECTED_LORA_SHA
            or lora_record.get('tensor_count') != 684 or lora_record.get('matrix_pairs') != 342
            or lora_record.get('rank') != 32 or lora_record.get('bytes') != lora.stat().st_size
            or lora_record.get('mtime_ns') != lora.stat().st_mtime_ns):
        raise ValueError('BeautyGRPO LoRA identity differs from its verification receipt')
    tensor_keys = lora_record.get('tensor_keys')
    if (not isinstance(tensor_keys, list) or len(tensor_keys) != 684
            or any(not isinstance(key, str) or not key.startswith('transformer.')
                   or not key.endswith(('.lora_A.weight', '.lora_B.weight')) for key in tensor_keys)):
        raise ValueError('BeautyGRPO LoRA tensor key inventory is incomplete')
    runtime = receipt.get('runtime') or {}
    if not isinstance(runtime, dict):
        raise ValueError('BeautyGRPO verification runtime record must be an object')
    expected_runtime = _runtime_identity(config)
    installed_runtime = {key: value for key, value in runtime.items() if key != 'identity_sha256'}
    # The installation audit proves immutable weights and their runtime recipe.
    # Editing orchestration code does not invalidate those content hashes. The
    # current worker identity is still part of preparation/cache provenance.
    if (runtime.get('identity_sha256') != _canonical_sha(installed_runtime)
            or {key: value for key, value in installed_runtime.items() if key != 'worker_sha256'}
            != {key: value for key, value in expected_runtime.items() if key != 'worker_sha256'}):
        raise ValueError('BeautyGRPO runtime identity differs from its verification receipt')
    if receipt.get('model_identity') != model_identity(config):
        raise ValueError('BeautyGRPO model identity differs from its verification receipt')
    if receipt.get('receipt_sha256') and receipt['receipt_sha256'] != sha(receipt_path):
        raise ValueError('BeautyGRPO verification receipt changed')
    return receipt


def verify_installation(path, *, receipt_path=None, progress=None):
    """Perform the full installation-time verification and return a receipt.

    This is the only routine that hashes all 25 base files and the LoRA.  The
    runtime loader consumes the resulting receipt and deliberately does not
    repeat those large reads.
    """
    config = _read_installation_config(path)
    from beautygrpo_nunchaku import installed_model
    installed_model(config, installation=True)
    catalog = config['_manifest_catalog']
    receipt = config.get('verification_receipt') or {}
    # This is the one explicit, installation-time full scan.  Runtime loading
    # consumes the resulting receipt and performs only stat/identity checks.
    verified_files = []
    for index, row in enumerate(catalog['completed_files']):
        name = row['path']
        target = (Path(config['base_model_root']) / name).resolve()
        if (not target.is_file() or not target.is_relative_to(Path(config['base_model_root']).resolve())
                or target.stat().st_size != row['bytes']):
            raise ValueError('Installed BeautyGRPO base file is missing or has an unexpected size: '+name)
        if progress:
            progress('beauty_model_verification', '正在安装期完整核验本地人像模型文件。',
                     .16 + .08*index/len(catalog['completed_files']))
        digest = sha(target)
        if digest != row['local_sha256'] or (row.get('expected_sha256') and row['expected_sha256'] != digest):
            raise ValueError('Local BeautyGRPO base bytes differ from the pinned manifest: '+name)
        verified_files.append({'path': name, 'bytes': row['bytes'], 'sha256': digest,
                               'verified': True, 'mtime_ns': target.stat().st_mtime_ns})
    base_evidence = {'base_manifest_sha256': config['base_manifest_sha256'],
                     'all_files_hash_verified': True, 'verified_file_count': len(verified_files),
                     'verification_schema': VERIFICATION_SCHEMA,
                     'verification_stage': 'installation'}
    keys = verify_lora_file(config['lora_path'])
    pairs = {}
    for key in keys:
        prefix, kind = key.rsplit('.lora_', 1)
        pairs.setdefault(prefix, {})[kind[0]] = True
    if len(pairs) != 342 or any(set(pair) != {'A', 'B'} for pair in pairs.values()):
        raise ValueError('BeautyGRPO LoRA matrix pairs are incomplete')
    config_path = Path(config['config_path']).resolve()
    expected_files = verified_files
    runtime = _runtime_identity(config)
    receipt = {'schema': VERIFICATION_SCHEMA, 'ready': True,
        'config_sha256': config['config_sha256'],
        'config_identity': {'schema': CONFIG_SCHEMA, 'ready': True,
                            'base_manifest_sha256': config['base_manifest_sha256'],
                            'model_identity': model_identity(config)},
        'manifest': {'path': str(Path(config['base_manifest']).resolve()),
                     'sha256': config['base_manifest_sha256'], 'repository': BASE_ID,
                     'revision': BASE_REVISION, 'files': expected_files},
        'base': {'root': str(Path(config['base_model_root']).resolve()), 'file_count': len(expected_files),
                 'files': expected_files, **base_evidence},
        'lora': {'path': str(Path(config['lora_path']).resolve()), 'sha256': EXPECTED_LORA_SHA,
                 'bytes': Path(config['lora_path']).stat().st_size,
                 'mtime_ns': Path(config['lora_path']).stat().st_mtime_ns, 'tensor_count': len(keys),
                 'matrix_pairs': len(pairs), 'rank': 32, 'dtype': 'bfloat16',
                 'tensor_keys': keys, 'all_matrix_values_verified': True},
        'runtime': {**runtime, 'identity_sha256': _canonical_sha(runtime)},
        'model_identity': model_identity(config), 'full_content_hashes': True,
        'vae_component': _vae_component(config, verify_hash=True),
        'gpu_inference': 0, 'network_requests': 0}
    if receipt_path is not None:
        target = Path(receipt_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        write_json(target, receipt)
    return receipt


def _load_weight_installation(path):
    """Validate config and its installation-time receipt without model hashing."""
    config = _read_installation_config(path)
    receipt_path = _verification_path(config, config['config_path'])
    if not receipt_path.is_file():
        raise ValueError('BeautyGRPO installation verification receipt is missing: '+str(receipt_path))
    receipt = json.loads(receipt_path.read_text(encoding='utf8'))
    _validate_verification_receipt(config, receipt, receipt_path=receipt_path)
    config['verification_receipt_path'] = str(receipt_path)
    config['verification_receipt_sha256'] = sha(receipt_path)
    config['verification_receipt'] = receipt
    config.pop('_manifest_catalog', None)
    # Pin the requested policy for this preparation. It is deliberately not
    # written into the installed config/receipt or presented as installation
    # verification of a new execution strategy.
    config['_execution_offload'] = selected_offload(config)
    return config


def _prompt_cache_path(config):
    return Path(config['config_path']).with_suffix('.prompt-embeddings.json')


def _prompt_cache_error(config, reason):
    return ValueError('BeautyGRPO installed prompt embeddings '+reason+'. Run the explicit '
        'beautygrpo_prepare.py --prepare-prompt-embeddings "'+config['config_path']
        +'" with the configured BeautyGRPO Python; inference never loads missing encoders.')


def _prompt_cache_runtime(config):
    """Small installed-code identities only; no model imports or weight reads."""
    from importlib import metadata
    if Path(sys.executable).resolve() != Path(config['python']).resolve():
        raise _prompt_cache_error(config, 'require the configured Python runtime')
    versions = {name: metadata.version(name) for name in
                ('torch', 'diffusers', 'transformers', 'peft', 'accelerate', 'safetensors')}
    pipeline_source = metadata.distribution('diffusers').locate_file(
        'diffusers/pipelines/flux/pipeline_flux_kontext.py')
    return {'python_path': str(Path(sys.executable).resolve()),
            'python_version': sys.version, 'versions': versions,
            'flux_kontext_pipeline_sha256': sha(pipeline_source)}


def _prompt_cache_binding(config):
    receipt = config['verification_receipt']
    encoder_files = [{key: row[key] for key in ('path', 'bytes', 'sha256', 'mtime_ns')}
        for row in receipt['manifest']['files']
        if row['path'].startswith(('text_encoder/', 'text_encoder_2/', 'tokenizer/', 'tokenizer_2/'))]
    if not encoder_files:
        raise _prompt_cache_error(config, 'lack a verified encoder inventory')
    return {'config_path': str(Path(config['config_path']).resolve()),
            'config_sha256': config['config_sha256'],
            'verification_receipt_sha256': config['verification_receipt_sha256'],
            'base_model_root': str(Path(config['base_model_root']).resolve()),
            'base_manifest_sha256': config['base_manifest_sha256'],
            'base_model_id': BASE_ID, 'base_revision': BASE_REVISION,
            'encoding': dict(PROMPT_ENCODING), 'tensors': dict(PROMPT_TENSORS),
            'encoder_files': encoder_files, 'gpu_id': config.get('gpu_id', 0),
            'torch_cpu_threads': config.get('torch_cpu_threads', 8)}


def _validate_prompt_cache_receipt(config, receipt, *, runtime=False):
    """Validate the small cache and pinned installation evidence, never weights."""
    if (not isinstance(receipt, dict) or receipt.get('schema') != PROMPT_CACHE_SCHEMA
            or receipt.get('ready') is not True or receipt.get('binding') != _prompt_cache_binding(config)):
        raise _prompt_cache_error(config, 'have a missing or stale receipt binding')
    if receipt.get('binding_sha256') != _canonical_sha(receipt['binding']):
        raise _prompt_cache_error(config, 'have a changed binding')
    runtime_record = receipt.get('runtime')
    if (not isinstance(runtime_record, dict)
            or receipt.get('runtime_sha256') != _canonical_sha(runtime_record)
            or (runtime and runtime_record != _prompt_cache_runtime(config))):
        raise _prompt_cache_error(config, 'have a changed encoding runtime')
    artifact = receipt.get('artifact') or {}
    token = artifact.get('sha256')
    expected = _prompt_cache_path(config).with_name(
        _prompt_cache_path(config).stem+'-'+str(token)+'.safetensors')
    if (not isinstance(token, str) or len(token) != 64 or artifact.get('path') != str(expected)
            or not expected.is_file()):
        raise _prompt_cache_error(config, 'have a missing tensor artifact')
    stat = expected.stat()
    if (artifact.get('bytes') != stat.st_size or artifact.get('mtime_ns') != stat.st_mtime_ns
            or artifact.get('ctime_ns') != stat.st_ctime_ns or sha(expected) != token):
        raise _prompt_cache_error(config, 'have a changed tensor artifact')
    # Safetensors headers are JSON; inspect shape/dtype without importing Torch.
    with expected.open('rb') as handle:
        header_size = int.from_bytes(handle.read(8), 'little')
        if not 0 < header_size <= min(stat.st_size - 8, 1024 * 1024):
            raise _prompt_cache_error(config, 'have an invalid tensor header')
        header = json.loads(handle.read(header_size))
    tensors = {name: row for name, row in header.items() if name != '__metadata__'}
    if (set(tensors) != set(PROMPT_TENSORS) or any(
            tensors[name].get('shape') != shape or tensors[name].get('dtype') != 'BF16'
            for name, shape in PROMPT_TENSORS.items())):
        raise _prompt_cache_error(config, 'have unexpected tensor shapes or dtype')
    return expected


def load_installation(path):
    config = _load_weight_installation(path)
    from beautygrpo_nunchaku import installed_model
    quantized_path, quantized_binding = installed_model(config)
    config['_quantized_transformer_binding'] = quantized_binding
    config['_quantized_transformer_path'] = str(quantized_path)
    receipt_path = _prompt_cache_path(config)
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding='utf8'))
        _validate_prompt_cache_receipt(config, receipt)
        config['_prompt_embedding_receipt'] = receipt
        config['_prompt_embedding_identity'] = sha(receipt_path)
    return config


def load_prompt_embeddings(config):
    """Required runtime cache; absence never takes an implicit encoding path."""
    receipt_path = _prompt_cache_path(config)
    if not receipt_path.is_file():
        raise _prompt_cache_error(config, 'are missing')
    receipt = json.loads(receipt_path.read_text(encoding='utf8'))
    artifact_path = _validate_prompt_cache_receipt(config, receipt, runtime=True)
    if config.get('_prompt_embedding_identity') != sha(receipt_path):
        raise _prompt_cache_error(config, 'changed after installation was loaded')
    from safetensors.torch import load_file
    import torch
    tensors = load_file(str(artifact_path), device='cpu')
    if any(tensor.dtype != torch.bfloat16 or list(tensor.shape) != PROMPT_TENSORS[name]
           or not bool(torch.isfinite(tensor).all()) for name, tensor in tensors.items()):
        raise _prompt_cache_error(config, 'contain invalid tensor values')
    return tensors, {'schema': PROMPT_CACHE_SCHEMA, 'receipt_path': str(receipt_path),
        'receipt_sha256': sha(receipt_path), 'artifact_sha256': receipt['artifact']['sha256'],
        'binding_sha256': receipt['binding_sha256'], 'runtime_sha256': receipt['runtime_sha256'],
        'encoding_device': receipt['encoding_device'], 'encoding': dict(PROMPT_ENCODING),
        'encoder_model_loading': False}


@contextmanager
def _measure_component_loading(records):
    """Observe the official Diffusers loader without replacing its behavior."""
    from diffusers.pipelines import pipeline_utils
    original = pipeline_utils.load_sub_model
    @wraps(original)
    def measured(*args, **kwargs):
        before, started = _runtime_counters(), time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            record = _timing_record('component_loading', started, before)
            record['component'] = kwargs.get('name')
            records.append(record)
    pipeline_utils.load_sub_model = measured
    try:
        yield
    finally:
        pipeline_utils.load_sub_model = original


def _prompt_encoding_device(torch, gpu_id):
    return {'device': 'cuda:'+str(gpu_id), 'name': torch.cuda.get_device_name(gpu_id),
            'capability': list(torch.cuda.get_device_capability(gpu_id)),
            'cuda_version': torch.version.cuda,
            'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'float32_matmul_precision': torch.get_float32_matmul_precision()}


def _publish_prompt_embeddings(config, tensors, runtime, encoding_device, stages):
    """Commit an explicit prepare atomically, with the receipt written last."""
    import torch
    from safetensors.torch import save_file
    if (set(tensors) != set(PROMPT_TENSORS) or any(
            tensor.dtype != torch.bfloat16 or list(tensor.shape) != PROMPT_TENSORS[name]
            or not bool(torch.isfinite(tensor).all()) for name, tensor in tensors.items())):
        raise _prompt_cache_error(config, 'prepare produced invalid tensors')
    receipt_path = _prompt_cache_path(config)
    token = uuid.uuid4().hex
    temporary = receipt_path.with_name(receipt_path.name+'.tmp-'+token)
    tensor_temporary = temporary.with_name(temporary.name+'.safetensors')
    try:
        save_file({name: tensor.detach().to(device='cpu').contiguous()
                   for name, tensor in tensors.items()}, str(tensor_temporary))
        digest = sha(tensor_temporary)
        artifact_path = receipt_path.with_name(receipt_path.stem+'-'+digest+'.safetensors')
        if artifact_path.is_file() and sha(artifact_path) == digest:
            tensor_temporary.unlink()
        else:
            os.replace(tensor_temporary, artifact_path)
        stat = artifact_path.stat()
        binding = _prompt_cache_binding(config)
        receipt = {'schema': PROMPT_CACHE_SCHEMA, 'ready': True,
            'binding': binding, 'binding_sha256': _canonical_sha(binding),
            'runtime': runtime, 'runtime_sha256': _canonical_sha(runtime),
            'encoding_device': encoding_device,
            'artifact': {'path': str(artifact_path), 'sha256': digest, 'bytes': stat.st_size,
                         'mtime_ns': stat.st_mtime_ns, 'ctime_ns': stat.st_ctime_ns},
            'prepare_profile': {'schema': 'beautygrpo_prompt_prepare_profile.v1', 'stages': stages},
            'encoder_parameters_omitted_bytes': 9770742272,
            'generated_images': 0, 'network_requests': 0}
        _validate_prompt_cache_receipt(config, receipt)
        write_json(temporary, receipt)
        os.replace(temporary, receipt_path)
        return receipt
    finally:
        temporary.unlink(missing_ok=True)
        tensor_temporary.unlink(missing_ok=True)


def prepare_prompt_embeddings(path, *, progress=None):
    """The sole explicit installation-time encoder execution; never inference."""
    config = _load_weight_installation(path)
    runtime = _prompt_cache_runtime(config)
    receipt_path = _prompt_cache_path(config)
    cached_receipt = None
    if receipt_path.is_file():
        try:
            receipt = json.loads(receipt_path.read_text(encoding='utf8'))
            _validate_prompt_cache_receipt(config, receipt, runtime=True)
            cached_receipt = receipt
        except (ValueError, OSError, KeyError):
            pass  # Explicit prepare repairs an invalid cache; inference cannot.
    import torch
    from accelerate import cpu_offload
    from diffusers import FluxKontextPipeline
    torch.set_num_threads(config.get('torch_cpu_threads', 8))
    gpu_id = config.get('gpu_id', 0)
    if not torch.cuda.is_available() or gpu_id >= torch.cuda.device_count():
        raise RuntimeError('Prompt embeddings prepare requires the configured CUDA device')
    encoding_device = _prompt_encoding_device(torch, gpu_id)
    if cached_receipt is not None and cached_receipt.get('encoding_device') == encoding_device:
        return {**cached_receipt, 'prepare_reused': True, 'receipt_path': str(receipt_path)}
    stages = []
    if progress:
        progress('beauty_prompt_prepare', '正在准备官方固定提示词的精确编码缓存。', .1)
    with _measure_component_loading(stages):
        pipe = FluxKontextPipeline.from_pretrained(config['base_model_root'],
            torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True,
            transformer=None, vae=None, image_encoder=None, feature_extractor=None)
    for name, classes in PROMPT_ENCODING['preload_module_classes'].items():
        component = getattr(pipe, name)
        if component.dtype != torch.bfloat16:
            raise _prompt_cache_error(config, 'prepare loaded an unexpected encoder dtype')
        cpu_offload(component, execution_device=torch.device('cuda:'+str(gpu_id)),
                    offload_buffers=True, preload_module_classes=classes)
    before, started = _runtime_counters(), time.perf_counter()
    with torch.cuda.device(gpu_id), torch.inference_mode(), torch.random.fork_rng(devices=[gpu_id]):
        prompt, pooled, _text_ids = pipe.encode_prompt(prompt=RECIPE['prompt'],
            prompt_2=None, device=torch.device('cuda:'+str(gpu_id)),
            num_images_per_prompt=1, max_sequence_length=512, lora_scale=None)
        tensors = {'prompt_embeds': prompt.detach().cpu(),
                   'pooled_prompt_embeds': pooled.detach().cpu()}
        encoding_device = _prompt_encoding_device(torch, gpu_id)
    stages.append(_timing_record('fixed_prompt_encoding_and_cpu_export', started, before))
    # Recheck only the cheap installation receipt/stat contract after encoding.
    if _prompt_cache_binding(config) != _prompt_cache_binding(_load_weight_installation(path)):
        raise _prompt_cache_error(config, 'installation changed during prepare')
    receipt = _publish_prompt_embeddings(config, tensors, runtime, encoding_device, stages)
    return {**receipt, 'prepare_reused': False, 'receipt_path': str(receipt_path)}


def model_identity(config):
    """The installation identity, including its originally audited policy."""
    return {'method': 'BeautyGRPO_original_generated_appearance', 'base_model_id': BASE_ID,
            'base_revision': BASE_REVISION, 'base_manifest_sha256': config['base_manifest_sha256'],
            'lora_sha256': EXPECTED_LORA_SHA, 'recipe': dict(RECIPE),
            'roi_policy': ROI_POLICY, 'composition_policy': COMPOSITION_POLICY,
            'offload': config['offload'], 'quantization': None,
            'source_texture_restoration': False, 'automatic_downloads': False,
            'fallback_to_base': False}


def inference_identity(config):
    profile = execution_profile(config)
    return {**model_identity(config), 'installation_offload': config['offload'],
            'offload': profile['offload'], 'quantization': profile['transformer']['quantization'], 'execution_profile': profile}


def verify_base(config, *, progress=None, installation=False):
    """Return base evidence without rehashing at normal render time.

    The explicit installation pass is the sole owner of the expensive content
    verification.  Existing callers that need that audit must say so with
    ``installation=True``.
    """
    if not installation:
        receipt = config.get('verification_receipt')
        base = receipt.get('base') if isinstance(receipt, dict) else None
        if not isinstance(base, dict) or base.get('file_count') != 25:
            raise ValueError('BeautyGRPO installation verification receipt is required')
        return {'base_manifest_sha256': config['base_manifest_sha256'],
                'all_files_hash_verified': True, 'verified_file_count': base['file_count'],
                'verification_schema': VERIFICATION_SCHEMA,
                'verification_receipt_path': config.get('verification_receipt_path'),
                'verification_stage': 'installation'}
    catalog = json.loads(Path(config['base_manifest']).read_text(encoding='utf8'))
    rows = catalog['completed_files']
    for index, row in enumerate(rows):
        if progress:
            progress('beauty_model_verification', '正在核验本地人像模型文件。', .16+.08*index/len(rows))
        if sha(Path(config['base_model_root'])/row['path']) != row['local_sha256']:
            raise ValueError('Local BeautyGRPO base bytes changed: '+row['path'])
    return {'base_manifest_sha256': config['base_manifest_sha256'], 'all_files_hash_verified': True,
            'verified_file_count': len(rows)}


def verify_lora_file(path, *, verify_hash=True):
    """Verify every official matrix, pair shape, rank, dtype and finite value."""
    if verify_hash and sha(path) != EXPECTED_LORA_SHA:
        raise ValueError('BeautyGRPO LoRA bytes differ from the pinned official file')
    from safetensors import safe_open
    import torch
    with safe_open(path, framework='pt', device='cpu') as opened:
        keys = list(opened.keys())
        if len(keys) != 684:
            raise ValueError('Expected all 684 BeautyGRPO LoRA tensors')
        pairs = {}
        for key in keys:
            if not key.startswith('transformer.') or not key.endswith(('.lora_A.weight', '.lora_B.weight')):
                raise ValueError('Unexpected BeautyGRPO adapter key: '+key)
            prefix, kind = key.rsplit('.lora_', 1)
            tensor = opened.get_tensor(key)
            if tensor.ndim != 2 or tensor.dtype != torch.bfloat16 or not bool(torch.isfinite(tensor).all()):
                raise ValueError('Invalid BeautyGRPO adapter matrix: '+key)
            pairs.setdefault(prefix, {})[kind[0]] = tuple(tensor.shape)
        if len(pairs) != 342 or any(set(pair) != {'A', 'B'} or pair['A'][0] != 32
                                    or pair['B'][1] != 32 for pair in pairs.values()):
            raise ValueError('BeautyGRPO requires 342 complete rank-32 matrix pairs')
    return keys


























def load_pipeline(config, *, progress=None):
    startup_started = time.perf_counter()
    keys = ((config.get('verification_receipt') or {}).get('lora') or {}).get('tensor_keys')
    if (not isinstance(keys, list) or len(keys) != 684
            or any(not isinstance(key, str) or not key.startswith('transformer.')
                   or not key.endswith(('.lora_A.weight', '.lora_B.weight')) for key in keys)):
        raise ValueError('BeautyGRPO LoRA tensor key inventory is incomplete')
    profile = execution_profile(config)
    identity = inference_identity(config)
    key = (str(Path(config['base_model_root']).resolve()), str(Path(config['lora_path']).resolve()),
           _canonical_sha(identity))
    if _PIPELINE_CACHE and key not in _PIPELINE_CACHE:
        raise ValueError('A running BeautyGRPO process cannot replace its installed model identity')
    if key in _PIPELINE_CACHE:
        pipe, evidence = _PIPELINE_CACHE[key]
        return pipe, {**evidence, 'pipeline_reused': True, 'verification_reused': True,
                      'runtime_import_seconds': 0., 'startup_seconds': 0.}
    load_stages=[]
    before, started = _runtime_counters(), time.perf_counter()
    prompt_tensors, prompt_evidence=load_prompt_embeddings(config)
    load_stages.append(_timing_record('installed_prompt_embeddings_loading', started, before))
    import_started = time.perf_counter()
    import torch, diffusers, importlib.metadata
    from diffusers import FluxKontextPipeline
    from beautygrpo_nunchaku import installed_model, load_transformer, bind_lora, PACKAGE_VERSION
    if importlib.metadata.version('nunchaku') != PACKAGE_VERSION:
        raise ValueError('Beauty requires the qualified Nunchaku package version')
    runtime_import_seconds = time.perf_counter()-import_started
    torch.set_num_threads(config.get('torch_cpu_threads', 8))
    if not torch.cuda.is_available():
        raise RuntimeError('Beauty requires its configured CUDA runtime')
    gpu_id=config.get('gpu_id', 0)
    if gpu_id >= torch.cuda.device_count(): raise RuntimeError('Beauty CUDA device is unavailable')
    torch.cuda.set_device(gpu_id)
    if prompt_evidence['encoding_device'] != _prompt_encoding_device(torch, gpu_id):
        raise _prompt_cache_error(config, 'were encoded on a different device or math configuration')
    base_evidence=verify_base(config, progress=progress)
    quantized_path, quantized_evidence=installed_model(config)
    if progress: progress('beauty_model_loading', '正在加载 GPU 人像美化模型。', .25)
    before, started = _runtime_counters(), time.perf_counter()
    storage_stages=[]
    transformer=load_transformer(quantized_path, torch_dtype=torch.bfloat16,
        device=f'cuda:{gpu_id}', io_profile=storage_stages)
    quantized_loading=_timing_record('quantized_transformer_loading', started, before)
    quantized_loading['storage_profile']={'schema':'beautygrpo_weight_storage.v1',
        'stages':storage_stages}
    load_stages.append(quantized_loading)
    before, started = _runtime_counters(), time.perf_counter()
    lora_evidence=bind_lora(transformer, config['lora_path'], keys)
    load_stages.append(_timing_record('complete_lora_conversion_and_binding', started, before))
    before, started = _runtime_counters(), time.perf_counter()
    with _measure_component_loading(load_stages):
        from diffusers import AutoencoderKL
        vae_root = Path(config.get('vae_model_root') or Path(config['base_model_root'])/'vae').resolve()
        vae = AutoencoderKL.from_pretrained(str(vae_root), torch_dtype=torch.bfloat16, local_files_only=True)
        pipe=FluxKontextPipeline.from_pretrained(config['base_model_root'], transformer=transformer, vae=vae,
            text_encoder=None, text_encoder_2=None, tokenizer=None, tokenizer_2=None,
            torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True).to(f'cuda:{gpu_id}')
    pipe.vae.enable_slicing();pipe.vae.enable_tiling()
    pipe._beauty_prompt_embeddings=prompt_tensors
    load_stages.append(_timing_record('vae_and_conditioning_loading', started, before))
    evidence={'model_identity':identity, 'base_verified':base_evidence,
        'quantized_transformer_verified':quantized_evidence, 'lora_verified':lora_evidence,
        'versions':{'torch':torch.__version__, 'diffusers':diffusers.__version__, 'nunchaku':PACKAGE_VERSION},
        'pipeline_reused':False, 'verification_reused':False,
        'runtime_import_seconds':runtime_import_seconds, 'startup_seconds':time.perf_counter()-startup_started,
        'model_load_seconds':sum(row['seconds'] for row in load_stages
            if 'seconds' in row and row['stage'] != 'component_loading'),
        'model_load_seconds_scope':'outer_loading_spans; nested storage/components excluded',
        'execution_profile':profile, 'execution_identity_sha256':_canonical_sha(profile),
        'installation_offload':config['offload'], 'offload_setup_seconds':0.,
        'model_load_profile':{'schema':'beautygrpo_model_load_profile.v1','stages':load_stages},
        'prompt_embedding_cache':prompt_evidence, 'torch_cpu_threads':torch.get_num_threads(),
        'cuda_memory_accounting':'Torch allocator counters exclude Nunchaku native CUDA allocations'}
    _PIPELINE_CACHE[key]=(pipe,evidence)
    return pipe,evidence



def _point_bounds(points):
    return [float(points[:,0].min()), float(points[:,1].min()), float(points[:,0].max()), float(points[:,1].max())]


def face_focused_box(region, source_size, fields):
    """A complete-face crop; qualified support expands and never shrinks it."""
    width, height = source_size
    points = np.asarray(region.get('landmarks'), np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) <= max(FACE_OVAL) or not np.isfinite(points).all():
        raise ValueError('Complete canonical face landmarks are required')
    box = _point_bounds(points[list(FACE_OVAL)])
    if min(box[2]-box[0], box[3]-box[1]) <= 1:
        raise ValueError('Canonical face extent is degenerate')
    parent = region['native_box']
    skin = np.asarray(fields['skin_weight'], np.float32)
    if skin.shape != (parent[3]-parent[1], parent[2]-parent[0]):
        raise ValueError('Parser fields do not match original native face context')
    yy, xx = np.where(skin > 0)
    if len(xx):
        box = [min(box[0], int(xx.min())+parent[0]), min(box[1], int(yy.min())+parent[1]),
               max(box[2], int(xx.max())+1+parent[0]), max(box[3], int(yy.max())+1+parent[1])]
    side = math.ceil(1.5*max(box[2]-box[0], box[3]-box[1]))
    cx, cy = (box[0]+box[2])*.5, (box[1]+box[3])*.5
    if side <= min(width, height):
        l = min(max(0, math.floor(cx-side/2)), width-side)
        t = min(max(0, math.floor(cy-side/2)), height-side)
        return [l,t,l+side,t+side]
    return [max(0, math.floor(cx-side/2)), max(0, math.floor(cy-side/2)),
            min(width, math.ceil(cx+side/2)), min(height, math.ceil(cy+side/2))]


def _crop_field(values, parent_box, box):
    """Place observed parent fields into the focused ROI, with no extrapolation."""
    shape = (box[3]-box[1], box[2]-box[0])
    out = np.zeros(shape, values.dtype)
    l,t = max(box[0],parent_box[0]), max(box[1],parent_box[1])
    r,b = min(box[2],parent_box[2]), min(box[3],parent_box[3])
    if l < r and t < b:
        out[t-box[1]:b-box[1],l-box[0]:r-box[0]] = values[t-parent_box[1]:b-parent_box[1],l-parent_box[0]:r-parent_box[0]]
    return out


def forehead_extension(points, oval, labels, visible, box):
    """Follow observed forehead skin upward to its hairline, in face axes.

    The canonical oval's top landmark is often halfway down the forehead. Only
    a hard-skin connected component seeded in the oval's upper fifth may extend
    that boundary. A local face-width corridor and the upper fifth prevent
    this operation from following skin-labelled hands or lower-face props.
    No probability, colour similarity, photograph coordinates or new inference
    is used, and the original generation context bounds the available pixels.
    """
    empty = np.zeros(oval.shape,bool)
    eye_a = (points[33]+points[133])*.5
    eye_b = (points[362]+points[263])*.5
    eye_axis = eye_b-eye_a
    length = float(np.linalg.norm(eye_axis))
    if length <= 1:
        return empty, {'method':'upward_connected_hard_skin.v1','state':'unavailable',
                       'reason':'degenerate_visible_eye_axis','added_native_pixels':0}
    across = eye_axis/length
    down = np.array([-across[1],across[0]])
    center = (eye_a+eye_b)*.5
    if np.dot(points[152]-center,down)<0:
        down *= -1
    face_points = points[list(FACE_OVAL)]-center
    u,v = face_points@across,face_points@down
    height = float(np.ptp(v))
    if height <= 1:
        return empty, {'method':'upward_connected_hard_skin.v1','state':'unavailable',
                       'reason':'degenerate_face_height','added_native_pixels':0}
    top_band_limit = float(v.min()+.2*height)
    yy,xx = np.ogrid[box[1]:box[3],box[0]:box[2]]
    pixel_u = ((xx+.5-center[0])*across[0]+(yy+.5-center[1])*across[1]).astype(np.float32)
    pixel_v = ((xx+.5-center[0])*down[0]+(yy+.5-center[1])*down[1]).astype(np.float32)
    corridor = (pixel_u>=u.min())&(pixel_u<=u.max())&(pixel_v<=top_band_limit)
    domain = (labels==2)&(visible>0)&corridor
    seeds = domain&oval
    if not seeds.any():
        return empty, {'method':'upward_connected_hard_skin.v1','state':'no_reliable_forehead_seed',
                       'added_native_pixels':0,'seed_band_fraction':.2}
    reached = binary_propagation(seeds,structure=np.ones((3,3),bool),mask=domain)
    extension = reached&~oval
    metadata = {'method':'upward_connected_hard_skin.v1','state':'observed' if extension.any() else 'no_extension',
                'seed_band_fraction':.2,'reliable_seed_pixels':int(seeds.sum()),
                'added_native_pixels':int(extension.sum()),'hard_material_label':2,
                'face_axis_from_visible_eyes':True,'within_existing_generation_crop':True,
                'nonface_material_crossed':False,'source_texture_restoration':False,
                'clipped_at_crop_top':bool(extension[0].any()),
                'limitations':'Hard 512px parsing bounds the hairline; this is not soft native hair matting.'}
    return extension,metadata


def composition_weight(region, fields, box):
    """Retain generated eyebrows, eyes and lips; blend boundaries and occluders."""
    parent = region['native_box']
    labels = _crop_field(np.asarray(fields['labels'], np.uint8), parent, box)
    visible = _crop_field(np.asarray(fields['visible_face_support'], np.float32), parent, box)
    if not np.isfinite(visible).all() or (visible < 0).any() or (visible > 1).any():
        raise ValueError('Invalid visible-face composition support')
    points = np.asarray(region['landmarks'], np.float64)
    polygon = points[list(FACE_OVAL)]-[box[0],box[1]]
    canvas = Image.new('L',(box[2]-box[0],box[3]-box[1]))
    ImageDraw.Draw(canvas).polygon([tuple(point) for point in polygon], fill=255)
    oval = np.asarray(canvas)>0
    extension,extension_policy = forehead_extension(points,oval,labels,visible,box)
    complete_face = oval|extension
    feather = max(1., float(np.ptp(polygon[:,1]))*.015)
    oval_weight = np.clip(distance_transform_edt(complete_face)/feather,0,1)
    face_material = np.isin(labels, FACE_MATERIAL_LABELS)&complete_face
    material_weight = np.clip(distance_transform_edt(face_material)/feather,0,1)
    status = (region.get('texture_eligibility') or {}).get('status')
    if status == 'unavailable':
        visible[:] = 0
    elif status == 'visible_upper_face_only':
        # Missing mouth is not permission to edit paper/hands. Fully visible
        # hard-parsed eyes/brows still belong to the generated upper face.
        eye_brow = np.isin(labels,(6,7,8,9))&oval
        visible = np.maximum(visible,np.clip(distance_transform_edt(eye_brow)/feather,0,1))
    elif status != 'full_visible_face':
        raise ValueError('Current qualified visible face status required')
    weight = np.minimum(oval_weight, material_weight)*visible
    weight = np.clip(weight,0,1).astype(np.float32)
    return weight, {'policy': COMPOSITION_POLICY, 'feather_native_pixels': feather,
                    'forehead_extension':extension_policy,
                    'face_feature_pixels_included': int(((weight>0)&np.isin(labels,(6,7,8,9,11,12,13,15))).sum()),
                    'composition_pixels': int((weight>0).sum()), 'qualification_status': status,
                    'nonface_material_excluded': True, 'skin_only': False, 'features_gate_applied': False,
                    'raw_probability_applied': False, 'source_texture_restoration': False,
                    'limitations': ['Parser labels and landmark visibility are conservative observations, not a learned general occluder detector.']
                        + (['Mouth/lower-face visibility unverified; only observed upper face including eyes and brows is composed.']
                           if status=='visible_upper_face_only' else [])}


def refresh_composition(record, output):
    """Change seams on completed generated RGB without models or GPU work.

    Preserve actual model-output hashes and runtime evidence. The inferred crop
    must stay identical; changing it requires a new generation rather than
    guessing a warp or stretching an unrelated output.
    """
    record = json.loads(json.dumps(record))
    if (record.get('schema') != 'beautygrpo_layer_prepare.v1'
            or record.get('inference_completed') is not True or not record.get('faces')):
        raise ValueError('Completed BeautyGRPO preparation is required for recomposition')
    region_asset = record['regions']
    regions_path = Path(region_asset['path'])
    if sha(regions_path) != region_asset['sha256']:
        raise ValueError('Original portrait parsing contract changed')
    regions = json.loads(regions_path.read_text(encoding='utf8'))
    if (regions.get('schema_version') != 'portrait_regions.v1'
            or regions.get('source_sha256') != record['model_input_sha256']
            or regions.get('source_native_size') != record['native_size']):
        raise ValueError('Recomposition parsing belongs to another display input')
    by_id = {region['id']:region for region in regions['regions']}
    output = Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    for index,face in enumerate(record['faces']):
        for key in ('generated_native_roi','model_output','model_input'):
            if sha(face[key]) != face[key+'_sha256']:
                raise ValueError('Actual BeautyGRPO pixels changed before recomposition: '+key)
        region = by_id[face['id']]
        with np.load(region['npz_path'],allow_pickle=False) as opened:
            fields = {key:opened[key] for key in ('labels','skin_weight','visible_face_support')}
        box = face_focused_box(region,record['native_size'],fields)
        if box != face['native_box']:
            raise ValueError('Face inference crop changed; existing generation cannot be reused')
        weight,policy = composition_weight(region,fields,box)
        if not weight.any():
            raise ValueError('Recomposition cannot invent a visible face')
        path = output/f'face-{index+1:02d}-weight.npy'
        np.save(path,weight,allow_pickle=False)
        face.update(composition_weight=str(path),composition_weight_sha256=sha(path),
                    composition_policy=policy['policy'],composition=policy)
    record['model_identity']['composition_policy'] = COMPOSITION_POLICY
    record['recomposition'] = {'policy':COMPOSITION_POLICY,'new_model_inferences':0,
                               'actual_generated_rgb_unchanged':True,'model_runtime_evidence_retained':True}
    return record


def prepare_roi(source, box, size=1024):
    crop = source.crop(tuple(box))
    scale = size/max(crop.size)
    dimensions = (max(1, round(crop.width*scale)),max(1, round(crop.height*scale)))
    resized = crop.resize(dimensions,Image.Resampling.LANCZOS)
    offset = ((size-resized.width)//2,(size-resized.height)//2)
    canvas = Image.new('RGB',(size,size),(127,127,127))
    canvas.paste(resized,offset)
    mapping = {'native_roi_box': list(box), 'native_roi_size': list(crop.size), 'model_size': [size,size],
               'resized_size': list(resized.size), 'padding_offset': list(offset), 'scale_xy': [resized.width/crop.width,resized.height/crop.height],
               'mapping': 'native_crop_then_aspect_preserving_resize_and_letterbox',
               'roi_policy': ROI_POLICY, 'generated_detail_origin': '1024_model_sampling_not_camera_native_detail',
               'geometric_warp_applied': False}
    return crop,canvas,mapping


def run(input_path, regions_path, config, output, *, progress=None):
    """Run every eligible face with one fully verified pipeline and fixed recipe."""
    input_path, regions_path, output = Path(input_path).resolve(),Path(regions_path).resolve(),Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose a new BeautyGRPO inference directory')
    output.mkdir(parents=True,exist_ok=True)
    started = time.perf_counter()
    regions = json.loads(regions_path.read_text(encoding='utf8'))
    with Image.open(input_path) as opened:
        source = ImageOps.exif_transpose(opened).convert('RGB')
    if (regions.get('schema_version') != 'portrait_regions.v1' or regions.get('source_sha256') != sha(input_path)
            or regions.get('source_native_size') != list(source.size)):
        raise ValueError('Portrait regions belong to another prepared display source')
    record = {'schema': SCHEMA, 'status': 'started', 'inference_completed': False,
              'source': {'path':str(input_path),'sha256':sha(input_path),'native_size':list(source.size)},
              'regions': {'path':str(regions_path),'sha256':sha(regions_path)},
              'model_identity':inference_identity(config),'faces':[],'skipped_faces':[],
              'producer_sha256':sha(__file__),'automatic_downloads':False,'fallback_to_base':False,
              'source_texture_restoration':False,'skin_only_residual':False}
    write_json(output/'result.json',record)
    try:
        prepared = []
        ids = set()
        for region in regions.get('regions',[]):
            if region['id'] in ids:
                raise ValueError('Duplicate detected face identifiers')
            ids.add(region['id'])
            with np.load(region['npz_path'],allow_pickle=False) as opened:
                fields = {key:opened[key] for key in ('labels','skin_weight','visible_face_support')}
            box = face_focused_box(region,source.size,fields)
            weight,policy = composition_weight(region,fields,box)
            if not np.any(weight):
                record['skipped_faces'].append({'id':region['id'],'reason':'no_visible_complete_face_ownership','policy':policy})
                continue
            # Folder names are generated locally; untrusted region IDs never
            # determine filesystem paths.
            directory = output/f'face-{len(prepared)+1:02d}'
            directory.mkdir()
            crop,canvas,mapping = prepare_roi(source,box,RECIPE['size'])
            crop.save(directory/'source-roi.png')
            canvas.save(directory/'model-input.png')
            np.save(directory/'composition-weight.npy',weight,allow_pickle=False)
            prepared.append((region,box,directory,crop,canvas,mapping,policy))
        if not prepared:
            record.update(status='no_faces_available',elapsed_seconds=time.perf_counter()-started)
            write_json(output/'result.json',record)
            return record
        pipe,evidence = load_pipeline(config,progress=progress)
        record.update(evidence)
        import torch
        gpu_id = config.get('gpu_id',0)
        torch.cuda.reset_peak_memory_stats(gpu_id)
        # Prompt embeddings are immutable for this fixed recipe. Keep one
        # execution-device copy for every face in this request instead of
        # repeating the same CPU->device transfer before each diffusion call.
        # The explicit per-face Generator below remains unchanged, so this
        # cache cannot couple or advance face sampling state.
        device_prompt_embeddings = None
        for index,(region,box,directory,crop,canvas,mapping,policy) in enumerate(prepared):
            sampling = time.perf_counter()
            if progress:
                progress('beauty_face_generation',f'正在生成第 {index+1}/{len(prepared)} 张脸的人像美化层。',.38+.48*index/len(prepared))
            def on_step_end(_pipe,step,_tensors,kwargs):
                if progress:
                    amount=.38+.48*(index+(step+1)/RECIPE['steps'])/len(prepared)
                    progress('beauty_face_generation',f'人脸 {index+1}/{len(prepared)}：{step+1}/{RECIPE["steps"]}。',amount)
                return kwargs
            # Diffusers invokes a callback once per denoising step.  Headless
            # workers have no progress consumer, so avoid installing a no-op
            # callback and its Python crossing on every step.  The callback is
            # still used verbatim for UI/progress callers.
            generation_kwargs = dict(
                image=canvas,guidance_scale=RECIPE['guidance'],
                num_inference_steps=RECIPE['steps'],generator=torch.Generator(f'cuda:{gpu_id}').manual_seed(RECIPE['seed']),
                width=RECIPE['size'],height=RECIPE['size'],max_sequence_length=512)
            if progress:
                generation_kwargs['callback_on_step_end'] = on_step_end
            with torch.cuda.device(gpu_id), torch.inference_mode(), _measure_inference(pipe) as inference_profile:
                # Retain completed profiling spans if inference itself fails.
                record['active_inference_profile'] = {'face_id': region['id'], **inference_profile}
                if device_prompt_embeddings is None:
                    before, transfer_started = _runtime_counters(), time.perf_counter()
                    device_prompt_embeddings = {name: tensor.to(device=pipe._execution_device)
                        for name, tensor in pipe._beauty_prompt_embeddings.items()}
                    inference_profile['stages'].append(_timing_record(
                        'prompt_embedding_device_transfer', transfer_started, before))
                else:
                    before, reuse_started = _runtime_counters(), time.perf_counter()
                    inference_profile['stages'].append(_timing_record(
                        'prompt_embedding_device_reuse', reuse_started, before))
                generation_kwargs.update(device_prompt_embeddings)
                generated = pipe(**generation_kwargs).images[0]
            if generated.size != (RECIPE['size'],RECIPE['size']):
                raise ValueError('BeautyGRPO generated canvas has changed size')
            generated = generated.convert('RGB')
            generated.save(directory/'model-output.png')
            x,y = mapping['padding_offset'];w,h = mapping['resized_size']
            unpadded = generated.crop((x,y,x+w,y+h))
            unpadded.save(directory/'unpadded-output.png')
            native = unpadded.resize(crop.size,Image.Resampling.LANCZOS)
            native_path = directory/'generated-native-roi.png';native.save(native_path)
            face = {'id':region['id'],'native_box':box,'mapping':mapping,
                    'parent_native_box':region['native_box'],'generated_native_roi':str(native_path),
                    'generated_native_roi_sha256':sha(native_path),
                    'composition_weight':str(directory/'composition-weight.npy'),
                    'composition_weight_sha256':sha(directory/'composition-weight.npy'),
                    'composition_policy':policy['policy'],'composition':policy,
                    'model_input':str(directory/'model-input.png'),'model_input_sha256':sha(directory/'model-input.png'),
                    'model_output':str(directory/'model-output.png'),'model_output_sha256':sha(directory/'model-output.png'),
                    'source_roi':str(directory/'source-roi.png'),'source_roi_sha256':sha(directory/'source-roi.png'),
                    'sampling_seconds':time.perf_counter()-sampling,'inference_profile':inference_profile,
                    'geometric_warp_applied':False,
                    'source_texture_restoration':False,'skin_only_residual':False}
            record['faces'].append(face)
            record.pop('active_inference_profile', None)
            record['status'] = 'face_complete'
            write_json(output/'result.json',record)
        record.update(status='inference_complete',inference_completed=True,
                      elapsed_seconds=time.perf_counter()-started,
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(gpu_id),
                      peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(gpu_id))
        write_json(output/'result.json',record)
        return record
    except Exception as exc:
        record.update(status='failed_no_complete_layer',inference_completed=False,
                      error_type=type(exc).__name__,error=str(exc),elapsed_seconds=time.perf_counter()-started)
        write_json(output/'result.json',record)
        (output/'error.txt').write_text(traceback.format_exc(),encoding='utf8')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-config',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--regions',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    config = load_installation(args.model_config)
    result = run(args.input,args.regions,config,args.output)
    print(json.dumps({'status':result['status'],'face_count':len(result['faces']),
                      'result':str((args.output/'result.json').resolve())},ensure_ascii=False),flush=True)
    return 0 if result['inference_completed'] else 2


if __name__ == '__main__':
    sys.exit(main())
