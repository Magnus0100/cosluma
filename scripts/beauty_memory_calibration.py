"""Strict local, source-specific Beauty memory evidence; unknown stays cold.

This file contains no private source identities. Calibration is an operator
installed local receipt, never an image-plan flag or a model-output cache hit.
"""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path

from asset_identity import sha256_file
from runtime_resources import code_path

SCHEMA = 'beauty_single_source_memory_calibration.v1'
PREPARE_FUNCTIONS = (
    'validate_options', 'stable_retouch_identity', '_source_display_preview', 'prepare_source',
    '_cache_binding', '_copy_layer', '_cache_result', '_predecode_binding', '_predecode_cache_result',
    '_publish_cache', '_run_regions', '_owned_cached_result', '_continued_source', 'prepare',
    'recompose_prepared')
PRODUCERS = ('beautygrpo_worker.py', 'beautygrpo_nunchaku.py', 'portrait_regions_worker.py',
    'portrait_regions_identity.py', 'portrait_beauty_layer.py', 'raw_support.py',
    'display_transform.py', 'native_source_baseline.py', 'native_array_storage.py', 'runtime_resources.py')
# Calibration and the cold path use one model-demand value. System headroom is
# applied once by runtime_memory.memory_budget; the old separate 22 GiB demand
# duplicated that reserve and blocked otherwise healthy 64 GiB hosts.
from runtime_memory import BEAUTY_COLD_COMMIT_BYTES
CALIBRATED_COMMIT_BYTES = BEAUTY_COLD_COMMIT_BYTES


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode('utf8')).hexdigest()


def calibration_path():
    from current_model_registry import settings_directory
    return settings_directory()/'beauty-memory-calibration.json'


def producer_fingerprint(directory=None):
    locate = (lambda name: Path(directory)/name) if directory else code_path
    parsed = ast.parse(locate('beautygrpo_prepare.py').read_text(encoding='utf-8-sig'))
    functions = {node.name: node for node in parsed.body if isinstance(node, ast.FunctionDef)}
    math = {name: hashlib.sha256(ast.dump(functions[name], include_attributes=False).encode('utf8')).hexdigest()
            for name in PREPARE_FUNCTIONS}
    return {'identity_policy': 'full_numeric_modules_and_explicit_prepare_function_AST.v1',
        'modules': {name: sha256_file(locate(name)) for name in PRODUCERS},
        'prepare_functions': math}


def profile_fingerprint(config, directory=None):
    from beautygrpo_worker import execution_profile, inference_identity
    profile = execution_profile(config)
    receipt = config.get('_prompt_embedding_receipt') or {}
    if (profile.get('offload') != 'nunchaku_cuda'
            or profile.get('prompt_conditioning', {}).get('source') != 'installed_exact_embeddings'
            or not config.get('_prompt_embedding_identity') or not receipt):
        raise ValueError('Calibration requires verified installed prompt embeddings')
    artifact = receipt.get('artifact') or {}
    # The installation loader has validated the receipt and actual artifact;
    # bind their identities rather than accepting a caller's cache-warm flag.
    proof = {'receipt_sha256': config['_prompt_embedding_identity'],
        'artifact_sha256': artifact.get('sha256'),
        'binding_sha256': receipt.get('binding_sha256'),
        'runtime_sha256': receipt.get('runtime_sha256')}
    if not all(isinstance(value, str) and len(value) == 64 for value in proof.values()):
        raise ValueError('Verified prompt embedding identity is incomplete')
    return {'execution_identity_sha256': digest(profile),
        'model_identity_sha256': digest(inference_identity(config)),
        'prompt_embedding_proof': proof,
        'face_registry_sha256': sha256_file(config['face_registry']),
        'producers': producer_fingerprint(directory)}


def select(run, root, config, options=None):
    """Return verified scope or None; no missing/invalid proof renews admission."""
    from runtime_memory import BEAUTY_COLD_PHYSICAL_BYTES, CPU_RESERVE_BYTES
    try:
        if not isinstance(run, dict) or root is None:
            return None
        # The calibration constrains this proven source to one face. Explicit
        # requests for multiple faces retain the uncalibrated cold policy.
        for supplied in (options or {}, run.get('options') or {}):
            if 'max_faces' in supplied and (type(supplied['max_faces']) is not int or supplied['max_faces'] != 1):
                return None
        if (run.get('options') or {}).get('native_source_retouch'):
            return None
        path = calibration_path()
        receipt = json.loads(path.read_text(encoding='utf8'))
        if receipt.get('schema') != SCHEMA or receipt.get('profile') != profile_fingerprint(config):
            return None
        scope = receipt['scope']; source = run['source']
        source_path = Path(source['path'])
        if not source_path.is_absolute(): source_path = Path(root)/source_path
        if (scope.get('source_sha256') != source.get('sha256')
                or sha256_file(source_path) != source['sha256']
                or 'native_source_retouch_identity' not in scope or scope['native_source_retouch_identity'] is not None
                or type(scope.get('verified_face_count')) is not int or scope['verified_face_count'] != 1
                or type(scope.get('effective_max_faces')) is not int or scope['effective_max_faces'] != 1):
            return None
        from raw_support import dimensions
        actual_size = list(dimensions(source_path))
        if (actual_size != scope.get('native_size')
                or source.get('oriented_size', actual_size) != actual_size):
            return None
        measured = receipt['measurement']
        peak = measured.get('kernel_peak_job_commit_bytes')
        if (measured.get('completed') is not True or type(measured.get('beauty_inferences')) is not int or measured['beauty_inferences'] != 1
                or measured.get('pipeline_reused') is not False
                or measured.get('encoder_model_loading') is not False
                or type(measured.get('new_qwen_submissions')) is not int or measured['new_qwen_submissions'] != 0
                or type(peak) is not int or not 0 < peak <= CALIBRATED_COMMIT_BYTES+CPU_RESERVE_BYTES
                or measured.get('minimum_system_available_commit_bytes', 0) < CPU_RESERVE_BYTES
                or measured.get('minimum_system_available_physical_bytes', 0) < CPU_RESERVE_BYTES):
            return None
        budget = receipt['budget']
        if budget != {'physical_bytes': BEAUTY_COLD_PHYSICAL_BYTES, 'commit_bytes': CALIBRATED_COMMIT_BYTES,
                      'reserve_bytes': CPU_RESERVE_BYTES}:
            return None
        return {'schema': SCHEMA, 'receipt_sha256': sha256_file(path),
            'profile_sha256': digest(receipt['profile']), 'source_sha256': source['sha256'],
            'native_size': actual_size, 'native_source_retouch_identity': None,
            'effective_max_faces': 1, 'budget': budget,
            'measured_peak_job_commit_bytes': peak,
            'basis': 'local completed fixed-profile single-source calibration; not a universal guarantee'}
    except (OSError, ValueError, TypeError, KeyError, AttributeError, SyntaxError):
        return None
