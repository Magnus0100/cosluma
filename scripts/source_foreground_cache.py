"""Cache native source foreground assets using the installed GPU worker runtime."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

from local_photo_analysis import file_identity, local_runtime_paths
from raw_support import RGB_DECODE_VERSION
from runtime_resources import code_path
from native_array_storage import VALUES_ENCODING
from cache_roots import cache_dir
from cache_store import exclusive_file_lock


PARAMETERS = {
    'max_edge': 0, 'tile': 512, 'overlap': 128, 'alpha_halo': 64, 'alpha_batch_size': 4,
    'fb_halo': 96, 'trimap_edge_fraction': .004,
    'trimap_foreground_threshold': .98, 'trimap_possible_threshold': .02,
    'foreground_background_estimator': 'cuda_multilevel_fb_jacobi_v1; small arrays pymatting_cpu',
    'fb_diagnostics': 'source_fb_diagnostics.sparse.v1',
    'native_values_encoding': VALUES_ENCODING,
}
ASSET_KEYS = ('alpha_path', 'old_background_path', 'premultiplied_foreground_path',
              'unknown_edge_support_path', 'fb_diagnostics_path')


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _resolve(root, path):
    path = Path(path)
    return (path if path.is_absolute() else root/path).resolve()


def _identity(root, source, saliency, semantics, runtime=None):
    # ``semantics`` is review-only input. It is deliberately excluded from
    # this producer identity so changing diagnostic labels cannot rerun Alpha.
    runtime = runtime or local_runtime_paths(root)
    model = Path(runtime['vitmatte_model_root'])
    identity = {
        'schema_version': 'source_foreground_cache.v2',
        'source_sha256': file_identity(source), 'decode_version': RGB_DECODE_VERSION,
        'saliency_sha256': file_identity(saliency),
        'worker_sha256': file_identity(code_path('source_foreground.py')),
        'cuda_solver_sha256': file_identity(code_path('native_cuda_matting.py')),
        'cuda_scope_sha256': file_identity(code_path('native_cuda_fields.py')),
        'storage_writer_sha256': file_identity(code_path('native_array_storage.py')),
        'parameters': PARAMETERS,
    }
    if runtime['matting_profile'] != 'vitmatte_small_distinctions':
        raise ValueError('Unsupported source matting profile: '+runtime['matting_profile'])
    identity.update(matting_profile=runtime['matting_profile'],
        model_weight_sha256=file_identity(model/'model.safetensors'),
        model_config_sha256=file_identity(model/'config.json'),
        model_preprocessor_sha256=file_identity(model/'preprocessor_config.json'))
    return identity


def _valid_assets(result, identity):
    """Missing observations cannot be reconstructed from a historical pair."""
    if not isinstance(result, dict) or result.get('schema_version') != 'source_foreground.v1':
        return False
    if not all(result.get(name) and Path(result[name]).is_file() for name in ASSET_KEYS):
        return False
    if result.get('native_values_encoding') != identity['parameters']['native_values_encoding']:
        return False
    descriptor = result.get('fb_diagnostics') or {}
    if (descriptor.get('schema_version') != PARAMETERS['fb_diagnostics'] or
            descriptor.get('source_sha256') != identity['source_sha256'] or
            descriptor.get('producer_sha256') != identity['worker_sha256'] or
            descriptor.get('coordinate_space') != 'oriented_source_pixels' or
            descriptor.get('canvas_size') != result.get('analysis_size') or
            descriptor.get('renderer_consumed') is not False or
            descriptor.get('support_expanded') is not False):
        return False
    if (not descriptor.get('path') or
            Path(descriptor['path']).resolve() != Path(result['fb_diagnostics_path']).resolve() or
            descriptor.get('sha256') != file_identity(result['fb_diagnostics_path'])):
        return False
    bindings = descriptor.get('paired_asset_sha256') or {}
    return all(bindings.get(name) == file_identity(result[name]) for name in ASSET_KEYS[:-1])


def prepare_source_foreground(root, source, *, source_hash=None, saliency_path, semantics_path=None,
                              registry_path=None, native_retouch_binding=None, cancel=None):
    """Return source_foreground.v1 plus identity/cache_hit and cache locations.

    Canonical values remain native float32 alpha / B_old / Fp / applicability.
    Sparse F/B diagnostics are mandatory producer observations, never an
    inferred upgrade of saved pairs; they do not authorize or alter lighting.
    Only the current content-bound cache is reused. No historical inspection
    imports or inferred upgrades are supported. Dependencies remain in the
    installed GPU runtime.
    """
    if cancel and cancel():
        raise InterruptedError('Source foreground preparation cancelled')
    root = Path(root).resolve()
    source, saliency = _resolve(root, source), _resolve(root, saliency_path)
    semantics = _resolve(root, semantics_path) if semantics_path else None
    started = time.perf_counter()
    paths = local_runtime_paths(root, registry_path=registry_path)
    identity = _identity(root, source, saliency, semantics, paths)
    if native_retouch_binding:
        from native_source_baseline import retouch_identity
        identity['native_source_retouch']=retouch_identity({'source':{'sha256':identity['source_sha256']},
            'options':{'native_source_retouch':native_retouch_binding}},root=root)
    diagnostic_identity = {**identity,
        'review_semantics_sha256': file_identity(semantics) if semantics else None}
    if source_hash and source_hash != identity['source_sha256']:
        raise ValueError('Provided source SHA does not match the actual source file')
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    cache_base = cache_dir(root, 'source-foreground')
    directory = cache_base/identity['source_sha256'][:16]/key
    directory.mkdir(parents=True, exist_ok=True)
    with exclusive_file_lock(directory.with_name(key+'.single-flight.lock'), cancel=cancel):
        if cancel and cancel():
            raise InterruptedError('Source foreground preparation cancelled while waiting for cache')
        result_path = directory/'result.json'
        if result_path.is_file():
            # A disposable worker receipt can be interrupted between writes (or
            # copied from an older incomplete run). Treat malformed JSON/schema or
            # a failed asset check as an ordinary miss so the existing Alpha cache
            # and current producer can recover. Post-worker reads below remain
            # strict: a worker that claims completion must return a valid bundle.
            try:
                cached = _read(result_path)
                reusable = (isinstance(cached, dict)
                            and cached.get('identity') == identity
                            and _valid_assets(cached, identity))
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                cached, reusable = None, False
            if reusable:
                cached = dict(cached)
                cached.setdefault('provenance', {})['input_identity'] = diagnostic_identity
                if isinstance(cached.get('alpha_cache'), dict):
                    producer = cached['alpha_cache']
                    cached['producer_alpha_cache'] = producer
                    cached['alpha_cache'] = {key: value for key, value in producer.items() if key != 'miss_reason'}
                    cached['alpha_cache'].update(cache_hit=True, cache_scope='completed_foreground_bundle',
                        model_inference_batches_this_attempt=0)
                return {**cached, 'cache_hit': True, 'cache_root': str(cache_base.resolve()),
                        'cache_lookup_seconds': round(time.perf_counter()-started, 3)}
        argv = [code_path('source_foreground.py'),
                '--source', source, '--saliency', saliency, '--output-dir', directory,
                '--alpha-cache-dir', cache_base/'alpha-refinement'/identity['source_sha256'][:16],
                '--model-dir', paths['vitmatte_model_root'],
                '--max-edge', '0', '--tile', '512', '--halo', '64']
        if semantics:
            argv.extend(['--semantics', semantics])
        if native_retouch_binding:
            binding_path=directory/'native-retouch-binding.json'
            binding_path.write_text(json.dumps(native_retouch_binding,ensure_ascii=False,indent=2),encoding='utf8')
            argv.extend(['--native-retouch-binding',binding_path])
        from reference_stage_worker import execute_stage
        metrics = execute_stage(paths['python'], root, argv, directory, cancel,
            stage='source_foreground', log_path=directory/'worker.log',
            registry_path=paths['registry_path'])
        if cancel and cancel():
            raise InterruptedError('Source foreground preparation cancelled')
        result = _read(result_path)
        if metrics.get('worker_timeline'):
            import workflow_timing
            workflow_timing.record_worker('source_foreground_worker',
                wall_seconds=metrics.get('wall_seconds'), worker_timeline=metrics['worker_timeline'],
                worker_pid=metrics.get('worker_pid'),
                native_field_execution=metrics.get('native_field_execution'),
                native_libraw_threading=metrics.get('native_libraw_threading'))
        if not _valid_assets(result, identity):
            raise RuntimeError('Source foreground producer lacks valid source-bound F/B diagnostics')
        result.update(identity=identity, cache_key=key, cache_root=str(cache_base.resolve()), cache_directory=str(directory),
                      result_path=str(result_path), cache_hit=False,
                      cache_lookup_seconds=round(time.perf_counter()-started, 3))
        result['provenance']['input_identity'] = diagnostic_identity
        result['provenance']['gpt_calls'] = 0
        temporary_result = result_path.with_name(result_path.name+'.tmp-'+str(os.getpid()))
        temporary_result.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        os.replace(temporary_result, result_path)
        return result
