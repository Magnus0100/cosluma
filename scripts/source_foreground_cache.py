"""Cache native source foreground assets using the installed GPU worker runtime."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

from local_photo_analysis import _worker, file_identity, local_runtime_paths
from raw_support import RGB_DECODE_VERSION
from runtime_resources import code_path


PARAMETERS = {
    'max_edge': 0, 'tile': 512, 'overlap': 128, 'alpha_halo': 64,
    'fb_halo': 96, 'trimap_edge_fraction': .004,
    'trimap_foreground_threshold': .98, 'trimap_possible_threshold': .02,
    'foreground_background_estimator': 'PyMatting estimate_foreground_ml1.1.16, linear source tiles',
    'fb_diagnostics': 'source_fb_diagnostics.sparse.v1',
}
ASSET_KEYS = ('alpha_path', 'old_background_path', 'premultiplied_foreground_path',
              'unknown_edge_support_path', 'fb_diagnostics_path')


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _resolve(root, path):
    path = Path(path)
    return (path if path.is_absolute() else root/path).resolve()


def _identity(root, source, saliency, semantics, runtime=None):
    runtime = runtime or local_runtime_paths(root)
    model = Path(runtime['vitmatte_model_root'])
    identity = {
        'schema_version': 'source_foreground_cache.v2',
        'source_sha256': file_identity(source), 'decode_version': RGB_DECODE_VERSION,
        'saliency_sha256': file_identity(saliency),
        'worker_sha256': file_identity(code_path('source_foreground.py')),
        'parameters': PARAMETERS,
        'review_semantics_sha256': file_identity(semantics) if semantics else None,
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
    if not all(result.get(name) and Path(result[name]).is_file() for name in ASSET_KEYS):
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
                              registry_path=None, native_retouch_binding=None):
    """Return source_foreground.v1 plus identity/cache_hit and cache locations.

    Canonical values remain native float32 alpha / B_old / Fp / applicability.
    Sparse F/B diagnostics are mandatory producer observations, never an
    inferred upgrade of saved pairs; they do not authorize or alter lighting.
    Only the current content-bound cache is reused. No historical inspection
    imports or inferred upgrades are supported. Dependencies remain in the
    installed GPU runtime.
    """
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
    if source_hash and source_hash != identity['source_sha256']:
        raise ValueError('Provided source SHA does not match the actual source file')
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    directory = root/'outputs/cache/source-foreground'/identity['source_sha256'][:16]/key
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory/'result.json'
    if result_path.is_file():
        cached = _read(result_path)
        if cached.get('identity') == identity and _valid_assets(cached, identity):
            return {**cached, 'cache_hit': True, 'cache_lookup_seconds': round(time.perf_counter()-started, 3)}
    argv = [paths['python'], '-X', 'utf8', code_path('source_foreground.py'),
            '--source', source, '--saliency', saliency, '--output-dir', directory,
            '--model-dir', paths['vitmatte_model_root'],
            '--max-edge', '0', '--tile', '512', '--halo', '64']
    if semantics:
        argv.extend(['--semantics', semantics])
    if native_retouch_binding:
        binding_path=directory/'native-retouch-binding.json'
        binding_path.write_text(json.dumps(native_retouch_binding,ensure_ascii=False,indent=2),encoding='utf8')
        argv.extend(['--native-retouch-binding',binding_path])
    _worker(argv, directory/'worker.log')
    result = _read(result_path)
    if not _valid_assets(result, identity):
        raise RuntimeError('Source foreground producer lacks valid source-bound F/B diagnostics')
    result.update(identity=identity, cache_key=key, cache_directory=str(directory),
                  result_path=str(result_path), cache_hit=False,
                  cache_lookup_seconds=round(time.perf_counter()-started, 3))
    result['provenance']['input_identity'] = identity
    result['provenance']['gpt_calls'] = 0
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return result
