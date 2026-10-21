"""Prepare an owned, editable BeautyGRPO portrait layer for the workbench.

The desktop calls an isolated configured runtime. Source preparation consumes
the same display transform and approved native repairs as the main pipeline.
Model generation and automatic parsing occur only here; frozen layer replay
needs neither installed weights nor a GPU.
"""
from __future__ import annotations

from runtime_resources import external_python_popen, external_python_run

import argparse
import atexit
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import threading
import uuid

from beautygrpo_worker import (load_installation, inference_identity, execution_profile,
                               selected_offload, OFFLOAD_ENV, sha, write_json,
                               verify_installation, _read_installation_config,
                               _verification_path, prepare_prompt_embeddings)
from portrait_regions_identity import regions_code_identity
from cache_store import exclusive_file_lock

SELECTION_SCHEMA = 'beautygrpo_model_selection.v1'
PREPARE_SCHEMA = 'beautygrpo_layer_prepare.v1'
GENERATED_CACHE_SCHEMA = 'beautygrpo_generated_cache.v2'
REGIONS_CACHE_SCHEMA = 'portrait_regions_cache.v2'

# The desktop process can serve several portrait requests during one session.
# Keep the configured CUDA runtime alive between requests so the expensive
# Python/Torch import and BeautyGRPO pipeline construction happen once.  The
# service is deliberately process-isolated from the desktop and is killed on
# cancellation, model identity changes, or interpreter exit.
_BEAUTY_SERVICE = None
_BEAUTY_SERVICE_LOCK = threading.Lock()


def selection_path():
    from current_model_registry import settings_directory
    return settings_directory()/'beautygrpo-selection.json'


def installation_path(options=None):
    options = options or {}
    selected = options.get('model_config') or os.environ.get('COSPLAY_BEAUTYGRPO_CONFIG')
    from current_model_registry import model_directory
    portable = model_directory()/'beautygrpo.json'
    # The external/source interpreter must follow its explicitly selected
    # standalone package, just as the frozen GUI and installed worker do.
    # A deliberate models-root override still takes precedence.
    package_root = os.environ.get('COSPLAY_PACKAGE_ROOT')
    if package_root and not os.environ.get('COSPLAY_MODELS_ROOT'):
        portable = Path(package_root).resolve()/'model/beautygrpo.json'
    if not selected and selection_path().is_file():
        selection = json.loads(selection_path().read_text(encoding='utf-8-sig'))
        if not isinstance(selection,dict) or selection.get('schema') != SELECTION_SCHEMA:
            raise ValueError('Saved BeautyGRPO model selection has an invalid schema')
        selected = selection.get('model_config')
    if not selected and portable.is_file(): selected = str(portable)
    if not isinstance(selected,str) or not selected or not Path(selected).is_absolute():
        raise ValueError('Configure the local BeautyGRPO installation with COSPLAY_BEAUTYGRPO_CONFIG '
                         'or CosplayWorkbench/config/beautygrpo-selection.json; '
                         'providers/beautygrpo.example.json describes the required paths')
    path = Path(selected).resolve()
    if not path.is_file():
        raise FileNotFoundError('Selected BeautyGRPO installation is unavailable: '+str(path))
    descriptor = json.loads(path.read_text(encoding='utf-8-sig'))
    if descriptor.get('schema') == 'beautygrpo_installation_pointer.v1':
        target = descriptor.get('installation')
        if not isinstance(target, str) or not target:
            raise ValueError('BeautyGRPO installation pointer requires a target')
        package = os.environ.get('COSPLAY_PACKAGE_ROOT')
        if package:
            target = target.replace('${PACKAGE_ROOT}', package)
        target = Path(target).expanduser()
        path = (target if target.is_absolute() else path.parent/target).resolve()
        if not path.is_file():
            raise FileNotFoundError('Selected BeautyGRPO installation is unavailable: '+str(path))
    return path


def validate_options(options):
    options = copy.deepcopy(options or {})
    if not isinstance(options,dict) or set(options)-{'model_config','max_faces'}:
        raise ValueError('BeautyGRPO preparation supports model_config and max_faces')
    maximum = options.get('max_faces',8)
    if type(maximum) is not int or not 1 <= maximum <= 32:
        raise ValueError('BeautyGRPO max_faces must lie in [1,32]')
    options['max_faces'] = maximum
    return options


def stable_retouch_identity(run,root):
    from native_source_baseline import retouch_identity
    retouch = retouch_identity(run,root=root)
    return {key:retouch[key] for key in ('delta_sha256','source_sha256','working_domain')} if retouch else None


def _source_display_preview(source):
    """Encode aligned bands through the unchanged preview recipe."""
    from raw_support import linear_preview
    from runtime_resources import ordered_native_map
    from PIL import Image
    height, width = source.shape[:2]
    if height <= 1024:
        return linear_preview(source)
    image = Image.new('RGB', (width, height))
    def encode_band(top):
        return top, linear_preview(source[top:top+1024])
    for top, band in ordered_native_map(encode_band, range(0, height, 1024), workers=2):
        image.paste(band, (0, top))
        if top == 0:
            image.info.update(band.info)
    return image


def prepare_source(run,root,directory):
    """Decode the immutable source plus native repair, then apply its D once."""
    from native_source_baseline import open_source_for_run
    from display_transform import (source_transform_for_run,source_transform_for_precision,
                                   prepare_source_display_inplace)
    source_path = Path(run['source']['path'])
    source_path = (source_path if source_path.is_absolute() else Path(root)/source_path).resolve()
    if sha(source_path) != run['source']['sha256']:
        raise ValueError('Original source changed before portrait preparation')
    source,precision = open_source_for_run(run,root=root)
    options = run.get('options') or {}
    plan = (options.get('appearance_revision') or {}).get('design_reconstruction_plan') or {}
    baseline = plan.get('source_baseline') or options.get('source_only_plan')
    transform = (source_transform_for_run(run) if baseline
                 else source_transform_for_precision(precision['source_precision']))
    if baseline and (baseline.get('source_precision') != precision['source_precision']
                     or baseline.get('decode_version') != precision['decode_version']):
        raise ValueError('Decoded source precision differs from the workbench display baseline')
    prepare_source_display_inplace(source,transform)
    model_input = Path(directory)/'source-display.png'
    _source_display_preview(source).save(model_input,compress_level=1)
    record = {'schema':PREPARE_SCHEMA,'working_domain':'display_linear_sRGB',
              'source_path':str(source_path),'source_sha256':run['source']['sha256'],
              'native_size':list(source.shape[1::-1]),'source_precision':precision,'source_display':transform,
              'source_display_consumptions':1,'native_source_retouch':stable_retouch_identity(run,root),
              'model_input':str(model_input.resolve()),'model_input_sha256':sha(model_input),
              'source_texture_restoration':False,'skin_only_residual':False}
    write_json(Path(directory)/'source-input.json',record)
    return record


def _cache_binding(record,config,options,*,predecode_prefix=False):
    """Require model-input bytes except for the explicit predecode prefix."""
    from runtime_resources import code_path
    from portrait_regions_worker import MODEL_SHA256
    from current_model_registry import load_registry
    registry = load_registry(registry_path=config['face_registry'])
    face_model = Path(registry['face_model_path'])
    regions_code = regions_code_identity(code_path('portrait_regions_worker.py'))
    binding = {'source_sha256':record['source_sha256'],'source_display':record['source_display'],
            'source_precision':record['source_precision'],'native_size':record['native_size'],
            'native_source_retouch':record['native_source_retouch'],
            'model_identity':inference_identity(config),'max_faces':options['max_faces'],
            'regions_checkpoint_sha256':MODEL_SHA256,'face_landmarker_sha256':sha(face_model),
            'regions_algorithm_identity': {'schema': regions_code['schema'],
                                           'sha256': regions_code['algorithm_sha256']},
            'beauty_producer_sha256':sha(code_path('beautygrpo_worker.py')),
            'prepare_producer_sha256':sha(__file__)}
    if not predecode_prefix:
        binding['model_input_sha256'] = record['model_input_sha256']
    return binding


def _copy_layer(result_path,directory,source_path):
    """Own cached pixels under this result; no saved state points into a cache."""
    from portrait_beauty_layer import load_result
    result,path = load_result(result_path)
    layer = copy.deepcopy(result)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True,exist_ok=False)
    for index,face in enumerate(layer['faces']):
        for key,suffix in (('pixels','.npz'),('model_output','-model.png')):
            if key not in face:
                continue
            asset = face[key]
            origin = Path(asset['path'])
            origin = origin if origin.is_absolute() else path.parent/origin
            if sha(origin) != asset['sha256']:
                raise ValueError('Cached generated portrait bytes changed')
            owned = directory/f'face-{index+1:02d}{suffix}'
            try:
                os.link(origin, owned)
            except OSError:
                shutil.copyfile(origin,owned)
            asset['path'] = str(owned)
    layer['source_path'] = str(source_path)
    owned_path = directory/'portrait-beauty-layer.json'
    write_json(owned_path,layer)
    return owned_path,layer


def _cache_result(root,binding):
    payload = json.dumps(binding,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
    key = hashlib.sha256(payload.encode('utf8')).hexdigest()
    directory = Path(root)/'outputs/cache/beautygrpo'/key
    entry = directory/'cache-entry.json'
    if not entry.is_file():
        return directory,None
    record = json.loads(entry.read_text(encoding='utf8'))
    layer = directory/'layer/portrait-beauty-layer.json'
    if record.get('schema') != GENERATED_CACHE_SCHEMA or record.get('binding') != binding:
        raise ValueError('Cached portrait inference binding differs')
    if sha(layer) != record.get('result_sha256'):
        raise ValueError('Cached portrait layer identity changed')
    return directory,layer


def _predecode_binding(run, config, options, root):
    """Build the cache identity available without decoding the source.

    A prepared source baseline already records the decoder version, source
    precision, display transform, and native canvas.  Those values are the
    complete inputs to ``prepare_source`` apart from the model-input bytes;
    the latter are deterministic for this binding and are deliberately left
    out here.  Source-only or legacy runs without that explicit baseline must
    use the normal decode path because their oriented dimensions/precision
    cannot be inferred safely from a path or extension.
    """
    options = options or {}
    from raw_support import RGB_DECODE_VERSION
    run_options = run.get('options') or {}
    appearance = run_options.get('appearance_revision') or {}
    plan = appearance.get('design_reconstruction_plan') or {}
    baseline = plan.get('source_baseline') or run_options.get('source_only_plan') or {}
    schema = baseline.get('schema')
    source_facts = run.get('source') or {}
    # Beauty is intentionally prepared before native scene analysis on a new
    # layered run. RAW source facts already bind the decoder version, native
    # canvas and precision, so a generated cache can be located without
    # decoding the 40 MB source just to recreate ``source-display.png``.
    # Raster inputs stay on the conservative baseline-only path because ICC
    # metadata can affect their display transform.
    facts_only = False
    if schema not in {'appearance_source_baseline.v1', 'source_only_display_plan.v1'}:
        if (source_facts.get('mode') == 'RAW'
                and source_facts.get('decode_version') == RGB_DECODE_VERSION
                and isinstance(source_facts.get('oriented_size'), list)
                and len(source_facts['oriented_size']) == 2):
            facts_only = True
        else:
            return None
    source = run.get('source') or {}
    source_sha256 = source.get('sha256')
    if (not isinstance(source_sha256, str)
            or (not facts_only and baseline.get('source_sha256') != source_sha256)):
        return None
    if not facts_only and baseline.get('decode_version') != RGB_DECODE_VERSION:
        return None
    from display_transform import source_transform_for_run, source_transform_for_precision
    transform = (source_transform_for_run(run) if not facts_only
                 else source_transform_for_precision('raw16_linear'))
    source_precision = baseline.get('source_precision') if not facts_only else 'raw16_linear'
    if source_precision not in {'raw16_linear', 'raster8_srgb'}:
        return None
    native_size = baseline.get('native_size') if not facts_only else source_facts.get('oriented_size')
    if native_size is None and schema == 'appearance_source_baseline.v1':
        assets = baseline.get('assets') or {}
        native_size = (assets.get('foreground_alpha') or {}).get('native_size')
        if native_size is None:
            native_size = (assets.get('foreground_alpha') or {}).get('size')
    if (not isinstance(native_size, list) or len(native_size) != 2
            or any(type(value) is not int or value <= 0 for value in native_size)):
        return None
    from native_source_baseline import retouch_identity
    native_retouch = retouch_identity(run, root=root)
    precision = {'source_precision': source_precision,
                 'working_precision': 'float32_linear',
                 'decode_version': (baseline.get('decode_version')
                                    if not facts_only else RGB_DECODE_VERSION)}
    if native_retouch is not None:
        precision.update(native_source_retouch=native_retouch,
                         native_retouch_consumptions=1)
    record = {'source_sha256': source_sha256, 'source_display': transform,
              'source_precision': precision,
              'native_size': native_size, 'native_source_retouch': native_retouch}
    return _cache_binding(record, config, options, predecode_prefix=True)


def _predecode_cache_result(root, run, config, options):
    """Find one valid generated cache entry without reading the source pixels.

    The generated cache key also contains the SHA of ``source-display.png``.
    We cannot derive that digest without rendering, so this index-like scan
    compares every other binding field.  An ambiguous or malformed directory
    falls back to the existing decode path rather than guessing a cache hit.
    """
    prefix = _predecode_binding(run, config, options, root)
    if prefix is None:
        return None
    source_path = Path((run.get('source') or {}).get('path', ''))
    source_path = (source_path if source_path.is_absolute() else Path(root)/source_path).resolve()
    # Preserve the ordinary path's source binding before reusing generated
    # pixels.  This reads the immutable file digest (usually served by the
    # stat-bound identity cache) but never decodes RAW samples or writes a
    # display preview.
    try:
        if not source_path.is_file() or sha(source_path) != prefix['source_sha256']:
            return None
    except OSError:
        return None
    cache_root = Path(root)/'outputs/cache/beautygrpo'
    if not cache_root.is_dir():
        return None
    matches = []
    for entry in cache_root.glob('*/cache-entry.json'):
        try:
            record = json.loads(entry.read_text(encoding='utf8'))
            binding = record.get('binding')
            if (record.get('schema') != GENERATED_CACHE_SCHEMA
                    or not isinstance(binding, dict)
                    or set(binding) != set(prefix) | {'model_input_sha256'}
                    or any(binding.get(key) != value for key, value in prefix.items())):
                continue
            digest = binding.get('model_input_sha256')
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(character not in '0123456789abcdef' for character in digest.lower())):
                continue
            directory = entry.parent
            layer = directory/'layer/portrait-beauty-layer.json'
            if not layer.is_file() or sha(layer) != record.get('result_sha256'):
                continue
            from portrait_beauty_layer import load_result
            loaded, _origin = load_result(layer)
            if (loaded.get('source_sha256') != prefix['source_sha256']
                    or loaded.get('native_size') != prefix['native_size']):
                continue
            matches.append((directory, layer))
        except (OSError, ValueError, TypeError):
            # A damaged/partial cache must never make preparation fail.  The
            # regular path will rebuild it and publish a complete entry.
            continue
    return matches[0] if len(matches) == 1 else None


def _publish_cache(directory,binding,result_path,source_path):
    from runtime_resources import code_path
    if directory.exists():
        return
    directory.parent.mkdir(parents=True,exist_ok=True)
    # Stage a complete owned layer; partial/cancelled inference never becomes a
    # cache hit. Model files and source photographs are never copied here.
    stage = directory.with_name('.'+directory.name+'-'+uuid.uuid4().hex)
    stage.mkdir()
    path,layer = _copy_layer(result_path,stage/'layer',source_path)
    final_layer_dir = directory/'layer'
    for face in layer['faces']:
        for key in ('pixels','model_output'):
            if key in face:
                face[key]['path'] = str(final_layer_dir/Path(face[key]['path']).name)
    write_json(path,layer)
    regions_code = regions_code_identity(code_path('portrait_regions_worker.py'))
    write_json(stage/'cache-entry.json',{'schema':GENERATED_CACHE_SCHEMA,'binding':binding,
                                       'result_sha256':sha(path),'generated_outputs_only':True,
                                       'producer_provenance': {
                                           'regions_producer_sha256': regions_code['producer_sha256']}})
    try:
        os.replace(stage,directory)
    except OSError:
        if not directory.exists():
            raise


def _run_regions(config,record,root,output,max_faces):
    from runtime_resources import code_path, isolated_python_environment
    from current_model_registry import load_registry
    from portrait_regions_worker import MODEL_SHA256
    output.mkdir()
    # Region parsing is deterministic for one prepared display source and one
    # pinned parser installation.  Keep the generated masks in a project
    # cache so a retry/revision does not pay for MediaPipe + SegFace again.
    # The cache is read-only on a hit; the request-owned result JSON is still
    # written below so downstream workers retain their usual receipt path.
    registry = load_registry(registry_path=config['face_registry'])
    face_model = Path(registry['face_model_path']).resolve()
    producer = Path(code_path('portrait_regions_worker.py')).resolve()
    regions_code = regions_code_identity(producer)
    configured_cache_root = os.environ.get('COSPLAY_BEAUTY_CACHE_ROOT', '').strip()
    if configured_cache_root:
        cache_base = Path(configured_cache_root).expanduser().resolve()
    else:
        d_cache = Path('D:/CosplayStudio/outputs/cache')
        cache_base = d_cache if Path('D:/').exists() else Path(root).resolve()/'outputs/cache'
    cache_root = cache_base/'portrait-regions'
    binding = {
        'schema': REGIONS_CACHE_SCHEMA,
        'source_sha256': record['model_input_sha256'],
        'native_size': list(record['native_size']),
        'max_faces': int(max_faces),
        'regions_checkpoint_sha256': MODEL_SHA256,
        'face_landmarker_sha256': sha(face_model),
        'regions_algorithm_identity': {'schema': regions_code['schema'],
                                      'sha256': regions_code['algorithm_sha256']},
        'regions_python': str(Path(config['regions_python']).resolve()),
        'regions_model_root': str(Path(config['regions_model_root']).resolve()),
        'cache_root': str(cache_base),
    }
    cache_payload = json.dumps(binding, sort_keys=True, separators=(',', ':'),
                               ensure_ascii=False, allow_nan=False)
    cache_key = hashlib.sha256(cache_payload.encode('utf8')).hexdigest()
    cache_directory = cache_root/cache_key
    cache_entry = cache_directory/'cache-entry.json'
    cache_result = cache_directory/'result.json'

    with exclusive_file_lock(cache_directory.with_name(cache_key+'.single-flight.lock')):
        def _valid_cached_result():
            try:
                entry = json.loads(cache_entry.read_text(encoding='utf8'))
                if (entry.get('schema') != REGIONS_CACHE_SCHEMA
                        or entry.get('binding') != binding
                        or entry.get('result_sha256') != sha(cache_result)):
                    return None
                result = json.loads(cache_result.read_text(encoding='utf8'))
                if (result.get('schema_version') != 'portrait_regions.v1'
                        or result.get('source_sha256') != binding['source_sha256']
                        or result.get('source_native_size') != binding['native_size']):
                    return None
                # The worker consumes only NPZ paths; keep the complete region
                # result and review assets immutable in the cache.  Reject any
                # path escaping the cache directory before handing it to a model.
                cache_base = cache_directory.resolve()
                for region in result.get('regions', []):
                    path = Path(region.get('npz_path', '')).resolve()
                    if (not path.is_file()
                            or os.path.commonpath((str(cache_base), str(path))) != str(cache_base)):
                        return None
                return result
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                return None

        cached = _valid_cached_result()
        if cached is not None:
            output.mkdir(parents=True, exist_ok=True)
            # Bind the review/source path to this request while retaining shared
            # immutable NPZ/mask assets.  This avoids copying tens of MiB per run.
            cached['source_path'] = str(Path(record['model_input']).resolve())
            cached['cache'] = {'schema': REGIONS_CACHE_SCHEMA, 'cache_hit': True,
                               'cache_key': cache_key, 'cache_root': str(cache_base),
                               'binding': binding,
                               'producer_provenance': {
                                   'regions_producer_sha256': cached.get('worker_sha256'),
                                   'requested_regions_producer_sha256': regions_code['producer_sha256']}}
            write_json(output/'result.json', cached)
            (output.parent/'regions.log').write_text(
                json.dumps({'cache_hit': True, 'cache_key': cache_key}, ensure_ascii=False) + '\n',
                encoding='utf8')
            return output/'result.json'

        command = [config['regions_python'],'-B','-X','utf8',str(code_path('portrait_regions_worker.py')),
                   '--input',record['model_input'],'--output',str(output),
                   '--model-root',config['regions_model_root'],'--registry',config['face_registry'],
                   '--device','cuda','--max-faces',str(max_faces)]
        environment = isolated_python_environment({'PYTHONDONTWRITEBYTECODE':'1',
                       'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'})
        log = output.parent/'regions.log'
        with log.open('wb') as stream:
            process = external_python_run(command,cwd=str(root),env=environment,stdout=stream,stderr=subprocess.STDOUT,
                                     creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if process.returncode:
            raise RuntimeError('Automatic portrait region parsing failed: '+log.read_text('utf8',errors='replace')[-2200:])

        # Publish only a complete result directory.  A cancellation or partial
        # parser output can therefore never become a future cache hit.  Cache
        # publication is best-effort; a valid miss remains fully supported when a
        # worktree is read-only or another process wins the same key race.
        try:
            result_path = output/'result.json'
            result = json.loads(result_path.read_text(encoding='utf8'))
            result['cache'] = {'schema': REGIONS_CACHE_SCHEMA, 'cache_hit': False,
                               'cache_key': cache_key, 'cache_root': str(cache_base),
                               'binding': binding,
                               'producer_provenance': {
                                   'regions_producer_sha256': result.get('worker_sha256'),
                                   'requested_regions_producer_sha256': regions_code['producer_sha256']}}
            write_json(result_path, result)
            cache_root.mkdir(parents=True, exist_ok=True)
            stage = cache_root/('.'+cache_key+'-'+uuid.uuid4().hex)
            shutil.copytree(output, stage)
            stage_result = stage/'result.json'
            published = json.loads(stage_result.read_text(encoding='utf8'))
            old_prefix = str(output.resolve())
            new_prefix = str(cache_directory.resolve())
            def rewrite(value):
                if isinstance(value, str):
                    return new_prefix + value[len(old_prefix):] if value.startswith(old_prefix) else value
                if isinstance(value, list):
                    return [rewrite(item) for item in value]
                if isinstance(value, dict):
                    return {key: rewrite(item) for key, item in value.items()}
                return value
            write_json(stage_result, rewrite(published))
            write_json(stage/'cache-entry.json', {'schema': REGIONS_CACHE_SCHEMA,
                                                  'binding': binding,
                                                  'producer_provenance': result['cache']['producer_provenance'],
                                                  'result_sha256': sha(stage_result)})
            try:
                os.replace(stage, cache_directory)
            except OSError:
                if not cache_directory.exists():
                    raise
                shutil.rmtree(stage, ignore_errors=True)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            try:
                if 'stage' in locals():
                    shutil.rmtree(stage, ignore_errors=True)
            except OSError:
                pass
        return output/'result.json'


def _owned_cached_result(cached,source_path,output,started,*,predecoded=False,progress=None,cancel=None):
    """Finish and verify owned pixels before declaring zero model demand."""
    from portrait_beauty_layer import load_result, _pixels
    if cancel and cancel():
        raise InterruptedError('Portrait beauty cache preparation cancelled')
    path,layer = _copy_layer(cached,output/'layer',source_path)
    owned,_path = load_result(path)
    if owned != layer:
        raise ValueError('Owned cached portrait layer differs from its receipt')
    for face in owned['faces']:
        asset = face['pixels']; pixel_path = Path(asset['path'])
        stat = pixel_path.stat(); box = face['native_box']
        _pixels(str(pixel_path),stat.st_mtime_ns,stat.st_size,asset['sha256'],
                (box[3]-box[1],box[2]-box[0]))
        if 'model_output' in face and sha(face['model_output']['path']) != face['model_output']['sha256']:
            raise ValueError('Owned cached model output changed')
    if cancel and cancel():
        raise InterruptedError('Portrait beauty cache preparation cancelled')
    result = {'ok':True,'no_change':False,'cache_hit':True,'result':str(path),'layer':layer,
              'generation_inferences_this_attempt':0,'seconds':time.perf_counter()-started,
              'runtime':{'cache_only':True,'additional_model_cpu_bytes':0,'service_requests':0}}
    if predecoded:
        result['predecode_cache_hit'] = True
    write_json(output/'result.json',result)
    if progress:
        progress('beauty_complete','已复用同一原片和算法的人像美化结果。',1.)
    return result


def _continued_source(run,root,output,config,options):
    """Accept only the current cache probe's complete, unchanged CPU input."""
    if not output.exists() or not any(output.iterdir()):
        return None
    expected = {'source-display.png','source-input.json','cache-source-preparation.json'}
    if {path.name for path in output.iterdir()} != expected:
        raise FileExistsError('Choose a new portrait preparation directory')
    marker = json.loads((output/'cache-source-preparation.json').read_text(encoding='utf8'))
    record_path = output/'source-input.json'
    record = json.loads(record_path.read_text(encoding='utf8'))
    source_path = Path(run['source']['path'])
    source_path = (source_path if source_path.is_absolute() else root/source_path).resolve()
    from display_transform import source_transform_for_run, source_transform_for_precision
    from raw_support import RGB_DECODE_VERSION
    plan = ((run.get('options') or {}).get('appearance_revision') or {}).get('design_reconstruction_plan') or {}
    baseline = plan.get('source_baseline') or (run.get('options') or {}).get('source_only_plan')
    precision = record.get('source_precision') or {}
    transform = (source_transform_for_run(run) if baseline
                 else source_transform_for_precision(precision.get('source_precision')))
    input_path = output/'source-display.png'
    binding = _cache_binding(record,config,options)
    if (marker.get('schema') != 'beautygrpo_cache_source_preparation.v1'
            or marker.get('run_id') != run.get('run_id')
            or marker.get('source_input_sha256') != sha(record_path)
            or marker.get('binding') != binding
            or record.get('schema') != PREPARE_SCHEMA
            or record.get('working_domain') != 'display_linear_sRGB'
            or record.get('source_texture_restoration') is not False
            or record.get('skin_only_residual') is not False
            or record.get('source_path') != str(source_path)
            or record.get('source_sha256') != run['source']['sha256']
            or sha(source_path) != run['source']['sha256']
            or record.get('native_source_retouch') != stable_retouch_identity(run,root)
            or record.get('source_display') != transform
            or precision.get('decode_version') != RGB_DECODE_VERSION
            or (baseline and precision.get('source_precision') != baseline.get('source_precision'))
            or record.get('source_display_consumptions') != 1
            or record.get('model_input') != str(input_path)
            or record.get('model_input_sha256') != sha(input_path)):
        raise ValueError('Prepared Beauty cache-probe input changed before model dispatch')
    metadata_size = (run.get('source') or {}).get('oriented_size')
    if metadata_size and list(metadata_size) != record.get('native_size'):
        raise ValueError('Prepared Beauty native canvas differs from imported source')
    return record


def prepare_cached_for_workbench(run,root,output,options=None,*,cancel=None,progress=None):
    """Complete an existing generated layer on CPU; a miss never runs models.

    Without a saved baseline, prepare_source runs once and its owned input is
    continued by the same isolated prepare entry only after the cold budget
    fits. Region cache alone cannot eliminate Beauty inference demand.
    """
    if cancel and cancel():
        raise InterruptedError('Portrait beauty cache preparation cancelled')
    root,output = Path(root).resolve(),Path(output).resolve()
    options = effective_options_for_run(run,root,options)
    config = load_installation(installation_path(options))
    started = time.perf_counter()
    import workflow_timing
    with workflow_timing.span('beauty_cache_source_probe', model_request=False, http_request=False):
        record = _continued_source(run,root,output,config,options)
    if record is None:
        with workflow_timing.span('beauty_cache_predecode_lookup', model_request=False, http_request=False):
            predecoded = _predecode_cache_result(root,run,config,options)
        if predecoded:
            output.mkdir(parents=True,exist_ok=True)
            source_path = Path(run['source']['path'])
            source_path = (source_path if source_path.is_absolute() else root/source_path).resolve()
            if cancel and cancel():
                raise InterruptedError('Portrait beauty cache preparation cancelled')
            with workflow_timing.span('beauty_cache_materialize', model_request=False, http_request=False):
                return _owned_cached_result(predecoded[1],source_path,output,started,
                                            predecoded=True,progress=progress,cancel=cancel)
        output.mkdir(parents=True,exist_ok=True)
        with workflow_timing.span('beauty_cache_source_prepare', model_request=False, http_request=False):
            record = prepare_source(run,root,output)
        binding = _cache_binding(record,config,options)
        write_json(output/'cache-source-preparation.json',{
            'schema':'beautygrpo_cache_source_preparation.v1','run_id':run.get('run_id'),
            'source_input_sha256':sha(output/'source-input.json'),'binding':binding})
    if cancel and cancel():
        raise InterruptedError('Portrait beauty cache preparation cancelled')
    with workflow_timing.span('beauty_cache_result_lookup', model_request=False, http_request=False):
        _directory,cached = _cache_result(root,_cache_binding(record,config,options))
    if cached:
        with workflow_timing.span('beauty_cache_materialize', model_request=False, http_request=False):
            return _owned_cached_result(cached,record['source_path'],output,started,progress=progress,cancel=cancel)
    return None


def prepare(run,root,output,options=None,*,progress=None):
    """Isolated runtime entry: detect, generate once, freeze and return a receipt."""
    from beautygrpo_worker import run as infer
    from portrait_beauty_layer import freeze_prepared
    root,output = Path(root).resolve(),Path(output).resolve()
    options = validate_options(options)
    config = load_installation(installation_path(options))
    record = _continued_source(run,root,output,config,options)
    output.mkdir(parents=True,exist_ok=True)
    started = time.perf_counter()
    # A complete source baseline lets us identify an existing generated layer
    # before touching a large RAW.  The fast path still verifies the frozen
    # layer and owns a fresh copy exactly as the ordinary cache path does.
    predecoded = _predecode_cache_result(root, run, config, options)
    if predecoded:
        _directory, cached = predecoded
        source_path = Path(run['source']['path'])
        source_path = (source_path if source_path.is_absolute() else root/source_path).resolve()
        return _owned_cached_result(cached,source_path,output,started,predecoded=True,progress=progress)
    if progress:
        progress('beauty_source_preparation','正在按当前原片显像准备人像美化输入。',.05)
    if record is None:
        record = prepare_source(run,root,output)
    record['model_identity'] = inference_identity(config)
    binding = _cache_binding(record,config,options)
    cache_directory,cached = _cache_result(root,binding)
    with exclusive_file_lock(cache_directory.with_name(cache_directory.name+'.single-flight.lock')):
        cache_directory,cached = _cache_result(root,binding)
        if cached:
            return _owned_cached_result(cached,record['source_path'],output,started,progress=progress)
        if progress:
            progress('beauty_face_regions','正在自动定位多张脸并观察可见脸部和遮挡边界。',.12)
        regions = _run_regions(config,record,root,output/'regions',options['max_faces'])
        worker = infer(record['model_input'],regions,config,output/'model',progress=progress)
        if not worker['inference_completed'] or not worker['faces']:
            result = {'ok':True,'no_change':True,'cache_hit':False,'result':None,'layer':None,
                      'reason':'no_visible_faces_available','generation_inferences_this_attempt':0,
                      'diagnostics':str(output/'model/result.json'),'seconds':time.perf_counter()-started}
            write_json(output/'result.json',result)
            if progress:
                progress('beauty_complete','未找到可编辑的可见人脸，保留当前结果。',1.)
            return result
        if worker['source']['sha256'] != record['model_input_sha256'] or worker['source']['native_size'] != record['native_size']:
            raise ValueError('Generated portrait does not match the prepared original source')
        regions_cache = None
        try:
            regions_receipt = json.loads(Path(regions).read_text(encoding='utf8'))
            candidate_cache = regions_receipt.get('cache')
            if isinstance(candidate_cache, dict) and candidate_cache.get('schema') == REGIONS_CACHE_SCHEMA:
                regions_cache = candidate_cache
        except (OSError, ValueError, TypeError):
            # The worker already validated the region receipt; this optional
            # provenance copy must never turn a valid Beauty result into a failure.
            pass
        record.update(faces=worker['faces'],regions=worker['regions'],model_identity=worker['model_identity'],
                      inference_completed=True,model_evidence={key:worker[key] for key in (
                          'base_verified','lora_verified','versions')} | {key:worker[key] for key in (
                          'execution_profile','execution_identity_sha256',
                          'installation_offload','offload_setup_seconds','model_load_seconds','runtime_import_seconds',
                          'startup_seconds','pipeline_reused','verification_reused','model_load_profile',
                          'prompt_embedding_cache','quantized_transformer_verified',
                          'cuda_memory_accounting') if key in worker},
                      model_result=str(output/'model/result.json'),model_result_sha256=sha(output/'model/result.json'),
                      regions_cache=regions_cache)
        write_json(output/'prepared.json',record)
        if progress:
            progress('beauty_layer_freeze','正在冻结原始生成外观和接缝权重，保留可编辑图层。',.91)
        path,layer = freeze_prepared(record,output/'layer')
        _publish_cache(cache_directory,binding,path,record['source_path'])
        result = {'ok':True,'no_change':False,'cache_hit':False,'result':str(path),'layer':layer,
                  'prepared':str(output/'prepared.json'),'generation_inferences_this_attempt':len(worker['faces']),
                  'seconds':time.perf_counter()-started}
        write_json(output/'result.json',result)
        if progress:
            progress('beauty_complete','人像美化层已生成，可对比原图、生成图与合成效果。',1.)
        return result


def _stop_process(process):
    if process.poll() is not None:
        return
    if os.name == 'nt':
        subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    else:
        import signal
        os.killpg(process.pid,signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _stop_beauty_service():
    """Terminate the persistent model process and close its log handle."""
    global _BEAUTY_SERVICE
    service = _BEAUTY_SERVICE
    _BEAUTY_SERVICE = None
    if not service:
        return
    process = service.get('process')
    try:
        if process and process.stdin:
            process.stdin.close()
    except (OSError, ValueError):
        pass
    if process:
        _stop_process(process)
    stream = service.get('stream')
    if stream:
        try:
            stream.close()
        except OSError:
            pass


atexit.register(_stop_beauty_service)


def _service_event(path, event):
    """Append one flushed JSON event to a request-owned progress log."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf8', newline='\n') as stream:
        stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()










def _service_code_identity():
    """Bind the loaded service to small sources using existing stat-bound SHA."""
    from runtime_resources import code_path
    return {'schema': 'beautygrpo_service_code.v1', 'sources': {
        name: sha(code_path(name)) for name in
        ('beautygrpo_worker.py', 'beautygrpo_prepare.py', 'beautygrpo_nunchaku.py')}}


def _start_beauty_service(config, root):
    """Start or reuse a configured-runtime service process."""
    global _BEAUTY_SERVICE
    config_path = str(Path(config['config_path']).resolve())
    config_sha256 = config.get('config_sha256') or sha(config_path)
    profile = execution_profile(config)
    code_identity = _service_code_identity()
    current = _BEAUTY_SERVICE
    if current and current['process'].poll() is None:
        if current.get('owner_pid') != os.getpid() or current.get('busy_request_id'):
            raise BeautyServiceBusyError('The current Beauty service is busy or belongs to another owner')
        if (current['config_path'] == config_path and current['config_sha256'] == config_sha256
                and current.get('execution_profile') == profile
                and current.get('code_identity') == code_identity):
            return current, True
        _stop_beauty_service()
    from runtime_resources import code_path, isolated_python_environment
    log_dir = Path(root).resolve() / 'outputs' / 'runtime'
    log_dir.mkdir(parents=True, exist_ok=True)
    process_log = log_dir / 'beautygrpo-service.log'
    stream = process_log.open('ab')
    command = [config['python'], '-B', '-X', 'utf8', str(code_path('beautygrpo_prepare.py')), '--service']
    environment = isolated_python_environment({'PYTHONDONTWRITEBYTECODE': '1',
                   'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                   OFFLOAD_ENV: profile['offload']})
    process = external_python_popen(command, cwd=str(root), env=environment, stdin=subprocess.PIPE,
                               stdout=stream, stderr=subprocess.STDOUT, text=True, encoding='utf8',
                               bufsize=1, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
                               start_new_session=os.name != 'nt')
    _BEAUTY_SERVICE = {'process': process, 'stream': stream, 'config_path': config_path,
                       'config_sha256': config_sha256, 'process_log': str(process_log),
                       'execution_profile': profile, 'code_identity': code_identity,
                       'started_at': time.time(), 'owner_pid': os.getpid(),
                       'busy_request_id': None, 'pipeline_loaded': False}
    return _BEAUTY_SERVICE, False


def prewarm_for_workbench(root, options=None, *, progress=None, timeout_seconds=900):
    """Load the verified Beauty pipeline before the first user task.

    The warm request uses the same persistent service later used by ordinary
    preparation. It has no image input and never submits inference, so startup
    can pay model/runtime initialization once while the workbench opens. A
    concurrently active Beauty request is left untouched and reported as busy.
    """
    options = validate_options(options)
    config_path = installation_path(options)
    config = load_installation(config_path)
    root = Path(root).resolve()
    with _BEAUTY_SERVICE_LOCK:
        service, reused = _start_beauty_service(config, root)
        if service.get('busy_request_id'):
            return {'state': 'busy_service_untouched', 'service_pid': service['process'].pid}
        if service.get('pipeline_loaded'):
            return {'state': 'pipeline_already_warm', 'service_pid': service['process'].pid,
                    'service_reused': reused}
        event_log = root/'outputs'/'runtime'/'beautygrpo-prewarm-events.log'
        event_log.parent.mkdir(parents=True, exist_ok=True)
        event_log.unlink(missing_ok=True)
        request_id = 'prewarm-' + uuid.uuid4().hex
        service['busy_request_id'] = request_id
        service['busy_kind'] = 'startup_prewarm'
        service['busy_event_log'] = str(event_log)
        request = {'kind': 'prewarm', 'request_id': request_id, 'root': str(root),
                   'event_log': str(event_log), 'options': {**options, 'model_config': str(config_path)}}
        try:
            service['process'].stdin.write(json.dumps(request, ensure_ascii=False)+'\n')
            service['process'].stdin.flush()
        except (OSError, ValueError) as error:
            _stop_beauty_service()
            raise RuntimeError('Beauty startup prewarm could not start: '+str(error)) from error
    started = time.monotonic(); terminal = None; offset = 0
    try:
        while terminal is None and time.monotonic()-started < timeout_seconds:
            if progress:
                progress('beauty_startup_warm', '正在后台预热人像美化模型；界面可先使用其他功能。', .05)
            if service['process'].poll() is not None:
                break
            offset, terminal = _read_service_events(event_log, offset, request_id, progress)
            if terminal is None:
                time.sleep(.1)
    except BaseException:
        with _BEAUTY_SERVICE_LOCK:
            if _BEAUTY_SERVICE is service and service.get('owner_pid') == os.getpid():
                service['busy_request_id'] = None
                service.pop('busy_kind', None)
                service.pop('busy_event_log', None)
                _stop_beauty_service()
        raise
    with _BEAUTY_SERVICE_LOCK:
        service['busy_request_id'] = None
        service.pop('busy_kind', None)
        service.pop('busy_event_log', None)
        if isinstance(terminal, dict) and terminal.get('ok'):
            service['pipeline_loaded'] = True
            return {'state': 'pipeline_warm', 'service_pid': service['process'].pid,
                    'service_reused': reused, 'pipeline_reused': terminal.get('pipeline_reused', False)}
        _stop_beauty_service()
    detail = (terminal or {}).get('error') if isinstance(terminal, dict) else None
    raise RuntimeError('Beauty startup prewarm failed: '+str(detail or 'service exited or timed out'))


def _read_service_events(path, offset, request_id, progress, *, stream=None):
    """Read request-bound JSON events; repeated polling reuses an open stream.

    Startup and terminal reconciliation use one-shot reads. Active preparation
    owns its read handle until completion, cancellation or failure.
    """
    path = Path(path)
    owned = stream is None
    if stream is None:
        if not path.is_file():
            return offset, None
        stream = path.open('r', encoding='utf8', errors='replace')
    try:
        stream.seek(offset)
        while True:
            line = stream.readline()
            if not line:
                break
            offset = stream.tell()
            try:
                event = json.loads(line)
            except (ValueError, TypeError):
                continue
            if event.get('request_id') != request_id:
                continue
            if event.get('type') == 'beautygrpo_progress' and progress:
                progress(event['stage'], event['message'], event['progress'])
            elif event.get('type') == 'beautygrpo_result':
                return offset, event
        return offset, None
    finally:
        if owned:
            stream.close()


def recompose_prepared(prepared_path,output):
    """CPU-only public seam update; preserve the verified real model outputs."""
    from beautygrpo_worker import refresh_composition
    from portrait_beauty_layer import freeze_prepared
    prepared_path,output = Path(prepared_path).resolve(),Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose a new portrait recomposition directory')
    output.mkdir(parents=True,exist_ok=True)
    original = json.loads(prepared_path.read_text(encoding='utf8'))
    record = refresh_composition(original,output/'composition')
    record['recomposition']['parent_prepared'] = {'path':str(prepared_path),'sha256':sha(prepared_path)}
    write_json(output/'prepared.json',record)
    path,layer = freeze_prepared(record,output/'layer')
    result = {'ok':True,'no_change':False,'cache_hit':True,'result':str(path),'layer':layer,
              'prepared':str(output/'prepared.json'),'generation_inferences_this_attempt':0,
              'actual_generated_rgb_unchanged':True}
    write_json(output/'result.json',result)
    return result




def _reconcile_service_terminal(service):
    """Observe an abandoned wait without erasing or stopping an active request."""
    if not service or service.get('owner_pid') != os.getpid():
        return False
    if service.get('busy_kind') == 'startup_prewarm':
        # The startup controller owns completion, including its final state
        # publication. Observing its event log must not release it early.
        return False
    request_id = service.get('busy_request_id')
    if not request_id:
        return True
    # A split request owns its terminal until join/cleanup. Merely observing it
    # must not let another request replace the service beneath its owner.
    if service.get('pending_request') is not None:
        return False
    event_log = service.get('busy_event_log')
    if not event_log:
        return False
    try:
        _offset, terminal = _read_service_events(Path(event_log), 0, request_id, None)
    except (OSError, ValueError, TypeError):
        return False
    if terminal is None:
        return False
    service['busy_request_id'] = None
    service.pop('busy_event_log', None)
    return True


def _memory_calibration_selection(run, root, options=None):
    if run is None or root is None:
        return None
    try:
        from beauty_memory_calibration import select, calibration_path
        if not calibration_path().is_file():
            return None
        config = load_installation(installation_path(options))
        return select(run,root,config,options)
    except (OSError, ValueError, TypeError, KeyError):
        return None


def effective_options_for_run(run,root,options=None):
    """The same single-face calibration scope owns lookup and real dispatch."""
    chosen = validate_options(options)
    calibrated = _memory_calibration_selection(run,root,options)
    if calibrated:
        chosen['max_faces'] = calibrated['effective_max_faces']
    return chosen


def beauty_memory_needed(options=None, *, run=None, root=None):
    """Additional physical/commit demand; a verified idle pipeline needs no reload."""
    from beautygrpo_broker import client
    remote = client()
    if remote is not None:
        observed = remote.call('memory', options=options, run=run)
        return {**observed['demand'], 'broker_warm_verified': observed['warm_verified'],
                'broker_service': observed['service']}
    from runtime_memory import BEAUTY_COLD_PHYSICAL_BYTES, BEAUTY_COLD_COMMIT_BYTES
    cold = {'physical_bytes': BEAUTY_COLD_PHYSICAL_BYTES, 'commit_bytes': BEAUTY_COLD_COMMIT_BYTES}
    calibrated = _memory_calibration_selection(run,root,options)
    if calibrated:
        cold = {**{key:calibrated['budget'][key] for key in ('physical_bytes','commit_bytes')},
            'calibration': calibrated, 'effective_options': {'max_faces':calibrated['effective_max_faces']}}
    if not _BEAUTY_SERVICE_LOCK.acquire(blocking=False):
        return cold
    try:
        service = _BEAUTY_SERVICE
        _reconcile_service_terminal(service)
        if (service and service.get('owner_pid') == os.getpid()
                and service['process'].poll() is None and not service.get('busy_request_id')
                and service.get('pipeline_loaded')
                and service.get('code_identity') == _service_code_identity()
                and service['config_path'] == str(installation_path(options).resolve())
                and service['config_sha256'] == sha(service['config_path'])
                and service.get('execution_profile') == execution_profile(load_installation(service['config_path']))):
            return {'physical_bytes': 0, 'commit_bytes': 0, **(
                {'calibration': calibrated, 'effective_options': {'max_faces':calibrated['effective_max_faces']}}
                if calibrated else {})}
        return cold
    finally:
        _BEAUTY_SERVICE_LOCK.release()


def release_idle_for_stage(*, next_stage_cpu_bytes, next_stage, next_stage_gpu=False, cancel=None):
    """Release an owned idle resident pipeline before another GPU stage.

    Existing native/generation boundaries are known GPU stages. Other GPU
    callers declare next_stage_gpu=True without importing a model runtime.
    Nonresident policies retain their existing CPU-pressure behavior.
    """
    if type(next_stage_gpu) is not bool:
        raise ValueError('next_stage_gpu must be a boolean')
    from beautygrpo_broker import client
    remote = client()
    started = time.monotonic()
    waited = False
    while True:
        if cancel and cancel():
            raise InterruptedError('GPU handoff cancelled while waiting for startup prewarm')
        values = dict(next_stage_cpu_bytes=next_stage_cpu_bytes, next_stage=next_stage,
                      next_stage_gpu=next_stage_gpu)
        receipt = remote.call('park', **values) if remote is not None else _release_owned_idle_for_stage(**values)
        if receipt['state'] != 'startup_prewarm_pending':
            if waited:
                receipt['startup_prewarm_wait_seconds'] = time.monotonic()-started
            return receipt
        if not waited:
            import workflow_timing
            workflow_timing.annotate(waiting_for_startup_prewarm=True)
        waited = True
        time.sleep(.1)


def _release_owned_idle_for_stage(*, next_stage_cpu_bytes, next_stage, next_stage_gpu=False):
    """One nonblocking owner-side observation; never wait inside an RPC."""
    if type(next_stage_gpu) is not bool:
        raise ValueError('next_stage_gpu must be a boolean')
    from runtime_memory import memory_budget, memory_status
    if not _BEAUTY_SERVICE_LOCK.acquire(blocking=False):
        return {'state': 'busy_service_untouched', 'next_stage': next_stage, 'reason':'service_lock_held'}
    try:
        service = _BEAUTY_SERVICE
        if not service:
            return {'state': 'not_running', 'next_stage': next_stage}
        if service.get('owner_pid') != os.getpid():
            return {'state': 'unowned_service_untouched', 'next_stage': next_stage}
        if service.get('busy_kind') == 'startup_prewarm' and service['process'].poll() is None:
            return {'state':'startup_prewarm_pending', 'next_stage':next_stage,
                    'service_pid':service['process'].pid}
        _reconcile_service_terminal(service)
        if service.get('busy_request_id'):
            return {'state': 'busy_service_untouched', 'next_stage': next_stage}
        if service['process'].poll() is not None:
            return {'state': 'not_running', 'next_stage': next_stage}
        code_identity = _service_code_identity()
        if service.get('code_identity') != code_identity:
            receipt = {'state': 'idle_owned_service_stopped', 'stop_reason': 'service_code_changed',
                       'next_stage': next_stage, 'service_pid': service['process'].pid,
                       'model_requests': 0, 'service_code_identity': service.get('code_identity'),
                       'requested_service_code_identity': code_identity}
            _stop_beauty_service()
            return {**receipt, 'memory_after': memory_status(),
                    'process_exit_confirmed': service['process'].poll() is not None}
        budget = memory_budget(next_stage_cpu_bytes)
        resident_gpu_handoff = (
            (service.get('execution_profile') or {}).get('offload') == 'nunchaku_cuda'
            and (next_stage_gpu or next_stage in {'native_analysis', 'generation'}))
        receipt = {'next_stage': next_stage, 'cpu_budget': budget,
                   'service_pid': service['process'].pid, 'model_requests': 0}
        if resident_gpu_handoff:
            receipt.update(stop_reason='resident_gpu_handoff', next_stage_gpu=True,
                           execution_offload='nunchaku_cuda')
        if budget['fits'] and not resident_gpu_handoff:
            return {**receipt, 'state': 'idle_pipeline_retained'}
        # Frozen layer files are already owned by the completed result. Stopping
        # this idle child releases its CPU/GPU allocation; no assets are removed.
        _stop_beauty_service()
        return {**receipt, 'state': 'idle_owned_service_stopped',
                'memory_after': memory_status(), 'process_exit_confirmed': service['process'].poll() is not None}
    finally:
        _BEAUTY_SERVICE_LOCK.release()


class BeautyPreparationRequest:
    """One owned request on the existing service; never stored in a manifest."""
    def __init__(self, service, request, snapshot, reused, started, cancel_request_path=None):
        self.service = service
        self.process = service['process']
        self.request_id = request['request_id']
        self.root = Path(request['root'])
        self.run_id = request['run'].get('run_id')
        self.output = Path(request['output'])
        self.event_log = Path(request['event_log'])
        self.snapshot = snapshot
        self.reused = reused
        self.started = started
        self.finished = None
        self.offset = 0
        self.terminal = None
        self.result = None
        self.closed = False
        self.cleanup = None
        self.cancel_request_path = (str(Path(cancel_request_path).resolve())
                                    if cancel_request_path else None)
        self.effective_options = copy.deepcopy(request.get('options') or {})


class BeautyServiceBusyError(RuntimeError):
    pass


def begin_for_workbench(run,root,output,options=None,*,cancel=None,cancel_request_path=None,wait_for_idle=False):
    """Dispatch once on the native service; join or clean up this token later."""
    options = effective_options_for_run(run,root,options)
    from beautygrpo_broker import client
    remote = client()
    if remote is not None:
        return remote.begin(run, root, output, options, cancel=cancel,
            cancel_request_path=cancel_request_path, wait_for_idle=wait_for_idle)
    while True:
        if cancel and cancel():
            raise InterruptedError('Portrait beauty preparation cancelled while waiting for the service')
        try:
            with _BEAUTY_SERVICE_LOCK:
                return _begin_for_workbench_locked(run, root, output, options, cancel=cancel,
                                                   cancel_request_path=cancel_request_path)
        except BeautyServiceBusyError:
            if not wait_for_idle:
                raise
        time.sleep(.05)


def _begin_for_workbench_locked(run,root,output,options=None,*,cancel=None,cancel_request_path=None):
    started = time.perf_counter()
    options = validate_options(options)
    config_path = installation_path(options)
    config = load_installation(config_path)
    root,output = Path(root).resolve(),Path(output).resolve()
    if cancel and cancel():
        raise InterruptedError('Portrait beauty preparation cancelled')
    # Reconcile before touching a new snapshot or unlinking an old log. A
    # progress callback can fail while the service safely finishes its request.
    if (_BEAUTY_SERVICE and _BEAUTY_SERVICE['process'].poll() is None
            and _BEAUTY_SERVICE.get('busy_request_id')
            and not _reconcile_service_terminal(_BEAUTY_SERVICE)):
        raise BeautyServiceBusyError('The previous Beauty request has no terminal result; service retained')
    output.parent.mkdir(parents=True,exist_ok=True)
    snapshot = output.with_name(output.name+'-input.json')
    request_id = uuid.uuid4().hex
    event_log = output.with_name(output.name+'-service-events.log')
    if event_log.exists():
        event_log.unlink()
    request = {'request_id': request_id, 'run': run, 'root': str(Path(root).resolve()),
               'output': str(output), 'event_log': str(event_log),
               'options': {**options, 'model_config': str(config_path)}}
    write_json(snapshot, request)
    service, reused = _start_beauty_service(config, root)
    process = service['process']
    if service.get('busy_request_id'):
        raise BeautyServiceBusyError('The previous Beauty request has no terminal result; service retained')
    service['busy_request_id'] = request_id
    service['busy_event_log'] = str(event_log)
    pending = BeautyPreparationRequest(service, request, snapshot, reused, started, cancel_request_path)
    service['pending_request'] = pending
    try:
        process.stdin.write(json.dumps(request, ensure_ascii=False) + '\n')
        process.stdin.flush()
    except (OSError, ValueError) as exc:
        _stop_beauty_service()
        raise RuntimeError('BeautyGRPO service is unavailable: '+str(exc)) from exc
    return pending


def _owns_request(pending):
    service = pending.service
    return (service.get('owner_pid') == os.getpid() and _BEAUTY_SERVICE is service
            and service.get('busy_request_id') == pending.request_id
            and service.get('pending_request') is pending)


def _finish_request(pending):
    if (pending.service.get('owner_pid') == os.getpid()
            and pending.service.get('busy_request_id') == pending.request_id
            and pending.service.get('pending_request') is pending):
        pending.service['busy_request_id'] = None
        pending.service.pop('busy_event_log', None)
        pending.service.pop('pending_request', None)
    pending.closed = True
    pending.finished = time.perf_counter()


def _collect_request_result(pending, wait_started):
    completed, output, service = pending.terminal, pending.output, pending.service
    if not completed.get('ok'):
        error = completed.get('error') or 'unknown service error'
        raise RuntimeError('Portrait beauty preparation failed: '+error)
    result_path = output/'result.json'
    if not result_path.is_file():
        raise RuntimeError('BeautyGRPO service returned no result receipt')
    result = json.loads(result_path.read_text(encoding='utf8'))
    if result.get('ok') is not True:
        raise ValueError('Portrait preparation did not return a completed receipt')
    model_result = output/'model/result.json'
    model_evidence = {}
    if model_result.is_file():
        try:
            model_evidence = json.loads(model_result.read_text(encoding='utf8'))
        except (OSError, ValueError):
            model_evidence = {}
    # A copied generated-cache receipt can say inference_completed even when
    # this process never constructed a pipeline. Residency comes only from a
    # live service terminal, never from historical model output metadata.
    if 'pipeline_loaded' in completed:
        service['pipeline_loaded'] = completed['pipeline_loaded'] is True
    result['runtime'] = {'service_schema': 'beautygrpo_persistent_service.v1',
                         'service_reused': pending.reused, 'request_id': pending.request_id,
                         'service_pid': pending.process.pid,
                         'wait_seconds': time.perf_counter()-wait_started,
                         'request_seconds': time.perf_counter()-pending.started,
                         'pipeline_reused': bool(model_evidence.get('pipeline_reused', pending.reused)),
                         'verification_reused': bool(model_evidence.get('verification_reused', pending.reused)),
                         'execution_profile': service['execution_profile'],
                         'service_code_identity': copy.deepcopy(service['code_identity']),
                         'effective_max_faces': (getattr(pending,'effective_options',{}) or {}).get('max_faces'),
                         'memory_admission': copy.deepcopy(getattr(pending,'memory_admission',None)),
                         'process_log': service['process_log'], 'event_log': str(pending.event_log)}
    write_json(result_path, result)
    pending.result = result
    return result


def _request_usage(pending):
    """Count complete receipts exactly; retain uncertainty for killed sampling."""
    if pending.result is not None:
        count = pending.result.get('generation_inferences_this_attempt')
        return {'generation_inferences_this_attempt': count, 'model_submissions': count,
                'model_submission_outcome': 'completed_receipt'}
    generated = len(list((pending.output/'model').glob('face-*/model-output.png')))
    generation_started = False
    observed_faces = set()
    if pending.event_log.is_file():
        for line in pending.event_log.read_text('utf8', errors='replace').splitlines():
            try:
                event = json.loads(line)
            except (ValueError, TypeError):
                continue
            if event.get('request_id') != pending.request_id or event.get('stage') != 'beauty_face_generation':
                continue
            generation_started = True
            # A denoising callback proves the pipeline was invoked; the earlier
            # face-start event alone does not prove submission completed.
            match = re.search(r'人脸\s+(\d+)/\d+：\d+/\d+', str(event.get('message', '')))
            if match:
                observed_faces.add(int(match.group(1)))
    count = None if generation_started or generated else 0
    return {'generation_inferences_this_attempt': count, 'model_submissions': count,
            'observed_model_inferences_minimum': max(generated, len(observed_faces)),
            'model_submission_outcome': 'unknown_after_sampling_started' if count is None else 'not_submitted'}


def cleanup_for_workbench(pending,*,reason='owner_cleanup',stop_completed=False):
    """Retain a terminal receipt or stop only this token's owned process tree."""
    from beautygrpo_broker import BrokerPreparationRequest
    if isinstance(pending, BrokerPreparationRequest):
        return pending.cleanup_remote(reason=reason, stop_completed=stop_completed)
    with _BEAUTY_SERVICE_LOCK:
        if pending.closed:
            if (stop_completed and pending.cleanup is None and _BEAUTY_SERVICE is pending.service
                    and pending.service.get('owner_pid') == os.getpid()
                    and not pending.service.get('busy_request_id')):
                _stop_beauty_service()
                pending.cleanup = {'state': 'completed_service_stopped', 'reason': reason,
                    'request_id': pending.request_id, 'process_exit_confirmed': pending.process.poll() is not None,
                    **_request_usage(pending)}
            return pending.cleanup or {'state': 'already_joined', 'request_id': pending.request_id}
        if not _owns_request(pending):
            return {'state': 'unowned_request_untouched', 'request_id': pending.request_id}
        state, error = 'owned_request_stopped', None
        try:
            _offset, pending.terminal = _read_service_events(pending.event_log, 0, pending.request_id, None)
            if pending.terminal is not None and pending.terminal.get('ok'):
                _collect_request_result(pending, time.perf_counter())
                state = 'completed_result_retained'
            elif pending.terminal is not None:
                state = 'failed_request_stopped'
        except Exception as exc:
            error = str(exc)
        finally:
            # On branch failure/cancellation there is no next local stage to
            # hand off to. Stop the owned service even if its receipt is complete.
            try:
                _stop_beauty_service()
            except Exception as exc:
                error = str(exc)
            finally:
                _finish_request(pending)
        receipt = {'state': state, 'reason': reason, 'request_id': pending.request_id,
                   'request_snapshot': str(pending.snapshot), 'event_log': str(pending.event_log),
                   'result_receipt': str(pending.output/'result.json'),
                   'service_pid': pending.process.pid,
                   'process_exit_confirmed': pending.process.poll() is not None,
                   'request_seconds': pending.finished-pending.started, **_request_usage(pending)}
        if error:
            receipt['error'] = error
        pending.cleanup = receipt
        return receipt


def cancel_pending_for_run(root,run_id,*,cancel_request_path):
    """Existing backend monitor may cancel only its run and marker-bound token."""
    with _BEAUTY_SERVICE_LOCK:
        service = _BEAUTY_SERVICE
        pending = service.get('pending_request') if service else None
        marker = str(Path(cancel_request_path).resolve())
        if (pending is None or pending.root != Path(root).resolve() or pending.run_id != run_id
                or pending.cancel_request_path != marker or not Path(marker).is_file()
                or not _owns_request(pending)):
            return {'state': 'no_owned_matching_request'}
    return cleanup_for_workbench(pending, reason='cancelled')


def join_for_workbench(pending,*,cancel=None,progress=None):
    """Await the dispatched request without holding the lock during remote HTTP."""
    from beautygrpo_broker import BrokerPreparationRequest
    if isinstance(pending, BrokerPreparationRequest):
        return pending.join(cancel=cancel, progress=progress)
    event_stream = None
    poll_sleep = .05
    wait_started = time.perf_counter()
    try:
        while True:
            if cancel and cancel():
                # A CUDA diffusion call cannot be safely interrupted through the
                # JSON protocol. Kill the isolated service and retain its partial
                # output for diagnostics; the next request rebuilds it cleanly.
                cleanup_for_workbench(pending, reason='cancelled')
                raise InterruptedError('Portrait beauty preparation cancelled; saved result retained')
            with _BEAUTY_SERVICE_LOCK:
                if pending.closed:
                    if pending.cleanup:
                        raise InterruptedError('Portrait beauty request stopped by its owner')
                    return pending.result
                if not _owns_request(pending):
                    raise RuntimeError('Beauty request no longer owns its service')
                if event_stream is None and pending.event_log.is_file():
                    event_stream = pending.event_log.open('r', encoding='utf8', errors='replace')
                previous_offset = pending.offset
                pending.offset, pending.terminal = _read_service_events(
                    pending.event_log, pending.offset, pending.request_id, progress, stream=event_stream)
                if pending.terminal is not None:
                    result = _collect_request_result(pending, wait_started)
                    _finish_request(pending)
                    if cancel and cancel():
                        raise InterruptedError('Portrait beauty preparation cancelled; generated layer not applied')
                    return result
                if pending.process.poll() is not None:
                    tail = Path(pending.service['process_log']).read_text('utf8', errors='replace')[-2400:]
                    raise RuntimeError('Portrait beauty service exited unexpectedly: '+tail)
            poll_sleep = (.05 if pending.offset != previous_offset else min(.5, poll_sleep * 1.35))
            time.sleep(poll_sleep)
    except BaseException:
        cleanup_for_workbench(pending, reason='join_failed', stop_completed=True)
        raise
    finally:
        if event_stream is not None:
            event_stream.close()


def prepare_for_workbench(run,root,output,options=None,*,cancel=None,progress=None):
    """Synchronous entry shares the same dispatch, await and owned cleanup."""
    pending = begin_for_workbench(run, root, output, options, cancel=cancel, wait_for_idle=True)
    return join_for_workbench(pending, cancel=cancel, progress=progress)




def service_main():
    """Serve newline-delimited preparation requests in one runtime process."""
    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            request_id = request['request_id']
            event_log = Path(request['event_log'])
            _service_event(event_log, {'type': 'beautygrpo_service_started', 'request_id': request_id,
                                       'pid': os.getpid(), 'timestamp': time.time()})
            def progress(stage, message, amount):
                _service_event(event_log, {'type': 'beautygrpo_progress', 'request_id': request_id,
                                           'stage': stage, 'message': message, 'progress': amount})
            if request.get('kind') == 'prewarm':
                # Startup warming loads the verified pipeline into this same
                # persistent service process. No source, face or image is
                # required, so the first user task reuses the loaded model.
                config = load_installation(Path(request['options']['model_config']))
                from beautygrpo_worker import load_pipeline
                _pipe, evidence = load_pipeline(config, progress=progress)
                _service_event(event_log, {'type': 'beautygrpo_result', 'request_id': request_id,
                                           'ok': True, 'kind': 'prewarm',
                                           'pipeline_loaded': True,
                                           'pipeline_reused': bool(evidence.get('pipeline_reused')),
                                           'runtime': evidence.get('execution_profile')})
            else:
                prepare(request['run'], Path(request['root']), Path(request['output']),
                        request.get('options'), progress=progress)
                from beautygrpo_worker import _PIPELINE_CACHE
                _service_event(event_log, {'type': 'beautygrpo_result', 'request_id': request_id,
                    'ok': True, 'pipeline_loaded': bool(_PIPELINE_CACHE)})
        except BaseException as exc:
            # Keep the process alive for subsequent requests.  The caller has
            # the detailed traceback in the persistent process log and the
            # request's result marker remains machine-readable.
            try:
                _service_event(Path(request.get('event_log', 'service-events.log')),
                               {'type': 'beautygrpo_result', 'request_id': request.get('request_id'),
                                'ok': False, 'error_type': type(exc).__name__, 'error': str(exc)})
            except BaseException:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--request-json',type=Path)
    inputs.add_argument('--recompose-prepared',type=Path)
    inputs.add_argument('--service',action='store_true')
    inputs.add_argument('--verify-installation',type=Path,
                        help='Verify installed weights once and prepare the official fixed-prompt embeddings')
    inputs.add_argument('--prepare-prompt-embeddings',type=Path,
                        help='Explicitly encode/cache the official fixed prompt with the configured CUDA runtime')
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if args.service:
        service_main()
        return
    if args.prepare_prompt_embeddings:
        if args.output is not None:
            parser.error('Prompt embeddings are installation-bound; --output is not supported')
        result = prepare_prompt_embeddings(args.prepare_prompt_embeddings)
        print(json.dumps({'ok': True, 'schema': result['schema'],
                          'receipt': result['receipt_path'],
                          'prepare_reused': result['prepare_reused'],
                          'binding_sha256': result['binding_sha256'],
                          'artifact_sha256': result['artifact']['sha256'],
                          'generated_images': 0}, ensure_ascii=False), flush=True)
        return
    if args.verify_installation:
        # This command should run under the configured BeautyGRPO interpreter.
        # The sole installation entry verifies weights and prepares its prompt
        # resource once. Normal portrait requests never construct encoders.
        config = _read_installation_config(args.verify_installation)
        receipt_path = _verification_path(config, config['config_path'])
        result = verify_installation(args.verify_installation, receipt_path=receipt_path)
        if args.output and args.output.resolve() != receipt_path:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, result)
        prompt = prepare_prompt_embeddings(args.verify_installation)
        print(json.dumps({'ok': True, 'schema': result['schema'],
                          'receipt': str(Path(receipt_path).resolve()),
                          'verified_file_count': result['base']['file_count'],
                          'lora_tensor_count': result['lora']['tensor_count'],
                          'prompt_embeddings_receipt': prompt['receipt_path'],
                          'prompt_embeddings_ready': True,
                          'prompt_embeddings_prepare_reused': prompt['prepare_reused'],
                          'generated_images': 0},
                         ensure_ascii=False), flush=True)
        return
    if args.output is None:
        parser.error('--output is required for a preparation request')
    if args.recompose_prepared:
        result = recompose_prepared(args.recompose_prepared,args.output)
        print(json.dumps({key:value for key,value in result.items() if key!='layer'},ensure_ascii=False),flush=True)
        return
    if not args.root:
        parser.error('Automatic preparation requires --root')
    request = json.loads(args.request_json.read_text(encoding='utf8'))
    def progress(stage,message,amount):
        print(json.dumps({'type':'beautygrpo_progress','stage':stage,'message':message,'progress':amount},ensure_ascii=False),flush=True)
    result = prepare(request['run'],args.root,args.output,request.get('options'),progress=progress)
    print(json.dumps({key:value for key,value in result.items() if key!='layer'},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
