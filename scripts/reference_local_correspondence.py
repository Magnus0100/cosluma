"""Content-bound local person correspondence; heavy inference is a worker job.

Maps locate reference evidence on the source proxy. They do not warp source
geometry, paste reference texture, or infer correspondence from background H.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image

from runtime_resources import code_path

SCHEMA = 'reference_local_correspondence.v1'
DINO3_L_PROFILE = 'dinov3_vitl16'
DINO3_L_DIRECTORY = Path(os.environ.get('COSPLAY_MODELS_ROOT', str(Path.home() / 'CosplayWorkbench/models'))) / 'dinov3-vitl16'
DINO3_L_REVISION = 'ea8dc2863c51be0a264bab82070e3e8836b02d51'
PREPROCESS = {
    'version': 'dinov3_aspect_preserving_patch16_no_crop_v1',
    'maximum_edge': 720, 'patch_size': 16, 'resize': 'PIL_BICUBIC',
    'mean': [.485, .456, .406], 'std': [.229, .224, .225],
    'coordinates': 'pixel_centres; align_corners_false',
    'material_zoom_maximum_edge': 448, 'material_zoom_if_tokens_below': 48,
    'material_crop_context_fraction': .15, 'material_crop_minimum_pixels': 96,
}
MATCHING = {
    'minimum_cosine': .70, 'minimum_top2_margin': .02,
    'minimum_token_material_fraction': .65, 'neighbour_radius_patches': 4.,
    'minimum_neighbours': 4, 'maximum_leave_one_out_error_patches': 1.25,
    'minimum_affine_scale': .25, 'maximum_affine_scale': 4.,
    'maximum_affine_condition': 4., 'maximum_triangle_edge_patches': 4.,
    'minimum_triangle_area_patches_squared': .15,
    'minimum_face_triangle_scale': .35, 'maximum_face_triangle_scale': 3.,
    'maximum_face_triangle_condition': 3.,
    'maximum_face_heldout_error_fraction': .06,
    'minimum_confidence': .15,
}
TOKEN_LAYOUT = {'class_tokens': 1, 'register_tokens': 4, 'embedding_dimension': 1024,
                'model_type': 'dinov3_vit', 'inference_dtype': 'float32'}


def algorithm_configuration(algorithm_profile=DINO3_L_PROFILE, *, model_directory=None):
    """The current ViT-L adapter rejects every older model/profile explicitly."""
    if algorithm_profile != DINO3_L_PROFILE:
        raise ValueError('Unsupported local correspondence algorithm profile: '+str(algorithm_profile))
    return {'algorithm_profile': algorithm_profile,
            'model_directory': Path(model_directory) if model_directory is not None else DINO3_L_DIRECTORY,
            'repository': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
            'revision': DINO3_L_REVISION,
            'expected_config': {'model_type': 'dinov3_vit', 'patch_size': 16,
                                'hidden_size': 1024, 'num_register_tokens': 4,
                                'num_hidden_layers': 24, 'num_attention_heads': 16},
            'preprocess': dict(PREPROCESS), 'matching': dict(MATCHING),
            'token_layout': dict(TOKEN_LAYOUT)}


def algorithm_profile_for_run(run):
    value = (run.get('options') or {}).get('algorithm_profile', DINO3_L_PROFILE)
    algorithm_configuration(value)
    return value


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def array_sha(values):
    values = np.ascontiguousarray(values)
    return hashlib.sha256(json.dumps({'shape': list(values.shape), 'dtype': str(values.dtype)},
        sort_keys=True).encode()+values.tobytes()).hexdigest()


def _face_observation(record):
    if record is None:
        return None
    if not isinstance(record, dict):
        raise ValueError('Face observations must be MediaPipe result dictionaries')
    return {key: record.get(key, [] if key in ('landmarks', 'protected_regions') else None)
            for key in ('method', 'size', 'landmarks', 'protected_regions', 'model_sha256', 'mediapipe_version')}


def local_correspondence_identity(root, algorithm_profile=DINO3_L_PROFILE, *, registry_path=None):
    """Pure file identity: no GPU libraries, models, image decoding or workers."""
    root = Path(root)
    from current_model_registry import load_registry
    registry = load_registry(root, registry_path=registry_path)
    adapter = algorithm_configuration(algorithm_profile, model_directory=registry['dinov3_model_root'])
    directory = adapter['model_directory']
    manifest_path = directory/'model-manifest.json'
    if not manifest_path.is_file():
        raise RuntimeError(algorithm_profile+' model is not installed: '+str(directory))
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    if (manifest.get('repository') != adapter['repository']
            or not manifest.get('revision') or not manifest.get('license')):
        raise ValueError('Local correspondence installation lacks repository/revision/license provenance')
    if manifest['revision'] != adapter['revision']:
        raise ValueError('DINOv3 installation differs from the pinned official revision')
    files = {}
    required = ('config.json', 'preprocessor_config.json', 'model.safetensors', 'README.md', 'LICENSE.md')
    for name in required:
        path = directory/name
        if not path.is_file():
            raise RuntimeError('Local correspondence required file is missing: '+str(path))
        files[name] = file_sha(path)
        if files[name] != manifest.get('files_sha256', {}).get(name):
            raise ValueError('Local correspondence model bytes differ from recorded installation: '+name)
    config = json.loads((directory/'config.json').read_text(encoding='utf8'))
    if any(config.get(key) != value for key, value in adapter['expected_config'].items()):
        raise ValueError('DINOv3 model token/config layout differs from the current ViT-L adapter')
    worker = code_path('reference_local_correspondence_worker.py')
    runtime = Path(registry['python'])
    if not worker.is_file() or not runtime.is_file():
        raise RuntimeError(algorithm_profile+' correspondence worker/runtime is unavailable')
    packages = {}
    site_packages = runtime.parent.parent/'Lib/site-packages'
    for name in ('torch', 'transformers', 'numpy', 'scipy', 'pillow', 'safetensors'):
        metadata = [path for path in site_packages.glob(name+'-*.dist-info/METADATA') if path.is_file()]
        if len(metadata) != 1:
            raise RuntimeError('Correspondence runtime has missing/ambiguous package metadata: '+name)
        version = next(line[9:] for line in metadata[0].read_text(encoding='utf8').splitlines() if line.startswith('Version: '))
        packages[name] = {'version': version, 'metadata_sha256': file_sha(metadata[0])}
    from raw_support import RGB_DECODE_VERSION
    identity = {'schema': SCHEMA, 'model': {'repository': manifest['repository'],
        'revision': manifest['revision'], 'license': manifest['license'], 'files_sha256': files},
        'module_sha256': file_sha(__file__), 'worker_sha256': file_sha(worker),
        'runtime_packages': packages,
        'preprocess': adapter['preprocess'], 'matching': adapter['matching'], 'decode_version': RGB_DECODE_VERSION}
    identity.update(algorithm_profile=algorithm_profile, token_layout=adapter['token_layout'],
                    quality_status='current_model_partial_evidence_not_visual_acceptance')
    if manifest.get('installation_source') is not None:
        identity['model']['installation_source'] = manifest['installation_source']
    return identity


def _cached_rgb(root, record):
    """Share RGB across model profiles only when the display decode agrees."""
    from raw_support import RGB_DECODE_VERSION
    root = Path(root)
    candidates = []
    observations = []
    for path in sorted((root/'outputs/cache/local-analysis'/record['sha256'][:16]).glob('*/analysis.json')):
        data = json.loads(path.read_text(encoding='utf8'))
        image_path = path.parent/'source-srgb.png'
        decode = data.get('analysis_identity') or {}
        if (data.get('source', {}).get('sha256') == record['sha256']
                and decode.get('decode_version') == RGB_DECODE_VERSION
                and decode.get('input_proxy') == 'aspect_preserving_2048_lanczos_srgb'
                and image_path.is_file()):
            size = data['source']['oriented_size']
            if record.get('oriented_size') is not None and size != record['oriented_size']:
                raise ValueError('Cached RGB canvas differs from the bound original image')
            from PIL import Image
            with Image.open(image_path) as image:
                if image.mode != 'RGB':
                    raise ValueError('Cached observation is not decoded RGB')
                # PNG bytes differ when LittleCMS writes a fresh ICC creation
                # date. Compare the actual worker input and all other ICC bytes;
                # the chosen file still gets its full byte hash in the binding.
                profile = bytearray(image.info.get('icc_profile', b''))
                if len(profile) >= 128:
                    profile[24:36] = bytes(12)
                observations.append((tuple(size), image.size,
                    array_sha(np.asarray(image)), hashlib.sha256(profile).hexdigest()))
            candidates.append((image_path, data['source']['oriented_size'], file_sha(image_path)))
    if candidates:
        if len(set(observations)) != 1:
            raise ValueError('Ambiguous cached source/reference RGB observations')
        return candidates[0]
    return None


def validate_local_correspondence(receipt, *, binding=None):
    """Load bound map bytes and reject stale, nonfinite or out-of-canvas evidence."""
    if receipt.get('schema') != SCHEMA:
        raise ValueError('Unknown local correspondence schema')
    algorithm_configuration((receipt.get('identity') or {}).get('algorithm_profile'))
    if binding is not None and receipt.get('binding') != binding:
        raise ValueError('Local correspondence input binding mismatch')
    if (receipt['binding'].get('producer_identity_sha256') is not None
            and canonical_sha(receipt.get('identity')) != receipt['binding']['producer_identity_sha256']):
        raise ValueError('Local correspondence producer provenance changed')
    path = Path(receipt['asset']['path'])
    if file_sha(path) != receipt['asset']['sha256']:
        raise ValueError('Local correspondence asset bytes changed')
    with np.load(path, allow_pickle=False) as data:
        maps = {name: data[name].copy() for name in ('ref_x', 'ref_y', 'confidence', 'valid')}
    shape = tuple(reversed(receipt['binding']['proxy_size']))
    if (any(value.shape != shape for value in maps.values()) or maps['valid'].dtype != bool
            or not all(np.isfinite(maps[key]).all() for key in ('ref_x', 'ref_y', 'confidence'))):
        raise ValueError('Invalid local correspondence map shape/domain')
    valid, confidence = maps['valid'], maps['confidence']
    rw, rh = receipt['reference_size']
    threshold = (receipt.get('identity') or {}).get('matching', MATCHING).get('minimum_confidence')
    if not isinstance(threshold, (int, float)) or not np.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError('Invalid frozen local correspondence confidence threshold')
    if (confidence.min() < 0 or confidence.max() > 1
            or np.any(valid & (confidence < threshold))
            or np.any(valid & ((maps['ref_x'] < 0) | (maps['ref_x'] > rw-1)
                              | (maps['ref_y'] < 0) | (maps['ref_y'] > rh-1)))
            or np.any(confidence[~valid] != 0) or np.any(maps['ref_x'][~valid] != -1)
            or np.any(maps['ref_y'][~valid] != -1)):
        raise ValueError('Invalid local correspondence confidence/coordinates')
    return maps


def prepare_local_correspondence(root, run, size, source_labels, reference_labels,
                                 source_faces=None, reference_faces=None, *, algorithm_profile=None):
    """Prepare original-reference coordinates on the original-source proxy.

Source labels describe the supplied proxy, or another full source canvas; ref
labels describe a full reference canvas. Optional face records are the existing
MediaPipe result dictionaries (landmarks and their own size). No manual masks
or sample-specific settings are consumed.
"""
    started = time.perf_counter()
    root = Path(root).resolve()
    size = tuple(size)
    if len(size) != 2 or any(not isinstance(n, (int, np.integer)) or n <= 0 for n in size):
        raise ValueError('Local correspondence requires a positive source proxy size')
    labels = {'source': np.asarray(source_labels), 'reference': np.asarray(reference_labels)}
    for role, values in labels.items():
        if values.ndim != 2 or values.size == 0 or not np.issubdtype(values.dtype, np.integer) or values.min() < 0:
            raise ValueError('Local correspondence requires integer full-canvas semantic labels: '+role)
    algorithm_profile = algorithm_profile if algorithm_profile is not None else algorithm_profile_for_run(run)
    from current_model_registry import load_registry
    registry_path = (run.get('options') or {}).get('model_registry')
    registry = load_registry(root, registry_path=registry_path)
    adapter = algorithm_configuration(algorithm_profile, model_directory=registry['dinov3_model_root'])
    identity = local_correspondence_identity(root, algorithm_profile=algorithm_profile, registry_path=registry_path)
    inputs = {}
    for role in ('source', 'reference'):
        path = Path(run[role]['path'])
        if file_sha(path) != run[role]['sha256']:
            raise ValueError('Local correspondence image content binding changed: '+role)
        cached = _cached_rgb(root, run[role])
        inputs[role] = {'original': path, 'cached': cached}
    source_faces, reference_faces = _face_observation(source_faces), _face_observation(reference_faces)
    faces = {'source': source_faces, 'reference': reference_faces}
    binding = {'source_sha256': run['source']['sha256'], 'reference_sha256': run['reference']['sha256'],
        'source_labels_sha256': array_sha(labels['source']), 'reference_labels_sha256': array_sha(labels['reference']),
        'proxy_size': list(size), 'source_faces_sha256': canonical_sha(source_faces),
        'reference_faces_sha256': canonical_sha(reference_faces), 'producer_identity_sha256': canonical_sha(identity),
        'cached_rgb_sha256': {role: inputs[role]['cached'][2] if inputs[role]['cached'] else None
                             for role in inputs}}
    key = canonical_sha(binding)[:24]
    cache = root/'outputs/cache/reference-local-correspondence'/algorithm_profile
    directory = cache/key
    directory.mkdir(parents=True, exist_ok=True)
    receipt_path = directory/'receipt.json'
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding='utf8'))
        validate_local_correspondence(receipt, binding=binding)
        return {'receipt': receipt, 'receipt_path': str(receipt_path), 'asset': receipt['asset'],
                'cache_hit': True, 'elapsed_seconds': round(time.perf_counter()-started, 3)}
    request = {'schema': SCHEMA, 'binding': binding, 'identity': identity,
               'faces': faces, 'images': {}, 'model_directory': str(adapter['model_directory']),
               'output_directory': str(directory)}
    for role, item in inputs.items():
        cached = item['cached']
        if cached:
            image_path, original_size, _ = cached
        else:
            from raw_support import open_rgb
            image = open_rgb(item['original'])
            original_size = list(image.size)
            image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
            image_path = directory/(role+'-srgb.png')
            image.save(image_path)
        label_path = directory/(role+'-labels.npz')
        np.savez_compressed(label_path, labels=labels[role])
        request['images'][role] = {'path': str(image_path), 'sha256': file_sha(image_path),
            'original_size': list(original_size), 'labels_path': str(label_path), 'labels_sha256': file_sha(label_path)}
    request_path = directory/'request.json'
    request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    worker = code_path('reference_local_correspondence_worker.py')
    runtime = Path(registry['python'])
    log_path = directory/'worker.log'
    with log_path.open('w', encoding='utf8') as stream:
        process = subprocess.run([str(runtime), '-B', '-X', 'utf8', str(worker), '--request', str(request_path)],
            stdout=stream, stderr=stream, timeout=900,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
            env={**os.environ, 'PYTHONIOENCODING': 'utf8'})
    if process.returncode:
        raise RuntimeError(algorithm_profile+' local correspondence failed; inspect '+str(log_path))
    receipt = json.loads(receipt_path.read_text(encoding='utf8'))
    validate_local_correspondence(receipt, binding=binding)
    return {'receipt': receipt, 'receipt_path': str(receipt_path), 'asset': receipt['asset'],
            'cache_hit': False, 'elapsed_seconds': round(time.perf_counter()-started, 3)}
