"""Sole current SAM3.1 static detector; masks are never alpha or replacement RGB."""
from __future__ import annotations

import hashlib
import copy
import json
import os
from pathlib import Path

IMAGE_PROFILE = 'sam31_static_detector_v1'
CODE_COMMIT = '0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc'
IMAGE_SHA = '0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6'
CHECKPOINT = 'sam3.1/sam3.1_multiplex.pt'
CHECKPOINT_BYTES = 3502755717
WEIGHT_REVISION = 'daa63191845a41281374e725f4c9e51c7a824460'
PUBLIC_METADATA_SHA = 'c832c851a291228f41cd0b22f4bfdf721ca05372bb50c8a0676a121aaf85f789'
CODE_CONTENT_SHA = 'd47d049f88e02a33da10605c8d85a7af13675ddc4a772bfd7802787afbf38282'
DETECTOR_TENSORS = 1166
DIRECTORY = Path(os.environ.get('COSPLAY_MODELS_ROOT', str(Path.home() / 'CosplayWorkbench/models'))) / 'sam3.1'
CONCEPTS = ('mirror', 'floor', 'shoes', 'feet', 'person', 'bag')
RUNTIME_PACKAGES = {'torch': '2.10.0+cu128', 'torchvision': '0.25.0+cu128',
    'numpy': '1.26.4', 'opencv-python': '4.11.0.86', 'sam3': '0.1.0',
    'triton-windows': '3.6.0.post26', 'ftfy': '6.1.1', 'setuptools': '80.9.0'}
RUNTIME_FILES = {
    'runtime/pyvenv.cfg': '4ae9e9749a4146aa07a6655c9b5d374247748a7299658f67cc6abe03a972b3e2',
    'runtime/Lib/site-packages/shared-model-runtime.pth': 'f63c773424140ba30158f8be898e1e363b5765b024c7c6de39a34996fa89375a'}
PREPROCESS = {'processor': 'official_Sam3Processor', 'resolution': 1008,
              'resize': 'square_1008_torchvision_v2', 'normalize_mean': [.5]*3,
              'normalize_std': [.5]*3, 'mask_logit_probability_threshold': .5,
              'instance_confidence_threshold': .5, 'precision': 'float32_with_bfloat16_autocast',
              'compile': False, 'detector': 'Sam3MultiplexDetector_SAM3VLBackboneTri',
              'checkpoint_loading': 'all_detector_tensors_strict', 'tracker_constructed': False,
              'flash_attention_3': False, 'rope_real': False}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def content_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def specification(profile=IMAGE_PROFILE):
    if profile != IMAGE_PROFILE:
        raise ValueError('Unsupported SAM3.1 static detector profile: '+str(profile))
    return {'profile': profile, 'model_version': 'SAM3.1', 'checkpoint_sha256': IMAGE_SHA,
            'checkpoint_relative_path': CHECKPOINT, 'checkpoint_bytes': CHECKPOINT_BYTES,
            'weight_repository': 'facebook/sam3.1', 'weight_revision': WEIGHT_REVISION,
            'official_code_commit': CODE_COMMIT, 'official_code_content_sha256': CODE_CONTENT_SHA,
            'detector_tensor_count': DETECTOR_TENSORS, 'preprocess': copy.deepcopy(PREPROCESS),
            'concepts': list(CONCEPTS), 'representation': 'binary_instance_masks_with_confidence',
            'hair_alpha': False, 'person_rgb_replacement': False,
            'quality_status': 'current_static_detector_no_claim_of_static_cos_quality_superiority'}


def model_identity(directory):
    """Bind 3.1 directly; historical installation manifests cannot authorize it."""
    directory = Path(directory).resolve()
    code = directory/'code'
    paths = [code/'LICENSE', code/'sam3/assets/bpe_simple_vocab_16e6.txt.gz',
             *sorted((code/'sam3').rglob('*.py'))]
    code_files = {}
    for path in paths:
        if not path.resolve().is_relative_to(directory) or not path.is_file():
            raise ValueError('Missing or escaped pinned SAM3.1 code/tokenizer/license')
        code_files[path.relative_to(directory).as_posix()] = sha(path)
    if content_sha(code_files) != CODE_CONTENT_SHA:
        raise ValueError('SAM3.1 official code/tokenizer/license differ from the pinned revision')
    public_path = directory/'sam31-public-metadata.json'
    if sha(public_path) != PUBLIC_METADATA_SHA:
        raise ValueError('SAM3.1 pinned public weight metadata differs')
    public = json.loads(public_path.read_text(encoding='utf8'))
    row = next((item for item in public.get('siblings', [])
                if item['rfilename'] == Path(CHECKPOINT).name), {})
    if (public.get('id') != 'facebook/sam3.1' or public.get('sha') != WEIGHT_REVISION
            or row.get('lfs', {}).get('sha256') != IMAGE_SHA or row.get('size') != CHECKPOINT_BYTES):
        raise ValueError('SAM3.1 official checkpoint metadata is incompatible')
    checkpoint = directory/CHECKPOINT
    if not checkpoint.resolve().is_relative_to(directory) or not checkpoint.is_file() or checkpoint.stat().st_size != CHECKPOINT_BYTES:
        raise ValueError('SAM3.1 static checkpoint missing, escaped or truncated')
    if sha(checkpoint) != IMAGE_SHA:
        raise ValueError('SAM3.1 static checkpoint content differs')
    for name, expected in RUNTIME_FILES.items():
        if sha(directory/name) != expected:
            raise ValueError('SAM3.1 isolated runtime file differs: '+name)
    files = {**code_files, **RUNTIME_FILES, CHECKPOINT: IMAGE_SHA,
             'sam31-public-metadata.json': PUBLIC_METADATA_SHA}
    from runtime_resources import code_path
    result = {'specification': specification(), 'files_sha256': files,
        'runtime': {'python': str(directory/'runtime/Scripts/python.exe'),
                    'packages': copy.deepcopy(RUNTIME_PACKAGES)},
        'adapter_sha256': sha(__file__), 'static_loader_sha256': sha(code_path('sam31_static_model.py'))}
    return {**result, 'content_sha256': content_sha(result)}


def image_state(checkpoint, expected_keys):
    """Select all detector tensors; reject missing and unexpected detector keys."""
    if 'model' in checkpoint and isinstance(checkpoint['model'], dict):
        checkpoint = checkpoint['model']
    selected = {key[len('detector.'):]: value for key, value in checkpoint.items()
                if key.startswith('detector.')}
    missing, unexpected = set(expected_keys)-set(selected), set(selected)-set(expected_keys)
    if missing or unexpected:
        raise ValueError('Incompatible SAM3.1 detector checkpoint: missing='+
                         ', '.join(sorted(missing)[:8])+'; unexpected='+', '.join(sorted(unexpected)[:8]))
    return selected
