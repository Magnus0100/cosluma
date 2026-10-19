"""Frozen source-label identity independent of optional scene contact layers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from asset_identity import sha256_file

SCHEMA = 'source_semantic_binding.v1'
ROLE = 'source_owned_review_semantics'


def _size(value, label):
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(type(item) is not int or item <= 0 for item in value)):
        raise ValueError(label + ' must contain two positive integer dimensions')
    return list(value)


def load_source_semantics(asset, *, source_sha256, native_size, proxy_size):
    """Validate stored labels and metadata without any model/cache lookup."""
    from local_semantics_worker import CLASS_NAMES
    from sapiens_model_profiles import profile
    if (len(CLASS_NAMES) != 29
            or [CLASS_NAMES[index] for index in (3, 6, 15)] != ['Face_Neck', 'Left_Hand', 'Right_Hand']):
        raise ValueError('Current parser taxonomy differs from the frozen semantic contract')
    if not isinstance(asset, dict) or asset.get('schema_version') != SCHEMA:
        raise ValueError('Missing current independent source semantic binding')
    native_size = _size(native_size, 'Source native canvas')
    proxy_size = _size(proxy_size, 'Source semantic proxy')
    if (asset.get('role') != ROLE or asset.get('coordinate_space') != 'source_proxy'
            or asset.get('source_sha256') != source_sha256
            or _size(asset.get('native_size'), 'Semantic native canvas') != native_size
            or _size(asset.get('size'), 'Semantic proxy canvas') != proxy_size):
        raise ValueError('Frozen source semantics have a different source/canvas binding')
    if (not isinstance(asset.get('path'), str) or not isinstance(asset.get('sha256'), str)
            or len(asset['sha256']) != 64):
        raise ValueError('Frozen source semantics require an explicit file/hash binding')
    path = Path(asset['path']).resolve()
    actual = sha256_file(path)
    if actual != asset['sha256']:
        raise ValueError('Frozen source semantic bytes differ from the plan')
    specification = profile('sapiens2_seg_0.4b', 'segmentation')
    taxonomy_sha = hashlib.sha256(json.dumps(CLASS_NAMES, separators=(',', ':')).encode()).hexdigest()
    if (asset.get('model_profile') != 'sapiens2_seg_0.4b'
            or asset.get('model') != specification['repository']
            or asset.get('model_revision') != specification['revision']
            or asset.get('checkpoint_sha256') != specification['checkpoint_sha256']
            or asset.get('taxonomy_sha256') != taxonomy_sha):
        raise ValueError('Frozen source semantics differ from the pinned model/taxonomy identity')
    with np.load(path, allow_pickle=False) as data:
        labels = data['labels']
    if (labels.ndim != 2 or labels.shape != (proxy_size[1], proxy_size[0])
            or not np.issubdtype(labels.dtype, np.integer)
            or labels.min() < 0 or labels.max() >= len(CLASS_NAMES)):
        raise ValueError('Frozen source labels have invalid shape or taxonomy range')
    return labels, {**asset, 'path': str(path), 'taxonomy': {
        'source': 'frozen independent source labels', 'contract_schema': SCHEMA,
        'model': specification['repository'], 'model_profile': 'sapiens2_seg_0.4b',
        'model_revision': specification['revision'], 'checkpoint_sha256': specification['checkpoint_sha256'],
        'class_names': list(CLASS_NAMES), 'class_names_sha256': taxonomy_sha,
        'inference_performed': False}}


def bind_source_semantics(result, *, source_sha256, native_size, proxy_size):
    """Publish only the analyzed source label asset, never an inferred locator."""
    path = Path(result['labels_path']).resolve()
    identity = sha256_file(path)
    asset = {'schema_version': SCHEMA, 'path': str(path), 'sha256': identity,
        'source_sha256': source_sha256, 'native_size': list(native_size), 'size': list(proxy_size),
        'coordinate_space': 'source_proxy', 'role': ROLE, 'representation': 'semantic_labels',
        'sampling': 'nearest', **{key: result.get(key) for key in (
            'model', 'model_profile', 'model_revision', 'checkpoint_sha256', 'config_sha256', 'taxonomy_sha256')}}
    load_source_semantics(asset, source_sha256=source_sha256, native_size=native_size, proxy_size=proxy_size)
    return asset
