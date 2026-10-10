"""Cached CPU face landmarks; semantic crop and source-derived eye/lip masks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from local_photo_analysis import _worker, file_identity
from runtime_resources import code_path


def prepare_face_protection(root, input_path, semantic_result, *, registry_path=None):
    root = Path(root)
    from current_model_registry import load_registry
    registry = load_registry(root, registry_path=registry_path)
    worker = code_path('local_face_protection_worker.py')
    model = Path(registry['face_model_path'])
    python = Path(registry['face_python'])
    input_path = Path(input_path)
    labels = Path(semantic_result['labels_path'])
    identity = {'input': file_identity(input_path), 'labels': file_identity(labels),
        'model': file_identity(model), 'worker': file_identity(worker),
        'parameters': {'num_faces': 1, 'semantic_crop': 'Face_Neck',
            'detection_confidence': .3, 'presence_confidence': .3}}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    directory = root/'outputs/cache/face-protection'/key
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory/'result.json'
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding='utf8'))
        return {**result, 'cache_hit': True}
    semantics = directory/'semantic-input.json'
    semantics.write_text(json.dumps(semantic_result), encoding='utf8')
    _worker([python, '-X', 'utf8', worker, '--input', input_path, '--semantics', semantics,
        '--model', model, '--output-dir', directory], directory/'worker.log')
    result = json.loads(result_path.read_text(encoding='utf8'))
    result.update(identity=identity, cache_hit=False)
    result_path.write_text(json.dumps(result, indent=2), encoding='utf8')
    return result
