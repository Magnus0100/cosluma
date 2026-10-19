"""Independent tiny installation fixtures; metadata checks only, no inference.

The fixture's SAM files deliberately cannot match pinned production identities.
Full-reference contract tests may mock the capability result, but these tests
exercise the real inspector and never regard dummy weights as model acceptance.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from unittest.mock import patch

from current_model_registry import PROFILES, SCHEMA
from sapiens_model_profiles import profile as sapiens_profile
import task_model_preflight as preflight


def _write(path, data=b'Public offline fixture; never an executable model/runtime.\n'):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _json(path, value):
    return _write(path, (json.dumps(value, sort_keys=True, indent=2) + '\n').encode())


def _metadata(site, name, version='fixture-only'):
    _write(site / (name.replace('-', '_') + '-' + version + '.dist-info/METADATA'),
        ('Metadata-Version: 2.1\nName: ' + name + '\nVersion: ' + version + '\n').encode())


def build_registry_fixture(root, *, name='models'):
    """Create a standalone small-file registry and return its absolute JSON path.

    This supplies valid required paths/config metadata without importing any
    runtime. The real inspector accepts its local structure but blocks full
    SAM readiness due to pinned byte identities/size. No real install is read.
    """
    root = Path(root).resolve()
    models = root / name
    human = models / 'human/runtime/Scripts/python.exe'
    repo = models / 'human/repo'
    values = {'schema': SCHEMA, 'ready': True, **PROFILES,
        'python': str(human), 'semantics_checkpoint': str(models / 'human/checkpoints/parser.safetensors'),
        'semantics_repo_dir': str(repo), 'vitmatte_model_root': str(models / 'vitmatte'),
        'dinov3_model_root': str(models / 'dinov3'),
        'scene_python': str(models / 'sam31/runtime/Scripts/python.exe'),
        'scene_model_root': str(models / 'sam31'),
        'birefnet_model_path': str(models / 'birefnet.onnx'), 'lama_checkpoint': str(models / 'lama.pt'),
        'background_sr_checkpoint': str(models / 'background-sr.pth'),
        'face_model_path': str(models / 'face/model.task'),
        'face_python': str(models / 'face/runtime/Scripts/python.exe'),
        'onnx_runtime_root': str(models / 'onnx')}
    for key in ('python', 'semantics_checkpoint', 'scene_python', 'birefnet_model_path',
                'lama_checkpoint', 'background_sr_checkpoint', 'face_model_path', 'face_python'):
        _write(values[key])
    spec = sapiens_profile(PROFILES['semantics_profile'], 'segmentation')
    for relative in ('sapiens/__init__.py', 'sapiens/dense/src/models/__init__.py',
                     'sapiens/dense/src/models/init_model.py', spec['config']):
        _write(repo / relative, b'# Offline fixture source; not executed.\n')
    site = human.parent.parent / 'Lib/site-packages'
    for name_ in ('torch', 'transformers', 'numpy', 'pillow', 'safetensors', 'opencv-python', 'scipy'):
        _metadata(site, name_)
    matting = Path(values['vitmatte_model_root'])
    _write(matting / 'model.safetensors')
    _json(matting / 'config.json', {'model_type': 'vitmatte', 'hidden_size': 384,
        'backbone_config': {'model_type': 'vitdet', 'hidden_size': 384, 'num_channels': 4}})
    _json(matting / 'preprocessor_config.json', {'image_processor_type': 'VitMatteImageProcessor'})
    dino = Path(values['dinov3_model_root'])
    _json(dino / 'config.json', preflight.DINO_CONFIG)
    _json(dino / 'preprocessor_config.json', {'image_processor_type': 'DINOv3ViTImageProcessorFast'})
    for relative in ('model.safetensors', 'README.md', 'LICENSE.md'):
        _write(dino / relative)
    _json(dino / 'model-manifest.json', {'repository': preflight.DINO_REPOSITORY,
        'revision': preflight.DINO_REVISION, 'license': 'Public offline test fixture only',
        'files_sha256': {relative: hashlib.sha256((dino / relative).read_bytes()).hexdigest()
            for relative in preflight.DINO_FILES}})
    sam = Path(values['scene_model_root'])
    _write(sam / 'code/LICENSE')
    _write(sam / 'code/sam3/assets/bpe_simple_vocab_16e6.txt.gz')
    _write(sam / 'code/sam3/__init__.py', b'# Offline fixture only.\n')
    _write(sam / preflight.CHECKPOINT)
    _json(sam / 'sam31-public-metadata.json', {'id': 'facebook/sam3.1', 'sha': preflight.WEIGHT_REVISION,
        'siblings': [{'rfilename': Path(preflight.CHECKPOINT).name, 'size': preflight.CHECKPOINT_BYTES,
            'lfs': {'sha256': preflight.IMAGE_SHA}}]})
    _write(sam / 'runtime/pyvenv.cfg', b'# Offline fixture; never launched.\n')
    _write(sam / 'runtime/Lib/site-packages/shared-model-runtime.pth', (str(site) + '\n').encode())
    scene_site = sam / 'runtime/Lib/site-packages'
    for name_, version in preflight.RUNTIME_PACKAGES.items():
        _metadata(scene_site, name_, version)
    _metadata(Path(values['face_python']).parent.parent / 'Lib/site-packages', 'mediapipe')
    onnx = Path(values['onnx_runtime_root'])
    _write(onnx / 'onnxruntime/__init__.py', b'# Offline fixture; never imported.\n')
    _write(onnx / 'onnxruntime/capi/onnxruntime_pybind11_state.pyd')
    registry = root / (name + '.json')
    _json(registry, values)
    return registry


def mocked_task_capability(config, *, mode='existing_reference', **kwargs):
    """Explicit model-capability mock for task/API contract tests, not acceptance."""
    requested = preflight.required_capabilities(mode,
        execution_strategy=kwargs.get('execution_strategy', 'layered_reference_transfer'),
        complete_post=kwargs.get('complete_post'),
        use_beautygrpo=kwargs.get('use_beautygrpo', False))
    required_path_keys = sorted({key for name in requested for key in preflight.CAPABILITY_KEYS.get(name, ())})
    return {'schema_version': preflight.SCHEMA, 'mode': mode,
        'execution_strategy': kwargs.get('execution_strategy', 'layered_reference_transfer'),
        'required_capabilities': sorted(requested), 'required_model_groups': [],
        'required_path_keys': required_path_keys, 'ready': True,
        'blocking_reasons': [], 'checks': [], 'file_structure_verified': False,
        'checked_identity_verified': False, 'recorded_identity_verified': False,
        'small_file_identity_verified': False, 'weight_checks': [], 'weight_content_verified': False,
        'inference_verified': False, 'verification_scope': 'mocked_model_capability_for_contract_test',
        'mocked': True, 'hash_bytes_read': 0, 'large_weight_bytes_hashed': 0,
        'network_requests': 0, 'model_inferences': 0}


def _row(result, key):
    return next(row for row in result['checks'] if row['key'] == key)


def _snapshot(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob('*') if path.is_file()}


def _adapter_metadata_matches():
    """Catch metadata drift without importing the numerical correspondence API."""
    tree = ast.parse((Path(__file__).resolve().parents[1]/'scripts/reference_local_correspondence.py').read_text(encoding='utf8'))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
        and node.name == 'algorithm_configuration')
    dictionary = next(node.value for node in ast.walk(function) if isinstance(node, ast.Return))
    values = {key.value: value for key, value in zip(dictionary.keys, dictionary.values) if isinstance(key, ast.Constant)}
    assert ast.literal_eval(values['expected_config']) == preflight.DINO_CONFIG
    assert ast.literal_eval(values['repository']) == preflight.DINO_REPOSITORY
    revision = next(node.value for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'DINO3_L_REVISION' for target in node.targets))
    assert ast.literal_eval(revision) == preflight.DINO_REVISION


def main():
    _adapter_metadata_matches()
    with tempfile.TemporaryDirectory(prefix='task-model-preflight-') as folder:
        root = Path(folder)
        registry = build_registry_fixture(root)
        config = json.loads(registry.read_text(encoding='utf8'))
        before = _snapshot(root)
        weights = {Path(config[key]).resolve() for key in ('semantics_checkpoint', 'birefnet_model_path',
            'lama_checkpoint', 'background_sr_checkpoint', 'face_model_path')}
        weights.update((Path(config[key]) / relative).resolve() for key, relative in
            (('vitmatte_model_root', 'model.safetensors'), ('dinov3_model_root', 'model.safetensors'),
             ('scene_model_root', preflight.CHECKPOINT)))
        original_digest = hashlib.file_digest
        hashed = []
        def guarded_digest(stream, algorithm):
            path = Path(stream.name).resolve()
            assert path not in weights, 'Preflight hashed full model weight bytes'
            hashed.append(path)
            return original_digest(stream, algorithm)
        modules_before = set(sys.modules)
        with patch.object(hashlib, 'file_digest', side_effect=guarded_digest), \
             patch.object(socket, 'create_connection', side_effect=AssertionError('Network requested')), \
             patch.object(subprocess, 'Popen', side_effect=AssertionError('Runtime launched')):
            local = preflight.inspect_task_models(config, mode='local_portrait')
            assert local['ready'] and local['file_structure_verified']
            # A local portrait must not be held hostage by full-reference
            # components that this route never opens.
            local_unrelated = dict(config)
            for key in ('dinov3_model_root', 'scene_python', 'scene_model_root',
                        'lama_checkpoint', 'background_sr_checkpoint',
                        'face_model_path', 'face_python'):
                local_unrelated[key] = str(root / 'missing-local-unrelated' / key)
            local_without_reference_models = preflight.inspect_task_models(
                local_unrelated, mode='local_portrait')
            assert local_without_reference_models['ready']
            assert local_without_reference_models['required_capabilities'] == ['source_silhouette']
            assert local_without_reference_models['required_model_groups'] == ['alpha_seed']
            assert not any(row['key'].startswith(('dinov3.', 'sam31.', 'face_runtime',
                                                   'background_sr_runtime'))
                           for row in local_without_reference_models['checks'])
            # The current local route also does not invoke the layered
            # semantic/matting stack.  Missing Sapiens2 and ViTMatte must not
            # turn an otherwise runnable silhouette edit into a blocked task.
            local_without_geometry = dict(local_unrelated)
            for key in ('python', 'semantics_checkpoint', 'semantics_repo_dir',
                        'vitmatte_model_root'):
                local_without_geometry[key] = str(root / 'missing-local-geometry' / key)
            local_silhouette_only = preflight.inspect_task_models(
                local_without_geometry, mode='local_portrait')
            assert local_silhouette_only['ready']
            assert not any(row['key'].startswith(('sapiens.', 'vitmatte.'))
                           for row in local_silhouette_only['checks'])
            assert not local['checked_identity_verified'] and not local['recorded_identity_verified']
            assert not local['weight_content_verified'] and not local['inference_verified']
            full = preflight.inspect_task_models(config)
            assert not full['ready'], 'Synthetic SAM identities were accepted'
            assert not _row(full, 'sam31.code_identity')['available']
            assert not _row(full, 'sam31.public_metadata.identity')['available']
            assert not _row(full, 'sam31.checkpoint.bytes')['available']
            assert _row(full, 'sam31.weight_provenance')['available']
            assert _row(full, 'dinov3.architecture')['available']
            assert preflight.inspect_task_models(config, mode='create_reference')['ready'] is False
            with patch('model_setup.beauty_status', return_value={'available': False,
                    'config_path': str(root/'missing-beauty.json'),
                    'error': 'Beauty installation receipt is missing'}) as beauty:
                checked = preflight.inspect_task_models(config, mode='create_reference', use_beautygrpo=True)
                assert not _row(checked, 'portrait_beauty.installation')['available']
                assert checked['capability_status']['portrait_beauty']['ready'] is False
                assert 'Beauty installation receipt is missing' in checked['blocking_reasons']
                beauty.assert_called_once()
            with patch('model_setup.beauty_status', side_effect=AssertionError('Unused Beauty inspected')):
                preflight.inspect_task_models(config, mode='local_portrait')
        assert hashed and full['hash_bytes_read'] > 0 and full['large_weight_bytes_hashed'] == 0
        assert not ({'torch', 'transformers', 'numpy', 'onnxruntime', 'mediapipe'} & (set(sys.modules) - modules_before))
        assert _snapshot(root) == before, 'Model preflight mutated installation'
        def modify(path, data, key, *, mode='existing_reference'):
            path = Path(path)
            original = path.read_bytes()
            try:
                if data is None:
                    path.unlink()
                else:
                    path.write_bytes(data)
                checked = preflight.inspect_task_models(config, mode=mode)
                assert not checked['ready'] and not _row(checked, key)['available'], key
            finally:
                path.write_bytes(original)
        matting = Path(config['vitmatte_model_root'])
        for name in ('config.json', 'preprocessor_config.json', 'model.safetensors'):
            key = {'config.json': 'vitmatte.config', 'preprocessor_config.json': 'vitmatte.preprocessor',
                'model.safetensors': 'vitmatte.checkpoint'}[name]
            modify(matting / name, None, key, mode='existing_reference')
        modify(matting / 'config.json', b'{"model_type":"vitmatte","hidden_size":768}',
            'vitmatte.architecture', mode='existing_reference')
        modify(matting / 'preprocessor_config.json', b'not json', 'vitmatte.preprocessor.json', mode='existing_reference')
        modify(config['semantics_checkpoint'], b'', 'sapiens.checkpoint', mode='existing_reference')
        spec = sapiens_profile(PROFILES['semantics_profile'], 'segmentation')
        modify(Path(config['semantics_repo_dir']) / spec['config'], None, 'sapiens.' + spec['config'], mode='existing_reference')
        bad_manifest = Path(config['semantics_checkpoint']).parent / 'model-manifest.json'
        _json(bad_manifest, {'repository': spec['repository'], 'revision': spec['revision'], 'checkpoint_sha256': '0'*64})
        assert not preflight.inspect_task_models(config, mode='existing_reference')['ready']
        bad_manifest.unlink()
        dino = Path(config['dinov3_model_root'])
        for name in preflight.DINO_FILES:
            key = 'dinov3.checkpoint' if name == 'model.safetensors' else 'dinov3.' + name
            modify(dino / name, None, key)
        modify(dino / 'README.md', b'changed after install', 'dinov3.README.md.identity')
        manifest = json.loads((dino / 'model-manifest.json').read_text())
        wrong = copy.deepcopy(manifest); wrong['revision'] = 'wrong'
        modify(dino / 'model-manifest.json', json.dumps(wrong).encode(), 'dinov3.provenance')
        wrong = copy.deepcopy(manifest); wrong['files_sha256']['model.safetensors'] = 'invalid'
        modify(dino / 'model-manifest.json', json.dumps(wrong).encode(), 'dinov3.weight_declaration')
        dino_config = copy.deepcopy(preflight.DINO_CONFIG); dino_config['hidden_size'] = 768
        modify(dino / 'config.json', json.dumps(dino_config).encode(), 'dinov3.architecture')
        sam = Path(config['scene_model_root'])
        modify(sam / preflight.CHECKPOINT, None, 'sam31.checkpoint')
        modify(sam / 'code/sam3/assets/bpe_simple_vocab_16e6.txt.gz', None,
            'sam31.code/sam3/assets/bpe_simple_vocab_16e6.txt.gz')
        modify(Path(config['onnx_runtime_root']) / 'onnxruntime/capi/onnxruntime_pybind11_state.pyd', None, 'onnx.native_module')
        metadata = Path(config['python']).parent.parent / 'Lib/site-packages/torch-fixture-only.dist-info/METADATA'
        modify(metadata, None, 'human_runtime.torch', mode='existing_reference')
        ambiguous = metadata.parent.parent / 'torch-second.dist-info/METADATA'
        _write(ambiguous, b'Name: torch\nVersion: second\n')
        assert not preflight.inspect_task_models(config, mode='existing_reference')['ready']
        ambiguous.unlink(); ambiguous.parent.rmdir()
        assert _snapshot(root) == before
    print(json.dumps({'ok': True, 'standalone_public_fixture': True,
        'missing_internal_files_blocked': True, 'known_small_identity_mismatch_blocked': True,
        'configuration_and_runtime_metadata_checked': True, 'weight_content_verified': False,
        'inference_verified': False, 'model_inferences': 0, 'network_requests': 0,
        'large_weight_bytes_hashed': 0, 'installation_writes_by_preflight': 0}))


if __name__ == '__main__':
    main()
