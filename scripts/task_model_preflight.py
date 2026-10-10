"""Read model files and small metadata only; never import a model runtime.

Readiness here means the required installation structure and checked metadata
have no known blockers. It does not certify weight contents or inference.
Full weight identities remain the responsibility of the execution adapters.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from current_model_registry import DIRECTORY_KEYS, PATH_KEYS, PATH_LABELS, validate_config
from sapiens_model_profiles import profile as sapiens_profile
from sam31_model_profile import (CHECKPOINT, CHECKPOINT_BYTES, CODE_CONTENT_SHA,
    IMAGE_SHA, PUBLIC_METADATA_SHA, RUNTIME_FILES, RUNTIME_PACKAGES, WEIGHT_REVISION)

SCHEMA = 'cosplay_task_model_preflight.v1'
SMALL_FILE_LIMIT = 16 * 1024 * 1024
LOCAL_KEYS = frozenset(('python', 'semantics_checkpoint', 'semantics_repo_dir', 'vitmatte_model_root'))
# Literal architecture/provenance from the current ViT-L correspondence adapter.
# Keeping this metadata-only module independent avoids importing numpy/PIL/torch.
DINO_REPOSITORY = 'facebook/dinov3-vitl16-pretrain-lvd1689m'
DINO_REVISION = 'ea8dc2863c51be0a264bab82070e3e8836b02d51'
DINO_CONFIG = {'model_type': 'dinov3_vit', 'patch_size': 16, 'hidden_size': 1024,
    'num_register_tokens': 4, 'num_hidden_layers': 24, 'num_attention_heads': 16}
DINO_FILES = ('config.json', 'preprocessor_config.json', 'model.safetensors', 'README.md', 'LICENSE.md')


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


class _Inspection:
    def __init__(self):
        self.checks = []
        self.blockers = []
        self.weights = []
        self.hash_bytes_read = 0

    def check(self, key, scope, okay, *, path=None, detail='', **extra):
        row = {'key': key, 'scope': scope, 'available': bool(okay),
            'status': 'checked' if okay else 'blocked', 'detail': detail, **extra}
        if path is not None:
            row['path'] = str(path)
        self.checks.append(row)
        if not okay:
            self.blockers.append(detail or ('Model preflight failed: ' + key))
        return bool(okay)

    def file(self, key, path, *, base=None, allow_empty=False):
        path = Path(path)
        try:
            okay = path.is_file() and (allow_empty or path.stat().st_size > 0)
            if base is not None:
                okay = okay and path.resolve().is_relative_to(Path(base).resolve())
        except OSError:
            okay = False
        return self.check(key, 'file_structure', okay, path=path,
            detail='' if okay else 'Missing, empty or escaped required model file: ' + str(path))

    def small_sha(self, path):
        path = Path(path)
        size = path.stat().st_size
        if size > SMALL_FILE_LIMIT:
            raise ValueError('Metadata/code file exceeds preflight read limit: ' + str(path))
        with path.open('rb') as stream:
            value = hashlib.file_digest(stream, 'sha256').hexdigest()
        self.hash_bytes_read += size
        return value

    def hash_check(self, key, path, expected):
        try:
            actual = self.small_sha(path)
            okay = _digest(expected) and actual == expected
            return self.check(key, 'small_file_content_identity', okay, path=path,
                expected_sha256=expected, actual_sha256=actual,
                detail='' if okay else 'Small model file differs from recorded/pinned identity: ' + str(path))
        except (OSError, ValueError) as exc:
            return self.check(key, 'small_file_content_identity', False, path=path, detail=str(exc))

    def json_file(self, key, path, *, base=None):
        if not self.file(key, path, base=base):
            return None
        try:
            if Path(path).stat().st_size > SMALL_FILE_LIMIT:
                raise ValueError('Model JSON exceeds preflight read limit')
            value = json.loads(Path(path).read_text(encoding='utf-8-sig'))
            if not isinstance(value, dict):
                raise ValueError('Model metadata must be a JSON object')
            self.check(key + '.observed_identity', 'observed_small_file_identity', True,
                path=path, sha256=self.small_sha(path),
                content_verified_against_profile=False)
            return value
        except (OSError, ValueError) as exc:
            self.check(key + '.json', 'configuration', False, path=path,
                detail='Invalid model metadata: ' + str(path) + ': ' + str(exc))
            return None

    def weight(self, key, path, *, base=None, recorded_sha256=None, expected_sha256=None, expected_bytes=None):
        okay = self.file(key, path, base=base)
        size = Path(path).stat().st_size if okay else None
        row = {'key': key, 'path': str(path), 'bytes': size,
            'content_verified': False, 'content_status': 'not_hashed_by_preflight',
            'recorded_sha256': recorded_sha256, 'expected_sha256': expected_sha256}
        self.weights.append(row)
        if expected_bytes is not None:
            self.check(key + '.bytes', 'file_structure', size == expected_bytes,
                path=path, bytes=size, expected_bytes=expected_bytes,
                detail='' if size == expected_bytes else 'Pinned model checkpoint is missing or truncated: ' + str(path))
        if recorded_sha256 is None and expected_sha256 is not None:
            self.check(key + '.expected_identity', 'profile_expected_identity', True,
                path=path, expected_sha256=expected_sha256, status='expected_not_content_verified')
        elif recorded_sha256 is not None:
            compatible = (_digest(recorded_sha256)
                and (expected_sha256 is None or recorded_sha256 == expected_sha256))
            self.check(key + '.recorded_identity', 'recorded_identity', compatible,
                path=path, recorded_sha256=recorded_sha256, expected_sha256=expected_sha256,
                detail='' if compatible else 'Recorded weight identity differs from current profile: ' + str(path))

    def packages(self, key, python, names, *, pinned=None, extra_sites=()):
        sites = [Path(python).parent.parent / 'Lib/site-packages', *extra_sites]
        versions = {}
        for name in names:
            candidates = []
            # Read metadata without importing or executing packages or .pth files.
            for site in sites:
                matches = sorted(site.glob(name.replace('-', '_') + '-*.dist-info/METADATA'))
                if matches:
                    candidates = matches
                    break
            try:
                if len(candidates) != 1:
                    raise ValueError('missing/ambiguous package metadata')
                metadata = candidates[0]
                if metadata.stat().st_size > SMALL_FILE_LIMIT:
                    raise ValueError('oversized package metadata')
                lines = metadata.read_text(encoding='utf8').splitlines()
                versions_found = [line[9:].strip() for line in lines if line.startswith('Version: ')]
                if len(versions_found) != 1 or not versions_found[0]:
                    raise ValueError('missing/ambiguous package version')
                version = versions_found[0]
                if pinned and version != pinned[name]:
                    raise ValueError('package version differs from current profile: ' + version)
                versions[name] = version
                self.check(key + '.' + name, 'runtime_metadata', True, path=metadata,
                    version=version, metadata_sha256=self.small_sha(metadata))
            except (OSError, ValueError) as exc:
                self.check(key + '.' + name, 'runtime_metadata', False,
                    detail='Model runtime ' + name + ': ' + str(exc))
        return versions


def _sapiens_and_matting(check, config):
    spec = sapiens_profile(config['semantics_profile'], 'segmentation')
    repo = Path(config['semantics_repo_dir'])
    for name in ('sapiens/__init__.py', 'sapiens/dense/src/models/__init__.py',
                 'sapiens/dense/src/models/init_model.py', spec['config']):
        if check.file('sapiens.' + name, repo / name, base=repo, allow_empty=name.endswith('__init__.py')):
            check.check('sapiens.' + name + '.observed_identity', 'observed_small_file_identity', True,
                path=repo / name, sha256=check.small_sha(repo / name),
                content_verified_against_profile=False)
    check.weight('sapiens.checkpoint', config['semantics_checkpoint'], expected_sha256=spec['checkpoint_sha256'])
    # If present, a declaration must agree with the pinned profile. Absence is
    # honestly unverified; the execution adapter still hashes the checkpoint.
    manifest_path = Path(config['semantics_checkpoint']).parent / 'model-manifest.json'
    if manifest_path.is_file():
        manifest = check.json_file('sapiens.manifest', manifest_path) or {}
        check.check('sapiens.manifest.profile', 'recorded_identity',
            manifest.get('repository') == spec['repository'] and manifest.get('revision') == spec['revision']
                and manifest.get('checkpoint_sha256') == spec['checkpoint_sha256'], path=manifest_path,
            detail='' if (manifest.get('repository') == spec['repository'] and manifest.get('revision') == spec['revision']
                and manifest.get('checkpoint_sha256') == spec['checkpoint_sha256']) else 'Sapiens2 recorded identity differs from current profile')
    check.packages('human_runtime', config['python'],
        ('torch', 'transformers', 'numpy', 'pillow', 'safetensors', 'opencv-python'))
    matting = Path(config['vitmatte_model_root'])
    check.weight('vitmatte.checkpoint', matting / 'model.safetensors', base=matting)
    model = check.json_file('vitmatte.config', matting / 'config.json', base=matting)
    if model is not None:
        backbone = model.get('backbone_config') or {}
        okay = (model.get('model_type') == 'vitmatte' and model.get('hidden_size') == 384
            and isinstance(backbone, dict) and backbone.get('model_type') == 'vitdet'
            and backbone.get('hidden_size') == 384 and backbone.get('num_channels') == 4)
        check.check('vitmatte.architecture', 'configuration', okay,
            detail='' if okay else 'ViTMatte configuration differs from current small distinctions profile')
    processor = check.json_file('vitmatte.preprocessor', matting / 'preprocessor_config.json', base=matting)
    if processor is not None:
        okay = processor.get('image_processor_type') == 'VitMatteImageProcessor'
        check.check('vitmatte.processor', 'configuration', okay,
            detail='' if okay else 'ViTMatte preprocessor differs from current adapter')


def _dinov3(check, config):
    directory = Path(config['dinov3_model_root'])
    manifest = check.json_file('dinov3.manifest', directory / 'model-manifest.json', base=directory) or {}
    okay = (manifest.get('repository') == DINO_REPOSITORY and manifest.get('revision') == DINO_REVISION
        and isinstance(manifest.get('license'), str) and bool(manifest['license'].strip()))
    check.check('dinov3.provenance', 'recorded_identity', okay,
        detail='' if okay else 'DINOv3 recorded repository/revision/license differs from current profile')
    declared = manifest.get('files_sha256')
    declared = declared if isinstance(declared, dict) else {}
    for name in DINO_FILES:
        path = directory / name
        if name == 'model.safetensors':
            check.weight('dinov3.checkpoint', path, base=directory, recorded_sha256=declared.get(name))
            check.check('dinov3.weight_declaration', 'recorded_identity', _digest(declared.get(name)),
                detail='' if _digest(declared.get(name)) else 'DINOv3 installation has no valid recorded checkpoint digest')
        elif check.file('dinov3.' + name, path, base=directory):
            check.hash_check('dinov3.' + name + '.identity', path, declared.get(name))
    model = check.json_file('dinov3.config', directory / 'config.json', base=directory)
    if model is not None:
        okay = all(model.get(key) == value for key, value in DINO_CONFIG.items())
        check.check('dinov3.architecture', 'configuration', okay,
            detail='' if okay else 'DINOv3 token/config layout differs from current ViT-L adapter')
    processor = check.json_file('dinov3.preprocessor', directory / 'preprocessor_config.json', base=directory)
    if processor is not None:
        okay = processor.get('image_processor_type') == 'DINOv3ViTImageProcessorFast'
        check.check('dinov3.processor', 'configuration', okay,
            detail='' if okay else 'DINOv3 preprocessor differs from installed current model')
    check.packages('correspondence_runtime', config['python'], ('scipy',))


def _sam31(check, config):
    directory = Path(config['scene_model_root'])
    code = directory / 'code'
    paths = [code / 'LICENSE', code / 'sam3/assets/bpe_simple_vocab_16e6.txt.gz',
        *sorted((code / 'sam3').rglob('*.py'))]
    files = {}
    for path in paths:
        if check.file('sam31.' + path.relative_to(directory).as_posix(), path, base=directory):
            try:
                files[path.relative_to(directory).as_posix()] = check.small_sha(path)
            except (OSError, ValueError) as exc:
                check.check('sam31.code_read', 'small_file_content_identity', False, detail=str(exc))
    actual = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    check.check('sam31.code_identity', 'small_file_content_identity', actual == CODE_CONTENT_SHA,
        expected_sha256=CODE_CONTENT_SHA, actual_sha256=actual,
        detail='' if actual == CODE_CONTENT_SHA else 'SAM3.1 official code/tokenizer/license differ from pinned revision')
    public_path = directory / 'sam31-public-metadata.json'
    public = check.json_file('sam31.public_metadata', public_path, base=directory) or {}
    check.hash_check('sam31.public_metadata.identity', public_path, PUBLIC_METADATA_SHA)
    siblings = public.get('siblings')
    rows = [row for row in siblings if isinstance(row, dict) and row.get('rfilename') == Path(CHECKPOINT).name] if isinstance(siblings, list) else []
    row = rows[0] if len(rows) == 1 else {}
    lfs = row.get('lfs') if isinstance(row.get('lfs'), dict) else {}
    okay = (public.get('id') == 'facebook/sam3.1' and public.get('sha') == WEIGHT_REVISION
        and lfs.get('sha256') == IMAGE_SHA and row.get('size') == CHECKPOINT_BYTES)
    check.check('sam31.weight_provenance', 'recorded_identity', okay,
        detail='' if okay else 'SAM3.1 public checkpoint metadata differs from pinned profile')
    check.weight('sam31.checkpoint', directory / CHECKPOINT, base=directory,
        recorded_sha256=lfs.get('sha256'), expected_sha256=IMAGE_SHA, expected_bytes=CHECKPOINT_BYTES)
    for name, expected in RUNTIME_FILES.items():
        path = directory / name
        if check.file('sam31.' + name, path, base=directory):
            check.hash_check('sam31.' + name + '.identity', path, expected)
    expected_python = directory / 'runtime/Scripts/python.exe'
    check.check('sam31.runtime_binding', 'configuration', Path(config['scene_python']).resolve() == expected_python.resolve(),
        detail='' if Path(config['scene_python']).resolve() == expected_python.resolve() else 'SAM3.1 Python differs from its isolated runtime')
    # A .pth path may supply shared packages; never execute import statements.
    sites = []
    pth = directory / 'runtime/Lib/site-packages/shared-model-runtime.pth'
    if pth.is_file() and pth.stat().st_size <= SMALL_FILE_LIMIT:
        for line in pth.read_text(encoding='utf8').splitlines():
            if line.strip() and not line.lstrip().startswith(('#', 'import ')):
                path = Path(line.strip())
                sites.append(path if path.is_absolute() else pth.parent / path)
    check.packages('scene_runtime', config['scene_python'], RUNTIME_PACKAGES,
        pinned=RUNTIME_PACKAGES, extra_sites=sites)


def inspect_task_models(config, *, mode='existing_reference'):
    """Return serializable capability evidence with zero inference/network work."""
    if mode not in {'local_portrait', 'existing_reference', 'create_reference'}:
        raise ValueError('Unsupported task model preflight mode: ' + str(mode))
    check = _Inspection()
    try:
        validate_config(config)
        check.check('registry.configuration', 'configuration', config['ready'],
            detail='' if config['ready'] else 'Installed-model registry is not enabled')
    except (OSError, ValueError) as exc:
        check.check('registry.configuration', 'configuration', False, detail=str(exc))
        config = config if isinstance(config, dict) else {}
    required = LOCAL_KEYS if mode == 'local_portrait' else frozenset(PATH_KEYS)
    for key in PATH_KEYS:
        if key not in required:
            continue
        value = config.get(key)
        path = Path(value) if isinstance(value, str) and value else None
        if key in DIRECTORY_KEYS:
            okay = bool(path and path.is_absolute() and path.is_dir())
            check.check(key, 'file_structure', okay, path=path,
                detail='' if okay else 'Missing installed-model directory: ' + PATH_LABELS[key])
        elif path and path.is_absolute():
            if key in {'python', 'scene_python', 'face_python', 'semantics_checkpoint'}:
                check.file(key, path)
            else:
                check.weight(key, path)
        else:
            check.check(key, 'file_structure', False, detail='Missing installed-model path: ' + PATH_LABELS[key])
    groups = [('human_matting', _sapiens_and_matting)]
    if mode != 'local_portrait':
        groups += [('dinov3', _dinov3), ('sam31', _sam31)]
    for key, inspect in groups:
        try:
            inspect(check, config)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            check.check(key + '.inspection', 'configuration', False, detail='Cannot inspect ' + key + ': ' + str(exc))
    if mode != 'local_portrait':
        try:
            onnx = Path(config['onnx_runtime_root'])
            check.file('onnx.runtime_module', onnx / 'onnxruntime/__init__.py', base=onnx)
            native = [*sorted((onnx / 'onnxruntime/capi').glob('onnxruntime_pybind11_state*.pyd')),
                *sorted((onnx / 'onnxruntime/capi').glob('onnxruntime_pybind11_state*.so'))]
            check.check('onnx.native_module', 'file_structure', len(native) == 1 and native[0].stat().st_size > 0,
                detail='' if len(native) == 1 and native[0].stat().st_size > 0 else 'ONNX runtime native module is missing/ambiguous')
            check.packages('face_runtime', config['face_python'], ('mediapipe',))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            check.check('auxiliary_runtime.inspection', 'configuration', False, detail=str(exc))
    structure = [row for row in check.checks if row['scope'] == 'file_structure']
    identities = [row for row in check.checks if row['scope'] in {'recorded_identity', 'small_file_content_identity'}]
    recorded = [row for row in check.checks if row['scope'] == 'recorded_identity']
    small = [row for row in check.checks if row['scope'] == 'small_file_content_identity']
    return {'schema_version': SCHEMA, 'mode': mode, 'ready': not check.blockers,
        'blocking_reasons': list(dict.fromkeys(check.blockers)), 'checks': check.checks,
        'file_structure_verified': bool(structure) and all(row['available'] for row in structure),
        'checked_identity_verified': bool(identities) and all(row['available'] for row in identities),
        'recorded_identity_verified': bool(recorded) and all(row['available'] for row in recorded),
        'small_file_identity_verified': bool(small) and all(row['available'] for row in small),
        'weight_checks': check.weights, 'weight_content_verified': False,
        'inference_verified': False, 'verification_scope': 'file_structure_small_file_hashes_recorded_identity',
        'hash_bytes_read': check.hash_bytes_read, 'large_weight_bytes_hashed': 0,
        'network_requests': 0, 'model_inferences': 0,
        'limitations': ['Weight contents are not hashed by preflight; execution adapters verify their full identities.',
            'File/runtime metadata does not prove that packages execute, GPU memory is sufficient, or inference succeeds.']}
