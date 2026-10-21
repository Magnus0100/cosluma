"""Read model files and small metadata only; never import a model runtime.

Readiness here means the required installation structure and checked metadata
have no known blockers. It does not certify weight contents or inference.
Full weight identities remain the responsibility of the execution adapters.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from pathlib import PureWindowsPath
import re

from current_model_registry import (CAPABILITY_KEYS, CAPABILITY_LABELS,
    DIRECTORY_KEYS, PATH_KEYS, PATH_LABELS, validate_config)
from sapiens_model_profiles import profile as sapiens_profile
from sam31_model_profile import (CHECKPOINT, CHECKPOINT_BYTES, CODE_CONTENT_SHA,
    IMAGE_SHA, PUBLIC_METADATA_SHA, RUNTIME_FILES, RUNTIME_PACKAGES, WEIGHT_REVISION)

SCHEMA = 'cosplay_task_model_preflight.v1'
SMALL_FILE_LIMIT = 16 * 1024 * 1024
LOCAL_KEYS = frozenset(('python', 'semantics_checkpoint', 'semantics_repo_dir', 'vitmatte_model_root'))
# Internal inspectors are split more finely than the public capability names:
# this keeps the checks readable while the route contract stays centralized in
# current_model_registry.CAPABILITY_KEYS.
CAPABILITY_PATH_KEYS = {
    'human_matting': frozenset(LOCAL_KEYS),
    'reference_correspondence': frozenset(('dinov3_model_root',)),
    'scene_understanding': frozenset(('scene_python', 'scene_model_root')),
    'alpha_seed': frozenset(('birefnet_model_path', 'onnx_runtime_root')),
    'background_repair': frozenset(('lama_checkpoint',)),
    'background_superresolution': frozenset(('python', 'background_sr_checkpoint')),
    'face_localization': frozenset(('face_model_path', 'face_python')),
}
# Literal architecture/provenance from the current ViT-L correspondence adapter.
# Keeping this metadata-only module independent avoids importing numpy/PIL/torch.
DINO_REPOSITORY = 'facebook/dinov3-vitl16-pretrain-lvd1689m'
DINO_REVISION = 'ea8dc2863c51be0a264bab82070e3e8836b02d51'
DINO_CONFIG = {'model_type': 'dinov3_vit', 'patch_size': 16, 'hidden_size': 1024,
    'num_register_tokens': 4, 'num_hidden_layers': 24, 'num_attention_heads': 16}
DINO_FILES = ('config.json', 'preprocessor_config.json', 'model.safetensors', 'README.md', 'LICENSE.md')


def required_capabilities(mode, *, execution_strategy='layered_reference_transfer',
                          complete_post=None, use_beautygrpo=False, editing_scope=None,
                          shaping_request=None):
    """Return the model capabilities consumed by one checked task route.

    This is deliberately a route contract rather than a list of everything
    installed on the machine.  Historical/experimental models are not part of
    ``CAPABILITY_KEYS`` and therefore never enter production preflight.
    """
    if mode not in {'local_portrait', 'existing_reference', 'create_reference'}:
        raise ValueError('Unsupported task model preflight mode: ' + str(mode))
    if execution_strategy != 'layered_reference_transfer':
        raise ValueError('Unsupported task execution strategy: ' + str(execution_strategy))
    if complete_post is None:
        complete_post = mode != 'local_portrait'
    if type(complete_post) is not bool:
        raise ValueError('complete_post must be a boolean')
    if uses_generated_appearance(mode,editing_scope):
        if use_beautygrpo:
            raise ValueError('Generated appearance does not consume a source-space beauty worker')
        if shaping_request is not None:
            from photo_shaping import normalize_request
            request = normalize_request(shaping_request)
            if mode != 'existing_reference':
                raise ValueError('Shaping model capabilities require a checked existing-reference task')
            if request['action'] == 'apply' and 'face' in request['parts']:
                return frozenset({'portrait_regions'})
        return frozenset()
    if shaping_request is not None:
        raise ValueError('Shaping model capabilities require the saved generated-appearance scope')
    required = set()
    if mode == 'local_portrait':
        # The local portrait route currently renders the source with the
        # silhouette-only adapter.  It does not invoke the layered semantic
        # geometry stack, so Sapiens2/ViTMatte must not block this route.
        required.add('source_silhouette')
    else:
        required.update(('source_geometry', 'reference_correspondence', 'scene_analysis'))
        if complete_post:
            required.update(('portrait_regions', 'background_completion', 'background_superresolution'))
    if use_beautygrpo:
        # BeautyGRPO has its own installation registry and is checked by its
        # preparation worker.  Keep it visible in the plan without pretending
        # its files belong to the analysis registry.
        required.add('portrait_beauty')
    return frozenset(required)


def uses_generated_appearance(mode,editing_scope=None):
    """Scope selects actual pixels; existing-reference includes checked iteration."""
    if editing_scope is None:return False
    from photo_editing import validate_scope
    validate_scope(editing_scope)
    return mode in {'create_reference','existing_reference'}


def _safe_path_text(value):
    if (not isinstance(value,str) or not value or any(ord(c)<32 for c in value)
            or '://' in value or value.startswith(('\\\\','//'))
            or PureWindowsPath(value).drive and not PureWindowsPath(value).is_absolute()):
        raise ValueError('Explicit registry contains invalid or unsafe local path syntax')
    tail=value[2:] if len(value)>1 and value[1]==':' else value
    if any(c in tail for c in '<>:"|?*'):
        raise ValueError('Explicit registry contains invalid or unsafe local path syntax')
    return value


def generated_registry_status(root,registry_path=None):
    """Validate explicit unused registry metadata, preserving ready:false honestly.

    No default analysis installation is discovered and no native path is probed.
    An explicit disabled/unconfigured registry is valid metadata for a task that
    does not consume it. Missing/malformed explicit files remain real blockers.
    """
    from current_model_registry import SCHEMA as REGISTRY_SCHEMA, resolve_config_paths
    selected=registry_path if registry_path is not None else os.environ.get('COSPLAY_MODEL_REGISTRY')
    result={'registry_path':None,'origin':'not_requested','config':{},'configured':False,
        'paths':[],'error':None,'native_models_required':False,'configuration_valid':None,
        'verification_scope':'unused_native_registry_not_inspected','inference_verified':False}
    if selected is None:return result
    try:
        text=_safe_path_text(str(selected))
        path=Path(text).expanduser()
        path=(path if path.is_absolute() else Path(root)/path).resolve()
        result.update(registry_path=str(path),origin='explicit' if registry_path is not None else 'environment')
        if not path.is_file():raise ValueError('Explicit analysis registry file is unavailable: '+str(path))
        if path.stat().st_size>SMALL_FILE_LIMIT:raise ValueError('Explicit analysis registry JSON exceeds the metadata read limit')
        config=json.loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(config,dict) or config.get('schema')!=REGISTRY_SCHEMA or type(config.get('ready')) is not bool:
            raise ValueError('Explicit analysis registry requires current schema and boolean readiness')
        for key in PATH_KEYS:
            if key in config and config[key] not in (None,''):_safe_path_text(config[key])
        result.update(config=resolve_config_paths(config,path.parent),configured=config['ready'],
            configuration_valid=True,verification_scope='explicit_unused_registry_schema_only')
    except (OSError,ValueError,TypeError,RuntimeError) as error:
        result.update(error=str(error),configuration_valid=False)
    return result


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

    def packages(self, key, python, names, *, pinned=None, extra_sites=(), strict_name=False):
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
                if strict_name:
                    names_found = [line[6:].strip().lower().replace('_', '-') for line in lines if line.startswith('Name: ')]
                    if names_found != [name.lower().replace('_', '-')]:
                        raise ValueError('missing/ambiguous or mismatched package name')
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
        ('torch', 'transformers', 'numpy', 'pillow', 'safetensors', 'opencv-python', 'scipy'))
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


def _background_superresolution(check, config):
    """Check only the runtime packages imported by the source-bound SR worker."""
    check.packages('background_sr_runtime', config['python'],
        ('torch', 'numpy', 'pillow', 'opencv-python', 'scipy'))


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


def inspect_task_models(config, *, mode='existing_reference',
                        execution_strategy='layered_reference_transfer',
                        complete_post=None, use_beautygrpo=False, editing_scope=None,
                        shaping_request=None):
    """Return route-scoped model evidence with zero inference/network work.

    Only the capabilities consumed by this route are inspected.  A model
    installed for another route is explicitly marked ``not_requested`` and
    cannot block this task.
    """
    requested = required_capabilities(mode, execution_strategy=execution_strategy,
        complete_post=complete_post, use_beautygrpo=use_beautygrpo,editing_scope=editing_scope,
        shaping_request=shaping_request)
    face_shaping = shaping_request is not None and requested == frozenset({'portrait_regions'})
    group_for = {
        'source_silhouette': ('alpha_seed',),
        'source_geometry': ('human_matting', 'alpha_seed'),
        'reference_correspondence': ('reference_correspondence',),
        'scene_analysis': ('scene_understanding',),
        'portrait_regions': ('face_localization',),
        'background_completion': ('background_repair',),
        'background_superresolution': ('background_superresolution',),
        'portrait_beauty': (),
    }
    # ``requested`` is a frozenset by contract; use a fixed worker order so
    # plan identities stay identical between the desktop process and the CLI.
    worker_order = ('human_matting', 'reference_correspondence',
        'scene_understanding', 'face_localization', 'background_repair',
        'background_superresolution', 'alpha_seed')
    capabilities = tuple(group for group in worker_order
                         if any(group in group_for[name] for name in requested))
    required = frozenset().union(*(CAPABILITY_PATH_KEYS[key] for key in capabilities)) if capabilities else frozenset()
    profile_for = {
        'source_geometry': ('semantics_profile', 'matting_profile'),
        'reference_correspondence': ('dinov3_profile',),
        'scene_analysis': ('scene_profile',),
    }
    required_profiles = frozenset(profile for name in requested for profile in profile_for.get(name, ()))
    check = _Inspection()
    try:
        # Validate only the route's paths.  In particular source-bound finish
        # must remain runnable with a registry containing just its SR runtime.
        if requested:
            validate_config(config, required_keys=required, required_profiles=required_profiles)
            check.check('registry.configuration', 'configuration', config['ready'],
                detail='' if config['ready'] else 'Installed-model registry is not enabled')
    except (OSError, ValueError) as exc:
        check.check('registry.configuration', 'configuration', False, detail=str(exc))
        config = config if isinstance(config, dict) else {}
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
            if face_shaping and key in {'face_model_path', 'face_python'}:
                if check.file(key, path):
                    try:
                        actual = check.small_sha(path)
                        check.check(key+'.observed_identity', 'observed_small_file_identity', True,
                            path=path, sha256=actual, actual_sha256=actual, bytes=path.stat().st_size,
                            content_verified_against_profile=False)
                    except (OSError, ValueError) as exc:
                        check.check(key+'.observed_identity', 'observed_small_file_identity', False,
                            path=path, detail=str(exc))
            elif key in {'python', 'scene_python', 'face_python', 'semantics_checkpoint'}:
                check.file(key, path)
            else:
                check.weight(key, path)
        else:
            check.check(key, 'file_structure', False, detail='Missing installed-model path: ' + PATH_LABELS[key])
    groups = []
    if 'human_matting' in capabilities:
        groups.append(('human_matting', _sapiens_and_matting))
    if 'background_superresolution' in capabilities:
        groups.append(('background_superresolution', _background_superresolution))
    if 'reference_correspondence' in capabilities:
        groups.append(('dinov3', _dinov3))
    if 'scene_understanding' in capabilities:
        groups.append(('sam31', _sam31))
    for key, inspect in groups:
        try:
            inspect(check, config)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            check.check(key + '.inspection', 'configuration', False, detail='Cannot inspect ' + key + ': ' + str(exc))
    if 'alpha_seed' in capabilities or 'face_localization' in capabilities:
        try:
            if 'alpha_seed' in capabilities:
                onnx = Path(config['onnx_runtime_root'])
                check.file('onnx.runtime_module', onnx / 'onnxruntime/__init__.py', base=onnx)
                native = [*sorted((onnx / 'onnxruntime/capi').glob('onnxruntime_pybind11_state*.pyd')),
                    *sorted((onnx / 'onnxruntime/capi').glob('onnxruntime_pybind11_state*.so'))]
                check.check('onnx.native_module', 'file_structure', len(native) == 1 and native[0].stat().st_size > 0,
                    detail='' if len(native) == 1 and native[0].stat().st_size > 0 else 'ONNX runtime native module is missing/ambiguous')
            if 'face_localization' in capabilities:
                check.packages('face_runtime', config['face_python'], ('mediapipe',), strict_name=face_shaping)
                if face_shaping:
                    for row in check.checks:
                        if row['key']=='face_runtime.mediapipe' and row['available']:
                            row['actual_sha256']=row['metadata_sha256']
        except (OSError, ValueError, TypeError, KeyError) as exc:
            check.check('auxiliary_runtime.inspection', 'configuration', False, detail=str(exc))
    beauty = None
    if 'portrait_beauty' in requested:
        from model_setup import beauty_status
        beauty = beauty_status()
        check.check('portrait_beauty.installation', 'recorded_identity', beauty['available'],
            path=beauty.get('config_path'), detail=beauty.get('error') or '')
    structure = [row for row in check.checks if row['scope'] == 'file_structure']
    identity_scopes = {'recorded_identity', 'small_file_content_identity'} | ({'observed_small_file_identity', 'runtime_metadata'} if face_shaping else set())
    identities = [row for row in check.checks if row['scope'] in identity_scopes]
    recorded = [row for row in check.checks if row['scope'] == 'recorded_identity']
    small = [row for row in check.checks if row['scope'] in (
        {'small_file_content_identity', 'observed_small_file_identity', 'runtime_metadata'} if face_shaping else {'small_file_content_identity'})]
    # Keep conceptual capability names in plans and expose the worker groups
    # used to produce the evidence.
    capability_status = {}
    known = set(CAPABILITY_KEYS) | {'portrait_beauty'}
    for name in sorted(known):
        if name in requested:
            capability_status[name] = {'status': 'required', 'ready':
                bool(beauty and beauty['available']) if name == 'portrait_beauty' else not bool(check.blockers),
                'label': CAPABILITY_LABELS.get(name, name),
                'keys': sorted(CAPABILITY_KEYS.get(name, ())),
                'delegated': name == 'portrait_beauty'}
        else:
            capability_status[name] = {'status': 'not_requested', 'ready': None,
                'label': CAPABILITY_LABELS.get(name, name),
                'keys': sorted(CAPABILITY_KEYS.get(name, ())), 'delegated': False}
    output_capabilities = sorted(requested)
    return {'schema_version': SCHEMA, 'mode': mode,
        'execution_strategy': execution_strategy,
        'required_capabilities': output_capabilities,
        'required_model_groups': list(capabilities),
        'optional_capabilities': sorted(known - set(requested)),
        'unused_capabilities': sorted(known - set(requested)),
        'capability_status': capability_status,
        'required_path_keys': [key for key in PATH_KEYS if key in required],
        'ready': not check.blockers,
        'blocking_reasons': list(dict.fromkeys(check.blockers)), 'checks': check.checks,
        'file_structure_verified': bool(structure) and all(row['available'] for row in structure),
        'checked_identity_verified': bool(identities) and all(row['available'] for row in identities),
        'recorded_identity_verified': bool(recorded) and all(row['available'] for row in recorded),
        'small_file_identity_verified': bool(small) and all(row['available'] for row in small),
        'weight_checks': check.weights, 'weight_content_verified': False,
        'inference_verified': False, 'native_registry_required':bool(requested),
        'verification_scope': 'file_structure_small_file_hashes_recorded_identity' if requested else 'native_models_not_requested',
        'hash_bytes_read': check.hash_bytes_read, 'large_weight_bytes_hashed': 0,
        'network_requests': 0, 'model_inferences': 0,
        'limitations': [('Only the selected bounded face-model bytes, Python and package metadata are hashed; this does not prove inference.'
            if face_shaping else 'Weight contents are not hashed by preflight; execution adapters verify their full identities.'),
            'File/runtime metadata does not prove that packages execute, GPU memory is sufficient, or inference succeeds.']}
