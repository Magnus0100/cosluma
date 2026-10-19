"""Recover a frozen source observation from its unedited paired native bundle.

The identity is photographic observation, not a claim that F/B is uniquely
correct. Reconstruct before downsampling so the alpha/B product stays paired.
"""
from __future__ import annotations

from functools import lru_cache
from dataclasses import dataclass, field
import copy
import hashlib
import json
import os
from pathlib import Path
import time
import zlib
import zipfile
import numpy as np
from PIL import Image
from asset_identity import sha256_file


def retouch_identity(run, *, root=None):
    """Frozen source-stage identity; never starts recognition or repairs pixels."""
    record = (run.get('options') or {}).get('native_source_retouch')
    if not record:
        return None
    if record.get('schema') != 'automatic_native_retouch_binding.v1' or record.get('source_sha256') != run['source']['sha256']:
        raise ValueError('Frozen automatic retouch is bound to a different RAW root')
    layer_asset = record['layer']
    path = Path(layer_asset['path'])
    path = path if path.is_absolute() else Path(root or run.get('root') or '.')/path
    digest = sha256_file(path)
    if digest != layer_asset['sha256']:
        raise ValueError('Frozen native retouch layer bytes changed')
    layer = json.loads(path.read_text(encoding='utf8'))
    if layer['source']['sha256'] != run['source']['sha256'] or layer['native_size'] != record['native_size']:
        raise ValueError('Frozen retouch source/canvas differs from the installed binding')
    from native_retouch_producer import frozen_working_domain
    domain = frozen_working_domain(layer)
    if record['working_domain'] != domain:
        raise ValueError('Frozen retouch binding has an invalid working domain')
    delta_path = Path(layer['delta']['path'])
    delta_path = delta_path if delta_path.is_absolute() else path.parent/delta_path
    delta_sha = sha256_file(delta_path)
    if delta_sha != layer['delta']['sha256']:
        raise ValueError('Frozen native retouch delta bytes changed')
    return {'schema': record['schema'], 'layer_sha256':digest,'delta_sha256':delta_sha,
            'source_sha256':record['source_sha256'],'native_size':record['native_size'],
            'working_domain':domain,'approved_patches':len(layer['patches']),
            'execution_stage':'before_display_alpha_decomposition_and_reference_TBS'}


def open_source_for_run(run, *, root=None):
    """Decode untouched root, consume its frozen source delta exactly once.

    This is for producer decoding before display/alpha decomposition. Saved
    native plans replay their already prepared Fp/B pair via source_for_plan;
    they must not add this difference to that pair again.
    """
    from raw_support import open_linear
    path = Path(run['source']['path'])
    path = path if path.is_absolute() else Path(root or run.get('root') or '.')/path
    values, precision = open_linear(path)
    record = (run.get('options') or {}).get('native_source_retouch')
    if not record:
        return values, precision
    identity = retouch_identity(run,root=root)
    from native_retouch_producer import apply_frozen_retouch, sha256
    if sha256(path) != run['source']['sha256']:
        raise ValueError('Original source changed before frozen retouch decode')
    layer_path = Path(record['layer']['path'])
    layer_path = layer_path if layer_path.is_absolute() else Path(root or run.get('root') or '.')/layer_path
    layer = json.loads(layer_path.read_text(encoding='utf8'))
    expected_domain = 'scene_linear_sRGB' if precision['source_precision']=='raw16_linear' else 'display_linear_sRGB'
    if identity['working_domain'] != expected_domain:
        raise ValueError('RAW/display domain mismatch before native retouch consumption')
    values = apply_frozen_retouch(values,layer,source_sha256=run['source']['sha256'],
        canvas_size=values.shape[1::-1],verify_original=True)
    return values,{**precision,'native_source_retouch':identity,'native_retouch_consumptions':1}


def retouch_baseline_record(run):
    """Carry owned closure into saved baseline without a replay-stage delta."""
    record = (run.get('options') or {}).get('native_source_retouch')
    return copy.deepcopy(record) if record else None


# Reference preparation launches one short-lived worker per producer stage.
# A stage-local source cache avoids decoding the same camera RAW in every
# worker while keeping the cache outside saved project assets.  The cache is
# opt-in through the coordinator environment only; direct callers retain the
# original decode path.
_REFERENCE_SOURCE_CACHE_ENV = 'COSPLAY_REFERENCE_SOURCE_CACHE'
_REFERENCE_SOURCE_CACHE_SCHEMA = 'reference_source_cache.v1'


class _ValidatedReferenceSourceCache(tuple):
    """Immutable result of one validated source-cache lookup.

    The private donor fast path may pass this object to the preparation entry
    point so it does not immediately repeat the same metadata/mmap lookup.
    A plain user-created tuple is never trusted as a validated cache result;
    ordinary callers keep the original lookup behavior.
    """
    __slots__ = ()

    def __new__(cls, values, precision, variant):
        return tuple.__new__(cls, (values, precision, variant))


def _reference_source_cache_dir() -> Path | None:
    value = os.environ.get(_REFERENCE_SOURCE_CACHE_ENV)
    if not value:
        return None
    path = Path(value).resolve()
    return path if path.is_dir() or not path.exists() else None


def _pair_locator_path(kind, identity):
    """Share a completed pair across canonical coordinator work directories."""
    directory = _reference_source_cache_dir()
    if directory is None or directory.parent.parent.name != 'reference-pipeline':
        return None
    # The coordinator already resolved cache_roots. Keep ad-hoc source-cache
    # callers local, and never discover historical work directories by scan.
    cache_base = directory.parent.parent.parent
    token = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode('utf8')).hexdigest()[:24]
    return cache_base/'reference-source-pairs'/kind/(token+'.json')


def _read_pair_locator(kind, identity):
    path = _pair_locator_path(kind, identity)
    if path is None:
        return None
    try:
        value = json.loads(path.read_text(encoding='utf8'))
        if (isinstance(value, dict) and value.get('schema') == 'reference_source_pair_locator.v1'
                and value.get('kind') == kind and value.get('identity') == identity
                and isinstance(value.get('record'), dict)):
            return value['record']
    except (OSError, ValueError, TypeError):
        pass
    return None


def _publish_pair_locator(kind, identity, record):
    path = _pair_locator_path(kind, identity)
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f'.{path.stem}.{os.getpid()}.tmp.json')
        temporary.write_text(json.dumps({'schema': 'reference_source_pair_locator.v1',
            'kind': kind, 'identity': identity, 'record': record}, sort_keys=True,
            separators=(',', ':'), allow_nan=False)+'\n', encoding='utf8')
        temporary.replace(path)
    except (OSError, ValueError, TypeError):
        pass  # Optional reuse must not fail a valid producer.


def _pair_array_stat(path):
    stat = path.stat()
    return {'file_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def _pair_array_crc(values):
    return zlib.crc32(memoryview(np.ascontiguousarray(values)).cast('B'))


def _publish_sparse_pair(directory, identity, arrays):
    if _pair_locator_path('sparse-appearance', identity) is None:
        return
    try:
        record = {'directory': str(directory), 'arrays': {
            name: {**_pair_array_stat(directory/(name+'.npy')), 'crc32': _pair_array_crc(values)}
            for name, values in arrays.items()}}
        _publish_pair_locator('sparse-appearance', identity, record)
    except (OSError, ValueError, TypeError):
        pass


def _cache_identity(run, *, root, precision, transform, beauty_identity, shape):
    # ``open_source_for_run`` already verifies this closure during a cache
    # write. Re-check the small JSON/delta closure on every cache load so a
    # stale preparation cannot silently borrow a source retouch.
    retouch = retouch_identity(run, root=root)
    return {
        'schema': _REFERENCE_SOURCE_CACHE_SCHEMA,
        'source_sha256': run['source']['sha256'],
        'source_precision': precision['source_precision'],
        'decode_version': precision['decode_version'],
        'native_size': [int(shape[1]), int(shape[0])],
        'source_display': copy.deepcopy(transform),
        'native_source_retouch': retouch,
        'portrait_beauty': copy.deepcopy(beauty_identity),
    }


def _load_reference_source_cache(run, *, root, transform, beauty_identity, writable=True):
    """Load a coordinator-owned source array when its identity still binds.

    The array digest is recorded at write time but deliberately not recomputed
    for every stage: this is an ephemeral, single-writer work directory and
    repeated full-file SHA was the dominant avoidable hot-path cost. Shape,
    dtype, byte size and the complete source identity are checked on load;
    final frozen assets still carry their normal content hashes.
    """
    directory = _reference_source_cache_dir()
    if directory is None:
        return None
    metadata_path = directory / 'metadata.json'
    try:
        metadata = json.loads(metadata_path.read_text(encoding='utf8'))
        if metadata.get('schema') != _REFERENCE_SOURCE_CACHE_SCHEMA:
            return None
        requested_variant = 'beauty' if beauty_identity is not None else 'display'
        # If an earlier worker was interrupted after publishing the display
        # variant but before replaying Beauty, reuse that display array and
        # finish the small Beauty step instead of decoding the RAW again.
        variants = [requested_variant] + (['display'] if requested_variant == 'beauty' else [])
        variant = next((candidate for candidate in variants
                        if isinstance(metadata.get('variants', {}).get(candidate), dict)
                        and (directory / f'source-{candidate}.npy').is_file()), None)
        if variant is None:
            return None
        name = f'source-{variant}.npy'
        record = metadata['variants'][variant]
        path = directory / name
        expected = dict(record.get('identity') or {})
        if expected.get('source_sha256') != run['source']['sha256']:
            return None
        variant_beauty = beauty_identity if variant == 'beauty' else None
        if expected.get('source_display') != transform or expected.get('portrait_beauty') != variant_beauty:
            return None
        if expected.get('native_source_retouch') != retouch_identity(run, root=root):
            return None
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        shape = tuple(array.shape)
        if (array.dtype != np.dtype('float32') or array.ndim != 3 or shape[-1] != 3
                or [int(shape[1]), int(shape[0])] != expected.get('native_size')
                or path.stat().st_size != int(record.get('file_bytes', -1))):
            return None
        precision = {
            'source_precision': expected['source_precision'],
            'working_precision': 'float32_linear',
            'decode_version': expected['decode_version'],
        }
        if expected.get('native_source_retouch'):
            precision.update(native_source_retouch=copy.deepcopy(expected['native_source_retouch']),
                             native_retouch_consumptions=1)
        # Most reference stages only read the bound source (resize, sampling
        # and F/B estimation).  Keep that hot path on the mmap instead of
        # copying a several-hundred-MiB native frame in every short-lived
        # worker.  Callers that own an in-place operation retain the explicit
        # writable=True contract.
        values = np.array(array, dtype=np.float32, copy=True) if writable else array
        return _ValidatedReferenceSourceCache(values, precision, variant)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def _write_reference_source_cache(directory, identity, values, *, variant):
    """Atomically publish one source variant and its binding metadata."""
    directory = Path(directory).resolve(); directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'source-{variant}.npy'
    temporary = directory / f'.source-{variant}.{os.getpid()}.tmp.npy'
    np.save(temporary, np.asarray(values, dtype=np.float32), allow_pickle=False)
    temporary.replace(path)
    record = {'identity': copy.deepcopy(identity), 'file_bytes': path.stat().st_size,
              'dtype': 'float32', 'shape': list(values.shape)}
    metadata_path = directory / 'metadata.json'
    try:
        metadata = json.loads(metadata_path.read_text(encoding='utf8'))
    except (OSError, ValueError, TypeError):
        metadata = {'schema': _REFERENCE_SOURCE_CACHE_SCHEMA, 'variants': {}}
    metadata.setdefault('schema', _REFERENCE_SOURCE_CACHE_SCHEMA)
    metadata.setdefault('variants', {})[variant] = record
    temporary_meta = directory / f'.metadata.{os.getpid()}.tmp.json'
    temporary_meta.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    temporary_meta.replace(metadata_path)


def source_for_reference_prepare(run, *, root=None, binding=None, apply_beauty=False,
                                 writable=True, _validated_cache=None):
    """Decode approved source, apply its explicit native D once, then fit T/B/S.

    Every preparation stage starts from the same original and approved delta.
    This entry point never consumes a saved display-domain Fp/B pair. Fitting
    requests generated beauty explicitly; immutable Fp/B freezing keeps it off.
    """
    from display_transform import source_transform_for_run, prepare_source_display_inplace
    from portrait_beauty_layer import identity_for_run
    transform = source_transform_for_run(run)
    beauty_identity = identity_for_run(run)
    requested_beauty = beauty_identity if apply_beauty else None
    # The donor path can hand back the exact immutable result of the lookup it
    # just performed. Only the private marker type is eligible; arbitrary
    # tuples are ignored and go through the normal strict loader. If its
    # variant or endpoint does not match this request, reload and revalidate
    # rather than consuming an alias from a different source/Beauty control.
    expected_variant = 'beauty' if requested_beauty is not None else 'display'
    if isinstance(_validated_cache, _ValidatedReferenceSourceCache):
        # A marker from the exact lookup may be consumed once. A mismatched
        # variant is rejected and reloaded, preserving display fallback when
        # the ordinary loader legitimately needs to replay Beauty.
        cached = (_validated_cache if _validated_cache[2] == expected_variant else
                  _load_reference_source_cache(run, root=root, transform=transform,
                      beauty_identity=requested_beauty, writable=writable))
    else:
        cached = _load_reference_source_cache(run, root=root, transform=transform,
            beauty_identity=requested_beauty, writable=writable)
    if cached is not None and writable and not cached[0].flags.writeable:
        cached = _ValidatedReferenceSourceCache(
            np.array(cached[0], dtype=np.float32, copy=True), cached[1], cached[2])
    if cached is not None:
        # Keep all existing precision, binding and source/Beauty checks below;
        # the fast path only removes duplicate lookup I/O.
        source, precision, cached_variant = cached
        baseline = run['options']['appearance_revision']['design_reconstruction_plan']['source_baseline']
        if (precision['source_precision'] != baseline['source_precision']
                or precision['decode_version'] != baseline['decode_version']):
            raise ValueError('Cached source precision/decoder differs from its preparation baseline')
        retouch = precision.get('native_source_retouch')
        if binding is not None:
            if (binding.get('source') != run['source']['sha256']
                    or binding.get('decode_version') != precision['decode_version']
                    or binding.get('source_display') != transform
                    or 'native_source_retouch' not in binding
                    or binding['native_source_retouch'] != retouch
                    or binding.get('portrait_beauty') != beauty_identity):
                raise ValueError('Preparation parameters belong to a different approved source or source transform')
        if apply_beauty and beauty_identity is not None and cached_variant != 'beauty':
            from portrait_beauty_layer import apply_for_run
            source, _execution = apply_for_run(source, run)
            cache_directory = _reference_source_cache_dir()
            if cache_directory is not None:
                identity = _cache_identity(run, root=root, precision=precision, transform=transform,
                    beauty_identity=beauty_identity, shape=source.shape)
                _write_reference_source_cache(cache_directory, identity, source, variant='beauty')
        return source, {**precision, 'source_display_consumptions': 1,
            'portrait_beauty': beauty_identity,
            'portrait_beauty_consumptions': int(bool(apply_beauty and beauty_identity)),
            'reference_working_source': 'beauty_appearance' if apply_beauty and beauty_identity else 'original_display_source'}, transform
    source, precision = open_source_for_run(run, root=root)
    baseline = run['options']['appearance_revision']['design_reconstruction_plan']['source_baseline']
    if (precision['source_precision'] != baseline['source_precision']
            or precision['decode_version'] != baseline['decode_version']):
        raise ValueError('Decoded source precision/decoder differs from its preparation baseline')
    retouch = precision.get('native_source_retouch')
    if binding is not None:
        if (binding.get('source') != run['source']['sha256']
                or binding.get('decode_version') != precision['decode_version']
                or binding.get('source_display') != transform
                or 'native_source_retouch' not in binding
                or binding['native_source_retouch'] != retouch
                or binding.get('portrait_beauty') != beauty_identity):
            raise ValueError('Preparation parameters belong to a different approved source or source transform')
    prepare_source_display_inplace(source, transform)
    cache_directory = _reference_source_cache_dir()
    if cache_directory is not None:
        identity = _cache_identity(run, root=root, precision=precision, transform=transform,
            beauty_identity=beauty_identity, shape=source.shape)
        # The display variant is the exact pre-Beauty source used by the native
        # freeze stage. It is written before optional Beauty replay.
        _write_reference_source_cache(cache_directory, {**identity, 'portrait_beauty': None}, source, variant='display')
    if apply_beauty:
        from portrait_beauty_layer import apply_for_run
        source, _execution = apply_for_run(source, run)
        if cache_directory is not None and beauty_identity is not None:
            _write_reference_source_cache(cache_directory, identity, source, variant='beauty')
    return source, {**precision, 'source_display_consumptions': 1,
        'portrait_beauty': beauty_identity,
        'portrait_beauty_consumptions': int(bool(apply_beauty and beauty_identity)),
        'reference_working_source': 'beauty_appearance' if apply_beauty and beauty_identity else 'original_display_source'}, transform


def _appearance_estimator_identity():
    from importlib.metadata import version
    from runtime_resources import code_path
    from native_cuda_fields import execution
    return {'producer_sha256': sha256_file(code_path('source_foreground.py')),
            'cuda_solver_sha256': sha256_file(code_path('native_cuda_matting.py')),
            'field_backend': execution()['backend'],
            'pymatting': version('PyMatting'), 'numpy': np.__version__,
            'scipy': version('scipy'), 'tile': 512, 'overlap': 128, 'halo': 96,
            'trimap': 'zero_and_one_known_fractional_unknown'}


def display_pair_identity(run, *, precision, transform, alpha_sha256, native_size):
    """Bind only immutable pre-Beauty source pixels and their F/B estimator."""
    identity = _cache_identity(run, root=None, precision=precision, transform=transform,
        beauty_identity=None, shape=(native_size[1], native_size[0], 3))
    identity.update(schema='reference_display_pair_assets.v1',
        working_domain='display_linear_sRGB', appearance_variant='display',
        immutable_source_pair_contains_beauty=False, alpha_sha256=alpha_sha256,
        estimator=_appearance_estimator_identity())
    return identity


_VERIFIED_NATIVE_ARRAY_PROOF = object()


def _native_asset_file_identity(path):
    stat = Path(path).stat()
    return (int(stat.st_dev), int(stat.st_ino), int(stat.st_size),
            int(stat.st_mtime_ns), int(stat.st_ctime_ns))


@dataclass(frozen=True, slots=True)
class VerifiedNativeArrayAsset:
    """A verified immutable archive, with no materialized RGB pixel buffer."""
    path: Path
    sha256: str
    shape: tuple
    dtype: str
    fortran_order: bool
    payload_bytes: int
    member_crc32: int
    file_identity: tuple
    _proof: object = field(repr=False, compare=False)

    def __deepcopy__(self, memo):
        return self

    def validate_asset(self, asset):
        if (self._proof is not _VERIFIED_NATIVE_ARRAY_PROOF
                or Path(asset['path']).resolve() != self.path
                or asset['sha256'] != self.sha256
                or tuple(asset['size'][::-1])+(3,) != self.shape
                or _native_asset_file_identity(self.path) != self.file_identity):
            raise ValueError('Verified native archive binding changed before materialization')


def _verify_native_rgb_archive(asset, expected_shape):
    """Keep full SHA and ZIP CRC checks while reading at most 4 MiB at a time."""
    path = Path(asset['path']).resolve()
    before = _native_asset_file_identity(path)
    if sha256_file(path) != asset['sha256']:
        raise ValueError('Frozen native RGB archive content identity changed')
    with zipfile.ZipFile(path) as archive:
        if archive.namelist().count('values.npy') != 1:
            raise ValueError('Frozen native RGB archive needs one values member')
        member = archive.getinfo('values.npy')
        with archive.open(member) as stream:
            version = np.lib.format.read_magic(stream)
            header = {(1, 0): np.lib.format.read_array_header_1_0,
                      (2, 0): np.lib.format.read_array_header_2_0}.get(version)
            if header is None:
                raise ValueError('Unsupported native NPY header version')
            shape, fortran, dtype = header(stream)
            if dtype != np.dtype('float32') or shape != expected_shape:
                raise ValueError('Frozen native RGB archive payload layout changed')
            payload_bytes = int(np.prod(shape, dtype=np.int64))*dtype.itemsize
            if member.file_size-stream.tell() != payload_bytes:
                raise ValueError('Frozen native RGB archive payload length changed')
            consumed = 0
            # Read to EOF: ZipExtFile validates the member CRC, including for
            # STORED archives. Do not replace this with a header-only check.
            while chunk := stream.read(4*1024*1024):
                consumed += len(chunk)
            if consumed != payload_bytes:
                raise ValueError('Frozen native RGB archive payload was truncated')
    if _native_asset_file_identity(path) != before:
        raise ValueError('Frozen native RGB archive changed during verification')
    return VerifiedNativeArrayAsset(path, asset['sha256'], shape, dtype.str, bool(fortran),
        payload_bytes, int(member.CRC), before, _VERIFIED_NATIVE_ARRAY_PROOF)


def load_display_pair_assets(run, identity):
    return _load_display_pair_assets(run, identity, freeze_assets=False)


def _load_display_pair_assets(run, identity, *, freeze_assets):
    """Reuse a frozen F/B closure only when all current source bindings match.

    The coordinator cache merely locates owned immutable NPZs. Their ordinary
    content identities use the existing stat-bound sidecar; no model or disk
    scan is introduced. A miss retains the current estimator path.
    """
    plan = run['options']['appearance_revision']['design_reconstruction_plan']
    baseline = plan.get('source_baseline') or {}
    saved = baseline.get('appearance_pair')
    if (isinstance(saved, dict) and baseline.get('schema') == 'appearance_source_baseline.v1'
            and baseline.get('working_domain') == 'display_linear_sRGB'
            and baseline.get('source_display_applied_once') is True):
        # Saved plans already materialize these roles. Store no duplicate live
        # locators in the receipt, so relocation or cache removal remains safe.
        assets = baseline.get('assets') or {}
        saved = {**saved, 'assets': {
            'foreground_premultiplied': assets.get('foreground_premultiplied'),
            'old_environment': assets.get('old_environment'),
            'edge_applicability': (plan.get('mask_roles') or {}).get('edge_applicability')}}
    else:
        saved = None
    candidates = [saved]
    directory = _reference_source_cache_dir()
    if directory is not None:
        try:
            candidates.append(json.loads((directory/'prepared-display-pair.json').read_text(encoding='utf8')))
        except (OSError, ValueError, TypeError):
            pass
    candidates.append(_read_pair_locator('display-assets', identity))
    native = tuple(identity['native_size'])
    for record in candidates:
        if not isinstance(record, dict) or record.get('identity') != identity:
            continue
        try:
            if (type(record.get('tiles')) is not int or record['tiles'] < 0
                    or not np.isfinite(record.get('source_reconstruction_max_error', float('nan')))
                    or record['source_reconstruction_max_error'] < 0):
                raise ValueError('Frozen display pair estimator receipt changed')
            assets = record['assets']
            values = {}
            for name, channels, representation in (('foreground_premultiplied', 3, 'premultiplied_RGB'),
                    ('old_environment', 3, 'straight_RGB'), ('edge_applicability', None, 'edge_confidence')):
                asset = assets[name]
                if (not isinstance(asset, dict) or asset.get('size') != list(native)
                        or asset.get('coordinate_space') != 'source_canvas'
                        or asset.get('working_space') != 'display_linear_sRGB'
                        or asset.get('representation') != representation
                        or asset['sha256'] != record['asset_sha256'][name]
                        or sha256_file(asset['path']) != asset['sha256']):
                    raise ValueError('Frozen display pair asset binding changed')
                shape = (native[1], native[0], channels) if channels else (native[1], native[0])
                if freeze_assets and name == 'old_environment':
                    values[name] = _verify_native_rgb_archive(asset, shape)
                    continue
                with np.load(asset['path'], allow_pickle=False) as resource:
                    array = resource['values']
                if array.dtype != np.dtype('float32') or array.shape != shape:
                    raise ValueError('Frozen display pair payload changed')
                # Freeze may reuse its foreground buffer for a new delta; NPZ
                # loading owns these arrays, leaving the shared bytes immutable.
                values[name] = array
            return values, copy.deepcopy(record)
        except (OSError, ValueError, TypeError, KeyError, EOFError, zipfile.BadZipFile, zlib.error):
            continue
    return None


def prepare_display_pair_for_freeze(source, alpha, run, identity):
    """Freeze reuses a verified old-B asset; its pixels are not a matching input."""
    reused = _load_display_pair_assets(run, identity, freeze_assets=True)
    if reused is not None:
        values, record = reused
        print('Reusing immutable pre-Beauty F/B assets without an old-B buffer', flush=True)
        return (values['old_environment'], values['foreground_premultiplied'],
                values['edge_applicability'], record['tiles'],
                record['source_reconstruction_max_error'], record)
    return prepare_display_pair(source, alpha, run, identity)


def prepare_display_pair(source, alpha, run, identity):
    """Prepare the same pre-Beauty F/B pixels, or load their immutable endpoint."""
    reused = load_display_pair_assets(run, identity)
    if reused is not None:
        values, record = reused
        print('Reusing immutable pre-Beauty display F/B assets', flush=True)
        return (values['old_environment'], values['foreground_premultiplied'],
                values['edge_applicability'], record['tiles'],
                record['source_reconstruction_max_error'], record)
    from source_foreground import estimate_source_background
    trimap = np.full(alpha.shape, 128, np.uint8)
    trimap[alpha == 0], trimap[alpha == 1] = 0, 255
    background, foreground, support, tiles, _, _ = estimate_source_background(source, alpha, trimap)
    reconstruction_error = 0.
    for top in range(0, len(source), 128):
        restored = foreground[top:top+128]+(1-alpha[top:top+128])[...,None]*background[top:top+128]
        reconstruction_error = max(reconstruction_error, float(np.max(np.abs(restored-source[top:top+128]))))
    return background, foreground, support, tiles, reconstruction_error, None


def publish_display_pair_assets(identity, assets, *, tiles, reconstruction_error):
    """Publish the small locator after the owned native asset set is complete."""
    record = {'identity': copy.deepcopy(identity), 'assets': copy.deepcopy(assets),
              'asset_sha256': {name: asset['sha256'] for name, asset in assets.items()},
              'tiles': int(tiles), 'source_reconstruction_max_error': float(reconstruction_error)}
    directory = _reference_source_cache_dir()
    if directory is not None:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            temporary = directory/f'.prepared-display-pair.{os.getpid()}.tmp.json'
            temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
            temporary.replace(directory/'prepared-display-pair.json')
        except OSError:
            pass  # A disposable locator must not fail valid owned source assets.
    _publish_pair_locator('display-assets', identity, record)
    return record


def background_for_reference_prepare(source, alpha, run, *, binding, precision,
                                     transform, apply_beauty=False, root=None):
    """Reuse the existing F/B estimator's immutable appearance-domain pair.

    Only the coordinator-owned preparation cache is eligible. Reference T/B/S
    fields never enter this identity: native_baseline and native_target start
    with identical pixels. Cache just fractional-coverage measurements; known
    foreground/background pixels are exact source copies. This avoids writing
    two additional full native RGB frames to the work disk.
    """
    from source_foreground import estimate_source_background

    started = time.perf_counter()
    beauty = precision.get('portrait_beauty')
    consumed = int(bool(apply_beauty and beauty))
    if (precision.get('source_display_consumptions') != 1
            or precision.get('portrait_beauty_consumptions') != consumed
            or source.dtype != np.dtype('float32') or alpha.dtype != np.dtype('float32')
            or source.shape != (*alpha.shape, 3)
            or binding.get('source') != run['source']['sha256']
            or binding.get('decode_version') != precision['decode_version']
            or binding.get('source_display') != transform
            or binding.get('native_source_retouch') != precision.get('native_source_retouch')
            or binding.get('portrait_beauty') != beauty):
        raise ValueError('Appearance F/B requires its bound display/Beauty source variant')
    plan = run['options']['appearance_revision']['design_reconstruction_plan']
    alpha_sha = plan['source_baseline']['assets']['foreground_alpha']['sha256']
    if binding.get('alpha') != alpha_sha:
        raise ValueError('Appearance F/B belongs to a different baseline alpha')
    identity = _cache_identity(run, root=root, precision=precision, transform=transform,
        beauty_identity=beauty if consumed else None, shape=source.shape)
    identity.update(schema='reference_appearance_pair.v1', working_domain='display_linear_sRGB',
        appearance_variant='beauty' if consumed else 'display', alpha_sha256=alpha_sha,
        estimator=_appearance_estimator_identity())
    cache_root = _reference_source_cache_dir()
    token = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:24]
    directory = cache_root/'appearance-pairs'/token if cache_root is not None else None
    unknown = np.flatnonzero(((alpha != 0) & (alpha != 1)).ravel())
    shapes = {'background': (len(unknown), 3), 'foreground': (len(unknown), 3),
              'edge-support': (len(unknown),)}
    cached = None
    shared_pair = _read_pair_locator('sparse-appearance', identity)
    candidates = [(directory, None)] if directory is not None else []
    if (shared_pair is not None and isinstance(shared_pair.get('directory'), str)
            and isinstance(shared_pair.get('arrays'), dict)):
        try:
            candidates.append((Path(shared_pair['directory']), shared_pair['arrays']))
        except (ValueError, TypeError, KeyError):
            pass
    cache_scope = None
    for candidate, integrity in candidates:
        try:
            metadata = json.loads((candidate/'metadata.json').read_text(encoding='utf8'))
            if isinstance(metadata, dict) and metadata.get('identity') == identity:
                cached = {}
                for name, shape in shapes.items():
                    path = candidate/(name+'.npy')
                    values = np.load(path, mmap_mode='r', allow_pickle=False)
                    if (values.dtype != np.dtype('float32') or values.shape != shape
                            or path.stat().st_size != metadata['arrays'][name]['file_bytes']):
                        raise ValueError('Incomplete appearance F/B cache')
                    if integrity is not None:
                        if (integrity[name] != {**_pair_array_stat(path), 'crc32': _pair_array_crc(values)}):
                            raise ValueError('Shared appearance F/B measurement changed')
                    cached[name] = values
                tiles = int(metadata['tiles'])
                cache_scope = 'shared' if integrity is not None else 'preparation'
                cached_directory = candidate
                break
        except (OSError, ValueError, TypeError, KeyError, EOFError):
            cached = None
    if cached is not None:
        old_background = np.array(source, dtype=np.float32, copy=True)
        foreground = np.array(source, dtype=np.float32, copy=True)
        edge_support = np.zeros_like(alpha)
        old_background[alpha == 1] = 0
        foreground[alpha == 0] = 0
        old_background.reshape(-1, 3)[unknown] = cached['background']
        foreground.reshape(-1, 3)[unknown] = cached['foreground']
        edge_support.ravel()[unknown] = cached['edge-support']
        receipt = {'cache_hit': True, 'appearance_variant': identity['appearance_variant'],
                   'directory': str(cached_directory), 'cache_scope': cache_scope,
                   'unknown_pixels': len(unknown),
                   'elapsed_seconds': round(time.perf_counter()-started, 6),
                   'estimator_calls': 0}
        if shared_pair is None and cache_scope == 'preparation':
            _publish_sparse_pair(cached_directory, identity, cached)
        print('Reusing bound appearance F/B edge measurements', len(unknown), flush=True)
        return old_background, foreground, edge_support, tiles, receipt

    trimap = np.full(alpha.shape, 128, np.uint8)
    trimap[alpha == 0], trimap[alpha == 1] = 0, 255
    print('Re-estimating only unknown appearance-domain edges', len(unknown), flush=True)
    estimate_started = time.perf_counter()
    old_background, foreground, edge_support, tiles, _, _ = estimate_source_background(source, alpha, trimap)
    estimate_seconds = time.perf_counter()-estimate_started
    cache_write_error = None
    if directory is not None:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            arrays = {}
            saved_values = {}
            for name, values in (('background', old_background.reshape(-1, 3)),
                                 ('foreground', foreground.reshape(-1, 3)),
                                 ('edge-support', edge_support.ravel())):
                path = directory/(name+'.npy')
                temporary = directory/f'.{name}.{os.getpid()}.tmp.npy'
                saved_values[name] = values[unknown]
                np.save(temporary, saved_values[name], allow_pickle=False)
                temporary.replace(path)
                arrays[name] = {'file_bytes': path.stat().st_size}
            # Publish completion only after all three immutable measurements exist.
            temporary = directory/f'.metadata.{os.getpid()}.tmp.json'
            temporary.write_text(json.dumps({'identity': identity, 'tiles': tiles,
                'arrays': arrays}, sort_keys=True, indent=2)+'\n', encoding='utf8')
            temporary.replace(directory/'metadata.json')
            _publish_sparse_pair(directory, identity, saved_values)
        except OSError as exc:
            # A disposable acceleration cache must not fail a valid composite.
            cache_write_error = str(exc)
    receipt = {'cache_hit': False, 'appearance_variant': identity['appearance_variant'],
               'directory': str(directory) if directory is not None else None,
               'unknown_pixels': len(unknown), 'estimate_seconds': round(estimate_seconds, 6),
               'elapsed_seconds': round(time.perf_counter()-started, 6), 'estimator_calls': 1}
    if cache_write_error:
        receipt['cache_write_error'] = cache_write_error
    return old_background, foreground, edge_support, tiles, receipt


def enabled(plan):
    return (plan.get('execution') or {}).get('source_reference') == 'paired_native_baseline'


@lru_cache(maxsize=2)
def _baseline(fp_file, alpha_file, bg_file, target_size):
    """Reconstruct one immutable paired baseline keyed by file identity.

    The locator paths are deliberately excluded from ``_PlaneFile`` equality.
    Materialized child revisions commonly hard-link the same Fp/A/B payloads
    under new paths; using those paths as an LRU key rebuilt the same full RGB
    array and retained both copies in one long-lived backend process.
    """
    from design_reconstruction import _planes
    fp = _planes(fp_file.path, fp_file.identity[2])['values']
    alpha = _planes(alpha_file.path, alpha_file.identity[2])['values'][0]
    background = _planes(bg_file.path, bg_file.identity[2])['values']
    width, height = alpha.size
    native = tuple(target_size) == (width, height)
    if native:
        original = np.empty((height, width, 3), np.float32)
    else:
        # Reconstruct the paired product at native resolution before reducing.
        # A proxy needs one native channel at a time, not an entire native RGB
        # image retained just to make a small preview.
        original = np.empty((target_size[1], target_size[0], 3), np.float32)
        plane = np.empty((height, width), np.float32)
    if native:
        # All three channels share alpha. Reconstruct each bounded HWC band
        # directly from the existing read-only NumPy backing, without Pillow
        # bridges or three repeated uncovered-alpha computations.
        for top in range(0, height, 256):
            bottom = min(height, top + 256)
            uncovered = 1 - np.asarray(alpha._array[top:bottom, :], dtype=np.float32)
            band = original[top:bottom]
            for channel in range(3):
                band[..., channel] = fp[channel]._array[top:bottom, :]
                band[..., channel] += uncovered * background[channel]._array[top:bottom, :]
    else:
        # Proxy reduction still reconstructs the native paired product one
        # channel at a time before the original Pillow resize. This bounds
        # scratch memory without changing the premultiplied alpha product.
        for channel in range(3):
            for top in range(0, height, 256):
                bottom = min(height, top + 256)
                uncovered = 1 - np.asarray(alpha._array[top:bottom, :], dtype=np.float32)
                plane[top:bottom] = fp[channel]._array[top:bottom, :]
                plane[top:bottom] += uncovered * background[channel]._array[top:bottom, :]
            original[..., channel] = np.asarray(Image.fromarray(plane).resize(
                target_size, Image.Resampling.BILINEAR), np.float32)
    return original


def source_for_plan(plan, size=None):
    baseline=plan.get('source_baseline') or {}
    if (baseline.get('schema')!='appearance_source_baseline.v1'
            or baseline.get('working_domain')!='display_linear_sRGB'):
        raise ValueError('Source replay requires the current explicit display-domain baseline')
    assets = (plan.get('source_baseline') or {}).get('assets')
    if not isinstance(assets,dict):raise ValueError('Current source replay requires its immutable original paired assets')
    fp, alpha, bg = (assets[key] for key in
        ('foreground_premultiplied','foreground_alpha','old_environment'))
    target = tuple(size or alpha['size'])
    files = [Path(asset['path']) for asset in (fp, alpha, bg)]
    from design_reconstruction import _plane_file
    locators = tuple(_plane_file(path) for path in files)
    return _baseline(*locators, target)


def background_for_view(source_view, foreground_view, alpha_view):
    """Effective old B for the paired reduced view, not independent B resampling."""
    uncovered = 1 - alpha_view
    contribution = np.maximum(source_view - foreground_view, 0)
    return np.divide(contribution, uncovered[..., None], out=np.zeros_like(contribution),
                     where=uncovered[..., None] > 0)


def prepare_source_only(source_linear, run, precision):
    """Establish a current local source plan and consume its source display once."""
    from display_transform import source_transform_for_precision,validate_source_transform,display_linear
    from raw_support import RGB_DECODE_VERSION
    if run.get('reference') or ((run.get('options') or {}).get('appearance_revision') or {}).get('design_reconstruction_plan'):
        raise ValueError('Source-only preparation cannot inherit reference rendering assets')
    source=run.get('source') or {}
    if not source.get('sha256') or precision.get('decode_version')!=RGB_DECODE_VERSION:
        raise ValueError('Current source-only preparation requires a bound source and current decoder')
    if precision.get('source_precision') not in {'raw16_linear','raster8_srgb'}:
        raise ValueError('Unsupported source precision; decode the original through the current source decoder')
    existing=(run.get('options') or {}).get('source_only_plan')
    if existing:
        if (existing.get('schema')!='source_only_display_plan.v1' or existing.get('working_domain')!='display_linear_sRGB'
                or existing.get('source_sha256')!=source['sha256']
                or existing.get('decode_version')!=RGB_DECODE_VERSION
                or existing.get('source_precision')!=precision['source_precision']
                or existing.get('native_size')!=list(source_linear.shape[1::-1])
                or existing.get('reference_assets_used') is not False):
            raise ValueError('Source-only plan is not bound to the current source')
        transform=validate_source_transform(existing['source_display'], precision['source_precision'])
        if existing.get('source_display_applied_once') is not True:
            raise ValueError('Source-only baseline must record one source preparation consumption')
    else:transform=source_transform_for_precision(precision['source_precision'])
    values=display_linear(np.asarray(source_linear,np.float32),transform)
    plan={'schema':'source_only_display_plan.v1','working_domain':'display_linear_sRGB',
        'source_sha256':source['sha256'],'native_size':list(values.shape[1::-1]),
        'source_precision':precision['source_precision'],'decode_version':RGB_DECODE_VERSION,
        'source_display':transform,'source_display_applied_once':True,'reference_assets_used':False}
    return values,plan
