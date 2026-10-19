"""Disposable native T+B acceleration; never a saved appearance dependency."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
import zlib

import numpy as np

from asset_identity import sha256_file
from runtime_resources import code_path

SCHEMA = 'reference_pre_scene_target_cache.v1'
BLOCK_COLUMNS = 128
CODE = ('render_reference_colour_composite.py', 'reference_colour_fit.py',
        'reference_colour_blend.py', 'reference_material_palette.py',
        'colour_space.py', 'portrait_beauty_layer.py', 'reference_pre_scene_cache.py',
        'native_cuda_fields.py', 'native_cuda_sampling.py')


def _json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode('utf8')).hexdigest()


def _crc(values):
    return zlib.crc32(memoryview(np.ascontiguousarray(values)).cast('B'))


def _stat(path):
    stat = path.stat()
    return {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def _context(parameters, colour_receipt, native, proxy):
    from native_source_baseline import _reference_source_cache_dir

    root = _reference_source_cache_dir()
    if root is None:
        return None
    variant = 'beauty' if parameters['binding'].get('portrait_beauty') else 'display'
    endpoint = root/f'source-{variant}.npy'
    metadata = json.loads((root/'metadata.json').read_text(encoding='utf8'))
    source_record = metadata['variants'][variant]
    source_identity = source_record['identity']
    binding = parameters['binding']
    if (source_identity['source_sha256'] != binding['source']
            or source_identity['source_display'] != binding['source_display']
            or source_identity.get('portrait_beauty') != binding.get('portrait_beauty')
            or source_identity['native_size'] != list(native)):
        return None
    stat = _stat(endpoint)
    if stat['bytes'] != source_record['file_bytes']:
        return None
    identity = {'schema': SCHEMA, 'parameters': _json_digest(parameters),
        'source_endpoint': source_identity, 'endpoint_stat': stat,
        'colour_asset': colour_receipt['asset']['sha256'],
        'colour_guard': colour_receipt.get('hair_correction_guard'),
        'native_size': list(native), 'proxy_size': list(proxy),
        'sampling': {'band_rows': 128, 'block_columns': BLOCK_COLUMNS,
                     'coordinates': 'native_pixel_centres_original_proxy_owner_guard'},
        'producer_code': {name: sha256_file(code_path(name)) for name in CODE}}
    from native_cuda_fields import execution
    identity['field_backend'] = execution()['backend']
    from native_cuda_sampling import sampling_profile
    identity['sampling_execution_profile'] = sampling_profile()
    directory = root/'pre-scene-target'/_json_digest(identity)[:24]
    return directory, endpoint, identity


class PreSceneWriter:
    def __init__(self, context):
        self.directory, _, self.identity = context
        self.directory.mkdir(parents=True, exist_ok=True)
        self.temporary = self.directory/f'.target-{os.getpid()}.tmp.npy'
        width, height = self.identity['native_size']
        self.values = np.lib.format.open_memmap(self.temporary, mode='w+',
            dtype=np.float32, shape=(height, width, 3))
        self.tiles = {}
        self.error = None
        self.write_seconds = 0.

    def write(self, box, values):
        if self.error is not None:
            return
        started = time.perf_counter()
        try:
            left, top, right, bottom = box
            self.values[top:bottom, left:right] = values
            self.tiles[str(top)] = {'box': list(box), 'crc32': _crc(values)}
        except (OSError, ValueError, TypeError) as exc:
            self.error = f'{type(exc).__name__}: {exc}'
        finally:
            self.write_seconds += time.perf_counter()-started

    def finish(self):
        started = time.perf_counter()
        try:
            if self.error is not None:
                return {'written': False, 'reason': self.error}
            self.values.flush()
            self.values._mmap.close()
            self.values = None
            target = self.directory/'target.npy'
            self.temporary.replace(target)
            record = {'schema': SCHEMA, 'identity': self.identity, 'target_stat': _stat(target),
                'tiles': self.tiles}
            temporary = self.directory/f'.metadata-{os.getpid()}.tmp.json'
            temporary.write_text(json.dumps(record, separators=(',', ':'))+'\n', encoding='utf8')
            temporary.replace(self.directory/'metadata.json')
            return {'written': True, 'tiles': len(self.tiles), 'bytes': target.stat().st_size,
                    'write_seconds': round(self.write_seconds, 6),
                    'publish_seconds': round(time.perf_counter()-started, 6)}
        except (OSError, ValueError, TypeError) as exc:
            return {'written': False, 'reason': f'{type(exc).__name__}: {exc}'}
        finally:
            if self.values is not None:
                self.values._mmap.close()
                self.values = None
            try:
                self.temporary.unlink(missing_ok=True)
            except OSError:
                pass


class PreSceneReader:
    def __init__(self, context):
        self.directory, endpoint, identity = context
        record = json.loads((self.directory/'metadata.json').read_text(encoding='utf8'))
        target = self.directory/'target.npy'
        if (record['schema'] != SCHEMA or record['identity'] != identity
                or record['target_stat'] != _stat(target)):
            raise ValueError('Temporary T+B identity or target bytes changed')
        width, height = identity['native_size']
        self.target = self.endpoint = None
        try:
            self.target = np.load(target, mmap_mode='r', allow_pickle=False)
            self.endpoint = np.load(endpoint, mmap_mode='r', allow_pickle=False)
            if any(value.dtype != np.float32 or value.shape != (height, width, 3)
                   for value in (self.target, self.endpoint)):
                raise ValueError('Temporary T+B target or source endpoint layout changed')
        except (OSError, ValueError, TypeError, EOFError):
            self.close()
            raise
        self.tiles = record['tiles']
        self.stats = {'available': True, 'hit_bands': 0, 'miss_bands': 0,
            'reused_opaque_pixels': 0, 'endpoint_mismatch_pixels': 0, 'corrupt_bands': 0}

    def read(self, box, working, alpha):
        result, stats = self._read_band(box, working, alpha)
        self._merge_stats(stats)
        return result

    def _merge_stats(self, stats):
        """The ordered consumer alone commits each completed band's counts."""
        for key, value in stats.items():
            self.stats[key] += value

    def _read_band(self, box, working, alpha):
        """Read independently; return counters without mutating shared state."""
        stats = {key: 0 for key in self.stats if key != 'available'}
        try:
            left, top, right, bottom = box
            tile = self.tiles[str(top)]
            if tile['box'] != list(box):
                raise ValueError('Temporary T+B sampling box changed')
            target = np.array(self.target[top:bottom, left:right], copy=True)
            if _crc(target) != tile['crc32']:
                stats['corrupt_bands'] += 1
                raise ValueError('Temporary T+B band checksum changed')
            opaque = alpha == 1
            endpoint = self.endpoint[top:bottom, left:right]
            equal = np.all(endpoint.view(np.uint32) == working.view(np.uint32), -1)
            valid = opaque & equal
            stats['endpoint_mismatch_pixels'] += int(np.count_nonzero(opaque & ~equal))
            # A few scattered hits would add work; only use a band that can
            # completely skip at least one original-coordinate T+B block.
            if not any(valid[:, first:min(first+BLOCK_COLUMNS, valid.shape[1])].all()
                       for first in range(0, valid.shape[1], BLOCK_COLUMNS)):
                stats['miss_bands'] += 1
                return None, stats
            stats['hit_bands'] += 1
            stats['reused_opaque_pixels'] += int(np.count_nonzero(valid))
            return (target, valid), stats
        except (OSError, ValueError, TypeError, KeyError, EOFError):
            stats['miss_bands'] += 1
            return None, stats

    def close(self):
        for name in ('target', 'endpoint'):
            value = getattr(self, name, None)
            if value is not None:
                value._mmap.close()
                setattr(self, name, None)


def open_writer(parameters, colour_receipt, native, proxy):
    try:
        context = _context(parameters, colour_receipt, native, proxy)
        return PreSceneWriter(context) if context is not None else None
    except (OSError, ValueError, TypeError, KeyError, EOFError):
        return None


def open_reader(parameters, colour_receipt, native, proxy):
    reader = None
    try:
        context = _context(parameters, colour_receipt, native, proxy)
        if context is not None:
            reader = PreSceneReader(context)
        return reader
    except (OSError, ValueError, TypeError, KeyError, EOFError):
        if reader is not None:
            reader.close()
        return None
