"""Lossless standard NPZ publication for native arrays and complete bundles."""
from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np

VALUES_ENCODING = 'standard_npz_single_values_stored.v1'


class NativeArrayRowWriter:
    """Publish one native array from ordered live row blocks, without scratch NPY.

    The producer owns this object until validation finishes. Each block is
    shape/dtype/finite checked before writing; completion covers every row.
    Its live proof cannot be supplied through a saved plan or JSON receipt.
    """
    def __init__(self, path, shape, *, dtype=np.float32):
        self.path = Path(path)
        self.shape, self.dtype = tuple(shape), np.dtype(dtype)
        if (len(self.shape) not in (2, 3) or any(int(n) != n or n <= 0 for n in self.shape)
                or len(self.shape) == 3 and self.shape[-1] != 3
                or self.dtype != np.dtype(np.float32)):
            raise ValueError('Native row writer requires a positive float32 scalar/RGB shape')
        self.shape = tuple(map(int, self.shape))
        descriptor, filename = tempfile.mkstemp(prefix=self.path.name+'.', suffix='.tmp.npz', dir=self.path.parent)
        os.close(descriptor)
        self._temporary = Path(filename)
        self._rows, self._published_stat = 0, None
        self._archive, self._stream = None, None
        try:
            self._archive = zipfile.ZipFile(self._temporary, 'w', compression=zipfile.ZIP_STORED)
            self._stream = self._archive.open('values.npy', 'w', force_zip64=True)
            np.lib.format.write_array_header_1_0(self._stream, {
                'descr': np.lib.format.dtype_to_descr(self.dtype),
                'fortran_order': False, 'shape': self.shape})
        except BaseException:
            self.close()
            raise

    def write(self, values):
        values = np.asarray(values)
        if (self._stream is None or values.dtype != self.dtype or values.ndim != len(self.shape)
                or values.shape[1:] != self.shape[1:] or not len(values)
                or self._rows+len(values) > self.shape[0]):
            raise ValueError('Native row blocks must cover the declared array once in row order')
        if not np.isfinite(values).all():
            raise ValueError('Appearance delta contains nonfinite values')
        self._stream.write(memoryview(np.ascontiguousarray(values)).cast('B'))
        self._rows += len(values)

    def publish(self):
        if self._stream is None or self._rows != self.shape[0]:
            raise ValueError('Cannot publish an incomplete native array')
        self._stream.close(); self._stream = None
        self._archive.close(); self._archive = None
        self._temporary.replace(self.path)
        stat = self.path.stat()
        self._published_stat = (stat.st_mtime_ns, stat.st_size, stat.st_ino)

    def validate_native_delta(self, path, native_size):
        """Validate this live publication; saved consumers still open the NPZ."""
        if (self._published_stat is None or self._rows != self.shape[0] or path is None
                or self.shape != (native_size[1], native_size[0], 3)
                or self.dtype != np.dtype(np.float32)
                or Path(path).resolve() != self.path.resolve()):
            raise ValueError('Appearance delta must be a complete live native float32 HWC publication')
        stat = self.path.stat()
        if self._published_stat != (stat.st_mtime_ns, stat.st_size, stat.st_ino):
            raise ValueError('Live native publication changed after its producer completed')

    def close(self):
        try:
            if self._stream is not None:
                self._stream.close(); self._stream = None
        finally:
            try:
                if self._archive is not None:
                    self._archive.close(); self._archive = None
            finally:
                if self._temporary.exists():
                    self._temporary.unlink()


def _save_array_bundle(path, members, *, compression, compresslevel=None, stored_members=frozenset()):
    """Share standard NPY members and atomic publication for fixed producers."""
    if not isinstance(members, Mapping) or not members:
        raise ValueError('Native array bundle requires named members')
    for name in members:
        if (not isinstance(name, str) or not name or name in {'.', '..'}
                or '/' in name or '\\' in name):
            raise ValueError('Native array members require a single-component name')
    path = Path(path)
    descriptor, filename = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp.npz', dir=path.parent)
    os.close(descriptor)
    temporary = Path(filename)
    try:
        with zipfile.ZipFile(temporary, 'w', compression=compression,
                             compresslevel=compresslevel) as archive:
            for name, values in members.items():
                entry = name+'.npy'
                if name in stored_members:
                    # ZipInfo selects the codec for this dense member. Sparse
                    # names retain the ZipFile's fixed DEFLATE1 setting.
                    entry = zipfile.ZipInfo(entry)
                    entry.compress_type = zipfile.ZIP_STORED
                with archive.open(entry, 'w', force_zip64=True) as stream:
                    np.lib.format.write_array(stream, np.asarray(values), allow_pickle=False)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_native_arrays(path, members):
    """Atomically publish native payloads as a complete standard stored NPZ.

    Preserve each member's dtype, byte order, shape, layout and float bits.
    Consumers retain their ordinary np.load keys; integer owners stay integer.
    """
    _save_array_bundle(path, members, compression=zipfile.ZIP_STORED)


def save_proxy_arrays(path, members):
    """Publish proxy fields/owners as DEFLATE1 and dense baseline RGB stored.

    This is a producer storage responsibility, not a user-selectable precision
    or quality mode. Native endpoints keep the separate fixed stored policy.
    """
    _save_array_bundle(path, members, compression=zipfile.ZIP_DEFLATED, compresslevel=1,
                       stored_members=frozenset({'baseline_rgb'}))


def save_native_array(path, values):
    """Publish one native values array using the same atomic bundle writer."""
    save_native_arrays(path, {'values': values})
