"""Standard multi-member NPZ layout, exact array bits and atomic failure."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from native_array_storage import save_native_array, save_native_arrays, save_proxy_arrays


def same_bits(actual, expected):
    assert actual.dtype.str == expected.dtype.str and actual.shape == expected.shape
    assert actual.tobytes(order='A') == expected.tobytes(order='A')


def check_writer(root, writer, policy, compression, members, before):
    target = root/(policy+'.npz')
    writer(target, members)
    with zipfile.ZipFile(target) as archive:
        assert archive.namelist() == [name+'.npy' for name in members]
        for item in archive.infolist():
            expected_codec = (zipfile.ZIP_STORED if policy == 'proxy-deflate1'
                              and item.filename == 'baseline_rgb.npy' else compression)
            assert item.compress_type == expected_codec
            if expected_codec == zipfile.ZIP_STORED:
                assert item.file_size == item.compress_size
            assert item.date_time == (1980, 1, 1, 0, 0, 0)
    with np.load(target, allow_pickle=False) as archive:
        assert archive.files == list(members)
        for name, expected in members.items():
            same_bits(archive[name], np.asarray(expected))
        assert archive['baseline_rgb'].flags.f_contiguous
        assert archive['owners'].dtype == np.dtype(np.int32)
    for name, array in members.items():
        assert np.asarray(array).tobytes(order='A') == before[name]

    protected = hashlib.sha256(target.read_bytes()).hexdigest()
    write = np.lib.format.write_array
    attempts = []
    def fail_second(stream, array, **kwargs):
        attempts.append(1)
        if len(attempts) == 2:
            raise IOError('Synthetic incomplete bundle')
        return write(stream, array, **kwargs)
    with patch.object(np.lib.format, 'write_array', side_effect=fail_second):
        try:
            writer(target, members)
        except IOError as error:
            assert str(error) == 'Synthetic incomplete bundle'
        else:
            raise AssertionError('Incomplete bundle was published')
    assert hashlib.sha256(target.read_bytes()).hexdigest() == protected
    assert not list(root.glob('*.tmp.npz'))
    for invalid in ({'../values': members['values']}, {'values/escape': members['values']}, {'': members['values']}, {},
                    {'values': np.array([object()], dtype=object)}):
        try:
            writer(target, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('Unsafe or pickled member was accepted')
        assert hashlib.sha256(target.read_bytes()).hexdigest() == protected
        assert not list(root.glob('*.tmp.npz'))


def main():
    float_bits = np.array([0x00000000, 0x80000000, 0x3f800000, 0x7fc12345], np.uint32).view(np.float32)
    values = np.resize(float_bits, (9, 11, 6))
    members = {'values': values,
        'owners': np.arange(99, dtype=np.int32).reshape(9, 11),
        'baseline_rgb': np.asfortranarray(np.arange(9*11*3, dtype='>f4').reshape(9, 11, 3)),
        'chroma_values': np.arange(18*22*10, dtype=np.float32).reshape(18, 22, 10)[::2, ::2],
        'valid': (np.arange(99).reshape(9, 11) % 3 == 0),
        'canvas_size': np.array([11, 9], dtype=np.int32),
        'scale': np.float32(-0.0)}
    before = {name: np.asarray(array).tobytes(order='A') for name, array in members.items()}
    with TemporaryDirectory(prefix='native-multi-array-') as temporary:
        root = Path(temporary)
        check_writer(root, save_native_arrays, 'native-stored', zipfile.ZIP_STORED, members, before)
        check_writer(root, save_proxy_arrays, 'proxy-deflate1', zipfile.ZIP_DEFLATED, members, before)
        # The one-array convenience call must retain its existing bytes/API.
        one = root/'single.npz'; bundle = root/'single-bundle.npz'
        save_native_array(one, values); save_native_arrays(bundle, {'values': values})
        assert one.read_bytes() == bundle.read_bytes()
    print('OK: complete keys, float bits/signed zeros/NaN payload, integer owners, endian/Fortran/strided arrays, '
          'fixed native STORED and sparse proxy DEFLATE1/dense RGB STORED, standard np.load, '
          'atomic failure preserves valid bundle, temporary cleanup, no model or GPU')


if __name__ == '__main__':
    main()
