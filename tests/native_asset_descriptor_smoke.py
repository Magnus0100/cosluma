"""Header, full content binding, ZIP CRC and truncation proof without RGB decode."""
from __future__ import annotations

import copy
from io import BytesIO
import json
from pathlib import Path
import struct
import tempfile
import zipfile
from unittest.mock import patch

import numpy as np

from native_source_baseline import _verify_native_rgb_archive, VerifiedNativeArrayAsset
from asset_identity import sha256_file


def record(path, shape):
    return {'path': str(path), 'sha256': sha256_file(path, force=True),
            'size': list(shape[1::-1]), 'representation': 'straight_RGB'}


def reject(function):
    try:
        function()
    except (ValueError, OSError, EOFError, zipfile.BadZipFile):
        return
    raise AssertionError('Changed, corrupt or truncated native member was accepted')


def write_member(path, values, *, stored=True, trim=0):
    encoded = BytesIO(); np.save(encoded, values, allow_pickle=False)
    payload = encoded.getvalue()
    if trim: payload = payload[:-trim]
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('values.npy', payload)


def main():
    values = np.arange(17*31*3, dtype=np.float32).reshape(17, 31, 3)/2048
    with tempfile.TemporaryDirectory(prefix='native-archive-descriptor-') as temporary:
        root = Path(temporary)
        for stored in (True, False):
            path = root/('stored.npz' if stored else 'deflated.npz')
            write_member(path, values, stored=stored)
            asset = record(path, values.shape)
            with patch('native_source_baseline.np.load', side_effect=AssertionError('Descriptor decoded RGB')):
                checked = _verify_native_rgb_archive(asset, values.shape)
            assert isinstance(checked, VerifiedNativeArrayAsset) and copy.deepcopy(checked) is checked
            assert checked.shape == values.shape and checked.payload_bytes == values.nbytes
            assert np.dtype(checked.dtype) == np.float32 and not checked.fortran_order
            checked.validate_asset(asset)
            reject(lambda: checked.validate_asset({**asset, 'sha256': '0'*64}))
            reject(lambda: checked.validate_asset({**asset, 'size': [31, 18]}))
            reject(lambda: _verify_native_rgb_archive({**asset, 'sha256': '0'*64}, values.shape))
        fpath = root/'fortran.npz'; write_member(fpath, np.asfortranarray(values))
        assert _verify_native_rgb_archive(record(fpath, values.shape), values.shape).fortran_order
        wrong = root/'wrong.npz'; write_member(wrong, values.astype(np.float64))
        reject(lambda: _verify_native_rgb_archive(record(wrong, values.shape), values.shape))
        short = root/'short.npz'; write_member(short, values, trim=4)
        reject(lambda: _verify_native_rgb_archive(record(short, values.shape), values.shape))
        # Update the ordinary SHA to the damaged bytes: CRC remains an
        # independent mandatory check, not a side effect of a hash mismatch.
        damaged = root/'bad-crc.npz'; write_member(damaged, values)
        with zipfile.ZipFile(damaged) as archive:
            offset = archive.getinfo('values.npy').header_offset
        with damaged.open('r+b') as handle:
            handle.seek(offset); signature, name_length, extra_length = struct.unpack('<4s22xHH', handle.read(30))
            assert signature == b'PK\x03\x04'
            handle.seek(offset+30+name_length+extra_length+130)
            byte = handle.read(1); handle.seek(-1, 1); handle.write(bytes([byte[0]^1]))
        reject(lambda: _verify_native_rgb_archive(record(damaged, values.shape), values.shape))
        truncated = root/'truncated.npz'; write_member(truncated, values)
        truncated.write_bytes(truncated.read_bytes()[:-12])
        reject(lambda: _verify_native_rgb_archive(record(truncated, values.shape), values.shape))
        path = root/'replaced.npz'; write_member(path, values)
        asset = record(path, values.shape); checked = _verify_native_rgb_archive(asset, values.shape)
        other = root/'replacement.npz'; write_member(other, values*.5); other.replace(path)
        reject(lambda: checked.validate_asset(asset))
    print(json.dumps({'schema': 'native_asset_descriptor_smoke.v1', 'passed': True,
        'STORED_and_DEFLATE_CRC_checked': True, 'no_np_load_or_RGB_buffer': True,
        'full_SHA_binding': True, 'C_and_Fortran_float32_headers': True,
        'wrong_dtype_shape_SHA_rejected': True, 'payload_and_archive_truncation_rejected': True,
        'CRC_rejected_with_matching_SHA': True, 'replace_before_materialization_rejected': True,
        'model_or_GPU_used': False}))


if __name__ == '__main__':main()
