"""Check ordinary NPZ consumers and float bits for native storage policies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np

from native_array_storage import save_native_array


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    # Include both signed zeros, subnormals, finite endpoints, and a quiet-NaN
    # payload: serialization must preserve bits without numeric normalization.
    bits = np.array([0x00000000, 0x80000000, 0x00000001, 0x80000001,
        0x3f800000, 0xbf800000, 0x7f7fffff, 0xff7fffff, 0x7fc01234], dtype='<u4')
    dense = np.resize(bits, 19*31*3).reshape(19, 31, 3).view('<f4')
    source = np.zeros((19, 31, 3), dtype='<f4')
    source[2:4, 5:8] = dense[2:4, 5:8]
    source[::3, ::4, 0] = np.float32(-0.)
    edge = source[..., 0].copy()
    report = []
    with tempfile.TemporaryDirectory(prefix='cosplay_native_encoding_') as directory:
        folder = Path(directory)
        # Control production writes np.memmap values, while freeze uses a
        # resident ndarray. Exercise both plus a non-contiguous ndarray.
        mapped = np.lib.format.open_memmap(folder/'control.npy', mode='w+',
            dtype='<f4', shape=dense.shape)
        mapped[:] = dense
        mapped.flush()
        cases = [('source-foreground', source), ('original-environment', source),
            ('edge-support', edge), ('automatic-delta', dense),
            ('colour-delta-strided', dense[:, ::-1]), ('lightness-delta-memmap', mapped)]
        try:
            for name, values in cases:
                before = values.view('<u4').copy()
                path = folder/(name+'.npz')
                save_native_array(path, values)
                with zipfile.ZipFile(path) as archive:
                    assert archive.namelist() == ['values.npy']
                    method = archive.getinfo('values.npy').compress_type
                    assert method == zipfile.ZIP_STORED
                # Standard existing consumer: no custom decoder or schema.
                with np.load(path, allow_pickle=False) as resource:
                    assert resource.files == ['values']
                    decoded = resource['values']
                assert decoded.shape == values.shape and decoded.dtype.str == values.dtype.str
                mismatch = int(np.count_nonzero(decoded.view('<u4') != before))
                assert mismatch == 0
                assert np.array_equal(values.view('<u4'), before)
                assert np.any(before == 0x00000000) and np.any(before == 0x80000000)
                assert not path.with_suffix('.tmp.npz').exists()
                report.append({'case': name, 'shape': list(values.shape),
                    'dtype': values.dtype.str, 'compression_method': method,
                    'float_bit_mismatches': mismatch, 'signed_zero_bits_preserved': True,
                    'input_bits_unchanged': True, 'ordinary_np_load': True})
        finally:
            mapped._mmap.close()
    result = {'schema': 'reference_native_array_encoding_smoke.v1', 'cases': report,
        'passed': True, 'models_or_pipeline_used': False}
    if args.report:
        args.report.write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    print('OK: six CPU NPZ fixtures; standard np.load; exact dtype/shape/float bits and signed zeros; one STORED policy; ndarray/strided/memmap; input unchanged')


if __name__ == '__main__':
    main()
