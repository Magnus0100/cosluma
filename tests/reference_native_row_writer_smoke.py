"""Small CPU check of standard row-stream NPZ and live publication ownership."""
from __future__ import annotations

from pathlib import Path
import tempfile
import threading
import zipfile

import numpy as np

from native_array_storage import NativeArrayRowWriter, save_native_array
from runtime_resources import ordered_native_map
from reference_appearance_stack import _validate_native_delta


def main():
    # The first item waits for the second worker, proving overlap without a
    # timing threshold. The consumer still receives the original order and
    # the iterator never pulls more than two inputs ahead of that consumer.
    second_started = threading.Event()
    pulled = []
    def items():
        for item in range(5):
            pulled.append(item)
            yield item
    def compute(item):
        if item == 0:
            assert second_started.wait(5), 'Independent bands did not overlap'
        elif item == 1:
            second_started.set()
        return item+100
    bands = ordered_native_map(compute, items(), workers=2)
    assert next(bands) == 100 and pulled == [0, 1]
    assert list(bands) == [101, 102, 103, 104]
    assert list(ordered_native_map(lambda item: item+100, range(5), workers=1)) == list(range(100, 105))
    for workers in (0, -1, 1.5):
        try:list(ordered_native_map(lambda item: item, [0], workers=workers))
        except ValueError:pass
        else:raise AssertionError('Invalid native worker count was accepted')
    def fail_second(item):
        if item == 1:raise ValueError('worker failure')
        return item
    bands = ordered_native_map(fail_second, range(5), workers=2)
    assert next(bands) == 0
    try:next(bands)
    except ValueError as error:assert str(error) == 'worker failure'
    else:raise AssertionError('Worker failure was lost')
    bits = np.array([0, 0x80000000, 1, 0x80000001, 0x3f800000,
                     0xbf800000, 0x7f7fffff, 0xff7fffff], dtype='<u4')
    values = np.resize(bits, 19*31*3).reshape(19, 31, 3).view('<f4')[:, ::-1]
    with tempfile.TemporaryDirectory(prefix='cosplay_native_rows_') as directory:
        root = Path(directory); target = root/'rows.npz'; ordinary = root/'ordinary.npz'
        writer = NativeArrayRowWriter(target, values.shape)
        try:
            for top in range(0, len(values), 7):
                writer.write(values[top:top+7])
            writer.publish()
            _validate_native_delta(writer, (31, 19), asset_path=target)
            save_native_array(ordinary, np.ascontiguousarray(values))
            assert target.read_bytes() == ordinary.read_bytes(), 'Standard stored NPZ bytes changed'
            with zipfile.ZipFile(target) as archive:
                assert archive.namelist() == ['values.npy']
                assert archive.getinfo('values.npy').compress_type == zipfile.ZIP_STORED
            with np.load(target, allow_pickle=False) as resource:
                assert resource.files == ['values']
                np.testing.assert_array_equal(resource['values'].view('<u4'), values.view('<u4'))
            target.touch()
            try:_validate_native_delta(writer, (31, 19), asset_path=target)
            except ValueError:pass
            else:raise AssertionError('Changed live publication retained producer approval')
        finally:
            writer.close()
        kept = root/'kept.npz'; save_native_array(kept, values)
        kept_bytes = kept.read_bytes()
        for case in ('incomplete', 'shape', 'dtype', 'nonfinite'):
            writer = NativeArrayRowWriter(kept, values.shape)
            try:
                if case == 'incomplete':
                    writer.write(values[:7]); writer.publish()
                elif case == 'shape':writer.write(values[:7, :-1])
                elif case == 'dtype':writer.write(values[:7].astype(np.float64))
                else:
                    invalid = values[:7].copy(); invalid[0, 0, 0] = np.nan
                    writer.write(invalid)
            except ValueError:pass
            else:raise AssertionError('Invalid row publication was accepted: '+case)
            finally:writer.close()
            assert kept.read_bytes() == kept_bytes, 'Failed publication replaced a valid asset'
        assert not list(root.glob('*.tmp.npz'))
    print('OK: bounded overlapping native bands retain order and propagate failure; ordered native rows; standard stored NPZ bytes/float32 bits/signed zeros/strided input; live proof invalidates changed file; incomplete/shape/dtype/nonfinite rejection; atomic failure preserves valid asset; temporary cleanup; CPU only')


if __name__ == '__main__':
    main()
