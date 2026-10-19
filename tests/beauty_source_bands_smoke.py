"""Source D and preview bands preserve exact pixels, metadata and PNG bytes."""
from __future__ import annotations
import io

import numpy as np

import beautygrpo_prepare as prepare
from display_transform import display_linear, source_transform_for_precision, prepare_source_display_inplace
from raw_support import linear_preview


def main():
    random = np.random.default_rng(1123)
    for height in (31, 1024, 1031, 3079):
        original = random.uniform(-.05, 4, (height, 37, 3)).astype(np.float32)
        original.reshape(-1)[:6] = np.array([0, -0., 1.e-45, .0031308, .8, 1], np.float32)
        for precision in ('raw16_linear','raster8_srgb'):
            transform = source_transform_for_precision(precision)
            expected = original.copy()
            for top in range(0, height, 128):
                expected[top:top+128] = display_linear(expected[top:top+128],transform)
            actual = original.copy()
            prepare_source_display_inplace(actual,transform)
            assert actual.dtype == expected.dtype and actual.tobytes() == expected.tobytes()
            actual.flags.writeable = False
            before = actual.tobytes()
            expected_image = linear_preview(expected)
            actual_image = prepare._source_display_preview(actual)
            assert actual_image.mode == expected_image.mode and actual_image.size == expected_image.size
            assert actual_image.tobytes() == expected_image.tobytes() and actual_image.info == expected_image.info
            expected_png, actual_png = io.BytesIO(), io.BytesIO()
            expected_image.save(expected_png,format='PNG',compress_level=1)
            actual_image.save(actual_png,format='PNG',compress_level=1)
            assert actual_png.getvalue() == expected_png.getvalue()
            assert actual.tobytes() == before and not actual.flags.writeable
    print('OK: source D/preview bands retain exact float bits, RGB8, ICC/decode metadata, PNG bytes and readonly input')


if __name__ == '__main__':
    main()
