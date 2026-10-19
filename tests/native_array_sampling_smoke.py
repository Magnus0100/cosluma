"""Small CPU checks for array-backed native planes against the prior Pillow sampler."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

from design_reconstruction import _ArrayPlane, _planes, _sample_channels, sample_asset


def legacy_planes(path):
    if path.suffix == '.npz':
        with np.load(path) as resource:
            arrays = {key: np.asarray(resource[key], dtype=np.float32) for key in resource.files}
    else:
        with Image.open(path) as image:
            arrays = {'values': np.asarray(image.convert('L'), dtype=np.float32) / 255}
    return {key: (tuple(Image.fromarray(values[..., index]) for index in range(values.shape[-1]))
                  if values.ndim == 3 else (Image.fromarray(values),)) for key, values in arrays.items()}


def legacy_sample(channels, box, size, sampling='bilinear'):
    left, top, right, bottom = box
    width, height = size
    fw, fh = channels[0].size
    sample_box = (left * fw / width, top * fh / height, right * fw / width, bottom * fh / height)
    method = Image.Resampling.NEAREST if sampling == 'nearest' else Image.Resampling.BILINEAR
    planes = [np.array(channel.resize((right-left, bottom-top), method,
        box=sample_box), dtype=np.float32) for channel in channels]
    return planes[0] if len(planes) == 1 else np.stack(planes, axis=-1)


def assert_bits(actual, expected):
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert actual.dtype == expected.dtype == np.dtype('float32')
    assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32))


def outcome(function):
    try:
        return function(), None
    except Exception as error:
        return None, (type(error).__name__, str(error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    checks = {'sampling_parity': 0, 'crop_parity': 0, 'resize_parity': 0, 'pillow_tuple_parity': 0}
    bits = np.array([0x00000000, 0x80000000, 0x00000001, 0x80000001,
        0x3f800000, 0xbf800000, 0x7f7fffff, 0xff7fffff, 0x7fc01234], dtype=np.uint32)
    native_bits = np.resize(bits, 7*11*3).reshape(7, 11, 3).view(np.float32)
    rng = np.random.default_rng(20261013)
    values = rng.normal(size=(7, 11, 3)).astype(np.float32)
    values[0, :2] = np.array([0., -0.], np.float32)[:, None]
    arrays = {'rgb': values, 'scalar': values[..., 0], 'singleton': values[..., :1],
        'native_bits': native_bits}
    native_size = (11, 7)
    sample_cases = [((0, 0, 11, 7), native_size), ((2, 1, 9, 6), native_size),
        ((10, 6, 11, 7), native_size), ((np.int64(1), np.int64(2), np.int64(4), np.int64(5)), native_size),
        ((0, 0, 19, 13), (19, 13)), ((3, 2, 16, 10), (19, 13)),
        ((0, 0, 5, 3), (5, 3)), ((1, 1, 4, 3), (5, 3)),
        ((-1, 0, 4, 3), native_size), ((0, 0, 12, 8), native_size),
        ((2, 1, 2, 4), native_size), ((2, 1, 5, 1), native_size),
        ((5, 4, 2, 1), native_size), ((1.5, 1., 5.5, 4.), native_size),
        ((1., 1., 5., 4.), native_size), ((0, 0, 1, 1), (0, 7)),
        ((False, False, True, True), native_size)]
    crop_cases = [None, (0, 0, 11, 7), (2, 1, 9, 6), (-2, -1, 4, 3),
        (0, 0, 12, 8), (2, 1, 2, 4), (1.5, 1.5, 5.5, 4.5), (5, 4, 2, 1)]
    with tempfile.TemporaryDirectory(prefix='cosplay_array_sampling_') as directory:
        folder = Path(directory)
        path = folder/'planes.npz'
        np.savez(path, **arrays)
        _planes.cache_clear()
        modified = path.stat().st_mtime_ns
        actual = _planes(str(path), modified)
        expected = legacy_planes(path)
        assert _planes(str(path), modified) is actual
        assert _planes.cache_info().maxsize == 12
        for key, channels in actual.items():
            owner = channels[0]._values
            assert not owner.flags.writeable
            assert all(isinstance(channel, _ArrayPlane) and channel._values is owner for channel in channels)
            for index, channel in enumerate(channels):
                view = np.asarray(channel, dtype=np.float32)
                assert not view.flags.writeable and np.shares_memory(view, owner)
                assert_bits(view, np.asarray(expected[key][index], dtype=np.float32))
                converted = np.asarray(channel, dtype=np.float64)
                assert converted.dtype == np.float64 and not np.shares_memory(converted, owner)
                copied = np.array(channel, dtype=np.float32)
                assert copied.flags.writeable and not np.shares_memory(copied, owner)
                for box in crop_cases:
                    got, got_error = outcome(lambda: np.asarray(channel.crop(box), np.float32))
                    want, want_error = outcome(lambda: np.asarray(expected[key][index].crop(box), np.float32))
                    assert got_error == want_error, (key, box, got_error, want_error)
                    if got_error is None:
                        assert_bits(got, want)
                    checks['crop_parity'] += 1
                for method in (Image.Resampling.NEAREST, Image.Resampling.BILINEAR):
                    for size, box in [((5, 3), None), ((19, 13), None), ((4, 3), (1.25, .5, 9.5, 6.5))]:
                        assert_bits(np.asarray(channel.resize(size, method, box=box), np.float32),
                            np.asarray(expected[key][index].resize(size, method, box=box), np.float32))
                        checks['resize_parity'] += 1
            for box, size in sample_cases:
                for sampling in ('nearest', 'bilinear'):
                    got, got_error = outcome(lambda: _sample_channels(channels, box, size, sampling))
                    want, want_error = outcome(lambda: legacy_sample(expected[key], box, size, sampling))
                    assert got_error == want_error, (key, box, size, sampling, got_error, want_error)
                    if got_error is None:
                        assert_bits(got, want)
                        assert got.flags.writeable and not np.shares_memory(got, owner)
                    checks['sampling_parity'] += 1
            before = owner.view(np.uint32).copy()
            full = _sample_channels(channels, (0, 0, 11, 7), native_size)
            roi = _sample_channels(channels, (2, 1, 9, 6), native_size)
            full[:] = 17.
            roi[:] = 19.
            assert np.array_equal(owner.view(np.uint32), before)
        # Existing runtimes also pass ordinary Pillow tuples directly.
        for box, size in sample_cases:
            got, got_error = outcome(lambda: _sample_channels(expected['rgb'], box, size))
            want, want_error = outcome(lambda: legacy_sample(expected['rgb'], box, size))
            assert got_error == want_error
            if got_error is None:
                assert_bits(got, want)
            checks['pillow_tuple_parity'] += 1
        # A stale caller timestamp cannot select an older file identity.
        np.savez(path, values=np.full((7, 11, 3), 23., np.float32))
        os.utime(path, ns=(modified+1_000_000_000, modified+1_000_000_000))
        fresh = _planes(str(path), path.stat().st_mtime_ns)
        assert fresh is not actual
        assert _planes(str(path), modified) is not actual
        assert np.all(sample_asset({'path': str(path)}, 'values', (2, 1, 9, 6), native_size) == 23.)
        # Same-inode aliases share arrays, but replacing the original locator
        # cannot redirect its remaining hardlink to different pixels.
        alias = folder/'alias.npz'
        os.link(path, alias)
        timestamp = path.stat().st_mtime_ns
        assert _planes(str(alias), timestamp) is _planes(str(path), timestamp)
        replacement = folder/'replacement.npz'
        np.savez(replacement, values=np.full((7, 11, 3), 9., np.float32))
        os.utime(replacement, ns=(timestamp, timestamp))
        replacement.replace(path)
        _planes.cache_clear()
        assert np.all(np.asarray(_planes(str(path), timestamp)['values'][0]) == 9.)
        assert np.all(np.asarray(_planes(str(alias), timestamp)['values'][0]) == 23.)
        # Raster alpha keeps the former grayscale / 255 conversion.
        raster = folder/'alpha.png'
        Image.fromarray(np.arange(77, dtype=np.uint8).reshape(7, 11)).save(raster)
        channels = _planes(str(raster), raster.stat().st_mtime_ns)['values']
        assert_bits(_sample_channels(channels, (2, 1, 9, 6), native_size),
            legacy_sample(legacy_planes(raster)['values'], (2, 1, 9, 6), native_size))
        _planes.cache_clear()
    result = {'schema': 'native_array_sampling_smoke.v1', 'passed': True,
        'checks': checks, 'float_bits_and_signed_zero': True, 'single_readonly_array_owner': True,
        'mutable_results_leave_cache_unchanged': True, 'mtime_invalidation': True,
        'models_or_pipeline_used': False, 'pillow_version': Image.__version__}
    if args.report:
        args.report.write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    print('OK: native array planes; exact prior Pillow float bits/errors; native/scaled/invalid ROIs; '
        'crop/resize and ordinary Pillow tuples; one readonly cache; mutable results; mtime invalidation')


if __name__ == '__main__':
    main()
