"""Scoped official GPU geometry/reflect checks; no CPU/CUDA bit-equality gate."""
import argparse
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
from scipy import ndimage as scipy
_DESKTOP_TEMP_ENV = {name: os.environ.get(name) for name in ('TEMP', 'TMP', 'TMPDIR')}
_DESKTOP_TEMP_CACHE = tempfile.tempdir
import native_cuda_ndimage as native


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    assert 'cupy' not in sys.modules and 'torch' not in sys.modules
    assert {name: os.environ.get(name) for name in _DESKTOP_TEMP_ENV} == _DESKTOP_TEMP_ENV
    assert tempfile.tempdir == _DESKTOP_TEMP_CACHE
    small = np.zeros((33, 45), bool); small[8:27, 12:33] = True
    with patch.object(native, '_load_backend', side_effect=AssertionError('Desktop imported GPU')):
        assert np.array_equal(native.distance_transform_edt(small), scipy.distance_transform_edt(small))
        with native.cuda_ndimage_scope():
            assert np.array_equal(native.distance_transform_edt(small), scipy.distance_transform_edt(small))
            assert native.median_filter(small.astype(np.float32), size=3).dtype == np.float32
            assert native.gaussian_filter(small.astype(np.float64), 1.25).dtype == np.float64
            assert native.ndimage_execution()['cpu_calls'] == 3
            assert native.ndimage_execution()['gpu_calls'] == 0
        assert not native.ndimage_execution()['enabled']
    assert 'cupy' not in sys.modules and 'torch' not in sys.modules
    assert {name: os.environ.get(name) for name in _DESKTOP_TEMP_ENV} == _DESKTOP_TEMP_ENV
    assert tempfile.tempdir == _DESKTOP_TEMP_CACHE
    mixed = np.zeros((1031, 1027), bool); mixed[19:1010, 29:998] = True
    # A qualified selected node cannot quietly rerun on the CPU.
    with (patch.object(native, '_load_backend', side_effect=RuntimeError('GPU deliberately absent')),
          patch.object(native.scipy_ndimage, 'distance_transform_edt', side_effect=AssertionError('GPU retried on CPU'))):
        try:
            with native.cuda_ndimage_scope():
                try:
                    native.distance_transform_edt(mixed)
                except RuntimeError as error:
                    assert 'CPU retry is disabled' in str(error)
                    assert len(native.ndimage_execution()['failures']) == 1
                    raise
        except RuntimeError:
            pass
        else:
            raise AssertionError('Selected GPU failure was concealed')
    assert not native.ndimage_execution()['enabled']
    with native.cuda_ndimage_scope():
        with native.cuda_ndimage_scope():
            native.distance_transform_edt(small)
            assert native.ndimage_execution()['cpu_calls'] == 1
        assert native.ndimage_execution()['cpu_calls'] == 0
        empty = np.zeros_like(mixed); solid = np.ones_like(mixed)
        with patch.object(native, '_load_backend', side_effect=AssertionError('Featureless EDT imported GPU')):
            assert np.array_equal(native.distance_transform_edt(empty), scipy.distance_transform_edt(empty))
            assert np.array_equal(native.distance_transform_edt(solid), scipy.distance_transform_edt(solid))
        assert all(c['selection_reason'] == 'featureless_EDT_preserve_SciPy' for c in native.ndimage_execution()['calls'])
    if args.cuda:
        # Only simulate an inherited cached C path; do not create/write it.
        cached_c = 'C:/cupy-cached-temp-negative-never-created'
        with (patch.dict(os.environ, {name: cached_c for name in ('TEMP', 'TMP', 'TMPDIR')}),
              patch.object(tempfile, 'tempdir', cached_c), native.cuda_ndimage_scope()):
            assert tempfile.gettempdir() == cached_c
            distance, indices = native.distance_transform_edt(mixed[:, ::-1], return_indices=True)
            mask = mixed[:, ::-1]
            assert distance.shape == mask.shape and distance.dtype == np.float64
            assert indices.shape == (2, *mask.shape) and indices.dtype == np.int32
            assert np.isfinite(distance).all() and np.all(distance >= 0) and np.all(distance[~mask] == 0)
            assert np.all((indices[0] >= 0)&(indices[0] < mask.shape[0]))
            assert np.all((indices[1] >= 0)&(indices[1] < mask.shape[1]))
            assert np.all(~mask[indices[0], indices[1]])
            y, x = np.indices(mask.shape)
            assert np.max(np.abs(np.hypot(y-indices[0], x-indices[1])-distance)) < 1e-10
            expected = scipy.distance_transform_edt(mask)
            for threshold in (1.5, 2., 4., 8., 18., 24., 83., 110.):
                assert np.array_equal(distance <= threshold, expected <= threshold)
            luminance = ((y*3+x*7)%29).astype(np.float32)/29
            median = native.median_filter(luminance[:, ::-1], size=3, mode='reflect')
            assert median.dtype == np.float32 and np.isfinite(median).all()
            assert np.max(np.abs(median-scipy.median_filter(luminance[:, ::-1], size=3, mode='reflect'))) < 1e-7
            precise = np.full(mask.shape, .345678901234567, np.float64)
            blurred = native.gaussian_filter(precise, 1.25, mode='reflect', truncate=4.)
            assert blurred.dtype == np.float64 and np.max(np.abs(blurred-precise)) < 1e-12
            colours = np.empty((*mask.shape, 3), np.float32); colours[:] = [.12, .34, .89]
            colour_blur = native.gaussian_filter(colours[:, ::-1], (1.25, 1.25, 0), mode='reflect')
            assert colour_blur.dtype == np.float32 and np.isfinite(colour_blur).all()
            assert np.max(np.abs(colour_blur-[.12, .34, .89])) < 2e-6
            report = native.ndimage_execution()
            assert report['gpu_calls'] == 4 and report['cpu_calls'] == 0 and not report['failures']
            assert report['cupy_pool_is_separate_from_Torch_counters']
            assert len(report['cuda_free_total_bytes']) == 2
            assert report['package_identity']['actual_nvrtc_version'] == [12, 8]
            assert report['package_identity']['packages']['cupy-cuda12x']['version'] == '14.2.0'
            assert report['package_identity']['kernel_cache'].startswith('D:')
            actual_temp = Path(tempfile.gettempdir()).resolve()
            assert actual_temp.drive.upper() == 'D:'
            assert Path(report['package_identity']['actual_compilation_temp']).resolve() == actual_temp
            assert all(Path(os.environ[name]).resolve() == actual_temp for name in ('TEMP', 'TMP', 'TMPDIR'))
            with tempfile.TemporaryDirectory(prefix='cupy-D-scratch-smoke-') as scratch:
                assert Path(scratch).resolve().is_relative_to(actual_temp)
            assert report['cupy_pool_retained_bytes'] >= report['cupy_pool_used_bytes']
        assert not native.ndimage_execution()['enabled']
    print('native ndimage smoke ok: desktop laziness, gates, explicit failure, scoped cleanup'+
          (', exact EDT geometry/indices/thresholds, reflect median/Gaussian, dtype/channels, real counters/package/memory, cached-C-temp override and actual D compilation scratch' if args.cuda else ''))


if __name__ == '__main__':
    main()
