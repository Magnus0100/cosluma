"""Boundary and worker isolation checks, without a CPU/CUDA equality gate."""
import argparse
import sys
from unittest.mock import patch

import numpy as np
import native_cuda_filters as filters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    small = np.ones((30, 40), np.float32)
    with patch.dict(sys.modules, {'torch': None}):
        with filters.cuda_gaussian_scope():
            assert not filters.cuda_filter_eligible(small, 20)
            assert np.allclose(filters.gaussian_filter(small, 20), 1)
            large = np.zeros((1024, 1024), np.float32)
            try:
                with patch.object(filters, 'scoped_gaussian_filter', side_effect=AssertionError('CUDA retried on CPU')):
                    filters.gaussian_filter(large, 20)
            except ModuleNotFoundError:
                pass
            else:
                raise AssertionError('Missing CUDA runtime was concealed')
    assert not filters.cuda_filter_eligible(large, 20)
    # Gaussian scope alone retains float64 on the CPU. An additional ndimage
    # scope can use official CuPy while preserving that same float64 dtype.
    with filters.cuda_gaussian_scope():
        precise = np.ones((1024, 1024), dtype=np.float64)
        assert not filters.cuda_filter_eligible(precise, 20)
        assert filters.gaussian_filter(precise, 8).dtype == precise.dtype
    if args.cuda:
        # Constant colour and edge impulses expose channel mixing, broken
        # normalisation, circular wraparound, and wrong reflection padding.
        large = np.empty((1031, 1027, 3), np.float32)
        large[:] = [.12, .34, .89]
        with filters.cuda_gaussian_scope():
            result = filters.gaussian_filter(large[:, ::-1], (31., 31., 0))
            assert result.shape == large.shape and result.dtype == np.float32
            assert np.isfinite(result).all()
            assert np.max(np.abs(result-[.12, .34, .89])) < 1e-4
            impulse = np.zeros((1031, 1027), np.float32)
            impulse[0, 0] = 1
            edge = filters.gaussian_filter(impulse, 20.)
            assert edge[0, 0] > edge[30, 30] > 0
            assert abs(float(edge[-1, -1])) < 1e-7
            assert abs(float(edge.sum())-1.) < 1e-4
            assert filters.filter_execution()['cuda_filter_calls'] == 2
        assert not filters.cuda_filter_eligible(large, 31)
    print('native CUDA filter smoke ok: scoped runtime, explicit failure, no equality gate'+
          (', constant colour, edge reflection, no wraparound' if args.cuda else ''))


if __name__ == '__main__':
    main()
