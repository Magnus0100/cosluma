"""Use the selected worker's CUDA runtime for large spatial Gaussian filters.

Small filters stay on the host to avoid transfer overhead. A selected CUDA
operation fails explicitly if CUDA cannot execute; it never silently retries
an expensive native filter on the CPU. No Torch import reaches the desktop.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import sys
from threading import Lock

import numpy as np
from scipy.fft import next_fast_len
from native_cuda_ndimage import gaussian_filter as scoped_gaussian_filter, ndimage_execution

_STATE = ContextVar('native_cuda_gaussian', default=None)
MIN_PIXELS = 1_000_000
MIN_SIGMA = 8.


@contextmanager
def cuda_gaussian_scope(device=None):
    """Enable CUDA only inside the already isolated model worker."""
    torch = sys.modules.get('torch')
    if device is None and torch is not None and torch.cuda.is_initialized():
        device = f'cuda:{torch.cuda.current_device()}'
    token = _STATE.set({'cuda_calls': 0, 'device': device, '_lock': Lock()})
    try:
        yield
    finally:
        _STATE.reset(token)


def _sigmas(values, sigma):
    return (float(sigma),)*values.ndim if np.isscalar(sigma) else tuple(float(v) for v in sigma)


def cuda_filter_eligible(values, sigma):
    return (_STATE.get() is not None and values.ndim in (2, 3)
            and values.dtype == np.dtype(np.float32)
            and values.shape[0]*values.shape[1] >= MIN_PIXELS
            and max(_sigmas(values, sigma)) >= MIN_SIGMA)


def filter_execution():
    from asset_identity import sha256_file
    state = _STATE.get()
    return {'large_filter_backend': 'torch_cuda_fft' if state is not None else 'scipy_cpu',
            'cuda_filter_calls': state['cuda_calls'] if state is not None else 0,
            'minimum_pixels': MIN_PIXELS, 'minimum_sigma': MIN_SIGMA,
            'boundary': 'half_sample_symmetric_reflect', 'truncate': 4.,
            'dtype': 'float32', 'non_float32_backend': 'scoped_ndimage_preserve_dtype',
            'ordinary_filter_execution': ndimage_execution(),
            'device': state['device'] if state is not None else None,
            'implementation_sha256': sha256_file(Path(__file__)),
            'CUDA_equality_gate': False}


def _device():
    state = _STATE.get()
    if state is None:
        raise RuntimeError('Native CUDA Gaussian scope is not active')
    import torch
    selected = torch.device(state['device'] or f'cuda:{torch.cuda.current_device()}')
    if selected.type != 'cuda':
        raise ValueError('Native CUDA Gaussian requires a CUDA device')
    if selected.index is None:
        selected = torch.device('cuda', torch.cuda.current_device())
    state['device'] = str(selected)
    return state['device']


def gaussian_filter(values, sigma):
    """Filter NumPy HW/HWC data, with zero sigma on the colour axis."""
    if not cuda_filter_eligible(values, sigma):
        return scoped_gaussian_filter(values, sigma)
    sigmas = _sigmas(values, sigma)
    if len(sigmas) != values.ndim or any(not np.isfinite(s) or s < 0 for s in sigmas):
        raise ValueError('Gaussian sigma must match dimensions and be finite/nonnegative')
    if values.ndim == 3 and sigmas[2] != 0:
        raise ValueError('Native CUDA Gaussian cannot mix colour channels')
    import torch
    # Sequential axes bound transient memory to one native image and FFT
    # workspace. Reflection is explicit; zero padding is only FFT workspace.
    with torch.inference_mode():
        device = _device()
        data = torch.as_tensor(np.ascontiguousarray(values), device=device, dtype=torch.float32)
        for axis, s in enumerate(sigmas):
            if s == 0:
                continue
            radius = int(4*s+.5)
            n = data.shape[axis]
            index = torch.arange(-radius, n+radius, device=device) % (2*n)
            index = torch.where(index < n, index, 2*n-1-index)
            padded = data.index_select(axis, index)
            length = next_fast_len(n+2*radius, real=True)
            x = torch.arange(-radius, radius+1, device=device, dtype=torch.float32)
            taps = torch.exp(-.5*(x/s)**2)
            taps /= taps.sum()
            kernel = torch.zeros(length, device=device, dtype=torch.float32)
            kernel[torch.arange(-radius, radius+1, device=device) % length] = taps
            spectrum = torch.fft.rfft(padded, n=length, dim=axis)
            del padded
            kernel_shape = [1]*data.ndim
            kernel_shape[axis] = -1
            spectrum *= torch.fft.rfft(kernel).reshape(kernel_shape)
            del data
            data = torch.fft.irfft(spectrum, n=length, dim=axis).narrow(axis, radius, n).contiguous()
            del spectrum
        result = data.cpu().numpy()
    state = _STATE.get()
    with state['_lock']:
        state['cuda_calls'] += 1
    return result
