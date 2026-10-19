"""Official CuPy image primitives, enabled only in isolated native workers.

Small nodes and featureless EDTs keep SciPy's mature host behavior. A selected
large GPU node fails explicitly; there is no expensive CPU retry. Importing
this module never imports Torch/CuPy or initializes CUDA in the desktop.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import hashlib
import importlib.metadata
import os
from pathlib import Path
import sys
from threading import RLock
import time

import numpy as np
from scipy import ndimage as scipy_ndimage

MIN_PIXELS = 1_000_000
QUALIFIED_PACKAGES = {
    'cupy-cuda12x': ('14.2.0', 'c9571d3b5f2e65758137e210f7fb3c3b34767f0af6b6ca04035a244b6141ee12'),
    'cuda-pathfinder': ('1.8.3', 'e29e59829c297a7a5233bd9cc71094fc5bddbd076951482670178f9eade39b1f'),
    'nvidia-cuda-runtime-cu12': ('12.8.90', 'c0c6027f01505bfed6c3b21ec546f69c687689aad5f1a377554bc6ca4aa993a8'),
}
_STATE = ContextVar('native_cuda_ndimage', default=None)
_BACKEND = None
_BACKEND_LOCK = RLock()
_DLL_DIRECTORIES = []
_DLL_LIBRARIES = []


@contextmanager
def cuda_ndimage_scope(device=0):
    """Enable the qualified backend for one already isolated native request."""
    if not isinstance(device, int) or device < 0:
        raise ValueError('Native ndimage device must be a nonnegative CUDA index')
    state = {'device': device, 'calls': [], 'failures': [], 'initialization_seconds': 0., '_lock': RLock()}
    token = _STATE.set(state)
    try:
        yield
    finally:
        _STATE.reset(token)


def _load_backend():
    global _BACKEND
    with _BACKEND_LOCK:
        if _BACKEND is not None:
            return _BACKEND
        site = Path(sys.prefix)/'Lib/site-packages'
        cuda = site/'nvidia/cuda_runtime'
        torch_lib = site/'torch/lib'
        if os.name != 'nt' or Path(sys.prefix).drive.upper() != 'D:':
            raise RuntimeError('Qualified native CuPy requires the selected Windows D runtime')
        required_paths = (cuda/'include/cuda_runtime.h', torch_lib/'nvrtc64_120_0.dll',
                          torch_lib/'nvrtc-builtins64_128.dll')
        if any(not path.is_file() for path in required_paths):
            raise RuntimeError('Qualified CUDA 12.8 headers or Torch NVRTC DLL are missing')
        packages = {}
        for name, (version, wheel_sha) in QUALIFIED_PACKAGES.items():
            distribution = importlib.metadata.distribution(name)
            if distribution.version != version:
                raise RuntimeError(f'Native CuPy requires qualified {name}=={version}')
            files = list(distribution.files or ())
            metadata_file = next(p for p in files if str(p).endswith('.dist-info/METADATA'))
            path = Path(distribution.locate_file(metadata_file)).resolve()
            if not path.is_relative_to(site.resolve()):
                raise RuntimeError('Native CuPy distribution is outside the selected runtime')
            packages[name] = {'version': distribution.version, 'metadata_path': str(path),
                'metadata_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'qualified_official_wheel_sha256': wheel_sha}
        kernel_cache = Path(sys.prefix).parent/'cache/cupy-cuda12x-kernels'
        kernel_cache.mkdir(parents=True, exist_ok=True)
        compilation_temp = Path(sys.prefix).parent/'cache/cupy-compilation-temp'
        compilation_temp.mkdir(parents=True, exist_ok=True)
        # Only this worker process receives these settings. Never select the
        # machine's CUDA 13 headers for the qualified 12.8 NVRTC compiler.
        os.environ['CUDA_PATH'] = str(cuda)
        os.environ['CUPY_CACHE_DIR'] = str(kernel_cache)
        # CuPy's NVRTC scratch uses tempfile independently of its kernel
        # cache. Reset a cached desktop C path before importing the backend.
        import tempfile
        for name in ('TEMP', 'TMP', 'TMPDIR'):
            os.environ[name] = str(compilation_temp)
        tempfile.tempdir = None
        actual_compilation_temp = Path(tempfile.gettempdir()).resolve()
        if actual_compilation_temp != compilation_temp.resolve() or actual_compilation_temp.drive.upper() != 'D:':
            raise RuntimeError('Native CuPy compilation scratch did not bind to D storage')
        dll_paths = (cuda/'bin', torch_lib, site/'cupy/.data/lib')
        for path in dll_paths:
            if path.is_dir():
                _DLL_DIRECTORIES.append(os.add_dll_directory(str(path)))
        os.environ['PATH'] = ';'.join(str(p) for p in dll_paths if p.is_dir())+';'+os.environ.get('PATH', '')
        # cuda-pathfinder prefers an already loaded library. Bind the actual
        # Torch NVRTC 12.8 DLL before its system search can choose CUDA 13.
        import ctypes
        nvrtc_library = ctypes.WinDLL(str(required_paths[1]))
        major, minor = ctypes.c_int(), ctypes.c_int()
        nvrtc_library.nvrtcVersion.argtypes = (ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
        nvrtc_library.nvrtcVersion.restype = ctypes.c_int
        if nvrtc_library.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor)) != 0 or (major.value, minor.value) != (12, 8):
            raise RuntimeError('The bound Torch NVRTC DLL is not version 12.8')
        _DLL_LIBRARIES.append(nvrtc_library)
        import cupy as cp
        from cupyx.scipy import ndimage
        from cupy_backends.cuda.libs import nvrtc
        nvrtc_version = tuple(nvrtc.getVersion())
        runtime = cp.cuda.runtime.runtimeGetVersion()
        if runtime // 1000 != 12 or nvrtc_version != (12, 8):
            raise RuntimeError(f'Native CuPy loaded linked runtime {runtime}, NVRTC {nvrtc_version}; expected CUDA 12 / NVRTC 12.8')
        _BACKEND = (cp, ndimage, {'packages': packages, 'cupy': cp.__version__,
            'cuda_runtime_linked_version': runtime, 'cuda_driver_version': cp.cuda.runtime.driverGetVersion(),
            'actual_nvrtc_version': list(nvrtc_version), 'nvrtc_dll': str(required_paths[1]),
            'cuda_headers': str(cuda/'include'), 'cuda_runtime_distribution': '12.8.90',
            'kernel_cache': str(kernel_cache), 'kernel_cache_policy': 'persistent_D_cache_retained_between_jobs',
            'actual_compilation_temp': str(actual_compilation_temp),
            'cuda_minor_compatibility': 'CuPy links CUDA 12.9; installed runtime/headers and actual NVRTC are CUDA 12.8; NVCC is not used'})
        return _BACKEND


def _selection(values, *, distance=False):
    if _STATE.get() is None:
        return False, 'no_native_worker_scope'
    if values.ndim not in ((2,) if distance else (2, 3)):
        return False, 'unsupported_spatial_shape'
    if values.shape[0]*values.shape[1] < MIN_PIXELS:
        return False, 'small_node'
    if distance and (not np.any(values) or np.all(values)):
        return False, 'featureless_EDT_preserve_SciPy'
    return True, 'large_native_node'


def _execute(name, values, args, kwargs, *, distance=False):
    array = np.asarray(values)
    selected, reason = _selection(array, distance=distance)
    state = _STATE.get()
    started = time.perf_counter()
    row = {'operation': name, 'shape': list(array.shape), 'dtype': array.dtype.str,
           'backend': 'cupy_cuda' if selected else 'scipy_cpu', 'selection_reason': reason}
    if not selected:
        result = getattr(scipy_ndimage, name)(values, *args, **kwargs)
    else:
        try:
            with _BACKEND_LOCK:
                initial = time.perf_counter()
                was_loaded = _BACKEND is not None
                cp, ndimage, _ = _load_backend()
                if not was_loaded:
                    state['initialization_seconds'] += time.perf_counter()-initial
                with cp.cuda.Device(state['device']):
                    device = cp.asarray(array)
                    cp.cuda.Stream.null.synchronize()
                    copied_in = time.perf_counter()
                    gpu_kwargs = dict(kwargs)
                    if distance:
                        if any(gpu_kwargs.get(key) is not None for key in ('distances', 'indices')):
                            raise ValueError('Large native EDT does not accept host output buffers')
                        gpu_kwargs['float64_distances'] = True
                    device_result = getattr(ndimage, name)(device, *args, **gpu_kwargs)
                    cp.cuda.Stream.null.synchronize()
                    computed = time.perf_counter()
                    result = tuple(cp.asnumpy(v) for v in device_result) if isinstance(device_result, tuple) else cp.asnumpy(device_result)
                    if distance and gpu_kwargs.get('return_indices'):
                        # PBA may compact indices to int16 for these sizes;
                        # retain SciPy's public NumPy int32 index contract.
                        result = (result[0], result[1].astype(np.int32, copy=False)) if isinstance(result, tuple) else result.astype(np.int32, copy=False)
                    finished = time.perf_counter()
                    row.update(h2d_and_initialization_seconds=copied_in-started,
                               kernel_and_JIT_seconds=computed-copied_in, d2h_seconds=finished-computed)
                    del device, device_result
        except Exception as error:
            with state['_lock']:
                state['failures'].append({**row, 'error_type': type(error).__name__, 'error': str(error)})
            raise RuntimeError(f'Selected native CuPy {name} failed; CPU retry is disabled: {error}') from error
    row['inclusive_seconds'] = time.perf_counter()-started
    if state is not None:
        with state['_lock']:
            state['calls'].append(row)
    return result


def distance_transform_edt(values, sampling=None, return_distances=True, return_indices=False, distances=None, indices=None):
    return _execute('distance_transform_edt', values, (), {'sampling': sampling,
        'return_distances': return_distances, 'return_indices': return_indices,
        'distances': distances, 'indices': indices}, distance=True)


def median_filter(values, *args, **kwargs):
    return _execute('median_filter', values, args, kwargs)


def gaussian_filter(values, sigma, *args, **kwargs):
    return _execute('gaussian_filter', values, (sigma, *args), kwargs)


def ndimage_execution():
    """Actual scoped calls and memory, including CuPy outside Torch counters."""
    state = _STATE.get()
    if state is None:
        return {'enabled': False, 'backend': 'scipy_cpu', 'minimum_spatial_pixels': MIN_PIXELS,
                'CUDA_equality_gate': False}
    with state['_lock']:
        calls, failures = deepcopy(state['calls']), deepcopy(state['failures'])
    report = {'enabled': True, 'backend': 'official_cupy_scoped', 'device': state['device'],
        'minimum_spatial_pixels': MIN_PIXELS, 'counts_scope': 'cumulative_current_request',
        'gpu_calls': sum(c['backend'] == 'cupy_cuda' for c in calls),
        'cpu_calls': sum(c['backend'] == 'scipy_cpu' for c in calls),
        'calls': calls, 'failures': failures, 'initialization_seconds': state['initialization_seconds'],
        'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'distance_algorithm': 'official_CuPy_exact_PBA_plus_float64',
        'equidistant_nearest_indices_may_differ': True, 'CUDA_equality_gate': False}
    if _BACKEND is not None and report['gpu_calls']:
        cp, _, identity = _BACKEND
        with cp.cuda.Device(state['device']):
            report.update(package_identity=deepcopy(identity),
                cupy_pool_used_bytes=cp.get_default_memory_pool().used_bytes(),
                cupy_pool_retained_bytes=cp.get_default_memory_pool().total_bytes(),
                cuda_free_total_bytes=list(cp.cuda.runtime.memGetInfo()),
                cupy_pool_is_separate_from_Torch_counters=True)
    return report
