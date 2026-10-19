"""Bounded readonly proxy-field residency for one native matching request.

The CPU owns original-coordinate indices/float64 weights and output QA. Only
explicitly admitted B/S fields and owners may be uploaded, and each reserved
bank uploads once on its first eligible GPU sample. No tensor crosses a request
boundary; desktop/out-of-scope and small nodes keep mature CPU code.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
import hashlib
import inspect
import sys
from threading import Lock
import time

import numpy as np

MIN_PIXELS = 200_000
MAX_RESIDENT_BYTES = 256*1024*1024
_CURRENT = ContextVar('native_sampling_request', default=None)


def sampling_profile():
    """Stable recipe identity, independent of dynamic calls and device state."""
    return {'schema': 'native_owner_sampling_request.v2', 'GPU_enabled_only_in_native_field_scope': True,
        'GPU_backend': 'torch_lazy_index_select_readonly_request_lease',
        'GPU_bank_admission': 'owning_snapshot_reserved_at_request_lazy_upload_on_first_eligible_sample',
        'small_or_unscoped_backend': 'numpy_cpu',
        'minimum_native_pixels': MIN_PIXELS, 'maximum_resident_bytes': MAX_RESIDENT_BYTES,
        'field_dtype': 'float32', 'owner_dtype': 'int32', 'coordinates': 'CPU_global_int32_indices_float64_weights',
        'CPU_CUDA_bit_equality_gate': False, 'cross_request_GPU_cache': False}


def _sha(values):
    digest = hashlib.sha256(str((values.shape, values.dtype.str)).encode())
    digest.update(memoryview(np.ascontiguousarray(values)).cast('B'))
    return digest.hexdigest()


@dataclass
class _Lease:
    entries: dict = field(default_factory=dict)
    report: dict = field(default_factory=lambda: {**sampling_profile(), 'gpu_calls': {}, 'cpu_calls': {},
        'admission': [], 'host_snapshot_bytes': 0, 'resident_H2D_bytes': 0, 'plan_support_H2D_bytes': 0,
        'result_D2H_bytes': 0, 'failures': [], 'lease_released': False,
        'source_and_snapshot_hashes_closed': False})
    lock: object = field(default_factory=Lock)
    torch: object = None
    device: object = None


def _entry_admission(entry):
    """Return the stable receipt row for one admitted source bank."""
    return entry.get('admission')


def _ensure_entry_device(lease, entry, fields):
    """Upload one reserved bank exactly when a validated GPU sample needs it.

    Admission deliberately performs the owning host snapshot, hashes and byte
    reservation, but does not import Torch or initialize a CUDA context.  The
    first eligible sampler holds the lease lock while constructing both banks;
    this makes concurrent ordered-map workers idempotent and prevents a partial
    value/owner upload from becoming a usable entry.
    """
    with lease.lock:
        if entry.get('bank') is not None and entry.get('owner_bank') is not None:
            return lease.torch, lease.device
        try:
            if lease.torch is None:
                import torch
                lease.torch, lease.device = torch, fields._device()
                lease.report.update(torch_version=torch.__version__, device=lease.device)
            torch = lease.torch
            bank = torch.tensor(entry['values'].reshape(-1, entry['values'].shape[-1]),
                                device=lease.device, dtype=torch.float32)
            owner_bank = torch.tensor(entry['owners'].reshape(-1),
                                      device=lease.device, dtype=torch.int32)
            # Publish the pair together.  If the owner upload failed, neither
            # temporary is retained by the lease and a later retry cannot use a
            # half-initialized request bank.
            entry['bank'], entry['owner_bank'] = bank, owner_bank
            entry['uploaded'] = True
            count = entry['reserved_bytes']
            lease.report['resident_H2D_bytes'] += count
            row = _entry_admission(entry)
            if row is not None:
                row.update(uploaded=True, resident_bytes=count)
            lease.report['banks_uploaded'] = lease.report.get('banks_uploaded', 0) + 1
            return torch, lease.device
        except BaseException:
            entry.pop('bank', None); entry.pop('owner_bank', None)
            raise


def _memory_report(lease, phase):
    cuda = lease.torch.cuda
    queries = {'cuda_free_total_'+phase+'_bytes': lambda: list(cuda.mem_get_info(lease.device))}
    if phase == 'before_release':
        queries.update(torch_peak_allocated_bytes=lambda: cuda.max_memory_allocated(lease.device),
                       torch_peak_reserved_bytes=lambda: cuda.max_memory_reserved(lease.device))
        lease.report['torch_peak_scope'] = 'worker_cumulative_since_callers_last_reset'
    else:
        queries['torch_live_allocated_after_release_bytes'] = lambda: cuda.memory_allocated(lease.device)
    for name, query in queries.items():
        try:
            lease.report[name] = query()
        except Exception as error:
            lease.report.setdefault('diagnostic_failures', []).append({'metric': name,
                'type': type(error).__name__, 'error': str(error)})


@contextmanager
def _request(bound, pairs):
    import native_cuda_fields as fields
    field_state = fields._STATE.get()
    if field_state is None:
        yield bound
        return
    if _CURRENT.get() is not None:
        raise RuntimeError('Native sampling leases cannot overlap in one request context')
    lease = _Lease(); token = _CURRENT.set(lease)
    started = time.perf_counter()
    try:
        lease.report.setdefault('reserved_bytes', 0)
        lease.report.setdefault('banks_uploaded', 0)
        native = tuple(bound.arguments.get('native', ()))
        for value_name, owner_name in pairs:
            original, owners = bound.arguments[value_name], bound.arguments[owner_name]
            if isinstance(original, np.ndarray) and original.ndim >= 2 and native and (
                    native == (original.shape[1], original.shape[0]) or native[0]*native[1] < MIN_PIXELS):
                lease.report.setdefault('admission_skips', []).append({'field': value_name,
                    'reason': 'native_identity_grid' if native == (original.shape[1], original.shape[0]) else 'small_native_canvas'})
                continue
            if (not isinstance(original, np.ndarray) or not isinstance(owners, np.ndarray)
                    or original.ndim != 3 or original.shape[-1] not in (4, 6) or original.dtype != np.float32
                    or owners.dtype != np.int32 or owners.shape != original.shape[:2]
                    or original.shape[0]*original.shape[1] < MIN_PIXELS):
                continue
            count = original.nbytes+owners.nbytes
            if lease.report['reserved_bytes']+count > MAX_RESIDENT_BYTES:
                raise ValueError('Native proxy sampling request exceeded its resident byte cap')
            # Owning copies establish immutable backing even when np.load or
            # a caller supplies readonly views into another mutable array.
            values_copy, owners_copy = original.copy(order='C'), owners.copy(order='C')
            values_copy.setflags(write=False); owners_copy.setflags(write=False)
            values_sha, owners_sha = _sha(values_copy), _sha(owners_copy)
            key = id(values_copy), id(owners_copy)
            lease.entries[key] = {'key': key, 'values': values_copy, 'owners': owners_copy, 'original': original,
                'original_owners': owners, 'values_sha': values_sha, 'owners_sha': owners_sha,
                'bank': None, 'owner_bank': None, 'uploaded': False,
                'reserved_bytes': count}
            bound.arguments[value_name], bound.arguments[owner_name] = values_copy, owners_copy
            lease.report['host_snapshot_bytes'] += count
            lease.report['reserved_bytes'] += count
            admission = {'field': value_name, 'shape': list(original.shape), 'dtype': original.dtype.str,
                'owner_dtype': owners.dtype.str, 'live_values_sha256': values_sha, 'live_owners_sha256': owners_sha,
                'reserved_bytes': count, 'resident_bytes': 0, 'uploaded': False}
            lease.entries[key]['admission'] = admission
            lease.report['admission'].append(admission)
        lease.report['admission_seconds'] = time.perf_counter()-started
        yield bound
    except BaseException as error:
        lease.report['failures'].append({'phase': 'request', 'type': type(error).__name__, 'error': str(error)})
        raise
    finally:
        cleanup = time.perf_counter()
        caller_error = sys.exception()
        try:
            try:
                if lease.torch is not None:
                    lease.torch.cuda.synchronize(lease.device)
                    _memory_report(lease, 'before_release')
                for entry in lease.entries.values():
                    if (entry['values'].flags.writeable or entry['owners'].flags.writeable
                            or _sha(entry['values']) != entry['values_sha'] or _sha(entry['owners']) != entry['owners_sha']
                            or _sha(entry['original']) != entry['values_sha'] or _sha(entry['original_owners']) != entry['owners_sha']):
                        raise ValueError('Native proxy sampling source/snapshot changed during the request')
                if lease.entries:
                    del entry
                lease.report['source_and_snapshot_hashes_closed'] = True
            except BaseException as error:
                lease.report['failures'].append({'phase': 'source_hash_closure', 'type': type(error).__name__, 'error': str(error)})
                if caller_error is None:
                    raise
            finally:
                for retained in lease.entries.values():
                    retained.pop('bank', None); retained.pop('owner_bank', None)
                if lease.entries:
                    del retained
                lease.entries.clear()
                lease.report.update(lease_released=True, live_entries_after_release=0,
                    cleanup_seconds=time.perf_counter()-cleanup,
                    actual_backend='failed_selected_GPU' if lease.report['failures'] else
                        'torch_cuda_lazy_request_lease' if lease.report['gpu_calls'] else 'numpy_cpu')
                if lease.torch is not None:
                    _memory_report(lease, 'after_release')
                try:
                    with field_state['_lock']:
                        field_state.setdefault('sampling_requests', []).append(lease.report)
                except Exception as error:
                    lease.report.setdefault('diagnostic_failures', []).append({'metric': 'field_scope_append',
                        'type': type(error).__name__, 'error': str(error)})
        finally:
            _CURRENT.reset(token)


def request_sampling_lease(*pairs):
    """Admit validated source banks while the actual joined matching runs."""
    def decorate(function):
        signature = inspect.signature(function)
        if inspect.isgeneratorfunction(function):
            @wraps(function)
            def generate(*args, **kwargs):
                with _request(signature.bind(*args, **kwargs), pairs) as bound:
                    iterator = function(*bound.args, **bound.kwargs)
                    try:
                        yield from iterator
                    finally:
                        iterator.close()
            return generate
        @wraps(function)
        def call(*args, **kwargs):
            with _request(signature.bind(*args, **kwargs), pairs) as bound:
                return function(*bound.args, **bound.kwargs)
        return call
    return decorate


def sample_scoped_maps(values, box, proxy_size, native_size, *, owners, grid, hair_face_support, sample_plan):
    lease = _CURRENT.get()
    if lease is None or owners is None:
        return None
    entry = lease.entries.get((id(values), id(owners)))
    label = 'S' if values.ndim == 3 and values.shape[-1] == 4 else 'B'
    shape = box[3]-box[1], box[2]-box[0]
    if entry is None or shape[0]*shape[1] < MIN_PIXELS or tuple(proxy_size) == tuple(native_size):
        with lease.lock:
            lease.report['cpu_calls'][label] = lease.report['cpu_calls'].get(label, 0)+1
        return None
    started = time.perf_counter()
    if (entry['values'] is not values or entry['owners'] is not owners or values.flags.writeable or owners.flags.writeable
            or owners.shape != values.shape[:2] or tuple(proxy_size) != (values.shape[1], values.shape[0])):
        raise ValueError('Native owner sampler cannot consume a changed/unrelated request bank')
    from render_reference_colour_composite import _sample_grid, _sample_plan
    grid = _sample_grid(box, proxy_size, native_size) if grid is None else grid
    plan = _sample_plan(grid, values.shape[:2]) if sample_plan is None else sample_plan
    if (len(grid) != 2 or grid[0].shape != shape or grid[1].shape != shape
            or plan['xx'].shape != shape or plan['yy'].shape != shape
            or tuple(plan.get('values_shape', ())) != values.shape[:2]):
        raise ValueError('Native owner sampler requires the original global coordinate plan')
    bank_length = values.shape[0]*values.shape[1]
    indices = (plan['owner_indices'], *plan['indices'])
    if len(indices) != 5 or len(plan['weights']) != 4:
        raise ValueError('Native owner sampler requires four original neighbours and weights')
    for index in indices:
        if not isinstance(index, np.ndarray) or index.shape != shape or index.dtype.kind not in 'iu':
            raise ValueError('Native owner sampler indices must be integer arrays on the target grid')
        low, high = index.min(), index.max()
        if low < 0 or high >= bank_length or high > np.iinfo(np.int32).max:
            raise ValueError('Native owner sampler indices must be integer, int32-safe and within its actual bank')
    for weight in plan['weights']:
        if (not isinstance(weight, np.ndarray) or weight.shape != shape or weight.dtype.kind not in 'fiu'
                or not np.isfinite(weight).all()):
            raise ValueError('Native owner sampler weights must be finite real arrays on the target grid')
    support = None if hair_face_support is None else np.asarray(hair_face_support)
    if support is not None and (support.shape != shape or not np.isfinite(support).all() or support.min() < 0 or support.max() > 1):
        raise ValueError('Native owner sampler requires valid source face/hair veto support')
    positions = weight = domain = result = total = dose = visible = None
    try:
        import native_cuda_fields as fields
        torch, device = _ensure_entry_device(lease, entry, fields)
        pixels, channels = shape[0]*shape[1], values.shape[-1]
        positions = torch.tensor(plan['owner_indices'].reshape(-1), device=device, dtype=torch.int32)
        domain = torch.index_select(entry['owner_bank'], 0, positions)
        result = torch.zeros((pixels, channels), device=device, dtype=torch.float32)
        total = torch.zeros(pixels, device=device, dtype=torch.float32)
        transferred = pixels*4
        for indices, weights in zip(plan['indices'], plan['weights']):
            positions = torch.tensor(indices.reshape(-1), device=device, dtype=torch.int32)
            weight = torch.tensor(weights.reshape(-1), device=device, dtype=torch.float64)
            dose = (weight*(torch.index_select(entry['owner_bank'], 0, positions) == domain)).to(torch.float32)
            result.add_(torch.index_select(entry['bank'], 0, positions)*dose[:, None]); total.add_(dose)
            transferred += positions.numel()*4+weight.numel()*8
        result.div_(torch.clamp(total, min=1e-12)[:, None])
        if support is not None:
            visible = torch.tensor(support.reshape(-1), device=device, dtype=torch.float32)
            dose = torch.where((domain == 4) | (domain == 1004), 1-visible, 1)
            result[:, :2 if channels == 6 else channels-1].mul_(dose[:, None]); transferred += support.nbytes
        output = result.cpu().numpy().reshape((*shape, channels))
        with lease.lock:
            lease.report['gpu_calls'][label] = lease.report['gpu_calls'].get(label, 0)+1
            lease.report['plan_support_H2D_bytes'] += transferred
            lease.report['result_D2H_bytes'] += output.nbytes
            lease.report['GPU_sample_thread_seconds_sum'] = lease.report.get('GPU_sample_thread_seconds_sum', 0.)+time.perf_counter()-started
        return output
    except Exception as error:
        with lease.lock:
            lease.report['failures'].append({'phase': 'GPU_sample', 'type': type(error).__name__, 'error': str(error)})
        raise RuntimeError('Selected native GPU owner gather failed; CPU retry is disabled: '+str(error)) from error
    finally:
        positions = weight = domain = result = total = dose = visible = None
