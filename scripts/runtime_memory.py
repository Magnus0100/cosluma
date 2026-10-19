"""Cheap Windows memory budgets for explicit model-to-model stage handoffs."""
from __future__ import annotations

import os

GIB = 1024 ** 3
# Pinned Nunchaku INT4 / complete Beauty LoRA / 1024 / 28 steps. The
# standalone real pilot reached 8.63 GiB WorkingSet and 20.08 GiB private
# commit. Keep the model demand and the system headroom as separate values;
# native CUDA allocations are not included in Torch allocator counters.
BEAUTY_COLD_PHYSICAL_BYTES = 20 * GIB
BEAUTY_COLD_COMMIT_BYTES = 20 * GIB
# A 64 GiB host needs only a small OS/application cushion after the measured
# model allocation. The previous 4 GiB reserve duplicated model headroom and
# rejected healthy machines when commit headroom was just under 24 GiB.
CPU_RESERVE_BYTES = 2 * GIB


def memory_status():
    if os.name != 'nt':
        return None
    import ctypes
    class MemoryStatus(ctypes.Structure):
        _fields_ = [('length', ctypes.c_uint32), ('load_percent', ctypes.c_uint32)] + [
            (name, ctypes.c_uint64) for name in (
                'total_physical', 'available_physical', 'total_commit', 'available_commit',
                'total_virtual', 'available_virtual', 'available_extended_virtual')]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MemoryStatus)]
    kernel.GlobalMemoryStatusEx.restype = ctypes.c_int
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return {'available_bytes': status.available_physical,
            'total_bytes': status.total_physical,
            'available_commit_bytes': status.available_commit,
            'load_percent': status.load_percent}


def memory_budget(next_stage_cpu_bytes, *, next_stage_commit_bytes=None, reserve_bytes=CPU_RESERVE_BYTES):
    """Check physical and commit demand separately; ordinary allocations use both."""
    if next_stage_commit_bytes is None:
        next_stage_commit_bytes = next_stage_cpu_bytes
    for value in (next_stage_cpu_bytes, next_stage_commit_bytes, reserve_bytes):
        if type(value) is not int or value < 0:
            raise ValueError('CPU allocation budgets must be nonnegative integer bytes')
    snapshot = memory_status()
    required_physical = next_stage_cpu_bytes + reserve_bytes
    required_commit = next_stage_commit_bytes + reserve_bytes
    fits = bool(snapshot and snapshot['available_bytes'] >= required_physical
                and snapshot['available_commit_bytes'] >= required_commit)
    return {'schema': 'cpu_stage_memory_budget.v2', 'snapshot': snapshot,
            'next_stage_cpu_bytes': next_stage_cpu_bytes, 'reserve_bytes': reserve_bytes,
            'next_stage_commit_bytes': next_stage_commit_bytes,
            'required_available_physical_bytes': required_physical,
            'required_available_commit_bytes': required_commit, 'fits': fits,
            'reason': ('budget_fits' if fits else
                       'memory_unknown' if snapshot is None else 'insufficient_available_memory')}


def native_analysis_cpu_bytes(run):
    # Native source/reference arrays and analysis models overlap. This is a
    # conservative estimate, retained in receipts; it does not inspect assets.
    sizes = [run.get('source', {}).get('oriented_size'),
             run.get('source', {}).get('native_size'),
             run.get('reference', {}).get('oriented_size')]
    pixels = max((size[0] * size[1] for size in sizes
                  if isinstance(size, (list, tuple)) and len(size) == 2
                  and all(type(n) is int and n > 0 for n in size)), default=0)
    return 16 * GIB + pixels * 64
