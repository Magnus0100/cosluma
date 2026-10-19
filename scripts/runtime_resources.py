"""Locate shipped Python sources independently of project data and models."""
from pathlib import Path
import os
import sys

from asset_identity import sha256_file


def configure_cpu_runtime(environment=None) -> dict:
    """Set the BLAS default before importing NumPy; keep explicit settings."""
    target = os.environ if environment is None else environment
    target.setdefault('OPENBLAS_NUM_THREADS', '1')
    return target


def isolated_python_environment(overrides=None) -> dict:
    """Keep storage/service settings while isolating a selected interpreter."""
    environment = {**os.environ, **(overrides or {})}
    for name in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONUSERBASE'):
        environment.pop(name, None)
    configure_cpu_runtime(environment)
    environment.update(PYTHONNOUSERSITE='1', PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    return environment


def ordered_native_map(compute, items, *, workers=2):
    """Bound independent native work while leaving ordered writes to callers."""
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    from contextvars import copy_context

    if not isinstance(workers, int) or workers < 1:
        raise ValueError('Native worker count must be a positive integer')
    workers = min(4, workers, os.cpu_count() or 1)
    if workers == 1:
        for item in items:
            yield compute(item)
        return
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='native-bands') as executor:
        pending = deque()
        for item in items:
            pending.append(executor.submit(copy_context().run, compute, item))
            if len(pending) == workers:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def code_root() -> Path:
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS).resolve()
    return Path(__file__).resolve().parent


def code_path(filename: str) -> Path:
    if Path(filename).name != filename or not filename.endswith('.py'):
        raise ValueError('Expected a Python source filename')
    return code_root() / filename


def resource_report(filenames) -> dict:
    paths = {name: code_path(name) for name in filenames}
    missing = [name for name, path in paths.items() if not path.is_file()]
    return {'code_root': str(code_root()), 'project_scripts_required': False,
            'source_count': len(paths), 'missing_sources': missing, 'ok': not missing,
            'sources': {name: {'path': str(path), 'sha256': sha256_file(path)}
                        for name, path in paths.items() if path.is_file()}}
