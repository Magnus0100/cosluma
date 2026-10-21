"""Locate shipped Python sources independently of project data and models."""
from pathlib import Path
from contextlib import contextmanager
import ntpath
import os
import subprocess
import sys
import threading

from asset_identity import sha256_file


_EXTERNAL_LAUNCH_LOCK = threading.RLock()


def _without_bundle_path(environment):
    """Remove frozen DLL directories only; keep independently installed runtimes."""
    bundle = getattr(sys, '_MEIPASS', None)
    if not getattr(sys, 'frozen', False) or not bundle or sys.platform != 'win32':
        return environment
    root = ntpath.normcase(ntpath.abspath(str(bundle)))
    for key in tuple(environment):
        if key.upper() != 'PATH':
            continue
        retained = []
        for entry in environment[key].split(';'):
            candidate = ntpath.normcase(ntpath.abspath(entry.strip().strip('"')))
            try:
                bundled = ntpath.commonpath((root, candidate)) == root
            except ValueError:
                bundled = False
            if not bundled:
                retained.append(entry)
        environment[key] = ';'.join(retained)
    return environment


@contextmanager
def _external_dll_search():
    """Restore the frozen parent's exact DLL directory immediately after spawn."""
    if not (sys.platform == 'win32' and getattr(sys, 'frozen', False)
            and getattr(sys, '_MEIPASS', None)):
        yield
        return
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    get_directory = kernel.GetDllDirectoryW
    get_directory.argtypes = (wintypes.DWORD, wintypes.LPWSTR)
    get_directory.restype = wintypes.DWORD
    set_directory = kernel.SetDllDirectoryW
    set_directory.argtypes = (wintypes.LPCWSTR,)
    set_directory.restype = wintypes.BOOL
    # SetDllDirectory is process-wide, so all external worker launches share
    # this lock. Never hold it while waiting for imports, inference or pipes.
    with _EXTERNAL_LAUNCH_LOCK:
        ctypes.set_last_error(0)
        size = get_directory(0, None)
        if not size and ctypes.get_last_error():
            raise ctypes.WinError(ctypes.get_last_error())
        previous = None
        if size:
            buffer = ctypes.create_unicode_buffer(size + 1)
            copied = get_directory(len(buffer), buffer)
            if not copied or copied >= len(buffer):
                raise ctypes.WinError(ctypes.get_last_error())
            previous = buffer.value
        if not set_directory(None):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            yield
        finally:
            if not set_directory(previous):
                raise ctypes.WinError(ctypes.get_last_error())


def external_python_popen(*args, **kwargs):
    """Launch a registered Python runtime without the frozen desktop's DLLs."""
    if getattr(sys, 'frozen', False):
        kwargs['env'] = isolated_python_environment(kwargs.get('env'))
    with _external_dll_search():
        return subprocess.Popen(*args, **kwargs)


def external_python_run(*args, input=None, capture_output=False, timeout=None,
                        check=False, **kwargs):
    """Use run semantics with DLL isolation limited to process creation."""
    if not getattr(sys, 'frozen', False):
        return subprocess.run(*args, input=input, capture_output=capture_output,
                              timeout=timeout, check=check, **kwargs)
    if input is not None:
        if kwargs.get('stdin') is not None:
            raise ValueError('stdin and input arguments may not both be used')
        kwargs['stdin'] = subprocess.PIPE
    if capture_output:
        if kwargs.get('stdout') is not None or kwargs.get('stderr') is not None:
            raise ValueError('stdout and stderr arguments may not be used with capture_output')
        kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    with external_python_popen(*args, **kwargs) as process:
        try:
            stdout, stderr = process.communicate(input, timeout=timeout)
        except subprocess.TimeoutExpired as error:
            process.kill()
            if sys.platform == 'win32':
                error.stdout, error.stderr = process.communicate()
            else:
                process.wait()
            raise
        except BaseException:
            process.kill()
            raise
        returncode = process.poll()
        if check and returncode:
            raise subprocess.CalledProcessError(returncode, process.args,
                                                output=stdout, stderr=stderr)
    return subprocess.CompletedProcess(process.args, returncode, stdout, stderr)


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
    if getattr(sys, 'frozen', False):
        # A directly launched EXE has no launcher-provided package variable.
        # External Python workers still need it to resolve portable model paths.
        environment.setdefault('COSPLAY_PACKAGE_ROOT', str(Path(sys.executable).resolve().parent))
    configure_cpu_runtime(environment)
    environment.update(PYTHONNOUSERSITE='1', PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    return _without_bundle_path(environment)


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
