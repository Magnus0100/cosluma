"""PyInstaller startup hook for bundled Windows native dependencies.

Runs before NumPy, rawpy, ONNX Runtime and Qt runtime hooks. Only directories
inside the frozen payload participate in DLL lookup. Keep AddDllDirectory
handles alive on sys: dropping a handle immediately removes that directory.
Project roots, model runtimes, Python module paths and the source runtime are
not rewritten here.
"""
from pathlib import Path
import os
import sys
from runtime_resources import configure_cpu_runtime

configure_cpu_runtime()


DIRECTORIES = ('.', 'numpy.libs', 'scipy.libs', 'rawpy', 'onnxruntime/capi',
               'PySide6', 'shiboken6')


def configure_native_paths():
    if not getattr(sys, 'frozen', False) or sys.platform != 'win32':
        return None
    payload = getattr(sys, '_MEIPASS', None)
    if not payload:
        raise RuntimeError('Frozen native DLL setup requires the PyInstaller payload root')
    root = Path(payload).resolve()
    if not root.is_dir():
        raise RuntimeError('Frozen native DLL payload directory is missing')
    existing = getattr(sys, '_cosplay_native_paths', None)
    if existing and existing['bundle_root'] == str(root):
        return existing
    handles = []
    report = {'schema': 'frozen_native_paths.v1', 'bundle_root': str(root),
              'registered_directories': [], 'missing_optional_directories': [],
              'errors': [], 'source_runtime_changed': False, 'module_path_changed': False}
    for name in DIRECTORIES:
        directory = (root/name).resolve()
        if not directory.is_relative_to(root):
            raise RuntimeError('Frozen native DLL directory escapes the payload')
        if not directory.is_dir():
            report['missing_optional_directories'].append(name)
            continue
        try:
            handle = os.add_dll_directory(str(directory))
        except OSError as error:
            report['errors'].append({'directory': str(directory), 'error': str(error)})
        else:
            handles.append(handle)
            report['registered_directories'].append(str(directory))
    # Runtime-hook globals may be replaced as the entry script starts. sys
    # provides a process-lifetime owner independent of __main__ execution.
    sys._cosplay_native_dll_handles = tuple(handles)
    sys._cosplay_native_paths = report
    return report


configure_native_paths()
