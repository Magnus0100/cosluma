"""Check DLL-handle lifetime and frozen-only isolation before the real build."""
import os
from pathlib import Path
import runpy
import sys
import tempfile
from unittest.mock import patch


def main():
    hook = Path(__file__).resolve().parents[1]/'scripts/frozen_native_paths.py'
    original_path = list(sys.path)
    original_environment = dict(os.environ)
    with patch.object(sys, 'frozen', False, create=True), patch.object(os, 'add_dll_directory') as add:
        source = runpy.run_path(str(hook))
        assert source['configure_native_paths']() is None and not add.called
    assert list(sys.path) == original_path and dict(os.environ) == original_environment
    with tempfile.TemporaryDirectory(prefix='cosplay_frozen_paths_') as folder:
        root = Path(folder).resolve()
        for name in ('numpy.libs', 'rawpy', 'PySide6', 'onnxruntime/capi'):
            (root/name).mkdir(parents=True)
        handles = []
        class Handle:
            def __init__(self, name): self.name, self.closed = name, False
            def close(self): self.closed = True
        def add(name):
            handle = Handle(name); handles.append(handle); return handle
        with patch.object(sys, 'frozen', True, create=True), patch.object(sys, '_MEIPASS', str(root), create=True), \
             patch.object(sys, 'platform', 'win32'), patch.object(sys, '_cosplay_native_paths', None, create=True), \
             patch.object(sys, '_cosplay_native_dll_handles', (), create=True), \
             patch.object(os, 'add_dll_directory', side_effect=add):
            frozen = runpy.run_path(str(hook))
            report = sys._cosplay_native_paths
            assert len(handles) == 5 and len(sys._cosplay_native_dll_handles) == 5
            assert not any(handle.closed for handle in handles)
            assert all(Path(name).is_relative_to(root) for name in report['registered_directories'])
            assert set(report['missing_optional_directories']) == {'scipy.libs', 'shiboken6'}
            assert frozen['configure_native_paths']() is report and len(handles) == 5
            assert list(sys.path) == original_path and dict(os.environ) == original_environment
            with patch.object(sys, '_cosplay_native_paths', None), patch.object(os, 'add_dll_directory', side_effect=OSError('test failure')):
                failure = frozen['configure_native_paths']()
                assert len(failure['errors']) == 5 and not failure['registered_directories']
    print('OK: frozen Windows payload only; retained DLL-directory handles; idempotent; absent sidecars optional; native failures recorded; source runtime/module paths/environment exact')


if __name__ == '__main__':
    main()
