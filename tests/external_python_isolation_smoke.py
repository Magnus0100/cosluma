"""Offline Windows DLL inheritance proof using only an isolated test process."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
from unittest.mock import patch

import runtime_resources as runtime


PROBE = '''import ctypes,json,os
try:
    library=ctypes.WinDLL('cosplay-runtime-probe.dll',winmode=0)
    buffer=ctypes.create_unicode_buffer(32768)
    ctypes.windll.kernel32.GetModuleFileNameW(ctypes.c_void_p(library._handle),buffer,len(buffer))
    loaded=buffer.value
except OSError:
    loaded=None
print(json.dumps({'loaded':loaded,'path':os.environ.get('PATH',''),'marker':os.environ.get('COSPLAY_DLL_FIXTURE')}))
'''


def main():
    # Ordinary source launches remain delegated to the existing subprocess API.
    with patch.object(sys, 'frozen', False, create=True), \
         patch.object(runtime.subprocess, 'run', return_value='source-result') as launch:
        assert runtime.external_python_run(['fixture'], timeout=12) == 'source-result'
        assert launch.call_args.kwargs['timeout'] == 12
        assert 'env' not in launch.call_args.kwargs
    with patch.object(sys, 'frozen', False, create=True), \
         patch.object(runtime.subprocess, 'Popen', return_value='source-process') as launch:
        assert runtime.external_python_popen(['fixture']) == 'source-process'
        assert launch.call_args.kwargs == {}

    with patch.object(sys, 'frozen', True, create=True), \
         patch.object(sys, '_MEIPASS', r'C:\Fixture\_internal', create=True), \
         patch.object(sys, 'platform', 'win32'):
        keep = [r'C:\Windows\System32', r'C:\Fixture\_internal-other',
                r'C:\OtherRuntime\Scripts', r'D:\Fixture\_internal', '']
        entries = [r'C:\FIXTURE\_INTERNAL', r'"c:/fixture/_internal/PySide6"',
                   r'C:\Fixture\_internal\numpy\..\scipy', *keep]
        environment = runtime._without_bundle_path({'Path': ';'.join(entries), 'TOKEN': 'keep'})
        assert environment == {'Path': ';'.join(keep), 'TOKEN': 'keep'}
    if sys.platform != 'win32':
        print(json.dumps({'ok': True, 'windows_dll_inheritance': 'requires Windows'}))
        return 0

    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.SetDllDirectoryW.argtypes = (ctypes.c_wchar_p,)
    kernel.SetDllDirectoryW.restype = ctypes.c_int
    kernel.GetDllDirectoryW.argtypes = (ctypes.c_ulong, ctypes.c_wchar_p)
    kernel.GetDllDirectoryW.restype = ctypes.c_ulong

    def directory():
        buffer = ctypes.create_unicode_buffer(32768)
        size = kernel.GetDllDirectoryW(len(buffer), buffer)
        assert size < len(buffer)
        return buffer.value

    original = directory()
    # The licensed VC runtime already belongs to this desktop Python install.
    # A unique name proves which private directory the child actually searched.
    dll = Path(sys._base_executable).parent / 'vcruntime140.dll'
    if not dll.is_file():
        raise AssertionError('Desktop Python VC runtime DLL is required for the Windows fixture')
    with tempfile.TemporaryDirectory(prefix='cosplay-dll-isolation-') as temporary:
        bundle = Path(temporary) / '_internal'
        bundle.mkdir()
        shutil.copyfile(dll, bundle / 'cosplay-runtime-probe.dll')
        assert kernel.SetDllDirectoryW(str(bundle))
        try:
            command = [sys.executable, '-B', '-c', PROBE]
            base_environment = dict(os.environ)
            base_environment['PATH'] = str(bundle) + ';' + base_environment.get('PATH', '')
            base_environment['COSPLAY_DLL_FIXTURE'] = 'retained'
            # The control excludes PATH, isolating SetDllDirectory inheritance.
            control_environment = {**base_environment, 'PATH': os.environ.get('PATH', '')}
            control = subprocess.run(command, env=control_environment, capture_output=True,
                                     text=True, check=True)
            assert Path(json.loads(control.stdout)['loaded']).parent == bundle
            with patch.object(sys, 'frozen', True, create=True), \
                 patch.object(sys, '_MEIPASS', str(bundle), create=True):
                child = runtime.external_python_popen(command, env=base_environment,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                assert directory() == str(bundle)
                stdout, stderr = child.communicate(timeout=15)
                assert child.returncode == 0, stderr
                receipt = json.loads(stdout)
                assert receipt['loaded'] is None, receipt
                assert str(bundle) not in receipt['path']
                assert receipt['marker'] == 'retained'
                assert base_environment['PATH'].startswith(str(bundle) + ';')

                # Popen failure must restore the real parent directory too.
                with patch.object(runtime.subprocess, 'Popen', side_effect=OSError('fixture spawn failure')):
                    try:
                        runtime.external_python_popen(command)
                    except OSError as error:
                        assert str(error) == 'fixture spawn failure'
                    else:
                        raise AssertionError('Spawn failure was swallowed')
                assert directory() == str(bundle)

                # Two launch threads must never save another thread's cleared
                # directory as their own parent state.
                observed = []
                def fake_launch(*args, **kwargs):
                    observed.append(directory())
                    return object()
                with patch.object(runtime.subprocess, 'Popen', side_effect=fake_launch), \
                     ThreadPoolExecutor(max_workers=2) as pool:
                    list(pool.map(lambda _: runtime.external_python_popen(command), range(8)))
                assert observed == [''] * 8
                assert directory() == str(bundle)

                # communicate/timeout/check keep run's existing semantics, and
                # the global DLL lock is already free while a child is running.
                created = threading.Event()
                actual_popen = subprocess.Popen
                def real_launch(*args, **kwargs):
                    process = actual_popen(*args, **kwargs)
                    created.set()
                    return process
                with patch.object(runtime.subprocess, 'Popen', side_effect=real_launch), \
                     ThreadPoolExecutor(max_workers=1) as pool:
                    pending = pool.submit(runtime.external_python_run,
                        [sys.executable, '-c', 'import time;time.sleep(.5);print("ready")'],
                        capture_output=True, text=True, check=True)
                    assert created.wait(5)
                    assert runtime._EXTERNAL_LAUNCH_LOCK.acquire(timeout=.25)
                    try:
                        assert directory() == str(bundle)
                        assert not pending.done()
                    finally:
                        runtime._EXTERNAL_LAUNCH_LOCK.release()
                    assert pending.result(timeout=10).stdout.strip() == 'ready'
                try:
                    runtime.external_python_run([sys.executable, '-c', 'raise SystemExit(7)'],
                        capture_output=True, text=True, check=True)
                except subprocess.CalledProcessError as error:
                    assert error.returncode == 7
                else:
                    raise AssertionError('check=True lost the child exit status')
                try:
                    runtime.external_python_run([sys.executable, '-c',
                        'import time;print("before timeout",flush=True);time.sleep(5)'],
                        capture_output=True, timeout=.3)
                except subprocess.TimeoutExpired as error:
                    assert b'before timeout' in error.stdout
                else:
                    raise AssertionError('Child timeout was ignored')
                assert directory() == str(bundle)
        finally:
            assert kernel.SetDllDirectoryW(original or None)
    print(json.dumps({'ok': True, 'private_dll_control_loaded': True,
        'external_child_private_dll_absent': True, 'parent_restored': True,
        'spawn_failure_restored': True, 'source_delegation': True,
        'concurrent_launches': True, 'wait_outside_dll_lock': True}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
