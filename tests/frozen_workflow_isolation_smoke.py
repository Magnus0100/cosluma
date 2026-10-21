"""Frozen one-click isolation uses owned external Python, never the GUI interpreter."""
from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

import backend_server as module
from backend_server import Backend
from job_store import process_stamp


def main():
    with tempfile.TemporaryDirectory(prefix='cosplay-frozen-isolation-') as folder:
        root = Path(folder)
        backend = Backend(root, embedded=True)
        source = root/'source.jpg'; source.write_bytes(b'not decoded by this test')
        cli = root/'_internal'/'project_cli.py'
        cli.parent.mkdir()
        cli.write_text('''import json,os,sys,time
from pathlib import Path
arguments=sys.argv[1:]
root=Path(arguments[arguments.index('--root')+1])
options=Path(arguments[arguments.index('--options-file')+1])
record=json.loads(options.read_text(encoding='utf-8'))
if record.get('character_notes')=='direct-exe':
 location='${PACKAGE_ROOT}/model/installation-marker.json'.replace('${PACKAGE_ROOT}',os.environ.get('COSPLAY_PACKAGE_ROOT',''))
 assert json.loads(Path(location).read_text())=={'installed':True}
if record.get('character_notes')=='wait': time.sleep(30)
time.sleep(.25)
print(json.dumps({'run_id':arguments[arguments.index('--run-id')+1],
 'sample':'fixture','status':'needs_visual_review'}))
''', encoding='utf-8')
        latest = root/'outputs/latest.json'; latest.parent.mkdir(exist_ok=True)
        latest.write_bytes(b'previous saved version remains byte-identical')
        saved = latest.read_bytes()
        runtime = Path(sys.executable).resolve()
        fake_exe = root/'cosplay-workbench.exe'; fake_exe.touch()
        (root/'model').mkdir()
        (root/'model/installation-marker.json').write_text('{"installed":true}',encoding='utf-8')
        submitted = []
        real_popen = subprocess.Popen

        def capture(argv, **kwargs):
            submitted.append((argv, kwargs))
            return real_popen(argv, **kwargs)

        def execute(name, **extra):
            backend.jobs.create({'kind': 'one_click', 'source': str(source)}, job_id=name)
            backend._run_one_click_job(name, {'source': str(source), 'generate': False, **extra})
            return backend.jobs.get(name)

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(sys, 'frozen', True, create=True))
            stack.enter_context(patch.object(sys, 'executable', str(fake_exe)))
            registry = stack.enter_context(patch('current_model_registry.load_registry', return_value={'python': str(runtime)}))
            stack.enter_context(patch.object(module, 'code_path', return_value=cli))
            stack.enter_context(patch.dict(os.environ, {'COSPLAY_PACKAGE_ROOT': str(root/'installed-package'),
                'PYTHONPATH': 'must-be-removed', 'PYTHONHOME': 'must-be-removed'}))
            command = stack.enter_context(patch.object(backend, 'command', side_effect=AssertionError('GUI must not execute production')))
            with patch.object(module.subprocess, 'Popen', side_effect=capture):
                healthy = execute('job-isolated-healthy', model_registry='selected-models.json')
            assert healthy['status'] == 'succeeded'
            registry.assert_called_once_with(root, registry_path='selected-models.json')
            argv, kwargs = submitted[-1]
            assert argv[:2] == [str(runtime), str(cli)] and argv[0] != str(fake_exe)
            assert kwargs['cwd'] == str(root) and kwargs['cwd'] != str(cli.parent)
            assert kwargs['env']['COSPLAY_PACKAGE_ROOT'] == str(root/'installed-package')
            assert 'PYTHONPATH' not in kwargs['env'] and 'PYTHONHOME' not in kwargs['env']
            assert kwargs['env']['PYTHONFAULTHANDLER'] == '1'
            assert kwargs['creationflags'] == getattr(subprocess, 'CREATE_NO_WINDOW', 0)
            assert healthy['child_process']['pid'] != os.getpid()
            assert healthy['child_process']['stamp']
            assert healthy['child_process']['owner_pid'] == os.getpid()
            assert healthy['child_process']['owner_stamp'] == process_stamp(os.getpid())
            assert healthy['execution_process_isolation'] == 'external_python'
            assert not backend._job_processes and latest.read_bytes() == saved

            # Direct double-click has no launcher environment. A real external
            # worker must still resolve ${PACKAGE_ROOT} against the frozen EXE.
            with patch.dict(os.environ):
                os.environ.pop('COSPLAY_PACKAGE_ROOT',None)
                with patch.object(module.subprocess,'Popen',side_effect=capture):
                    assert execute('job-direct-exe',character_notes='direct-exe')['status']=='succeeded'
                assert submitted[-1][1]['env']['COSPLAY_PACKAGE_ROOT']==str(root.resolve())

            # A native fast-fail is a child result: no abort, resubmission or publication in parent.
            for code in (-1073740791, 0xc0000409):
                process = type('Exited', (), {'pid': 999999, 'returncode': code,
                    'poll': lambda self: self.returncode, 'communicate': lambda self, **kw: ('', '')})()
                with patch.object(module.subprocess, 'Popen', return_value=process) as spawn:
                    failed = execute('job-native-'+str(abs(code)))
                assert spawn.call_count == 1
                assert failed['status'] == 'failed' and failed['result_summary']['exit_code'] == code
                assert '0xc0000409' in failed['error']['message']
                assert latest.read_bytes() == saved and not backend._job_processes
            with patch.object(module.subprocess, 'Popen', side_effect=OSError('Cannot start runtime')) as spawn:
                failed = execute('job-spawn-error')
            assert spawn.call_count == 1 and failed['status'] == 'failed'
            assert latest.read_bytes() == saved

            # Missing runtime is rejected before options/process execution.
            registry.return_value = {'python': str(root/'missing-python.exe')}
            with patch.object(module.subprocess, 'Popen') as spawn:
                failed = execute('job-missing-runtime')
            assert failed['status'] == 'failed' and spawn.call_count == 0
            assert not (backend.jobs.path('job-missing-runtime')/'workflow-options.json').exists()
            registry.return_value = {'python': str(fake_exe)}
            with patch.object(module.subprocess, 'Popen') as spawn:
                assert execute('job-no-exe-recursion')['status'] == 'failed'
            spawn.assert_not_called()
            registry.return_value = {'python': str(runtime)}
            with patch.object(module, 'code_path', return_value=root/'missing-cli.py'), \
                 patch.object(module.subprocess, 'Popen') as spawn:
                assert execute('job-missing-cli')['status'] == 'failed'
            spawn.assert_not_called()

            # Cancellation stops only the retained child with the durable creation identity.
            backend.jobs.create({'kind': 'one_click', 'source': str(source)}, job_id='job-owned-cancel')
            worker = threading.Thread(target=backend._run_one_click_job, args=('job-owned-cancel',
                {'source': str(source), 'generate': False, 'character_notes': 'wait'}))
            backend._job_workers['job-owned-cancel'] = worker
            with patch.object(module.subprocess, 'Popen', side_effect=capture):
                worker.start()
                deadline = time.monotonic()+8
                while not backend.jobs.get('job-owned-cancel').get('child_process') and time.monotonic()<deadline:
                    time.sleep(.02)
                child = backend.jobs.get('job-owned-cancel').get('child_process') or {}
                assert child.get('stamp')
                backend.jobs.cancel('job-owned-cancel')
                worker.join(10)
            assert not worker.is_alive() and backend.jobs.get('job-owned-cancel')['status'] == 'cancelled'
            assert process_stamp(child['pid']) is None and latest.read_bytes() == saved
            # A foreign or recycled PID is refused, rather than passed to tree cleanup.
            foreign = type('Live', (), {'pid': os.getpid(), 'poll': lambda self: None})()
            backend._job_processes['job-owned-cancel'] = foreign
            with patch('beautygrpo_prepare._stop_process') as stop:
                try: backend._stop_owned_desktop_workflow('job-owned-cancel', foreign)
                except RuntimeError: pass
                else: raise AssertionError('Foreign ownership accepted')
                stop.assert_not_called()
            backend._job_processes.clear()
            with patch.object(module.subprocess, 'Popen', side_effect=capture):
                assert execute('job-parent-still-healthy')['status'] == 'succeeded'
            command.assert_not_called()
        # Source embedded mode keeps its existing in-process test/CLI behavior.
        with patch.object(backend, 'command', return_value={'ok': True, 'exit_code': 0,
                'stdout': '{}', 'stderr': ''}) as command, patch.object(module.subprocess, 'Popen') as spawn:
            assert execute('job-source-compatibility')['status'] == 'succeeded'
            command.assert_called_once(); spawn.assert_not_called()
        assert backend._worker_slots.acquire(blocking=False)
        backend._worker_slots.release()
    print(json.dumps({'ok': True, 'frozen_external_runtime': True, 'native_exit_does_not_abort_parent': True,
                     'owned_tree_cancellation': True, 'latest_preserved': True, 'source_embedded_compatible': True,
                     'direct_exe_child_portable_model_paths': True}))


if __name__ == '__main__': main()
