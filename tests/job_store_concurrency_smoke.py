"""Project mutation locks serialize independent backends and processes."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from job_store import JobStore


def spawn(code, *args):
    return subprocess.Popen([sys.executable, '-B', '-c', code, *map(str, args)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def finish(process):
    out, error = process.communicate(timeout=10)
    assert process.returncode == 0, error
    return out.strip()


def main():
    with tempfile.TemporaryDirectory(prefix='job-mutations-') as folder:
        root = Path(folder)
        store = JobStore(root)
        code = '''import sys,time
from pathlib import Path
from job_store import JobStore,DuplicateActiveJobError
root=Path(sys.argv[1]); gate=root/'start'
while not gate.exists(): time.sleep(.01)
try:
    job=JobStore(root).create({'execution_binding':'same-binding'},job_id='creator-'+sys.argv[2])
    print('created')
except DuplicateActiveJobError: print('duplicate')
'''
        processes = [spawn(code, root, index) for index in range(6)]
        (root/'start').write_text('start')
        results = [finish(process) for process in processes]
        assert results.count('created') == 1 and results.count('duplicate') == 5, results
        current = store.list()[0]
        store.update(current['job_id'], status='succeeded')

        # Nested mutation calls and separate stores in the same thread share
        # depth accounting; there is no second OS-lock acquisition/deadlock.
        with store.mutation(), store.mutation(), JobStore(root).mutation():
            shared = store.create({}, job_id='shared-updates')
        update_code = '''import sys
from job_store import JobStore
s=JobStore(sys.argv[1])
for n in range(8): s.update('shared-updates', **{'field_'+sys.argv[2]+'_'+str(n):n})
'''
        for process in [spawn(update_code, root, index) for index in range(4)]:
            finish(process)
        final = store.get('shared-updates')
        assert len([key for key in final if key.startswith('field_')]) == 32
        assert len(store.events('shared-updates')) == 33
        store.update('shared-updates', status='succeeded')

        create_after = '''import sys
from pathlib import Path
from job_store import JobStore
root=Path(sys.argv[1]); (root/'attempting').write_text('attempting')
JobStore(root).create({},job_id='after-config')
'''
        with store.configuration_change():
            process = spawn(create_after, root)
            deadline = time.monotonic() + 5
            while not (root/'attempting').exists() and time.monotonic() < deadline:
                time.sleep(.01)
            assert (root/'attempting').exists()
            time.sleep(.08)
            assert process.poll() is None and not store.path('after-config').exists()
        finish(process)
        try:
            with JobStore(root).configuration_change():
                raise AssertionError('Configuration write entered while a job was queued')
        except ValueError:
            pass
        print(json.dumps({'ok': True, 'cross_process_duplicate_prevented': True,
            'no_lost_snapshot_or_events': True, 'configuration_and_creation_serialized': True,
            'nested_stores_reentrant': True}))


if __name__ == '__main__':
    main()
