"""Actual separate-process ownership must survive a second backend's queries."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import time

from backend_server import Backend
from job_store import process_stamp


def main():
    assert process_stamp(os.getpid())
    with tempfile.TemporaryDirectory(prefix='job-owner-') as temporary:
        root = Path(temporary)
        code = '''from pathlib import Path
import sys,time
from job_store import JobStore
s=JobStore(Path(sys.argv[1]))
j=s.create({'kind':'one_click'},job_id='owned-elsewhere')
s.update(j['job_id'],status='running',stage='rendering',progress=.7)
j=s.create({'kind':'one_click'},job_id='cancel-owned-elsewhere')
s.update(j['job_id'],status='running',stage='rendering',progress=.7)
while not (s.root/'exit-owner').exists():time.sleep(.02)
'''
        owner = subprocess.Popen([sys.executable,'-B','-c',code,str(root)])
        try:
            deadline = time.monotonic()+10
            file = root/'outputs/jobs/owned-elsewhere/job.json'
            while not file.is_file() and time.monotonic()<deadline:
                time.sleep(.02)
            viewer = Backend(root)
            while viewer.jobs.get('owned-elsewhere')['status'] != 'running' and time.monotonic()<deadline:
                time.sleep(.02)
            while not (root/'outputs/jobs/cancel-owned-elsewhere/job.json').exists() and time.monotonic()<deadline:
                time.sleep(.02)
            before = file.read_bytes()
            for _ in range(3):
                assert viewer.job('owned-elsewhere')['job']['status'] == 'running'
                assert all(row['status']=='running' for row in viewer.jobs_list()['jobs'])
            assert file.read_bytes()==before, 'A reader changed another live process job'
            assert viewer.cancel_job('cancel-owned-elsewhere')['job']['status']=='cancelling'
            assert viewer.job('cancel-owned-elsewhere')['job']['status']=='cancelling'
            (root/'exit-owner').touch()
            assert owner.wait(10)==0
            recovered = viewer.job('owned-elsewhere')['job']
            assert recovered['status']=='failed' and recovered['stage']=='interrupted'
            assert viewer.job('cancel-owned-elsewhere')['job']['status']=='cancelled'
        finally:
            if owner.poll() is None:
                owner.terminate();owner.wait(10)
    print('actual separate-process job ownership passed')


if __name__=='__main__':
    main()
