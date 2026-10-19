"""HTTP selection returns before slow native preparation; cancellation is durable.

The gated preparation is synthetic. This tests orchestration, not inference.
"""
import json
from pathlib import Path
import tempfile
import threading
import time
import urllib.request
from unittest.mock import patch

from backend_server import Backend, LocalHTTPServer, make_handler
from backend_batch_export_smoke import saved
import single_source_workflow as workflow


def wait(backend, identifier, predicate):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        record = backend.jobs.get(identifier)
        if predicate(record):
            return record
        time.sleep(.01)
    raise AssertionError(record)


def main():
    with tempfile.TemporaryDirectory(prefix='cosplay-select-job-') as folder:
        root = Path(folder).resolve()
        manifest = saved(root, 'parent')
        parent = json.loads(manifest.read_text('utf8'))
        parent['variants'] = [{'variant_id': 'v1', 'candidate_path': parent['source']['path']}]
        workflow.write_json(manifest, parent)
        before = manifest.read_bytes()
        backend = Backend(root)
        server = LocalHTTPServer(('127.0.0.1', 0), make_handler(backend))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

        def post(body):
            request = urllib.request.Request(f'http://127.0.0.1:{server.server_port}/api/source-runs/parent/select-variant',
                data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
            with opener.open(request, timeout=2) as response:
                return json.load(response)

        entered, release = threading.Event(), threading.Event()

        def slow_prepare(_root, _parent, _variant, *, progress):
            progress('selected-child')
            entered.set()
            assert release.wait(5)
            return json.loads(saved(root, 'selected-child').read_text('utf8'))

        try:
            with patch.object(workflow, 'select_source_variant', side_effect=slow_prepare) as prepare:
                started = post({'variant_id': 'v1'})
                assert started['async'] and entered.wait(2) and not release.is_set()
                record = wait(backend, started['job_id'], lambda row: row['stage'] == 'reference_preparation')
                assert record['status'] == 'running' and record['run_id'] == 'selected-child'
                release.set()
                terminal = wait(backend, started['job_id'], lambda row: row['status'] == 'succeeded')
                assert terminal['run_id'] == 'selected-child' and prepare.call_count == 1
                assert manifest.read_bytes() == before

            def failed_prepare(_root, _parent, _variant, *, progress):
                path = saved(root, 'failed-child')
                child = json.loads(path.read_text('utf8'))
                child.update(parent_run_id='parent', status='rendering')
                workflow.write_json(path, child)
                progress('failed-child')
                raise ValueError('Synthetic native preparation failure')

            with patch.object(workflow, 'select_source_variant', side_effect=failed_prepare):
                failed = post({'variant_id': 'v1'})
                wait(backend, failed['job_id'], lambda row: row['status'] == 'failed')
                _, child = workflow.load_run(root, 'failed-child')
                assert child['status'] == 'blocked' and child['error']['candidate_retained']
                assert 'Synthetic native preparation failure' in child['error']['message']
                assert manifest.read_bytes() == before

            # A queued cancellation cannot create a child or render anything.
            backend._worker_slots.acquire()
            try:
                with patch.object(workflow, 'select_source_variant', side_effect=AssertionError('Cancelled selection rendered')):
                    cancelled = post({'variant_id': 'v1'})
                    backend.cancel_job(cancelled['job_id'])
                    ended = wait(backend, cancelled['job_id'], lambda row: row['status'] == 'cancelled')
                    assert ended['run_id'] is None
            finally:
                backend._worker_slots.release()
            stale = backend.jobs.create({'kind': 'select_reference_variant', 'run_id': None, 'parent_run_id': 'parent'})
            interrupted = Backend(root).job(stale['job_id'])['job']
            assert interrupted['status'] == 'failed' and interrupted['stage'] == 'interrupted'
        finally:
            release.set()
            server.shutdown(); server.server_close(); thread.join(5)
    print('OK: immediate HTTP selection job, observable slow preparation, preserved parent, queued cancellation and restart interruption; synthetic preparation')


if __name__ == '__main__':
    main()
