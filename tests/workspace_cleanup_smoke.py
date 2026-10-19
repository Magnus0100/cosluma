"""Reference closure and direct deletion without crossing workspace links."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from unittest.mock import patch

import workspace_cleanup as cleanup_module
from workspace_cleanup import CleanupBusyError, cleanup, cleanup_lock


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf8')
    return path


def run(root, identifier, status='failed', **fields):
    folder = root/'outputs/runs'/identifier
    write(folder/'manifest.json', {'kind': 'single_input_run', 'run_id': identifier, 'status': status, **fields})
    (folder/'pixels.bin').write_bytes(b'irrelevant failed pixels')
    return folder


def job(root, identifier, status='failed', **fields):
    folder = root/'outputs/jobs'/identifier
    write(folder/'job.json', {'job_id': identifier, 'status': status, **fields})
    return folder


def file_snapshot(root):
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob('*')
            if path.is_file() and path.name not in {'.workspace-cleanup.lock', '.job-mutations.lock'}}


def main():
    from qt_workbench import capability_declaration
    capabilities = capability_declaration()
    assert capabilities['project_cleanup_supported'] is True
    assert capabilities['project_cleanup_scope'] == 'unreferenced_stopped_workspace_records'
    with tempfile.TemporaryDirectory(prefix='workspace-cleanup-') as folder:
        base = Path(folder)
        root = base/'workspace'; root.mkdir()
        parent = run(root, 'failed-parent')
        grandparent = run(root, 'cancelled-grandparent', 'cancelled')
        write(parent/'parent.json', {'parent_run_id': grandparent.name})
        live = run(root, 'pending', 'needs_visual_review', parent_run_id=parent.name)
        relative_parent = run(root, 'relative-parent', 'failed')
        write(live/'relative-reference.json', {'asset': '../relative-parent/pixels.bin'})
        run(root, 'approved', 'accepted')
        run(root, 'delivered', 'delivered')
        run(root, 'awaiting', 'awaiting_reference')
        run(root, 'unknown-status', 'future_status')
        active = job(root, 'active', 'running', edited_run={'parent_run_id': 'failed-job-parent'})
        run(root, 'failed-job-parent')
        pending_job = job(root, 'pending-job', 'queued', run_id='failed-active-parent')
        run(root, 'failed-active-parent', 'failed')
        cache_parent = run(root, 'cache-parent')
        write(root/'outputs/cache/saved-cache/receipt.json', {'source': str(cache_parent/'pixels.bin')})
        delivery_parent = run(root, 'delivery-parent')
        write(root/'outputs/final_delivery/package/delivery.json', {'run_id': delivery_parent.name})
        current_parent = run(root, 'project-current-parent')
        write(root/'project.json', {'state': {'last_source_only_run': str(current_parent/'manifest.json')}})
        run(root, 'stopped-approved', 'failed', review={'status': 'accepted'})
        stopped_run = run(root, 'unreferenced-failure')
        stopped_job = job(root, 'unreferenced-job', 'cancelled', run_id=stopped_run.name)
        run(root, 'stopped-cycle-a', 'blocked', parent_run_id='stopped-cycle-b')
        run(root, 'stopped-cycle-b', 'cancelled', parent_run_id='stopped-cycle-a')
        # A recent-menu index is not an ownership root for a failed run.
        write(root/'outputs/settings/recent_projects.json', {'items': [{'run_id': stopped_run.name}]})
        before = file_snapshot(root)
        preview = cleanup(root)
        assert preview['ok'], preview
        assert preview['run_count'] == 3 and preview['job_count'] == 1, preview
        assert preview['removed_files'] == 0 and preview['archive_created'] is False
        assert file_snapshot(root) == before
        assert not any(path.name == 'archive' for path in root.rglob('*'))
        assert cleanup(root, min_age_seconds=3600)['file_count'] == 0
        result = cleanup(root, apply=True)
        assert result['ok'], result
        assert result['removed_runs'] == 3 and result['removed_jobs'] == 1
        assert result['deleted_bytes'] == preview['bytes'] and result['removed_files'] == preview['file_count']
        assert not stopped_run.exists() and not stopped_job.exists()
        for relative, data in before.items():
            if not any(relative.startswith(row['relative'] + '/') for row in preview['candidates']):
                assert (root/relative).read_bytes() == data
        assert cleanup(root)['file_count'] == 0

        # Preview/apply cannot race each other, even from another process.
        with cleanup_lock(root):
            try:
                cleanup(root)
            except CleanupBusyError:
                pass
            else:
                raise AssertionError('Nested cleanup bypassed workspace lock')
            code = ('import sys; from workspace_cleanup import cleanup, CleanupBusyError\n'
                    'try: cleanup(sys.argv[1])\n'
                    'except CleanupBusyError: sys.exit(23)\n'
                    'sys.exit(1)\n')
            held = subprocess.run([sys.executable, '-B', '-c', code, str(root)], capture_output=True, text=True)
            assert held.returncode == 23, held.stderr

        # An apply always rescans: a parent newly referenced after preview is
        # protected. A change during the scan also prevents deletion.
        referenced = run(root, 'new-reference')
        assert cleanup(root)['run_count'] == 1
        write(live/'new.json', {'parent_run_id': referenced.name})
        assert cleanup(root, apply=True)['run_count'] == 0 and referenced.exists()
        changing = run(root, 'changing')
        with patch.object(cleanup_module, '_unchanged', return_value=False):
            changed = cleanup(root, apply=True)
        assert not changed['ok'] and changing.exists()

        # Corrupt ownership evidence fails closed rather than assuming no
        # references. The failed record's data remains intact.
        invalid = root/'outputs/cache/invalid/receipt.json'
        invalid.parent.mkdir(parents=True); invalid.write_text('{', encoding='utf8')
        uncertain = cleanup(root, apply=True)
        assert not uncertain['ok'] and changing.exists()
        invalid.unlink()
        assert cleanup(root, apply=True)['ok'] and not changing.exists()

        outside = base/'outside'; outside.mkdir()
        protected_file = outside/'important.bin'; protected_file.write_bytes(b'outside data')
        linked = root/'outputs/runs/linked'
        if os.name == 'nt':
            subprocess.run(['cmd', '/c', 'mklink', '/J', str(linked), str(outside)],
                           check=True, capture_output=True)
        else:
            linked.symlink_to(outside, target_is_directory=True)
        # A top-level link is never traversed or removed.
        assert cleanup(root, apply=True)['ok']
        assert protected_file.read_bytes() == b'outside data' and linked.exists()
        nested = run(root, 'contains-link')/'linked-directory'
        if os.name == 'nt':
            subprocess.run(['cmd', '/c', 'mklink', '/J', str(nested), str(outside)],
                           check=True, capture_output=True)
        else:
            nested.symlink_to(outside, target_is_directory=True)
        assert not cleanup(root, apply=True)['ok']
        assert nested.exists() and protected_file.read_bytes() == b'outside data'

        # The frozen/backend HTTP endpoint previews and applies the same
        # policy, and refuses both while a project task is active.
        from backend_server import Backend, build_server
        http_root = base/'http-workspace'
        backend = Backend(http_root)
        stale = run(http_root, 'http-failed')
        server = build_server(http_root, port=0)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        def request(body=None):
            data = json.dumps(body).encode('utf8') if body is not None else None
            pending = urllib.request.Request(f'http://127.0.0.1:{server.server_port}/api/cleanup',
                data=data, headers={'Content-Type': 'application/json'})
            try:
                response = urllib.request.urlopen(pending, timeout=5)
            except urllib.error.HTTPError as error:
                assert error.code == 409
                response = error
            with response:
                return json.loads(response.read())
        try:
            preview = request()
            assert preview['ok'] and preview['available'] and preview['run_count'] == 1
            backend.jobs.create({'kind': 'smoke'}, job_id='active-http')
            assert request()['blocked'] and request({'apply': True})['blocked'] and stale.exists()
            backend.jobs.update('active-http', status='cancelled')
            applied = request({'apply': True})
            assert applied['ok'] and applied['removed_runs'] == 1 and applied['removed_jobs'] == 1
            assert not stale.exists() and request()['file_count'] == 0
        finally:
            server.shutdown(); server.server_close(); server_thread.join(5)
        print(json.dumps({'ok': True, 'reference_closure_preserved': True, 'direct_deletion_only': True,
            'preview_apply_locked': True, 'concurrent_changes_fail_closed': True, 'junctions_untouched': True,
            'http_preview_apply_and_active_task_guard': True}))


if __name__ == '__main__':
    main()
