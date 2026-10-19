"""Saved product approval, validated recovery and durable job snapshots."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

from PIL import Image

from backend_server import Backend
from job_store import JobStore
import job_store
import single_source_workflow as workflow
from task_model_preflight_smoke import build_registry_fixture, mocked_task_capability


def snapshot_contract(root):
    store = JobStore(root)
    store.create({'kind': 'one_click'}, job_id='durable')
    target = store.path('durable') / 'job.json'
    original_replace = os.replace
    seen = []

    def transient(source, destination):
        seen.append(Path(source))
        assert json.loads(Path(source).read_text('utf8'))['execution_status'] == 'running'
        if len(seen) < 3:
            raise PermissionError('transient Windows reader lock')
        original_replace(source, destination)

    with patch.object(job_store.os, 'replace', side_effect=transient), \
            patch.object(job_store.os, 'fsync', wraps=os.fsync) as flushed, \
            patch.object(job_store.time, 'sleep') as slept:
        current = store.update('durable', status='running')
    assert current['execution_status'] == 'running' and current['product_state'] == 'processing'
    assert len(seen) == 3 and len(set(seen)) == 1 and flushed.call_count == 1
    assert slept.call_count == 2
    stable = target.read_bytes()
    with patch.object(job_store.os, 'replace', side_effect=PermissionError('persistent lock')), \
            patch.object(job_store.time, 'sleep'):
        try:
            store.update('durable', status='succeeded')
        except PermissionError:
            pass
        else:
            raise AssertionError('Persistent snapshot failure was hidden')
    assert target.read_bytes() == stable
    assert not list(target.parent.glob('*.tmp'))
    next_paths = []

    def final_replace(source, destination):
        next_paths.append(Path(source))
        original_replace(source, destination)

    with patch.object(job_store.os, 'replace', side_effect=final_replace):
        store.update('durable', status='succeeded')
    assert next_paths[0] != seen[0]


def main():
    with tempfile.TemporaryDirectory(prefix='product-state-') as folder:
        root = Path(folder)
        snapshot_contract(root/'snapshots')
        backend = Backend(root/'project')
        source = root/'source.png'
        Image.new('RGB', (48, 32), '#284568').save(source)
        registry = build_registry_fixture(root/'registry')
        with patch('task_model_preflight.inspect_task_models', side_effect=mocked_task_capability):
            checked = backend.task_plan({'source': str(source), 'model_registry': str(registry),
                'creative_task': {'schema_version': 'cosplay_task_intent.v1', 'mode': 'local_portrait'}})
        with patch('runtime_preflight.preflight_for_run', return_value={'status': 'ready'}), \
                patch('source_alpha.source_silhouette', side_effect=lambda _root, _source, image, **kw:
                      (Image.new('L', image.size, 255), {'backend': 'offline CPU fixture'})):
            run = workflow.create_run(workflow.parse_args(['--root', str(backend.root),
                '--source', str(source), '--run-id', 'current', '--local-only']))
        run['task_intent'] = copy.deepcopy(checked['plan']['intent'])
        run['task_plan'] = copy.deepcopy(checked['plan'])
        path = backend.root/'outputs/runs/current/manifest.json'
        workflow.write_json(path, run)
        recovery = backend.recovery()
        assert recovery['recoverable'] and recovery['source_current']['run_id'] == 'current'
        current_bytes = path.read_bytes()

        # Invalid/stopped history remains readable, but never becomes recovery.
        for identifier, changes in (
                ('failed', {'status': 'failed'}), ('cancelled', {'status': 'cancelled'}),
                ('blocked', {'status': 'blocked'}), ('accepted-history', {'status': 'accepted'}),
                ('bad-revision', {'revision': 0}), ('legacy-plan', {'task_plan': None}),
                ('pending-save', {'continuation_edit': {'render_state': 'pending'}}),
                ('missing-master', {'artifacts': {**run['artifacts'], 'render_master': 'missing.tif'}})):
            row = copy.deepcopy(run)
            row.update(run_id=identifier, **changes)
            row['render']['run_id'] = identifier
            workflow.write_json(backend.root/f'outputs/runs/{identifier}/manifest.json', row)
        # A live job cannot make a stopped run resumable. An orphaned worker
        # snapshot is reconciled before deciding whether it owns any run.
        backend.jobs.create({'kind': 'one_click', 'run_id': 'failed'}, job_id='active-stopped')
        backend.jobs.update('active-stopped', status='running')
        backend._job_workers['active-stopped'] = object()
        orphan = copy.deepcopy(run)
        orphan.update(run_id='orphan', status='imported')
        workflow.write_json(backend.root/'outputs/runs/orphan/manifest.json', orphan)
        backend.jobs.create({'kind': 'one_click', 'run_id': 'orphan'}, job_id='orphan-worker')
        backend.jobs.update('orphan-worker', status='running')
        recovery = backend.recovery()
        assert recovery['source_current']['run_id'] == 'current'
        assert [row['run_id'] for row in recovery['source_runs'] if row['recoverable']] == ['current']
        assert backend.jobs.get('orphan-worker')['execution_status'] == 'failed'
        assert path.read_bytes() == current_bytes

        # Candidate selection can recover a current create-reference task,
        # but a status string alone cannot invent an available candidate.
        with patch('task_model_preflight.inspect_task_models', side_effect=mocked_task_capability), \
                patch('reference_provider.provider_health', return_value={'ready': True}):
            candidates_checked = backend.task_plan({'source': str(source), 'model_registry': str(registry),
                'creative_task': {'schema_version': 'cosplay_task_intent.v1', 'mode': 'create_reference'}})
        candidate = copy.deepcopy(run)
        candidate.update(run_id='candidate-selection', status='awaiting_reference',
            task_intent=candidates_checked['plan']['intent'], task_plan=candidates_checked['plan'],
            variants=[{'candidate_path': str(source)}])
        candidate_path = backend.root/'outputs/runs/candidate-selection/manifest.json'
        workflow.write_json(candidate_path, candidate)
        assert any(row['run_id'] == 'candidate-selection' and row['recoverable'] for row in backend.recovery()['source_runs'])
        candidate['variants'][0]['candidate_path'] = str(root/'missing-candidate.png')
        workflow.write_json(candidate_path, candidate)
        assert not any(row['run_id'] == 'candidate-selection' and row['recoverable'] for row in backend.recovery()['source_runs'])
        source_bytes = source.read_bytes()
        source.write_bytes(b'changed source')
        assert not backend.recovery()['recoverable']
        source.write_bytes(source_bytes)

        # Workflow execution success does not grant visual/product acceptance.
        with patch.object(backend, 'command', return_value={'ok': True, 'exit_code': 0,
                'stdout': json.dumps(run), 'stderr': ''}):
            result = backend._one_click_sync({'source': str(source), 'generate': False})
        job_id = result['job_id']
        saved = backend.jobs.get(job_id)
        assert saved['status'] == saved['execution_status'] == 'succeeded'
        assert saved['product_state'] == 'needs_visual_review' and '尚未交付' in saved['message']
        assert not (backend.root/'outputs/final_delivery').exists()
        for stopped in ('failed', 'cancelled'):
            backend.jobs.create({'kind': 'one_click', 'run_id': 'current'}, job_id=stopped)
            backend.jobs.update(stopped, status=stopped)
        stopped_bytes = {key: (backend.jobs.path(key)/'job.json').read_bytes() for key in ('failed', 'cancelled')}
        checks = {key: 'pass' for key in workflow.VISUAL_CHECK_KEYS}
        review_path = path.parent/'visual_review.json'
        human_review_before = review_path.read_bytes()
        with patch.object(workflow, 'accept_run', side_effect=AssertionError('Agent entered human approval gate')):
            agent = backend.accept_source_run('current', {'review_kind': 'agent', 'reviewer': 'assistant',
                'checks': checks, 'observations': ['Thumbnail comparison only; native details remain unexamined.']})
        assert agent['product_state'] == 'pending_review' and agent['human_approval'] is False
        assert agent['run']['agent_review']['status'] == 'observations_only'
        assert 'checks' not in agent['run']['agent_review']
        assert review_path.read_bytes() == human_review_before
        assert not (backend.root/'outputs/final_delivery').exists()
        assert backend.jobs.get(job_id) == saved
        try:
            backend.accept_source_run('current', {'reviewer': 'fixture', 'checks': {**checks, 'identity': 'unknown'}})
        except RuntimeError:
            pass
        else:
            raise AssertionError('Unknown visual check bypassed approval')
        assert backend.jobs.get(job_id) == saved
        accepted = backend.accept_source_run('current', {'reviewer': 'fixture', 'checks': checks})
        assert accepted['product_state'] == 'delivered'
        final = backend.jobs.get(job_id)
        assert final['status'] == final['execution_status'] == 'succeeded'
        assert final['finished_at'] == saved['finished_at']
        assert final['product_state'] == 'delivered' and final['result_summary']['current_revision_visual_approved']
        assert backend.jobs.events(job_id)[-1]['type'] == 'product_approved'
        assert all((backend.jobs.path(key)/'job.json').read_bytes() == value for key, value in stopped_bytes.items())
        assert not backend.recovery()['recoverable']
        print(json.dumps({'ok': True, 'execution_and_product_states_separate': True,
            'approval_gate_preserved': True, 'recovery_validated': True, 'atomic_fsynced_snapshots': True}))


if __name__ == '__main__':
    main()
