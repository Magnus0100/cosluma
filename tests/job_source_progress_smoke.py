"""Live tiny-image float rendering with a CPU matte plus external job regressions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time
from unittest.mock import patch

from PIL import Image

from backend_server import Backend
from job_source_progress import SourceJobProgress
from job_store import TERMINAL
import single_source_workflow as workflow
from task_model_preflight_smoke import build_registry_fixture, mocked_task_capability


# A real subprocess with ordinary persisted run events, not a model/producer.
# Gates prove that the backend receives stages before stdout/CLI completion.
CLI_FIXTURE = r'''
import argparse, hashlib, json, sys, time
from pathlib import Path
p=argparse.ArgumentParser()
for key in ('root','source','run-id','options-file'): p.add_argument('--'+key)
a,_=p.parse_known_args()
root=Path(a.root); directory=root/'outputs/runs'/a.run_id
directory.mkdir(parents=True,exist_ok=True)
control=root/'control'/a.run_id; control.mkdir(parents=True,exist_ok=True)
log=directory/'logs/run.log.jsonl';log.parent.mkdir()
source=Path(a.source)
run={'kind':'single_input_run','run_id':a.run_id,'revision':1,'sample':'fixture',
     'source':{'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()},
     'status':'imported','events':[],'options':{'local_only':True}}
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf8')
def event(status):
    row={'status':status,'at':str(time.time_ns())};run['events'].append(row);run['status']=status
    with log.open('a',encoding='utf8') as stream:stream.write(json.dumps(row)+'\n')
    write(directory/'manifest.json',run)
def gate(name):
    while not (control/name).exists():time.sleep(.02)
event('imported');event('candidate_ready')
write(directory/'automatic-native-retouch/input-run.json',run)
attempt={'owner_run_id':a.run_id,'source_sha256':run['source']['sha256'],'status':'reserved'}
write(directory/'automatic-native-retouch/attempt.json',attempt)
gate('allow-retouch')
attempt['status']='frozen_source_stage_ready';write(directory/'automatic-native-retouch/attempt.json',attempt)
event('source_retouch_prepared');event('rendering');gate('allow-render')
event('needs_visual_review');gate('allow-review')
if (control/'fail').exists():event('blocked')
print(json.dumps(run),flush=True)
sys.exit(2 if run['status']=='blocked' else 0)
'''


def wait_for(backend, job_id, predicate, *, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = backend.jobs.get(job_id)
        if predicate(job):
            return job
        if job['status'] in TERMINAL:
            raise AssertionError('Unexpected terminal state: ' + repr(job))
        time.sleep(.02)
    raise AssertionError('Timed out: ' + repr(backend.jobs.get(job_id)))


def terminal(backend, job_id):
    return wait_for(backend, job_id, lambda job: job['status'] in TERMINAL)


def release(root, run_id, name):
    path = root / 'control' / run_id / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('test gate released\n', encoding='utf8')


def checked_start(backend, body):
    """Enter through the current checked task boundary with CPU metadata only."""
    registry = build_registry_fixture(backend.root/'CPU-model-fixture')
    request = {key:value for key,value in body.items() if key not in {'local_only','generate'}}
    request.update(model_registry=str(registry), creative_task={
        'schema_version':'cosplay_task_intent.v1','mode':'local_portrait',
        'services':{'text_vision':False,'visual_review':False}})
    with patch('task_model_preflight.inspect_task_models', side_effect=mocked_task_capability):
        checked = backend.task_plan(request)
        return backend.start_task({**checked['request'], 'plan_id':checked['plan']['plan_id']})


def external_live(root, source, *, run_id=None, cancel=False, fail=False):
    backend = Backend(root)
    cli = root / 'fixture_cli.py'
    cli.write_text(CLI_FIXTURE, encoding='utf8')
    backend.cli = cli
    started = checked_start(backend, {'source': str(source), 'local_only': True, 'generate': False,
        **({'run_id': run_id} if run_id else {})})
    job_id = started['job_id']
    running = wait_for(backend, job_id, lambda job: job['stage'] == 'source-retouch')
    assigned = running['run_id']
    assert assigned == run_id if run_id else assigned == 'run-' + job_id.removeprefix('job-')
    assert running['status'] == 'running' and running['progress'] == 0.50
    if cancel:
        backend.cancel_job(job_id)
        ended = terminal(backend, job_id)
        assert ended['status'] == 'cancelled' and ended['run_id'] == assigned
        return backend, job_id, assigned
    release(root, assigned, 'allow-retouch')
    rendering = wait_for(backend, job_id, lambda job: job['stage'] == 'rendering')
    assert rendering['status'] == 'running' and rendering['progress'] == 0.70
    release(root, assigned, 'allow-render')
    reviewed = wait_for(backend, job_id, lambda job: job['stage'] == 'review')
    assert reviewed['status'] == 'running' and reviewed['progress'] == 0.85
    if fail:
        release(root, assigned, 'fail')
    release(root, assigned, 'allow-review')
    ended = terminal(backend, job_id)
    assert ended['status'] == ('blocked' if fail else 'succeeded')
    assert ended['result_summary']['quality_inferred_from_worker_success'] is False
    events = backend.jobs.events(job_id)
    for stage in ('source-retouch', 'rendering', 'review'):
        assert any(event['stage'] == stage and event['status'] == 'running' for event in events)
    assert all(event['progress'] is None or 0 <= event['progress'] <= 1 for event in events if event['type'] != 'completed')
    identities = [json.dumps(event.get('data', {}).get('workflow_event'), sort_keys=True)
        for event in events if event.get('data', {}).get('workflow_event')]
    assert len(identities) == len(set(identities)), 'Manifest/log milestones duplicated'
    return backend, job_id, assigned


def late_events_preserved(backend, job_id, run_id, source):
    before = backend.jobs.get(job_id)
    journal_before = backend.jobs.events(job_id)
    observer = SourceJobProgress(backend.root, run_id, source)
    path = backend.root / 'outputs/runs' / run_id / 'manifest.json'
    run = json.loads(path.read_text())
    run['events'] = [{'status': 'rendering', 'at': 'late-after-terminal'}]
    path.write_text(json.dumps(run), encoding='utf8')
    backend._observe_source_job(job_id, observer)
    backend._update_live_one_click_job(job_id, status='succeeded', stage='complete', progress=1,
        message='late success callback')
    assert backend.jobs.get(job_id) == before
    assert backend.jobs.events(job_id) == journal_before


def main():
    with tempfile.TemporaryDirectory(prefix='job-source-progress-') as folder:
        root = Path(folder)
        source = root / 'source.png'
        Image.new('RGB', (64, 48), '#395568').save(source)
        source_bytes = source.read_bytes()
        # Actual production local image rendering, held before CLI completion.
        embedded = Backend(root / 'actual-image', embedded=True)
        project_identity = (embedded.root / 'project.json').read_bytes()
        rendered, resume = threading.Event(), threading.Event()
        actual_render = workflow.render_local_source
        def gated_render(project, run):
            result = actual_render(project, run)
            rendered.set()
            assert resume.wait(10), 'Actual render gate was not released'
            return result
        with patch.object(workflow, 'render_local_source', side_effect=gated_render), \
                patch('runtime_preflight.preflight_for_run', return_value={'status':'ready','CPU_fixture':True}), \
                patch('source_alpha.source_silhouette', side_effect=lambda _root,_source,image,**kw:
                    (Image.new('L',image.size,255),{'backend':'CPU_contract_matte_not_model_inference'})):
            started = checked_start(embedded, {'source': str(source), 'local_only': True, 'generate': False,
                'preserve_project_selection': True})
            job_id = started['job_id']
            try:
                assert rendered.wait(15), 'Actual small-image rendering did not execute'
                live = wait_for(embedded, job_id, lambda job: job['stage'] == 'rendering')
                assert live['run_id'] and live['status'] == 'running' and live['progress'] == 0.70
            finally:
                resume.set()
            finished = terminal(embedded, job_id)
        assert finished['status'] == 'succeeded'
        manifest = json.loads((embedded.root / 'outputs/runs' / finished['run_id'] / 'manifest.json').read_text())
        artifact = Path(manifest['artifacts']['auto_draft'])
        artifact = artifact if artifact.is_absolute() else embedded.root / artifact
        with Image.open(artifact) as image:
            assert image.size == (64, 48)
        assert (embedded.root / 'project.json').read_bytes() == project_identity
        success, success_id, assigned = external_live(root / 'external', source)
        late_events_preserved(success, success_id, assigned, source)
        explicit, explicit_id, assigned = external_live(root / 'explicit', source, run_id='user-chosen-run')
        assert explicit.jobs.get(explicit_id)['initial_run_id'] == 'user-chosen-run'
        cancelled, cancelled_id, assigned = external_live(root / 'cancelled', source, cancel=True)
        late_events_preserved(cancelled, cancelled_id, assigned, source)
        failed, failed_id, assigned = external_live(root / 'failed', source, fail=True)
        late_events_preserved(failed, failed_id, assigned, source)
        # Cancellation while waiting for a slot allocates no run or options.
        queued = Backend(root / 'queued', embedded=True)
        queued._worker_slots.acquire()
        try:
            requested = checked_start(queued, {'source': str(source), 'local_only': True, 'generate': False,
                'run_id': 'requested-but-never-started'})
            queued_id = requested['job_id']
            assert queued.jobs.get(queued_id)['run_id'] is None
            queued.cancel_job(queued_id)
            ended = terminal(queued, queued_id)
            assert ended['status'] == 'cancelled' and ended['run_id'] is None
            assert not (queued.root / 'outputs/runs').exists()
            assert not (queued.jobs.path(queued_id) / 'workflow-options.json').exists()
        finally:
            queued._worker_slots.release()
        # Unrelated and partial records cannot impersonate this source/run.
        bound_root = root / 'binding'
        observer = SourceJobProgress(bound_root, 'owned', source,
            source_sha256=hashlib.sha256(source_bytes).hexdigest())
        manifest_path = bound_root / 'outputs/runs/owned/manifest.json'
        manifest_path.parent.mkdir(parents=True)
        other = {'run_id': 'other', 'source': {'path': str(source), 'sha256': 'f'*64},
            'events': [{'status': 'rendering', 'at': 'unrelated'}]}
        manifest_path.write_text(json.dumps(other), encoding='utf8')
        assert observer.poll() == []
        other.update(run_id='owned'); other['source']['sha256'] = hashlib.sha256(source_bytes).hexdigest()
        manifest_path.write_text('{', encoding='utf8')
        assert observer.poll() == []
        other['events'] = []
        manifest_path.write_text(json.dumps(other), encoding='utf8')
        log = manifest_path.parent / 'logs/run.log.jsonl'; log.parent.mkdir()
        log.write_bytes(b'{"status":"rendering","at":"partial"}')
        assert observer.poll() == []
        with log.open('ab') as stream:
            stream.write(b'\n')
        assert [row['stage'] for row in observer.poll()] == ['rendering']
        assert observer.poll() == []
        # Existing explicit IDs cannot replay pre-job files on a failed open.
        stale = SourceJobProgress(bound_root, 'owned', source)
        assert stale.poll() == []
        # Candidate progress comes from actual persisted submission outcomes,
        # never from elapsed time or arbitrary log detail strings.
        candidates_root = root / 'candidate-progress'
        candidate_observer = SourceJobProgress(candidates_root, 'candidate-run', source, owns_new_run=True)
        candidate_manifest = candidates_root / 'outputs/runs/candidate-run/manifest.json'
        candidate_manifest.parent.mkdir(parents=True)
        candidate_run = {'run_id': 'candidate-run', 'source': {'path': str(source),
            'sha256': hashlib.sha256(source_bytes).hexdigest()}, 'status': 'generating',
            'generation_budget': {'requested_variants': 4, 'submissions': [{'status': 'succeeded'},
                {'status': 'submitted_outcome_unknown'}]}, 'events': []}
        candidate_manifest.write_text(json.dumps(candidate_run), 'utf8')
        observed = candidate_observer.poll()
        assert len(observed) == 1 and observed[0]['candidate_counts'] == {'completed': 1, 'submitted': 2, 'requested': 4}
        assert observed[0]['progress'] == .2875 and '1/4' in observed[0]['message']
        assert candidate_observer.poll() == []
        # Completion markers provide honest later stages without following or
        # reading unrelated child jobs. Observation never writes these files.
        completion = manifest_path.parent / 'system-post-completion/attempt.json'
        completion.parent.mkdir()
        record = {'initial_run_id': 'owned', 'result_run_id': 'owned-child',
            'source_sha256': hashlib.sha256(source_bytes).hexdigest(), 'stages': []}
        for status, expected in (('completing_materials', 'material_finish'),
                ('reviewing_final_revision', 'review'), ('repairing_final_locally', 'final_local_repair')):
            record['status'] = status
            completion.write_text(json.dumps(record), encoding='utf8')
            before = completion.read_bytes()
            assert [row['stage'] for row in observer.poll()] == [expected]
            assert completion.read_bytes() == before
        # No observer update can replace an already acknowledged cancellation.
        cancelling_backend = Backend(bound_root)
        job = cancelling_backend.jobs.create({'kind': 'one_click', 'run_id': 'owned'})
        cancelling_backend.jobs.update(job['job_id'], status='running', stage='rendering')
        cancelling_backend.jobs.cancel(job['job_id'])
        before = cancelling_backend.jobs.get(job['job_id'])
        journal_before = cancelling_backend.jobs.events(job['job_id'])
        cancelling_backend._update_live_one_click_job(job['job_id'], stage='review', message='late progress')
        assert cancelling_backend.jobs.get(job['job_id']) == before
        assert cancelling_backend.jobs.events(job['job_id']) == journal_before
        assert source.read_bytes() == source_bytes
        recovered = Backend(success.root)
        assert recovered.jobs.get(success_id)['run_id'] == success.jobs.get(success_id)['run_id']
    print(json.dumps({'ok': True, 'actual_tiny_image_render_observed_live': True,
        'external_stages_before_stdout_completion': True, 'short_owned_run_id': True,
        'explicit_run_id_preserved': True, 'queued_cancel_run_id_null': True,
        'active_cancel_preserved': True, 'late_terminal_events_ignored': True,
        'partial_unrelated_records_rejected': True, 'stage_milestone_percentages': True,
        'model_inferences': 0, 'network_requests': 0}))


if __name__ == '__main__':
    main()
