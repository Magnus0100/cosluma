"""Measured ETA evidence, bounded live polling and source/run binding."""
from __future__ import annotations

import copy
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from job_source_progress import SourceJobProgress
from workflow_timing import measured_progress, progress_text


def main():
    start = datetime.now(timezone.utc).timestamp()
    spans = [dict(id='0', name='production_workflow', parent_id=None, status='running',
                  start_offset_seconds=0., inclusive_seconds=35., metadata={}),
             dict(id='1', name='source_foreground', parent_id='0', status='running',
                  start_offset_seconds=10., inclusive_seconds=25., metadata={'cache_hit': False})]
    live = dict(schema='cosplay_workflow_timing.v1', request_id='live', run_ids=['owned'],
        operation='production_workflow', status='running', created_at=datetime.fromtimestamp(start, timezone.utc).isoformat(),
        workload_key='same-source-task-registry', whole_wall_seconds=35., spans=spans)
    history = []
    for index, duration in enumerate((100., 110., 120.)):
        sample = copy.deepcopy(live)
        sample.update(request_id='success-' + str(index), status='complete', whole_wall_seconds=duration)
        for span in sample['spans']: span['status'] = 'complete'
        history.append(sample)
    estimated = measured_progress(live, history, now=start+35.)
    assert estimated['eta_seconds'] == 75 and estimated['estimate_samples'] == 3
    assert estimated['stage_elapsed_seconds'] == 25.
    assert '3 次同类实测' in progress_text(estimated)
    assert '估算中' in progress_text(measured_progress(live, history[:2], now=start+35.))
    for field, replacement in (('status', 'failed'), ('workload_key', 'different-source')):
        incompatible = [{**sample, field: replacement} for sample in history]
        assert measured_progress(live, incompatible, now=start+35.)['eta_seconds'] is None
    cold = copy.deepcopy(history)
    for sample in cold: sample['spans'][1]['metadata']['cache_hit'] = True
    assert measured_progress(live, cold, now=start+35.)['eta_seconds'] is None
    assert measured_progress(live, history, now=start+200.)['eta_seconds'] is None

    with tempfile.TemporaryDirectory(prefix='cosplay-live-timing-') as directory:
        root = Path(directory)
        source = root/'source.png'; source.write_bytes(b'fixture-source')
        clock = [0.]
        observer = SourceJobProgress(root, 'owned', source, owns_new_run=True, clock=lambda: clock[0])
        timing = root/'outputs/timings/workflow-live.json'; timing.parent.mkdir(parents=True)
        timing.write_text(json.dumps(live), encoding='utf8')
        manifest = root/'outputs/runs/owned/manifest.json'; manifest.parent.mkdir(parents=True)
        run = {'run_id': 'owned', 'source': {'path': str(source), 'sha256': 'a'*64},
            'workflow_timing': {'path': str(timing)}, 'events': [{'status': 'rendering', 'at': 'first'}]}
        manifest.write_text(json.dumps(run), encoding='utf8')
        original = timing.read_bytes()
        with patch('workflow_timing.time.time', return_value=start+35.):
            result = observer.poll()
            assert result[-1]['timing']['module_label'] == '人物前景准备'
            assert result[-1]['progress'] == .7 and '阶段已用 25.0 秒' in result[-1]['message']
            assert observer.poll() == []
        clock[0] = 1.
        with patch('workflow_timing.time.time', return_value=start+36.):
            heartbeat = observer.poll()
        assert heartbeat[-1]['timing']['stage_elapsed_seconds'] == 26.
        assert timing.read_bytes() == original, 'Observation cannot modify measured evidence'
        # A stale workflow milestone cannot move the progress bar backwards.
        run['events'] = [{'status': 'candidate_ready', 'at': 'stale'}]
        manifest.write_text(json.dumps(run), encoding='utf8')
        assert all(row.get('progress', .7) >= .7 for row in observer.poll())
        # A new module is emitted immediately rather than waiting for heartbeat.
        live['spans'].append(dict(id='2', parent_id='1', name='reference_fit', status='running',
            start_offset_seconds=36., inclusive_seconds=0., metadata={}))
        timing.write_text(json.dumps(live), encoding='utf8')
        with patch('workflow_timing.time.time', return_value=start+36.):
            assert observer.poll()[-1]['timing']['module_label'] == '参考外观拟合'
        # Unrelated timing runs cannot impersonate an owned job.
        live['run_ids'] = ['other']
        temporary = timing.with_suffix('.tmp')
        temporary.write_text(json.dumps(live), encoding='utf8'); temporary.replace(timing)
        clock[0] = 2.
        assert observer.poll() == []
    print(json.dumps({'ok': True, 'matching_successful_samples_only': True,
        'no_eta_from_failures_or_cache_mismatch': True, 'one_second_elapsed_updates': True,
        'new_stage_immediate': True, 'monotonic_evidence_progress': True}))


if __name__ == '__main__': main()
