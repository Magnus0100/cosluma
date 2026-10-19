"""Task-centre inspection cannot restart an in-flight production task."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PIL import Image
import qt_workbench as qt
import single_source_workflow as workflow


class Window(qt.Workbench):
    def refresh(self):
        pass


class Pool:
    def __init__(self):
        self.tasks = []

    def start(self, task):
        self.tasks.append(task)


def main():
    app = qt.QApplication.instance() or qt.QApplication([])
    with tempfile.TemporaryDirectory(prefix='active-task-view-') as directory:
        root = Path(directory)
        source = root/'source.png'
        Image.new('RGB', (48, 32), '#345678').save(source)
        with patch('runtime_preflight.preflight_for_run', return_value={'status': 'ready'}), \
                patch('source_alpha.source_silhouette', side_effect=lambda _root, _source, image, **kw:
                    (Image.new('L', image.size, 255), {'backend': 'CPU fixture'})):
            run = workflow.create_run(workflow.parse_args(['--root', str(root), '--source', str(source),
                '--local-only', '--run-id', 'run-current']))
        run.update(status='candidate_ready', reference={'path': str(source), 'sha256': run['source']['sha256']})
        actions = [{'action_id': name, 'label': name, 'supported': True, 'reason': 'checkpoint fixture'} for name in
            ('retry_local_processing', 'prepare_reconstruction', 'inspect_native_evidence', 'inspect_scene_producer')]
        report = {'whole_work': {'result_state': 'needs_local_recovery', 'result_label': '准备尚未完成',
            'open_findings': [], 'recovery_actions': actions}}
        window = Window(root, 'http://127.0.0.1:1')
        pool = Pool(); window.pool = pool
        window.source_run_id = 'run-current'
        current = {'job_id': 'job-current', 'kind': 'one_click', 'status': 'running', 'run_id': 'run-current'}
        window.job_rows = {'job-current': current}
        window.active_job_id = 'job-current'
        window.render_pending = True
        intent = {'stage': 'production', 'mode': 'existing_reference', 'source': str(source)}
        window.simple_creation = intent
        window.populate_jobs({'jobs': [current]})
        row = window.jobs_tree.topLevelItem(0)
        window.select_job(row)
        requested = pool.tasks[-1]
        assert requested.method == 'GET' and requested.path == '/api/source-runs/run-current'
        assert requested.label == 'job-run-loaded'
        requested.signals.result.emit({'run': run, 'stage_report': report})
        assert window.render_pending and window.simple_creation is intent
        assert window.source_run_id == 'run-current'
        assert window.active_job_id == 'job-current'
        reconstruct = Mock(); window.reconstruct_design = reconstruct

        def choose(action_id):
            panel = window.product_result_panels[0]
            index = next(index for index in range(panel['selector'].count())
                         if (panel['selector'].itemData(index) or {}).get('action_id') == action_id)
            panel['selector'].setCurrentIndex(index)
            window.update_product_action_buttons()
            return panel['button']

        # Buttons and direct action dispatch independently block mutation.
        for status in ('queued', 'running', 'cancelling'):
            window.job_rows['job-current']['status'] = status
            for action in actions[:2]:
                assert not choose(action['action_id']).isEnabled()
                before = len(pool.tasks)
                window.run_product_action(action)
                assert len(pool.tasks) == before
            assert not reconstruct.called
        # Even a stale local busy flag cannot bypass the active-job guard.
        window.render_pending = False
        assert not choose('retry_local_processing').isEnabled()
        before = len(pool.tasks)
        window.task('POST', '/api/source-runs/run-current/retry', {'generate': False}, 'local-retry')
        assert len(pool.tasks) == before
        # Read-only evidence and cancel stay usable while production is active.
        evidence = Mock(); window.open_product_evidence = evidence
        assert choose('inspect_native_evidence').isEnabled()
        window.run_product_action(actions[2]); evidence.assert_called_once_with(native=True)
        window.cancel_selected_job()
        assert pool.tasks[-1].path == '/api/jobs/job-current/cancel'
        assert pool.tasks[-1].method == 'POST'

        window.result('job-status', {'job': {**current, 'status': 'succeeded'}})
        window.result('job-run-loaded', {'run': {**run, 'status': 'needs_visual_review'}, 'stage_report': report})
        assert not window.render_pending and window.simple_creation is None

        # A terminal observation releases the guard. Opening the stopped run
        # permits normal local recovery again without a second running task.
        for status in ('failed', 'blocked', 'cancelled'):
            window.simple_creation = copy.deepcopy(intent)
            window.result('job-status', {'job': {**current, 'status': status}})
            assert window.simple_creation is None
            window.result('job-run-loaded', {'run': {**run, 'status': status}, 'stage_report': report})
            assert not window.render_pending and not window._active_production_job()
            window._session_dirty = False
            assert choose('prepare_reconstruction').isEnabled()
            window.run_product_action(actions[1]); assert reconstruct.called
            reconstruct.reset_mock()
        window.run_product_action(actions[0])
        assert pool.tasks[-1].path == '/api/source-runs/run-current/retry'
        assert pool.tasks[-1].method == 'POST'
        window.render_pending = False; window.simple_creation = None
        window.close(); app.processEvents()
        print(json.dumps({'ok': True, 'task_centre_running_view_preserves_busy': True,
            'queued_running_cancelling_reentry_blocked': True, 'read_only_and_cancel_available': True,
            'stopped_task_local_recovery_available': True, 'network_requests': 0, 'model_inferences': 0}))


if __name__ == '__main__':
    main()
