"""Exercise Qt poll scheduling across slow replies, SSE fallback and job switches."""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import qt_workbench as qt
from design_reconstruction_ui_smoke import Pool, Window


def pump(app, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.005)


def main():
    app = qt.QApplication.instance() or qt.QApplication([])
    with tempfile.TemporaryDirectory(prefix='cosplay-poll-') as temporary:
        window = Window(Path(temporary), 'http://127.0.0.1:1')
        pool = Pool(); window.pool = pool
        window.observe_job('slow-job')
        for _ in range(50):
            window.poll_active_job()
            window.job_stream_error('test disconnected stream')
            window._schedule_job_poll(750)
        pump(app, .18)
        assert len([t for t in pool.tasks if getattr(t, 'label', '') == 'job-status']) == 1
        assert len([t for t in pool.tasks if getattr(t, 'label', '') == 'job-events']) == 1

        status = next(t for t in pool.tasks if getattr(t, 'label', '') == 'job-status')
        status.signals.result.emit({'job': {'job_id': 'slow-job', 'status': 'running', 'progress': .2}})
        status.signals.finished.emit()
        assert window.job_poll_timer.isActive()
        for _ in range(50): window._schedule_job_poll(100)
        pump(app, .18)
        statuses = [t for t in pool.tasks if getattr(t, 'label', '') == 'job-status']
        assert len(statuses) == 2, 'duplicate scheduling must create one subsequent request'
        statuses[-1].signals.error.emit('test slow network')
        statuses[-1].signals.finished.emit()
        assert window.job_poll_timer.isActive(), 'ordinary status failures must recover too'

        old_events = next(t for t in pool.tasks if getattr(t, 'label', '') == 'job-events')
        window.observe_job('new-job')
        assert any(getattr(t, 'path', '') == '/api/jobs/new-job' for t in pool.tasks)
        old_events.signals.finished.emit()
        # A response from the old observation must not terminate the new job.
        statuses[-1].signals.result.emit({'job': {'job_id': 'slow-job', 'status': 'failed'}})
        assert window.job_poll_terminal_observation != window.job_observation
        new_status = next(t for t in reversed(pool.tasks) if getattr(t, 'path', '') == '/api/jobs/new-job')
        new_status.signals.result.emit({'job': {'job_id': 'new-job', 'status': 'succeeded'}})
        new_status.signals.finished.emit()
        assert not window.job_poll_timer.isActive()
        before = len(pool.tasks)
        window.job_stream_finished(); window.job_stream_error('late close')
        window.poll_active_job(); pump(app, .18)
        assert len(pool.tasks) == before, 'terminal jobs must not restart polling'
        window.close(); app.processEvents()
    print({'ok': True, 'one_poll_chain': True, 'slow_reply_deduplicated': True,
           'error_recovery': True, 'stale_observation_ignored': True})


if __name__ == '__main__':
    main()
