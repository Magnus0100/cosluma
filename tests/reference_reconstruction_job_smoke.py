#!/usr/bin/env python3
"""Exercise local reference preparation jobs without models or generation.

The pipeline is mocked intentionally: this verifies lifecycle and UI contracts,
not colour quality, real model execution, or a cross-photo production workflow.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
import types
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from backend_server import Backend
from session_draft import digest


TERMINAL = {'succeeded', 'failed', 'blocked', 'cancelled'}


def wait_job(backend, job_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = backend.job(job_id)['job']
        if job['status'] in TERMINAL:
            with backend._job_lock:
                worker = backend._job_workers.get(job_id)
            if worker:
                worker.join(timeout=2)
            return job
        time.sleep(.01)
    raise AssertionError('Local reference preparation did not finish')


def ui_contract(root, plan):
    import qt_workbench as qt
    assert qt.QT_AVAILABLE, 'Qt runtime is required for the UI contract smoke'
    app = qt.QApplication.instance() or qt.QApplication([])

    class Window(qt.Workbench):
        def refresh(self):
            pass

    window = Window(root, 'http://127.0.0.1:1')
    # Asset validity belongs to the real pipeline/consumer smokes. Here only
    # isolate the view/serialization contract from the mock plan's fake assets.
    window._paired_subject_state = lambda *args: {'reconstruction_required': False}
    window.load_design_reconstruction(plan)
    assert window.design_reconstruct_button.text() == '准备仿色融合 / 重试'
    assert '不新增生图' in window.design_reconstruct_button.toolTip()
    assert '0 为当前源人物底色' in window.design_layer_help.text()
    assert '不撤销人工笔刷' in window.design_layer_help.text()
    assert '不独立修改' in window.design_layer_help.text()
    assert window.design_layer_strength.minimum() == 0
    assert window.design_layer_strength.maximum() == 1
    before = copy.deepcopy(plan)
    window.design_layer_strength.setValue(.4)
    window.source_light_timer.stop()
    assert window.design_reconstruction_plan['layers'][0]['strength'] == .4
    assert plan == before, 'UI must not mutate the pipeline response'
    payload = window._source_edit_payload()
    assert payload['appearance_revision']['design_reconstruction_plan']['layers'][0]['strength'] == .4
    assert not any(key in payload for key in ('generate', 'provider', 'prompt'))
    window.close()
    app.processEvents()


def main():
    with tempfile.TemporaryDirectory(prefix='cosplay-reference-job-') as temporary:
        root = Path(temporary)
        parent_id = 'fixture-parent'
        directory = root / 'outputs' / 'runs' / parent_id
        directory.mkdir(parents=True)
        source = root / 'source.png'
        reference = root / 'reference.png'
        source.write_bytes(b'source fixture')
        reference.write_bytes(b'reference fixture')
        run = {'kind': 'single_input_run', 'run_id': parent_id, 'sample': 'fixture',
            'revision': 40, 'status': 'needs_visual_review',
            'source': {'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()},
            'reference': {'path': str(reference), 'sha256': hashlib.sha256(reference.read_bytes()).hexdigest()},
            'options': {'render_mode': 'native_source_design'}}
        manifest = directory / 'manifest.json'
        manifest.write_text(json.dumps(run), encoding='utf-8')
        pointer = root / 'project.json'
        pointer.write_text(json.dumps({'state': {'last_source_run': parent_id}}), encoding='utf-8')
        retained = (manifest.read_bytes(), pointer.read_bytes())
        edits = {'appearance_revision': {'params': {'global': {'exposure': .1}}},
            'manual_layers': {'local_colour_brush': {'enabled': True,
                'strokes': [{'id': str(index), 'points': [[.2, .3]]} for index in range(4)]}}}
        plan = {'schema_version': 'design_reconstruction_plan.v1', 'enabled': True,
            'layers': [{'id': 'reference-appearance', 'kind': 'reference_appearance_stack',
                'label': '整体仿色融合', 'enabled': True, 'strength': 1., 'assets': {}}]}
        module = types.ModuleType('reference_pipeline')
        backend = Backend(root, embedded=True)
        calls = []

        def no_generation(*args, **kwargs):
            raise AssertionError('Reference preparation must never invoke generation or the CLI')

        backend.command = no_generation
        backend.one_click = no_generation

        def assert_retained(job_id):
            job = backend.jobs.get(job_id)
            assert (manifest.read_bytes(), pointer.read_bytes()) == retained
            assert job['input_sha256'] == digest(edits)
            assert job['edited_run']['options']['manual_layers'] == edits['manual_layers']
            assert len(job['edited_run']['options']['manual_layers']['local_colour_brush']['strokes']) == 4
            assert sorted(path.name for path in (root / 'outputs' / 'runs').iterdir()) == [parent_id]
            assert job['project_pointer_updated'] is False and job['generates_reference'] is False

        def successful(project, edited, *, rebuild, cancel, progress):
            calls.append(copy.deepcopy(edited))
            assert project == root and rebuild is False and cancel() is False
            progress('transfer', '正在准备原图与参考图仿色', .35)
            progress({'stage': 'composition', 'message': '正在准备融合资产', 'progress': .85})
            return {'ok': True, 'design_reconstruction_plan': copy.deepcopy(plan), 'cache_hit': True}

        with patch.dict(sys.modules, {'reference_pipeline': module}):
            module.prepare = successful
            started = backend.reconstruct_source_design(parent_id, edits)
            success_id = started['job_id']
            success = wait_job(backend, success_id)
            assert success['status'] == 'succeeded', success
            assert success['result_summary']['cache_hit'] is True
            assert success['result_summary']['visual_acceptance'] is False
            receipt = backend.job(success_id)['reconstruction_result']
            assert receipt['design_reconstruction_plan'] == plan
            assert receipt['input_sha256'] == digest(edits)
            assert {event['stage'] for event in backend.job_events(success_id)['events']} >= {'transfer', 'composition', 'prepared'}
            assert_retained(success_id)
            # Completed receipts remain available after backend restart.
            assert Backend(root).job(success_id)['reconstruction_result'] == receipt
            stale_snapshot = {**success, 'status': 'running', 'stage': 'transfer'}
            assert backend._reconcile_reconstruction_job(stale_snapshot)['stage'] == 'prepared'
            target = backend.jobs.path(success_id) / 'prepared-plan.json'
            original = target.read_bytes()
            target.write_bytes(original + b' ')
            try:
                backend.job(success_id)
                raise AssertionError('Changed preparation receipt should be rejected')
            except ValueError:
                pass
            target.write_bytes(original)

            def failed(*args, **kwargs):
                kwargs['progress']('transfer', '仿色准备失败', .2)
                raise RuntimeError('fixture fit failure')

            module.prepare = failed
            failure_id = backend.reconstruct_source_design(parent_id, edits)['job_id']
            failure = wait_job(backend, failure_id)
            assert failure['status'] == 'failed' and failure['stage'] == 'transfer', failure
            assert 'fixture fit failure' in failure['error']
            assert_retained(failure_id)

            module.prepare = lambda *args, **kwargs: {'ok': False, 'error': 'fixture incomplete preparation'}
            incomplete_id = backend.reconstruct_source_design(parent_id, edits)['job_id']
            assert wait_job(backend, incomplete_id)['status'] == 'failed'
            assert_retained(incomplete_id)

            entered = threading.Event()

            def cancellable(*args, **kwargs):
                kwargs['progress']('native_fit', '正在计算原生仿色', .3)
                entered.set()
                deadline = time.monotonic() + 3
                while not kwargs['cancel']():
                    if time.monotonic() > deadline:
                        raise AssertionError('Cancel callback was not signalled')
                    time.sleep(.01)
                raise InterruptedError('Reference preparation cancelled')

            module.prepare = cancellable
            cancelled_id = backend.reconstruct_source_design(parent_id, edits)['job_id']
            assert entered.wait(timeout=2)
            backend.cancel_job(cancelled_id)
            assert wait_job(backend, cancelled_id)['status'] == 'cancelled'
            assert not (backend.jobs.path(cancelled_id) / 'prepared-plan.json').exists()
            assert_retained(cancelled_id)

            # A killed backend leaves only the snapshot: listing must not show
            # it as still computing or pretend that execution has resumed.
            orphan = backend.jobs.create({'kind': 'native_design_reconstruction', 'run_id': parent_id,
                'input_sha256': digest(edits), 'edited_run': calls[0],
                'project_pointer_updated': False, 'generates_reference': False})
            orphan_id = orphan['job_id']
            backend.jobs.update(orphan_id, status='running', stage='transfer')
            recovered = Backend(root)
            listed = {job['job_id']: job for job in recovered.jobs_list()['jobs']}
            assert listed[orphan_id]['status'] == 'failed' and listed[orphan_id]['stage'] == 'interrupted'
            assert_retained(orphan_id)

            # Retrying is an explicit new local job with the same retained
            # inputs, and never reopens an interrupted terminal snapshot.
            module.prepare = successful
            retried_id = backend.reconstruct_source_design(parent_id, edits)['job_id']
            assert wait_job(backend, retried_id)['status'] == 'succeeded'
            assert backend.job(orphan_id)['job']['status'] == 'failed'
            assert_retained(retried_id)

            try:
                backend.reconstruct_source_design(parent_id,{**edits,'controls_only':True})
                raise AssertionError('Retired control upgrade accepted')
            except ValueError: pass

            ui_contract(root, plan)
        print(json.dumps({'ok': True, 'mock_pipeline': True, 'local_prepare_success_failure_cancel_restart_retry': True,
            'stage_events_persisted': True, 'parent_pointer_and_four_strokes_preserved': True,
            'receipt_hash_checked': True, 'no_generation': True, 'overall_strength_ui_contract': True,
            'retired_control_upgrade_rejected':True,
            'real_model_visual_or_end_to_end_acceptance': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
