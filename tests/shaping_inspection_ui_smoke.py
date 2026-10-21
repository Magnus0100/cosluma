"""Cold inspection stays off Qt; cached UI facts never authorize adoption."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import unittest
from unittest.mock import patch

from PySide6.QtTest import QTest
import qt_workbench as qt
import photo_shaping_smoke as shaping_fixture
import shaping_inspection as inspection
from shaping_ui_smoke import InspectionPool, Window, wait_inspections
from workbench_store import saved_result_state


class PaintCounter(qt.QObject):
    def __init__(self, parent):
        super().__init__(parent); self.count = 0
    def eventFilter(self, watched, event):
        if event.type() == qt.QEvent.Paint: self.count += 1
        return False


class ShapingInspectionUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = qt.QApplication.instance() or qt.QApplication([])
        cls.fixture = shaping_fixture.ShapingTests(methodName='runTest'); cls.fixture.setUp()
        cls.parent = copy.deepcopy(cls.fixture.parent)
        _, cls.candidate = cls.fixture.start(cls.fixture.request(part='face'))

    @classmethod
    def tearDownClass(cls):
        qt.QThreadPool.globalInstance().waitForDone(30000)
        cls.fixture.doCleanups()

    def setUp(self):
        self.root = self.fixture.root; self.backups = {}
        self.window = Window(self.root, 'http://127.0.0.1:1'); self.window.pool = InspectionPool()
        self.window.load_source_run(self.parent, selection_explicit=True)
        self.window.show(); wait_inspections(self.window, self.app)
        self.controls = self.window.shaping_panels[-1]
        self.assertTrue(self.window.shaping_parent_state()['enabled'])
        self.selected = saved_result_state(self.root)
        self.hook = patch('sys.excepthook'); self.unhandled = self.hook.start()

    def tearDown(self):
        self.window.session_timer.stop(); self.window.job_poll_timer.stop()
        self.window.close(); self.app.processEvents()
        qt.QThreadPool.globalInstance().waitForDone(30000); self.app.processEvents()
        for path, (data, stat) in self.backups.items():
            path.write_bytes(data); os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.unhandled.assert_not_called(); self.hook.stop()

    def keep(self, path):
        path = Path(path)
        if path not in self.backups: self.backups[path] = (path.read_bytes(), path.stat())
        return path

    def wait_for(self, predicate, timeout=5):
        deadline = time.monotonic()+timeout
        while not predicate():
            self.app.processEvents(); QTest.qWait(5)
            self.assertLess(time.monotonic(), deadline, 'Expected Qt/worker observation did not arrive; '+
                self.window.operation_state.text()+'; '+str((self.window.shaping_parent_fact or {}).get('reason')))

    def hold(self, function):
        started = threading.Event(); release = threading.Event(); main_thread = threading.get_ident()
        self.addCleanup(release.set)
        def delayed(*args, **kwargs):
            self.assertNotEqual(threading.get_ident(), main_thread, 'Authority check entered GUI thread')
            started.set()
            if not release.wait(5): raise AssertionError('Test did not release read-only worker')
            return function(*args, **kwargs)
        return delayed, started, release

    def receive(self, run=None):
        run = copy.deepcopy(run or self.candidate)
        state = self.window.shaping_parent_state()
        self.assertTrue(state['enabled'], state['reason'])
        self.window.shaping_execution = {'token':'fixture-'+str(time.monotonic_ns()),
            'parent':state['parent'], 'revision':self.window.shaping_revision,
            'request':run['options']['shaping_state']['request'], 'cancelled':False}
        self.window.handle_shaping_result('shaping-candidate:'+self.window.shaping_execution['token'], {'run':run})

    def preview(self):
        self.receive(); wait_inspections(self.window, self.app)
        self.assertTrue(self.window.shaping_candidate, self.window.operation_state.text())
        self.assertTrue(self.controls['adopt'].isEnabled())
        self.assertEqual(self.window.source_run_id, self.parent['run_id'])
        self.assertEqual(saved_result_state(self.root), self.selected)

    def test_cold_parent_check_keeps_paint_and_strength_edits_responsive_and_coalesced(self):
        import appearance_assets
        appearance_assets.clear_proof_cache()
        delayed, started, release = self.hold(inspection.inspect_parent)
        actual_audit = appearance_assets.force_audit; audit_threads = []
        def observed_audit(*args, **kwargs):
            audit_threads.append(threading.get_ident())
            self.assertNotEqual(threading.get_ident(), threading.main_thread().ident)
            return actual_audit(*args, **kwargs)
        paint = PaintCounter(self.window); self.window.installEventFilter(paint)
        ticks = []; timer = qt.QTimer(self.window); timer.setInterval(5); timer.timeout.connect(lambda:ticks.append(1))
        with patch('shaping_inspection.inspect_parent', side_effect=delayed), patch('appearance_assets.force_audit', side_effect=observed_audit):
            self.window.shaping_parent_fact = None
            before = len([t for t in self.window.pool.tasks if isinstance(t, qt.ShapingInspectionTask)])
            begun = time.monotonic(); self.window.update_shaping_actions()
            self.assertLess(time.monotonic()-begun, .2)
            self.wait_for(started.is_set); timer.start(); self.window.update()
            self.controls['parts']['face'][0].click()
            self.controls['parts']['face'][1].setValue(.45)
            for _ in range(12): self.window.update_shaping_actions()
            QTest.qWait(80); self.app.processEvents(); timer.stop()
            self.assertGreater(len(ticks), 3); self.assertGreater(paint.count, 0)
            self.assertEqual(self.window.shaping_parts, {'face':.45})
            self.assertFalse(self.controls['check'].isEnabled())
            after = len([t for t in self.window.pool.tasks if isinstance(t, qt.ShapingInspectionTask)])
            self.assertEqual(after, before+1, 'Strength/UI refreshes repeated the same cold parent audit')
            release.set(); wait_inspections(self.window, self.app)
        self.assertTrue(audit_threads)
        self.assertTrue(self.controls['check'].isEnabled())
        self.assertEqual(saved_result_state(self.root), self.selected)

    def test_cancel_during_candidate_check_discards_late_preview_and_selection(self):
        delayed, started, release = self.hold(inspection.inspect_candidate)
        with patch('shaping_inspection.inspect_candidate', side_effect=delayed):
            self.receive(); self.wait_for(started.is_set)
            self.assertTrue(self.controls['cancel'].isEnabled())
            begun = time.monotonic(); self.controls['cancel'].click()
            self.assertLess(time.monotonic()-begun, .2)
            self.assertNotIn('candidate', self.window.shaping_inspections)
            release.set(); qt.QThreadPool.globalInstance().waitForDone(30000); self.app.processEvents()
        self.assertIsNone(self.window.shaping_candidate)
        self.assertEqual(self.window.previews['final'].path, Path(self.parent['artifacts']['auto_draft']))
        self.assertEqual(saved_result_state(self.root), self.selected)

    def test_late_parent_and_candidate_checks_cannot_replace_newer_view(self):
        delayed, started, release = self.hold(inspection.inspect_candidate)
        with patch('shaping_inspection.inspect_candidate', side_effect=delayed):
            self.receive(); self.wait_for(started.is_set)
            self.window.load_source_run(self.candidate, record_selection=False)
            release.set(); wait_inspections(self.window, self.app)
            qt.QThreadPool.globalInstance().waitForDone(30000); self.app.processEvents()
        self.assertEqual(self.window.source_run_id, self.candidate['run_id'])
        self.assertIsNone(self.window.shaping_candidate)
        self.assertEqual(self.window.shaping_parent_state()['parent']['run_id'], self.candidate['run_id'])
        self.assertEqual(saved_result_state(self.root), self.selected)

    def test_late_cold_parent_fact_does_not_enable_the_previous_version(self):
        delayed, started, release = self.hold(inspection.inspect_parent)
        original = inspection.inspect_parent
        def only_old_parent(root, run, **kwargs):
            return delayed(root, run, **kwargs) if run['run_id'] == self.parent['run_id'] else original(root, run, **kwargs)
        with patch('shaping_inspection.inspect_parent', side_effect=only_old_parent):
            self.window.shaping_parent_fact = None; self.window.update_shaping_actions()
            self.wait_for(started.is_set)
            self.window.load_source_run(self.candidate, record_selection=False)
            wait_inspections(self.window, self.app)
            release.set(); qt.QThreadPool.globalInstance().waitForDone(30000); self.app.processEvents()
        self.assertEqual(self.window.shaping_parent_state()['parent']['run_id'], self.candidate['run_id'])
        self.assertEqual(saved_result_state(self.root), self.selected)

    def test_adopt_rechecks_actual_bytes_after_ui_fact_and_preserves_current_on_tamper(self):
        self.preview()
        delayed, started, release = self.hold(inspection.inspect_candidate)
        receipt = json.loads((Path(self.candidate['options']['photo_appearance']['path'])/'receipt.json').read_text('utf8'))
        master = self.keep(Path(self.candidate['options']['photo_appearance']['path'])/receipt['files']['master']['file'])
        with patch('shaping_inspection.inspect_candidate', side_effect=delayed):
            self.controls['adopt'].click(); self.wait_for(started.is_set)
            original, stat = self.backups[master]; changed = bytearray(original); changed[-1] ^= 1
            master.write_bytes(changed); os.utime(master, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            release.set(); wait_inspections(self.window, self.app)
        self.assertEqual(self.window.source_run_id, self.parent['run_id'])
        self.assertEqual(saved_result_state(self.root), self.selected)
        self.assertIn('未采用', self.window.operation_state.text())

    def test_wrong_permission_and_unknown_schema_never_become_preview(self):
        manifest = self.keep(self.root/'outputs/runs'/self.candidate['run_id']/'manifest.json')
        invalid = copy.deepcopy(self.candidate); invalid['task_intent']['services']['visual_review'] = False
        manifest.write_text(json.dumps(invalid), encoding='utf8')
        self.receive(invalid); wait_inspections(self.window, self.app)
        self.assertIsNone(self.window.shaping_candidate)
        self.assertIn('权限', self.window.operation_state.text())
        descriptor = self.candidate['options']['photo_appearance']
        receipt = self.keep(Path(descriptor['path'])/'receipt.json')
        value = json.loads(receipt.read_text('utf8')); value['schema'] = 'cosluma_unknown_appearance.v999'
        receipt.write_text(json.dumps(value), encoding='utf8')
        unknown = copy.deepcopy(self.candidate)
        unknown['options']['photo_appearance']['receipt_sha256'] = hashlib.sha256(receipt.read_bytes()).hexdigest()
        manifest.write_text(json.dumps(unknown), encoding='utf8')
        self.receive(unknown); wait_inspections(self.window, self.app)
        self.assertIsNone(self.window.shaping_candidate)
        self.assertEqual(saved_result_state(self.root), self.selected)

    def test_malformed_state_and_separate_saved_draft_tamper_fail_closed(self):
        invalid = copy.deepcopy(self.parent); invalid['options']['shaping_state'] = ['invalid']
        fact = inspection.inspect_parent(self.root, invalid)
        self.assertFalse(fact['ok']); self.assertIn('无效', fact['reason'])
        candidate = copy.deepcopy(self.candidate)
        descriptor = candidate['options']['photo_appearance']
        manifest = self.keep(self.root/'outputs/runs'/candidate['run_id']/'manifest.json')
        # The saved render may live outside its typed asset directory. Prime
        # the shared stat memo, then replace same-size bytes and restore mtime.
        separate = self.root/'separate-saved-shaping-preview.png'
        separate.write_bytes((Path(descriptor['path'])/'preview.png').read_bytes())
        self.addCleanup(lambda:separate.unlink(missing_ok=True))
        candidate['artifacts']['auto_draft'] = str(separate)
        manifest.write_text(json.dumps(candidate), encoding='utf8')
        from asset_identity import sha256_file
        good = sha256_file(separate); stat = separate.stat()
        data = bytearray(separate.read_bytes()); data[-1] ^= 1; separate.write_bytes(data)
        os.utime(separate, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        calls = []
        def observed(path, **kwargs):
            if Path(path) == separate: calls.append(kwargs.get('force'))
            return sha256_file(path, **kwargs)
        with patch('asset_identity.sha256_file', side_effect=observed):
            self.receive(candidate); wait_inspections(self.window, self.app)
        self.assertNotEqual(hashlib.sha256(separate.read_bytes()).hexdigest(), good)
        self.assertEqual(calls, [True], 'Separate saved preview was checked using a stale stat memo')
        self.assertIsNone(self.window.shaping_candidate)
        self.assertEqual(saved_result_state(self.root), self.selected)

    def test_adopt_uses_nonwaiting_existing_commit_and_only_loads_after_success(self):
        self.preview()
        from cache_store import exclusive_file_lock
        import workbench_store
        # Use the actual store location, not a made-up lock fixture.
        recent = workbench_store._settings_dir(self.root)/'recent_projects.json'
        commit = workbench_store.select_saved_result
        with patch('workbench_store.select_saved_result', wraps=commit) as observed:
            with exclusive_file_lock(recent.with_suffix('.lock'), timeout=0):
                self.controls['adopt'].click(); wait_inspections(self.window, self.app)
            self.assertEqual(self.window.source_run_id, self.parent['run_id'])
            self.assertEqual(saved_result_state(self.root), self.selected)
            self.assertTrue(self.window.shaping_candidate)
            self.assertIn('可重试', self.window.operation_state.text())
            self.controls['adopt'].click(); wait_inspections(self.window, self.app)
            self.assertEqual(observed.call_count, 2, 'load_source_run performed a second default waiting selection')
            for call in observed.call_args_list:
                self.assertEqual(call.kwargs['lock_timeout'], 0)
                self.assertIs(call.kwargs['return_state'], False)
                self.assertTrue(call.kwargs['expected_manifest_sha256'])
                self.assertIsNotNone(call.kwargs['guard'])
        self.assertEqual(self.window.source_run_id, self.candidate['run_id'])
        self.assertEqual(saved_result_state(self.root)['selected_saved_run_id'], self.candidate['run_id'])

    def test_failed_session_checkpoint_before_load_keeps_durable_selection_and_candidate(self):
        self.preview()
        def fail_checkpoint(): self.window._session_checkpoint_failed = True
        import workbench_store
        with patch.object(self.window, '_checkpoint_session', side_effect=fail_checkpoint), patch(
                'workbench_store.select_saved_result', wraps=workbench_store.select_saved_result) as commit:
            self.controls['adopt'].click(); wait_inspections(self.window, self.app)
            commit.assert_not_called()
        self.assertEqual(self.window.source_run_id, self.parent['run_id'])
        self.assertEqual(saved_result_state(self.root), self.selected)
        self.assertTrue(self.window.shaping_candidate)
        self.assertNotIn('已采用', self.window.operation_state.text())
        self.window._session_checkpoint_failed = False

    def test_manifest_replacement_between_fresh_check_and_commit_preserves_parent(self):
        self.preview()
        import workbench_store
        commit = workbench_store.select_saved_result
        manifest = self.keep(self.root/'outputs/runs'/self.candidate['run_id']/'manifest.json')
        def replaced(*args, **kwargs):
            value = json.loads(manifest.read_text('utf8')); value['independent_test_change'] = True
            manifest.write_text(json.dumps(value), encoding='utf8')
            return commit(*args, **kwargs)
        with patch('workbench_store.select_saved_result', side_effect=replaced):
            self.controls['adopt'].click(); wait_inspections(self.window, self.app)
        self.assertEqual(self.window.source_run_id, self.parent['run_id'])
        self.assertEqual(saved_result_state(self.root), self.selected)
        self.assertTrue(self.window.shaping_candidate)

    def test_close_during_adopt_check_discards_callback_without_writing_project(self):
        self.preview()
        delayed, started, release = self.hold(inspection.inspect_candidate)
        before = {p:p.read_bytes() for p in [self.root/'project.json',
            self.root/'outputs/runs'/self.parent['run_id']/'manifest.json',
            self.root/'outputs/runs'/self.candidate['run_id']/'manifest.json']}
        import workbench_store
        recent = workbench_store._settings_dir(self.root)/'recent_projects.json'; before[recent] = recent.read_bytes()
        with patch('shaping_inspection.inspect_candidate', side_effect=delayed), patch('workbench_store.select_saved_result') as commit:
            self.controls['adopt'].click(); self.wait_for(started.is_set)
            self.window.close(); self.app.processEvents(); release.set()
            qt.QThreadPool.globalInstance().waitForDone(30000); self.app.processEvents()
            commit.assert_not_called()
        self.assertEqual({p:p.read_bytes() for p in before}, before)


if __name__ == '__main__': unittest.main()
