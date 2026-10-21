"""Real Qt shaping controls and callbacks; captured backend boundaries only.

The saved typed parent is made by the shared offline fixture. Shaping plan and
job responses below are UI transport stubs, not model execution or pixel QA.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image

import qt_workbench as qt
from design_reconstruction_ui_smoke import Window, Pool
from scoped_production_smoke import ScopedProductionTests
from single_source_workflow import copy_source_run
from workbench_store import saved_result_state


class InspectionPool(Pool):
    """Capture HTTP tasks, but execute real local checks on actual Qt workers."""
    def start(self, task):
        super().start(task)
        if isinstance(task, qt.ShapingInspectionTask): qt.QThreadPool.globalInstance().start(task)


def wait_inspections(window, app, timeout=30):
    deadline = time.monotonic()+timeout
    while window.shaping_inspections:
        app.processEvents()
        if time.monotonic()>deadline: raise AssertionError('Read-only shaping worker did not finish')
        time.sleep(.002)
    app.processEvents()


def geometric_controls(app):
    from photo_shaping_smoke import ShapingTests
    fixture=ShapingTests(methodName='runTest');fixture.setUp()
    window=None
    try:
        parent=fixture.parent
        _,candidate=fixture.start(fixture.request(part='face'))
        window=Window(fixture.root,'http://127.0.0.1:1');window.pool=InspectionPool()
        window.load_source_run(parent,selection_explicit=True);window.show();app.processEvents()
        wait_inspections(window,app)
        controls=window.shaping_panels[-1];before=saved_result_state(fixture.root)
        state=window.shaping_parent_state();assert state['enabled'],state['reason']
        invalid=copy.deepcopy(candidate)
        invalid['task_intent']['services']={'text_vision':False,'visual_review':False}
        assert not window.validate_shaping_candidate(invalid)
        assert window.source_run_id==parent['run_id'] and saved_result_state(fixture.root)==before
        window._start_shaping_inspection('candidate',candidate,parent_run=parent,parent=state['parent'],
            request=candidate['options']['shaping_state']['request'])
        wait_inspections(window,app)
        assert window.validate_shaping_candidate(candidate)
        assert window.previews['final'].path==Path(candidate['artifacts']['auto_draft'])
        assert window.source_run_id==parent['run_id'] and saved_result_state(fixture.root)==before
        controls['before'].click();app.processEvents()
        assert window.previews['final'].path==Path(fixture.u['path'])/'preview.png'
        controls['before'].click();controls['adopt'].click();wait_inspections(window,app)
        assert window.source_run_id==candidate['run_id']
        assert saved_result_state(fixture.root)['selected_saved_run_id']==candidate['run_id']
        assert controls['parts']['face'][0].isChecked() and controls['restore'].isEnabled()
        controls['restore'].click()
        task=window.pool.tasks[-1]
        assert task.path=='/api/tasks/plan'
        assert task.body['shaping_request']=={'action':'restore'}
        assert task.body['creative_task']['services']=={'text_vision':False,'visual_review':False}
        owned_u=candidate['options']['shaping_state']['base_appearance']
        assert owned_u['asset_id']==fixture.u['asset_id']
        assert task.body['reference']==str(Path(owned_u['path'])/'preview.png')
    finally:
        if window is not None:
            window.session_timer.stop();window.job_poll_timer.stop();window.close();app.processEvents()
            qt.QThreadPool.globalInstance().waitForDone(30000)
        fixture.doCleanups()


def main():
    app = qt.QApplication.instance() or qt.QApplication([])
    fixture = ScopedProductionTests(); fixture.setUp()
    window = None
    try:
        _, parent = fixture.start(fixture.request(fixture.sources[0]))
        original = copy.deepcopy(parent)
        candidate = copy_source_run(fixture.root, parent['run_id'])
        from photo_shaping import authorize
        from photo_appearance import load_asset, freeze_candidate, validate_asset
        from asset_identity import sha256_file
        from float_appearance import srgb_to_linear
        binding = {'source_sha256':parent['source']['sha256'], 'scope':parent['task_plan']['editing_scope'],
            'design_id':parent['generation_design']['observed_design']['design_id']}
        base = candidate['options']['photo_appearance']
        loaded = load_asset(base, **binding); native = np.asarray(loaded['master']).copy()
        height, width = native.shape[:2]; core = [width//4, height//4, 3*width//4, 3*height//4]
        generated = fixture.root/'ui-synthetic-generated.png'
        Image.new('RGB', (width, height), (110, 160, 190)).save(generated)
        provider = fixture.root/'ui-synthetic-provider.json'
        provider.write_text(json.dumps({'status':'reference_candidate', 'source':parent['source'],
            'asset':{'sha256':sha256_file(generated)}, 'provider':'offline_ui_fixture', 'model':'synthetic'}), encoding='utf8')
        coverage = np.zeros((height, width), np.float32)
        x0,y0,x1,y1 = core; coverage[y0:y1,x0:x1] = .5
        patch_rgb = srgb_to_linear(np.asarray(Image.open(generated), np.float32)/255)
        native[y0:y1,x0:x1] = native[y0:y1,x0:x1]*.5+patch_rgb[y0:y1,x0:x1]*.5
        shaped = freeze_candidate(fixture.root/'ui-synthetic-appearance', kind='mixed_regional', master=native,
            parent=base, coverage=coverage, generation={'image':{'path':str(generated), 'sha256':sha256_file(generated)},
                'provider_receipt':{'path':str(provider), 'sha256':sha256_file(provider)},
                'core_box':core, 'context_box':[0,0,width,height]}, **binding)
        receipt = validate_asset(shaped, **binding)
        auth = authorize(fixture.root, parent_run_id=parent['run_id'], source=parent['source'],
            reference={'path':str(Path(parent['options']['photo_appearance']['path'])/'preview.png'),
                'sha256':validate_asset(parent['options']['photo_appearance'], **binding)['files']['preview']['sha256']},
            scope=binding['scope'], request={'action':'apply', 'parts':{'waist':.3}})
        candidate['options']['photo_appearance'] = shaped
        candidate['task_intent']['mode'] = 'existing_reference'
        candidate['options']['task_intent']['mode'] = 'existing_reference'
        candidate['task_plan']['mode'] = 'existing_reference'
        candidate['task_plan']['intent']['mode'] = 'existing_reference'
        candidate['task_plan'].pop('editing_initialization', None)
        candidate['task_plan'].pop('editing_autofinish', None)
        candidate['task_plan']['shaping_request'] = {'action':'apply', 'parts':{'waist':.3}}
        candidate['task_plan']['shaping_authorization'] = auth
        candidate['artifacts']['auto_draft'] = str(Path(shaped['path'])/'preview.png')
        candidate['reference'].update(path=candidate['artifacts']['auto_draft'], sha256=receipt['files']['preview']['sha256'])
        candidate['options']['shaping_state'] = {
            'schema':'cosluma_shaping_state.v1', 'base_appearance':copy.deepcopy(base),
            'enabled':True, 'request':{'action':'apply', 'parts':{'waist':.3}},
            'authorization':auth, 'selected_asset_id':shaped['asset_id']}
        # Synthetic registered pixels exercise real typed/state checks, while
        # the plan and job responses remain captured UI transport stubs.
        manifest = fixture.root/'outputs/runs'/candidate['run_id']/'manifest.json'
        manifest.write_text(json.dumps(candidate), encoding='utf8')
        with patch('sys.excepthook') as unhandled:
            window = Window(fixture.root, 'http://127.0.0.1:1'); window.pool = InspectionPool()
            window.load_source_run(parent, selection_explicit=True); window.show(); app.processEvents()
            wait_inspections(window,app)
            controls = window.shaping_panels[-1]
            assert not window.shaping_parts
            assert all(not row[0].isChecked() for row in controls['parts'].values())
            assert not controls['check'].isEnabled() and not controls['restore'].isEnabled()
            selected = saved_result_state(fixture.root)

            def check_apply():
                controls['parts']['waist'][0].setChecked(True)
                controls['parts']['waist'][1].setValue(.3)
                controls['check'].click()
                pending = window.pool.tasks[-1]
                assert pending.path == '/api/tasks/plan'
                request = copy.deepcopy(pending.body)
                assert request['shaping_request'] == {'action':'apply', 'parts':{'waist':.3}}
                assert request['reference'] == str(Path(parent['options']['photo_appearance']['path'])/'preview.png')
                assert request['parent_run_id'] == parent['run_id'] and request['preserve_project_selection'] is True
                assert request['creative_task']['services'] == {'text_vision':True, 'visual_review':True}
                assert not {'editing_iteration', 'editing_autofinish'} & set(request)
                plan = {'ready':True, 'plan_id':'ui-checked-shaping', 'mode':'existing_reference',
                    'shaping_request':request['shaping_request'], 'shaping_authorization':{},
                    'editing_scope':copy.deepcopy(parent['task_plan']['editing_scope']),
                    'image_budget':{'shaping':1}, 'services':{'text_vision':True, 'visual_review':True,
                        'text_vision_requests_upper_bound':3}}
                pending.signals.result.emit({'ok':True, 'request':request, 'plan':plan})
                assert controls['start'].isEnabled()
                return request, plan

            def begin_candidate():
                before_selection = saved_result_state(fixture.root)
                request, plan = check_apply()
                controls['start'].click(); task = window.pool.tasks[-1]
                assert task.path == '/api/tasks/start' and task.body['plan_id'] == plan['plan_id']
                assert task.body['async'] and task.body['preserve_project_selection']
                assert window.product_run['run_id'] == parent['run_id']
                task.signals.result.emit({'ok':True, 'run':copy.deepcopy(candidate)})
                wait_inspections(window,app)
                assert window.shaping_candidate
                assert window.previews['final'].path == Path(candidate['artifacts']['auto_draft'])
                assert window.product_run['run_id'] == parent['run_id']
                assert saved_result_state(fixture.root) == before_selection
                return task

            begin_candidate()
            window.result('job-run-loaded', {'run':copy.deepcopy(candidate)})
            assert window.source_run_id == parent['run_id'] and saved_result_state(fixture.root) == selected
            zoom = window.previews['final'].zoom
            controls['before'].click(); app.processEvents()
            assert window.previews['final'].path == Path(parent['options']['photo_appearance']['path'])/'preview.png'
            assert abs(window.previews['final'].zoom-zoom) < 1.
            controls['before'].click(); app.processEvents()
            assert window.previews['final'].path == Path(candidate['artifacts']['auto_draft'])
            controls['cancel'].click()
            assert not window.shaping_candidate
            assert window.product_run['run_id'] == parent['run_id']
            assert window.previews['final'].path == Path(parent['artifacts']['auto_draft'])
            assert saved_result_state(fixture.root) == selected

            # Real apply remains behind a separate checked start; changing a
            # part after checking invalidates the pending plan.
            check_apply(); controls['parts']['waist'][0].click()
            count = len(window.pool.tasks); controls['start'].click()
            assert len(window.pool.tasks) == count and window.creative_task_plan is None
            controls['parts']['waist'][0].click()

            # Busy/dirty/stale parents cannot submit or adopt a candidate.
            window._session_dirty = True; window.update_shaping_actions()
            assert not controls['check'].isEnabled()
            count = len(window.pool.tasks); window.plan_shaping('apply')
            assert len(window.pool.tasks) == count
            window._session_dirty = False; window.render_pending = True; window.update_shaping_actions()
            assert not controls['check'].isEnabled()
            window.render_pending = False; window.update_shaping_actions()
            begin_candidate(); controls['parts']['waist'][1].setValue(.4)
            assert window.shaping_candidate is None
            controls['parts']['waist'][1].setValue(.3)
            begin_candidate(); window.product_run['options']['shaping_state'] = {'enabled':False}
            window.adopt_shaping_candidate()
            assert window.product_run['run_id'] == parent['run_id']
            window.load_source_run(parent, selection_explicit=True)
            wait_inspections(window,app)

            # Cancel before an asynchronous start response still cancels the
            # returned job, and a late saved child cannot switch the view.
            check_apply(); controls['start'].click(); started = window.pool.tasks[-1]
            token = window.shaping_execution['token']; controls['cancel'].click()
            started.signals.result.emit({'ok':True, 'async':True, 'job_id':'shape-ui-cancelled'})
            assert any(getattr(task, 'path', '') == '/api/jobs/shape-ui-cancelled/cancel' for task in window.pool.tasks)
            window.result('job-status', {'job':{'job_id':'shape-ui-cancelled', 'status':'cancelled'}})
            window.result('shaping-candidate:'+token, {'run':candidate})
            assert window.product_run['run_id'] == parent['run_id'] and not window.shaping_candidate
            assert saved_result_state(fixture.root) == selected

            check_apply(); controls['start'].click(); late_start = window.pool.tasks[-1]
            window.load_source_run(candidate, selection_explicit=True)
            late_start.signals.result.emit({'ok':True, 'run':candidate})
            assert window.source_run_id == candidate['run_id'] and not window.shaping_candidate
            window.load_source_run(parent, selection_explicit=True)
            wait_inspections(window,app)

            # A changed version rejects a late plan without starting a task.
            controls['parts']['waist'][0].setChecked(True)
            controls['check'].click(); pending = window.pool.tasks[-1]
            window.load_source_run(candidate, selection_explicit=True)
            pending.signals.result.emit({'ok':True, 'request':pending.body,
                'plan':{'ready':True, 'plan_id':'late'}})
            assert not window.creative_task_plan
            window.load_source_run(parent, selection_explicit=True)
            wait_inspections(window,app)
            begin_candidate(); controls['adopt'].click(); wait_inspections(window,app)
            assert window.source_run_id == candidate['run_id']
            assert saved_result_state(fixture.root)['selected_saved_run_id'] == candidate['run_id']
            assert controls['restore'].isEnabled() and controls['parts']['waist'][0].isChecked()
            for button in window.editing_improve_buttons:
                assert not button.isHidden() and not button.isEnabled()
                assert button.text() == '普通反馈编辑（形体结果暂不支持）'
                assert '当前形体结果支持改力度、恢复和导出；普通反馈继续编辑尚未支持' in button.toolTip()
                count = len(window.pool.tasks); button.click()
                assert len(window.pool.tasks) == count
            count = len(window.pool.tasks); window.plan_editing_improvement()
            assert len(window.pool.tasks) == count and '普通反馈继续编辑尚未支持' in window.operation_state.text()

            # Restore uses the owned U preview, no generation/vision services,
            # and still checks a task before publishing its restored child.
            controls['restore'].click(); pending = window.pool.tasks[-1]
            assert pending.path == '/api/tasks/plan'
            assert pending.body['shaping_request'] == {'action':'restore'}
            assert pending.body['creative_task']['services'] == {'text_vision':False, 'visual_review':False}
            assert pending.body['reference'] == str(Path(base['path'])/'preview.png')
            assert pending.body['parent_run_id'] == candidate['run_id']
            assert window.source_run_id == candidate['run_id']

            # A terminal job fetches a candidate using its own callback; it
            # never follows the ordinary job-run-loaded auto-selection path.
            restored = copy.deepcopy(candidate); restored['run_id'] = 'ui-restored-child'
            restored['parent_run_id'] = candidate['run_id']
            restored['result_publication']['run_id'] = restored['run_id']
            restored['options']['photo_appearance'] = copy.deepcopy(base)
            restored['artifacts']['auto_draft'] = str(Path(base['path'])/'preview.png')
            base_receipt = validate_asset(base, **binding)
            restored['reference'].update(path=restored['artifacts']['auto_draft'],
                sha256=base_receipt['files']['preview']['sha256'])
            restore_auth = copy.deepcopy(auth)
            restore_auth.update(parent_run_id=candidate['run_id'], parent_manifest_sha256=sha256_file(manifest),
                parent_appearance=shaped, parent_image_sha256=receipt['files']['preview']['sha256'],
                request={'action':'restore'}, manual_provider_target=None, manual_target=None)
            from photo_editing import identity
            restore_auth['authorization_id'] = identity({key:value for key,value in restore_auth.items() if key!='authorization_id'})
            restored['options']['shaping_state'] = {'schema':'cosluma_shaping_state.v1',
                'base_appearance':copy.deepcopy(base), 'enabled':False, 'request':{'action':'restore'},
                'authorization':restore_auth, 'selected_asset_id':base['asset_id']}
            restored['task_plan'].update(shaping_request={'action':'restore'}, shaping_authorization=restore_auth)
            for intent in (restored['task_intent'],restored['options']['task_intent'],restored['task_plan']['intent']):
                intent['services']={'text_vision':False,'visual_review':False}
            restored_manifest = fixture.root/'outputs/runs'/restored['run_id']/'manifest.json'
            restored_manifest.parent.mkdir(parents=True)
            restored_manifest.write_text(json.dumps(restored), encoding='utf8')
            pending.signals.result.emit({'ok':True, 'request':pending.body,
                'plan':{'ready':True, 'plan_id':'restore-ui-plan', 'mode':'existing_reference',
                    'shaping_request':{'action':'restore'}, 'shaping_authorization':restore_auth,
                    'image_budget':{'shaping':0}, 'services':{'text_vision':False, 'visual_review':False,
                        'text_vision_requests_upper_bound':0}}})
            controls['start'].click(); started = window.pool.tasks[-1]
            assert started.body['shaping_request'] == {'action':'restore'}
            started.signals.result.emit({'ok':True, 'async':True, 'job_id':'shape-ui-restore'})
            window.result('job-status', {'job':{'job_id':'shape-ui-restore', 'status':'succeeded',
                'run_id':restored['run_id']}})
            completed = window.pool.tasks[-1]
            assert completed.label.startswith('shaping-candidate:')
            completed.signals.result.emit({'ok':True, 'run':restored})
            wait_inspections(window,app)
            assert window.source_run_id == candidate['run_id'] and window.shaping_candidate
            controls['adopt'].click(); wait_inspections(window,app)
            assert window.source_run_id == restored['run_id'] and not window.shaping_parts
            assert not controls['restore'].isEnabled()
            assert window.product_run['options']['photo_appearance']['asset_id'] == base['asset_id']
            # Reopening restores the saved off state and retains owned U.
            window.load_source_run(restored, selection_explicit=True)
            wait_inspections(window,app)
            assert not controls['parts']['waist'][0].isChecked()
            assert window.product_run['options']['shaping_state']['base_appearance'] == base
            assert fixture.backend.source_run(parent['run_id'])['run']['options']['photo_appearance'] == original['options']['photo_appearance']
            unhandled.assert_not_called()
    finally:
        if window is not None:
            window.session_timer.stop(); window.job_poll_timer.stop(); window.close(); app.processEvents()
            qt.QThreadPool.globalInstance().waitForDone(30000)
        fixture.doCleanups()
    geometric_controls(app)
    print(json.dumps({'ok':True, 'real_qt_actions_and_signals':True,
        'actual_geometric_candidate_preview_and_explicit_adoption':True,
        'shaping_backend_responses':'captured UI transport stubs', 'default_off_and_explicit_parts':True,
        'checked_apply_and_restore':True, 'preview_does_not_select':True, 'explicit_adopt_selects':True,
        'cancel_busy_dirty_stale_and_late_guarded':True, 'actual_model_inferences':0}))


if __name__ == '__main__': main()
