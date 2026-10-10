"""CPU/offscreen cloth-light job and UI contracts; model preparation is mocked."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import types
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from backend_server import Backend
from current_stack_fixture import prepare_fixture
from design_reconstruction_ui_smoke import Pool, Window, last, write_run
from reference_reconstruction_job_smoke import wait_job
from session_draft import digest
import qt_workbench as qt


KIND = 'cloth_boundary_light'


def boundary_plan(plan):
    result = copy.deepcopy(plan)
    result['layers'] = [row for row in result['layers'] if row['kind'] != 'source_cloth_boundary_light']
    result['layers'].append({'id': 'source_cloth_boundary_light', 'kind': 'source_cloth_boundary_light',
        'label': '参考肩袖受光', 'enabled': True, 'strength': 1.,
        'params': {'schema': 'source_cloth_boundary_light.v1'}, 'assets': {}})
    return result


def project(root):
    root.mkdir(parents=True, exist_ok=True)
    run = prepare_fixture(root/'assets', native=(32, 48))
    run.update(root=str(root), run_id='hair-parent', sample='hair-fixture', status='needs_visual_review')
    run['options']['manual_layers'] = {}
    run['options']['appearance_revision']['params'] = {}
    final = root/'final.png'; qt_image = qt.QImage(32, 48, qt.QImage.Format_RGB32)
    qt_image.fill(qt.QColor('#706050')); qt_image.save(str(final))
    run['artifacts'] = {'auto_draft': str(final), 'reference_candidate': run['reference']['path'],
        'render_scene_master': run['options']['appearance_revision']['design_reconstruction_plan']['source_baseline']['assets']['foreground_premultiplied']['path']}
    run['options']['appearance_revision']['design_reconstruction_plan']['layers'].append({
        'id': 'source_boundary_light', 'kind': 'source_boundary_light', 'label': '参考发梢受光',
        'enabled': True, 'strength': .5, 'params': {'schema': 'source_boundary_light.v1'}, 'assets': {}})
    write_run(root, run)
    (root/'project.json').write_text(json.dumps({'kind': 'project', 'workflow_mode': 'single_source',
        'state': {'last_source_run': run['run_id']}}), encoding='utf8')
    return run


def run_smoke():
    checks = 0
    def require(value):
        nonlocal checks
        assert value; checks += 1
    def rejected(call):
        nonlocal checks
        try: call()
        except ValueError: checks += 1
        else: raise AssertionError('Invalid cloth preparation request accepted')
    require(qt.QT_AVAILABLE)
    app = qt.QApplication.instance() or qt.QApplication([])
    with tempfile.TemporaryDirectory(prefix='cos-cloth-workbench-') as temporary:
        base = Path(temporary); root = base/'api'; run = project(root)
        plan = run['options']['appearance_revision']['design_reconstruction_plan']
        edits = {'appearance_revision': copy.deepcopy(run['options']['appearance_revision']), 'manual_layers': {}}
        body = {**copy.deepcopy(edits), 'preparation_kind': KIND}
        manifest = root/'outputs/runs/hair-parent/manifest.json'; pointer = root/'project.json'
        retained = (manifest.read_bytes(), pointer.read_bytes())
        backend = Backend(root, embedded=True)
        def forbidden(*args, **kwargs): raise AssertionError('Cloth preparation entered generation/default preparation')
        backend.command = forbidden; backend.one_click = forbidden
        worker = types.ModuleType('cloth_boundary_worker'); reference = types.ModuleType('reference_pipeline')
        reference.prepare = forbidden; calls = []
        hair_worker = types.ModuleType('boundary_light_worker'); hair_worker.prepare_for_workbench = forbidden
        def keep(job_id, current_edits=None):
            current_edits = edits if current_edits is None else current_edits
            job = backend.jobs.get(job_id)
            require((manifest.read_bytes(), pointer.read_bytes()) == retained)
            require(job['input_sha256'] == digest(current_edits) and job['preparation_kind'] == KIND)
            require(job['edited_run']['options']['appearance_revision'] == current_edits['appearance_revision'])
            require(job['project_pointer_updated'] is False and job['generates_reference'] is False)
            require([path.name for path in (root/'outputs/runs').iterdir()] == ['hair-parent'])
        def success(project_root, edited, output, *, cancel, progress):
            require(project_root == root and not Path(output).exists() and cancel() is False)
            require(Path(output).is_relative_to(root/'outputs/jobs'))
            Path(output).mkdir(); calls.append(copy.deepcopy(edited))
            progress('reference_matte', '正在准备肩袖受光参考', .3)
            return {'ok': True, 'design_reconstruction_plan': boundary_plan(edited['options']['appearance_revision']['design_reconstruction_plan']), 'cache_hit': True,
                'no_change': False, 'receipt': {'state': 'prepared', 'changed_pixels': 7, 'image_generation_submissions': 0}}
        with patch.dict(sys.modules, {'cloth_boundary_worker': worker, 'boundary_light_worker': hair_worker, 'reference_pipeline': reference}):
            worker.prepare_for_workbench = success
            started = backend.reconstruct_source_design(run['run_id'], body); job_id = started['job_id']
            require(started['job']['preparation_kind'] == KIND)
            require(any(row['kind'] == 'source_boundary_light' for row in started['job']['edited_run']['options']['appearance_revision']['design_reconstruction_plan']['layers']))
            completed = wait_job(backend, job_id); require(completed['status'] == 'succeeded')
            receipt = backend.job(job_id)['reconstruction_result']
            require(receipt['input_sha256'] == digest(edits) and receipt['preparation_kind'] == KIND)
            require(receipt['preparation_contract'] == 'source_cloth_boundary_light.v1')
            require(receipt['cache_hit'] is True and receipt['no_change'] is False and receipt['receipt']['changed_pixels'] == 7)
            require('肩袖受光' in completed['message'] and completed['result_summary']['visual_acceptance'] is False)
            require(Backend(root).job(job_id)['reconstruction_result'] == receipt); keep(job_id)
            # A re-signed envelope still cannot change the durable job kind.
            receipt_path = backend.jobs.path(job_id)/'prepared-plan.json'; original = receipt_path.read_bytes()
            bad = copy.deepcopy(receipt); bad['preparation_kind'] = 'reference_appearance'
            bad_bytes = json.dumps(bad).encode('utf8'); receipt_path.write_bytes(bad_bytes)
            summary = copy.deepcopy(completed['result_summary']); summary['prepared_plan_sha256'] = hashlib.sha256(bad_bytes).hexdigest()
            backend.jobs.update(job_id, result_summary=summary); rejected(lambda: backend.job(job_id))
            receipt_path.write_bytes(original); backend.jobs.update(job_id, result_summary=completed['result_summary'])
            job_count = len(backend.jobs.list())
            for kind in ('unsupported', None, []):
                rejected(lambda kind=kind: backend.reconstruct_source_design(run['run_id'], {**body, 'preparation_kind': kind}))
            for mutation in ('no-appearance', 'old-appearance', 'duplicate-appearance', 'duplicate-boundary-on', 'duplicate-boundary-off', 'missing-baseline'):
                changed = copy.deepcopy(body); changed_plan = changed['appearance_revision']['design_reconstruction_plan']
                if mutation == 'no-appearance': changed_plan['layers'].pop(0)
                elif mutation == 'old-appearance': changed_plan['layers'][0]['params']['schema'] = 'reference_appearance_stack.v1'
                elif mutation == 'duplicate-appearance': changed_plan['layers'].append(copy.deepcopy(changed_plan['layers'][0]))
                elif mutation == 'missing-baseline': changed_plan.pop('source_baseline')
                else:
                    changed['appearance_revision']['design_reconstruction_plan'] = boundary_plan(changed_plan)
                    changed_plan = changed['appearance_revision']['design_reconstruction_plan']
                    changed_plan['layers'][-1]['enabled'] = mutation == 'duplicate-boundary-on'
                    changed_plan['layers'].append(copy.deepcopy(changed_plan['layers'][-1]))
                rejected(lambda changed=changed: backend.reconstruct_source_design(run['run_id'], changed))
            require(len(backend.jobs.list()) == job_count and len(calls) == 1)
            for enabled in (True, False):
                replacing = copy.deepcopy(body)
                replacing_plan = boundary_plan(plan); replacing_plan['layers'][-1]['enabled'] = enabled
                replacing_plan['layers'][-1]['strength'] = .35
                replacing['appearance_revision']['design_reconstruction_plan'] = replacing_plan
                replacing['appearance_revision']['params'] = {'background': {'exposure': .2}}
                replacing_edits = {key: value for key, value in replacing.items() if key != 'preparation_kind'}
                replacement_id = backend.reconstruct_source_design(run['run_id'], replacing)['job_id']
                require(wait_job(backend, replacement_id)['status'] == 'succeeded')
                replaced = backend.job(replacement_id)['reconstruction_result']['design_reconstruction_plan']
                require(sum(row['kind'] == 'source_cloth_boundary_light' for row in replaced['layers']) == 1)
                require(next(row for row in replaced['layers'] if row['kind'] == 'source_boundary_light') ==
                    next(row for row in replacing_plan['layers'] if row['kind'] == 'source_boundary_light'))
                require(next(row for row in replaced['layers'] if row['kind'] == 'source_cloth_boundary_light')['strength'] == 1)
                keep(replacement_id, replacing_edits)
            def no_change(project_root, edited, output, **kwargs):
                require(not Path(output).exists())
                return {'ok': True, 'design_reconstruction_plan': copy.deepcopy(edited['options']['appearance_revision']['design_reconstruction_plan']), 'cache_hit': True,
                    'no_change': True, 'receipt': {'changed_pixels': 0}}
            worker.prepare_for_workbench = no_change
            empty_id = backend.reconstruct_source_design(run['run_id'], body)['job_id']; empty = wait_job(backend, empty_id)
            require(empty['status'] == 'succeeded' and '依据不足' in empty['message'])
            require(backend.job(empty_id)['reconstruction_result']['design_reconstruction_plan'] == plan); keep(empty_id)
            empty_replace_id = backend.reconstruct_source_design(run['run_id'], replacing)['job_id']
            require(wait_job(backend, empty_replace_id)['status'] == 'succeeded')
            require(backend.job(empty_replace_id)['reconstruction_result']['design_reconstruction_plan'] == replacing_plan)
            keep(empty_replace_id, replacing_edits)
            invalid_prepared = []
            for key in ('source_sha256', 'candidate_sha256', 'canvas', 'duplicate-layer'):
                bad_plan = boundary_plan(replacing_plan)
                if key == 'canvas': bad_plan[key]['source_oriented_size'] = [64, 48]
                elif key == 'duplicate-layer': bad_plan['layers'].append(copy.deepcopy(bad_plan['layers'][-1]))
                else: bad_plan[key] = '0'*64
                invalid_prepared.append({'ok': True, 'design_reconstruction_plan': bad_plan, 'no_change': False})
            for result in (None, {'ok': False}, {'ok': True, 'design_reconstruction_plan': None},
                    {'ok': True, 'design_reconstruction_plan': plan, 'no_change': False},
                    {'ok': True, 'design_reconstruction_plan': boundary_plan(plan), 'no_change': True}, *invalid_prepared):
                worker.prepare_for_workbench = lambda *args, result=result, **kwargs: copy.deepcopy(result)
                failure_id = backend.reconstruct_source_design(run['run_id'], replacing)['job_id']
                failure = wait_job(backend, failure_id)
                require(failure['status'] == 'failed' and '肩袖受光' in failure['message']); keep(failure_id, replacing_edits)
            entered = threading.Event()
            def cancellable(*args, cancel, progress, **kwargs):
                progress('native_boundary', '正在准备肩袖受光', .5); entered.set()
                deadline = time.monotonic()+3
                while not cancel():
                    if time.monotonic() > deadline: raise AssertionError('Cancel not propagated')
                    time.sleep(.01)
                raise InterruptedError('cancelled')
            worker.prepare_for_workbench = cancellable
            cancel_id = backend.reconstruct_source_design(run['run_id'], replacing)['job_id']; require(entered.wait(2))
            cancelled = backend.cancel_job(cancel_id)['job']; require('肩袖受光' in cancelled['message'])
            require(wait_job(backend, cancel_id)['status'] == 'cancelled')
            require(not (backend.jobs.path(cancel_id)/'prepared-plan.json').exists()); keep(cancel_id, replacing_edits)

        ui_root = base/'ui'; ui_run = project(ui_root); ui_plan = copy.deepcopy(ui_run['options']['appearance_revision']['design_reconstruction_plan'])
        window = Window(ui_root, 'http://127.0.0.1:1'); pool = Pool(); window.pool = pool
        window.load_source_run(ui_run); window.show(); app.processEvents()
        require(window.cloth_boundary_light_button.text() == '补充参考肩袖受光' and window.cloth_boundary_light_button.isEnabled())
        original_plan = copy.deepcopy(window.design_reconstruction_plan)
        def submit(identifier):
            window.cloth_boundary_light_button.click(); request = last(pool, 'design-reconstruct-job')
            require(request.body['preparation_kind'] == KIND and request.path.endswith('/reconstruct-design'))
            editing = {key: value for key, value in request.body.items() if key != 'preparation_kind'}
            require(editing == window._source_edit_payload())
            require(not window.cloth_boundary_light_button.isEnabled())
            request.signals.result.emit({'ok': True, 'async': True, 'job_id': identifier,
                'job': {'preparation_kind': KIND}}); app.processEvents()
            require('肩袖受光' in window.operation_state.text())
            return editing
        def finish(identifier, editing, *, status='succeeded', prepared=None, no_change=False, stale=False):
            job = {'kind': 'native_design_reconstruction', 'job_id': identifier, 'preparation_kind': KIND, 'status': status}
            result = {'run_id': ui_run['run_id'], 'source_sha256': ui_run['source']['sha256'],
                'candidate_sha256': ui_run['reference']['sha256'], 'input_sha256': 'stale' if stale else digest(editing),
                'preparation_kind': KIND, 'no_change': no_change, 'design_reconstruction_plan': prepared or copy.deepcopy(ui_plan)}
            window.result('job-status', {'job': job, 'reconstruction_result': result}); app.processEvents()
        editing = submit('ui-no-change'); version = window.source_light_version
        previews = sum(getattr(task, 'label', '').startswith('source-light-preview:') for task in pool.tasks)
        finish('ui-no-change', editing, no_change=True)
        require(window.design_reconstruction_plan == original_plan and not window._session_dirty)
        require(window.source_light_version == version and '依据不足' in window.operation_state.text())
        require(sum(getattr(task, 'label', '').startswith('source-light-preview:') for task in pool.tasks) == previews)
        for status in ('cancelled', 'failed'):
            editing = submit('ui-'+status); finish('ui-'+status, editing, status=status)
            require(window.design_reconstruction_plan == original_plan and '肩袖受光' in window.operation_state.text())
            require(window.cloth_boundary_light_button.isEnabled())
        editing = submit('ui-stale'); finish('ui-stale', editing, prepared=boundary_plan(ui_plan), stale=True)
        require(window.design_reconstruction_plan == original_plan and '精修已有变化' in window.operation_state.text())
        window.cloth_boundary_light_button.click(); request = last(pool, 'design-reconstruct-job')
        request.signals.error.emit('worker unavailable'); app.processEvents()
        require(window.design_reconstruction_plan == original_plan and window.cloth_boundary_light_button.isEnabled())
        require('肩袖受光' in window.source_light_state.text() and '当前编辑保留' in window.source_light_state.text())
        missing = copy.deepcopy(ui_plan); missing['layers'].pop(0); window.load_design_reconstruction(missing)
        require(not window.cloth_boundary_light_button.isEnabled())
        count = len(pool.tasks); window.prepare_cloth_boundary_light(); require(len(pool.tasks) == count)
        window.load_design_reconstruction(ui_plan)
        editing = submit('ui-prepared'); prepared = boundary_plan(ui_plan)
        immutable = copy.deepcopy(prepared); finish('ui-prepared', editing, prepared=prepared)
        require(window.design_reconstruction_plan == prepared and window._session_dirty and window.design_reconstruction_prepared)
        require(window.cloth_boundary_light_button.isEnabled() and '肩袖受光' in window.operation_state.text())
        require(window.cloth_boundary_light_button.text() == '重新协调肩袖暖光')
        require(window.hair_boundary_light_button.isEnabled() and window.hair_boundary_light_button.text() == '重新准备参考发梢受光')
        editing = submit('ui-reprepared'); replacement = boundary_plan(prepared)
        replacement['layers'][-1]['label'] = '重新协调的肩袖暖光'
        finish('ui-reprepared', editing, prepared=replacement)
        require(window.design_reconstruction_plan == replacement and sum(row['kind'] == 'source_cloth_boundary_light'
            for row in replacement['layers']) == 1)
        prepared = replacement; immutable = copy.deepcopy(prepared)
        for status in ('succeeded', 'cancelled', 'failed'):
            preserved = (copy.deepcopy(window.design_reconstruction_plan), window.source_light_version, window._session_dirty)
            previews = sum(getattr(task, 'label', '').startswith('source-light-preview:') for task in pool.tasks)
            editing = submit('ui-existing-'+status)
            finish('ui-existing-'+status, editing, status=status, prepared=copy.deepcopy(prepared), no_change=status == 'succeeded')
            require((window.design_reconstruction_plan, window.source_light_version, window._session_dirty) == preserved)
            require(window.cloth_boundary_light_button.isEnabled())
            require(sum(getattr(task, 'label', '').startswith('source-light-preview:') for task in pool.tasks) == previews)
        window.design_stack_list.setCurrentRow(len(prepared['layers'])-1)
        window.design_layer_strength.setValue(.5); window.design_layer_enabled.setChecked(False); window.source_light_timer.stop()
        require(prepared == immutable and window.cloth_boundary_light_button.isEnabled())
        window.save_source_lights(); save = last(pool, 'source-adjust')
        saved_plan = save.body['appearance_revision']['design_reconstruction_plan']
        require(saved_plan['layers'][-1]['strength'] == .5 and saved_plan['layers'][-1]['enabled'] is False)
        require('preparation_kind' not in save.body and set(save.body) == {'appearance_revision', 'manual_layers'})
        child = copy.deepcopy(ui_run); child.update(run_id='hair-child', revision=2, parent_run_id=ui_run['run_id'])
        child['options']['appearance_revision'] = copy.deepcopy(save.body['appearance_revision'])
        child['options']['manual_layers'] = copy.deepcopy(save.body['manual_layers'])
        child['render'] = {'run_id': child['run_id'], 'revision': 2, 'operator': {'render_mode': 'native_source_design'}}
        write_run(ui_root, child); save.signals.result.emit({'ok': True, 'run': child}); app.processEvents()
        require(window.source_run_id == 'hair-child' and not window._session_dirty)
        window.close(); app.processEvents()
        reopened = Window(ui_root, 'http://127.0.0.1:1'); reopened.pool = Pool(); reopened.load_source_run(child)
        require(reopened.design_reconstruction_plan == saved_plan and reopened.cloth_boundary_light_button.isEnabled())
        require(reopened.cloth_boundary_light_button.text() == '重新协调肩袖暖光')
        reopened.hair_boundary_light_button.click(); hair_request = last(reopened.pool, 'design-reconstruct-job')
        require(hair_request.body['preparation_kind'] == 'hair_boundary_light')
        require(any(row['kind'] == 'source_cloth_boundary_light' for row in
            hair_request.body['appearance_revision']['design_reconstruction_plan']['layers']))
        hair_request.signals.error.emit('mock hair worker unavailable'); app.processEvents()
        require(reopened.design_reconstruction_plan == saved_plan)
        reopened.design_stack_list.setCurrentRow(len(saved_plan['layers'])-1)
        require(reopened.design_layer_strength.value() == .5 and not reopened.design_layer_enabled.isChecked())
        for captured in (pool, reopened.pool):
            require(not any(task.path == '/api/one-click' or task.path.endswith(('/retry', '/prepare', '/recognize', '/repair-final'))
                for task in captured.tasks if hasattr(task, 'path')))
        reopened.close(); app.processEvents()
    return {'ok': True, 'checks': checks, 'mock_boundary_preparation_only': True, 'current_hair_layer_accepted_and_preserved': True,
        'default_pipeline_not_invoked': True, 'job_preparation_kind_persisted_edit_digest_unchanged': True,
        'duplicate_missing_appearance_unknown_kind_and_invalid_worker_results_rejected': True,
        'cancel_and_failure_preserve_parent_and_pointer': True, 'no_change_preserves_ui_without_preview_or_dirty_state': True,
        'ui_current_payload_stale_rejection_and_same_kind_repreparation': True,
        'api_replaces_one_existing_layer_on_or_off_and_preserves_hair': True,
        'ui_preview_save_and_reopen_existing_child_flow': True,
        'model_inferences': 0, 'image_generation_submissions': 0, 'visual_acceptance': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path); args = parser.parse_args()
    report = run_smoke()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(report, ensure_ascii=False))
