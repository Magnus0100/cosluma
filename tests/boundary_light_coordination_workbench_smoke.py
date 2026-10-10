"""CPU/offscreen atomic hair and cloth API/UI gates; preparation is mocked."""
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
from cloth_boundary_workbench_smoke import project
from design_reconstruction_ui_smoke import Pool, Window, last, write_run
from reference_reconstruction_job_smoke import wait_job
from session_draft import digest
import qt_workbench as qt

KIND = 'boundary_light_coordination'
CONTRACT = 'source_boundary_light_coordination.v1'
KINDS = ('source_boundary_light', 'source_cloth_boundary_light')


def pair(plan):
    result = copy.deepcopy(plan)
    for kind in KINDS:
        current = next((row for row in result['layers'] if row['kind'] == kind), None)
        if current is None:
            current = {'id': kind, 'kind': kind, 'label': kind, 'enabled': True,
                       'strength': 1., 'params': {'schema': kind+'.v1'}, 'assets': {}}
            result['layers'].append(current)
        current['label'] = 'Coordinated '+kind
    return result


def run_smoke():
    checks = 0
    def require(value):
        nonlocal checks
        assert value; checks += 1
    def rejected(call):
        nonlocal checks
        try: call()
        except ValueError: checks += 1
        else: raise AssertionError('Invalid coordinated preparation request accepted')
    require(qt.QT_AVAILABLE)
    app = qt.QApplication.instance() or qt.QApplication([])
    with tempfile.TemporaryDirectory(prefix='cos-boundary-coordination-ui-') as temporary:
        base = Path(temporary); root = base/'api'; run = project(root)
        plan = run['options']['appearance_revision']['design_reconstruction_plan']
        plan = pair(plan)
        for index, kind in enumerate(KINDS):
            layer = next(row for row in plan['layers'] if row['kind'] == kind)
            layer.update(id='retained-'+kind, enabled=index == 0, strength=.3+.2*index)
        run['options']['appearance_revision']['design_reconstruction_plan'] = plan
        write_run(root, run)
        edits = {'appearance_revision': copy.deepcopy(run['options']['appearance_revision']), 'manual_layers': {}}
        edits['appearance_revision']['params'] = {'background': {'exposure': .2}}
        body = {**copy.deepcopy(edits), 'preparation_kind': KIND}
        manifest = root/'outputs/runs/hair-parent/manifest.json'; pointer = root/'project.json'
        retained = (manifest.read_bytes(), pointer.read_bytes())
        backend = Backend(root, embedded=True)
        def forbidden(*args, **kwargs): raise AssertionError('Coordination entered default or individual preparation')
        backend.command = forbidden; backend.one_click = forbidden
        worker = types.ModuleType('boundary_light_coordination_worker')
        mocks = {'boundary_light_coordination_worker': worker}
        for name in ('reference_pipeline', 'boundary_light_worker', 'cloth_boundary_worker'):
            module = types.ModuleType(name); module.prepare = forbidden; module.prepare_for_workbench = forbidden
            mocks[name] = module
        calls = []; validations = []
        def validate(prepared, current):
            require(current['options']['appearance_revision']['design_reconstruction_plan'] == prepared)
            validations.append(copy.deepcopy(prepared)); return True
        worker.validate_prepared_pair = validate
        def keep(identifier, current_edits=None):
            current_edits = edits if current_edits is None else current_edits
            job = backend.jobs.get(identifier)
            require((manifest.read_bytes(), pointer.read_bytes()) == retained)
            require(job['input_sha256'] == digest(current_edits) and job['preparation_kind'] == KIND)
            require(job['edited_run']['options']['appearance_revision'] == current_edits['appearance_revision'])
            require(job['project_pointer_updated'] is False and job['generates_reference'] is False)
            require([path.name for path in (root/'outputs/runs').iterdir()] == ['hair-parent'])
        def success(project_root, edited, output, *, cancel, progress):
            require(project_root == root and not Path(output).exists() and cancel() is False)
            require(Path(output).name == 'boundary-coordination-preparation')
            Path(output).mkdir(); calls.append(copy.deepcopy(edited))
            progress('hair_boundary_light', '发梢 1/2', .1)
            progress('cloth_boundary_light', '肩袖 2/2', .55)
            return {'ok': True, 'changed': True, 'design_reconstruction_plan': pair(
                edited['options']['appearance_revision']['design_reconstruction_plan']), 'cache_hit': True,
                'no_change': False, 'receipt': {'schema': CONTRACT, 'atomic_commit': True,
                    'stages': ['hair_boundary_light', 'cloth_boundary_light'], 'image_generation_submissions': 0}}
        with patch.dict(sys.modules, mocks):
            # Observe the second stage while the result is still unavailable.
            entered = threading.Event(); release = threading.Event()
            def staged(*args, **kwargs):
                result = success(*args, **kwargs); entered.set()
                if not release.wait(3): raise AssertionError('Atomic gate not released')
                return result
            worker.prepare_for_workbench = staged
            identifier = backend.reconstruct_source_design(run['run_id'], body)['job_id']
            require(entered.wait(2))
            require(backend.jobs.get(identifier)['stage'] == 'cloth_boundary_light')
            require(not (backend.jobs.path(identifier)/'prepared-plan.json').exists())
            keep(identifier); release.set()
            completed = wait_job(backend, identifier); require(completed['status'] == 'succeeded')
            receipt = backend.job(identifier)['reconstruction_result']
            require(receipt['preparation_contract'] == CONTRACT and receipt['preparation_kind'] == KIND)
            require(receipt['receipt']['atomic_commit'] and len(validations) == 1)
            prepared = receipt['design_reconstruction_plan']
            require(prepared == pair(plan) and calls[0]['options']['appearance_revision'] == edits['appearance_revision'])
            for kind in KINDS:
                old = next(row for row in plan['layers'] if row['kind'] == kind)
                new = next(row for row in prepared['layers'] if row['kind'] == kind)
                require(all(old[key] == new[key] for key in ('id', 'enabled', 'strength')))
                require(sum(row['kind'] == kind for row in prepared['layers']) == 1)
            require(Backend(root).job(identifier)['reconstruction_result'] == receipt); keep(identifier)
            receipt_path = backend.jobs.path(identifier)/'prepared-plan.json'; receipt_bytes = receipt_path.read_bytes()
            bad = copy.deepcopy(receipt); bad['preparation_kind'] = 'hair_boundary_light'
            data = json.dumps(bad).encode('utf8'); receipt_path.write_bytes(data)
            summary = copy.deepcopy(completed['result_summary']); summary['prepared_plan_sha256'] = hashlib.sha256(data).hexdigest()
            backend.jobs.update(identifier, result_summary=summary); rejected(lambda: backend.job(identifier))
            receipt_path.write_bytes(receipt_bytes); backend.jobs.update(identifier, result_summary=completed['result_summary'])
            count = len(backend.jobs.list())
            for mutation in ('no-appearance', 'old-appearance', 'duplicate-appearance', 'missing-baseline', *KINDS):
                changed = copy.deepcopy(body); changed_plan = changed['appearance_revision']['design_reconstruction_plan']
                if mutation == 'no-appearance': changed_plan['layers'].pop(0)
                elif mutation == 'old-appearance': changed_plan['layers'][0]['params']['schema'] = 'reference_appearance_stack.v1'
                elif mutation == 'duplicate-appearance': changed_plan['layers'].append(copy.deepcopy(changed_plan['layers'][0]))
                elif mutation == 'missing-baseline': changed_plan.pop('source_baseline')
                else: changed_plan['layers'].append(copy.deepcopy(next(row for row in changed_plan['layers'] if row['kind'] == mutation)))
                rejected(lambda changed=changed: backend.reconstruct_source_design(run['run_id'], changed))
            require(len(backend.jobs.list()) == count and len(calls) == 1)
            def no_change(project_root, edited, output, **kwargs):
                return {'ok': True, 'no_change': True, 'cache_hit': True, 'receipt': {'schema': CONTRACT},
                    'design_reconstruction_plan': copy.deepcopy(edited['options']['appearance_revision']['design_reconstruction_plan'])}
            worker.prepare_for_workbench = no_change
            identifier = backend.reconstruct_source_design(run['run_id'], body)['job_id']
            require(wait_job(backend, identifier)['status'] == 'succeeded')
            require('没有新的可用改动' in backend.jobs.get(identifier)['message'])
            require(backend.job(identifier)['reconstruction_result']['design_reconstruction_plan'] == plan); keep(identifier)
            valid_result = success(root, {'options': {'appearance_revision': edits['appearance_revision']}}, base/'boundary-coordination-preparation', cancel=lambda: False, progress=lambda *a: None)
            invalid = [None, {'ok': False}]
            for mutation in ('contract', 'missing-hair', 'missing-cloth', 'duplicate-hair', 'source_sha256',
                    'candidate_sha256', 'canvas', 'other-layer', 'id', 'enabled', 'strength', 'no-change'):
                result = copy.deepcopy(valid_result); changed_plan = result['design_reconstruction_plan']
                if mutation == 'contract': result['receipt']['schema'] = 'wrong.v1'
                elif mutation.startswith('missing-'):
                    kind = KINDS[0 if mutation.endswith('hair') else 1]
                    changed_plan['layers'] = [row for row in changed_plan['layers'] if row['kind'] != kind]
                elif mutation == 'duplicate-hair': changed_plan['layers'].append(copy.deepcopy(changed_plan['layers'][-2]))
                elif mutation == 'canvas': changed_plan['canvas']['source_oriented_size'] = [64, 48]
                elif mutation in ('source_sha256', 'candidate_sha256'): changed_plan[mutation] = '0'*64
                elif mutation == 'other-layer': changed_plan['layers'][0]['strength'] = .17
                elif mutation == 'no-change':
                    result['no_change'] = True; changed_plan['layers'][-1]['label'] = 'not retained'
                else: changed_plan['layers'][-1][mutation] = {'id': 'changed', 'enabled': True, 'strength': .9}[mutation]
                invalid.append(result)
            for result in invalid:
                worker.prepare_for_workbench = lambda *args, result=result, **kwargs: copy.deepcopy(result)
                identifier = backend.reconstruct_source_design(run['run_id'], body)['job_id']
                require(wait_job(backend, identifier)['status'] == 'failed')
                require(not (backend.jobs.path(identifier)/'prepared-plan.json').exists()); keep(identifier)
            worker.prepare_for_workbench = success
            worker.validate_prepared_pair = lambda *args: (_ for _ in ()).throw(ValueError('stale pair'))
            identifier = backend.reconstruct_source_design(run['run_id'], body)['job_id']
            require(wait_job(backend, identifier)['status'] == 'failed')
            require(not (backend.jobs.path(identifier)/'prepared-plan.json').exists()); keep(identifier)
            worker.validate_prepared_pair = validate; entered.clear()
            def cancellable(*args, cancel, progress, **kwargs):
                progress('cloth_boundary_light', '肩袖 2/2', .55); entered.set()
                deadline = time.monotonic()+3
                while not cancel():
                    if time.monotonic() > deadline: raise AssertionError('Cancel not propagated')
                    time.sleep(.01)
                raise InterruptedError('cancelled')
            worker.prepare_for_workbench = cancellable
            identifier = backend.reconstruct_source_design(run['run_id'], body)['job_id']; require(entered.wait(2))
            require('发梢与肩袖受光' in backend.cancel_job(identifier)['job']['message'])
            require(wait_job(backend, identifier)['status'] == 'cancelled')
            require(not (backend.jobs.path(identifier)/'prepared-plan.json').exists()); keep(identifier)

        ui_root = base/'ui'; ui_run = project(ui_root)
        ui_plan = pair(ui_run['options']['appearance_revision']['design_reconstruction_plan'])
        for kind in KINDS:
            layer = next(row for row in ui_plan['layers'] if row['kind'] == kind)
            layer.update(enabled=False, strength=.35)
        ui_run['options']['appearance_revision']['design_reconstruction_plan'] = copy.deepcopy(ui_plan); write_run(ui_root, ui_run)
        window = Window(ui_root, 'http://127.0.0.1:1'); pool = Pool(); window.pool = pool
        window.load_source_run(ui_run); window.show(); app.processEvents()
        require(window.boundary_light_coordination_button.isEnabled())
        require(window.boundary_light_coordination_button.text() == '同时更新边界受光')
        require('同时更新边界受光' in window.hair_boundary_light_button.toolTip())
        original = copy.deepcopy(window.design_reconstruction_plan)
        def submit(identifier):
            window.boundary_light_coordination_button.click(); request = last(pool, 'design-reconstruct-job')
            require(request.body['preparation_kind'] == KIND and request.path.endswith('/reconstruct-design'))
            editing = {key: value for key, value in request.body.items() if key != 'preparation_kind'}
            require(editing == window._source_edit_payload())
            require(not any(button.isEnabled() for button in (window.boundary_light_coordination_button,
                window.hair_boundary_light_button, window.cloth_boundary_light_button)))
            request.signals.result.emit({'ok': True, 'async': True, 'job_id': identifier, 'job': {'preparation_kind': KIND}})
            app.processEvents(); return editing
        def finish(identifier, editing, *, status='succeeded', prepared=None, no_change=False, stale=False):
            job = {'kind': 'native_design_reconstruction', 'job_id': identifier, 'preparation_kind': KIND, 'status': status}
            receipt = {'run_id': ui_run['run_id'], 'source_sha256': ui_run['source']['sha256'],
                'candidate_sha256': ui_run['reference']['sha256'], 'input_sha256': 'stale' if stale else digest(editing),
                'preparation_kind': KIND, 'preparation_contract': CONTRACT, 'no_change': no_change,
                'design_reconstruction_plan': prepared or copy.deepcopy(ui_plan)}
            window.result('job-status', {'job': job, 'reconstruction_result': receipt}); app.processEvents()
        for status in ('succeeded', 'cancelled', 'failed'):
            retained_ui = (copy.deepcopy(window.design_reconstruction_plan), window.source_light_version, window._session_dirty)
            previews = sum(getattr(task, 'label', '').startswith('source-light-preview:') for task in pool.tasks)
            editing = submit('ui-'+status); finish('ui-'+status, editing, status=status, no_change=status == 'succeeded')
            require((window.design_reconstruction_plan, window.source_light_version, window._session_dirty) == retained_ui)
            require(sum(getattr(task, 'label', '').startswith('source-light-preview:') for task in pool.tasks) == previews)
            require(window.boundary_light_coordination_button.isEnabled())
            require('发梢与肩袖受光' in window.operation_state.text())
            if status == 'succeeded':
                require('没有新的可用改动' in window.operation_state.text() and '没有新的可用改动' in window.source_light_state.text())
        require(window.design_reconstruction_plan == original)
        editing = submit('ui-stale'); finish('ui-stale', editing, prepared=pair(ui_plan), stale=True)
        require(window.design_reconstruction_plan == original and '精修已有变化' in window.operation_state.text())
        window.boundary_light_coordination_button.click(); last(pool, 'design-reconstruct-job').signals.error.emit('worker unavailable'); app.processEvents()
        require(window.design_reconstruction_plan == original and window.boundary_light_coordination_button.isEnabled())
        for mutation in ('missing-appearance', *KINDS):
            unavailable = copy.deepcopy(ui_plan)
            if mutation == 'missing-appearance': unavailable['layers'].pop(0)
            else: unavailable['layers'].append(copy.deepcopy(next(row for row in unavailable['layers'] if row['kind'] == mutation)))
            window.load_design_reconstruction(unavailable)
            require(not window.boundary_light_coordination_button.isEnabled())
            count = len(pool.tasks); window.prepare_boundary_light_coordination(); require(len(pool.tasks) == count)
        window.load_design_reconstruction(ui_plan)
        editing = submit('ui-prepared'); prepared = pair(ui_plan)
        for row in prepared['layers']:
            if row['kind'] in KINDS: row['label'] += ' refreshed'
        immutable = copy.deepcopy(prepared); version = window.source_light_version
        finish('ui-prepared', editing, prepared=prepared)
        require(window.design_reconstruction_plan == prepared and window.design_reconstruction_prepared and window._session_dirty)
        require(window.source_light_version == version+1 and window.boundary_light_coordination_button.isEnabled())
        preview = next(task for task in reversed(pool.tasks) if getattr(task, 'label', '').startswith('source-light-preview:'))
        require(preview.body['appearance_revision']['design_reconstruction_plan'] == prepared)
        for kind in KINDS:
            require(sum(row['kind'] == kind for row in window.design_reconstruction_plan['layers']) == 1)
            row = next(row for row in window.design_reconstruction_plan['layers'] if row['kind'] == kind)
            require(row['enabled'] is False and row['strength'] == .35)
        require(prepared == immutable)
        window.save_source_lights(); save = last(pool, 'source-adjust')
        saved_plan = save.body['appearance_revision']['design_reconstruction_plan']
        require(saved_plan == prepared and set(save.body) == {'appearance_revision', 'manual_layers'})
        child = copy.deepcopy(ui_run); child.update(run_id='joint-child', revision=2, parent_run_id=ui_run['run_id'])
        child['options']['appearance_revision'] = copy.deepcopy(save.body['appearance_revision'])
        child['options']['manual_layers'] = copy.deepcopy(save.body['manual_layers'])
        child['render'] = {'run_id': child['run_id'], 'revision': 2, 'operator': {'render_mode': 'native_source_design'}}
        write_run(ui_root, child); save.signals.result.emit({'ok': True, 'run': child}); app.processEvents()
        require(window.source_run_id == 'joint-child' and not window._session_dirty)
        window.close(); app.processEvents()
        reopened = Window(ui_root, 'http://127.0.0.1:1'); reopened.pool = Pool(); reopened.load_source_run(child)
        require(reopened.design_reconstruction_plan == saved_plan and reopened.boundary_light_coordination_button.isEnabled())
        require('同时更新边界受光' in reopened.hair_boundary_light_button.toolTip())
        require(not any(task.path == '/api/one-click' or task.path.endswith(('/retry', '/prepare', '/recognize', '/repair-final'))
            for task in pool.tasks if hasattr(task, 'path')))
        reopened.close(); app.processEvents()
    return {'ok': True, 'checks': checks, 'atomic_two_stage_publication_api_gate': True,
        'current_contract_pair_scope_switches_doses_guarded': True, 'durable_kind_digest_and_receipt_restart_verified': True,
        'failure_cancel_stale_and_no_change_preserve_edit_parent_pointer': True,
        'ui_applies_both_once_with_one_preview_explicit_save_and_reopen': True,
        'synthetic_fixture_mock_coordinator_and_pair_validator': True,
        'model_inferences': 0, 'image_generation_submissions': 0, 'visual_acceptance': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path); args = parser.parse_args()
    report = run_smoke()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(report, ensure_ascii=False))
