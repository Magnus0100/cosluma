"""One evidence-bound CPU subject/environment response and an owned native child."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

SCHEMA = 'automatic_subject_environment_completion.v1'
RECORD_KEY = 'automatic_subject_environment_completion'
PRODUCER = 'subject_environment_response'
EXTERNAL_REVIEW_KEY = 'current_subject_environment_review_evidence'
EXTERNAL_REVIEW_SCHEMA = 'subject_environment_external_review_binding.v1'
ARTIFACTS = ('auto_draft', 'render_master', 'render_scene_master')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def _path(root, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path(root)/path).resolve()


def _cancelled(args):
    value = getattr(args, 'cancel_request_path', None)
    return bool(value and Path(value).exists())


def required_by_review(run):
    review = run.get('cosplay_post_review') or {}
    final = ((review.get('checks') or {}).get('scene_integration') or {}).get('final') or {}
    return bool(review.get('status') == 'reviewed' and review.get('actual_api_evidence_valid') is True
        and final.get('status') == 'fail' and final.get('correction_stage') == 'scene_composition'
        and final.get('cause') == 'retouch_missing')


def _diagnosis(root, run):
    """Reopen the saved comprehensive QA, including external original receipts."""
    from system_post_completion import _review_binding_current
    from cosplay_post_review import validate_saved_comprehensive_review
    from native_subject_detail import validate_diagnosed_plate
    if not required_by_review(run):
        return None
    if not _review_binding_current(run):
        raise ValueError('Subject environment diagnosis is stale for the current native endpoint')
    review = run['cosplay_post_review']; external = run.get(EXTERNAL_REVIEW_KEY)
    if external is not None:
        if (not isinstance(external, dict) or external.get('schema') != EXTERNAL_REVIEW_SCHEMA
                or not external.get('path') or not external.get('sha256')):
            raise ValueError('Subject environment external review needs an explicit read-only evidence binding')
        path = _path(root, external['path'])
        if sha(path) != external['sha256']:
            raise ValueError('Subject environment external review bytes differ from their evidence hash')
    else:
        path = Path(root)/'outputs/runs'/run['run_id']/'cosplay-post-review/review.json'
    saved = read(path); contextual = copy.deepcopy(review)
    if external is not None and 'run_binding' not in saved:
        contextual.pop('run_binding', None)
    if saved != contextual:
        raise ValueError('Subject environment diagnosis differs from the saved actual receipt')
    images, binding = review['binding']['images'], review['run_binding']
    for key in ARTIFACTS:
        if sha(_path(root, run['artifacts'][key])) != binding['render_artifacts'][key]:
            raise ValueError('Subject environment diagnosis does not bind every current native master')
    if binding.get('final_sha256') != binding['render_artifacts']['auto_draft']:
        raise ValueError('Subject environment final pixels differ from the current draft')
    if external is not None:
        identity = {key: run[key] for key in ('run_id', 'revision')}
        identity.update(source_sha256=run['source']['sha256'], reference_sha256=run['reference']['sha256'])
        selection = saved['native_selection']
        if selection.get('run_binding') != identity:
            raise ValueError('Subject environment external native selection belongs to another endpoint')
        size = run['source'].get('oriented_size') or run['options']['appearance_revision']['design_reconstruction_plan']['canvas']['source_oriented_size']
        final_path = _path(root, run['artifacts']['auto_draft']); selected_final = selection['final']
        if (selection.get('native_size') != size or images['source']['oriented_size'] != size
                or images['final']['oriented_size'] != size or selected_final.get('size') != size
                or _path(root, images['final']['path']) != final_path
                or _path(root, selected_final['path']) != final_path
                or images['final']['sha256'] != binding['final_sha256']
                or selected_final.get('sha256') != binding['final_sha256']):
            raise ValueError('Subject environment external final path, native size or pixels changed')
    checked, proof = validate_saved_comprehensive_review(_path(root, saved['binding']['images']['source']['path']),
        _path(root, saved['binding']['images']['reference']['path']), path, run['generation_design'])
    if checked != saved or not required_by_review({**run, 'cosplay_post_review': checked}):
        raise ValueError('Subject environment diagnosis cannot be reopened from actual visible boards')
    evidence = {'path': str(path.resolve()), 'sha256': sha(path)}
    if external is not None and evidence['sha256'] != external['sha256']:
        raise ValueError('Subject environment external review changed during validation')
    plate = validate_diagnosed_plate(root, run)
    return {'evidence': evidence, 'failed_criteria': ['scene_integration'],
        'final_sha256': binding['final_sha256'], 'diagnostic_producer_run_id': binding['run_id'],
        'consumer_run_id': run['run_id'], 'actual_saved_evidence_verified': True,
        'external_review_consumed_read_only': external is not None, 'plate_binding': plate,
        'evidence_assets': [evidence, *proof, plate['metadata'], *plate.get('evidence_assets', [])]}


def prepare_subject_environment(root, run, output, *, cancel=None):
    """Only the selected local CPU worker can prepare an environment response."""
    from current_model_registry import load_registry
    from runtime_resources import code_path
    root, output = Path(root).resolve(), Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    if cancel and cancel():
        raise InterruptedError('Subject environment cancelled; parent retained')
    python = Path(load_registry(root, registry_path=run.get('options', {}).get('model_registry'))['python'])
    if not python.is_file():
        raise ValueError('Selected subject environment CPU runtime is unavailable')
    snapshot = output/'subject-environment-parent.json'; write(snapshot, run)
    command = [str(python), '-B', '-X', 'utf8', str(code_path('subject_environment_worker.py')),
        '--root', str(root), '--run', str(snapshot), '--output', str(output)]
    log = output/'subject-environment.log'
    with log.open('ab') as stream:
        process = subprocess.Popen(command, cwd=str(root), stdout=stream, stderr=subprocess.STDOUT,
            env={**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'},
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            while process.poll() is None:
                if cancel and cancel():
                    process.terminate()
                    try: process.wait(timeout=5)
                    except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)
                    raise InterruptedError('Subject environment cancelled; parent retained')
                time.sleep(.25)
        finally:
            if process.poll() is None:
                process.kill(); process.wait(timeout=5)
        if process.returncode:
            raise ValueError('Local subject environment CPU preparation failed: '+log.read_text(encoding='utf-8', errors='replace')[-2500:])
    if cancel and cancel():
        raise InterruptedError('Subject environment cancelled; parent retained')
    return read(output/'prepared-subject-environment.json')


def _save_child(root, parent, plan, record):
    import single_source_workflow as workflow
    path, child = workflow._source_revision(root, parent['run_id'], snapshot=False, design_plan_override=plan)
    child.pop(EXTERNAL_REVIEW_KEY, None); child.pop('saved_render_copy', None)
    child.setdefault('options', {})['preserve_project_selection'] = True
    child[RECORD_KEY] = copy.deepcopy(record)
    child['cosplay_post_review'] = {'status': 'unavailable', 'diagnostic_gate': 'unknown',
        'actual_api_evidence_valid': False, 'automatic_accepted': False, 'visual_acceptance': False,
        'reason': 'New subject environment pixels require current whole/context/native QA'}
    child['continuation_edit'] = {'schema_version': 'continuation_edit.v1',
        'coordinate_space': 'source_canvas_normalized', 'render_state': 'pending',
        'requested_at': workflow.now(), 'edit_count': 1, 'edits': [{'kind': PRODUCER}],
        'source_sha256': parent['source']['sha256'], 'reference_sha256': parent['reference']['sha256']}
    workflow.write_json(path, child)
    return child


def complete_subject_environment_once(root, run, args, *, bounded=True):
    """One durable local attempt; parents, QA, budgets and selection stay exact."""
    import single_source_workflow as workflow
    root = Path(root).resolve()
    if not bounded:
        raise ValueError('Subject environment completion must be bounded to one attempt')
    if run.get(RECORD_KEY):
        return {'handled': True, 'run': run, 'status': 'already_attempted', 'reason': 'saved_local_attempt_not_repeated'}
    try:
        diagnosis = _diagnosis(root, run)
    except (KeyError, OSError, ValueError, TypeError) as exc:
        return {'handled': True, 'run': run, 'status': 'subject_environment_evidence_unsupported', 'reason': str(exc)}
    if diagnosis is None:
        return {'handled': False, 'run': run, 'status': 'not_required_by_current_diagnosis'}
    directory = root/'outputs/runs'/run['run_id']; output = directory/'automatic-subject-environment-completion'
    output.mkdir(parents=True, exist_ok=True); marker = output/'attempt.json'
    record = {'schema': SCHEMA, 'status': 'reserved', 'attempt': 1,
        'initial_run_id': run['run_id'], 'result_run_id': run['run_id'], 'diagnosis': diagnosis,
        'input_binding': {'run_id': run['run_id'], 'revision': run['revision'],
            'source_sha256': run['source']['sha256'], 'reference_sha256': run['reference']['sha256'],
            'final_sha256': diagnosis['final_sha256']}, 'image_generation_submissions': 0,
        'text_vision_submissions': 0, 'model_inferences': 0, 'automatic_accepted': False,
        'visual_acceptance': False, 'original_draft_retained': True, 'project_selection_changed': False,
        'scope': 'supported_source_hair_and_clothing_response; no_background_or_generated_subject'}
    try:
        with marker.open('x', encoding='utf-8') as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
    except FileExistsError:
        retained = copy.deepcopy(run)
        try:
            previous = read(marker)
            if isinstance(previous, dict) and previous.get('schema') == SCHEMA and previous.get('initial_run_id') == run['run_id']:
                retained[RECORD_KEY] = previous
        except (OSError, ValueError, TypeError):
            pass
        return {'handled': True, 'run': retained, 'status': 'already_attempted', 'reason': 'persistent_local_attempt_marker'}
    retained = copy.deepcopy(run); preserved = {}
    def persist(): write(marker, record)
    def retained_result():
        persist(); retained[RECORD_KEY] = copy.deepcopy(record)
        return {'handled': True, 'run': retained, 'status': record['status'], 'reason': record.get('reason')}
    try:
        paths = [directory/'manifest.json', _path(root, run['source']['path']), _path(root, run['reference']['path'])]
        paths += [_path(root, run['artifacts'][key]) for key in ARTIFACTS]
        paths += [_path(root, row['path']) for row in diagnosis['evidence_assets']]
        paths += [path for path in directory.glob('*/attempt.json') if path != marker]
        if (root/'project.json').is_file(): paths.append(root/'project.json')
        preserved = {str(path): sha(path) for path in paths}
        workflow.check_generation_cancel(args)
        prepared = prepare_subject_environment(root, run, output, cancel=lambda: _cancelled(args))
        receipt = prepared['receipt']
        if any(receipt.get(key, 0) != 0 for key in ('model_inferences', 'image_generation_submissions', 'text_vision_submissions')):
            raise ValueError('Subject environment worker exceeded its CPU-only request contract')
        if receipt.get('visual_acceptance', False) is not False or receipt.get('automatic_accepted', False) is not False:
            raise ValueError('Subject environment numerical preparation cannot grant artistic acceptance')
        record.update(repair_receipt=receipt, reason=prepared.get('reason') or receipt.get('reason'),
            changed_pixels=(receipt.get('checks') or {}).get('changed_pixels', 0))
        if not prepared.get('changed'):
            record['status'] = ('subject_environment_evidence_unsupported' if receipt.get('state') in
                {'unsupported', 'unknown'} else 'subject_environment_rejected')
            return retained_result()
        qualification = receipt.get('qualification') or {}; checks = receipt.get('checks') or {}
        if (receipt.get('state') != 'prepared' or record['changed_pixels'] <= 0
                or any(qualification.get(key) is not True for key in
                    ('qualified', 'actual_saved_evidence_verified', 'holdout_improved'))
                or any(checks.get(key) is not False for key in
                    ('alpha_changed', 'source_geometry_changed', 'reference_texture_imported', 'unknown_support_changed'))):
            raise ValueError('Subject environment worker claimed change without verified safe holdout improvement')
        if (receipt.get('source_sha256') != run['source']['sha256']
                or receipt.get('reference_sha256') != run['reference']['sha256']
                or receipt.get('diagnosis') != diagnosis['evidence']):
            raise ValueError('Subject environment response differs from the current diagnosis/source/reference')
        if any(sha(path) != digest for path, digest in preserved.items()):
            raise ValueError('Subject environment worker changed a frozen parent asset')
        workflow.check_generation_cancel(args)
        child = _save_child(root, run, prepared['design_reconstruction_plan'], record)
        record['candidate_run_id'] = child['run_id']; persist()
        workflow.check_generation_cancel(args)
        rendered = workflow.adjust_source_run(root, child['run_id']); child_dir = root/'outputs/runs'/child['run_id']
        if (rendered.get('run_id') != child['run_id'] or rendered.get('parent_run_id') != run['run_id']
                or rendered.get('source', {}).get('sha256') != run['source']['sha256']
                or rendered.get('reference', {}).get('sha256') != run['reference']['sha256']
                or rendered.get('options', {}).get('preserve_project_selection') is not True
                or rendered.get('continuation_edit', {}).get('render_state') != 'saved'):
            raise ValueError('Subject environment child lineage/render/preserve contract failed')
        if rendered.get('generation_budget') != run.get('generation_budget'):
            raise ValueError('Subject environment repair changed the saved image request budget')
        review = rendered.get('cosplay_post_review') or {}
        if review.get('actual_api_evidence_valid') is True or review.get('visual_acceptance') is True:
            raise ValueError('Subject environment child inherited old visual approval')
        for key in ARTIFACTS:
            path = _path(root, rendered['artifacts'][key])
            if child_dir not in path.parents or not path.is_file():
                raise ValueError('Subject environment child is missing an owned native artifact')
        record.update(status='subject_environment_child_saved', result_run_id=rendered['run_id'],
            original_draft_retained=True, child_native_artifacts_owned=True, requires_new_visual_QA=True)
        persist(); rendered[RECORD_KEY] = copy.deepcopy(record)
        workflow.write_json(child_dir/'manifest.json', rendered)
        return {'handled': True, 'run': rendered, 'status': record['status'], 'reason': record.get('reason')}
    except (InterruptedError, workflow.WorkflowCancelled):
        record.update(status='subject_environment_cancelled', reason='cancelled; parent_retained', cancelled=True)
    except Exception as exc:
        record.update(status='subject_environment_rejected', reason=str(exc)[:500], error_type=type(exc).__name__)
    finally:
        for path, digest in preserved.items():
            if sha(path) != digest:
                raise ValueError('Subject environment repair changed a source, parent master, QA or project selection')
    return retained_result()
