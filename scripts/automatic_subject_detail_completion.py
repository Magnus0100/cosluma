"""One current diagnosis-bound CPU subject texture repair and an owned native child."""
from __future__ import annotations

from runtime_resources import external_python_popen

import copy
import json
import os
from pathlib import Path
import subprocess
import time

from asset_identity import sha256_file

SCHEMA = 'automatic_subject_detail_completion.v1'
RECORD_KEY = 'automatic_subject_detail_completion'
ARTIFACTS = ('auto_draft', 'render_master', 'render_scene_master')
EXTERNAL_REVIEW_KEY = 'current_subject_detail_review_evidence'
EXTERNAL_REVIEW_SCHEMA = 'subject_detail_external_review_binding.v1'
DIAGNOSTIC_STATUSES = frozenset({'support_unmeasurable', 'detail_preserved_diagnostic_conflict',
    'structured_loss_confirmed', 'measurement_inconclusive', 'recovery_rejected'})


def _diagnostic_summary(receipt):
    """Persist measurements separately from the unchanged artistic verdict."""
    value = receipt.get('diagnostic_summary')
    if value is None:
        return None
    if (not isinstance(value, dict) or value.get('status') not in DIAGNOSTIC_STATUSES
            or value.get('visual_qa_unchanged') is not True
            or value.get('visual_acceptance', False) is not False
            or not isinstance(value.get('owners'), list)
            or not isinstance(value.get('next_action'), str)):
        raise ValueError('Subject diagnostic summary needs explicit measurement and unchanged visual QA facts')
    return copy.deepcopy(value)


def _result(run, status, reason=None):
    record = run.get(RECORD_KEY) or {}
    result = {'handled': True, 'run': run, 'status': status, 'reason': reason}
    if record.get('diagnostic_summary'):
        result['diagnostic_summary'] = copy.deepcopy(record['diagnostic_summary'])
    return result


def sha(path):
    """Return the stat-bound identity used by every production worker.

    Subject-detail completion is called after the native render and revisits
    the same source/child assets while assembling its receipt.  Reusing the
    shared identity cache avoids rereading large masters in this worker
    process while preserving full hashing on first access or after a stat
    change.
    """
    return sha256_file(path)


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


def prepare_subject_detail(root, run, output, *, cancel=None):
    """Launch only the selected CPU array worker, never an inference service."""
    from current_model_registry import load_registry
    from runtime_resources import code_path
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if cancel and cancel():
        raise InterruptedError('Subject detail cancelled; parent retained')
    python = Path(load_registry(root, registry_path=run.get('options', {}).get('model_registry'))['python'])
    if not python.is_file():
        raise ValueError('Selected subject CPU runtime is unavailable')
    snapshot = output/'subject-detail-parent.json'; write(snapshot, run)
    command = [str(python), '-B', '-X', 'utf8', str(code_path('native_subject_detail.py')),
               '--root', str(root), '--run', str(snapshot), '--output', str(output)]
    log = output/'subject-detail.log'
    with log.open('ab') as stream:
        process = external_python_popen(command, cwd=str(root), stdout=stream, stderr=subprocess.STDOUT,
            env={**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'},
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            while process.poll() is None:
                if cancel and cancel():
                    process.terminate()
                    try: process.wait(timeout=5)
                    except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)
                    raise InterruptedError('Subject detail cancelled; parent retained')
                time.sleep(.25)
        finally:
            if process.poll() is None:
                process.kill(); process.wait(timeout=5)
        if process.returncode:
            raise ValueError('Local subject CPU preparation failed: '+log.read_text(encoding='utf-8', errors='replace')[-2500:])
    if cancel and cancel():
        raise InterruptedError('Subject detail cancelled; parent retained')
    return read(output/'prepared-plan.json')


def _diagnosis(root, run):
    from native_subject_detail import qualified_review_windows, validate_diagnosed_plate
    from system_post_completion import _review_binding_current
    from cosplay_post_review import validate_saved_comprehensive_review
    windows = qualified_review_windows(run)
    if not windows:
        return None
    if not _review_binding_current(run):
        raise ValueError('Subject detail diagnosis is stale for the current native endpoint')
    review = run['cosplay_post_review']
    external = run.get(EXTERNAL_REVIEW_KEY)
    if external is not None:
        if (not isinstance(external, dict) or external.get('schema') != EXTERNAL_REVIEW_SCHEMA
                or not external.get('path') or not external.get('sha256')):
            raise ValueError('Subject external review needs its explicit read-only evidence binding')
        path = _path(root, external['path'])
        if sha(path) != external['sha256']:
            raise ValueError('Subject external review bytes differ from the explicit evidence hash')
    else:
        path = Path(root)/'outputs/runs'/run['run_id']/'cosplay-post-review/review.json'
    saved = read(path)
    contextual = copy.deepcopy(review)
    if external is not None and 'run_binding' not in saved:
        # Original review stays byte-for-byte intact. Only its consumer adds
        # the full endpoint binding that _review_binding_current just checked.
        contextual.pop('run_binding', None)
    if saved != contextual:
        raise ValueError('Subject diagnosis differs from its actual saved receipt')
    images, binding = review['binding']['images'], review['run_binding']
    for key in ARTIFACTS:
        if sha(_path(root, run['artifacts'][key])) != binding['render_artifacts'][key]:
            raise ValueError('Subject diagnosis does not bind all current native masters')
    if external is not None:
        expected_identity = {key: run[key] for key in ('run_id', 'revision')}
        expected_identity.update(source_sha256=run['source']['sha256'],
                                 reference_sha256=run['reference']['sha256'])
        selection = saved['native_selection']
        if selection.get('run_binding') != expected_identity:
            raise ValueError('Subject external native selection belongs to a different endpoint')
        native_size = run['source'].get('oriented_size')
        if native_size is None:
            native_size = run['options']['appearance_revision']['design_reconstruction_plan']['canvas']['source_oriented_size']
        final_path = _path(root, run['artifacts']['auto_draft'])
        native_final = selection['final']
        if (selection.get('native_size') != native_size
                or images['source']['oriented_size'] != native_size
                or images['final']['oriented_size'] != native_size
                or native_final.get('size') != native_size
                or _path(root, images['final']['path']) != final_path
                or _path(root, native_final['path']) != final_path
                or images['final']['sha256'] != binding['final_sha256']
                or native_final.get('sha256') != binding['final_sha256']):
            raise ValueError('Subject external final path, native size or pixels differ from the current endpoint')
    saved_images = saved['binding']['images']
    checked, proof = validate_saved_comprehensive_review(_path(root, saved_images['source']['path']),
        _path(root, saved_images['reference']['path']), path, run['generation_design'])
    if checked != saved or qualified_review_windows({**run, 'cosplay_post_review': checked}) != windows:
        raise ValueError('Subject evidence cannot be reopened from actual paired native boards')
    evidence = {'path': str(path.resolve()), 'sha256': sha(path)}
    if external is not None and evidence['sha256'] != external['sha256']:
        raise ValueError('Subject external review changed during validation')
    plate_binding = validate_diagnosed_plate(root, run)
    return {'evidence': evidence,
            'reviewed_windows': windows, 'final_sha256': binding['final_sha256'],
            'diagnostic_producer_run_id': binding['run_id'], 'consumer_run_id': run['run_id'],
            'actual_saved_evidence_verified': True, 'plate_binding': plate_binding,
            'external_review_consumed_read_only': external is not None,
            'evidence_assets': [evidence, *proof, plate_binding['metadata'], *plate_binding.get('evidence_assets', [])]}


def _save_child(root, parent, plan, record):
    import single_source_workflow as workflow
    path, child = workflow._source_revision(root, parent['run_id'], snapshot=False, design_plan_override=plan)
    child.pop(EXTERNAL_REVIEW_KEY, None)
    child.setdefault('options', {})['preserve_project_selection'] = True
    child[RECORD_KEY] = copy.deepcopy(record)
    child['cosplay_post_review'] = {'status': 'unavailable', 'diagnostic_gate': 'unknown',
        'actual_api_evidence_valid': False, 'automatic_accepted': False, 'visual_acceptance': False,
        'reason': 'New subject pixels require their own final whole/native diagnosis'}
    child['continuation_edit'] = {'schema_version': 'continuation_edit.v1',
        'coordinate_space': 'source_canvas_normalized', 'render_state': 'pending',
        'requested_at': workflow.now(), 'edit_count': 1,
        'edits': [{'kind': 'automatic_subject_detail'}],
        'source_sha256': parent['source']['sha256'], 'reference_sha256': parent['reference']['sha256']}
    workflow.write_json(path, child)
    return child


def complete_subject_detail_once(root, run, args, *, bounded=True):
    """Make at most one local attempt; saved parents and project selection stay exact."""
    import single_source_workflow as workflow
    root = Path(root).resolve()
    if not bounded:
        raise ValueError('Subject detail completion must be bounded to one attempt')
    if run.get(RECORD_KEY):
        return _result(run, 'already_attempted')
    try:
        diagnosis = _diagnosis(root, run)
    except (KeyError, OSError, ValueError, TypeError) as exc:
        return {'handled': True, 'run': run, 'status': 'subject_evidence_unsupported', 'reason': str(exc)}
    if diagnosis is None:
        return {'handled': False, 'run': run, 'status': 'not_required_by_current_diagnosis'}
    directory = root/'outputs/runs'/run['run_id']; output = directory/'automatic-subject-detail-completion'
    output.mkdir(parents=True, exist_ok=True); marker = output/'attempt.json'
    record = {'schema': SCHEMA, 'status': 'reserved', 'attempt': 1,
        'initial_run_id': run['run_id'], 'result_run_id': run['run_id'], 'diagnosis': diagnosis,
        'input_binding': {'run_id': run['run_id'], 'revision': run['revision'],
            'source_sha256': run['source']['sha256'], 'reference_sha256': run['reference']['sha256'],
            'final_sha256': diagnosis['final_sha256']},
        'image_generation_submissions': 0, 'text_vision_submissions': 0, 'model_inferences': 0,
        'automatic_accepted': False, 'visual_acceptance': False,
        'original_draft_retained': True, 'project_selection_changed': False}
    try:
        with marker.open('x', encoding='utf-8') as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
    except FileExistsError:
        # Reopening the immutable parent must not hide a completed no-change
        # diagnosis. This is history only; it never starts another worker.
        retained = copy.deepcopy(run)
        try:
            previous = read(marker)
            if (isinstance(previous, dict) and previous.get('schema') == SCHEMA
                    and previous.get('initial_run_id') == run['run_id']):
                retained[RECORD_KEY] = previous
        except (OSError, ValueError, TypeError):
            pass
        return _result(retained, 'already_attempted', 'persistent_local_attempt_marker')
    retained = copy.deepcopy(run); preserved = {}
    def persist(): write(marker, record)
    def retained_result():
        persist(); retained[RECORD_KEY] = copy.deepcopy(record)
        return _result(retained, record['status'], record.get('reason'))
    try:
        paths = [directory/'manifest.json', _path(root, run['source']['path']), _path(root, run['reference']['path'])]
        paths += [_path(root, run['artifacts'][key]) for key in ARTIFACTS]
        paths += [_path(root, row['path']) for row in diagnosis['evidence_assets']]
        if (root/'project.json').is_file(): paths.append(root/'project.json')
        preserved = {str(path): sha(path) for path in paths}
        workflow.check_generation_cancel(args)
        prepared = prepare_subject_detail(root, run, output, cancel=lambda: _cancelled(args))
        receipt = prepared['receipt']; qualification = prepared.get('qualification') or {}
        if any(receipt.get(key, 0) != 0 for key in (
                'model_inferences', 'image_generation_submissions', 'text_vision_submissions')):
            raise ValueError('Subject detail worker exceeded its local CPU-only request contract')
        record.update(qualification=qualification, repair_receipt=receipt,
                      reason=receipt.get('reason'), changed_pixels=(receipt.get('checks') or {}).get('changed_pixels', 0))
        summary = _diagnostic_summary(receipt)
        if summary is not None:
            if (receipt.get('source_sha256') != run['source']['sha256']
                    or receipt.get('reference_sha256') != run['reference']['sha256']
                    or receipt.get('diagnosis') != diagnosis['evidence']):
                raise ValueError('Subject diagnostic summary differs from the current diagnosis/source/reference')
            record['diagnostic_summary'] = summary
            record['numerical_gate_grants_visual_acceptance'] = False
        if not prepared.get('changed'):
            record['status'] = 'subject_evidence_unsupported' if not qualification.get('qualified') else 'subject_detail_rejected'
            return retained_result()
        if not qualification.get('qualified') or receipt.get('state') != 'prepared' or record['changed_pixels'] <= 0:
            raise ValueError('Subject worker claimed a change without qualified nonzero native detail')
        if (receipt.get('source_sha256') != run['source']['sha256']
                or receipt.get('reference_sha256') != run['reference']['sha256']
                or receipt.get('diagnosis') != diagnosis['evidence']):
            raise ValueError('Subject worker result differs from the current diagnosis/source/reference')
        if any(sha(path) != digest for path, digest in preserved.items()):
            raise ValueError('Subject worker changed a frozen parent asset before child creation')
        workflow.check_generation_cancel(args)
        child = _save_child(root, run, prepared['design_reconstruction_plan'], record)
        record['candidate_run_id'] = child['run_id']; persist()
        workflow.check_generation_cancel(args)
        rendered = workflow.adjust_source_run(root, child['run_id'])
        child_dir = root/'outputs/runs'/child['run_id']
        if (rendered.get('run_id') != child['run_id'] or rendered.get('parent_run_id') != run['run_id']
                or rendered.get('source', {}).get('sha256') != run['source']['sha256']
                or rendered.get('reference', {}).get('sha256') != run['reference']['sha256']
                or rendered.get('options', {}).get('preserve_project_selection') is not True
                or rendered.get('continuation_edit', {}).get('render_state') != 'saved'):
            raise ValueError('Subject child lineage/render/preserve contract failed')
        for key in ARTIFACTS:
            path = _path(root, rendered['artifacts'][key])
            if child_dir not in path.parents or not path.is_file():
                raise ValueError('Subject child is missing an owned native artifact')
        record.update(status='subject_detail_child_saved', result_run_id=rendered['run_id'],
            original_draft_retained=True, child_native_artifacts_owned=True)
        persist(); rendered[RECORD_KEY] = copy.deepcopy(record)
        workflow.write_json(child_dir/'manifest.json', rendered)
        return _result(rendered, record['status'], record.get('reason'))
    except (InterruptedError, workflow.WorkflowCancelled):
        record.update(status='subject_detail_rejected', reason='cancelled; parent_retained', cancelled=True)
    except Exception as exc:
        record.update(status='subject_detail_rejected', reason=str(exc)[:500], error_type=type(exc).__name__)
    finally:
        for path, digest in preserved.items():
            if sha(path) != digest:
                raise ValueError('Subject completion changed a source, parent master, diagnosis or project selection')
    return retained_result()
