"""One diagnosis-led local background audit/repair and an owned native child."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

SCHEMA = 'automatic_scene_completion.v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def prepare_scene_consistency(root, run, output, *, diagnosis=None, cancel=None):
    """Run all numeric audit/production in the selected CPU model runtime."""
    from current_model_registry import load_registry
    from runtime_resources import code_path
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    python = Path(load_registry(root, registry_path=run.get('options', {}).get('model_registry'))['python'])
    if not python.is_file():
        raise ValueError('Selected scene CPU runtime is unavailable')
    snapshot = output/'scene-consistency-parent.json'; write(snapshot, run)
    command = [str(python), '-B', '-X', 'utf8', str(code_path('scene_completion_worker.py')),
               '--root', str(root), '--run', str(snapshot), '--output', str(output)]
    if diagnosis is not None:
        evidence = output/'scene-diagnosis.json'; write(evidence, diagnosis)
        command += ['--diagnosis', str(evidence)]
    log = output/'scene-consistency.log'
    with log.open('ab') as stream:
        process = subprocess.Popen(command, cwd=str(root), stdout=stream, stderr=subprocess.STDOUT,
            env={**os.environ, 'OPENBLAS_NUM_THREADS':'1', 'OMP_NUM_THREADS':'1'},
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            while process.poll() is None:
                if cancel and cancel():
                    process.terminate()
                    try: process.wait(timeout=5)
                    except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)
                    raise InterruptedError('Scene consistency cancelled; parent retained')
                time.sleep(.25)
        finally:
            if process.poll() is None:
                process.kill(); process.wait(timeout=5)
        if process.returncode:
            raise ValueError('Local scene CPU preparation failed: '+log.read_text(encoding='utf-8', errors='replace')[-2500:])
    return read(output/'prepared-scene-consistency.json')


def _diagnosis(root, run):
    import cosplay_post_review as visual
    review = run.get('cosplay_post_review') or {}
    checks = review.get('checks') or {}
    failures = [name for name in ('effects_space','scene_integration')
        if checks.get(name,{}).get('final',{}).get('status') == 'fail'
        and checks[name]['final'].get('correction_stage') == 'scene_composition'
        and checks[name]['final'].get('cause') in ('transfer_incomplete','composition_defect')]
    if not failures:
        return None
    if review.get('status') != 'reviewed' or review.get('actual_api_evidence_valid') is not True:
        raise ValueError('Local scene repair requires actual final diagnosis')
    directory=Path(root)/'outputs/runs'/run['run_id']/'cosplay-post-review'
    saved=read(directory/'review.json')
    if saved != review:
        raise ValueError('Scene diagnosis differs from its actual saved receipt')
    images=review['binding']['images']; binding=review['run_binding']
    for key in ('auto_draft','render_master','render_scene_master'):
        path=Path(run['artifacts'][key]); path=path if path.is_absolute() else Path(root)/path
        if sha(path) != binding['render_artifacts'][key]:
            raise ValueError('Final scene diagnosis is stale for this native endpoint')
    if (binding.get('run_id') != run['run_id'] or binding.get('revision') != run['revision']
            or binding['source_sha256'] != run['source']['sha256']
            or binding['reference_sha256'] != run['reference']['sha256']
            or binding['final_sha256'] != images['final']['sha256']
            or binding['render_artifacts']['auto_draft'] != binding['final_sha256']):
        raise ValueError('Scene diagnosis input identity changed')
    checked,_=visual.validate_saved_comprehensive_review(Path(images['source']['path']),
        Path(images['reference']['path']),directory/'review.json',run['generation_design'])
    if checked != review:
        raise ValueError('Scene diagnosis cannot be reopened from actual visible evidence')
    return {'evidence':{'path':str((directory/'review.json').resolve()),'sha256':sha(directory/'review.json')},
        'failed_criteria':failures,'final_sha256':binding['final_sha256'],
        'diagnostic_producer_run_id':binding['run_id'],'consumer_run_id':run['run_id']}


def complete_scene_once(root, run, args, *, bounded=True):
    import single_source_workflow as workflow
    root=Path(root).resolve()
    if not bounded:
        raise ValueError('Local scene repair must be bounded to one attempt')
    try:
        diagnosis=_diagnosis(root,run)
    except (KeyError,OSError,ValueError,TypeError) as exc:
        return {'handled':True,'run':run,'status':'scene_evidence_unsupported','reason':str(exc)}
    if diagnosis is None:
        return {'handled':False,'run':run,'status':'not_required_by_current_diagnosis','reason':'no_final_scene_failure'}
    if run.get('automatic_scene_completion'):
        return {'handled':True,'run':run,'status':'already_attempted','reason':'saved_local_attempt_not_repeated'}
    directory=root/'outputs/runs'/run['run_id'];output=directory/'automatic-scene-completion'
    output.mkdir(parents=True,exist_ok=True);marker=output/'attempt.json'
    record={'schema':SCHEMA,'status':'reserved','attempt':1,'initial_run_id':run['run_id'],
        'result_run_id':run['run_id'],'diagnosis':diagnosis,'image_generation_submissions':0,
        'text_vision_submissions':0,'automatic_accepted':False,'visual_acceptance':False,
        'original_draft_retained':True,'project_selection_changed':False}
    try:
        with marker.open('x',encoding='utf-8') as stream:
            import json
            json.dump(record,stream,ensure_ascii=False,indent=2)
    except FileExistsError:
        return {'handled':True,'run':run,'status':'already_attempted','reason':'persistent_local_attempt_marker'}
    retained=copy.deepcopy(run);preserved={}
    def persist():write(marker,record)
    try:
        paths=[directory/'manifest.json',Path(run['source']['path']),Path(run['reference']['path'])]
        paths += [Path(run['artifacts'][key]) for key in ('auto_draft','render_master','render_scene_master')]
        if (root/'project.json').is_file():paths.append(root/'project.json')
        preserved={str(path):sha(path) for path in paths}
        workflow.check_generation_cancel(args)
        prepared=prepare_scene_consistency(root,run,output,diagnosis=diagnosis,
            cancel=lambda:bool(getattr(args,'cancel_request_path',None) and Path(args.cancel_request_path).exists()))
        audit=prepared['audit']
        record.update(audit=audit['receipt_asset'],audit_status=audit['status'],reason=audit.get('reason'))
        persist()
        if audit['status'] != 'repair_eligible':
            record['status']='audited_no_repair' if audit['status']=='audited_no_repair' else 'scene_evidence_unsupported'
            retained['automatic_scene_completion']=copy.deepcopy(record);persist()
            return {'handled':True,'run':retained,'status':record['status'],'reason':record.get('reason')}
        record['repair_receipt']=str((output/'consistent-background/background-consistency.json').resolve())
        if not prepared['changed']:
            record.update(status='scene_repair_rejected',reason='qualified_observation_error_not_improved')
            retained['automatic_scene_completion']=copy.deepcopy(record);persist()
            return {'handled':True,'run':retained,'status':record['status'],'reason':record['reason']}
        workflow.check_generation_cancel(args)
        path,child=workflow._source_revision(root,run['run_id'],snapshot=False,
                                             design_plan_override=prepared['design_reconstruction_plan'])
        child.setdefault('options',{})['preserve_project_selection']=True
        child['automatic_scene_completion']=copy.deepcopy(record)
        child['cosplay_post_review']={'status':'unavailable','diagnostic_gate':'unknown',
            'actual_api_evidence_valid':False,'visual_acceptance':False,
            'reason':'New background pixels require current final whole/native diagnosis'}
        child['continuation_edit']={'schema_version':'continuation_edit.v1','coordinate_space':'source_canvas_normalized',
            'render_state':'pending','requested_at':workflow.now(),'edit_count':1,
            'edits':[{'kind':'automatic_background_consistency'}],
            'source_sha256':run['source']['sha256'],'reference_sha256':run['reference']['sha256']}
        workflow.write_json(path,child);record['candidate_run_id']=child['run_id'];persist()
        child=workflow.adjust_source_run(root,child['run_id'])
        if child.get('continuation_edit',{}).get('render_state') != 'saved':
            raise ValueError('Local background child did not save its native render')
        record.update(status='scene_consistency_child_saved',result_run_id=child['run_id'],
            reason='SR_data_consistency_improved; effect_readability_requires_actual_review',
            original_draft_retained=False,relative_RMSE_improvement=prepared['receipt']['relative_RMSE_improvement'])
        persist();child['automatic_scene_completion']=copy.deepcopy(record)
        workflow.write_json(root/'outputs/runs'/child['run_id']/'manifest.json',child)
        return {'handled':True,'run':child,'status':record['status'],'reason':record['reason']}
    except (InterruptedError,workflow.WorkflowCancelled):
        record.update(status='scene_repair_cancelled',reason='cancelled; parent_retained')
    except Exception as exc:
        record.update(status='scene_repair_rejected',reason=str(exc)[:500],error_type=type(exc).__name__)
    finally:
        for path,digest in preserved.items():
            if sha(path) != digest:
                raise ValueError('Local scene repair changed a source, parent master or project selection')
    persist();retained['automatic_scene_completion']=copy.deepcopy(record)
    return {'handled':True,'run':retained,'status':record['status'],'reason':record.get('reason')}
