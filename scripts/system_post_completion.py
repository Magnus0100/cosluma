"""Complete one fresh Cos task in dependency order, with bounded stage receipts."""
from __future__ import annotations

import copy
import json
from pathlib import Path

SCHEMA = 'system_post_completion.v2'


def _eligible(run, args):
    from generation_design import DESIGN_VERSION
    from single_source_workflow import task_visual_review_allowed
    if not task_visual_review_allowed(run, args):
        return False
    design = run.get('generation_design') or {}
    task = run.get('task_intent') or (run.get('options') or {}).get('task_intent') or getattr(args, 'task_intent', {}) or {}
    return bool((task.get('mode') in ('existing_reference', 'create_reference')
        or int(getattr(args, 'cosplay_correction_budget', 0)) == 1
        or getattr(args, 'final_repair_only', False))
        and (getattr(args, 'generate', False) or getattr(args, 'complete_post', False))
        and not getattr(args, 'preserve_effective_prompt', False)
        and design.get('schema_version') == DESIGN_VERSION and design.get('status') == 'ready'
        and run.get('status') == 'needs_visual_review' and run.get('reference'))


def _review_current(run):
    review = run.get('cosplay_post_review') or {}
    return bool(review.get('status') == 'reviewed' and review.get('actual_api_evidence_valid') is True
                and _review_binding_current(run))


def _review_binding_current(run):
    """Pixel identity also binds an incomplete attempt; it never grants repair."""
    review = run.get('cosplay_post_review') or {}
    binding = review.get('run_binding') or {}
    matches = (binding.get('run_id') == run.get('run_id') and binding.get('revision') == run.get('revision')
        and binding.get('source_sha256') == run['source']['sha256']
        and binding.get('reference_sha256') == run['reference']['sha256'])
    if not matches:
        return False
    import single_source_workflow as workflow
    try:
        for role in ('source', 'reference'):
            role_path = Path(run[role]['path'])
            if not role_path.is_absolute():
                role_path = Path(run.get('root') or '.')/role_path
            if workflow.sha256(role_path) != run[role]['sha256']:
                return False
        final = ((review.get('binding') or {}).get('images') or {})['final']
        final_path = Path(final['path'])
        if not final_path.is_absolute():
            final_path = Path(run.get('root') or '.')/final_path
        if (not binding.get('final_sha256') or binding['final_sha256'] != final.get('sha256')
                or workflow.sha256(final_path) != binding['final_sha256']):
            return False
        actual = {key:workflow.sha256(Path(value) if Path(value).is_absolute() else
                  Path(run.get('root') or '.')/value) for key,value in (run.get('artifacts') or {}).items()
                  if key in ('auto_draft', 'render_master', 'render_scene_master') and value}
        return bool(actual and actual == binding.get('render_artifacts'))
    except (KeyError, OSError, TypeError, ValueError):
        return False


def complete_once(root, run, args):
    """Material correction, scene correction, independent effects, then final QA.

    Ordinary opens/renders cannot enter. Each producer owns its exclusive
    attempt marker. This orchestration also reserves before entry so an
    interrupted completion cannot silently repeat review or image requests.
    New pixels always become an independently owned child, never a parent edit.
    """
    import single_source_workflow as workflow
    from automatic_material_completion import complete_materials_once, has_skin_reference_failure
    from cosplay_post_completion import continue_cosplay_post
    from automatic_foreground_effects import complete_effects_once

    root = Path(root).resolve()
    if not _eligible(run, args):
        return run
    directory = root/workflow.RUNS_DIR/run['run_id']/'system-post-completion'
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory/'attempt.json'
    if marker.exists():
        previous = workflow.read_json(marker)
        result_id = previous.get('result_run_id')
        if previous.get('status') in ('completed', 'completed_with_findings') and result_id:
            _, saved = workflow.load_run(root, result_id)
            if saved['source']['sha256'] != run['source']['sha256']:
                raise ValueError('System completion result belongs to a different source')
            if _review_current(saved):
                return saved
            stale = copy.deepcopy(saved)
            stale['system_post_completion'] = {**copy.deepcopy(previous),
                'status': 'stale_draft_retained', 'actual_review_current': False,
                'preserved_completed_receipt': str(marker), 'network_requests_sent_on_reopen': 0}
            stale['cosplay_post_review'] = {**copy.deepcopy(saved.get('cosplay_post_review') or {}),
                'status': 'stale', 'actual_evidence_matches_current_pixels': False,
                'reason': 'Saved pixels no longer match the completed review; opening cannot grant another request budget.'}
            return stale
        # An explicit new complete-post task creates a new child and a new
        # marker. Reopening this attempt does not grant another network budget.
        return run
    final_only = bool(getattr(args, 'final_repair_only', False))
    reference_budget = workflow.task_image_budget(run, args, 'reference_correction', default=1)
    effect_budget = workflow.task_image_budget(run, args, 'independent_effect', default=1)
    text_vision_allowed = workflow.task_text_vision_allowed(run, args)
    record = {'schema': SCHEMA, 'status': 'reserved', 'attempt': 1,
        'initial_run_id': run['run_id'], 'result_run_id': run['run_id'],
        'source_sha256': run['source']['sha256'], 'stages': [],
        'stage_order': ['native_source_retouch', 'material_finish', 'reference_scene', 'independent_effects', 'final_local_repair', 'final_review'],
        'reference_image_budget': reference_budget,
        'independent_effect_image_budget': effect_budget,
        'final_repair_only': final_only,
        'local_repair_round_budget': 1, 'local_repair_rounds': 0,
        'automatic_visual_acceptance': False, 'project_selection_changed': False}
    try:
        with marker.open('x', encoding='utf-8') as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2)
    except FileExistsError:
        return run
    current = run
    project = root/'project.json'
    project_sha = workflow.sha256(project) if project.is_file() else None

    def persist():
        record['result_run_id'] = current['run_id']
        workflow.write_json(marker, record)

    def stage(name, status, **evidence):
        record['stages'].append({'id': name, 'status': status, 'run_id': current['run_id'], **evidence})
        persist()

    def save_current():
        workflow.write_json(root/workflow.RUNS_DIR/current['run_id']/'manifest.json', current)

    attempted_endpoints = set()

    def diagnose_current_once():
        if _review_current(current):
            return
        artifacts = current.get('artifacts') or {}
        endpoint = (current['run_id'], current.get('revision'),
            tuple((key, workflow.sha256(Path(artifacts[key]) if Path(artifacts[key]).is_absolute() else root/artifacts[key]))
                  for key in ('auto_draft', 'render_master', 'render_scene_master') if artifacts.get(key)))
        # A partial result from create_run already attempted these exact pixels.
        # Unknown stays open; another automatic stage cannot resend successful
        # boards merely because one response failed. A changed child gets QA.
        if endpoint in attempted_endpoints or _review_binding_current(current):
            return
        attempted_endpoints.add(endpoint)
        workflow.diagnose_cosplay_post(root, current, args)
        save_current()

    try:
        workflow.check_generation_cancel(args)
        # A new completion child may contain inherited observations. Refresh
        # them before *any* producer can use the diagnosis as authority.
        if not _review_current(current):
            diagnose_current_once()
        retouch = current.get('automatic_native_retouch') or {}
        binding = (current.get('options') or {}).get('native_source_retouch') or {}
        stage('native_source_retouch', retouch.get('status', 'frozen_source_retouch_reused' if binding else 'not_prepared'),
              approved_patches=binding.get('approved_patches', 0), source_geometry_changed=False)
        record['status'] = 'completing_materials'
        persist()
        if has_skin_reference_failure(current) and text_vision_allowed:
            result = complete_materials_once(root, current, args, bounded=True)
            current = result['run']
            stage('material_finish', result.get('status', 'not_handled'))
            save_current()
            if not _review_current(current):
                diagnose_current_once()
        else:
            stage('material_finish', 'text_vision_disabled' if not text_vision_allowed else 'not_required_by_current_diagnosis' if _review_current(current)
                  else 'diagnosis_unavailable', image_submissions=0)
        workflow.check_generation_cancel(args)
        record['status'] = 'completing_reference_scene'
        persist()
        scene_args = copy.deepcopy(args)
        scene_args._material_completion_checked = True
        # The source-stage binding follows a scene child without another
        # source recognizer call; its new owner is established by create_run.
        scene_args.native_source_retouch = copy.deepcopy((current.get('options') or {}).get('native_source_retouch'))
        if not final_only and reference_budget:
            current = continue_cosplay_post(root, current, scene_args)
        correction = current.get('cosplay_post_completion') or {}
        stage('reference_scene', 'not_requested_local_repair' if final_only else 'image_budget_zero' if not reference_budget else
              correction.get('status', 'not_required_by_current_diagnosis' if _review_current(current)
              else 'diagnosis_unavailable'), image_submissions=0 if not reference_budget else correction.get('correction_submissions', 0))
        save_current()
        workflow.check_generation_cancel(args)
        record['status'] = 'completing_independent_effects'
        persist()
        result = ({'run': current, 'status': 'not_requested_local_repair',
                   'reason': 'Final local repair cannot submit a new independent effect image.'}
                  if final_only else complete_effects_once(root, current, args, fresh_generation=True, bounded=True))
        current = result['run']
        effects = current.get('automatic_foreground_effects') or {}
        record['unfulfilled_independent_effects'] = result.get('unfulfilled_effect_ids', [])
        stage('independent_effects', result['status'], reason=result.get('reason'),
              image_submissions=effects.get('independent_effect_image_submissions', 0),
              actual_visual_gate=effects.get('actual_visual_gate', 'unknown'))
        save_current()
        workflow.check_generation_cancel(args)
        record['status'] = 'reviewing_final_revision'
        persist()
        if not _review_current(current):
            diagnose_current_once()
        from cosplay_repair_plan import plan_final_repairs
        repair_plan = plan_final_repairs(current, current_review=_review_current(current))
        record['final_repair_plan'] = repair_plan
        outcomes = []
        # Final-only defects do not enter reference generation. Scene verdicts
        # first audit actual assets; the scene producer may return no qualified
        # correction. Each local producer has its own durable single attempt.
        for producer in ('qualified_material_finish', 'background_consistency',
                         'subject_environment_response', 'native_subject_detail', 'native_receiver_detail'):
            workflow.check_generation_cancel(args)
            eligible = any(row['producer'] == producer and row['pixel_repair_qualified']
                           for row in repair_plan['tasks'])
            if not eligible:
                continue
            record.update(status='repairing_final_locally', local_repair_rounds=1)
            persist()
            if producer == 'qualified_material_finish':
                from automatic_material_completion import has_material_failure
                if not has_material_failure(current) or not text_vision_allowed:
                    continue
                result = complete_materials_once(root, current, args, bounded=True)
            elif producer == 'background_consistency':
                from automatic_scene_completion import complete_scene_once
                result = complete_scene_once(root, current, args, bounded=True)
            elif producer == 'subject_environment_response':
                from automatic_subject_environment_completion import complete_subject_environment_once
                result = complete_subject_environment_once(root, current, args, bounded=True)
            elif producer == 'native_subject_detail':
                from automatic_subject_detail_completion import complete_subject_detail_once
                result = complete_subject_detail_once(root, current, args, bounded=True)
            else:
                from automatic_receiver_detail_completion import complete_receiver_detail_once
                result = complete_receiver_detail_once(root, current, args, bounded=True)
            previous_id = current['run_id']
            current = result['run']
            outcomes.append({'producer': producer, 'status': result.get('status', 'unavailable'),
                             'reason': result.get('reason'), 'input_run_id': previous_id,
                             'result_run_id': current['run_id'],
                             'new_owned_revision': current['run_id'] != previous_id})
            save_current()
            if current['run_id'] != previous_id or not _review_current(current):
                diagnose_current_once()
            repair_plan = plan_final_repairs(current, current_review=_review_current(current))
        stage('final_local_repair', 'checked_with_outcomes' if outcomes else 'not_qualified_by_current_diagnosis',
              outcomes=outcomes, image_submissions=0,
              repair_plan=repair_plan, automatic_visual_acceptance=False)
        record['remaining_repair_plan'] = repair_plan
        workflow.check_generation_cancel(args)
        review = current.get('cosplay_post_review') or {}
        failures = [key for key, row in (review.get('checks') or {}).items()
                    if row.get('final', {}).get('status') == 'fail']
        unknowns = [key for key, row in (review.get('checks') or {}).items()
                    if row.get('final', {}).get('status') == 'unknown']
        stage('final_review', review.get('status', 'unavailable'), diagnostic_gate=review.get('diagnostic_gate', 'unknown'),
              actual_api_evidence_valid=review.get('actual_api_evidence_valid') is True,
              failed_checks=failures, unknown_checks=unknowns, visual_acceptance=False)
        findings_open = bool(failures or unknowns or record['unfulfilled_independent_effects'] or not _review_current(current))
        record.update(status='completed_with_findings' if findings_open else 'completed',
                      final_diagnostic_gate=review.get('diagnostic_gate', 'unknown'),
                      failed_checks=failures, unknown_checks=unknowns,
                      all_stages_checked_once=True,
                      execution_state='finished',
                      visual_state='findings_open' if findings_open else 'ready_for_human_review')
    except workflow.WorkflowCancelled:
        record.update(status='cancelled_draft_retained', all_stages_checked_once=False, execution_state='cancelled')
    except Exception as exc:
        record.update(status='unavailable_draft_retained', error_type=type(exc).__name__, error=str(exc),
                      all_stages_checked_once=False, execution_state='unavailable')
    finally:
        if project_sha is not None and workflow.sha256(project) != project_sha:
            raise ValueError('System post completion changed project selection')
        persist()
        current['system_post_completion'] = copy.deepcopy(record)
        budget = current.setdefault('generation_budget', {})
        effects = current.get('automatic_foreground_effects') or {}
        budget['reference_correction_budget'] = reference_budget
        budget['independent_effect_image_budget'] = effect_budget
        budget['independent_effect_image_submissions'] = effects.get('independent_effect_image_submissions', 0)
        if 'regional_refinement_image_submissions' in budget:
            # Recompute from disjoint roles, never add regional submissions to
            # a total that may already include the same frozen region IDs.
            budget['total_image_submissions'] = (budget.get('image_submissions', 0)
                + budget.get('correction_image_submissions', 0)
                + budget['regional_refinement_image_submissions'])
        budget['total_image_submissions_including_effects'] = (budget.get('total_image_submissions', budget.get('image_submissions', 0))
            + budget['independent_effect_image_submissions'])
        save_current()
    return current
