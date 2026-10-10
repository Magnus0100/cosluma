"""Bounded diagnosis-led reference correction and fresh native composition.

Only a new complete generation task can enter this coordinator. A generated
correction is composed in a fresh staging child and must pass its bound native
review before selection. Children never request another image or recurse into correction.
"""
from __future__ import annotations

import copy
from pathlib import Path

from generation_design import DESIGN_VERSION


def _reference_analysis(root, run):
    """Choose only the current explicitly selected model identity, without inference."""
    from local_photo_analysis import analysis_identity
    from reference_refinement import _read
    try:
        expected = analysis_identity(root, role='candidate',
            registry_path=run.get('options', {}).get('model_registry'))
    except (OSError, RuntimeError, ValueError):
        return None
    reference_sha = run['reference']['sha256']
    directory = Path(root)/'outputs/cache/local-analysis'/reference_sha[:16]
    matching = []
    for path in directory.glob('*/analysis.json'):
        record = _read(path)
        if (record.get('analysis_identity') == expected and record.get('analysis_role') == 'candidate'
                and record.get('source', {}).get('sha256') == reference_sha):
            matching.append(path)
    if len(matching) > 1:
        raise ValueError('Ambiguous frozen reference semantic analysis')
    return matching[0] if matching else None


def continue_cosplay_post(root: Path, run: dict, args) -> dict:
    import single_source_workflow as workflow
    import reference_refinement as refinement

    if workflow.task_image_budget(run, args, 'reference_correction', default=1) < 1:
        return run
    if not workflow.task_text_vision_allowed(run, args):
        return run

    budget = int(getattr(args, 'cosplay_correction_budget', 0))
    design = run.get('generation_design') or {}
    if (budget != 1 or not (getattr(args, 'generate', False) or getattr(args, 'complete_post', False))
            or getattr(args, 'preserve_effective_prompt', False)
            or design.get('schema_version') != DESIGN_VERSION
            or design.get('status') != 'ready'
            or run.get('status') != 'needs_visual_review'):
        return run
    # Re-entering this task cannot reset a spent budget or replace its outcome.
    # Existing candidates may be separately re-evaluated without this entry.
    previous_completion = run.get('cosplay_post_completion') or {}
    if previous_completion and not (getattr(args, '_material_completion_checked', False)
            and previous_completion.get('status') in ('material_child_pending', 'qualified_material_child_saved')
            and previous_completion.get('correction_submissions', 0) == 0):
        return run
    review = run.get('cosplay_post_review') or {}
    if review.get('status') != 'reviewed':
        return run
    # Skin/material faults use qualified colour targets on the frozen source
    # endpoint. Unknown targets retain the draft; they cannot trigger another
    # whole-reference generation or random skin prompt refinement.
    from automatic_material_completion import complete_materials_once, has_skin_reference_failure
    if has_skin_reference_failure(run) and not getattr(args, '_material_completion_checked', False):
        material = complete_materials_once(root, run, args, bounded=True)
        if material['handled']:
            completed = material['run']
            if completed['run_id'] != run['run_id']:
                return completed
            # Unsupported skin targets cannot authorize skin generation. An
            # independent scene/effect failure may still spend the existing
            # single image budget with an explicit non-skin instruction filter.
            run = completed
            review = run.get('cosplay_post_review') or {}
    actions = [row['reference'] for name, row in (review.get('checks') or {}).items()
               if name != 'skin_materials'
               and isinstance(row, dict) and isinstance(row.get('reference'), dict)
               and row['reference'].get('status') == 'fail'
               and row['reference'].get('correction_stage') == 'reference_generation']
    if not actions:
        return run

    directory = root / workflow.RUNS_DIR / run['run_id']
    manifest = directory / 'manifest.json'
    output = directory / 'reference-material-correction'
    record = {'schema': 'cosplay_post_completion.v1', 'status': 'pending',
              'correction_budget': 1, 'correction_submissions': 0,
              'initial_run_id': run['run_id'], 'result_run_id': run['run_id'],
              'automatic_accepted': False, 'visual_acceptance': False,
              'project_selection_changed_by_correction': False,
              'edit_protocol': 'multi_reference_masked',
              'actual_generation_image_role': 'scene_reference_only',
              'source_identity_role': 'actual_identity_QA_and_fresh_native_source_composition'}
    run['cosplay_post_completion'] = record

    def persist():
        workflow.write_json(manifest, run)

    def retain_spent_budget(submission_count=0):
        # A reserved request remains spent if transport or receipt saving
        # raises. Never infer zero from a missing/unknown provider response.
        if (output/'attempt.json').exists():
            marker=refinement._read(output/'attempt.json')
            if marker.get('attempt') == 1 and marker.get('budget') == 1:
                submission_count=1
        record['correction_submissions']=max(record['correction_submissions'],submission_count)
        shared=run.setdefault('generation_budget', {})
        shared['correction_image_submissions']=record['correction_submissions']
        shared['total_image_submissions']=shared.get('image_submissions',0)+record['correction_submissions']

    # Inspecting/repairing a reference must not mutate any existing master.
    preserved = {str(path): workflow.sha256(Path(path)) for path in
                 (run['source']['path'], run['reference']['path'],
                  *[run['artifacts'][key] for key in
                    ('auto_draft', 'render_master', 'render_scene_master')])}
    project_path = root / 'project.json'
    project_sha = workflow.sha256(project_path) if project_path.exists() else None
    try:
        workflow.check_generation_cancel(args)
        record['status'] = 'correcting_reference'
        persist()
        images = review['binding']['images']
        profile = run.get('character_profile') or {}
        vision_options = dict(provider=args.provider,
            vision_model=getattr(args, 'vision_model', None) or
                (profile.get('provenance') or {}).get('model'),
            api_base=args.api_base, api_url=args.api_url, api_key_env=args.api_key_env)
        result = refinement.refine_reference_once(Path(images['source']['path']),
            Path(run['reference']['path']), directory/'cosplay-post-review/review.json',
            design, output, original_source=Path(run['source']['path']),
            sample=run['sample'], provider=args.provider, model=args.model,
            api_base=args.api_base, api_url=args.api_url, api_key_env=args.api_key_env,
            reference_analysis=_reference_analysis(root, run), mask_root=root,
            review_options=vision_options,
            correction_mode=refinement.MASKED_MODE, generation_only=True,
            correction_criteria=('identity_makeup', 'concept_hierarchy', 'effects_space', 'scene_integration'))
        retain_spent_budget(result.get('generation_attempts', 0))
        record.update(generation_status=result.get('status'),
                      generation_receipt=str(output/'refinement.json'),
                      evaluation=str(output/'evaluation.json'))
        if result.get('status') != 'generated':
            record.update(status='correction_unavailable', error=result.get('error'))
            persist()
            return run
        workflow.check_generation_cancel(args)
        child_args = copy.deepcopy(args)
        child_args.generate = False
        child_args.force = False
        child_args.local_only = False
        child_args.cosplay_correction_budget = 0
        child_args.cosplay_review_only = True
        child_args.preserve_project_selection = True
        child_args.run_id = run['run_id']+'-material-correction'
        child_args.parent_run_id = run['run_id']
        child_args.source = Path(run['source']['path'])
        child_args.reference = Path(result['candidate'])
        child_args.generation_id = result['generation_evidence']['response_id']
        provenance = refinement._read(result['provenance'])
        child_args.prompt = child_args.effective_prompt = provenance['prompt']
        child_args.preserve_effective_prompt = True
        child_args.generation_design = copy.deepcopy(design)
        child_args.native_source_retouch = copy.deepcopy((run.get('options') or {}).get('native_source_retouch'))
        # Rebuild every reference-dependent asset. Source editing controls
        # remain, but a prepared parent stack cannot follow a new reference.
        edits = copy.deepcopy(getattr(args, 'appearance_revision', None) or {})
        edits.pop('design_reconstruction_plan', None)
        child_args.appearance_revision = edits or None
        child_args.reference_pipeline_inputs = None
        child_args.variant_count = 1
        child_args.variant_styles = []
        record.update(status='composing_corrected_child', candidate_run_id=child_args.run_id)
        persist()
        parent_sha = workflow.sha256(manifest)
        child = workflow.create_run(child_args)
        if workflow.sha256(manifest) != parent_sha:
            raise ValueError('Corrected composition changed its saved parent')
        child['reference_refinement'] = copy.deepcopy(result)
        record['child_status'] = child.get('status')
        if child.get('status') != 'needs_visual_review':
            record.update(status='corrected_local_processing_failed',
                          original_draft_retained=True, child_error=child.get('error'))
            persist()
            return run
        child_directory=(root/workflow.RUNS_DIR/child['run_id']).resolve()
        if (child.get('parent_run_id') != run['run_id']
                or child.get('source',{}).get('sha256') != run['source']['sha256']
                or child.get('reference',{}).get('sha256') != workflow.sha256(child_args.reference)):
            raise ValueError('Corrected child source/reference lineage does not match')
        plan=child.get('options',{}).get('appearance_revision',{}).get('design_reconstruction_plan') or {}
        if (plan.get('source_sha256') != run['source']['sha256']
                or plan.get('candidate_sha256') != workflow.sha256(child_args.reference)):
            raise ValueError('Corrected child did not prepare its current source/reference pair')
        for key in ('auto_draft','render_master','render_scene_master'):
            path=Path(child.get('artifacts',{}).get(key,'')).resolve()
            if child_directory not in path.parents or not path.is_file():
                raise ValueError('Corrected child has missing or parent-owned render assets')
        # Review the actual native staging result. Generated pixels retain the
        # reference role; they cannot stand in for the composed final image.
        evaluation = refinement.evaluate_reference_refinement(output,
            candidate_review_path=child_directory/'cosplay-post-review/review.json')
        record['evaluation_status'] = evaluation['status']
        guard = refinement.validate_composition_eligibility(output,
            candidate=Path(result['candidate']), source_sha256=run['source']['sha256'],
            design=design)
        record['reference_gate'] = guard
        if not guard.get('eligible_for_composition'):
            record.update(status='correction_rejected', original_draft_retained=True,
                          rejected_staging_run_id=child['run_id'])
            persist()
            return run
        record.update(status='corrected_child_saved', result_run_id=child['run_id'])
        # Store the outcome before returning the separately reviewed child.
        persist()
        child['cosplay_post_completion'] = copy.deepcopy(record)
        child['generation_budget'] = copy.deepcopy(run.get('generation_budget') or {})
        child['generation_budget'].update(origin_run_id=run['run_id'],
                                          local_child_image_submissions=0)
        workflow.write_json(root/workflow.RUNS_DIR/child['run_id']/'manifest.json', child)
        return child
    except workflow.WorkflowCancelled:
        retain_spent_budget()
        record.update(status='correction_cancelled', original_draft_retained=True)
        persist()
        return run
    except Exception as exc:
        retain_spent_budget()
        record.update(status='correction_unavailable', error=str(exc), original_draft_retained=True)
        persist()
        return run
    finally:
        for path, digest in preserved.items():
            if workflow.sha256(Path(path)) != digest:
                raise ValueError('Reference correction changed a preserved source or master')
        if project_sha is not None and workflow.sha256(project_path) != project_sha:
            raise ValueError('Reference correction changed the project selection')
