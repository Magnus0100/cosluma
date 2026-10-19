"""Bounded material-target completion for newly reviewed native Cos drafts.

Actual source/reference diagnosis, selected cached semantics and the complete
frozen appearance are required. Text vision qualifies targets; generation,
segmentation inference, parent pixel mutation and project selection are absent.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from cache_roots import cache_dir


SCHEMA = 'automatic_material_completion.v1'


def final_skin_regions(review):
    """Supported failed skin owners, excluding fabric and dedicated makeup."""
    supported={'face':'face','face_upper':'face','face_lower':'face',
        'left_arm':'left_arm','right_arm':'right_arm','left_hand':'left_hand',
        'right_hand':'right_hand','legs':'legs'}
    regions=set()
    for row in review.get('local_observations') or []:
        verdict=(row.get('checks') or {}).get('skin_materials') or {}
        if (row.get('region') in supported and verdict.get('status')=='fail'
                and verdict.get('correction_stage')=='subject_appearance'
                and verdict.get('cause')=='composition_defect'):
            regions.add(supported[row['region']])
    return sorted(regions)


def has_skin_reference_failure(run):
    review = run.get('cosplay_post_review') or {}
    return (review.get('status') == 'reviewed'
            and ((review.get('checks') or {}).get('skin_materials') or {}).get('reference', {}).get('status') == 'fail')


def has_material_failure(run):
    """Reference defects or a diagnosed final skin-colour failure, never unknown."""
    if has_skin_reference_failure(run):
        return True
    review = run.get('cosplay_post_review') or {}
    final = ((review.get('checks') or {}).get('skin_materials') or {}).get('final') or {}
    return bool(review.get('status') == 'reviewed' and final.get('status') == 'fail'
        and final.get('correction_stage') == 'subject_appearance'
        and final.get('cause') in ('transfer_incomplete', 'composition_defect', 'reference_defect'))


def _diagnosis_target(run):
    return 'reference' if has_skin_reference_failure(run) else 'final'


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _path(root, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path(root)/path).resolve()


def _frozen_stack(run):
    plan = (run.get('options', {}).get('appearance_revision') or {}).get('design_reconstruction_plan') or {}
    stacks = [row for row in plan.get('layers', []) if row.get('kind') == 'reference_appearance_stack']
    if len(stacks) != 1:
        return None
    stack = stacks[0]
    assets = stack.get('assets', {})
    if (not stack.get('enabled') or any(float(stack.get(name, 1)) != 1 for name in
            ('strength', 'colour_strength', 'lightness_strength')) or not all(isinstance(assets.get(name), dict)
            and assets[name].get('path') and assets[name].get('sha256')
            for name in ('automatic_delta', 'preparation_snapshot'))):
        return None
    return stack


def selected_cached_analysis(root, run, *, role, source_sha256):
    """Locate exactly one current identity; never use mtime or start inference."""
    from local_photo_analysis import analysis_identity
    if role not in ('source', 'candidate'):
        raise ValueError('Unsupported material analysis role')
    expected = analysis_identity(root, role=role,
        registry_path=run.get('options', {}).get('model_registry'))
    directory = cache_dir(root, 'local-analysis')/source_sha256[:16]
    matches = []
    for path in directory.glob('*/analysis.json'):
        record = _read(path)
        if (record.get('analysis_identity') == expected and record.get('analysis_role') == role
                and record.get('source', {}).get('sha256') == source_sha256):
            matches.append(path.resolve())
    if len(matches) != 1:
        raise ValueError('Current '+role+' material analysis is '+('ambiguous' if matches else 'not cached'))
    return matches[0]


def _diagnosis(root, run, directory):
    import reference_refinement
    import single_source_workflow as workflow
    review = run['cosplay_post_review']
    if review.get('schema') != 'cosplay_post_review.v2' or review.get('actual_api_evidence_valid') is not True:
        raise ValueError('Automatic material completion requires actual API diagnosis evidence')
    diagnosis_path = directory/'cosplay-post-review/review.json'
    saved = _read(diagnosis_path)
    for key in ('status', 'actual_api_evidence_valid', 'binding', 'checks', 'run_binding'):
        if saved.get(key) != review.get(key):
            raise ValueError('In-memory diagnosis differs from its saved actual evidence')
    root_binding = review.get('run_binding') or {}
    if (root_binding.get('run_id') != run['run_id'] or root_binding.get('revision') != run['revision']
            or root_binding.get('source_sha256') != run['source']['sha256']
            or root_binding.get('reference_sha256') != run['reference']['sha256']):
        raise ValueError('Material diagnosis does not belong to this RAW root/reference revision')
    images = review['binding']['images']
    source_display = _path(root, images['source']['path'])
    source = _path(root, run['source']['path'])
    reference = _path(root, run['reference']['path'])
    if (workflow.sha256(source) != run['source']['sha256']
            or workflow.sha256(reference) != run['reference']['sha256']
            or workflow.sha256(reference) != images['reference']['sha256']
            or workflow.sha256(source_display) != images['source']['sha256']):
        raise ValueError('Bound source display, root source or reference bytes changed')
    checked, _, _ = reference_refinement._bound_review(source_display, reference, diagnosis_path,
        run['generation_design'])
    target = _diagnosis_target(run)
    if checked['checks']['skin_materials'][target]['status'] != 'fail':
        raise ValueError('No actual skin/material diagnosis is available')
    if target == 'final':
        final = checked['checks']['skin_materials']['final']
        if (final.get('correction_stage') != 'subject_appearance'
                or final.get('cause') not in ('transfer_incomplete','composition_defect','reference_defect')):
            raise ValueError('Final material failure is not a supported subject-colour owner')
        if not final_skin_regions(checked):
            raise ValueError('Final material failure has no actual diagnosed skin-owner colour scope')
        for role in ('auto_draft','render_master','render_scene_master'):
            actual = _path(root, run['artifacts'][role])
            if workflow.sha256(actual) != root_binding.get('render_artifacts',{}).get(role):
                raise ValueError('Final material diagnosis is stale for the actual parent endpoint')
        if (workflow.sha256(_path(root,run['artifacts']['auto_draft'])) != root_binding.get('final_sha256')
                or root_binding.get('final_sha256') != images['final']['sha256']):
            raise ValueError('Final material diagnosis does not bind the actual current final pixels')
    return diagnosis_path, source_display, source, reference


def _save_staging_child(root, parent, plan, record):
    """Freeze preserve-selection before a render can update project state.

    Existing adjust_source_run resumes native pending children in place. The
    child owns the new plan first; the saved parent is never temporarily edited.
    """
    import single_source_workflow as workflow
    path, child = workflow._source_revision(root, parent['run_id'], snapshot=False,
                                           design_plan_override=plan)
    child.setdefault('options', {})['preserve_project_selection'] = True
    finishes = [(row.get('params') or {}).get('material_finish') for row in plan.get('layers', [])
                if row.get('kind') == 'reference_appearance_stack']
    if len(finishes) == 1 and isinstance(finishes[0], dict) and finishes[0].get('algorithm_profile'):
        from material_solver_profiles import solver_configuration
        profile = finishes[0]['algorithm_profile']
        solver_configuration(profile)
        child['options']['material_finish_profile'] = profile
    child['automatic_material_completion'] = copy.deepcopy(record)
    child['cosplay_post_completion'] = {
        'schema': 'cosplay_post_completion.v1', 'status': 'material_child_pending',
        'initial_run_id': parent['run_id'], 'result_run_id': child['run_id'],
        'correction_budget': 1, 'correction_submissions': 0,
        'automatic_accepted': False, 'visual_acceptance': False,
        'project_selection_changed_by_correction': False,
    }
    child['continuation_edit'] = {
        'schema_version': 'continuation_edit.v1', 'coordinate_space': 'source_canvas_normalized',
        'render_state': 'pending', 'requested_at': workflow.now(), 'edit_count': 1,
        'edits': [{'kind': 'automatic_qualified_material_finish'}],
        'source_sha256': parent['source']['sha256'], 'reference_sha256': parent['reference']['sha256'],
    }
    workflow.write_json(path, child)
    return child


def complete_materials_once(root, run, args, *, bounded=True):
    """Return a new child or retained draft plus a durable material outcome.

    This is the new-task hook, not a saved-project migration. Unsupported or
    unavailable skin evidence stops this material branch without a Qwen retry.
    One attempt marker prevents API repetition even after process interruption.
    """
    import material_anchor_review
    import reference_pipeline
    import single_source_workflow as workflow
    root = Path(root).resolve()
    if not bounded:
        raise ValueError('Automatic material completion must be bounded to one attempt')
    if not has_material_failure(run):
        return {'handled': False, 'run': run}
    target = _diagnosis_target(run)
    if target == 'reference' and (run.get('automatic_material_completion') or run.get('cosplay_post_completion')):
        return {'handled': True, 'run': run, 'status': 'already_completed_or_attempted'}
    review = run.get('cosplay_post_review') or {}
    if review.get('actual_api_evidence_valid') is not True or _frozen_stack(run) is None:
        return {'handled': True, 'run': run, 'status': 'material_evidence_or_frozen_stack_unavailable'}
    if (_frozen_stack(run).get('params') or {}).get('material_finish'):
        return {'handled': True, 'run': run, 'status': 'material_endpoint_already_finished',
                'reason':'A new qualification requires an upstream rebase; material colour is not stacked'}
    if target=='final' and not final_skin_regions(review):
        return {'handled':True,'run':run,'status':'material_colour_scope_unavailable',
                'reason':'Final skin/material criterion lacks a diagnosed skin owner; fabric and makeup are unsupported'}
    directory = root/workflow.RUNS_DIR/run['run_id']
    output = directory/'automatic-material-completion'
    output.mkdir(parents=True, exist_ok=True)
    marker_path = output/'attempt.json'
    if marker_path.exists():
        previous = _read(marker_path)
        result_id = previous.get('result_run_id')
        if (previous.get('status') == 'qualified_material_child_saved' and result_id
                and result_id != run['run_id']):
            _, saved_child = workflow.load_run(root, result_id)
            if (saved_child.get('parent_run_id') != run['run_id']
                    or saved_child.get('source', {}).get('sha256') != run['source']['sha256']
                    or saved_child.get('reference', {}).get('sha256') != run['reference']['sha256']
                    or saved_child.get('options', {}).get('preserve_project_selection') is not True):
                raise ValueError('Completed material child no longer matches its bounded source/reference lineage')
            return {'handled': True, 'run': saved_child, 'status': 'reopened_completed_material_child'}
        retained = copy.deepcopy(run)
        retained['automatic_material_completion'] = previous
        return {'handled': True, 'run': retained, 'status': 'already_attempted'}
    record = {'schema': SCHEMA, 'status': 'reserved', 'bounded': True, 'attempt': 1,
        'initial_run_id': run['run_id'], 'result_run_id': run['run_id'],
        'image_generation_submissions': 0, 'semantic_inference_submissions': 0,
        'text_vision_allowed': True, 'automatic_accepted': False, 'visual_acceptance': False,
        'original_draft_retained': True, 'project_selection_changed': False,
        'parent_manifest_modified': False,
        'diagnosis_target': _diagnosis_target(run),
        'remaining_reference_failures': [name for name, row in (review.get('checks') or {}).items()
            if name not in ('skin_materials', 'native_detail') and isinstance(row, dict)
            and row.get('reference', {}).get('status') == 'fail']}
    # Exclusive reservation, before any text request or native fit starts.
    with marker_path.open('x', encoding='utf8') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
    manifest_path = directory/'manifest.json'
    paths = [manifest_path, _path(root, run['source']['path']), _path(root, run['reference']['path'])]
    paths += [_path(root, run['artifacts'][name]) for name in
              ('auto_draft', 'render_master', 'render_scene_master') if run.get('artifacts', {}).get(name)]
    project_path = root/'project.json'
    if project_path.exists():
        paths.append(project_path)
    preserved = {}
    retained = copy.deepcopy(run)
    child = None
    def persist():
        workflow.write_json(marker_path, record)
    try:
        preserved = {str(path): workflow.sha256(path) for path in paths}
        workflow.check_generation_cancel(args)
        diagnosis_path, source_display, source, reference = _diagnosis(root, run, directory)
        stack = _frozen_stack(run)
        for name in ('automatic_delta', 'preparation_snapshot'):
            asset = stack['assets'][name]
            if workflow.sha256(_path(root, asset['path'])) != asset['sha256']:
                raise ValueError('Frozen parent appearance '+name+' bytes changed')
        source_analysis = selected_cached_analysis(root, run, role='source', source_sha256=run['source']['sha256'])
        reference_analysis = selected_cached_analysis(root, run, role='candidate', source_sha256=run['reference']['sha256'])
        record.update(status='qualifying_material_targets', source_analysis=str(source_analysis),
            reference_analysis=str(reference_analysis), diagnosis=str(diagnosis_path))
        persist()
        targets_directory = output/'material-targets'
        model = (getattr(args, 'vision_model', None)
                 or (run.get('character_profile') or {}).get('provenance', {}).get('model') or 'gpt-6.1-sol')
        material_anchor_review.main([
            '--root', str(root), '--source-display', str(source_display), '--original-source', str(source),
            '--source-analysis', str(source_analysis), '--reference', str(reference),
            '--reference-analysis', str(reference_analysis), '--diagnosis', str(diagnosis_path),
            '--diagnosis-target', record['diagnosis_target'], '--current-run', str(manifest_path),
            '--output', str(targets_directory), '--provider', getattr(args, 'provider', None) or 'ccswitch_sub2api',
            '--vision-model', model,
        ])
        inputs_path = targets_directory/'material-target-inputs.json'
        _, _, qualification, _ = material_anchor_review.load_inputs(inputs_path)
        record['material_inputs'] = str(inputs_path)
        if not qualification.get('targets', {}).get('skin'):
            record.update(status='material_targets_unknown_or_rejected', original_draft_retained=True)
            persist()
            retained['automatic_material_completion'] = copy.deepcopy(record)
            return {'handled': True, 'run': retained, 'status': record['status']}
        workflow.check_generation_cancel(args)
        record.update(status='preparing_qualified_material_finish', qualification_sha256=qualification['qualification_sha256'])
        persist()
        prepared = reference_pipeline.prepare_material_finish(root, run, inputs_path,
            output=output/'native-material-finish',
            cancel=lambda: bool(getattr(args, 'cancel_request_path', None)
                and Path(args.cancel_request_path).exists()))
        workflow.check_generation_cancel(args)
        record.update(status='saving_material_child', original_draft_retained=False)
        persist()
        staging = _save_staging_child(root, run, prepared['design_reconstruction_plan'], record)
        record['candidate_run_id'] = staging['run_id']
        persist()
        # Staging already materialized the replacement plan for its own owner.
        # Resume it without replacing that owned plan by a parent-bound copy.
        child = workflow.adjust_source_run(root, staging['run_id'])
        child_dir = (root/workflow.RUNS_DIR/child['run_id']).resolve()
        if (child['run_id'] != staging['run_id'] or child.get('parent_run_id') != run['run_id']
                or child.get('source', {}).get('sha256') != run['source']['sha256']
                or child.get('reference', {}).get('sha256') != run['reference']['sha256']
                or child.get('options', {}).get('preserve_project_selection') is not True
                or child.get('continuation_edit', {}).get('render_state') != 'saved'):
            raise ValueError('Automatic material child lineage/render/preserve-selection contract failed')
        for name in ('auto_draft', 'render_master', 'render_scene_master'):
            path = _path(root, child.get('artifacts', {}).get(name, ''))
            if child_dir not in path.parents or not path.is_file():
                raise ValueError('Automatic material child render is missing or parent-owned')
        record.update(status='qualified_material_child_saved', result_run_id=child['run_id'],
            original_draft_retained=False, upstream_T_B_S_refitted=False,
            independent_effect_requests=prepared.get('independent_effect_requests', []))
        persist()
        child['automatic_material_completion'] = copy.deepcopy(record)
        child['cosplay_post_completion'].update(status='qualified_material_child_saved', result_run_id=child['run_id'])
        # The parent API saw parent pixels. Its rebased paths/old verdicts are
        # not actual QA of the newly rendered child.
        child['parent_material_diagnosis'] = {'path': str(diagnosis_path), 'sha256': workflow.sha256(diagnosis_path)}
        child['cosplay_post_review'] = {'status': 'unavailable', 'actual_api_evidence_valid': False,
            'diagnostic_gate': 'unknown', 'automatic_accepted': False, 'visual_acceptance': False,
            'reason': 'New material child requires independent review; parent diagnosis remains separately bound.'}
        child['generation_budget'] = copy.deepcopy(run.get('generation_budget') or {})
        child['generation_budget'].update(origin_run_id=run['run_id'], local_child_image_submissions=0)
        workflow.write_json(child_dir/'manifest.json', child)
        return {'handled': True, 'run': child, 'status': record['status']}
    except Exception as exc:
        record.update(status='material_completion_unavailable', error_type=type(exc).__name__,
            error=str(exc)[:500], original_draft_retained=True)
        persist()
        retained['automatic_material_completion'] = copy.deepcopy(record)
        return {'handled': True, 'run': retained, 'status': record['status']}
    finally:
        for path, expected in preserved.items():
            if workflow.sha256(Path(path)) != expected:
                raise ValueError('Automatic material completion changed its saved parent/source/master/project selection')
