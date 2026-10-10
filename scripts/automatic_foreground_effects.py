"""Bound one independent RGBA request, real effect QA, and an owned native child.

Only the root fresh-task/explicit-complete caller grants entry. Historical
inspection, imported preview, ordinary re-render and unavailable quality never
submit an image. The attempt marker survives failures and cannot reset a budget.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image
from effect_conditioning import METHOD, prompt_from_condition

from foreground_effects import (KIND, accept_layer, bind_result, binding, declaration,
    digest, identity, prepare_effects, produce_request, read, validate_layer, validate_text_only_graph, write)

SCHEMA = 'automatic_foreground_effects.v1'


def _path(root, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path(root)/path).resolve()


def _verify_generation(request, layer):
    """Transport success is evidenced independently of visual effect quality."""
    provenance = read(layer['assets']['provider_provenance']['path'])
    if (provenance.get('asset_role') != 'effect_layer'
            or provenance.get('asset', {}).get('sha256') != layer['assets']['generated_rgba']['sha256']
            or digest(layer['assets']['generated_rgba']['path']) != layer['assets']['generated_rgba']['sha256']
            or provenance.get('source', {}).get('sha256') != request['scene_context']['sha256']
            or provenance.get('prompt') != request['prompt']
            or provenance.get('alpha', {}).get('preserved') is not True):
        raise ValueError('Effect provider did not preserve this actual request/output/alpha')
    contract = provenance.get('provider_contract') or {}
    submitted = [row for row in contract.get('transport', {}).get('events', [])
                 if row.get('event') in {'provider_attempt', 'comfyui_submitted'}]
    if (len(submitted) != 1 or any(row.get('retrying') or row.get('event') == 'provider_retry'
            for row in contract.get('transport', {}).get('events', []))
            or not contract.get('response', {}).get('id')
            or contract.get('request', {}).get('prompt') != request['prompt']
            or contract.get('comfyui', {}).get('workflow_sha256') != request['workflow']['sha256']):
        raise ValueError('One actual frozen RGBA workflow submission must be evidenced without retries')
    api_input = contract['request']['api_input']
    if (contract['request'].get('input_sha256') != api_input.get('sha256')
            or digest(api_input['path']) != api_input['sha256']
            or (api_input.get('source') or api_input).get('sha256') != request['condition']['sha256']):
        raise ValueError('Actual effect generation condition derivative differs from selected reference')
    if (request.get('generation_method') != METHOD
            or request.get('photo_RGB_conditioned') is not False
            or digest(request['canvas_node']['path']) != request['canvas_node']['sha256']):
        raise ValueError('Actual effect generation has no verified text-only canvas implementation')
    if request['prompt'] != prompt_from_condition(request['effect_condition']):
        raise ValueError('Actual effect generation positive text contains non-whitelisted context')
    runtime_asset = layer['assets'].get('generation_runtime') or {}
    if not runtime_asset or digest(runtime_asset['path']) != runtime_asset['sha256']:
        raise ValueError('Actual effect service implementation evidence is missing or changed')
    runtime = read(runtime_asset['path'])
    if (runtime.get('loaded_node_sha256') != request['canvas_node']['sha256']
            or runtime.get('loaded_node_verified_via') != 'actual_service_object_info'):
        raise ValueError('Actual service did not load the frozen effect canvas implementation')
    with Image.open(api_input['path']) as supplied:
        if np.any(np.asarray(supplied) != 0):
            raise ValueError('Effect model API upload contains pixels instead of the empty canvas')
    graph = contract.get('comfyui', {}).get('request', {}).get('prompt')
    if not isinstance(graph, dict):
        raise ValueError('Actual submitted effect workflow body is missing')
    validate_text_only_graph(graph, request['canvas_latent']['size'], positive_prompt=request['prompt'])
    return {'response_id': contract['response']['id'], 'submission_count': 1,
        'actual_api_input': copy.deepcopy(api_input),
        'workflow_sha256': request['workflow']['sha256'],
        'provider_provenance_sha256': layer['assets']['provider_provenance']['sha256'],
        'output_sha256': layer['assets']['generated_rgba']['sha256'],
        'generation_method': request['generation_method'], 'source_photo_RGB_conditioned': False,
        'actual_empty_canvas_upload_verified': True, 'native_canvas_channels': 64,
        'actual_service_runtime': copy.deepcopy(runtime_asset),
        'transport_success_is_visual_acceptance': False}


def _save_staging_child(root, parent, plan, record):
    import single_source_workflow as workflow
    path, child = workflow._source_revision(root, parent['run_id'], snapshot=False, design_plan_override=plan)
    child.setdefault('options', {})['preserve_project_selection'] = True
    child['automatic_foreground_effects'] = copy.deepcopy(record)
    child['continuation_edit'] = {'schema_version': 'continuation_edit.v1',
        'coordinate_space': 'source_canvas_normalized', 'render_state': 'pending',
        'requested_at': workflow.now(), 'edit_count': 1,
        'edits': [{'kind': 'automatic_independent_effect'}],
        'source_sha256': parent['source']['sha256'], 'reference_sha256': parent['reference']['sha256']}
    child['cosplay_post_review'] = {'status': 'unavailable', 'actual_api_evidence_valid': False,
        'diagnostic_gate': 'unknown', 'automatic_accepted': False, 'visual_acceptance': False,
        'reason': 'Independent effect child requires final whole-frame/native diagnosis; effect QA is scoped.'}
    workflow.write_json(path, child)
    return child


def complete_effects_once(root, run, args, *, fresh_generation=False, bounded=True):
    """Return {handled, run, status}; caller persists the returned run manifest.

    Fresh generation or explicit complete-post authorization is established by
    the root caller, never inferred from a saved effect plan. Maximum one newly
    submitted RGBA and one actual text-vision request. Extra declarations remain
    deferred instead of causing hidden generation loops.
    """
    import single_source_workflow as workflow
    from generation_design import DESIGN_VERSION
    from automatic_foreground_effects_review import review_effect
    root = Path(root).resolve()
    if not bounded:
        raise ValueError('Automatic effect completion must be bounded to one request')
    if not fresh_generation:
        return {'handled': False, 'run': run, 'status': 'fresh_task_authorization_required'}
    design = run.get('generation_design') or {}
    plan = (run.get('options', {}).get('appearance_revision') or {}).get('design_reconstruction_plan') or {}
    if (design.get('schema_version') != DESIGN_VERSION or design.get('status') != 'ready'
            or run.get('status') != 'needs_visual_review' or not plan.get('enabled')
            or not run.get('auto_draft')):
        return {'handled': False, 'run': run, 'status': 'complete_native_draft_required'}
    specs, reason = declaration(run, plan)
    if not specs:
        return {'handled': False, 'run': run, 'status': 'skipped', 'reason': reason,
            'independent_effect_image_submissions': 0}
    if workflow.task_image_budget(run, args, 'independent_effect', default=1) < 1:
        ready = {row['id'].removeprefix('effect_'): row for row in plan.get('layers', [])
                 if row.get('kind') == KIND and row.get('provenance', {}).get('quality_status') == 'accepted_visual_review'}
        missing = [row['id'] for row in specs if row['id'] not in ready]
        for row in ready.values():
            validate_layer(row, run, plan, require_accepted=True)
        return {'handled': False, 'run': run, 'status': 'blocked_by_image_budget',
                'reason': 'This task forbids new independent effect images; declared increments remain unfulfilled.',
                'independent_effect_image_submissions': 0,
                'unfulfilled_effect_ids': missing} if missing else {
                    'handled': True, 'run': run, 'status': 'effects_already_prepared',
                    'independent_effect_image_submissions': 0}
    if not workflow.task_text_vision_allowed(run, args):
        return {'handled': False, 'run': run, 'status': 'text_vision_disabled',
                'reason': 'Actual effect review requires text/vision permission.',
                'independent_effect_image_submissions': 0,
                'unfulfilled_effect_ids': [row['id'] for row in specs]}
    if run.get('automatic_foreground_effects'):
        return {'handled': True, 'run': run, 'status': 'already_completed_or_attempted'}
    directory = root/workflow.RUNS_DIR/run['run_id']
    output = directory/'automatic-foreground-effects'
    output.mkdir(parents=True, exist_ok=True)
    marker_path = output/'attempt.json'
    retained = copy.deepcopy(run)
    if marker_path.exists():
        retained['automatic_foreground_effects'] = read(marker_path)
        return {'handled': True, 'run': retained, 'status': 'already_completed_or_attempted'}
    record = {'schema': SCHEMA, 'status': 'pending', 'attempt': 1,
        'initial_run_id': run['run_id'], 'result_run_id': run['run_id'],
        'generation_budget': 1, 'independent_effect_image_submissions': 0,
        'visual_review_budget': 1, 'visual_review_submissions': 0,
        'actual_visual_gate': 'unknown', 'automatic_accepted': False, 'human_final_acceptance': False,
        'source_person_RGB_used': False, 'physical_depth_established': False,
        'baked_back_effects_duplicated': False, 'upstream_T_B_S_refitted': False,
        'original_draft_retained': True, 'project_selection_changed': False,
        'declared_effect_count': len(specs), 'deferred_effect_ids': [],
        'frozen_effect_binding': binding(run, plan)}
    # Exclusive creation prevents concurrent coordinator entry, and reserves
    # the durable attempt before either a network image or a visual request.
    try:
        with marker_path.open('x', encoding='utf8') as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2)
    except FileExistsError:
        retained['automatic_foreground_effects'] = read(marker_path)
        return {'handled': True, 'run': retained, 'status': 'already_completed_or_attempted'}
    paths = [directory/'manifest.json', _path(root, run['source']['path']), _path(root, run['reference']['path'])]
    paths += [_path(root, run['artifacts'][key]) for key in
              ('auto_draft', 'render_master', 'render_scene_master') if run.get('artifacts', {}).get(key)]
    if (root/'project.json').is_file():
        paths.append(root/'project.json')
    preserved = {str(path): workflow.sha256(path) for path in paths}

    def persist():
        write(marker_path, record)

    def retained_result():
        retained['automatic_foreground_effects'] = copy.deepcopy(record)
        return {'handled': True, 'run': retained, 'status': record['status']}

    try:
        workflow.check_generation_cancel(args)
        record['status'] = 'preparing_effect_requests'
        persist()
        prepared = prepare_effects(root, run, plan, output_dir=output/'effect-assets')
        current = prepared['plan']
        pending_layers = [row for row in current.get('layers', []) if row.get('kind') == KIND
                          and row.get('provenance', {}).get('quality_status') != 'accepted_visual_review']
        accepted = [row for row in current.get('layers', []) if row.get('kind') == KIND
                    and row.get('provenance', {}).get('quality_status') == 'accepted_visual_review']
        for ready in accepted:
            validate_layer(ready, run, current, require_accepted=True)
        if accepted and not pending_layers and not prepared['requests']:
            record.update(status='effects_already_prepared', original_draft_retained=True,
                actual_visual_gate='pass', effect_generation_cached=True)
            persist()
            return retained_result()
        if pending_layers:
            layer = pending_layers[0]
            request = read(layer['assets']['contract']['path'])['generation_request']
            record['effect_generation_cached'] = True
        elif prepared['requests']:
            request = prepared['requests'][0]
            record['effect_generation_cached'] = False
            if (request['specification']['slot'] == 'front'
                    and request['specification']['occlusion'] != 'protect_face_hands'):
                raise ValueError('Automatic front effects require frozen source face/hand protection')
            workflow.check_generation_cancel(args)
            # Reserve before invoking the provider. Even a failed or unknown
            # transport outcome cannot reduce this count or grant a retry.
            record.update(status='generating_independent_rgba', independent_effect_image_submissions=1,
                          selected_request_identity=request['identity'])
            persist()
            if produce_request(request['request_path']) != 0:
                raise RuntimeError('Independent RGBA provider returned failure')
            workflow.check_generation_cancel(args)
            layer = bind_result(run, current, request, request['generated_image_path'],
                request['generated_provenance_path'], output/'selected-effect')
        else:
            record.update(status='effect_unavailable', reason='No generated effect or pending request')
            persist()
            return retained_result()
        record['selected_effect_id'] = request['specification']['id']
        if (request['specification']['slot'] == 'front'
                and request['specification']['occlusion'] != 'protect_face_hands'):
            raise ValueError('Automatic front effects require frozen source face/hand protection')
        record['selected_request_identity'] = request['identity']
        record['deferred_effect_ids'] = [row['id'] for row in specs if row['id'] != request['specification']['id']
            and not any(row['id'] == read(ready['assets']['contract']['path'])['specification']['id'] for ready in accepted)]
        record['actual_generation_evidence'] = _verify_generation(request, layer)
        validate_layer(layer, run, current, require_accepted=False)
        workflow.check_generation_cancel(args)
        record.update(status='reviewing_actual_rgba_and_composite', visual_review_submissions=1)
        persist()
        review = review_effect(run, current, layer, output/'visual-review',
            provider=getattr(args, 'provider', None), vision_model=getattr(args, 'vision_model', None) or 'gpt-6.1-sol',
            api_base=getattr(args, 'api_base', None), api_url=getattr(args, 'api_url', None),
            api_key_env=getattr(args, 'api_key_env', None))
        record.update(actual_visual_gate=review['status'], visual_review={'path': str(output/'visual-review/review.json'),
            'sha256': digest(output/'visual-review/review.json')})
        if review['status'] != 'pass':
            record.update(status='effect_visual_'+review['status'], original_draft_retained=True)
            persist()
            return retained_result()
        workflow.check_generation_cancel(args)
        layer = accept_layer(layer, run, current, review=review)
        current['layers'] = [row for row in current['layers'] if row.get('kind') != KIND]+accepted+[layer]
        current.setdefault('provenance', {})['independent_effects'] = {'status': 'prepared',
            'request_identities': [row['provenance']['request_identity'] for row in accepted+[layer]],
            'deferred_effect_ids': record['deferred_effect_ids'], 'baked_back_effects_duplicated': False,
            'source_person_RGB_used': False, 'physical_depth_established': False}
        record['status'] = 'saving_effect_child'
        persist()
        staging = _save_staging_child(root, run, current, record)
        record['candidate_run_id'] = staging['run_id']
        persist()
        workflow.check_generation_cancel(args)
        child = workflow.adjust_source_run(root, staging['run_id'])
        child_dir = (root/workflow.RUNS_DIR/child['run_id']).resolve()
        if (child['run_id'] != staging['run_id'] or child.get('parent_run_id') != run['run_id']
                or child.get('source', {}).get('sha256') != run['source']['sha256']
                or child.get('reference', {}).get('sha256') != run['reference']['sha256']
                or child.get('options', {}).get('preserve_project_selection') is not True
                or child.get('continuation_edit', {}).get('render_state') != 'saved'):
            raise ValueError('Automatic independent-effect child lineage/render/preserve contract failed')
        for key in ('auto_draft', 'render_master', 'render_scene_master'):
            path = _path(root, child.get('artifacts', {}).get(key, ''))
            if child_dir not in path.parents or not path.is_file():
                raise ValueError('Effect child is missing an owned native artifact')
        owned_plan = child['options']['appearance_revision']['design_reconstruction_plan']
        for row in owned_plan['layers']:
            if row.get('kind') == KIND and row.get('enabled'):
                validate_layer(row, child, owned_plan)
        record.update(status='independent_effect_child_saved', result_run_id=child['run_id'],
            original_draft_retained=False, child_native_artifacts_owned=True)
        persist()
        child['automatic_foreground_effects'] = copy.deepcopy(record)
        child['generation_budget'] = copy.deepcopy(run.get('generation_budget') or {})
        child['generation_budget'].update(origin_run_id=(run.get('generation_budget') or {}).get('origin_run_id', run['run_id']),
            independent_effect_image_submissions=record['independent_effect_image_submissions'],
            local_child_image_submissions=0)
        child['cosplay_post_review'] = {'status': 'unavailable', 'actual_api_evidence_valid': False,
            'diagnostic_gate': 'unknown', 'automatic_accepted': False, 'visual_acceptance': False,
            'reason': 'Saved effect child needs its own final review; effect-only QA cannot approve native final.'}
        workflow.write_json(child_dir/'manifest.json', child)
        return {'handled': True, 'run': child, 'status': record['status']}
    except workflow.WorkflowCancelled:
        record.update(status='effect_completion_cancelled', original_draft_retained=True)
        persist()
        return retained_result()
    except Exception as exc:
        record.update(status='effect_completion_unavailable', error_type=type(exc).__name__,
            error=str(exc)[:500], original_draft_retained=True)
        persist()
        return retained_result()
    finally:
        for path, expected in preserved.items():
            if workflow.sha256(Path(path)) != expected:
                raise ValueError('Effect completion changed saved parent/source/master/project selection')
