"""Read-only multi-source release evidence, never a renderer or visual approval.

Supply saved manifest paths and explicit case labels. Header/hash checks are
storage evidence; strict saved QA replay is opt-in and makes no API requests.
Historical producers and missing evidence remain visible. No model, service,
portable operation, project selection or image mutation is performed here.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = 'cosplay_major_release_evaluation.v1'
CASE_SCHEMA = 'cosplay_major_release_cases.v1'
CHECKS = ('identity_makeup', 'concept_hierarchy', 'effects_space',
          'skin_materials', 'native_detail', 'scene_integration')
HUMAN_CHECKS = ('identity', 'face_eyes', 'hands_fingers', 'hair_halo',
                'outline_light', 'garment_texture', 'background_light_effects',
                'overall_impression')
MASTERS = ('auto_draft', 'render_master', 'render_scene_master')
MODES = ('local_portrait', 'existing_reference', 'create_reference')
BUDGET_KEYS = ('initial_reference', 'reference_correction', 'independent_effect')
MODEL_PROFILES = {'semantics_profile': 'sapiens2_seg_0.4b',
                  'matting_profile': 'vitmatte_small_distinctions',
                  'dinov3_profile': 'dinov3_vitl16',
                  'scene_profile': 'sam31_static_detector_v1'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf8')).hexdigest()


def resolve(value, root):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else Path(root) / path).resolve()


def descriptor(value, root):
    if not isinstance(value, dict) or not value.get('path'):
        return {'status': 'missing', 'expected_sha256': None}
    path = resolve(value['path'], root)
    result = {'path': str(path), 'expected_sha256': value.get('sha256')}
    if not path.is_file():
        return {**result, 'status': 'missing'}
    actual = sha(path)
    expected = value.get('sha256')
    return {**result, 'actual_sha256': actual, 'bytes': path.stat().st_size,
            'status': 'verified' if expected and actual == expected else
            'hash_mismatch' if expected else 'present_unbound'}


def native_header(path, key, size):
    """Inspect real storage types/ICC, without claiming complete pixel replay."""
    from PIL import Image, ImageCms
    import numpy as np
    import tifffile
    path = Path(path)
    if key == 'auto_draft':
        with Image.open(path) as image:
            actual_size, mode = list(image.size), image.mode
            profile = image.info.get('icc_profile')
            srgb = bool(profile and 'srgb' in ImageCms.getProfileDescription(
                ImageCms.ImageCmsProfile(io.BytesIO(profile))).casefold())
            if image.format != 'PNG' or mode != 'RGB' or actual_size != size or not srgb:
                raise ValueError('Preview must be native RGB PNG with sRGB ICC')
        return {'status': 'verified_header', 'size': actual_size, 'mode': mode,
                'actual_bit_depth': 8, 'sRGB_ICC': srgb}
    with tifffile.TiffFile(path) as file:
        page = file.pages[0]
        floating = key == 'render_scene_master'
        expected_dtype = np.dtype('float32' if floating else 'uint16')
        if (len(file.pages) != 1 or list(page.shape) != [size[1], size[0], 3]
                or page.dtype != expected_dtype or page.photometric != 2 or page.planarconfig != 1):
            raise ValueError('Master must be native contiguous RGB with actual required precision')
        profile = page.tags.get(34675)
        metadata = (file.shaped_metadata or ({},))[0] or {}
        if floating:
            if (profile is not None or metadata.get('working_domain') != 'display_linear_sRGB'
                    or metadata.get('display_transform_applied') is not True):
                raise ValueError('Float master must declare display-linear domain and no nonlinear ICC')
        else:
            if profile is None or 'srgb' not in ImageCms.getProfileDescription(
                    ImageCms.ImageCmsProfile(io.BytesIO(profile.value))).casefold():
                raise ValueError('True uint16 master must embed sRGB ICC')
        return {'status': 'verified_header', 'size': size, 'actual_dtype': str(page.dtype),
                'actual_bit_depth': 32 if floating else 16,
                'working_domain': metadata.get('working_domain'),
                'sRGB_ICC': not floating, 'whole_pixel_replay_performed': False}


def task_evidence(run, source_sha, reference_sha):
    options = run.get('options') or {}
    intent = run.get('task_intent') or options.get('task_intent') or {}
    plan = run.get('task_plan') or options.get('task_plan') or {}
    if not intent or not plan:
        return {'status': 'missing_current_task_contract', 'mode': intent.get('mode'),
                'historical_run_not_relabelled': True}
    errors = []
    mode = intent.get('mode')
    if (intent.get('schema_version') != 'cosplay_task_intent.v1' or mode not in MODES
            or plan.get('schema_version') != 'cosplay_task_plan.v1'
            or plan.get('intent') != intent or plan.get('mode') != mode or plan.get('ready') is not True):
        errors.append('Task intent/plan schema, route or readiness differs')
    if (run.get('task_intent') and options.get('task_intent')
            and run['task_intent'] != options['task_intent']):
        errors.append('Top-level and options intent differ')
    if (run.get('task_plan') and options.get('task_plan')
            and run['task_plan'] != options['task_plan']):
        errors.append('Top-level and options plans differ')
    inputs = plan.get('inputs') or {}
    if (inputs.get('source') or {}).get('sha256') != source_sha:
        errors.append('Task plan belongs to a different source')
    if mode == 'existing_reference' and (inputs.get('reference') or {}).get('sha256') != reference_sha:
        errors.append('Task plan belongs to a different reference')
    budget = plan.get('image_budget') or {}
    if (set(budget) != set(BUDGET_KEYS) or any(type(v) is not int or v < 0 for v in budget.values())
            or (mode != 'create_reference' and any(budget.values()))):
        errors.append('Task image budget does not match its route')
    plan_id = plan.get('plan_id')
    if not isinstance(plan_id, str) or len(plan_id) != 64:
        errors.append('Missing frozen plan identity')
    services = plan.get('services') or {}
    if mode == 'local_portrait' and (services.get('text_vision') is not False
                                    or services.get('image_generation') is not False):
        errors.append('Local portrait plan grants external services')
    for key in ('preflight_network_requests', 'preflight_model_inferences', 'preflight_jobs_created'):
        if plan.get(key) != 0:
            errors.append('No explicit zero-side-effect preflight: ' + key)
    counters = []
    for name in ('task_execution_receipt', 'system_post_completion', 'cosplay_post_completion',
                 'automatic_foreground_effects', 'automatic_receiver_detail_completion',
                 'automatic_material_completion', 'automatic_scene_completion'):
        receipt = run.get(name)
        if not isinstance(receipt, dict):
            continue
        owner = receipt.get('owner_run_id') or receipt.get('result_run_id') or receipt.get('run_id')
        if owner and owner != run.get('run_id'):
            continue
        for key in ('image_submissions', 'image_generation_submissions', 'generation_submissions',
                    'reference_image_submissions', 'independent_effect_image_submissions'):
            value = receipt.get(key)
            if type(value) is int:
                counters.append({'receipt': name, 'field': key, 'value': value})
                if mode != 'create_reference' and value > 0:
                    errors.append('Zero-image task has an observed image submission')
    return {'status': 'invalid' if errors else 'bound_record', 'mode': mode,
            'plan_id': plan_id, 'image_budget': budget, 'services': {
                key: services.get(key) for key in ('text_vision', 'image_generation', 'offline',
                'sent_content', 'text_vision_requests_upper_bound')}, 'errors': errors,
            'observed_submission_counters': counters,
            'complete_request_ledger_verified': False,
            'counter_scope': 'explicit current-owner top-level receipts; no ancestor sums',
            'shared_task': options.get('shared_task') or run.get('shared_task')}


def design_evidence(run, source_sha, reference_sha, task):
    options = run.get('options') or {}
    design = run.get('generation_design') or options.get('generation_design') or {}
    if task.get('mode') == 'local_portrait':
        return {'status': 'not_required_for_local_portrait'}
    binding = design.get('binding') or {}
    result = {'schema_version': design.get('schema_version'), 'status_recorded': design.get('status'),
              'design_origin': design.get('design_origin'),
              'source_sha256': binding.get('source_sha256') or design.get('source_sha256'),
              'reference_sha256': binding.get('reference_sha256') or design.get('reference_sha256')}
    if task.get('mode') == 'existing_reference':
        intent = run.get('task_intent') or options.get('task_intent')
        pair = (result['source_sha256'] == source_sha and result['reference_sha256'] == reference_sha
                and binding.get('task_intent_sha256') == digest(intent)
                and design.get('status') == 'ready' and design.get('design_origin') == 'existing_reference')
        board = descriptor(design.get('observation_board'), Path(run['_evaluation_root']))
        return {**result, 'status': 'paired_bound_record' if pair and board['status'] == 'verified'
                else 'missing_or_mismatched_paired_design', 'observation_board': board,
                'ready_is_not_visual_acceptance': True}
    return {**result, 'status': 'recorded_historical_design' if design else 'missing_design',
            'current_major_producer_qualified': False}


def model_evidence(run):
    profiles = []
    def visit(value):
        if isinstance(value, dict):
            selected = {key: value[key] for key in MODEL_PROFILES if key in value}
            if selected and selected not in profiles:
                profiles.append(selected)
            for key, item in value.items():
                if key not in {'cosplay_post_review', 'render'}:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    visit(run.get('options') or {})
    plan = run.get('task_plan') or (run.get('options') or {}).get('task_plan') or {}
    selected = (plan.get('capabilities') or {}).get('model_profiles')
    if selected and selected not in profiles:
        profiles.append(selected)
    conflicts = [p for p in profiles if any(p.get(k, v) != v for k, v in MODEL_PROFILES.items())]
    return {'status': 'profile_conflict' if conflicts else 'recorded_profiles' if profiles
            else 'missing_frozen_model_profiles', 'profiles': profiles,
            'registry_path_recorded': (run.get('options') or {}).get('model_registry'),
            'installed_registry_not_used_to_relabel_producer': True,
            'inference_performed_by_evaluator': False}


def quality_evidence(run, root, hashes, verify=False):
    qa = run.get('cosplay_post_review') or {}
    binding = qa.get('run_binding') or {}
    identity_match = bool(qa.get('status') == 'reviewed' and qa.get('actual_api_evidence_valid') is True
        and binding.get('run_id') == run.get('run_id') and binding.get('revision') == run.get('revision')
        and binding.get('source_sha256') == (run.get('source') or {}).get('sha256')
        and binding.get('reference_sha256') == (run.get('reference') or {}).get('sha256')
        and all(hashes.get(k) and (binding.get('render_artifacts') or {}).get(k) == hashes[k]
                for k in MASTERS) and binding.get('final_sha256') == hashes.get('auto_draft'))
    checks = {key: ((qa.get('checks') or {}).get(key, {}).get('final') or {}).get('status', 'unknown')
              for key in CHECKS}
    replay = {'status': 'not_requested', 'visual_evidence_verified': False}
    if verify and identity_match:
        try:
            from cosplay_post_review import validate_saved_comprehensive_review
            images = qa['binding']['images']
            review_path = root / 'outputs/runs' / run['run_id'] / 'cosplay-post-review/review.json'
            design = run.get('generation_design') or (run.get('options') or {}).get('generation_design') or {}
            rebuilt, proof = validate_saved_comprehensive_review(images['source']['path'],
                images['reference']['path'], review_path, design)
            if rebuilt != qa:
                raise ValueError('Manifest QA differs from independently rebuilt saved response evidence')
            replay = {'status': 'verified_saved_response_replay', 'visual_evidence_verified': True,
                      'bound_files_checked': len(proof), 'new_service_requests': 0}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            replay = {'status': 'failed', 'visual_evidence_verified': False, 'error': str(exc)[:500]}
    sampling = qa.get('native_sampling_coverage') or {}
    missing = sampling.get('unsupported_regions') or []
    unobserved = sampling.get('unobserved_pairs') or []
    checks_pass = identity_match and all(value == 'pass' for value in checks.values())
    return {'status': 'current_scoped_record' if identity_match else 'missing_or_stale',
            'six_final_checks': checks, 'six_pass_recorded': checks_pass,
            'strict_replay': replay, 'native_detail_scope': qa.get('native_detail_scope'),
            'observed_pair_ids': sampling.get('actually_observed_pairs') or [],
            'requested_pair_ids': sampling.get('requested_pairs') or [],
            'unobserved_pair_ids': unobserved, 'unsupported_regions': missing,
            'region_sampling': [{k: row.get(k) for k in ('region', 'sampled_fraction',
                'support_basis', 'native_region_completely_certified')} for row in sampling.get('regions', [])],
            'quality_state': 'failed_in_observed_scope' if identity_match and 'fail' in checks.values()
                else 'verified_scoped_pass_with_coverage_gaps' if checks_pass and replay['visual_evidence_verified']
                and (missing or unobserved) else 'verified_scoped_pass' if checks_pass and replay['visual_evidence_verified']
                else 'recorded_scoped_pass_unreplayed' if checks_pass else 'unknown',
            'whole_image_native_acceptance': False, 'human_final_approval_granted': False}


def human_evidence(run, root, hashes):
    path = (run.get('artifacts') or {}).get('visual_review')
    if not path:
        return {'status': 'missing', 'approved': False}
    path = resolve(path, root)
    if not path.is_file():
        return {'status': 'missing', 'approved': False}
    record = read(path)
    identity = record.get('run_id') == run.get('run_id') and record.get('revision') == run.get('revision')
    approved = (identity and record.get('status') == 'accepted' and bool(record.get('reviewer'))
                and all((record.get('checks') or {}).get(k) == 'pass' for k in HUMAN_CHECKS))
    pixel = record.get('render_artifacts') or {}
    pixel_bound = bool(approved and all(hashes.get(k) and pixel.get(k) == hashes[k] for k in MASTERS))
    return {'status': 'approved_bound' if pixel_bound else 'legacy_approval_not_pixel_bound'
            if approved else 'not_approved', 'approved_recorded': approved,
            'current_master_binding_verified': pixel_bound, 'review_sha256': sha(path),
            'reviewer': record.get('reviewer'), 'checks': record.get('checks') or {},
            'declared_scope': record.get('review_scope') or record.get('scope')}


def delivery_evidence(case, root, run, hashes):
    records = []
    for item in case.get('delivery_receipts') or []:
        bound = descriptor(item, root)
        if bound['status'] != 'verified':
            records.append(bound)
            continue
        receipt = read(bound['path'])
        identity = (receipt.get('schema') == 'saved_native_delivery_verification.v1'
            and receipt.get('run_id') == run.get('run_id') and receipt.get('revision') == run.get('revision')
            and receipt.get('source_sha256') == run['source']['sha256'])
        protected = receipt.get('preserved_inputs') or {}
        current_masters_bound = all(str(resolve(run['artifacts'][k], root)) in protected
            and protected[str(resolve(run['artifacts'][k], root))] == hashes.get(k) for k in MASTERS)
        stages = receipt.get('stages') or {}
        exported = (stages.get('tiff16_export') or {}).get('path')
        export_exact = bool(exported and Path(exported).is_file()
                            and sha(exported) == hashes.get('render_master'))
        imported = stages.get('portable') or {}
        imported_manifest = (Path(imported.get('imported_root', '')) / 'outputs/runs'
                             / str(imported.get('imported_run_id')) / 'manifest.json')
        imported_exact = False
        if imported_manifest.is_file():
            imported_run = read(imported_manifest)
            imported_root = imported_manifest.parents[3]
            imported_exact = all(Path(resolve((imported_run.get('artifacts') or {}).get(k, ''),
                imported_root)).is_file() and sha(resolve(imported_run['artifacts'][k], imported_root)) == hashes.get(k)
                for k in MASTERS)
        valid = (identity and current_masters_bound and receipt.get('status') == 'passed'
            and receipt.get('preserved_inputs_unchanged') is True
            and receipt.get('frozen_render_identity_unchanged') is True
            and receipt.get('external_api_requests') == 0 and export_exact and imported_exact
            and (stages.get('saved_reopen') or {}).get('status') == 'passed'
            and imported.get('status') == 'passed')
        records.append({**bound, 'status': 'bound_delivery_record' if valid else 'incomplete_or_stale',
            'current_masters_bound': current_masters_bound, 'actual_TIFF16_export_exact': export_exact,
            'actual_imported_masters_exact': imported_exact,
            'native_ROI_replay_recorded': imported.get('imported_native_ROI_exact') is True,
            'pixel_replay_performed_by_evaluator': False})
    return {'status': 'bound_delivery_record' if any(r['status'] == 'bound_delivery_record' for r in records)
            else 'missing_or_unverified', 'records': records,
            'portable_or_export_operations_performed': False}


def evaluate_case(case, verify=False):
    labels = {key: case.get(key) for key in ('case_id', 'source_group', 'split', 'label_basis')}
    if labels['split'] not in {None, 'baseline', 'development', 'held_out'}:
        raise ValueError('Unsupported explicit sample split')
    if not case.get('manifest'):
        source = descriptor({'path': case.get('source'), 'sha256': case.get('source_sha256')}, Path.cwd())
        reference = descriptor({'path': case.get('reference'), 'sha256': case.get('reference_sha256')}, Path.cwd())
        return {**labels, 'status': 'pending_saved_run', 'source': source, 'reference': reference,
                'workflow_state': 'not_run', 'quality_state': 'unknown',
                'held_out_label_is_caller_declaration': labels['split'] == 'held_out'}
    path = Path(case['manifest']).resolve()
    initial_sha = sha(path)
    run = read(path)
    root = path.parents[3] if path.parent.parent.name == 'runs' else path.parent
    run['_evaluation_root'] = str(root)
    source = descriptor(run.get('source'), root)
    reference = descriptor(run.get('reference'), root)
    expected = case.get('source_sha256')
    errors = []
    if source['status'] != 'verified' or (expected and source.get('actual_sha256') != expected):
        errors.append('Source bytes differ or lack explicit saved identity')
    if run.get('reference') and reference['status'] != 'verified':
        errors.append('Reference bytes differ or lack saved identity')
    if case.get('reference_sha256') and reference.get('actual_sha256') != case['reference_sha256']:
        errors.append('Reference differs from the explicitly registered case')
    plan = ((run.get('options') or {}).get('appearance_revision') or {}).get('design_reconstruction_plan') or {}
    size = ((plan.get('canvas') or {}).get('source_oriented_size')
            or ((run.get('options') or {}).get('source_only_plan') or {}).get('native_size')
            or (run.get('source') or {}).get('oriented_size'))
    masters, hashes = {}, {}
    for key in MASTERS:
        value = (run.get('artifacts') or {}).get(key)
        try:
            if not value or not size or len(size) != 2:
                raise ValueError('Missing saved native asset or canvas')
            target = resolve(value, root)
            hashes[key] = sha(target)
            masters[key] = {**native_header(target, key, size), 'path': str(target), 'sha256': hashes[key]}
        except (OSError, ValueError, IndexError) as exc:
            masters[key] = {'status': 'invalid_or_missing', 'error': str(exc)[:500]}
            errors.append('Native ' + key + ': ' + str(exc)[:160])
    task = task_evidence(run, source.get('actual_sha256'), reference.get('actual_sha256'))
    design = design_evidence(run, source.get('actual_sha256'), reference.get('actual_sha256'), task)
    quality = quality_evidence(run, root, hashes, verify)
    parent = {'status': 'not_declared'}
    if run.get('parent_run_id'):
        parent_path = path.parent.parent / run['parent_run_id'] / 'manifest.json'
        if parent_path.is_file():
            old = read(parent_path)
            matched = old.get('source', {}).get('sha256') == run.get('source', {}).get('sha256')
            parent = {'status': 'source_lineage_matched' if matched else 'source_lineage_mismatch',
                'run_id': old.get('run_id'), 'manifest_sha256': sha(parent_path),
                'parent_pre_execution_unchanged_proof': 'not_provided'}
            if not matched:
                errors.append('Parent source lineage differs')
        else:
            parent = {'status': 'missing_parent_manifest', 'run_id': run['parent_run_id']}
    if sha(path) != initial_sha:
        errors.append('Manifest changed while evaluation was in progress')
    if task['status'] == 'invalid':
        errors.extend(task['errors'])
    workflow = 'saved_native_assets_verified' if not errors else 'storage_or_contract_invalid'
    render = run.get('render') or {}
    version = run.get('software_version')
    current_producer = bool(isinstance(version, str) and version.startswith('0.4.')
        and render.get('software_version') == version and render.get('run_id') == run.get('run_id')
        and render.get('revision') == run.get('revision') and not run.get('saved_render_copy'))
    producer = {'software_version': version, 'render_software_version': render.get('software_version'),
        'recorded_identity_preserved': True, 'current_major_render_recorded': current_producer,
        'code_byte_identity_verified': False,
        'current_major_producer_qualified': current_producer and task['status'] == 'bound_record'}
    return {**labels, 'manifest': str(path), 'manifest_sha256': initial_sha,
        'run_id': run.get('run_id'), 'revision': run.get('revision'),
        'saved_status': run.get('status'), 'status': 'evaluated', 'errors': errors,
        'workflow_state': workflow, 'quality_state': quality['quality_state'],
        'source': source, 'reference': reference, 'task': task, 'design': design,
        'model_selection': model_evidence(run), 'producer': producer, 'native_masters': masters,
        'parent': parent, 'quality': quality, 'human': human_evidence(run, root, hashes),
        'delivery': delivery_evidence(case, root, run, hashes),
        'held_out_label_is_caller_declaration': labels['split'] == 'held_out'}


def evaluate(cases, minimum=3, verify=False):
    if type(minimum) is not int or minimum < 1:
        raise ValueError('Minimum source count must be a positive integer')
    results = []
    for case in cases:
        try:
            results.append(evaluate_case(case, verify))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            results.append({'case_id': case.get('case_id'), 'manifest': case.get('manifest'),
                'split': case.get('split'), 'source_group': case.get('source_group'),
                'status': 'unreadable', 'errors': [str(exc)[:500]], 'quality_state': 'unknown'})
    saved = [r for r in results if r['status'] == 'evaluated']
    sources = sorted({r['source']['actual_sha256'] for r in saved if r['source']['status'] == 'verified'})
    invariants = len(sources) >= minimum and all(not r.get('errors') for r in results)
    held = [r for r in results if r.get('split') == 'held_out']
    development_SHA = {r.get('source', {}).get('actual_sha256') for r in results
                       if r.get('split') == 'development'}
    held_development_overlap = sorted({r.get('source', {}).get('actual_sha256') for r in held
        if r.get('source', {}).get('actual_sha256') in development_SHA and r.get('source', {}).get('actual_sha256')})
    major = bool(saved) and all(r['task']['status'] == 'bound_record'
        and r['producer']['current_major_producer_qualified']
        and (r['model_selection']['status'] == 'recorded_profiles' or r['task']['mode'] == 'local_portrait')
        and r['quality']['strict_replay']['visual_evidence_verified']
        and r['quality']['six_pass_recorded'] and r['delivery']['status'] == 'bound_delivery_record'
        and r['design']['status'] in {'paired_bound_record', 'not_required_for_local_portrait'} for r in saved)
    held_pass = bool(held) and not held_development_overlap and all(r['status'] == 'evaluated'
        and r['quality']['strict_replay']['visual_evidence_verified'] and r['quality']['six_pass_recorded']
        and r['delivery']['status'] == 'bound_delivery_record' for r in held)
    return {'schema': SCHEMA, 'evaluation_completed': True,
        'evaluated_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'evaluator_identity': {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)},
        'storage_input_invariants_met': invariants, 'minimum_independent_source_SHA': minimum,
        'independent_saved_source_SHA_count': len(sources), 'source_SHA': sources,
        'source_groups': sorted({r['source_group'] for r in results if r.get('source_group')}),
        'group_and_split_basis': 'explicit caller labels; no filename, person or scene inference',
        'distinct_person_or_scene_count_verified': False,
        'release_evidence_ready': bool(invariants and major and held_pass),
        'human_final_work_approval_granted': False, 'artistic_generalization_claimed': False,
        'registered_held_out_case_count': len(held), 'held_out_evidence_complete': held_pass,
        'held_out_source_SHA_also_labelled_development': held_development_overlap,
        'held_out_independence_verified_by_tool': False,
        'workflow_state': 'storage_baseline_complete' if invariants else 'baseline_incomplete',
        'quality_state': 'release_evidence_complete_scoped' if invariants and major and held_pass else 'missing_current_release_evidence',
        'operations': {'network_requests': 0, 'model_inferences': 0, 'generated_images': 0,
            'portable_operations': 0, 'project_selection_changes': 0}, 'cases': results}


def load_cases(args):
    cases = []
    if args.case_list:
        payload = read(args.case_list)
        if payload.get('schema') != CASE_SCHEMA or not isinstance(payload.get('cases'), list):
            raise ValueError('Use the explicit current release case-list schema')
        for row in payload['cases']:
            if not isinstance(row, dict):
                raise ValueError('Every case must be an object')
            value = dict(row)
            for key in ('manifest', 'source', 'reference'):
                if value.get(key):
                    value[key] = str(resolve(value[key], args.case_list.parent))
            cases.append(value)
    cases.extend({'manifest': str(path.resolve())} for path in args.run)
    if not cases:
        raise ValueError('Provide --case-list or at least one explicit --run manifest')
    return cases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', type=Path, default=[], help='Saved manifest path; repeatable')
    parser.add_argument('--case-list', type=Path, help='Explicit source-group / split / evidence declarations')
    parser.add_argument('--output', required=True, type=Path, help='New evaluation JSON outside saved run storage')
    parser.add_argument('--require-min-sources', type=int, default=3)
    parser.add_argument('--verify-review-evidence', action='store_true', help='Pure saved response/pixel proof replay; no new QA')
    args = parser.parse_args(argv)
    cases = load_cases(args)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Use a new output; evaluation does not replace prior evidence: ' + str(output))
    for case in cases:
        if case.get('manifest') and output.is_relative_to(Path(case['manifest']).resolve().parent.parent):
            raise ValueError('Evaluation output must stay outside saved run storage')
        for key in ('manifest', 'source', 'reference'):
            if case.get(key) and output == Path(case[key]).resolve():
                raise ValueError('Evaluation cannot replace an input')
    report = evaluate(cases, args.require_min_sources, args.verify_review_evidence)
    report['case_list_identity'] = ({'path': str(args.case_list.resolve()), 'sha256': sha(args.case_list)}
                                  if args.case_list else None)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print(json.dumps({'evaluation_completed': True, 'output': str(output),
        'independent_saved_sources': report['independent_saved_source_SHA_count'],
        'storage_input_invariants_met': report['storage_input_invariants_met'],
        'release_evidence_ready': report['release_evidence_ready'],
        'human_final_work_approval_granted': False}, ensure_ascii=False))
    return 0 if report['storage_input_invariants_met'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
