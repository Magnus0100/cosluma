"""One user task contract and a side-effect-free, content-bound execution plan."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from asset_identity import sha256_file
from appearance_read_scope import saved_parent_reads

INTENT_SCHEMA = 'cosplay_task_intent.v1'
PLAN_SCHEMA = 'cosplay_task_plan.v2'
MODES = {'local_portrait': '本地原片精修', 'existing_reference': '已有参考完整后期',
         'create_reference': '创作参考并完成后期'}
REQUEST_FIELDS = frozenset(('source', 'reference', 'preset', 'provider', 'model', 'vision_model',
    'api_base', 'api_url', 'api_key_env', 'character_name', 'character_notes', 'prompt', 'input_prompt',
    'variant_count', 'variant_styles', 'model_registry', 'algorithm_profile', 'material_finish_profile',
    'sample', 'run_id', 'parent_run_id',
    'generation_design', 'character_profile', 'appearance_revision', 'manual_layers', 'native_source_retouch',
    'preset_snapshot', 'effective_prompt', 'prompt_template_version', 'size', 'alpha_backend',
    'denoise_strength', 'residual', 'preserve_project_selection', 'defer_reference_selection', 'creative_task',
    'async', 'sync', 'plan_id', 'generate', 'local_only', 'complete_post', 'final_repair_only',
    'execution_strategy', 'preparation_policy', 'creative_direction', 'editing_request', 'regional_resume_authorization',
    'regional_candidate_reuse_authorization', 'editing_iteration', 'editing_autofinish', 'shaping_request'))
SHARED_FIELDS = frozenset(('preset', 'provider', 'model', 'vision_model', 'api_base', 'api_url',
    'api_key_env', 'character_name', 'character_notes', 'prompt', 'input_prompt',
    'variant_count', 'variant_styles', 'model_registry', 'algorithm_profile', 'material_finish_profile',
    'size', 'defer_reference_selection', 'creative_task', 'execution_strategy', 'preparation_policy', 'editing_request', 'editing_autofinish'))

EXECUTION_STRATEGIES = frozenset({'layered_reference_transfer'})
PREPARATION_POLICIES = frozenset({'prepare_current', 'replay_frozen'})


def normalize_preparation_policy(value=None, *, default='replay_frozen'):
    policy = default if value is None else value
    if not isinstance(policy, str) or policy not in PREPARATION_POLICIES:
        raise ValueError('Unsupported preparation policy: ' + str(policy))
    return policy


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf8')).hexdigest()


def sha(path):
    return sha256_file(path)


def normalize_task_intent(value):
    if not isinstance(value, dict) or set(value) - {'schema_version', 'mode', 'brief', 'constraints', 'services', 'regional_refinement'}:
        raise ValueError('Provide the current creative task intent')
    if value.get('schema_version') != INTENT_SCHEMA or value.get('mode') not in MODES:
        raise ValueError('Unsupported creative task schema or route')
    brief = value.get('brief', '')
    if not isinstance(brief, str) or len(brief) > 16000:
        raise ValueError('Creative brief must be text of at most 16000 characters')
    constraints = value.get('constraints', {})
    services = value.get('services', {})
    if not isinstance(constraints, dict) or set(constraints) - {'preserve_background', 'no_effects', 'background_mode', 'reference_role', 'refinement_profile'}:
        raise ValueError('Unsupported creative task constraints')
    create=value['mode']=='create_reference'
    if create and (any(key in constraints for key in ('reference_role','refinement_profile'))
            or 'regional_refinement' in value):
        raise ValueError('Retired creation route: choose a current editing scope instead of reference roles, profiles or native regional generation')
    if not isinstance(services, dict) or set(services) - {'text_vision', 'visual_review'}:
        raise ValueError('Unsupported creative task service policy')
    boolean_options = [(key, item) for key, item in constraints.items() if key not in {'background_mode', 'reference_role', 'refinement_profile'}]
    for key, item in [*boolean_options, *services.items()]:
        if type(item) is not bool:
            raise ValueError('Task option must be a boolean: ' + key)
    local = value['mode'] == 'local_portrait'
    if local and services.get('text_vision') is True:
        raise ValueError('Local portrait tasks do not call external text/vision services')
    text_vision = services.get('text_vision', create)
    visual_review = services.get('visual_review', create and text_vision)
    if visual_review and (local or not text_vision):
        raise ValueError('Automatic visual review requires an external text/vision service')
    result = {'schema_version': INTENT_SCHEMA, 'mode': value['mode'], 'brief': brief.strip(),
        'constraints': {key: constraints.get(key, local) for key in ('preserve_background', 'no_effects')},
        'services': {'text_vision': text_vision, 'visual_review': visual_review}}
    if 'background_mode' in constraints:
        if constraints['background_mode'] != 'enhance_source':
            raise ValueError('Unsupported explicit background mode')
        if local or result['constraints']['preserve_background']:
            raise ValueError('Source enhancement requires a reference task with preserve_background false')
        from creative_direction import has_no_effects_veto
        result['constraints']['background_mode'] = 'enhance_source'
        result['constraints']['no_effects'] |= has_no_effects_veto(brief)
    if 'reference_role' in constraints:
        reference_role = constraints['reference_role']
        if (not isinstance(reference_role, str) or reference_role not in {'same_photo_finish', 'source_conditioned_design'}
                or value['mode'] != 'existing_reference'
                or result['constraints'].get('background_mode') != 'enhance_source'
                or 'regional_refinement' in value):
            raise ValueError('Explicit reference roles require the supported enhance_source route and no regional requests')
        result['constraints']['reference_role'] = reference_role
    if 'refinement_profile' in constraints:
        from preset_store import normalize_refinement_selector
        if (value['mode'] != 'existing_reference'
                or result['constraints'].get('reference_role') != 'source_conditioned_design'
                or result['constraints'].get('background_mode') != 'enhance_source'
                or 'regional_refinement' in value):
            raise ValueError('Refinement recommendation requires the source-conditioned single-photo route')
        selector = normalize_refinement_selector(constraints['refinement_profile'])
        result['constraints']['refinement_profile'] = selector
        from creative_direction import has_no_effects_veto
        override_text = '\n'.join([brief, *selector.get('overrides', {}).values()])
        result['constraints']['no_effects'] |= has_no_effects_veto(override_text) or '无特效' in override_text
    if 'regional_refinement' in value:
        from creative_direction import normalize_regional_refinement
        result['regional_refinement'] = normalize_regional_refinement(value['regional_refinement'],
            mode=value['mode'], background_mode=result['constraints'].get('background_mode'))
    return result


def autofinish_policy(value, intent, editing, *, supplied=False):
    """One canonical source-creation policy; manual iteration stays separate."""
    available = editing is not None and intent['mode'] == 'create_reference'
    if not available:
        if supplied:
            raise ValueError('Automatic finishing requires a scoped create-reference task')
        return None
    if value is None and not supplied:
        enabled=int(intent['services']['visual_review'])
        value = {'maximum_improvements': 2*enabled, 'maximum_zero_image_finishes':enabled}
    if (not isinstance(value, dict) or 'maximum_improvements' not in value
            or set(value)-{'maximum_improvements','maximum_zero_image_finishes'}):
        raise ValueError('Automatic finishing requires maximum_improvements')
    count = value['maximum_improvements']
    if type(count) is not int or not 0 <= count <= 8:
        raise ValueError('Automatic improvements must be an integer from zero to eight')
    if count and not intent['services']['visual_review']:
        raise ValueError('Automatic improvements require independent visual review')
    finish=value.get('maximum_zero_image_finishes',0)
    if type(finish) is not int or not 0<=finish<=1 or (finish and (not count or not intent['services']['visual_review'])):
        raise ValueError('One zero-image finish requires enabled automatic improvements and independent review')
    return {'maximum_improvements': count,'maximum_zero_image_finishes':finish}


def image_budget(intent, variant_count=1, editing=None, iteration=None, autofinish=None, shaping=None):
    intent = normalize_task_intent(intent)
    count = variant_count
    if type(count) is not int or not 1 <= count <= 8:
        raise ValueError('Candidate count must be an integer from 1 to 8')
    create = intent['mode'] == 'create_reference'
    if create and editing is None:
        from photo_editing import editing_scope
        editing = editing_scope(preserve_background=intent['constraints']['preserve_background'],
                                no_effects=intent['constraints']['no_effects'])
    budget = {'initial_reference': count if create else 0,
            'reference_correction': 0, 'independent_effect': 0}
    if shaping is not None:
        from photo_shaping import normalize_request,image_submissions
        shaping = normalize_request(shaping)
        if intent['mode'] != 'existing_reference' or editing is None or count != 1 or iteration is not None or autofinish is not None:
            raise ValueError('Shaping requires one separately checked saved input')
        return {**budget, 'shaping': image_submissions(shaping)}
    if editing is not None:
        # Scoped scene/light/effects are baked into one coherent candidate.
        # No separate legacy correction or FX submission is implemented here.
        budget['reference_correction'] = 0
        budget['independent_effect'] = 0
        if iteration is not None:
            budget['editing_refinement']=iteration.get('maximum_image_submissions',iteration['maximum_improvements'])
        elif create:
            policy = autofinish_policy(autofinish, intent, editing, supplied=autofinish is not None)
            budget['editing_refinement'] = policy['maximum_improvements']
    if 'regional_refinement' in intent:
        regional = intent['regional_refinement']
        budget['regional_refinement'] = regional['maximum_submissions'] if intent['mode'] == 'existing_reference' else 0
    return budget


def task_brief(intent):
    intent = normalize_task_intent(intent)
    lines = [intent['brief']] if intent['brief'] else []
    if intent['constraints']['preserve_background']:
        lines.append('保留原背景，不更换或重建场景。')
    if intent['constraints'].get('background_mode') == 'enhance_source':
        lines.append('保留原场景结构、主要物件、机位与透视，只适当清理和增强原景；禁止完整换景。')
    if intent['constraints'].get('reference_role') == 'same_photo_finish':
        lines.append('参考是用户声明的同照精修标尺；原片提供几何和原生纹理，参考仅提供影调及局部低频色映射；执行时核验配准，不直接合成参考RGB。')
    if intent['constraints'].get('reference_role') == 'source_conditioned_design':
        lines.append('只以本次原片生成同源美术设计参考，不读取或复刻用户人工成片；原片拥有几何和原生纹理，参考仅提供影调与局部低频色映射，执行时核验实际生成来源及配准。')
    if intent['constraints']['no_effects']:
        lines.append('不要添加特效。')
    return '\n'.join(lines)


def _validate_resume_proof(proof, intent, inputs):
    import re
    keys = {'schema', 'resume_from_run_id', 'origin_run_id', 'previous_plan_id',
        'source_sha256', 'reference_sha256', 'reservation_sha256',
        'reservation_binding_sha256', 'previous_direction_id', 'previous_manifest_sha256'}
    if (not isinstance(proof, dict) or set(proof) != keys
            or proof.get('schema') != 'regional_resume_authorization.v1'
            or proof.get('resume_from_run_id') != intent['regional_refinement']['resume_from_run_id']):
        raise ValueError('Regional resume requires its checked immutable authorization')
    for key in ('resume_from_run_id', 'origin_run_id'):
        if not isinstance(proof[key], str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', proof[key]):
            raise ValueError('Unsafe regional resume identity')
    for key in keys - {'schema', 'resume_from_run_id', 'origin_run_id'}:
        if not isinstance(proof[key], str) or not re.fullmatch('[0-9a-f]{64}', proof[key]):
            raise ValueError('Regional resume proof must contain checked SHA256 identities')
    if any(proof[role+'_sha256'] != (inputs.get(role) or {}).get('sha256') for role in ('source', 'reference')):
        raise ValueError('Regional resume proof source/reference differs from checked inputs')


def _resume_file(root, path):
    """No links, traversal or mutable state repair while checking authorization."""
    path = Path(path)
    if not path.is_relative_to(root):
        raise ValueError('Regional resume evidence must remain within its workspace')
    for current in (path, *path.parents):
        if current == root:
            break
        if current.is_symlink() or (hasattr(current, 'is_junction') and current.is_junction()):
            raise ValueError('Regional resume evidence must not traverse links')
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise ValueError('Regional resume evidence is unavailable or malformed') from exc
    if not isinstance(value, dict):
        raise ValueError('Regional resume evidence must be an object')
    return value, hashlib.sha256(raw).hexdigest()


def verify_regional_resume(root, intent, inputs, direction, *, executing_plan_id=None):
    """Pure preflight proof; the worker rechecks it under the origin journal lock."""
    import re
    root = Path(root).resolve()
    regional = intent.get('regional_refinement') or {}
    failed_id = regional.get('resume_from_run_id')
    if (intent.get('mode') != 'existing_reference' or regional.get('maximum_submissions') != 1
            or not isinstance(failed_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', failed_id)):
        raise ValueError('No explicit zero-submission regional resume intent')
    previous, manifest_sha = _resume_file(root, root/'outputs/runs'/failed_id/'manifest.json')
    if previous.get('run_id') != failed_id or previous.get('status') not in {'blocked', 'failed', 'cancelled'}:
        raise ValueError('Regional resume requires a stopped failed parent')
    options = previous.get('options') or {}
    previous_intent = previous.get('task_intent') or options.get('task_intent')
    previous_plan = previous.get('task_plan') or options.get('task_plan')
    saved_task_intent_for_replanning(previous_plan, previous_intent,
        source_sha256=(inputs.get('source') or {}).get('sha256'),
        reference_sha256=(inputs.get('reference') or {}).get('sha256'))
    if any((previous.get(role) or {}).get('sha256') != (inputs.get(role) or {}).get('sha256') for role in ('source', 'reference')):
        raise ValueError('Regional resume must reuse the exact failed source and global reference')
    old_regional = previous_intent.get('regional_refinement') or {}
    if (old_regional.get('maximum_submissions') != 1 or len(old_regional.get('regions') or []) != 1
            or old_regional['regions'] != regional['regions']):
        raise ValueError('Regional resume cannot change the original one-region instruction or core')
    previous_direction = previous_plan.get('creative_direction') or {}
    if (previous_intent.get('brief') != intent['brief']
            or previous_intent.get('constraints') != intent['constraints']
            or previous_intent.get('services') != intent['services']
            or previous_direction.get('character') != direction.get('character')
            or previous_direction.get('main_effect') != direction.get('main_effect')):
        raise ValueError('Regional resume must retain the original role, brief and constraints')
    from creative_direction import MARKER
    if previous_direction.get('prompt', '').split(MARKER, 1)[0] != direction['prompt'].split(MARKER, 1)[0]:
        raise ValueError('Regional resume cannot change the original editable prompt')
    origin = (previous.get('generation_budget') or {}).get('origin_run_id') or failed_id
    if not isinstance(origin, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', origin):
        raise ValueError('Unsafe original regional reservation owner')
    for job_path in (root/'outputs/jobs').glob('*/job.json'):
        job, _ = _resume_file(root, job_path)
        if executing_plan_id and (job.get('task_plan') or {}).get('plan_id') == executing_plan_id:
            # The checked continuation's own running job is not its parent.
            # The origin journal lock still arbitrates concurrent consumption.
            continue
        if job.get('status') in {'queued', 'running', 'cancelling'} and (
                job.get('run_id') in {failed_id, origin} or job.get('parent_run_id') in {failed_id, origin}
                or (((job.get('task_intent') or {}).get('regional_refinement') or {}).get('resume_from_run_id') == failed_id)):
            raise ValueError('Regional resume parent or another continuation is still active')
    reservation, reservation_sha = _resume_file(root,
        root/'outputs/regional-reservations'/origin/'regional-refinement.json')
    def submitted_id(value):
        if isinstance(value, dict):
            return any((key.replace('_', '').lower() in {'promptid', 'providerids'} and bool(item))
                or submitted_id(item) for key, item in value.items() if key != 'attempt_history')
        if isinstance(value, list):
            return any(submitted_id(item) for item in value)
        return False
    if (reservation.get('schema') != 'regional_scene_refinement.v1'
            or reservation.get('state') != 'failed_no_submission'
            or type(reservation.get('model_submissions')) is not int or reservation['model_submissions'] != 0
            or reservation.get('request_started') not in (None, False) or submitted_id(reservation)):
        raise ValueError('Regional reservation is not proven unsubmitted; never grant an unknown request another allowance')
    binding = reservation.get('binding') or {}
    if (binding.get('source') != inputs['source']['sha256'] or binding.get('reference') != inputs['reference']['sha256']
            or binding.get('core_box') != regional['regions'][0]['core_box']
            or binding.get('checked_plan_id') != previous_plan['plan_id']
            or binding.get('direction_id') != previous_direction.get('direction_id')):
        raise ValueError('Regional failed reservation differs from its original checked task')
    proof = {'schema': 'regional_resume_authorization.v1', 'resume_from_run_id': failed_id,
        'origin_run_id': origin, 'previous_plan_id': previous_plan['plan_id'],
        'source_sha256': inputs['source']['sha256'], 'reference_sha256': inputs['reference']['sha256'],
        'previous_manifest_sha256': manifest_sha, 'reservation_sha256': reservation_sha,
        'reservation_binding_sha256': digest(binding), 'previous_direction_id': previous_direction['direction_id']}
    _validate_resume_proof(proof, intent, inputs)
    return proof


def revalidate_regional_resume(root, plan):
    """Call just before consumption, under the worker's origin reservation lock."""
    intent = normalize_task_intent(plan.get('intent'))
    inputs = plan.get('inputs') or {}
    from creative_direction import validate_direction
    direction = validate_direction(plan.get('creative_direction'), intent,
        source_sha256=(inputs.get('source') or {}).get('sha256'))
    actual = verify_regional_resume(root, intent, inputs, direction, executing_plan_id=plan.get('plan_id'))
    if actual != plan.get('regional_resume_authorization'):
        raise ValueError('Regional resume evidence changed after checking the task plan')
    return actual


def _validate_candidate_reuse_proof(proof, intent, inputs):
    import re
    identities = {'previous_plan_id', 'previous_direction_id', 'source_sha256', 'reference_sha256',
        'previous_manifest_sha256', 'reservation_sha256', 'reservation_binding_sha256'}
    keys = identities | {'schema', 'reuse_generated_from_run_id', 'origin_run_id', 'prompt_id',
        'conditioning_policy', 'assets', 'original_core_box', 'requested_core_box'}
    roles = {'raw_candidate', 'raw_provenance', 'input_context', 'reliable_halo',
        'condition_manifest', 'permission_mask', 'request', 'input_image'}
    if (not isinstance(proof, dict) or set(proof) != keys
            or proof.get('schema') != 'regional_candidate_reuse_authorization.v1'
            or proof.get('conditioning_policy') != 'single_native_ROI_CPU_permission.v3'
            or proof.get('reuse_generated_from_run_id') != intent['regional_refinement']['reuse_generated_from_run_id']
            or not isinstance(proof.get('assets'), dict) or set(proof['assets']) != roles):
        raise ValueError('Generated regional candidate requires its frozen one-photo reuse proof')
    for key in identities:
        if not isinstance(proof[key], str) or not re.fullmatch('[0-9a-f]{64}', proof[key]):
            raise ValueError('Generated candidate proof requires content hashes')
    for key in ('reuse_generated_from_run_id', 'origin_run_id', 'prompt_id'):
        if not isinstance(proof[key], str) or not re.fullmatch('[A-Za-z0-9_-]{1,128}', proof[key]):
            raise ValueError('Generated candidate proof has unsafe run/provider identity')
    for key in ('original_core_box', 'requested_core_box'):
        box = proof[key]
        if (not isinstance(box, list) or len(box) != 4 or any(type(v) is not int or v < 0 for v in box)
                or not 0 < box[2]-box[0] <= 768 or not 0 < box[3]-box[1] <= 768):
            raise ValueError('Generated candidate proof has an invalid native core')
    if proof['requested_core_box'] != intent['regional_refinement']['regions'][0]['core_box']:
        raise ValueError('Candidate reuse proof differs from explicitly requested CPU core')
    for role in roles:
        record = proof['assets'][role]
        if (not isinstance(record, dict) or set(record) != {'path', 'sha256'}
                or not isinstance(record['path'], str) or not Path(record['path']).is_absolute()
                or not isinstance(record['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', record['sha256'])):
            raise ValueError('Generated candidate proof has an invalid frozen asset')
    if any(proof[role+'_sha256'] != (inputs.get(role) or {}).get('sha256') for role in ('source', 'reference')):
        raise ValueError('Generated candidate proof differs from source/reference')


def verify_regional_candidate_reuse(root, intent, inputs, direction, *, executing_plan_id=None):
    """Read returned ONE-photo pixels; grants no request and never changes the journal."""
    import re
    import numpy as np
    from PIL import Image
    from creative_direction import MARKER
    root = Path(root).resolve()
    regional = intent.get('regional_refinement') or {}
    failed_id = regional.get('reuse_generated_from_run_id')
    if (intent.get('mode') != 'existing_reference' or regional.get('maximum_submissions') != 0
            or intent.get('constraints', {}).get('background_mode') != 'enhance_source'
            or len(regional.get('regions') or []) != 1 or not isinstance(failed_id, str)
            or not re.fullmatch('[A-Za-z0-9_-]{1,128}', failed_id)):
        raise ValueError('No explicit zero-generation candidate reuse intent')
    previous, manifest_sha = _resume_file(root, root/'outputs/runs'/failed_id/'manifest.json')
    if previous.get('run_id') != failed_id or previous.get('status') not in {'blocked', 'failed', 'cancelled'}:
        raise ValueError('Candidate reuse requires a stopped failed run')
    options = previous.get('options') or {}
    old_intent = previous.get('task_intent') or options.get('task_intent')
    old_plan = previous.get('task_plan') or options.get('task_plan')
    saved_task_intent_for_replanning(old_plan, old_intent,
        source_sha256=inputs['source']['sha256'], reference_sha256=inputs['reference']['sha256'])
    if any((previous.get(role) or {}).get('sha256') != inputs[role]['sha256'] for role in ('source', 'reference')):
        raise ValueError('Candidate reuse cannot change source or global reference')
    old_regional = old_intent.get('regional_refinement') or {}
    old_direction = old_plan.get('creative_direction') or {}
    old_regions = old_regional.get('regions') or []
    old_region_contract = [{key:value for key,value in row.items() if key != 'core_box'} for row in old_regions]
    requested_region_contract = [{key:value for key,value in row.items() if key != 'core_box'} for row in regional['regions']]
    if (old_regional.get('maximum_submissions') != 1 or len(old_regions) != 1 or old_region_contract != requested_region_contract
            or old_intent.get('brief') != intent['brief'] or old_intent.get('constraints') != intent['constraints']
            or old_intent.get('services') != intent['services']
            or old_direction.get('character') != direction['character']
            or old_direction.get('main_effect') != direction['main_effect']
            or old_direction.get('prompt', '').split(MARKER, 1)[0] != direction['prompt'].split(MARKER, 1)[0]):
        raise ValueError('Candidate reuse cannot change original role, brief, effect or region')
    origin = (previous.get('generation_budget') or {}).get('origin_run_id') or failed_id
    if not isinstance(origin, str) or not re.fullmatch('[A-Za-z0-9_-]{1,128}', origin):
        raise ValueError('Unsafe generated candidate origin')
    for job_path in (root/'outputs/jobs').glob('*/job.json'):
        job, _ = _resume_file(root, job_path)
        if executing_plan_id and (job.get('task_plan') or {}).get('plan_id') == executing_plan_id:
            continue
        if job.get('status') in {'queued', 'running', 'cancelling'} and (
                job.get('run_id') in {failed_id, origin} or job.get('parent_run_id') in {failed_id, origin}):
            raise ValueError('Generated candidate parent is still active')
    ledger, ledger_sha = _resume_file(root, root/'outputs/regional-reservations'/origin/'regional-refinement.json')
    prompt_id = ledger.get('prompt_id')
    if (ledger.get('schema') != 'regional_scene_refinement.v1' or ledger.get('state') != 'generated_rejected'
            or ledger.get('provider_returned') is not True or type(ledger.get('model_submissions')) is not int
            or ledger['model_submissions'] != 1 or not isinstance(prompt_id, str)
            or not re.fullmatch('[A-Za-z0-9_-]{1,128}', prompt_id)):
        raise ValueError('Candidate reuse requires one known returned, rejected generation')
    binding = ledger.get('binding') or {}
    if (binding.get('source') != inputs['source']['sha256'] or binding.get('reference') != inputs['reference']['sha256']
            or binding.get('core_box') != old_regions[0]['core_box']
            or binding.get('checked_plan_id') != old_plan['plan_id']
            or binding.get('direction_id') != old_direction.get('direction_id')):
        raise ValueError('Returned candidate differs from its original checked binding')
    requested_core = regional['regions'][0]['core_box']
    original_context = binding.get('context_box')
    native = binding.get('native_size')
    if (not isinstance(original_context, list) or len(original_context) != 4
            or any(type(v) is not int for v in original_context)
            or not isinstance(native, list) or len(native) != 2 or any(type(v) is not int or v < 1 for v in native)
            or not 0 <= requested_core[0] < requested_core[2] <= native[0]
            or not 0 <= requested_core[1] < requested_core[3] <= native[1]
            or not original_context[0] <= requested_core[0] < requested_core[2] <= original_context[2]
            or not original_context[1] <= requested_core[1] < requested_core[3] <= original_context[3]):
        raise ValueError('Explicit candidate CPU core must remain inside the original raw context and native canvas')
    if native != ((old_plan.get('inputs') or {}).get('source') or {}).get('regional_native_size'):
        raise ValueError('Retained candidate canvas differs from original metadata preflight')
    def file_asset(record):
        if not isinstance(record, dict) or not isinstance(record.get('path'), str):
            raise ValueError('Generated candidate asset is missing')
        path = Path(record['path'])
        if not path.is_absolute() or not path.is_relative_to(root):
            raise ValueError('Generated candidate assets must remain in the workspace')
        for ancestor in (path, *path.parents):
            if ancestor == root:
                break
            if ancestor.is_symlink() or (hasattr(ancestor, 'is_junction') and ancestor.is_junction()):
                raise ValueError('Generated candidate assets must not traverse links')
        if not path.is_file():
            raise ValueError('Generated candidate asset no longer exists')
        current_sha = sha(path)
        if record.get('sha256', current_sha) != current_sha:
            raise ValueError('Generated candidate asset content changed')
        return {'path': str(path), 'sha256': current_sha}
    for role in ('raw_candidate', 'raw_provenance'):
        if not isinstance(ledger.get(role), dict) or not isinstance(ledger[role].get('sha256'), str):
            raise ValueError('Returned candidate must have its original recorded asset hash')
    assets = {role: file_asset(ledger.get(role)) for role in ('raw_candidate', 'raw_provenance')}
    directory = Path(assets['raw_candidate']['path']).parent.parent
    for role, filename in (('input_context', 'input-context.npz'), ('reliable_halo', 'reliable-halo.npz'),
                           ('condition_manifest', 'conditions.json'), ('permission_mask', 'permission.png'), ('request', 'request.json')):
        assets[role] = file_asset({'path': str(directory/filename)})
    conditions, _ = _resume_file(root, assets['condition_manifest']['path'])
    condition_inputs = conditions.get('inputs') or []
    if (len(condition_inputs) != 1 or condition_inputs[0].get('role') != 'source'
            or condition_inputs[0].get('coordinate_space') != 'source_canvas_roi'
            or (conditions.get('permission_mask') or {}).get('transmitted_to_model') is not False):
        raise ValueError('Only the single local photograph conditioning policy may be reused')
    assets['input_image'] = file_asset(condition_inputs[0])
    if file_asset(conditions.get('permission_mask')) != assets['permission_mask']:
        raise ValueError('Candidate permission asset differs from its conditions')
    provenance, _ = _resume_file(root, assets['raw_provenance']['path'])
    contract = provenance.get('provider_contract') or {}
    actual = contract.get('comfyui') or {}
    graph = (actual.get('request') or {}).get('prompt') or {}
    encoders = [row.get('inputs') or {} for row in graph.values() if row.get('class_type') == 'CosplayTextEncodeQwenImage21']
    uploads = actual.get('condition_inputs') or []
    events = (contract.get('transport') or {}).get('events') or []
    submitted = [event for event in events if event.get('event') == 'comfyui_submitted']
    loaders = [(key, row.get('inputs') or {}) for key, row in graph.items() if row.get('class_type') == 'LoadImage']
    if (len(uploads) != 1 or len(encoders) != 1
            or [key for key in encoders[0] if key.startswith('images.')] != ['images.image_1']
            or len(loaders) != 1 or encoders[0].get('images.image_1') != [loaders[0][0], 0]
            or loaders[0][1].get('image') != (uploads[0].get('upload') or {}).get('name')
            or len(submitted) != 1 or submitted[0].get('prompt_id') != prompt_id
            or provenance.get('generation_id') != prompt_id or actual.get('prompt_id') != prompt_id
            or (contract.get('response') or {}).get('id') != prompt_id
            or (submitted[0].get('request') or {}).get('prompt') != graph
            or encoders[0].get('prompt') != binding.get('prompt') or provenance.get('prompt') != binding.get('prompt')
            or file_asset(uploads[0]) != assets['input_image']
            or (provenance.get('source') or {}).get('sha256') != inputs['source']['sha256']
            or file_asset(provenance.get('asset')) != assets['raw_candidate']):
        raise ValueError('Actual transport does not prove the one-photo returned local candidate')
    request, _ = _resume_file(root, assets['request']['path'])
    if (((request.get('run') or {}).get('task_plan') or {}).get('plan_id') != old_plan['plan_id']
            or (request.get('plan') or {}).get('canvas', {}).get('source_oriented_size') != binding.get('native_size')):
        raise ValueError('Candidate request differs from the original checked native canvas')
    def pixels_sha(values):
        values = np.ascontiguousarray(values)
        return hashlib.sha256(str((values.shape, values.dtype.str)).encode()+values.tobytes()).hexdigest()
    with np.load(assets['input_context']['path'], allow_pickle=False) as archive:
        context = archive['values']
    with np.load(assets['reliable_halo']['path'], allow_pickle=False) as archive:
        reliable = archive['values']
    core, box = binding.get('core_box'), binding.get('context_box')
    if (context.shape != (1024, 1024, 3) or reliable.shape != (1024, 1024)
            or not np.isfinite(context).all() or not np.isfinite(reliable).all()
            or np.any((reliable != 0) & (reliable != 1)) or not isinstance(box, list) or len(box) != 4
            or box[2]-box[0] != 1024 or box[3]-box[1] != 1024
            or not 0 <= core[0]-box[0] < core[2]-box[0] <= 1024
            or not 0 <= core[1]-box[1] < core[3]-box[1] <= 1024
            or pixels_sha(context) != binding.get('context_pixels_sha256')
            or pixels_sha(context[core[1]-box[1]:core[3]-box[1],core[0]-box[0]:core[2]-box[0]]) != binding.get('base_core_pixels_sha256')):
        raise ValueError('Retained candidate context differs from its original base binding')
    permission = np.zeros((1024, 1024), bool)
    permission[core[1]-box[1]:core[3]-box[1],core[0]-box[0]:core[2]-box[0]] = True
    with Image.open(assets['permission_mask']['path']) as image:
        mask = np.asarray(image)
    if mask.shape != permission.shape or not np.array_equal(mask, permission.astype('uint8')*255) or pixels_sha(permission) != binding.get('permission_pixels_sha256'):
        raise ValueError('Retained candidate permission differs from the frozen native core')
    with Image.open(assets['raw_candidate']['path']) as image:
        if image.size != (1024, 1024):
            raise ValueError('Retained model output has a different canvas')
    proof = {'schema': 'regional_candidate_reuse_authorization.v1',
        'reuse_generated_from_run_id': failed_id, 'origin_run_id': origin,
        'previous_plan_id': old_plan['plan_id'], 'previous_direction_id': old_direction['direction_id'],
        'source_sha256': inputs['source']['sha256'], 'reference_sha256': inputs['reference']['sha256'],
        'previous_manifest_sha256': manifest_sha, 'reservation_sha256': ledger_sha,
        'reservation_binding_sha256': digest(binding), 'prompt_id': prompt_id,
        'conditioning_policy': 'single_native_ROI_CPU_permission.v3', 'assets': assets,
        'original_core_box': binding['core_box'], 'requested_core_box': requested_core}
    _validate_candidate_reuse_proof(proof, intent, inputs)
    return proof


def revalidate_regional_candidate_reuse(root, plan):
    intent = normalize_task_intent(plan.get('intent'))
    from creative_direction import validate_direction
    direction = validate_direction(plan.get('creative_direction'), intent,
        source_sha256=((plan.get('inputs') or {}).get('source') or {}).get('sha256'))
    proof = verify_regional_candidate_reuse(root, intent, plan['inputs'], direction, executing_plan_id=plan.get('plan_id'))
    if proof != plan.get('regional_candidate_reuse_authorization'):
        raise ValueError('Returned candidate evidence changed after checking the task')
    return proof


def _validate_refinement_inheritance(proof, intent, inputs):
    selector = intent['constraints'].get('refinement_profile')
    inherited = selector is not None and intent['mode'] == 'existing_reference'
    if not inherited:
        if proof is not None:
            raise ValueError('Unexpected refinement inheritance without an existing checked recommendation')
        return
    from preset_store import get_refinement_profile
    if (not isinstance(proof, dict) or proof.get('schema') != 'refinement_profile_inheritance.v1'
            or proof.get('profile') != selector
            or proof.get('profile_sha256') != get_refinement_profile(selector)['profile_sha256']
            or proof.get('source_sha256') != (inputs.get('source') or {}).get('sha256')
            or proof.get('reference_sha256') != (inputs.get('reference') or {}).get('sha256')):
        raise ValueError('Existing refinement recommendation requires its saved checked inheritance')
    import re
    if not isinstance(proof.get('parent_run_id'), str) or not re.fullmatch(r'run-[A-Za-z0-9_-]{1,120}', proof['parent_run_id']):
        raise ValueError('Invalid saved refinement origin')
    for key in ('previous_plan_id', 'previous_direction_id', 'previous_manifest_sha256'):
        if not isinstance(proof.get(key), str) or not re.fullmatch('[0-9a-f]{64}', proof[key]):
            raise ValueError('Incomplete saved refinement identity')


def _refinement_inheritance(root, intent, inputs, parent_run_id):
    """Pure read proof; an old no-profile run can never acquire this default."""
    import re
    if not isinstance(parent_run_id, str) or not re.fullmatch(r'run-[A-Za-z0-9_-]{1,120}', parent_run_id):
        raise ValueError('Existing refinement recommendation requires a saved parent run')
    base = root/'outputs/runs'
    path = base/parent_run_id/'manifest.json'
    if not path.resolve().is_relative_to(base.resolve()):
        raise ValueError('Saved refinement origin escapes its project')
    try:
        previous = json.loads(path.read_text(encoding='utf8'))
    except (OSError, ValueError) as exc:
        raise ValueError('Saved refinement origin is unavailable or invalid') from exc
    if not isinstance(previous, dict):
        raise ValueError('Saved refinement origin must be a run manifest')
    if (previous.get('run_id') != parent_run_id
            or previous.get('status') not in {'needs_visual_review', 'accepted', 'delivered', 'blocked', 'failed', 'cancelled'}
            or (previous.get('source') or {}).get('sha256') != inputs['source']['sha256']
            or (previous.get('reference') or {}).get('sha256') != inputs['reference']['sha256']):
        raise ValueError('Saved refinement source/reference or inactive state differs')
    old_intent = previous.get('task_intent') or (previous.get('options') or {}).get('task_intent')
    old_plan = previous.get('task_plan') or (previous.get('options') or {}).get('task_plan')
    verified = saved_task_intent_for_replanning(old_plan, old_intent,
        source_sha256=inputs['source']['sha256'], reference_sha256=inputs['reference']['sha256'])
    selector = intent['constraints']['refinement_profile']
    if (verified['constraints'].get('refinement_profile') != selector
            or verified.get('brief') != intent.get('brief')):
        raise ValueError('Saved refinement profile/overrides or brief differs; check a new creative task')
    from preset_store import get_refinement_profile
    proof = {'schema': 'refinement_profile_inheritance.v1', 'parent_run_id': parent_run_id,
        'previous_plan_id': old_plan['plan_id'], 'previous_direction_id': old_plan['creative_direction']['direction_id'],
        'previous_manifest_sha256': sha(path), 'source_sha256': inputs['source']['sha256'],
        'reference_sha256': inputs['reference']['sha256'], 'profile': copy.deepcopy(selector),
        'profile_sha256': get_refinement_profile(selector)['profile_sha256']}
    _validate_refinement_inheritance(proof, intent, inputs)
    return proof


def validate_task_plan(plan, intent):
    intent = normalize_task_intent(intent)
    if not isinstance(plan, dict) or plan.get('schema_version') != PLAN_SCHEMA:
        raise ValueError('Current task execution requires a checked task plan')
    if plan.get('intent') != intent or plan.get('mode') != intent['mode'] or plan.get('ready') is not True:
        raise ValueError('Task plan does not match its current intent or is blocked')
    budget = plan.get('image_budget')
    if not isinstance(budget, dict) or any(type(value) is not int or value < 0 for value in budget.values()):
        raise ValueError('Task image budget must contain nonnegative integer limits')
    scope = plan.get('editing_scope')
    if intent['mode']=='create_reference' and (scope is None or 'editing_request' not in plan):
        raise ValueError('Creation requires the current checked editing scope')
    from scoped_workflow import SOURCE_NATIVE_INITIALIZATION
    initialization=plan.get('editing_initialization')
    if intent['mode']=='create_reference':
        if initialization!=SOURCE_NATIVE_INITIALIZATION:
            raise ValueError('Creation requires the current source-native initialization policy')
    elif initialization is not None:
        raise ValueError('Source-native initialization belongs to a new creation task')
    if scope is not None or 'editing_request' in plan:
        from photo_editing import editing_scope
        expected_scope = editing_scope(plan['editing_request'], local=intent['mode']=='local_portrait',
            preserve_background=intent['constraints']['preserve_background'],
            no_effects=intent['constraints']['no_effects'])
        if scope != expected_scope:
            raise ValueError('Editing scope differs from the checked task permissions')
    auto = autofinish_policy(plan.get('editing_autofinish'), intent, scope,
                            supplied='editing_autofinish' in plan)
    if scope is not None and intent['mode']=='create_reference' and plan.get('editing_autofinish') != auto:
        raise ValueError('Scoped source creation requires its canonical automatic finishing policy')
    if auto is not None and plan.get('editing_iteration') is not None:
        raise ValueError('Automatic finishing cannot share a manual improvement request')
    shaping = plan.get('shaping_request')
    expected = image_budget(intent, budget.get('initial_reference', 0)
                            if intent['mode'] == 'create_reference' else 1, scope,plan.get('editing_iteration'),auto,shaping)
    if shaping is not None:
        from photo_shaping import normalize_request, validate_authorization_metadata
        shaping = normalize_request(shaping)
        proof = plan.get('shaping_authorization') or {}
        apply = shaping['action'] == 'apply'
        if (scope is None or intent['mode'] != 'existing_reference' or plan.get('editing_iteration') is not None
                or proof.get('request') != shaping
                or proof.get('base_image_sha256') != plan['inputs']['reference']['sha256']
                or intent['services'] != {'text_vision': apply, 'visual_review': apply}
                or (plan.get('services') or {}).get('text_vision_requests_upper_bound') != (3 if apply else 0)):
            raise ValueError('Shaping differs from its checked input, permission or budget')
        validate_authorization_metadata(proof, source_sha256=plan['inputs']['source']['sha256'], scope=scope)
        from task_model_preflight import required_capabilities
        required = required_capabilities(intent['mode'], execution_strategy=plan.get('execution_strategy'),
            complete_post=True, editing_scope=scope, shaping_request=shaping)
        if required:
            installation = (plan.get('capabilities') or {}).get('model_preflight') or {}
            if installation.get('required_capabilities') != sorted(required) or installation.get('ready') is not True:
                raise ValueError('Face shaping requires its checked portrait-region installation')
        if plan.get('generation_submissions_upper_bound')!=sum(expected.values()):
            raise ValueError('Shaping image limit differs from its checked permission')
    elif plan.get('shaping_authorization') is not None:
        raise ValueError('Unexpected shaping authorization')
    if auto is not None:
        expected_vision = (2+2*budget['initial_reference']+2*(auto['maximum_improvements']+auto['maximum_zero_image_finishes'])
            if intent['services']['visual_review'] else 1+budget['initial_reference'])
        if (plan.get('services') or {}).get('text_vision_requests_upper_bound') != expected_vision:
            raise ValueError('Automatic finishing vision budget differs from its checked policy')
    if plan.get('editing_iteration') is not None:
        from photo_edit_iteration import iteration_request, SCHEMA as REFINEMENT_SCHEMA
        iteration=iteration_request(plan['editing_iteration'])
        proof=plan.get('editing_refinement_authorization') or {}
        if (scope is None or intent['mode']!='existing_reference'
                or proof.get('schema')!=REFINEMENT_SCHEMA or proof.get('iteration')!=iteration
                or proof.get('scope_id')!=scope['scope_id']
                or proof.get('source_sha256')!=plan['inputs']['source']['sha256']
                or proof.get('base_image_sha256')!=plan['inputs']['reference']['sha256']
                or (plan.get('services') or {}).get('text_vision_requests_upper_bound')!=2*iteration['maximum_improvements']):
            raise ValueError('Editing improvement differs from its checked authorization or budget')
    if plan.get('image_budget') != expected:
        raise ValueError('Task image budget differs from its route')
    _validate_refinement_inheritance(plan.get('refinement_inheritance'), intent, plan.get('inputs') or {})
    from creative_direction import validate_reference_contract
    validate_reference_contract(plan.get('reference_contract'), intent, plan.get('inputs') or {})
    resume = (intent.get('regional_refinement') or {}).get('resume_from_run_id')
    proof = plan.get('regional_resume_authorization')
    if resume:
        _validate_resume_proof(proof, intent, plan.get('inputs') or {})
    elif proof is not None:
        raise ValueError('Unexpected regional resume authorization')
    if plan.get('final_repair_only') and budget.get('regional_refinement', 0):
        raise ValueError('Final local repair cannot resume regional image generation')
    reuse = (intent.get('regional_refinement') or {}).get('reuse_generated_from_run_id')
    if reuse:
        _validate_candidate_reuse_proof(plan.get('regional_candidate_reuse_authorization'), intent, plan.get('inputs') or {})
        if plan.get('final_repair_only'):
            raise ValueError('Final repair cannot replace frozen upstream pixels with a rejected candidate')
    elif 'regional_candidate_reuse_authorization' in plan:
        raise ValueError('Unexpected generated candidate reuse proof')
    if intent['constraints'].get('background_mode') == 'enhance_source' and scope is None:
        from creative_direction import validate_direction
        validate_direction(plan.get('creative_direction'), intent,
            source_sha256=((plan.get('inputs') or {}).get('source') or {}).get('sha256'))
    elif 'creative_direction' in plan:
        raise ValueError('Unexpected creative direction without explicit source enhancement')
    if not isinstance(plan.get('plan_id'), str) or len(plan['plan_id']) != 64:
        raise ValueError('Task plan has no checked content identity')
    strategy = plan.get('execution_strategy')
    if strategy not in EXECUTION_STRATEGIES:
        raise ValueError('Task plan has no checked execution strategy')
    normalize_preparation_policy(plan.get('preparation_policy'))
    if plan.get('final_repair_only') is True and plan.get('preparation_policy', 'replay_frozen') != 'replay_frozen':
        raise ValueError('Final local repair must keep its frozen upstream preparation')
    from cos_node_registry import compile_workflow
    if plan.get('layer_workflow') != compile_workflow(intent['mode'],editing_scope=scope,
            visual_review=intent['services']['visual_review'],editing_iteration=plan.get('editing_iteration'),
            editing_autofinish=auto,editing_initialization=initialization,
            initial_variant_count=budget.get('initial_reference',1),shaping_request=shaping):
        raise ValueError('Task layer workflow differs from the current checked node contracts')
    return copy.deepcopy(plan)


def saved_task_intent_for_replanning(plan, intent, *, source_sha256, reference_sha256=None):
    """Replan only a saved task that satisfies the current execution contract."""
    intent = normalize_task_intent(intent)
    validate_task_plan(plan, intent)
    inputs = plan.get('inputs')
    source = inputs.get('source') if isinstance(inputs, dict) else None
    if (not isinstance(source, dict) or not source_sha256 or source.get('sha256') != source_sha256):
        raise ValueError('Saved task source binding differs from the saved version')
    reference = inputs.get('reference')
    if intent['mode'] == 'existing_reference':
        if (not isinstance(reference, dict) or not reference_sha256 or reference.get('sha256') != reference_sha256):
            raise ValueError('Saved task reference binding differs from the saved version')
    elif reference is not None or (intent['mode'] == 'local_portrait' and reference_sha256 is not None):
        raise ValueError('Saved task has an unexpected reference binding')
    return copy.deepcopy(intent)


def _input(root, value, role):
    from raw_support import SOURCE_SUFFIXES
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Select an image for ' + role)
    path = Path(value).expanduser()
    path = (path if path.is_absolute() else root / path).resolve()
    suffixes = SOURCE_SUFFIXES if role == 'source' else {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    if path.suffix.lower() not in suffixes or not path.is_file():
        raise ValueError('Missing or unsupported ' + role + ' image')
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def _region_canvas(source):
    """Read oriented image headers only; no fallback image decode in planning."""
    from raw_support import is_raw
    path = Path(source['path'])
    if is_raw(path):
        import rawpy
        with rawpy.imread(str(path)) as raw:
            width, height = int(raw.sizes.width), int(raw.sizes.height)
            if int(raw.sizes.flip or 0) in {5,6,7,8}:
                width, height = height, width
    else:
        from PIL import Image
        with Image.open(path) as image:
            width, height = image.size
            if image.getexif().get(274) in {5,6,7,8}:
                width, height = height, width
    if width <= 0 or height <= 0:
        raise ValueError('Native regional canvas metadata is unavailable')
    return [width,height]


def _services(request, intent):
    if not intent['services']['text_vision']:
        generation = intent['mode'] == 'create_reference'
        services = {'text_vision': False, 'visual_review': False, 'image_generation': generation, 'offline': not generation,
                'configured': True, 'network_probed': False, 'sent_content': [],
                'text_vision_requests_upper_bound': 0,
                'price': '不调用文本/视觉规划服务；生图费用另计' if generation else '不调用外部服务'}
        if (intent.get('regional_refinement') or {}).get('maximum_submissions'):
            # This opt-in has already passed the local-Qwen provider gate.
            # Existing-reference resume forbids a new global reference, while
            # still making one real local image-model submission possible.
            services.update(image_generation=True, local_image_inference=True,
                image_generation_location='local_machine',
                regional_image_submissions_upper_bound=1,
                price='本机Qwen图像推理；不调用外部文本/视觉服务。模型提交次数与GPU用时见执行回执，不据此推定外部API费用。')
        if intent['constraints'].get('reference_role') == 'source_conditioned_design':
            services.update(local_image_inference=generation, image_generation_location='local_machine',
                global_image_submissions_upper_bound=1 if generation else 0,
                offline=True,
                price=('本机Qwen仅以原片生成设计参考；不调用外部文本/视觉服务，实际模型提交与GPU用时见回执。'
                    if generation else '复用已核验的同源Qwen设计参考；新增生图0次，执行CPU外观映射，不调用外部文本/视觉服务。'))
        return services
    try:
        from vision_transport import _connection
        config, endpoint, _credential, protocol = _connection(**{key: request.get(key)
            for key in ('provider', 'api_base', 'api_url', 'api_key_env')})
        connection = {'configured': True, 'provider': config['registry_id'], 'endpoint': endpoint,
                      'protocol': protocol}
    except (ValueError, RuntimeError, OSError) as exc:
        connection = {'configured': False, 'reason': str(exc)[:400]}
    sent_content = ['原片观察图', '已有参考或生成参考', '创作要求']
    if intent['services']['visual_review']:
        sent_content.append('成片与匹配原尺寸审片窗口')
    return {**connection, 'text_vision': True, 'visual_review': intent['services']['visual_review'],
        'image_generation': intent['mode'] == 'create_reference',
        'offline': False, 'network_probed': False,
        'sent_content': sent_content,
        'text_vision_requests_upper_bound': None,
        'text_vision_budget_basis': 'bounded_stage_attempts; actual board count depends on observed regions',
        'price': '价格未知；实际服务请求与用量保存于任务回执'}


def _storage(root, request, intent):
    """Detect the current Windows owned-asset filename limit before paid work."""
    stem = Path(request['source']).stem
    stamp = '20000101-000000000000'
    parent_id = str(request.get('run_id') or ('run-' + stamp + '-' + stem + '-00000000'))
    sample = str(request.get('sample') or stem)
    child_id = sample + '-r999-' + stamp
    ids = (parent_id, child_id) if intent['mode'] != 'local_portrait' else (parent_id,)
    if any(not value or value in {'.', '..'} or '/' in value or '\\' in value for value in ids):
        raise ValueError('Invalid run or sample identifier')
    directories = [root / 'outputs/runs' / value / ('inputs/design-reconstruction'
        if intent['mode'] != 'local_portrait' else 'automatic-native-retouch') for value in ids]
    longest = max(len(str(path).encode('utf-16-le')) // 2 for path in directories)
    name = 73 if intent['mode'] != 'local_portrait' else len('worker-result.json')
    required = longest + 1 + name + len('.rewrite.tmp')
    return {'ready': required <= 240, 'workspace': str(root), 'path_limit_utf16': 240,
            'required_path_characters': required, 'basis': 'current_owned_asset_and_atomic_rewrite_paths',
            'write_performed': False}


@saved_parent_reads
def plan_task(root, body):
    """Read metadata/config/files only. No directories, jobs, inference or network."""
    from current_contract import validate_current_request
    from current_model_registry import registry_status
    from generation_prompt import DEFAULT_PROMPT
    root = Path(root).resolve()
    if not isinstance(body, dict) or set(body) - REQUEST_FIELDS:
        raise ValueError('Unsupported creative task request fields')
    validate_current_request(body)
    for key, current in (('algorithm_profile', 'dinov3_vitl16'),
                         ('material_finish_profile', 'owner_adaptive_multiscale_skin_v2')):
        if key in body and body[key] != current:
            raise ValueError('Only the current algorithm is supported: ' + key)
    intent = normalize_task_intent(body.get('creative_task'))
    if intent['mode']=='create_reference':
        if any(key in body for key in ('creative_direction','regional_resume_authorization','regional_candidate_reuse_authorization')):
            raise ValueError('Retired creation direction or regional proof; choose a current editing scope')
        body=copy.deepcopy(body)
        body.setdefault('editing_request',{})
    if intent['constraints'].get('reference_role') in {'same_photo_finish', 'source_conditioned_design'} and (
            'regional_resume_authorization' in body or 'regional_candidate_reuse_authorization' in body):
        raise ValueError('Explicit appearance reference roles cannot inherit regional resume or generated-candidate proofs')
    if (intent['constraints'].get('background_mode') == 'enhance_source'
            and intent['services']['text_vision'] and 'editing_request' not in body):
        raise ValueError('Source enhancement currently uses a checked offline design; external planning is not enabled on this route')
    strategy = body.get('execution_strategy')
    if strategy is None:
        strategy = 'layered_reference_transfer'
    if strategy not in EXECUTION_STRATEGIES:
        raise ValueError('Unsupported execution strategy: ' + str(strategy))
    final_repair = body.get('final_repair_only', False)
    if type(final_repair) is not bool or (final_repair and intent['mode'] != 'existing_reference'):
        raise ValueError('Final local repair requires an existing-reference task')
    policy = normalize_preparation_policy(body.get('preparation_policy'),
        default='replay_frozen' if final_repair else 'prepare_current')
    if final_repair and policy != 'replay_frozen':
        raise ValueError('Final local repair must keep its frozen upstream preparation')
    if final_repair and (intent.get('regional_refinement') or {}).get('maximum_submissions'):
        raise ValueError('Final local repair cannot grant regional image generation')
    if final_repair and (intent.get('regional_refinement') or {}).get('reuse_generated_from_run_id'):
        raise ValueError('Final local repair must replay its saved upstream pixels')
    local, supplied = intent['mode'] == 'local_portrait', intent['mode'] == 'existing_reference'
    scope = None
    if 'editing_request' in body:
        from photo_editing import editing_scope
        scope = editing_scope(body['editing_request'], local=local,
            preserve_background=intent['constraints']['preserve_background'], no_effects=intent['constraints']['no_effects'])
        if supplied and 'editing_iteration' not in body and 'shaping_request' not in body:
            raise ValueError('Observed scoped editing currently starts from the original; imported-reference planning is not yet enabled')
        if intent['constraints'].get('reference_role'):
            raise ValueError('Scoped generated appearance cannot also use a source-coefficient reference role')
        if intent.get('regional_refinement'):
            raise ValueError('Scoped generation cannot inherit the retired source-native regional editing route')
        if body.get('variant_styles'):
            raise ValueError('Scoped candidates share one observed design; select the visual style in editing_request')
        if not local and (body.get('denoise_strength') or body.get('manual_layers')
                or body.get('native_source_retouch') or body.get('appearance_revision')):
            raise ValueError('Source-space pixel controls cannot be applied to generated appearance')
    shaping = None
    if 'shaping_request' in body:
        from photo_shaping import normalize_request
        shaping = normalize_request(body['shaping_request'])
        apply = shaping['action'] == 'apply'
        if (scope is None or not supplied or final_repair or 'editing_iteration' in body
                or 'editing_autofinish' in body or body.get('variant_count',1) != 1
                or body.get('defer_reference_selection',False)
                or intent['services'] != {'text_vision':apply,'visual_review':apply}):
            raise ValueError('Shaping needs one saved typed parent and its separate apply/restore service policy')
    iteration=None
    if 'editing_iteration' in body:
        from photo_edit_iteration import iteration_request
        if scope is None or not supplied or not intent['services']['text_vision'] or not intent['services']['visual_review']:
            raise ValueError('Improvement requires a scoped saved reference, visual decisions and independent review')
        iteration=iteration_request(body['editing_iteration'])
    autofinish = autofinish_policy(body.get('editing_autofinish'), intent, scope,
                                  supplied='editing_autofinish' in body)
    if autofinish is not None and iteration is not None:
        raise ValueError('Automatic finishing and a manual improvement task are separate budgets')
    if body.get('generate') is True and (local or supplied):
        raise ValueError('This route forbids all image generation')
    if local and body.get('reference'):
        raise ValueError('Local portrait tasks do not consume a reference')
    if not supplied and body.get('reference'):
        raise ValueError('Use the existing-reference route for a supplied reference')
    request = {key: copy.deepcopy(value) for key, value in body.items()
               if key not in {'creative_task', 'plan_id', 'generate', 'local_only', 'complete_post'}}
    if shaping is not None and shaping['action']=='restore':
        from photo_shaping import frozen_restore_source
        inputs={'source':frozen_restore_source(root,body.get('parent_run_id'),body.get('source'),scope)}
    else:
        inputs = {'source': _input(root, body.get('source'), 'source')}
    request['source'] = inputs['source']['path']
    if ((intent.get('regional_refinement') or {}).get('maximum_submissions')
            or (intent.get('regional_refinement') or {}).get('reuse_generated_from_run_id')):
        from reference_provider import provider_config
        regional_provider = provider_config(**{key: body.get(key) for key in
            ('provider','model','api_base','api_url','api_key_env')})
        if regional_provider.get('kind') != 'comfyui' or not str(regional_provider.get('registry_id','')).startswith('qwen_image'):
            raise ValueError('Native regional editing currently requires a local Qwen ComfyUI provider')
        canvas = _region_canvas(inputs['source'])
        for region in intent['regional_refinement']['regions']:
            if region['core_box'][2] > canvas[0] or region['core_box'][3] > canvas[1]:
                raise ValueError('Regional core lies outside the native source canvas')
        inputs['source']['regional_native_size'] = canvas
    if supplied:
        inputs['reference'] = _input(root, body.get('reference'), 'reference')
        request['reference'] = inputs['reference']['path']
        if shaping is not None:
            from photo_shaping import authorize,image_submissions
            from photo_request_budget import manual_provider_target
            shaping_authorization = authorize(root,parent_run_id=body.get('parent_run_id'),
                source=inputs['source'],reference=inputs['reference'],scope=scope,request=shaping,
                provider_target=manual_provider_target(request,image_generation=bool(image_submissions(shaping)))
                    if shaping['action']=='apply' else None)
            request['shaping_request'] = copy.deepcopy(shaping)
        if iteration is not None:
            from photo_edit_iteration import authorize
            from photo_request_budget import manual_provider_target
            refinement_authorization=authorize(root,parent_run_id=body.get('parent_run_id'),
                source=inputs['source'],reference=inputs['reference'],scope=scope,iteration=iteration,
                provider_target=manual_provider_target(request,image_generation=
                    iteration.get('maximum_image_submissions',iteration['maximum_improvements'])>0))
            request['editing_iteration']=copy.deepcopy(iteration)
        if intent['constraints'].get('reference_role') == 'source_conditioned_design':
            from creative_direction import verify_source_conditioned_reference
            inputs['reference']['source_conditioned_binding'] = verify_source_conditioned_reference(
                root, inputs['source']['sha256'], inputs['reference'])
    from creative_direction import reference_contract_for
    reference_contract = reference_contract_for(intent, inputs)
    count = body.get('variant_count', 1)
    if intent['constraints'].get('background_mode') == 'enhance_source' and scope is None:
        if count != 1 or body.get('variant_styles') or body.get('defer_reference_selection', False):
            raise ValueError('Source enhancement supports one checked candidate only; variant styles and deferred selection are unsupported')
    input_prompt = str(body.get('input_prompt') or body.get('prompt') or DEFAULT_PROMPT)
    selector = intent['constraints'].get('refinement_profile')
    if selector is not None:
        from creative_direction import refinement_overrides
        from preset_store import get_preset, get_refinement_profile
        profile = get_refinement_profile(selector)
        actual = refinement_overrides(input_prompt, body.get('character_notes', ''))
        if 'overrides' in selector and selector['overrides'] != actual:
            raise ValueError('Profile override context differs from actual prompt/character notes')
        selector['overrides'] = actual
        intent = normalize_task_intent(intent)
        if body.get('preset') not in (None, '', profile['preset_id']):
            raise ValueError('Recommended profile differs from the selected artistic preset')
        preset = get_preset(profile['preset_id'])
        if 'preset_snapshot' in body and body['preset_snapshot'] != preset:
            raise ValueError('Recommended preset snapshot differs from the bundled product profile')
        request.update(preset=profile['preset_id'], preset_snapshot=copy.deepcopy(preset))
    budget = image_budget(intent, count, scope,iteration,autofinish,shaping)
    brief = task_brief(intent)
    request.update(task_intent=intent, creative_task=intent, execution_strategy=strategy, preparation_policy=policy,
        input_prompt=input_prompt, prompt=input_prompt, variant_count=count,
        generate=intent['mode'] == 'create_reference', local_only=local, complete_post=not local,
        cosplay_correction_budget=budget['reference_correction'],
        preserve_project_selection=bool(body.get('preserve_project_selection', True)))
    # Keep the user's editable prompt separate from deterministic task constraints.
    request['task_brief'] = brief
    if scope is not None:
        request['editing_request'] = {key:scope[key] for key in ('work_type','style','target_setting','scene_target','language')}
    if autofinish is not None:
        request['editing_autofinish'] = copy.deepcopy(autofinish)
    direction = None
    resume_authorization = None
    reuse_authorization = None
    if intent['constraints'].get('background_mode') == 'enhance_source' and scope is None:
        from creative_direction import compile_direction
        direction = compile_direction(intent, source_sha256=inputs['source']['sha256'],
            character_name=body.get('character_name', ''), character_notes=body.get('character_notes', ''),
            input_prompt=input_prompt)
        if 'creative_direction' in body and body['creative_direction'] != direction:
            raise ValueError('Supplied creative direction differs from the checked task')
        request['creative_direction'] = direction
        request['prompt'] = request['effective_prompt'] = direction['prompt']
        if (intent.get('regional_refinement') or {}).get('resume_from_run_id'):
            resume_authorization = verify_regional_resume(root, intent, inputs, direction)
            if ('regional_resume_authorization' in body
                    and body['regional_resume_authorization'] != resume_authorization):
                raise ValueError('Supplied regional resume proof differs from current failed evidence')
            request['regional_resume_authorization'] = copy.deepcopy(resume_authorization)
        if (intent.get('regional_refinement') or {}).get('reuse_generated_from_run_id'):
            reuse_authorization = verify_regional_candidate_reuse(root, intent, inputs, direction)
            if ('regional_candidate_reuse_authorization' in body
                    and body['regional_candidate_reuse_authorization'] != reuse_authorization):
                raise ValueError('Supplied candidate reuse proof differs from current returned evidence')
            request['regional_candidate_reuse_authorization'] = copy.deepcopy(reuse_authorization)
    elif 'creative_direction' in body:
        raise ValueError('Creative direction requires explicit source enhancement')
    if resume_authorization is None and 'regional_resume_authorization' in body:
        raise ValueError('Regional resume proof requires an explicit unsubmitted continuation')
    if reuse_authorization is None and 'regional_candidate_reuse_authorization' in body:
        raise ValueError('Generated candidate proof requires an explicit zero-generation reuse task')
    # Node-specific checks apply only to enabled production capabilities.
    from task_model_preflight import inspect_task_models,uses_generated_appearance,generated_registry_status
    generated_appearance=uses_generated_appearance(intent['mode'],scope)
    restoring = shaping is not None and shaping['action']=='restore'
    face_apply = shaping is not None and shaping['action']=='apply' and 'face' in shaping['parts']
    models = ({'configured':True,'configuration_valid':True,'paths':[],'config':{},'registry_path':None,
               'scope':'no_model_required_for_frozen_restore'} if restoring else
        generated_registry_status(root,body.get('model_registry')) if generated_appearance and not face_apply else
        registry_status(root, registry_path=body.get('model_registry'),required_keys=(), required_profiles=()))
    beauty_requested = (scope is None and (bool((body.get('appearance_revision') or {}).get('portrait_beauty'))
        or (not local and strategy == 'layered_reference_transfer')))
    installation = ({'ready':True,'blocking_reasons':[],'required_path_keys':[],
        'scope':'frozen_pixel_restore_no_model_requirements','checks':[]} if restoring else
        inspect_task_models(models.get('config', {}), mode=intent['mode'],
        execution_strategy=strategy, complete_post=not local,
        use_beautygrpo=beauty_requested,editing_scope=scope,shaping_request=shaping))
    if generated_appearance and not restoring:
        from runtime_preflight import preflight_for_run,RuntimePreflightError
        try:
            installation['desktop_runtime']=preflight_for_run(root,registry_path=body.get('model_registry'),
                mode=intent['mode'],execution_strategy=strategy,editing_scope=scope,require_local_models=False)
        except RuntimePreflightError as error:
            installation['ready']=False
            installation['blocking_reasons']=[*installation['blocking_reasons'],str(error)]
    required_keys = set(installation.get('required_path_keys') or ())
    paths = [{key: row[key] for key in ('key', 'label', 'kind', 'available', 'status')}
             for row in models['paths'] if row['key'] in required_keys]
    blockers = []
    if scope is not None and not local and not restoring and not intent['services']['text_vision']:
        blockers.append('当前图像编辑设计需要启用视觉观察与规划；不能把离线模板称为已观察原图。')
    storage = _storage(root, request, intent)
    if not storage['ready']:
        blockers.append('工作区或任务名称过长，无法安全保存完整工程；请使用较短的工作区路径或任务名称。')
    if ((generated_appearance and models.get('error')) or
            ((not generated_appearance or face_apply) and not models['configured'])):
        blockers.append('当前分析模型配置不可用：' + str(models.get('error') or '尚未配置'))
    blockers.extend('缺少 ' + row['label'] for row in paths if not row['available'])
    blockers.extend(installation['blocking_reasons'])
    services = _services(request, intent)
    if scope is not None:
        services['text_vision_requests_upper_bound'] = (2*iteration['maximum_improvements'] if iteration is not None else
            0 if local else 2+2*count+2*sum((autofinish or {}).get(key,0) for key in ('maximum_improvements','maximum_zero_image_finishes'))
            if intent['services']['visual_review'] else 1+count)
        services['text_vision_budget_basis'] = ('separate_checked_iteration_decision_and_review_reservations'
            if iteration is not None else 'shared_design_and_source_baseline_review_then_bounded_initial_and_improvement_operations')
    if shaping is not None:
        services['text_vision_requests_upper_bound'] = 0 if restoring else 3
        services['text_vision_budget_basis'] = ('exact_frozen_restore' if restoring else
            'checked_unshaped_input_review_decision_and_candidate_review')
    if not services['configured']:
        blockers.append('文本/视觉服务配置不可用：' + str(services.get('reason') or '尚未配置'))
    if (shaping is not None and budget['shaping']>0) or intent['mode'] == 'create_reference' or (iteration is not None and
            iteration.get('maximum_image_submissions',iteration['maximum_improvements'])>0):
        from reference_provider import provider_health
        health = provider_health(**{key: request.get(key) for key in
            ('provider', 'api_base', 'api_url', 'model', 'api_key_env')})
        services['generation_configured'] = bool(health.get('ready'))
        services['image_generation']=True
        if not health.get('ready'):
            blockers.append('生图服务未就绪：' + str(health.get('reason') or health.get('status')))
    defer_selection = body.get('defer_reference_selection', False)
    if type(defer_selection) is not bool:
        raise ValueError('defer_reference_selection must be a boolean')
    registry_file = Path(models['registry_path']) if models.get('registry_path') else None
    config_identity = {'path': str(registry_file) if registry_file else None,
        'sha256': sha(registry_file) if registry_file and registry_file.is_file() else None}
    capabilities = {'models': paths, 'configuration_verified': models['configured'],
        'native_models_required':not generated_appearance or face_apply,
        'native_registry_metadata_valid':models.get('configuration_valid'),
        'model_preflight': installation,
        'storage_preflight': storage,
        'model_profiles': {k: v for k, v in models.get('config', {}).items() if k.endswith('profile')},
        'registry_identity': config_identity, 'verification_scope': 'installation_structure_small_file_and_recorded_identity',
        'inference_verified': False}
    stage_names = (['原片解码与本地精修', '高清保存', '人工审片与导出'] if local else
        [('原片与参考成对设计' if supplied else '角色与本图设计')
         if intent['services']['text_vision'] else '用户要求、所选风格与源图绑定',
         '复用已有参考' if supplied else '创作风格参考', '原片精修与原生人物基线',
         '仿色、材质与明暗协调', '背景、接触与已拥有的特效合成',
         *(['整图和匹配原尺寸审片', '有资格的一轮本地修复与复审']
           if intent['services']['visual_review'] else []), '人工审片、重开与导出'])
    if direction is not None:
        stage_names[stage_names.index('背景、接触与已拥有的特效合成')] = '保留原景结构、原生背景与有据角色效果合成'
        if budget.get('regional_refinement', 0):
            stage_names.insert(stage_names.index('仿色、材质与明暗协调'), '一次许可背景局部精修；失败保留原版本')
        elif reuse_authorization is not None:
            stage_names.insert(stage_names.index('仿色、材质与明暗协调'), '复用已返回的单区候选并重新核验合成；新增生图0次')
    if reference_contract is not None:
        index = stage_names.index('仿色、材质与明暗协调')
        verification = ('执行时核验用户声明的同照参考配准与有效范围' if reference_contract['role'] == 'same_photo_finish'
            else '执行时核验Qwen同源生成来源、配准与有效范围')
        stage_names[index:index+1] = [verification,
            'CPU影调与色彩归一（条件启用）', '源图引导的局部低频色映射系数场校正（条件启用）']
    if scope is not None and not local:
        stage_names=['观察原图并形成具体编辑设计','准备原图原生精度基线',
            *(['评价原片基线：自然可信度与创作缺口'] if intent['services']['visual_review'] else []),
            '模型选择并执行首次摄影调整或图像编辑',
            *(['独立审片与有界条件纠偏'] if intent['services']['visual_review'] else []),
            '保存较好完整版本与母版','人工审片与导出']
    plan = {'schema_version': PLAN_SCHEMA, 'mode': intent['mode'], 'label': MODES[intent['mode']],
        'intent': intent, 'inputs': inputs, 'ready': not blockers, 'blocking_reasons': blockers,
        'stages': [{'id': str(i+1), 'label': label} for i, label in enumerate(stage_names)],
        'capabilities': capabilities, 'services': services, 'image_budget': budget,
        'generation_submissions_upper_bound': sum(budget.values()),
        'candidate_selection': {'required': bool(intent['mode'] == 'create_reference' and defer_selection),
                               'default_variant_count': count if intent['mode'] == 'create_reference' else 0,
                               'selection_action': 'select_variant' if intent['mode'] == 'create_reference' and defer_selection else None},
        'outputs': ['原尺寸PNG', '真实TIFF16/32母版', '可继续编辑的产品工程'],
        'limitations': ['自动审片只证明实际观察范围，人工最终批准单独记录。',
            '自动投影阴影、镜面或湿地反射生成尚未支持。',
            '共享创作要求不等于已验证的套图风格一致算法。'],
        'estimated_seconds': None, 'estimate_basis': '没有足够的当前硬件任务统计',
        'preflight_network_requests': 0, 'preflight_model_inferences': 0, 'preflight_jobs_created': 0}
    from cos_node_registry import compile_workflow
    plan['execution_strategy'] = strategy
    if scope is not None:
        plan['editing_scope'] = scope
        plan['editing_request'] = copy.deepcopy(request['editing_request'])
    if autofinish is not None:
        plan['editing_autofinish'] = copy.deepcopy(autofinish)
    if intent['mode']=='create_reference':
        from scoped_workflow import SOURCE_NATIVE_INITIALIZATION
        plan['editing_initialization']=copy.deepcopy(SOURCE_NATIVE_INITIALIZATION)
    if iteration is not None:
        plan['label']='按独立审片改进'
        plan['editing_iteration']=copy.deepcopy(iteration)
        plan['editing_refinement_authorization']=refinement_authorization
        plan['stages']=[{'id':str(i+1),'label':label} for i,label in enumerate((
            '核验已保存成片与独立审片','模型选择主要纠偏工具',
            '执行模型选择的摄影调整或图像编辑','独立复审并保留较好完整版本','保存子版本与导出'))]
    if shaping is not None:
        from photo_shaping import execution_method
        plan.update(label='恢复形体调整前版本' if restoring else '可撤销形体调整',
            shaping_request=copy.deepcopy(shaping),shaping_authorization=shaping_authorization)
        labels = (('核验并拥有调整前母版','原样恢复并保存子版本') if restoring else
            ('核验并拥有调整前母版','观察调整前成片','定位获准部位并选择局部工具',
             '从同一母版计算轮廓形变' if execution_method(shaping)=='face_contour_deform' else '生成一次局部调整',
             '独立审片','保存待采用候选'))
        plan['stages']=[{'id':str(i+1),'label':label} for i,label in enumerate(labels)]
    plan['preparation_policy'] = policy
    plan['final_repair_only'] = final_repair
    plan['layer_workflow'] = compile_workflow(intent['mode'],editing_scope=scope,
        visual_review=intent['services']['visual_review'],editing_iteration=iteration,
        editing_autofinish=autofinish,editing_initialization=plan.get('editing_initialization'),
        initial_variant_count=count,shaping_request=shaping)
    if direction is not None:
        plan['creative_direction'] = copy.deepcopy(direction)
    if reference_contract is not None:
        plan['reference_contract'] = reference_contract
    if selector is not None and supplied:
        plan['refinement_inheritance'] = _refinement_inheritance(root, intent, inputs, body.get('parent_run_id'))
    if resume_authorization is not None:
        plan['regional_resume_authorization'] = copy.deepcopy(resume_authorization)
    if reuse_authorization is not None:
        plan['regional_candidate_reuse_authorization'] = copy.deepcopy(reuse_authorization)
    execution = {key: value for key, value in request.items() if key not in {'async', 'sync'}}
    plan['plan_id'] = digest({'plan': plan, 'request': execution})
    public_request = {key: value for key, value in request.items()
                      if key not in {'task_intent', 'task_brief', 'cosplay_correction_budget'}}
    return {'ok': True, 'plan': plan, 'request': public_request}


def execution_request(root, body):
    checked = plan_task(root, body)
    plan = checked['plan']
    if body.get('plan_id') != plan['plan_id']:
        raise ValueError('Inputs, creative requirements or configuration changed; check the task plan again')
    if not plan['ready']:
        raise ValueError('Task is blocked: ' + '; '.join(plan['blocking_reasons']))
    request = checked['request']
    request['task_intent'] = request.pop('creative_task')
    request['cosplay_correction_budget'] = plan['image_budget']['reference_correction']
    request['task_plan'] = plan
    return request


def plan_batch(root, body):
    if not isinstance(body, dict) or set(body) - SHARED_FIELDS - {'sources', 'items', 'plan_id'}:
        raise ValueError('A batch shares creative requirements, never another image\'s geometry or masks')
    if ('sources' in body) == ('items' in body):
        raise ValueError('Provide either sources or explicit source/reference pairs')
    rows = body.get('items') if 'items' in body else [{'source': value} for value in body.get('sources', [])]
    if not isinstance(rows, list) or not 1 <= len(rows) <= 50:
        raise ValueError('Provide between 1 and 50 batch images')
    shared = {key: copy.deepcopy(value) for key, value in body.items() if key in SHARED_FIELDS}
    intent = normalize_task_intent(shared.get('creative_task'))
    items = []
    seen = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) - {'source', 'reference', 'sample'}:
            raise ValueError('Batch items contain only their own source, reference and sample')
        checked = plan_task(root, {**shared, **row, 'preserve_project_selection': True})
        key = checked['plan']['inputs']['source']['sha256']
        if key in seen:
            raise ValueError('Duplicate source image in the batch')
        seen.add(key)
        items.append({'index': index, **checked})
    if 'regional_refinement' in intent and (intent['regional_refinement']['maximum_submissions']
            or intent['regional_refinement'].get('reuse_generated_from_run_id')):
        raise ValueError('Native regional geometry cannot be shared across batch sources')
    budget = {key: sum(row['plan']['image_budget'].get(key, 0) for row in items)
              for key in dict.fromkeys(key for row in items for key in row['plan']['image_budget'])}
    plan = {'schema_version': 'cosplay_batch_task_plan.v1', 'shared_task': intent,
        'items': items, 'ready': all(row['plan']['ready'] for row in items),
        'image_budget_total': budget, 'per_image_geometry_shared': False,
        'automatic_style_consistency_claimed': False}
    plan['plan_id'] = digest(plan)
    return {'ok': True, 'batch_plan': plan, 'request': copy.deepcopy(body)}
