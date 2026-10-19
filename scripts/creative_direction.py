"""Deterministic, user-supplied art direction; no image analysis or services."""
from __future__ import annotations

import hashlib
import json
import re

SCHEMA = 'cosplay_creative_direction.v1'
MARKER = '\n\nCHECKED SOURCE ENHANCEMENT DIRECTION\n'
HARD_LOCKS = ('identity', 'facial_geometry', 'makeup', 'expression', 'pose',
              'hands', 'costume', 'hair_silhouette', 'camera', 'perspective')


def reference_contract_for(intent, inputs):
    """Describe declared use only; no photograph recognition or registration."""
    role = (intent.get('constraints') or {}).get('reference_role')
    if role is None:
        return None
    if role == 'source_conditioned_design':
        if (intent.get('mode') not in {'create_reference', 'existing_reference'}
                or intent['constraints'].get('background_mode') != 'enhance_source'
                or 'regional_refinement' in intent):
            raise ValueError('Source-conditioned design requires a source-enhancement reference task without regional requests')
        source_sha = (inputs.get('source') or {}).get('sha256')
        if not isinstance(source_sha, str) or not re.fullmatch('[0-9a-f]{64}', source_sha):
            raise ValueError('Source-conditioned design requires the checked original source')
        reference = inputs.get('reference')
        verified = None
        if intent['mode'] == 'create_reference':
            if reference is not None:
                raise ValueError('Source-conditioned generation cannot consume a supplied finish image')
        else:
            verified = (reference or {}).get('source_conditioned_binding')
            if (not isinstance(verified, dict) or verified.get('source_sha256') != source_sha
                    or verified.get('reference_sha256') != (reference or {}).get('sha256')):
                raise ValueError('Continuing source-conditioned design requires verified same-source generated provenance')
            for key in ('reference_sha256', 'reference_provenance_sha256'):
                if not isinstance(verified.get(key), str) or not re.fullmatch('[0-9a-f]{64}', verified[key]):
                    raise ValueError('Generated reference provenance identity is missing')
            if not isinstance(verified.get('provider_prompt_id'), str) or not re.fullmatch('[A-Za-z0-9_-]{1,128}', verified['provider_prompt_id']):
                raise ValueError('Generated reference has no real provider prompt identity')
        return {'schema': 'source_conditioned_design_reference.v1', 'role': role,
            'source_sha256': source_sha, 'reference_sha256': verified['reference_sha256'] if verified else None,
            'geometry_owner': 'source', 'texture_owner': 'source_native',
            'transfer_scope': 'tone_and_source_guided_local_colour_coefficients',
            'reference_rgb_compositing_allowed': False, 'registration_status': 'not_checked',
            'reference_binding': 'generated_source_verified' if verified else 'after_generation',
            'reference_provenance_sha256': verified['reference_provenance_sha256'] if verified else None,
            'provider_prompt_id': verified['provider_prompt_id'] if verified else None}
    if (role != 'same_photo_finish' or intent.get('mode') != 'existing_reference'
            or intent['constraints'].get('background_mode') != 'enhance_source'
            or 'regional_refinement' in intent):
        raise ValueError('Same-photo finish requires an existing source-enhancement reference without regional requests')
    hashes = {}
    for key in ('source', 'reference'):
        value = (inputs.get(key) or {}).get('sha256')
        if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
            raise ValueError('Same-photo finish requires both checked input hashes')
        hashes[key+'_sha256'] = value
    return {'schema': 'same_photo_finish_reference.v1', 'role': role, **hashes,
        'geometry_owner': 'source', 'texture_owner': 'source_native',
        'transfer_scope': 'tone_and_source_guided_local_colour_coefficients',
        'reference_rgb_compositing_allowed': False, 'registration_status': 'not_checked'}


def validate_reference_contract(contract, intent, inputs):
    expected = reference_contract_for(intent, inputs)
    if expected is None:
        if contract is not None:
            raise ValueError('Reference-use contract requires explicit same_photo_finish intent')
        return None
    if (not isinstance(contract, dict) or contract != expected
            or any(type(contract.get(key)) is not type(value) for key, value in expected.items())):
        raise ValueError('Same-photo reference contract differs from its checked input ownership and scope')
    return expected


def verify_source_conditioned_reference(root, source_sha256, reference_record):
    """Read actual Qwen source-only generation evidence; never infer alignment."""
    from pathlib import Path
    from asset_identity import sha256_file
    root = Path(root).resolve()
    if not isinstance(source_sha256, str) or not re.fullmatch('[0-9a-f]{64}', source_sha256):
        raise ValueError('Generated art direction needs a checked original source hash')
    if not isinstance(reference_record, dict) or not isinstance(reference_record.get('path'), str):
        raise ValueError('Generated reference record is missing')
    def actual_file(record):
        if not isinstance(record, dict) or not isinstance(record.get('path'), str):
            raise ValueError('Source-conditioned generation evidence is missing an asset')
        path = Path(record['path'])
        path = path if path.is_absolute() else root/path
        if not path.is_file():
            raise ValueError('Source-conditioned generation asset is unavailable')
        actual_sha = sha256_file(path)
        if record.get('sha256') != actual_sha:
            raise ValueError('Source-conditioned generation asset differs from its provenance')
        return {'path': str(path.resolve()), 'sha256': actual_sha}
    reference = actual_file(reference_record)
    provenance_path = reference_record.get('provenance')
    if provenance_path is None:
        path = Path(reference['path'])
        provenance_path = str(path.with_name(path.stem+'.provenance.json'))
    if not isinstance(provenance_path, str):
        raise ValueError('Generated reference provenance path must be explicit text')
    provenance_path = Path(provenance_path)
    provenance_path = provenance_path if provenance_path.is_absolute() else root/provenance_path
    try:
        raw = provenance_path.read_bytes()
        provenance = json.loads(raw)
    except (OSError, ValueError) as error:
        raise ValueError('An existing reference requires actual source-conditioned generation provenance') from error
    if not isinstance(provenance, dict):
        raise ValueError('Generated reference provenance is not an object')
    provider = provenance.get('provider')
    contract = provenance.get('provider_contract') or {}
    comfy = contract.get('comfyui') or {}
    events = (contract.get('transport') or {}).get('events') or []
    submitted = [event for event in events if event.get('event') == 'comfyui_submitted']
    completed = [event for event in events if event.get('event') == 'comfyui_completed']
    prompt_id = provenance.get('generation_id')
    api_input = (contract.get('request') or {}).get('api_input') or {}
    condition_inputs = comfy.get('condition_inputs') or []
    graph = (comfy.get('request') or {}).get('prompt') or {}
    encoders = [row.get('inputs') or {} for row in graph.values() if row.get('class_type') == 'CosplayTextEncodeQwenImage21']
    loaders = [(key,row.get('inputs') or {}) for key,row in graph.items() if row.get('class_type') == 'LoadImage']
    if (provenance.get('status') != 'reference_candidate'
            or provider not in {'qwen_image_2_1', 'qwen_image_2_1_fast', 'qwen_image_2_1_uncensored', 'qwen_image_2_1_uncensored_fast'}
            or (provenance.get('source') or {}).get('sha256') != source_sha256
            or not isinstance(prompt_id, str) or not re.fullmatch('[A-Za-z0-9_-]{1,128}', prompt_id)
            or comfy.get('prompt_id') != prompt_id or (contract.get('response') or {}).get('id') != prompt_id
            or len(submitted) != 1 or submitted[0].get('prompt_id') != prompt_id
            or len(completed) != 1 or completed[0].get('prompt_id') != prompt_id
            or (comfy.get('execution_status') or {}).get('completed') is not True
            or (comfy.get('execution_status') or {}).get('status_str') != 'success'
            or len(condition_inputs) != 1 or condition_inputs[0].get('role') != 'source'
            or len(encoders) != 1 or len(loaders) != 1
            or [key for key in encoders[0] if key.startswith('images.')] != ['images.image_1']
            or encoders[0].get('images.image_1') != [loaders[0][0],0]
            or loaders[0][1].get('image') != (condition_inputs[0].get('upload') or {}).get('name')
            or (submitted[0].get('request') or {}).get('prompt') != graph
            or encoders[0].get('prompt') != provenance.get('prompt')
            or (contract.get('request') or {}).get('prompt') != provenance.get('prompt')):
        raise ValueError('Reference is not a proven one-source Qwen-generated design')
    if actual_file(provenance.get('asset')) != reference:
        raise ValueError('Generated art-direction output does not match the actual reference')
    source = provenance['source']
    if actual_file(source)['sha256'] != source_sha256:
        raise ValueError('Generation original source bytes changed')
    uploaded = actual_file(condition_inputs[0])
    if uploaded != actual_file(api_input) or (contract.get('request') or {}).get('input_sha256') != uploaded['sha256']:
        raise ValueError('Actual Qwen image input differs from its sole recorded source condition')
    if api_input.get('derived') is True:
        if actual_file(api_input.get('source'))['sha256'] != source_sha256:
            raise ValueError('Generation derivative is not bound to the original source')
    elif uploaded['sha256'] != source_sha256:
        raise ValueError('Generation used a different photograph instead of the original source')
    return {'source_sha256': source_sha256, 'reference_sha256': reference['sha256'],
        'reference_provenance_sha256': hashlib.sha256(raw).hexdigest(), 'provider_prompt_id': prompt_id,
        'provenance': {'path': str(provenance_path.resolve()), 'sha256': hashlib.sha256(raw).hexdigest()},
        'input_image': uploaded, 'conditioning_policy': 'one_original_source_photo',
        'registration_status': 'not_checked'}


def bind_source_conditioned_reference(root, plan, reference_record):
    """Bind returned source-only art direction without mutating the checked plan."""
    import copy
    intent, inputs = plan.get('intent') or {}, plan.get('inputs') or {}
    if (intent.get('constraints') or {}).get('reference_role') != 'source_conditioned_design':
        raise ValueError('Source-conditioned reference binding requires that explicit checked role')
    descriptor = validate_reference_contract(plan.get('reference_contract'), intent, inputs)
    binding = verify_source_conditioned_reference(root, descriptor['source_sha256'], reference_record)
    result = copy.deepcopy(descriptor)
    result.update(reference_sha256=binding['reference_sha256'], reference_binding='generated_source_verified',
        reference_provenance_sha256=binding['reference_provenance_sha256'], provider_prompt_id=binding['provider_prompt_id'])
    if intent['mode'] == 'existing_reference' and result != descriptor:
        raise ValueError('Generated reference identity changed after checking the continuation plan')
    return result


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf8')).hexdigest()


def normalize_regional_refinement(value, *, mode, background_mode):
    if (not isinstance(value, dict)
            or set(value) - {'maximum_submissions', 'coordinate_space', 'regions', 'resume_from_run_id', 'reuse_generated_from_run_id'}):
        raise ValueError('Unsupported regional refinement contract')
    maximum = value.get('maximum_submissions')
    if type(maximum) is not int or maximum not in (0, 1):
        raise ValueError('Regional refinement permits at most one explicit submission')
    if value.get('coordinate_space', 'source_canvas') != 'source_canvas':
        raise ValueError('Regional refinement coordinates must use the native source_canvas')
    regions = value.get('regions')
    reuse = value.get('reuse_generated_from_run_id')
    if 'reuse_generated_from_run_id' in value:
        if (not isinstance(reuse, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', reuse)
                or mode != 'existing_reference' or maximum != 0 or background_mode != 'enhance_source'
                or 'resume_from_run_id' in value):
            raise ValueError('Generated candidate reuse requires one retained source region and zero new submissions')
    if not isinstance(regions, list) or len(regions) != (1 if reuse is not None else maximum):
        raise ValueError('Regional refinement must bind exactly its authorized region count')
    resume = value.get('resume_from_run_id')
    if 'resume_from_run_id' in value:
        if (not isinstance(resume, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', resume)
                or mode != 'existing_reference' or maximum != 1):
            raise ValueError('Regional resume requires one explicit existing-reference allowance and safe failed run ID')
    if maximum and (mode not in {'create_reference', 'existing_reference'} or background_mode != 'enhance_source'):
        raise ValueError('Regional generation requires an explicit source-enhancement reference task')
    normalized = []
    for region in regions:
        if (not isinstance(region, dict) or not {'id', 'core_box'} <= set(region)
                or set(region) - {'id', 'core_box', 'instruction'}):
            raise ValueError('A region requires id, native core_box and optional instruction')
        identifier, box = region['id'], region['core_box']
        if not isinstance(identifier, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', identifier):
            raise ValueError('Regional refinement id must be a short safe identifier')
        if (not isinstance(box, list) or len(box) != 4
                or any(type(item) is not int or item < 0 for item in box)
                or box[0] >= box[2] or box[1] >= box[3]
                or box[2]-box[0] > 768 or box[3]-box[1] > 768):
            raise ValueError('Regional core_box must be a positive integer source rectangle')
        row = {'id': identifier, 'core_box': box.copy()}
        if 'instruction' in region:
            instruction = region['instruction']
            if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 4000:
                raise ValueError('Regional instruction must be nonempty bounded text')
            row['instruction'] = instruction.strip()
        normalized.append(row)
    result = {'maximum_submissions': maximum, 'coordinate_space': 'source_canvas', 'regions': normalized}
    if resume is not None:
        result['resume_from_run_id'] = resume
    if reuse is not None:
        result['reuse_generated_from_run_id'] = reuse
    return result


def has_no_effects_veto(text):
    return bool(re.search(r'(?:不要|禁止|不添加|不加|不用)\s*(?:添加|使用)?\s*(?:任何|额外)?\s*特效'
        r'|\b(?:no|without)\s+(?:(?:special|visual|magic|magical)\s+)?(?:effects?|vfx|fx)\b'
        r'|\b(?:do\s+not|don[\'’]t|never)\s+(?:add|use|create)\s+(?:any\s+)?'
        r'(?:(?:special|visual|magic|magical)\s+)?(?:effects?|vfx|fx)\b',
        text, re.IGNORECASE))


def refinement_overrides(input_prompt='', character_notes=''):
    """Keep editable text, excluding known product and managed derived prose."""
    if (not isinstance(input_prompt, str) or len(input_prompt) > 64000
            or not isinstance(character_notes, str) or len(character_notes) > 16000):
        raise ValueError('Recommended profile requires bounded actual request text')
    from generation_prompt import DEFAULT_PROMPT, STYLE_MARKER, DESIGN_MARKER, CHARACTER_MARKER
    from preset_store import get_refinement_profile
    profile = get_refinement_profile({'id': 'source_refinement', 'version': 1})
    def editable(text):
        # The UI exposes the complete editable text, including appended user
        # vetoes after generated sections. Only known complete product prose
        # and header tokens can be removed safely; an unknown section has no
        # reliable end marker. Its remaining context may restrict, never grant.
        text = text.replace(DEFAULT_PROMPT, '').replace(profile['prompt'], '')
        for marker in (MARKER, DESIGN_MARKER, CHARACTER_MARKER, STYLE_MARKER):
            text = text.replace(marker.strip(), '')
        return text.strip()
    return {'input_prompt': editable(input_prompt), 'character_notes': editable(character_notes)}


def recommendation_for(intent, *, input_prompt='', character_notes=''):
    selector = (intent.get('constraints') or {}).get('refinement_profile')
    if selector is None:
        return None
    from preset_store import get_refinement_profile
    profile = get_refinement_profile(selector)
    actual = refinement_overrides(input_prompt, character_notes)
    if selector.get('overrides', {'input_prompt': '', 'character_notes': ''}) != actual:
        raise ValueError('Recommended profile overrides differ from the actual editable request')
    return {'schema': 'product_refinement_recommendation.v1', 'id': profile['id'],
        'version': profile['version'], 'origin': 'bundled_product_recommendation',
        'selection': 'requested_profile', 'profile_sha256': profile['profile_sha256'],
        'effective_goals': profile['goals'],
        'request_overrides': {'brief_sha256': digest(intent.get('brief', '')),
            'input_prompt_sha256': digest(actual['input_prompt']),
            'character_notes_sha256': digest(actual['character_notes']),
            'no_effects': intent['constraints']['no_effects']}}


def skin_cast_contract_for(intent):
    """Compile explicit brief permission, not masks, measured skin or activation.

    Missing means no permission for this optional operator. Character notes and
    generated prompts cannot grant it. Domain vetoes win even across sentences.
    """
    constraints = intent.get('constraints') or {}
    if (constraints.get('reference_role') != 'source_conditioned_design'
            or constraints.get('background_mode') != 'enhance_source'):
        return None
    domains = {
        'face': r'脸|面部|脸颊|\b(?:face|facial|cheeks?)\b',
        'arms': r'手臂|前臂|上臂|胳膊|\b(?:arms?|forearms?)\b',
        'hands': r'手指|手掌|手背|手腕|手(?!臂)|\b(?:hands?|fingers?|wrists?)\b'}
    generic_skin = r'皮肤|肤色|\bskin\b'
    cast = r'杂光|偏色|色带|条带|光斑|污染色|红绿光|\b(?:casts?|colou?r\s+bands?|stripes?|contamination)\b'
    cleanup = r'消除|去除|清除|修正|校正|减轻|清理|去杂光|去偏色|\b(?:remove|correct|reduce|clean)\b'
    unify = r'统一|协调|一致|\b(?:unify|harmonize|harmonise|consistent)\b'
    negative = r'不要|禁止|不必|无需|不允许|不需要|不用|(?:^|\s|请|但|也|并|且)别|不做|不去|不统一|不校正|\b(?:no|never|without|do\s+not|don[\x27’]t)\b'
    coloured_light = r'彩灯|彩色照明|彩色光|舞台颜色|红色灯带|红光|绿光|红绿光|(?:红|绿|蓝|橙)(?:色)?(?:舞台)?(?:灯)?光|\b(?:colou?red\s+lights?|stage\s+(?:lights?|colou?rs?))\b'
    selector = constraints.get('refinement_profile')
    profile = None
    if selector is not None:
        from preset_store import get_refinement_profile
        profile = get_refinement_profile(selector)
        coloured_light += r'|\b(?:colou?red\s+lighting|stage\s+lighting|(?:red|green|blue|orange)\s+(?:stage\s+)?(?:light|lighting))\b'
    rows, denied = {key: [] for key in domains}, set()
    if profile:
        for key in rows:
            rows[key].append({'instruction': '产品推荐：协调本人肤色与不一致源杂光；保留所选照明、肤色与自然阴影。',
                'operations': ['harmonize_skin_chroma'], 'cast_targets': ['source_lighting_inconsistency'],
                'source': 'bundled_product_recommendation'})
    overrides = selector.get('overrides', {}) if profile else {}
    clause_separator = r'[。；;\n，,]|但是|不过|但' if profile else r'[。；;\n，,]'
    clauses = [(part, True) for part in re.split(clause_separator, intent.get('brief', ''))]
    for value in overrides.values():
        clauses.extend((part, False) for part in re.split(clause_separator+r'|\.', value))
    for clause, grants_permission in clauses:
        clause = clause.strip()
        if not clause:
            continue
        selected = [key for key, pattern in domains.items() if re.search(pattern, clause, re.I)]
        skin = bool(re.search(generic_skin, clause, re.I))
        targets = selected or (list(domains) if skin else [])
        veto = bool(re.search(negative, clause, re.I)
            and (re.search(cleanup, clause, re.I) or re.search(cast, clause, re.I)
                 or re.search(unify, clause, re.I)))
        if profile:
            actions = r'消除|去除|去掉|去杂光|去偏色|校正|修正|清理|统一|协调|修饰|精修|美化|改变|处理|调整|动|修|改|\b(?:remove|correct|reduce|clean|unify|harmonize|retouch|edit|change|beautify|touch)\b'
            commands = negative+r'|(?:都|手背|手掌|手指|手臂|前臂|肤色|皮肤|面部|脸|手|臂)别'
            denial = bool(re.search('(?:'+commands+r')[^。；;,，\n]{0,20}(?:'+actions+')', clause, re.I)
                or re.search(r'不(?:改(?:变)?|处理|修饰|美化|精修|调整|动|修|去(?:除|掉|杂光|偏色))', clause))
            geometry_only = (re.search(r'手势|姿态|姿势|脸型|五官|几何|轮廓|比例|手指排列|\b(?:pose|gesture|geometry|anatomy|proportions?)\b', clause, re.I)
                and not re.search(r'肤色|颜色|色彩|偏色|杂光|照明|光照|受光|彩灯|\b(?:colou?r|casts?|lighting|skin\s+tone)\b', clause, re.I))
            if geometry_only:
                denial = False
            veto = denial and bool(targets or re.search(cast+'|'+unify, clause, re.I))
            if targets:
                veto |= bool(re.search(r'(?:保留|保持)(?:脸|面部|手臂|手部|手掌|手背|前臂|皮肤|肤色)(?:的肤色)?(?:完全)?(?:原样|不变)', clause))
        keep_light = (re.search(r'保留|保持|不改变|不要改变|不得改变|不改|\b(?:keep|preserve|retain)\b', clause, re.I)
            and re.search(coloured_light, clause, re.I))
        if veto or keep_light:
            # Environment-only preservation is not a skin veto. An unscoped
            # explicit "do not remove casts" veto is deliberately conservative.
            if targets:
                denied.update(targets)
            elif not re.search(r'背景|环境|衣服|衣料|道具|\b(?:background|environment|costume|prop)\b', clause, re.I):
                denied.update(domains)
            continue
        if profile and re.search(r'只|仅|\bonly\b', clause, re.I) and re.search(actions, clause, re.I):
            if targets:
                denied.update(set(domains)-set(targets))
            elif re.search(r'衣服|衣料|服装|背景|道具|\b(?:cloth|clothing|costume|background|props?)\b', clause, re.I):
                denied.update(domains)
        if not grants_permission:
            continue
        operations = []
        if re.search(cleanup, clause, re.I) and re.search(cast, clause, re.I):
            operations.append('remove_declared_unwanted_casts')
        if re.search(unify, clause, re.I) and (targets or skin):
            operations.append('harmonize_skin_chroma')
        if operations:
            for key in targets:
                rows[key].append({'instruction': clause, 'operations': operations,
                    'cast_targets': [name for name, pattern in (
                        ('red', r'红|\bred\b'), ('green', r'绿|青|\b(?:green|cyan)\b'),
                        ('colour_band', r'条|带|\b(?:bands?|stripes?)\b'))
                        if re.search(pattern, clause, re.I)] or ['user_declared_cast']})
    authorized = {key: value for key, value in rows.items() if value and key not in denied}
    if not authorized:
        return None
    result = {'schema': 'skin_cast_contract.v1',
        'permission_source': 'checked_product_recommendation' if profile else 'explicit_task_brief',
        'domains': authorized, 'vetoed_domains': sorted(denied),
        'anchor': 'reliable_face_skin_chroma_in_requested_lighting',
        'target_temperature': 'follow_user_intent',
        'preserve': ['source_geometry', 'native_texture', 'local_shading',
            'makeup_and_lip_colour', 'relative_intrinsic_nail_colour',
            'user_preserved_coloured_lighting'],
        'rgb_changes_allowed': True, 'reference_rgb_sampling_allowed': False,
        'activation_status': 'not_evaluated',
        'consumer_requirements': ['verified_domain_masks', 'reliable_skin_anchor',
            'bounded_illumination_correction', 'preserve_relative_material_colour']}
    if profile:
        result['recommendation_binding'] = {'id': profile['id'], 'version': profile['version'],
            'profile_sha256': profile['profile_sha256'], 'overrides_sha256': digest(overrides)}
    return result


def _source_photographic_goals(user_text, *, no_effects, extended_soft=False):
    """Expand declared photographic goals, never choose a palette or invent lore."""
    result = []
    clauses = re.split(r'[。；;\n]', user_text)
    cleanup = any(re.search(
        r'(?:消除|去除|清除|修正|校正|减轻|清理)[^。；;\n]{0,80}(?:杂光|偏色|色带|条带|光斑|污染色|红绿光)'
        r'|\b(?:remove|correct|reduce|clean)\b[^.;\n]{0,70}\b(?:casts?|colou?r(?:ed)?\s+(?:bands?|stripes?)|chromatic\s+contamination|unwanted\s+colou?r)\b',
        clause, re.IGNORECASE) and not re.search(
        r'(?:不要|禁止|不必|无需|不允许)[^。；;\n]{0,16}(?:消除|去除|清除|修正|校正|减轻|清理)'
        r'|\b(?:do\s+not|don[\'’]t|never)\s+(?:remove|correct|reduce|clean)\b', clause, re.IGNORECASE)
        for clause in clauses)
    if cleanup:
        result.append('仅在用户明确要求清理杂光的部位校色：让皮肤、浅色道具、珠饰与衣料保持各自可信的底色和材质，不被环境色一起染灰、染脏；用户要求保留的妆色和有意彩色照明不去除。')
        if re.search(r'手|臂|\b(?:arms?|hands?)\b', user_text, re.IGNORECASE):
            result.append('手与手臂需要校色时，连续处理整只手、手腕到前臂的受光过渡，清除要求移除的长条彩色光斑和局部污染，不只清脸却留下手臂条带；保留手指、关节、肤纹及自然阴影。')
    if re.search(r'干净[^。；;\n]{0,12}(?:肤|脸)|明净[^。；;\n]{0,12}(?:肤|脸)|自然肤|面部精修|脸部精修|\bclean\b[^.;\n]{0,20}\b(?:skin|face)\b', user_text, re.IGNORECASE):
        result.append('按用户要求完成脸肤修饰：干净而有体积，眼妆、眉形、唇色与细节各自清晰；皮肤保留自然纹理与柔和明暗，不磨成塑料，不把干净理解为全脸漂白。')
    soft_phrase = (r'柔和(?:中性|冷调|暖调|冷色|暖色|的|自然|人像|\s){0,3}主光|柔主光|柔光'
                   if extended_soft else r'柔和主光|柔主光|柔光')
    soft = re.search(soft_phrase+r'|\bsoft\b[^.;\n]{0,30}\b(?:key|light)\b', user_text, re.IGNORECASE)
    no_soft = re.search(r'(?:不要|不加|禁止|无需|不需要)[^。；;\n]{0,12}(?:'+soft_phrase+r')|\b(?:no|without)\s+soft\b', user_text, re.IGNORECASE)
    if soft and not no_soft:
        result.append('按所选明暗目标以柔主光塑造人物，让脸、手和主要道具获得清楚、柔和的亮度层次，避免主体灰暗塌平；环境仍可保留用户指定的夜景、暗调或暖调，不整体拉白。')
    elif re.search(r'主光|布光|曝光|\b(?:key\s+light|exposure|lighting)\b', user_text, re.IGNORECASE):
        result.append('主光的方向、软硬与冷暖服从用户原文，人物与主要道具应有可读的明暗塑形；不要用整图同色滤镜替代真实的光照层次。')
    rim = re.search(r'轮廓光|边缘光|发丝[^。；;\n]{0,14}(?:光|rim)|发缘[^。；;\n]{0,10}光|\brim\b', user_text, re.IGNORECASE)
    no_rim = re.search(r'(?:不要|不加|禁止|无需)[^。；;\n]{0,12}(?:轮廓光|边缘光|rim)|\b(?:no|without)\s+rim\b', user_text, re.IGNORECASE)
    if rim and not no_rim:
        result.append('用户指定的发丝边缘受光仅落在合理朝向的局部细丝，细窄、克制、带渐变，冷暖按原文；发内保持原有深浅与丝纹，不形成整片饱和彩色发带或均匀全身描边。')
    if re.search(r'空间|纵深|景深|层次|氛围|电影|\b(?:depth|cinematic|atmosphere)\b', user_text, re.IGNORECASE):
        result.append('在原有建筑、布景和透视内组织前景、人物与后景：以用户要求的亮度、虚实和色彩差异建立纵深，让人物成为视觉中心，不更换场地或把背景压成无层次黑块。')
    if not no_effects and re.search(r'雾|\b(?:fog|mist|haze)\b', user_text, re.IGNORECASE) and not re.search(
            r'(?:不要|不加|禁止|无需)[^。；;\n]{0,8}雾|\b(?:no|without)\s+(?:fog|mist|haze)\b', user_text, re.IGNORECASE):
        result.append('只按用户要求呈现或保留薄雾：用低对比、分层的局部空气感衔接空间，保持脸、眼妆、手指与道具细节清楚，不以整图蒙白代替纵深。')
    if not no_effects and re.search(r'前景|叶|\b(?:foreground|leaves|foliage)\b', user_text, re.IGNORECASE) and not re.search(
            r'(?:不要|不加|禁止|无需|不需要)[^。；;\n]{0,8}(?:前景|叶)|\b(?:no|without)\s+(?:foreground|leaves|foliage)\b', user_text, re.IGNORECASE):
        result.append('按用户指定处理原有或明确要求的前景物，保留相应固有色、虚焦和遮挡关系，让前景衬托人物；不随意新增叶片、复制素材或遮住脸与手。')
    return result


def compile_direction(intent, *, source_sha256, character_name='', character_notes='', input_prompt=''):
    """Freeze declared intent, never infer lore, skills or photographic observations."""
    if (intent.get('constraints') or {}).get('background_mode') != 'enhance_source':
        raise ValueError('Explicit enhance_source intent required for offline direction')
    for key, value, limit in (('character_name', character_name, 256),
                              ('character_notes', character_notes, 16000),
                              ('input_prompt', input_prompt, 64000)):
        if not isinstance(value, str) or len(value) > limit:
            raise ValueError(key + ' must be bounded text')
    if not isinstance(source_sha256, str) or not re.fullmatch('[0-9a-f]{64}', source_sha256):
        raise ValueError('Creative direction requires the bound source SHA256')
    brief = intent.get('brief', '')
    user_text = '\n'.join(text for text in (brief, character_notes) if text)
    # Only original editable prose participates in veto detection. A previous
    # generated direction must never manufacture a new user prohibition.
    base = input_prompt.split(MARKER, 1)[0].rstrip()
    if intent['constraints'].get('refinement_profile') is not None:
        base = refinement_overrides(input_prompt, character_notes)['input_prompt']
    no_effects = intent['constraints']['no_effects'] or has_no_effects_veto(user_text + '\n' + base)
    effects = [part.strip() for part in re.split(r'[。；;\n]', user_text)
               if re.search(r'主效|特效|意象|\beffects?\b|\bmotif\b', part, re.IGNORECASE)]
    effect = {'mode': 'none' if no_effects else 'user_described' if effects else 'not_specified',
              'description': '不要添加任何特效。' if no_effects else '；'.join(effects)
                  if effects else '用户未指定主效；按可见场景与角色气质选择克制的氛围处理，也可不加特效；不臆造具体角色技能。',
              'ownership': 'one_global_source_canvas_object; regional crops never duplicate it',
              'basis': 'user_instructions_only; not_verified_character_ability'}
    scene = '延续原景的室内或室外属性、建筑布局、主要道具、机位、透视及主体位置；允许整理、局部替换干扰性的背景细节及调整景深和受光，整体仍是原来的地点，不完整换景。'
    light = '在原场景和原人物几何内协调有来源的主光、明暗与氛围；保留原生纹理，不用统一滤镜或全身描边代替光照设计。'
    character = {'name': character_name.strip(), 'notes': character_notes.strip(),
                 'basis': 'user_provided_unverified', 'research_performed': False}
    direction = {'schema_version': SCHEMA, 'source_sha256': source_sha256,
        'intent_sha256': digest(intent), 'origin': 'offline_user_contract',
        'character': character, 'background_mode': 'enhance_source',
        'scene': scene, 'light': light, 'main_effect': effect,
        'hard_locks': list(HARD_LOCKS), 'brief': brief,
        'regional_policy': 'native source_canvas background only; alpha-zero and safety margin verified by worker',
        'network_requests': 0, 'model_inferences': 0, 'automatic_visual_acceptance': False}
    same_photo = intent['constraints'].get('reference_role') == 'same_photo_finish'
    source_design = intent['constraints'].get('reference_role') == 'source_conditioned_design'
    if same_photo or source_design:
        direction['reference_role'] = intent['constraints']['reference_role']
    recommendation = recommendation_for(intent, input_prompt=base, character_notes=character_notes)
    if recommendation is not None:
        direction['recommendation'] = recommendation
        direction['origin'] = 'offline_checked_product_recommendation_with_request_overrides'
        direction['recommendation_prompt_policy'] = 'source_refinement_character.v3'
    material_refinement = skin_cast_contract_for(intent)
    if material_refinement is not None:
        direction['material_refinement'] = material_refinement
    lines = ['本冻结合同中的保留项与禁止项优先于原始提示词、预设或其他创作建议。',
        '用户声明角色：' + character['name'] if character['name'] else '角色未声明；不得自动识别或编造角色能力。',
        '用户角色说明：' + character['notes'] if character['notes'] else '',
        '创作要求：' + brief if brief else '', scene, light,
        '主效：' + effect['description'],
        '主效在全画布只定义一次；区域仅细化所见片段，不重复生成主效，不增加新人物。',
        '保留本人身份、五官几何、妆造、神态、姿态、手势、服装和发型轮廓；保留原生细节。',
        '以上仅编译用户要求，未经角色研究或图像分析；不推断任何角色技能。']
    if same_photo:
        lines += ['参考图是用户声明的同照精修效果标尺，不是新原图；执行时仍须核验同照配准质量。',
            '原片拥有几何与原生纹理；参考只提供影调和源图引导的局部低频色映射系数，不直接合成参考RGB。',
            '薄雾、红叶等要求按用户原文保留；此光色映射合同不授予复制参考素材或新增实体图像的权限。']
    if source_design:
        # The image editor receives art direction, not the renderer's ownership,
        # coefficient or registration implementation. Those remain frozen above.
        model_notes = (intent['constraints']['refinement_profile'].get('overrides', {}).get('character_notes', '')
                       if recommendation is not None else character['notes'])
        lines = ['以用户明确的创作目标、冷暖和禁止项为准；保留要求优先，不擅自换风格或推断角色技能。',
            '用户声明角色：' + character['name'] if character['name'] else '角色未声明；以原片可见妆造为准，不编造角色能力。',
            '用户角色说明：' + model_notes if model_notes else '',
            '本次创作要求：' + brief if brief else '', scene,
            '以这张原片完成所要求的后期，保留本人身份、妆造、姿态、服装道具及机位构图。',
            '人物优先：让眼神、妆容、唇色和衣料色彩有表现力，以有层次的主光和柔和过渡形成精致的角色写真。暗调或恐怖题材仍保留干净的皮肤高光、眼部可读性与丰富暗部，不把全图一起压暗、染灰或蒙上一层蓝色。',
            '背景以局部明暗、虚实与色彩衬托人物；改变应同时体现在人像精修、光影塑形和角色气氛，避免只换色，也避免背景比人物更抢眼。']
        if effect['mode'] == 'none':
            lines.append('主效要求：' + effect['description'])
        else:
            lines.append('允许贴合角色气质和原场景的克制氛围特效；选择有叙事作用的局部点缀并保留相应受光与遮挡，不以全屏浓雾、强光束或堆叠符号代替设计，不凭角色名编造能力。')
        if recommendation is not None:
            from preset_store import get_refinement_profile
            profile = get_refinement_profile(intent['constraints']['refinement_profile'])
            lines += ['以下为产品推荐目标，服从用户原文与保留项，不是用户逐字要求。', profile['prompt']]
            lines.append('Apply the requested key illumination to the hands and existing props as well as the face. '
                'When neutral key light is requested, reduce conflicting red and green light bands across those surfaces; '
                'retain their separate material colours, aged paint, wood grain and metallic reflections. '
                'Do not paint all objects white or make hands identical to face pixels. '
                'Explicitly preserved coloured lighting, warm light and original material colours take precedence.')
        lines += _source_photographic_goals(user_text, no_effects=no_effects, extended_soft=recommendation is not None)
        lines.append('交付一张主次明确、材质可辨、局部过渡连续的完整作品；不增加未要求的人物、技能、文字、水印或对比面板。')
    # A previously compiled marker is not editable user prose; never stack it.
    direction['prompt'] = base + MARKER + '\n'.join(line for line in lines if line)
    direction['direction_id'] = digest(direction)
    return direction


def validate_direction(direction, intent, *, source_sha256):
    if not isinstance(direction, dict) or direction.get('schema_version') != SCHEMA:
        raise ValueError('Checked source enhancement requires frozen creative direction')
    prompt = direction.get('prompt')
    if not isinstance(prompt, str) or prompt.count(MARKER) != 1:
        raise ValueError('Creative direction has no unique compiled prompt')
    character = direction.get('character') or {}
    expected = compile_direction(intent, source_sha256=source_sha256,
        character_name=character.get('name', ''), character_notes=character.get('notes', ''),
        input_prompt=prompt.split(MARKER, 1)[0])
    if direction != expected:
        raise ValueError('Creative direction differs from its frozen user contract')
    return expected
