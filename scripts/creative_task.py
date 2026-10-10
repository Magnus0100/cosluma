"""One user task contract and a side-effect-free, content-bound execution plan."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

INTENT_SCHEMA = 'cosplay_task_intent.v1'
PLAN_SCHEMA = 'cosplay_task_plan.v1'
MODES = {'local_portrait': '本地原片精修', 'existing_reference': '已有参考完整后期',
         'create_reference': '创作参考并完成后期'}
REQUEST_FIELDS = frozenset(('source', 'reference', 'preset', 'provider', 'model', 'vision_model',
    'api_base', 'api_url', 'api_key_env', 'character_name', 'character_notes', 'prompt', 'input_prompt',
    'variant_count', 'variant_styles', 'model_registry', 'algorithm_profile', 'material_finish_profile',
    'sample', 'run_id', 'parent_run_id',
    'generation_design', 'character_profile', 'appearance_revision', 'manual_layers', 'native_source_retouch',
    'preset_snapshot', 'effective_prompt', 'prompt_template_version', 'size', 'alpha_backend',
    'denoise_strength', 'residual', 'preserve_project_selection', 'creative_task',
    'async', 'sync', 'plan_id', 'generate', 'local_only', 'complete_post'))
SHARED_FIELDS = frozenset(('preset', 'provider', 'model', 'vision_model', 'api_base', 'api_url',
    'api_key_env', 'character_name', 'character_notes', 'prompt', 'input_prompt',
    'variant_count', 'variant_styles', 'model_registry', 'algorithm_profile', 'material_finish_profile',
    'size', 'creative_task'))


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf8')).hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def normalize_task_intent(value):
    if not isinstance(value, dict) or set(value) - {'schema_version', 'mode', 'brief', 'constraints', 'services'}:
        raise ValueError('Provide the current creative task intent')
    if value.get('schema_version') != INTENT_SCHEMA or value.get('mode') not in MODES:
        raise ValueError('Unsupported creative task schema or route')
    brief = value.get('brief', '')
    if not isinstance(brief, str) or len(brief) > 16000:
        raise ValueError('Creative brief must be text of at most 16000 characters')
    constraints = value.get('constraints', {})
    services = value.get('services', {})
    if not isinstance(constraints, dict) or set(constraints) - {'preserve_background', 'no_effects'}:
        raise ValueError('Unsupported creative task constraints')
    if not isinstance(services, dict) or set(services) - {'text_vision'}:
        raise ValueError('Unsupported creative task service policy')
    for key, item in [*constraints.items(), *services.items()]:
        if type(item) is not bool:
            raise ValueError('Task option must be a boolean: ' + key)
    local = value['mode'] == 'local_portrait'
    if local and services.get('text_vision') is True:
        raise ValueError('Local portrait tasks do not call external text/vision services')
    return {'schema_version': INTENT_SCHEMA, 'mode': value['mode'], 'brief': brief.strip(),
        'constraints': {key: constraints.get(key, local) for key in ('preserve_background', 'no_effects')},
        'services': {'text_vision': services.get('text_vision', not local)}}


def image_budget(intent, variant_count=1):
    intent = normalize_task_intent(intent)
    count = variant_count
    if type(count) is not int or not 1 <= count <= 8:
        raise ValueError('Candidate count must be an integer from 1 to 8')
    create = intent['mode'] == 'create_reference'
    return {'initial_reference': count if create else 0,
            'reference_correction': 1 if create else 0, 'independent_effect': 1 if create else 0}


def task_brief(intent):
    intent = normalize_task_intent(intent)
    lines = [intent['brief']] if intent['brief'] else []
    if intent['constraints']['preserve_background']:
        lines.append('保留原背景，不更换或重建场景。')
    if intent['constraints']['no_effects']:
        lines.append('不要添加特效。')
    return '\n'.join(lines)


def image_generation_allowed(run_or_args):
    if isinstance(run_or_args, dict):
        intent = run_or_args.get('task_intent') or (run_or_args.get('options') or {}).get('task_intent')
    else:
        intent = getattr(run_or_args, 'task_intent', None)
    return not intent or normalize_task_intent(intent)['mode'] == 'create_reference'


def validate_task_plan(plan, intent):
    intent = normalize_task_intent(intent)
    if not isinstance(plan, dict) or plan.get('schema_version') != PLAN_SCHEMA:
        raise ValueError('Current task execution requires a checked task plan')
    if plan.get('intent') != intent or plan.get('mode') != intent['mode'] or plan.get('ready') is not True:
        raise ValueError('Task plan does not match its current intent or is blocked')
    expected = image_budget(intent, (plan.get('image_budget') or {}).get('initial_reference', 0)
                            if intent['mode'] == 'create_reference' else 1)
    if plan.get('image_budget') != expected:
        raise ValueError('Task image budget differs from its route')
    if not isinstance(plan.get('plan_id'), str) or len(plan['plan_id']) != 64:
        raise ValueError('Task plan has no checked content identity')
    return copy.deepcopy(plan)


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


def _services(request, intent):
    if not intent['services']['text_vision']:
        return {'text_vision': False, 'image_generation': False, 'offline': True,
                'configured': True, 'network_probed': False, 'sent_content': [],
                'text_vision_requests_upper_bound': 0, 'price': '不调用外部服务'}
    try:
        from vision_transport import _connection
        config, endpoint, _credential, protocol = _connection(**{key: request.get(key)
            for key in ('provider', 'api_base', 'api_url', 'api_key_env')})
        connection = {'configured': True, 'provider': config['registry_id'], 'endpoint': endpoint,
                      'protocol': protocol}
    except (ValueError, RuntimeError, OSError) as exc:
        connection = {'configured': False, 'reason': str(exc)[:400]}
    return {**connection, 'text_vision': True, 'image_generation': intent['mode'] == 'create_reference',
        'offline': False, 'network_probed': False,
        'sent_content': ['原片观察图', '已有参考或生成参考', '成片与匹配原尺寸审片窗口', '创作要求'],
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
    local, supplied = intent['mode'] == 'local_portrait', intent['mode'] == 'existing_reference'
    if body.get('generate') is True and (local or supplied):
        raise ValueError('This route forbids all image generation')
    if local and body.get('reference'):
        raise ValueError('Local portrait tasks do not consume a reference')
    if not supplied and body.get('reference'):
        raise ValueError('Use the existing-reference route for a supplied reference')
    request = {key: copy.deepcopy(value) for key, value in body.items()
               if key not in {'creative_task', 'plan_id', 'generate', 'local_only', 'complete_post'}}
    inputs = {'source': _input(root, body.get('source'), 'source')}
    request['source'] = inputs['source']['path']
    if supplied:
        inputs['reference'] = _input(root, body.get('reference'), 'reference')
        request['reference'] = inputs['reference']['path']
    count = body.get('variant_count', 1)
    budget = image_budget(intent, count)
    if not local and not intent['services']['text_vision']:
        policy_block = ['完整参考后期需文本/视觉计划与审片；当前任务未允许该服务。可使用本地原片路线继续离线编辑。']
    else:
        policy_block = []
    brief = task_brief(intent)
    input_prompt = str(body.get('input_prompt') or body.get('prompt') or DEFAULT_PROMPT)
    request.update(task_intent=intent, creative_task=intent,
        input_prompt=input_prompt, prompt=input_prompt, variant_count=count,
        generate=intent['mode'] == 'create_reference', local_only=local, complete_post=not local,
        cosplay_correction_budget=1 if intent['mode'] == 'create_reference' else 0,
        preserve_project_selection=bool(body.get('preserve_project_selection', True)))
    # Keep the user's editable prompt separate from deterministic task constraints.
    request['task_brief'] = brief
    models = registry_status(root, registry_path=body.get('model_registry'))
    required_keys = {'python', 'semantics_checkpoint', 'semantics_repo_dir', 'vitmatte_model_root'} if local else {
        row['key'] for row in models['paths']}
    paths = [{key: row[key] for key in ('key', 'label', 'kind', 'available', 'status')}
             for row in models['paths'] if row['key'] in required_keys]
    blockers = list(policy_block)
    storage = _storage(root, request, intent)
    if not storage['ready']:
        blockers.append('工作区或任务名称过长，无法安全保存完整工程；请使用较短的工作区路径或任务名称。')
    if not models['configured']:
        blockers.append('当前分析模型配置不可用：' + str(models.get('error') or '尚未配置'))
    blockers.extend('缺少 ' + row['label'] for row in paths if not row['available'])
    from task_model_preflight import inspect_task_models
    installation = inspect_task_models(models.get('config', {}), mode=intent['mode'])
    blockers.extend(installation['blocking_reasons'])
    services = _services(request, intent)
    if not services['configured']:
        blockers.append('文本/视觉服务配置不可用：' + str(services.get('reason') or '尚未配置'))
    if intent['mode'] == 'create_reference':
        from reference_provider import provider_health
        health = provider_health(**{key: request.get(key) for key in
            ('provider', 'api_base', 'api_url', 'model', 'api_key_env')})
        services['generation_configured'] = bool(health.get('ready'))
        if not health.get('ready'):
            blockers.append('生图服务未就绪：' + str(health.get('reason') or health.get('status')))
    registry_file = Path(models['registry_path']) if models.get('registry_path') else None
    config_identity = {'path': str(registry_file) if registry_file else None,
        'sha256': sha(registry_file) if registry_file and registry_file.is_file() else None}
    capabilities = {'models': paths, 'configuration_verified': models['configured'],
        'model_preflight': installation,
        'storage_preflight': storage,
        'model_profiles': {k: v for k, v in models.get('config', {}).items() if k.endswith('profile')},
        'registry_identity': config_identity, 'verification_scope': 'installation_structure_small_file_and_recorded_identity',
        'inference_verified': False}
    stage_names = (['原片解码与本地精修', '高清保存', '人工审片与导出'] if local else
        ['原片与参考成对设计' if supplied else '角色与本图设计',
         '复用已有参考' if supplied else '创作风格参考', '原片精修与原生人物基线',
         '仿色、材质与明暗协调', '背景、接触与已拥有的特效合成',
         '整图和匹配原尺寸审片', '有资格的一轮本地修复与复审', '人工审片、重开与导出'])
    plan = {'schema_version': PLAN_SCHEMA, 'mode': intent['mode'], 'label': MODES[intent['mode']],
        'intent': intent, 'inputs': inputs, 'ready': not blockers, 'blocking_reasons': blockers,
        'stages': [{'id': str(i+1), 'label': label} for i, label in enumerate(stage_names)],
        'capabilities': capabilities, 'services': services, 'image_budget': budget,
        'generation_submissions_upper_bound': sum(budget.values()),
        'outputs': ['原尺寸PNG', '真实TIFF16/32母版', '可继续编辑的产品工程'],
        'limitations': ['自动审片只证明实际观察范围，人工最终批准单独记录。',
            '自动投影阴影、镜面或湿地反射生成尚未支持。',
            '共享创作要求不等于已验证的套图风格一致算法。'],
        'estimated_seconds': None, 'estimate_basis': '没有足够的当前硬件任务统计',
        'preflight_network_requests': 0, 'preflight_model_inferences': 0, 'preflight_jobs_created': 0}
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
    request['cosplay_correction_budget'] = 1 if plan['mode'] == 'create_reference' else 0
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
    budget = {key: sum(row['plan']['image_budget'][key] for row in items)
              for key in ('initial_reference', 'reference_correction', 'independent_effect')}
    plan = {'schema_version': 'cosplay_batch_task_plan.v1', 'shared_task': intent,
        'items': items, 'ready': all(row['plan']['ready'] for row in items),
        'image_budget_total': budget, 'per_image_geometry_shared': False,
        'automatic_style_consistency_claimed': False}
    plan['plan_id'] = digest(plan)
    return {'ok': True, 'batch_plan': plan, 'request': copy.deepcopy(body)}


def format_task_plan(plan):
    lines = [plan['label'], '可以开始制作' if plan['ready'] else '请先补齐以下配置']
    lines += plan['blocking_reasons']
    lines.append('处理步骤：' + ' → '.join(row['label'] for row in plan['stages']))
    lines.append('图像生成提交上限：' + str(plan['generation_submissions_upper_bound']))
    lines.append('外部文本/视觉服务：' + ('会使用；' + plan['services']['price']
                 if plan['services']['text_vision'] else '不使用'))
    lines.append('模型检查：安装结构、小文件与记录身份；完整权重内容和推理由实际处理核验。')
    return '\n'.join(lines)
