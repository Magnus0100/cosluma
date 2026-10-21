"""A real vision-model decision adapter for the checked creative session.

The coordinator provides capabilities; the model chooses a useful operation or
stops. This adapter cannot execute tools, expand scope or replenish a budget.
Unknown calls remain evidence-bearing attempts and are never resubmitted here.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

from asset_identity import sha256_file
from current_model_registry import atomic_json
import photo_edit_session as session
from photo_editing import compile_edit_prompt, identity, validate_scope
from photo_session_review import prepare_board

SCHEMA = 'cosluma_photo_session_decision.v1'


def _asset(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError('Decision image is not a file')
    return {'path': str(path), 'sha256': sha256_file(path)}


def _images(images):
    if (not isinstance(images, list) or [row.get('role') for row in images] not in
            (['source'], ['source', 'best'], ['source', 'best', 'working'], ['source', 'best', 'previous_attempt'])):
        raise ValueError('Decision requires an actual source and optional current best')
    for row in images:
        if set(row) != {'role', 'path', 'sha256'} or _asset(row['path']) != {key: row[key] for key in ('path', 'sha256')}:
            raise ValueError('Decision image identity changed')


def _counts(value):
    if not isinstance(value, dict) or set(value) != set(session.COUNTERS):
        raise ValueError('Decision requires every checked budget counter')
    if any(type(count) is not int or count < 0 for count in value.values()):
        raise ValueError('Decision budget counters must be nonnegative integers')
    return copy.deepcopy(value)


def _text(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 16000:
        raise ValueError('Decision requires bounded nonempty descriptive text')


def _capabilities(context):
    scope = validate_scope(context['scope'])
    from photo_shaping import operations as permitted_operations,validate_authorization,shaping_operations,shaping_tool
    shaping=context.get('shaping_authorization')
    if shaping is not None:
        validate_authorization(shaping,source_sha256=context['source_original']['sha256'],scope=scope,
            design_id=context['design']['design_id'],base_appearance=context['shaping_base_appearance'],
            source_appearance=context['appearance_assets']['source'])
    allowed=permitted_operations(scope,shaping)
    compile_edit_prompt(context['design'], scope, 'gpt_image')
    _images(context['images'])
    remaining = _counts(context['remaining_budget'])
    if remaining['vision_requests'] < 1:
        raise ValueError('No checked request remains for this model decision')
    remaining['vision_requests'] -= 1  # advance reserves this call after building its context.
    tools = context.get('tools')
    if not isinstance(tools, dict):
        raise ValueError('Coordinator-owned decision capabilities required')
    available, unavailable = {}, {}
    for name, tool in tools.items():
        _text(name)
        if (not isinstance(tool, dict)
                or set(tool)-{'operations', 'costs', 'producer_identity', 'requires_region', 'requires_working', 'parameter_schema', 'inputs_available'}
                or not {'operations', 'costs', 'producer_identity', 'requires_region'} <= set(tool)):
            raise ValueError('Unexpected coordinator tool descriptor')
        if type(tool['requires_region']) is not bool:
            raise ValueError('Decision tool requires an explicit region capability')
        requires_working = tool.get('requires_working', False)
        if type(requires_working) is not bool:
            raise ValueError('Decision tool requires an explicit working-input capability')
        session.validate_parameter_schema(tool.get('parameter_schema'))
        if 'inputs_available' in tool and type(tool['inputs_available']) is not bool:
            raise ValueError('Tool actual-input admission must be a boolean fact')
        operations = tool['operations']; _text(tool['producer_identity'])
        if (not isinstance(operations, list) or not operations or len(set(operations)) != len(operations)
                or set(operations)-set(allowed)):
            raise ValueError('Tool capability expands the editing scope')
        if shaping is not None and (name!=shaping_tool(shaping) or set(operations)!=set(shaping_operations(shaping))):
            raise ValueError('Shaping cannot offer a joint, photographic or unselected-part tool')
        costs = _counts(tool['costs'])
        if costs['tool_calls'] or costs['improvement_steps']:
            raise ValueError('Tool descriptor cannot charge session-owned counters')
        required = {**costs, 'tool_calls': 1,
            'improvement_steps':0 if context.get('operation_stage')=='initial' else int(len(context['images']) > 1)}
        required['vision_requests'] += int(context.get('visual_review',True))
        exceeded = [key for key in session.COUNTERS if required[key] > remaining[key]]
        if context.get('zero_image_only') and costs['image_submissions']:
            unavailable[name] = {'reason':'checked_zero_image_finish_only'}
        elif tool.get('inputs_available') is False or requires_working and tool.get('inputs_available') is not True:
            unavailable[name] = {'reason': 'actual_working_input_not_admitted' if requires_working else 'actual_inputs_unavailable'}
        elif requires_working and not any(row['role'] == 'working' for row in context['images']):
            unavailable[name] = {'reason': 'reviewed_working_candidate_required'}
        elif tool['requires_region'] and len(context['images']) < 2:
            unavailable[name] = {'reason': 'current_best_required'}
        elif exceeded:
            unavailable[name] = {'reason': 'checked_budget_insufficient', 'counters': exceeded}
        else:
            available[name] = copy.deepcopy(tool)
    return available, unavailable, remaining


def _feedback(context):
    values = context.get('feedback')
    if not isinstance(values, list):
        raise ValueError('Decision context requires independent feedback, even when empty')
    result = []
    for value in values:
        if not isinstance(value, dict):
            raise ValueError('Independent decision feedback must be an object')
        observed = value.get('observed_images')
        if (not isinstance(observed, list) or not all(isinstance(image, dict) for image in observed)
                or [image.get('role') for image in observed] not in
                    (['source', 'candidate'], ['source', 'candidate', 'previous_best'])
                or any(set(image) != {'role', 'sha256'} or not isinstance(image['sha256'], str)
                       or len(image['sha256']) != 64 or any(c not in '0123456789abcdef' for c in image['sha256'])
                       for image in observed)):
            raise ValueError('Independent feedback requires its actual observed image identities')
        row = {}
        for axis in session.AXES:
            finding = value.get(axis)
            if (not isinstance(finding, dict) or set(finding) != {'status', 'findings'}
                    or finding['status'] not in session.RANK or not isinstance(finding['findings'], list)
                    or not finding['findings']):
                raise ValueError('Decision feedback must retain separate quality axes')
            for text in finding['findings']:
                _text(text)
            row[axis] = copy.deepcopy(finding)
        _text(value.get('primary_issue'))
        row.update(primary_issue=value['primary_issue'], comparison=value.get('comparison'),
            candidate_sha256=observed[1]['sha256'])
        if len(observed) == 3:
            row['previous_best_sha256'] = observed[2]['sha256']
        result.append(row)
    return result


def _producer():
    from runtime_resources import code_path
    return {name: sha256_file(code_path(name)) for name in ('photo_session_decision.py',
        'photo_edit_session.py', 'photo_editing.py', 'photo_session_review.py',
        'vision_transport.py', 'raw_support.py','photo_shaping.py')}


def _prepare_input(images, directory):
    if len(images) == 1:
        # There is no finished candidate yet. Send the actual source, including
        # RAW through transport's normal decoder, without inventing a second panel.
        mapping = {'schema': 'cosluma_decision_image_mapping.v1', 'images': images,
            'panels': [{'panel': 'single_source', 'role': 'source'}],
            'transport_image': {key: images[0][key] for key in ('path', 'sha256')},
            'inspection_scope': 'whole_image', 'native_detail_acceptance': False}
    else:
        board = prepare_board(images, directory/'board', decision=True)
        mapping = {'schema': 'cosluma_decision_image_mapping.v1', 'images': images,
            'panels': board['panels'],
            'transport_image': board['board'], 'review_board': board,
            'inspection_scope': 'whole_image', 'native_detail_acceptance': False}
    atomic_json(directory/'image-mapping.json', mapping)
    return mapping


def _validate_mapping(mapping, images):
    if mapping['images'] != images or _asset(mapping['transport_image']['path']) != mapping['transport_image']:
        raise ValueError('Frozen decision image mapping changed')
    if 'review_board' in mapping:
        board = mapping['review_board']
        saved = json.loads((Path(board['board']['path']).parent/'review-board.json').read_text(encoding='utf-8'))
        if saved != board or board['board'] != mapping['transport_image']:
            raise ValueError('Frozen decision board roles changed')


def _payload(value, available, shaping_authorization=None):
    if not isinstance(value, dict) or value.get('action') not in {'tool', 'stop'}:
        raise ValueError('Vision model must choose one available tool or stop')
    if value['action'] == 'stop':
        if set(value) != {'action', 'reason'}:
            raise ValueError('Model stop response contains unsupported fields')
        _text(value['reason'])
    else:
        if value.get('tool') not in available:
            raise ValueError('Model chose an unavailable or unaffordable tool')
        tool = available[value['tool']]
        fields = {'action', 'tool', 'operations', 'instruction', 'reason'}
        schema = session.validate_parameter_schema(tool.get('parameter_schema'))
        if schema:
            fields.add('parameters')
            session.validate_parameters(value.get('parameters'), schema)
        elif 'parameters' in value:
            if not isinstance(value['parameters'],dict) or value['parameters']:
                raise ValueError('Selected tool does not support parameters')
            # Preserve the raw transport response; an empty object adds no arguments.
            value={key:part for key,part in value.items() if key!='parameters'}
        if tool['requires_region']:
            fields.add('region')
            session.validate_region(value.get('region'))
        if set(value) != fields:
            raise ValueError('Model fields differ from the selected tool capability')
        operations = value['operations']
        if (not isinstance(operations, list) or not operations or len(set(operations)) != len(operations)
                or set(operations)-set(available[value['tool']]['operations'])):
            raise ValueError('Model chose an operation outside its actual tool capability')
        _text(value['instruction']); _text(value['reason'])
        if shaping_authorization is not None:
            from photo_shaping import shaping_operations,shaping_tool
            if value['tool']!=shaping_tool(shaping_authorization) or set(value['operations'])!=set(shaping_operations(shaping_authorization)):
                raise ValueError('The decision must cover exactly the explicitly selected shaping parts')
    return copy.deepcopy(value)


def _bind(payload, provenance, mapping, images):
    _validate_mapping(mapping, images)
    if (not isinstance(provenance, dict) or provenance.get('source') != mapping['transport_image']
            or provenance.get('network_used') is not True or provenance.get('generation_used') is not False
            or not isinstance(provenance.get('model'), str) or not provenance['model'].strip()):
        raise ValueError('Actual decision transport evidence does not match the supplied images')
    return {**payload, 'observed_images': [{'role': row['role'], 'sha256': row['sha256']} for row in images],
        'provenance': {**copy.deepcopy(provenance), 'model_requests': 1,
            'decision_image_mapping': copy.deepcopy(mapping), 'native_detail_acceptance': False}}


def decide_images(context, output, *, provider=None, vision_model=None, api_base=None,
                  api_url=None, api_key_env=None):
    """One model decision for a reservation made by ``session.advance``.

    The caller must pass this callback only under the checked task's text/vision
    authorization. A ready exact attempt may be reopened. Failed, cancelled or
    unknown attempts cannot be retried using the same directory.
    """
    from vision_transport import request_vision_json
    import workflow_timing
    available, unavailable, remaining = _capabilities(context)
    feedback = _feedback(context)
    images = copy.deepcopy(context['images'])
    output = Path(output).resolve()
    connection = {'provider': provider, 'vision_model': vision_model, 'api_base': api_base,
                  'api_url': api_url, 'api_key_env': api_key_env}
    environment = {name: os.environ.get(name) for name in ('COSPLAY_VISION_PROVIDER',
        'COSPLAY_VISION_MODEL', 'COSPLAY_VISION_API_BASE', 'COSPLAY_VISION_API_URL')}
    binding = {'context_sha256': identity(context), 'producer': _producer(),
               'connection_identity': identity({'selected': connection, 'environment': environment})}
    destination = output/'decision.json'
    if destination.is_file():
        saved = json.loads(destination.read_text(encoding='utf-8'))
        if saved.get('schema') != SCHEMA or saved.get('binding') != binding:
            raise ValueError('Saved decision belongs to another context, producer or connection')
        if saved.get('status') != 'ready':
            raise RuntimeError('Existing decision outcome must be reconciled, never silently resubmitted')
        mapping = json.loads((output/'image-mapping.json').read_text(encoding='utf-8'))
        if mapping != saved['image_mapping']:
            raise ValueError('Saved decision image roles changed')
        result = _bind(_payload(saved['proposed_decision'], available,context.get('shaping_authorization')), saved['provenance'], mapping, images)
        if result != saved['result']:
            raise ValueError('Saved model decision changed')
        return result
    output.mkdir(parents=True, exist_ok=True)
    record = {'schema': SCHEMA, 'status': 'reserved', 'binding': binding,
        'model_requests_attempted': 0, 'image_submissions': 0,
        'available_tools': available, 'unavailable_tools': unavailable,
        'automatic_resubmission': False, 'human_approved': False}
    with destination.open('x', encoding='utf-8') as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2); handle.flush(); os.fsync(handle.fileno())
    try:
        with workflow_timing.span('creative_decision_board', model_request=False, http_request=False):
            mapping = _prepare_input(images, output)
        record['image_mapping'] = mapping
        model_context = {'session_id': context['session_id'], 'scope': context['scope'], 'design': context['design'],
            'user_intent': copy.deepcopy(context.get('user_intent', {})),
            'images': [{'role': row['role'], 'sha256': row['sha256']} for row in images],
            'feedback': feedback, 'tools': available, 'unavailable_tools': unavailable,
            'review_objectives': copy.deepcopy(context.get('review_objectives', [])),
            'creative_delivery_eligibility':copy.deepcopy(context.get('creative_delivery_eligibility')),
            'remaining_budget_after_this_decision': remaining,
            'operation_stage':context.get('operation_stage'),'visual_review':context.get('visual_review',True),
            'zero_image_only':context.get('zero_image_only',False),
            'independent_evaluation_reserved_per_tool':int(context.get('visual_review',True)), 'human_approved': False}
        prompt = session.decision_prompt(model_context)+'\nIMAGE MAPPING: '+json.dumps(mapping['panels'])+\
            '\nIMAGE MAPPING lists physical panels from left to right. Roles joined by / share one panel '+\
            'because their bytes and decoder branch are identical. Best is the current best complete photograph; '+\
            'working is a reviewed work-in-progress candidate, not the accepted best. '+\
            'previous_attempt is a reviewed observation only, never an authorized tool input. '+\
            'When only the source is supplied, no finished candidate exists. Tools not listed under tools are unavailable. '+\
            'The displayed remaining budget excludes this decision. Independent review requests per tool: '+\
            str(int(context.get('visual_review',True)))+'. A review-disabled draft has no independent quality verdict. '+\
            'Write instruction and reason in '+('Chinese.' if context['scope']['language'] == 'zh' else 'English.')
        if context.get('shaping_authorization') is not None:
            from photo_shaping import part_limits,shaping_tool
            selected_parts=list(context['shaping_authorization']['request']['parts'])
            prompt += ('\nEXPLICIT OPT-IN SHAPING\n'+part_limits(context['shaping_authorization'])+
                '\nUse only '+shaping_tool(context['shaping_authorization'])+' with exactly all listed shape operations. Locate those parts from the actual unshaped best. '
                'Selected parts: '+', '.join(selected_parts)+'. Every selected part needs its own visible contour opportunity '
                'and preservation evidence; a visible face is not evidence that selected waist, arms or legs are available. '
                'The amount is fixed by the checked request: do not invent parameters. '
                'Stop if the selected parts or person cannot be identified confidently or are outside the photograph. '
                'Do not guess an off-frame body or infer an occluded contour. '
                'Do not substitute scene editing, exposure changes or joint editing for shaping. '
                'Only a tool explicitly requiring a region accepts a region field; otherwise the fixed canonical operation selects its own bound support.')
            if shaping_tool(context['shaping_authorization'])=='edit_region':
                prompt += ' Choose one bounded region containing the selected changes and necessary immediate context.'
            if 'face' in selected_parts:
                prompt += ('\nThe selected face uses narrow canonical cheek/jaw geometry support bound to the same unshaped U. '
                    'The central face, eyes, brows, nose and lips, and all non-band pixels within its face envelope are '
                    'hard-protected; another selected body operation cannot reopen that protection. '
                    'The deterministic face tool uses the checked amount and fixed canonical support; instruction cannot change its amount, field or sampling. It creates no new texture or light and spends zero image submissions. '
                    'This geometry is an editing-permission constraint, not proof of actual visibility, material ownership, '
                    'occlusion, naturalness or successful shaping. Judge those facts from the supplied pixels; stop when '
                    'the required selected contours cannot be observed confidently.')
        record.update(prompt=prompt, model_requests_attempted=1)
        atomic_json(destination, record)
        with workflow_timing.span('creative_tool_decision', model_request=True, http_request=True):
            payload, provenance = request_vision_json(Path(mapping['transport_image']['path']), output/'transport', prompt, **connection)
        record.update(proposed_decision=payload, provenance=provenance)
        _images(images)
        result = _bind(_payload(payload, available,context.get('shaping_authorization')), provenance, mapping, images)
        record.update(status='ready', result=result)
    except BaseException as error:
        record.update(status='interrupted' if isinstance(error, (InterruptedError, KeyboardInterrupt)) else 'failed',
            error={'type': type(error).__name__, 'message': str(error)[:1200]})
        atomic_json(destination, record)
        raise
    atomic_json(destination, record)
    return result
