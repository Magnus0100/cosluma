"""Actual image-board review adapter for the coordinator's editing session.

Only the configured vision transport makes a model request. Image roles and
execution provenance come from frozen bytes, never from the model's claims.
Saved drafts survive any review error; no review grants human approval.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import uuid

from PIL import Image, ImageDraw, ImageOps

from asset_identity import sha256_file
from current_model_registry import atomic_json
import photo_edit_session as session
from photo_editing import identity

SCHEMA = 'cosluma_photo_session_review.v1'


def _asset(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': sha256_file(path)}


def _verify_images(images, *, decision=False):
    if type(decision) is not bool:
        raise ValueError('Board purpose must be an explicit boolean')
    roles = [row.get('role') for row in images]
    allowed = ((['source', 'best'], ['source', 'best', 'working'], ['source', 'best', 'previous_attempt']) if decision else
               (['source', 'candidate'], ['source', 'candidate', 'previous_best']))
    if roles not in allowed:
        raise ValueError('Independent review needs ordered source/candidate/optional previous_best images')
    for image in images:
        if _asset(image['path']) != {key: image[key] for key in ('path', 'sha256')}:
            raise ValueError('Independent review image bytes changed')


def _producer():
    from runtime_resources import code_path
    return {name: sha256_file(code_path(name)) for name in (
        'photo_session_review.py', 'photo_edit_session.py', 'photo_editing.py',
        'vision_transport.py', 'raw_support.py', 'photo_request_budget.py','photo_shaping.py')}


def _connection_identity(connection):
    environment = {name: os.environ.get(name) for name in ('COSPLAY_VISION_PROVIDER',
        'COSPLAY_VISION_MODEL', 'COSPLAY_VISION_API_BASE', 'COSPLAY_VISION_API_URL')}
    return identity({'selected': connection, 'environment': environment})


def _panel_groups(images):
    """Group byte-identical inputs only within the same decoder branch."""
    from raw_support import is_raw
    groups = {}; ordered = []
    for index, image in enumerate(images):
        branch = 'raw' if is_raw(Path(image['path'])) else 'raster'
        key = (image['sha256'], branch)
        if key not in groups:
            group = {'indices': [], 'roles': [], 'decode_branch': branch}
            groups[key] = group; ordered.append(group)
        groups[key]['indices'].append(index)
        groups[key]['roles'].append(image['role'])
    return ordered


def review_region(context):
    """Derive inspection coordinates from actual local pixels, never design prose."""
    assets=context.get('appearance_assets') or {}
    candidate=assets.get('candidate')
    if candidate is None:return None
    from appearance_assets import validate_asset
    source=context['source_original']
    if _asset(source['path'])!={key:source[key] for key in ('path','sha256')}:
        raise ValueError('Regional review original bytes changed')
    contract={'source_sha256':source['sha256'],'scope':context['scope'],
              'design_id':context['design']['design_id']}
    receipt=validate_asset(candidate,**contract)
    images={row['role']:row for row in context['images']}
    if receipt['files']['preview']['sha256']!=images['candidate']['sha256']:
        raise ValueError('Regional review candidate differs from its typed preview')
    if context.get('shaping_baseline'):
        from photo_shaping import validate_authorization
        auth=context.get('shaping_authorization')
        validate_authorization(auth,**contract,base_appearance=context['shaping_base_appearance'],
            source_appearance=assets.get('source'))
        if (candidate['asset_id']!=context['shaping_base_appearance']['asset_id']
                or candidate['receipt_sha256']!=context['shaping_base_appearance']['receipt_sha256']):
            raise ValueError('Shaping baseline review must observe the actual unshaped input')
        return None
    if receipt['kind'] not in {'mixed_regional','geometric_transform'}:return None
    geometric=receipt['kind']=='geometric_transform'
    canvas=receipt['canvas_size'];core=(receipt['operation']['box'] if geometric else receipt['generation']['core_box'])
    if core==[0,0,*canvas]:return None
    bound_receipts={'candidate':receipt}
    for role,image in images.items():
        if role=='candidate':continue
        if role not in assets:
            raise ValueError('Regional review image has no matching native appearance')
        prior=validate_asset(assets[role],**contract)
        if (prior['files']['preview']['sha256']!=image['sha256'] or prior['canvas_size']!=canvas):
            raise ValueError('Regional review image differs from its native appearance')
        bound_receipts[role]=prior
    parent_role=None;parent_receipt=None
    for role in ('previous_best','source'):
        if role not in bound_receipts:continue
        prior=bound_receipts[role]
        actual={'asset_id':prior['asset_id'],'master_sha256':prior['files']['master']['sha256'],
                'preview_sha256':prior['files']['preview']['sha256']}
        if actual==receipt['parent']:
            parent_role=role;parent_receipt=prior;break
    if parent_role is None:
        raise ValueError('Regional review has no matching actual input parent')
    margin=max(1,min(core[2]-core[0],core[3]-core[1])//10)
    box=[max(0,core[0]-margin),max(0,core[1]-margin),
         min(canvas[0],core[2]+margin),min(canvas[1],core[3]+margin)]
    result={'canvas_size':canvas,'box':box,'core_box':core,
        'generation_context_box':None if geometric else receipt['generation']['context_box'],
        'candidate_asset_id':receipt['asset_id'],'candidate_receipt_sha256':candidate['receipt_sha256'],
        'actual_parent_role':parent_role,'parent_asset_id':parent_receipt['asset_id'],
        'parent_master_sha256':parent_receipt['files']['master']['sha256']}
    if geometric:
        result.update(operation_kind='geometric_transform',
            coverage_semantics='coordinate_displacement_support_not_new_generation',
            exact_protected_pixels_inside_core=True)
    return result


def review_prompt(context):
    prompt=session.evaluation_prompt(context)+'\nBOARD: panels from left to right are '+', '.join(
        ' / '.join(group['roles']) for group in _panel_groups(context['images']))+\
        '. Roles joined by / share one panel because their bytes and decoder branch are identical. '+\
        'Assess only what the supplied whole-image board supports; '+\
        'native-resolution detail has not been inspected. Return only the requested quality fields.'
    region=review_region(context)
    if region is not None:
        axes=('preserving the same two quality axes and additionally assessing shaping_fulfillment. '
            if context.get('shaping_authorization') else 'preserving the same two quality axes. ')
        prompt += ('\nREGIONAL OBSERVATION: the first row remains the complete photographs; '
            'judge whole-image quality and visual hierarchy first. The second row repeats the same '
            'roles in the same order, cropped at the SAME native canvas box. It shows the actual '
            'local edit plus nearby context, not a new composition or an additional edit. Only '
            'core_box is the permitted edit; surrounding context is not claimed changed. Inspect '
            'the requested local result and its integration, '+axes+
            'The actual input parent is explicitly identified; previous_best is not assumed to be '
            'that parent. Tiles are resampled previews, not native 1:1 certification or added camera '
            'detail. Use unknown for features too small or unclear; do not invent defects or request '
            'optional tiny-detail processing merely because a crop is supplied. Input profile: detailed_board.\n'
            +json.dumps(region,ensure_ascii=False,sort_keys=True))
    return prompt


def prepare_board(images, output, *, decision=False, region=None):
    """Freeze a role-labelled display-sRGB board and original-image mapping.

    Whole-image review can identify composition/lighting/design problems. It
    does not certify native-resolution skin, hair or fine prop geometry.
    """
    from raw_support import open_rgb
    images = copy.deepcopy(images); _verify_images(images, decision=decision)
    if decision and region is not None:raise ValueError('Tool decision boards do not use regional review crops')
    if region is not None:
        canvas=region['canvas_size'];box=region['box'];core=region['core_box']
        if (len(canvas)!=2 or len(box)!=4 or len(core)!=4
                or any(type(n) is not int for n in [*canvas,*box,*core])
                or not (0<=box[0]<=core[0]<core[2]<=box[2]<=canvas[0]
                    and 0<=box[1]<=core[1]<core[3]<=box[3]<=canvas[1])):
            raise ValueError('Regional review needs one valid native canvas box and edit core')
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    destination = output/'review-board.png'; mapping_path = output/'review-board.json'
    if destination.exists() or mapping_path.exists():
        raise FileExistsError('Review board already exists; do not replace frozen evidence')
    width, height, label_height, gutter = 512, 768, 36, 12
    groups = _panel_groups(images)
    first_row_height=height+label_height+2*gutter
    board = Image.new('RGB', (len(groups)*width+(len(groups)+1)*gutter,
        first_row_height if region is None else 2*(height+label_height)+3*gutter), '#17191c')
    draw = ImageDraw.Draw(board); mapping = [None]*len(images); panels = []
    for index, group in enumerate(groups):
        image = images[group['indices'][0]]
        original = open_rgb(Path(image['path']))
        try:
            if region is not None and list(original.size)!=region['canvas_size']:
                raise ValueError('Regional review images do not share the actual native canvas')
            tile = ImageOps.contain(original, (width, height), Image.Resampling.LANCZOS)
            x = gutter+index*(width+gutter)+(width-tile.width)//2
            y = gutter+label_height+(height-tile.height)//2
            board.paste(tile, (x, y))
            label = ' / '.join(group['roles'])
            draw.text((gutter+index*(width+gutter), gutter+8), label, fill='white')
            geometry = {'decoded_size': list(original.size), 'board_rect': [x, y, tile.width, tile.height]}
            panels.append({'panel': label, 'roles': group['roles'], 'decode_branch': group['decode_branch'],
                           'image_sha256': image['sha256'], **geometry})
            for logical_index in group['indices']:
                mapping[logical_index] = {**images[logical_index], **geometry, 'panel_index': index}
            tile.close()
            if region is not None:
                crop=original.crop(region['box'])
                try:
                    tile=ImageOps.contain(crop,(width,height),Image.Resampling.LANCZOS)
                    ry=first_row_height+label_height+(height-tile.height)//2
                    rx=gutter+index*(width+gutter)+(width-tile.width)//2
                    board.paste(tile,(rx,ry))
                    draw.text((gutter+index*(width+gutter),first_row_height+8),label+' (local edit)',fill='white')
                    details={'native_box':region['box'],'core_box':region['core_box'],
                        'crop_size':list(crop.size),'board_rect':[rx,ry,tile.width,tile.height],
                        'native_pixels_per_board_pixel':[crop.width/tile.width,crop.height/tile.height],
                        'upscaled':tile.width>crop.width or tile.height>crop.height}
                    panels[-1]['region']=copy.deepcopy(details)
                    for logical_index in group['indices']:mapping[logical_index]['region']=copy.deepcopy(details)
                    tile.close()
                finally:crop.close()
        finally:
            original.close()
    temporary = output/(uuid.uuid4().hex+'.png')
    try:
        with temporary.open('xb') as handle:
            board.save(handle, format='PNG'); handle.flush()
            os.fsync(handle.fileno())
        _verify_images(images, decision=decision)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True); board.close()
    result = {'schema': 'cosluma_independent_review_board.v1', 'images': mapping, 'panels': panels,
        'board': _asset(destination), 'color_domain': 'display_srgb',
        'inspection_scope': 'whole_image', 'native_detail_acceptance': False}
    if region is not None:
        result.update(region=copy.deepcopy(region),inspection_scope='whole_image_and_edit_region',
                      input_profile='detailed_board')
    atomic_json(mapping_path, result)
    return result


def _validate_payload(payload, has_best, *, shaping=False):
    axes=(*session.AXES,'shaping_fulfillment') if shaping else session.AXES
    if not isinstance(payload, dict) or set(payload) != {*axes, 'comparison', 'primary_issue'}:
        raise ValueError('Vision review must return quality findings only; provenance is adapter-owned')
    for axis in axes:
        row = payload[axis]
        if (not isinstance(row, dict) or set(row) != {'status', 'findings'}
                or row['status'] not in session.RANK or not isinstance(row['findings'], list)
                or not row['findings'] or any(not isinstance(text, str) or not text.strip()
                                             for text in row['findings'])):
            raise ValueError('Vision review must separately explain both quality axes')
    if payload['comparison'] not in ({'better', 'equivalent', 'worse', 'unknown'} if has_best else {'first_candidate'}):
        raise ValueError('Vision comparison differs from actual supplied images')
    if not isinstance(payload['primary_issue'], str) or not payload['primary_issue'].strip():
        raise ValueError('Vision review requires one main finding or an explicit none')
    return copy.deepcopy(payload)


def evaluate_images(context, output, *, provider=None, vision_model=None,
                    api_base=None, api_url=None, api_key_env=None):
    """Perform one reserved independent observation, or reuse its exact receipt."""
    from vision_transport import request_vision_json
    import workflow_timing
    images = context['images']; _verify_images(images)
    shaping=bool(context.get('shaping_authorization') and not context.get('shaping_baseline'))
    region=review_region(context)
    input_profile='standard' if region is None else 'detailed_board'
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    connection = {'provider': provider, 'vision_model': vision_model, 'api_base': api_base,
                  'api_url': api_url, 'api_key_env': api_key_env}
    binding = {'context_sha256': identity(context), 'images': images, 'producer': _producer(),
               'connection_identity': _connection_identity(connection),
               'inspection':{'region':region,'input_profile':input_profile}}
    destination = output/'review.json'
    if destination.exists():
        saved = json.loads(destination.read_text(encoding='utf-8'))
        if saved.get('schema') != SCHEMA or saved.get('binding') != binding:
            raise ValueError('Existing review belongs to other images, objective or code')
        if saved.get('status') != 'ready':
            raise RuntimeError('Review outcome is unresolved; query the existing attempt instead of resubmitting')
        board = saved['board']
        if _asset(board['board']['path']) != board['board']:
            raise ValueError('Frozen review board changed')
        mapping = json.loads((output/'board'/'review-board.json').read_text(encoding='utf-8'))
        if mapping != board:
            raise ValueError('Frozen review role mapping changed')
        if board.get('region')!=region:
            raise ValueError('Frozen review crop differs from the actual candidate edit')
        payload = _validate_payload(saved['proposed_review'], len(images) == 3,shaping=shaping)
        result = _bind_result(payload, saved['provenance'], board, images)
        if result != saved['result']:
            raise ValueError('Saved review result changed')
        return result
    record = {'schema': SCHEMA, 'status': 'reserved', 'binding': binding,
              'model_requests_attempted': 0, 'image_submissions': 0,
              'automatic_resubmission': False, 'human_approved': False}
    with destination.open('x', encoding='utf-8') as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2); handle.flush(); os.fsync(handle.fileno())
    try:
        with workflow_timing.span('independent_review_board', model_request=False, http_request=False):
            board = prepare_board(images, output/'board',region=region)
        record['board'] = board
        prompt = review_prompt(context)
        record['prompt'] = prompt
        record['model_requests_attempted'] = 1
        atomic_json(destination, record)
        with workflow_timing.span('independent_image_review', model_request=True, http_request=True):
            payload, provenance = request_vision_json(Path(board['board']['path']), output/'transport', prompt,
                **connection,**({'input_profile':input_profile} if region is not None else {}))
        record.update(proposed_review=payload, provenance=provenance)
        _verify_images(images)
        result = _bind_result(_validate_payload(payload, len(images) == 3,shaping=shaping), provenance, board, images)
        record.update(status='ready', result=result)
    except BaseException as error:
        record.update(status='interrupted' if isinstance(error, (InterruptedError, KeyboardInterrupt)) else 'failed',
            error={'type': type(error).__name__, 'message': str(error)[:1200]})
        atomic_json(destination, record)
        raise
    atomic_json(destination, record)
    return result


def _bind_result(payload, provenance, board, images):
    if (not isinstance(provenance, dict) or provenance.get('source') != board['board']
            or provenance.get('network_used') is not True or provenance.get('generation_used') is not False
            or not provenance.get('model') or _asset(board['board']['path']) != board['board']
            or board.get('region') is not None and provenance.get('input_profile')!='detailed_board'):
        raise ValueError('Vision transport did not observe the frozen actual-image board')
    return {**payload, 'observed_images': [{'role': row['role'], 'sha256': row['sha256']} for row in images],
        'provenance': {**copy.deepcopy(provenance), 'model_requests': 1,
            'review_board': board['board'], 'review_inspection_scope': board['inspection_scope'],
            'native_detail_acceptance': False,
            **({'regional_observation':copy.deepcopy(board['region'])} if board.get('region') is not None else {})}}


def _path(root, path):
    path = Path(path)
    return (path if path.is_absolute() else root/path).resolve()


def _services_allowed(run, args):
    intents = [value for value in (run.get('task_intent'), (run.get('options') or {}).get('task_intent'),
               getattr(args, 'task_intent', None)) if isinstance(value, dict)]
    return bool(intents and not getattr(args, 'local_only', False) and all(
        value.get('mode') != 'local_portrait' and value.get('services', {}).get('text_vision') is True
        and value.get('services', {}).get('visual_review') is True for value in intents))


def review_saved_run(root, run, args):
    """Attach one independent review to a saved scoped draft, preserving pixels.

    The caller saves ``run`` atomically through its existing publication path.
    No repair or generation runs here. A checked scope needs an explicit finite
    services.text_vision_requests_upper_bound; this adapter never grants budget.
    """
    if not _services_allowed(run, args):
        run['photo_edit_session'] = {'status': 'not_requested', 'human_approved': False,
            'model_requests_attempted': 0, 'reason': 'Independent visual review is disabled by the task'}
        return run
    if (run.get('task_plan') or {}).get('editing_iteration') is not None:
        # Its coordinator already owns a separate 2K decision/review session.
        # Reading it must not relabel its evidence or spend the origin quota.
        return run
    root = Path(root).resolve()
    directory = None
    origin_budget = None
    ticket = None
    invocation_attempts = 0
    try:
        import photo_request_budget as requests
        from runtime_resources import code_path
        plan = run.get('task_plan') or {}
        if plan.get('ready') is not True or not plan.get('plan_id'):
            raise ValueError('Independent editing review requires a ready checked production plan')
        generation = run.get('generation_design') or {}
        design = generation.get('observed_design')
        scope = plan.get('editing_scope')
        if not design or not scope:
            raise ValueError('Independent editing review requires an actual observed scoped design')
        source = _path(root, run['source']['path']); source_asset = _asset(source)
        if source_asset['sha256'] != run['source']['sha256'] or generation.get('source_sha256') != source_asset['sha256']:
            raise ValueError('Saved draft and observed design source bytes differ')
        image = _path(root, (run.get('artifacts') or {})['auto_draft'])
        generation_budget = run.get('generation_budget') or {}
        images_used = generation_budget.get('total_image_submissions_including_effects',
            generation_budget.get('total_image_submissions', generation_budget.get('image_submissions', 0)))
        if type((plan.get('services') or {}).get('text_vision_requests_upper_bound')) is not int:
            raise ValueError('Scoped review has no finite checked vision-request budget')
        run_id = str(run['run_id'])
        if not run_id or run_id in {'.', '..'} or '/' in run_id or '\\' in run_id:
            raise ValueError('Invalid scoped review run identity')
        directory = root/'outputs'/'runs'/run_id/'editing-session'
        origin_budget = requests.prepare(root, run)
        options = run.get('options') or {}
        connection = {key: getattr(args, key, None) or options.get(key) for key in
            ('provider', 'vision_model', 'api_base', 'api_url', 'api_key_env')}
        state = session.snapshot(directory) if (directory/'session.json').is_file() else None
        limits = {'image_submissions': 0, 'vision_requests': 1, 'tool_calls': 0,
                  'local_repairs': 0, 'improvement_steps': 0}
        if state is not None:
            saved_limits = state['binding']['budget']
            if (any(state['binding']['initial_usage'].values()) or
                    any(saved_limits[key] for key in saved_limits if key != 'vision_requests')
                    or saved_limits['vision_requests'] not in (0, 1)):
                raise ValueError('Historical session usage is retained as evidence, not relabelled as a current review')
            limits = saved_limits
            if state['candidates'] and state['candidates'][0]['image']['sha256'] != sha256_file(image):
                raise ValueError('Current saved draft differs from this review session; retain the prior review as historical')
        if state is None or not state['candidates'] or state['candidates'][0]['review'] is None:
            context_for_identity = {'scope': scope, 'design': design,
                'user_intent': (generation.get('art_direction') or {}).get('user_intent') or {},
                'images': [{'role': 'source', **source_asset}, {'role': 'candidate', **_asset(image)}]}
            key = requests.endpoint(source_sha256=source_asset['sha256'], candidate_sha256=sha256_file(image),
                scope_id=scope['scope_id'], design_id=design['design_id'], prompt=review_prompt(context_for_identity),
                connection_identity=_connection_identity(connection), decoder_identity=sha256_file(code_path('raw_support.py')))
            ticket = requests.reserve(origin_budget, key, run_id=run_id)
            if state is None:
                limits['vision_requests'] = int(ticket['owner'])
        state = session.open_session(directory, source=source, scope=scope, design=design,
            execution_identity=identity({'plan_id': plan['plan_id'], 'run_id': run_id}),
            producer_identity=identity(_producer()), budget=limits,
            user_intent=(generation.get('art_direction') or {}).get('user_intent'))
        if not state['candidates']:
            provenance = {'enclosing_checked_task': plan['plan_id'], 'run_id': run_id,
                          'saved_artifact': _asset(image), 'origin_budget': origin_budget}
            if ticket and not ticket['owner']:
                state = session.adopt_reviewed_candidate(directory, image, review=ticket['result'], provenance=provenance)
            else:
                state = session.adopt_candidate(directory, image, provenance=provenance)
        elif ticket and not ticket['owner'] and state['pending_candidate_id']:
            unresolved = [row for row in state['attempts'] if row['kind'] == 'evaluate'
                          and row['status'] in {'reserved', 'unknown', 'interrupted'}]
            if len(unresolved) != 1:
                raise ValueError('Existing review session needs explicit evidence reconciliation')
            state = session.reconcile_attempt(directory, unresolved[0]['attempt_id'], outcome='completed',
                evidence={'origin_budget': ticket['origin_budget']}, result=ticket['result'], tools=[])
        def evaluate(context):
            nonlocal invocation_attempts
            if ticket is None or ticket['owner'] is not True:
                raise RuntimeError('Review transport requires the current origin-budget reservation')
            destination = directory/'reviews'/context['attempt_id']
            invocation_attempts += 1
            try:
                result = evaluate_images(context, destination, **connection)
                requests.finish(ticket, result=result, evidence=_asset(destination/'review.json'), actual_model_requests=1)
                return result
            except BaseException:
                record_path = destination/'review.json'
                record = json.loads(record_path.read_text(encoding='utf-8')) if record_path.is_file() else {}
                requests.finish(ticket, evidence=_asset(record_path) if record_path.is_file() else None,
                                actual_model_requests=record.get('model_requests_attempted'))
                raise
        def no_decision(_context):
            raise RuntimeError('Saved-draft review does not authorize creative tool decisions')
        cancel_path = getattr(args, 'cancel_request_path', None)
        if state['phase'] == 'evaluate':
            state = session.advance(directory, tools=[], decide=no_decision, evaluate=evaluate,
                model_identity=identity({'connection': connection, 'producer': _producer()}),
                cancel=lambda: bool(cancel_path and Path(cancel_path).is_file()))
        if state['status'] == 'active' and state['phase'] == 'decide':
            # No repair is authorized here. The zero remaining local vision
            # budget stops deterministically, without invoking no_decision.
            state = session.advance(directory, tools=[], decide=no_decision, evaluate=evaluate,
                model_identity=identity({'connection': connection, 'producer': _producer()}))
        selected = session.selected_output(directory)
        # This adapter is deliberately one review; it does not run another
        # decision call when the objective is incomplete.
        run['photo_edit_session'] = {'status': state['status'], 'phase': state['phase'],
            'session_id': state['session_id'], 'path': str(directory/'session.json'),
            'reserved_budget_usage': state['used'], 'automated_target_met': state['automated_target_met'],
            'human_approved': False, 'independent_evaluation': (selected['candidate'] or {}).get('review'),
            'stop_reason': state['stop_reason'], 'native_detail_acceptance': False,
            'automatic_image_retry': False, 'origin_request_budget': requests.snapshot(origin_budget),
            'historical_image_submissions': images_used, 'historical_design_requests': 1,
            'review_attempts_this_invocation': invocation_attempts,
            'review_evidence_reused': bool(ticket and not ticket['owner'])}
        run.setdefault('artifacts', {})['photo_edit_session'] = str(directory/'session.json')
    except Exception as error:
        run['photo_edit_session'] = {'status': 'needs_attention', 'human_approved': False,
            'automated_target_met': False, 'automatic_image_retry': False,
            'error': {'type': type(error).__name__, 'message': str(error)[:1200]},
            'saved_draft_preserved': True, 'review_attempts_this_invocation': invocation_attempts}
        if origin_budget is not None:
            run['photo_edit_session']['origin_request_budget'] = requests.snapshot(origin_budget)
        if directory is not None and (directory/'session.json').is_file():
            run['photo_edit_session']['path'] = str(directory/'session.json')
            run.setdefault('artifacts', {})['photo_edit_session'] = str(directory/'session.json')
    return run
