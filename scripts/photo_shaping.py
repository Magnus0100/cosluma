"""Checked opt-in regional shaping and exact immutable-baseline restoration.

This modifier consumes current typed pixels through the production coordinator.
It does not replay old sessions, infer approval or reconstruct a baseline.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from asset_identity import sha256_file
from photo_editing import identity, validate_scope

SCHEMA = 'cosluma_shaping_authorization.v1'
STATE_SCHEMA = 'cosluma_shaping_state.v1'
PARTS = ('face', 'waist', 'arms', 'legs')
_AUTH_KEYS = {'schema', 'parent_run_id', 'parent_manifest_sha256', 'parent_appearance',
    'base_appearance', 'source_appearance', 'source_sha256', 'parent_image_sha256',
    'base_image_sha256', 'base_master_sha256', 'scope_id', 'design_id', 'request',
    'manual_provider_target', 'manual_target', 'authorization_id'}


def normalize_request(value):
    if not isinstance(value, dict) or not isinstance(value.get('action'), str) or value['action'] not in {'apply', 'restore'}:
        raise ValueError('Shaping requires an explicit apply or restore action')
    if value['action'] == 'restore':
        if set(value) != {'action'}:
            raise ValueError('Restore accepts only its explicit action')
        return {'action': 'restore'}
    if set(value) != {'action', 'parts'} or not isinstance(value['parts'], dict):
        raise ValueError('Shaping apply requires explicit selected parts')
    if set(value['parts']) - set(PARTS):
        raise ValueError('Unsupported shaping part')
    parts = {}
    for part in PARTS:
        if part not in value['parts']:
            continue
        amount = value['parts'][part]
        if type(amount) not in (int, float) or not math.isfinite(amount) or not 0 <= amount <= 1:
            raise ValueError('Shaping amounts must be finite numbers from zero to one')
        if amount:
            parts[part] = float(amount)
    if not parts:
        raise ValueError('Select at least one nonzero shaping part')
    return {'action': 'apply', 'parts': parts}


def execution_method(request):
    request = normalize_request(request)
    if request['action'] == 'restore':
        return 'exact_restore'
    if 'face' in request['parts']:
        if set(request['parts']) != {'face'}:
            raise ValueError('Face contour deformation currently requires a separate face-only task')
        return 'face_contour_deform'
    return 'edit_region'


def image_submissions(request):
    return int(execution_method(request) == 'edit_region')


def shaping_tool(auth):
    return execution_method(_contract(auth)['request'])


def _sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Shaping requires exact content identities')
    return value


def _component(value):
    if not isinstance(value, str) or not value or value in {'.', '..'} or any(c in value for c in '/\\:'):
        raise ValueError('Shaping requires a saved parent run identity')
    return value


def _descriptor(value):
    if (not isinstance(value, dict) or set(value) != {'path', 'asset_id', 'receipt_sha256'}
            or not isinstance(value['path'], str) or not value['path'].strip()):
        raise ValueError('Shaping requires a typed appearance descriptor')
    _sha(value['asset_id']); _sha(value['receipt_sha256'])
    return {k: value[k] for k in ('asset_id', 'receipt_sha256')}


def _contract(auth):
    if not isinstance(auth, dict) or set(auth) != _AUTH_KEYS or auth.get('schema') != SCHEMA:
        raise ValueError('Current checked shaping authorization required')
    if auth['authorization_id'] != identity({k: v for k, v in auth.items() if k != 'authorization_id'}):
        raise ValueError('Shaping authorization identity changed')
    _component(auth['parent_run_id'])
    for key in ('parent_manifest_sha256', 'source_sha256', 'parent_image_sha256',
                'base_image_sha256', 'base_master_sha256', 'scope_id', 'design_id'):
        _sha(auth[key])
    for key in ('parent_appearance', 'base_appearance', 'source_appearance'):
        _descriptor(auth[key])
    request = normalize_request(auth['request'])
    if auth['request'] != request:
        raise ValueError('Shaping request is not canonical')
    from photo_request_budget import manual_target_key
    expected = None
    if request['action'] == 'apply':
        expected = manual_target_key(source_sha256=auth['source_sha256'], base_sha256=auth['base_image_sha256'],
            scope_id=auth['scope_id'], design_id=auth['design_id'], provider_target=auth['manual_provider_target'],
            shaping_request=request)
    elif auth['manual_provider_target'] is not None:
        raise ValueError('Exact restore must not resolve a model target')
    if auth['manual_target'] != expected:
        raise ValueError('Shaping semantic request target changed')
    return copy.deepcopy(auth)


def validate_authorization(auth, *, source_sha256, scope, design_id=None,
                           base_appearance=None, source_appearance=None):
    """Validate checked semantics and current U using optional owned IO locators.

    Historical authorization bytes remain unchanged after relocation. Its source
    descriptor is validated as bytes when an owned source override is supplied;
    the baseline receipt already verifies its own source-origin contract.
    """
    auth = validate_authorization_metadata(auth, source_sha256=source_sha256, scope=scope, design_id=design_id)
    scope = validate_scope(scope)
    from appearance_assets import validate_asset
    base = auth['base_appearance'] if base_appearance is None else base_appearance
    if _descriptor(base) != _descriptor(auth['base_appearance']):
        raise ValueError('Owned shaping baseline differs from checked U')
    binding = {'source_sha256': source_sha256, 'scope': scope, 'design_id': auth['design_id']}
    receipt = validate_asset(base, **binding)
    if (receipt['files']['preview']['sha256'] != auth['base_image_sha256']
            or receipt['files']['master']['sha256'] != auth['base_master_sha256']):
        raise ValueError('Shaping baseline pixels changed')
    if source_appearance is not None:
        if _descriptor(source_appearance) != _descriptor(auth['source_appearance']):
            raise ValueError('Owned source baseline differs from checked source')
        original = validate_asset(source_appearance, **binding)
        if original['kind'] != 'source_baseline' or original['source_origin'] != receipt['source_origin'] or original['canvas_size'] != receipt['canvas_size']:
            raise ValueError('Shaping baseline does not share its original source canvas')
    return auth


def validate_authorization_metadata(auth, *, source_sha256, scope, design_id=None):
    """Verify immutable checked semantics without touching historical IO paths."""
    auth = _contract(auth); scope = validate_scope(scope)
    if auth['source_sha256'] != source_sha256 or auth['scope_id'] != scope['scope_id'] or (
            design_id is not None and auth['design_id'] != design_id):
        raise ValueError('Shaping authorization belongs to different source/scope/design')
    return auth


def validate_state(state, *, current, source_sha256, scope, design_id):
    """Return the canonical saved state after verifying current and owned U."""
    if not isinstance(state, dict) or set(state) != {'schema', 'base_appearance', 'enabled', 'request', 'authorization', 'selected_asset_id'}:
        raise ValueError('Current shaping state required')
    if state['schema'] != STATE_SCHEMA or type(state['enabled']) is not bool:
        raise ValueError('Shaping state schema or enabled flag changed')
    request = normalize_request(state['request'])
    auth = validate_authorization(state['authorization'], source_sha256=source_sha256, scope=scope,
        design_id=design_id, base_appearance=state['base_appearance'])
    if request != state['request'] or request != auth['request'] or state['enabled'] is not (request['action'] == 'apply'):
        raise ValueError('Saved shaping action differs from its checked request')
    from appearance_assets import validate_asset
    receipt = validate_asset(current, source_sha256=source_sha256, scope=scope, design_id=design_id)
    base = validate_asset(state['base_appearance'], source_sha256=source_sha256, scope=scope, design_id=design_id)
    if (state['selected_asset_id'] != current['asset_id'] or receipt['source_origin'] != base['source_origin']
            or receipt['canvas_size'] != base['canvas_size']):
        raise ValueError('Saved shaping state differs from actual current pixels')
    if not state['enabled'] and _descriptor(current) != _descriptor(state['base_appearance']):
        raise ValueError('Restored current pixels must be the exact owned baseline')
    if state['enabled'] and current['asset_id'] == state['base_appearance']['asset_id']:
        raise ValueError('An unchanged baseline cannot claim completed shaping')
    expected_kind = 'geometric_transform' if execution_method(request) == 'face_contour_deform' else 'mixed_regional'
    if state['enabled'] and (receipt['kind'] != expected_kind or receipt['parent'] != {
            'asset_id': base['asset_id'], 'master_sha256': base['files']['master']['sha256'],
            'preview_sha256': base['files']['preview']['sha256']}):
        raise ValueError('Completed shaping requires its actual checked operation directly from U')
    if state['enabled'] and expected_kind == 'geometric_transform' and receipt['operation']['amount'] != request['parts']['face']:
        raise ValueError('Geometric shaping amount differs from its checked request')
    return copy.deepcopy(state)


def shaping_operations(auth):
    request = _contract(auth)['request']
    return ['shape_' + part for part in PARTS if part in request.get('parts', {})]


def operations(scope, authorization=None):
    scope = validate_scope(scope)
    result = list(scope['allowed_operations'])
    if authorization is not None:
        auth = _contract(authorization)
        if auth['scope_id'] != scope['scope_id']:
            raise ValueError('Shaping permissions belong to another scope')
        result += shaping_operations(auth)
    return result


def part_limits(auth):
    request = _contract(auth)['request']
    if request['action'] != 'apply':
        return 'Restore the exact saved baseline; no generated edit is authorized.'
    selected = ', '.join(f'{part}: {amount:g} of the explicit normalized 0..1 strength' for part, amount in request['parts'].items())
    return ('Explicit opt-in shaping exception: gently narrow only these selected contours: ' + selected +
        '. Strength is the checked user request, not permission for the model to choose a new amount. '
        'Preserve all unselected body parts, identity, facial features, makeup, expression, gaze, mouth opening, '
        'pose, height, limb length, costume seams and texture, important props, hands, scene geometry and lighting. '
        'Face selection permits only natural face-contour narrowing, never replacement or beautification of facial features. '
        'No pose changes, stretching, background bending, or whole-canvas shaping.')


def _path(root, value):
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def frozen_restore_source(root, parent_run_id, source_path, scope):
    """Return checked source identity from owned frozen planes, without RAW IO."""
    root = Path(root).resolve(); scope = validate_scope(scope)
    manifest = root / 'outputs/runs' / _component(parent_run_id) / 'manifest.json'
    manifest_sha = sha256_file(manifest)
    parent = json.loads(manifest.read_text(encoding='utf8'))
    if (parent.get('run_id') != parent_run_id or (parent.get('result_publication') or {}).get('state') != 'committed'
            or (parent.get('task_plan') or {}).get('editing_scope') != scope):
        raise ValueError('Frozen restore requires its committed current scoped parent')
    source = parent.get('source') or {}
    if not isinstance(source_path, (str, Path)) or _path(root, source_path) != _path(root, source['path']):
        raise ValueError('Frozen restore cannot substitute a different missing source')
    source_sha = _sha(source.get('sha256'))
    if (type(source.get('bytes')) is not int or source['bytes'] < 1
            or ((parent.get('task_plan') or {}).get('inputs') or {}).get('source', {}).get('sha256') != source_sha):
        raise ValueError('Frozen restore lost its checked original identity')
    options = parent.get('options') or {}; current = options.get('photo_appearance')
    design_id = parent['generation_design']['observed_design']['design_id']
    binding = {'source_sha256': source_sha, 'scope': scope, 'design_id': design_id}
    state = validate_state(options.get('shaping_state'), current=current, **binding)
    validate_authorization(state['authorization'], **binding, base_appearance=state['base_appearance'],
        source_appearance=parent.get('initial_source_appearance'))
    from appearance_assets import validate_asset
    receipt = validate_asset(current, **binding)
    draft = _path(root, parent['artifacts']['auto_draft'])
    if sha256_file(draft) != receipt['files']['preview']['sha256'] or sha256_file(manifest) != manifest_sha:
        raise ValueError('Frozen restore parent pixels or manifest changed')
    return {'path': str(_path(root, source['path'])), 'sha256': source_sha, 'bytes': source['bytes']}


def _parent_requests_known(root, parent):
    """Read actual original reservation ledgers, never trust a copied summary."""
    import photo_request_budget as budget
    current = parent; visited = set()
    while True:
        run_id = _component(current['run_id'])
        if run_id in visited:
            raise ValueError('Cyclic request origin history')
        visited.add(run_id)
        auth = (current.get('task_plan') or {}).get('shaping_authorization') or (current.get('task_plan') or {}).get('editing_refinement_authorization')
        if auth and auth.get('manual_target'):
            if not budget._manual_path(root, auth['manual_target']).is_file():
                raise ValueError('Saved manual request guard is unavailable; restore its original ledger')
            budget.check_manual_target(root, auth['manual_target'])
            outcome = budget.classify_manual_outcome(root, auth['manual_target'], run_id=run_id, require_session=True)
            if outcome['status'] != 'known' or outcome.get('session_status') == 'active':
                raise ValueError('Inspect the existing unresolved parent request before shaping')
        plan = current.get('task_plan') or {}
        if plan.get('mode') == 'create_reference':
            origin = (current.get('generation_budget') or {}).get('origin_run_id') or run_id
            if origin != run_id:
                current = json.loads((root / 'outputs/runs' / _component(origin) / 'manifest.json').read_text(encoding='utf8'))
                continue
            path = root / 'outputs/request-budgets' / run_id / 'vision.json'
            binding, _ = budget._origin(root, current)
            state = budget._read(path)
            if state['binding'] != binding:
                raise ValueError('Original request ledger differs from its checked source/design')
            if 'editing_initialization' in binding and not isinstance(state.get('operations'), list):
                raise ValueError('Original native operation reservations are unavailable')
            if any(row.get('status') != 'ready' for row in state['reviews'] + state.get('operations', [])):
                raise ValueError('Original model request outcome remains unknown; retain its reservations')
            report = budget._report(path, state)
            recorded = current.get('generation_budget') or {}
            if (report['reserved_image_requests'] < recorded.get('total_image_submissions', 0)
                    or report['reserved_vision_requests'] < recorded.get('text_vision_submissions', 0)):
                raise ValueError('Original request reservations disappeared; do not replenish them')
            return
        previous = current.get('parent_run_id') or (current.get('generation_budget') or {}).get('historical_generation_parent_run_id')
        if not previous:
            raise ValueError('Original request ledger is unavailable; inspect the original task')
        current = json.loads((root / 'outputs/runs' / _component(previous) / 'manifest.json').read_text(encoding='utf8'))


def authorize(root, *, parent_run_id, source, reference, scope, request, provider_target=None):
    root = Path(root).resolve(); scope = validate_scope(scope); request = normalize_request(request)
    execution_method(request)
    parent_run_id = _component(parent_run_id)
    manifest = root / 'outputs/runs' / parent_run_id / 'manifest.json'
    manifest_sha = sha256_file(manifest)
    parent = json.loads(manifest.read_text(encoding='utf8'))
    if parent.get('run_id') != parent_run_id or (parent.get('result_publication') or {}).get('state') != 'committed':
        raise ValueError('Shaping requires its completely saved parent version')
    if (parent.get('task_plan') or {}).get('editing_scope') != scope:
        raise ValueError('Shaping cannot change the parent creative scope')
    from generated_appearance_render import validate_run
    validate_run(root, parent)
    source_path = _path(root, parent['source']['path'])
    source_sha = (_sha(parent['source']['sha256']) if request['action'] == 'restore' and not source_path.exists()
                  else sha256_file(source_path))
    if source_sha != parent['source']['sha256'] or source.get('sha256') != source_sha or _path(root, source['path']) != source_path:
        raise ValueError('Shaping source differs from the saved original')
    design_id = parent['generation_design']['observed_design']['design_id']
    options = parent.get('options') or {}
    current = options.get('photo_appearance'); original = parent.get('initial_source_appearance')
    from appearance_assets import validate_asset
    binding = {'source_sha256': source_sha, 'scope': scope, 'design_id': design_id}
    current_receipt = validate_asset(current, **binding)
    source_receipt = validate_asset(original, **binding)
    if source_receipt['kind'] != 'source_baseline' or source_receipt['source_origin'] != current_receipt['source_origin']:
        raise ValueError('Saved current pixels differ from their source baseline')
    draft = _path(root, parent['artifacts']['auto_draft'])
    if sha256_file(draft) != current_receipt['files']['preview']['sha256']:
        raise ValueError('Saved parent output differs from its current typed preview')
    prior = options.get('shaping_state')
    if prior is not None:
        prior = validate_state(prior, current=current, **binding)
        base = prior['base_appearance']
    else:
        if request['action'] == 'restore':
            raise ValueError('This saved version has no shaping baseline to restore')
        base = current
    base_receipt = validate_asset(base, **binding)
    base_preview = Path(base['path']) / base_receipt['files']['preview']['file']
    if reference.get('sha256') != base_receipt['files']['preview']['sha256'] or _path(root, reference['path']) != base_preview.resolve():
        raise ValueError('Shaping must condition on its exact unchanged adjustment-before baseline U')
    target = None
    if request['action'] == 'apply':
        _parent_requests_known(root, parent)
        from photo_request_budget import manual_target_key, check_manual_target
        target = manual_target_key(source_sha256=source_sha, base_sha256=base_receipt['files']['preview']['sha256'],
            scope_id=scope['scope_id'], design_id=design_id, provider_target=provider_target, shaping_request=request)
        check_manual_target(root, target)
    elif provider_target is not None:
        raise ValueError('Exact restore must not resolve model credentials or a provider target')
    if sha256_file(manifest) != manifest_sha:
        raise ValueError('Parent changed while checking shaping')
    result = {'schema': SCHEMA, 'parent_run_id': parent_run_id, 'parent_manifest_sha256': manifest_sha,
        'parent_appearance': copy.deepcopy(current), 'base_appearance': copy.deepcopy(base),
        'source_appearance': copy.deepcopy(original), 'source_sha256': source_sha,
        'parent_image_sha256': current_receipt['files']['preview']['sha256'],
        'base_image_sha256': base_receipt['files']['preview']['sha256'],
        'base_master_sha256': base_receipt['files']['master']['sha256'],
        'scope_id': scope['scope_id'], 'design_id': design_id, 'request': request,
        'manual_provider_target': copy.deepcopy(provider_target), 'manual_target': target}
    result['authorization_id'] = identity(result)
    return result


def revalidate(root, plan):
    auth = plan['shaping_authorization']; inputs = plan['inputs']
    current = authorize(root, parent_run_id=auth['parent_run_id'], source=inputs['source'],
        reference=inputs['reference'], scope=plan['editing_scope'], request=plan['shaping_request'],
        provider_target=auth['manual_provider_target'])
    if current != auth:
        raise ValueError('Saved shaping parent or inputs changed after planning')
    return current


def execute(root, args):
    """Entered only by the current checked coordinator, with one request guard."""
    root = Path(root).resolve(); auth = revalidate(root, args.task_plan)
    if auth['request']['action'] == 'restore':
        return _execute_authorized(root, args, auth)
    from photo_request_budget import manual_provider_target, manual_target_guard
    target = manual_provider_target(vars(args), image_generation=bool(image_submissions(auth['request'])))
    if target != auth['manual_provider_target']:
        raise ValueError('Shaping provider target changed after planning')
    with manual_target_guard(root, auth['manual_target'], run_id=args.run_id):
        return _execute_authorized(root, args, auth)


def _reference_record(root, asset, receipt):
    import single_source_workflow as workflow
    preview = Path(asset['path']) / receipt['files']['preview']['file']
    result = workflow.candidate_record(preview, root, provider='frozen_appearance_reuse',
        model=None, prompt='', generation_id=None)
    result['photo_appearance'] = copy.deepcopy(asset)
    return result


def _execute_authorized(root, args, auth):
    import single_source_workflow as workflow
    from appearance_assets import validate_asset, materialize_asset
    plan = copy.deepcopy(args.task_plan); scope = plan['editing_scope']
    path, run = workflow._source_revision(root, auth['parent_run_id'], snapshot=False, child_run_id=args.run_id)
    options = run.setdefault('options', {})
    owned_u = (options.get('shaping_state') or {}).get('base_appearance') or options['photo_appearance']
    binding = {'source_sha256': run['source']['sha256'], 'scope': scope, 'design_id': auth['design_id']}
    if _descriptor(owned_u) != _descriptor(auth['base_appearance']):
        owned_u = materialize_asset(auth['base_appearance'], path.parent / 'inputs/photo-appearances' /
            auth['base_appearance']['asset_id'], **binding)
    validate_authorization(auth, **binding, base_appearance=owned_u, source_appearance=run['initial_source_appearance'])
    u_receipt = validate_asset(owned_u, **binding)
    run.update(task_plan=copy.deepcopy(plan), task_intent=copy.deepcopy(args.task_intent))
    options.update(task_plan=copy.deepcopy(plan), task_intent=copy.deepcopy(args.task_intent),
        editing_request=copy.deepcopy(plan['editing_request']), shaping_request=copy.deepcopy(auth['request']),
        preserve_project_selection=True, photo_appearance=copy.deepcopy(owned_u))
    options.pop('photo_appearance_selection', None)
    for key in ('provider', 'model', 'vision_model', 'api_base', 'api_url', 'api_key_env', 'model_registry'):
        value = getattr(args, key, None)
        if value is not None:
            options[key] = str(value) if isinstance(value, Path) else value
    # A fresh checked session owns U/current/source. Parent navigation retains
    # history without importing stale executable sessions or unrelated variants.
    for key in ('derived_edit_candidate', 'editing_iteration_result', 'editing_autofinish_result',
                'variants', 'selected_variant_id', 'appearance_history', 'shaping_geometry'):
        run.pop(key, None)
    run['reference'] = _reference_record(root, owned_u, u_receipt)
    workflow._begin_result_publication(run, operation='shaping_' + auth['request']['action'], checked=True)
    run['generation_budget'] = {'origin_run_id': run['run_id'],
        'max_image_submissions': image_submissions(auth['request']), 'image_submissions': 0,
        'total_image_submissions': 0, 'text_vision_submissions': 0, 'automatic_image_retry': False,
        'historical_generation_parent_run_id': auth['parent_run_id'], 'planning_requests_this_attempt': 0}
    log_path = path.parent / 'logs/run.log.jsonl'

    def cancelled():
        marker = getattr(args, 'cancel_request_path', None)
        return bool(marker and Path(marker).is_file())

    def saved_state(current, *, enabled):
        return {'schema': STATE_SCHEMA, 'base_appearance': copy.deepcopy(owned_u), 'enabled': enabled,
            'request': copy.deepcopy(auth['request']), 'authorization': copy.deepcopy(auth),
            'selected_asset_id': current['asset_id']}

    if auth['request']['action'] == 'restore':
        options['shaping_state'] = saved_state(owned_u, enabled=False)
        run['shaping_result'] = {'authorization_id': auth['authorization_id'], 'action': 'restore',
            'base_asset_id': owned_u['asset_id'], 'selected_asset_id': owned_u['asset_id'],
            'new_requests': {'image_submissions': 0, 'vision_requests': 0}, 'exact_frozen_reuse': True,
            'human_approved': False, 'independent_observation': False}
        workflow.write_json(path, run)
        try:
            workflow.check_generation_cancel(args)
            rendered = workflow.render_candidate(root, run, args)
            if sha256_file(rendered['outputs']['auto_draft']) != auth['base_image_sha256']:
                raise ValueError('Restored preview differs from its exact saved baseline')
            workflow.check_generation_cancel(args)
            run.update(auto_draft=rendered, render=rendered['metadata'], review=rendered['review'],
                artifacts=rendered['outputs'], delivery_status='auto_draft')
            workflow.append_event(run, 'needs_visual_review', detail='exact_shaping_baseline_restored', log_path=log_path)
        except BaseException as error:
            workflow.append_event(run, 'cancelled' if cancelled() else 'needs_attention',
                detail='restore_failed_parent_preserved', log_path=log_path)
            run['shaping_result']['failure'] = {'type': type(error).__name__, 'message': str(error)[:1600]}
            workflow.write_json(path, run)
            raise
        workflow.write_json(path, run)
        workflow.update_project_state(root, run)
        return run

    import photo_edit_session as session
    from photo_edit_tools import make_tools
    from photo_session_decision import decide_images
    from photo_session_review import evaluate_images
    from photo_edit_finish import drive_session, save_selected_candidate, session_summary
    from runtime_resources import code_path
    face_support = None
    if 'face' in auth['request']['parts']:
        from photo_shaping_geometry import prepare_geometry
        import workflow_timing
        workflow.check_generation_cancel(args)
        with workflow_timing.span('shaping_face_geometry', model_request=False, http_request=False,
                native_analysis_capability='face_localization', supports_verified_cache=True):
            face_support = prepare_geometry(root, owned_u, **binding,
                output=path.parent / 'shaping-geometry', registry_path=options.get('model_registry'),
                cancel=cancelled, preflight=plan['capabilities']['model_preflight'])
        run['shaping_geometry'] = copy.deepcopy(face_support)
        workflow.write_json(path, run)
    budget = {'image_submissions': image_submissions(auth['request']), 'vision_requests': 3, 'local_repairs': 1,
              'tool_calls': 1, 'improvement_steps': 1}
    original_intent = copy.deepcopy((run['generation_design'].get('art_direction') or {}).get('user_intent') or {})
    user_intent = {key: str(original_intent.get(key) or '') for key in ('brief', 'character_name', 'character_notes')}
    producer_names = ['photo_shaping.py', 'photo_edit_finish.py', 'photo_edit_session.py',
        'photo_session_decision.py', 'photo_session_review.py', 'appearance_assets.py']
    if execution_method(auth['request'])=='face_contour_deform':
        producer_names += ['photo_shaping_geometry.py','face_contour_deform.py','geometric_appearance.py','photo_shaping_tool.py']
    else:
        producer_names += ['photo_edit_tools.py']
    producer = {name: sha256_file(code_path(name)) for name in producer_names}
    directory = path.parent / 'editing-session'
    state = session.open_session(directory, source=run['source']['path'], scope=scope,
        design=run['generation_design']['observed_design'], execution_identity=plan['plan_id'],
        producer_identity=identity(producer), budget=budget, user_intent=user_intent,
        appearance_required=True, source_appearance=run['initial_source_appearance'],
        shaping_authorization=auth, shaping_base_appearance=owned_u)
    state = session.adopt_typed_parent(directory, owned_u, authorization=auth)
    connection = {key: getattr(args, key, None) or options.get(key) for key in
        ('provider', 'vision_model', 'api_base', 'api_url', 'api_key_env')}
    provider_options = {key: getattr(args, key, None) or options.get(key) for key in
        ('provider', 'model', 'api_base', 'api_url', 'api_key_env', 'size')}
    if execution_method(auth['request']) == 'face_contour_deform':
        from photo_shaping_tool import make_tool
        tools = [make_tool(path.parent / 'editing-tools', scope=scope, authorization=auth,
            base_appearance=owned_u, support_descriptor=face_support, session_directory=directory, cancel=cancelled)]
    else:
        tools = make_tools(root, path.parent / 'editing-tools', scope=scope, session_directory=directory,
            provider_options=provider_options, cancel=cancelled, shaping_authorization=auth,
            shaping_base_appearance=owned_u)
    if any(tool.name != shaping_tool(auth) for tool in tools):
        raise ValueError('Shaping may offer only its checked operation')

    def decide(context):
        return decide_images(context, path.parent / 'editing-decisions' / context['attempt_id'], **connection)

    def evaluate(context):
        return evaluate_images(context, path.parent / 'editing-reviews' / context['attempt_id'], **connection)

    def record_usage(current):
        run['generation_budget'].update(image_submissions=current['used']['image_submissions'],
            total_image_submissions=current['used']['image_submissions'], text_vision_submissions=current['used']['vision_requests'])
        workflow.write_json(path, run)

    workflow.write_json(path, run)
    driven = drive_session(directory, tools=tools, decide=decide, evaluate=evaluate,
        model_identity=identity({'connection': connection, 'producer': producer}), cancel=cancelled, after_step=record_usage)
    state = driven['state']; selected = driven['selected']; failure = driven['failure']
    completed = (selected is not None and selected.get('appearance_asset') is not None
        and selected['appearance_asset']['asset_id'] != owned_u['asset_id'] and selected.get('review') is not None
        and selected['image']['sha256'] != auth['base_image_sha256'] and state['automated_target_met'] is True)
    run['photo_edit_session'] = session_summary(directory, state, selected)
    run['shaping_result'] = {'authorization_id': auth['authorization_id'], 'action': 'apply',
        'base_asset_id': owned_u['asset_id'], 'selected_asset_id': (selected or {}).get('appearance_asset', {}).get('asset_id'),
        'completed': completed, 'new_requests': copy.deepcopy(state['used']), 'failure': failure, 'human_approved': False}
    if driven['cancelled'] or failure is not None or not completed:
        workflow.append_event(run, 'cancelled' if driven['cancelled'] else 'needs_attention',
            detail='shaping_incomplete_parent_preserved', log_path=log_path)
        workflow.write_json(path, run)
        return run
    # Own this new session's declared closure while the checked parent still
    # exists. Historical authorization locators stay immutable and resolve by
    # receipt identity after either parent or workspace relocation.
    state = session.own_evidence(directory, source_directory=directory, source_state=state)
    selected = session.selected_output(directory)['candidate']
    options['shaping_state'] = saved_state(selected['appearance_asset'], enabled=True)
    workflow.append_event(run, 'rendering', detail='save_reviewed_shaping_candidate', log_path=log_path)
    save_selected_candidate(root, run, args, selected=selected, session_directory=directory, cancel=cancelled)
    run['artifacts']['photo_edit_session'] = str(directory / 'session.json')
    run['photo_edit_session'] = session_summary(directory, state, selected)
    validate_state(run['options']['shaping_state'], current=run['options']['photo_appearance'], **binding)
    workflow.append_event(run, 'needs_visual_review', detail='shaping_candidate_requires_explicit_adoption', log_path=log_path)
    workflow.write_json(path, run)
    workflow.update_project_state(root, run)
    return run
