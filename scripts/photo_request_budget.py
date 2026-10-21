"""Origin-bound image/vision reservations shared by saved candidate revisions.

The ledger preserves checked limits and actual historical evidence. It never
submits a model request, changes a task plan, refunds an unknown attempt or
turns an inherited design into another current model call.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import uuid
from contextlib import contextmanager
from urllib.parse import urlsplit

from asset_identity import sha256_file
from cache_store import exclusive_file_lock
from current_model_registry import atomic_json
from photo_editing import identity

SCHEMA = 'cosluma_photo_request_budget.v1'


class ReviewPending(RuntimeError):
    """The existing endpoint must be inspected; another call is forbidden."""


class BudgetExhausted(RuntimeError):
    pass


def text_sha(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _component(value):
    if (not isinstance(value, str) or not value or value in {'.', '..'}
            or '/' in value or '\\' in value or ':' in value):
        raise ValueError('A saved origin run identity is required')
    return value


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _asset(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': sha256_file(path)}


def endpoint(*, source_sha256, candidate_sha256, scope_id, design_id, prompt,
             connection_identity, decoder_identity, previous_best_sha256=None,source_preview_sha256=None):
    """Stable across revisions; includes actual objective and selected connection."""
    result = {'source_sha256': source_sha256, 'candidate_sha256': candidate_sha256,
        'scope_id': scope_id, 'design_id': design_id, 'prompt_sha256': text_sha(prompt),
        'connection_identity': connection_identity, 'decoder_identity': decoder_identity}
    if previous_best_sha256 is not None:
        result['previous_best_sha256'] = previous_best_sha256
    if source_preview_sha256 is not None:result['source_preview_sha256']=source_preview_sha256
    return result


def _origin(root, run):
    if (run.get('task_plan') or {}).get('editing_iteration') is not None:
        raise ValueError('Editing iteration owns its separately checked decision/review budget')
    origin_id = _component((run.get('generation_budget') or {}).get('origin_run_id') or run.get('run_id'))
    if origin_id == run.get('run_id'):
        origin = run
    else:
        origin = _read_json(root/'outputs/runs'/origin_id/'manifest.json')
    plan = origin.get('task_plan') or {}
    generation = origin.get('generation_design') or {}
    art = generation.get('art_direction') or {}
    source_sha = (origin.get('source') or {}).get('sha256')
    scope = plan.get('editing_scope') or {}
    limit = (plan.get('services') or {}).get('text_vision_requests_upper_bound')
    provenance = art.get('provenance') or {}
    if (origin.get('run_id') != origin_id or plan.get('mode') != 'create_reference'
            or plan.get('editing_iteration') is not None or not plan.get('plan_id')
            or type(limit) is not int or limit < 1 or not scope.get('scope_id')
            or generation.get('source_sha256') != source_sha or art.get('status') != 'ready'
            or (provenance.get('source') or {}).get('sha256') != source_sha
            or provenance.get('network_used') is not True or provenance.get('generation_used') is not False
            or not provenance.get('model') or art.get('model_requests_attempted') != 1
            or generation.get('text_vision_submissions') != 1):
        raise ValueError('Origin budget requires its checked limit and actual design-observation evidence')
    if ((run.get('task_plan') or {}).get('plan_id') != plan['plan_id']
            or (run.get('source') or {}).get('sha256') != source_sha
            or (run.get('generation_design') or {}).get('observed_design', {}).get('design_id')
               != generation.get('observed_design', {}).get('design_id')):
        raise ValueError('Saved revision does not share this origin plan/source/design')
    binding = {'origin_run_id': origin_id, 'origin_plan_id': plan['plan_id'],
        'source_sha256': source_sha, 'scope_id': scope['scope_id'],
        'design_id': generation['observed_design']['design_id'], 'checked_vision_limit': limit,
        'design_evidence_id': identity({'binding': art.get('binding'), 'provenance': provenance}),
        'checked_image_limit': sum(plan.get('image_budget', {}).values()),
        'checked_initial_review_limit': (plan.get('image_budget') or {}).get('initial_reference', 0)
            if (origin.get('task_intent') or {}).get('services', {}).get('visual_review') is True else 0,
        'checked_improvement_limit': (plan.get('editing_autofinish') or {}).get('maximum_improvements', 0),
        'checked_zero_image_finish_limit': (plan.get('editing_autofinish') or {}).get('maximum_zero_image_finishes',0)}
    from photo_edit_session import INITIALIZATION
    if plan.get('editing_initialization')==INITIALIZATION:
        from photo_appearance import validate_asset
        source_asset=origin.get('initial_source_appearance')
        source_receipt=validate_asset(source_asset,source_sha256=source_sha,scope=scope,
            design_id=generation['observed_design']['design_id'])
        if source_receipt['kind']!='source_baseline':raise ValueError('Origin needs its actual source float baseline')
        binding.update(editing_initialization=copy.deepcopy(INITIALIZATION),
            checked_initial_operation_limit=plan['image_budget']['initial_reference'],
            visual_review=bool(origin['task_intent']['services']['visual_review']),
            checked_initial_review_limit=int(origin['task_intent']['services']['visual_review']),
            source_asset_id=source_asset['asset_id'],source_preview_sha256=source_receipt['files']['preview']['sha256'],
            source_master_sha256=source_receipt['files']['master']['sha256'])
    return binding, origin


def _historical_reviews(root, binding, current):
    """Import unique persisted attempts, including unresolved pre-upgrade ones.

    This one-time migration reads metadata only. It does not require historical
    pixels to still exist in order to preserve a known spent reservation.
    """
    manifests = {path.parent.name: path for path in (root/'outputs/runs').glob('*/manifest.json')}
    identifiers = set(manifests) | {current['run_id']}
    found = {}; recorded_highwater = 1
    for run_id in sorted(identifiers):
        value = current if run_id == current['run_id'] else _read_json(manifests[run_id])
        if ((value.get('task_plan') or {}).get('plan_id') != binding['origin_plan_id']
                or (value.get('source') or {}).get('sha256') != binding['source_sha256']):
            continue
        prior = (value.get('photo_edit_session') or {}).get('origin_request_budget') or {}
        if prior.get('origin_plan_id') == binding['origin_plan_id'] and prior.get('improvement_rounds_reserved', 0):
            raise ValueError('Historical improvement reservations require their original resource ledger; do not reset rounds')
        if (prior.get('origin_plan_id') == binding['origin_plan_id']
                and type(prior.get('reserved_vision_requests')) is int):
            recorded_highwater = max(recorded_highwater, prior['reserved_vision_requests'])
        session_path = root/'outputs/runs'/run_id/'editing-session/session.json'
        if not session_path.is_file():
            if (value.get('photo_edit_session') or {}).get('independent_evaluation'):
                raise ValueError('Historical review evidence has no persistent attempt ledger; inspect it before another call')
            continue
        state = _read_json(session_path)
        if (state.get('session_id') != identity(state.get('binding'))
                or (state.get('binding') or {}).get('source', {}).get('sha256') != binding['source_sha256']):
            raise ValueError('Historical review session identity differs')
        for attempt in state.get('attempts', []):
            if attempt.get('kind') != 'evaluate':
                continue  # Reused parent observations are not new requests.
            count = (attempt.get('costs') or {}).get('vision_requests')
            if type(count) is not int or count != 1:
                raise ValueError('Historical review reservation count is not trustworthy')
            context = attempt.get('context') or {}
            images = context.get('images') or []
            if (attempt.get('context_sha256') != identity(context) or len(images) != 2
                    or images[0].get('role') != 'source' or images[0].get('sha256') != binding['source_sha256']
                    or images[1].get('role') != 'candidate'):
                raise ValueError('Historical origin review is not bound to these input images')
            receipt_path = session_path.parent/'reviews'/attempt['attempt_id']/'review.json'
            receipt = _read_json(receipt_path) if receipt_path.is_file() else {}
            response = receipt.get('result') or attempt.get('response')
            provenance = (response or {}).get('provenance') or receipt.get('provenance') or {}
            request_id = ('response:'+str(provenance['response_id']) if provenance.get('response_id') else
                          'attempt:'+str(attempt['attempt_id']))
            key = None
            if receipt.get('prompt') and (receipt.get('binding') or {}).get('connection_identity'):
                key = endpoint(source_sha256=binding['source_sha256'], candidate_sha256=images[1]['sha256'],
                    scope_id=binding['scope_id'], design_id=binding['design_id'], prompt=receipt['prompt'],
                    connection_identity=receipt['binding']['connection_identity'],
                    decoder_identity=(receipt['binding'].get('producer') or {}).get('raw_support.py'))
            ready = (receipt.get('status') == 'ready' and response == receipt.get('result')
                     and provenance.get('network_used') is True and provenance.get('generation_used') is False)
            record = {'reservation_id': request_id, 'endpoint': key,
                'candidate_sha256': images[1]['sha256'], 'status': 'ready' if ready else 'unknown',
                'reserved_requests': 1, 'owner_run_id': run_id, 'historical_import': True,
                'result': copy.deepcopy(response) if ready else None,
                'evidence': _asset(receipt_path) if receipt_path.is_file() else _asset(session_path),
                'actual_model_requests': 1 if ready else receipt.get('model_requests_attempted')}
            if request_id in found and found[request_id] != record:
                # Identical provider responses can be referenced by several
                # revisions. Preserve the first actual request, not references.
                if found[request_id].get('result') != record.get('result'):
                    raise ValueError('Conflicting historical review request evidence')
            else:
                found[request_id] = record
    if recorded_highwater > 1+len(found):
        raise ValueError('Historical origin reservations are missing; restore their ledger instead of granting replacement quota')
    return list(found.values())


def _read(path):
    state = _read_json(path)
    if (state.get('schema') != SCHEMA or state.get('binding_id') != identity(state.get('binding'))
            or state.get('design_reserved_requests') != 1):
        raise ValueError('Origin request ledger binding changed')
    rows = state.get('reviews')
    if (not isinstance(rows, list) or any(row.get('reserved_requests') != 1 for row in rows)
            or len({row['reservation_id'] for row in rows}) != len(rows)):
        raise ValueError('Origin review reservations changed')
    if 'operations' in state:
        if (not isinstance(state['operations'], list) or type(state.get('initial_images_reserved')) is not int
                or state['initial_images_reserved'] < 0):
            raise ValueError('Origin image reservations changed')
        improvement_ordinal=0
        for row in state['operations']:
            tool_images = (_image_cost(row['tool_costs']) if row.get('tool_costs') is not None else 1)
            review_vision = 0 if row.get('reused_evidence') is True else 1
            native='editing_initialization' in state['binding']
            initial=native and row.get('context',{}).get('operation_stage')=='initial'
            expected_cost={'vision_requests':review_vision if row['kind']=='post_review' else int(row['kind']=='decision'),
                'image_submissions':tool_images if row['kind']=='tool' else 0,
                'improvement_rounds':int(row['kind']=='decision' and not initial)}
            if native:expected_cost['initial_operations']=int(row['kind']=='decision' and initial)
            if row['kind']=='decision' and not initial:
                zero_only=improvement_ordinal>=state['binding']['checked_improvement_limit']
                if (row['context'].get('zero_image_only',False) is not zero_only
                        or improvement_ordinal>=state['binding']['checked_improvement_limit']+state['binding']['checked_zero_image_finish_limit']):
                    raise ValueError('Origin decision no longer matches its checked round admission')
                if zero_only:
                    offered=[tool for tool in (row['context'].get('tools') or {}).values() if tool.get('inputs_available') is True]
                    if not offered or any(_image_cost(tool.get('costs'))!=0 for tool in offered):
                        raise ValueError('A saved final decision offered an image-generating tool')
                improvement_ordinal+=1
            if (row.get('kind') not in {'decision', 'tool', 'post_review'} or
                    row.get('context_sha256') != identity(row.get('context')) or
                    row.get('operation_id') != identity({'kind': row['kind'], 'context': row['context']}) or
                    row.get('costs') != expected_cost):
                raise ValueError('Origin operation cost differs from its phase')
            if row.get('reused_evidence') is True and (row['kind'] != 'post_review' or row.get('actual_model_requests') != 0):
                raise ValueError('Only actual cached review evidence can avoid a new model reservation')
        if len({row['reservation_id'] for row in state['operations']}) != len(state['operations']):
            raise ValueError('Duplicate origin operation reservation')
        for row in state['operations']:
            if row['kind'] == 'tool' and row.get('tool_costs') is not None:
                lease = next((item for item in state['operations'] if item['reservation_id'] == row['lease']), None)
                descriptor = ((lease or {}).get('context', {}).get('tools') or {}).get(
                    row['context'].get('decision', {}).get('tool')) or {}
                if (not lease or lease['kind'] != 'decision' or lease['status'] != 'ready'
                        or descriptor.get('inputs_available') is False
                        or (lease['context'].get('zero_image_only') and _image_cost(row['tool_costs'])!=0)
                        or _decision_payload(lease['result']) != row['context'].get('decision')
                        or _image_cost(descriptor.get('costs')) != _image_cost(row['tool_costs'])
                        or not _tool_images_match(lease['context'], row['context'], descriptor)):
                    raise ValueError('Origin tool cost no longer matches its actual decision capability')
    return state


def _report(path, state):
    operations = state.get('operations', [])
    reserved = (state['design_reserved_requests']+sum(row['reserved_requests'] for row in state['reviews'])
                +sum(row['costs']['vision_requests'] for row in operations))
    images = state.get('initial_images_reserved', 0)+sum(row['costs']['image_submissions'] for row in operations)
    rounds = sum(row['costs']['improvement_rounds'] for row in operations)
    return {'path': str(path), 'origin_run_id': state['binding']['origin_run_id'],
        'origin_plan_id': state['binding']['origin_plan_id'],
        'checked_vision_limit': state['binding']['checked_vision_limit'],
        'reserved_vision_requests': reserved, 'historical_design_requests': 1,
        'review_reservations': len(state['reviews']), 'reserved_image_requests': images,
        'historical_initial_image_requests': state.get('initial_images_reserved', 0),
        'checked_image_limit': state['binding'].get('checked_image_limit'),
        'checked_improvement_limit': state['binding'].get('checked_improvement_limit', 0),
        'checked_zero_image_finish_limit':state['binding']['checked_zero_image_finish_limit'],
        'zero_image_finish_rounds_reserved':sum(row['kind']=='decision' and row['context'].get('zero_image_only',False) for row in operations),
        'improvement_rounds_reserved': rounds,
        'initial_operations_reserved':sum(row['costs'].get('initial_operations',0) for row in operations),
        'checked_initial_operation_limit':state['binding'].get('checked_initial_operation_limit',0),
        'initial_operation_image_submissions':sum(row['costs']['image_submissions'] for row in operations
            if row['context'].get('operation_stage')=='initial'),
        'improvement_image_submissions':sum(row['costs']['image_submissions'] for row in operations
            if row['context'].get('operation_stage')=='improvement' or 'operation_stage' not in row['context']),
        'unknown_image_reservations':sum(row['costs']['image_submissions'] for row in operations if row['status']!='ready'),
        'historical_limit_exceeded': (reserved > state['binding']['checked_vision_limit']
            or images > state['binding'].get('checked_image_limit', images)
            or rounds > state['binding']['checked_improvement_limit']+state['binding']['checked_zero_image_finish_limit']),
        'unknown_operation_reservations': sum(row['status'] != 'ready' for row in operations),
        'confirmed_vision_requests': (1+sum(row.get('actual_model_requests') == 1 for row in state['reviews'])
            +sum(row['costs']['vision_requests'] for row in operations if row['status'] == 'ready')),
        'confirmed_image_submissions': state.get('initial_images_reserved', 0)
            +sum(row['costs']['image_submissions'] for row in operations if row['status'] == 'ready'),
        'unknown_review_reservations': sum(row['status'] not in {'ready'} for row in state['reviews'])}


def prepare(root, run):
    """Execution-only initialization; never called by task-plan."""
    root = Path(root).resolve(); binding, origin_run = _origin(root, run)
    generation_budget = origin_run.get('generation_budget') or {}
    submissions = generation_budget.get('submissions')
    initial_images = (sum(row.get('status') != 'rejected_before_submission' for row in submissions)
                      if isinstance(submissions, list) else generation_budget.get('image_submissions', 0))
    image_history_limit=(binding['checked_image_limit'] if 'editing_initialization' in binding
                         else origin_run['task_plan']['image_budget']['initial_reference'])
    if type(initial_images) is not int or not 0 <= initial_images <= image_history_limit:
        raise ValueError('Initial image reservation history differs from the checked origin')
    path = root/'outputs/request-budgets'/binding['origin_run_id']/'vision.json'
    with exclusive_file_lock(path.with_suffix('.lock'), timeout=30):
        if 'editing_initialization' in binding:
            if not path.exists() and initial_images:
                raise ValueError('Source-native initialization cannot hide pre-session image submissions')
            initial_images=0  # All native images are charged through actual tool leases.
        if ('editing_initialization' in binding and not path.exists()
                and any(row.get('photo_edit_session') for row in origin_run.get('variants',[]))):
            raise ReviewPending('Source-native operation history requires its original ledger; do not replenish it')
        if path.exists():
            state = _read(path)
            if 'operations' not in state:
                previous = copy.deepcopy(state['binding'])
                if any(binding.get(key) != value for key, value in previous.items()):
                    raise ValueError('Historical origin limits changed; do not upgrade its allowance')
                state.update(binding=binding, binding_id=identity(binding), operations=[],
                    initial_images_reserved=initial_images,
                    resource_budget_upgrade={'previous_ledger_sha256': sha256_file(path), 'previous_binding': previous})
                atomic_json(path, state)
            if state['binding'] != binding:
                raise ValueError('The saved origin limit or observed design changed; do not rewrite its budget')
        else:
            state = {'schema': SCHEMA, 'binding': binding, 'binding_id': identity(binding),
                'design_reserved_requests': 1, 'reviews': _historical_reviews(root, binding, run),
                'operations': [], 'initial_images_reserved': initial_images,
                'automatic_refund': False, 'automatic_resubmission': False}
            atomic_json(path, state)
    return _report(path, state)


def snapshot(locator):
    path = Path(locator['path']).resolve()
    with exclusive_file_lock(path.with_suffix('.lock'), timeout=30):
        return _report(path, _read(path))


def _ready(row, key):
    result = row.get('result')
    expected = [{'role': 'source', 'sha256': key.get('source_preview_sha256',key['source_sha256'])},
                {'role': 'candidate', 'sha256': key['candidate_sha256']}]
    if key.get('previous_best_sha256'):
        expected.append({'role': 'previous_best', 'sha256': key['previous_best_sha256']})
    provenance = (result or {}).get('provenance') or {}
    board = provenance.get('review_board') or {}
    if (not isinstance(result, dict) or result.get('observed_images') != expected
            or provenance.get('source') != board or not provenance.get('model')
            or provenance.get('network_used') is not True or provenance.get('generation_used') is not False
            or _asset(board['path']) != board):
        raise ValueError('Existing review no longer has matching actual-image evidence; no new request was granted')
    evidence = row.get('evidence') or {}
    if _asset(evidence['path']) != evidence:
        raise ValueError('Existing review receipt bytes changed; its reservation is retained')
    receipt = _read_json(evidence['path'])
    if (receipt.get('schema') != 'cosluma_photo_session_review.v1' or receipt.get('status') != 'ready'
            or receipt.get('result') != result or text_sha(receipt.get('prompt', '')) != key['prompt_sha256']
            or (receipt.get('binding') or {}).get('connection_identity') != key['connection_identity']
            or ((receipt.get('binding') or {}).get('producer') or {}).get('raw_support.py') != key['decoder_identity']):
        raise ValueError('Cached review differs from its actual transport receipt')
    return copy.deepcopy(result)


def reserve(locator, key, *, run_id):
    """Reserve once before transport, or return identical completed evidence."""
    path = Path(locator['path']).resolve(); run_id = _component(run_id)
    with exclusive_file_lock(path.with_suffix('.lock'), timeout=30):
        state = _read(path)
        if 'editing_initialization' in state['binding'] and (
                key.get('candidate_sha256')!=state['binding']['source_preview_sha256']
                or key.get('source_preview_sha256')!=state['binding']['source_preview_sha256']
                or key.get('previous_best_sha256')):
            raise ValueError('Only the shared typed source baseline uses the initial review slot')
        if (key.get('source_sha256') != state['binding']['source_sha256']
                or key.get('scope_id') != state['binding']['scope_id']
                or key.get('design_id') != state['binding']['design_id']):
            raise ValueError('Review endpoint differs from its checked origin')
        for row in state['reviews']:
            if row.get('endpoint') == key:
                if row['status'] == 'ready':
                    return {'path': str(path), 'reservation_id': row['reservation_id'], 'owner': False,
                            'result': _ready(row, key), 'origin_budget': _report(path, state)}
                raise ReviewPending('An identical review is pending or unknown; inspect its existing receipt, never resubmit')
            if row['status'] != 'ready' and row.get('candidate_sha256') == key['candidate_sha256']:
                raise ReviewPending('A review of these pixels is unresolved; changing revision or model does not retry it')
        if any(row['kind'] == 'post_review' and row['status'] != 'ready'
               and any(image['role'] == 'candidate' and image['sha256'] == key['candidate_sha256']
                       for image in row['context']['images']) for row in state.get('operations', [])):
            raise ReviewPending('A correction review of these pixels remains unresolved')
        report = _report(path, state)
        if (report['reserved_vision_requests'] >= report['checked_vision_limit']
                or len(state['reviews']) >= state['binding']['checked_initial_review_limit']):
            raise BudgetExhausted('The original checked vision budget is exhausted; no review request was submitted')
        row = {'reservation_id': uuid.uuid4().hex, 'endpoint': copy.deepcopy(key),
            'candidate_sha256': key['candidate_sha256'], 'status': 'reserved',
            'reserved_requests': 1, 'owner_run_id': run_id, 'historical_import': False,
            'actual_model_requests': None, 'result': None, 'evidence': None}
        state['reviews'].append(row); atomic_json(path, state)
        return {'path': str(path), 'reservation_id': row['reservation_id'], 'owner': True,
                'origin_budget': _report(path, state)}


def finish(ticket, *, result=None, evidence=None, actual_model_requests=None):
    """Persist outcome without refunding its reservation or making another call."""
    if ticket.get('owner') is not True:
        raise ValueError('Only the reserved review owner can record its outcome')
    path = Path(ticket['path']).resolve()
    with exclusive_file_lock(path.with_suffix('.lock'), timeout=30):
        state = _read(path)
        row = next(item for item in state['reviews'] if item['reservation_id'] == ticket['reservation_id'])
        if row['status'] == 'ready':
            if result != row['result']:
                raise ValueError('Completed review facts are immutable')
            return _report(path, state)
        row.update(status='ready' if result is not None else 'unknown', result=copy.deepcopy(result),
            evidence=copy.deepcopy(evidence), actual_model_requests=actual_model_requests)
        if result is not None:
            _ready(row, row['endpoint'])
        atomic_json(path, state)
        return _report(path, state)


def _decision_payload(result):
    return {key: value for key, value in result.items() if key not in {'provenance', 'observed_images'}}


def _image_cost(costs):
    if (not isinstance(costs, dict) or set(costs)-{'image_submissions', 'vision_requests', 'local_repairs', 'tool_calls', 'improvement_steps'}
            or any(type(value) is not int or value < 0 for value in costs.values())
            or costs.get('image_submissions', 0) not in (0, 1) or costs.get('vision_requests', 0) != 0
            or costs.get('tool_calls', 0) or costs.get('improvement_steps', 0)):
        raise ValueError('Origin tool needs its actual checked zero/one image cost and no hidden vision call')
    return costs.get('image_submissions', 0)


def _tool_images_match(decision_context, tool_context, descriptor):
    expected = [image for image in decision_context['images'] if image['role'] != 'previous_attempt']
    if not descriptor.get('requires_working'):
        expected = [image for image in expected if image['role'] != 'working']
    return expected == tool_context.get('images')


def _check_context(state, context):
    binding = state['binding']; images = context.get('images') or []
    if (not images or images[0].get('role') != 'source'
            or (context.get('source_original') or images[0]).get('sha256') != binding['source_sha256']
            or context.get('scope', {}).get('scope_id') != binding['scope_id']
            or context.get('design', {}).get('design_id') != binding['design_id']):
        raise ValueError('Operation objective or source differs from the checked origin')
    for image in images:
        if _asset(image['path']) != {key: image[key] for key in ('path', 'sha256')}:
            raise ValueError('Operation input pixels changed')
    if 'editing_initialization' in binding:
        from photo_appearance import validate_asset
        if (context.get('variant_id') not in {'v'+str(i) for i in range(1,binding['checked_initial_operation_limit']+1)}
                or context.get('operation_stage') not in {'baseline','initial','improvement'}
                or context.get('visual_review') != binding['visual_review']):
            raise ValueError('Source operation stage, variant or review permission differs')
        for image in images:
            receipt=validate_asset(context['appearance_assets'][image['role']],
                source_sha256=binding['source_sha256'],scope=context['scope'],design_id=binding['design_id'])
            if receipt['files']['preview']['sha256']!=image['sha256']:
                raise ValueError('Actual source-native context preview and float master differ')
            if image['role']=='source' and (receipt['asset_id']!=binding['source_asset_id']
                    or receipt['files']['master']['sha256']!=binding['source_master_sha256']):
                raise ValueError('Operation source is not this origin float baseline')


def reserve_operation(locator, kind, context, *, review_endpoint=None, tool_costs=None, reuse_review=None):
    """Atomic origin reservation; a decision leases exactly one improvement round.

    Tools and their independent reviews must follow that actual decision in the
    same persistent session. Unknown outcomes consume the lease permanently.
    """
    if kind not in {'decision', 'tool', 'post_review'}:
        raise ValueError('Unsupported origin operation')
    path = Path(locator['path']).resolve()
    operation_id = identity({'kind': kind, 'context': context})
    with exclusive_file_lock(path.with_suffix('.lock'), timeout=30):
        state = _read(path); _check_context(state, context)
        for row in state['operations']:
            if row['operation_id'] == operation_id:
                if row['status'] != 'ready':
                    raise ReviewPending('This operation is unresolved; no implicit resubmission')
                _validate_operation_result(row, row['result'])
                return {'path': str(path), 'reservation_id': row['reservation_id'],
                        'owner': False, 'result': copy.deepcopy(row['result'])}
        report = _report(path, state); binding = state['binding']
        lease = None
        declared_costs = None
        native='editing_initialization' in binding
        initial=native and context['operation_stage']=='initial'
        if kind == 'decision':
            zero_only=not initial and report['improvement_rounds_reserved']>=binding['checked_improvement_limit']
            if context.get('zero_image_only',False) is not zero_only:
                raise BudgetExhausted('The checked decision round changed; do not switch its offered tools implicitly')
            if zero_only:
                offered=[tool for tool in (context.get('tools') or {}).values() if tool.get('inputs_available') is True]
                if (not offered or any(_image_cost(tool.get('costs'))!=0 for tool in offered)):
                    raise ValueError('The last checked round may offer only actual zero-image tools')
            if initial:
                if (report['initial_operations_reserved']>=binding['checked_initial_operation_limit']
                        or any(row['kind']=='decision' and row['context'].get('operation_stage')=='initial'
                               and row['context'].get('variant_id')==context['variant_id'] for row in state['operations'])):
                    raise BudgetExhausted('This checked variant already reserved its one first operation')
            if ((not initial and report['improvement_rounds_reserved'] >= binding['checked_improvement_limit']+binding['checked_zero_image_finish_limit'])
                    or report['reserved_vision_requests']+1+int(binding.get('visual_review',True)) > binding['checked_vision_limit']):
                raise BudgetExhausted('No checked improvement round with tool and review allowance remains')
            if any(row['kind'] == 'decision' and row['session_id'] == context['session_id']
                   and row['status'] != 'ready' for row in state['operations']):
                raise ReviewPending('An earlier decision in this session is unresolved')
            observed = [(row['role'], row['sha256']) for row in context['images']]
            if any(row['kind'] in {'decision', 'tool'} and row['status'] != 'ready'
                   and [(image['role'], image['sha256']) for image in row['context']['images']] == observed
                   for row in state['operations']):
                raise ReviewPending('A decision on these actual images is unresolved across revisions')
        else:
            decisions = [row for row in state['operations'] if row['kind'] == 'decision'
                         and row['session_id'] == context['session_id'] and row['status'] == 'ready'
                         and row['result'].get('action') == 'tool']
            for decision in reversed(decisions):
                tools = [row for row in state['operations'] if row['kind'] == 'tool'
                         and row['lease'] == decision['reservation_id']]
                if kind == 'tool' and not tools and _decision_payload(decision['result']) == context.get('decision'):
                    descriptor = (decision['context'].get('tools') or {}).get(context['decision']['tool']) or {}
                    if descriptor.get('inputs_available') is False:
                        raise ValueError('The actual decision did not offer this tool')
                    if not _tool_images_match(decision['context'], context, descriptor):
                        continue
                    declared_costs = descriptor.get('costs')
                    expected_images = _image_cost(declared_costs)
                    if decision['context'].get('zero_image_only') and expected_images!=0:
                        raise ValueError('A zero-image finish cannot authorize image generation')
                    if tool_costs is not None and _image_cost(tool_costs) != expected_images:
                        raise ValueError('Executed tool image cost differs from its actual offered capability')
                    lease = decision['reservation_id']; break
                if kind == 'post_review' and len(tools) == 1 and tools[0]['status'] == 'ready':
                    candidate = next((image for image in context['images'] if image['role'] == 'candidate'), {})
                    previous = next((image for image in context['images'] if image['role'] == 'previous_best'), None)
                    tool_best = next((image for image in tools[0]['context']['images'] if image['role'] == 'best'), None)
                    if (tools[0]['output_image']['sha256'] == candidate.get('sha256')
                            and (previous is None or tool_best and previous['sha256'] == tool_best['sha256'])
                            and not any(row['kind'] == 'post_review' and row['lease'] == decision['reservation_id']
                                        for row in state['operations'])):
                        lease = decision['reservation_id']; break
            if lease is None:
                raise BudgetExhausted('No unused actual decision/tool lease authorizes this operation')
        costs = {'vision_requests': int(kind != 'tool' and reuse_review is None),
                 'image_submissions': _image_cost(declared_costs) if kind == 'tool' else 0,
                 'improvement_rounds': int(kind == 'decision' and not initial)}
        if native:costs['initial_operations']=int(kind=='decision' and initial)
        if (report['reserved_vision_requests']+costs['vision_requests'] > binding['checked_vision_limit']
                or report['reserved_image_requests']+costs['image_submissions'] > binding['checked_image_limit']):
            raise BudgetExhausted('The original checked resource limit is exhausted')
        row = {'reservation_id': uuid.uuid4().hex, 'operation_id': operation_id, 'kind': kind,
               'session_id': context['session_id'], 'context': copy.deepcopy(context),
               'context_sha256': identity(context), 'lease': lease, 'costs': costs,
               'status': 'reserved', 'result': None}
        if kind == 'tool':
            row['tool_costs'] = copy.deepcopy(declared_costs)
        if kind == 'post_review':
            row['review_endpoint'] = copy.deepcopy(review_endpoint)
            if reuse_review is not None:
                board = Path(reuse_review['provenance']['review_board']['path'])
                row.update(reused_evidence=True, actual_model_requests=0, actual_image_submissions=0,
                    status='ready', result=copy.deepcopy(reuse_review), evidence=_asset(board.parent.parent/'review.json'))
                _validate_operation_result(row, reuse_review)
        elif reuse_review is not None:
            raise ValueError('Only an independent post-review can reuse existing evidence')
        state['operations'].append(row); atomic_json(path, state)
        return {'path': str(path), 'reservation_id': row['reservation_id'], 'owner': reuse_review is None,
                **({'result': copy.deepcopy(reuse_review)} if reuse_review is not None else {})}


def _validate_operation_result(row, result):
    if not isinstance(result, dict):
        raise ValueError('Origin operation needs actual callback result')
    provenance = result.get('provenance') or {}
    if row['kind'] == 'tool':
        image = _asset(result['image'])
        evidence = provenance.get('tool_attempt') or {}
        if (_asset(evidence['path']) != evidence or provenance.get('full_candidate', {}).get('sha256') != image['sha256']
                or result.get('resources', {}).get('image_submissions') != row['costs']['image_submissions']):
            raise ValueError('Tool result has no matching actual image request evidence')
        if row.get('output_image') and row['output_image'] != image:
            raise ValueError('Completed tool output changed')
        return image
    expected = [{'role': image['role'], 'sha256': image['sha256']} for image in row['context']['images']]
    if (result.get('observed_images') != expected or not provenance.get('model')
            or provenance.get('network_used') is not True or provenance.get('generation_used') is not False):
        raise ValueError('Vision operation has no matching actual-image observation')
    if row['kind'] == 'decision':
        if result.get('action')=='tool':
            descriptor=(row['context'].get('tools') or {}).get(result.get('tool'))
            if (not descriptor or descriptor.get('inputs_available') is False
                    or (row['context'].get('zero_image_only') and
                        (descriptor.get('inputs_available') is not True or _image_cost(descriptor.get('costs'))!=0))):
                raise ValueError('Decision selected a tool outside its actual checked admission')
        mapping = provenance.get('decision_image_mapping') or {}
        if (mapping.get('images') != row['context']['images']
                or _asset(mapping['transport_image']['path']) != mapping['transport_image']
                or provenance.get('source') != mapping['transport_image']):
            raise ValueError('Decision transport differs from the frozen actual images')
    else:
        board = provenance.get('review_board') or {}
        if _asset(board['path']) != board or provenance.get('source') != board:
            raise ValueError('Review transport differs from the frozen actual board')
        if row.get('review_endpoint'):
            _ready(row, row['review_endpoint'])


def finish_operation(ticket, *, result=None):
    if ticket.get('owner') is not True:
        raise ValueError('Only the reserved operation owner can finish it')
    path = Path(ticket['path']).resolve()
    with exclusive_file_lock(path.with_suffix('.lock'), timeout=30):
        state = _read(path)
        row = next(row for row in state['operations'] if row['reservation_id'] == ticket['reservation_id'])
        if row['status'] == 'ready':
            if result != row['result']:
                raise ValueError('Completed origin operation facts are immutable')
            return _report(path, state)
        if result is not None:
            if row['kind'] == 'post_review':
                board = Path(result['provenance']['review_board']['path'])
                row['evidence'] = _asset(board.parent.parent/'review.json')
                if ticket.get('review_endpoint'):
                    row['review_endpoint'] = copy.deepcopy(ticket['review_endpoint'])
                row['result'] = copy.deepcopy(result)
            output = _validate_operation_result(row, result)
            if row['kind'] == 'tool':
                row['output_image'] = output
        row.update(status='ready' if result is not None else 'unknown', result=copy.deepcopy(result),
            actual_model_requests=(row['costs']['vision_requests'] if result is not None else None),
            actual_image_submissions=(row['costs']['image_submissions'] if result is not None else None))
        atomic_json(path, state)
        return _report(path, state)


class OriginHooks:
    """Callback decorators only; the existing Session remains the sole driver."""

    def __init__(self, root, run, connection):
        self.locator = prepare(root, run)
        self.run_id = run['run_id']
        self.connection = {key: connection.get(key) for key in
                           ('provider', 'vision_model', 'api_base', 'api_url', 'api_key_env')}
        self.user_intent = copy.deepcopy((run.get('generation_design') or {}).get('art_direction', {}).get('user_intent') or {})

    def report(self):
        return snapshot(self.locator)

    def remaining(self):
        report = self.report()
        path = Path(self.locator['path'])
        with exclusive_file_lock(path.with_suffix('.lock'), timeout=30):
            state = _read(path)
        return {'image_submissions': max(0, report['checked_image_limit']-report['reserved_image_requests']),
                'vision_requests': max(0, report['checked_vision_limit']-report['reserved_vision_requests']),
                'improvement_rounds': max(0, report['checked_improvement_limit']+report['checked_zero_image_finish_limit']-report['improvement_rounds_reserved']),
                'zero_image_only':report['improvement_rounds_reserved']>=report['checked_improvement_limit'],
                'initial_review_slots': max(0, state['binding']['checked_initial_review_limit']-len(state['reviews'])),
                'initial_operations':max(0,report['checked_initial_operation_limit']-report['initial_operations_reserved'])}

    def session_budget(self):
        report=self.report();count = report['checked_improvement_limit'];finishes=report['checked_zero_image_finish_limit']
        with exclusive_file_lock(Path(self.locator['path']).with_suffix('.lock'),timeout=30):
            binding=_read(Path(self.locator['path']))['binding']
        if 'editing_initialization' in binding:
            review=binding['visual_review']
            return {'image_submissions':1+count,'vision_requests':1+2*int(review)+2*(count+finishes),
                'tool_calls':1+count+finishes,'local_repairs':1+count+finishes,'improvement_steps':count+finishes}
        return {'image_submissions': count, 'vision_requests': 1+2*count, 'tool_calls': count,
                'local_repairs': count, 'improvement_steps': count}

    def initial_usage(self):
        return {key: 0 for key in self.session_budget()}

    def _review_key(self, context):
        from photo_session_review import _connection_identity, review_prompt
        from runtime_resources import code_path
        context = copy.deepcopy(context); context.setdefault('user_intent', self.user_intent)
        images = context['images']
        return endpoint(source_sha256=(context.get('source_original') or images[0])['sha256'], candidate_sha256=images[1]['sha256'],
            scope_id=context['scope']['scope_id'], design_id=context['design']['design_id'],
            prompt=review_prompt(context), connection_identity=_connection_identity(self.connection),
            decoder_identity=sha256_file(code_path('raw_support.py')),
            previous_best_sha256=images[2]['sha256'] if len(images) == 3 else None,
            source_preview_sha256=images[0]['sha256'] if context.get('source_original') else None)

    def cached_review(self, context):
        key = self._review_key(context); path = Path(self.locator['path'])
        with exclusive_file_lock(path.with_suffix('.lock'), timeout=30):
            state = _read(path)
            for row in state['reviews']:
                if row.get('endpoint') == key:
                    if row['status'] != 'ready':
                        raise ReviewPending('The existing review remains unresolved')
                    return _ready(row, key)
            for row in state['operations']:
                if row['kind'] != 'post_review':
                    continue
                saved_key = row.get('review_endpoint') or {}
                # An adopted completed repair can retain its original three
                # image comparison. Recompile that original role arrangement
                # under today's supplied objective; never relabel it as a new
                # two-image observation or drop the actual previous best.
                original = copy.deepcopy(row['context'])
                for field in ('scope','design','user_intent'):
                    if field in context:original[field]=copy.deepcopy(context[field])
                matches = saved_key == key or (
                    saved_key.get('source_sha256') == key['source_sha256']
                    and saved_key.get('candidate_sha256') == key['candidate_sha256']
                    and len(context['images']) == 2 and len(original['images']) == 3
                    and saved_key == self._review_key(original))
                if not matches:
                    continue
                if row['status'] != 'ready':
                    raise ReviewPending('The existing correction review remains unresolved')
                _check_context(state, original)
                _validate_operation_result(row, row['result'])
                return copy.deepcopy(row['result'])
        return None

    def _wrap_operation(self, callback, kind, *, tool_costs=None):
        def wrapped(context):
            ticket = reserve_operation(self.locator, kind, context,
                review_endpoint=self._review_key(context) if kind == 'post_review' else None,
                tool_costs=tool_costs,
                reuse_review=self.cached_review(context) if kind == 'post_review' else None)
            if not ticket['owner']:
                return ticket['result']
            try:
                if kind == 'post_review':
                    ticket['review_endpoint'] = self._review_key(context)
                result = callback(context)
                finish_operation(ticket, result=result)
                return result
            except BaseException:
                finish_operation(ticket)
                raise
        return wrapped

    def wrap_decide(self, callback):
        return self._wrap_operation(callback, 'decision')

    def wrap_tools(self, tools):
        from dataclasses import replace
        return [replace(tool, execute=self._wrap_operation(tool.execute, 'tool', tool_costs=copy.deepcopy(tool.costs))) for tool in tools]

    def wrap_evaluate(self, callback):
        def wrapped(context):
            path = Path(self.locator['path']); candidate = context['images'][1]['sha256']
            with exclusive_file_lock(path.with_suffix('.lock'), timeout=30):
                state = _read(path)
                post = any(row['kind'] == 'tool' and row['status'] == 'ready'
                           and row['session_id'] == context['session_id']
                           and row['output_image']['sha256'] == candidate for row in state['operations'])
            if post:
                return self._wrap_operation(callback, 'post_review')(context)
            ticket = reserve(self.locator, self._review_key(context), run_id=self.run_id)
            if not ticket['owner']:
                return ticket['result']
            try:
                result = callback(context)
                board = Path(result['provenance']['review_board']['path'])
                finish(ticket, result=result, evidence=_asset(board.parent.parent/'review.json'), actual_model_requests=1)
                return result
            except BaseException:
                finish(ticket)
                raise
        return wrapped


def make_hooks(root, run, connection):
    return OriginHooks(root, run, connection)


MANUAL_GUARD_SCHEMA = 'cosluma_manual_target_guard.v1'


class ManualTargetBlocked(ValueError):
    """An identical paid target is active or has an unresolved external call."""


def _endpoint_identity(value):
    parsed = urlsplit(str(value or ''))
    # Credential rotation, signed queries and local source snapshots are not
    # new model targets. Never persist authorization or URL user information.
    return identity({'scheme': parsed.scheme.lower(), 'host': parsed.hostname,
                     'port': parsed.port, 'path': parsed.path.rstrip('/')})


def manual_provider_target(options, *, image_generation=True):
    """Resolve public transport identity read-only; CPU work needs no image readiness."""
    import os
    from vision_transport import _connection
    values = options if isinstance(options, dict) else vars(options)
    selected = {key: values.get(key) for key in ('provider', 'model', 'api_base', 'api_url', 'api_key_env')}
    if image_generation:
        import reference_provider
        config = reference_provider.provider_config(**selected)
        kind = config.get('kind')
        image = {'kind': kind, 'endpoint_id': _endpoint_identity(config.get('endpoint')),
                 'model': (str(config.get('registry_id'))+':'+Path(str(config.get('model') or '')).name
                           if kind in {'comfyui', 'local'} else str(config.get('model') or 'auto'))}
    else:
        # Unused image settings cannot grant a fresh paid vision allowance.
        # Provider/base still flow into the actual vision resolver below.
        image = {'kind': 'not_requested'}
    try:
        vision_config, endpoint, _secret, _protocol = _connection(**{key: selected[key] for key in
            ('provider', 'api_base', 'api_url', 'api_key_env')})
        vision = {'kind': vision_config['kind'], 'endpoint_id': _endpoint_identity(endpoint)}
    except (ValueError, RuntimeError, OSError):
        # A missing credential is configuration, never permission to issue a
        # call. The normal task preflight still owns connectivity/readiness.
        vision = {'kind': 'unresolved', 'endpoint_id': _endpoint_identity(
            os.environ.get('COSPLAY_VISION_API_URL') or os.environ.get('COSPLAY_VISION_API_BASE')
            or values.get('api_url') or values.get('api_base'))}
    vision['model'] = str(values.get('vision_model') or os.environ.get('COSPLAY_VISION_MODEL') or 'auto')
    return {'image': image, 'vision': vision}


def manual_target_key(*, source_sha256, base_sha256, scope_id, design_id, user_feedback='', provider_target=None,
                      shaping_request=None):
    """One semantic target across child IDs, budgets, reasons and code versions."""
    if not isinstance(user_feedback, str) or len(user_feedback) > 8000:
        raise ValueError('Manual guard feedback must be bounded explicit user text')
    result = {'source_sha256': source_sha256, 'base_sha256': base_sha256,
              'scope_id': scope_id, 'design_id': design_id, 'user_feedback': user_feedback.strip(),
              'provider_target': copy.deepcopy(provider_target)}
    if shaping_request is not None:
        from photo_shaping import normalize_request
        request = normalize_request(shaping_request)
        if request['action'] != 'apply' or user_feedback.strip():
            raise ValueError('Shaping target requires one explicit apply request without unrelated feedback')
        result['shaping_request'] = request
    if any(not isinstance(result[key], str) or len(result[key]) != 64
           or any(char not in '0123456789abcdef' for char in result[key])
           for key in ('source_sha256', 'base_sha256', 'scope_id', 'design_id')):
        raise ValueError('Manual guard requires exact source/base/scope/design identities')
    json.dumps(result, allow_nan=False)
    return result


def _manual_path(root, binding):
    return Path(root).resolve()/'outputs/request-budgets/manual-targets'/identity(binding)/'guard.json'


def _manual_read(path, binding):
    saved = _read_json(path)
    if (saved.get('schema') != MANUAL_GUARD_SCHEMA or saved.get('binding') != binding
            or saved.get('target_id') != identity(binding) or not isinstance(saved.get('attempts'), list)
            or not saved['attempts']):
        raise ValueError('Manual target guard identity changed; inspect existing evidence')
    return saved


def check_manual_target(root, binding):
    """Read-only planning check: no lock creation, writes, network or model calls."""
    path = _manual_path(root, binding)
    if not path.exists():
        return {'status': 'clear', 'target_id': identity(binding)}
    state = _manual_read(path, binding); latest = state['attempts'][-1]
    if latest['status'] in {'active', 'unknown'}:
        raise ManualTargetBlocked('This exact manual target has an active or unknown request in '+
            latest['run_id']+'; inspect its real existing receipts instead of resubmitting under a new child ID')
    return {'status': 'clear', 'target_id': state['target_id'], 'previous_run_id': latest['run_id']}


def _record_matches(asset):
    return isinstance(asset, dict) and _asset(asset['path']) == {key: asset[key] for key in ('path', 'sha256')}


def _known_vision_attempt(run, attempt):
    kind = attempt['kind']; name = 'decision' if kind == 'decide' else 'review'
    output = run/('editing-decisions' if kind == 'decide' else 'editing-reviews')/attempt['attempt_id']
    receipt = output/(name+'.json')
    if not receipt.is_file():
        return False, []
    value = _read_json(receipt); evidence = [_asset(receipt)]
    expected_schema = 'cosluma_photo_session_decision.v1' if kind == 'decide' else 'cosluma_photo_session_review.v1'
    if (value.get('schema') != expected_schema
            or (value.get('binding') or {}).get('context_sha256') != identity(attempt['context'])):
        return False, evidence
    if value.get('model_requests_attempted') == 0:
        return True, evidence
    provenance = value.get('provenance') or {}
    if (provenance.get('network_used') is True and provenance.get('generation_used') is False
            and provenance.get('model') and _record_matches(provenance.get('source'))):
        return True, evidence
    failure = output/'transport/response-failure.json'
    if failure.is_file():
        terminal = _read_json(failure)
        if (terminal.get('schema') == 'vision_response_failure.v1'
                and terminal.get('failure_kind') in {'provider_failure', 'completed_invalid_json'}
                and _record_matches(terminal.get('request_binding'))
                and _record_matches(terminal.get('response_payload'))):
            return True, [*evidence, _asset(failure)]
    error = value.get('error') or {}
    if error.get('type') == 'VisionHTTPError' and str(error.get('message', '')).startswith('vision provider HTTP '):
        return True, evidence
    if (not (output/'transport/request-binding.json').exists()
            and (error.get('type') in {'ValueError', 'FileNotFoundError', 'PermissionError', 'IsADirectoryError'}
                 or error.get('type') == 'RuntimeError' and 'credentials are unavailable' in str(error.get('message', '')))):
        return True, evidence  # Current transport writes this binding before any model POST.
    return False, evidence


def _known_image_attempt(run, attempt):
    output = run/'editing-tools'/attempt['attempt_id']; path = output/'tool-attempt.json'
    if not path.is_file():
        return False, []
    tool = _read_json(path); evidence = [_asset(path)]
    binding = tool.get('binding') or {}
    if (tool.get('schema') != 'cosluma_photo_edit_tool.v1'
            or binding.get('attempt_id') != attempt['attempt_id']
            or binding.get('context_sha256') != identity(attempt['context'])):
        return False, evidence
    transports=[]
    for receipt in (output/'provider').glob('edited.transport-*.json'):
        try:
            value=_read_json(receipt)
            if isinstance(value,dict):transports.append((receipt,value))
        except (OSError,ValueError):
            continue  # Unreadable evidence never establishes a terminal call.
    rejected=((output/'provider/edited.quality-rejection.json').exists() or any(
        value.get('state')=='rejected_quality' or 'quality_rejection_path' in value or 'quality_rejection_sha256' in value
        for _receipt,value in transports))
    if rejected:
        # A rejection cannot fall through to a state-only success/error check.
        # Its returned bytes remain rejected; this observes a terminal request,
        # never candidate registration, a refund or authority to resubmit.
        if not output.resolve().is_relative_to(run.resolve()):return False,evidence
        for receipt,transport in transports:
            known,proof=_known_quality_rejection(output,attempt,tool,receipt,transport)
            if known:return True,[*evidence,*proof]
        return False,evidence
    if _record_matches(tool.get('generated')) and _record_matches(tool.get('provider_receipt')):
        return True, evidence
    if 'generator_input' not in tool and tool.get('state') in {'needs_attention', 'interrupted'}:
        return True, evidence  # Durable tool record failed before provider submission preparation.
    for receipt,transport in transports:
        request = transport.get('request') or {}
        if (transport.get('schema') != 'reference_transport_attempt.v1'
                or (transport.get('source') or {}).get('sha256') != attempt['context']['images'][0]['sha256']
                or request.get('requested_prompt', request.get('prompt')) != (tool.get('compiled_prompt') or {}).get('prompt')):
            continue
        evidence.append(_asset(receipt))
        if transport.get('state') in {'response_received', 'registered', 'rejected_input_quality'}:
            return True, evidence
        events = transport.get('events') or []
        if any(row.get('event') == 'comfyui_execution_failed' for row in events):
            return True, evidence
        if any(row.get('event') == 'provider_attempt' and type(row.get('http_status')) is int for row in events):
            return True, evidence
        if (transport.get('error_type') == 'HTTPError'
                and not any(row.get('event') == 'comfyui_submitted' for row in events)):
            return True, evidence
    return False, evidence


def _known_quality_rejection(output, attempt, tool, receipt, transport):
    """Verify one owned complete response and its unchanged rejected bytes."""
    try:
        provider=(output/'provider').resolve();context=attempt['context']
        original=context.get('source_original') or context['images'][0]
        source={key:original[key] for key in ('path','sha256')}
        best=next(row for row in context['images'] if row.get('role')=='best')
        prompt=tool['compiled_prompt']['prompt'];request=transport['request'];response=transport['response']
        binding=tool['binding'];generator=tool['generator_input'];api=transport['api_input']
        if (not provider.is_relative_to(output.resolve()) or receipt.resolve().parent!=provider
                or transport.get('schema')!='reference_transport_attempt.v1'
                or transport.get('state')!='rejected_quality' or transport.get('candidate_registered') is not False
                or not isinstance(transport.get('provider'),str) or not transport['provider'].startswith('qwen_image')
                or not context.get('session_id') or binding.get('session_id')!=context['session_id']
                or binding.get('source')!=source or transport.get('source')!=source or not _record_matches(source)
                or binding.get('current_best')!=best or not _record_matches(best)
                or not isinstance(prompt,str) or not prompt.strip() or tool['compiled_prompt'].get('sha256')!=identity(prompt)
                or request.get('method')!='POST' or request.get('requested_prompt',request.get('prompt'))!=prompt
                or request.get('prompt')!=prompt  # The recorded current Qwen recipe is verbatim.
                or not _record_matches(generator) or not Path(generator['path']).resolve().is_relative_to(output.resolve())
                or not _record_matches(api) or Path(api['path']).resolve()!=Path(generator['path']).resolve()
                or api['sha256']!=generator['sha256'] or api.get('bytes')!=Path(generator['path']).stat().st_size
                or response.get('mode')!='comfyui_view' or not isinstance(response.get('id'),str) or not response['id']
                or type(response.get('image_bytes')) is not int or response['image_bytes']<=0
                or not isinstance(response.get('generated_model'),str) or not response['generated_model']
                or tool.get('generated') is not None or tool.get('provider_receipt') is not None
                or (provider/'edited.png').exists() or (provider/'edited.provenance.json').exists()):
            return False,[]
        conditions=transport.get('condition_inputs')
        inputs=[api] if conditions is None else [row['api_input'] for row in conditions]
        if (not inputs or inputs[0]['sha256']!=api['sha256'] or any(not _record_matches(row)
                or not Path(row['path']).resolve().is_relative_to(output.resolve()) for row in inputs)):
            return False,[]
        events=transport['events']
        uploads=[row for row in events if row.get('event')=='comfyui_uploaded']
        submitted=[row for row in events if row.get('event')=='comfyui_submitted']
        completed=[row for row in events if row.get('event')=='comfyui_completed']
        if (len(uploads)!=len(inputs) or len(submitted)!=1 or len(completed)!=1
                or any(row.get('event')=='comfyui_execution_failed' for row in events)
                or any(row.get('index')!=index or row.get('input_sha256')!=image['sha256']
                    or row.get('bytes')!=Path(image['path']).stat().st_size
                    for index,(row,image) in enumerate(zip(uploads,inputs),1))
                or submitted[0].get('prompt_id')!=response['id'] or completed[0].get('prompt_id')!=response['id']):
            return False,[]
        graph=submitted[0]['request']['prompt'];status=completed[0]['status']
        messages=status.get('messages') or []
        graph_sha=hashlib.sha256(json.dumps(graph,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        names=['/'.join(part for part in (str(row['upload'].get('subfolder') or ''),row['upload']['name']) if part)
            for row in uploads]
        if (not isinstance(graph,dict) or submitted[0].get('submitted_workflow_sha256')!=graph_sha
                or any(not isinstance(row['upload'].get('name'),str) or not row['upload']['name'] for row in uploads)
                or not _known_qwen_rejection_flow(graph,completed[0].get('output_node'),names,prompt,
                    request.get('model'),conditions,response['generated_model'])
                or status.get('completed') is not True or status.get('status_str')!='success'
                or not any(isinstance(row,list) and len(row)==2 and row[0]=='execution_success'
                    and isinstance(row[1],dict) and row[1].get('prompt_id')==response['id'] for row in messages)
                or any(isinstance(row,list) and row and row[0] in {'execution_error','execution_interrupted'} for row in messages)):
            return False,[]
        rejection_path=Path(transport['quality_rejection_path']).resolve()
        rejected_path=Path(transport['rejected_path']).resolve()
        if (rejection_path!=provider/'edited.quality-rejection.json'
                or not _record_matches({'path':str(rejection_path),'sha256':transport['quality_rejection_sha256']})
                or rejected_path.parent!=provider or not rejected_path.name.startswith('edited.quality-rejected-')
                or not _record_matches({'path':str(rejected_path),'sha256':response['output_sha256']})
                or rejected_path.stat().st_size!=response['image_bytes']):
            return False,[]
        rejection=_read_json(rejection_path);quality=rejection.get('quality') or {};metrics=quality.get('metrics') or {}
        if (rejection.get('schema')!='reference_quality_rejection.v1' or rejection.get('state')!='rejected_quality'
                or rejection.get('source')!=source or rejection.get('provider')!=transport['provider']
                or rejection.get('model')!=response['generated_model'] or rejection.get('output_sha256')!=response['output_sha256']
                or Path(rejection.get('rejected_path') or '').resolve()!=rejected_path
                or rejection.get('candidate_registered') is not False or type(rejection.get('image_generation_submissions')) is not int
                or rejection['image_generation_submissions']!=1 or transport.get('quality')!=quality
                or quality.get('schema')!='qwen_grid_quality_gate.v1' or quality.get('context')!='generated_output'
                or quality.get('grid_policy')!='reject_corroborated_periodic_grid' or quality.get('passed') is not False
                or metrics.get('bytes')!=response['image_bytes']):
            return False,[]
        if quality.get('decodable') is True:
            if (quality.get('grid_detected') is not True or quality.get('threshold_exceeded') is not True
                    or (quality.get('periodicity') or {}).get('corroborated') is not True
                    or metrics.get('sha256')!=response['output_sha256']):return False,[]
        elif quality.get('decodable') is not False or not isinstance(metrics.get('error_type'),str) or not metrics['error_type']:
            return False,[]
        return True,[_asset(receipt),_asset(rejection_path),_asset(rejected_path),
            *[_asset(Path(row['path'])) for row in inputs],_asset(Path(request['model']))]
    except (OSError,ValueError,KeyError,TypeError,AttributeError,StopIteration):
        return False,[]


def _known_qwen_rejection_flow(graph, output_node, names, prompt, workflow_path, conditions, generated_model):
    """Follow the returned output, then verify the recorded workflow binding.

    This only inspects bounded JSON and known pure workflow construction. No
    graph executes, weights load, or disconnected prompt/image proves usage.
    """
    from comfyui_adapter import prepare_condition_workflow
    try:
        def node(identifier,classes):
            if not isinstance(identifier,str) or identifier not in graph:raise ValueError('Missing live node')
            value=graph[identifier]
            if value.get('class_type') not in classes or not isinstance(value.get('inputs'),dict):
                raise ValueError('Unknown live node type')
            return value['inputs']
        def linked(value,classes,port):
            if (not isinstance(value,list) or len(value)!=2 or type(value[1]) is not int or value[1]!=port):
                raise ValueError('Wrong live output port')
            return value[0],node(value[0],classes)
        save=node(output_node,{'SaveImage'})
        _decoder,decoder=linked(save.get('images'),{'VAEDecode'},0)
        _sampler,sampler=linked(decoder.get('samples'),{'KSampler'},0)
        encoder_id,encoder=linked(sampler.get('latent_image'),{'TextEncodeQwenImage21','CosplayTextEncodeQwenImage21'},2)
        positive,_=linked(sampler.get('positive'),{'TextEncodeQwenImage21','CosplayTextEncodeQwenImage21'},0)
        negative,_=linked(sampler.get('negative'),{'TextEncodeQwenImage21','CosplayTextEncodeQwenImage21'},1)
        if positive!=encoder_id or negative!=encoder_id or encoder.get('prompt')!=prompt:return False
        cfg=sampler.get('cfg')
        if isinstance(cfg,list):
            if cfg!=[encoder_id,3] or type(cfg[1]) is not int:return False
        elif type(cfg) not in (int,float):return False
        expected_ports={f'images.image_{index}' for index in range(1,len(names)+1)}
        if {key for key in encoder if key.startswith('images.image_')}!=expected_ports:return False
        for index,name in enumerate(names,1):
            _image,image=linked(encoder[f'images.image_{index}'],{'LoadImage'},0)
            if image.get('image')!=name:return False
        vae_id,_vae=linked(encoder.get('vae'),{'VAELoader'},0)
        decode_vae,_=linked(decoder.get('vae'),{'VAELoader'},0)
        if decode_vae!=vae_id:return False
        linked(encoder.get('clip'),{'CLIPLoader'},0)
        _cache,cache=linked(sampler.get('model'),{'QwenImage21Cache'},0)
        linked(cache.get('model'),{'UNETLoader','UnetLoaderGGUF'},0)
        path=Path(workflow_path)
        if not path.is_absolute() or not path.is_file() or path.suffix.lower()!='.json' or path.stat().st_size>1024*1024:return False
        document=_read_json(path)
        if str(document.get('output_node'))!=output_node or document.get('model_name')!=generated_model:return False
        coordinate=(conditions[0].get('coordinate_space') if conditions else None)
        if conditions and (not isinstance(coordinate,str) or any(row.get('coordinate_space')!=coordinate for row in conditions)):
            return False
        expected=prepare_condition_workflow(document,names,prompt,coordinate_space=coordinate)
        observed=copy.deepcopy(graph)
        # Equivalent source LoadImage aliases are valid only at the real ordered
        # encoder ports. Unused nodes cannot substitute for a consumed image.
        for key in expected_ports:
            observed[encoder_id]['inputs'][key]=copy.deepcopy(expected[encoder_id]['inputs'][key])
        if document.get('randomize_seed'):
            for key,value in expected.items():
                if value['class_type']=='KSampler':
                    seed=graph[key]['inputs'].get('seed')
                    if type(seed) is not int or not 0<=seed<2**64:return False
                    value['inputs']['seed']=seed
        return observed==expected
    except (OSError,ValueError,KeyError,TypeError,AttributeError):
        return False


def classify_manual_outcome(root, binding, *, run_id, require_session=False):
    """Classify only this owner's actual persistent attempts and terminal receipts."""
    run = Path(root).resolve()/'outputs/runs'/_component(run_id)
    session_path = run/'editing-session/session.json'
    if not session_path.is_file():
        started = any((run/name).exists() for name in ('editing-decisions', 'editing-reviews', 'editing-tools'))
        return {'status': 'unknown' if require_session or started else 'known', 'reason': 'no_session_evidence', 'evidence': []}
    state = _read_json(session_path); evidence = [_asset(session_path)]
    user_feedback = (state.get('user_intent') or {}).get('user_feedback', '').strip()
    session_binding = state.get('binding') or {}
    shaping = binding.get('shaping_request')
    shaping_auth = session_binding.get('shaping_authorization') or {}
    shaping_matches = (shaping is None or shaping_auth.get('request') == shaping
        and shaping_auth.get('manual_target') == binding
        and shaping_auth.get('base_image_sha256') == binding['base_sha256'])
    expected_usage = copy.deepcopy(session_binding.get('initial_usage') or {})
    for attempt in state.get('attempts', []):
        for key, value in (attempt.get('costs') or {}).items():
            expected_usage[key] = expected_usage.get(key, 0)+value
    if (state.get('schema') != 'cosluma_photo_edit_session.v1'
            or state.get('session_id') != identity(session_binding) or state.get('used') != expected_usage
            or session_binding.get('source', {}).get('sha256') != binding['source_sha256']
            or (state.get('scope') or {}).get('scope_id') != binding['scope_id']
            or (state.get('design') or {}).get('design_id') != binding['design_id']
            or user_feedback != binding['user_feedback']
            or not shaping_matches
            or not state.get('candidates') or state['candidates'][0].get('image', {}).get('sha256') != binding['base_sha256']):
        return {'status': 'unknown', 'reason': 'owner_evidence_binding_differs', 'evidence': evidence}
    unknown = []
    for attempt in state.get('attempts', []):
        costs = attempt.get('costs') or {}
        if attempt.get('kind') in {'decide', 'evaluate'} and costs.get('vision_requests', 0):
            known, proof = _known_vision_attempt(run, attempt)
        elif attempt.get('kind') == 'tool' and costs.get('image_submissions', 0):
            known, proof = _known_image_attempt(run, attempt)
        else:
            continue  # CPU work and adopted observations submit no external model request.
        evidence.extend(proof)
        if not known:
            unknown.append({'attempt_id': attempt['attempt_id'], 'kind': attempt['kind']})
    return {'status': 'unknown' if unknown else 'known', 'reason': 'unresolved_external_attempt' if unknown else 'terminal_receipts_known',
            'unknown_attempts': unknown, 'session_status': state.get('status'), 'evidence': evidence}


@contextmanager
def manual_target_guard(root, binding, *, run_id):
    """Atomic execution claim; finally retains unknown calls across all child IDs."""
    run_id = _component(run_id); path = _manual_path(root, binding); token = uuid.uuid4().hex
    with exclusive_file_lock(path.with_suffix('.lock'), timeout=30):
        check_manual_target(root, binding)
        state = (_manual_read(path, binding) if path.exists() else
                 {'schema': MANUAL_GUARD_SCHEMA, 'target_id': identity(binding), 'binding': copy.deepcopy(binding), 'attempts': []})
        state['attempts'].append({'run_id': run_id, 'token': token, 'status': 'active',
                                  'automatic_resubmission': False, 'automatic_expiry': False})
        atomic_json(path, state)
    try:
        yield {'target_id': identity(binding), 'path': str(path), 'run_id': run_id}
    finally:
        try:
            outcome = classify_manual_outcome(root, binding, run_id=run_id)
        except (ValueError, OSError, KeyError, TypeError) as error:
            outcome = {'status': 'unknown', 'reason': 'inspect_invalid_existing_evidence', 'error_type': type(error).__name__}
        with exclusive_file_lock(path.with_suffix('.lock'), timeout=30):
            state = _manual_read(path, binding); latest = state['attempts'][-1]
            if latest['token'] == token:
                latest.update(status='unknown' if outcome['status'] == 'unknown' else 'known', outcome=outcome)
                atomic_json(path, state)


def reconcile_manual_target(root, binding):
    """Unlock only from real terminal owner evidence; never from PID, age or assertions."""
    path = _manual_path(root, binding)
    with exclusive_file_lock(path.with_suffix('.lock'), timeout=30):
        state = _manual_read(path, binding); latest = state['attempts'][-1]
        outcome = classify_manual_outcome(root, binding, run_id=latest['run_id'], require_session=True)
        original_unknown = {row['attempt_id'] for row in (latest.get('outcome') or {}).get('unknown_attempts', [])}
        session_file = Path(root).resolve()/'outputs/runs'/latest['run_id']/'editing-session/session.json'
        current_ids = {row['attempt_id'] for row in _read_json(session_file).get('attempts', [])} if session_file.is_file() else set()
        if outcome['status'] != 'known' or outcome.get('session_status') == 'active':
            raise ManualTargetBlocked('No stopped-owner terminal evidence resolves this target; retain the existing request guard')
        if not original_unknown <= current_ids:
            raise ManualTargetBlocked('Historical unknown attempts disappeared; missing records never grant a new request')
        latest.update(status='known', resolution=outcome)
        atomic_json(path, state)
        return {'status': 'known', 'target_id': state['target_id'], 'run_id': latest['run_id'], 'evidence': outcome['evidence']}
