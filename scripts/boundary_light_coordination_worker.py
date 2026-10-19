"""Atomically prepare current hair and shoulder/sleeve boundary-light layers.

Reuse the existing producers, in that order, in one isolated local worker.
Only a complete current pair can replace the submitted edit snapshot.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from boundary_light_geometry_context import request_geometry_scope

CONTRACT = 'source_boundary_light_coordination.v1'
KINDS = ('source_boundary_light', 'source_cloth_boundary_light')
STAGES = ('hair_boundary_light', 'cloth_boundary_light')


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def _plan(run):
    return run['options']['appearance_revision']['design_reconstruction_plan']


def validate_prepared_pair(plan, run):
    """Require fresh endpoint bindings even for an off or zero-dose layer."""
    from design_reconstruction import validate_current_native_plan
    import boundary_light_runtime as hair
    import cloth_boundary_runtime as cloth
    validate_current_native_plan(plan)
    for runtime in (hair, cloth):
        layers = [row for row in plan['layers'] if row['kind'] == runtime.KIND]
        if len(layers) != 1:
            raise ValueError('Coordinated boundary preparation requires one hair and one shoulder layer')
        layer = layers[0]
        state = runtime.validate_layer(layer, plan, run)
        if state['bypass_reason'] not in (None, 'boundary_light_disabled', 'cloth_boundary_light_disabled'):
            raise ValueError('Coordinated boundary preparation cannot retain a stale bypass')
    return True


def _stage_failure_receipt(stages, *, stage, kind, error, state='failed'):
    """Build a durable receipt when one coordinated stage cannot finish.

    The coordinator deliberately re-raises the original error so callers keep
    the existing failure semantics.  The receipt is the machine-readable
    companion: it records completed stages, the failed stage, and that no
    partial plan was committed.  This prevents a cloth-only result from being
    mistaken for a complete hair+cloth preparation after a local failure.
    """
    detail = str(error).strip() or type(error).__name__
    return {
        'schema': CONTRACT,
        'state': state,
        'stages': [*stages, {
            'stage': stage,
            'kind': kind,
            'changed': False,
            'status': 'cancelled' if state == 'cancelled' else 'failed',
            'error_type': type(error).__name__,
            'error': detail[-2000:],
        }],
        'atomic_commit': False,
        'partial_plan_discarded': True,
        'failed_stage': stage,
        'reason': 'stage_cancelled' if state == 'cancelled' else 'stage_failed',
        'image_generation_submissions': 0,
        'source_alpha_changed': False,
        'source_geometry_changed': False,
        'artistic_acceptance': False,
    }


@request_geometry_scope
def prepare(root, run, output, *, cancel=None, progress=None):
    from design_reconstruction import validate_current_native_plan
    import boundary_light_worker as hair_worker
    import cloth_boundary_worker as cloth_worker
    original = copy.deepcopy(run); original_plan = _plan(original)
    validate_current_native_plan(original_plan)
    for kind in KINDS:
        if sum(row.get('kind') == kind for row in original_plan['layers']) > 1:
            raise ValueError('Coordinated boundary preparation cannot replace duplicate layers')
    candidate = copy.deepcopy(original)
    root, output = Path(root).resolve(), Path(output).resolve()
    if cancel and cancel(): raise InterruptedError('Boundary coordination cancelled; submitted plan retained')
    output.mkdir(parents=True, exist_ok=False)
    import workflow_timing
    with workflow_timing.span('boundary_registration_preflight', model_request=None, http_request=False):
        preflight = cloth_worker.registration_upper_bound(root, original, original_plan)
    if cancel and cancel(): raise InterruptedError('Boundary coordination cancelled; submitted plan retained')
    if preflight and preflight.get('rejected'):
        receipt = {'schema':CONTRACT, 'state':'no_change', 'stages':[], 'atomic_commit':True,
            'reason':'insufficient_pair_registration_anchors', 'registration_upper_bound':preflight,
            'model_tile_inferences_this_attempt':preflight['model_tile_inferences_this_attempt'],
            'image_generation_submissions':0, 'source_alpha_changed':False,
            'source_geometry_changed':False, 'artistic_acceptance':False}
        write(output/'qualification.json',receipt)
        return {'changed':False, 'no_change':True, 'receipt':receipt,
            'design_reconstruction_plan':copy.deepcopy(original_plan), 'ok':True,
            'cache_hit':preflight['reference_matte_cache_hit']}
    stages = []
    for index, (kind, stage, worker, message) in enumerate(zip(KINDS, STAGES,
            (hair_worker, cloth_worker), ('正在更新参考发梢受光（1/2）。', '正在按新发梢端点更新肩袖暖光（2/2）。'))):
        if cancel and cancel(): raise InterruptedError('Boundary coordination cancelled; submitted plan retained')
        if progress: progress(stage, message, .1 if index == 0 else .55)
        if cancel and cancel(): raise InterruptedError('Boundary coordination cancelled before the next stage')
        before = copy.deepcopy(_plan(candidate))
        try:
            result = worker.prepare(root, candidate, output/stage)
        except Exception as exc:
            # Persist the failed stage before re-raising.  The parent manifest
            # remains untouched, while the coordinator output stays useful to
            # the job API and to a later retry.
            receipt = _stage_failure_receipt(
                stages, stage=stage, kind=kind, error=exc,
                state='cancelled' if isinstance(exc, InterruptedError) else 'failed')
            write(output/'qualification.json', receipt)
            raise
        if cancel and cancel(): raise InterruptedError('Boundary coordination cancelled; staged result not applied')
        if not isinstance(result, dict) or not isinstance(result.get('changed'), bool) or not isinstance(result.get('design_reconstruction_plan'), dict):
            raise ValueError('Boundary coordination stage returned an incomplete result')
        after = result['design_reconstruction_plan']
        validate_current_native_plan(after)
        if (after['source_sha256'] != original_plan['source_sha256'] or after['candidate_sha256'] != original_plan['candidate_sha256']
                or after['canvas'] != original_plan['canvas']
                or {k: v for k, v in before.items() if k != 'layers'} != {k: v for k, v in after.items() if k != 'layers'}
                or [row for row in before['layers'] if row.get('kind') != kind] != [row for row in after['layers'] if row.get('kind') != kind]):
            raise ValueError('Boundary coordination stage changed an unrelated submitted edit')
        if not result['changed'] and after != before:
            raise ValueError('Boundary coordination no-change must retain the exact submitted stage plan')
        previous = next((row for row in before['layers'] if row.get('kind') == kind), None)
        replacements = [row for row in after['layers'] if row.get('kind') == kind]
        if result['changed'] and len(replacements) != 1:
            raise ValueError('Boundary coordination stage must upsert exactly one layer')
        if previous and result['changed'] and any(replacements[0].get(key) != previous.get(key) for key in ('id', 'enabled', 'strength')):
            raise ValueError('Boundary coordination must preserve the submitted layer switch and strength')
        # Preserve the exact old transport for equal frozen pixels so a ZIP
        # timestamp/recompression cannot change the downstream hair digest.
        if previous and result['changed']:
            from native_subject_detail_runtime import _read
            frozen, _ = _read(replacements[0]['assets']['frozen_delta'])
            if hair_worker.identical_frozen_output(previous, frozen):
                after = before
                result = {**result, 'changed': False, 'receipt': {
                    **result.get('receipt', {}), 'state': 'no_change', 'reason': 'identical_frozen_pixels'}}
        candidate['options']['appearance_revision']['design_reconstruction_plan'] = copy.deepcopy(after)
        stages.append({'stage': stage, 'kind': kind, 'changed': result['changed'], 'receipt': result.get('receipt', {})})
    changed = any(stage['changed'] for stage in stages)
    final = _plan(candidate)
    # No evidence for a missing member means the partial candidate is discarded.
    complete = all(sum(row.get('kind') == kind for row in final['layers']) == 1 for kind in KINDS)
    reason = None
    if changed and not complete:
        changed = False; final = copy.deepcopy(original_plan); reason = 'incomplete_pair_retained_submitted_plan'
    elif changed or complete:
        validate_prepared_pair(final, candidate)
    if not changed: final = copy.deepcopy(original_plan)
    if cancel and cancel(): raise InterruptedError('Boundary coordination cancelled; complete result not applied')
    assert run == original, 'Boundary coordination mutated the caller edit snapshot'
    receipt = {'schema': CONTRACT, 'state': 'prepared' if changed else 'no_change', 'stages': stages,
        'atomic_commit': True, 'preparation_order': list(STAGES), 'reason': reason or ('unchanged_submitted_pair' if not changed else None),
        'reference_observation_cache_reused': bool(stages[1]['receipt'].get('reference_matte_cache_hit')),
        'model_tile_inferences_this_attempt': sum(int(s['receipt'].get('model_tile_inferences_this_attempt', 0)) for s in stages)
            + int((preflight or {}).get('model_tile_inferences_this_attempt', 0)),
        'image_generation_submissions': 0, 'source_alpha_changed': False, 'source_geometry_changed': False,
        'preserved_layer_switches_and_strengths': True, 'artistic_acceptance': False}
    from native_cuda_ndimage import ndimage_execution
    receipt['native_ndimage_execution'] = ndimage_execution()
    result = {'changed': changed, 'no_change': not changed, 'receipt': receipt,
        'design_reconstruction_plan': final, 'ok': True,
        'cache_hit': (all(bool(s['receipt'].get('reference_matte_cache_hit')) for s in stages)
                      and (preflight is None or bool(preflight.get('reference_matte_cache_hit'))))}
    write(output/'qualification.json', receipt)
    return result


def prepare_for_workbench(root, run, output, *, cancel=None, progress=None):
    """Run both stages in one isolated worker, publishing only its final plan."""
    from current_model_registry import load_registry
    from runtime_resources import code_path
    from reference_stage_worker import execute_stage
    import workflow_timing
    root, output = Path(root).resolve(), Path(output).resolve()
    if cancel and cancel(): raise InterruptedError('Boundary coordination cancelled')
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = output.with_name(output.name+'-input.json'); write(snapshot, run)
    python = load_registry(root, registry_path=run['options'].get('model_registry'))['python']
    log = output.with_name(output.name+'-worker.log')
    if progress: progress(STAGES[0], '正在协调发梢与肩袖受光；两阶段完成后一起应用，不新增生图。', .05)
    try:
        metrics = execute_stage(python, root,
            [str(code_path('boundary_light_coordination_worker.py')),
             '--root', str(root), '--run-json', str(snapshot), '--output', str(output)],
            output.parent, cancel, stage='boundary_coordination', log_path=log, progress=progress,
            registry_path=run['options'].get('model_registry'))
        workflow_timing.record_worker('boundary_coordination', wall_seconds=metrics.get('wall_seconds'),
            worker_timeline=metrics.get('worker_timeline'), worker_pid=metrics.get('worker_pid'),
            native_field_execution=metrics.get('native_field_execution'))
    except (RuntimeError, InterruptedError, ValueError) as error:
        # A worker can fail before it reaches ``prepare`` (for example during
        # import or registry loading).  Keep the same durable failure contract
        # as an in-stage exception so callers never have to infer the state
        # from a missing file or an unstructured log tail.
        qualification = output/'qualification.json'
        if not qualification.is_file():
            output.mkdir(parents=True, exist_ok=True)
            write(qualification, _stage_failure_receipt(
                [], stage='boundary_coordination', kind='boundary_coordination',
                error=error, state='cancelled' if isinstance(error, InterruptedError) else 'failed'))
        if isinstance(error, InterruptedError):
            raise
        raise ValueError('Boundary coordination failed; complete submitted plan retained: '+str(error)) from error
    if cancel and cancel(): raise InterruptedError('Boundary coordination cancelled; staged result not applied')
    return json.loads((output/'result.json').read_text('utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--run-id'); inputs.add_argument('--run-json', type=Path)
    # Child revision publication belongs to the backend coordinator. The
    # worker only prepares and persists its isolated result receipt.
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import single_source_workflow as workflow
    run = json.loads(args.run_json.read_text('utf8')) if args.run_json else workflow.load_run(args.root, args.run_id)[1]
    def progress(stage, message, amount):
        print(json.dumps({'type': 'boundary_coordination_progress', 'stage': stage, 'message': message,
                          'progress': amount}, ensure_ascii=False), flush=True)
    from native_cuda_filters import cuda_gaussian_scope
    from native_cuda_ndimage import cuda_ndimage_scope
    with cuda_gaussian_scope(), cuda_ndimage_scope():
        result = prepare(args.root, run, args.output, progress=progress)
    write(args.output/'result.json', result)
    print(json.dumps({key: value for key, value in result.items() if key != 'design_reconstruction_plan'}, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
