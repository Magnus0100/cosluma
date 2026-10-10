"""Atomically prepare current hair and shoulder/sleeve boundary-light layers.

Reuse the existing producers, in that order, in one isolated local worker.
Only a complete current pair can replace the submitted edit snapshot.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import time

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
    stages = []
    for index, (kind, stage, worker, message) in enumerate(zip(KINDS, STAGES,
            (hair_worker, cloth_worker), ('正在更新参考发梢受光（1/2）。', '正在按新发梢端点更新肩袖暖光（2/2）。'))):
        if cancel and cancel(): raise InterruptedError('Boundary coordination cancelled; submitted plan retained')
        if progress: progress(stage, message, .1 if index == 0 else .55)
        if cancel and cancel(): raise InterruptedError('Boundary coordination cancelled before the next stage')
        before = copy.deepcopy(_plan(candidate))
        result = worker.prepare(root, candidate, output/stage)
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
        'model_tile_inferences_this_attempt': sum(int(s['receipt'].get('model_tile_inferences_this_attempt', 0)) for s in stages),
        'image_generation_submissions': 0, 'source_alpha_changed': False, 'source_geometry_changed': False,
        'preserved_layer_switches_and_strengths': True, 'artistic_acceptance': False}
    result = {'changed': changed, 'no_change': not changed, 'receipt': receipt,
        'design_reconstruction_plan': final, 'ok': True,
        'cache_hit': all(bool(s['receipt'].get('reference_matte_cache_hit')) for s in stages)}
    write(output/'qualification.json', receipt)
    return result


def prepare_for_workbench(root, run, output, *, cancel=None, progress=None):
    """Run both stages in one isolated worker, publishing only its final plan."""
    from current_model_registry import load_registry
    from runtime_resources import code_path
    root, output = Path(root).resolve(), Path(output).resolve()
    if cancel and cancel(): raise InterruptedError('Boundary coordination cancelled')
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = output.with_name(output.name+'-input.json'); write(snapshot, run)
    python = load_registry(root, registry_path=run['options'].get('model_registry'))['python']
    log = output.with_name(output.name+'-worker.log')
    if progress: progress(STAGES[0], '正在协调发梢与肩袖受光；两阶段完成后一起应用，不新增生图。', .05)
    with log.open('wb') as stream, log.open('r', encoding='utf8', errors='replace') as reader:
        process = subprocess.Popen([python, '-B', '-X', 'utf8', str(code_path('boundary_light_coordination_worker.py')),
            '--root', str(root), '--run-json', str(snapshot), '--output', str(output)], cwd=str(root),
            stdout=stream, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        def stop():
            if process.poll() is not None: return
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            else: process.terminate()
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)
        pending = ''
        def consume_progress():
            nonlocal pending
            pending += reader.read()
            lines = pending.split('\n'); pending = lines.pop()
            for line in lines:
                try: event = json.loads(line)
                except (ValueError, TypeError): continue
                if isinstance(event, dict) and event.get('type') == 'boundary_coordination_progress' and progress:
                    progress(event['stage'], event['message'], event['progress'])
        try:
            while process.poll() is None:
                if cancel and cancel():
                    stop(); raise InterruptedError('Boundary coordination cancelled; complete submitted plan retained')
                consume_progress(); time.sleep(.25)
            consume_progress()
        finally: stop()
    if process.returncode:
        raise ValueError('Boundary coordination failed; complete submitted plan retained: '+log.read_text('utf8', errors='replace')[-2200:])
    if cancel and cancel(): raise InterruptedError('Boundary coordination cancelled; staged result not applied')
    return json.loads((output/'result.json').read_text('utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--run-id'); inputs.add_argument('--run-json', type=Path)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--save-child', action='store_true')
    args = parser.parse_args()
    import single_source_workflow as workflow
    run = json.loads(args.run_json.read_text('utf8')) if args.run_json else workflow.load_run(args.root, args.run_id)[1]
    def progress(stage, message, amount):
        print(json.dumps({'type': 'boundary_coordination_progress', 'stage': stage, 'message': message,
                          'progress': amount}, ensure_ascii=False), flush=True)
    result = prepare(args.root, run, args.output, progress=progress)
    if args.save_child and result['changed']:
        from boundary_light_worker import save_current_snapshot
        path, child = save_current_snapshot(args.root, run, result['design_reconstruction_plan'])
        child['cosplay_post_review'] = {'status': 'unavailable', 'actual_api_evidence_valid': False,
            'visual_acceptance': False, 'reason': 'Coordinated hair/shoulder pixels require current visual review'}
        child['boundary_light_coordination_preparation'] = result['receipt']; workflow.write_json(path, child)
        result.update(child_manifest=str(path), child_run_id=child['run_id'])
    write(args.output/'result.json', result)
    print(json.dumps({key: value for key, value in result.items() if key != 'design_reconstruction_plan'}, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
