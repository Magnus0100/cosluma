"""Read small frozen preparation receipts and current producer code identities.

This compares saved T/B/S preparation and recorded production nodes with the
installed producer closure. It never decodes native arrays, loads models, prepares
assets, rewrites provenance, or changes the saved version's replay contract.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from asset_identity import sha256_file

SCHEMA = 'cosplay_preparation_currentness.v1'


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':')).encode('utf8')).hexdigest()


def preparation_currentness(root: Path | str, run: dict) -> dict:
    """Report currentness of the actual frozen snapshot's code/input closure."""
    root = Path(root)
    report = {'schema': SCHEMA, 'run_id': run.get('run_id'), 'revision': run.get('revision'),
        'scope': 'reference_appearance_preparation', 'state': 'not_prepared',
        'is_current': False, 'reprepare_required': False, 'changed_producers': [],
        'stale_scopes': [], 'frozen_identity': None, 'current_identity': None,
        'basis': 'frozen_preparation_snapshot_code_closure', 'native_assets_decoded': False,
        'model_inferences': 0, 'saved_version_modified': False,
        'open_behavior': 'replay_frozen', 'redo_behavior': 'prepare_current_in_new_revision'}
    plan = ((run.get('options') or {}).get('appearance_revision') or {}).get('design_reconstruction_plan') or {}
    stacks = [row for row in plan.get('layers', []) if row.get('kind') == 'reference_appearance_stack']
    if not stacks:
        report['reason'] = '当前版本尚无冻结的人物与场景融合准备。'
        return report
    try:
        if len(stacks) != 1:
            raise ValueError('Expected one frozen reference appearance stack')
        asset = stacks[0]['assets']['preparation_snapshot']
        path = Path(asset['path'])
        path = path if path.is_absolute() else root/path
        if sha256_file(path) != asset['sha256']:
            raise ValueError('Frozen preparation snapshot identity differs from its receipt')
        snapshot = json.loads(path.read_text(encoding='utf-8-sig'))
        inputs = snapshot.get('pipeline_inputs') or {}
        saved = inputs.get('code')
        if not isinstance(saved, dict) or not saved or not all(
                isinstance(name, str) and isinstance(digest, str) for name, digest in saved.items()):
            raise ValueError('Frozen preparation has no current pipeline input/code contract')
        # reference_pipeline imports only coordinator helpers at module load;
        # its identity reads Python files, never native assets or model data.
        from reference_pipeline import RECIPE, _code_identity
        current = _code_identity(root)
        frozen_recipe = inputs.get('recipe') or snapshot.get('recipe')
        report['frozen_identity'] = {'recipe': frozen_recipe, 'code': saved,
                                     'identity_sha256': _digest({'recipe': frozen_recipe, 'code': saved})}
        report['current_identity'] = {'recipe': RECIPE, 'code': current,
                                      'identity_sha256': _digest({'recipe': RECIPE, 'code': current})}
        changed = [{'name': name, 'frozen_sha256': saved.get(name), 'current_sha256': current.get(name)}
                   for name in sorted(set(saved) | set(current)) if saved.get(name) != current.get(name)]
        report['changed_producers'] = changed
        report['reference_appearance_is_current'] = not changed and frozen_recipe == RECIPE
        report['is_current'] = report['reference_appearance_is_current']
        style_layers=[row for row in plan.get('layers',[]) if row.get('kind')=='registered_reference_style']
        if style_layers:
            if len(style_layers)!=1:raise ValueError('Expected one frozen registered style operator')
            from registered_reference_style import producer_code
            asset=style_layers[0]['assets']['contract'];path=Path(asset['path'])
            path=path if path.is_absolute() else root/path
            if sha256_file(path)!=asset['sha256']:
                raise ValueError('Frozen registered-style contract identity differs from its receipt')
            saved_style=json.loads(path.read_text(encoding='utf-8-sig'))
            actual_style=producer_code();old_style=saved_style['identity']['producer_code']
            changed_style=[name for name in sorted(set(actual_style)|set(old_style))
                if actual_style.get(name)!=old_style.get(name)]
            report['registered_reference_style']={'is_current':not changed_style,
                'changed_producers':changed_style,'coefficients_decoded':False}
            report['is_current']=report['is_current'] and not changed_style
            if changed_style:report['stale_scopes'].append('registered_reference_style')
        checked = run.get('task_plan') or (run.get('options') or {}).get('task_plan') or {}
        graph = checked.get('layer_workflow')
        if isinstance(graph, dict) and graph.get('order'):
            from cos_node_registry import compile_workflow
            from cos_workflow_cache import compare, SCOPES
            current_graph = compile_workflow(checked.get('mode'))
            graph_delta = compare(graph, current_graph)
            report['scope'] = 'saved_local_production'
            report['execution_graph'] = {'frozen_graph_id': graph.get('graph_id'),
                'current_graph_id': current_graph.get('graph_id'),
                'changed_nodes': graph_delta['changed'], 'invalidated_nodes': graph_delta['invalidated'],
                'pixel_cache_validation_inferred': False}
            report['is_current'] = report['is_current'] and not graph_delta['invalidated']
            report['stale_scopes'] = list(dict.fromkeys([*report['stale_scopes'],
                *(SCOPES.get(node, node) for node in graph_delta['invalidated'])]))
        report['state'] = 'current' if report['is_current'] else 'stale'
        report['reason'] = ('冻结准备及已记录执行节点与当前安装的 producer 身份一致。' if report['is_current'] else
                            '此版本保留旧算法的冻结结果；明确重新制作会准备当前算法并保存新版本。')
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        report.update(state='unavailable', is_current=False, reason=str(error))
    report['reprepare_required'] = not report['is_current']
    if report['reprepare_required'] and (not report.get('reference_appearance_is_current') or not report['stale_scopes']):
        report['stale_scopes'] = list(dict.fromkeys(
            ['reference_appearance', 'boundary_light', 'composite'] + report['stale_scopes']))
    return report
