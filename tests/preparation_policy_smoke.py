"""Explicit redo uses current producers; replay and local repair keep history."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

import creative_task as task
from backend_server import Backend, _execution_binding
from current_stack_fixture import prepare_fixture
from generation_design import DESIGN_VERSION
from preparation_currentness import preparation_currentness
from production_coordinator import execute_workflow, route_contract
import reference_pipeline as pipeline
import single_source_workflow as workflow
from task_model_preflight_smoke import build_registry_fixture, mocked_task_capability


def rejected(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('Unchecked preparation policy was accepted')


def main():
    with tempfile.TemporaryDirectory(prefix='preparation-policy-') as folder:
        root = Path(folder)
        run = prepare_fixture(root/'assets')
        plan = run['options']['appearance_revision']['design_reconstruction_plan']
        layer = plan['layers'][0]
        inputs = {'recipe': pipeline.RECIPE, 'code': pipeline._code_identity(root)}
        snapshot = root/'preparation-snapshot.json'
        snapshot.write_text(json.dumps({'recipe': pipeline.RECIPE, 'pipeline_inputs': inputs}), encoding='utf8')
        layer['assets']['preparation_snapshot'] = {'path': str(snapshot), 'sha256': pipeline._sha(snapshot)}
        before = snapshot.read_bytes()
        with patch('numpy.load', side_effect=AssertionError('Currentness decoded native pixels')):
            current = preparation_currentness(root, run)
        assert current['state'] == 'current' and not current['reprepare_required']
        assert not current['native_assets_decoded']
        from runtime_resources import code_path
        from cos_node_registry import compile_workflow
        run['task_plan'] = {'mode': 'existing_reference', 'layer_workflow': compile_workflow('existing_reference')}
        assert preparation_currentness(root, run)['scope'] == 'saved_local_production'
        changed_boundary = root/'changed-boundary-worker.py'
        changed_boundary.write_bytes(code_path('boundary_light_worker.py').read_bytes() + b'\n# upgraded boundary producer\n')
        with patch('runtime_resources.code_path', side_effect=lambda name:
                changed_boundary if name == 'boundary_light_worker.py' else code_path(name)):
            boundary_stale = preparation_currentness(root, run)
        assert boundary_stale['reference_appearance_is_current']
        assert not boundary_stale['is_current'] and boundary_stale['state'] == 'stale'
        assert boundary_stale['execution_graph']['changed_nodes'] == ['boundary']
        assert 'boundary_layer' in boundary_stale['stale_scopes']
        changed = root/'changed-reference-automatic-layers.py'
        changed.write_bytes(code_path('reference_automatic_layers.py').read_bytes() + b'\n# producer update\n')
        current_inputs = lambda _root, _run: {'recipe': pipeline.RECIPE, 'code': pipeline._code_identity(_root)}
        with patch.object(pipeline, '_input_identity', side_effect=current_inputs), \
                patch('reference_appearance_stack.validate_layer'):
            assert pipeline._prepared_matches(root, run, {'design_reconstruction_plan': plan})
            with patch('runtime_resources.code_path', side_effect=lambda name:
                    changed if name == 'reference_automatic_layers.py' else code_path(name)):
                with patch('numpy.load', side_effect=AssertionError('Currentness decoded native pixels')):
                    stale = preparation_currentness(root, run)
                assert stale['state'] == 'stale' and stale['reprepare_required']
                assert [item['name'] for item in stale['changed_producers']] == ['reference_automatic_layers.py']
                assert not pipeline._prepared_matches(root, run, {'design_reconstruction_plan': plan})
        assert snapshot.read_bytes() == before

        # Opening/saving uses the explicit default replay behavior even when
        # its historical task plan was created with prepare_current.
        run['task_plan'] = {'preparation_policy': 'prepare_current'}
        run['options']['appearance_revision']['portrait_beauty'] = {'strength': .7, 'fixture': 'frozen Beauty'}
        run['options']['manual_layers'] = {'fixture': 'unchanged manual layer'}
        layer.update(enabled=False, strength=.63, colour_strength=.4, lightness_strength=.8)
        original = copy.deepcopy(run)
        replacement = copy.deepcopy(plan)
        replacement['layers'][0].update(enabled=True, strength=1., colour_strength=1., lightness_strength=1.)
        replacement['layers'][0]['params']['producer_fixture'] = 'new current producer'
        with patch('design_reconstruction.materialize_plan', side_effect=lambda _root, _run, value: copy.deepcopy(value)), \
                patch('scene_design.bind_plan_receiver_intent', side_effect=lambda _run, value: value), \
                patch('design_scene_assets.prepare_native_receivers', side_effect=lambda _root, _run, value: value), \
                patch('design_reconstruction_pipeline.ready', return_value=True), \
                patch('runtime_memory.native_analysis_cpu_bytes', return_value=0), \
                patch('beautygrpo_prepare.release_idle_for_stage', return_value={}), \
                patch('start_qwen_image.release_for_run', return_value={'state': 'not_managed_qwen'}), \
                patch.object(pipeline, 'prepare', return_value={'cache_hit': False, 'design_reconstruction_plan': replacement}) as prepare:
            workflow._refresh_design_plan(root, run)
            prepare.assert_not_called()
            assert run['options'] == original['options']
            workflow._refresh_design_plan(root, run, preparation_policy='prepare_current')
            assert prepare.call_count == 1
        prepared = run['options']['appearance_revision']['design_reconstruction_plan']['layers'][0]
        assert prepared['params']['producer_fixture'] == 'new current producer'
        for key in ('enabled', 'strength', 'colour_strength', 'lightness_strength'):
            assert prepared[key] == original['options']['appearance_revision']['design_reconstruction_plan']['layers'][0][key]
        assert run['options']['appearance_revision']['portrait_beauty'] == original['options']['appearance_revision']['portrait_beauty']
        assert run['options']['manual_layers'] == original['options']['manual_layers']
        assert run['preparation_execution']['frozen_parent_replayed'] is False
        assert snapshot.read_bytes() == before

        repair_run = copy.deepcopy(original)
        repair_run['task_plan'] = {'execution_strategy': 'layered_reference_transfer', 'preparation_policy': 'replay_frozen'}
        with patch.object(workflow, '_refresh_design_plan') as refresh, \
                patch('boundary_light_coordination_worker.prepare_for_workbench', side_effect=AssertionError('Local repair upgraded boundary')):
            assert workflow._prepare_task_boundary_light(root, repair_run,
                SimpleNamespace(final_repair_only=True, preparation_policy='replay_frozen'))
        assert refresh.call_args.kwargs['preparation_policy'] == 'replay_frozen'

        # The complete checked task contract owns the preparation policy;
        # caller overrides and stale plan IDs cannot switch it after planning.
        registry = build_registry_fixture(root/'models')
        intent = task.normalize_task_intent({'schema_version': task.INTENT_SCHEMA, 'mode': 'existing_reference'})
        public = {'source': run['source']['path'], 'reference': run['reference']['path'],
            'model_registry': str(registry), 'creative_task': intent,
            'appearance_revision': original['options']['appearance_revision'],
            'manual_layers': original['options']['manual_layers']}
        with patch('task_model_preflight.inspect_task_models', side_effect=mocked_task_capability):
            checked = task.plan_task(root, public)
            frozen = task.plan_task(root, {**public, 'preparation_policy': 'replay_frozen'})
            assert checked['plan']['preparation_policy'] == 'prepare_current'
            assert frozen['plan']['plan_id'] != checked['plan']['plan_id']
            request = task.execution_request(root, {**checked['request'], 'plan_id': checked['plan']['plan_id']})
            assert route_contract(request)['preparation_policy'] == 'prepare_current'
            rejected(lambda: route_contract({**request, 'preparation_policy': 'replay_frozen'}))
            rejected(lambda: task.execution_request(root, {**checked['request'],
                'preparation_policy': 'replay_frozen', 'plan_id': checked['plan']['plan_id']}))
            assert _execution_binding(request) != _execution_binding({**request, 'preparation_policy': 'replay_frozen'})

            backend = Backend(root)
            parent = copy.deepcopy(original)
            parent.update(run_id='parent', sample='fixture', status='needs_visual_review', auto_draft={'saved': True},
                task_intent=intent, task_plan=checked['plan'],
                generation_design={'schema_version': DESIGN_VERSION, 'status': 'ready'})
            parent['options'].update(task_intent=intent, task_plan=checked['plan'],
                algorithm_profile='dinov3_vitl16', model_registry=str(registry))
            original_parent = copy.deepcopy(parent)
            with patch.object(workflow, 'load_run', return_value=(root/'parent.json', parent)), \
                    patch.object(backend, 'one_click', side_effect=lambda body: body), \
                    patch('system_model_choices.options_for_choice', return_value={'model_registry': str(registry)}):
                retried = backend.retry_source_run('parent', {})
                completed = backend.complete_source_run('parent', {})
                repaired = backend.repair_final_run('parent', {})
            for execution in (retried, completed):
                assert execution['preparation_policy'] == execution['task_plan']['preparation_policy'] == 'prepare_current'
                assert execution['generate'] is False and sum(execution['task_plan']['image_budget'].values()) == 0
                assert execution['appearance_revision'] == original_parent['options']['appearance_revision']
                assert execution['manual_layers'] == original_parent['options']['manual_layers']
            assert repaired['preparation_policy'] == repaired['task_plan']['preparation_policy'] == 'replay_frozen'
            assert repaired['final_repair_only'] is True
            assert parent == original_parent

            # Follow the checked retry through the real coordinator/create
            # boundary. Only numeric producers/renderer are CPU fixtures.
            parent_path = root/'outputs/runs/parent/manifest.json'
            workflow.write_json(parent_path, parent)
            parent_bytes = parent_path.read_bytes()
            args = workflow.parse_args(['--root', str(root), '--source', run['source']['path'],
                '--reference', run['reference']['path'], '--run-id', 'current-retry',
                '--parent-run-id', 'parent', '--complete-post'])
            workflow.apply_workflow_settings(args, retried)
            stages = []
            def prepare_endpoint(_root, _run, **kwargs):
                stages.append('current_prepare')
                return {'cache_hit': False, 'design_reconstruction_plan': copy.deepcopy(replacement)}
            def boundary_endpoint(_root, _run, output, **kwargs):
                stages.append('boundary')
                fresh = _run['options']['appearance_revision']['design_reconstruction_plan']
                assert fresh['layers'][0]['params']['producer_fixture'] == 'new current producer'
                return {'ok': True, 'no_change': True, 'design_reconstruction_plan': fresh, 'receipt': {}}
            def render_endpoint(_root, saved, _args):
                stages.append('render')
                assert saved.get('_native_plan_prepared_for_render') is True
                output = root/'outputs/runs'/saved['run_id']/'render.png'
                output.parent.mkdir(parents=True, exist_ok=True)
                from PIL import Image
                Image.new('RGB', (24, 20), '#506080').save(output)
                return {'outputs': {'auto_draft': str(output)},
                        'metadata': {'run_id': saved['run_id'], 'revision': saved['revision']},
                        'review': {'checks': {key: 'not_checked' for key in workflow.VISUAL_CHECK_KEYS}}}
            with patch('design_reconstruction.materialize_plan', side_effect=lambda _root, _run, value: copy.deepcopy(value)), \
                    patch('scene_design.bind_plan_receiver_intent', side_effect=lambda _run, value: value), \
                    patch('design_scene_assets.prepare_native_receivers', side_effect=lambda _root, _run, value: value), \
                    patch('design_reconstruction_pipeline.ready', return_value=True), \
                    patch('runtime_memory.native_analysis_cpu_bytes', return_value=0), \
                    patch('beautygrpo_prepare.release_idle_for_stage', return_value={}), \
                    patch('start_qwen_image.release_for_run', return_value={'state': 'not_managed_qwen'}), \
                    patch('runtime_preflight.preflight_for_run', return_value={'status': 'ready'}), \
                    patch('automatic_native_retouch._validate_beauty_source_before_preparation'), \
                    patch('portrait_beauty_layer.materialize_for_run', side_effect=lambda _root, _run, value: value), \
                    patch.object(pipeline, 'prepare', side_effect=prepare_endpoint), \
                    patch.object(workflow, '_prepare_task_portrait_beauty'), \
                    patch('boundary_light_coordination_worker.prepare_for_workbench', side_effect=boundary_endpoint), \
                    patch.object(workflow, 'render_candidate', side_effect=render_endpoint), \
                    patch.object(workflow, 'diagnose_cosplay_post'):
                child = execute_workflow(root, args)
            assert stages == ['current_prepare', 'boundary', 'render'], stages
            assert child['parent_run_id'] == 'parent' and child['revision'] == parent['revision'] + 1
            assert child['production_contract']['preparation_policy'] == 'prepare_current'
            assert child['generation_budget']['image_submissions'] == 0
            assert all(value == 'not_checked' for value in child['review']['checks'].values())
            assert child['status'] == 'needs_visual_review' and child['delivery_status'] == 'auto_draft'
            assert parent_path.read_bytes() == parent_bytes and snapshot.read_bytes() == before
        with patch('workbench_store.record_recent') as recent:
            for status in ('failed', 'blocked', 'cancelled', 'awaiting_reference'):
                workflow._record_saved_result(root, {**parent, 'status': status})
            recent.assert_not_called()
            workflow._record_saved_result(root, {**parent, 'artifacts': {'auto_draft': 'saved.png'}})
            recent.assert_called_once()
        print(json.dumps({'ok': True, 'checked_preparation_policy_bound': True,
            'explicit_redo_current_ordinary_save_frozen': True, 'controls_beauty_manual_preserved': True,
            'small_snapshot_code_currentness': True, 'old_cache_rejected_after_producer_update': True,
            'failed_and_candidate_not_recorded_as_saved': True, 'model_inferences': 0, 'network_requests': 0}))


if __name__ == '__main__':
    main()
