"""Real saved-image completion plans; only the model capability layer is mocked."""
import copy
import json
import os
from pathlib import Path
import tempfile
import urllib.request
from unittest.mock import patch

from PIL import Image

import backend_server
import creative_task as task
import project_cli
import single_source_workflow as workflow
import task_model_preflight
from generation_design import DESIGN_VERSION
from system_model_choices import options_for_choice, choice_for_options
from task_model_preflight_smoke import build_registry_fixture, mocked_task_capability


def rejected(call):
    try:
        call()
    except (ValueError, FileNotFoundError, RuntimeError):
        return
    raise AssertionError('Unsafe or incomplete saved completion was accepted')


def checked_zero_image_task(request, parent):
    plan = task.validate_task_plan(request['task_plan'], request['task_intent'])
    assert plan['ready'] and plan['mode'] == 'existing_reference'
    assert all(value == 0 for value in plan['image_budget'].values())
    assert plan['generation_submissions_upper_bound'] == 0
    assert request['cosplay_correction_budget'] == 0 and request['generate'] is False
    assert plan['inputs']['source']['sha256'] == parent['source']['sha256']
    assert plan['inputs']['reference']['sha256'] == parent['reference']['sha256']
    assert request['task_intent']['constraints'] == parent['task_intent']['constraints']
    assert request['task_intent']['brief'] == parent['task_intent']['brief']
    assert any(value.get('mocked') is True for value in plan['capabilities'].values() if isinstance(value, dict))


def main():
    with tempfile.TemporaryDirectory(prefix='system-completion-routes-') as folder:
        root = Path(folder).resolve()
        registry = build_registry_fixture(root)
        alternative_registry = build_registry_fixture(root, name='alternative-models')
        source, reference = root / 'source.png', root / 'reference.png'
        Image.new('RGB', (32, 24), '#384d65').save(source)
        Image.new('RGB', (30, 22), '#a487b0').save(reference)
        backend = object.__new__(backend_server.Backend); backend.root = root
        intent = task.normalize_task_intent({'schema_version': task.INTENT_SCHEMA,
            'mode': 'create_reference', 'brief': '保持妆造、低饱和暗调',
            'constraints': {'preserve_background': True, 'no_effects': True},
            'services': {'text_vision': True}})
        parent = {'kind': 'single_input_run', 'input_mode': 'single_source', 'run_id': 'saved',
            'sample': 'portrait', 'status': 'needs_visual_review', 'auto_draft': {'saved': True},
            'source': {'path': str(source), 'sha256': task.sha(source)},
            'reference': {'path': str(reference), 'sha256': task.sha(reference)},
            'task_intent': intent,
            'generation_design': {'schema_version': DESIGN_VERSION, 'status': 'ready'},
            'options': {'render_mode': 'native_source_design', 'algorithm_profile': 'dinov3_vitl16',
                'model_registry': str(registry), 'task_intent': copy.deepcopy(intent),
                'manual_layers': {'tone_curves': {'enabled': True}},
                'appearance_revision': {'params': {'contrast': .1}, 'design_reconstruction_plan': {'old': True}}}}
        original = copy.deepcopy(parent)
        manifest = root / 'manifest.json'
        manifest.write_text(json.dumps(parent, ensure_ascii=False), encoding='utf8')
        pointer = root / 'active-project.json'
        pointer.write_text(json.dumps({'active_run_id': 'saved'}), encoding='utf8')
        protected = {path: path.read_bytes() for path in (source, reference, manifest, pointer)}
        env = {'COSPLAY_MODEL_REGISTRY': str(registry), 'COSPLAY_MODEL_SETTINGS': str(root / 'selection.json')}
        with patch.dict(os.environ, env), \
             patch.object(workflow, 'load_run', return_value=(manifest, parent)), \
             patch.object(backend, 'one_click', side_effect=lambda request: copy.deepcopy(request)) as dispatch, \
             patch.object(task_model_preflight, 'inspect_task_models', side_effect=mocked_task_capability) as capability, \
             patch('vision_transport._connection', return_value=({'registry_id': 'offline-service-fixture'},
                'https://example.invalid/v1/responses', 'never-sent-test-credential', 'responses')), \
             patch.object(urllib.request, 'urlopen', side_effect=AssertionError('Completion planning requested network')):
            request = backend.complete_source_run('saved', {'algorithm_profile': 'dinov3_vitl16'})
            checked_zero_image_task(request, parent)
            assert parent == original
            assert request['generate'] is False and request['complete_post'] is True
            assert request['source'] == parent['source']['path'] and request['reference'] == parent['reference']['path']
            assert request['parent_run_id'] == 'saved' and request['preserve_project_selection'] is True
            assert 'design_reconstruction_plan' not in request['appearance_revision']
            assert request['appearance_revision']['params'] == parent['options']['appearance_revision']['params']
            assert request['manual_layers'] == parent['options']['manual_layers']
            parsed = project_cli.build_parser().parse_args(backend._one_click_args(request, request['source']))
            assert parsed.complete_post and parsed.generate is False
            assert parsed.algorithm_profile == 'dinov3_vitl16'
            settings = json.loads(parsed.options_json)
            args = workflow.parse_args(['--source', request['source'], '--complete-post'])
            workflow.apply_workflow_settings(args, settings)
            assert args.preserve_project_selection and args.cosplay_correction_budget == 0
            assert args.task_intent == request['task_intent'] and args.task_plan == request['task_plan']
            repaired = backend.repair_final_run('saved', {})
            checked_zero_image_task(repaired, parent)
            assert repaired['final_repair_only'] and repaired['cosplay_correction_budget'] == 0
            assert repaired['generate'] is False and repaired['complete_post'] is True
            local = project_cli.build_parser().parse_args(backend._one_click_args(repaired, repaired['source']))
            assert local.final_repair_only and local.complete_post and local.generate is False
            assert repaired['parent_run_id'] == 'saved' and repaired['preserve_project_selection']
            rejected(lambda: backend.repair_final_run('saved', {'generate': True}))
            saved_parent = copy.deepcopy(parent)
            saved_plan = {'source_sha256': parent['source']['sha256'],
                'candidate_sha256': parent['reference']['sha256'],
                'layers': [{'kind': 'reference_appearance_stack', 'assets': {'frozen': True}}]}
            saved_parent['options']['appearance_revision']['design_reconstruction_plan'] = saved_plan
            with patch.object(workflow, 'load_run', return_value=(root/'manifest.json', saved_parent)):
                same = backend.complete_source_run('saved', {})
                checked_zero_image_task(same, parent)
                assert same['appearance_revision']['design_reconstruction_plan'] == saved_plan
                same['appearance_revision']['design_reconstruction_plan']['layers'].clear()
                assert saved_parent['options']['appearance_revision']['design_reconstruction_plan'] == saved_plan
                changed = backend.complete_source_run('saved', {'model_registry': str(alternative_registry)})
                checked_zero_image_task(changed, parent)
                assert 'design_reconstruction_plan' not in changed['appearance_revision']
                assert changed['task_plan']['plan_id'] != same['task_plan']['plan_id']
                rejected(lambda: backend.complete_source_run('saved', {'model_registry': str(root / 'missing-models.json')}))
            for change in ({'continuation_edit': {'render_state': 'pending'}},
                           {'generation_design': {'status': 'unavailable'}}, {'auto_draft': None}):
                with patch.object(workflow, 'load_run', return_value=(root/'manifest.json', {**parent, **change})):
                    rejected(lambda: backend.complete_source_run('saved', {}))
            for path in (source, reference):
                original_bytes = path.read_bytes()
                try:
                    Image.new('RGB', (32, 24), 'red').save(path)
                    rejected(lambda: backend.complete_source_run('saved', {}))
                finally:
                    path.write_bytes(original_bytes)
            denied = copy.deepcopy(parent)
            denied['task_intent']['services']['text_vision'] = False
            with patch.object(workflow, 'load_run', return_value=(manifest, denied)):
                rejected(lambda: backend.complete_source_run('saved', {}))
            called_before = dispatch.call_count
            with patch.object(task_model_preflight, 'inspect_task_models', return_value={
                    **mocked_task_capability({}, mode='existing_reference'), 'ready': False,
                    'blocking_reasons': ['Explicit fixture model capability failure']}):
                rejected(lambda: backend.complete_source_run('saved', {}))
            assert dispatch.call_count == called_before, 'Blocked capability reached execution'
            assert capability.call_count >= 8
            standard = options_for_choice('current', root)
            assert choice_for_options(standard) == 'current'
            rejected(lambda: options_for_choice('unknown', root))
        assert parent == original
        assert all(path.read_bytes() == content for path, content in protected.items())
    print(json.dumps({'ok': True, 'explicit_completion_cli_and_backend': True,
        'saved_parent_preserved': True, 'changed_analysis_reprepares': True,
        'same_analysis_preserves_saved_layers': True,
        'manual_controls_preserved': True, 'pending_and_incomplete_rejected': True,
        'real_source_reference_and_checked_task_plans': True,
        'saved_source_reference_digest_changes_rejected': True,
        'parent_constraints_preserved': True, 'all_image_budgets_zero': True,
        'model_capability_mocked': True, 'model_acceptance_claimed': False,
        'scene_regeneration_requested': False, 'network_requests': 0}))


if __name__ == '__main__':
    main()
