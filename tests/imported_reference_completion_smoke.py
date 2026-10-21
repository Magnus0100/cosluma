"""Offline imported pair, complete chain and direct zero-image-budget regressions."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import ExitStack

from PIL import Image
import numpy as np
import automatic_foreground_effects as effects
import cosplay_post_completion as scene
import foreground_effects
import generation_design as design
import single_source_workflow as workflow
import system_post_completion as system
from creative_task import normalize_task_intent, image_budget


def brief(intent, *, baked=False):
    result = {key: 'Observed source and reference; retain actual identity and texture.' for key in
        ('summary', 'scene_analysis', 'character_basis', 'subject_retouch_plan', 'spatial_integration_plan',
         'lighting_plan', 'palette_material_plan', 'tradeoffs', 'reference_application', 'english_art_direction')}
    result.update(scene_transformation_plan={'mode': 'refine', 'description': 'Use the observed setting.',
        'visible_changes': ['Finish observed colour and contrast.']},
        primary_effect_plan={'mode': 'none', 'description': 'Clean photographic finish.', 'reason': 'Keep the person readable.'},
        foreground_effect_plan={'enabled': False, 'layers': []}, user_constraints={},
        reference_background_policy={'mode': 'preserve_source' if intent['constraints']['preserve_background'] else 'use_reference',
            'reason': 'Follow the explicit background ownership.'},
        reference_effect_observation={'present': baked, 'description': 'Observed smoke.' if baked else 'No added effects observed.'})
    return result


def main():
    with tempfile.TemporaryDirectory(prefix='imported-complete-') as folder, ExitStack() as isolated:
        isolated.enter_context(patch('runtime_preflight.preflight_for_run',return_value={'status':'ready','CPU_fixture':True}))
        isolated.enter_context(patch.object(workflow,'_prepare_task_portrait_beauty',return_value=None))
        isolated.enter_context(patch.object(workflow,'_prepare_task_boundary_light',return_value=False))
        isolated.enter_context(patch('urllib.request.urlopen',side_effect=AssertionError('Unexpected actual network')))
        root = Path(folder)
        source, reference = root/'source.png', root/'reference.png'
        Image.new('RGB', (20, 24), '#805050').save(source)
        Image.new('RGB', (28, 18), '#506080').save(reference)
        project = root/'project.json'; project.write_text('{"selection":"retained"}')
        hashes = {str(path): workflow.sha256(path) for path in (source, reference, project)}
        intent = normalize_task_intent({'schema_version': 'cosplay_task_intent.v1', 'mode': 'existing_reference',
            'brief': 'Keep this makeup and finish the imported reference.',
            'constraints': {}, 'services': {'text_vision': True,'visual_review':True}})
        plan = {'schema_version': 'cosplay_task_plan.v2', 'plan_id': 'a'*64, 'mode': intent['mode'],
            'intent': intent, 'ready': True, 'image_budget': image_budget(intent),
            'execution_strategy':'layered_reference_transfer','preparation_policy':'replay_frozen',
            'inputs': {role: {'sha256': hashes[str(path)]} for role, path in (('source', source), ('reference', reference))}}
        from cos_node_registry import compile_workflow
        plan['layer_workflow'] = compile_workflow(intent['mode'])
        settings = {'task_intent': intent, 'task_plan': plan}
        calls = []
        def observe(board, output, prompt, **kwargs):
            calls.append('paired_design')
            assert 'SOURCE on the left' in prompt and 'Image generation' in prompt
            with Image.open(board) as raster:
                assert raster.size == (1024, 576)
            return brief(intent), {'network_used': True, 'generation_used': False, 'fixture_only': True}
        with patch('vision_transport.request_vision_json', side_effect=observe) as vision:
            paired = design.prepare_existing_reference_design(source, reference, root/'pair', intent)
            assert paired['status'] == 'ready' and paired['reference']['sha256'] == hashes[str(reference)]
            assert paired['observation_board']['panels']['source']['sha256'] == hashes[str(source)]
            assert paired['image_submissions'] == 0 and paired['text_vision_submissions'] == 1
            assert design.prepare_existing_reference_design(source, reference, root/'pair', intent) == paired
            assert vision.call_count == 1
        changed = copy.deepcopy(intent); changed['brief'] += ' More subdued.'
        assert design.existing_reference_binding(hashes[str(source)], hashes[str(reference)], changed) != paired['binding']
        assert design.existing_reference_binding(hashes[str(source)], 'b'*64, intent) != paired['binding']
        forbidden = copy.deepcopy(intent); forbidden['constraints']['no_effects'] = True
        with patch('vision_transport.request_vision_json', return_value=(brief(forbidden, baked=True), {})):
            conflict = design.prepare_existing_reference_design(source, reference, root/'conflict', forbidden)
        assert conflict['status'] == 'unavailable' and 'baked effects incompatible' in conflict['error']
        preserve = copy.deepcopy(forbidden); preserve['constraints']['preserve_background'] = True
        with patch('vision_transport.request_vision_json', return_value=(brief(preserve, baked=True), {})):
            preserved = design.prepare_existing_reference_design(source, reference, root/'preserve', preserve)
        assert preserved['status'] == 'ready' and preserved['design_brief']['user_constraints']['no_effects']
        assert preserved['design_brief']['user_constraints']['preserve_background']

        args = workflow.parse_args(['--root', str(root), '--source', str(source), '--reference', str(reference), '--run-id', 'imported'])
        workflow.apply_workflow_settings(args, settings)
        args.preserve_project_selection = True
        args.generation_design = paired
        master = root/'final.png'; Image.new('RGB', (20, 24), '#805850').save(master)
        def render(_root, run, _args):
            calls.append('render')
            return {'metadata': {}, 'review': {}, 'outputs': {'auto_draft': str(master),
                'render_master': str(master), 'render_scene_master': str(master)}}
        def retouch(_root, run, _args, **kwargs):
            calls.append('source_retouch'); run['automatic_native_retouch'] = {'status': 'fixture_checked'}
            return {'run': run, 'status': 'fixture_checked'}
        def diagnose(_root, run, _args):
            calls.append('review')
            sha = workflow.sha256(master)
            run['cosplay_post_review'] = {'status': 'reviewed', 'actual_api_evidence_valid': True,
                'diagnostic_gate': 'pass', 'checks': {},
                'binding': {'images': {'final': {'path': str(master), 'sha256': sha}}},
                'run_binding': {'run_id': run['run_id'], 'revision': run['revision'],
                    'source_sha256': run['source']['sha256'], 'reference_sha256': run['reference']['sha256'],
                    'final_sha256': sha, 'render_artifacts': {key: sha for key in run['artifacts']}}}
        repairs = {'tasks': [{'producer': key, 'pixel_repair_qualified': True} for key in
                            ('qualified_material_finish', 'background_consistency', 'native_receiver_detail')]}
        def local(key):
            def complete(_root, run, _args, **kwargs):
                calls.append(key); return {'run': run, 'status': 'fixture_checked'}
            return complete
        with patch.object(workflow, 'render_candidate', side_effect=render), \
             patch.object(workflow, 'diagnose_cosplay_post', side_effect=diagnose), \
             patch('automatic_native_retouch.prepare_once', side_effect=retouch), \
             patch('automatic_material_completion.has_skin_reference_failure', return_value=True), \
             patch('automatic_material_completion.has_material_failure', return_value=True), \
             patch('automatic_material_completion.complete_materials_once', side_effect=local('material')), \
             patch('automatic_scene_completion.complete_scene_once', side_effect=local('background')), \
             patch('automatic_receiver_detail_completion.complete_receiver_detail_once', side_effect=local('receiver')), \
             patch('cosplay_repair_plan.plan_final_repairs', return_value=repairs), \
             patch('reference_provider.main', side_effect=AssertionError('Reference image submitted')), \
             patch('reference_refinement.refine_reference_once', side_effect=AssertionError('Scene image submitted')), \
             patch.object(effects, 'produce_request', side_effect=AssertionError('RGBA image submitted')):
            saved = workflow.create_run(args)
            assert saved['software_version'] == workflow.SOFTWARE_VERSION and saved['status'] == 'needs_visual_review', saved.get('error')
        receipt = saved['system_post_completion']
        assert receipt['all_stages_checked_once'] and receipt['reference_image_budget'] == receipt['independent_effect_image_budget'] == 0
        assert saved['task_intent'] == saved['options']['task_intent'] == intent
        assert 'source_retouch' in calls and 'background' in calls and 'receiver' in calls and 'material' in calls
        assert saved['generation_budget']['total_image_submissions_including_effects'] == 0
        assert system._eligible(saved, args)
        assert scene.continue_cosplay_post(root, saved, args) is saved
        saved['options']['appearance_revision'] = {'design_reconstruction_plan': {'enabled': True, 'layers': []}}
        with patch.object(effects, 'declaration', return_value=([{'id': 'missing-accent'}], 'required')):
            result = effects.complete_effects_once(root, saved, args, fresh_generation=True)
        assert result['status'] == 'blocked_by_image_budget' and result['unfulfilled_effect_ids'] == ['missing-accent']
        with patch.object(foreground_effects, 'declaration', return_value=([{'id': 'missing-accent'}], 'required')), \
             patch.object(foreground_effects, 'generation_request', side_effect=AssertionError('Runtime/request prepared')):
            result = foreground_effects.prepare_effects(root, saved, {'layers': []}, generate=True)
        assert result['status'] == 'blocked_by_image_budget' and result['requests'] == []
        blocked_request = root/'blocked-effect-request.json'
        blocked_request.write_text(json.dumps({'task_image_policy': {'mode': 'existing_reference', 'independent_effect': 0}}))
        try: foreground_effects.produce_request(blocked_request)
        except ValueError as exc: assert 'forbids independent effect' in str(exc)
        else: raise AssertionError('Direct frozen effect producer ignored zero image policy')
        policy_run = {**saved, 'task_intent': preserve}
        native_plan = {'layers': [{'id': key, 'kind': key, 'enabled': True} for key in
            ('background_design', 'contact_shadow', 'independent_effect', 'reference_appearance_stack')]}
        frozen = workflow.apply_task_render_constraints(native_plan, policy_run)
        assert [row['enabled'] for row in frozen['layers']] == [False, False, False, True]
        assert native_plan['layers'][0]['enabled'] is True
        # Verify the ownership policy at actual full and ROI pixel endpoints.
        from automatic_foreground_effects_smoke import fixture as native_fixture
        from native_source_baseline import source_for_plan
        from design_reconstruction import apply_plan
        from native_roi_render import render_native_roi
        native_run, _, _, _ = native_fixture(root/'native-preserve-pixels')
        native_run['task_intent'] = preserve
        pixel_plan = native_run['options']['appearance_revision']['design_reconstruction_plan']
        constrained = workflow.apply_task_render_constraints(pixel_plan, native_run)
        native_run['options']['appearance_revision']['design_reconstruction_plan'] = constrained
        size = tuple(constrained['canvas']['source_oriented_size'])
        original = source_for_plan(constrained, size)
        final, _ = apply_plan(original, original.copy(), constrained, Image.new('L', size, 255), run=native_run)
        np.testing.assert_array_equal(final, original)
        box = (0, 70, 50, 125)
        roi, _ = render_native_roi(native_run, box)
        np.testing.assert_array_equal(roi, final[70:125, 0:50])
        offline = copy.deepcopy(intent); offline['services']['text_vision'] = False
        denied_args = copy.copy(args); denied_args.task_intent = offline
        denied_args.task_plan = {**plan, 'intent': offline}
        with patch('vision_transport.request_vision_json', side_effect=AssertionError('Forbidden service request')):
            try: workflow.create_run(denied_args)
            except ValueError as exc: assert 'text/vision' in str(exc)
            else: raise AssertionError('Complete reference task ran without service permission')
        assert all(workflow.sha256(Path(path)) == value for path, value in hashes.items())
    print(json.dumps({'ok': True, 'paired_single_request_binding': True, 'baked_effect_conflict_explicit': True,
        'full_existing_reference_chain': True, 'direct_image_producers_blocked': True,
        'qualified_local_stages_checked': True, 'explicit_source_background_ownership': True,
        'source_reference_project_preserved': True, 'actual_api_requests': 0, 'model_inferences': 0}))


if __name__ == '__main__':
    main()
