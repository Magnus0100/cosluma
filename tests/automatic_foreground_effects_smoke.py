"""CPU image/vision mocks, real RGBA contracts, previews, ownership and child save.

No external request, GPU or AI effect is made. The synthetic raster and mocked
text verdict verify bounded orchestration; they are not visual quality evidence.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch

import numpy as np
from PIL import Image

import automatic_foreground_effects as automatic
import automatic_foreground_effects_review as visual
from cosplay_post_review_smoke import fake_transport
from foreground_effects import (KIND, asset, declaration, digest, generation_request,
    identity, prepare_effects, produce_request, read, RECIPE, validate_layer, write)
from effect_conditioning import compile_condition, prompt_from_condition, validate_condition
from generation_design import DESIGN_VERSION
import single_source_workflow as workflow


def response(status='pass'):
    return {'schema': visual.OBSERVATION_SCHEMA, 'checks': {name: {
        'status': status, 'reason': 'CPU mocked visual response; no actual API or AI quality claim.',
        'evidence': [] if status == 'unknown' else [{'basis': 'visible_pixels',
            'panels': ['composite_after', 'isolated_rgba', 'actual_alpha', 'consumed_alpha'],
            'observation': 'The known synthetic smoke raster changes the lower scene region.'}]}
        for name in visual.CHECKS}}


def fixture(root, *, multiple=False, baked=False, missing_semantics=False, unprotected=False):
    root.mkdir(parents=True)
    size = [96, 144]
    w, h = size
    inputs = root/'inputs'
    inputs.mkdir()
    workflow_path = root/'providers/workflows/qwen-image-2.1-effect-rgba-text-only.json'
    workflow_path.parent.mkdir(parents=True)
    shutil.copyfile(Path(__file__).resolve().parents[1]/'providers/workflows/qwen-image-2.1-effect-rgba-text-only.json', workflow_path)
    node_target = root/'providers/comfyui/cosplay_effect_canvas.py'
    node_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__).resolve().parents[1]/'providers/comfyui/cosplay_effect_canvas.py', node_target)
    source = inputs/'source.png'
    reference = inputs/'reference.png'
    Image.new('RGB', size, '#726660').save(source)
    Image.new('RGB', size, '#656053').save(reference)
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[84:122, 13:84] = [128, 90, 40, 110]
    rgba_path = inputs/'synthetic-raster-not-AI-generated.png'
    Image.fromarray(rgba).save(rgba_path)
    alpha = np.zeros((h, w), np.float32)
    alpha[22:122, 29:68] = 1
    labels = np.zeros((h, w), np.uint8)
    labels[28:51, 36:61] = 3
    labels[82:95, 29:41] = 6
    labels[82:95, 55:67] = 15

    def field(name, values, key='values'):
        path = inputs/(name+'.npz')
        np.savez_compressed(path, **{key: values})
        return asset(path, size=size, coordinate_space='source_canvas', color_space='linear_sRGB')

    fp = np.full((h, w, 3), .15, np.float32)*alpha[..., None]
    fp_asset, alpha_asset, old_asset = field('fp', fp), field('alpha', alpha), field('old', np.full_like(fp, .1))
    background = {'id': 'background_design', 'kind': 'background_design', 'enabled': True, 'strength': 1.,
        'assets': {'foreground_premultiplied': fp_asset, 'old_environment': old_asset,
            'background': field('background', np.full_like(fp, .08)), 'validity': field('validity', np.ones((h, w), np.float32))}}
    background['assets']['background'].update(representation='sampling_validity_associated_RGB',validity_associated=True)
    plan = {'schema_version': 'design_reconstruction_plan.v1', 'working_domain':'display_linear_sRGB', 'enabled': True,
        'source_sha256': digest(source), 'candidate_sha256': digest(reference),
        'canvas': {'source_oriented_size': size, 'source_proxy_size': size, 'candidate_mapping': {
            'candidate_original_to_final_original': np.eye(3).tolist(), 'scene_registration_supported': True}},
        'layers': [background], 'mask_roles': {'foreground_alpha': alpha_asset},
        'source_baseline': {'schema':'appearance_source_baseline.v1', 'working_domain':'display_linear_sRGB',
            'source_sha256': digest(source), 'source_precision': 'raster8_srgb',
            'decode_version': __import__('raw_support').RGB_DECODE_VERSION,
            'source_display': __import__('display_transform').source_transform_for_precision('raster8_srgb'),
            'source_display_applied_once': True,
            'assets': {'foreground_premultiplied': fp_asset, 'foreground_alpha': alpha_asset, 'old_environment': old_asset}},
        'execution': {'source_reference': 'paired_native_baseline'}, 'provenance': {}}
    if not missing_semantics:
        plan['reconstruction_observations'] = {'source_semantics': field('labels', labels, key='labels')}
    specs = [{'id': 'front_smoke', 'slot': 'front', 'operation': 'over',
        'description': 'CPU fixture only: one thin newly requested foreground wisp below the waist.',
        'effect_design': {'material':'scattering_smoke','colour_rgb8':[255,181,82],
                          'extent_2d':[0,.6,1,1],'density':'sparse'},
        'occlusion': 'in_front' if unprotected else 'protect_face_hands', 'increment': True, 'baked_in_reference': False}]
    if multiple:
        specs.append({**specs[0], 'id': 'front_particles', 'operation': 'emission_add',
                      'effect_design': {**specs[0]['effect_design'],'material':'luminous_particles'}})
    brief = {'primary_effect_plan': {'mode': 'scene_effect', 'description': 'Fixture only'},
        'foreground_effect_plan': {'enabled': True, 'layers': specs},
        'spatial_integration_plan': 'Fixture: thin foreground smoke below waist.'}
    if baked:
        brief.pop('foreground_effect_plan')
        brief['spatial_integration_plan'] = 'Keep foreground floor unobscured; smoke is behind her.'
    directory = root/workflow.RUNS_DIR/'initial'
    directory.mkdir(parents=True)
    artifacts = {}
    for name in ('auto_draft', 'render_master', 'render_scene_master'):
        path = directory/(name+'.png')
        Image.new('RGB', size, '#857467').save(path)
        artifacts[name] = str(path)
    run = {'root': str(root), 'kind': 'single_input_run', 'input_mode': 'single_source',
        'run_id': 'initial', 'sample': 'fixture', 'revision': 1, 'status': 'needs_visual_review',
        'source': {**asset(source), 'oriented_size': size}, 'reference': asset(reference),
        'artifacts': artifacts, 'auto_draft': {'outputs': artifacts, 'metadata': {}, 'review': {'status': 'not_checked'}},
        'generation_design': {'schema_version': DESIGN_VERSION, 'status': 'ready', 'design_brief': brief},
        'generation_budget': {'image_submissions': 1, 'total_image_submissions': 1},
        'options': {'render_mode': 'native_source_design', 'appearance_revision': {'design_reconstruction_plan': plan}}}
    write(directory/'manifest.json', run)
    write(root/'project.json', {'state': {'selected_run_id': 'preserve-previous'}})
    args = workflow.parse_args(['--root', str(root), '--source', str(source), '--generate'])
    args.provider, args.vision_model = 'cpu_mock', 'mock_vision'
    preserved = {str(path): digest(path) for path in
        (directory/'manifest.json', source, reference, root/'project.json', *map(Path, artifacts.values()))}
    return run, args, rgba_path, preserved


def main():
    checks = []
    with tempfile.TemporaryDirectory(prefix='automatic-effects-cpu-') as name:
        parent = Path(name)

        def case(label, *, status='pass', multiple=False, baked=False, missing_semantics=False,
                 unprotected=False, transport_error=False, bad_transport=False, qa_error=False,
                 render_error=False, cancel_before=False, cancel_after=False, tamper_response=False,
                 tamper_panel=False, fresh=True, pixel_condition=False, photo_graph=False,
                 wrong_canvas=False, stale_runtime=False, wrong_latent=False, role_graph=False):
            root = parent/label
            run, args, rgba_path, preserved = fixture(root, multiple=multiple, baked=baked,
                missing_semantics=missing_semantics, unprotected=unprotected)
            seen = {'images': 0, 'vision': 0, 'render': 0}
            if cancel_before:
                args.cancel_request_path = root/'cancel-request.json'
                args.cancel_request_path.write_text('{}', encoding='utf8')

            def producer(request_path):
                seen['images'] += 1
                request = read(request_path)
                if transport_error:
                    raise OSError('CPU mock submitted transport failed before view')
                generated, sidecar = Path(request['generated_image_path']), Path(request['generated_provenance_path'])
                generated.parent.mkdir(parents=True)
                shutil.copyfile(rgba_path, generated)
                actual = generated.parent/'actual-api-condition.png'
                shutil.copyfile(request['condition']['path'], actual)
                if pixel_condition:
                    Image.new('RGBA', Image.open(actual).size, (1, 0, 0, 0)).save(actual)
                write(request['runtime_receipt_path'], {
                    'loaded_node_sha256': 'wrong-loaded-code' if stale_runtime else request['canvas_node']['sha256'],
                    'loaded_node_verified_via': 'actual_service_object_info', 'test_scope': 'CPU mock only'})
                from comfyui_adapter import prepare_workflow
                submitted_graph = prepare_workflow(read(request['workflow']['path']), 'empty-condition.png', request['prompt'])
                if photo_graph:
                    next(row for row in submitted_graph.values() if row['class_type'] == 'TextEncodeQwenImage21')['inputs']['images.image_1'] = ['4', 0]
                if role_graph:
                    next(row for row in submitted_graph.values() if row['class_type'] == 'TextEncodeQwenImage21')['inputs']['prompt'] += ' A portrait of a woman.'
                if wrong_canvas:
                    next(row for row in submitted_graph.values() if row['class_type'] == 'CosplayEmptyQwen21Canvas')['inputs']['width'] += 32
                if wrong_latent:
                    next(row for row in submitted_graph.values() if row['class_type'] == 'KSampler')['inputs']['latent_image'] = ['5', 2]
                write(sidecar, {'status': 'reference_candidate', 'asset_role': 'effect_layer', 'alpha': {'preserved': True},
                    'source': request['scene_context'], 'asset': asset(generated), 'prompt': request['prompt'],
                    'generation_id': 'cpu-mock-generation', 'provider_contract': {
                        'request': {'prompt': request['prompt'], 'input_sha256': digest(actual), 'api_input': {
                            **asset(actual), 'source': request['condition'], 'derived': False}}, 'response': {'id': 'cpu-mock-generation'},
                        'transport': {'events': [{'event': 'comfyui_submitted'}]*(2 if bad_transport else 1)},
                        'comfyui': {'workflow_sha256': request['workflow']['sha256'], 'request': {'prompt': submitted_graph}}}})
                if cancel_after:
                    args.cancel_request_path = root/'cancel-request.json'
                    args.cancel_request_path.write_text('{}', encoding='utf8')
                return 0

            def text_transport(source, output, prompt, **kwargs):
                seen['vision'] += 1
                if qa_error:
                    raise OSError('CPU mock text service unavailable')
                return fake_transport(response(status))(source, output, prompt, **kwargs)

            def render(_root, child, _args):
                seen['render'] += 1
                assert child['parent_run_id'] == 'initial' and child['options']['preserve_project_selection'] is True
                assert child['continuation_edit']['render_state'] == 'pending' and not child.get('auto_draft')
                current = child['options']['appearance_revision']['design_reconstruction_plan']
                effects = [row for row in current['layers'] if row.get('kind') == KIND]
                assert len(effects) == 1 and effects[0]['enabled'] is True
                validate_layer(effects[0], child, current)
                for row in effects[0]['assets'].values():
                    assert str(child['run_id']) in row['path'], row['path']
                if render_error:
                    raise OSError('CPU mock native render interrupted')
                outputs = {}
                folder = root/workflow.RUNS_DIR/child['run_id']
                for artifact in ('auto_draft', 'render_master', 'render_scene_master'):
                    path = folder/(artifact+'.png')
                    Image.new('RGB', (96, 144), '#857462').save(path)
                    outputs[artifact] = str(path)
                return {'outputs': outputs, 'metadata': {}, 'review': {'status': 'not_checked'}}

            with patch.object(automatic, 'produce_request', side_effect=producer), \
                 patch.object(visual.vision_transport, 'request_vision_json', side_effect=text_transport), \
                 patch.object(workflow, 'render_candidate', side_effect=render), \
                 patch('reference_pipeline.prepare', side_effect=AssertionError('must not refit T/B/S')):
                result = automatic.complete_effects_once(root, copy.deepcopy(run), args, fresh_generation=fresh)
                again = automatic.complete_effects_once(root, copy.deepcopy(run), args, fresh_generation=fresh)
                assert seen['images'] <= 1 and seen['vision'] <= 1 and seen['render'] <= 1
                assert again['run']['run_id'] == 'initial'
            for path, sha in preserved.items():
                assert digest(path) == sha, path
            if result['run']['run_id'] != 'initial':
                child = result['run']
                assert child['continuation_edit']['render_state'] == 'saved'
                assert child['cosplay_post_review']['actual_api_evidence_valid'] is False
                assert child['automatic_foreground_effects']['human_final_acceptance'] is False
                assert child['generation_budget']['independent_effect_image_submissions'] == 1
                current = child['options']['appearance_revision']['design_reconstruction_plan']
                effect = next(row for row in current['layers'] if row.get('kind') == KIND)
                contract = read(effect['assets']['contract']['path'])
                review = contract['visual_review']
                if tamper_response:
                    path = Path(review['evidence']['response']['path'])
                    changed = read(path)
                    changed['checks']['scene_integration']['status'] = 'fail'
                    write(path, changed)
                if tamper_panel:
                    packet = read(review['evidence']['packet']['path'])
                    Image.new('RGB', (4, 4), '#ffffff').save(packet['panels'][3]['asset']['path'])
                if tamper_response or tamper_panel:
                    try:
                        validate_layer(effect, child, current)
                    except ValueError:
                        checks.append(label+'-changed-actual-evidence-rejected')
                    else:
                        raise AssertionError('Edited actual evidence was accepted after child ownership')
            return result, seen, root

        result, seen, _ = case('one-accepted', multiple=True)
        assert result['status'] == 'independent_effect_child_saved' and seen == {'images': 1, 'vision': 1, 'render': 1}, (seen, result['run'].get('automatic_foreground_effects'))
        assert result['run']['automatic_foreground_effects']['deferred_effect_ids'] == ['front_particles']
        checks.append('one-required-RGBA-plus-one-QA-owned-child-extra-layer-deferred')
        for status in ('fail', 'unknown'):
            result, seen, _ = case('qa-'+status, status=status)
            assert result['status'] == 'effect_visual_'+status and seen == {'images': 1, 'vision': 1, 'render': 0}
            checks.append(status+'-keeps-parent-draft-no-enabled-layer')
        for label, options in [('not-fresh', {'fresh': False}), ('baked', {'baked': True}),
                                ('missing-semantic', {'missing_semantics': True}),
                                ('unprotected', {'unprotected': True}), ('cancel-before', {'cancel_before': True})]:
            result, seen, _ = case(label, **options)
            assert seen == {'images': 0, 'vision': 0, 'render': 0}
            checks.append(label+'-zero-image-and-QA')
        for label, options, expected in [
            ('transport-fail', {'transport_error': True}, {'images': 1, 'vision': 0, 'render': 0}),
            ('duplicate-submission', {'bad_transport': True}, {'images': 1, 'vision': 0, 'render': 0}),
            ('cancel-after', {'cancel_after': True}, {'images': 1, 'vision': 0, 'render': 0}),
            ('qa-failure', {'qa_error': True}, {'images': 1, 'vision': 1, 'render': 0}),
            ('render-failure', {'render_error': True}, {'images': 1, 'vision': 1, 'render': 1})]:
            result, seen, root = case(label, **options)
            assert seen == expected and result['run']['run_id'] == 'initial'
            assert result['run']['automatic_foreground_effects']['independent_effect_image_submissions'] == 1
            checks.append(label+'-spent-budget-marker-parent-preserved')
            if label == 'render-failure':
                pending = [read(path) for path in (root/workflow.RUNS_DIR).glob('*/manifest.json') if path.parent.name != 'initial']
                assert len(pending) == 1 and not pending[0].get('auto_draft')
                assert pending[0]['continuation_edit']['render_state'] == 'interrupted'
                checks.append('interrupted-child-no-inherited-exportable-master')
        for label, options in [('photo-API-pixels', {'pixel_condition': True}),
                               ('photo-graph-condition', {'photo_graph': True}),
                               ('wrong-native-canvas', {'wrong_canvas': True}),
                               ('stale-loaded-canvas', {'stale_runtime': True}),
                               ('unused-empty-canvas-photo-latent', {'wrong_latent': True}),
                               ('role-prompt-in-actual-graph', {'role_graph': True})]:
            result, seen, _ = case(label, **options)
            assert seen == {'images': 1, 'vision': 0, 'render': 0} and result['run']['run_id'] == 'initial'
            checks.append(label+'-refused-before-QA-and-child')
        case('tamper-response', tamper_response=True)
        case('tamper-visible-panel', tamper_panel=True)
        root = parent/'frozen-native-implementation'
        run, _, _, _ = fixture(root)
        plan = run['options']['appearance_revision']['design_reconstruction_plan']
        request = prepare_effects(root, run, plan)['requests'][0]
        node = root/'providers/comfyui/cosplay_effect_canvas.py'
        node.write_bytes(node.read_bytes()+b'\n# Deliberate CPU version-change fixture.\n')
        with patch('foreground_effects_runtime.ensure_effect_runtime', side_effect=AssertionError('must fail before service startup')), \
             patch('foreground_effects.resource_path', return_value=node):
            try:
                produce_request(request['request_path'])
            except ValueError:
                checks.append('changed-bundled-canvas-refused-before-service')
            else:
                raise AssertionError('Changed service implementation was accepted')
        spec = declaration(run, plan)[0][0]
        from installed_resources import resource_path
        original_resource_path = resource_path
        def updated_resource_path(relative):
            return node if relative.endswith('cosplay_effect_canvas.py') else original_resource_path(relative)
        with patch('foreground_effects.resource_path', side_effect=updated_resource_path):
            updated = generation_request(root, run, plan, spec, root/'updated-request')
        assert updated['identity'] != request['identity']
        checks.append('canvas-code-version-is-part-of-request-identity')
        # A saved/new project's data directory never needs source templates.
        providers = (root/'providers').resolve()
        assert providers.is_relative_to(root.resolve())
        shutil.rmtree(providers)
        detached = generation_request(root, run, plan, spec, root/'installed-request')
        assert detached['canvas_node']['sha256'] == digest(original_resource_path('providers/comfyui/cosplay_effect_canvas.py'))
        assert not (root/'providers').exists()
        checks.append('source-install-resources-independent-of-project-data')
        shipped = parent/'frozen-installation'
        for relative in ('providers/comfyui/cosplay_effect_canvas.py',
                         'providers/workflows/qwen-image-2.1-effect-rgba-text-only.json'):
            target = shipped/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original_resource_path(relative), target)
        from effect_code_identity import FILES
        from runtime_resources import code_path
        for name in FILES:
            target = shipped/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(code_path(name), target)
        import installed_resources
        with patch.object(installed_resources.sys, 'frozen', True, create=True), \
             patch.object(installed_resources.sys, '_MEIPASS', str(shipped), create=True):
            frozen_request = generation_request(root, run, plan, spec, root/'frozen-installed-request')
        assert frozen_request['canvas_node']['sha256'] == detached['canvas_node']['sha256']
        assert frozen_request['prompt'] == detached['prompt']
        checks.append('frozen-install-resources-independent-of-project-data')
        photo_workflow = read(request['workflow']['path'])
        photo_workflow['asset_contract']['method'] = 'photo_edit_reference_v1'
        photo_path = root/'old-photo-workflow.json'
        write(photo_path, photo_workflow)
        try:
            generation_request(root, run, plan, spec, root/'refused-photo', workflow_path=photo_path)
        except ValueError:
            checks.append('legacy-photo-effect-workflow-refused-before-service')
        else:
            raise AssertionError('Legacy photo extraction workflow was accepted')
        root = parent/'effect-semantic-isolation'
        run, _, _, _ = fixture(root)
        plan = run['options']['appearance_revision']['design_reconstruction_plan']
        spec = declaration(run, plan)[0][0]
        clean = generation_request(root, run, plan, spec, root/'clean-request')
        polluted = copy.deepcopy(run)
        polluted['generation_design']['design_brief'].update(
            lighting_plan='Photograph of a woman: face, eyes, mouth, hands, hair, skin highlights.',
            palette_material_plan='Slate-blue costume, charcoal hair, brown leather handbag, ivory shoes.')
        polluted['generation_design']['design_brief']['foreground_effect_plan']['layers'][0]['description'] += \
            ' Preserve woman, person, portrait, face, eyes, mouth, skin, costume, hair, handbag and shoes.'
        polluted_spec = declaration(polluted, plan)[0][0]
        isolated = generation_request(root, polluted, plan, polluted_spec, root/'isolated-request')
        assert clean['prompt'] == isolated['prompt']
        import re
        assert not re.search(r'\b(person|portrait|face|eyes|mouth|hands?|costume|cloth|hair|handbag|bag|shoes?|skin|woman|human|dress|leather|iris|sclera)\b', isolated['prompt'], re.I)
        assert 'woman' in isolated['automatic_context']['lighting_intent_QA_only']
        checks.append('full-role-lighting-material-prose-and-negative-role-nouns-never-reach-model-text')
        for label, mutate in [
            ('role-key', lambda c: c.update(role='woman')),
            ('role-material', lambda c: c.update(material='portrait')),
            ('role-colour', lambda c: c.update(colour_rgb8='blue costume')),
            ('role-ambient', lambda c: c['ambient'].update(lighting_plan='face light')),
            ('nonfinite-extent', lambda c: c.update(extent_2d=[0, 0, float('nan'), 1])),
            ('empty-extent', lambda c: c.update(extent_2d=[.5, 0, .4, 1]))]:
            invalid = copy.deepcopy(clean['effect_condition'])
            mutate(invalid)
            try:
                validate_condition(invalid)
            except ValueError:
                checks.append(label+'-whitelist-refused')
            else:
                raise AssertionError('Polluted/invalid effect schema was accepted')
        corrupted = copy.deepcopy(clean)
        corrupted['prompt'] += ' A woman in blue costume.'
        corrupted['identity'] = identity({'binding': corrupted['binding'], 'specification': corrupted['specification'],
            'workflow_sha256': corrupted['workflow']['sha256'], 'canvas_node_sha256': corrupted['canvas_node']['sha256'],
            'effect_condition': corrupted['effect_condition'], 'fx_code_identity': corrupted['fx_code_identity'],
            'prompt': corrupted['prompt'], 'recipe': RECIPE})
        path = root/'corrupted-request.json'
        write(path, corrupted)
        with patch('foreground_effects_runtime.ensure_effect_runtime', side_effect=AssertionError('must fail before service startup')):
            try:
                produce_request(path)
            except ValueError:
                checks.append('role-text-with-valid-request-identity-refused-before-service')
            else:
                raise AssertionError('Positive prompt role text was accepted with recomputed identity')
        emission = compile_condition({**spec, 'operation': 'emission_add',
            'effect_design':{**spec['effect_design'],'material':'luminous_particles'}}, plan)
        assert emission['material'] == 'luminous_particles' and 'smoke wisp' not in prompt_from_condition(emission)
        checks.append('emission-uses-whitelisted-radiance-material')
        from effect_layout import bound_dose, layout_transform, PARAMETERS
        from design_scene_assets import warp_values
        alpha = np.zeros((144, 96), np.float32)
        alpha[10:90, 25:70] = .8
        matrix, visibility, bbox = layout_transform(alpha, [0, .6, .48, 1], [96, 144])
        assert bbox == [25, 10, 70, 90] and matrix[0, 0] == matrix[1, 1]
        raw = np.dstack([np.repeat((alpha*.7)[..., None], 3, axis=-1), alpha])
        mapped, _ = warp_values(raw, matrix, [96, 144])
        mapped *= visibility[..., None]
        assert np.max(mapped[visibility == 0]) == 0
        checks.append('alpha-support-uniform-fit-and-hard-frozen-extent')
        dosed, dose = bound_dose(mapped, np.full((144, 96), .03, np.float32), visibility)
        assert dose['RGB_appearance_multiplier'] > 0 and dose['alpha_density_multiplier'] > 0
        assert abs(dose['actual_effect_alpha_q95']-PARAMETERS['density_presets']['scattering_smoke'][0]) < 1e-6
        assert dose['actual_effect_alpha_max'] <= PARAMETERS['density_presets']['scattering_smoke'][1]+1e-7
        assert abs(dose['actual_material_luminance_q95']-dose['target_material_luminance_q95']) < 1e-6
        darker, dark_dose = bound_dose(mapped, np.full((144, 96), .003, np.float32), visibility)
        np.testing.assert_array_equal(dosed[..., 3], darker[..., 3])
        assert dark_dose['RGB_appearance_multiplier'] < dose['RGB_appearance_multiplier']
        checks.append('RGB-follows-local-light-alpha-independent-fixed-material-transparency')
        try:
            bound_dose(mapped, np.zeros((144, 96), np.float32), visibility)
        except ValueError:
            checks.append('unlit-extent-refuses-invented-emitted-response')
        else:
            raise AssertionError('Unlit effect extent received an automatic emitted response')
        assert 'effect_layout.py' in clean['fx_code_identity']['files']
        assert clean['fx_code_identity']['files']['effect_layout.py'] == digest(Path(__file__).resolve().parents[1]/'scripts/effect_layout.py')
        checks.append('FX-request-code-identity-includes-current-layout-implementation')
        import effect_code_identity
        changed = copy.deepcopy(clean['fx_code_identity'])
        changed['files']['effect_layout.py'] = '0'*64
        with patch.object(effect_code_identity, 'current_code_identity', return_value=changed), \
             patch('foreground_effects_runtime.ensure_effect_runtime', side_effect=AssertionError('must fail before service startup')):
            try:
                produce_request(root/'clean-request/generation-request.json')
            except ValueError:
                checks.append('FX-layout-code-change-refuses-frozen-request-before-service')
            else:
                raise AssertionError('Changed effect layout implementation was accepted')
        accepted_result, _, _ = case('consumer-code')
        accepted_run = accepted_result['run']
        assert accepted_result['status'] == 'independent_effect_child_saved', accepted_run.get('automatic_foreground_effects')
        accepted_plan = accepted_run['options']['appearance_revision']['design_reconstruction_plan']
        accepted_layer = next(row for row in accepted_plan['layers'] if row.get('kind') == KIND)
        with patch.object(effect_code_identity, 'current_code_identity', return_value=changed):
            try:
                validate_layer(accepted_layer, accepted_run, accepted_plan)
            except ValueError:
                checks.append('current-consumer-refuses-outdated-FX-producer-code')
            else:
                raise AssertionError('Accepted consumer ignored changed FX code')
        from effect_sampling import sample_frozen_effect
        rng = np.random.default_rng(31)
        data = rng.random((67, 43), dtype=np.float32)
        path = root/'noninteger-grid.npz'
        np.savez_compressed(path, alpha=data)
        sampling_asset = asset(path)
        full = sample_frozen_effect(sampling_asset, 'alpha', (0, 0, 333, 501), (333, 501))
        for box in [(0, 0, 21, 31), (57, 211, 189, 383), (300, 470, 333, 501)]:
            roi = sample_frozen_effect(sampling_asset, 'alpha', box, (333, 501))
            np.testing.assert_array_equal(roi, full[box[1]:box[3], box[0]:box[2]])
        checks.append('noninteger-native-global-grid-full-ROI-bitwise-equal')
    print(json.dumps({'ok': True, 'checks_passed': len(checks), 'checks': checks,
        'scope': 'CPU synthetic RGBA, mock image/text transport and native final output; real preview compositor/ownership/staging/save',
        'external_image_generation': 0, 'external_text_requests': 0, 'GPU_inference': 0}, indent=2))


if __name__ == '__main__':
    main()
