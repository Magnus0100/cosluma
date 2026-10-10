"""Real RGBA raster fixture: artistic slot algebra, native ROI and frozen portable replay."""
from __future__ import annotations

import copy
import argparse
import json
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch

import numpy as np
from PIL import Image

from foreground_effects import (KIND, accept_layer, apply_slot, asset, bind_result,
                               binding, declaration, decode_rgba, digest, prepare_effects, read, validate_layer, write)
import automatic_foreground_effects_review as effect_review
from cosplay_post_review_smoke import fake_transport


def main(output_dir=None):
    from design_reconstruction import apply_plan, materialize_plan, sample_asset
    from native_roi_render import render_native_roi
    from native_source_baseline import source_for_plan
    from portable_project import export_project, import_project

    with tempfile.TemporaryDirectory(prefix='cosplay_effect_slots_') as temp:
        root = Path(temp)
        inputs = root/'inputs'; inputs.mkdir()
        width, height = 96, 144
        size = [width, height]
        rgba = np.zeros((height, width, 4), np.uint8)
        rgba[..., :3] = [255, 255, 255]  # invisible RGB must not bleed after resampling
        rgba[16:120, 12:84, :3] = [190, 70, 30]
        rgba[16:48, 12:84, 3] = 64
        rgba[48:96, 12:84, 3] = 128
        rgba[96:120, 12:84, 3] = 192
        rgba_path = inputs/'synthetic-effect-not-AI-generated.png'
        Image.fromarray(rgba).save(rgba_path)
        p, a = decode_rgba(rgba_path)
        expected_rgb = ((np.array([190, 70, 30], np.float32)/255+.055)/1.055)**2.4
        np.testing.assert_allclose(p[60, 30], expected_rgb*(128/255), rtol=0, atol=1e-7)
        assert np.max(p[a == 0]) == 0
        opaque = inputs/'opaque-rgba.png'; Image.new('RGBA', size, (80, 90, 100, 255)).save(opaque)
        try: decode_rgba(opaque)
        except ValueError: pass
        else: raise AssertionError('Opaque effect was accepted')
        alpha = np.zeros((height, width), np.float32)
        alpha[30:118, 26:70] = 1
        alpha[25:30, 26:70] = .3
        fp = np.zeros((height, width, 3), np.float32)+np.array([.12, .2, .3], np.float32)
        fp *= alpha[..., None]
        old = np.full_like(fp, .08)
        target = np.full_like(fp, .18)

        def field(name, values, key='values'):
            path = inputs/(name+'.npz')
            np.savez_compressed(path, **{key: values})
            return asset(path, size=size, coordinate_space='source_canvas', color_space='linear_sRGB')

        source_path = inputs/'source.png'; Image.new('RGB', size, (40, 80, 120)).save(source_path)
        reference_path = inputs/'reference.png'; Image.new('RGB', size, (70, 60, 50)).save(reference_path)
        labels = np.zeros((height, width), np.uint8)
        labels[30:50, 35:61] = 3
        labels[65:80, 34:45] = 6
        labels[65:80, 51:62] = 15
        semantics = field('source-semantics', labels, key='labels')
        alpha_asset, fp_asset, old_asset = field('alpha', alpha), field('fp', fp), field('old', old)
        background = {'id': 'background_design', 'kind': 'background_design', 'enabled': True,
                      'strength': 1., 'assets': {'foreground_premultiplied': fp_asset, 'old_environment': old_asset,
                                               'background': field('background', target),
                                               'validity': field('validity', np.ones(alpha.shape, np.float32))}}
        background['assets']['background'].update(representation='sampling_validity_associated_RGB',validity_associated=True)
        plan = {'schema_version': 'design_reconstruction_plan.v1', 'working_domain':'display_linear_sRGB', 'enabled': True,
                'source_sha256': digest(source_path), 'candidate_sha256': digest(reference_path),
                'canvas': {'source_oriented_size': size, 'candidate_mapping': {
                    'candidate_original_to_final_original': np.eye(3).tolist(), 'scene_registration_supported': True}},
                'layers': [background], 'mask_roles': {'foreground_alpha': alpha_asset},
                'reconstruction_observations': {'source_semantics': semantics},
                'source_baseline': {'schema':'appearance_source_baseline.v1', 'working_domain':'display_linear_sRGB',
                                   'source_display_applied_once': True, 'assets': {'foreground_premultiplied': fp_asset,
                                             'foreground_alpha': alpha_asset, 'old_environment': old_asset}},
                'execution': {'source_reference': 'paired_native_baseline'}, 'provenance': {}}
        specs = [{'id': 'back_smoke', 'slot': 'behind', 'operation': 'over', 'occlusion': 'source_person'},
                 {'id': 'front_smoke', 'slot': 'front', 'operation': 'over', 'occlusion': 'protect_face_hands'},
                 {'id': 'front_light', 'slot': 'front', 'operation': 'emission_add', 'occlusion': 'protect_face_hands'}]
        specs = [{**x, 'description': 'Synthetic smoke/particles raster fixture only; not an AI image acceptance.',
                  'increment': True, 'baked_in_reference': False,
                  'effect_design': {'material': 'luminous_particles' if x['operation']=='emission_add' else 'scattering_smoke',
                    'colour_rgb8': [255,181,82], 'extent_2d': [0,.6,1,1], 'density': 'sparse'}} for x in specs]
        brief = {'primary_effect_plan': {'mode': 'scene_effect', 'description': 'Fixture fog', 'reason': 'Verify operators'},
                 'spatial_integration_plan': 'A thin foreground mist crosses the lower garment.',
                 'foreground_effect_plan': {'enabled': True, 'layers': specs}}
        run = {'root': str(root), 'kind': 'single_input_run', 'input_mode': 'single_source',
               'run_id': 'fixture', 'sample': 'fixture', 'revision': 1,
               'source': {'path': str(source_path), 'sha256': digest(source_path), 'oriented_size': size},
               'reference': {'path': str(reference_path), 'sha256': digest(reference_path)},
               'generation_design': {'design_brief': brief},
               'options': {'render_mode': 'native_source_design', 'appearance_revision': {'design_reconstruction_plan': plan}}}
        workflow = inputs/'workflow.json'
        shutil.copy2(Path(__file__).resolve().parents[1]/'providers/workflows/qwen-image-2.1-effect-rgba-text-only.json', workflow)
        node_target = root/'providers/comfyui/cosplay_effect_canvas.py'
        node_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__).resolve().parents[1]/'providers/comfyui/cosplay_effect_canvas.py', node_target)
        # Current ordinary portrait and behind-only smoke do not produce images.
        portrait = copy.deepcopy(run); portrait['generation_design']['design_brief']['primary_effect_plan']['mode'] = 'none'
        assert prepare_effects(root, portrait, plan)['status'] == 'skipped'
        baked = copy.deepcopy(run)
        baked['generation_design']['design_brief'].pop('foreground_effect_plan')
        baked['generation_design']['design_brief']['spatial_integration_plan'] = 'Keep foreground stone unobscured; smoke softer behind her.'
        assert prepare_effects(root, baked, plan)['status'] == 'skipped'
        baked['generation_design']['design_brief']['spatial_integration_plan'] = 'A thin foreground smoke layer is required in front of the person.'
        assert prepare_effects(root, baked, plan)['status'] == 'skipped', 'Prose cannot authorize an undeclared image request'
        pending = prepare_effects(root, run, plan, workflow_path=workflow)
        assert pending['status'] == 'pending_generation' and len(pending['requests']) == 3

        def synthetic_provider(request_path):
            request = read(request_path)
            image_path, provenance_path = Path(request['generated_image_path']), Path(request['generated_provenance_path'])
            image_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(rgba_path, image_path)
            write(provenance_path, {'asset_role': 'effect_layer', 'alpha': {'preserved': True},
                'source': request['scene_context'], 'asset': asset(image_path), 'prompt': request['prompt'],
                'generation_claim': 'CPU synthetic raster fixture; no AI generation performed'})
            return 0

        with patch('foreground_effects.produce_request', side_effect=synthetic_provider) as producer:
            prepared = prepare_effects(root, run, plan, generate=True, workflow_path=workflow)
            assert producer.call_count == 3
        plan = prepared['plan']
        assert prepared['status'] == 'pending_visual_review'
        assert all(not x['enabled'] for x in plan['layers'] if x.get('kind') == KIND)
        unaccepted = copy.deepcopy(plan['layers'][1])
        try:
            accept_layer(unaccepted, run, plan, review={'status': 'pass',
                'effect_sha256': unaccepted['assets']['generated_rgba']['sha256']})
        except ValueError:
            pass
        else:
            raise AssertionError('A pass bool and effect SHA became actual visual acceptance')
        mock_verdicts = {'schema': effect_review.OBSERVATION_SCHEMA, 'checks': {
            key: {'status': 'pass', 'reason': 'CPU mocked visual response; synthetic fixture only.',
                'evidence': [{'basis': 'visible_pixels', 'panels': ['composite_after', 'isolated_rgba', 'actual_alpha', 'consumed_alpha'],
                    'observation': 'Known synthetic smoke field is visible and separate from source pixels.'}]}
            for key in effect_review.CHECKS}}
        for i, layer in enumerate(plan['layers']):
            if layer.get('kind') != KIND: continue
            with patch.object(effect_review.vision_transport, 'request_vision_json', side_effect=fake_transport(mock_verdicts)):
                review = effect_review.review_effect(run, plan, layer, inputs/('mock-visual-'+str(i)),
                    provider='cpu_mock', vision_model='mock_vision')
            plan['layers'][i] = accept_layer(layer, run, plan, review=review)
        run['options']['appearance_revision']['design_reconstruction_plan'] = plan
        # Refresh uses a different output cache and must retain owned accepted
        # layers rather than returning new requests or deleting their controls.
        refreshed = prepare_effects(root, run, plan, output_dir=inputs/'different-empty-producer-cache', workflow_path=workflow)
        assert refreshed['status'] == 'prepared' and not refreshed['requests']
        assert all(row['enabled'] for row in refreshed['plan']['layers'] if row.get('kind') == KIND)
        source = source_for_plan(plan, size)
        geometry = Image.fromarray(np.uint8(alpha*255))
        source_bytes = {name: digest(row['path']) for name, row in plan['source_baseline']['assets'].items()}
        full, summary = apply_plan(source, source.copy(), plan, geometry, run=run, include_optics=False)
        import effect_code_identity
        import effect_layout
        old_code = read(next(row for row in plan['layers'] if row.get('kind') == KIND)['assets']['contract']['path'])['fx_code_identity']
        changed_code = copy.deepcopy(old_code)
        changed_code['files']['effect_layout.py'] = '0'*64
        with patch.object(effect_code_identity, 'current_code_identity', return_value=changed_code), \
             patch.object(effect_layout, 'PARAMETERS', {'future_producer_parameter': 99}), \
             patch.object(effect_layout, 'measure_before_luminance', side_effect=AssertionError('saved replay must not remeasure environment')), \
             patch('foreground_effects.compile_condition', side_effect=AssertionError('saved replay must not recompile scene condition')):
            try:
                prepare_effects(root, run, plan, output_dir=inputs/'refused-after-producer-update')
            except ValueError:
                pass
            else:
                raise AssertionError('Old producer contract was silently accepted by current FX')
        wrong_abi = copy.deepcopy(plan['layers'][1])
        contract = read(wrong_abi['assets']['contract']['path'])
        contract['operation_abi'] = 'unknown-future-operator'
        incompatible = inputs/'incompatible-operation-abi.json'
        write(incompatible, contract)
        wrong_abi['assets']['contract'] = asset(incompatible)
        try:
            validate_layer(wrong_abi, run, plan, require_accepted=False)
        except ValueError:
            pass
        else:
            raise AssertionError('Unsupported operator ABI silently changed saved pixels')
        sampled = lambda layer, key: sample_asset(layer['assets']['effect'], key, (0, 0, width, height), size)
        back, front, light = [x for x in plan['layers'] if x.get('kind') == KIND]
        expected_background = sampled(back, 'premultiplied_rgb')+(1-sampled(back, 'alpha')[..., None])*target
        expected = fp+(1-alpha[..., None])*expected_background
        # Behind is hidden at alpha1; alpha.3 attenuates the background once, not twice.
        behind_only = copy.deepcopy(plan)
        for x in behind_only['layers']:
            if x.get('kind') == KIND and x['slot'] == 'front': x['enabled'] = False
        rendered_back, _ = apply_plan(source, source.copy(), behind_only, geometry, run=run, include_optics=False)
        np.testing.assert_allclose(rendered_back, expected, rtol=0, atol=1e-7)
        np.testing.assert_array_equal(rendered_back[alpha == 1], fp[alpha == 1])
        expected = sampled(front, 'premultiplied_rgb')+(1-sampled(front, 'alpha')[..., None])*expected
        expected += sampled(light, 'emission')
        np.testing.assert_allclose(full, expected, rtol=0, atol=1e-7)
        assert np.max(sampled(front, 'alpha')[labels == 3]) == 0
        np.testing.assert_array_equal(full[labels == 3], fp[labels == 3])
        assert {x['slot'] for x in summary['layers'] if x['kind'] == KIND} == {'front', 'behind'}
        off = copy.deepcopy(plan)
        for x in off['layers']:
            if x.get('kind') == KIND: x['enabled'] = False
        none, _ = apply_plan(source, source.copy(), off, geometry, run=run, include_optics=False)
        np.testing.assert_allclose(none, fp+(1-alpha[..., None])*target, rtol=0, atol=1e-7)
        box = (15, 32, 79, 102)
        roi, _ = render_native_roi(run, box)
        np.testing.assert_allclose(roi, full[box[1]:box[3], box[0]:box[2]], rtol=0, atol=1e-7)
        assert source_bytes == {name: digest(row['path']) for name, row in plan['source_baseline']['assets'].items()}
        # A portable run owns RGBA, associated field and contract independently of disposable producer cache.
        owned = materialize_plan(root, run, plan)
        run['options']['appearance_revision']['design_reconstruction_plan'] = owned
        directory = root/'outputs/runs/fixture'; directory.mkdir(parents=True, exist_ok=True)
        write(directory/'manifest.json', run)
        package = root/'fixture.cosplay'
        export_project(root, 'fixture', package)
        imported = import_project(root/'moved', package, run_id='moved-fixture')['run']
        moved = imported['options']['appearance_revision']['design_reconstruction_plan']
        relocated, _ = apply_plan(source, source.copy(), moved, geometry, run=imported, include_optics=False)
        np.testing.assert_array_equal(relocated, full)
        bad = copy.deepcopy(run); bad['reference']['sha256'] = '0'*64
        try: validate_layer(front, bad, plan)
        except ValueError: pass
        else: raise AssertionError('Stale scene reference accepted')
        try: validate_layer(unaccepted, run, plan)
        except ValueError: pass
        else: raise AssertionError('Transport success became visual acceptance')
        report = {'ok': True, 'fixture': 'actual 8-bit RGBA raster; synthetic, not AI generated',
            'visual_response': 'CPU mocked text transport; no real API quality acceptance',
            'current_display_prepared_baseline_explicit': True,
            'pass_bool_and_effect_sha_rejected': True, 'actual_review_evidence_closure_bound': True,
            'owned_accepted_refresh_survives_empty_producer_cache': True,
            'old_producer_contract_explicitly_rejected_without_remeasurement': True,
            'unsupported_operator_ABI_readonly_master_required': True,
            'ordinary_and_baked_back_skip': True, 'automatic_producer_calls': 3,
            'straight_srgb_to_linear_associated': True, 'hidden_RGB_no_bleed': True,
            'behind_person_coverage_once': True, 'front_over_once': True,
            'emission_separate_additive': True, 'semantic_face_hands_protection': True,
            'native_full_roi_equal': True, 'source_RGB_assets_unchanged': True,
            'toggle_off_baseline': True, 'portable_replay_exact': True, 'stale_and_unreviewed_refused': True}
        if output_dir:
            from raw_support import linear_preview
            destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(package, destination/'synthetic-effects-fixture.cosplay')
            shutil.copy2(rgba_path, destination/rgba_path.name)
            np.savez_compressed(destination/'synthetic-effect-native-render.npz', values=full)
            panels = [linear_preview(x) for x in (none, rendered_back, full, relocated)]
            canvas = Image.new('RGB', (width*4, height))
            for index, image in enumerate(panels): canvas.paste(image, (index*width, 0))
            canvas.save(destination/'synthetic-effects-baseline-behind-front-replay.png')
            write(destination/'fixture-proof.json', report)
        print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    main(parser.parse_args().output_dir)
