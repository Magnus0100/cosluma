"""Real CPU freeze/replay proof with a synthetic matching operator, no inference."""
from __future__ import annotations

from contextlib import ExitStack
import copy
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import cv2
import numpy as np

from current_stack_fixture import prepare_fixture
from portrait_beauty_backend_smoke import prepared, write
from display_transform import source_transform_for_precision
from native_source_baseline import source_for_reference_prepare, source_for_plan
from portrait_beauty_layer import (apply_for_run, identity_for_run, materialize_for_run, sha256,
    hair_correction_guard_for_run, sample_visible_face_support, VISIBLE_FACE_POLICY)
from pipeline_stage_report import build_stage_report, read_stage_report, format_stage_report
from reference_appearance_controls import corner_targets
from reference_appearance_prepare import freeze
from reference_pipeline import _input_identity, attach_reference_support
from float_appearance import render_source_float
from runtime_resources import code_path


def field(path, values, **extra):
    np.savez_compressed(path, **values)
    return {'path': str(path), 'sha256': sha256(path), **extra}


def fixture(root):
    run = prepare_fixture(root/'assets', native=(32, 24))
    run.update(root=str(root), run_id='beauty-fit-fixture')
    plan = run['options']['appearance_revision']['design_reconstruction_plan']
    # An observed scene receipt is carried through freeze, never re-created
    # from the currently selected runtime or used to access models here.
    plan.setdefault('provenance', {})['local_model_registry'] = {
        'path': str(root/'fixture-registry.json'), 'sha256': 'synthetic_scene_registry_receipt',
        'semantics_profile': 'sapiens2_seg_0.4b', 'matting_profile': 'vitmatte_small_distinctions',
        'scene_profile': 'synthetic_fixture_scene', 'dinov3_profile': 'synthetic_fixture_dinov3'}
    baseline = plan['source_baseline']
    baseline['source_sha256'] = run['source']['sha256']
    baseline['source_display'] = source_transform_for_precision('raster8_srgb')
    plan['canvas']['candidate_mapping'] = {'candidate_original_to_final_original': np.eye(3).tolist()}
    plan['mask_roles']['identity_protection'] = field(root/'protection.npz',
        {'values': np.zeros((24, 32), np.float32)}, size=[32, 24])
    return run


def operator(values):
    # A nonlinear deterministic stand-in exposes an incorrectly chosen source
    # basis; this is a data-flow test, not a colour-fitting quality evaluation.
    return (np.sqrt(np.asarray(values, np.float32))*.67+.013).astype(np.float32)


def frozen_case(root, run, name):
    match = root/name
    match.mkdir()
    plan = run['options']['appearance_revision']['design_reconstruction_plan']
    current_identity = identity_for_run(run)
    weights = field(match/'source-weights.npz', {'values': np.ones((24, 32, 1), np.float32)})
    from reference_colour_fit import REGIONS
    parameters = {'support_policy': 'reference_support.v1', 'operators': {}, 'observations': {},
        'region_order': list(REGIONS),
        'binding': {'source': run['source']['sha256'], 'reference': run['reference']['sha256'],
            'source_display': source_transform_for_precision('raster8_srgb'),
            'native_source_retouch': None, 'portrait_beauty': current_identity,
            'decode_version': plan['source_baseline']['decode_version'],
            'alpha': plan['mask_roles']['foreground_alpha']['sha256'],
            'source_weights_sha256': weights['sha256'],
            'algorithm_sha256': sha256(code_path('reference_colour_fit.py'))}}
    write(match/'parameters.json', parameters)
    write(match/'composite-receipt.json', {'parameters_sha256': sha256(match/'parameters.json'),
        'renderer_sha256': sha256(code_path('render_reference_colour_composite.py')),
        'output_sha256': 'synthetic_matching_operator_not_an_image_inference'})
    zeros = np.zeros((24, 32, 3), np.float32)
    scene_asset = field(match/'scene.npz', {'baseline_rgb': zeros})
    guard=hair_correction_guard_for_run(run)
    guard_binding=guard['binding'] if guard else None
    raw_colour=np.zeros((24,32,6),np.float32);raw_scene=np.zeros((24,32,4),np.float32)
    if guard:
        raw_colour[..., :3]=[.02,-.01,.6];raw_scene[:]=[.5,0,0,.8]
    colour = {'schema': 'reference_spatial_colour.v2', 'support_policy': 'reference_support.v1',
        'feature_chroma_policy': 'beauty_feature_chroma_with_supported_ambient.v1',
        'regions': {},'hair_correction_guard':guard_binding}
    scene = {'schema': 'reference_scene_balance.v1', 'support_policy': 'reference_support.v1',
        'asset': scene_asset, 'regions': {},'hair_correction_guard':guard_binding}
    seen = {'donor_source': [], 'matching_input': [], 'donor_B_S': [], 'native_B_S': []}
    def donors(source, *args, **kwargs):
        seen['donor_source'].append(source.copy())
        # Current freeze deliberately removes the unused donor colour field.
        # donor_fields receives scene balance and the frozen matched baseline
        # in positional slots 4 and 5; native matching still receives colour
        # and scene by keyword below.
        seen['donor_B_S'].append((args[4].copy(),args[5].copy()))
        return zeros
    def matching(values, *args, **kwargs):
        seen['matching_input'].append(values.copy())
        seen['native_B_S'].append((kwargs['reference_blend'].copy(),kwargs['scene_balance'].copy()))
        kwargs['donor_gamut_report'].update(policy='synthetic_matching_fixture',
            requested_outside_sdr_pixels=0, source_outside_premultiplied_sdr_pixels=0,
            maximum_requested_excursion=0, maximum_source_excursion=0,
            maximum_operator_increased_excursion=0, minimum_applied_dose=1)
        return operator(values)
    def receiver(_run, receiver_plan, _output):
        return {'design_reconstruction_plan': receiver_plan, 'receiver_preservation': {
            'state': 'synthetic_fixture_no_receiver_edit', 'qualification': {}, 'checks': {}}}
    with ExitStack() as context:
        context.enter_context(patch('reference_colour_blend.validate_layer', return_value=(raw_colour, colour)))
        context.enter_context(patch('reference_scene_balance.validate_scene_layer', return_value=(raw_scene, scene, zeros)))
        context.enter_context(patch('render_reference_colour_composite.field_owners', return_value=np.full((24,32),4,np.int32)))
        context.enter_context(patch('render_reference_colour_composite.donor_fields', side_effect=donors))
        context.enter_context(patch('render_reference_colour_composite.match_foreground', side_effect=matching))
        context.enter_context(patch('reference_support.donor_support_report', return_value={'policy_executed': True, 'regions': {}}))
        context.enter_context(patch('native_receiver_preservation.prepare_native_receiver_preservation', side_effect=receiver))
        result = freeze(run, match, root/(name+'-frozen'))
    registry = plan['provenance']['local_model_registry']
    frozen_registry = result['design_reconstruction_plan']['provenance']['local_model_registry']
    assert frozen_registry == registry and frozen_registry is not registry
    original_path = registry['path']
    registry['path'] = 'changed_after_freeze'
    assert frozen_registry['path'] == original_path
    registry['path'] = original_path
    if guard:
        visible=sample_visible_face_support(guard,(0,0,32,24),(32,24))
        expected_B=raw_colour.copy();expected_B[...,:2]*=(1-visible)[...,None]
        expected_S=raw_scene.copy();expected_S[...,:3]*=(1-visible)[...,None]
        for consumed_B,consumed_S in seen['donor_B_S']:
            np.testing.assert_array_equal(consumed_B,expected_S)
            np.testing.assert_array_equal(consumed_S,zeros)
        for consumed_B,consumed_S in seen['native_B_S']:
            np.testing.assert_array_equal(consumed_B,expected_B)
            np.testing.assert_array_equal(consumed_S,expected_S)
    attach_reference_support(root, result['design_reconstruction_plan'], root/(name+'-support'))
    return result, seen


def read(asset):
    with np.load(asset['path'], allow_pickle=False) as data:
        return data['values'].copy()


def main():
    checks = []
    with tempfile.TemporaryDirectory(prefix='cos-beauty-fit-') as temporary:
        root = Path(temporary)
        raw_run = fixture(root)
        raw, raw_precision, _ = source_for_reference_prepare(raw_run)
        no_beauty, unchanged_precision, _ = source_for_reference_prepare(raw_run, apply_beauty=True)
        np.testing.assert_array_equal(no_beauty, raw)
        assert raw_precision['portrait_beauty_consumptions'] == unchanged_precision['portrait_beauty_consumptions'] == 0
        plain, plain_seen = frozen_case(root, raw_run, 'plain')
        checks.append('real_freeze_preserves_original_scene_registry_as_independent_metadata')
        plain_plan = plain['design_reconstruction_plan']
        np.testing.assert_array_equal(read(plain_plan['source_baseline']['assets']['foreground_premultiplied']), raw)
        np.testing.assert_array_equal(plain_seen['matching_input'][0], raw)
        plain_layer = plain_plan['layers'][0]
        np.testing.assert_array_equal(read(plain_layer['assets']['automatic_delta']), operator(raw)-raw)
        assert plain_layer['params']['fit_beauty_identity'] is None
        checks.append('no_beauty_source_decode_baseline_and_delta_are_numerically_unchanged')

        beauty_run = copy.deepcopy(raw_run)
        created = prepared(beauty_run, root/'beauty-model-fixture')
        # This fixture certifies its synthetic complete visible-face support;
        # it is a source-flow proof, not a learned material/quality result.
        for face in created['layer']['faces']:
            face.update(composition_policy=VISIBLE_FACE_POLICY,composition={'policy':VISIBLE_FACE_POLICY,
                'qualification_status':'full_visible_face','nonface_material_excluded':True,
                'skin_only':False,'features_gate_applied':False,'raw_probability_applied':False,
                'source_texture_restoration':False})
        write(Path(created['result']),created['layer'])
        beauty_run['options']['appearance_revision']['portrait_beauty'] = {
            'result': created['result'], 'strength': 1., 'disabled_faces': []}
        beautiful, precision, _ = source_for_reference_prepare(beauty_run, apply_beauty=True)
        expected_beauty = apply_for_run(raw, beauty_run)[0]
        from automatic_native_retouch import prepare_once
        with patch('automatic_native_retouch.subprocess.Popen', side_effect=AssertionError('Must not alter upstream source after beauty')):
            retained = prepare_once(root, beauty_run)
        assert retained['status'] == 'source_locked_by_portrait_beauty'
        assert retained['run']['options'] == beauty_run['options']
        assert not (root/'outputs/runs'/beauty_run['run_id']/'automatic-native-retouch').exists()
        changed = copy.deepcopy(beauty_run)
        changed['source']['sha256'] = 'different-source'
        try:
            prepare_once(root, changed)
        except ValueError:
            pass
        else:
            raise AssertionError('Source lock must validate the beauty input binding')
        np.testing.assert_array_equal(beautiful, expected_beauty)
        assert not np.array_equal(beautiful, raw) and precision['portrait_beauty_consumptions'] == 1
        untouched, original_precision, _ = source_for_reference_prepare(beauty_run, apply_beauty=False)
        np.testing.assert_array_equal(untouched, raw)
        assert original_precision['portrait_beauty_consumptions'] == 0
        checks.append('fit_explicitly_consumes_generated_beauty_once_immutable_decode_does_not')

        frozen, seen = frozen_case(root, beauty_run, 'beauty')
        plan = frozen['design_reconstruction_plan']
        layer = plan['layers'][0]
        immutable_fp = read(plan['source_baseline']['assets']['foreground_premultiplied'])
        np.testing.assert_array_equal(immutable_fp, raw)
        np.testing.assert_array_equal(source_for_plan(plan), raw)
        np.testing.assert_array_equal(seen['donor_source'][0], expected_beauty)
        np.testing.assert_array_equal(seen['matching_input'][0], expected_beauty)
        delta = read(layer['assets']['automatic_delta'])
        np.testing.assert_array_equal(delta, operator(expected_beauty)-expected_beauty)
        assert not np.array_equal(delta, operator(expected_beauty)-raw)
        assert frozen['checks']['immutable_source_pair_contains_beauty'] is False
        assert layer['params']['fit_beauty_identity'] == identity_for_run(beauty_run)
        checks.append('real_freeze_keeps_original_Fp_B_and_anchors_only_TBS_delta_to_beauty_working_Fp')
        checks.append('real_freeze_proxy_donors_and_native_tiles_consume_same_hair_visibility_veto_once')

        replay_run = copy.deepcopy(beauty_run)
        replay_run['options']['appearance_revision']['design_reconstruction_plan'] = plan
        replay, _ = render_source_float(raw, replay_run)
        np.testing.assert_allclose(replay, operator(expected_beauty), rtol=0, atol=3e-8)
        assert not np.allclose(replay, operator(expected_beauty)+(expected_beauty-raw), rtol=0, atol=1e-6)
        alpha = read(plan['source_baseline']['assets']['foreground_alpha'])
        corners, _, _ = corner_targets(expected_beauty, expected_beauty+delta, alpha)
        for role in ('colour_delta', 'lightness_delta'):
            np.testing.assert_array_equal(read(layer['assets'][role]), corners[role]-expected_beauty)
        control_receipt = json.loads(Path(layer['assets']['control_receipt']['path']).read_text('utf8'))
        assert control_receipt['endpoint_contract']['beauty_applied_before_corner_derivation'] is True
        assert control_receipt['binding']['portrait_beauty'] == identity_for_run(beauty_run)
        checks.append('actual_renderer_does_not_double_beauty_and_control_corners_use_beauty_source')

        with patch('reference_pipeline._registry_identity', return_value={'sha256': 'fixture-registry',
                'semantics_profile': 'fixture', 'matting_profile': 'fixture'}), \
             patch('reference_pipeline._feature_identity', return_value={'available': True, 'identity': 'fixture'}):
            before = _input_identity(root, beauty_run)
            relocated = copy.deepcopy(beauty_run)
            relocated['run_id'] = 'relocated'
            relocated['options']['appearance_revision']['portrait_beauty'] = materialize_for_run(root, relocated,
                relocated['options']['appearance_revision']['portrait_beauty'])
            assert _input_identity(root, relocated) == before
            changed = copy.deepcopy(beauty_run)
            changed['options']['appearance_revision']['portrait_beauty']['strength'] = .5
            assert _input_identity(root, changed)['portrait_beauty'] != before['portrait_beauty']
            parameters = json.loads((root/'beauty/parameters.json').read_text('utf8'))
            try:
                source_for_reference_prepare(changed, binding=parameters['binding'], apply_beauty=True)
            except ValueError:
                pass
            else:
                raise AssertionError('A fit stage accepted a different beauty source')
        # A later dose edit replays the saved response; it does not refit or
        # claim the fit source still matches. Saving remains a local operation.
        changed = copy.deepcopy(replay_run)
        changed['options']['appearance_revision']['portrait_beauty']['strength'] = .5
        assert identity_for_run(changed) != layer['params']['fit_beauty_identity']
        modified, _ = render_source_float(raw, changed)
        np.testing.assert_allclose(modified, apply_for_run(raw, changed)[0]+delta, rtol=0, atol=3e-8)
        checks.append('beauty_cache_identity_is_locator_independent_and_dose_edits_do_not_refit_or_block_replay')

        # Reports compare metadata, preserving the renderer's legal frozen
        # replay. No pixel hashes, matching workers or model tasks are run.
        with patch('portrait_beauty_layer.sha256', side_effect=AssertionError('Status must not hash assets')):
            matched_report = read_stage_report(root, replay_run)
            matching = matched_report['portrait_beauty_reference_fit']
            assert matching['state'] == 'matched' and matching['fit_dependency_matches'] is True
            assert matching['current_strength'] == 1. and matching['disabled_faces'] == []
            assert matching['reference_reprepare_recommended'] is False
            assert matched_report['reference_appearance_stack']['prepared']
            changed_report = read_stage_report(root, changed)
            dependency = changed_report['portrait_beauty_reference_fit']
            assert dependency['state'] == 'changed' and dependency['fit_dependency_matches'] is False
            assert dependency['current_strength'] == .5 and dependency['reference_reprepare_recommended']
            assert not changed_report['reconstruction_required']
            assert dependency['frozen_replay_blocked_by_fit_dependency'] is False
            assert changed_report['reference_appearance_stack']['prepared']
            assert '重新准备参考匹配' in format_stage_report(changed_report)
            disabled_run = copy.deepcopy(replay_run)
            disabled_run['options']['appearance_revision']['portrait_beauty']['disabled_faces'] = ['face-01']
            disabled_report = read_stage_report(root, disabled_run)
            assert disabled_report['portrait_beauty_reference_fit']['disabled_faces'] == ['face-01']
            assert disabled_report['portrait_beauty_reference_fit']['state'] == 'changed'
            assert not disabled_report['reconstruction_required']
            zero_run = copy.deepcopy(replay_run)
            zero_run['options']['appearance_revision']['portrait_beauty']['strength'] = 0.
            assert read_stage_report(root, zero_run)['portrait_beauty_reference_fit']['state'] == 'changed'
            plain_run = copy.deepcopy(raw_run)
            plain_run['options']['appearance_revision']['design_reconstruction_plan'] = plain_plan
            plain_state = read_stage_report(root, plain_run)['portrait_beauty_reference_fit']
            assert plain_state['state'] == 'matched' and plain_state['current_strength'] is None
            removed_run = copy.deepcopy(replay_run)
            removed_run['options']['appearance_revision'].pop('portrait_beauty')
            assert read_stage_report(root, removed_run)['portrait_beauty_reference_fit']['state'] == 'changed'
            relocated['options']['appearance_revision']['design_reconstruction_plan'] = plan
            assert read_stage_report(root, relocated)['portrait_beauty_reference_fit']['state'] == 'matched'
            relative_run = copy.deepcopy(replay_run)
            relative_run['options']['appearance_revision']['portrait_beauty']['result'] = str(
                Path(created['result']).relative_to(root))
            assert read_stage_report(root, relative_run)['portrait_beauty_reference_fit']['state'] == 'matched'
        checks.append('read_only_stage_report_detects_strength_disabled_zero_removed_and_relocated_fit_inputs')

        retained = copy.deepcopy(replay_run)
        missing_run = copy.deepcopy(replay_run)
        missing_run['options']['appearance_revision']['portrait_beauty']['result'] = str(root/'missing-beauty.json')
        missing_report = read_stage_report(root, missing_run)
        missing_state = missing_report['portrait_beauty_reference_fit']
        assert missing_state['state'] == 'unavailable' and missing_state['error']
        assert missing_state['fit_dependency_matches'] is None
        assert missing_state['reference_reprepare_recommended'] is None
        corrupt_path = root/'corrupt-beauty.json'
        corrupt_path.write_text('{broken', encoding='utf8')
        corrupt_run = copy.deepcopy(missing_run)
        corrupt_run['options']['appearance_revision']['portrait_beauty']['result'] = str(corrupt_path)
        assert read_stage_report(root, corrupt_run)['portrait_beauty_reference_fit']['state'] == 'unavailable'
        corrupt_path.write_text('[]', encoding='utf8')
        assert read_stage_report(root, corrupt_run)['portrait_beauty_reference_fit']['state'] == 'unavailable'
        metadata_missing = copy.deepcopy(replay_run)
        metadata_missing['options']['appearance_revision']['design_reconstruction_plan']['layers'][0]['params'].pop('fit_beauty_identity')
        metadata_state = read_stage_report(root, metadata_missing)['portrait_beauty_reference_fit']
        assert metadata_state['state'] == 'not_recorded' and metadata_state['reference_reprepare_recommended']
        # The pure reducer does not perform hidden file reads when callers
        # have only manifest/presence observations.
        with patch('portrait_beauty_layer.load_result', side_effect=AssertionError('Reducer must not load files')):
            pure_report = build_stage_report(replay_run, {'assets': {}})
        assert pure_report['portrait_beauty_reference_fit']['state'] == 'unavailable'
        assert replay_run == retained
        checks.append('report_handles_missing_corrupt_and_unobserved_identity_without_mutating_or_blocking_replay')
        assert not any(name in sys.modules for name in ('torch', 'diffusers', 'beautygrpo_model_worker'))
    print(json.dumps({'ok': True, 'checks': checks, 'gpu_calls': 0, 'model_inferences': 0,
        'scope': 'real source decode, asset freeze, corner derivation and rendering; synthetic matching operator, not image quality'}))


if __name__ == '__main__':
    main()
