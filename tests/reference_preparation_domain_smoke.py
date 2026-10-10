"""CPU contract and numeric checks for source D, approved source and final encoding."""
from __future__ import annotations

import copy
import json
from unittest.mock import patch

import numpy as np

import display_transform as display
import native_source_baseline as source
from raw_support import RGB_DECODE_VERSION


def fixture(precision):
    transform = display.source_transform_for_precision(precision)
    return {'source': {'sha256': 'a'*64}, 'options': {'appearance_revision': {
        'design_reconstruction_plan': {'schema_version': 'design_reconstruction_plan.v1',
            'working_domain': transform['input_domain'], 'source_sha256': 'a'*64,
            'source_baseline': {'schema': 'source_preparation_baseline.v1',
                'working_domain': transform['input_domain'], 'source_sha256': 'a'*64,
                'decode_version': RGB_DECODE_VERSION, 'source_precision': precision,
                'source_display': transform, 'source_display_applied_once': False}}}}}


def main():
    passed = []
    run = fixture('raw16_linear')
    baseline = run['options']['appearance_revision']['design_reconstruction_plan']['source_baseline']
    original = np.asarray([[[.97, .88, .84], [2., 1.1, .4]]], np.float32)
    delta = np.asarray([[[.03, -.02, .01], [.05, .01, .02]]], np.float32)
    identity = {'source_sha256': 'a'*64, 'working_domain': 'scene_linear_sRGB',
                'approved_patches': 1, 'delta_sha256': 'b'*64}
    precision = {'source_precision': 'raw16_linear', 'decode_version': RGB_DECODE_VERSION,
                 'native_source_retouch': identity, 'native_retouch_consumptions': 1}
    with patch.object(source, 'open_source_for_run', side_effect=lambda *args, **kwargs: (original.copy()+delta, precision)):
        values, actual, transform = source.source_for_reference_prepare(run)
        expected = display.display_linear(original+delta, transform)
        np.testing.assert_array_equal(values, expected)
        assert not np.array_equal(values, display.display_linear(original, transform))
        assert not np.array_equal(values, display.display_linear(expected, transform))
        assert actual['source_display_consumptions'] == actual['native_retouch_consumptions'] == 1
        passed.append('approved_nonzero_source_delta_then_D_exactly_once_before_TBS')
        binding = {'source': 'a'*64, 'decode_version': RGB_DECODE_VERSION,
                   'source_display': transform, 'native_source_retouch': identity}
        for _ in range(3):
            refit, _, _ = source.source_for_reference_prepare(run, binding=binding)
            np.testing.assert_array_equal(refit, expected)
        passed.append('each_stage_redecodes_same_bound_approved_source_without_accumulation')
        for name, mutate in (
            ('different_source_delta_identity', lambda row: row['native_source_retouch'].update(delta_sha256='changed')),
            ('parameters_missing_source_delta_binding', lambda row: row.pop('native_source_retouch')),
            ('final_encoder_substituted_for_source_D', lambda row: row.update(source_display={'method': 'clip_sdr'}))):
            changed = copy.deepcopy(binding); mutate(changed)
            try: source.source_for_reference_prepare(run, binding=changed)
            except ValueError: passed.append(name)
            else: raise AssertionError('Wrong source fitting binding accepted: '+name)
    for name, mutate in (
        ('missing_baseline_schema', lambda plan: plan['source_baseline'].pop('schema')),
        ('missing_source_transform', lambda plan: plan['source_baseline'].pop('source_display')),
        ('missing_source_domain', lambda plan: plan['source_baseline'].pop('working_domain')),
        ('missing_plan_domain', lambda plan: plan.pop('working_domain')),
        ('wrong_source_identity', lambda plan: plan['source_baseline'].update(source_sha256='other')),
        ('wrong_source_input_domain', lambda plan: plan['source_baseline']['source_display'].update(input_domain='display_linear_sRGB')),
        ('missing_explicit_unconsumed_state', lambda plan: plan['source_baseline'].pop('source_display_applied_once'))):
        changed = copy.deepcopy(run); mutate(changed['options']['appearance_revision']['design_reconstruction_plan'])
        try: display.source_transform_for_run(changed)
        except ValueError: passed.append(name)
        else: raise AssertionError('Implicit source preparation accepted: '+name)
    for precision_name in ('raw16_linear', 'raster8_srgb'):
        try: display.plan_for_run(fixture(precision_name))
        except ValueError: passed.append(precision_name+'_unprepared_baseline_cannot_encode_final')
        else: raise AssertionError('Unprepared source entered final render')
    prepared = copy.deepcopy(run)
    plan = prepared['options']['appearance_revision']['design_reconstruction_plan']
    plan['working_domain'] = 'display_linear_sRGB'
    plan['source_baseline'].update(schema='appearance_source_baseline.v1',
                                  working_domain='display_linear_sRGB', source_display_applied_once=True)
    final = display.plan_for_run(prepared)
    assert final['method'] == 'clip_sdr' and final['input_domain'] == 'display_linear_sRGB'
    np.testing.assert_array_equal(display.display_linear(expected, final), expected)
    assert display.source_transform_for_run(prepared) == transform
    passed.append('completed_baseline_refit_reads_original_D_and_final_only_encodes_display')
    explicit_execution = copy.deepcopy(prepared)
    existing = explicit_execution['options']['appearance_revision']['design_reconstruction_plan']
    existing['source_baseline'].pop('source_display_applied_once')
    existing['execution'] = {'source_display_applied_at_native_prepare': True}
    assert display.plan_for_run(explicit_execution) == final
    passed.append('independent_final_encoding_accepts_explicit_already_consumed_execution')
    local = {'source': {'sha256': 'c'*64}, 'options': {}}
    raster = np.asarray([[[.97, .88, .84]]], np.float32)
    values, local_plan = source.prepare_source_only(raster, local,
        {'source_precision': 'raster8_srgb', 'decode_version': RGB_DECODE_VERSION})
    np.testing.assert_array_equal(values, raster)
    local['options']['source_only_plan'] = local_plan
    assert display.plan_for_run(local)['method'] == 'clip_sdr'
    passed.append('display_decoded_raster_preserves_source_values_and_final_has_no_second_shoulder')
    print(json.dumps({'schema': 'reference_preparation_domain_smoke.v1', 'status': 'passed',
        'checks_passed': len(passed), 'checks': passed, 'network_requests': 0,
        'model_requests': 0, 'image_generation_requests': 0, 'gpu_requests': 0}))


if __name__ == '__main__':
    main()
