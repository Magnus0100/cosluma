"""Small CPU checks for palette support ROI, optionally paired with its producer."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

import reference_material_palette as palette_math
from colour_space import from_lab
from reference_colour_fit import REGIONS, fit


def same_bits(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert actual.tobytes(order='C') == expected.tobytes(order='C')


def rejected(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('Invalid palette/input contract was accepted')


def main(baseline_path=None, parameters_path=None, report_path=None):
    previous = None
    if baseline_path:
        spec = importlib.util.spec_from_file_location('frozen_palette_roi_fixture', baseline_path)
        previous = importlib.util.module_from_spec(spec); spec.loader.exec_module(previous)
    rgb = lambda lab: from_lab(np.asarray(lab, np.float32)).astype(np.float32)
    training = np.broadcast_to(rgb([.55, .02, .03]), (96, 128, 3)).copy()
    target = np.broadcast_to(rgb([.55, .035, .015]), training.shape).copy()
    labels = np.ones(training.shape[:2], np.uint8)
    palette = palette_math.fit_palette(training, target, labels, labels, np.ones(labels.shape), fit([]))
    assert palette['regions']['clothing']['status'] == 'supported'
    supported = palette['regions']['clothing']
    palette['regions'] = {name: copy.deepcopy(supported) for name in ('face', 'clothing', 'hair')}
    palette['regions']['face']['source_skin_L_floor'] = .54
    palette['regions']['face']['model']['nodes'] = (np.asarray(supported['model']['nodes'])*-.5).tolist()
    palette['regions']['hair']['model']['nodes'] = (np.asarray(supported['model']['nodes'])*.25).tolist()
    values = np.broadcast_to(rgb([.55, .02, .03]), (64, 96, 3)).copy()
    values[0, 0] = [-0., 0., np.nextafter(np.float32(0), np.float32(1))]
    values[18:25, 24:31] = [1., 0., 0.]
    source = np.broadcast_to(rgb([.55, .02, .03]), values.shape).copy()
    source[13:17, 15:19, 0] = 1.2
    source[21:23, 36:39, 1] = -.02
    weights = np.zeros((*values.shape[:2], len(REGIONS)), np.float32)
    names = list(REGIONS); face, cloth, hair = (names.index(name) for name in ('face', 'clothing', 'hair'))
    weights[8:28, 12:40, face] = .25
    weights[16:37, 23:51, cloth] = .45
    weights[21:31, 30:46, hair] = .15
    cases = []; snapshots = [array.tobytes() for array in (values, source, weights)]

    def check(name, baseline, immutable, fields, model=palette, strength=1.):
        output = palette_math.apply_palette(baseline, model, fields, source_values=immutable, strength=strength)
        if previous:
            same_bits(output, previous.apply_palette(baseline, model, fields, source_values=immutable, strength=strength))
        cases.append({'case': name, 'strength': strength, 'exact': True, 'paired_frozen_producer': previous is not None})
        return output

    for strength in (1., .37, 0.):
        result = check('overlap_skin_floor_source_excursions_gamut', values, source, weights, strength=strength)
        crop = np.s_[7:45, 9:59]
        same_bits(palette_math.apply_palette(values[crop], palette, weights[crop], source_values=source[crop], strength=strength), result[crop])
        same_bits(result[0:1, 0:1], values[0:1, 0:1])
    assert snapshots == [array.tobytes() for array in (values, source, weights)]
    assert np.any(palette_math.apply_palette(values, palette, weights, source_values=source) != values)
    check('reversed_strided_inputs', values[:, ::-1], source[:, ::-1], weights[:, ::-1])
    precise = values.astype(float); precise[1:7] += 1e-12
    check('float64_nonzero_output_contract', precise, source.astype(float), weights)
    check('float64_zero_strength_bits', precise, source.astype(float), weights, strength=0.)
    for column in (31, 95):
        thin = np.zeros_like(weights); thin[9:35, column, cloth] = .8
        check('single_column_'+str(column), values, source, thin)
    check('original_one_column_canvas', values[:, 31:32], source[:, 31:32], weights[:, 31:32])
    one = np.zeros_like(weights); one[18, 27, cloth] = 1.
    isolated = np.full_like(source, 1.2); isolated[18, 27] = rgb([.55, .02, .03]); isolated[55, 70] = rgb([.55, .02, .03])
    check('single_valid_roi_pixel_original_multiple', values, isolated, one)
    isolated[55, 70] = 1.2
    check('single_valid_source_pixel_total', values, isolated, one)
    gamut_one = np.zeros_like(weights); gamut_one[18, 30, cloth] = 1.
    gamut_source = np.full_like(source, 1.2); gamut_source[18, 30] = rgb([.55, .02, .03])
    gamut_source[55, 70] = rgb([.55, .02, .03])
    check('single_gamut_pixel_with_outside_inactive_bad_pixels', values, gamut_source, gamut_one)
    check('weights_below_existing_gate', values.astype(float), source, weights*1e-9)
    check('all_zero_weights_float32_contract', values.astype(float), source, weights*0)
    zero = copy.deepcopy(palette)
    for row in zero['regions'].values(): row['model']['nodes'] = (np.asarray(row['model']['nodes'])*0).tolist()
    same_bits(check('zero_residual_supported_palette', values, source, weights, model=zero), values)
    reversed_palette = copy.deepcopy(palette); reversed_palette['regions'] = dict(reversed(list(palette['regions'].items())))
    check('saved_region_accumulation_order', values, source, weights, model=reversed_palette)
    for module in (palette_math, previous):
        if module is None: continue
        forged = copy.deepcopy(palette); forged['regions']['hair']['model']['nodes'][0] = [.1, 0.]
        rejected(lambda: module.apply_palette(values, forged, weights*0, source_values=source))
        rejected(lambda: module.apply_palette(values, forged, weights*0, source_values=source, strength=0))
        damaged = weights.copy(); damaged[-1, -1, cloth] = np.nan
        rejected(lambda: module.apply_palette(values, palette, damaged, source_values=source))
        damaged_source = source.copy(); damaged_source[-1, -1, 0] = np.nan
        rejected(lambda: module.apply_palette(values, palette, weights*0, source_values=damaged_source))
        damaged_values = values.copy(); damaged_values[-1, -1, 0] = -.01
        rejected(lambda: module.apply_palette(damaged_values, palette, weights*0, source_values=source))
    if parameters_path:
        actual = json.loads(Path(parameters_path).read_text('utf8'))['automatic_material_palette']
        for strength in (1., .37):
            check('formal_run_palette_parameters', values, source, weights, model=actual, strength=strength)
        identity_weights = np.zeros_like(weights); identity_weights[..., cloth] = 1.
        same_bits(check('formal_run_identity_region_weights', values, source, identity_weights, model=actual), values)
    transforms = []; gamut_calls = []
    to_lab, from_lab_original = palette_math.to_lab, palette_math.from_lab
    def observe_lab(array):
        transforms.append({'shape': list(array.shape), 'pixels': int(array.size//3)})
        return to_lab(array)
    def observe_rgb(array):
        gamut_calls.append({'shape': list(array.shape), 'pixels': int(array.size//3)})
        return from_lab_original(array)
    with patch.object(palette_math, 'to_lab', side_effect=observe_lab), patch.object(palette_math, 'from_lab', side_effect=observe_rgb):
        palette_math.apply_palette(values, palette, weights, source_values=source)
    assert transforms[1]['shape'] == [29, 39, 3]
    assert gamut_calls[0]['shape'] == [29, 39, 3] and len(gamut_calls) > 1, 'Fixture must exercise bounded gamut search'
    no_work = []
    with patch.object(palette_math, 'to_lab', side_effect=lambda array: no_work.append(array.shape)):
        same_bits(palette_math.apply_palette(values, palette, weights*0, source_values=source), values)
    assert not no_work
    assert snapshots == [array.tobytes() for array in (values, source, weights)]
    report = {'schema': 'reference_material_palette_support_roi_smoke.v1', 'passed': True, 'cases': cases,
        'Lab_transform_calls': transforms, 'gamut_calls': gamut_calls, 'before_tile_pixels': values.size//3,
        'bounded_replay_pixels': gamut_calls[0]['pixels'], 'zero_support_has_no_Lab_or_gamut_work': True,
        'unused_palette_and_outside_support_input_checks_preserved': True, 'input_bits_unchanged': True,
        'inactive_signed_zero_subnormal_bits_preserved': True, 'models_or_GPU_used': False,
        'large_asset_array_loaded': False, 'performance_benchmark': False}
    if report_path: Path(report_path).write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--parameters', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args(); main(args.baseline, args.parameters, args.report)
