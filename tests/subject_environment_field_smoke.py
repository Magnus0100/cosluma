"""Independent fixed analytic fixtures for local environment response fields."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from reference_scene_balance import apply_scene_balance
from reference_colour_fit import to_lab
from reference_automatic_layers import lowpass
from subject_environment_field import (CONFIG, prepare_environment_field, _training_residual,
                                       _split_reference_observations, _fit, _guided_upsample,
                                       GUIDANCE_POLICY, _macro_guidance, _regularize_response)


def fixture(h=128, w=160, owner_id=4, confidence=.8):
    yy, xx = np.mgrid[:h, :w]
    labels = np.zeros((h, w), np.int32); labels[12:-12, 12:-12] = owner_id
    alpha = (labels != 0).astype(np.float32)
    # Source-owned gentle photographic variation; no reference image texture
    # is assigned to the source, even when desired appearance changes.
    neutral = .18 + .002*np.sin(2*np.pi*xx/19) + .002*np.cos(2*np.pi*yy/31)
    current = neutral[..., None]*np.array([.95, 1., 1.05], np.float32)
    current = current.astype(np.float32)
    maps = {'ref_x': xx.astype(np.float32), 'ref_y': yy.astype(np.float32),
            'confidence': np.full((h, w), confidence, np.float32), 'valid': labels != 0}
    return current, labels, alpha, maps


def reference_for(current, labels, values):
    field = np.zeros((*labels.shape, 4), np.float32)
    field[..., :3] = values
    field[labels == 0] = 0
    field[..., 3] = labels != 0
    return apply_scene_balance(current, field)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    cases = []
    def run(name, current, labels, alpha, maps, reference, *, reference_labels=None, expected=None):
        copies = [v.copy() for v in (current, labels, alpha, reference)]
        map_copies = {k: v.copy() for k, v in maps.items()}
        field, report = prepare_environment_field(current, current, reference, labels,
            labels if reference_labels is None else reference_labels, alpha, maps)
        allowed = np.isin(labels, (4, 1, 13, 23)) & (alpha >= CONFIG['opaque_alpha']) & maps['valid']
        invariants = {'finite': bool(np.isfinite(field).all()),
            'unknown_face_background_alpha_boundary_zero': bool(np.all(field[~allowed] == 0)),
            'ev_bounds': bool(field[..., 0].min() >= CONFIG['minimum_ev']-1e-6 and field[..., 0].max() <= CONFIG['maximum_ev']+1e-6),
            'ab_bound': bool(np.linalg.norm(field[..., 1:3], axis=-1).max() <= CONFIG['maximum_ab']+1e-6),
            'inputs_unchanged': bool(all(np.array_equal(a, b) for a, b in zip((current, labels, alpha, reference), copies))
                                     and all(np.array_equal(maps[k], v) for k, v in map_copies.items())),
            'coverage_not_confidence_dose': report['confidence_is_effect_dose'] is False,
            'visual_quality_not_claimed': report['visual_acceptance'] is False}
        for owner in report['owners']:
            if owner['state'] == 'prepared':
                validation = owner['spatial_validation']
                invariants['independent_'+str(owner['semantic_class_id'])] = bool(
                    validation['passed'] and validation['training_error_used_for_acceptance'] is False
                    and len(validation['folds']) == CONFIG['holdout_folds']
                    and all(v['target_values_used_by_fit'] is False and v['gamut_and_highlight_response_included']
                            for v in validation['folds']))
        changed = report['changed_pixels'] > 0
        row = {'name': name, 'changed': changed, 'expected': expected,
               'passed': bool(all(invariants.values()) and (expected is None or changed == expected)),
               'invariants': invariants, 'report': report}
        cases.append(row)
        return field, row

    current, labels, alpha, maps = fixture()
    run('already_matched_endpoint_is_identity', current, labels, alpha, maps, current.copy(), expected=False)
    reference = reference_for(current, labels, np.array([.2, .006, .009], np.float32))
    field, response = run('uniform_observed_warm_response', current, labels, alpha, maps, reference, expected=True)
    lower_maps = {k: v.copy() for k, v in maps.items()}; lower_maps['confidence'][:] = .2
    lower_field, lower = run('lower_confidence_is_data_weight_not_fixed_dose', current, labels, alpha,
                           lower_maps, reference, expected=True)
    response['invariants']['uniform_dose_invariant_to_lower_confidence'] = bool(np.max(np.abs(field-lower_field)) < 2e-5)
    response['passed'] &= all(response['invariants'].values())
    yy, xx = np.mgrid[:labels.shape[0], :labels.shape[1]]
    for sign in (1, -1):
        desired = np.zeros((*labels.shape, 3), np.float32)
        position = sign*(xx-labels.shape[1]/2)/(labels.shape[1]/2)
        desired[..., 0] = .22*np.maximum(position, 0)
        desired[..., 1] = .01*position
        desired[..., 2] = .015*position
        reference = reference_for(current, labels, desired)
        result, row = run('directional_response_sign_'+str(sign), current, labels, alpha, maps, reference, expected=True)
        left = (labels == 4) & (xx < 64); right = (labels == 4) & (xx > 96)
        row['invariants']['observed_direction_changes_output_side'] = bool(
            sign*(np.median(result[..., 2][right])-np.median(result[..., 2][left])) > .003)
        row['passed'] &= all(row['invariants'].values())
    desired = np.zeros((*labels.shape, 3), np.float32)
    desired[..., 0] = .15*np.abs((xx-labels.shape[1]/2)/(labels.shape[1]/2))
    desired[..., 1] = .006*np.sign(xx-labels.shape[1]/2)
    desired[..., 2] = .012*np.sign(xx-labels.shape[1]/2)
    run('independent_two_sided_response', current, labels, alpha, maps,
        reference_for(current, labels, desired), expected=True)
    empty = {k: v.copy() for k, v in maps.items()}; empty['valid'][:] = False
    run('no_correspondence_is_exact_zero', current, labels, alpha, empty, reference, expected=False)
    wrong_labels = np.where(labels != 0, 23, 0).astype(np.int32)
    run('different_semantic_owner_is_rejected', current, labels, alpha, maps, reference,
        reference_labels=wrong_labels, expected=False)
    unknown = {k: v.copy() for k, v in maps.items()}; unknown['valid'][45:85, 65:95] = False
    run('unknown_correspondence_hole_stays_zero', current, labels, alpha, unknown,
        reference_for(current, labels, np.array([.2, .006, .009], np.float32)), expected=True)
    boundary_alpha = alpha.copy(); boundary_alpha[12:16] = .7
    run('semiopaque_source_boundary_stays_zero', current, labels, boundary_alpha, maps,
        reference_for(current, labels, np.array([.2, .006, .009], np.float32)), expected=True)
    # Deliberately broken same-owner random geometry. Its noisy residual must
    # not validate at genuinely withheld blocks. Coherent wrong geometry is
    # an upstream independent correspondence problem, explicitly reported.
    desired = np.zeros((*labels.shape, 3), np.float32)
    pattern = np.where(xx < 80, -1., 1.)*np.where(yy < 64, -1., 1.)
    desired[..., 0] = .2*pattern; desired[..., 1] = .012*pattern; desired[..., 2] = .016*pattern
    broken = {k: v.copy() for k, v in maps.items()}
    selected = np.flatnonzero(labels.ravel() == 4); permutation = np.random.default_rng(82993).permutation(selected)
    for k in ('ref_x', 'ref_y'): broken[k].ravel()[selected] = maps[k].ravel()[permutation]
    run('shuffled_same_owner_correspondence_fails_spatial_holdout', current, labels, alpha, broken,
        reference_for(current, labels, desired), expected=False)
    # Original material contrast partitions the field solve even where Sapiens
    # assigns both panels the same clothing class. The shaded panel has no
    # reference response and must not acquire its neighbour's warmth.
    cloth, cl, ca, cm = fixture(owner_id=23)
    cloth[:, :80] *= .15; cloth[:, 80:] *= 2
    desired = np.zeros((*cl.shape, 3), np.float32)
    desired[:, 80:] = [.2, .008, .012]
    result, row = run('same_owner_black_white_boundary_and_occluded_dark_side', cloth, cl, ca, cm,
        reference_for(cloth, cl, desired), expected=True)
    dark = (cl == 23) & (xx < 76)
    row['invariants']['dark_material_does_not_receive_adjacent_response'] = bool(np.max(np.abs(result[dark, :3])) < 2e-5)
    row['passed'] &= all(row['invariants'].values())
    face_labels = labels.copy(); face_labels[12:36, 12:48] = 3
    run('face_and_background_never_receive_field', current, face_labels, alpha, maps,
        reference_for(current, labels, np.array([.2, .006, .009], np.float32)), expected=True)
    # Real pixel response, rather than the algebraic proposed EV/ab field,
    # drives the validation even near the output gamut/highlight boundary.
    bright = np.full_like(current, [.89, .93, .97])
    run('highlight_gamut_response_in_actual_holdout_operator', bright, labels, alpha, maps,
        reference_for(bright, labels, np.array([.2, .006, .009], np.float32)), expected=None)
    # Perturb every withheld target dramatically. A fit which has leaked even
    # the target filter's range guide or scale would change its prediction.
    domain = labels == 4; guide_lab = to_lab(current)
    source_low, _, _ = lowpass(guide_lab, domain)
    raw = to_lab(reference_for(current, labels, np.array([.2, .006, .009], np.float32)))
    held = domain & ((xx//8+2*(yy//8)) % CONFIG['holdout_folds'] == 0)
    training = domain & ~held
    poisoned = raw.copy(); poisoned[held] = [.95, .25, -.2]
    ra, oa = _training_residual(raw, source_low, training)
    rb, ob = _training_residual(poisoned, source_low, training)
    fa, sa = _fit(ra, np.where(training, .8, 0).astype(np.float32), domain, domain, guide_lab)
    fb, sb = _fit(rb, np.where(training, .8, 0).astype(np.float32), domain, domain, guide_lab)
    invariants = {'withheld_targets_do_not_enter_training_filter_or_guide': bool(np.array_equal(ra, rb) and oa == ob),
                  'withheld_targets_do_not_change_trained_predictions': bool(np.array_equal(fa, fb)),
                  'both_fits_converged': bool(sa['converged'] and sb['converged'])}
    cases.append({'name': 'withheld_target_perturbation_cannot_change_prediction', 'changed': False,
        'expected': False, 'passed': bool(all(invariants.values())), 'invariants': invariants, 'report': {'owners': []}})
    ids = (xx % 40)+40*yy; held = domain & (xx < 40)
    training, excluded = _split_reference_observations(domain, held, ids)
    invariants = {'duplicate_reference_pixels_excluded': excluded > 0,
                  'training_and_held_target_reference_ids_disjoint': bool(
                      np.intersect1d(np.unique(ids[training]), np.unique(ids[held])).size == 0)}
    cases.append({'name': 'duplicate_reference_observation_is_not_independent_training_data', 'changed': False,
        'expected': False, 'passed': bool(all(invariants.values())), 'invariants': invariants, 'report': {'owners': []}})
    rng = np.random.default_rng(7261)
    noisy_reference = np.clip(current+rng.normal(0, .015, current.shape).astype(np.float32), 0, 1)
    run('reference_sensor_noise_is_not_a_validated_environment_response', current, labels, alpha, maps,
        noisy_reference, expected=False)
    # The production analysis-grid limit, with disconnected material domains,
    # very sparse training anchors and an entirely withheld isolated island.
    # Numeric convergence is measured against Ax=b, not a visual error score.
    sy, sx = np.mgrid[:512, :336]
    domain = np.ones((512, 336), bool); domain[254:258] = False; domain[:, 166:170] = False
    domain[470:500, 290:325] = False; domain[480:488, 303:311] = True
    guide = np.zeros((512, 336, 3), np.float32); guide[..., 0] = np.where(sx < 168, .35, .7)
    sparse = domain & (sy % 40 == 5) & (sx % 40 == 5)
    confidence = np.where(sparse, .02, 0).astype(np.float32)
    residual = np.empty_like(guide)
    residual[sx < 168] = [.12, .006, .008]; residual[sx >= 168] = [.18, -.004, -.007]
    numeric_field, numeric = _fit(residual, confidence, domain, domain, guide)
    anchored = domain.copy(); anchored[480:488, 303:311] = False
    invariants = {'large_sparse_disconnected_grid_converged': numeric['converged'],
        'real_rhs_relative_residual_within_fixed_tolerance': bool(max(numeric['rhs_relative_residuals']) <= CONFIG['relative_tolerance']),
        'no_material_leakage_or_dose_shrink': bool(np.max(np.abs(numeric_field[anchored]-residual[anchored])) < 2e-5),
        'unanchored_withheld_component_explicit_zero': bool(numeric['unanchored_zero_solution_pixels'] == 64
            and np.all(numeric_field[480:488, 303:311] == 0)),
        'outside_domain_zero': bool(np.all(numeric_field[~domain] == 0))}
    cases.append({'name': '512x336_sparse_anchors_material_domains_and_unanchored_holdout', 'changed': True,
        'expected': True, 'passed': bool(all(invariants.values())), 'invariants': invariants,
        'numeric_solver': numeric, 'report': {'owners': []}})
    # Native weave/checkerboard texture must not modulate a proxy
    # low-frequency field. Coarse guidance edges remain protected.
    small_h, small_w = 8, 8
    small_guide = np.zeros((small_h, small_w, 3), np.float32)
    small_guide[:, small_w//2:, 0] = .45
    small_domain = np.ones((small_h, small_w), bool)
    values = np.zeros((small_h, small_w, 3), np.float32)
    values[:, :small_w//2, 2] = .02
    values[:, small_w//2:, 2] = .08
    full_domain = np.ones((64, 64), bool)
    textured = _guided_upsample(values, small_guide, small_domain, full_domain)
    invariants = {
        'policy_is_explicit': GUIDANCE_POLICY == 'validated_proxy_owner_edge_v2',
        'field_stays_low_frequency_inside_material': bool(np.max(np.abs(np.diff(textured[:, 8:24, 2], axis=1))) < .01),
        'coarse_edge_response_survives': bool(np.median(textured[:, 36:44, 2])-np.median(textured[:, 20:28, 2]) > .03),
    }
    cases.append({'name': 'proxy_only_upsampling_preserves_low_frequency_and_coarse_edges',
        'changed': False, 'expected': False, 'passed': bool(all(invariants.values())),
        'invariants': invariants, 'report': {'guidance_policy': GUIDANCE_POLICY, 'owners': []}})
    # Nonuniform observations and sparse held blocks exercise conductance;
    # a constant residual would be unchanged even with a broken guide.
    gy, gx = np.mgrid[:96, :96]; material = np.ones((96, 96), bool)
    residual = np.zeros((96, 96, 3), np.float32)
    residual[..., 0] = np.where(gx >= 48, .08+.1*gx/96+.025*np.sin(gy/11), 0)
    residual[..., 1] = np.where(gx >= 48, .006*np.sin(gy/21), 0)
    residual[..., 2] = np.where(gx >= 48, .01*gx/96, 0)
    confidence = np.where((gy//8+2*(gx//8)) % 5 != 0, .8, 0).astype(np.float32)
    source_guide = np.zeros_like(residual); source_guide[..., 0] = np.where(gx < 48, .25, .7)
    noisy_guide = source_guide+np.random.default_rng(9371).normal(0, .01, residual.shape).astype(np.float32)
    smooth_low = _macro_guidance(source_guide, material)
    grain_low = _macro_guidance(noisy_guide, material)
    fit_smooth, solve_smooth = _fit(residual, confidence, material, material, smooth_low)
    fit_grain, solve_grain = _fit(residual, confidence, material, material, grain_low)
    invariants = {
        'filtered_guides_are_finite': bool(np.isfinite(grain_low).all()),
        'nonflat_EV_prediction_stable_under_proxy_grain': bool(np.max(np.abs(fit_grain[..., 0]-fit_smooth[..., 0])) < .005),
        'nonflat_ab_prediction_stable_under_proxy_grain': bool(np.max(np.abs(fit_grain[..., 1:]-fit_smooth[..., 1:])) < .0005),
        'large_material_edge_does_not_import_adjacent_response': bool(np.all(fit_grain[gx < 48] == 0)),
        'both_guidance_solves_converged': bool(solve_smooth['converged'] and solve_grain['converged']),
    }
    cases.append({'name': 'nonflat_predictions_with_held_blocks_are_stable_under_proxy_grain',
        'changed': False, 'expected': False, 'passed': bool(all(invariants.values())),
        'invariants': invariants, 'report': {'guidance_policy': GUIDANCE_POLICY, 'owners': []}})
    # Reproduce the real failure mechanism: a narrow piping observation gets
    # positive EV amid negative fabric EV. It must not survive as bright
    # islands, but a large black/white material boundary must survive.
    h, w = 128, 160; domain = np.ones((h, w), bool)
    guide = np.zeros((h, w, 3), np.float32); guide[..., 0] = .72
    guide[:, :80, 0] = .25; guide[:, 110:112, 0] = .3
    values = np.zeros_like(guide); values[:, :80, 0] = .12
    values[:, 80:, 0] = -.2; values[::4, 110:112, 0] = .5
    macro = _macro_guidance(guide, domain)
    constrained, numeric = _regularize_response(values, domain, macro)
    invariants = {
        'narrow_piping_not_a_macro_material_partition': bool(np.max(np.abs(macro[:, 110:112, 0]-.72)) < 1e-6),
        'bright_piping_islands_removed': bool(np.max(constrained[8:-8, 106:116, 0]) < -.1),
        'broad_dark_material_retains_its_separate_response': bool(np.max(np.abs(constrained[:, :72, 0]-.12)) < 2e-5),
        'response_solver_converged': numeric['converged'],
    }
    cases.append({'name': 'macro_response_removes_piping_EV_islands_without_crossing_large_material_edge',
        'changed': False, 'expected': False, 'passed': bool(all(invariants.values())),
        'invariants': invariants, 'report': {'guidance_policy': GUIDANCE_POLICY, 'owners': []}})
    h, w = 192, 256; yy, xx = np.mgrid[:h, :w]
    source = np.full((h, w, 3), .42, np.float32)+(.005*np.sin(yy/40))[..., None]
    piping = abs(xx-(62+yy//8)) <= 1
    source[piping] = [.025, .07, .1]
    response = np.zeros((h, w, 4), np.float32)
    response[..., 0] = -.18-.04*yy/h; response[..., 1:3] = [.005, .007]; response[..., 3] = 1
    plain = np.full((h, w, 3), .42, np.float32)+(.005*np.sin(yy/40))[..., None]
    reference = apply_scene_balance(plain, response)
    offset = abs(xx-(67+yy//8)) <= 1
    reference[offset] = apply_scene_balance(source, response)[piping]
    labels = np.full((h, w), 23, np.int32); alpha = np.ones((h, w), np.float32)
    maps = {'ref_x': xx.astype(np.float32), 'ref_y': yy.astype(np.float32),
            'confidence': np.full((h, w), .8, np.float32), 'valid': np.ones((h, w), bool)}
    field, row = run('displaced_narrow_trim_cannot_reverse_the_macro_lighting_response',
                    source, labels, alpha, maps, reference, expected=True)
    row['invariants']['macro_darkening_has_no_positive_piping_islands'] = bool(np.max(field[..., 0][piping]) <= 1e-6)
    row['passed'] &= all(row['invariants'].values())
    domain = labels == 23; training = (yy//8+2*(xx//8)) % 5 != 0
    guide = _macro_guidance(to_lab(source), domain)
    current_low = to_lab(plain); target = to_lab(reference); poisoned = target.copy(); poisoned[~training] = [.99, .2, -.2]
    ra, _ = _training_residual(target, current_low, training, domain=domain, guide=guide)
    rb, _ = _training_residual(poisoned, current_low, training, domain=domain, guide=guide)
    invariants = {'macro_filter_held_targets_do_not_change_training_numerator_guide_or_support': bool(np.array_equal(ra, rb))}
    cases.append({'name': 'source_macro_domain_filter_does_not_read_withheld_reference_targets',
        'changed': False, 'expected': False, 'passed': bool(all(invariants.values())),
        'invariants': invariants, 'report': {'guidance_policy': GUIDANCE_POLICY, 'owners': []}})
    report = {'schema': 'subject_environment_field_smoke.v1', 'config': CONFIG,
              'cases': cases, 'passed': bool(all(row['passed'] for row in cases)),
              'limits': ['Analytic fixtures do not prove real-image artistic quality.',
                         'Spatial colour holdout cannot certify coherent but false correspondence geometry.']}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print(json.dumps({'passed': report['passed'], 'cases': len(cases),
        'failed': [{'name': row['name'], 'invariants': row['invariants'],
                    'owners': [{'id': o['semantic_class_id'], 'reason': o['reason'],
                                'improvement': o.get('spatial_validation', {}).get('improvement')}
                               for o in row['report']['owners']]} for row in cases if not row['passed']]}, ensure_ascii=False))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__': main()
