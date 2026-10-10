"""Independent CPU fixtures for measured source-owned subject detail.

The fixtures use fixed analytic structures and uniform algorithm parameters.
Results retain failed counterexamples; this suite never tunes producer limits.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt, convolve1d

from native_subject_detail import CONFIG, LUMA, qualified_review_windows, recover_bands, _bands, sha, diagnosed_endpoint_equivalence, measure_diagnosed_final


def _structure(size=384, angle_degrees=0, frequency_scale=1.):
    yy, xx = np.mgrid[:size, :size]
    xx, yy = xx/frequency_scale, yy/frequency_scale
    if angle_degrees:
        angle = np.deg2rad(angle_degrees)
        xx, yy = xx*np.cos(angle)+yy*np.sin(angle), -xx*np.sin(angle)+yy*np.cos(angle)
    log = (-1.5 + .06*np.sin(2*np.pi*(xx+.31*yy)/76)
           + .05*np.cos(2*np.pi*(yy-.23*xx)/38)
           + .045*np.sin(2*np.pi*(xx+.61*yy)/20)
           + .035*np.cos(2*np.pi*(yy+.21*xx)/10)
           + .025*np.sin(2*np.pi*(xx-.45*yy)/6))
    return log.astype(np.float32)


def _rgb(log, chroma=(1.12, .98, .86)):
    chroma = np.asarray(chroma, np.float32)
    chroma = chroma/(chroma@LUMA)
    return np.exp(log)[..., None]*chroma


def _diagnosis():
    checks = {key: {'status': 'pass'} for key in ('identity_makeup', 'skin_materials', 'scene_integration')}
    checks['native_detail'] = {'status': 'fail', 'cause': 'retouch_missing', 'correction_stage': 'native_retouch',
        'evidence': [{'basis': 'visible_pixels', 'panels': ['source', 'final']}]}
    row = {'pair_id': 'fixture', 'region': 'left_arm', 'scope': 'matched_source_final_native_pair_only',
           'native_box': [18, 24, 318, 340], 'source_panel': 'source', 'final_panel': 'final', 'checks': checks}
    return {'cosplay_post_review': {'status': 'reviewed', 'actual_api_evidence_valid': True,
                                  'local_observations': [row]}}


def _tensor_probe(field, support, derivative='central'):
    """Diagnostic only: evaluate fixed directional-structure alternatives."""
    records = []
    scales = [1., 2., 4., 8.]
    for band_index, band in enumerate(_bands(field, np.asarray(scales))):
        if derivative == 'fourth_order':
            kernel = np.asarray([1, -8, 0, 8, -1], np.float32)/12
            gy, gx = [convolve1d(band, kernel, axis=axis, mode='nearest') for axis in (0, 1)]
        elif derivative == 'gaussian_derivative':
            gy, gx = [gaussian_filter(field, scales[band_index], order=order, mode='nearest')
                      - gaussian_filter(field, scales[band_index+1], order=order, mode='nearest')
                      for order in ((1, 0), (0, 1))]
        else:
            gy, gx = np.gradient(band)
        values = []
        for y in range(0, support.shape[0], 16):
            for x in range(0, support.shape[1], 16):
                selected = support[y:y+16, x:x+16]
                if selected.sum() < 128:
                    continue
                def tensor(gyv, gxv, select):
                    a, b, c = np.mean(gxv[select]**2), np.mean(gyv[select]**2), np.mean(gxv[select]*gyv[select])
                    coherence = float(np.hypot(a-b, 2*c)/max(a+b, 1e-12))
                    angle = float(.5*np.arctan2(2*c, a-b))
                    return coherence, angle
                by, bx = gy[y:y+16, x:x+16], gx[y:y+16, x:x+16]
                coherence, angle = tensor(by, bx, selected)
                sub = []
                for sy in (0, 8):
                    for sx in (0, 8):
                        ss = selected[sy:sy+8, sx:sx+8]
                        if ss.sum() < 32:
                            continue
                        co, an = tensor(by[sy:sy+8, sx:sx+8], bx[sy:sy+8, sx:sx+8], ss)
                        sub.append((co, float(np.cos(2*(an-angle)))))
                values.append((coherence, min([p[0] for p in sub], default=0), min([p[1] for p in sub], default=-1)))
        array = np.asarray(values)
        records.append({'band': band_index, 'blocks': len(values),
            'coherence_quantiles': np.quantile(array[:, 0], [.1, .5, .9]).tolist(),
            'minimum_subblock_coherence_quantiles': np.quantile(array[:, 1], [.1, .5, .9]).tolist(),
            'minimum_direction_agreement_quantiles': np.quantile(array[:, 2], [.1, .5, .9]).tolist(),
            'coherence_above_point35_fraction': float(np.mean(array[:, 0] > .35)),
            'parent_coherence_above_point35_and_all_subblock_agreement_point8_fraction':
                float(np.mean((array[:, 0] > .35) & (array[:, 2] > .8))),
            'all_subblocks_coherence_above_point35_and_agreement_point8_fraction':
                float(np.mean((array[:, 1] > .35) & (array[:, 2] > .8)))})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    results = []
    log = _structure(); donor = _rgb(log)
    owner = np.zeros(log.shape, bool); owner[8:-8, 8:-8] = True
    allowed = np.ones(log.shape, bool)

    def case(name, src, target, expected_recovery, own=owner, permitted=allowed, native_short_side=384):
        before = target.copy(); frozen_source = src.copy(); frozen_owner = own.copy(); frozen_allowed = permitted.copy()
        after, weight, checks = recover_bands(src, target, permitted, own, native_short_side=native_short_side)
        sigmas = checks['sigmas']; erosion = 4*sigmas[-1]+1
        recovered = checks['changed_pixels'] > 0
        invariants = {'same_shape': after.shape == target.shape,
            'source_unchanged': bool(np.array_equal(src, frozen_source)),
            'target_unchanged': bool(np.array_equal(target, before)),
            'owner_unchanged': bool(np.array_equal(own, frozen_owner)),
            'permission_unchanged': bool(np.array_equal(permitted, frozen_allowed)),
            'unknown_unchanged': bool(np.array_equal(after[~own], target[~own])),
            'unreviewed_unchanged': bool(np.array_equal(after[~permitted], target[~permitted])),
            'owner_edge_unchanged': bool(np.array_equal(after[distance_transform_edt(own) <= erosion], target[distance_transform_edt(own) <= erosion])),
            'zero_weight_unchanged': bool(np.array_equal(after[weight <= 0], target[weight <= 0])),
            'finite_nonnegative': bool(np.isfinite(after).all() and np.all(after >= 0))}
        chroma_before = target/(target@LUMA)[..., None]
        chroma_after = after/(after@LUMA)[..., None]
        chroma_error = float(np.max(np.abs(chroma_after-chroma_before)))
        invariants['current_chroma_preserved'] = chroma_error < 1e-6
        safe = permitted & own & (distance_transform_edt(own) > erosion)
        low_drift = gaussian_filter(np.log(np.maximum(after@LUMA, 1e-5))
                                   -np.log(np.maximum(target@LUMA, 1e-5)), sigmas[-1], mode='nearest')
        low_drift_q95 = float(np.quantile(np.abs(low_drift[safe]), .95)) if safe.any() else 0.
        invariants['current_coarse_luminance_preserved'] = low_drift_q95 < .004
        if not recovered:
            invariants['rejection_preserves_all_pixels'] = bool(np.array_equal(after, target) and not weight.any())
        else:
            invariants['recovery_has_both_measured_band_qualifications'] = bool(
                len(checks.get('band_diagnostics', [])) == 2 and all(item['accepted_loss'] for item in checks['band_diagnostics']))
            invariants['recovery_improves_measured_bands'] = bool(checks['after_band_error'] < checks['before_band_error']*.95)
        row = {'name': name, 'expected_recovery': expected_recovery, 'recovered': recovered,
               'passed': bool((expected_recovery is None or recovered == expected_recovery) and all(invariants.values())),
               'invariants': invariants, 'maximum_chroma_error': chroma_error,
               'independent_low_log_drift_q95': low_drift_q95, 'checks': checks}
        results.append(row)
        return after, weight, row

    blur = np.stack([gaussian_filter(donor[..., c], 1.8) for c in range(3)], axis=-1)
    case('structured_blur_recovery', donor, blur, True)
    case('structured_blur_after_global_colour_and_exposure', donor,
         blur*np.array([.72, 1.09, 1.27], np.float32)*.73, True)
    for angle in (17, 43, 90, 180):
        rotated = _rgb(_structure(angle_degrees=angle))
        rotated_blur = np.stack([gaussian_filter(rotated[..., c], 1.8) for c in range(3)], axis=-1)
        case('structured_blur_rotation_'+str(angle), rotated, rotated_blur, True)
    case('unchanged_structure', donor, donor.copy(), False)
    case('pure_exposure', donor, donor*.73, False)
    case('global_rgb_grading_constant_chroma', donor, donor*np.array([.72, 1.09, 1.27], np.float32), False)
    case('uniform_log_tone_contrast_compression', donor, _rgb(.64*log-.5), False)
    case('constant_source', np.full_like(donor, .2), np.full_like(donor, .13), False)

    rng = np.random.default_rng(82993)
    noise_log = -1.5 + rng.normal(0, .045, log.shape).astype(np.float32)
    noise_src = _rgb(noise_log)
    noise_target = np.stack([gaussian_filter(noise_src[..., c], 1.8) for c in range(3)], axis=-1)
    case('white_camera_noise_donor', noise_src, noise_target, False)
    # Sensor demosaicing or previous denoising can correlate camera noise.
    # There is still no deterministic scene texture in this fixture.
    correlated_noise = gaussian_filter(rng.normal(0, .08, log.shape).astype(np.float32), 1)
    case('correlated_camera_noise_donor', _rgb(-1.5+correlated_noise),
         _rgb(-1.5+gaussian_filter(correlated_noise, 1.8)), False)
    for seed in (486, 1027, 16803):
        random_noise = np.random.default_rng(seed).normal(0, .08, log.shape).astype(np.float32)
        for sigma in (.7, 1., 1.5, 2.):
            correlated = gaussian_filter(random_noise, sigma)
            case('correlated_camera_noise_seed_'+str(seed)+'_sigma_'+str(sigma), _rgb(-1.5+correlated),
                 _rgb(-1.5+gaussian_filter(correlated, 1.8)), False)
    blurred_log = gaussian_filter(log, 6)
    case('blurred_donor_without_fine_structure', _rgb(blurred_log), _rgb(gaussian_filter(blurred_log, 4)), False)

    # Fine detail can be coloured while low-frequency shape is neutral. A
    # global colour transform changes its luminance without filtering texture.
    yy, xx = np.mgrid[:384, :384]
    coarse = -.0 + .05*np.sin(2*np.pi*(xx+.31*yy)/76) + .045*np.cos(2*np.pi*(yy-.23*xx)/38)
    fine = .045*np.sin(2*np.pi*(xx+.61*yy)/20) + .035*np.cos(2*np.pi*(yy+.21*xx)/10) + .025*np.sin(2*np.pi*(xx-.45*yy)/6)
    coloured = np.repeat(np.exp(-1.5+coarse)[..., None], 3, -1).astype(np.float32)
    coloured[..., 0] *= np.exp(3*fine).astype(np.float32)
    case('global_rgb_grading_spatial_chroma', coloured, coloured*np.array([.25, 1.15, 1.0], np.float32), False)
    slow_exposure = (.8+.01*np.sin(2*np.pi*xx/300)).astype(np.float32)
    case('spatial_chroma_grading_and_slow_exposure', coloured,
         coloured*np.array([.25,1.15,1.0],np.float32)*slow_exposure[...,None], False)
    case('structured_blur_after_grading_and_slow_exposure', donor,
         blur*np.array([.72,1.09,1.27],np.float32)*slow_exposure[...,None], True)

    # A large photograph uses the largest *geometrically* measurable scale.
    # Measurements do not trigger a second chance at a smaller frequency.
    thin = np.zeros_like(owner); thin[8:-8,122:262] = True
    scaled_donor = _rgb(_structure(frequency_scale=1.546875))
    scaled_blur = np.stack([gaussian_filter(scaled_donor[...,c],1.8*1.546875) for c in range(3)],-1)
    _, _, thin_blur = case('native6336_thin_owner_multi_axis_structure_rejected', scaled_donor, scaled_blur, False,
                          own=thin, native_short_side=6336)
    parallel_log = np.full_like(log, -1.5)
    for amplitude, wavelength in ((.06,76),(.05,38),(.045,20),(.035,10),(.025,6)):
        parallel_log += amplitude*np.sin(2*np.pi*(xx+.31*yy)/(wavelength*1.546875))
    parallel_donor = _rgb(parallel_log)
    parallel_blur = np.stack([gaussian_filter(parallel_donor[...,c],3*1.546875) for c in range(3)],-1)
    case('native6336_thin_owner_directionally_stable_blur', parallel_donor, parallel_blur, True,
         own=thin, native_short_side=6336)
    _, _, thin_same = case('native6336_thin_owner_unchanged', donor, donor.copy(), False,
                          own=thin, native_short_side=6336)
    _, _, thin_colour = case('native6336_thin_owner_colour_exposure', coloured,
         coloured*np.array([.25,1.15,1.0],np.float32)*slow_exposure[...,None], False,
         own=thin, native_short_side=6336)
    case('native6336_thin_owner_white_noise', noise_src, noise_target, False,
         own=thin, native_short_side=6336)
    case('native6336_thin_owner_correlated_noise', _rgb(-1.5+correlated_noise),
         _rgb(-1.5+gaussian_filter(correlated_noise,1.8)), False, own=thin, native_short_side=6336)
    floor_owner = np.zeros_like(owner); floor_owner[8:-8,153:231] = True
    _, _, floor_noise = case('native6336_floor_scale_white_noise_rejected', noise_src, noise_target, False,
                             own=floor_owner, native_short_side=6336)
    floor_noise['invariants']['floor_is_measurable_without_noise_authority'] = (
        floor_noise['checks']['scale_selection']['ready'] and floor_noise['checks']['sigmas'][0]==1.)
    floor_noise['passed'] &= all(floor_noise['invariants'].values())
    narrow = np.zeros_like(owner); narrow[8:-8,168:216] = True
    _, _, too_narrow = case('native6336_unmeasurable_at_floor', donor, blur, False,
                           own=narrow, native_short_side=6336)
    tiny_allowed = np.zeros_like(owner); tiny_allowed[180:192,180:192] = True
    _, _, too_few_blocks = case('native6336_enough_pixels_not_enough_blocks', donor, blur, False,
                               permitted=tiny_allowed, native_short_side=6336)
    geometries = [row['checks']['scale_selection'] for row in (thin_blur,thin_same,thin_colour)]
    thin_blur['invariants']['largest_ready_scale_selected'] = geometries[0]['selected_scale'] == 1.546875
    thin_same['invariants']['target_does_not_affect_scale'] = all(row == geometries[0] for row in geometries)
    thin_same['invariants']['no_rescale_after_no_loss'] = not geometries[1]['retry_after_measurement']
    too_narrow['invariants']['unmeasured_is_not_no_loss'] = too_narrow['checks']['measurement_state']=='not_measured'
    too_few_blocks['invariants']['spatial_blocks_gate_readiness'] = not too_few_blocks['checks']['scale_selection']['ready']
    for row in (thin_blur,thin_same,too_narrow,too_few_blocks):
        row['passed'] &= all(row['invariants'].values())

    region_owner = owner.copy(); region_owner[150:230, 150:230] = False
    reviewed = np.zeros(log.shape, bool); reviewed[32:352, 28:356] = True
    # Whether independent support qualifies or conservatively rejects, owner
    # holes, unreviewed pixels and protected owner edges must remain exact.
    case('unknown_owner_hole_and_review_permission', donor, blur, None,
         own=region_owner, permitted=reviewed)
    case('owner_interior_too_narrow', donor, blur, False, own=np.pad(np.ones((384, 48), bool), ((0, 0), (168, 168))))
    case('unknown_ownership_everywhere', donor, blur, False, own=np.zeros(log.shape, bool))

    # The donor is the frozen approved baseline. A raw blemish omitted by that
    # baseline must never become available for recovery. This checks recover's
    # dataflow, rather than pretending recover itself validates the binding.
    raw_blemish = .12*np.exp(-((xx-192)**2+(yy-192)**2)/(2*3**2)).astype(np.float32)
    raw = _rgb(log-raw_blemish)
    approved = donor.copy()
    approved_after, _, row = case('approved_retouch_donor_excludes_raw_blemish', approved, blur, True)
    raw_after, _, raw_checks = recover_bands(raw, blur, allowed, owner, native_short_side=384)
    row['raw_blemish_not_available'] = bool(np.array_equal(approved, donor))
    row['approved_result_is_independent_of_unused_raw'] = bool(np.max(np.abs(approved_after-raw_after)) > 0)
    row['passed'] &= row['raw_blemish_not_available'] and row['approved_result_is_independent_of_unused_raw']

    diagnosis = _diagnosis()
    good = qualified_review_windows(diagnosis)
    qualification = {'valid_semantic_window': len(good) == 1,
                     'dynamic_review_coordinates_preserved': good[0]['native_box'] == [18, 24, 318, 340]}
    dim_target = _rgb(.95*log)*.01
    dim_final = np.stack([gaussian_filter(dim_target[...,c],3) for c in range(3)],-1)
    qualification['dark_downstream_blur_not_hidden_by_small_absolute_rgb_error'] = (
        np.max(np.abs(dim_final-dim_target)) < 5e-4
        and not diagnosed_endpoint_equivalence(dim_target,dim_final,owner)['matches'])
    qualification['same_float_endpoint_matches'] = diagnosed_endpoint_equivalence(dim_target,dim_target.copy(),owner)['matches']
    qualification['absent_endpoint_support_cannot_prove_equivalence'] = not diagnosed_endpoint_equivalence(
        dim_target,dim_final,np.zeros_like(owner))['matches']
    very_dark = np.full_like(donor,1e-7)
    qualification['dark_values_below_measurement_floor_still_checked'] = not diagnosed_endpoint_equivalence(
        very_dark,very_dark*2,owner)['matches']
    qualification['zero_channel_mismatch_not_floored_away'] = not diagnosed_endpoint_equivalence(
        np.zeros_like(donor),very_dark,owner)['matches']
    fixed_sigmas = [1.,2.,4.,8.]
    actual_same = measure_diagnosed_final(donor*.01,dim_target,allowed,owner,sigmas=fixed_sigmas)
    actual_blur = measure_diagnosed_final(donor*.01,dim_final,allowed,owner,sigmas=fixed_sigmas)
    qualification['actual_final_preservation_measured_at_fixed_scale'] = actual_same['preserved']
    qualification['actual_dark_final_blur_cannot_inherit_appearance_preservation'] = not actual_blur['preserved']
    qualification['actual_final_measurement_never_reselects_scale_or_restores'] = (
        not actual_same['scale_reselected'] and not actual_blur['pixel_restoration_attempted'])
    center = np.zeros_like(owner); center[120:264,120:264] = True
    halo_final = donor.copy(); halo_final[~center] *= .05
    halo_measurement = measure_diagnosed_final(donor,halo_final,center,owner,sigmas=fixed_sigmas)
    halo_control = measure_diagnosed_final(donor,donor,center,owner,sigmas=fixed_sigmas)
    qualification['actual_final_reads_filter_halo_even_when_centers_match'] = (
        diagnosed_endpoint_equivalence(donor,halo_final,center)['matches']
        and halo_measurement['band_diagnostics'] != halo_control['band_diagnostics'])
    for mutation in ('unreviewed', 'unevidenced', 'unknown_scene', 'unknown_skin', 'unknown_identity', 'uncited', 'unknown_region', 'noninteger_box', 'wrong_stage', 'wrong_cause'):
        changed = copy.deepcopy(diagnosis); review = changed['cosplay_post_review']; item = review['local_observations'][0]
        if mutation == 'unreviewed': review['status'] = 'unknown'
        elif mutation == 'unevidenced': review['actual_api_evidence_valid'] = False
        elif mutation.startswith('unknown_') and mutation != 'unknown_region':
            item['checks'][{'unknown_scene':'scene_integration','unknown_skin':'skin_materials','unknown_identity':'identity_makeup'}[mutation]]['status'] = 'unknown'
        elif mutation == 'uncited': item['checks']['native_detail']['evidence'] = []
        elif mutation == 'unknown_region': item['region'] = 'unknown_skin_colour'
        elif mutation == 'noninteger_box': item['native_box'][0] = 18.5
        elif mutation == 'wrong_stage': item['checks']['native_detail']['correction_stage'] = 'reference_generation'
        elif mutation == 'wrong_cause': item['checks']['native_detail']['cause'] = 'composition_defect'
        qualification[mutation+'_rejected'] = not qualified_review_windows(changed)
    qualification['passed'] = all(qualification.values())
    report = {'schema': 'native_subject_detail_independent_smoke.v1', 'producer_config': CONFIG,
              'producer_sha256': sha(Path(__file__).resolve().parents[1]/'scripts/native_subject_detail.py'),
              'smoke_sha256': sha(__file__),
              'cases': results, 'qualification': qualification,
              'raw_donor_probe': {'recover_bands_is_not_an_asset_binding_validator': True,
                  'different_raw_donor_produces_different_result': bool(np.max(np.abs(approved_after-raw_after)) > 0),
                  'raw_input_reason': raw_checks['reason'], 'upstream_binding_test_still_required': True},
              'structure_tensor_probe': {
                  'fixed_analytic_structure': _tensor_probe(log, owner & (distance_transform_edt(owner) > 33)),
                  'correlated_camera_noise': _tensor_probe(-1.5+correlated_noise, owner & (distance_transform_edt(owner) > 33))},
              'rotation_derivative_probe': {str(angle): {mode: _tensor_probe(_structure(angle_degrees=angle),
                  owner & (distance_transform_edt(owner) > 33), derivative=mode)
                  for mode in ('central', 'fourth_order', 'gaussian_derivative')} for angle in (0, 17, 43)},
              'passed': bool(all(row['passed'] for row in results) and qualification['passed']),
              'limits': ['Synthetic fixtures do not establish real-image visual quality.',
                         'Passing these noise fixtures does not prove rejection of arbitrary directional or correlated camera noise.',
                         'Frozen approved retouch asset binding is validated upstream, not by recover_bands.',
                         'No alpha input or resampling operation exists in recover_bands; production geometry and alpha require integration tests.']}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print(json.dumps({'passed': report['passed'], 'output': str(args.output) if args.output else None,
                      'case_count': len(results), 'recovered_case_count': sum(row['recovered'] for row in results),
                      'failures': [{'name': row['name'], 'passed': row['passed'], 'reason': row['checks']['reason'],
                                 'changed_pixels': row['checks']['changed_pixels'], 'bands': row['checks'].get('band_diagnostics')}
                                for row in results if not row['passed']],
                      'qualification_passed': qualification['passed']}, ensure_ascii=False))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
