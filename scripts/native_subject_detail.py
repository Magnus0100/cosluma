"""Measured source-owned detail recovery, frozen before manual foreground edits.

Vision localizes an observation; it cannot establish that a filter removed
texture. Two adjacent structured luminance bands must independently lose
contrast relative to the coarse tone response. Unchanged texture, source noise,
unknown ownership and downstream effects cannot authorize a new pixel layer.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, convolve

SCHEMA = 'native_subject_detail.v1'
RECIPE = 'source_owned_geometry_scaled_log_luminance_bands_v2'
KIND = 'source_subject_detail'
DOMAIN = 'display_linear_sRGB'
LUMA = np.array([.2126, .7152, .0722], np.float32)
CONFIG = {'native_scale_reference': 2048, 'minimum_sigma': 1.,
          'band_sigmas': [1., 2., 4., 8.], 'minimum_owner_pixels': 128,
          'minimum_spatial_blocks': 4, 'block_sigma_multiple': 8.,
          'minimum_block_pixels': 32, 'loss_ratio_upper_quantile': .9,
          'maximum_relative_fine_gain': .9, 'minimum_structure_correlation': .7,
          'minimum_band_noise_ratio': 3., 'minimum_band_rms': .001,
          'minimum_structure_coherence': .35, 'minimum_subblock_axis_cosine': .8,
          'minimum_coherent_block_fraction': .25,
          'opaque_alpha': .995, 'maximum_native_roi_pixels': 2_000_000,
          'maximum_log_correction': .12, 'maximum_low_log_drift_q95': .004,
          'minimum_error_improvement': .05}
CONFIG.update(scale_selection='largest_source_geometry_ready_power_of_two',
              photometric_exposure_band_rms_fraction=.1,
              maximum_endpoint_log_difference=1e-6,
              diagnostic_preserved_gain_range=[.9, 1.1],
              diagnostic_preserved_correlation=.98)
# These are photographic location owners, not bare-skin/fabric classification.
# No colour threshold or reference mask supplies ownership.
REGIONS = {'left_hand': (6,), 'right_hand': (15,),
           'left_arm': (7, 11), 'right_arm': (16, 20),
           'upper_clothing': (1, 23), 'lower_clothing': (13,), 'hair': (4,)}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def array_sha(values):
    from native_retouch_producer import array_sha as digest
    return digest(values)


def qualified_review_windows(run):
    review = run.get('cosplay_post_review') or {}
    if review.get('status') != 'reviewed' or review.get('actual_api_evidence_valid') is not True:
        return []
    result = []
    for row in review.get('local_observations') or []:
        checks = row.get('checks') or {}; detail = checks.get('native_detail') or {}
        box = row.get('native_box')
        panels = {row.get('source_panel'), row.get('final_panel')}
        cited = {panel for ev in detail.get('evidence') or [] if ev.get('basis') == 'visible_pixels'
                 for panel in ev.get('panels') or []}
        if (row.get('region') not in REGIONS or row.get('scope') != 'matched_source_final_native_pair_only'
                or detail.get('status') != 'fail' or detail.get('cause') != 'retouch_missing'
                or detail.get('correction_stage') != 'native_retouch'
                or any((checks.get(key) or {}).get('status') != 'pass'
                       for key in ('identity_makeup', 'skin_materials', 'scene_integration'))
                or not isinstance(box, list) or len(box) != 4
                or any(type(value) is not int for value in box) or None in panels or not panels <= cited):
            continue
        result.append({'pair_id': row['pair_id'], 'region': row['region'], 'native_box': box.copy(),
                       'source_panel': row['source_panel'], 'final_panel': row['final_panel'],
                       'owner_class_ids': list(REGIONS[row['region']]), 'failed_criteria': ['native_detail']})
    return result


def validate_diagnosed_plate(root, run):
    """Require the exact saved pair and appearance endpoint, not just a source ID."""
    from native_receiver_detail import validate_diagnosed_plate as validate_pair
    from reference_appearance_stack import validate_layer
    from source_semantic_binding import load_source_semantics
    result = validate_pair(root, run)
    plan = run['options']['appearance_revision']['design_reconstruction_plan']
    saved = read(Path(root)/'outputs/runs'/run['run_id']/'manifest.json')
    for key in ('denoise_strength', 'manual_layers'):
        if run.get('options', {}).get(key) != saved.get('options', {}).get(key):
            raise ValueError('Subject detail needs the saved foreground controls')
    if run['options']['appearance_revision'] != saved['options']['appearance_revision']:
        raise ValueError('Subject detail foreground edits differ from the diagnosed endpoint')
    stack = next(row for row in plan['layers'] if row['kind'] == 'reference_appearance_stack')
    validate_layer(stack, plan, run)
    metadata = read(result['metadata']['path'])
    executed = next(row for row in metadata['operator']['design_reconstruction_plan']['layers']
                    if row['id'] == stack['id'])
    from design_reconstruction import reference_stack_execution_weights
    weights = reference_stack_execution_weights(stack)
    if executed.get('assets') != stack['assets'] or executed.get('bank_weights') != weights['bank_weights']:
        raise ValueError('Subject detail appearance differs from the native render actually reviewed')
    semantics = plan['mask_roles']['source_semantics']
    load_source_semantics(semantics, source_sha256=run['source']['sha256'],
        native_size=plan['canvas']['source_oriented_size'], proxy_size=plan['canvas']['source_proxy_size'])
    observed = run['cosplay_post_review']['native_selection']
    if observed.get('semantic_binding', {}).get('sha256') != semantics['sha256']:
        raise ValueError('Subject owner labels differ from the actual native QA')
    assets = [*plan['source_baseline']['assets'].values(), semantics,
              plan['mask_roles']['foreground_alpha'], *stack['assets'].values()]
    if plan['mask_roles'].get('identity_protection'):
        assets.append(plan['mask_roles']['identity_protection'])
    for asset in assets:
        if sha(asset['path']) != asset['sha256']:
            raise ValueError('Subject detail donor/owner/appearance asset changed')
    result['evidence_assets'] = [record(asset['path']) for asset in assets]
    result.update(source_owner_verified=True, frozen_approved_donor_only=True)
    return result


def _noise_estimate(field, support):
    # Diagonal 2x2 difference cancels a locally affine exposure field. Its L2
    # kernel norm is 2; structured texture inflates this estimate conservatively.
    kernel = np.array([[1., -1.], [-1., 1.]], np.float32)
    residual = convolve(field, kernel, mode='nearest')[support]
    return float(np.median(np.abs(residual-np.median(residual)))/(.67448975*2))


def _dog_noise_gain(s1, s2):
    radius = int(np.ceil(4*s2)); xx = np.arange(-radius, radius+1, dtype=float)
    kernels = []
    for sigma in (s1, s2):
        k = np.exp(-xx*xx/(2*sigma*sigma)); k /= k.sum(); kernels.append(np.outer(k, k))
    return float(np.sqrt(np.sum((kernels[0]-kernels[1])**2)))


def _bands(field, sigmas):
    blurred = [gaussian_filter(field, sigma, mode='nearest', truncate=4) for sigma in sigmas]
    return [blurred[i]-blurred[i+1] for i in range(len(blurred)-1)]


def select_geometry_scale(allowed, owner, *, native_short_side):
    """Choose once using source ownership, never target pixels or measured loss."""
    initial = max(CONFIG['minimum_sigma'], native_short_side/CONFIG['native_scale_reference'])
    scales = [initial]
    while scales[-1] > CONFIG['minimum_sigma']:
        scales.append(max(CONFIG['minimum_sigma'], scales[-1]/2))
    distance = distance_transform_edt(owner)
    candidates = []; selected = None; selected_support = None
    for scale in scales:
        sigmas = np.array(CONFIG['band_sigmas'])*scale
        support = allowed & owner & (distance > 4*sigmas[-1]+1)
        block = max(16, int(np.ceil(scale*CONFIG['block_sigma_multiple'])))
        blocks = sum(int(support[y:y+block,x:x+block].sum()) >= CONFIG['minimum_block_pixels']
                     for y in range(0,support.shape[0],block) for x in range(0,support.shape[1],block))
        ready = int(support.sum()) >= CONFIG['minimum_owner_pixels'] and blocks >= CONFIG['minimum_spatial_blocks']
        candidates.append({'scale': float(scale), 'permitted_pixels': int(support.sum()),
                           'spatial_blocks': int(blocks), 'block_size': block, 'geometry_ready': bool(ready)})
        if ready and selected is None:
            selected = scale; selected_support = support
    if selected is None:
        selected = scales[-1]
        selected_support = allowed & owner & (distance > 32*selected+1)
    selection = {'initial_scale': float(initial), 'selected_scale': float(selected),
        'ready': any(row['geometry_ready'] for row in candidates), 'candidates': candidates,
        'basis': 'source_owner_alpha_protection_and_review_window_only',
        'selected_before_target_noise_or_loss': True, 'retry_after_measurement': False,
        'scope': 'selected_frequency_bands_only; larger_unmeasurable_bands_remain_unknown'}
    return selected, selected_support, distance, selection


def _pointwise_photometric_explanation(donor, target, support, sigmas, block):
    """Refuse recovery for exact channel grading with negligible fine exposure.

    This is an exclusion proof, never permission to restore pixels. A shared
    exposure ratio from a real blur contains the removed high frequencies and
    must fail this proof even when RGB reconstruction is exact.
    """
    if not np.all(donor[support] > 1e-4) or not np.all(target[support] > 1e-5):
        return None
    ratio = target/np.maximum(donor, 1e-8)
    exposure = ratio[...,1]
    if np.any(exposure[support] <= 0):
        return None
    gains = [float(np.median(ratio[...,c][support]/exposure[support])) for c in range(3)]
    reconstructed = donor*np.asarray(gains, np.float32)*exposure[...,None]
    if np.max(np.abs(target[support]-reconstructed[support]), initial=0) > 2e-6:
        return None
    bands = _bands(np.log(np.maximum(exposure, 1e-8)), sigmas)[:2]
    maximum_rms = []
    for band in bands:
        values = []
        for y in range(0,support.shape[0],block):
            for x in range(0,support.shape[1],block):
                sl = np.s_[y:y+block,x:x+block]; keep = support[sl]
                if keep.any():
                    values.append(float(np.sqrt(np.mean(band[sl][keep].astype(float)**2))))
        maximum_rms.append(max(values, default=0.))
    limit = CONFIG['minimum_band_rms']*CONFIG['photometric_exposure_band_rms_fraction']
    if any(value >= limit for value in maximum_rms):
        return None
    return {'relative_channel_gains': gains, 'maximum_exposure_fine_band_rms': maximum_rms,
            'exclusion_rms_limit': limit, 'maximum_rgb_reconstruction_error':
            float(np.max(np.abs(target[support]-reconstructed[support]), initial=0))}


def _block_measurements(source_bands, target_bands, support, *, block, noise, gradients):
    rows = []
    def axis(gx, gy, selected):
        if selected.sum() < 8:
            return 0., np.zeros(2)
        x, y = gx[selected].astype(float), gy[selected].astype(float)
        a, b, c = float(x@x), float(y@y), float(x@y)
        direction = np.array([a-b, 2*c]); length = np.linalg.norm(direction)
        return float(length/max(a+b, 1e-12)), direction/max(length, 1e-12)
    for y in range(0, support.shape[0], block):
        for x in range(0, support.shape[1], block):
            sl = np.s_[y:y+block, x:x+block]; selected = support[sl]
            if selected.sum() < CONFIG['minimum_block_pixels']:
                continue
            measures = []
            for index, (sb, tb) in enumerate(zip(source_bands, target_bands)):
                a, b = sb[sl][selected].astype(float), tb[sl][selected].astype(float)
                # Robustly trim strong isolated edges, never classify their colour.
                keep = np.abs(a-np.median(a)) <= max(np.quantile(np.abs(a-np.median(a)), .95), 1e-8)
                a, b = a[keep], b[keep]; a -= a.mean(); b -= b.mean()
                energy = float(a@a); rms = float(np.sqrt(energy/max(1, len(a))))
                gain = float(a@b/max(energy, 1e-12))
                corr = float(a@b/max(np.sqrt(energy*(b@b)), 1e-12))
                gy, gx = [gradient[sl] for gradient in gradients[index]]
                coherence, direction = axis(gx, gy, selected)
                alignment = []
                for yy in (0, block//2):
                    for xx in (0, block//2):
                        quarter = np.s_[yy:yy+block//2, xx:xx+block//2]
                        _, local_axis = axis(gx[quarter], gy[quarter], selected[quarter])
                        alignment.append(float(direction@local_axis))
                coherent = (coherence >= CONFIG['minimum_structure_coherence']
                            and min(alignment) >= CONFIG['minimum_subblock_axis_cosine'])
                measures.append({'gain': gain, 'correlation': corr, 'source_rms': rms,
                    'direction_coherence': coherence, 'subblock_axis_alignment_min': min(alignment),
                    'coherent_structure': bool(coherent),
                    'structured': rms >= max(CONFIG['minimum_band_rms'],
                        CONFIG['minimum_band_noise_ratio']*noise[index])})
            coarse = measures[-1]
            rows.append({'box': [x, y, min(x+block, support.shape[1]), min(y+block, support.shape[0])],
                         'bands': measures, 'coarse_gain': coarse['gain']})
    return rows


def _measure_frequency_response(donor, target, support, sigmas):
    """Shared fixed-scale measurements; this function cannot select a scale."""
    source_log, target_log = [np.log(np.maximum(rgb@LUMA, 1e-5)) for rgb in (donor, target)]
    sb, tb = _bands(source_log, sigmas), _bands(target_log, sigmas)
    noise = _noise_estimate(source_log, support)
    predicted = [noise*_dog_noise_gain(a, b) for a, b in zip(sigmas[:-1], sigmas[1:])]
    block = max(16, int(np.ceil(sigmas[0]*CONFIG['block_sigma_multiple'])))
    gradients = [tuple(gaussian_filter(source_log, a, order=order, mode='nearest', truncate=4)
                       -gaussian_filter(source_log, b, order=order, mode='nearest', truncate=4)
                       for order in ((1,0),(0,1))) for a,b in zip(sigmas[:-1],sigmas[1:])]
    blocks = _block_measurements(sb, tb, support, block=block, noise=predicted, gradients=gradients)
    valid = [row for row in blocks if row['bands'][-1]['structured']
             and .25 <= row['coarse_gain'] <= 4.
             and row['bands'][-1]['correlation'] >= CONFIG['minimum_structure_correlation']]
    diagnostics = []; qualified_bands = []
    for index in range(2):
        structured = [row for row in valid if row['bands'][index]['structured']
                and row['bands'][index]['correlation'] >= CONFIG['minimum_structure_correlation']]
        used = [row for row in structured if row['bands'][index]['coherent_structure']]
        ratios = [row['bands'][index]['gain']/row['coarse_gain'] for row in used]
        upper = float(np.quantile(ratios, CONFIG['loss_ratio_upper_quantile'])) if ratios else None
        fraction = len(used)/max(1, len(structured))
        measured_ratios = [row['bands'][index]['gain']/row['coarse_gain'] for row in structured]
        correlations = [row['bands'][index]['correlation'] for row in structured]
        lower_gain, median_gain, upper_gain = (np.quantile(measured_ratios, [.1,.5,.9]).tolist()
                                              if measured_ratios else [None]*3)
        median_correlation = float(np.median(correlations)) if correlations else None
        low, high = CONFIG['diagnostic_preserved_gain_range']
        preserved = (len(structured) >= CONFIG['minimum_spatial_blocks'] and lower_gain >= low
                     and upper_gain <= high and median_correlation > CONFIG['diagnostic_preserved_correlation'])
        accepted = (len(used) >= CONFIG['minimum_spatial_blocks'] and upper < CONFIG['maximum_relative_fine_gain']
                    and fraction >= CONFIG['minimum_coherent_block_fraction'])
        diagnostics.append({'band': index, 'structured_blocks': len(structured), 'coherent_blocks': len(used),
                            'coherent_fraction': fraction, 'relative_gain_upper': upper,
                            'measured_relative_gain_q10': lower_gain, 'measured_relative_gain_median': median_gain,
                            'measured_relative_gain_q90': upper_gain, 'measured_correlation_median': median_correlation,
                            'measured_band_preserved': bool(preserved), 'accepted_loss': bool(accepted)})
        qualified_bands.append(bool(accepted))
    return source_log, target_log, sb, tb, noise, predicted, blocks, valid, diagnostics, qualified_bands


def measure_diagnosed_final(donor, final, allowed, owner, *, sigmas):
    """Read actual final bands at the already chosen source geometry scale."""
    sigmas = np.asarray(sigmas, float)
    support = allowed & owner & (distance_transform_edt(owner) > 4*sigmas[-1]+1)
    if not np.isfinite(final).all() or np.any(final < 0) or not support.any():
        return {'preserved': False, 'band_diagnostics': [], 'measured_blocks': 0,
                'reason': 'actual_final_not_measurable'}
    *_, blocks, valid, bands, qualified = _measure_frequency_response(donor, final, support, sigmas)
    return {'preserved': bool(all(row['measured_band_preserved'] for row in bands)),
        'sigmas': sigmas.tolist(), 'permitted_pixels': int(support.sum()),
        'band_diagnostics': bands, 'measured_blocks': len(blocks), 'scale_reselected': False,
        'endpoint': 'actual_diagnosed_final_with_its_full_filter_input',
        'pixel_restoration_attempted': False, 'visual_acceptance': False}


def recover_bands(donor, target, allowed, owner, *, native_short_side):
    """Return unchanged RGB unless structured fine-band loss is reproducible.

    Current chromatic ratios and coarse tone are retained. Restoration is a
    multiplicative luminance response from the already retouched source, not
    general sharpening. The finest camera noise band is never reintroduced.
    """
    donor, target = np.asarray(donor, np.float32), np.asarray(target, np.float32)
    allowed, owner = np.asarray(allowed, bool), np.asarray(owner, bool)
    if (donor.shape != target.shape or donor.shape != (*owner.shape, 3)
            or allowed.shape != owner.shape or not np.isfinite(donor).all()
            or not np.isfinite(target).all() or np.any(donor < 0) or np.any(target < 0)):
        raise ValueError('Subject detail needs finite nonnegative same-coordinate RGB and ownership')
    scale, support, distance, selection = select_geometry_scale(allowed, owner, native_short_side=native_short_side)
    sigmas = np.array(CONFIG['band_sigmas'])*scale
    checks = {'reason': 'insufficient_independent_owner_interior', 'permitted_pixels': int(support.sum()),
              'changed_pixels': 0, 'sigmas': sigmas.tolist(), 'visual_acceptance': False,
              'scale_selection': selection, 'measurement_state': 'not_measured',
              'diagnostic_status': 'support_unmeasurable'}
    if not selection['ready']:
        return target.copy(), np.zeros_like(owner, np.float32), checks
    checks['diagnostic_status'] = 'measurement_inconclusive'
    block = max(16, int(np.ceil(scale*CONFIG['block_sigma_multiple'])))
    # A channel gain can change Y's texture spectrum when the photograph has
    # chromatic detail. It is an exactly pointwise colour operation, not lost
    # spatial detail. Prove that explanation before interpreting log-Y bands.
    gains = []
    explained = True
    for channel in range(3):
        selected = support & (donor[...,channel] > 1e-4)
        if selected.sum() < CONFIG['minimum_owner_pixels']:
            explained = False; break
        gain = float(np.median(target[...,channel][selected]/donor[...,channel][selected]))
        error = np.abs(target[...,channel][support]-donor[...,channel][support]*gain)
        if not np.isfinite(gain) or gain <= 0 or error.max(initial=0) > 2e-6:
            explained = False; break
        gains.append(gain)
    if explained:
        checks.update(reason='pointwise_channel_grading_preserves_spatial_detail',
                      exact_channel_gains=gains)
        return target.copy(), np.zeros_like(owner, np.float32), checks
    photometric = _pointwise_photometric_explanation(donor, target, support, sigmas, block)
    if photometric is not None:
        checks.update(reason='pointwise_channel_grading_with_low_frequency_exposure',
                      photometric_explanation=photometric)
        return target.copy(), np.zeros_like(owner, np.float32), checks
    source_log, target_log, sb, tb, noise, predicted, blocks, valid, diagnostics, qualified_bands = (
        _measure_frequency_response(donor, target, support, sigmas))
    checks.update(band_diagnostics=diagnostics, measured_blocks=blocks,
                  estimated_log_noise_sigma=noise, band_noise_sigma=predicted, measurement_state='measured')
    if not all(qualified_bands):
        if all(row['measured_band_preserved'] for row in diagnostics):
            checks.update(reason='selected_bands_preserved_but_visual_diagnosis_failed',
                          diagnostic_status='detail_preserved_diagnostic_conflict')
        else:
            checks['reason'] = 'diagnosed_but_no_measured_structured_detail_loss'
        return target.copy(), np.zeros_like(owner, np.float32), checks
    checks['diagnostic_status'] = 'recovery_rejected'
    coarse_gain = float(np.median([row['coarse_gain'] for row in valid]))
    gains = [float(np.median([row['bands'][i]['gain'] for row in valid
             if row['bands'][i]['structured']])) for i in range(2)]
    qualified_support = np.zeros_like(support)
    for row in valid:
        if all(row['bands'][i]['coherent_structure'] and row['bands'][i]['structured']
               and row['bands'][i]['correlation'] >= CONFIG['minimum_structure_correlation']
               and row['bands'][i]['gain']/row['coarse_gain'] < CONFIG['maximum_relative_fine_gain'] for i in range(2)):
            x,y,r,b=row['box'];qualified_support[y:b,x:r] = True
    support &= qualified_support
    if support.sum() < CONFIG['minimum_owner_pixels']:
        checks['reason'] = 'insufficient_joint_structured_loss_support'
        return target.copy(), np.zeros_like(owner,np.float32), checks
    correction = sum(sb[i]*max(0., coarse_gain-gains[i]) for i in range(2))
    # Inward feather protects both independent owner and reviewed-window edges.
    weight = np.minimum(np.clip((distance-4*sigmas[-1]-1)/max(2., sigmas[1]), 0, 1),
                        np.clip(distance_transform_edt(support)/max(2., sigmas[1]), 0, 1))
    correction *= weight
    # Remove local coarse drift created by spatial permission without extending
    # the support or borrowing low-frequency colour from the source.
    for _ in range(2):
        correction -= gaussian_filter(correction, sigmas[-1], mode='nearest')*weight
    correction = np.clip(correction, -CONFIG['maximum_log_correction'], CONFIG['maximum_log_correction'])
    correction[~support] = 0
    upper = np.log(1/np.maximum(target.max(-1), 1e-8))
    correction = np.minimum(correction, np.maximum(upper, 0))
    after = target*np.exp(correction[..., None]).astype(np.float32)
    after[~support] = target[~support]
    after_log = np.log(np.maximum(after@LUMA, 1e-5)); ab = _bands(after_log, sigmas)
    before_error = sum(float(np.mean((tb[i][support]-coarse_gain*sb[i][support])**2)) for i in range(2))
    after_error = sum(float(np.mean((ab[i][support]-coarse_gain*sb[i][support])**2)) for i in range(2))
    drift = gaussian_filter(after_log-target_log, sigmas[-1], mode='nearest')[support]
    changed = np.any(after != target, axis=-1)
    checks.update(coarse_tone_gain=coarse_gain, before_band_error=before_error, after_band_error=after_error,
        low_log_drift_q95=float(np.quantile(np.abs(drift), .95)),
        low_log_drift_mean=float(np.mean(drift)), excluded_pixels_unchanged=bool(not np.any(changed & ~support)),
        maximum_log_correction=float(np.max(np.abs(correction))), owner_leakage_pixels=int((changed & ~owner).sum()))
    if (not np.isfinite(after).all() or after_error >= before_error*(1-CONFIG['minimum_error_improvement'])
            or checks['low_log_drift_q95'] > CONFIG['maximum_low_log_drift_q95'] or not changed.any()):
        checks['reason'] = 'recovery_did_not_improve_bands_without_coarse_drift'
        return target.copy(), np.zeros_like(weight), checks
    checks.update(reason='measured_structured_detail_recovered', changed_pixels=int(changed.sum()),
                  diagnostic_status='structured_loss_confirmed')
    return after, weight.astype(np.float32), checks


def _unions(boxes):
    result = []
    for box in boxes:
        candidate = list(box); rest = []
        for old in result:
            if candidate[0] < old[2] and old[0] < candidate[2] and candidate[1] < old[3] and old[1] < candidate[3]:
                candidate = [min(candidate[0], old[0]), min(candidate[1], old[1]),
                             max(candidate[2], old[2]), max(candidate[3], old[3])]
            else:
                rest.append(old)
        result = rest+[candidate]
    # A merge may span an earlier disjoint box. Iterate until stable.
    return _unions(result) if len(result) < len(boxes) else result


def diagnostic_summary(receipt):
    owners = []
    for observation in receipt.get('observations') or []:
        for row in observation['owners']:
            actual = row.get('diagnosed_final_measurement') or {}
            bands = actual.get('band_diagnostics') or row.get('band_diagnostics') or []
            owners.append({'native_box': observation['native_box'], 'semantic_class_id': row['semantic_class_id'],
                'status': row['diagnostic_status'], 'reason': row['reason'],
                'permitted_pixels': row['permitted_pixels'], 'measured_blocks': actual.get('measured_blocks',len(row.get('measured_blocks') or [])),
                'measurement_endpoint': actual.get('endpoint','appearance_before_manual_edits'),
                'measurement_state': row['measurement_state'], 'scale_selection': row['scale_selection'],
                'band_diagnostics': bands, 'fine_gain_medians': [band['measured_relative_gain_median'] for band in bands]})
    counts = {status: sum(row['status']==status for row in owners) for status in (
        'support_unmeasurable', 'detail_preserved_diagnostic_conflict', 'structured_loss_confirmed',
        'measurement_inconclusive', 'recovery_rejected')}
    # Report unresolved mixed-owner cases individually, without diluting them
    # into a global claim that no texture was lost.
    status = next((key for key in ('structured_loss_confirmed', 'detail_preserved_diagnostic_conflict',
        'recovery_rejected', 'measurement_inconclusive', 'support_unmeasurable') if counts[key]), 'recovery_rejected')
    actions = {
        'support_unmeasurable': '补充独立原尺寸窗口及源语义支持；当前局部未测量，保留未知。',
        'detail_preserved_diagnostic_conflict': '复核原片与成片的同尺寸局部诊断，区分纹理损失与缺少修饰；保留原失败项，不重复已占用的自动尝试。',
        'structured_loss_confirmed': '保存独立恢复子版本，并对新像素重新进行整图及局部审片。',
        'measurement_inconclusive': '检查所选频带的结构、噪声和仿色解释；未证明可恢复损失，保留原失败项。',
        'recovery_rejected': '按记录原因检查恢复改善或后续图层归属；保持当前像素和原失败项。'}
    return {'schema': 'subject_detail_diagnostic_summary.v1', 'status': status, 'owner_counts': counts,
        'owners': owners, 'changed_pixels': receipt['checks']['changed_pixels'],
        'visual_acceptance': False, 'visual_qa_unchanged': True,
        'scope': 'selected_measurable_bands_only; unmeasured_owners_and_larger_bands_remain_unknown',
        'next_action': actions[status]}


def diagnosed_endpoint_equivalence(target, final, support):
    """Require float endpoint equality, including relative changes in dark RGB."""
    if not support.any():
        return {'matches': False, 'checked_pixels': 0, 'maximum_log_rgb_difference': None}
    before, observed = target[support].astype(float), final[support].astype(float)
    finite = np.isfinite(observed).all() and np.all(observed >= 0)
    positive = (before>0) & (observed>0)
    zeros_equal = bool(np.all(before[~positive] == observed[~positive]))
    error = float(np.max(np.abs(np.log(observed[positive]/before[positive])), initial=0)) if finite else None
    return {'matches': bool(finite and zeros_equal and error <= CONFIG['maximum_endpoint_log_difference']),
        'checked_pixels': int(support.sum()),
        'maximum_log_rgb_difference': error, 'zero_channels_exactly_equal': zeros_equal,
        'maximum_allowed_log_rgb_difference': CONFIG['maximum_endpoint_log_difference'],
        'basis': 'within_numeric_endpoint_tolerance_on_pixels_allowed_to_change'}


def prepare_subject_detail(root, run, output):
    from automatic_subject_detail_completion import _diagnosis
    from design_reconstruction import sample_asset, reference_stack_execution_weights
    from reference_appearance_stack import apply_stack
    from source_semantic_binding import load_source_semantics
    from native_detail import read_scene_crop
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    diagnosis = _diagnosis(root, run)
    if not diagnosis:
        raise ValueError('Subject detail needs current actually reviewed native windows')
    windows = diagnosis['reviewed_windows']
    plan = run['options']['appearance_revision']['design_reconstruction_plan']; result = copy.deepcopy(plan)
    receipt = {'schema': SCHEMA, 'recipe': RECIPE, 'state': 'unknown', 'config': CONFIG.copy(),
        'producer_sha256': sha(__file__), 'source_sha256': run['source']['sha256'],
        'reference_sha256': run['reference']['sha256'], 'diagnosis': diagnosis['evidence'],
        'reviewed_windows': windows, 'visual_acceptance': False, 'model_inferences': 0,
        'image_generation_submissions': 0, 'checks': {'changed_pixels': 0},
        'qualification': {'qualified': False}, 'source_rgb_warped': False,
        'donor': 'frozen_approved_source_baseline; never raw unretouched pixels'}
    def finish(reason, changed=False):
        receipt['reason'] = reason
        receipt['diagnostic_summary'] = diagnostic_summary(receipt)
        write(output/'subject-detail.json', receipt)
        result.setdefault('provenance', {})['native_subject_detail'] = copy.deepcopy(receipt)
        prepared = {'ok': True, 'changed': changed, 'design_reconstruction_plan': result,
                    'receipt': receipt, 'qualification': receipt['qualification']}
        write(output/'prepared-plan.json', prepared); return prepared
    if any(row['kind'] == KIND for row in plan['layers']):
        return finish('saved_subject_detail_not_applied_twice')
    if run['options'].get('denoise_strength', 0):
        return finish('explicit_user_denoise_not_automatically_undone')
    native = list(plan['canvas']['source_oriented_size']); baseline = plan['source_baseline']['assets']
    stack = next(row for row in plan['layers'] if row['kind'] == 'reference_appearance_stack')
    weights = reference_stack_execution_weights(stack)['bank_weights']
    if not any(weights.values()):
        return finish('no_active_automatic_appearance_response')
    if plan['mask_roles'].get('edited_edge_light_applicability'):
        return finish('edited_coverage_requires_independent_detail_qualification')
    labels, semantics = load_source_semantics(plan['mask_roles']['source_semantics'],
        source_sha256=run['source']['sha256'], native_size=native, proxy_size=plan['canvas']['source_proxy_size'])
    mapped = np.asarray(labels)
    size = native; scale = max(1., min(size)/CONFIG['native_scale_reference']); margin = int(np.ceil(40*scale))+2
    boxes = _unions([row['native_box'] for row in windows])
    if any(not (0 <= b[0] < b[2] <= size[0] and 0 <= b[1] < b[3] <= size[1]) for b in boxes):
        raise ValueError('Reviewed subject window outside native canvas')
    processing = [[max(0,b[0]-margin), max(0,b[1]-margin), min(size[0],b[2]+margin), min(size[1],b[3]+margin)] for b in boxes]
    if sum((b[2]-b[0])*(b[3]-b[1]) for b in processing) > CONFIG['maximum_native_roi_pixels']:
        return finish('reviewed_subject_windows_exceed_bounded_native_roi_budget')
    arrays = {}; patches = []; observations = []; permitted = 0; changed_count = 0
    for index, (box, core_box) in enumerate(zip(processing, boxes)):
        x,y,r,b = box; h,w = b-y,r-x
        original_alpha = sample_asset(baseline['foreground_alpha'], 'values', box, size)
        alpha = sample_asset(plan['mask_roles']['foreground_alpha'], 'values', box, size)
        fp = sample_asset(baseline['foreground_premultiplied'], 'values', box, size)
        current_fp = sample_asset(next(row for row in plan['layers'] if row['kind']=='background_design')['assets']['foreground_premultiplied'], 'values', box, size)
        target_fp, _ = apply_stack(current_fp, stack, plan, run, box=box, canvas_size=size)
        donor = np.divide(fp, original_alpha[...,None], out=np.zeros_like(fp), where=original_alpha[...,None]>1e-5)
        target = np.divide(target_fp, alpha[...,None], out=np.zeros_like(fp), where=alpha[...,None]>1e-5)
        opaque = (alpha>=CONFIG['opaque_alpha']) & (original_alpha>=CONFIG['opaque_alpha'])
        xi = np.clip(((np.arange(x,r)+.5)*mapped.shape[1]/size[0]).astype(int),0,mapped.shape[1]-1)
        yi = np.clip(((np.arange(y,b)+.5)*mapped.shape[0]/size[1]).astype(int),0,mapped.shape[0]-1)
        owner_labels = mapped[np.ix_(yi,xi)]
        protect = plan['mask_roles'].get('identity_protection')
        if protect:
            opaque &= sample_asset(protect, 'protection', box, size) < .01
        final, _ = read_scene_crop(Path(run['artifacts']['render_scene_master']), box)
        after = target.copy(); owner_rows = []
        for semantic_id in sorted({owner for row in windows for owner in row['owner_class_ids']}):
            allowed = np.zeros((h,w), bool)
            for row in windows:
                if semantic_id not in row['owner_class_ids']: continue
                l,t,rr,bb = row['native_box']; l,t,rr,bb = max(l,x),max(t,y),min(rr,r),min(bb,b)
                if l<rr and t<bb: allowed[t-y:bb-y,l-x:rr-x] = True
            if not allowed.any():
                continue
            owner = (owner_labels==semantic_id) & opaque
            candidate, weight, checks = recover_bands(donor, target, allowed, owner, native_short_side=min(size))
            permitted += checks['permitted_pixels']
            # A foreground-only correction cannot compensate a downstream fog,
            # manual colour change or effect. Preserve that separate owner.
            support = weight>0
            if checks['diagnostic_status'] == 'detail_preserved_diagnostic_conflict':
                actual = measure_diagnosed_final(donor, final, allowed, owner, sigmas=checks['sigmas'])
                checks['diagnosed_final_measurement'] = actual
                if not actual['preserved']:
                    checks.update(reason='actual_final_bands_require_their_downstream_owner',
                                  diagnostic_status='recovery_rejected')
            if checks['changed_pixels']:
                checks['diagnosed_endpoint_equivalence'] = diagnosed_endpoint_equivalence(target, final, support)
            if checks.get('diagnosed_endpoint_equivalence', {}).get('matches') is False:
                checks.update(reason='downstream_pixels_require_their_own_owner', changed_pixels=0,
                              diagnostic_status='recovery_rejected')
            elif checks['changed_pixels']:
                after[support] = candidate[support]
            owner_rows.append({'semantic_class_id': semantic_id, **checks})
        observations.append({'native_box': box, 'owners': owner_rows})
        delta = (after-target)*alpha[...,None]
        # Freeze only the reviewed core: halos are computation, not permission.
        l,t,rr,bb=core_box; sl=np.s_[t-y:bb-y,l-x:rr-x]; delta=delta[sl].copy()
        changed=np.any(delta!=0,axis=-1); changed_count += int(changed.sum())
        if changed.any():
            key=f'delta_{index}'; arrays[key]=delta
            patches.append({'native_box': core_box, 'array_key': key, 'delta_array_sha256': array_sha(delta),
                            'changed_pixels': int(changed.sum()), 'target_patch_sha256': array_sha(target_fp[sl])})
    receipt.update(observations=observations, checks={'changed_pixels': changed_count,
        'alpha_changed': False, 'source_geometry_changed': False, 'reference_rgb_detail_used': False,
        'approved_source_retouch_reapplied': False, 'source_chroma_restored': False})
    receipt['qualification'] = {'qualified': any(row['scale_selection']['ready']
        for observation in observations for row in observation['owners']),
        'permitted_pixels': permitted, 'actual_paired_QA_verified': True,
        'owner': semantics, 'material_colour_classification_used': False,
        'scope': 'reviewed_source_semantic_interiors_only; other pixels unknown'}
    if not patches:
        summary = diagnostic_summary(receipt)
        return finish({'support_unmeasurable': 'subject_detail_not_measured_source_support_insufficient',
            'detail_preserved_diagnostic_conflict': 'subject_detail_measurement_conflicts_with_visual_diagnosis',
            'measurement_inconclusive': 'subject_detail_measured_loss_not_proven',
            'recovery_rejected': 'subject_detail_recovery_rejected'}[summary['status']])
    np.savez_compressed(output/'subject-detail-delta.npz', **arrays)
    binding = {'source_sha256': run['source']['sha256'], 'reference_sha256': run['reference']['sha256'],
        'source_fp_sha256': baseline['foreground_premultiplied']['sha256'],
        'source_alpha_sha256': baseline['foreground_alpha']['sha256'],
        'alpha_sha256': plan['mask_roles']['foreground_alpha']['sha256'],
        'semantic_sha256': plan['mask_roles']['source_semantics']['sha256'],
        'appearance_banks': {role: stack['assets'][role]['sha256'] for role in weights if weights[role]},
        'appearance_weights': weights, 'native_size': size, 'working_domain': DOMAIN}
    frozen={'schema': SCHEMA, 'recipe': RECIPE, 'binding': binding, 'delta': record(output/'subject-detail-delta.npz'),
            'patches': patches, 'donor_assets': copy.deepcopy(baseline), 'owner_asset': copy.deepcopy(plan['mask_roles']['source_semantics']),
            'approved_retouch_binding': copy.deepcopy(plan['source_baseline'].get('native_source_retouch'))}
    write(output/'frozen-subject-detail.json', frozen)
    receipt['state']='prepared'; finish('measured_source_subject_detail_prepared', True)
    result['layers'].insert(result['layers'].index(next(row for row in result['layers'] if row['kind']=='reference_appearance_stack'))+1,
        {'id':KIND,'kind':KIND,'label':'原片人物细节恢复','enabled':True,'strength':1.,
         'params':{'schema':SCHEMA,'recipe':RECIPE,'binding':binding},
         'assets':{'frozen_delta':record(output/'frozen-subject-detail.json'),
                   'qualification_receipt':record(output/'subject-detail.json')}})
    prepared={'ok':True,'changed':True,'design_reconstruction_plan':result,
              'receipt':receipt,'qualification':receipt['qualification']}
    write(output/'prepared-plan.json',prepared);return prepared


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path);parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    result=prepare_subject_detail(args.root,read(args.run),args.output)
    print(json.dumps({'changed':result['changed'],'reason':result['receipt']['reason']}))


if __name__=='__main__':main()
