"""Constrained reference colour matching in a shared display-linear domain.

Reference patches or independent material distributions supply colour
observations, never output pixels or geometry. Distribution evidence cannot
certify the spatial correspondence required by later colour fields.
This module deliberately has no matting, image generation, or scene relighting.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter
from scipy.optimize import least_squares, minimize
from scipy.interpolate import PchipInterpolator
from scipy.spatial import ConvexHull

from colour_space import to_lab, from_lab

SCHEMA = 'reference_colour_fit.appearance.v2'
REGIONS = {
    'face': [3], 'left_arm': [7, 11], 'right_arm': [16, 20],
    'left_hand': [6], 'right_hand': [15],
    'legs': [5, 8, 12, 14, 17, 21, 22], 'hair': [4],
    'clothing': [1, 10, 13, 19, 23], 'shoes': [9, 18], 'bag': [200],
}
SKIN = {'face', 'left_arm', 'right_arm', 'left_hand', 'right_hand', 'legs'}
SETTINGS = {'patch_size': 24, 'minimum_patch_fraction': .70,
            'minimum_anchors': 3, 'conservative_exposure_ev': .5, 'maximum_exposure_ev': 1.,
            'maximum_chroma_shift': .05, 'matrix_singular_values': [.75, 1.25],
            'maximum_matrix_condition': 20., 'protection_residual_strength': .15,
            'protection_chroma_strength': .15,
            'distribution_minimum_samples': 48, 'distribution_minimum_cells': 6,
            'distribution_maximum_exposure_ev': .5}


def release_feature_colour(parameters):
    """New preparations apply colour without an identity-mask attenuation.

    Keep saved recipes immutable. Lightness weights and material observation
    eligibility remain separate from this colour application policy.
    """
    import copy
    result = copy.deepcopy(parameters)
    result['settings']['protection_chroma_strength'] = 1.
    result['protection_policy'] = {
        'mode': 'split_lightness_chroma',
        'colour_identity_protection': False,
        'structural_L_strength': result['settings']['protection_residual_strength'],
        'protected_ab_strength': 1.,
        'anchor_protection': 'distinct_material_features_excluded_from_skin_fitting',
        'authored_jaw_gate': 'structural_weights_greater_than_0.9; unchanged_support',
        'reference_geometry_or_high_frequency_imported': False,
        'weight_assets': {'lightness': 'values', 'chroma': 'chroma_values'}}
    if result.get('feature_palette_policy'):
        result['protection_policy']['source_feature_palette'] = result['feature_palette_policy']
    return result


def _weighted_huber_loss(delta, observation_weights):
    """Keep the Huber threshold in colour units; leave priors quadratic.

    Residuals already include sample/material weights. Scaling the threshold
    by those same weights preserves the intended mean robust objective when
    sampling density changes. Regularization follows the observations.
    """
    thresholds = delta*np.asarray(observation_weights).ravel()
    def loss(squared):
        result = np.stack([squared.copy(), np.ones_like(squared), np.zeros_like(squared)])
        roots = np.sqrt(np.maximum(squared[:len(thresholds)], 1e-30))
        active = np.flatnonzero(roots > thresholds)
        t, r = thresholds[active], roots[active]
        result[0, active] = 2*t*r-t*t
        result[1, active] = t/r
        result[2, active] = -.5*t/(r**3)
        return result
    return loss


def _validate_images(source, reference, source_labels, reference_labels, alpha, protection):
    shape = source.shape
    if shape[-1] != 3 or reference.shape != shape:
        raise ValueError('Source and proposed reference anchors must share a canvas')
    if any(x.shape != shape[:2] for x in (source_labels, reference_labels, alpha, protection)):
        raise ValueError('Masks must use the source canvas')
    if not all(np.isfinite(x).all() for x in (source, reference, alpha, protection)):
        raise ValueError('Nonfinite colour observation')
    if alpha.min() < 0 or alpha.max() > 1 or protection.min() < 0 or protection.max() > 1:
        raise ValueError('Invalid alpha or protection observation')


def collect_anchors(source, reference, source_labels, reference_labels, alpha,
                    protection, reference_protection, validity, settings=None):
    """H proposes positions; homogeneous semantic patches reject poor proposals."""
    cfg = {**SETTINGS, **(settings or {})}
    _validate_images(source, reference, source_labels, reference_labels, alpha, protection)
    if any(x.shape != alpha.shape or not np.isfinite(x).all() or x.min() < 0 or x.max() > 1
           for x in (reference_protection, validity)):
        raise ValueError('Invalid reference protection or mapping validity')
    sl, rl = to_lab(source), to_lab(reference)
    usable_s = (alpha >= .995) & (protection < .05) & (sl[..., 0] > .08) & (source.max(-1) < .98)
    usable_r = (reference_protection < .05) & (validity > .999) & (rl[..., 0] > .08) & (reference.max(-1) < .98)
    anchors, rejected, overlaps = [], {}, {}
    for name, ids in REGIONS.items():
        step = 6 if name in {'face', 'hair'} or name.endswith('_hand') else 12 if name in {'left_arm', 'right_arm', 'bag'} else int(cfg['patch_size'])
        sm, rm = np.isin(source_labels, ids), np.isin(reference_labels, ids)
        union = np.count_nonzero(sm | rm)
        overlap = np.count_nonzero(sm & rm) / max(union, 1)
        overlaps[name] = float(overlap)
        # A background homography is not proof of matching person geometry.
        if overlap < .55:
            rejected[name] = 'semantic_geometry_overlap_below_0.55'
            continue
        safe_s = binary_erosion(sm, iterations=2) & usable_s
        safe_r = binary_erosion(rm, iterations=2) & usable_r
        count = 0
        ys, xs = np.nonzero(sm)
        if not len(xs):
            rejected[name] = 'source_region_absent'
            continue
        for top in range(int(ys.min()), int(ys.max()) + 1 - step, step):
            for left in range(int(xs.min()), int(xs.max()) + 1 - step, step):
                box = (slice(top, top+step), slice(left, left+step))
                valid = safe_s[box] & safe_r[box]
                if float(valid.mean()) < cfg['minimum_patch_fraction']:
                    continue
                sv, rv = sl[box][valid], rl[box][valid]
                # Only gross structural disagreement is rejected. Low-texture
                # cheeks cannot be certified by a texture correlation score.
                a, b = sl[box][..., 0], rl[box][..., 0]
                a, b = gaussian_filter(a, 1), gaussian_filter(b, 1)
                if a[valid].std() > .012 and b[valid].std() > .012:
                    correlation = float(np.corrcoef(a[valid], b[valid])[0, 1])
                    if correlation < -.1:
                        continue
                else:
                    correlation = None
                anchors.append({'region': name, 'xy': [left+step/2, top+step/2],
                    'box': [left, top, left+step, top+step],
                    'sample_mask_bits': np.packbits(valid.ravel()).tobytes().hex(),
                    'source': np.median(sv, axis=0).tolist(),
                    'reference': np.median(rv, axis=0).tolist(),
                    'pixels': int(valid.sum()), 'structure_correlation': correlation})
                count += 1
        if count < cfg['minimum_anchors']:
            anchors = [a for a in anchors if a['region'] != name]
            rejected[name] = 'insufficient_homogeneous_corresponding_patches'
    return anchors, {'rejected_regions': rejected, 'semantic_overlap': overlaps,
                     'correspondence': 'coarse_background_H_plus_material_patch_rejection; not_dense_correspondence'}


def _material_distribution(image, labels, protection, coverage, cfg, name, lab, skin_family=None):
    """Summarize one material on its own canvas; no paired positions exist."""
    image = np.asarray(image)
    if (image.ndim != 3 or image.shape[-1] != 3 or labels.shape != image.shape[:2]
            or protection.shape != labels.shape or coverage.shape != labels.shape
            or not all(np.isfinite(v).all() for v in (image, protection, coverage))
            or image.min() < 0 or image.max() > 1
            or min(protection.min(), coverage.min()) < 0
            or max(protection.max(), coverage.max()) > 1):
        raise ValueError('Invalid independent material observation')
    mask = binary_erosion(np.isin(labels, REGIONS[name]), iterations=2)
    mask &= (coverage >= .995) & (protection < .05)
    mask &= (lab[..., 0] > .15) & (lab[..., 0] < .95) & (image.max(-1) < .98)
    material_guard = None
    if name.endswith('_hand'):
        # A hand semantic class includes black cosplay gloves. Exposed skin
        # needs radiometric support from this image's own observed face skin;
        # reference glove pixels must never become a skin-colour target.
        if not skin_family or not skin_family.get('accepted'):
            return {'samples': 0, 'cells': 0, 'accepted': False,
                    'rejected_reason': 'hand_material_not_certified_without_observed_skin_family'}
        floor = max(.15, .65*float(skin_family['quantiles'][0][0]))
        mask &= lab[..., 0] >= floor
        material_guard = {'mode': 'source_L_exposed_skin_support', 'minimum_L': floor,
                          'transition_L': .02, 'family_region': 'face',
                          'reason': 'exclude_dark_glove_material_inside_hand_semantics'}
    count = int(mask.sum())
    step = 3 if name in SKIN and name != 'face' else 6
    rows = []
    ys, xs = np.nonzero(mask)
    if len(xs):
        for top in range(int(ys.min()), int(ys.max())+1, step):
            for left in range(int(xs.min()), int(xs.max())+1, step):
                box = (slice(top, top+step), slice(left, left+step))
                valid = mask[box]
                if valid.size != step*step or valid.mean() < .7:
                    continue
                rows.append(np.median(lab[box][valid], axis=0))
    result = {'samples': count, 'cells': len(rows), 'accepted': False,
              'sampling': 'independent_eroded_same_material_cells; no_coordinate_pairs'}
    if material_guard:
        result['material_guard'] = material_guard
    if count < cfg['distribution_minimum_samples'] or len(rows) < cfg['distribution_minimum_cells']:
        return {**result, 'rejected_reason': 'insufficient_safe_material_samples'}
    rows = np.asarray(rows)
    center = np.median(rows, axis=0)
    spread = float(np.quantile(np.linalg.norm(rows[:, 1:]-center[1:], axis=1), .9))
    split_delta = np.abs(np.median(rows[::2], axis=0)-np.median(rows[1::2], axis=0))
    result.update(center=center.tolist(), quantiles=np.quantile(rows, [.25, .5, .75], axis=0).tolist(),
                  chroma_spread_q90=spread, split_center_delta=split_delta.tolist(),
                  validation_centers=[np.median(rows[::2], axis=0).tolist(),
                                      np.median(rows[1::2], axis=0).tolist()])
    if spread > (.07 if name in SKIN else .12) or np.linalg.norm(split_delta[1:]) > .025 or split_delta[0] > .12:
        return {**result, 'rejected_reason': 'unstable_or_mixed_material_distribution'}
    return {**result, 'accepted': True}


def collect_material_distributions(source, reference, source_labels, reference_labels,
                                   alpha, protection, reference_protection, *, settings=None):
    """Independent same-material statistics, including different image sizes.

    Semantic interiors, source coverage and protected facial features reject
    contamination. Native reference observations never depend on background H
    or person shape overlap. Small or unstable materials remain unsupported.
    """
    cfg = {**SETTINGS, **(settings or {})}
    reference_coverage = np.ones(reference.shape[:2], np.float32)
    source_lab, reference_lab = to_lab(source), to_lab(reference)
    source_face = _material_distribution(source, source_labels, protection, alpha, cfg, 'face', source_lab)
    reference_face = _material_distribution(reference, reference_labels, reference_protection,
                                            reference_coverage, cfg, 'face', reference_lab)
    return {name: {'source': source_face if name == 'face' else _material_distribution(
                    source, source_labels, protection, alpha, cfg, name, source_lab, source_face),
                   'reference': reference_face if name == 'face' else _material_distribution(
                    reference, reference_labels, reference_protection, reference_coverage, cfg, name,
                    reference_lab, reference_face), 'mode': 'independent_material_distribution',
                   'paired_anchor_count': 0, 'spatial_correspondence_allowed': False}
            for name in REGIONS}


def _distribution_operator(name, observation, cfg):
    source, reference = observation['source'], observation['reference']
    if not source['accepted'] or not reference['accepted']:
        return None
    x, y = np.asarray(source['quantiles']), np.asarray(reference['quantiles'])
    center_x, center_y = np.asarray(source['center']), np.asarray(reference['center'])
    raw_bias = center_y[1:]-center_x[1:]
    bias = raw_bias*min(1., cfg['maximum_chroma_shift']/max(float(np.linalg.norm(raw_bias)), 1e-9))
    requested = float(np.median(3*np.log2(y[:, 0]/x[:, 0])))
    cap = cfg['distribution_maximum_exposure_ev']
    exposure = float(least_squares(lambda v: tone(x[:, 0], v[0])-y[:, 0],
                                  [np.clip(requested, -cap, cap)], bounds=([-cap], [cap]),
                                  loss='soft_l1', f_scale=.02).x[0])
    before, after = [], []
    for sx, ry in zip(source['validation_centers'], reference['validation_centers']):
        sx, ry = np.asarray(sx), np.asarray(ry)
        before.append(float(np.linalg.norm(ry[1:]-sx[1:])))
        after.append(float(np.linalg.norm(ry[1:]-sx[1:]-bias)))
    before_error, after_error = float(np.mean(before)), float(np.mean(after))
    # This validates distribution representatives, never per-pixel matches.
    if before_error > .003 and after_error > .85*before_error:
        observation['rejected_reason'] = 'independent_split_distribution_shift_not_stable'
        return None
    light_before = float(np.mean(abs(x[:, 0]-y[:, 0])))
    light_after = float(np.mean(abs(tone(x[:, 0], exposure)-y[:, 0])))
    light_valid = light_after <= light_before+1e-7
    if not light_valid:
        exposure = 0.
    return {'matrix': np.eye(2).tolist(), 'bias': bias.tolist(), 'exposure_ev': exposure,
            **({'source_material_guard': source['material_guard']} if source.get('material_guard') else {}),
            'requested_exposure_ev': requested, 'exposure_limited': abs(requested) > cap,
            'chroma_limited': bool(np.linalg.norm(raw_bias) > cfg['maximum_chroma_shift']), 'anchors': 0,
            'evidence': {'mode': 'material_distribution', 'accepted': True,
                'source_samples': source['samples'], 'reference_samples': reference['samples'],
                'source_cells': source['cells'], 'reference_cells': reference['cells'],
                'paired_anchor_count': 0, 'spatial_correspondence_allowed': False,
                'source_region': name, 'reference_region': name, 'lightness_statistical_tone': light_valid,
                'validation': {'method': 'independent_split_distribution_centers_and_L_quantiles; not_paired_error',
                    'before_chroma_center_error': before_error, 'after_chroma_center_error': after_error,
                    'before_L_quantile_MAE': light_before, 'after_L_quantile_MAE': light_after}}}


def fit_material_distributions(parameters, observations):
    """Fill missing T operators without changing paired fits or spatial gates."""
    import copy
    result = copy.deepcopy(parameters)
    cfg = {**SETTINGS, **result['settings']}
    result['settings'] = cfg
    for name in REGIONS:
        if name in result['operators']:
            continue
        op = _distribution_operator(name, observations[name], cfg)
        if op is not None:
            result['operators'][name] = op
    # An independently observed source hand/arm may borrow a supported skin
    # palette shift when its reference material is absent. This is an explicit
    # family prior, not direct evidence for that hand or a brightness match.
    for name in ('left_arm', 'right_arm', 'left_hand', 'right_hand'):
        if name in result['operators'] or not observations[name]['source']['accepted']:
            continue
        parents = (name.replace('_hand', '_arm'), 'face') if name.endswith('_hand') else ('face',)
        for parent in parents:
            parent_op = result['operators'].get(parent)
            family = observations[parent]
            if not parent_op or not family['source']['accepted'] or not family['reference']['accepted']:
                continue
            source = observations[name]['source']; reference = family['reference']
            source_ab = np.asarray(source['center'])[1:]
            parent_ab = np.asarray(family['source']['center'])[1:]
            if np.linalg.norm(source_ab-parent_ab) > .08:
                continue
            bias = parent_ab @ np.asarray(parent_op['matrix']).T+parent_op['bias']-parent_ab
            bias *= min(1., cfg['maximum_chroma_shift']/max(float(np.linalg.norm(bias)), 1e-9))
            result['operators'][name] = {'matrix': np.eye(2).tolist(), 'bias': bias.tolist(),
                'exposure_ev': 0., 'anchors': 0, 'parent_region': parent,
                **({'source_material_guard': source['material_guard']} if source.get('material_guard') else {}),
                'evidence': {'mode': 'skin_family_prior', 'accepted': True,
                    'source_samples': source['samples'], 'reference_samples': reference['samples'],
                    'source_cells': source['cells'], 'reference_cells': reference['cells'],
                    'paired_anchor_count': 0, 'spatial_correspondence_allowed': False,
                    'source_region': name, 'reference_region': parent, 'parent_region': parent,
                    'lightness_statistical_tone': False,
                    'validation': {'method': 'source_skin_family_compatibility_and_supported_parent_palette',
                                   'source_family_chroma_distance': float(np.linalg.norm(source_ab-parent_ab))}}}
            break
    if result['operators']:
        result.pop('fallback', None)
    result['distribution_policy'] = 'independent_same_material_statistics.v1'
    face = result['operators'].get('face', {})
    if face.get('evidence', {}).get('mode') == 'material_distribution':
        result['feature_palette_policy'] = {
            'mode': 'source_feature_chroma_only_face_distribution_bias',
            'source_feature_labels': [24, 25], 'parent_region': 'face',
            'colour_operator': 'identity_matrix_plus_supported_face_bias',
            'native_L_and_feature_contrast_retained': True,
            'reference_feature_pixels_used': False, 'spatial_correspondence_claimed': False}
    return result


def _bounded_matrix(matrix, cfg):
    u, s, vt = np.linalg.svd(matrix)
    matrix = u @ np.diag(np.clip(s, *cfg['matrix_singular_values'])) @ vt
    matrix[0, 1] = np.clip(matrix[0, 1], -.15, .15)
    matrix[1, 0] = np.clip(matrix[1, 0], -.15, .15)
    matrix[0, 0] = np.clip(matrix[0, 0], .75, 1.25)
    matrix[1, 1] = np.clip(matrix[1, 1], .75, 1.25)
    if np.linalg.det(matrix) <= 0:
        return np.eye(2)
    # Projection of off-diagonals may change the singular-value bounds.
    for _ in range(10):
        s = np.linalg.svd(matrix, compute_uv=False)
        if s.min() >= cfg['matrix_singular_values'][0]-1e-7 and s.max() <= cfg['matrix_singular_values'][1]+1e-7:
            break
        matrix = .8 * matrix + .2 * np.eye(2)
    return matrix


def tone(lightness, exposure_ev):
    """Exposure-like midtone gain with a monotone, bounded highlight endpoint."""
    gain = 2**(exposure_ev/3)
    # Keep the start tangent below 2.8 times the remaining secant. A fixed
    # knee with a large fitted gain can overshoot white and reverse highlights.
    knee = min(.7, (2.8-gain)/(1.8*gain)) if gain > 1 else .7
    x = np.clip(lightness, 0, 1)
    result = gain*x
    t = np.clip((x-knee)/(1-knee), 0, 1)
    y0 = gain*knee
    secant = (1-y0)/(1-knee)
    m0, m1 = gain, secant*max(.4, 1-5*abs(gain-1))
    curved = (2*t**3-3*t**2+1)*y0 + (t**3-2*t**2+t)*(1-knee)*m0
    curved += (-2*t**3+3*t**2) + (t**3-t**2)*(1-knee)*m1
    return np.where(x > knee, curved, result)


def _fit_exposure(rows, cfg, parent=None):
    """Fit one material's monotone tone scalar; never copy reference shading.

    The conservative fit remains the fallback. A larger gain or a separate
    hand value is accepted only when its actual bounded curve predicts heldout
    observations better, with stable estimates across the three folds.
    """
    x = np.array([a['source'][0] for a in rows]); y = np.array([a['reference'][0] for a in rows])
    requested = float(np.median(3*np.log2(np.maximum(y, .01)/np.maximum(x, .01))))
    fallback = parent['exposure_ev'] if parent else float(np.clip(requested, -cfg['conservative_exposure_ev'], cfg['conservative_exposure_ev']))
    low, high = -cfg['maximum_exposure_ev'], cfg['maximum_exposure_ev']
    if parent:
        low, high = max(low, fallback-.35), min(high, fallback+.35)
    def solve(indices):
        def residual(v):
            return np.r_[(tone(x[indices], v[0])-y[indices])/np.sqrt(len(indices)),
                         (.02 if parent else .01)*(v[0]-fallback)]
        return float(least_squares(residual, [np.clip(fallback, low, high)], bounds=([low], [high]),
                                   loss=_weighted_huber_loss(.012, np.full(len(indices), 1/np.sqrt(len(indices))))).x[0])
    result = {'exposure_ev': fallback, 'requested_exposure_ev': requested,
              'exposure_validation': {'accepted': False, 'reason': 'insufficient_or_unimproved_holdout'},
              'exposure_inherited_from_arm': bool(parent)}
    if len(rows) >= 9:
        order = np.argsort(x, kind='stable')
        baseline_error, new_error, baseline_mae, new_mae, estimates = [], [], [], [], []
        for fold in range(3):
            held = order[fold::3]; train = np.setdiff1d(np.arange(len(rows)), held)
            trial = solve(train); estimates.append(trial)
            old = tone(x[held], fallback)-y[held]; new = tone(x[held], trial)-y[held]
            baseline_error.extend(old**2); new_error.extend(new**2)
            baseline_mae.extend(abs(old)); new_mae.extend(abs(new))
        improvement = 1-float(np.mean(new_error))/max(float(np.mean(baseline_error)), 1e-12)
        mae_improvement = float(np.mean(baseline_mae)-np.mean(new_mae))
        stable = float(np.ptp(estimates)) <= .15
        accepted = improvement >= .10 and mae_improvement >= .003 and stable
        result['exposure_validation'] = {'accepted': bool(accepted), 'heldout_error_improvement': improvement,
            'heldout_absolute_error_reduction': mae_improvement, 'fold_estimates_ev': estimates,
            'stable_fold_estimates': stable, 'validation': 'three_folds_interleaved_by_source_L; actual_bounded_curve',
            'bounded_range_ev': [low, high], 'fallback_ev': fallback}
        if accepted:
            result['exposure_ev'] = solve(np.arange(len(rows)))
            result['exposure_inherited_from_arm'] = False
    result['exposure_limited'] = abs(result['exposure_ev']-requested) > 1e-6
    return result


def _apply_tone(lightness, operator):
    curve = operator.get('tone_curve')
    if not curve:
        return tone(lightness, operator['exposure_ev'])
    return PchipInterpolator(curve['source_knots'], curve['target_knots'])(np.clip(lightness, 0, 1)).astype(np.float32)


def fit_tone_curves(anchors, parameters, *, regions=('left_hand', 'right_hand', 'right_arm')):
    """A material tone curve can lift dark skin without over-lifting fingers.

    Only source lightness is the input. No spatial reference lightness is copied.
    Three measured controls, positive bounded slopes, fixed black/white and
    regularization preserve native texture and ordered brightness.
    """
    from copy import deepcopy
    result = deepcopy(parameters)
    for name in regions:
        rows = [a for a in anchors if a['region'] == name]
        op = parameters['operators'].get(name)
        if op is None or op.get('evidence', {}).get('mode', 'paired_patch') != 'paired_patch' or len(rows) < 18:
            continue
        x = np.array([a['source'][0] for a in rows]); y = np.array([a['reference'][0] for a in rows])
        knots = np.quantile(x, [.05, .5, .95])
        if knots[0] <= .04 or knots[-1] >= .9 or np.diff(knots).min() < .02:
            continue
        baseline = tone(knots, op['exposure_ev'])
        source_knots = np.r_[0, knots, 1]
        def solve(indices):
            def objective(controls):
                prediction = PchipInterpolator(source_knots, np.r_[0, controls, 1])(x[indices])
                error = prediction-y[indices]
                huber = np.where(abs(error) < .02, error**2, .04*abs(error)-.0004)
                return np.mean(huber)+.10*np.mean((controls-baseline)**2)
            constraints = [{'type': 'ineq', 'fun': lambda c: np.diff(c)-.6*np.diff(knots)},
                           {'type': 'ineq', 'fun': lambda c: 1.6*np.diff(knots)-np.diff(c)}]
            bounds = list(zip(np.maximum(baseline-.04, .01), np.minimum(baseline+.04, .99)))
            optimized = minimize(objective, baseline, method='SLSQP', constraints=constraints, bounds=bounds,
                                 options={'ftol': 1e-11, 'maxiter': 100})
            if not optimized.success or np.diff(optimized.x).min() <= 0:
                raise ValueError('Unsupported monotone tone fit')
            return optimized.x
        order = np.argsort(x, kind='stable')
        old, new, estimates = [], [], []
        try:
            for fold in range(3):
                held = order[fold::3]; train = np.setdiff1d(np.arange(len(rows)), held)
                controls = solve(train); estimates.append(controls)
                old.extend(tone(x[held], op['exposure_ev'])-y[held])
                new.extend(PchipInterpolator(source_knots, np.r_[0, controls, 1])(x[held])-y[held])
            old, new = np.array(old), np.array(new)
            improvement = 1-float(np.mean(new**2))/max(float(np.mean(old**2)), 1e-12)
            mae_improvement = float(np.mean(abs(old))-np.mean(abs(new)))
            stable = float(np.ptp(np.array(estimates), axis=0).max()) <= .015
            if improvement < .10 or mae_improvement < .003 or not stable:
                continue
            controls = solve(np.arange(len(rows)))
        except ValueError:
            continue
        result['operators'][name]['tone_curve'] = {'source_knots': source_knots.tolist(),
            'target_knots': np.r_[0, controls, 1].tolist(), 'maximum_control_residual': .04,
            'interior_secant_bounds': [.6, 1.6], 'heldout_error_improvement': improvement,
            'heldout_absolute_error_reduction': mae_improvement, 'stable_fold_estimates': stable,
            'validation': 'three_folds_interleaved_by_source_L; bounded_monotone_curve',
            'source_only_coordinate': 'OKLab_L', 'reference_spatial_lightness_imported': False}
        derivative = PchipInterpolator(source_knots, np.r_[0, controls, 1]).derivative()(np.linspace(0, 1, 10001))
        result['operators'][name]['tone_curve']['actual_derivative_range'] = [float(derivative.min()), float(derivative.max())]
    return result


def _shade_design(levels, knots):
    levels = np.clip(levels, knots[0], knots[-1])
    columns = [np.interp(levels, knots, np.eye(3)[i]) for i in range(3)]
    return np.stack(columns, axis=-1)


def _shade_colour(field, lightness):
    colour = _shade_design(lightness, field['knots']) @ np.asarray(field['nodes'], np.float32)
    low, high = field['observed_range']
    support = np.clip(1-np.maximum(low-lightness, lightness-high)/field.get('support_margin', .05), 0, 1)
    return colour*support[..., None]


def fit_conditional_residual(anchors, parameters, *, regions=('hair', 'left_hand', 'right_hand', 'right_arm')):
    """A colour-only bias conditioned on one source colour axis.

    Hair/right skin use source a (warm/green cast); the left hand uses source L. Three smooth
    controls preserve source lightness/texture. Unsupported input colours and
    failed heldouts fall back to A. Bounds are included in heldout prediction.
    """
    from copy import deepcopy
    result = deepcopy(parameters)
    result['conditional_chroma_residual'] = {}
    for name in regions:
        op = parameters['operators'].get(name)
        rows = [a for a in anchors if a['region'] == name]
        if op is None or op.get('evidence', {}).get('mode', 'paired_patch') != 'paired_patch' or len(rows) < 12:
            continue
        x = np.array([a['source'] for a in rows]); y = np.array([a['reference'] for a in rows])
        channel = 0 if name == 'left_hand' else 1
        coordinate = x[:, channel]
        # Hair highlights occupy a small area. A 90th-percentile endpoint can
        # still describe only dark hair, leaving the observed warm tail out.
        quantiles = [.1, .5, .98] if name == 'hair' else [.1, .5, .9] if not channel else [.05, .5, .95]
        knots = np.quantile(coordinate, quantiles)
        if np.min(np.diff(knots)) < (.002 if channel else .015):
            continue
        design = _shade_design(coordinate, knots)
        error = y[:, 1:]-(x[:, 1:] @ np.asarray(op['matrix']).T+op['bias'])
        maximum = .02 if name.endswith('_hand') else .03
        groups = np.digitize(coordinate, [(knots[0]+knots[1])/2, (knots[1]+knots[2])/2])
        if np.min(np.bincount(groups, minlength=3)) < 6:
            continue
        def solve(indices):
            counts = np.maximum(np.bincount(groups[indices], minlength=3), 1)
            balance = 1/np.sqrt(3*counts[groups[indices]])
            def residual(v):
                nodes = v.reshape(3, 2)
                return np.r_[((design[indices] @ nodes-error[indices])*balance[:, None]).ravel(),
                             (.06*np.diff(nodes, axis=0)).ravel(), (.025*nodes).ravel()]
            nodes = least_squares(residual, np.zeros(6),
                loss=_weighted_huber_loss(.006, np.repeat(balance, 2))).x.reshape(3, 2)
            nodes *= np.minimum(1, maximum/np.maximum(np.linalg.norm(nodes, axis=1), 1e-9))[:, None]
            return nodes
        ordered = np.argsort(coordinate, kind='stable')
        base_error = test_error = 0.
        for fold in range(3):
            heldout = ordered[fold::3]; training = np.setdiff1d(np.arange(len(rows)), heldout)
            trial = solve(training)
            base_error += float(np.mean(error[heldout]**2))
            test_error += float(np.mean((design[heldout] @ trial-error[heldout])**2))
        improvement = 1-test_error/max(base_error, 1e-12)
        if improvement < .10:
            continue
        result['conditional_chroma_residual'][name] = {'knots': knots.tolist(), 'nodes': solve(np.arange(len(rows))).tolist(),
            'observed_range': [float(coordinate.min()), float(coordinate.max())], 'maximum_node_shift': maximum,
            'source_channel': channel, 'support_margin': .01 if channel else .05,
            'anchors': len(rows), 'heldout_error_improvement': improvement,
            'validation': 'three_folds_interleaved_by_source_axis; bounded_prediction',
            'changes_lightness': False, 'coordinate': f'source_OKLab_{"a" if channel else "L"}_before_A'}
    return result


def fit_local_residual(anchors, parameters, canvas_size, *, regions=None, strength=1.):
    """Fit four colour-only controls per supported material, after A.

    Coordinates refer to the source canvas; no reference pixels are sampled
    during application. Missing spatial support rejects the local model.
    """
    from copy import deepcopy
    result = deepcopy(parameters)
    result['local_chroma_residual'] = {}
    selected = set(regions if regions is not None else (SKIN | {'clothing'}))
    for name in parameters['operators']:
        if name not in selected:
            continue
        if parameters['operators'][name].get('evidence', {}).get('spatial_correspondence_allowed') is False:
            continue
        if name in {'left_hand', 'right_hand', 'right_arm'} and name in parameters.get('conditional_chroma_residual', {}):
            continue
        rows = [a for a in anchors if a['region'] == name]
        if len(rows) < 6:
            continue
        xy = np.array([a['xy'] for a in rows], float)/np.asarray(canvas_size)
        low, high = xy.min(0), xy.max(0)
        if np.any(high-low < .015):
            continue
        uv = (xy-low)/(high-low)
        u, v = uv.T
        hand_axis = name.endswith('_hand')
        design = np.stack([1-u, u], axis=-1) if hand_axis else np.stack([(1-u)*(1-v), u*(1-v), (1-u)*v, u*v], axis=-1)
        condition = float(np.linalg.cond(design))
        controls = 2 if hand_axis else 4
        if np.linalg.matrix_rank(design, tol=.001) < controls or condition > 30:
            continue
        op = parameters['operators'][name]
        x = np.array([a['source'][1:] for a in rows]); y = np.array([a['reference'][1:] for a in rows])
        residuals = y-(x @ np.asarray(op['matrix']).T+op['bias'])
        shade = parameters.get('conditional_chroma_residual', {}).get(name)
        if shade:
            residuals -= _shade_colour(shade, np.array([a['source'][shade['source_channel']] for a in rows]))
        residuals -= np.median(residuals, axis=0)
        # A large reference error alone does not activate B. There must be
        # opposite spatial colour residuals within this material.
        if not np.any((np.quantile(residuals, .1, axis=0) < -.004) &
                      (np.quantile(residuals, .9, axis=0) > .004)):
            continue
        maximum = min(.02 if name.endswith('_hand') else .035, float(np.quantile(np.linalg.norm(residuals, axis=1), .95)))
        def expanded(nodes):
            return nodes[[0, 1, 0, 1]] if hand_axis else nodes
        def bounded(nodes):
            nodes = nodes.copy()
            nodes *= np.minimum(1, maximum/np.maximum(np.linalg.norm(nodes, axis=1), 1e-9))[:, None]
            for _ in range(3):
                four = expanded(nodes)
                edges = np.linalg.norm(four[[0, 0, 1, 2]]-four[[1, 2, 3, 3]], axis=1)
                limit = .025 if name.endswith('_hand') else .04
                if edges.max() <= limit:
                    break
                nodes *= limit/edges.max()
            return nodes
        def residual(ctrl, subset=None):
            nodes = ctrl.reshape(controls, 2)
            four = expanded(nodes)
            edges = four[[0, 0, 1, 2]]-four[[1, 2, 3, 3]]
            active = np.arange(len(rows)) if subset is None else subset
            return np.r_[((design[active] @ nodes-residuals[active])/np.sqrt(len(active))).ravel(),
                         (.03*edges).ravel(), (.02*nodes).ravel(), .10*nodes.mean(0)]
        # A distribution of errors is not proof of a spatial colour trend.
        # Spatial controls must improve held-out patches over a zero field.
        order = np.argsort(xy[:, 0], kind='stable') if hand_axis else np.arange(len(rows))
        base_error = test_error = 0.
        folds = 3 if hand_axis else 1
        for fold in range(folds):
            heldout = order[fold::3]
            training = np.setdiff1d(np.arange(len(rows)), heldout)
            if np.linalg.matrix_rank(design[training], tol=.001) < controls:
                test_error = float('inf')
                break
            trial = bounded(least_squares(lambda v: residual(v, training), np.zeros(controls*2),
                loss=_weighted_huber_loss(.004, np.full(2*len(training), 1/np.sqrt(len(training))))).x.reshape(controls, 2))
            base_error += float(np.mean(residuals[heldout]**2))
            test_error += float(np.mean((design[heldout] @ trial-residuals[heldout])**2))
        improvement = 1-test_error/max(base_error, 1e-12)
        if improvement < .10:
            continue
        fitted = least_squares(residual, np.zeros(controls*2),
            loss=_weighted_huber_loss(.004, np.full(2*len(rows), 1/np.sqrt(len(rows))))).x.reshape(controls, 2)
        raw = fitted.copy()
        # Controls are constrained by regularization, rank and held-out error.
        # Do not also weaken reliable interior samples merely because a curved
        # arm or cheek does not occupy the corners of its rectangular bounds.
        fitted = expanded(bounded(fitted))
        result['local_chroma_residual'][name] = {
            'nodes': fitted.tolist(), 'bounds_normalized': [low.tolist(), high.tolist()],
            'strength': strength, 'maximum_node_shift': maximum, 'condition': condition,
            'requested_maximum_node_shift': float(np.linalg.norm(raw, axis=1).max()),
            'changes_lightness': False, 'anchors': len(rows),
            'spatial_axes': ['x'] if hand_axis else ['x', 'y'],
            'validation': 'three_folds_interleaved_by_source_x; bounded_prediction' if hand_axis else 'heldout_every_third_patch; bounded_prediction',
            'heldout_error_improvement': improvement,
            'support_hull_equations': ConvexHull(xy).equations.tolist(),
            'support_margin_normalized': (np.array([48, 48])/np.asarray(canvas_size)).tolist()}
    return result


def _local_colour(field, shape, origin, canvas_size):
    height, width = shape
    x = (np.arange(width)+origin[0]+.5)/canvas_size[0]
    y = (np.arange(height)+origin[1]+.5)/canvas_size[1]
    low, high = np.asarray(field['bounds_normalized'])
    u = np.clip((x-low[0])/(high[0]-low[0]), 0, 1)[None, :]
    v = np.clip((y-low[1])/(high[1]-low[1]), 0, 1)[:, None]
    nodes = np.asarray(field['nodes'], np.float32)
    colour = ((1-u)*(1-v))[..., None]*nodes[0] + (u*(1-v))[..., None]*nodes[1]
    colour += ((1-u)*v)[..., None]*nodes[2] + (u*v)[..., None]*nodes[3]
    margin = np.asarray(field['support_margin_normalized'])
    sx = np.clip(1-np.maximum(low[0]-x, x-high[0])/margin[0], 0, 1)
    sy = np.clip(1-np.maximum(low[1]-y, y-high[1])/margin[1], 0, 1)
    support = sx[None, :]*sy[:, None]
    for a, b, c in field['support_hull_equations']:
        outside = a*x[None, :]+b*y[:, None]+c
        support = np.minimum(support, np.clip(1-np.maximum(outside, 0)/min(margin), 0, 1))
    return colour*support[..., None]*field['strength']


def fit(anchors, *, settings=None):
    cfg = {**SETTINGS, **(settings or {})}
    for row in anchors:
        if (row.get('region') not in REGIONS or any(np.asarray(row.get(key)).shape != (3,)
                or not np.isfinite(np.asarray(row[key], float)).all() for key in ('source', 'reference'))):
            raise ValueError('Invalid colour anchor observation')
    names = [n for n in REGIONS if sum(a['region'] == n for a in anchors) >= cfg['minimum_anchors']]
    if not names:
        return {'schema': SCHEMA, 'working_domain': 'display_linear_sRGB', 'settings': cfg,
            'operators': {}, 'matrix_fit': {'estimated': False, 'rank': 0, 'condition': None},
            'fallback': {'mode': 'identity_statistical', 'statistics_available': False,
                'reason': 'no_trustworthy_same_material_anchors', 'source_appearance_preserved': True},
            'reference_pixels_in_output': False, 'geometry_warp': False}
    matrix = np.eye(2)
    centered_s, centered_r, weights = [], [], []
    for name in names:
        if name not in SKIN or name.endswith('_hand'):
            continue
        rows = [a for a in anchors if a['region'] == name]
        x = np.array([a['source'][1:] for a in rows])
        y = np.array([a['reference'][1:] for a in rows])
        centered_s.append(x - np.median(x, axis=0))
        centered_r.append(y - np.median(y, axis=0))
        weights.extend([1/np.sqrt(len(rows))] * len(rows))
    condition, rank, estimated = None, 0, False
    if centered_s:
        x, y = np.concatenate(centered_s), np.concatenate(centered_r)
        rank = int(np.linalg.matrix_rank(x, tol=.001))
        condition = float(np.linalg.cond(x))
        if rank == 2 and condition <= cfg['maximum_matrix_condition']:
            w = np.array(weights)[:, None]
            def residual(v):
                m = v.reshape(2, 2)
                return np.r_[((x @ m.T-y)*w).ravel(), .015*(m-np.eye(2)).ravel()]
            opt = least_squares(residual, np.eye(2).ravel(), loss=_weighted_huber_loss(.006, np.repeat(w.ravel(), 2)))
            matrix = _bounded_matrix(opt.x.reshape(2, 2), cfg)
            estimated = True
    operators = {}
    for name in names:
        rows = [a for a in anchors if a['region'] == name]
        x = np.array([a['source'] for a in rows]); y = np.array([a['reference'] for a in rows])
        m = matrix if name in SKIN else np.eye(2)
        diffs = y[:, 1:] - x[:, 1:] @ m.T
        target = np.median(diffs, axis=0)
        parent = operators.get(name.replace('_hand', '_arm')) if name.endswith('_hand') else None
        if name.endswith('_hand') and parent is None:
            continue
        parent_bias = np.asarray(parent['bias']) if parent else np.zeros(2)
        def solve_bias(indices):
            def bias_residual(v):
                return np.r_[((v-diffs[indices])/np.sqrt(len(indices))).ravel(),
                             .20*(v-parent_bias) if parent else .10*v]
            b = least_squares(bias_residual, np.median(diffs[indices], axis=0),
                loss=_weighted_huber_loss(.008, np.full(2*len(indices), 1/np.sqrt(len(indices))))).x
            if parent:
                extra = b-parent_bias
                extra *= min(1., .025/max(float(np.linalg.norm(extra)), 1e-9))
                b = parent_bias+extra
            raw_norm = float(np.linalg.norm(b))
            b *= min(1., cfg['maximum_chroma_shift']/max(raw_norm, 1e-9))
            return b, raw_norm
        b, raw_norm = solve_bias(np.arange(len(rows)))
        if parent:
            xy = np.array([a['xy'] for a in rows])
            bias_accepted, bias_improvement = False, None
            old_errors, new_errors = [], []
            if len(rows) >= 9 and np.all(np.ptp(xy, axis=0) >= 12):
                order = np.argsort(xy[:, 0], kind='stable')
                for fold in range(3):
                    held = order[fold::3]; train = np.setdiff1d(np.arange(len(rows)), held)
                    trial, _ = solve_bias(train)
                    old_errors.extend(((diffs[held]-parent_bias)**2).ravel())
                    new_errors.extend(((diffs[held]-trial)**2).ravel())
                old_error, new_error = float(np.mean(old_errors)), float(np.mean(new_errors))
                bias_improvement = 1-new_error/max(old_error, 1e-12)
                bias_accepted = new_error < .9*old_error
            if not bias_accepted:
                b = parent_bias.copy()
                raw_norm = float(np.linalg.norm(b))
        exposure = _fit_exposure(rows, cfg, parent)
        operators[name] = {'matrix': m.tolist(), 'bias': b.tolist(), **exposure,
            'chroma_limited': raw_norm > cfg['maximum_chroma_shift'], 'anchors': len(rows),
            'evidence': {'mode': 'paired_patch', 'accepted': True,
                'source_samples': sum(int(a.get('pixels', 1)) for a in rows),
                'reference_samples': sum(int(a.get('pixels', 1)) for a in rows),
                'source_cells': len(rows), 'reference_cells': len(rows),
                'paired_anchor_count': len(rows), 'spatial_correspondence_allowed': True,
                'source_region': name, 'reference_region': name, 'lightness_statistical_tone': True}}
        if parent:
            operators[name].update(parent_region=name.replace('_hand', '_arm'),
                heldout_error_improvement=bias_improvement, chroma_inherited_from_arm=not bias_accepted,
                extra_chroma_limit=.025, bias_prior_strength=.20,
                bias_validation='three_folds_interleaved_by_source_x; actual_bounded_bias')
    return {'schema': SCHEMA, 'working_domain': 'display_linear_sRGB', 'settings': cfg,
        'operators': operators, 'matrix_fit': {'estimated': estimated, 'rank': rank,
        'condition': condition if condition is not None and np.isfinite(condition) else None},
        'reference_pixels_in_output': False, 'geometry_warp': False}


def region_weights(labels, protection, parameters, *, chroma=False, source_values=None):
    """Material weights with independently chosen L or ab protection dose.

    Protection controls application strength, not reference anchor eligibility.
    """
    protection = np.asarray(protection)
    if (protection.shape != labels.shape or not np.isfinite(protection).all()
            or protection.min() < 0 or protection.max() > 1):
        raise ValueError('Invalid source feature protection')
    structural = parameters['settings']['protection_residual_strength']
    if 'protection_chroma_strength' not in parameters['settings']:
        raise ValueError('Current matching parameters require explicit chroma protection strength')
    dose = parameters['settings']['protection_chroma_strength'] if chroma else structural
    if not np.isfinite(dose) or not 0 <= dose <= 1:
        raise ValueError('Invalid source feature protection strength')
    guarded = any(op.get('source_material_guard') for op in parameters['operators'].values())
    if guarded and (source_values is None or source_values.shape != (*labels.shape, 3)
                    or not np.isfinite(source_values).all()):
        raise ValueError('Source material guards require explicit source RGB on the mask canvas')
    source_lightness = to_lab(source_values)[..., 0] if guarded else None
    palette_regions = {}
    if chroma and parameters.get('automatic_material_palette'):
        from reference_material_palette import validate_palette
        palette_regions = validate_palette(parameters['automatic_material_palette'])['regions']
    result = []
    for name, ids in REGIONS.items():
        feature_palette = parameters.get('feature_palette_policy', {})
        if (chroma and name == 'face' and feature_palette.get('mode') ==
                'source_feature_chroma_only_face_distribution_bias'):
            ids = ids+feature_palette['source_feature_labels']
        mask = np.isin(labels, ids).astype(np.float32)
        material_dose = None
        guard = parameters['operators'].get(name, {}).get('source_material_guard')
        if guard:
            material_dose = np.clip((source_lightness-guard['minimum_L'])/guard['transition_L'], 0, 1)
            mask *= material_dose
        for hand in ('left_hand', 'right_hand'):
            if (not parameters.get('distribution_policy') and name == hand.replace('_hand', '_arm')
                    and hand not in parameters['operators']):
                mask = np.maximum(mask, np.isin(labels, REGIONS[hand]).astype(np.float32))
        if (name not in parameters['operators']
                and palette_regions.get(name, {}).get('status') != 'supported'):
            mask *= 0
        blurred = gaussian_filter(mask, 1.5)
        palette_floor = palette_regions.get(name, {}).get('source_skin_L_floor')
        if palette_floor is not None:
            if (source_values is None or source_values.shape != (*labels.shape, 3)
                    or not np.isfinite(source_values).all()):
                raise ValueError('Palette skin support requires explicit source RGB')
            palette_L = to_lab(source_values)[..., 0]
            blurred *= np.clip((palette_L-palette_floor)/.02, 0, 1)
        result.append(blurred*material_dose if material_dose is not None else blurred)
    weights = np.stack(result, axis=-1)
    weights /= np.maximum(weights.sum(-1, keepdims=True), 1)
    weights *= (1-(1-dose)*protection[..., None])
    if not chroma and parameters.get('feature_palette_policy', {}).get('mode') == 'source_feature_chroma_only_face_distribution_bias':
        # Gaussian mask feathering must not introduce face brightness into a
        # source lip. Only the separate chroma weights inherit its palette.
        weights[np.isin(labels, parameters['feature_palette_policy']['source_feature_labels']),
                list(REGIONS).index('face')] = 0
    return weights.astype(np.float32)


def apply(values, weights, parameters, *, strength=1., origin=(0, 0), canvas_size=None, chroma_weights=None):
    """Apply native pixels with sampled parameter masks; never resize the person."""
    if parameters.get('schema') != SCHEMA or parameters.get('working_domain') != 'display_linear_sRGB':
        raise ValueError('Colour parameters require the explicit appearance domain')
    if not 0 <= strength <= 1 or weights.shape != (*values.shape[:2], len(REGIONS)):
        raise ValueError('Invalid matching strength or source-coordinate weights')
    if values.ndim != 3 or values.shape[-1] != 3 or not np.isfinite(values).all():
        raise ValueError('Invalid source appearance values')
    for name, operator in parameters.get('operators', {}).items():
        matrix = np.asarray(operator.get('matrix'), dtype=float)
        bias = np.asarray(operator.get('bias'), dtype=float)
        exposure = float(operator.get('exposure_ev', float('nan')))
        if (name not in REGIONS or matrix.shape != (2, 2) or bias.shape != (2,)
                or not np.isfinite(matrix).all() or not np.isfinite(bias).all()
                or not np.isfinite(exposure)):
            raise ValueError('Invalid saved material colour operator')
    feature_palette = parameters.get('feature_palette_policy', {})
    if feature_palette.get('mode') == 'source_feature_chroma_only_face_distribution_bias':
        face_operator = parameters['operators'].get('face', {})
        if (face_operator.get('evidence', {}).get('mode') != 'material_distribution'
                or not np.array_equal(face_operator.get('matrix'), np.eye(2))
                or 'face' in parameters.get('conditional_chroma_residual', {})
                or 'face' in parameters.get('local_chroma_residual', {})):
            raise ValueError('Source feature tint requires only a supported face distribution bias')
    policy = parameters.get('protection_policy', {})
    split = policy == 'split_lightness_chroma' or (isinstance(policy, dict)
                                                and policy.get('mode') == 'split_lightness_chroma')
    if chroma_weights is None:
        if split:
            raise ValueError('Split protection requires explicit chroma_weights')
        chroma_weights = weights
    for name, field in (('lightness', weights), ('chroma', chroma_weights)):
        if (field.shape != weights.shape or not np.isfinite(field).all()
                or field.min() < 0 or field.max() > 1+1e-6
                or np.any(field.sum(-1) > 1+1e-6)):
            raise ValueError(f'Invalid {name} region weights')
    if strength == 0:
        return values.copy()
    lab = to_lab(values)
    delta = np.zeros_like(lab)
    for index, name in enumerate(REGIONS):
        op = parameters['operators'].get(name)
        if op is None:
            continue
        target = lab.copy()
        target[..., 0] = _apply_tone(lab[..., 0], op)
        target[..., 1:] = lab[..., 1:] @ np.asarray(op['matrix'], np.float32).T + op['bias']
        shade = parameters.get('conditional_chroma_residual', {}).get(name)
        if shade:
            target[..., 1:] += _shade_colour(shade, lab[..., shade['source_channel']])
        field = parameters.get('local_chroma_residual', {}).get(name)
        if field:
            if canvas_size is None:
                raise ValueError('Local residuals require explicit source canvas coordinates')
            target[..., 1:] += _local_colour(field, values.shape[:2], origin, canvas_size)
        for edit in parameters.get('authored_chroma_adjustments', []):
            if edit['region'] != name:
                continue
            if canvas_size is None:
                raise ValueError('Authored colour strokes require source canvas coordinates')
            from local_colour_brush import stroke_coverage
            shift = np.asarray(edit['chroma_shift'], np.float32)
            if shift.shape != (2,) or not np.isfinite(shift).all() or np.linalg.norm(shift) > .012001:
                raise ValueError('Authored chroma adjustment exceeds its bounded colour-only contract')
            high, low = edit['source_a_ramp']
            if not low < high < 0:
                raise ValueError('Authored green-tail support must use an explicit source colour range')
            gate = np.clip((high-lab[..., 1])/(high-low), 0, 1)
            gate = gate*gate*(3-2*gate)
            # Protected eyes/lips and semantic boundaries cannot receive this
            # authored skin correction; only the fully supported face does.
            gate *= np.clip((weights[..., index]-.9)/.1, 0, 1)
            if any(not 0 <= edit['stroke'].get(key, default) <= 1 for key, default in (('opacity', 1.), ('hardness', .5))):
                raise ValueError('Authored stroke opacity and hardness must remain in [0, 1]')
            ox, oy = origin
            coverage = stroke_coverage(edit['stroke'], (ox, oy, ox+values.shape[1], oy+values.shape[0]), canvas_size)
            target[..., 1:] += (coverage*gate)[..., None]*shift
        delta[..., 0] += (target[..., 0]-lab[..., 0])*weights[..., index]
        delta[..., 1:] += (target[..., 1:]-lab[..., 1:])*chroma_weights[..., index, None]
    target = lab + delta*strength
    rgb = from_lab(target)
    # Chroma compression preserves perceptual lightness; no photographic tone
    # shoulder is applied to an already display-referred reference or result.
    for _ in range(12):
        bad = (rgb.min(-1) < 0) | (rgb.max(-1) > 1)
        if not np.any(bad):
            break
        target[..., 1:][bad] *= .9
        rgb[bad] = from_lab(target[bad])
    result = np.clip(rgb, 0, 1).astype(np.float32)
    operator_indices = [i for i, name in enumerate(REGIONS) if name in parameters['operators']]
    inactive = ((weights[..., operator_indices].sum(-1) <= 1e-8)
                & (chroma_weights[..., operator_indices].sum(-1) <= 1e-8))
    result[inactive] = values[inactive]
    if parameters.get('automatic_material_palette'):
        from reference_material_palette import apply_palette
        result = apply_palette(result, parameters['automatic_material_palette'], chroma_weights,
                               source_values=values, strength=strength)
    return result
