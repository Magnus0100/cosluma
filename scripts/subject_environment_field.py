"""Fixed-policy low-frequency residuals at the current subject endpoint.

The caller validates the immutable correspondence binding before calling this
array-only producer. Confidence weights observations, not the artistic dose.
Spatial block holdouts are never passed to the field fit. They are evaluated
through the actual scene-balance pixel operator, including its gamut response.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, median_filter, gaussian_filter
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import LinearOperator, cg, splu

from reference_automatic_layers import lowpass, sample_reference
from reference_colour_fit import to_lab
from reference_scene_balance import apply_scene_balance

SCHEMA = 'subject_environment_field.v1'
RECIPE = 'validated_local_endpoint_residual_with_spatial_holdouts.v1'
OWNER_IDS = (4, 1, 13, 23)
GUIDANCE_POLICY = 'validated_proxy_owner_edge_v2'
CONFIG = {
    'opaque_alpha': .995, 'minimum_lowpass_support': .1,
    'maximum_analysis_long_side': 512, 'minimum_observed_pixels': 128,
    'minimum_training_pixels': 64, 'minimum_heldout_pixels': 24,
    'minimum_spatial_blocks': 8, 'minimum_block_observations': 8,
    'holdout_folds': 5, 'holdout_block_extent_fraction': .06,
    'minimum_holdout_block_side': 4, 'minimum_validation_improvement': .05,
    'maximum_fold_regression': .01, 'minimum_improved_fold_fraction': .6,
    'lambda': .5, 'unknown_zero_prior': .5, 'relative_tolerance': 1e-6,
    'maximum_cg_iterations': 192,
    'numeric_fallback': 'symmetric_ordered_sparse_LU_with_verified_rhs_relative_residual',
    'edge_scale_floor': .012, 'edge_scale_ceiling': .06,
    'edge_cutoff_sigmas': 3., 'minimum_ev': -.25, 'maximum_ev': .5,
    'maximum_ab': .025, 'minimum_observation_lightness': .03,
    'geometry_taper_extent_fraction': .006, 'minimum_taper_pixels': 2.,
    'zero_residual_tolerance': 2e-5,
    'response_scale_extent_fraction': .02, 'minimum_response_sigma': 2.,
    'macro_guidance_median_side': 7,
    'macro_observation_owner_scale_fraction': 1/24,
}


def _inputs(current, guide, reference, sl, rl, alpha, maps):
    current, guide, reference = [np.asarray(v, np.float32) for v in (current, guide, reference)]
    sl, rl, alpha = np.asarray(sl), np.asarray(rl), np.asarray(alpha, np.float32)
    if (current.ndim != 3 or current.shape[-1] != 3 or guide.shape != current.shape
            or reference.ndim != 3 or reference.shape[-1] != 3 or not current.size or not reference.size
            or sl.shape != current.shape[:2] or rl.shape != reference.shape[:2]
            or alpha.shape != sl.shape or sl.dtype.kind not in 'iu' or rl.dtype.kind not in 'iu'
            or sl.min() < 0 or rl.min() < 0):
        raise ValueError('Subject environment inputs need same-coordinate RGB, alpha and integer owners')
    for value in (current, guide, reference, alpha):
        if not np.isfinite(value).all() or value.min() < 0 or value.max() > 1:
            raise ValueError('Subject environment inputs require finite display-linear SDR values')
    if not isinstance(maps, dict) or any(k not in maps for k in ('ref_x', 'ref_y', 'confidence', 'valid')):
        raise ValueError('Subject environment needs independently validated local maps')
    mapped = {k: np.asarray(maps[k]) for k in ('ref_x', 'ref_y', 'confidence', 'valid')}
    if any(value.shape != sl.shape or not np.isfinite(value).all() for value in mapped.values()):
        raise ValueError('Subject environment maps must share the source proxy canvas')
    if (mapped['confidence'].min() < 0 or mapped['confidence'].max() > 1
            or not np.isin(mapped['valid'], (0, 1)).all()):
        raise ValueError('Subject environment map confidence/validity range is invalid')
    return current, guide, reference, sl, rl, alpha, mapped


def _resize(value, size, nearest=False):
    import cv2
    return cv2.resize(value, tuple(size), interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)


def _conductance(guide, domain):
    dh = np.linalg.norm(np.diff(guide, axis=1), axis=-1)
    dv = np.linalg.norm(np.diff(guide, axis=0), axis=-1)
    sh, sv = domain[:, 1:] & domain[:, :-1], domain[1:] & domain[:-1]
    values = np.concatenate((dh[sh], dv[sv]))
    scale = float(np.clip(np.quantile(values, .65) if values.size else 0,
                          CONFIG['edge_scale_floor'], CONFIG['edge_scale_ceiling']))
    def weight(delta, same):
        return np.exp(-.5*(delta/scale)**2)*same*(delta <= CONFIG['edge_cutoff_sigmas']*scale)
    return weight(dh, sh), weight(dv, sv), scale


def _fit(residual, confidence, eligible, domain, guide, *, smoothing=None):
    """No identity prior at observed or deliberately withheld observations."""
    gh, gv, edge_scale = _conductance(guide, domain)
    mass = confidence.astype(np.float64) + CONFIG['unknown_zero_prior']*(~eligible)
    degree = np.zeros_like(mass)
    degree[:, 1:] += gh; degree[:, :-1] += gh
    degree[1:] += gv; degree[:-1] += gv
    lam = CONFIG['lambda'] if smoothing is None else float(smoothing)
    n = mass.size
    nodes = np.arange(n).reshape(mass.shape)
    hr, hc = nodes[:, :-1].ravel(), nodes[:, 1:].ravel()
    vr, vc = nodes[:-1].ravel(), nodes[1:].ravel()
    hw, vw = (-lam*gh).ravel(), (-lam*gv).ravel()
    selected_h, selected_v = hw != 0, vw != 0
    row = np.concatenate((np.arange(n), hr[selected_h], hc[selected_h], vr[selected_v], vc[selected_v]))
    col = np.concatenate((np.arange(n), hc[selected_h], hr[selected_h], vc[selected_v], vr[selected_v]))
    data = np.concatenate(((mass+lam*degree).ravel(), hw[selected_h], hw[selected_h], vw[selected_v], vw[selected_v]))
    matrix = coo_matrix((data, (row, col)), shape=(n, n)).tocsc()
    matrix.eliminate_zeros()
    # A completely withheld connected component has no data or prior anchor.
    # Its Laplacian has a constant null space; the zero minimum-norm solution
    # is explicit and is still scored against the withheld targets. Anchoring
    # only these null-space components changes no observed-data solution.
    component_count, components = connected_components(matrix, directed=False)
    anchored = np.bincount(components, weights=mass.ravel(), minlength=component_count) > 0
    unanchored = ~anchored[components]
    if unanchored.any():
        matrix += diags(unanchored.astype(float), format='csc')
        mass.ravel()[unanchored] = 1
    def action(values):
        out = mass*values
        flux = lam*gh*(values[:, :-1]-values[:, 1:])
        out[:, :-1] += flux; out[:, 1:] -= flux
        flux = lam*gv*(values[:-1]-values[1:])
        out[:-1] += flux; out[1:] -= flux
        return out
    operator = LinearOperator((n, n), matvec=lambda v: action(v.reshape(mass.shape)).ravel(), dtype=float)
    diagonal = mass+lam*degree
    preconditioner = LinearOperator((n, n), matvec=lambda v: v/diagonal.ravel(), dtype=float)
    solutions = np.zeros((n, 3), float); right_sides = np.zeros((n, 3), float)
    counts = []; cg_info = []; initial_relative = []; initial_absolute = []; fallback = False
    def measured_error(values, rhs):
        difference = matrix@values-rhs
        absolute, rhs_norm = float(np.linalg.norm(difference)), float(np.linalg.norm(rhs))
        relative = absolute/rhs_norm if rhs_norm else (0. if absolute == 0 else float('inf'))
        return relative, absolute, rhs_norm
    for channel in range(3):
        rhs = (confidence*residual[..., channel]).astype(float).ravel(); count = [0]
        def step(_value): count[0] += 1
        values, info = cg(operator, rhs, M=preconditioner, rtol=CONFIG['relative_tolerance'], atol=1e-10,
                          maxiter=CONFIG['maximum_cg_iterations'], callback=step)
        if info < 0 or not np.isfinite(values).all():
            raise ValueError('Subject environment field solver failed')
        relative, absolute, _ = measured_error(values, rhs)
        solutions[:, channel] = values; right_sides[:, channel] = rhs
        counts.append(count[0]); cg_info.append(int(info))
        initial_relative.append(relative); initial_absolute.append(absolute)
        fallback |= info != 0 or relative > CONFIG['relative_tolerance']
    factorization_error = None
    if fallback:
        try:
            factor = splu(matrix, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0., options={'SymmetricMode': True})
            solutions = factor.solve(right_sides)
        except (RuntimeError, ValueError) as exc:
            factorization_error = type(exc).__name__
    final_errors = [measured_error(solutions[:, c], right_sides[:, c]) for c in range(3)]
    converged = (factorization_error is None and np.isfinite(solutions).all()
                 and all(np.isfinite(value[0]) and value[0] <= CONFIG['relative_tolerance'] for value in final_errors))
    result = solutions.reshape((*mass.shape, 3)).astype(np.float32)
    result[~domain] = 0
    return _project(result), {'converged': bool(converged), 'iterations': counts,
        'primary_method': 'Jacobi_preconditioned_CG', 'cg_info': cg_info,
        'primary_rhs_relative_residuals': initial_relative, 'primary_absolute_residuals': initial_absolute,
        'direct_fallback_used': bool(fallback), 'direct_fallback_method': CONFIG['numeric_fallback'] if fallback else None,
        'factorization_error': factorization_error, 'matrix_variables': n, 'matrix_nonzeros': int(matrix.nnz),
        'unanchored_zero_solution_pixels': int(unanchored.sum()),
        'unanchored_components': int(np.count_nonzero(~anchored)),
        'rhs_relative_residuals': [value[0] for value in final_errors],
        'absolute_residuals': [value[1] for value in final_errors], 'rhs_norms': [value[2] for value in final_errors],
        'measured_residual_tolerance': CONFIG['relative_tolerance'], 'smoothing_lambda': lam,
        'numeric_residual_measured_before_artistic_projection': True,
        'conductance_edge_scale': edge_scale, 'identity_prior_at_observed_pixels': False,
        'semantic_boundary_conductance': 0., 'strong_material_edge_cutoff': CONFIG['edge_cutoff_sigmas']*edge_scale}


def _project(values):
    result = np.asarray(values, np.float32).copy()
    result[..., 0] = np.clip(result[..., 0], CONFIG['minimum_ev'], CONFIG['maximum_ev'])
    norm = np.linalg.norm(result[..., 1:3], axis=-1)
    result[..., 1:3] *= np.minimum(1, CONFIG['maximum_ab']/np.maximum(norm, 1e-20))[..., None]
    return result


def _macro_guidance(guide, domain):
    """Remove narrow texture from correction guidance, never from output RGB."""
    if not domain.any(): return np.zeros_like(guide)
    # Extend the owner's edge before the statistic, so zeros outside a small
    # hair/cloth owner cannot bias its guidance. Owner validity remains exact.
    _, indices = distance_transform_edt(~domain, return_indices=True)
    extended = guide[tuple(indices)]
    side = CONFIG['macro_guidance_median_side']
    return np.stack([median_filter(extended[..., k], size=side, mode='nearest')
                     for k in range(3)], axis=-1).astype(np.float32)


def _regularize_response(values, domain, guide):
    """Impose an explicit macro response scale on the field itself.

    A field with +EV piping islands beside -EV fabric can pass sparse colour
    prediction yet damage continuous trim. A Laplacian response smoother on
    persistent source-material edges constrains the delivered field; exactly
    the same operation is scored at withheld observations before acceptance.
    """
    sigma = max(CONFIG['minimum_response_sigma'],
                max(domain.shape)*CONFIG['response_scale_extent_fraction'])
    response, solve = _fit(values, domain.astype(np.float32), domain, domain,
                           guide, smoothing=sigma*sigma)
    solve['response_sigma'] = sigma
    return response, solve


def _macro_observations(values, observations, domain, guide):
    """Filter both photographs on the same persistent source material domains.

    Reference line placement cannot define a lighting partition. Held target
    values are never copied to the numerator, weights, guide or denominator.
    """
    gh, gv, _ = _conductance(guide, domain)
    h, w = domain.shape; nodes = np.arange(h*w).reshape((h, w))
    hr, hc = nodes[:, :-1][gh > 0], nodes[:, 1:][gh > 0]
    vr, vc = nodes[:-1][gv > 0], nodes[1:][gv > 0]
    row = np.concatenate((hr, hc, vr, vc))
    col = np.concatenate((hc, hr, vc, vr))
    graph = coo_matrix((np.ones(len(row)), (row, col)), shape=(h*w, h*w)).tocsr()
    _, owners = connected_components(graph, directed=False)
    owners = owners.reshape((h, w))
    sigma = max(CONFIG['minimum_response_sigma'],
                np.sqrt(float(domain.sum()))*CONFIG['macro_observation_owner_scale_fraction'])
    output = np.zeros_like(values, np.float32); support = np.zeros(domain.shape, np.float32)
    for owner in np.unique(owners[domain]):
        part = domain & (owners == owner)
        yy, xx = np.nonzero(part)
        crop = np.s_[yy.min():yy.max()+1, xx.min():xx.max()+1]
        selected = part[crop] & observations[crop]
        if not selected.any(): continue
        weight = gaussian_filter(selected.astype(np.float32), sigma, mode='constant')
        observed = np.zeros_like(values[crop], np.float32)
        observed[selected] = values[crop][selected]
        numerator = gaussian_filter(observed, (sigma, sigma, 0), mode='constant')
        result = numerator/np.maximum(weight[..., None], 1e-20)
        output[crop][part[crop]] = result[part[crop]]
        support[crop][part[crop]] = weight[part[crop]]
    return output, support, sigma


def proxy_response_guidance(guide_lab, labels):
    """The same bounded macro owner guidance used by native interpolation."""
    result = np.zeros_like(guide_lab, np.float32)
    for owner in OWNER_IDS:
        domain = labels == owner
        yy, xx = np.nonzero(domain)
        if not yy.size: continue
        crop = np.s_[yy.min():yy.max()+1, xx.min():xx.max()+1]
        h, w = domain[crop].shape
        factor = min(1., CONFIG['maximum_analysis_long_side']/max(h, w))
        size = (max(1, round(w*factor)), max(1, round(h*factor)))
        low, _, _ = lowpass(guide_lab[crop], domain[crop])
        small_domain = _resize(domain[crop].astype(np.uint8), size, True).astype(bool)
        macro = _macro_guidance(_resize(low, size, True), small_domain)
        result[crop] = _resize(macro, (w, h))
    return result


def _taper(eligible):
    yy, xx = np.nonzero(eligible)
    if not len(xx): return np.zeros_like(eligible, np.float32)
    extent = max(int(yy.max()-yy.min()+1), int(xx.max()-xx.min()+1))
    width = max(CONFIG['minimum_taper_pixels'], extent*CONFIG['geometry_taper_extent_fraction'])
    distance = distance_transform_edt(np.pad(eligible, 1, constant_values=False))[1:-1, 1:-1]
    weight = np.clip((distance-.5)/width, 0, 1)
    return (weight*weight*(3-2*weight)).astype(np.float32)


def _pack(values, eligible, taper):
    field = np.zeros((*eligible.shape, 4), np.float32)
    field[..., :3] = values*taper[..., None]
    field[~eligible] = 0
    active = np.any(field[..., :3] != 0, axis=-1)
    field[..., 3] = np.where(active, taper, 0)
    return field


def _error(lab, target):
    ev = 3*np.log2(np.maximum(lab[..., 0], .03)/np.maximum(target[..., 0], .03))
    delta = np.concatenate((ev[..., None]/CONFIG['maximum_ev'],
                            (lab[..., 1:]-target[..., 1:])/CONFIG['maximum_ab']), axis=-1)
    return np.sum(delta.astype(float)**2, axis=-1)


def _training_residual(raw_target, current_low, training, *, domain=None, guide=None):
    """Split before target filtering: withheld values are not even read here.

    The zero-initialized array also prevents withheld values from becoming the
    edge guide, filter scale estimate or denominator's sampled numerator.
    Source observations may be shared; reference observations may not.
    """
    observed = np.zeros_like(raw_target, np.float32)
    observed[training] = raw_target[training]
    if domain is None:
        target_low, support, sigma = lowpass(observed, training)
    else:
        target_low, support, sigma = _macro_observations(observed, training, domain, guide)
    residual = np.zeros_like(raw_target, np.float32)
    ev = 3*np.log2(np.maximum(target_low[..., 0], .03)/np.maximum(current_low[..., 0], .03))
    residual[..., 0][training] = ev[training]
    residual[..., 1:][training] = (target_low[..., 1:]-current_low[..., 1:])[training]
    return residual, {'target_filter_split_before_fit': True,
        'target_filter_domain': 'source_macro_material' if domain is not None else 'training_observations',
        'target_filter_observation_pixels': int(training.sum()), 'target_filter_sigma': sigma,
        'minimum_training_lowpass_support': float(support[training].min()) if training.any() else None}


def _evaluate(current_rgb, target_lab, field, selected, confidence):
    after = apply_scene_balance(current_rgb, field)
    before_lab, after_lab = to_lab(current_rgb), to_lab(after)
    weights = confidence[selected].astype(float)
    mass = max(float(weights.sum()), 1e-20)
    before = float(np.sum(_error(before_lab, target_lab)[selected]*weights)/mass)
    error = float(np.sum(_error(after_lab, target_lab)[selected]*weights)/mass)
    achieved_ev = 3*np.log2(np.maximum(after_lab[..., 0], .03)/np.maximum(before_lab[..., 0], .03))
    achieved = np.concatenate((achieved_ev[..., None], after_lab[..., 1:]-before_lab[..., 1:]), axis=-1)
    difference = np.abs(achieved-field[..., :3])[selected]
    return {'before_error': before, 'after_error': error,
        'improvement': float((before-error)/max(before, 1e-20)),
        'actual_pixel_operator': 'reference_scene_balance.apply_scene_balance',
        'gamut_and_highlight_response_included': True,
        'requested_vs_actual_response_abs_q50_95': np.quantile(difference, [.5, .95], axis=0).tolist(),
        'heldout_pixels': int(selected.sum()), 'heldout_confidence_weight': float(weights.sum())}


def _guided_upsample(values, small_guide, small_domain, domain):
    """Upsample a field from proxy guidance without importing fine texture.

    Only validated proxy guidance supplies edges. Native weave and grain do
    not participate in the low-frequency correction field.
    """
    if values.shape[:2] == domain.shape: return values
    h, w = domain.shape; sh, sw = small_domain.shape
    ys = np.linspace(0, sh-1, h); xs = np.linspace(0, sw-1, w)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.minimum(y0+1, sh-1), np.minimum(x0+1, sw-1)
    fy, fx = ys-y0, xs-x0
    result = np.zeros((*domain.shape, 3), np.float32); denominator = np.zeros(domain.shape, np.float32)
    _, _, scale = _conductance(small_guide, small_domain)
    nearest_y = np.clip(np.floor(ys+.5).astype(int), 0, sh-1)
    nearest_x = np.clip(np.floor(xs+.5).astype(int), 0, sw-1)
    centre = small_guide[np.ix_(nearest_y, nearest_x)]
    for iy, wy in ((y0, 1-fy), (y1, fy)):
        for ix, wx in ((x0, 1-fx), (x1, fx)):
            grid = (iy[:, None], ix[None, :])
            delta = np.linalg.norm(small_guide[grid]-centre, axis=-1)
            weight = wy[:, None]*wx[None, :]*domain*small_domain[grid]
            weight *= np.exp(-.5*(delta/scale)**2)*(delta <= CONFIG['edge_cutoff_sigmas']*scale)
            result += values[grid]*weight[..., None]; denominator += weight
    return result/np.maximum(denominator[..., None], 1e-20)


def _split_reference_observations(eligible, held, reference_ids):
    """A reference pixel observed by the held set is unavailable to training."""
    shared = np.isin(reference_ids, np.unique(reference_ids[held]))
    training = eligible & ~held & ~shared
    return training, int(np.count_nonzero(eligible & ~held & shared))


def prepare_environment_field(current_rgb, source_guide_rgb, reference_rgb,
                              source_labels, reference_labels, alpha, maps):
    """Return preweighted EV/ab/coverage and an auditable numeric receipt.

    Passing spatial validation does not certify correspondence geometry or
    artistic quality. The worker owns upstream binding and fresh visual QA.
    """
    current, guide, reference, sl, rl, alpha, maps = _inputs(
        current_rgb, source_guide_rgb, reference_rgb, source_labels, reference_labels, alpha, maps)
    current_lab, guide_lab, reference_lab = to_lab(current), to_lab(guide), to_lab(reference)
    sampled_labels = sample_reference(rl, maps, nearest=True).astype(np.int32)
    inside = ((maps['ref_x'] >= 0) & (maps['ref_x'] <= reference.shape[1]-1)
              & (maps['ref_y'] >= 0) & (maps['ref_y'] <= reference.shape[0]-1))
    output = np.zeros((*sl.shape, 4), np.float32); owners = []
    for owner_id in OWNER_IDS:
        domain = (sl == owner_id) & (alpha >= CONFIG['opaque_alpha'])
        row = {'semantic_class_id': owner_id, 'state': 'rejected', 'source_owner_pixels': int(domain.sum()),
               'changed_pixels': 0, 'visual_acceptance': False}
        owners.append(row)
        if not domain.any() or not np.any(rl == owner_id):
            row['reason'] = 'source_or_reference_owner_absent'; continue
        source_low, ss, source_sigma = lowpass(current_lab, domain)
        # Nearest observation ownership is explicit. Bilinear observations
        # could mix a held-out reference pixel into a neighbouring training
        # observation before the spatial validation split.
        target = sample_reference(reference_lab, maps, nearest=True)
        eligible = (domain & maps['valid'].astype(bool) & inside & (maps['confidence'] > 0)
                    & (sampled_labels == owner_id) & (ss >= CONFIG['minimum_lowpass_support'])
                    )
        confidence = np.where(eligible, maps['confidence'], 0).astype(np.float32)
        row.update(observed_pixels=int(eligible.sum()), source_observation_sigma=source_sigma,
                   reference_observation_sampling='nearest_raw_Lab; split_before_target_filter',
                   observed_source_fraction=float(eligible.sum()/max(1, domain.sum())))
        if int(eligible.sum()) < CONFIG['minimum_observed_pixels']:
            row['reason'] = 'insufficient_independent_local_observations'; continue
        direct = np.concatenate((3*np.log2(np.maximum(target[..., 0], .03)/np.maximum(current_lab[..., 0], .03))[..., None],
                                  target[..., 1:]-current_lab[..., 1:]), axis=-1).astype(np.float32)
        normalized = direct/np.array([CONFIG['maximum_ev'], CONFIG['maximum_ab'], CONFIG['maximum_ab']], np.float32)
        if np.max(np.abs(normalized[eligible]), initial=0) <= CONFIG['zero_residual_tolerance']:
            row.update(state='identity', reason='current_endpoint_already_matches_reference_low_frequencies'); continue
        yy, xx = np.nonzero(domain)
        y0, y1, x0, x1 = int(yy.min()), int(yy.max()+1), int(xx.min()), int(xx.max()+1)
        crop = np.s_[y0:y1, x0:x1]; h, w = domain[crop].shape
        factor = min(1., CONFIG['maximum_analysis_long_side']/max(h, w))
        size = (max(1, round(w*factor)), max(1, round(h*factor)))
        ca = _resize(confidence[crop], size, True); ea = ca > 0
        da = _resize(domain[crop].astype(np.uint8), size, True).astype(bool)
        # The conductance guide is itself policy-filtered before the proxy
        # solve.  Using raw proxy Lab here lets weave/grain alter edge scale
        # and therefore the entire correction field even when native
        # interpolation is texture-invariant.
        guide_low, _, guide_sigma = lowpass(guide_lab[crop], domain[crop])
        ga = _macro_guidance(_resize(guide_low, size, True), da)
        source_low_a = _resize(source_low[crop], size, True)
        rgb_a = _resize(current[crop], size, True); target_a = _resize(target[crop], size, True)
        source_low_a, _, _ = _macro_observations(to_lab(rgb_a), da, da, ga)
        rx = _resize(np.clip(maps['ref_x'][crop], 0, reference.shape[1]-1), size, True)
        ry = _resize(np.clip(maps['ref_y'][crop], 0, reference.shape[0]-1), size, True)
        reference_ids = np.rint(ry).astype(np.int64)*reference.shape[1]+np.rint(rx).astype(np.int64)
        taper_a = _taper(ea)
        by, bx = np.mgrid[:ea.shape[0], :ea.shape[1]]
        block = max(CONFIG['minimum_holdout_block_side'], round(min(ea.shape)*CONFIG['holdout_block_extent_fraction']))
        block_ids = (by//block)*int(np.ceil(ea.shape[1]/block)) + bx//block
        used_blocks = [int(value) for value in np.unique(block_ids[ea])
                       if np.count_nonzero(ea & (block_ids == value)) >= CONFIG['minimum_block_observations']]
        row.update(analysis_shape=list(ea.shape), analysis_sampling='nearest_owner_observation_and_source_guide',
                   guidance_policy=GUIDANCE_POLICY, guidance_sigma=float(guide_sigma),
                   validation_block_size=block, observed_spatial_blocks=len(used_blocks))
        if len(used_blocks) < CONFIG['minimum_spatial_blocks'] or int(ea.sum()) < CONFIG['minimum_observed_pixels']:
            row['reason'] = 'insufficient_spatial_holdout_support'; continue
        folds = (by//block+2*(bx//block)) % CONFIG['holdout_folds']
        validation = []; fitting_failed = False
        for fold in range(CONFIG['holdout_folds']):
            held = ea & (folds == fold)
            training, shared_count = _split_reference_observations(ea, held, reference_ids)
            if (held.sum() < CONFIG['minimum_heldout_pixels']
                    or training.sum() < CONFIG['minimum_training_pixels']):
                fitting_failed = True; break
            training_residual, observation = _training_residual(target_a, source_low_a, training, domain=da, guide=ga)
            fitted, solve = _fit(training_residual, np.where(training, ca, 0), ea, da, ga)
            fitted, response_solve = _regularize_response(fitted, da, ga)
            tested = _pack(fitted, ea, taper_a)
            values = _evaluate(rgb_a, target_a, tested, held, ca)
            validation.append({'fold': fold, 'training_pixels': int(training.sum()),
                'heldout_blocks': len(np.unique(block_ids[held])), 'target_values_used_by_fit': False,
                'shared_reference_pixels_excluded_from_training': shared_count,
                'heldout_target_basis': 'raw_nearest_reference_Lab; not_target_filtered_with_training',
                'training_observation': observation, 'solver': solve,
                'macro_response_solver': response_solve, **values})
            fitting_failed |= not solve['converged'] or not response_solve['converged']
        if validation:
            mass = sum(v['heldout_confidence_weight'] for v in validation)
            before = sum(v['before_error']*v['heldout_confidence_weight'] for v in validation)/mass
            after = sum(v['after_error']*v['heldout_confidence_weight'] for v in validation)/mass
            improvement = (before-after)/max(before, 1e-20)
        else:
            before = after = improvement = 0.
        passed = (not fitting_failed and len(validation) == CONFIG['holdout_folds']
                  and improvement >= CONFIG['minimum_validation_improvement']
                  and all(v['improvement'] >= -CONFIG['maximum_fold_regression'] for v in validation)
                  and sum(v['improvement'] > 0 for v in validation)/max(1, len(validation)) >= CONFIG['minimum_improved_fold_fraction'])
        row['spatial_validation'] = {'passed': bool(passed), 'folds': validation,
            'before_error': float(before), 'after_error': float(after), 'improvement': float(improvement),
            'metrics': 'confidence_weighted_actual_pixel_EV_ab_error_on_withheld_spatial_blocks',
            'training_error_used_for_acceptance': False, 'correspondence_geometry_certified': False}
        if not passed:
            row['reason'] = 'withheld_spatial_pixels_did_not_validate_actual_response'; continue
        final_residual, final_observation = _training_residual(target_a, source_low_a, ea, domain=da, guide=ga)
        fitted, solve = _fit(final_residual, ca, ea, da, ga)
        fitted, response_solve = _regularize_response(fitted, da, ga)
        if not solve['converged'] or not response_solve['converged']:
            row['reason'] = 'final_numeric_field_solve_not_converged'; continue
        fine = _guided_upsample(fitted, ga, da, domain[crop])
        packed = _pack(_project(fine), eligible[crop], _taper(eligible[crop]))
        output[crop] += packed
        row.update(state='prepared', reason='actual_response_validated_on_independent_spatial_holdouts',
                   solver=solve, macro_response_solver=response_solve,
                   changed_pixels=int(np.count_nonzero(np.any(packed[..., :3] != 0, axis=-1))),
                   final_observation=final_observation,
                   applied_ev_q10_50_90=np.quantile(packed[..., 0][eligible[crop]], [.1, .5, .9]).tolist(),
                   maximum_ab=float(np.linalg.norm(packed[..., 1:3], axis=-1).max()),
                   confidence_is_effect_dose=False)
    allowed = np.isin(sl, OWNER_IDS) & (alpha >= CONFIG['opaque_alpha'])
    if np.any(output[~allowed] != 0) or not np.isfinite(output).all():
        raise ValueError('Subject environment field violated source ownership')
    report = {'schema': SCHEMA, 'recipe': RECIPE, 'config': dict(CONFIG), 'owners': owners,
        'guidance_policy': GUIDANCE_POLICY,
        'state': 'prepared' if np.any(output[..., :3] != 0) else 'identity_or_unsupported',
        'changed_pixels': int(np.count_nonzero(np.any(output[..., :3] != 0, axis=-1))),
        'working_domain': 'display_linear_sRGB', 'field_representation': 'preweighted_EV_da_db_and_geometric_QA_coverage',
        'coverage_consumed_at_preparation': True, 'confidence_is_effect_dose': False,
        'source_geometry_warped': False, 'alpha_modified': False,
        'reference_RGB_or_texture_used_as_output': False, 'visual_acceptance': False,
        'binding_validation_owned_by_caller': True,
        'limits': ['Spatial holdouts test prediction at withheld observations, not artistic quality.',
                   'A smooth but geometrically wrong same-owner map cannot be certified by colour validation.',
                   'Regions without independently qualified maps remain unchanged.',
                   'Only bounded proxy low-frequency observations drive the frozen field.']}
    return output, report
