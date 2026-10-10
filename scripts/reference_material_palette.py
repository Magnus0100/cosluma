"""Automatic material colour groups with an explicit unmatched transport bin.

The producer observes independent source/reference semantic interiors. It never
asserts pixel correspondence. The consumer adds a bounded ab residual after T,
using immutable source colours and source-coordinate semantic weights only.
No reference pixels, model runtime, spatial warping or new lightness are needed
to replay this JSON-safe palette on a native tile.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
from scipy.ndimage import binary_erosion

from reference_colour_fit import REGIONS, apply, region_weights
from colour_space import to_lab, from_lab


SCHEMA = 'reference_material_palette.v1'
SETTINGS = {
    'cell_size': 4, 'spatial_split_block': 32, 'minimum_cells': 36,
    'minimum_group_cells': 8, 'maximum_groups': 4,
    'feature_scale_floor': [.06, .012, .012], 'group_separation': .9,
    'soft_width': .55, 'transport_entropy': .065, 'dustbin_cost': .45,
    'minimum_matched_fraction': .55, 'maximum_residual_chroma': .025,
    'maximum_total_chroma': .065, 'minimum_relative_improvement': .10,
    'minimum_absolute_improvement': .0005, 'maximum_fold_disagreement': .009,
    'support_radius': 3., 'support_margin_Lab': [.04, .018, .018],
}


def _rgb(values, name):
    values = np.asarray(values)
    if (values.ndim != 3 or values.shape[-1] != 3 or not values.size
            or not np.issubdtype(values.dtype, np.number)
            or not np.isfinite(values).all() or values.min() < 0 or values.max() > 1):
        raise ValueError(f'{name} requires finite display-linear RGB in [0,1]')
    return values.astype(np.float32, copy=False)


def _mask(value, shape, name, *, labels=False):
    value = np.asarray(value)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f'{name} must be finite on its own RGB canvas')
    if labels:
        if value.min() < 0 or np.any(value != np.round(value)):
            raise ValueError('Semantic labels must be nonnegative integers')
    elif value.min() < 0 or value.max() > 1:
        raise ValueError(f'{name} must be in [0,1]')
    return value


def _digest(value):
    if isinstance(value, np.ndarray):
        return hashlib.sha256(value.tobytes()).hexdigest()
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _baseline(source, labels, parameters, protection):
    # Do not recursively fit/replay a palette being replaced by this producer.
    parameters = {k: v for k, v in parameters.items()
                  if k not in ('material_palette', 'automatic_material_palette')}
    weights = region_weights(labels, protection, parameters, source_values=source)
    chroma = region_weights(labels, protection, parameters, chroma=True, source_values=source)
    return apply(source, weights, parameters, chroma_weights=chroma,
                 canvas_size=(source.shape[1], source.shape[0]))


def _samples(rgb, lab, labels, alpha, protection, ids, *, family_floor=None):
    """Uniform safe cells; split whole spatial blocks, never colour ranks."""
    cfg = SETTINGS
    selected = binary_erosion(np.isin(labels, ids), iterations=2)
    selected &= (alpha >= .995) & (protection < .05)
    selected &= (lab[..., 0] > .10) & (lab[..., 0] < .96) & (rgb.max(-1) < .985)
    if family_floor is not None:
        selected &= lab[..., 0] >= family_floor
    step = cfg['cell_size']; rows = []; locations = []; folds = []
    ys, xs = np.nonzero(selected)
    if not len(xs):
        return np.empty((0, 3)), np.empty((0, 2), int), np.empty(0, int)
    for y in range(int(ys.min()), int(ys.max())+1, step):
        for x in range(int(xs.min()), int(xs.max())+1, step):
            mask = selected[y:y+step, x:x+step]
            if mask.shape != (step, step) or mask.mean() < .75:
                continue
            rows.append(np.median(lab[y:y+step, x:x+step][mask], axis=0))
            locations.append([x+step//2, y+step//2])
            folds.append(((x//cfg['spatial_split_block'])+2*(y//cfg['spatial_split_block'])) % 3)
    return np.asarray(rows, float).reshape(-1, 3), np.asarray(locations, int).reshape(-1, 2), np.asarray(folds, int)


def _normalization(rows):
    low, high = np.quantile(rows, [.05, .95], axis=0)
    # Midrange normalization stays independent of the area of black/white
    # groups. A median/IQR normalizer would change roles when their proportions
    # swap and incorrectly make source black closest to reference white.
    center = (low+high)/2
    scale = np.maximum(high-low, SETTINGS['feature_scale_floor'])
    return center, scale


def _soft(rows, model):
    normalized = (rows-np.asarray(model['center']))/np.asarray(model['scale'])
    centers = np.asarray(model['centers'])
    squared = np.sum((normalized[:, None]-centers[None])**2, axis=-1)
    logits = -squared/(2*SETTINGS['soft_width']**2)
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights, np.sqrt(squared.min(axis=1))


def _groups(rows):
    """Deterministic robust colour groups; independent per-image normalization."""
    if len(rows) < SETTINGS['minimum_cells']:
        return None
    center, scale = _normalization(rows)
    z = np.clip((rows-center)/scale, -5, 5)
    order_by_L = np.argsort(z[:, 0], kind='stable')
    # Start at an observed tonal mode, not the between-mode median of a
    # 50/50 black/white material. That median may describe no real material.
    seeds = [z[order_by_L[len(order_by_L)//10]]]
    for _ in range(1, SETTINGS['maximum_groups']):
        distance = np.min(np.sum((z[:, None]-np.asarray(seeds)[None])**2, axis=-1), axis=1)
        # Gross tails do not create a transport group on their own.
        candidates = np.argsort(distance, kind='stable')
        index = candidates[max(0, len(candidates)-max(1, len(candidates)//25))]
        if distance[index] < SETTINGS['group_separation']**2:
            break
        seeds.append(z[index])
    centers = np.asarray(seeds)
    for _ in range(24):
        distances = np.sum((z[:, None]-centers[None])**2, axis=-1)
        assignment = distances.argmin(axis=1)
        updated = []
        for i in range(len(centers)):
            members = z[assignment == i]
            if len(members) >= SETTINGS['minimum_group_cells']:
                updated.append(np.median(members, axis=0))
        if not updated:
            return None
        updated = np.asarray(updated)
        if updated.shape == centers.shape and np.max(abs(updated-centers)) < 1e-5:
            centers = updated; break
        centers = updated
    order = np.lexsort((centers[:, 2], centers[:, 1], centers[:, 0]))
    return {'center': center.tolist(), 'scale': scale.tolist(), 'centers': centers[order].tolist()}


def _representatives(rows, model, *, minimum=None):
    memberships, _ = _soft(rows, model)
    hard = memberships.argmax(axis=1)
    representatives = []; counts = []
    for i in range(len(model['centers'])):
        selected = rows[hard == i]
        count = len(selected); counts.append(count)
        representatives.append(np.median(selected, axis=0) if count else np.zeros(3))
    return np.asarray(representatives), np.asarray(counts)


def _spectrum(lightness):
    # Broad dark/bright material classes survive unequal area proportions.
    # They supplement semantic labels and normalized features; nearest raw
    # chroma alone is never accepted as evidence of material correspondence.
    return np.stack([np.clip((lightness-.23)/.10, 0, 1),
                     np.clip((lightness-.60)/.14, 0, 1)], axis=-1)


def _transport(source_model, reference_model, source_rep, reference_rep):
    zs = np.asarray(source_model['centers']); zr = np.asarray(reference_model['centers'])
    normalized_cost = np.sum((zs[:, None]-zr[None])**2 * [1., .25, .25], axis=-1)
    spectrum_cost = 1.8*np.sum((_spectrum(source_rep[:, 0])[:, None]-
                               _spectrum(reference_rep[:, 0])[None])**2, axis=-1)
    cost = .35*normalized_cost+spectrum_cost
    n, m = cost.shape
    extended = np.full((n+1, m+1), SETTINGS['dustbin_cost'], float)
    extended[:n, :m] = cost; extended[-1, -1] = 0.
    # Equal mass per observed group prevents a large white area from forcing
    # black clothing into white merely to satisfy area/histogram proportions.
    a = np.r_[np.full(n, 1/n), 1.]; b = np.r_[np.full(m, 1/m), 1.]
    kernel = np.exp(-extended/SETTINGS['transport_entropy'])
    u = np.ones(n+1); v = np.ones(m+1)
    for _ in range(180):
        u = a/np.maximum(kernel@v, 1e-30)
        v = b/np.maximum(kernel.T@u, 1e-30)
    plan = u[:, None]*kernel*v[None]
    real = plan[:n, :m]; fraction = np.clip(real.sum(axis=1)/a[:n], 0, 1)
    relation = real/np.maximum(real.sum(axis=1, keepdims=True), 1e-30)
    # A diffuse row that spans distinct dark/bright materials is unresolved.
    for i in range(n):
        targets = reference_rep[:, 0]
        variance = np.sum(relation[i]*(targets-np.sum(relation[i]*targets))**2)
        if variance > .0225:
            fraction[i] = 0.
    accepted = fraction >= SETTINGS['minimum_matched_fraction']
    return relation, fraction, accepted, {'cost': cost.tolist(), 'plan': real.tolist(),
        'unmatched_source_fraction': (1-fraction).tolist(), 'source_groups': n, 'reference_groups': m,
        'entropy': SETTINGS['transport_entropy'], 'dustbin_cost': SETTINGS['dustbin_cost'],
        'group_mass_policy': 'equal_observed_group_mass; area_proportions_not_forced',
        'association_features': 'independently_normalized_Lab_plus_broad_L_material_class; not_nearest_raw_colour',
        'material_truth': 'semantic_interiors_and_stable_split_evidence; association_is_distribution_only',
        'spatial_correspondence_allowed': False}


def _limited(delta, maximum):
    norm = np.linalg.norm(delta, axis=-1, keepdims=True)
    return delta*np.minimum(1, maximum/np.maximum(norm, 1e-30))


def _fit_model(source_rows, baseline_rows, reference_rows):
    source_model, reference_model = _groups(source_rows), _groups(reference_rows)
    if source_model is None or reference_model is None:
        return None
    source_rep, source_count = _representatives(source_rows, source_model)
    reference_rep, reference_count = _representatives(reference_rows, reference_model)
    if min(source_count.min(), reference_count.min()) < SETTINGS['minimum_group_cells']:
        return None
    relation, confidence, accepted, receipt = _transport(source_model, reference_model,
                                                         source_rep, reference_rep)
    source_soft, _ = _soft(source_rows, source_model)
    hard = source_soft.argmax(axis=1)
    baseline_rep = np.array([np.median(baseline_rows[hard == i], axis=0)
                             for i in range(len(source_model['centers']))])
    target_ab = relation@reference_rep[:, 1:]
    nodes = _limited(target_ab-baseline_rep[:, 1:], SETTINGS['maximum_residual_chroma'])
    nodes[~accepted] = 0.
    # Joint source->T->palette colour budget uses the same bound for all groups.
    combined = baseline_rep[:, 1:]+nodes-source_rep[:, 1:]
    allowance = np.maximum(SETTINGS['maximum_total_chroma'],
                           np.linalg.norm(baseline_rep[:, 1:]-source_rep[:, 1:], axis=1))
    budget = np.minimum(1, allowance/np.maximum(np.linalg.norm(combined, axis=1), 1e-30))
    projected = combined*budget[:, None]+source_rep[:, 1:]-baseline_rep[:, 1:]
    nodes = _limited(projected, SETTINGS['maximum_residual_chroma']); nodes[~accepted] = 0.
    source_model.update(nodes=nodes.tolist(), source_group_samples=source_count.tolist(),
        matched_fraction=confidence.tolist(), accepted_groups=accepted.tolist(),
        source_support_low=np.quantile(source_rows, .01, axis=0).tolist(),
        source_support_high=np.quantile(source_rows, .99, axis=0).tolist(),
        reference_model=reference_model, relation=relation.tolist(), transport=receipt)
    return source_model


def _residual(rows, model):
    weights, distance = _soft(rows, model)
    dose = np.clip((SETTINGS['support_radius']-distance)/.75, 0, 1)
    matched_probability = weights@np.asarray(model['accepted_groups'], float)
    matched_dose = np.clip((matched_probability-.25)/.5, 0, 1)
    dose *= matched_dose*matched_dose*(3-2*matched_dose)
    low = np.asarray(model['source_support_low']); high = np.asarray(model['source_support_high'])
    margin = np.asarray(SETTINGS['support_margin_Lab'])
    outside = np.maximum(np.maximum(low-rows, rows-high), 0)/margin
    dose *= np.clip(1-outside.max(axis=1), 0, 1)
    return _limited((weights@np.asarray(model['nodes']))*dose[:, None],
                    SETTINGS['maximum_residual_chroma'])


def _validation(source_rows, baseline_rows, reference_rows, model):
    source_weights, _ = _soft(source_rows, model)
    source_hard = source_weights.argmax(axis=1)
    reference_rep, reference_count = _representatives(reference_rows, model['reference_model'])
    target = np.asarray(model['relation'])@reference_rep[:, 1:]
    residual = _residual(source_rows, model)
    base_errors = []; new_errors = []; mass = []
    for i, allowed in enumerate(model['accepted_groups']):
        selected = source_hard == i
        used_reference = np.asarray(model['relation'])[i] > .05
        if not allowed or selected.sum() < 4 or np.any(reference_count[used_reference] < 4):
            continue
        before = np.median(baseline_rows[selected, 1:], axis=0)
        after = np.median(baseline_rows[selected, 1:]+residual[selected], axis=0)
        base_errors.append(float(np.linalg.norm(before-target[i])))
        new_errors.append(float(np.linalg.norm(after-target[i])))
        mass.append(int(selected.sum()))
    if not mass:
        return {'accepted': False, 'reason': 'no_independently_held_material_group_support'}
    before, after = float(np.mean(base_errors)), float(np.mean(new_errors))
    accepted = (before-after >= SETTINGS['minimum_absolute_improvement']
                and after <= (1-SETTINGS['minimum_relative_improvement'])*before)
    return {'accepted': bool(accepted), 'reason': 'stable_held_distribution_improvement' if accepted else
        'no_stable_held_distribution_improvement', 'before_chroma_error': before,
        'after_chroma_error': after, 'held_source_cells': sum(mass), 'held_groups': len(mass),
        'metric': 'independent_spatial_block_group_representatives_after_actual_soft_mixture; not_paired_pixel_error'}


def fit_palette(source_rgb, reference_rgb, source_labels, reference_labels, source_alpha, parameters,
                *, source_protection=None, reference_protection=None, baseline_rgb=None):
    """Fit one global-config, colour-only, source-driven palette after current T.

    Source and reference may have different sizes or poses. Optional automatic
    face-protection fields exclude fit observations only. An actual native T
    baseline reduced to this source canvas may be supplied by the producer;
    otherwise the receipt explicitly records the proxy T approximation.
    """
    source = _rgb(source_rgb, 'Source'); reference = _rgb(reference_rgb, 'Reference')
    source_labels = _mask(source_labels, source.shape[:2], 'Source labels', labels=True)
    reference_labels = _mask(reference_labels, reference.shape[:2], 'Reference labels', labels=True)
    alpha = _mask(source_alpha, source.shape[:2], 'Source alpha')
    sp = _mask(np.zeros(alpha.shape) if source_protection is None else source_protection,
               alpha.shape, 'Source automatic feature protection')
    rp = _mask(np.zeros(reference.shape[:2]) if reference_protection is None else reference_protection,
               reference.shape[:2], 'Reference automatic feature protection')
    try:
        json.dumps(parameters, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError('T parameters must be finite JSON-safe evidence') from error
    baseline = _baseline(source, source_labels, parameters, sp) if baseline_rgb is None else _rgb(baseline_rgb, 'T baseline')
    if baseline.shape != source.shape:
        raise ValueError('T baseline must use the source canvas')
    sl, rl, bl = to_lab(source), to_lab(reference), to_lab(baseline)
    # Gloves and exposed skin share semantic hand labels. Both images derive
    # their own exposed-skin floor from observed face interiors, never from the
    # closest target colour or a photo-specific rectangle.
    family_floors = []
    for rgb, lab, labels, coverage, protection in ((source, sl, source_labels, alpha, sp),
            (reference, rl, reference_labels, np.ones(reference.shape[:2]), rp)):
        skin, _, _ = _samples(rgb, lab, labels, coverage, protection, REGIONS['face'])
        family_floors.append(max(.15, .65*float(np.quantile(skin[:, 0], .25))) if len(skin) >= SETTINGS['minimum_cells'] else None)
    result = {'schema': SCHEMA, 'working_domain': 'display_linear_sRGB', 'region_order': list(REGIONS),
        'settings': SETTINGS.copy(), 'regions': {}, 'changes_lightness': False,
        'geometry_warp': False, 'reference_pixels_in_output': False,
        'baseline': 'native_T_reduced' if baseline_rgb is not None else 'T_proxy_approximation',
        'binding': {'source_rgb': _digest(source), 'reference_rgb': _digest(reference),
            'source_labels': _digest(source_labels), 'reference_labels': _digest(reference_labels),
            'source_alpha': _digest(alpha), 'T_baseline': _digest(baseline)}}
    for name, ids in REGIONS.items():
        if name.endswith('_hand') and any(floor is None for floor in family_floors):
            result['regions'][name] = {'status': 'identity', 'reason': 'no_auto_skin_evidence_to_separate_glove'}
            continue
        floor_s, floor_r = family_floors if name.endswith('_hand') else (None, None)
        sr, xy, sf = _samples(source, sl, source_labels, alpha, sp, ids, family_floor=floor_s)
        rr, _, rf = _samples(reference, rl, reference_labels, np.ones(reference.shape[:2]), rp, ids, family_floor=floor_r)
        record = {'source_cells': len(sr), 'reference_cells': len(rr), 'paired_anchor_count': 0,
                  'spatial_correspondence_allowed': False, 'nonzero_colour_residual': False}
        if min(len(sr), len(rr)) < SETTINGS['minimum_cells']:
            result['regions'][name] = {**record, 'status': 'identity', 'reason': 'insufficient_independent_safe_material_cells'}
            continue
        half = SETTINGS['cell_size']//2
        baseline_rows = np.asarray([np.median(bl[y-half:y+half, x-half:x+half].reshape(-1, 3), axis=0)
                                    for x, y in xy], float)
        model = _fit_model(sr, baseline_rows, rr)
        if model is None or not any(model['accepted_groups']):
            result['regions'][name] = {**record, 'status': 'identity', 'reason': 'no_supported_transport_groups'}
            continue
        validations = []; predictions = []
        for fold in range(3):
            training = _fit_model(sr[sf != fold], baseline_rows[sf != fold], rr[rf != fold])
            if training is None:
                validations.append({'accepted': False, 'reason': 'insufficient_spatial_split_training_material'})
                continue
            validations.append(_validation(sr[sf == fold], baseline_rows[sf == fold], rr[rf == fold], training))
            predictions.append(_residual(sr, training))
        stable = bool(len(predictions) == 3 and np.quantile(np.linalg.norm(
            np.stack(predictions)-np.median(np.stack(predictions), axis=0), axis=-1), .9)
            <= SETTINGS['maximum_fold_disagreement'])
        if not stable or not all(row['accepted'] for row in validations):
            result['regions'][name] = {**record, 'status': 'identity', 'reason': 'independent_spatial_splits_not_stable_or_not_improved',
                'validation': {'folds': validations, 'stable_fold_predictions': stable}}
            continue
        result['regions'][name] = {**record, 'status': 'supported', 'mode': 'soft_source_colour_groups_with_dustbin_transport',
            'nonzero_colour_residual': bool(np.max(np.linalg.norm(model['nodes'], axis=1)) > 1e-7),
            'model': model, 'validation': {'folds': validations, 'stable_fold_predictions': stable,
                'split_method': 'three_disjoint_source_and_reference_32px_spatial_blocks; no_position_pairing'},
            'source_skin_L_floor': floor_s, 'source_material_guard': 'exposed_skin_only' if floor_s else None}
    # No ndarray/numpy scalar or nonfinite evidence may enter saved projects.
    return json.loads(json.dumps(result, allow_nan=False))


def validate_palette(palette):
    try:
        json.dumps(palette, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError('Palette must be finite JSON-safe evidence') from error
    if (palette.get('schema') != SCHEMA or palette.get('working_domain') != 'display_linear_sRGB'
            or palette.get('region_order') != list(REGIONS) or palette.get('settings') != SETTINGS
            or palette.get('changes_lightness') is not False or palette.get('geometry_warp') is not False
            or palette.get('reference_pixels_in_output') is not False or not isinstance(palette.get('regions'), dict)):
        raise ValueError('Invalid material palette contract')
    for name, row in palette['regions'].items():
        if name not in REGIONS or row.get('status') not in ('supported', 'identity'):
            raise ValueError('Invalid palette material status')
        if row['status'] == 'identity':
            continue
        model = row.get('model', {})
        centers = np.asarray(model.get('centers'), float); nodes = np.asarray(model.get('nodes'), float)
        if (centers.ndim != 2 or centers.shape[1] != 3 or not 1 <= len(centers) <= SETTINGS['maximum_groups']
                or nodes.shape != (len(centers), 2) or np.linalg.norm(nodes, axis=-1).max() > SETTINGS['maximum_residual_chroma']+1e-8
                or np.asarray(model.get('center')).shape != (3,) or np.asarray(model.get('scale')).shape != (3,)
                or np.any(np.asarray(model.get('scale')) < np.asarray(SETTINGS['feature_scale_floor']))
                or np.asarray(model.get('source_support_low')).shape != (3,)
                or np.asarray(model.get('source_support_high')).shape != (3,)
                or np.any(np.asarray(model['source_support_low']) > model['source_support_high'])
                or row.get('paired_anchor_count') != 0 or row.get('spatial_correspondence_allowed') is not False
                or row.get('validation', {}).get('stable_fold_predictions') is not True
                or len(row.get('validation', {}).get('folds', [])) != 3
                or not all(v.get('accepted') is True for v in row['validation']['folds'])):
            raise ValueError('Invalid or unsupported saved soft material group')
    return palette


def apply_palette(values, palette, weights, *, source_values, strength=1.):
    """Add colour-only residual to T; native replay depends only on source RGB.

    Crops have no origin/canvas dependency. Caller crops immutable source RGB
    and frozen source-coordinate region weights with the same pixel box.
    """
    untouched_values = np.asarray(values)
    values = _rgb(values, 'T values')
    source = np.asarray(source_values)
    if (source.ndim != 3 or source.shape[-1] != 3 or not source.size
            or not np.issubdtype(source.dtype, np.number) or not np.isfinite(source).all()):
        raise ValueError('Immutable source colours must be finite RGB')
    source_domain_support = ((source.min(-1) >= 0) & (source.max(-1) <= 1)).reshape(-1)
    validate_palette(palette)
    weights = np.asarray(weights)
    if (source.shape != values.shape or weights.shape != (*values.shape[:2], len(REGIONS))
            or not np.isfinite(weights).all() or weights.min() < 0 or weights.max() > 1+1e-6
            or np.any(weights.sum(-1) > 1+1e-6) or not np.isfinite(strength) or not 0 <= strength <= 1):
        raise ValueError('Invalid palette replay source/weights/strength')
    active = [name for name, row in palette['regions'].items() if row['status'] == 'supported']
    if not active or strength == 0:
        return untouched_values.copy()
    source_lab = np.zeros(values.shape, np.float32)
    source_lab.reshape(-1, 3)[source_domain_support] = to_lab(
        source.reshape(-1, 3)[source_domain_support].astype(np.float32))
    baseline_lab = to_lab(values)
    flat_source = source_lab.reshape(-1, 3).astype(float)
    delta = np.zeros((len(flat_source), 2), float)
    flat_weights = weights.reshape(-1, len(REGIONS))
    for name in active:
        row = palette['regions'][name]; mask = flat_weights[:, list(REGIONS).index(name)]
        selected = (mask > 1e-8) & source_domain_support
        if not np.any(selected):
            continue
        shift = _residual(flat_source[selected], row['model'])
        if row.get('source_skin_L_floor') is not None:
            dose = np.clip((flat_source[selected, 0]-row['source_skin_L_floor'])/.02, 0, 1)
            shift *= dose[:, None]
        delta[selected] += shift*mask[selected, None]
    delta = _limited(delta*strength, SETTINGS['maximum_residual_chroma'])
    base_shift = baseline_lab.reshape(-1, 3)[:, 1:]-flat_source[:, 1:]
    combined = base_shift+delta
    allowance = np.maximum(SETTINGS['maximum_total_chroma'], np.linalg.norm(base_shift, axis=1))
    # Solve the step along this residual, without altering the existing T.
    aa = np.sum(delta*delta, axis=1); bb = 2*np.sum(base_shift*delta, axis=1)
    cc = np.sum(base_shift*base_shift, axis=1)-allowance**2
    root = (-bb+np.sqrt(np.maximum(bb*bb-4*aa*cc, 0)))/np.maximum(2*aa, 1e-30)
    fraction = np.where(np.linalg.norm(combined, axis=1) > allowance, np.clip(root, 0, 1), 1.)
    delta *= fraction[:, None]
    target = baseline_lab.copy(); target[..., 1:] += delta.reshape(*values.shape[:2], 2).astype(np.float32)
    rgb = from_lab(target)
    bad = (rgb.min(-1) < 0) | (rgb.max(-1) > 1)
    # Gamut correction retracts only the added residual, retaining T and L.
    if np.any(bad):
        low = np.zeros(int(bad.sum())); high = np.ones_like(low)
        original = baseline_lab[bad]; changes = target[bad]-original
        for _ in range(14):
            trial = (low+high)/2
            tested = from_lab(original+changes*trial[:, None])
            valid = (tested.min(-1) >= 0) & (tested.max(-1) <= 1)
            low = np.where(valid, trial, low); high = np.where(valid, high, trial)
        rgb[bad] = from_lab(original+changes*low[:, None])
    result = np.clip(rgb, 0, 1).astype(np.float32)
    inactive = np.linalg.norm(delta, axis=1).reshape(values.shape[:2]) <= 1e-10
    result[inactive] = values[inactive]
    return result
