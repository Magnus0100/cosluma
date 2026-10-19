"""Frozen source-owned skin illumination, separate from reference RGB transfer.

The hand operator is a bounded diagonal *linear RGB* illumination change. Arm
chroma uses a low-frequency Lab offset. Their anatomical transition is inside
the proximal hand, never a hard hand/arm application boundary.
"""
from __future__ import annotations

import copy
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

from asset_identity import sha256_file

SCHEMA = 'source_skin_illumination.v1'
HAND_GAIN_FIT_POLICY = 'exposure_separated_bounded_illuminant.v1'
HAND_SPATIAL_POLICY = {'id': 'source_owned_slow_hand_RGB_illumination.v1',
    'sigma_at_2048': 24., 'minimum_component_samples': 128,
    'minimum_observed_cells': 3, 'minimum_axis_span': .3,
    'minimum_local_weight': .025, 'minimum_local_effective_samples': 12.,
    'sampling': 'component_associated_log_RGB_global_pixel_centres.v1',
    'fallback': 'original_constant_hand_gain_no_new_observation'}
PROFILE = {'id': SCHEMA, 'maximum_proxy_edge': 2048, 'field_edge': 512,
    'arm_low_sigma_at_1600': 4.6875, 'arm_chroma_dose': .95,
    'maximum_arm_ab_offset': .065, 'maximum_L_shift': .035,
    'minimum_L_shift': -.015, 'face_L_offset': .04, 'L_coupling': .25,
    'maximum_hand_channel_EV': .5, 'wrist_distance_at_2048': 45,
    'external_fade_at_2048': 3., 'face_inset_at_2048': 8.,
    'sampling': 'global_pixel_centres; fine_semantic_union; native_opaque_packbits',
    'hand_colour_policy': 'one_positive_diagonal_linear_RGB_gain_per_hand',
    'nail_policy': 'shared_illumination; preserve_relative_intrinsic_colour_not_RGB_identity'}
ARM_IDS = (7, 11, 16, 20)
HAND_IDS = (6, 15)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode('utf8')).hexdigest()


def _asset(path, **metadata):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha256_file(path), **metadata}


def _load(asset):
    path = Path(asset['path'])
    if sha256_file(path) != asset.get('sha256'):
        raise ValueError('Skin source asset identity differs')
    with np.load(path, allow_pickle=False) as data:
        for name in ('labels', 'values', 'protection', 'alpha'):
            if name in data:
                return data[name].copy()
        if len(data.files) == 1:
            return data[data.files[0]].copy()
    raise ValueError('Skin source array is ambiguous')


def checked_context(run, native):
    """No retrospective permission: only a frozen checked material contract."""
    from creative_direction import skin_cast_contract_for
    task = run.get('task_plan') or {}
    intent = run.get('task_intent') or task.get('intent') or {}
    goal = (task.get('creative_direction') or {}).get('material_refinement')
    if goal is None:
        return None
    if (task.get('ready') is not True or task.get('intent') != intent
            or goal != skin_cast_contract_for(intent)
            or (intent.get('constraints') or {}).get('reference_role') != 'source_conditioned_design'):
        raise ValueError('Skin illumination requires the exact checked user permission')
    if not set(goal.get('domains', {})) & {'arms', 'hands'}:
        return None
    plan = (((run.get('options') or {}).get('appearance_revision') or {})
            .get('design_reconstruction_plan') or {})
    source = (run.get('source') or {}).get('sha256')
    roles = plan.get('mask_roles') or {}
    baseline = plan.get('source_baseline') or {}
    if (not source or plan.get('source_sha256') != source or baseline.get('source_sha256') != source
            or baseline.get('schema') != 'appearance_source_baseline.v1'
            or plan.get('canvas', {}).get('source_oriented_size') != list(native)):
        raise ValueError('Skin material masks do not belong to this source canvas')
    selected = {name: copy.deepcopy(roles.get(name)) for name in
                ('source_semantics', 'identity_protection', 'foreground_alpha')}
    if any(not isinstance(value, dict) for value in selected.values()):
        raise ValueError('Source-owned semantic, identity and alpha masks are required')
    sem = selected['source_semantics']
    if (sem.get('schema_version') != 'source_semantic_binding.v1'
            or sem.get('source_sha256') != source or sem.get('sampling') != 'nearest'
            or sem.get('coordinate_space') != 'source_proxy' or sem.get('native_size') != list(native)):
        raise ValueError('Skin semantics lack a checked source binding')
    identity = selected['identity_protection']; alpha = selected['foreground_alpha']
    if (identity.get('representation') != 'eyes_lips_protection'
            or identity.get('coordinate_space') != 'source_proxy'
            or identity.get('size') != sem.get('size')
            or alpha.get('coordinate_space') != 'source_canvas'
            or alpha.get('size') != list(native)
            or alpha.get('sha256') != (baseline.get('assets', {}).get('foreground_alpha') or {}).get('sha256')):
        raise ValueError('Skin identity/alpha coverage differs from the source baseline')
    for asset in selected.values():
        if sha256_file(asset['path']) != asset.get('sha256'):
            raise ValueError('Skin material source asset hash differs')
    return {'schema': SCHEMA, 'goal': copy.deepcopy(goal), 'source_sha256': source,
        'native_size': list(native), 'source_assets': selected,
        'binding': {'goal_sha256': _digest(goal), 'source_sha256': source,
            'native_size': list(native), 'source_assets': {key: value['sha256'] for key, value in selected.items()},
            'profile': PROFILE, 'hand_gain_fit_policy': HAND_GAIN_FIT_POLICY,
            'hand_spatial_fit_policy': HAND_SPATIAL_POLICY}}


def _policy(goal, domain):
    operations, colours = set(), set()
    for item in goal.get('domains', {}).get(domain, []):
        operations.update(item.get('operations', []))
        colours.update(item.get('cast_targets', []))
    general = 'harmonize_skin_chroma' in operations or 'user_declared_cast' in colours
    return {'enabled': bool(operations), 'general': general, 'colours': colours}


def _permitted_ab_delta(current, anchor, policy):
    delta = anchor-current
    if policy['general']:
        return delta
    result = np.zeros_like(delta)
    colours = policy['colours']
    if 'red' in colours: result[..., 0] += np.minimum(delta[..., 0], 0)
    if 'green' in colours: result[..., 0] += np.maximum(delta[..., 0], 0)
    if 'yellow' in colours: result[..., 1] += np.minimum(delta[..., 1], 0)
    if 'blue' in colours: result[..., 1] += np.maximum(delta[..., 1], 0)
    return result


def _fit_hand_gain(rgb, desired):
    """Preserve illuminant ratios before spending the finite exposure budget.

    A common log-gain offset changes exposure, not white balance. Independent
    channel clipping confounds the two when multiple channels saturate.
    The colour span is contracted only if no common exposure can fit the
    existing +/-0.5 EV box; the receipt explicitly records that shortfall.
    """
    import registered_reference_style as style
    rgb, desired = np.asarray(rgb, np.float64), np.asarray(desired, np.float64)
    if rgb.shape != (3,) or desired.shape != (3,) or not np.isfinite([rgb, desired]).all() or np.any(rgb < 0):
        raise ValueError('Invalid hand illuminant observations')
    limit = PROFILE['maximum_hand_channel_EV']
    raw = np.log2(np.maximum(desired, 1e-6)/np.maximum(rgb, 1e-6))
    common = float((raw.max()+raw.min())*.5)
    colour = raw-common
    dose = min(1., (2*limit)/max(float(np.ptp(raw)), 1e-12))
    base_L = float(style._to_lab(rgb.astype(np.float32))[0])
    floor_L = max(0., base_L+PROFILE['minimum_L_shift'])
    ceiling_L = base_L+PROFILE['maximum_L_shift']

    def candidate(d):
        chroma = colour*d
        lower, upper = -limit-float(chroma.min()), limit-float(chroma.max())
        # OKLab L is homogeneous of degree 1/3 in linear RGB. Thus a
        # common EV offset has a monotone exact lightness interval, without
        # altering the chromatic ratio or requiring iterative pixel edits.
        zero_L = float(style._to_lab((rgb*np.exp2(chroma)).astype(np.float32))[0])
        if zero_L > 1e-12:
            if floor_L > 0: lower = max(lower, 3*np.log2(floor_L/zero_L))
            upper = min(upper, 3*np.log2(ceiling_L/zero_L))
        return chroma, lower, upper

    chroma, lower, upper = candidate(dose)
    if lower > upper:
        # Preserve both lightness bounds and the channel box. If they cannot
        # coexist with the requested colour, reduce colour rather than
        # sacrificing hand exposure or relaxing any existing constraint.
        low, high = 0., dose
        for _ in range(28):
            trial = (low+high)*.5
            _, lo, hi = candidate(trial)
            if lo <= hi: low = trial
            else: high = trial
        dose = low
        chroma, lower, upper = candidate(dose)
    if lower > upper+1e-10:
        raise ValueError('No feasible bounded hand illuminant')
    exposure = float(np.clip(common, lower, upper))
    ev = np.clip(chroma+exposure, -limit, limit)
    gain = np.exp2(ev).astype(np.float32)
    actual_L = float(style._to_lab((rgb*gain).astype(np.float32))[0])
    return gain, ev, {'policy': HAND_GAIN_FIT_POLICY,
        'unclipped_gain_EV': raw.tolist(), 'requested_common_exposure_EV': common,
        'colour_dose': float(dose), 'colour_target_limited': bool(dose < 1.-1e-7),
        'feasible_common_exposure_EV': [lower, upper], 'common_exposure_EV': exposure,
        'removed_common_exposure_EV': common-exposure,
        'sample_RGB_lightness_before': base_L, 'sample_RGB_lightness_after': actual_L,
        'sample_RGB_lightness_shift': actual_L-base_L,
        'sample_RGB_lightness_floor': floor_L, 'sample_RGB_lightness_ceiling': ceiling_L,
        'exposure_target_limited': bool(abs(exposure-common) > 1e-7),
        'lightness_scope': 'robust median RGB; not a per-pixel skin or specular exposure guarantee'}


def _bounded_hand_log_field(rgb, desired):
    """Vector form of the existing channel/exposure box, for slow field nodes."""
    import registered_reference_style as style
    raw = np.log2(np.maximum(desired, 1e-6)/np.maximum(rgb, 1e-6))
    common = (raw.max(-1)+raw.min(-1))*.5
    colour = raw-common[..., None]
    limit = PROFILE['maximum_hand_channel_EV']
    dose = np.minimum(1., 2*limit/np.maximum(np.ptp(raw, axis=-1), 1e-12))
    L = style._to_lab(rgb)[..., 0]
    floor, ceiling = np.maximum(0., L+PROFILE['minimum_L_shift']), L+PROFILE['maximum_L_shift']
    def bounds(d):
        c = colour*d[..., None]
        lo, hi = -limit-c.min(-1), limit-c.max(-1)
        z = style._to_lab(rgb*np.exp2(c))[..., 0]
        lo = np.maximum(lo, 3*np.log2(np.maximum(floor, 1e-12)/np.maximum(z, 1e-12)))
        hi = np.minimum(hi, 3*np.log2(np.maximum(ceiling, 1e-12)/np.maximum(z, 1e-12)))
        return c, lo, hi
    c, lo, hi = bounds(dose)
    bad = lo > hi
    if bad.any():
        lower, upper = np.zeros_like(dose), dose.copy()
        for _ in range(28):
            trial = (lower+upper)*.5
            _, a, b = bounds(trial)
            ok = a <= b
            lower, upper = np.where(ok, trial, lower), np.where(ok, upper, trial)
        dose = np.where(bad, lower, dose)
        c, lo, hi = bounds(dose)
    if np.any(lo > hi+1e-7): raise ValueError('No bounded hand field exposure interval')
    ev = np.clip(c+np.clip(common, lo, hi)[..., None], -limit, limit).astype(np.float32)
    actual_L = style._to_lab(rgb*np.exp2(ev))[..., 0]
    if (not np.isfinite(ev).all() or np.any(actual_L < floor-2e-6)
            or np.any(actual_L > ceiling+2e-6)):
        raise ValueError('Bounded hand field node verification failed')
    return ev, dose, actual_L-L


def _fit_spatial_hand(source, lab, hm, reliable, anchor_ab, policy, hand_L, grid, next_id):
    """Estimate slow illumination from observed skin, never pigment/texture RGB.

    Fine connected components are established before reduction. Unsupported
    components retain the constant gain; observation support is not a dose hull.
    """
    from scipy.ndimage import gaussian_filter, distance_transform_edt, label
    import registered_reference_style as style
    components, count = label(hm)
    fine_owner = np.zeros(hm.shape, np.int32)
    field_owner = np.zeros(grid[::-1], np.int32)
    field = np.zeros((*grid[::-1], 3), np.float32)
    reports = []
    small_components = _resize(components.astype(np.int32), grid, nearest=True)
    scale = max(hm.shape)/2048
    sigma = max(.5, HAND_SPATIAL_POLICY['sigma_at_2048']*max(grid)/2048)
    minimum = max(32, round(HAND_SPATIAL_POLICY['minimum_component_samples']*scale*scale))
    for component in range(1, count+1):
        domain = components == component
        observed = reliable & domain
        row = {'component': component, 'source_pixels': int(domain.sum()),
               'observed_pixels': int(observed.sum()), 'state': 'constant_gain_fallback'}
        reports.append(row)
        if observed.sum() < minimum:
            row['reason'] = 'insufficient_reliable_skin_samples'; continue
        y, x = np.nonzero(domain); oy, ox = np.nonzero(observed)
        span = np.array([np.ptp(ox)/max(1, np.ptp(x)), np.ptp(oy)/max(1, np.ptp(y))])
        cells = np.unique(np.minimum(1, (ox-x.min())*2//max(1, np.ptp(x)+1))
            +2*np.minimum(1, (oy-y.min())*2//max(1, np.ptp(y)+1)))
        row.update(observed_axis_span=span.tolist(), observed_cells=int(len(cells)))
        if len(cells) < HAND_SPATIAL_POLICY['minimum_observed_cells'] or span.min() < HAND_SPATIAL_POLICY['minimum_axis_span']:
            row['reason'] = 'insufficient_spatial_skin_evidence'; continue
        occupied = small_components == component
        if not occupied.any():
            row['reason'] = 'fine_component_lost_on_field_grid'; continue
        weight = _resize(observed.astype(np.float32), grid)
        mass = gaussian_filter(weight, sigma)
        squared = gaussian_filter(weight*weight, sigma/np.sqrt(2)) / (4*np.pi*sigma*sigma)
        ess = mass*mass/np.maximum(squared, 1e-12)
        good = occupied & (mass >= HAND_SPATIAL_POLICY['minimum_local_weight']) & (ess >= HAND_SPATIAL_POLICY['minimum_local_effective_samples'])
        if good.sum() < 8:
            row['reason'] = 'insufficient_low_frequency_node_support'; continue
        local = np.stack([gaussian_filter(_resize(source[..., i]*observed, grid), sigma)
            / np.maximum(mass, 1e-8) for i in range(3)], -1)
        local = np.clip(local[good], 0, 1)
        target = style._to_lab(local)
        target[..., 1:] += _permitted_ab_delta(target[..., 1:], anchor_ab, policy)
        target[..., 0] = np.clip(target[..., 0]+PROFILE['arm_chroma_dose']*hand_L, .1, .95)
        ev, dose, shift = _bounded_hand_log_field(local, style._from_lab(target))
        component_field = np.zeros_like(field); component_field[good] = ev
        _, nearest = distance_transform_edt(~good, return_indices=True)
        component_field[occupied] = component_field[nearest[0][occupied], nearest[1][occupied]]
        field[occupied] = component_field[occupied]
        field_owner[occupied] = next_id; fine_owner[domain] = next_id
        row.update(state='source_skin_illumination_estimated', owner_id=next_id,
            observed_nodes=int(good.sum()), estimated_nodes=int((occupied & ~good).sum()),
            colour_dose_minimum=float(dose.min()), bounded_sample_L_shift=[float(shift.min()),float(shift.max())],
            source_intrinsic_colour_verified=False)
        next_id += 1
    return field, field_owner, fine_owner, reports, next_id


def _resize(values, size, *, nearest=False):
    from PIL import Image
    method = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    if values.ndim == 2:
        return np.asarray(Image.fromarray(values).resize(size, method))
    return np.stack([np.asarray(Image.fromarray(values[..., i]).resize(size, method))
                     for i in range(values.shape[-1])], -1)


def _proximal_weight(hand, arm, distance):
    """Geodesic inside-hand distance; occluded/separate fingers keep hand WB."""
    from scipy.ndimage import binary_dilation
    if not hand.any(): return np.zeros(hand.shape, np.float32)
    d = np.full(hand.shape, float(distance), np.float32)
    frontier = binary_dilation(arm) & hand
    reached = np.zeros_like(hand)
    for step in range(1, distance+1):
        if not frontier.any(): break
        d[frontier] = step
        reached |= frontier
        frontier = binary_dilation(frontier) & hand & ~reached
    t = np.clip(d/distance, 0, 1)
    return (t*t*(3-2*t))*hand


def fit(styled_proxy, labels, protection, opaque_proxy, goal):
    """Fit from source-carried output only, never reference/human-target RGB."""
    from scipy.ndimage import gaussian_filter, distance_transform_edt
    import registered_reference_style as style
    source = np.asarray(styled_proxy, np.float32)
    if (source.ndim != 3 or source.shape[-1] != 3 or source.shape[:2] != labels.shape
            or labels.shape != protection.shape or labels.shape != opaque_proxy.shape
            or max(labels.shape) > PROFILE['maximum_proxy_edge']
            or not np.isfinite(source).all() or source.min() < 0 or source.max() > 1
            or not np.isfinite(protection).all() or not np.isfinite(opaque_proxy).all()):
        raise ValueError('Invalid source-carried skin observations')
    domains = goal.get('domains', {})
    if (goal.get('schema') != 'skin_cast_contract.v1'
            or goal.get('reference_rgb_sampling_allowed') is not False
            or goal.get('rgb_changes_allowed') is not True
            or set(domains) & set(goal.get('vetoed_domains', []))):
        raise ValueError('Invalid skin material permission contract')
    arm_policy, hand_policy = _policy(goal, 'arms'), _policy(goal, 'hands')
    if not arm_policy['enabled'] and not hand_policy['enabled']:
        raise ValueError('No checked arm or hand illumination permission')
    scale = max(labels.shape)/2048
    lab = style._to_lab(source)
    face = (labels == 3) & (protection < .01) & (opaque_proxy > .999)
    face &= distance_transform_edt(face) > max(1, PROFILE['face_inset_at_2048']*scale)
    face &= (lab[..., 0] > .1) & (lab[..., 0] < .95)
    minimum = max(32, round(128*scale*scale))
    if face.sum() < minimum: raise ValueError('No reliable source face skin anchor')
    lo, hi = np.quantile(lab[face, 1:], [.2, .8], axis=0)
    face &= ((lab[..., 1:] >= lo) & (lab[..., 1:] <= hi)).all(-1)
    if face.sum() < minimum: raise ValueError('Insufficient robust face colour evidence')
    anchor = np.median(lab[face], axis=0)
    # The existing source-carried face under the requested lighting is the
    # chroma reference. No universal "skin swatch" or complexion is imposed.
    anchor_ab = anchor[1:].astype(np.float32)
    arm = np.isin(labels, ARM_IDS) & arm_policy['enabled'] & (opaque_proxy > .999)
    hand = np.isin(labels, HAND_IDS) & hand_policy['enabled'] & (opaque_proxy > .999)
    union = arm | hand
    if union.sum() < minimum: raise ValueError('No reliable requested source skin domain')
    h, w = labels.shape
    factor = min(1., PROFILE['field_edge']/max(h, w))
    grid = (max(2, round(w*factor)), max(2, round(h*factor)))
    small = _resize(lab, grid)
    weight = _resize(arm.astype(np.float32), grid)
    sigma = PROFILE['arm_low_sigma_at_1600']*max(grid)/1600
    denominator = gaussian_filter(weight, sigma)
    low = np.stack([gaussian_filter(small[..., i]*weight, sigma)/np.maximum(denominator, 1e-8)
                    for i in range(3)], -1)
    offset = np.zeros_like(low)
    L_shift = 0.
    if arm.any():
        L_shift = float(np.clip((float(anchor[0])-PROFILE['face_L_offset']-float(np.median(lab[..., 0][arm])))*PROFILE['L_coupling'],
            PROFILE['minimum_L_shift'], PROFILE['maximum_L_shift']))
        offset[..., 0] = L_shift
        offset[..., 1:] = _permitted_ab_delta(low[..., 1:], anchor_ab, arm_policy)
        length = np.linalg.norm(offset[..., 1:], axis=-1)
        offset[..., 1:] *= np.minimum(1., PROFILE['maximum_arm_ab_offset']/np.maximum(length, 1e-8))[..., None]
        offset *= PROFILE['arm_chroma_dose']
    small_labels = _resize(labels.astype(np.uint8), grid, nearest=True)
    hand_weights = np.zeros((*labels.shape, 2), np.float32)
    owner = np.zeros(labels.shape, np.uint8)
    gains = np.ones((2, 3), np.float32)
    hand_log_gain = np.zeros((2, *grid[::-1], 3), np.float32)
    hand_field_owner = np.zeros((2, *grid[::-1]), np.int32)
    hand_component_owner = np.zeros(labels.shape, np.int32)
    next_component_id = 1
    hand_reports = []
    for index, (hid, aids) in enumerate(((6, (7, 11)), (15, (16, 20)))):
        hm = (labels == hid) & hand
        am = np.isin(labels, aids) & arm
        owner[hm | am] = index+1
        if not hm.any(): continue
        inner = hm & (distance_transform_edt(hm) > max(1, 3*scale))
        if inner.sum() < minimum: raise ValueError('Insufficient hand illumination evidence')
        llo, lhi = np.quantile(lab[..., 0][inner], [.2, .8])
        chroma = np.linalg.norm(lab[..., 1:], axis=-1)
        cmax = min(.065, float(np.quantile(chroma[inner], .8)))
        reliable = inner & (lab[..., 0] >= llo) & (lab[..., 0] <= lhi) & (chroma <= cmax)
        if reliable.sum() < minimum: raise ValueError('Hand has no reliable majority-skin WB samples')
        rgb = np.median(source[reliable], axis=0)
        observed = np.median(lab[reliable], axis=0)
        target = observed.copy()
        target[1:] += _permitted_ab_delta(observed[1:], anchor_ab, hand_policy)
        # A shared exposure component retains hand creases and relative colour.
        hand_L = L_shift if arm.any() else float(np.clip((anchor[0]-.04-observed[0])*.25, -.015, .035))
        target[0] = np.clip(target[0]+PROFILE['arm_chroma_dose']*hand_L, .1, .95)
        desired = style._from_lab(target)
        gains[index], log_gain, gain_fit = _fit_hand_gain(rgb, desired)
        spatial, spatial_owner, fine_owner, spatial_reports, next_component_id = _fit_spatial_hand(
            source, lab, hm, reliable, anchor_ab, hand_policy, hand_L, grid, next_component_id)
        hand_log_gain[index], hand_field_owner[index] = spatial, spatial_owner
        hand_component_owner[fine_owner > 0] = fine_owner[fine_owner > 0]
        if am.any():
            hand_weights[..., index] = _proximal_weight(hm, am,
                max(2, round(PROFILE['wrist_distance_at_2048']*scale)))
            small_arm = np.isin(small_labels, aids)
            if small_arm.any():
                _, nearest = distance_transform_edt(~small_arm, return_indices=True)
                missing = small_labels == hid
                offset[missing] = offset[nearest[0][missing], nearest[1][missing]]
            else:
                raise ValueError('Fine wrist evidence vanished on the arm coefficient grid')
        else:
            hand_weights[..., index][hm] = 1.
        hand_reports.append({'semantic_id': hid, 'sample_pixels': int(reliable.sum()),
            'gain_RGB': gains[index].tolist(), 'gain_EV': log_gain.tolist(),
            'gain_fit': gain_fit,
            'spatial_illumination': spatial_reports,
            'samples': 'central 20-80% L; robust low-chroma majority; bright specular/high-chroma pigment excluded',
            'nail_RGB_identity_claimed': False})
    signed = (distance_transform_edt(union)-distance_transform_edt(~union)).astype(np.float32)
    return {'arm_offset': offset.astype(np.float32), 'hand_weights': hand_weights,
        'owner': owner, 'signed_domain': signed, 'gains': gains,
        'hand_log_gain': hand_log_gain, 'hand_field_owner': hand_field_owner,
        'hand_component_owner': hand_component_owner}, {
        'profile': PROFILE, 'goal': copy.deepcopy(goal), 'processed_domains': [d for d in ('arms', 'hands') if d in domains],
        'face_anchor_pixels': int(face.sum()), 'face_anchor_OKLab': anchor.tolist(),
        'bounded_anchor_ab': anchor_ab.tolist(), 'arm_L_shift': L_shift,
        'hands': hand_reports, 'hand_gain_fit_policy': HAND_GAIN_FIT_POLICY,
        'hand_spatial_fit_policy': copy.deepcopy(HAND_SPATIAL_POLICY), 'source_texture_bitexact': False,
        'reference_RGB_sampled': False, 'geometry_changed': False,
        'boundary_uncertainty': 'Fine learned labels and inward fade leave unknown outer skin pixels uncorrected; no permission expansion.'}


def prepare(styled_proxy, context, native, output):
    """Return owned field/domain assets plus immutable binding evidence."""
    from PIL import Image
    expected = {'goal_sha256': _digest(context['goal']), 'source_sha256': context['source_sha256'],
        'native_size': list(native), 'profile': PROFILE,
        'source_assets': {key: value['sha256'] for key, value in context['source_assets'].items()},
        'hand_gain_fit_policy': HAND_GAIN_FIT_POLICY, 'hand_spatial_fit_policy': HAND_SPATIAL_POLICY}
    if any(context.get('binding', {}).get(key) != value for key, value in expected.items()):
        raise ValueError('Skin numerical preparation context binding differs')
    labels = _load(context['source_assets']['source_semantics'])
    protection = _load(context['source_assets']['identity_protection'])
    alpha = _load(context['source_assets']['foreground_alpha'])
    if (list(context['native_size']) != list(native) or alpha.shape != (native[1], native[0])
            or list(labels.shape[::-1]) != context['source_assets']['source_semantics'].get('size')
            or labels.shape != protection.shape or not np.isfinite(alpha).all()
            or alpha.min() < 0 or alpha.max() > 1):
        raise ValueError('Skin source mask canvases or coverage are invalid')
    opaque_proxy = np.asarray(Image.fromarray(alpha).resize(labels.shape[::-1], Image.Resampling.BILINEAR))
    fields, evidence = fit(styled_proxy, labels, protection, opaque_proxy, context['goal'])
    packed = np.packbits((alpha > .999).reshape(-1), bitorder='little')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    field = output/'skin-field.npz'
    np.savez_compressed(field, arm_offset=fields['arm_offset'], gains=fields['gains'],
        hand_log_gain=fields['hand_log_gain'], hand_field_owner=fields['hand_field_owner'])
    domain = output/'skin-domain.npz'
    np.savez_compressed(domain, owner=fields['owner'], signed_domain=fields['signed_domain'],
        hand_weights=fields['hand_weights'], opaque_packed=packed, native_size=np.asarray(native, np.int32),
        hand_component_owner=fields['hand_component_owner'])
    assets = {'skin_field': _asset(field, representation='bounded_skin_Lab_offset_and_linear_RGB_gains'),
        'skin_domain': _asset(domain, size=list(labels.shape[::-1]), native_size=list(native),
            coordinate_space='source_canvas_via_source_proxy', representation='fine_semantic_domain_native_opaque_packbits')}
    evidence.update(schema=SCHEMA, binding=context['binding'],
        assets_sha256={key: value['sha256'] for key, value in assets.items()},
        model_submissions=0, human_target_RGB_sampled=False)
    receipt = output/'skin-receipt.json'
    receipt.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    assets['skin_receipt'] = _asset(receipt, representation='source_skin_illumination_receipt')
    return assets, evidence


@lru_cache(maxsize=4)
def _frozen_arrays(field_path, field_sha, field_mtime, domain_path, domain_sha, domain_mtime):
    if sha256_file(field_path) != field_sha or sha256_file(domain_path) != domain_sha:
        raise ValueError('Frozen skin field/domain hash differs')
    with np.load(field_path, allow_pickle=False) as f, np.load(domain_path, allow_pickle=False) as d:
        result = {key: f[key].copy() for key in ('arm_offset', 'gains')}
        result.update({key: d[key].copy() for key in ('owner', 'signed_domain', 'hand_weights', 'opaque_packed', 'native_size')})
        optional = ('hand_log_gain' in f, 'hand_field_owner' in f, 'hand_component_owner' in d)
        if any(optional) and not all(optional): raise ValueError('Incomplete frozen hand illumination field')
        if all(optional):
            result.update(hand_log_gain=f['hand_log_gain'].copy(), hand_field_owner=f['hand_field_owner'].copy(),
                hand_component_owner=d['hand_component_owner'].copy())
    shape = result['owner'].shape; size = result['native_size']
    if (len(shape) != 2 or max(shape) > 2048 or min(shape) < 2
            or result['owner'].dtype != np.uint8 or result['owner'].max() > 2
            or result['signed_domain'].shape != shape or result['hand_weights'].shape != (*shape, 2)
            or result['arm_offset'].ndim != 3 or result['arm_offset'].shape[-1] != 3
            or max(result['arm_offset'].shape[:2]) > 512 or min(result['arm_offset'].shape[:2]) < 2
            or result['gains'].shape != (2, 3)
            or any(result[key].dtype != np.float32 for key in ('arm_offset','gains','signed_domain','hand_weights'))
            or size.shape != (2,) or not np.issubdtype(size.dtype,np.integer) or np.any(size < 2)
            or result['opaque_packed'].dtype != np.uint8
            or result['opaque_packed'].size != (int(size[0])*int(size[1])+7)//8
            or any(not np.isfinite(result[key]).all() for key in ('arm_offset', 'gains', 'signed_domain', 'hand_weights'))
            or result['gains'].min() < 2**(-.5)-1e-6 or result['gains'].max() > 2**.5+1e-6
            or result['hand_weights'].min() < 0 or result['hand_weights'].max() > 1
            or np.max(abs(result['arm_offset'][..., 0])) > .035+1e-6
            or np.linalg.norm(result['arm_offset'][..., 1:], axis=-1).max() > .065+1e-6):
        raise ValueError('Invalid bounded frozen skin illumination arrays')
    if 'hand_log_gain' in result:
        ev, coarse, fine = (result[k] for k in ('hand_log_gain','hand_field_owner','hand_component_owner'))
        if (ev.shape != (2, *result['arm_offset'].shape[:2], 3) or coarse.shape != ev.shape[:-1]
                or fine.shape != shape or ev.dtype != np.float32 or coarse.dtype != np.int32 or fine.dtype != np.int32
                or not np.isfinite(ev).all() or np.abs(ev).max() > PROFILE['maximum_hand_channel_EV']+1e-6
                or coarse.min() < 0 or fine.min() < 0 or fine.max() > np.prod(shape)
                or not set(np.unique(coarse)).issubset(set(np.unique(fine)))
                or np.any((fine > 0) & (result['owner'] == 0))):
            raise ValueError('Invalid bounded spatial hand illumination arrays')
    for values in result.values(): values.flags.writeable = False
    return result


def validate(assets, evidence, binding):
    if (evidence.get('schema') != SCHEMA or evidence.get('profile') != PROFILE
            or evidence.get('binding') != binding or evidence.get('reference_RGB_sampled') is not False
            or evidence.get('model_submissions') != 0):
        raise ValueError('Frozen skin illumination evidence differs')
    if ('hand_gain_fit_policy' in evidence or 'hand_gain_fit_policy' in binding) and (
            evidence.get('hand_gain_fit_policy') != HAND_GAIN_FIT_POLICY
            or binding.get('hand_gain_fit_policy') != HAND_GAIN_FIT_POLICY):
        raise ValueError('Frozen hand gain fit policy binding differs')
    for key in ('skin_field', 'skin_domain'):
        if evidence.get('assets_sha256', {}).get(key) != assets[key].get('sha256'):
            raise ValueError('Frozen skin illumination receipt/asset binding differs')
    f, d = assets['skin_field'], assets['skin_domain']
    if sha256_file(f['path']) != f['sha256'] or sha256_file(d['path']) != d['sha256']:
        raise ValueError('Frozen skin field/domain identity differs')
    result = _frozen_arrays(f['path'], f['sha256'], Path(f['path']).stat().st_mtime_ns,
        d['path'], d['sha256'], Path(d['path']).stat().st_mtime_ns)
    present = 'hand_log_gain' in result
    declared = 'hand_spatial_fit_policy' in evidence or 'hand_spatial_fit_policy' in binding
    if (present != declared or declared and (evidence.get('hand_spatial_fit_policy') != HAND_SPATIAL_POLICY
            or binding.get('hand_spatial_fit_policy') != HAND_SPATIAL_POLICY)):
        raise ValueError('Frozen spatial hand illumination policy binding differs')
    return result


def _sample_hand_gain(fields, index, box, canvas):
    """Associated bilinear field sampling never borrows another component."""
    from scipy.ndimage import map_coordinates
    gh, gw = fields['hand_log_gain'].shape[1:3]
    x = (np.arange(box[0],box[2],dtype=np.float64)+.5)*gw/canvas[0]-.5
    y = (np.arange(box[1],box[3],dtype=np.float64)+.5)*gh/canvas[1]-.5
    x, y = np.clip(x,0,gw-1), np.clip(y,0,gh-1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0+1,gw-1), np.minimum(y0+1,gh-1)
    h, w = fields['hand_component_owner'].shape
    fx = (np.arange(box[0],box[2],dtype=np.float64)+.5)*w/canvas[0]-.5
    fy = (np.arange(box[1],box[3],dtype=np.float64)+.5)*h/canvas[1]-.5
    xx, yy = np.meshgrid(fx,fy)
    expected = map_coordinates(fields['hand_component_owner'],[yy,xx],order=0,mode='nearest')
    total = np.zeros(expected.shape,np.float64)
    ev = np.zeros((*expected.shape,3),np.float64)
    for ys, xs, weight in ((y0,x0,(1-(y-y0))[:,None]*(1-(x-x0))[None,:]),
                           (y0,x1,(1-(y-y0))[:,None]*(x-x0)[None,:]),
                           (y1,x0,(y-y0)[:,None]*(1-(x-x0))[None,:]),
                           (y1,x1,(y-y0)[:,None]*(x-x0)[None,:])):
        owner = fields['hand_field_owner'][index,ys[:,None],xs[None,:]]
        weight = weight*(expected > 0)*(owner == expected)
        total += weight
        ev += fields['hand_log_gain'][index,ys[:,None],xs[None,:]]*weight[...,None]
    gain = np.broadcast_to(fields['gains'][index],ev.shape).copy()
    active = total > 1e-12
    gain[active] = np.exp2(ev[active]/total[active,None]).astype(np.float32)
    return gain


def apply(values, fields, box, canvas, strength=1.):
    """Pure pointwise native/explicit-preview application, no OpenCV/model."""
    from scipy.ndimage import map_coordinates
    import registered_reference_style as style
    if strength == 0: return values.copy(), {'applied': False}
    if not np.isfinite(strength) or not 0 <= strength <= 1: raise ValueError('Invalid skin dose')
    h, w = fields['owner'].shape
    x=(np.arange(box[0],box[2],dtype=np.float64)+.5)*w/canvas[0]-.5
    y=(np.arange(box[1],box[3],dtype=np.float64)+.5)*h/canvas[1]-.5
    xx, yy = np.meshgrid(x,y)
    owner=map_coordinates(fields['owner'],[yy,xx],order=0,mode='nearest')
    signed=map_coordinates(fields['signed_domain'],[yy,xx],order=1,mode='nearest')
    edge=max(.25,PROFILE['external_fade_at_2048']*max(h,w)/2048)
    gate=np.clip(signed/edge,0,1); gate=gate*gate*(3-2*gate)
    original=fields['native_size']; nw,nh=map(int,original)
    nx=np.minimum(nw-1,np.floor((np.arange(box[0],box[2],dtype=np.float64)+.5)*nw/canvas[0]).astype(np.int64))
    ny=np.minimum(nh-1,np.floor((np.arange(box[1],box[3],dtype=np.float64)+.5)*nh/canvas[1]).astype(np.int64))
    bit=ny[:,None]*nw+nx[None,:]
    opaque=((fields['opaque_packed'][bit//8]>>(bit%8))&1)>0
    gate *= (owner>0)&opaque
    if not np.any(gate > 0):
        return values.copy(), {'applied': True, 'active_pixels': 0,
            'gamut_dose_minimum': 1., 'gamut_attenuated_pixels': 0, 'outside_permission_exact': True,
            'source_geometry_changed': False, 'source_texture_bitexact': False,
            'nail_RGB_bitexact': False, 'hand_policy': HAND_SPATIAL_POLICY['id'] if 'hand_log_gain' in fields else PROFILE['hand_colour_policy'],
            **({'hand_spatial_fit_policy': HAND_SPATIAL_POLICY['id']} if 'hand_log_gain' in fields else {})}
    arm_delta=style._sample(fields['arm_offset'],box,canvas)
    lab=style._to_lab(values)
    mapped=style._from_lab(lab+arm_delta)
    weights=np.stack([map_coordinates(fields['hand_weights'][...,i],[yy,xx],order=1,mode='nearest')
                      for i in range(2)],-1)
    for index in range(2):
        wh=weights[...,index]*(owner==index+1)
        if 'hand_log_gain' in fields and not np.any(wh > 0):
            continue
        gain = _sample_hand_gain(fields,index,box,canvas) if 'hand_log_gain' in fields else fields['gains'][index]
        hand=values*gain
        mapped=mapped+(hand-mapped)*wh[...,None]
    mapped=values+(mapped-values)*gate[...,None]
    output,gamut=style._ray_gamut(values,mapped,strength)
    active=gate>0
    output[~active]=values[~active]
    return output, {'applied': True, 'active_pixels': int(active.sum()),
        'gamut_dose_minimum': float(gamut[active].min()) if active.any() else 1.,
        'gamut_attenuated_pixels': int(np.count_nonzero((gamut<1)&active)),
        'outside_permission_exact': bool(np.array_equal(output[~active],values[~active])),
        'source_geometry_changed': False, 'source_texture_bitexact': False,
        'nail_RGB_bitexact': False, 'hand_policy': HAND_SPATIAL_POLICY['id'] if 'hand_log_gain' in fields else PROFILE['hand_colour_policy'],
        **({'hand_spatial_fit_policy': HAND_SPATIAL_POLICY['id']} if 'hand_log_gain' in fields else {})}
