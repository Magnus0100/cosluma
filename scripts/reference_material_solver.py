"""Current owner-adaptive chroma residuals from reviewed same-material crops.

Only source-owned colour differences are filtered. Original L, geometry, native
RGB texture and independently owned feature pigment remain consumer inputs.
Actual qualification pixels and API verdicts are reopened and strictly validated.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import (binary_fill_holes, distance_transform_edt,
                           gaussian_filter, label)

import reference_material_evidence as evidence

SCHEMA = 'qualified_material_targets.v2'
SOLVER_SCHEMA = 'qualified_source_material_residual.v2'
PROFILE_ID = 'owner_adaptive_multiscale_skin_v2'
DOMAIN = evidence.DOMAIN
REGIONS = evidence.REGIONS
SETTINGS = {
    'profile_id': PROFILE_ID,
    'maximum_ab': .04,
    'minimum_pixels': 24,
    'owner_width_quantile': .9,
    'owner_width_divisor': 64.,
    'minimum_sigma_proxy': 1.,
    'minimum_sigma_native': 3.,
    'pyramid_multipliers': [1., 2., 4.],
    'continuity_sigma_fraction': .5,
    'outer_transition_native': 3.,
    'minimum_outer_transition_proxy': 1.,
    'source_hand_L_floor_minimum': .15,
    'source_hand_L_floor_ratio': .65,
    'source_hand_L_transition': .02,
    'feature_ambient': 'constant_source_neighbour_delta_per_feature_component',
}


def _module_sha():
    with Path(__file__).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def qualify_material_targets(source, reference, diagnosis, anchor_review):
    """Produce the current qualification directly from actual reviewed crops."""
    result = evidence.reviewed_material_evidence(source, reference, diagnosis, anchor_review)
    result.pop('qualification_sha256')
    result.update(schema=SCHEMA, solver_profile=PROFILE_ID, algorithm_sha256=_module_sha(),
                  settings=copy.deepcopy(SETTINGS), evidence_settings=copy.deepcopy(evidence.SETTINGS))
    result['qualification_sha256'] = evidence._digest(result)
    return result


def load_inputs(path):
    from material_anchor_review import load_inputs as load_current
    return load_current(path)


def _validate_qualification(qualification):
    body = {key: value for key, value in qualification.items() if key != 'qualification_sha256'}
    if (qualification.get('schema') != SCHEMA or qualification.get('domain') != DOMAIN
            or qualification.get('solver_profile') != PROFILE_ID
            or qualification.get('algorithm_sha256') != _module_sha()
            or qualification.get('evidence_algorithm_sha256') != evidence._module_sha()
            or qualification.get('settings') != SETTINGS
            or qualification.get('evidence_settings') != evidence.SETTINGS
            or 'evidence_v1' in qualification
            or qualification.get('qualification_sha256') != evidence._digest(body)):
        raise ValueError('Only intact current material targets v2 are supported; prepare fresh reviewed inputs')


def _sigma(owner, native_scale):
    geometry = binary_fill_holes(owner)
    yy, xx = np.nonzero(geometry)
    short_bbox = float(min(xx.max()-xx.min()+1, yy.max()-yy.min()+1))
    interior_width = 2*float(np.quantile(distance_transform_edt(geometry)[geometry], SETTINGS['owner_width_quantile']))
    width = min(short_bbox, interior_width)
    value = max(SETTINGS['minimum_sigma_proxy'], SETTINGS['minimum_sigma_native']/native_scale,
                width/SETTINGS['owner_width_divisor'])
    return value, {'short_bbox': short_bbox, 'interior_width': interior_width,
                   'material_width': width, 'sigma_proxy': value, 'sigma_native': value*native_scale,
                   'scale_basis': 'source_connected_material_geometry_and_explicit_native_pixel_scale'}


def _box(owner, margin):
    yy, xx = np.nonzero(owner)
    return (slice(max(0, int(yy.min())-margin), min(owner.shape[0], int(yy.max())+margin+1)),
            slice(max(0, int(xx.min())-margin), min(owner.shape[1], int(xx.max())+margin+1)))


def _normalized(values, support, sigma):
    weights = support.astype(np.float32)
    denominator = gaussian_filter(weights, sigma)
    channels = values.ndim == 3
    blurred = gaussian_filter(values*(weights[..., None] if channels else weights),
                              (sigma, sigma, 0) if channels else sigma)
    return blurred/np.maximum(denominator[..., None] if channels else denominator, 1e-8)


def _feature_ambient(rgb, labels, features, field, skin, native_scale):
    """Features inherit a common ambient delta, never skin ab(L) or RGB.

    A constant per independently connected feature retains its own chroma
    differences exactly before gamut limiting; original L remains unchanged.
    """
    result = field.copy()
    face_skin = skin & (labels == 3) & (field[..., 2] > 0)
    independent = np.isin(labels, [24, 25, 26, 27, 28])
    for values in features.values():
        independent |= np.asarray(values) > 1e-5
    if not face_skin.any() or not independent.any():
        return result, {'status': 'identity', 'reason': 'missing_supported_source_face_or_independent_features'}
    sigma, scale = _sigma(face_skin | (independent & (labels == 3)), native_scale)
    nearby = distance_transform_edt(~face_skin) <= max(2., 4*sigma)
    permitted = independent & nearby
    components, count = label(permitted)
    rows = []
    for index in range(1, count+1):
        own = components == index
        if not own.any():
            continue
        distance = distance_transform_edt(~own)
        ring = face_skin & (distance > 0) & (distance <= max(2., 3*sigma))
        if ring.sum() < 3:
            rows.append({'component': index, 'status': 'identity', 'reason': 'no_supported_ambient_ring'}); continue
        ambient = evidence._limited(np.median(result[ring, :2], axis=0)[None, None], SETTINGS['maximum_ab'])[0, 0]
        result[own, :2] = ambient
        result[own, 2] = 1
        rows.append({'component': index, 'status': 'supported', 'source_feature_pixels': int(own.sum()),
                     'source_skin_ring_pixels': int(ring.sum()), 'ambient_delta_ab': ambient.tolist(),
                     'source_feature_chroma_differences_preserved_before_gamut': True})
    # Only skin in a narrow feature neighbourhood is feathered toward the
    # feature's inherited ambient. Its pigment does not enter this blur.
    face_domain = face_skin | permitted
    if permitted.any():
        smooth = _normalized(result[..., :2], face_domain, sigma)
        distance = distance_transform_edt(~permitted)
        transition = face_skin & (distance <= 3*sigma)
        weight = np.clip(1-distance/np.maximum(3*sigma, 1e-8), 0, 1)
        result[transition, :2] = ((1-weight[..., None])*result[..., :2]+weight[..., None]*smooth)[transition]
    return result, {'status': 'supported', 'mode': SETTINGS['feature_ambient'], 'scale': scale,
                    'components': rows, 'skin_endpoint_applied_to_feature': False,
                    'reference_feature_rgb_used': False, 'source_feature_pigment_is_base': True,
                    'changes_lightness': False, 'native_visual_acceptance': False}


def _fit_material_field(rgb, labels, features, diagnosed_regions, target_cells, *,
                        source_rgb=None, native_pixel_scale=1., feature_ambient=True):
    """Array core; production callers must validate bound qualifications first."""
    rgb = np.asarray(rgb, np.float32)
    lab = evidence.to_lab(rgb)
    original_lab = evidence.to_lab(rgb if source_rgb is None else source_rgb)
    owners = evidence._owners(labels, features)
    field = np.zeros((*labels.shape, 3), np.float32)
    rows = {}
    if len(target_cells) == 0:
        return field, {region: {'status': 'identity', 'reason': 'no_actual_qualified_skin_target'} for region in diagnosed_regions}, {'status': 'identity'}
    if not np.isfinite(native_pixel_scale) or native_pixel_scale <= 0:
        raise ValueError('Invalid native pixel scale')
    # Semantic hands are not proof of exposed skin. The floor comes from known
    # diagnosed non-hand source skin, never reference brightness or a target.
    known = np.zeros(labels.shape, bool)
    for region in diagnosed_regions:
        if not region.endswith('_hand'):
            known |= owners[region]
    floor = (max(SETTINGS['source_hand_L_floor_minimum'], SETTINGS['source_hand_L_floor_ratio']*
                 float(np.quantile(original_lab[known, 0], .25)))
             if int(known.sum()) >= SETTINGS['minimum_pixels'] else None)
    valid_owners, guard_weights = {}, {}
    for region in diagnosed_regions:
        own = owners[region].copy()
        guard = np.ones(labels.shape, np.float32)
        if region.endswith('_hand'):
            if floor is None:
                rows[region] = {'status': 'identity', 'reason': 'no_known_source_skin_to_separate_glove'}; continue
            guard = np.clip((original_lab[..., 0]-floor)/SETTINGS['source_hand_L_transition'], 0, 1)
            own &= guard > 0
        numbered, count = label(own)
        areas = np.bincount(numbered.ravel())
        supported = [index for index in range(1, count+1) if areas[index] >= SETTINGS['minimum_pixels']]
        own = np.isin(numbered, supported)
        if own.sum() < SETTINGS['minimum_pixels']:
            rows[region] = {'status': 'identity', 'reason': 'insufficient_known_exposed_source_owner',
                            'source_skin_L_floor': floor if region.endswith('_hand') else None}; continue
        valid_owners[region], guard_weights[region] = own, guard
    skin = np.zeros(labels.shape, bool)
    for own in valid_owners.values():
        skin |= own
    scales = []
    raw_delta = np.zeros((*labels.shape, 2), np.float32)
    for region, owner in valid_owners.items():
        components, count = label(owner)
        component_rows = []
        for index in range(1, count+1):
            own = components == index
            if own.sum() < SETTINGS['minimum_pixels']:
                continue
            sigma, scale = _sigma(own, native_pixel_scale)
            scales.append(sigma)
            box = _box(own, int(np.ceil(4*sigma*max(SETTINGS['pyramid_multipliers']))))
            local = own[box]
            current = lab[box]
            differences = []
            for multiplier in SETTINGS['pyramid_multipliers']:
                low = _normalized(current, local, sigma*multiplier)
                endpoint = evidence._conditional_endpoint(target_cells, low[..., 0])
                differences.append(endpoint-low[..., 1:])
            # Coarse family shift plus two explicit residual bands. With unit
            # support the telescoping sum equals the finest supported cast;
            # coarse statistics cannot conceal a remaining medium-size patch.
            coarse = differences[2]
            middle = differences[1]-differences[2]
            fine = differences[0]-differences[1]
            correction = coarse+middle+fine
            raw_delta[box][local] = correction[local]
            component_rows.append({'component': index, 'source_pixels': int(own.sum()), 'scale': scale,
                                   'pyramid_sigma_proxy': [sigma*m for m in SETTINGS['pyramid_multipliers']],
                                   'bands': {'coarse_median_ab': np.median(np.linalg.norm(coarse[local], axis=-1)).item(),
                                             'middle_median_ab': np.median(np.linalg.norm(middle[local], axis=-1)).item(),
                                             'fine_median_ab': np.median(np.linalg.norm(fine[local], axis=-1)).item()}})
        rows[region] = {'status': 'supported' if component_rows else 'identity', 'source_pixels': int(owner.sum()),
                        'components': component_rows, 'source_skin_L_floor': floor if region.endswith('_hand') else None,
                        'source_material_guard': 'known_source_exposed_skin' if region.endswith('_hand') else 'diagnosed_source_semantic_skin'}
    # Smooth colour differences across internal arm/hand label interfaces. The
    # true material exterior alone controls taper; semantic joint borders do not.
    shared_components, count = label(skin)
    corrected = raw_delta.copy()
    continuity_rows = []
    for index in range(1, count+1):
        own = shared_components == index
        if not own.any():
            continue
        local_sigmas = [component['scale']['sigma_proxy'] for row in rows.values() for component in row.get('components', [])]
        sigma = max(SETTINGS['minimum_sigma_proxy']*.5, min(local_sigmas, default=1.)*SETTINGS['continuity_sigma_fraction'])
        box = _box(own, int(np.ceil(4*sigma)))
        smoothed = _normalized(raw_delta[box], own[box], sigma)
        corrected[box][own[box]] = smoothed[own[box]]
        continuity_rows.append({'component': index, 'pixels': int(own.sum()), 'sigma_proxy': sigma,
                                'internal_semantic_boundary_taper': False})
    outer = skin.copy()
    if 'face' in valid_owners:
        independent = np.isin(labels, [24, 25, 26, 27, 28])
        for values in features.values():
            independent |= np.asarray(values) > 1e-5
        nearby = distance_transform_edt(~valid_owners['face']) <= max(2., 4*min(scales, default=1.))
        outer |= independent & nearby
    band = max(SETTINGS['minimum_outer_transition_proxy'], SETTINGS['outer_transition_native']/native_pixel_scale)
    taper = np.clip(distance_transform_edt(outer)/band, 0, 1).astype(np.float32)*skin
    for region, owner in valid_owners.items():
        taper[owner] *= guard_weights[region][owner]
    corrected = evidence._limited(corrected, SETTINGS['maximum_ab'])*taper[..., None]
    field[..., :2] = corrected
    field[..., 2] = taper
    if feature_ambient and 'face' in valid_owners:
        field, feature_receipt = _feature_ambient(rgb, labels, features, field, skin, native_pixel_scale)
    else:
        feature_receipt = {'status': 'identity', 'reason': 'disabled_or_no_known_supported_source_face'}
    field[..., :2] = evidence._limited(field[..., :2], SETTINGS['maximum_ab'])
    dose = evidence._gamut_dose(lab, field[..., :2])
    field[..., :2] *= dose[..., None]
    field[..., 2] *= dose
    feature_receipt['gamut_dose_may_attenuate_ambient'] = True
    feature_receipt['shared_skin_continuity'] = continuity_rows
    feature_receipt['outer_transition_proxy'] = band
    return field.astype(np.float32), rows, feature_receipt


def fit_qualified_skin_colour(source, reference, qualification, diagnosis, *, source_base=None,
                              feature_ambient=True):
    _validate_qualification(qualification)
    validation = evidence.validate_material_evidence(source, reference, qualification, diagnosis,
                                                      source_base=source_base)
    original, labels, features, binding = validation['observations']['source']
    rgb = validation['source_rgb']
    target_cells = validation['target_cells']
    native_scale = 1.
    if binding.get('display_parent'):
        path = evidence._file(binding['display_parent'])['path']
        with Image.open(path) as displayed:
            native_scale = min(displayed.width/labels.shape[1], displayed.height/labels.shape[0])
    field, rows, feature_receipt = _fit_material_field(rgb, labels, features, diagnosis['regions'], target_cells,
                        source_rgb=original, native_pixel_scale=native_scale, feature_ambient=feature_ambient)
    parameters = {'schema': SOLVER_SCHEMA, 'domain': DOMAIN, 'solver_profile': PROFILE_ID,
                  'binding': binding, 'algorithm_sha256': _module_sha(), 'settings': copy.deepcopy(SETTINGS),
                  'qualification_sha256': qualification['qualification_sha256'],
                                    'evidence_algorithm_sha256': evidence._module_sha(),
                  'source_colour_base': validation['source_colour_base'],
                  'native_pixel_scale': native_scale, 'regions': rows, 'feature_ambient': feature_receipt,
                  'field_array_sha256': evidence.array_sha(field), 'coordinate_space': 'source_display',
                  'channels': ['preweighted_delta_a', 'preweighted_delta_b', 'coverage_metadata_already_consumed'],
                  'changes_lightness': False, 'geometry_warp': False, 'reference_rgb_paste': False,
                  'feature_target_policy': 'independent_pigment_with_source_neighbour_ambient_only',
                  'source_high_frequency_is_consumer_input': True,
                  'target_quality_proof': 'actual_current_v2_crop_qualification; synthetic_checks_are_not_visual_acceptance'}
    return {'parameters': parameters, 'source_ab_field': field,
            'receipt': {'schema': SOLVER_SCHEMA, 'parameters_sha256': evidence._digest(parameters),
                        'field_array_sha256': evidence.array_sha(field), 'solver_profile': PROFILE_ID,
                        'qualified_target': qualification, 'visual_acceptance': False, 'native_detail_status': 'unknown'}}
