"""Content-bound evidence for reference effects; product reads small JSON only.

Material overlap is a coarse rejection test, never dense person correspondence.
Preparation, nonzero effect consumption and human visual acceptance are separate.
"""
from __future__ import annotations

import hashlib
import json
import copy
import math
from pathlib import Path

SCHEMA = 'reference_support.v1'
REGION_NAMES = ('face', 'left_arm', 'right_arm', 'left_hand', 'right_hand',
                'legs', 'hair', 'clothing', 'shoes', 'bag', 'lips')
BINDING_KEYS = ('source_sha256', 'reference_sha256', 'alpha_sha256',
                'source_fp_sha256', 'canvas_size', 'automatic_delta_sha256')
MATERIAL_TRANSITION_SCHEMA = 'qualified_material_support_transition.v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf8')).hexdigest()


def content_record(value):
    if isinstance(value, dict):
        return {key: content_record(item) for key, item in value.items()
                if key != 'path' and not key.endswith('_path') and key != 'master_path'}
    if isinstance(value, list):
        return [content_record(item) for item in value]
    return value


def donor_support_report(fields, weights, parameters):
    import numpy as np
    if (fields.ndim != 3 or fields.shape[-1] != 5 or fields.shape[:2] != weights.shape[:2]
            or not np.isfinite(fields).all() or not np.isfinite(weights).all()):
        raise ValueError('Invalid donor support evidence')
    from reference_colour_fit import REGIONS
    owner, ownership = weights.argmax(-1), weights.max(-1)
    return {'regions': {name: {'eligible_pixels': int(np.count_nonzero(
        (fields[..., 4] > 0) & (owner == index) & (ownership >= .25)))}
        for index, name in enumerate(REGIONS)}, 'cross_material_borrowing': False,
        'nonzero_extension_pixels': int(np.count_nonzero((fields[..., 4] > 0) & (
            (fields[..., 0] != 1) | np.any(fields[..., 1:4] != 0, axis=-1)))),
        'policy_executed': True}


def region_correspondence(source_mask, reference_mask, *, anchors=0,
                          face_alignment=None, minimum_pixels=16):
    """Reject spatial transfer lacking independently observed material geometry.

    A held-out face landmark fit can certify a coarse facial colour map. Other
    materials require both semantic shape overlap and accepted patch evidence.
    Neither option certifies dense correspondence or permits output warping.
    """
    import numpy as np
    sm, rm = np.asarray(source_mask), np.asarray(reference_mask)
    if sm.shape != rm.shape or sm.ndim != 2 or sm.dtype != bool or rm.dtype != bool:
        raise ValueError('Correspondence requires equal-canvas boolean material masks')
    sc, rc = int(sm.sum()), int(rm.sum())
    iou = float((sm & rm).sum()/max(int((sm | rm).sum()), 1))
    evidence = {'source_pixels': sc, 'reference_pixels': rc, 'semantic_iou': iou,
                'accepted_material_anchors': int(anchors), 'dense_correspondence': False,
                'background_H_is_person_correspondence': False}
    if min(sc, rc) < minimum_pixels:
        return {**evidence, 'status': 'unsupported', 'allowed': False,
                'reason': 'insufficient_same_material_support'}
    if iou < .55:
        return {**evidence, 'status': 'unsupported', 'allowed': False,
                'reason': 'semantic_geometry_overlap_below_0.55'}
    landmark_ok = bool(face_alignment and face_alignment.get('method') ==
        'reference_only_similarity_from_source_reference_landmarks'
        and face_alignment.get('check_errors_proxy_px')
        and all(np.isfinite(x) and 0 <= x <= 3
                for x in face_alignment['check_errors_proxy_px']))
    if face_alignment is not None and not landmark_ok:
        return {**evidence, 'status': 'unsupported', 'allowed': False,
                'reason': 'unvalidated_person_face_correspondence'}
    if not landmark_ok and anchors < 3:
        return {**evidence, 'status': 'unsupported', 'allowed': False,
                'reason': 'no_accepted_material_patch_correspondence'}
    return {**evidence, 'status': 'partial', 'allowed': True,
            'reason': ('heldout_face_landmarks_and_material_overlap' if landmark_ok else
                       'material_shape_overlap_and_accepted_patches; coarse_only')}


def _support(status, reason, **evidence):
    return {'status': status, 'reason': reason, **evidence}


def _statistical_support(name, operator, observations):
    """Distribution colour evidence never certifies a spatial person mapping."""
    anchors = int(operator.get('anchors', 0))
    evidence = operator.get('evidence') or {}
    rejected = observations.get('rejected_regions', {}).get(name, 'no_accepted_same_material_evidence')
    if not evidence:
        if anchors > 0:
            raise ValueError('Recorded anchors require current explicit material evidence')
        return _support('unsupported', rejected, mode='identity', accepted_anchors=0,
            paired_anchor_count=0, spatial_correspondence_not_implied=True,
            lightness_statistical_tone=False)
    mode = evidence.get('mode')
    if mode not in ('paired_patch', 'material_distribution', 'skin_family_prior'):
        raise ValueError('Unknown material statistical evidence mode')
    if evidence.get('accepted') is not True:
        return _support('unsupported', 'statistical_evidence_not_accepted', mode=mode,
            accepted_anchors=0, paired_anchor_count=0, lightness_statistical_tone=False)
    counts = {key: evidence.get(key) for key in ('source_samples', 'reference_samples',
                                               'source_cells', 'reference_cells', 'paired_anchor_count')}
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts.values()):
        raise ValueError('Invalid material statistical evidence sample counts')
    if evidence.get('source_region') != name or not evidence.get('reference_region'):
        raise ValueError('Material statistical evidence belongs to a different region')
    if mode == 'paired_patch':
        if anchors <= 0 or counts['paired_anchor_count'] != anchors:
            raise ValueError('Paired patch evidence requires recorded paired anchors')
    elif (anchors != 0 or counts['paired_anchor_count'] != 0
          or evidence.get('spatial_correspondence_allowed') is not False):
        raise ValueError('Unpaired distribution/prior evidence cannot authorize spatial correspondence')
    if mode == 'material_distribution' and evidence.get('reference_region') != name:
        raise ValueError('Direct material distribution cannot borrow another material')
    if mode == 'skin_family_prior':
        parents = {'left_arm': ('face',), 'right_arm': ('face',),
                   'left_hand': ('left_arm', 'face'), 'right_hand': ('right_arm', 'face')}
        parent = operator.get('parent_region') or evidence.get('parent_region')
        if (name not in parents or parent not in parents[name]
                or evidence.get('reference_region') != parent
                or any(value is not None and value != parent for value in
                       (operator.get('parent_region'), evidence.get('parent_region')))):
            raise ValueError('Skin family prior requires its own arm or face palette parent')
        if (evidence.get('lightness_statistical_tone') is not False
                or operator.get('exposure_ev') != 0 or operator.get('tone_curve')
                or operator.get('matrix') != [[1., 0.], [0., 1.]]):
            raise ValueError('Skin family prior permits a chroma bias only; no inherited lightness')
    supported = all(counts[key] > 0 for key in ('source_samples', 'reference_samples', 'source_cells', 'reference_cells'))
    if mode == 'paired_patch':
        supported = True
    lightness = evidence.get('lightness_statistical_tone') is True
    return _support(('partial' if mode == 'skin_family_prior' else 'supported') if supported else 'unsupported',
        ('skin_family_prior; direct_region_reference_not_established' if mode == 'skin_family_prior' else
         'independent_same_material_distributions; no_spatial_pairing' if mode == 'material_distribution' else
         'accepted_same_material_paired_patches') if supported else 'empty_statistical_population',
        mode=mode, accepted_anchors=anchors, **counts,
        source_region=evidence['source_region'], reference_region=evidence['reference_region'],
        parent_region=operator.get('parent_region') or evidence.get('parent_region'),
        spatial_correspondence_not_implied=True,
        lightness_statistical_tone=bool(supported and lightness), validation=evidence.get('validation', {}))


def _source_feature_palette_support(parameters, observations):
    """Describe a source lip tint without inventing direct lip observations."""
    policy = parameters.get('feature_palette_policy') or {}
    if not policy:
        return None
    if (policy.get('mode') != 'source_feature_chroma_only_face_distribution_bias'
            or policy.get('source_feature_labels') != [24, 25]
            or policy.get('parent_region') != 'face'
            or policy.get('colour_operator') != 'identity_matrix_plus_supported_face_bias'
            or policy.get('native_L_and_feature_contrast_retained') is not True
            or policy.get('reference_feature_pixels_used') is not False
            or policy.get('spatial_correspondence_claimed') is not False):
        raise ValueError('Invalid source feature palette inheritance policy')
    face = parameters.get('operators', {}).get('face', {})
    parent = _statistical_support('face', face, observations)
    if (parent['status'] != 'supported' or parent.get('mode') != 'material_distribution'
            or face.get('matrix') != [[1., 0.], [0., 1.]]
            or parameters.get('operators', {}).get('lips')
            or 'face' in parameters.get('local_chroma_residual', {})
            or 'face' in parameters.get('conditional_chroma_residual', {})):
        raise ValueError('Source feature palette requires only a supported face distribution bias')
    return _support('partial', 'source_feature_palette_inheritance; no_direct_lip_reference',
        mode='source_feature_palette_inheritance', source_region='lips', reference_region='face',
        parent_region='face', parent_evidence_mode=parent['mode'],
        source_feature_labels=policy['source_feature_labels'], direct_region_reference=False,
        reference_feature_pixels_used=False, accepted_anchors=0, paired_anchor_count=0,
        spatial_correspondence_not_implied=True, lightness_statistical_tone=False,
        native_L_and_feature_contrast_retained=True,
        validation={'method': 'inherit_supported_face_distribution_chroma_bias_only'})


def build_reference_support(parameters, colour_receipt, scene_receipt, *, binding, donor=None):
    """Derive evidence from prepared receipts without pixels, models or filenames."""
    binding = {key: binding.get(key) for key in BINDING_KEYS}
    if any(value is None for value in binding.values()):
        raise ValueError('Support report requires source/reference/alpha/Fp/canvas/delta identity')
    operators = parameters.get('operators', {})
    obs = parameters.get('observations', {})
    blend = (colour_receipt or {}).get('regions', {})
    scene = (scene_receipt or {}).get('regions', {})
    policy_executed = (parameters.get('support_policy') == SCHEMA
        and (colour_receipt or {}).get('support_policy') == SCHEMA
        and (scene_receipt or {}).get('support_policy') == SCHEMA)
    if not policy_executed:
        raise ValueError('Current support preparation requires all current support-policy receipts')
    feature_palette = _source_feature_palette_support(parameters, obs)
    regions = {}
    supported_operators = {}
    for name in REGION_NAMES:
        op = operators.get(name, {})
        statistics = (feature_palette if name == 'lips' and feature_palette is not None else
                      _statistical_support(name, op, obs))
        inherited_palette = statistics.get('mode') == 'source_feature_palette_inheritance'
        op = op if statistics['status'] != 'unsupported' else {}
        if op:
            supported_operators[name] = op
        b = blend.get(name, {})
        s = scene.get('head_hair' if name == 'hair' else name, {})
        d = (donor or {}).get('regions', donor or {}).get(name, {})
        geometry = b.get('correspondence_support') or s.get('correspondence_support')
        if not geometry:
            geometry = _support('unsupported', 'person_correspondence_not_recorded',
                semantic_iou=obs.get('semantic_overlap', {}).get(name),
                background_H_is_person_correspondence=False, dense_correspondence=False)
        present = bool(op or b.get('eligible_pixels') or s.get('eligible_pixels'))
        spatial = _support('partial' if b.get('eligible_pixels', 0) > 0 else 'unsupported',
            b.get('rejected') or ('coarse_material_colour_map' if b.get('eligible_pixels') else
                                 'no_prepared_spatial_colour_support'),
            eligible_pixels=int(b.get('eligible_pixels', 0)),
            maximum_applied_chroma_shift=float(b.get('maximum_applied_chroma_shift', 0)))
        tone_supported = statistics.get('lightness_statistical_tone', False)
        light = _support('partial' if tone_supported or s.get('eligible_pixels', 0) > 0 else 'unsupported',
            'same_material_statistical_tone_or_supported_spatial_balance' if tone_supported or s.get('eligible_pixels')
                else s.get('rejected', 'no_new_lightness_support'),
            statistical_tone=bool(tone_supported), statistical_evidence_mode=statistics.get('mode'),
            spatial_eligible_pixels=int(s.get('eligible_pixels', 0)),
            applied_ev_range=s.get('applied_ev_range', [0., 0.]))
        regions[name] = {'geometry': geometry,
            'material': (_support('partial', 'source_feature_material_with_inherited_face_palette; no_direct_lip_reference')
                if inherited_palette else _support('supported' if present else 'unsupported',
                    'same_material_observations' if present else 'no_accepted_material_evidence')),
            'statistics': statistics, 'spatial_colour': spatial, 'lightness': light,
            'donor': _support('partial' if d.get('eligible_pixels', 0) else 'unsupported',
                'same_material_only' if (donor or {}).get('policy_executed') and d.get('eligible_pixels') else
                'no_same_material_donor; source_preserved' if (donor or {}).get('policy_executed') else
                'not_recorded; saved_target_replayed; source_preservation_not_verified',
                eligible_pixels=int(d.get('eligible_pixels', 0)),
                policy_executed=bool((donor or {}).get('policy_executed'))),
            'degradation': (('supported_face_palette_chroma_only; native_feature_L_retained; no_direct_lip_reference'
                            if inherited_palette else 'identity_for_absent_statistics')
                + '; zero_spatial_dose_when_unsupported; source_edges_preserved_without_donor'
                if policy_executed and (donor or {}).get('policy_executed') else
                'saved_target_replayed; new_degradation_policy_not_verified'),
            'policy_executed': policy_executed}
        palette = parameters.get('automatic_material_palette', {}).get('regions', {}).get(name)
        if palette:
            regions[name]['statistics']['automatic_palette'] = palette
    stages = {}
    for stage, columns in (
            ('reference_colour_match', ('statistics',)),
            ('reference_colour_blend', ('spatial_colour',)),
            ('reference_scene_balance', ('lightness',)),
            ('edge_donor_extension', ('donor',))):
        supported = [name for name, row in regions.items() if any(
            row[column]['status'] != 'unsupported' for column in columns)]
        # S does not consume T's statistical tone. Keep its support separate.
        if stage == 'reference_scene_balance':
            supported = [name for name, row in regions.items()
                         if row['lightness']['spatial_eligible_pixels'] > 0]
        stages[stage] = {'status': 'partial' if supported else 'unsupported',
            'supported_regions': supported,
            'unsupported_regions': [name for name in regions if name not in supported],
            'effect_consumption': 'not_established_by_preparation', 'visual_acceptance': False}
        if stage == 'reference_colour_match':
            colour_nonzero = any(op.get('bias') != [0., 0.] or op.get('matrix') != [[1.,0.],[0.,1.]]
                for op in supported_operators.values())
            colour_nonzero |= bool(parameters.get('local_chroma_residual') or parameters.get('conditional_chroma_residual')
                                   or parameters.get('authored_chroma_adjustments'))
            colour_nonzero |= any(row.get('nonzero_colour_residual', False) for row in
                parameters.get('automatic_material_palette', {}).get('regions', {}).values())
            lightness_nonzero = any((op.get('exposure_ev', 0) != 0 or op.get('tone_curve')) and
                regions[name]['lightness']['statistical_tone'] for name, op in supported_operators.items())
        elif stage == 'reference_colour_blend':
            colour_nonzero = any(row.get('maximum_applied_chroma_shift', 0) > 0 for row in blend.values())
            lightness_nonzero = False
        elif stage == 'reference_scene_balance':
            colour_nonzero = any(row.get('maximum_ab', 0) > 0 for row in scene.values())
            lightness_nonzero = any(any(value != 0 for value in row.get('applied_ev_range', [])) for row in scene.values())
        else:
            colour_nonzero = lightness_nonzero = bool((donor or {}).get('nonzero_extension_pixels', 0))
        stages[stage].update(colour_effect_nonzero=bool(colour_nonzero), lightness_effect_nonzero=bool(lightness_nonzero))
    report = {'schema': SCHEMA, 'binding': binding,
        'evidence_identity': {'parameters': digest(content_record(parameters)),
            'colour_receipt': digest(content_record(colour_receipt)), 'scene_receipt': digest(content_record(scene_receipt)),
            'colour_asset': (colour_receipt or {}).get('asset', {}).get('sha256'),
            'scene_asset': (scene_receipt or {}).get('asset', {}).get('sha256')},
        'regions': regions, 'stages': stages,
        'summary': {'state': 'partial' if any(x['status'] == 'partial' for x in stages.values()) else 'unsupported',
            'editable_degradation': True, 'background_and_editor_can_continue': True},
        'read_only': True, 'visual_acceptance': False,
        'evidence_epoch': 'current_support_policy',
        'policy_executed': policy_executed,
        'person_geometry_warped': False, 'reference_texture_imported': False,
        'cross_material_donor_borrowing': False if (donor or {}).get('policy_executed') else 'not_verified'}
    report['identity_sha256'] = digest(report)
    return report


def report_state(report, expected_binding):
    """Small metadata integrity check, never an image-content verification."""
    if not isinstance(report, dict) or not report:
        return {'valid': False, 'errors': ['reference_support_missing'], 'state': 'unsupported', 'report': None,
                'reason': 'current_preparation_requires_region_support_report'}
    errors = []
    try:
        unsigned = {key: value for key, value in report.items() if key != 'identity_sha256'}
        if report.get('schema') != SCHEMA or report.get('identity_sha256') != digest(unsigned):
            errors.append('reference_support_identity_mismatch')
        if any(report.get('binding', {}).get(key) != expected_binding.get(key) for key in BINDING_KEYS):
            errors.append('reference_support_input_binding_mismatch')
        if report.get('visual_acceptance') is not False or report.get('read_only') is not True:
            errors.append('reference_support_invalid_acceptance_claim')
        if (report.get('summary', {}).get('state') not in ('partial', 'unsupported')
                or report.get('evidence_epoch') != 'current_support_policy'
                or report.get('policy_executed') is not True):
            errors.append('reference_support_invalid_evidence_state')
        regions = report.get('regions', {})
        if set(regions) != set(REGION_NAMES):
            errors.append('reference_support_region_schema_mismatch')
        for row in regions.values():
            for column in ('geometry', 'material', 'statistics', 'spatial_colour', 'lightness', 'donor'):
                entry = row.get(column, {})
                if entry.get('status') not in ('supported', 'partial', 'unsupported'):
                    errors.append('reference_support_invalid_region_state')
        stages = report.get('stages', {})
        if set(stages) != {'reference_colour_match', 'reference_colour_blend',
                           'reference_scene_balance', 'edge_donor_extension'}:
            errors.append('reference_support_stage_schema_mismatch')
        if any(row.get('status') not in ('partial', 'unsupported') or row.get('visual_acceptance') is not False
               for row in stages.values()):
            errors.append('reference_support_invalid_stage_state')
        transition = report.get('material_transition')
        if transition is not None:
            parent = transition.get('parent_binding', {})
            differences = {key for key in BINDING_KEYS
                           if parent.get(key) != report.get('binding', {}).get(key)}
            if (transition.get('schema') != MATERIAL_TRANSITION_SCHEMA
                    or differences != {'automatic_delta_sha256'}
                    or transition.get('child_automatic_delta_sha256') != report.get('binding', {}).get('automatic_delta_sha256')
                    or not transition.get('parent_support_identity_sha256')
                    or not transition.get('material_invariants_sha256')
                    or transition.get('qualified_material_finish_added') is not True
                    or transition.get('original_preparation_support_retained') is not True
                    or transition.get('new_reference_support_inferred') is not False
                    or transition.get('visual_acceptance') is not False):
                errors.append('reference_support_invalid_material_transition')
    except (TypeError, ValueError, AttributeError):
        errors.append('reference_support_invalid_metadata')
    return {'valid': not errors, 'errors': errors,
            'state': report.get('summary', {}).get('state', 'unsupported') if not errors else 'unsupported',
            'report': report if not errors else None, 'reason': 'prepared_evidence_only'}


def _material_projection(receipt):
    """Stable material invariants, independent of rewritten dependency paths."""
    base = receipt['base_recipe']
    if (receipt.get('schema') != 'qualified_material_finish.v1'
            or receipt.get('algorithm_profile') != 'owner_adaptive_multiscale_skin_v2'
            or receipt.get('working_domain') != 'display_linear_sRGB'
            or base.get('schema') != 'qualified_material_finish.v1'
            or base.get('domain') != 'display_linear_sRGB'
            or base.get('basis') != 'complete_frozen_parent_foreground'
            or receipt.get('applied_to') != 'complete_parent_T_B_S_foreground_before_single_composition'
            or receipt.get('original_T_B_S_reapplied') is not False
            or receipt.get('scene_EV_refitted') is not False
            or receipt.get('outside_field_exact') is not True
            or receipt.get('coverage_modified') is not False
            or receipt.get('reference_geometry_or_RGB_pasted') is not False
            or receipt.get('field_after_opaque_coverage_gate') is not True
            or receipt.get('alpha_consumption') != 'associated_delta_once'
            or receipt.get('visual_acceptance') is not False
            or type(receipt.get('native_changed_pixels')) is not int
            or receipt['native_changed_pixels'] < 0
            or not math.isfinite(float(receipt['maximum_OKLab_L_error']))
            or not 0 <= float(receipt['maximum_OKLab_L_error']) <= 4e-6):
        raise ValueError('Material residual does not preserve its declared parent appearance')
    binding = {'source_sha256': base['source']['sha256'],
        'reference_sha256': base['reference']['sha256'],
        'source_fp_sha256': base['foreground']['sha256'], 'alpha_sha256': base['alpha']['sha256'],
        'canvas_size': base['native_size'], 'automatic_delta_sha256': base['delta']['sha256']}
    if (base['foreground'].get('size') != binding['canvas_size']
            or base['alpha'].get('size') != binding['canvas_size']
            or base['delta'].get('size') != binding['canvas_size']):
        raise ValueError('Material residual parent assets do not share its native canvas')
    return {'parent_binding': binding, 'schema': receipt['schema'],
        'algorithm_profile': receipt['algorithm_profile'],
        'native_changed_pixels': receipt['native_changed_pixels'],
        'maximum_OKLab_L_error': receipt['maximum_OKLab_L_error'],
        'field_sha256': receipt['field']['sha256'],
        'parent_geometry_coverage_and_lightness_preserved': True,
        'original_T_B_S_reapplied': False, 'visual_acceptance': False}


def material_child_support(parent_report, current_binding, receipt):
    """Bind support during current material production, never repair saved input.

    The checked parent evidence keeps its region and stage scope. Only the new
    producer's explicit residual may change the delta binding.
    """
    projection = _material_projection(receipt)
    parent_binding = projection['parent_binding']
    if not report_state(parent_report, parent_binding)['valid']:
        raise ValueError('Current material production requires intact bound parent support')
    changed = {key for key in BINDING_KEYS if parent_binding.get(key) != current_binding.get(key)}
    if changed - {'automatic_delta_sha256'}:
        raise ValueError('Material child cannot replace its source/reference/coverage/canvas')
    if not changed:
        return copy.deepcopy(parent_report)
    result = copy.deepcopy(parent_report)
    result.pop('identity_sha256')
    result['binding'] = copy.deepcopy(current_binding)
    result['material_transition'] = {'schema': MATERIAL_TRANSITION_SCHEMA,
        'parent_support_identity_sha256': parent_report['identity_sha256'],
        'parent_binding': copy.deepcopy(parent_binding),
        'child_automatic_delta_sha256': current_binding['automatic_delta_sha256'],
        'material_invariants_sha256': digest(projection),
        'qualified_material_finish_added': True, 'original_preparation_support_retained': True,
        'new_reference_support_inferred': False, 'visual_acceptance': False}
    result['identity_sha256'] = digest(result)
    if not report_state(result, current_binding)['valid']:
        raise ValueError('Current material support publication failed its binding contract')
    return result
