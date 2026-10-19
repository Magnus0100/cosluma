"""CPU regressions for medium cast, semantic wrist seams and owned pigment.

Synthetic array fixtures prove bandwidth/ownership behaviour, not crop QA or
final photographic acceptance. Optional actual evidence smoke makes no API call.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

import reference_material_evidence as evidence
import reference_material_solver as new
from reference_colour_blend import apply_colour_layer


def fixture():
    shape = (384, 576)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    labels = np.zeros(shape, np.uint8)
    labels[38:132, 184:286] = 3
    labels[174:300, 112:182] = 7
    labels[300:366, 112:182] = 6
    labels[174:336, 330:400] = 16
    labels[336:376, 330:400] = 15
    eyes = np.zeros(shape, np.float32)
    eyes[64:72, 200:219] = 1
    eyes[64:72, 250:269] = 1
    labels[99:109, 218:249] = 24
    lips = (labels == 24).astype(np.float32)
    skin = labels > 0
    lab = np.zeros((*shape, 3), np.float32)
    lab[..., 0] = .20
    lab[..., 1:] = [.012, .008]
    lab[skin, 0] = .6+.006*np.cos(xx[skin]*np.pi)
    lab[skin, 1] = .035+.0018*np.cos(xx[skin]*np.pi)
    lab[skin, 2] = .022+.0018*np.cos(yy[skin]*np.pi)
    lab[labels == 3, 1] -= .018
    left = np.isin(labels, [7, 6])
    patch = np.exp(-((xx-147)**2+(yy-259)**2)/(2*8.**2))
    lab[left, 1] -= (.005+.026*patch)[left]
    lab[left, 2] += (.001+.009*patch)[left]
    lab[labels == 16, 1] += .026
    lab[labels == 15, 0] = .105  # Black glove semantically labelled hand.
    lab[labels == 15, 1:] = [.003, .005]
    lab[eyes > .5, 1] = -.021+.008*np.sin(xx[eyes > .5]*2.)
    lab[eyes > .5, 2] = .009+.007*np.cos(yy[eyes > .5]*1.7)
    lab[lips > .5, 1] = .119+.005*np.sin(xx[lips > .5]*2.1)
    lab[lips > .5, 2] = .019+.003*np.cos(yy[lips > .5]*1.8)
    rgb = evidence.from_lab(lab).astype(np.float32)
    assert rgb.min() >= 0 and rgb.max() <= 1
    cells = np.array([[lightness, .035, .022] for lightness in np.linspace(.50, .70, 32)])
    return rgb, labels, {'eyes': eyes, 'lips': lips}, cells, patch


def synthetic():
    rgb, labels, features, cells, patch = fixture()
    regions = ['face', 'left_arm', 'left_hand', 'right_arm', 'right_hand']
    field, rows, feature_receipt = new._fit_material_field(rgb, labels, features, regions, cells,
                                               source_rgb=rgb, native_pixel_scale=1, feature_ambient=False)
    before = evidence.to_lab(rgb)
    result = apply_colour_layer(rgb, field)
    after = evidence.to_lab(result)
    clean = np.array([.035, .022])
    core = (patch > .75) & (labels == 7)
    old_error = float(np.median(np.linalg.norm(before[core, 1:]-clean, axis=-1)))
    new_error = float(np.median(np.linalg.norm(after[core, 1:]-clean, axis=-1)))
    assert new_error < old_error*.45, (old_error, new_error)
    assert np.max(abs(before[..., 0]-after[..., 0])) < 3e-6
    assert np.array_equal(result[labels == 0], rgb[labels == 0])
    assert not field[labels == 15].any(), 'Black semantic-hand glove was recoloured as bare skin'
    assert np.array_equal(result[labels == 15], rgb[labels == 15])
    independent = (features['eyes'] > .5) | (features['lips'] > .5)
    assert not field[independent].any()
    assert np.array_equal(result[independent], rgb[independent])
    assert np.linalg.norm(field[..., :2], axis=-1).max() <= new.SETTINGS['maximum_ab']+1e-6
    cases = 6
    # A contiguous arm and hand have an internal label border, not a skin edge.
    joint_lab = before.copy()
    left = np.isin(labels, [7, 6])
    joint_lab[left, 1:] = [.008, .022]
    joint_rgb = evidence.from_lab(joint_lab).astype(np.float32)
    new_joint, _, _ = new._fit_material_field(joint_rgb, labels, features, ['left_arm', 'left_hand'], cells,
                                            source_rgb=joint_rgb, feature_ambient=False)
    joint = np.zeros(labels.shape, bool)
    joint[298:302, 136:160] = True
    new_joint_result = evidence.to_lab(apply_colour_layer(joint_rgb, new_joint))
    old_joint_error = float(np.median(np.linalg.norm(joint_lab[joint, 1:]-clean, axis=-1)))
    new_joint_error = float(np.median(np.linalg.norm(new_joint_result[joint, 1:]-clean, axis=-1)))
    assert new_joint_error < old_joint_error*.05, (old_joint_error, new_joint_error)
    cases += 1
    # Ambient is common feature material delta, never skin endpoint substitution.
    ambient_field, _, ambient_receipt = new._fit_material_field(rgb, labels, features, regions, cells,
                                                   source_rgb=rgb, feature_ambient=True)
    ambient_result = evidence.to_lab(apply_colour_layer(rgb, ambient_field))
    assert np.max(abs(before[..., 0]-ambient_result[..., 0])) < 3e-6
    assert np.median(ambient_result[features['lips'] > .5, 1]) > .10
    assert np.median(ambient_result[features['eyes'] > .5, 1]) < .01
    for name in ['eyes', 'lips']:
        own = features[name] > .5
        assert np.linalg.norm(ambient_field[own, :2], axis=-1).mean() > .005
        assert abs(float(np.std(ambient_result[own, 1])-np.std(before[own, 1]))) < 1e-5
    assert ambient_receipt['skin_endpoint_applied_to_feature'] is False
    cases += 1
    # Original 1--3 native-pixel chroma texture must not become a cast field.
    own = (labels == 7)
    checker = np.cos(np.mgrid[:labels.shape[0], :labels.shape[1]][1]*np.pi)
    interior = own & (distance_transform_edt(own) > 18) & (patch < .01)
    old_coefficient = float(np.mean((before[..., 1]-.035)[interior]*checker[interior]))
    new_coefficient = float(np.mean((after[..., 1]-.035)[interior]*checker[interior]))
    assert abs(new_coefficient-old_coefficient) < .00005, (old_coefficient, new_coefficient)
    assert min(component['scale']['sigma_native'] for row in rows.values() for component in row.get('components', [])) >= 3
    cases += 1
    missing, missing_rows, _ = new._fit_material_field(rgb, labels, features, ['right_hand'], cells,
                                                       source_rgb=rgb, feature_ambient=False)
    assert not missing.any() and missing_rows['right_hand']['status'] == 'identity'
    empty, _, _ = new._fit_material_field(rgb, labels, features, regions, np.empty((0, 3)))
    assert not empty.any()
    unknown, _, _ = new._fit_material_field(rgb, np.zeros_like(labels), features, ['left_arm'], cells)
    assert not unknown.any()
    cases += 3
    return {'cases_passed': cases, 'adaptive_minimum_sigma_native': 3.,
            'medium_patch_ab_error': {'source_before': old_error, 'v2_owner_adaptive': new_error, 'ratio': new_error/old_error},
            'internal_wrist_ab_error': {'source_before': old_joint_error, 'v2_shared_skin_taper': new_joint_error},
            'native_checker_chroma_projection': {'before': old_coefficient, 'after': new_coefficient},
            'source_L_retained': True, 'outside_source_owner_and_black_glove_exact': True,
            'independent_feature_exact_when_ambient_disabled': True,
            'feature_pigment_contrast_retained_when_ambient_enabled': True}


def actual_evidence(inputs):
    source, reference, qualification, diagnosis=new.load_inputs(inputs)
    result=new.fit_qualified_skin_colour(source,reference,qualification,diagnosis)
    assert np.isfinite(result['source_ab_field']).all()
    changed=copy.deepcopy(qualification);changed['settings']['minimum_sigma_native']=1
    changed['qualification_sha256']=evidence._digest({key:value for key,value in changed.items() if key!='qualification_sha256'})
    try:new.fit_qualified_skin_colour(source,reference,changed,diagnosis)
    except ValueError:pass
    else:raise AssertionError('Changed current solver accepted')
    return {'cases_passed':2,'qualification_sha256':qualification['qualification_sha256']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    summary = {'schema': 'qualified_material_targets_v2_smoke.v1', 'solver_profile': new.PROFILE_ID,
               'algorithm_sha256': new._module_sha(), 'evidence_algorithm_sha256': evidence._module_sha(),
               'network_requests': 0, 'gpu_requests': 0, 'synthetic': synthetic()}
    if args.inputs:
        summary['actual_evidence_validation'] = actual_evidence(args.inputs)
    summary['status'] = 'passed'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        evidence.visual._write(args.output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
