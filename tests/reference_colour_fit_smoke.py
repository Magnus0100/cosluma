"""Check safety failures and identity behaviour of the appearance matcher."""
from __future__ import annotations

import copy
import numpy as np
from reference_colour_fit import collect_anchors, collect_material_distributions, fit_material_distributions, fit, fit_tone_curves, fit_conditional_residual, fit_local_residual, apply, region_weights, REGIONS, tone, _apply_tone, _fit_exposure, _bounded_matrix, SETTINGS, to_lab, from_lab, release_feature_colour


def main():
    rng = np.random.default_rng(8)
    values = rng.uniform(.03, .5, (120, 120, 3)).astype(np.float32)
    labels = np.full((120, 120), 3, np.uint8)
    ones, zeros = np.ones((120, 120), np.float32), np.zeros((120, 120), np.float32)
    anchors, _ = collect_anchors(values, values, labels, labels, ones, zeros, zeros, ones)
    params = fit(anchors)
    weights = region_weights(labels, zeros, params)
    identity = apply(values, weights, params)
    assert np.max(np.abs(identity-values)) < 2e-6, 'Same reference must not shift its source'
    assert np.array_equal(apply(values, weights, params, strength=0), values)
    bad, observation = collect_anchors(values, values, labels, np.full_like(labels, 23), ones, zeros, zeros, ones)
    assert not bad and observation['rejected_regions']['face'], 'Wrong material correspondence must be rejected'
    protected, _ = collect_anchors(values, values, labels, labels, ones, ones, zeros, ones)
    assert not protected, 'Protected face features must not become fit anchors'
    uncertain, _ = collect_anchors(values, values, labels, labels, ones*.9, zeros, zeros, ones)
    assert not uncertain, 'Uncertain coverage must not become a colour observation'
    # Collinear skin colours cannot identify a full colour matrix. A severe
    # requested correction must be bounded rather than accepted as an exact fit.
    rows = [{'region': 'face', 'xy': [i, i], 'source': [.4, .01*i, .02*i],
             'reference': [.8, .2, -.2]} for i in range(6)]
    bounded = fit(rows)
    assert not bounded['matrix_fit']['estimated']
    assert np.array_equal(bounded['operators']['face']['matrix'], np.eye(2))
    op = bounded['operators']['face']
    assert abs(op['exposure_ev']) <= SETTINGS['maximum_exposure_ev'] and op['exposure_limited']
    assert np.linalg.norm(op['bias']) <= .05000001 and op['chroma_limited']
    assert (_bounded_matrix(-np.eye(2), SETTINGS).diagonal() > 0).all(), 'Chroma rotation must remain near identity'
    levels = np.linspace(0, 1, 1001)
    for ev in np.linspace(-SETTINGS['maximum_exposure_ev'], SETTINGS['maximum_exposure_ev'], 41):
        mapped = tone(levels, ev)
        assert mapped[0] == 0 and abs(mapped[-1]-1) < 1e-8
        assert (np.diff(mapped) > 0).all(), 'Highlight matching must preserve ordered levels'
    bright = np.repeat(np.array([.75, .8, .9], np.float32)[:, None, None], 3, axis=-1)
    bright_weights = np.zeros((3, 1, len(REGIONS)), np.float32); bright_weights[..., 0] = 1
    highlight_params = copy.deepcopy(bounded); highlight_params['operators']['face']['bias'] = [0, 0]
    highlight_params['operators']['face']['exposure_ev'] = SETTINGS['maximum_exposure_ev']
    bright_out = apply(bright, bright_weights, highlight_params)
    assert np.diff(bright_out[:, 0, 0]).min() > .005 and bright_out.max() < 1, 'Legal gain must not collapse bright values'
    no_weights = np.zeros((*values.shape[:2], len(REGIONS)), np.float32)
    assert np.array_equal(apply(values, no_weights, bounded), values), 'Unknown regions must retain their source'
    lip_labels = np.full_like(labels, 24)
    lip_weights = region_weights(lip_labels, zeros, bounded)
    assert not lip_weights.any() and REGIONS['bag'] == [200], 'Accessory labels must not collide with lips'
    # Native crop uses the same pointwise operator and source-coordinate masks.
    full = apply(values, weights, bounded)
    crop = apply(values[19:77, 24:91], weights[19:77, 24:91], bounded)
    assert np.array_equal(crop, full[19:77, 24:91]), 'Crop evaluation must not refit or normalize colours'
    spatial_rows = []
    for y in np.linspace(12, 108, 5):
        for x in np.linspace(12, 108, 5):
            spatial_rows.append({'region': 'face', 'xy': [x, y], 'source': [.5, .015, .02],
                'reference': [.5, .015+.016*(x/120-.5), .02+.012*(y/120-.5)]})
    local = fit_local_residual(spatial_rows, fit(spatial_rows), (120, 120),
        source_labels=labels, source_lab=np.full(values.shape, [.5, .015, .02], np.float32),
        source_alpha=np.ones(labels.shape, np.float32))
    assert 'face' in local['local_chroma_residual'], 'Supported spatial colour trend must be recoverable'
    neutral_skin = np.broadcast_to(from_lab(np.array([.5, .015, .02], np.float32)), values.shape).copy()
    local_weights = np.zeros_like(weights); local_weights[..., 0] = 1
    native = apply(neutral_skin, local_weights, local, canvas_size=(120, 120))
    cropped = apply(neutral_skin[19:77, 24:91], local_weights[19:77, 24:91], local, origin=(24, 19), canvas_size=(120, 120))
    assert np.array_equal(cropped, native[19:77, 24:91]), 'Local colour fields must use absolute source coordinates'
    before_l, after_l = to_lab(neutral_skin)[..., 0], to_lab(native)[..., 0]
    assert np.max(np.abs(before_l-after_l)) < 2e-6, 'Local colour residual must not import reference lightness'
    # Warm hair needs observed colour support rather than dark-hair extrapolation.
    # A conditional colour bias must preserve texture lightness and fade
    # completely outside its measured input-colour support.
    hair_rows = [{'region': 'hair', 'xy': [i % 10, i // 10], 'source': [l, .02+.04*(l-.2)/.25, .03],
                  'reference': [l, .02+.025*(l-.2)/.25, .03-.025*(l-.2)/.25]}
                 for i, l in enumerate(np.linspace(.15, .45, 60))]
    hair_a = fit(hair_rows)
    hair_b = fit_conditional_residual(hair_rows, hair_a)
    assert 'hair' in hair_b['conditional_chroma_residual'], 'Observed warm colour trend must be recoverable'
    shade = hair_b['conditional_chroma_residual']['hair']
    assert np.linalg.norm(shade['nodes'], axis=1).max() <= .030001
    source_hair = from_lab(np.stack([np.linspace(.15, .45, 120), .02+.04*(np.linspace(.15,.45,120)-.2)/.25, np.full(120, .03)], -1))[:, None]
    hair_weights = np.zeros((120, 1, len(REGIONS)), np.float32)
    hair_weights[..., list(REGIONS).index('hair')] = 1
    shade_a = apply(source_hair, hair_weights, hair_a)
    shade_b = apply(source_hair, hair_weights, hair_b)
    assert np.max(np.abs(to_lab(shade_a)[..., 0]-to_lab(shade_b)[..., 0])) < 2e-6
    assert np.array_equal(apply(source_hair[21:77], hair_weights[21:77], hair_b), shade_b[21:77])
    unsupported = from_lab(np.array([.7, .08, .03], np.float32))[None, None]
    assert np.array_equal(apply(unsupported, hair_weights[:1], hair_a), apply(unsupported, hair_weights[:1], hair_b))
    tone_rows = [{'region': 'right_arm', 'xy': [i % 10, i // 10], 'source': [l, .02, .03],
                  'reference': [.37+.75*(l-.27), .02, .03]}
                 for i, l in enumerate(np.linspace(.27, .46, 60))]
    curved = fit_tone_curves(tone_rows, fit(tone_rows))
    curved_op = curved['operators']['right_arm']
    assert 'tone_curve' in curved_op, 'Supported compressed skin tones must lift shadows without over-lifting bright fingers'
    levels = np.linspace(0, 1, 10001)
    mapped = _apply_tone(levels, curved_op)
    assert mapped[0] == 0 and mapped[-1] == 1 and np.diff(mapped).min() > 0
    source_arm = from_lab(np.stack([np.linspace(.27, .46, 120), np.full(120, .02), np.full(120, .03)], -1))[:, None]
    arm_weights = np.zeros((120, 1, len(REGIONS)), np.float32)
    arm_weights[..., list(REGIONS).index('right_arm')] = 1
    arm_out = apply(source_arm, arm_weights, curved)
    assert np.array_equal(apply(source_arm[21:77], arm_weights[21:77], curved), arm_out[21:77])
    assert np.array_equal(apply(source_arm, arm_weights, curved, strength=0), source_arm)
    # More observations of the same distribution must not weaken outlier
    # resistance. Eight grossly bright targets cannot move a robust exposure.
    robust_rows = [{'source': [l, .02, .03], 'reference': [float(tone(l, .8)), .02, .03]}
                   for l in np.linspace(.2, .5, 40)]
    robust_rows += [{'source': [.3, .02, .03], 'reference': [.85, .02, .03]} for _ in range(8)]
    ev = _fit_exposure(robust_rows, SETTINGS)['exposure_ev']
    dense_ev = _fit_exposure(robust_rows*5, SETTINGS)['exposure_ev']
    assert .75 < ev < .9 and abs(dense_ev-ev) < 1e-5, 'Sampling density must not change the robust colour-unit threshold'
    # A hand already sharing its arm colour may still need a distinct tone.
    independent_rows = []
    for name in ('right_arm', 'right_hand'):
        for i, l in enumerate(np.linspace(.2, .4, 36)):
            independent_rows.append({'region': name, 'xy': [12*(i % 6), 12*(i // 6)],
                'source': [l, .02, .03], 'reference': [float(tone(l, .3 if name.endswith('_hand') else 0)), .02, .03]})
    independent_hand = fit(independent_rows)['operators']['right_hand']
    assert independent_hand['chroma_inherited_from_arm'] and not independent_hand['exposure_inherited_from_arm']
    assert abs(independent_hand['exposure_ev']-.3) < .02, 'A failed extra colour bias must not skip a valid hand brightness fit'
    # An explicitly authored jaw stroke is separate from automatic fitting.
    # It may change only colour inside its source-coordinate feathered ROI.
    authored = copy.deepcopy(params)
    authored['operators']['face'].update(matrix=np.eye(2).tolist(), bias=[0, 0], exposure_ev=0)
    authored['authored_chroma_adjustments'] = [{'region': 'face', 'chroma_shift': [.007, -.002],
        'source_a_ramp': [-.004, -.020], 'stroke': {'points': [[.5, .5]], 'native_canvas_size': [120, 120],
        'radius_px': 28, 'hardness': .6, 'opacity': 1}}]
    green_skin = np.broadcast_to(from_lab(np.array([.4, -.015, .025], np.float32)), values.shape).copy()
    authored_weights = local_weights.copy(); authored_weights[40:55, 40:55, 0] = .15
    no_edit = copy.deepcopy(authored); no_edit.pop('authored_chroma_adjustments')
    baseline_skin = apply(green_skin, authored_weights, no_edit)
    authored_skin = apply(green_skin, authored_weights, authored, canvas_size=(120, 120))
    assert np.max(np.abs(to_lab(authored_skin)[..., 0]-to_lab(baseline_skin)[..., 0])) < 2e-6
    assert np.array_equal(authored_skin[:20], baseline_skin[:20]), 'Outside the authored stroke must remain unchanged'
    assert np.array_equal(authored_skin[40:55, 40:55], baseline_skin[40:55, 40:55]), 'Protected face features must reject the authored stroke'
    assert to_lab(authored_skin)[60, 60, 1] > to_lab(baseline_skin)[60, 60, 1]+.004, 'The intended green jaw colour must move toward its measured target'
    assert np.array_equal(apply(green_skin[19:77, 24:91], authored_weights[19:77, 24:91], authored,
                               origin=(24, 19), canvas_size=(120, 120)), authored_skin[19:77, 24:91])
    # Feature structure and colour protection are independent application doses.
    split = copy.deepcopy(authored)
    split['settings']['protection_chroma_strength'] = .65
    split['protection_policy'] = {'mode': 'split_lightness_chroma'}
    split.pop('authored_chroma_adjustments')
    split['operators']['face'].update(exposure_ev=.3, bias=[.008, -.018])
    feature_mask = zeros.copy(); feature_mask[35:80, 35:80] = 1
    structure = region_weights(labels, feature_mask, split)
    chroma = region_weights(labels, feature_mask, split, chroma=True)
    missing = copy.deepcopy(split)
    missing['settings'].pop('protection_chroma_strength')
    try:
        region_weights(labels, feature_mask, missing, chroma=True)
    except ValueError:
        pass
    else:
        raise AssertionError('Missing current chroma protection was interpreted as an old recipe')
    assert abs(structure[50,50,0]-.15) < 1e-6 and abs(chroma[50,50,0]-.65) < 1e-6
    split_skin = np.broadcast_to(from_lab(np.array([.45, .015, .025],np.float32)), values.shape).copy()
    structure_colour = apply(split_skin, structure, split, chroma_weights=structure)
    relaxed_colour = apply(split_skin, structure, split, chroma_weights=chroma)
    structure_lab, split_lab = to_lab(structure_colour), to_lab(relaxed_colour)
    assert np.max(abs(structure_lab[...,0]-split_lab[...,0])) < 2e-6, 'Relaxing colour must retain protected source structure L'
    assert np.max(abs(structure_lab[45:70,45:70,1:]-split_lab[45:70,45:70,1:])) > .005, 'Split protection must actually release colour'
    released = release_feature_colour(split)
    assert split['settings']['protection_chroma_strength'] == .65, 'Saved recipes must remain immutable'
    full_colour = region_weights(labels, feature_mask, released, chroma=True)
    assert np.array_equal(full_colour, region_weights(labels, zeros, released, chroma=True)), 'Identity mask must not attenuate new colour application'
    assert np.array_equal(region_weights(labels, feature_mask, released), structure), 'Colour release must retain lightness weights'
    released_out = apply(split_skin, structure, released, chroma_weights=full_colour)
    assert np.max(abs(to_lab(released_out)[..., 0]-split_lab[..., 0])) < 2e-6, 'Colour release must retain native L on this in-gamut fixture'
    assert np.max(abs(released_out-relaxed_colour)) > .001, 'Removing colour attenuation must have an actual effect'
    assert np.array_equal(apply(split_skin, structure, released, chroma_weights=full_colour, strength=0), split_skin)
    assert np.array_equal(apply(split_skin[19:77, 24:91], structure[19:77, 24:91], released, chroma_weights=full_colour[19:77, 24:91]), released_out[19:77, 24:91])
    assert np.array_equal(apply(split_skin,structure,split,chroma_weights=chroma,strength=0),split_skin)
    assert np.array_equal(apply(split_skin,no_weights,split,chroma_weights=no_weights),split_skin)
    assert np.array_equal(apply(split_skin[19:77,24:91],structure[19:77,24:91],split,
                                chroma_weights=chroma[19:77,24:91]),relaxed_colour[19:77,24:91])
    try:
        apply(split_skin,structure,split)
    except ValueError:
        pass
    else:
        raise AssertionError('Split policy cannot silently omit chroma weights')
    for invalid in (np.full_like(chroma,.2), np.full_like(chroma,-.1),
                    np.full_like(chroma,np.nan), chroma[:119]):
        try:
            apply(split_skin,structure,split,chroma_weights=invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid or over-unity chroma weights must fail')
    # Person displacement rejects paired/spatial support, but cannot erase a
    # separately observed same-material palette. Different canvases are valid.
    shape=(96,96)
    source_lab=np.broadcast_to(np.array([.5,.018,.024],np.float32),(*shape,3)).copy()
    reference_lab=source_lab.copy();reference_lab[...,1:]+=[.012,-.019]
    material_source,material_reference=from_lab(source_lab),from_lab(reference_lab)
    source_mask=np.zeros(shape,np.uint8);reference_mask=source_mask.copy()
    source_mask[10:86,5:37]=3;reference_mask[10:86,59:91]=3
    safe=np.ones(shape,np.float32);unprotected=np.zeros(shape,np.float32)
    rows,obs=collect_anchors(material_source,material_reference,source_mask,reference_mask,
                             safe,unprotected,unprotected,safe)
    assert not rows and obs['semantic_overlap']['face']==0
    distributions=collect_material_distributions(material_source,material_reference,source_mask,reference_mask,
                                                  safe,unprotected,unprotected)
    stat=fit_material_distributions(fit(rows),distributions)
    op=stat['operators']['face'];evidence=op['evidence']
    assert op['anchors']==0 and evidence['mode']=='material_distribution'
    assert evidence['paired_anchor_count']==0 and evidence['spatial_correspondence_allowed'] is False
    assert evidence['source_samples']>48 and evidence['reference_cells']>=6
    stat_weights=region_weights(source_mask,unprotected,stat)
    adapted=apply(material_source,stat_weights,stat)
    center=to_lab(adapted)[30:65,12:30].mean((0,1))
    assert np.linalg.norm(center[1:]-reference_lab[0,0,1:]) < .001
    assert np.max(abs(to_lab(adapted)[...,0]-source_lab[...,0])) < 2e-6
    assert np.array_equal(adapted[:,50:],material_source[:,50:]), 'Statistical colour remains source-mask bound'
    # Distribution-only face tint can adapt source lip colour without copying
    # reference makeup, a skin matrix, native L or lip/skin contrast.
    lip_source=source_mask.copy();lip_source[40:55,12:30]=24
    stat['settings']['protection_chroma_strength']=.65
    stat['protection_policy']={'mode':'split_lightness_chroma'}
    lip_protection=unprotected.copy();lip_protection[40:55,12:30]=1
    stat['operators']['face']['exposure_ev']=.4
    lip_L=region_weights(lip_source,lip_protection,stat)
    lip_ab=region_weights(lip_source,lip_protection,stat,chroma=True)
    lip_output=apply(material_source,lip_L,stat,chroma_weights=lip_ab)
    assert not lip_L[40:55,12:30,0].any() and lip_ab[45:50,16:25,0].min()>.6
    assert np.max(abs(to_lab(lip_output)[40:55,12:30,0]-source_lab[40:55,12:30,0]))<2e-6
    assert np.mean(to_lab(lip_output)[45:50,16:25,1]-source_lab[45:50,16:25,1])>.006
    assert stat['feature_palette_policy']['reference_feature_pixels_used'] is False
    # Canonical mouth illumination is independent of its pigment/opening. A
    # strong face exposure cannot stop at a lip label and produce a dark edge.
    lit_labels=np.full(shape,3,np.uint8);lit_labels[40:55,12:30]=24
    lit_labels[46:48,18:25]=26
    lit_lab=np.broadcast_to(np.array([.5,.018,.024],np.float32),(*shape,3)).copy()
    lit_lab[40:55,12:30]=[.35,.09,.025]
    lit_lab[46:48,18:25]=[.12,0,0]
    lit_rgb=from_lab(lit_lab)
    lit_protection=np.zeros(shape,np.float32);lit_protection[40:55,12:30]=1
    illumination=copy.deepcopy(stat);illumination['operators']['face']['exposure_ev']=1
    original_ab=region_weights(lit_labels,lit_protection,illumination,chroma=True)
    illumination['feature_material_domains']={'source_regions':[
        {'region':'lips','polygon':[[11,39],[30,39],[30,55],[11,55]],
         'inner_polygon':[[17,45],[25,45],[25,48],[17,48]]}]}
    lit_L=region_weights(lit_labels,lit_protection,illumination)
    lit_ab=region_weights(lit_labels,lit_protection,illumination,chroma=True)
    assert np.array_equal(lit_ab,original_ab), 'Mouth illumination cannot expand or change chroma weights'
    assert np.array_equal(lit_L[...,0],np.ones(shape,np.float32)), 'Skin, lip and opening share the face L dose'
    lit_output=apply(lit_rgb,lit_L,illumination,chroma_weights=lit_ab)
    assert np.max(abs(to_lab(lit_output)[...,0]-tone(to_lab(lit_rgb)[...,0],1)))<2e-6
    assert REGIONS['face']==[3] and lit_labels[42,20]==24 and lit_labels[46,20]==26
    assert lit_protection[42,20]==lit_protection[46,20]==1, 'Input feature masks must remain immutable'
    # Reference geometry and dimensions do not certify or obstruct statistics.
    small=material_reference[::2,::2];small_labels=reference_mask[::2,::2]
    independent=collect_material_distributions(material_source,small,source_mask,small_labels,
                                                safe,unprotected,unprotected[::2,::2])
    assert 'face' in fit_material_distributions(fit([]),independent)['operators']
    # Excluded makeup/feature, matte edge and wrong-material pixels never
    # influence the distribution. Missing or tiny samples remain identity.
    changed=material_source.copy();feature=unprotected.copy()
    changed[20:35,10:30]=from_lab(np.array([.5,-.08,.08],np.float32));feature[20:35,10:30]=1
    excluded=collect_material_distributions(changed,material_reference,source_mask,reference_mask,
                                             safe,feature,unprotected)
    assert np.max(abs(np.asarray(excluded['face']['source']['center'])-source_lab[0,0]))<2e-6
    no_material=collect_material_distributions(material_source,material_reference,source_mask,
                                               np.zeros_like(reference_mask),safe,unprotected,unprotected)
    assert not fit_material_distributions(fit([]),no_material)['operators']
    uncertain=collect_material_distributions(material_source,material_reference,source_mask,reference_mask,
                                              safe*.9,unprotected,unprotected)
    assert not fit_material_distributions(fit([]),uncertain)['operators']
    # A hand may have independent material evidence without any arm. When
    # absent in the reference, its explicit family prior imports only palette.
    hands=source_mask.copy();hands[30:65,50:73]=6
    direct_hand_labels=reference_mask.copy();direct_hand_labels[30:65,10:33]=6
    hand_distributions=collect_material_distributions(material_source,material_reference,hands,direct_hand_labels,
                                                       safe,unprotected,unprotected)
    direct_hand=fit_material_distributions(fit([]),hand_distributions)['operators']['left_hand']
    assert direct_hand['evidence']['mode']=='material_distribution' and direct_hand['anchors']==0
    family_distributions=collect_material_distributions(material_source,material_reference,hands,reference_mask,
                                                        safe,unprotected,unprotected)
    prior=fit_material_distributions(fit([]),family_distributions)['operators']['left_hand']
    assert prior['evidence']['mode']=='skin_family_prior' and prior['evidence']['reference_region']=='face'
    assert prior['exposure_ev']==0 and prior['evidence']['lightness_statistical_tone'] is False
    # Sapiens hand labels include cosplay gloves. Independent exposed-skin
    # observations and application must exclude that distinct dark material.
    glove_source=material_source.copy();glove_reference=material_reference.copy()
    glove_source[30:48,50:73]=from_lab(np.array([.22,.001,.001],np.float32))
    glove_reference[30:48,10:33]=from_lab(np.array([.22,.001,.001],np.float32))
    glove_obs=collect_material_distributions(glove_source,glove_reference,hands,direct_hand_labels,
                                              safe,unprotected,unprotected)
    glove_fit=fit_material_distributions(fit([]),glove_obs)
    assert glove_fit['operators']['left_hand']['source_material_guard']['minimum_L']>.3
    glove_w=region_weights(hands,unprotected,glove_fit,source_values=glove_source)
    assert not glove_w[30:48,50:73,list(REGIONS).index('left_hand')].any()
    glove_out=apply(glove_source,glove_w,glove_fit)
    assert np.array_equal(glove_out[30:48,50:73],glove_source[30:48,50:73])
    try:
        region_weights(hands,unprotected,glove_fit)
    except ValueError:
        pass
    else:
        raise AssertionError('Glove-aware material weights cannot omit their source colour support')
    paired_before=copy.deepcopy(params)
    preserved=fit_material_distributions(params,distributions)
    assert preserved['operators']['face']==paired_before['operators']['face'] and params==paired_before
    # A stronger colour dose must not expand the authored jaw stroke into the
    # structurally protected features or outside the existing source ROI.
    split_authored = copy.deepcopy(authored)
    split_authored['settings']['protection_chroma_strength'] = .65
    split_authored['protection_policy'] = {'mode': 'split_lightness_chroma'}
    split_no_edit = copy.deepcopy(split_authored); split_no_edit.pop('authored_chroma_adjustments')
    stronger_chroma = local_weights.copy(); stronger_chroma[40:55,40:55,0] = .65
    split_baseline = apply(green_skin,authored_weights,split_no_edit,chroma_weights=stronger_chroma)
    split_jaw = apply(green_skin,authored_weights,split_authored,chroma_weights=stronger_chroma,canvas_size=(120,120))
    assert np.array_equal(split_jaw[40:55,40:55],split_baseline[40:55,40:55]), 'ab relaxation cannot expand jaw into protected features'
    assert np.array_equal(split_jaw[:20],split_baseline[:20]), 'ab relaxation cannot expand jaw outside its stroke'
    assert np.array_equal(apply(green_skin[19:77,24:91],authored_weights[19:77,24:91],split_authored,
                               chroma_weights=stronger_chroma[19:77,24:91],origin=(24,19),canvas_size=(120,120)),split_jaw[19:77,24:91])
    wrong_domain = copy.deepcopy(params); wrong_domain['working_domain'] = 'scene_linear'
    try:
        apply(values, weights, wrong_domain)
    except ValueError:
        pass
    else:
        raise AssertionError('Scene/appearance ambiguity must fail explicitly')
    print('PASS: identity, paired anchor rejection, independent displaced/different-canvas material statistics, excluded feature/matte samples, unsupported identity, arm-independent hand statistics, explicit chroma-only skin prior, unchanged paired operators, validated exposure/highlights, monotone tone curve, native crops, split L/ab protection and domain rejection')


if __name__ == '__main__':
    main()
