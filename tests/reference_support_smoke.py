"""Real preparation degradation and evidence checks, including edge SDR dose."""
from __future__ import annotations
import copy
import json
import numpy as np

from reference_colour_fit import (fit, REGIONS, region_weights, apply,
    collect_anchors, collect_material_distributions, fit_material_distributions)
from reference_colour_blend import apply_colour_layer
from render_reference_colour_composite import donor_fields, match_foreground, bounded_edge_extension
from reference_support import build_reference_support, donor_support_report, report_state


def rejected(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('Corrupt input accepted by degraded path')


def main():
    source=np.full((20,24,3),[.2,.3,.4],np.float32)
    alpha=np.full(source.shape[:2],.5,np.float32)
    parameters=fit([])
    weights=region_weights(np.ones(alpha.shape,np.uint8),np.zeros_like(alpha),parameters)
    assert not weights.any()
    assert np.array_equal(apply(source,weights,parameters),source)
    donors=donor_fields(source,alpha,weights,parameters)
    assert np.all(donors[...,0]==1) and not donors[...,1:].any()
    fp=source*alpha[...,None]
    matched=match_foreground(fp,alpha,weights,donors,np.ones_like(alpha),parameters,
        origin=(0,0),canvas_size=(24,20))
    assert np.array_equal(matched,fp)
    for bad in (float('nan'),-.01,1.01):
        damaged=alpha.copy();damaged[0,0]=bad
        rejected(lambda:donor_fields(source,damaged,weights,parameters))
    damaged=source.copy();damaged[0,0,0]=float('nan')
    rejected(lambda:donor_fields(damaged,alpha,weights,parameters))
    corrupt=np.zeros_like(source);corrupt[0,0,0]=float('nan')
    rejected(lambda:donor_fields(source,alpha,weights,parameters,reference_blend=corrupt))
    rejected(lambda:apply_colour_layer(source,corrupt,strength=0))
    rejected(lambda:fit([{'region':'face','source':[.2,float('nan'),0.], 'reference':[.2,0.,0.]}]))
    # Known material edges only; an untyped nearby pixel receives zero dose.
    supported=fit([]);supported['operators']={'clothing':{'matrix':np.eye(2).tolist(),
        'bias':[0.,0.],'exposure_ev':0.,'anchors':3}}
    known=np.zeros_like(weights);known[:,:12,list(REGIONS).index('clothing')]=1
    with_donors=donor_fields(source,np.ones_like(alpha),known,supported)
    assert with_donors[:,:12,4].any() and not with_donors[:,12:,4].any()
    # Reproduce the real DSC02631 sleeve excursion before source-alpha blend.
    sleeve_fp=np.array([[[.81590480,.82778138,.74266887]]],np.float32)
    sleeve_alpha=np.array([[.96051037]],np.float32)
    request=np.array([[[1.12234282,1.15340948,1.03819132]]],np.float32)
    evidence={}
    bounded=bounded_edge_extension(sleeve_fp,sleeve_alpha,request,evidence)
    assert bounded.max()<=sleeve_alpha.max()+1e-7
    assert evidence['requested_outside_sdr_pixels']==1 and evidence['maximum_requested_excursion']>1.
    dose=(bounded-sleeve_fp)/request
    assert np.ptp(dose)<1e-6 and evidence['minimum_applied_dose']<.2
    old=np.array([[[.99232632,.79893917,.77568501]]],np.float32)
    a=np.array([[.98524451]],np.float32)
    evidence={};bounded=bounded_edge_extension(old,a,request,evidence)
    assert np.array_equal(old,bounded) and evidence['maximum_source_excursion']>.007
    binding={'source_sha256':'source','reference_sha256':'reference','alpha_sha256':'alpha',
        'source_fp_sha256':'fp','canvas_size':[24,20],'automatic_delta_sha256':'delta'}
    parameters['support_policy']='reference_support.v1'
    receipts={'support_policy':'reference_support.v1','regions':{}}
    report=build_reference_support(parameters,receipts,receipts,binding=binding,
        donor=donor_support_report(donors,weights,parameters))
    assert report['summary']['state']=='unsupported' and report['policy_executed']
    assert all(x['status']=='unsupported' for x in report['stages'].values())
    assert report_state(report,binding)['valid']
    damaged=copy.deepcopy(report);damaged['regions']['face']['statistics']['status']='supported'
    assert not report_state(damaged,binding)['valid']
    assert not report_state(report,{**binding,'source_sha256':'different-source'})['valid']
    rejected(lambda:build_reference_support({}, {}, {}, binding=binding))
    relocated=copy.deepcopy(receipts);relocated['asset']={'path':'A:/one.npz','sha256':'owned'}
    first=build_reference_support(parameters,relocated,receipts,binding=binding)
    relocated['asset']['path']='B:/two.npz'
    second=build_reference_support(parameters,relocated,receipts,binding=binding)
    assert first==second
    # Statistical colour/tone populations need no positional correspondence.
    # Their evidence cannot open B/S or masquerade as paired observations.
    distributions=copy.deepcopy(parameters)
    clothing={'matrix':[[1.,0.],[0.,1.]],'bias':[.02,-.015], 'exposure_ev':.2,'anchors':0,
        'evidence':{'mode':'material_distribution','accepted':True,
            'source_samples':120,'reference_samples':90,'source_cells':12,'reference_cells':9,
            'paired_anchor_count':0,'spatial_correspondence_allowed':False,
            'lightness_statistical_tone':True,'source_region':'clothing','reference_region':'clothing',
            'validation':{'accepted_distribution_population':True}}}
    distributions['operators']={'clothing':clothing}
    report=build_reference_support(distributions,receipts,receipts,binding=binding)
    row=report['regions']['clothing']
    assert row['statistics']['status']=='supported' and row['statistics']['mode']=='material_distribution'
    assert row['statistics']['accepted_anchors']==0 and row['statistics']['paired_anchor_count']==0
    assert row['geometry']['status']=='unsupported' and row['spatial_colour']['status']=='unsupported'
    assert row['lightness']['statistical_tone'] and row['lightness']['spatial_eligible_pixels']==0
    assert report['stages']['reference_colour_match']['colour_effect_nonzero']
    assert report['stages']['reference_colour_match']['lightness_effect_nonzero']
    assert report['stages']['reference_colour_blend']['status']=='unsupported'
    assert report['stages']['reference_scene_balance']['status']=='unsupported'
    assert report_state(report,binding)['valid']
    different=copy.deepcopy(distributions)
    different['operators']['clothing']['evidence']['reference_samples']=91
    changed=build_reference_support(different,receipts,receipts,binding=binding)
    assert changed['identity_sha256']!=report['identity_sha256']
    prior=copy.deepcopy(clothing)
    prior['parent_region']='face'
    prior['exposure_ev']=0.
    prior['evidence'].update(mode='skin_family_prior',source_region='left_hand',reference_region='face',
                             parent_region='face',lightness_statistical_tone=False)
    distributions['operators']['left_hand']=prior
    report=build_reference_support(distributions,receipts,receipts,binding=binding)
    hand=report['regions']['left_hand']
    assert hand['statistics']['status']=='partial' and hand['statistics']['reference_region']=='face'
    assert hand['statistics']['parent_region']=='face' and hand['geometry']['status']=='unsupported'
    assert not hand['lightness']['statistical_tone']
    for group,key,value in (
            ('evidence','lightness_statistical_tone',True), ('operator','exposure_ev',.2),
            ('operator','tone_curve',{'identity':False}), ('operator','matrix',[[1.1,0.],[0.,1.]]),
            ('evidence','reference_region','clothing'), ('operator','parent_region','right_arm'),
            ('evidence','parent_region','right_arm')):
        corrupt=copy.deepcopy(distributions)
        item=corrupt['operators']['left_hand']
        (item['evidence'] if group=='evidence' else item)[key]=value
        rejected(lambda:build_reference_support(corrupt,receipts,receipts,binding=binding))
    corrupt=copy.deepcopy(distributions)
    item=corrupt['operators'].pop('left_hand')
    item['evidence']['source_region']='clothing'
    corrupt['operators']['clothing']=item
    rejected(lambda:build_reference_support(corrupt,receipts,receipts,binding=binding))
    for key, value in (('spatial_correspondence_allowed',True),('paired_anchor_count',3),
                       ('reference_region','hair'),('source_samples',float('nan'))):
        corrupt=copy.deepcopy(distributions)
        corrupt['operators']['clothing']['evidence'][key]=value
        rejected(lambda:build_reference_support(corrupt,receipts,receipts,binding=binding))
    empty=copy.deepcopy(distributions)
    empty['operators']={'clothing':{'matrix':[[1.,0.],[0.,1.]],'bias':[0.,0.],'exposure_ev':0.,'anchors':0}}
    identity=build_reference_support(empty,receipts,receipts,binding=binding)
    assert identity['stages']['reference_colour_match']['status']=='unsupported'
    assert not identity['stages']['reference_colour_match']['colour_effect_nonzero']
    # Actual unpaired fitter on source/reference materials at different places.
    # The colours transfer while background H supplies zero valid patch pairs.
    canvas=(128,96)
    src=np.full((*canvas,3),[.22,.27,.32],np.float32)
    ref=np.full((*canvas,3),[.30,.24,.20],np.float32)
    source_labels=np.zeros(canvas,np.uint8);source_labels[:,:48]=1
    reference_labels=np.zeros(canvas,np.uint8);reference_labels[:,48:]=1
    opaque=np.ones(canvas,np.float32);protect=np.zeros(canvas,np.float32)
    anchors,_=collect_anchors(src,ref,source_labels,reference_labels,opaque,protect,protect,opaque)
    assert not anchors
    observations=collect_material_distributions(src,ref,source_labels,reference_labels,opaque,protect,protect)
    fitted=fit_material_distributions(fit(anchors),observations)
    fitted['support_policy']='reference_support.v1'
    fitted['observations']={'material_distributions':observations}
    actual_op=fitted['operators']['clothing']
    assert actual_op['anchors']==0 and actual_op['evidence']['mode']=='material_distribution'
    w=region_weights(source_labels,protect,fitted)
    actual=apply(src,w,fitted)
    assert np.max(abs(actual-src))>.01
    real_report=build_reference_support(fitted,receipts,receipts,binding=binding)
    assert real_report['regions']['clothing']['statistics']['status']=='supported'
    assert real_report['regions']['clothing']['geometry']['status']=='unsupported'
    assert real_report['stages']['reference_colour_match']['colour_effect_nonzero']
    assert real_report['stages']['reference_scene_balance']['status']=='unsupported'
    # A truly absent direct hand target remains an explicit skin family prior,
    # with partial evidence and no claimed matched hand brightness.
    source_labels[:]=0;reference_labels[:]=0
    source_labels[4:44,4:44]=3;reference_labels[4:44,4:44]=3
    source_labels[75:115,50:90]=6
    observations=collect_material_distributions(src,ref,source_labels,reference_labels,opaque,protect,protect)
    fitted=fit_material_distributions(fit([]),observations)
    hand=fitted['operators']['left_hand']
    assert hand['evidence']['mode']=='skin_family_prior' and hand['anchors']==0
    fitted['support_policy']='reference_support.v1'
    real_report=build_reference_support(fitted,receipts,receipts,binding=binding,
                                       donor={'policy_executed':True,'regions':{}})
    assert real_report['regions']['left_hand']['statistics']['status']=='partial'
    assert real_report['regions']['left_hand']['statistics']['reference_region']=='face'
    assert not real_report['regions']['left_hand']['lightness']['statistical_tone']
    assert real_report['regions']['left_hand']['geometry']['status']=='unsupported'
    # Source lips inherit a supported face palette, without direct lip samples,
    # paired positions, a spatial gate or the face's statistical brightness.
    lip=real_report['regions']['lips']
    assert lip['statistics']['status']=='partial' and lip['material']['status']=='partial'
    assert lip['statistics']['mode']=='source_feature_palette_inheritance'
    assert lip['statistics']['parent_region']=='face' and not lip['statistics']['direct_region_reference']
    assert not lip['statistics']['reference_feature_pixels_used']
    assert lip['statistics']['paired_anchor_count']==0 and not lip['lightness']['statistical_tone']
    assert lip['geometry']['status']=='unsupported' and lip['spatial_colour']['status']=='unsupported'
    assert 'native_feature_L_retained' in lip['degradation'] and 'identity_for_absent_statistics' not in lip['degradation']
    assert 'lips' in real_report['stages']['reference_colour_match']['supported_regions']
    assert real_report['stages']['reference_colour_blend']['status']=='unsupported'
    assert real_report['stages']['reference_scene_balance']['status']=='unsupported'
    assert report_state(real_report,binding)['valid']
    for key,value in (('source_feature_labels',[24]),('parent_region','hair'),
                      ('native_L_and_feature_contrast_retained',False),
                      ('reference_feature_pixels_used',True),('spatial_correspondence_claimed',True)):
        corrupt=copy.deepcopy(fitted);corrupt['feature_palette_policy'][key]=value
        rejected(lambda:build_reference_support(corrupt,receipts,receipts,binding=binding))
    corrupt=copy.deepcopy(fitted);corrupt['operators']['face']['matrix']=[[1.1,0.],[0.,1.]]
    rejected(lambda:build_reference_support(corrupt,receipts,receipts,binding=binding))
    corrupt=copy.deepcopy(fitted);corrupt['operators'].pop('face')
    rejected(lambda:build_reference_support(corrupt,receipts,receipts,binding=binding))
    without_palette=copy.deepcopy(fitted);without_palette.pop('feature_palette_policy')
    direct=build_reference_support(without_palette,receipts,receipts,binding=binding)
    assert direct['regions']['lips']['statistics']['status']=='unsupported'
    print(json.dumps({'ok':True,'current_support_and_donor_math':True,
        'no_donor_preserves_source':True,'no_cross_material_borrowing':True,
        'corruption_not_concealed':True,'sleeve_edge_delta_bounded_before_composition':evidence,
        'reports_content_bound_path_independent_current_only':True,
        'real_unpaired_material_statistics_colour_without_spatial_support':True,
        'skin_family_prior_partial_not_direct_hand_or_tone_support':True,
        'skin_prior_rejects_cross_material_parent_and_lightness':True,
        'lip_palette_partial_no_direct_reference_or_spatial_or_L_support':True}))


if __name__=='__main__':main()
