"""Correspondence hulls cannot become skin illumination boundaries."""
from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np

from reference_adaptive_fusion import solve_field
from reference_automatic_layers import scene_field


def main():
    h,w=96,128
    yy,xx=np.mgrid[:h,:w]
    material=((xx-66)/46)**2+((yy-48)/38)**2<1
    labels=np.zeros((h,w),np.int32)
    labels[material]=3
    labels[:,116:]=4
    labels[3:8,3:8]=3  # separate same-owner island with no observations
    valid=material&(yy>25+.28*xx)
    confidence=valid.astype(np.float32)*.94
    lab=np.full((h,w,3),[.45,.015,-.01],np.float32)
    residual=np.full((h,w),-.18,np.float32)
    original_lab,original_confidence=lab.copy(),confidence.copy()
    old,_=solve_field(residual,confidence,labels,lab,maximum_norm=.25)
    field,report=solve_field(residual,confidence,labels,lab,maximum_norm=.25,
        minimum_value=-.25,maximum_value=.25,continuation_domain=labels==3)
    expected=-.18*.94/(.94+.5*.06)
    np.testing.assert_allclose(field[material],expected,atol=8e-6)
    assert abs(float(old[24,64]))<1e-5
    assert abs(float(field[24,64]))>.17
    assert not field[~material].any()
    assert np.array_equal(lab,original_lab) and np.array_equal(confidence,original_confidence)
    assert report['continuation']['continued_source_pixels']>0
    assert not report['continuation']['reference_correspondence_extended']
    assert report['converged']
    bounded,bounded_report=solve_field(residual*8,confidence,labels,lab,maximum_norm=.25,
        minimum_value=-.25,maximum_value=.25,continuation_domain=labels==3)
    np.testing.assert_allclose(bounded[material],-.25,atol=2e-6)
    assert bounded_report['continuation']['projected_converged']
    assert bounded_report['continuation']['projected_kkt_residual']<=1e-6
    # Unobserved residual bytes are not data and cannot become a target.
    altered=residual.copy();altered[~valid]=.23
    same,_=solve_field(altered,confidence,labels,lab,maximum_norm=.25,
        minimum_value=-.25,maximum_value=.25,continuation_domain=labels==3)
    np.testing.assert_allclose(same,field,atol=8e-6)
    rotated,_=solve_field(np.rot90(residual),np.rot90(confidence),np.rot90(labels),np.rot90(lab),
        maximum_norm=.25,minimum_value=-.25,maximum_value=.25,continuation_domain=np.rot90(labels==3))
    np.testing.assert_allclose(rotated,np.rot90(field),atol=8e-6)

    # Both sides retain independent red/green source illumination. A strong
    # same-skin colour edge stops field continuation; no source pixels blur.
    dual=lab.copy();dual[:,:w//2,1]=-.05;dual[:,w//2:,1]=.05
    desired=np.where(xx<w//2,-.12,.12).astype(np.float32)
    two_tone,_=solve_field(desired,confidence,labels,dual,maximum_norm=.25,
        minimum_value=-.25,maximum_value=.25,continuation_domain=labels==3)
    np.testing.assert_allclose(two_tone[material&(xx<w//2)],-.12*.94/.97,atol=8e-6)
    np.testing.assert_allclose(two_tone[material&(xx>=w//2)],.12*.94/.97,atol=5e-5)
    left_only=confidence.copy();left_only[:,w//2:]=0
    isolated,_=solve_field(desired,left_only,labels,dual,maximum_norm=.25,
        continuation_domain=labels==3)
    assert not isolated[:,w//2:].any()

    # Exercise production S routing on face and arm owners, with an angled
    # partial correspondence domain and unchanged semantic/visibility data.
    maps={'ref_x':np.where(valid,xx,-1).astype(np.float32),
          'ref_y':np.where(valid,yy,-1).astype(np.float32),
          'confidence':confidence,'valid':valid}
    reference=lab.copy();reference[material,0]*=2**(-.12/3)
    alpha=(labels>0).astype(np.float32)
    for owner,region in ((3,'face'),(7,'left_arm'),(16,'right_arm'),(6,'left_hand'),(15,'right_hand'),(5,'legs')):
        current_labels=labels.copy();current_labels[material]=owner
        actual,reports,_=scene_field(lab,reference,current_labels,current_labels,alpha,
            np.zeros_like(alpha),maps,{'regions':{}})
        assert reports[region]['automatic_solver']['continuation']['continued_source_pixels']>0
        assert actual[24,64,0]<-.11,region
        assert not actual[labels==4,:3].any()
        assert np.max(np.abs(actual[material,0]-actual[material,0].mean()))<1e-5

    # Full-resolution upsampling must retain a real source edge and must not
    # fill an unobserved component, even when it shares the skin owner ID.
    large=(560,560)
    by,bx=np.mgrid[:large[0],:large[1]]
    guide=np.full((*large,3),[.45,-.05,0],np.float32);guide[:,280:,1]=.05
    evidence=((by>220+.15*bx)&(bx<280)).astype(np.float32)
    result,big=solve_field(np.full(large,-.12,np.float32),evidence,np.full(large,3,np.int32),guide,
        maximum_norm=.25,continuation_domain=np.ones(large,bool))
    assert big['source_guided_upsampling'] and not result[:,280:].any()
    assert result[60,120]<-.119
    # Iterative failure uses a checked sparse solve on the reachable domain;
    # a nearest-neighbour initial guess cannot silently become a result.
    with patch('reference_adaptive_fusion.cg', return_value=(np.zeros(h*w,np.float32),1)):
        fallback,fallback_report=solve_field(residual,confidence,np.full((h,w),3,np.int32),lab,
            maximum_norm=.25,continuation_domain=np.ones((h,w),bool))
        assert fallback_report['continuation']['sparse_fallback']['residual_norm']<=fallback_report['continuation']['sparse_fallback']['required_residual_norm']
        assert fallback[0,0]<-.17
    with patch('reference_adaptive_fusion.cg', return_value=(np.zeros(h*w,np.float32),1)), \
            patch('reference_adaptive_fusion.spsolve', side_effect=lambda matrix,rhs:np.full(len(rhs),np.nan)):
        try:
            solve_field(residual,confidence,np.full((h,w),3,np.int32),lab,maximum_norm=.25,
                continuation_domain=np.ones((h,w),bool))
        except ValueError as error:
            assert 'residual check' in str(error)
        else:
            raise AssertionError('Unconverged continuation was accepted')
    print(json.dumps({'ok':True,'angled_hull_not_an_illumination_edge':True,
        'face_and_arm_hand_routes':True,'source_RGB_and_confidence_unchanged':True,
        'red_green_source_edge_and_disconnected_material_preserved':True,
        'fine_grid_edge_isolation':True,'unconverged_solution_rejected':True}))


if __name__=='__main__':main()
