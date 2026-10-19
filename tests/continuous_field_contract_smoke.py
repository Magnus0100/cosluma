"""B/S evidence, fine topology, bounded vectors and continuous sampling dose."""
from __future__ import annotations
import json
import numpy as np
from unittest.mock import patch
from scipy.ndimage import zoom
from reference_adaptive_fusion import solve_field
from reference_automatic_layers import colour_field,scene_field
from reference_scene_balance import scene_field_owners
from render_reference_colour_composite import sample_maps


def main():
    h,w=96,128;yy,xx=np.mgrid[:h,:w]
    labels=np.full((h,w),3,np.int32);lab=np.full((h,w,3),[.45,.01,.01],np.float32)
    valid=yy>25+.28*xx;c=valid.astype(np.float32)*.94
    maps={'ref_x':np.where(valid,xx,-1).astype(np.float32),
        'ref_y':np.where(valid,yy,-1).astype(np.float32),'confidence':c,'valid':valid}
    ref=lab.copy();ref[...,1]+=.03
    b,reports=colour_field(lab,ref,labels,labels,np.ones((h,w),np.float32),maps,{'regions':{}})
    assert np.max(np.abs(b[...,0]-.03*.94/.97))<2e-5
    assert reports['face']['observed_pixels']==int(valid.sum())
    # The two-channel constraint is a ball, not two independent scalar clips.
    residual=np.stack((.09*np.sin(xx/19),.09*np.cos(yy/23)),axis=-1).astype(np.float32)
    vector,report=solve_field(residual,c,labels,lab,maximum_norm=.04,continuation_domain=np.ones((h,w),bool))
    bounded=report['continuation']['bounded_vector']
    assert bounded['converged'] and bounded['factorizations']==1
    assert max(bounded['primal_residual'],bounded['dual_residual'],bounded['projected_kkt_residual'])<=1e-6
    assert np.linalg.norm(vector,axis=-1).max()<=.040001
    assert np.ptp(vector[...,0])>.02 and np.ptp(vector[...,1])>.02
    # An iterative solver's success flag cannot certify an incorrect vector.
    with patch('reference_adaptive_fusion.cg',return_value=(np.zeros(h*w,np.float32),0)):
        corrected,checked=solve_field(residual,c,labels,lab,maximum_norm=.04,continuation_domain=np.ones((h,w),bool))
    assert np.any(corrected)
    assert all(row['converged'] and row['actual_residual_norm']<=row['required_residual_norm']
               for row in checked['continuation']['unconstrained_linear_residuals'])
    assert all(row['reason']=='cg_actual_residual_check' for row in checked['continuation']['sparse_fallbacks'])
    # Missing and infinitesimal evidence must have the same limiting field;
    # a confidence boundary is neither a material boundary nor a zero dose.
    step=np.full((48,80),.94,np.float32);step[:,40:]=0
    tiny=step.copy();tiny[:,40:]=1e-7
    guide=np.full((48,80,3),[.45,.01,.01],np.float32)
    owner=np.full((48,80),3,np.int32);domain=np.ones((48,80),bool)
    for residual in (np.full((48,80),.15,np.float32),
                     np.broadcast_to(np.array([.015,-.012],np.float32),(48,80,2)).copy()):
        absent,absent_report=solve_field(residual,step,owner,guide,maximum_norm=.25,continuation_domain=domain)
        epsilon,_=solve_field(residual,tiny,owner,guide,maximum_norm=.25,continuation_domain=domain)
        np.testing.assert_allclose(epsilon,absent,atol=2e-6)
        np.testing.assert_allclose(absent,residual*(.94/.97),atol=2e-6)
        diagonal=(np.indices((48,80))[0]>.2*np.indices((48,80))[1]+10)
        varied=np.where(diagonal,.94,.2).astype(np.float32)
        result,_=solve_field(residual,varied,owner,guide,maximum_norm=.25,continuation_domain=domain)
        expected_q=float((varied.astype(np.float64)**2).sum()/varied.sum(dtype=np.float64))
        np.testing.assert_allclose(result,residual*(expected_q/(expected_q+.5*(1-expected_q))),atol=3e-6)
        rotated,_=solve_field(np.rot90(residual),np.rot90(varied),np.rot90(owner),np.rot90(guide),
            maximum_norm=.25,continuation_domain=np.rot90(domain))
        np.testing.assert_allclose(rotated,np.rot90(result),atol=3e-6)
        assert absent_report['energy_terms']['prior']==0

    # Coarse nearest sampling misses this source barrier. Its fine graph
    # components must nonetheless remain separate during solve and upsampling.
    n=560;sampled=np.unique(zoom(np.arange(n),(512/n),order=0,mode='nearest',prefilter=False))
    edge=int(np.setdiff1d(np.arange(150,410),sampled)[12])
    guide=np.full((n,n,3),[.45,.01,.01],np.float32);guide[:,edge,0]=.1
    confidence=np.zeros((n,n),np.float32);confidence[200:,50:edge-20]=1
    result,fine=solve_field(np.full((n,n),-.12,np.float32),confidence,np.full((n,n),3,np.int32),guide,
        maximum_norm=.25,continuation_domain=np.ones((n,n),bool))
    assert edge not in sampled and fine['source_guided_upsampling']
    assert result[100,edge-30]<-.119 and not result[:,edge+1:].any()
    # The same applies to a one-pixel actual occluder and isolated material.
    material=np.full((n,n),3,np.int32);material[:,edge]=4
    result,_=solve_field(np.full((n,n),-.12,np.float32),confidence,material,np.full((n,n,3),[.45,0,0],np.float32),
        maximum_norm=.25,continuation_domain=material==3)
    assert not result[:,edge+1:].any() and not result[:,edge].any()
    # A microscopic observed component that decimation drops is reported and
    # receives exact zero; the main component cannot fill it by proximity.
    missing=int(np.setdiff1d(np.arange(180,380),sampled)[8])
    material=np.full((n,n),3,np.int32);material[missing-1:missing+2,missing-1:missing+2]=0
    material[missing,missing]=3;confidence[material!=3]=0;confidence[missing,missing]=1
    result,dropped=solve_field(np.full((n,n),-.12,np.float32),confidence,material,
        np.full((n,n,3),[.45,0,0],np.float32),maximum_norm=.25,continuation_domain=material==3)
    assert result[missing,missing]==0
    assert dropped['continuation']['omitted_observed_components']>=1
    assert dropped['continuation']['omitted_observation_pixels']>=1

    # Protection is an independently sampled continuous dose, not an owner
    # ID discontinuity at p=.5. Exercise actual S producer and native sampler.
    labels=np.full((16,32),3,np.int32);source=np.full((16,32,3),[.4,0,0],np.float32)
    target=source.copy();target[...,0]=.6
    y,x=np.indices(labels.shape,dtype=np.float32)
    full={'ref_x':x,'ref_y':y,'confidence':np.ones(labels.shape,np.float32),'valid':np.ones(labels.shape,bool)}
    protection=np.broadcast_to(np.linspace(.1,.9,32,dtype=np.float32),(16,32)).copy()
    field,_,_=scene_field(source,target,labels,labels,np.ones(labels.shape,np.float32),protection,full,{'regions':{}})
    owners=scene_field_owners(labels,np.zeros(labels.shape,bool))
    assert np.unique(owners).tolist()==[3]
    sampled=sample_maps(field,(0,0,320,160),(32,16),(320,160),owners=owners)
    assert np.max(np.abs(np.diff(sampled[80,:,0])))<.001
    np.testing.assert_allclose(sampled[80,5:315:10,0],field[8,:31,0],atol=.001)
    print(json.dumps({'ok':True,'B_vector_and_S_share_continuation':True,'bounded_vector_ADMM_KKT_checked':True,
        'fine_one_pixel_colour_and_occluder_barriers_preserved':True,'lost_components_fail_closed_and_reported':True,
        'soft_protection_native_interpolation_continuous':True}))


if __name__=='__main__':main()
