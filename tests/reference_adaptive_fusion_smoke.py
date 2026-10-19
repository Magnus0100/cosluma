"""Numeric evidence, boundary, constraint and large-grid solver contracts."""
from __future__ import annotations
import json,time
import numpy as np
from reference_adaptive_fusion import CONFIG,solve_field,_upsample


def reference_upsample(field, small_labels, small_lab, labels, lab, scale):
    """Frozen four-corner sampler for ownership/edge and gather regression."""
    h,w=labels.shape;sh,sw=small_labels.shape
    result=np.zeros((h,w,field.shape[-1]),np.float32)
    xs=np.linspace(0,sw-1,w,dtype=np.float32);x0=np.floor(xs).astype(int);x1=np.minimum(x0+1,sw-1);fx=xs-x0
    ys=np.linspace(0,sh-1,h,dtype=np.float32)
    for top in range(0,h,128):
        end=min(h,top+128);yy=ys[top:end];y0=np.floor(yy).astype(int);y1=np.minimum(y0+1,sh-1);fy=yy-y0
        denominator=np.zeros((end-top,w),np.float32);numerator=np.zeros((end-top,w,field.shape[-1]),np.float32)
        for iy,wy in ((y0,1-fy),(y1,fy)):
            for ix,wx in ((x0,1-fx),(x1,fx)):
                grid=(iy[:,None],ix[None,:])
                distance=np.linalg.norm(lab[top:end]-small_lab[grid],axis=-1)
                weight=(wy[:,None]*wx[None,:])*(labels[top:end]==small_labels[grid])
                weight*=np.exp(-.5*(distance/scale)**2)*(distance<=CONFIG['edge_cutoff_sigmas']*scale)
                numerator+=field[grid]*weight[...,None];denominator+=weight
        result[top:end]=numerator/np.maximum(denominator[...,None],1e-12)
    return result


def main():
    rng=np.random.default_rng(671)
    guide=rng.uniform([.25,-.02,-.02],[.65,.02,.02],(131,197,3)).astype(np.float32)
    owners=np.ones((131,197),np.int32);owners[:,99:]=2
    field=rng.uniform(-.03,.03,(131,197,3)).astype(np.float32)
    fine_labels=np.repeat(np.repeat(owners,2,0),2,1)
    fine_lab=np.repeat(np.repeat(guide,2,0),2,1)
    fine_lab[:,199:,0]=.99
    for channels in (1,2,3):
        # Reverse/strided storage also exercises row-major gathers that must
        # preserve the original two-index lookup and channel accumulation.
        args=(field[:,::-1,:channels],owners[:,::-1],guide[:,::-1],fine_labels[:,::-1],fine_lab[:,::-1],.05)
        actual=_upsample(*args);expected=reference_upsample(*args)
        assert np.array_equal(actual.view(np.uint32),expected.view(np.uint32))
    assert not _upsample(field,owners,guide,fine_labels,fine_lab,.05)[:,199:].any()
    h,w=61,73
    lab=np.full((h,w,3),[.5,.02,-.01],np.float32)
    labels=np.ones((h,w),np.uint8);c=np.ones((h,w),np.float32)
    r=np.full((h,w,2),[.03,-.02],np.float32)
    zero,report=solve_field(r,np.zeros_like(c),labels,lab,maximum_norm=.05)
    assert np.array_equal(zero,np.zeros_like(r)) and report['exact_zero'] and report['iterations']==0
    identity,_=solve_field(np.zeros_like(r),c,labels,lab,maximum_norm=.05)
    assert np.array_equal(identity,np.zeros_like(r)), 'same-image residual must be exact identity'
    uniform,report=solve_field(r,c,labels,lab,maximum_norm=.05)
    np.testing.assert_allclose(uniform,r,rtol=0,atol=4e-7)
    assert report['converged'] and report['energy']<1e-7
    # Flat low-confidence observations have a known analytic shrinkage; local
    # confidence remains effective even when whole-frame overlap is tiny.
    uniform_low,_=solve_field(r,c*.2,labels,lab,maximum_norm=.05)
    np.testing.assert_allclose(uniform_low,r*(.2/(.2+.5*.8)),rtol=0,atol=4e-7)
    sparse=np.zeros_like(c);sparse[26:34,31:39]=1
    local,local_report=solve_field(r,sparse,labels,lab,maximum_norm=.05)
    assert local[29,34,0]>.005 and local_report['whole_frame_overlap_gate_used'] is False
    assert np.linalg.norm(local[0,0])<1e-8
    # Material classes cannot borrow corrections, including a nearby island of
    # the same class whose connected component contains no observations.
    split=labels.copy();split[:,w//2:]=2
    evidence=np.zeros_like(c);evidence[:, :w//2]=1
    blocked,_=solve_field(r,evidence,split,lab,maximum_norm=.05)
    assert np.array_equal(blocked[:,w//2:],np.zeros_like(blocked[:,w//2:]))
    islands=split.copy();islands[:,w//2+5:]=1
    isolated,_=solve_field(r,evidence,islands,lab,maximum_norm=.05)
    assert np.array_equal(isolated[:,w//2+5:],np.zeros_like(isolated[:,w//2+5:]))
    # A strong black/white source edge inside the clothing class is also a
    # barrier. Prefiltering the target cannot undermine the graph conductance.
    edge_lab=lab.copy();edge_lab[:,:w//2,0]=.1;edge_lab[:,w//2:,0]=.8
    colour_stop,_=solve_field(r,evidence,labels,edge_lab,maximum_norm=.05)
    assert np.array_equal(colour_stop[:,w//2:],np.zeros_like(colour_stop[:,w//2:]))
    desired=np.full((h,w),.04,np.float32);desired[:,:w//2]=-.04
    edge_result,_=solve_field(desired,c,labels,edge_lab,maximum_norm=.05)
    np.testing.assert_allclose(edge_result,desired,rtol=0,atol=5e-7)
    # Low-texture gradients and sharp native texture are source-owned. Adding a
    # constant correction preserves every adjacent-pixel difference exactly.
    yy,xx=np.mgrid[:h,:w];texture_lab=lab.copy()
    texture_lab[...,0]=.4+.0004*xx+.015*((xx+yy)%2)
    constant,_=solve_field(np.full((h,w),.02,np.float32),c,labels,texture_lab,maximum_norm=.04)
    corrected=texture_lab[...,0]+constant
    np.testing.assert_allclose(np.diff(corrected,axis=1),np.diff(texture_lab[...,0],axis=1),rtol=0,atol=1e-7)
    bounded,bounds=solve_field(r*10,c,labels,lab,maximum_norm=.025,minimum_value=-.01,maximum_value=.02)
    assert np.linalg.norm(bounded,axis=-1).max()<=.025001
    assert bounded.min()>=-.010001 and bounded.max()<=.020001 and bounds['projected_iterations']>0
    for kwargs in ({'maximum_norm':float('nan')},{'maximum_norm':-.1},
                   {'maximum_norm':.05,'minimum_value':.001}, {'maximum_norm':.05,'maximum_value':-.001}):
        try:
            solve_field(r,c,labels,lab,**kwargs)
            raise AssertionError('Invalid bounds accepted')
        except ValueError: pass
    for role in ('residual','confidence','labels','source_lab'):
        args=[r.copy(),c.copy(),labels.copy(),lab.copy()]
        index={'residual':0,'confidence':1,'labels':2,'source_lab':3}[role]
        if role=='labels':args[index]=args[index].astype(np.float32)
        else:args[index].flat[0]=float('nan')
        try:
            solve_field(*args,maximum_norm=.05)
            raise AssertionError('Invalid input accepted: '+role)
        except ValueError: pass
    try:
        solve_field(r,c*1.01,labels,lab,maximum_norm=.05)
        raise AssertionError('Confidence outside range accepted')
    except ValueError:pass
    # Production-sized proxy, with source-guided return to its original grid.
    # This is a real numeric solve, not an allocation or mocked convergence.
    big_shape=(2048,1365)
    by,bx=np.mgrid[:big_shape[0],:big_shape[1]]
    big_r=(.015+.004*np.sin(bx/91)*np.cos(by/117)).astype(np.float32)
    big_c=(.6+.3*np.cos(by/103)).astype(np.float32);big_labels=np.ones(big_shape,np.uint8)
    big_lab=np.full((*big_shape,3),[.5,.02,-.01],np.float32)
    started=time.perf_counter();big,large=solve_field(big_r,big_c,big_labels,big_lab,maximum_norm=.03)
    elapsed=time.perf_counter()-started
    assert big.shape==big_shape and max(large['analysis_shape'])<=CONFIG['maximum_analysis_long_side']
    assert large['source_guided_upsampling'] and large['converged']
    assert np.isfinite(big).all() and big.min()>0 and big.max()<=.030001
    assert max(large['iterations'])>0 and np.ptp(big)>.003
    assert elapsed<30, ('Production proxy solver exceeded bounded-runtime smoke',elapsed)
    print(json.dumps({'ok':True,'zero_evidence_and_same_image_exact_identity':True,
        'analytic_uniform_solution_and_low_confidence_shrinkage':True,'local_evidence_without_global_overlap_gate':True,
        'no_cross_material_or_unsupported_component_leakage':True,'source_colour_edge_stops_propagation':True,
        'native_texture_and_low_texture_gradient_retained':True,'constraints_and_invalid_input_checked':True,
        'production_proxy_analysis_shape':large['analysis_shape'],'production_proxy_seconds':round(elapsed,3),
        'production_proxy_iterations':large['iterations'],'reference_rim_synthesis':False}))


if __name__=='__main__':main()
