"""Evidence-weighted correction fields; never smooth or replace source pixels.

The solver works on a bounded analysis grid and returns a source-guided field
in the caller's grid. Spatial evidence belongs to the caller's correspondence
map. Label overlap, background brightness or this optimization cannot invent it.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, label, zoom
from scipy.sparse.linalg import LinearOperator, cg

SCHEMA = 'reference_adaptive_fusion.v1'
CONFIG = {'maximum_analysis_long_side': 512, 'lambda': .75, 'mu': .5,
    'relative_tolerance': 1e-5, 'maximum_cg_iterations': 96,
    'maximum_projected_iterations': 128, 'edge_scale_floor': .012,
    'edge_scale_ceiling': .06, 'edge_cutoff_sigmas': 3.,
    'lowpass_scale_fractions': [.003, .010, .030]}


def _inputs(residual, confidence, labels, source_lab):
    r,c,labels,lab = map(np.asarray,(residual,confidence,labels,source_lab))
    scalar = r.ndim == 2
    if (r.ndim not in (2,3) or not r.size or (not scalar and not 1 <= r.shape[2] <= 3)
            or c.shape != r.shape[:2] or labels.shape != c.shape or lab.shape != (*c.shape,3)
            or labels.dtype.kind not in 'iu' or labels.dtype.kind == 'b'
            or not all(np.isfinite(value).all() for value in (r,c,lab))
            or c.min() < 0 or c.max() > 1 or lab[...,0].min() < 0 or lab[...,0].max() > 1):
        raise ValueError('Finite residual, confidence [0,1], integer labels and source OKLab must share a nonempty grid')
    return (r[...,None] if scalar else r).astype(np.float32),c.astype(np.float32),labels,lab.astype(np.float32),scalar


def _project(values, maximum_norm, lower, upper):
    result=np.clip(values,lower,upper)
    norms=np.linalg.norm(result,axis=-1)
    active=norms > maximum_norm
    if not active.any(): return result
    if not np.isfinite(lower) and not np.isfinite(upper):
        return result*np.minimum(1,maximum_norm/np.maximum(norms,1e-30))[...,None]
    # The exact Euclidean projection onto box intersected with a centred ball.
    # Zero is required feasible, so this monotone multiplier always brackets.
    selected=values[active];lo=np.zeros(len(selected),np.float32)
    hi=np.maximum(np.linalg.norm(selected,axis=-1)/max(maximum_norm,1e-30),1)
    for _ in range(24):
        middle=(lo+hi)*.5
        trial=np.clip(selected/(1+middle[:,None]),lower,upper)
        too_large=np.linalg.norm(trial,axis=-1)>maximum_norm
        lo=np.where(too_large,middle,lo);hi=np.where(too_large,hi,middle)
    result[active]=np.clip(selected/(1+hi[:,None]),lower,upper)
    return result


def _resize(values, shape, order=1):
    factors=(shape[0]/values.shape[0],shape[1]/values.shape[1])+(1,)*(values.ndim-2)
    return zoom(values,factors,order=order,mode='nearest',prefilter=False)


def _analysis(r,c,labels,lab):
    h,w=c.shape;factor=min(1,CONFIG['maximum_analysis_long_side']/max(h,w))
    shape=(max(1,round(h*factor)),max(1,round(w*factor)))
    if shape==(h,w): return r,c,labels,lab
    # Correspondence confidence is evidence, not an image to blur. Sampling
    # it together with its owner and guide keeps zero-evidence/strong-colour
    # boundaries intact. The following lowpass normalizes inside each owner.
    return (_resize(r,shape,0),_resize(c,shape,0),
            _resize(labels,shape,0),_resize(lab,shape,0))


def _lowpass(r,c,labels,lab,edge_scale):
    """Normalize each scale inside its own material, then select by variance."""
    output=np.zeros_like(r);scale_use=np.zeros(3,np.float64)
    extent=min(c.shape)
    sigmas=[max(.65,f*extent) for f in CONFIG['lowpass_scale_fractions']]
    barrier=np.zeros(c.shape,bool)
    horizontal=(labels[:,1:]!=labels[:,:-1]) | (np.linalg.norm(lab[:,1:]-lab[:,:-1],axis=-1)>CONFIG['edge_cutoff_sigmas']*edge_scale)
    vertical=(labels[1:]!=labels[:-1]) | (np.linalg.norm(lab[1:]-lab[:-1],axis=-1)>CONFIG['edge_cutoff_sigmas']*edge_scale)
    barrier[:,1:]|=horizontal;barrier[:,:-1]|=horizontal;barrier[1:]|=vertical;barrier[:-1]|=vertical
    distance=distance_transform_edt(~barrier) if barrier.any() else np.full(c.shape,np.inf)
    for name in np.unique(labels[c>0]):
        mask=labels==name;weight=c*mask
        numerator=np.zeros_like(r);denominator=np.zeros(c.shape,np.float32)
        for index,sigma in enumerate(sigmas):
            support=gaussian_filter(weight,sigma)
            mean=gaussian_filter(r*weight[...,None],(sigma,sigma,0))/np.maximum(support[...,None],1e-12)
            square=gaussian_filter(r*r*weight[...,None],(sigma,sigma,0))/np.maximum(support[...,None],1e-12)
            variance=np.maximum(square-mean*mean,0).mean(-1)
            guide_mean=gaussian_filter(lab*weight[...,None],(sigma,sigma,0))/np.maximum(support[...,None],1e-12)
            guide_square=gaussian_filter(lab*lab*weight[...,None],(sigma,sigma,0))/np.maximum(support[...,None],1e-12)
            guide_variance=np.maximum(guide_square-guide_mean*guide_mean,0).mean(-1)
            # Gaussian support is material-normalized, but can still jump a
            # same-material black/white colour edge. Retain the observed broad
            # residual near detected boundaries; do not preblur across them.
            mean=np.where((distance>=4*sigma)[...,None],mean,r)
            # High confidence/variation favours the narrow scale. Flat uncertain
            # regions favour broader statistics; no sample-specific controls.
            reliability=support/(1+variance/.0004+guide_variance/.0004)
            reliability*=1+(1-c)*(index+1)
            reliability*=mask & (support>1e-8)
            numerator+=mean*reliability[...,None];denominator+=reliability
            scale_use[index]+=float(reliability[mask & (c>0)].sum())
        output[mask]=(numerator/np.maximum(denominator[...,None],1e-12))[mask]
    total=max(float(scale_use.sum()),1e-30)
    return output,{'sigmas_analysis_pixels':sigmas,'selected_scale_weight_fraction':(scale_use/total).tolist(),
        'selection':'same_material_normalized_lowpass; local_residual_and_source_variance_and_confidence',
        'source_boundary_prefilter':'retain_observed_residual_within_four_scale_sigmas_of_semantic_or_strong_Lab_edge'}


def _conductance(labels,lab):
    dh=np.linalg.norm(lab[:,1:]-lab[:,:-1],axis=-1)
    dv=np.linalg.norm(lab[1:]-lab[:-1],axis=-1)
    same_h=labels[:,1:]==labels[:,:-1];same_v=labels[1:]==labels[:-1]
    observed=np.concatenate([dh[same_h],dv[same_v]])
    scale=float(np.clip(np.quantile(observed,.65) if observed.size else 0,
        CONFIG['edge_scale_floor'],CONFIG['edge_scale_ceiling']))
    def weight(delta,same):
        return (np.exp(-.5*(delta/scale)**2)*same*(delta<=CONFIG['edge_cutoff_sigmas']*scale)).astype(np.float32)
    return weight(dh,same_h),weight(dv,same_v),scale


def _upsample(field,small_labels,small_lab,labels,lab,scale):
    if field.shape[:2]==labels.shape: return field
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


def solve_field(residual,confidence,labels,source_lab,*,maximum_norm,minimum_value=None,maximum_value=None):
    """Return a bounded correction field and its numeric evidence receipt.

    Confidence is local spatial evidence, independently verified by the caller.
    ``minimum_value``/``maximum_value`` are scalar component bounds containing
    zero. The Euclidean ``maximum_norm`` is per pixel. No global IoU test is used.
    """
    r,c,labels,lab,scalar=_inputs(residual,confidence,labels,source_lab)
    lower=-np.inf if minimum_value is None else float(minimum_value)
    upper=np.inf if maximum_value is None else float(maximum_value)
    maximum_norm=float(maximum_norm)
    if (not np.isfinite(maximum_norm) or maximum_norm<0 or lower>0 or upper<0 or lower>upper
            or (minimum_value is not None and not np.isfinite(lower))
            or (maximum_value is not None and not np.isfinite(upper))):
        raise ValueError('Finite bounds must include zero; maximum_norm must be finite and nonnegative')
    report={'schema':SCHEMA,'config':dict(CONFIG),'source_shape':list(c.shape),'channels':r.shape[-1],
        'confidence_pixels':int(np.count_nonzero(c)),'confidence_mean':float(c.mean()),
        'bounds':{'maximum_norm':maximum_norm,'minimum_value':minimum_value,'maximum_value':maximum_value},
        'source_pixels_modified':False,'source_geometry_warped':False,'reference_texture_imported':False,
        'rim_synthesis_implemented':False,'visual_acceptance':False,
        'objective':'sum(c*|d-r_lowpass|^2)+lambda*sum(g*|di-dj|^2)+mu*sum((1-c)*|d|^2)'}
    if not c.any() or not r[c>0].any() or maximum_norm==0:
        result=np.zeros_like(r);report.update(status='identity',exact_zero=True,iterations=0,energy=0.,converged=True)
        return (result[...,0] if scalar else result),report
    yy,xx=np.nonzero(c);padding=max(4,round(min(c.shape)*.01))
    box=(max(0,int(yy.min())-padding),min(c.shape[0],int(yy.max())+padding+1),
         max(0,int(xx.min())-padding),min(c.shape[1],int(xx.max())+padding+1))
    y0,y1,x0,x1=box;crop=(slice(y0,y1),slice(x0,x1))
    ar,ac,al,ag=_analysis(r[crop],c[crop],labels[crop],lab[crop])
    gh,gv,scale=_conductance(al,ag);lam=CONFIG['lambda'];mu=CONFIG['mu']
    target,lowpass_report=_lowpass(ar,ac,al,ag,scale)
    mass=ac+mu*(1-ac);degree=np.zeros_like(ac)
    degree[:,1:]+=gh;degree[:,:-1]+=gh;degree[1:]+=gv;degree[:-1]+=gv
    def action(values):
        out=mass*values
        flux=lam*gh*(values[:,:-1]-values[:,1:]);out[:,:-1]+=flux;out[:,1:]-=flux
        flux=lam*gv*(values[:-1]-values[1:]);out[:-1]+=flux;out[1:]-=flux
        return out
    size=ac.size;operator=LinearOperator((size,size),matvec=lambda value:action(value.reshape(ac.shape)).ravel(),dtype=np.float32)
    diagonal=mass+lam*degree
    preconditioner=LinearOperator((size,size),matvec=lambda value:value/diagonal.ravel(),dtype=np.float32)
    solved=np.zeros_like(target);iterations=[];converged=True
    for channel in range(target.shape[-1]):
        rhs=(ac*target[...,channel]).ravel();counter=[0]
        def step(_value):counter[0]+=1
        initial=(ac*target[...,channel]/mass).ravel()
        values,info=cg(operator,rhs,x0=initial,M=preconditioner,rtol=CONFIG['relative_tolerance'],atol=1e-8,
            maxiter=CONFIG['maximum_cg_iterations'],callback=step)
        if info<0 or not np.isfinite(values).all(): raise ValueError('Adaptive field solver failed')
        solved[...,channel]=values.reshape(ac.shape);iterations.append(counter[0]);converged &= info==0
    bounded=_project(solved,maximum_norm,lower,upper)
    projected_iterations=0
    if not np.array_equal(bounded,solved):
        step_size=1/float((mass+2*lam*degree).max());current=bounded
        for iteration in range(CONFIG['maximum_projected_iterations']):
            gradient=np.stack([action(current[...,channel])-ac*target[...,channel]
                for channel in range(current.shape[-1])],axis=-1)
            following=_project(current-step_size*gradient,maximum_norm,lower,upper)
            delta=float(np.max(np.abs(following-current)));current=following;projected_iterations=iteration+1
            if delta<=1e-6:break
        bounded=current
    # No-evidence disconnected semantic classes never receive another class's
    # correction. Bilateral upsampling returns only a field, never source RGB.
    fine=_upsample(bounded,al,ag,labels[crop],lab[crop],scale)
    fine=_project(fine,maximum_norm,lower,upper)
    result=np.zeros_like(r);result[crop]=fine
    supported=np.zeros(c.shape,bool)
    for name in np.unique(labels[c>0]):
        components,count=label(labels==name)
        accepted=np.unique(components[c>0]);accepted=accepted[accepted>0]
        supported|=np.isin(components,accepted)
    result[~supported]=0
    if (not np.isfinite(result).all() or np.linalg.norm(result,axis=-1).max()>maximum_norm+1e-6
            or result.min()<lower-1e-6 or result.max()>upper+1e-6):
        raise ValueError('Adaptive solver violated correction bounds')
    data=float(np.sum(ac[...,None]*(bounded-target)**2,dtype=np.float64))
    prior=float(mu*np.sum((1-ac[...,None])*bounded**2,dtype=np.float64))
    smooth=float(lam*(np.sum(gh[...,None]*np.diff(bounded,axis=1)**2,dtype=np.float64)
        +np.sum(gv[...,None]*np.diff(bounded,axis=0)**2,dtype=np.float64)))
    report.update(status='prepared',exact_zero=bool(not result.any()),converged=bool(converged),
        iterations=iterations,projected_iterations=projected_iterations,analysis_shape=list(ac.shape),
        analysis_sampling='nearest_evidence_owner_and_source_guide; then_same_material_normalized_lowpass',
        active_crop_yxyx=list(box),source_guided_upsampling=ac.shape!=c[crop].shape,
        conductance_edge_scale=scale,semantic_boundary_conductance=0.,lowpass=lowpass_report,
        energy=data+prior+smooth,energy_terms={'data':data,'prior':prior,'smooth':smooth},
        maximum_applied_norm=float(np.linalg.norm(result,axis=-1).max()),
        changed_pixels=int(np.count_nonzero(np.any(result!=0,axis=-1))),
        whole_frame_overlap_gate_used=False)
    return (result[...,0] if scalar else result),report
