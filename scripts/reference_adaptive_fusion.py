"""Evidence-weighted correction fields; never smooth or replace source pixels.

The solver works on a bounded analysis grid and returns a source-guided field
in the caller's grid. Spatial evidence belongs to the caller's correspondence
map. Label overlap, background brightness or this optimization cannot invent it.
"""
from __future__ import annotations

import numpy as np
import time
from scipy.ndimage import distance_transform_edt, gaussian_filter, label, zoom
from scipy.sparse.linalg import LinearOperator, cg, spsolve, splu
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

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


def _lowpass(r,c,labels,lab,edge_scale,*,sigmas=None):
    """Normalize each scale inside its own material, then select by variance."""
    output=np.zeros_like(r);scale_use=np.zeros(3,np.float64)
    extent=min(c.shape)
    sigmas=[max(.65,f*extent) for f in CONFIG['lowpass_scale_fractions']] if sigmas is None else sigmas
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
        'scale_weight_sum':float(scale_use.sum()),
        'selection':'same_material_normalized_lowpass; local_residual_and_source_variance_and_confidence',
        'source_boundary_prefilter':'retain_observed_residual_within_four_scale_sigmas_of_semantic_or_strong_Lab_edge'}


def _component_lowpass(r,c,labels,lab,edge_scale):
    """Evaluate only actual data rows, with unchanged four-sigma filter closure."""
    sigmas=[max(.65,f*min(c.shape)) for f in CONFIG['lowpass_scale_fractions']]
    padding=int(np.ceil(4*max(sigmas)))+1
    output=np.zeros_like(r);weights=np.zeros(3,np.float64);count=0
    for owner in np.unique(labels[c>0]):
        data=(labels==owner)&(c>0)
        yy,xx=np.nonzero(data)
        if not len(yy):continue
        y0,y1=max(0,int(yy.min())-padding),min(c.shape[0],int(yy.max())+padding+1)
        x0,x1=max(0,int(xx.min())-padding),min(c.shape[1],int(xx.max())+padding+1)
        region=np.s_[y0:y1,x0:x1];own=labels[region]==owner
        local,report=_lowpass(r[region],c[region]*own,labels[region],lab[region],edge_scale,sigmas=sigmas)
        selected=data[region];output[region][selected]=local[selected]
        weights+=np.asarray(report['selected_scale_weight_fraction'])*report['scale_weight_sum'];count+=1
    return output,{'sigmas_analysis_pixels':sigmas,
        'selected_scale_weight_fraction':(weights/max(weights.sum(),1e-30)).tolist(),
        'selection':'fine_component_normalized_lowpass_at_observed_data_only',
        'component_count':count,'filter_halo_pixels':padding,
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
    guide_flat=small_lab.reshape(-1,3);labels_flat=small_labels.reshape(-1)
    field_flat=field.reshape(-1,field.shape[-1])
    for top in range(0,h,128):
        end=min(h,top+128);yy=ys[top:end];y0=np.floor(yy).astype(int);y1=np.minimum(y0+1,sh-1);fy=yy-y0
        denominator=np.zeros((end-top,w),np.float32);numerator=np.zeros((end-top,w,field.shape[-1]),np.float32)
        for iy,wy in ((y0,1-fy),(y1,fy)):
            for ix,wx in ((x0,1-fx),(x1,fx)):
                # Row-major gather preserves the four-corner arithmetic while
                # avoiding two-index broadcasting in each native band.
                grid=iy[:,None]*sw+ix[None,:]
                distance=np.linalg.norm(lab[top:end]-np.take(guide_flat,grid,axis=0),axis=-1)
                weight=(wy[:,None]*wx[None,:])*(labels[top:end]==np.take(labels_flat,grid,axis=0))
                weight*=np.exp(-.5*(distance/scale)**2)*(distance<=CONFIG['edge_cutoff_sigmas']*scale)
                numerator+=np.take(field_flat,grid,axis=0)*weight[...,None];denominator+=weight
        result[top:end]=numerator/np.maximum(denominator[...,None],1e-12)
    return result


def _fine_topology(domain, labels, lab, confidence):
    """Preserve source/owner/occluder connectivity before analysis decimation."""
    gh,gv,_=_conductance(labels,lab)
    gh *= domain[:,1:]&domain[:,:-1]
    gv *= domain[1:]&domain[:-1]
    nodes=np.flatnonzero(domain.ravel())
    remap=np.full(domain.size,-1,np.int32);remap[nodes]=np.arange(len(nodes))
    grid=np.arange(domain.size).reshape(domain.shape)
    left,top=gh>0,gv>0
    rows=np.concatenate((remap[grid[:,:-1][left]],remap[grid[:-1][top]]))
    columns=np.concatenate((remap[grid[:,1:][left]],remap[grid[1:][top]]))
    graph=coo_matrix((np.ones(len(rows),np.uint8),(rows,columns)),shape=(len(nodes),len(nodes))).tocsr()
    _,components=connected_components(graph,directed=False)
    topology=np.zeros(domain.size,np.int32);topology[nodes]=components+1
    topology=topology.reshape(domain.shape)
    observed=np.unique(topology[confidence>0])
    reachable=domain&np.isin(topology,observed)
    return topology,reachable,observed


def _vector_bounded_solution(matrix, rhs, initial, maximum_norm, lower, upper):
    """ADMM with one shared sparse factorization and an exact convex prox."""
    from scipy.sparse import eye
    rho=.1
    factor=splu(matrix+rho*eye(matrix.shape[0],format='csc'))
    z=_project(initial,maximum_norm,lower,upper).astype(np.float64)
    u=np.zeros_like(z)
    primal=dual=kkt=float('inf')
    for iteration in range(1024):
        x=factor.solve(rhs+rho*(z-u))
        relaxed=1.6*x-.6*z
        following=_project(relaxed+u,maximum_norm,lower,upper)
        primal=float(np.max(np.abs(x-following)))
        dual=float(rho*np.max(np.abs(following-z)))
        u+=relaxed-following;z=following
        if max(primal,dual)<=1e-6 or iteration%16==15:
            gradient=matrix@z-rhs
            kkt=float(np.max(np.abs(z-_project(z-gradient,maximum_norm,lower,upper))))
            if max(primal,dual,kkt)<=1e-6:
                return z,{'solver':'shared_sparse_ADMM_exact_box_ball_prox',
                    'iterations':iteration+1,'rho':rho,'primal_residual':primal,
                    'dual_residual':dual,'projected_kkt_residual':kkt,
                    'converged':True,'factorizations':1}
    raise ValueError(f'Bounded vector field did not converge: primal={primal}; dual={dual}; KKT={kkt}')


def solve_field(residual,confidence,labels,source_lab,*,maximum_norm,minimum_value=None,maximum_value=None,
                continuation_domain=None, allow_relaxed_continuation=False):
    """Return a bounded correction field and its numeric evidence receipt.

    Confidence is local spatial evidence, independently verified by the caller.
    ``minimum_value``/``maximum_value`` are scalar component bounds containing
    zero. The Euclidean ``maximum_norm`` is per pixel. No global IoU test is used.
    """
    r,c,labels,lab,scalar=_inputs(residual,confidence,labels,source_lab)
    if continuation_domain is not None:
        continuation_domain = np.asarray(continuation_domain)
        if (continuation_domain.dtype != np.dtype(bool) or continuation_domain.shape != c.shape
                or np.any((c>0)&~continuation_domain)):
            raise ValueError('Field continuation requires a boolean material domain containing all observations')
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
    yy,xx=np.nonzero(c if continuation_domain is None else continuation_domain)
    padding=max(4,round(min(c.shape)*.01))
    box=(max(0,int(yy.min())-padding),min(c.shape[0],int(yy.max())+padding+1),
         max(0,int(xx.min())-padding),min(c.shape[1],int(xx.max())+padding+1))
    y0,y1,x0,x1=box;crop=(slice(y0,y1),slice(x0,x1))
    fine_topology=fine_reachable=fine_observed=None
    if continuation_domain is not None:
        fine_topology,fine_reachable,fine_observed=_fine_topology(
            continuation_domain[crop],labels[crop],lab[crop],c[crop])
        fine_confidence=c[crop].astype(np.float64)
        sums=np.bincount(fine_topology.ravel(),weights=fine_confidence.ravel())
        squares=np.bincount(fine_topology.ravel(),weights=(fine_confidence*fine_confidence).ravel())
        quality=np.divide(squares,sums,out=np.zeros_like(sums),where=sums>0)
        quality=np.clip(quality,0,1)
        denominator=quality+CONFIG['mu']*(1-quality)
        component_strength=np.divide(quality,denominator,out=np.zeros_like(quality),where=denominator>0)
    analysis_labels=fine_topology if fine_topology is not None else labels[crop]
    ar,ac,al,ag=_analysis(r[crop],c[crop],analysis_labels,lab[crop])
    gh,gv,scale=_conductance(al,ag);lam=CONFIG['lambda'];mu=CONFIG['mu']
    reachable = None
    if continuation_domain is not None:
        domain = (_resize(continuation_domain[crop], ac.shape, 0).astype(bool)
                  if ac.shape != continuation_domain[crop].shape else continuation_domain[crop])
        gh *= domain[:,1:]&domain[:,:-1]
        gv *= domain[1:]&domain[:-1]
        # A source colour edge, occluder, or material boundary can disconnect
        # a surface. Only graph components with actual correspondence data
        # may receive a continued illumination field.
        grid = np.arange(ac.size).reshape(ac.shape)
        left, top = gh>0, gv>0
        rows = np.concatenate((grid[:,:-1][left], grid[:-1][top]))
        columns = np.concatenate((grid[:,1:][left], grid[1:][top]))
        graph = coo_matrix((np.ones(len(rows),np.uint8),(rows,columns)),shape=(ac.size,ac.size)).tocsr()
        _, components = connected_components(graph, directed=False)
        components = components.reshape(ac.shape)
        reachable = domain & np.isin(components,np.unique(components[ac>0]))
    target,lowpass_report=(_component_lowpass(ar,ac,al,ag,scale) if continuation_domain is not None
                          else _lowpass(ar,ac,al,ag,scale))
    mass=ac+mu*(1-ac)
    if reachable is not None:
        # Confidence weights observations continuously, including c -> 0.
        # A local zero prior would make epsilon evidence resurrect a hard
        # application boundary. Reliability instead scales the target once
        # for its complete fine source component, independent of hull area.
        mass = np.where(reachable,ac,1).astype(np.float32)
        target *= component_strength[al][...,None].astype(np.float32)
        report['continuation'] = {'policy': 'same_material_source_guided_harmonic.v2',
            'analysis_domain_pixels': int(domain.sum()), 'analysis_reachable_pixels': int(reachable.sum()),
            'analysis_observed_pixels': int(np.count_nonzero(ac)),
            'reference_correspondence_extended': False, 'source_RGB_smoothed': False,
            'observation_confidence_changed': False,
            'component_reliability': 'q=sum(c*c)/sum(c); target_gain=q/(q+mu*(1-q))',
            'component_reliability_mu':mu,
            'observed_component_target_gain_range': [float(component_strength[fine_observed].min()),
                                                       float(component_strength[fine_observed].max())],
            'maximum_cg_iterations': max(512,CONFIG['maximum_cg_iterations']),
            'bounded_solver': ('sparse_primal_dual_active_set_when_needed' if scalar
                               else 'shared_sparse_ADMM_exact_box_ball_prox_when_needed'),
            'maximum_active_set_iterations': 32,
            'unobserved_zero_prior': False, 'unobserved_disconnected_components': 'exact_zero',
            'boundaries': 'semantic_owner; explicit_material_domain; strong_source_OKLab_edge'}
        represented=np.unique(al[ac>0])
        omitted=np.setdiff1d(fine_observed,represented)
        report['continuation'].update(fine_topology_before_decimation=True,
            fine_observed_components=len(fine_observed),omitted_observed_components=len(omitted),
            omitted_observation_pixels=int(np.count_nonzero((c[crop]>0)&np.isin(fine_topology,omitted))),
            omitted_component_policy='exact_zero; not_declared_covered')
        report['objective'] = ('sum(c*|d-component_gain*r_lowpass|^2)'
                               '+lambda*sum(g*|di-dj|^2); no_local_zero_prior')
    degree=np.zeros_like(ac)
    degree[:,1:]+=gh;degree[:,:-1]+=gh;degree[1:]+=gv;degree[:-1]+=gv
    def action(values):
        out=mass*values
        flux=lam*gh*(values[:,:-1]-values[:,1:]);out[:,:-1]+=flux;out[:,1:]-=flux
        flux=lam*gv*(values[:-1]-values[1:]);out[:-1]+=flux;out[1:]-=flux
        return out
    size=ac.size;operator=LinearOperator((size,size),matvec=lambda value:action(value.reshape(ac.shape)).ravel(),dtype=np.float32)
    diagonal=mass+lam*degree
    preconditioner=LinearOperator((size,size),matvec=lambda value:value/diagonal.ravel(),dtype=np.float32)
    continuation_matrix = None
    continuation_indices = None
    def sparse_continuation_matrix():
        nonlocal continuation_matrix, continuation_indices
        if continuation_matrix is None:
            continuation_indices = np.flatnonzero(reachable.ravel())
            remap = np.full(ac.size,-1,np.int32)
            remap[continuation_indices] = np.arange(len(continuation_indices))
            edge_values = np.concatenate((gh[left],gv[top])).astype(np.float64)
            accepted = (remap[rows]>=0)&(remap[columns]>=0)
            a,b = remap[rows[accepted]],remap[columns[accepted]]
            nodes = np.arange(len(continuation_indices))
            continuation_matrix = coo_matrix((np.concatenate((
                diagonal.ravel()[continuation_indices].astype(np.float64),
                -lam*edge_values[accepted],-lam*edge_values[accepted])),
                (np.concatenate((nodes,a,b)),np.concatenate((nodes,b,a)))),
                shape=(len(nodes),len(nodes))).tocsc()
        return continuation_matrix, continuation_indices
    solved=np.zeros_like(target);iterations=[];converged=True
    for channel in range(target.shape[-1]):
        rhs=(ac*target[...,channel]).ravel();counter=[0]
        def step(_value):counter[0]+=1
        initial=np.divide(ac*target[...,channel],mass,out=np.zeros_like(ac),where=mass>0)
        if reachable is not None:
            nearest = distance_transform_edt(ac<=0,return_distances=False,return_indices=True)
            same_component = components == components[tuple(nearest)]
            unknown = reachable&(ac==0)&same_component
            initial[unknown] = initial[tuple(nearest)][unknown]
        initial=initial.ravel()
        values,info=cg(operator,rhs,x0=initial,M=preconditioner,rtol=CONFIG['relative_tolerance'],atol=1e-8,
            maxiter=max(512,CONFIG['maximum_cg_iterations']) if reachable is not None else CONFIG['maximum_cg_iterations'],callback=step)
        if info<0 or not np.isfinite(values).all(): raise ValueError('Adaptive field solver failed')
        cg_info=info
        required=max(1e-8,CONFIG['relative_tolerance']*float(np.linalg.norm(rhs)))
        actual_residual=float(np.linalg.norm(operator.matvec(values)-rhs))
        linear_method='CG'
        if reachable is not None and (info != 0 or actual_residual>required):
            started_direct=time.perf_counter()
            matrix,indices=sparse_continuation_matrix()
            cg_values = values.copy()
            values=np.zeros(ac.size,np.float64)
            values[indices]=spsolve(matrix,rhs[indices].astype(np.float64))
            residual_norm=float(np.linalg.norm(operator.matvec(values)-rhs))
            required=max(1e-8,CONFIG['relative_tolerance']*float(np.linalg.norm(rhs)))
            if not np.isfinite(values).all() or residual_norm>required:
                if not allow_relaxed_continuation:
                    raise ValueError('Sparse source-guided continuation failed its actual residual check')
                # A generated scene-design reference may intentionally change
                # the background while the source still owns the person.  In
                # that route the sparse lightness field is a bounded polish
                # layer, not the authority for the new scene plate. Keep the
                # finite CG estimate and record the relaxed residual for
                # review instead of rejecting the whole scene reconstruction.
                values = cg_values
                report.setdefault('continuation', {}).setdefault('relaxed_residuals', []).append({
                    'channel': channel, 'cg_info': int(cg_info),
                    'cg_actual_residual_norm': actual_residual,
                    'required_residual_norm': required,
                    'sparse_residual_norm': residual_norm,
                    'reason': 'scene_design_reference_background_change'})
                residual_norm = actual_residual
                linear_method = 'reachable_sparse_relaxed_cg'
                info = 0
            else:
                fallback={'reason':'cg_iteration_budget' if cg_info else 'cg_actual_residual_check',
                    'channel':channel,'cg_info':int(cg_info),'cg_actual_residual_norm':actual_residual,
                    'reachable_unknowns':len(indices),'matrix_nonzeros':int(matrix.nnz),
                    'residual_norm':residual_norm,'required_residual_norm':required,
                    'seconds':time.perf_counter()-started_direct}
                report['continuation']['sparse_fallback']=fallback
                report['continuation'].setdefault('sparse_fallbacks',[]).append(fallback)
                actual_residual=residual_norm;linear_method='reachable_sparse_direct'
                info=0
        if reachable is not None:
            report['continuation'].setdefault('unconstrained_linear_residuals',[]).append({
                'channel':channel,'method':linear_method,'actual_residual_norm':actual_residual,
                'required_residual_norm':required,'cg_info':int(cg_info),
                'converged':bool(info==0 and actual_residual<=required)})
        solved[...,channel]=values.reshape(ac.shape);iterations.append(counter[0]);converged &= info==0
    bounded=_project(solved,maximum_norm,lower,upper)
    if reachable is not None:
        bounded[~reachable]=0
    projected_iterations=0
    projected_converged=True
    if reachable is not None and not scalar and not np.array_equal(bounded,solved):
        matrix,indices=sparse_continuation_matrix()
        rhs=np.stack([(ac*target[...,channel]).ravel()[indices] for channel in range(target.shape[-1])],axis=-1).astype(np.float64)
        initial=bounded.reshape(-1,bounded.shape[-1])[indices].astype(np.float64)
        started_bounded=time.perf_counter()
        values,vector_report=_vector_bounded_solution(matrix,rhs,initial,maximum_norm,lower,upper)
        vector_report['seconds']=time.perf_counter()-started_bounded
        report['continuation']['bounded_vector']=vector_report
        report['continuation']['projected_converged']=True
        report['continuation']['projected_kkt_residual']=vector_report['projected_kkt_residual']
        bounded.reshape(-1,bounded.shape[-1])[indices]=values.astype(np.float32)
        projected_iterations=vector_report['iterations']
    elif reachable is not None and not np.array_equal(bounded,solved):
        # Scalar EV has box constraints. Solve their KKT system on the
        # reachable surface, rather than stopping a slow projected diffusion
        # while an unsupported forehead/arm still retains its initial value.
        matrix,indices=sparse_continuation_matrix()
        low,high=max(lower,-maximum_norm),min(upper,maximum_norm)
        values=bounded[...,0].ravel()[indices].astype(np.float64)
        rhs=(ac*target[...,0]).ravel()[indices].astype(np.float64)
        projected_converged=False
        for iteration in range(32):
            gradient=matrix@values-rhs
            kkt=float(np.max(np.abs(values-np.clip(values-gradient,low,high))))
            if kkt<=1e-6:
                projected_converged=True
                break
            fixed=((values<=low+1e-9)&(gradient>=0))|((values>=high-1e-9)&(gradient<=0))
            free=~fixed
            if not free.any():
                break
            values[free]=spsolve(matrix[free][:,free],rhs[free]-matrix[free][:,fixed]@values[fixed])
            values=np.clip(values,low,high)
            projected_iterations=iteration+1
        gradient=matrix@values-rhs
        kkt=float(np.max(np.abs(values-np.clip(values-gradient,low,high))))
        projected_converged=kkt<=1e-6
        report['continuation']['projected_kkt_residual']=kkt
        report['continuation']['projected_converged']=projected_converged
        report['continuation']['maximum_active_set_iterations']=32
        if not projected_converged:
            raise ValueError('Bounded source-guided continuation failed its projected KKT check')
        bounded[...,0].ravel()[indices]=values.astype(np.float32)
    elif not np.array_equal(bounded,solved):
        step_size=1/float((mass+2*lam*degree).max());current=bounded
        for iteration in range(max(256,CONFIG['maximum_projected_iterations']) if reachable is not None else CONFIG['maximum_projected_iterations']):
            gradient=np.stack([action(current[...,channel])-ac*target[...,channel]
                for channel in range(current.shape[-1])],axis=-1)
            following=_project(current-step_size*gradient,maximum_norm,lower,upper)
            delta=float(np.max(np.abs(following-current)));current=following;projected_iterations=iteration+1
            if delta<=1e-6:break
        bounded=current
    # No-evidence disconnected semantic classes never receive another class's
    # correction. Bilateral upsampling returns only a field, never source RGB.
    fine=_upsample(bounded,al,ag,analysis_labels,lab[crop],scale)
    fine=_project(fine,maximum_norm,lower,upper)
    result=np.zeros_like(r);result[crop]=fine
    supported=np.zeros(c.shape,bool)
    for name in np.unique(labels[c>0]):
        components,count=label(labels==name)
        accepted=np.unique(components[c>0]);accepted=accepted[accepted>0]
        supported|=np.isin(components,accepted)
    if continuation_domain is not None:
        supported[crop] &= fine_reachable&np.isin(fine_topology,np.unique(al[ac>0]))
        supported &= continuation_domain
    result[~supported]=0
    if (not np.isfinite(result).all() or np.linalg.norm(result,axis=-1).max()>maximum_norm+1e-6
            or result.min()<lower-1e-6 or result.max()>upper+1e-6):
        raise ValueError('Adaptive solver violated correction bounds')
    data=float(np.sum(ac[...,None]*(bounded-target)**2,dtype=np.float64))
    prior_weights = np.zeros_like(ac) if reachable is not None else 1-ac
    prior=float(mu*np.sum(prior_weights[...,None]*bounded**2,dtype=np.float64))
    smooth=float(lam*(np.sum(gh[...,None]*np.diff(bounded,axis=1)**2,dtype=np.float64)
        +np.sum(gv[...,None]*np.diff(bounded,axis=0)**2,dtype=np.float64)))
    report.update(status='prepared',exact_zero=bool(not result.any()),converged=bool(converged),
        convergence_scope='analysis_grid; bounded_KKT_separate_from_unconstrained_linear_residual',
        iterations=iterations,projected_iterations=projected_iterations,analysis_shape=list(ac.shape),
        analysis_sampling='nearest_evidence_owner_and_source_guide; then_same_material_normalized_lowpass',
        active_crop_yxyx=list(box),source_guided_upsampling=ac.shape!=c[crop].shape,
        conductance_edge_scale=scale,semantic_boundary_conductance=0.,lowpass=lowpass_report,
        energy=data+prior+smooth,energy_terms={'data':data,'prior':prior,'smooth':smooth},
        maximum_applied_norm=float(np.linalg.norm(result,axis=-1).max()),
        changed_pixels=int(np.count_nonzero(np.any(result!=0,axis=-1))),
        whole_frame_overlap_gate_used=False)
    if reachable is not None:
        report['continuation']['continued_source_pixels'] = int(np.count_nonzero(
            np.any(result!=0,axis=-1)&(c==0)))
    return (result[...,0] if scalar else result),report
