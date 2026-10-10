"""Source-coordinate clone and screened-Poisson healing on linear branches."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates


def _heal_sample(current, donor, eligible, radius):
    """Transfer donor gradients while target boundary/light set the patch tone.

    Solving in log-linear RGB keeps exposure/HDR and avoids an 8-bit OpenCV
    round trip. The screened term anchors broad shading to the photographed
    target. Donor texture is still sampled, so the user should choose matching
    material; this is a healing brush, not automatic skin generation.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import cg
    domain=eligible>1e-3
    ys,xs=np.nonzero(domain)
    count=len(xs)
    if not count:
        return donor
    h,w=domain.shape
    index=np.full((h,w),-1,np.int32)
    index[ys,xs]=np.arange(count)
    screen=1/max(radius*.4,1.)**2
    diagonal=np.full(count,screen,np.float32)
    rows=[];cols=[]
    log_current=np.log(np.maximum(current,1e-6))
    log_difference=np.log(np.maximum(donor,1e-6))-log_current
    rhs=np.zeros((count,3),np.float32)
    for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
        ny,nx=ys+dy,xs+dx
        valid=(ny>=0)&(ny<h)&(nx>=0)&(nx<w)
        center=np.flatnonzero(valid)
        neighbor=index[ny[valid],nx[valid]]
        diagonal[center]+=1
        rhs[center]+=log_difference[ys[valid],xs[valid]]-log_difference[ny[valid],nx[valid]]
        inside=neighbor>=0
        rows.append(center[inside]);cols.append(neighbor[inside])
    offrow=np.concatenate(rows);offcol=np.concatenate(cols)
    matrix=coo_matrix((np.concatenate((diagonal,-np.ones(len(offrow),np.float32))),
                      (np.concatenate((np.arange(count),offrow)),
                       np.concatenate((np.arange(count),offcol)))),shape=(count,count)).tocsr()
    corrected=current.copy()
    for channel in range(3):
        delta,_=cg(matrix,rhs[:,channel],rtol=1e-4,atol=0,maxiter=160)
        corrected[ys,xs,channel]=np.exp(log_current[ys,xs,channel]+delta)
    return corrected


def empty_clone_layer():
    return {'schema_version':'clone_retouch.v1','enabled':True,'strokes':[]}


def apply_clone_retouch(values, layer, *, branch, subject_alpha, premultiplied_alpha=None):
    """Sample the current same-branch image; strokes never alter source files.

    A donor rectangle is frozen per stroke so painting over the donor cannot
    smear its own output. Foreground samples are unassociated with donor alpha,
    then associated with target alpha; background donor pixels must be outside
    the original person. Radius is expressed in the recorded native canvas.
    """
    if not layer.get('enabled',True):
        return values
    result=np.asarray(values,np.float32)
    height,width=result.shape[:2];size=np.asarray([width-1,height-1],np.float32)
    alpha=np.asarray(subject_alpha,np.float32)
    for stroke in layer.get('strokes',[]):
        scope=stroke.get('scope','foreground')
        if branch not in (None,scope) or not stroke.get('points'):
            continue
        points=np.asarray(stroke['points'],np.float32)*size
        offset=np.asarray(stroke['source_anchor'],np.float32)*size-points[0]
        native=stroke['native_canvas_size']
        radius=max(.5,float(stroke['radius_px'])*min(width,height)/min(native))
        left,top=np.maximum(np.floor(points.min(0)-radius-1),0).astype(int)
        right,bottom=np.minimum(np.ceil(points.max(0)+radius+2),[width,height]).astype(int)
        if right<=left or bottom<=top:
            continue
        mask=np.zeros((bottom-top,right-left),np.float32)
        hardness=float(stroke.get('hardness',.5));opacity=float(stroke.get('opacity',1.))
        centers=[points[0]]
        for a,b in zip(points[:-1],points[1:]):
            count=max(1,int(np.ceil(np.linalg.norm(b-a)/max(radius*.25,.5))))
            centers.extend(a+(b-a)*t for t in np.linspace(0,1,count+1)[1:])
        for cx,cy in centers:
            x0=max(left,int(np.floor(cx-radius)));x1=min(right,int(np.ceil(cx+radius+1)))
            y0=max(top,int(np.floor(cy-radius)));y1=min(bottom,int(np.ceil(cy+radius+1)))
            yy,xx=np.mgrid[y0:y1,x0:x1]
            distance=np.hypot(xx-cx,yy-cy)/radius
            falloff=np.clip((1-distance)/max(1-hardness,1e-5),0,1)
            # Smooth feather, max accumulation prevents pen-speed dependence.
            stamp=falloff*falloff*(3-2*falloff)
            patch=mask[y0-top:y1-top,x0-left:x1-left]
            np.maximum(patch,stamp,out=patch)
        yy,xx=np.mgrid[top:bottom,left:right]
        donor_x,donor_y=xx+offset[0],yy+offset[1]
        x0=max(0,int(np.floor(donor_x.min())));x1=min(width,int(np.ceil(donor_x.max()))+2)
        y0=max(0,int(np.floor(donor_y.min())));y1=min(height,int(np.ceil(donor_y.max()))+2)
        if x1<=x0 or y1<=y0:
            continue
        donor=result[y0:y1,x0:x1].copy()
        local=[donor_y-y0,donor_x-x0]
        sampled=np.stack([map_coordinates(donor[...,c],local,order=1,mode='constant',cval=0,prefilter=False)
                          for c in range(3)],-1)
        sample_alpha=map_coordinates(alpha[y0:y1,x0:x1],local,order=1,mode='constant',cval=0,prefilter=False)
        valid=(donor_x>=0)&(donor_x<=width-1)&(donor_y>=0)&(donor_y<=height-1)
        target_alpha=alpha[top:bottom,left:right]
        weight=mask*opacity*valid
        if scope=='foreground':
            weight*=sample_alpha>.01
            if premultiplied_alpha is not None:
                sampled=np.divide(sampled,sample_alpha[...,None],out=np.zeros_like(sampled),where=sample_alpha[...,None]>.01)
                sampled*=np.asarray(premultiplied_alpha)[top:bottom,left:right,None]
            else:
                weight*=target_alpha
        else:
            weight*=sample_alpha<.01
            if branch is None:
                weight*=1-target_alpha
        current=result[top:bottom,left:right]
        if stroke.get('mode','clone')=='heal':
            # Healing is evaluated on straight RGB. A transparent foreground
            # must never donate alpha-associated dark values to another edge.
            association=(np.asarray(premultiplied_alpha)[top:bottom,left:right,None]
                         if premultiplied_alpha is not None else None)
            if association is not None:
                straight_current=np.divide(current,association,out=np.zeros_like(current),where=association>1e-3)
                straight_sample=np.divide(sampled,association,out=np.zeros_like(sampled),where=association>1e-3)
                eligible=weight*(association[...,0]>.01)
                sampled=_heal_sample(straight_current,straight_sample,eligible,radius)*association
            else:
                sampled=_heal_sample(current,sampled,weight,radius)
        current[:]=current*(1-weight[...,None])+sampled*weight[...,None]
    return result
