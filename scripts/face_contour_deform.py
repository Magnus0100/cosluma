"""Checked canonical scanline inverse sampling, without generated appearance.

The complete central protection and every pixel outside the admitted cheek/jaw
band stay exact. RGB and existing generated provenance use the same map.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import importlib
import importlib.metadata
import math
from pathlib import Path

import numpy as np
import scipy
from scipy.ndimage import distance_transform_edt, map_coordinates

from asset_identity import sha256_file
from photo_editing import identity
from runtime_resources import code_path

SCHEMA='cosluma_face_contour_mapping.v1'
PARAMETERS={'face_grid_pitch':1.,'support_erosion':float(np.sqrt(2)+1),
    'minimum_component_gap':1.,'inward_gap_fraction':.6,'face_width_cap_fraction':.03,
    'outward_gap_cap_fraction':.75,'chin_tanh_face_fraction':.2,
    'invalid_row_transition_face_fraction':.015,'interpolation':'bilinear_display_linear_RGB_and_G',
    'zero_policy':'exact_original_copy_without_interpolation'}
ARRAYS=('displacement','protected','allowed','edit_support')
_VERIFIED=set()


def _cancel(cancel):
    if cancel is not None and cancel():raise InterruptedError('Cancelled face contour deformation')


def _amount(value):
    if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1:
        raise ValueError('Face contour amount must be a finite number from zero to one')
    return float(value)


def _size(value):
    if not isinstance(value,(list,tuple)) or len(value)!=2 or any(type(v) is not int or v<1 for v in value):
        raise ValueError('Face contour mapping needs its native integer canvas size')
    return list(value)


def producer_identity():
    """Current operation/runtime bytes, without probing detector installations."""
    runtime={}
    for package in ('numpy','scipy'):
        distribution=importlib.metadata.distribution(package)
        files=[file for file in distribution.files or () if str(file).endswith('.dist-info/METADATA')]
        if len(files)!=1:raise ValueError('Face contour runtime package metadata is ambiguous')
        path=Path(distribution.locate_file(files[0]))
        runtime[package]={'version':distribution.version,'metadata_sha256':sha256_file(path)}
    binaries={}
    for name in ('numpy._core._multiarray_umath','scipy.ndimage._nd_image','scipy.spatial._qhull'):
        module=importlib.import_module(name)
        binaries[name]=sha256_file(module.__file__)
    return {'code':{name:sha256_file(code_path(name)) for name in
        ('face_contour_deform.py','photo_shaping_geometry.py','portrait_regions_worker.py')},
        'runtime':runtime,'native_sampling_binaries':binaries,
        'numpy_version':np.__version__,'scipy_version':scipy.__version__}


def _smooth(value):
    value=np.clip(value,0,1)
    return value*value*(3-2*value)


def _intersections(chain,row):
    lengths=np.linalg.norm(np.diff(chain,axis=0),axis=1);total=float(lengths.sum());offset=0.;hits=[]
    for a,b,length in zip(chain,chain[1:],lengths):
        if abs(b[1]-a[1])<1e-9:
            if abs(row-a[1])<1e-9:return [],'ambiguous_horizontal_contour'
        elif a[1]<=row<b[1] or b[1]<=row<a[1]:
            fraction=(row-a[1])/(b[1]-a[1])
            hits.append((float(a[0]+fraction*(b[0]-a[0])),float((offset+fraction*length)/total)))
        offset+=float(length)
    return hits,None


def _jacobian(displacement,h,*,cancel=None):
    minimum=1.;maximum=1.;gradient=0.
    for start in range(0,displacement.shape[0]-1,128):
        _cancel(cancel)
        block=np.asarray(displacement[start:min(start+129,len(displacement))],np.float64)
        dx=np.diff(block,axis=1);dy=np.diff(block,axis=0)
        gx=np.maximum(np.abs(dx[:-1]),np.abs(dx[1:]));gy=np.maximum(np.abs(dy[:,:-1]),np.abs(dy[:,1:]))
        gradient=max(gradient,float(np.sqrt(gx*gx+gy*gy).max(initial=0)))
        for x in (dx[:-1],dx[1:]):
            for y in (dy[:,:-1],dy[:,1:]):
                determinant=1+h[0]*x+h[1]*y
                minimum=min(minimum,float(determinant.min(initial=1)));maximum=max(maximum,float(determinant.max(initial=1)))
    return {'jacobian_min':minimum,'jacobian_max':maximum,'maximum_gradient_norm':gradient}


def _seal(mapping):
    hashes={name:hashlib.sha256(np.ascontiguousarray(mapping[name]).tobytes()).hexdigest() for name in ARRAYS}
    content={key:value for key,value in mapping.items() if key not in {*ARRAYS,'mapping_id','array_sha256'}}
    mapping['array_sha256']=hashes;mapping['mapping_id']=identity({**content,'array_sha256':hashes})
    return mapping


def build_mapping(detections,canvas_size,amount,*,cancel=None):
    """Freeze a deterministic native-box scalar displacement along eye axis h."""
    import photo_shaping_geometry as geometry
    _cancel(cancel);canvas_size=_size(canvas_size);amount=_amount(amount)
    if not isinstance(detections,dict):raise ValueError('Canonical detections are required')
    planes,info=geometry._geometry(detections,canvas_size,cancel=cancel)
    points=np.asarray(detections['faces'][0]['landmarks'],np.float64)
    eye_a=(points[33]+points[133])*.5;eye_b=(points[362]+points[263])*.5
    h=eye_b-eye_a;h/=np.linalg.norm(h);vertical=np.array([-h[1],h[0]])
    if (points[152]-(eye_a+eye_b)*.5)@vertical<0:vertical*=-1
    origin=points[152];matrix=np.stack((h,vertical),axis=1)
    left,top,right,bottom=info['box'];shape=(bottom-top,right-left)
    protected=planes['P'].copy();allowed=(planes['J']>0)
    reasons=Counter();counts=Counter();row_min=analytic_min=1.;max_row_change=0.;max_control=0.
    displacement=np.zeros(shape,np.float32)
    if amount:
        corners=np.asarray([[left,top],[right,top],[right,bottom],[left,bottom]],np.float64)
        projected=(corners-origin)@matrix;pitch=PARAMETERS['face_grid_pitch']
        q0,v0=np.floor(projected.min(axis=0)/pitch)*pitch-2*pitch
        q1,v1=np.ceil(projected.max(axis=0)/pitch)*pitch+2*pitch
        qs=np.arange(q0,q1+pitch*.5,pitch);vs=np.arange(v0,v1+pitch*.5,pitch)
        safe=np.zeros((len(vs),len(qs)),bool);distance=distance_transform_edt(allowed)
        for row in range(0,len(vs),64):
            _cancel(cancel);stop=min(row+64,len(vs));q=np.broadcast_to(qs,(stop-row,len(qs)));v=np.broadcast_to(vs[row:stop,None],q.shape)
            x=origin[0]+h[0]*q+vertical[0]*v-left;y=origin[1]+h[1]*q+vertical[1]*v-top
            safe[row:stop]=map_coordinates(distance,[y,x],order=1,prefilter=False,mode='constant',cval=0.)>PARAMETERS['support_erosion']
        chain=(points[info['canonical_lower_indices']]-origin)@matrix;controls=[{} for _ in vs]
        for row,v in enumerate(vs):
            if row%64==0:_cancel(cancel)
            hits,error=_intersections(chain,float(v))
            if error:reasons[error]+=1;continue
            if not hits:reasons['no_contour_crossing']+=1;continue
            edges=np.flatnonzero(np.diff(np.pad(safe[row].astype(np.int8),(1,1))))
            components=[(qs[a],qs[b-1]) for a,b in zip(edges[::2],edges[1::2]) if b-a>=2];assigned={}
            for contour,position in hits:
                matches=[index for index,(a,b) in enumerate(components) if a<contour<b]
                if len(matches)!=1:reasons['contour_not_in_one_safe_J_component']+=1;continue
                assigned.setdefault(matches[0],[]).append((contour,position))
            for component,values in assigned.items():
                if len(values)!=1:reasons['multiple_contours_in_component']+=1;continue
                contour,position=values[0];a,b=components[component]
                if abs(contour)<1e-8:reasons['chin_midline_fixed']+=1;continue
                side=1 if contour>0 else -1
                inner,outer=(a,b) if side>0 else (-b,-a);canonical=abs(contour)
                inward=canonical-inner;outward=outer-canonical
                if min(inward,outward)<PARAMETERS['minimum_component_gap']:reasons['insufficient_component_gap']+=1;continue
                if side in controls[row]:controls[row][side]=None;reasons['ambiguous_repeated_side']+=1;continue
                taper=float(_smooth(min(position,1-position)/geometry.PARAMETERS['arc_endpoint_taper_fraction']))
                lateral=abs(float(np.tanh(contour/(PARAMETERS['chin_tanh_face_fraction']*info['face_width']))))
                controls[row][side]=(inner,outer,canonical,inward,outward,taper,lateral);counts['unambiguous_safe_side_rows']+=1
        tapers={}
        for side in (-1,1):
            valid=np.array([isinstance(row.get(side),tuple) for row in controls])
            separation=distance_transform_edt(np.pad(valid,(1,1)))[1:-1]*pitch
            tapers[side]=_smooth(np.maximum(separation-pitch,0)/(PARAMETERS['invalid_row_transition_face_fraction']*info['face_width']))
        face=np.zeros((len(vs),len(qs)),np.float32)
        for row,control in enumerate(controls):
            if row%64==0:_cancel(cancel)
            for side,value in control.items():
                if not isinstance(value,tuple):continue
                inner,outer,c,inward,outward,taper,lateral=value
                delta=min(amount*PARAMETERS['inward_gap_fraction']*inward,
                    amount*PARAMETERS['face_width_cap_fraction']*info['face_width'],PARAMETERS['outward_gap_cap_fraction']*outward)
                delta*=taper*lateral*tapers[side][row]
                if delta<=0:continue
                target=c-delta
                if not inner<target<c<outer:raise ValueError('Face contour controls would reverse their order')
                analytic_min=min(analytic_min,1-1.5*delta/(outer-target));max_control=max(max_control,delta)
                u=side*qs;selected=(u>=inner)&(u<=outer)
                values=np.where(u<=target,delta*_smooth((u-inner)/(target-inner)),delta*_smooth((outer-u)/(outer-target)))
                face[row,selected]+=side*values[selected]
        row_min=float(np.diff(qs[None,:]+face,axis=1).min())/pitch
        max_row_change=float(np.abs(np.diff(face,axis=0)).max(initial=0))
        if row_min<=0 or analytic_min<=0:raise ValueError('Face contour scanline map is not strictly monotone')
        for row in range(0,shape[0],96):
            _cancel(cancel);stop=min(row+96,shape[0]);yy,xx=np.indices((stop-row,shape[1]),dtype=np.float64);yy+=row
            x=xx+left-origin[0];y=yy+top-origin[1]
            q=(h[0]*x+h[1]*y-q0)/pitch;v=(vertical[0]*x+vertical[1]*y-v0)/pitch
            displacement[row:stop]=map_coordinates(face,[v,q],order=1,prefilter=False,mode='constant',cval=0.)
    stats={**_jacobian(displacement,h,cancel=cancel),'face_grid_minimum_q_derivative':row_min,
        'analytic_scanline_minimum_q_derivative':analytic_min,'maximum_adjacent_face_row_change':max_row_change,
        'maximum_control_delta':max_control,'maximum_displacement':float(np.abs(displacement).max(initial=0)),
        'modified_native_pixels':int(np.count_nonzero(displacement)),
        'invalid_scanline_reasons':dict(reasons),'control_counts':dict(counts),
        'whole_current_P_exact':True,'J0_exact':True,'geometric_acceptance_only':True,'artistic_acceptance':False}
    mapping={'schema':SCHEMA,'canvas_size':canvas_size,'amount':amount,'box':info['box'],'h':h.tolist(),
        'canonical_landmarks':points.tolist(),'source_detections_id':identity(detections),
        'parameters':copy.deepcopy(PARAMETERS),'producer':producer_identity(),'stats':stats,
        'displacement':displacement,'protected':protected,'allowed':allowed,'edit_support':displacement!=0}
    _seal(mapping);_validate_structure(mapping,cancel=cancel);_VERIFIED.add(mapping['mapping_id'])
    if len(_VERIFIED)>16:_VERIFIED.pop()
    return mapping


def _validate_structure(mapping,*,cancel=None):
    _cancel(cancel)
    expected={'schema','canvas_size','amount','box','h','canonical_landmarks','source_detections_id','parameters',
        'producer','stats','array_sha256','mapping_id',*ARRAYS}
    if not isinstance(mapping,dict) or set(mapping)!=expected or mapping['schema']!=SCHEMA:
        raise ValueError('Complete current face contour mapping required')
    if (not isinstance(mapping['stats'],dict) or not isinstance(mapping['source_detections_id'],str)
            or len(mapping['source_detections_id'])!=64
            or any(value not in '0123456789abcdef' for value in mapping['source_detections_id'])):
        raise ValueError('Face contour mapping needs its original detection identity and statistics')
    size=_size(mapping['canvas_size']);amount=_amount(mapping['amount']);box=mapping['box']
    if (not isinstance(box,list) or len(box)!=4 or any(type(v) is not int for v in box)
            or not 0<=box[0]<box[2]<=size[0] or not 0<=box[1]<box[3]<=size[1]):
        raise ValueError('Face contour box lies outside its actual native canvas')
    h=np.asarray(mapping['h'],np.float64)
    if h.shape!=(2,) or not np.isfinite(h).all() or abs(float(np.linalg.norm(h))-1)>1e-10:
        raise ValueError('Face contour direction must be a finite unit eye axis')
    shape=(box[3]-box[1],box[2]-box[0]);d=mapping['displacement'];p=mapping['protected'];a=mapping['allowed'];e=mapping['edit_support']
    if any(not isinstance(value,np.ndarray) or value.shape!=shape for value in (d,p,a,e)):
        raise ValueError('Face contour mapping arrays do not match their native box')
    if d.dtype!=np.float32 or p.dtype!=bool or a.dtype!=bool or e.dtype!=bool or not np.isfinite(d).all():
        raise ValueError('Face contour mapping precision or finite fields changed')
    if np.any(d[p]!=0) or np.any(d[~a]!=0) or not np.array_equal(e,d!=0) or np.any(p&a):
        raise ValueError('Face contour mapping violates complete P or J0 protection')
    if np.any(d[[0,-1]]!=0) or np.any(d[:,[0,-1]]!=0):raise ValueError('Face contour boundary displacement must be zero')
    if amount==0 and np.any(d) or amount>0 and not np.any(d):raise ValueError('Face contour request has no valid effective motion')
    for row in range(0,len(d),128):
        _cancel(cancel);end=min(row+128,len(d));yy,xx=np.indices((end-row,shape[1]),dtype=np.float64);yy+=row
        sx=xx+box[0]+h[0]*d[row:end];sy=yy+box[1]+h[1]*d[row:end]
        if sx.min()<0 or sy.min()<0 or sx.max()>size[0]-1 or sy.max()>size[1]-1:
            raise ValueError('Face contour sampling leaves the actual native canvas')
    actual=_jacobian(d,h,cancel=cancel)
    if actual['jacobian_min']<=0 or any(mapping['stats'].get(key)!=value for key,value in actual.items()):
        raise ValueError('Face contour map folds or its checked Jacobian evidence changed')
    if mapping['producer']!=producer_identity() or mapping['parameters']!=PARAMETERS:
        raise ValueError('Face contour producer or fixed strategy changed')
    check=dict(mapping);_seal(check)
    if check['mapping_id']!=mapping['mapping_id'] or check['array_sha256']!=mapping['array_sha256']:
        raise ValueError('Face contour mapping content identity changed')
    return mapping


def verify_mapping(mapping,detections,canvas_size,amount,*,cancel=None):
    """Rebuild from owned detections; self-consistent rewritten fields fail."""
    _validate_structure(mapping,cancel=cancel)
    expected=build_mapping(detections,canvas_size,amount,cancel=cancel)
    if expected['mapping_id']!=mapping['mapping_id'] or any(not np.array_equal(expected[key],mapping[key]) for key in ARRAYS):
        raise ValueError('Face contour mapping does not derive from its checked canonical detections')
    return expected


def _array(value,shape,name,*,cancel=None):
    if not isinstance(value,np.ndarray) or value.dtype!=np.float32 or value.shape!=shape:
        raise ValueError(name+' must retain its actual float32 native dimensions')
    for row in range(0,len(value),128):
        _cancel(cancel);part=value[row:row+128]
        if not np.isfinite(part).all() or np.any(part<0) or np.any(part>1):raise ValueError(name+' must be finite normalized display-linear values')


def sample_mapping(master,G,mapping,*,cancel=None):
    """Return complete float RGB/G with identical inverse-sampling coordinates."""
    _validate_structure(mapping,cancel=cancel)
    if mapping['mapping_id'] not in _VERIFIED:
        # An independent typed verifier supplies the full original detections.
        # For ad-hoc restored IO, compare the deterministic geometry as well;
        # the historical full-detection identity remains metadata, never reset.
        expected=build_mapping({'faces':[{'landmarks':mapping['canonical_landmarks']}]},mapping['canvas_size'],mapping['amount'],cancel=cancel)
        if (any(expected[key]!=mapping[key] for key in
                ('canvas_size','amount','box','h','canonical_landmarks','parameters','producer','stats'))
                or any(not np.array_equal(expected[key],mapping[key]) for key in ARRAYS)):
            raise ValueError('Unverified face contour map differs from its canonical geometry')
    width,height=mapping['canvas_size'];_array(master,(height,width,3),'RGB master',cancel=cancel)
    if not (type(G) is int and G in (0,1)):_array(G,(height,width),'Generated coverage',cancel=cancel)
    result=np.array(master,copy=True);transported=G if type(G) is int else np.array(G,copy=True)
    if mapping['amount']==0:_cancel(cancel);return result,transported
    left,top,right,bottom=mapping['box'];d=mapping['displacement'];h=np.asarray(mapping['h'],np.float64)
    changed=d!=0;padding=int(math.ceil(float(np.abs(d).max())))+2
    x0=max(0,left-padding);y0=max(0,top-padding);x1=min(width,right+padding);y1=min(height,bottom+padding)
    actual_change=False
    for row in range(0,len(d),96):
        _cancel(cancel);end=min(row+96,len(d));yy,xx=np.indices((end-row,right-left),dtype=np.float64);yy+=row
        sy=yy+top+h[1]*d[row:end];sx=xx+left+h[0]*d[row:end];coordinates=[sy-y0,sx-x0]
        active=changed[row:end];target=result[top+row:top+end,left:right]
        for channel in range(3):
            sampled=map_coordinates(np.asarray(master[y0:y1,x0:x1,channel]),coordinates,order=1,prefilter=False,mode='nearest')
            target[...,channel][active]=sampled[active]
        if not np.array_equal(target[active],master[top+row:top+end,left:right][active]):actual_change=True
        if type(G) is not int:
            sampled=map_coordinates(np.asarray(G[y0:y1,x0:x1]),coordinates,order=1,prefilter=False,mode='nearest')
            transported[top+row:top+end,left:right][active]=sampled[active]
    _cancel(cancel)
    if not actual_change:raise ValueError('Face contour sampling made no actual RGB change')
    protected=mapping['protected']|~changed
    if not np.array_equal(result[top:bottom,left:right][protected],master[top:bottom,left:right][protected]):
        raise ValueError('Face contour sampling changed protected original RGB')
    if type(G) is not int and not np.array_equal(transported[top:bottom,left:right][protected],G[top:bottom,left:right][protected]):
        raise ValueError('Face contour sampling changed protected provenance coverage')
    return result,transported
