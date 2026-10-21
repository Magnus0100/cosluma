"""Checked bounded mesh-weight inverse sampling, without generated appearance.

The old field squeezed one narrowing per scanline into the thin admitted band, so
the 10 px jaw displacement had to be absorbed by a 28 px ramp: measured forward
strain reached 2.03 and the sampling Jacobian fell to 0.49 (a fold). This module
keeps the same contract (complete central protection exact, everything outside
the admitted envelope exact, strength recomputed only from U) but replaces the
per-pixel mapping with a mesh-weight deformation.

Field construction (selected method: bounded harmonic mesh weights)
--------------------------------------------------------------------
* Mesh: regular lattice in the canonical eye frame (q along the eye axis h, v
  across it) with cell size face_grid_pitch*mesh_pitch_fraction*face_width
  (about 6.7 native px on a 1118 px face, 88k nodes on the 662 master).
* Handles: for every lattice row that crosses the checked lower contour, the
  single mesh node nearest that contour carries the per-row narrowing delta
  (the same strength semantics as before: inward gap fraction, face-width cap
  and outward gap cap, tapered at the arc ends, near the chin midline and over
  the strain-derived reach of every handle gap).
* Anchors: every node inside the complete central protection P (dilated by one
  cell so the interpolated field cannot step at the protection border), every
  node outside the admitted envelope E and every lattice-border node are fixed
  at zero displacement.
* Free nodes minimise the Dirichlet energy (the harmonic term of bounded
  biharmonic weights, Jacobsen/Baran/Sorkine-Hornung, SIGGRAPH 2011). Selection
  rationale, measured on the real 662 master at amount .3: the biharmonic and
  screened (lambda .2/1.0) energies overshoot the handle value (raw peak
  12.3 px against a 10.0 px handle) and fold (Jacobian min 0.18 to 0.51), while
  the harmonic energy is bounded by its boundary data (maximum principle: no
  overshoot, no fold), which is why the box-constrained weights are used
  without the biharmonic term. A mass-spring/ARAP energy would need a rest
  configuration and rotation fitting, and Green Coordinates need the boundary
  integral of the cage; both add machinery this scalar field does not need.
* Sampling: one inverse bilinear resample per pixel with cv2.remap
  (INTER_LINEAR, BORDER_REPLICATE); zero displacement keeps the exact original
  copy without interpolation.

Strain reporting (never one ambiguous number)
---------------------------------------------
The checked bound (PARAMETERS forward_strain_min/forward_strain_max, 0.8/1.25)
applies to the reciprocal area ratio 1/det(ds/dx) of the sampling Jacobian, and
the amplitude limiter enforces it together with the central-difference principal
stretch of the forward deformation gradient (ds/dx)^-1. stats carries, by name:
forward_strain_min/max (area ratio, the bound quantity), principal_stretch_min/
max (central difference), principal_stretch_envelope_min/max (conservative
one-sided over-estimate, audit only, not enforced), central_area_strain_min/max
(cross reference), the unlimited_* values the request would produce without the
limiter, strain_limited_fraction and delivered_fraction_of_request, plus a
strain_metric descriptor naming each quantity. natural_strain_bound_met is true
only when both enforced metrics are inside the window.

Row change reporting (three readings, one bound)
------------------------------------------------
maximum_adjacent_face_row_change is the gated quantity: the largest change of one
side's control series between adjacent mesh rows, divided by the mesh pitch. The
series is indexed by mesh row, so a handle that moves to a neighbouring lattice
column keeps its value in place. maximum_adjacent_face_column_change is the same
difference read inside one lattice column; it is audit only, because it also
moves when a handle shifts by a cell (measured 1.4891 and 1.4884 there against
this bound while the row series reads 0.2068 and 0.1554).
maximum_adjacent_field_row_change is the delivered displacement change between
adjacent native pixel rows (0.2583 and 0.2149 px), the quantity a viewer can see.
row_change_metric names all three. A control keeps its own row: when the nearest
node is unusable the handle moves to the nearest usable column instead of being
dropped.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import importlib
import importlib.metadata
import math
from pathlib import Path

import cv2
import numpy as np
import scipy
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt, label, map_coordinates
from scipy.sparse import diags, identity as sparse_identity, kron
from scipy.sparse.linalg import spsolve

from asset_identity import sha256_file
from photo_editing import identity
from runtime_resources import code_path

SCHEMA='cosluma_face_contour_mapping.v2'
PARAMETERS={'face_grid_pitch':1.,'mesh_pitch_fraction':.006,'support_erosion':float(np.sqrt(2)+1),
    'minimum_component_gap':1.,'inward_gap_fraction':.6,'face_width_cap_fraction':.03,
    'outward_gap_cap_fraction':.75,'ramp_peak_derivative':1.5,
    'forward_strain_min':.8,'forward_strain_max':1.25,
    'chin_tanh_face_fraction':.2,'invalid_row_transition_face_fraction':.015,
    'handle_reach_multiple':9.,'anchor_margin_cells':1,'mesh_relaxation_passes':2,'mesh_relaxation_weight':.5,'strain_limited_amplitude':True,
    'handle_node_search_cells':3,
    'weight_operator':'bounded_harmonic_mesh_weights',
    'interpolation':'cv2_remap_bilinear_display_linear_RGB_and_G',
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
    # The mesh solve and the cv2.remap sampler are new runtime dependencies, recorded
    # beside the legacy audited set. appearance_assets._current_verifier rebuilds
    # 'runtime' and 'native_sampling_binaries' from its own fixed audit list, so extra
    # names inside those two keys would break saved-asset verification; the new bytes
    # are bound as their own re-derived key instead.
    sampling={}
    for name in ('scipy.sparse._sparsetools','scipy.sparse.linalg._dsolve._superlu','cv2'):
        sampling[name]=sha256_file(importlib.import_module(name).__file__)
    return {'code':{name:sha256_file(code_path(name)) for name in
        ('face_contour_deform.py','photo_shaping_geometry.py','portrait_regions_worker.py')},
        'runtime':runtime,'native_sampling_binaries':binaries,
        'numpy_version':np.__version__,'scipy_version':scipy.__version__,
        'sampling_binaries':sampling,'opencv_version':cv2.__version__}


def _smooth(value):
    value=np.clip(value,0,1)
    return value*value*(3-2*value)


def _row_series_step(series):
    """Largest change between adjacent rows of one side's control series.

    The series is indexed by mesh row, so a handle that moves to a neighbouring
    lattice column keeps its value in place. Reading the same difference inside
    one lattice column instead would report that placement change as a step.
    """
    values=np.asarray(series,np.float64)
    if values.size<2:return 0.
    return float(np.abs(np.diff(values)).max(initial=0.))


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


def _principal_stretch(x,y,h):
    """Local principal stretches of the forward map for one gradient estimate.

    The sampler reads the source at s = x + h*d(x), so the sampling Jacobian is
    A = I + h (grad d)^T and the content (forward) deformation gradient is F =
    A^-1. The determinant of A is the area ratio used by the checked bound; the
    singular values of F are the physically meaningful local stretches. Both are
    returned because an area ratio alone cannot say how much any one direction
    stretches.
    """
    a=1.+h[0]*x;b=h[0]*y;c=h[1]*x;e=1.+h[1]*y
    area=a*e-b*c
    first=a*a+c*c;second=a*b+c*e;third=b*b+e*e
    trace=first+third;root=np.sqrt(np.maximum(trace*trace-4.*area*area,0.))
    sigma_max=np.sqrt(np.maximum((trace+root)/2.,1e-30));sigma_min=np.sqrt(np.maximum((trace-root)/2.,1e-30))
    return area,float((1./sigma_max).min(initial=1.)),float((1./sigma_min).max(initial=1.))


def _jacobian(displacement,h,*,cancel=None):
    """Measure the field under three named metrics, never one ambiguous number.

    determinant  : area ratio det(A) = 1 + h.grad(d); its reciprocal is what the
                   checked forward-strain bound and the amplitude limiter use.
    principal    : local principal stretches of F = A^-1 from central
                   differences, the O(h^2) estimate of the same field.
    envelope     : the maximum over the four one-sided difference combinations,
                   a deliberately conservative over-estimate kept for audit.
    """
    minimum=1.;maximum=1.;gradient=0.
    principal_min=1.;principal_max=1.;envelope_min=1.;envelope_max=1.
    central_min=1.;central_max=1.
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
                _area,low,high=_principal_stretch(x,y,h)
                envelope_min=min(envelope_min,low);envelope_max=max(envelope_max,high)
        cx=np.zeros_like(block);cy=np.zeros_like(block)
        if block.shape[1]>2:cx[:,1:-1]=(block[:,2:]-block[:,:-2])/2.
        if block.shape[0]>2:cy[1:-1]=(block[2:]-block[:-2])/2.
        cx=cx[1:-1,1:-1];cy=cy[1:-1,1:-1]
        determinant=1+h[0]*cx+h[1]*cy
        central_min=min(central_min,float(determinant.min(initial=1)));central_max=max(central_max,float(determinant.max(initial=1)))
        _area,low,high=_principal_stretch(cx,cy,h)
        principal_min=min(principal_min,low);principal_max=max(principal_max,high)
    return {'jacobian_min':minimum,'jacobian_max':maximum,'maximum_gradient_norm':gradient,
        'principal_stretch_min':principal_min,'principal_stretch_max':principal_max,
        'principal_stretch_envelope_min':envelope_min,'principal_stretch_envelope_max':envelope_max,
        'central_area_strain_min':central_min,'central_area_strain_max':central_max}


def _seal(mapping):
    hashes={name:hashlib.sha256(np.ascontiguousarray(mapping[name]).tobytes()).hexdigest() for name in ARRAYS}
    content={key:value for key,value in mapping.items() if key not in {*ARRAYS,'mapping_id','array_sha256'}}
    mapping['array_sha256']=hashes;mapping['mapping_id']=identity({**content,'array_sha256':hashes})
    return mapping


def _laplacian(rows,cols):
    vertical=diags([1.,-2.,1.],[-1,0,1],shape=(rows,rows)).tocsr()
    horizontal=diags([1.,-2.,1.],[-1,0,1],shape=(cols,cols)).tocsr()
    return (kron(vertical,sparse_identity(cols))+kron(sparse_identity(rows),horizontal)).tocsr()


def _solve_weights(free,handle_value,*,cancel=None):
    """Dirichlet minimiser of the mesh Dirichlet energy over the free nodes."""
    rows,cols=free.shape;A=_laplacian(rows,cols)
    _cancel(cancel)
    isolated=0
    if free.any():
        components,count=label(free)
        touching=set(np.unique(components[binary_dilation(~free,np.ones((3,3),bool))]))
        for index in range(1,count+1):
            if index not in touching:
                free=free&(components!=index);isolated+=1
    if not free.any():raise ValueError('Face contour mesh has no free node between its anchors and handles')
    values=np.asarray(handle_value,np.float64).ravel();mask=free.ravel()
    free_index=np.flatnonzero(mask);fixed_index=np.flatnonzero(~mask)
    solution=spsolve(A[free_index][:,free_index].tocsc(),(-A[free_index][:,fixed_index]@values[fixed_index]))
    if solution.shape!=(len(free_index),) or not np.isfinite(solution).all():
        raise ValueError('Face contour mesh weights could not be solved')
    values[free_index]=solution
    relaxed=values.reshape(rows,cols)
    passes=int(PARAMETERS['mesh_relaxation_passes']);weight=float(PARAMETERS['mesh_relaxation_weight'])
    for _ in range(passes):
        _cancel(cancel);padded=np.pad(relaxed,1,mode='edge')
        average=(padded[:-2,1:-1]+padded[2:,1:-1]+padded[1:-1,:-2]+padded[1:-1,2:])/4.
        relaxed=np.where(free,(1.-weight)*relaxed+weight*average,relaxed)
    return relaxed.astype(np.float32),{'mesh_free_nodes':int(len(free_index)),'mesh_relaxation_passes':passes,
        'mesh_anchor_nodes':int(len(fixed_index)),'mesh_isolated_components':isolated}


def _limited_fraction(measured,strain_min,strain_max):
    """Largest amplitude fraction that still fits the checked strain window.

    Two metrics are enforced so the reported bound cannot be read as a promise
    about a different quantity: the area ratio (det of the sampling Jacobian)
    and the central-difference principal stretch. The one-sided envelope is
    reported for audit only; it over-estimates the field and enforcing it would
    discard deliverable strength for a discretisation artefact.
    """
    fraction=1.
    for low,high in (('jacobian_min','jacobian_max'),('principal_stretch_min','principal_stretch_max')):
        if measured[high]>strain_max:
            fraction=min(fraction,(strain_max-1.)/max(measured[high]-1.,1e-12))
        if measured[low]<strain_min:
            fraction=min(fraction,(1.-strain_min)/max(1.-measured[low],1e-12))
    return float(max(min(fraction,1.),0.))


def build_mapping(detections,canvas_size,amount,*,cancel=None):
    """Freeze a deterministic bounded mesh-weight displacement along eye axis h."""
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
    protected=planes['P'].copy();envelope=planes['E'].copy();band=(planes['J']>0)
    # The edit support of this construction is the admitted envelope minus the
    # complete central protection: that is where the mesh may legally act at all.
    allowed=envelope&~protected
    reasons=Counter();counts=Counter();max_control=0.;max_handle_step=0.;row_step=0.
    mesh_info={'mesh_free_nodes':0,'mesh_anchor_nodes':0,'mesh_rows':0,'mesh_columns':0,'mesh_nodes':0,'mesh_handle_nodes':0,'mesh_pitch':0.,'mesh_relaxation_passes':0,'weight_operator':PARAMETERS['weight_operator'],'strain_headroom':0.}
    displacement=np.zeros(shape,np.float32)
    # Strain budget: a smoothstep ramp of width w moving delta has peak |d'| =
    # 1.5*delta/w, so a compressed ramp needs w >= 6*delta and a stretched ramp
    # w >= 7.5*delta for forward strain inside 0.8-1.25. The mesh solves the
    # whole admitted envelope instead of one thin band, and every handle gap is
    # emptied over the reach that budget needs; the measured field still decides
    # natural_strain_bound_met.
    strain_min=float(PARAMETERS['forward_strain_min']);strain_max=float(PARAMETERS['forward_strain_max'])
    if not 0<strain_min<1<strain_max:raise ValueError('Invalid face contour strain bound')
    peak=float(PARAMETERS['ramp_peak_derivative'])
    inner_ramp_ratio=peak*strain_min/(1-strain_min)
    outer_ramp_ratio=peak*strain_max/(strain_max-1.)
    target_delta=amount*float(PARAMETERS['face_width_cap_fraction'])*info['face_width']
    mesh_pitch=float(PARAMETERS['face_grid_pitch'])*float(PARAMETERS['mesh_pitch_fraction'])*info['face_width']
    if not mesh_pitch>0:raise ValueError('Face contour mesh pitch must be positive')
    if amount:
        corners=np.asarray([[left,top],[right,top],[right,bottom],[left,bottom]],np.float64)
        projected=(corners-origin)@matrix
        q0,v0=np.floor(projected.min(axis=0)/mesh_pitch)*mesh_pitch-2*mesh_pitch
        q1,v1=np.ceil(projected.max(axis=0)/mesh_pitch)*mesh_pitch+2*mesh_pitch
        qs=np.arange(q0,q1+mesh_pitch*.5,mesh_pitch);vs=np.arange(v0,v1+mesh_pitch*.5,mesh_pitch)
        rows,cols=len(vs),len(qs)
        mesh_info={'mesh_rows':rows,'mesh_columns':cols,'mesh_nodes':int(rows*cols),'mesh_pitch':float(mesh_pitch)}
        # 1. Band distance per node: only used to read the admitted contour gap per row.
        distance=distance_transform_edt(band);safe=np.zeros((rows,cols),bool)
        for start in range(0,rows,64):
            _cancel(cancel);stop=min(start+64,rows)
            q=np.broadcast_to(qs,(stop-start,cols));v=np.broadcast_to(vs[start:stop,None],q.shape)
            x=origin[0]+h[0]*q+vertical[0]*v-left;y=origin[1]+h[1]*q+vertical[1]*v-top
            safe[start:stop]=map_coordinates(distance,[y,x],order=1,prefilter=False,mode='constant',cval=0.)>PARAMETERS['support_erosion']
        # 2. Handles: one node per (row, side) at the checked lower contour.
        chain=(points[info['canonical_lower_indices']]-origin)@matrix;controls=[{} for _ in vs];raw={}
        for row,v in enumerate(vs):
            if row%64==0:_cancel(cancel)
            if row%8192==0 and row:counts['rows_checked']+=8192
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
                raw[(row,side)]=min(amount*PARAMETERS['inward_gap_fraction']*inward,
                    target_delta,PARAMETERS['outward_gap_cap_fraction']*outward)*lateral
        # 3. Every handle gap (a missing row, an arc end or the chin end of the
        #    covered contour) must empty over the reach the strain budget needs,
        #    otherwise the mesh has to absorb the whole narrowing across one cell.
        reach=max(float(PARAMETERS['handle_reach_multiple'])*(max(raw.values()) if raw else 0.),
            float(PARAMETERS['invalid_row_transition_face_fraction'])*info['face_width'])
        tapers={}
        for side in (-1,1):
            valid=np.array([isinstance(row.get(side),tuple) for row in controls])
            separation=distance_transform_edt(np.pad(valid,(1,1)))[1:-1]*mesh_pitch
            tapers[side]=_smooth(np.maximum(separation-mesh_pitch,0)/reach)
        # 4. Anchors (complete P dilated by the anchor margin, outside the
        #    envelope, lattice border) and the mesh handle nodes.
        q=np.broadcast_to(qs,(rows,cols));v=np.broadcast_to(vs[:,None],(rows,cols))
        x=origin[0]+h[0]*q+vertical[0]*v-left;y=origin[1]+h[1]*q+vertical[1]*v-top
        _cancel(cancel)
        guarded=map_coordinates(protected.astype(np.float32),[y,x],order=0,prefilter=False,mode='constant',cval=1.)>0.5
        inside=map_coordinates(envelope.astype(np.float32),[y,x],order=0,prefilter=False,mode='constant',cval=0.)>0.5
        gap_in=distance_transform_edt(~guarded)*mesh_pitch
        gap_out=distance_transform_edt(inside)*mesh_pitch
        free=inside&~guarded
        for _ in range(int(PARAMETERS['anchor_margin_cells'])):
            free=binary_erosion(free,np.ones((3,3),bool),border_value=0)
        free[0,:]=free[-1,:]=False;free[:,0]=free[:,-1]=False
        headroom=min(strain_max-1.,1.-strain_min)
        handles=np.zeros((rows,cols),bool);handle_value=np.zeros((rows,cols),np.float32)
        # One control per (row, side) lands on the nearest usable node of that
        # row. That node is not always usable: it can sit on the eroded safe
        # band's border or on an anchor cell. Dropping the control there left the
        # neighbouring row holding the whole narrowing, and that abrupt on/off is
        # exactly what an adjacent-row change reports (measured 9.9884 px between
        # adjacent lattice rows on the real 662 master, 15.5178 px on 685, i.e.
        # 1.4891 and 1.4884 per lattice row). The control therefore stays on its
        # own row and moves to the nearest usable column instead, which keeps the
        # row series continuous without changing the requested narrowing.
        series={-1:np.zeros(rows,np.float32),1:np.zeros(rows,np.float32)}
        search=int(PARAMETERS['handle_node_search_cells'])
        for row,control in enumerate(controls):
            for side,value in control.items():
                if not isinstance(value,tuple):continue
                inner,outer,c,inward,outward,taper,lateral=value
                delta=raw[(row,side)]*taper*tapers[side][row]
                if delta<=0:continue
                nearest=int(np.argmin(np.abs(qs-side*c)));column=None
                for offset in range(0,search+1):
                    for candidate in ((nearest,) if offset==0 else (nearest-offset,nearest+offset)):
                        if 0<=candidate<cols and safe[row,candidate] and free[row,candidate] and not handles[row,candidate]:
                            column=candidate;break
                    if column is not None:break
                if column is None:
                    reasons['handle_node_not_usable']+=1;continue
                handles[row,column]=True;handle_value[row,column]=side*delta
                series[side][row]=np.float32(side*delta);max_control=max(max_control,delta)
        counts['mesh_handle_nodes']=int(handles.sum())
        if not handles.any():raise ValueError('Face contour mesh has no usable handle')
        lattice,solved=_solve_weights(free&~handles,handle_value,cancel=cancel)
        mesh_info.update(solved)
        mesh_info.update({'mesh_handle_nodes':int(handles.sum()),'weight_operator':PARAMETERS['weight_operator'],
            'strain_headroom':float(headroom)})
        lattice=np.clip(lattice,-max_control,max_control).astype(np.float32)
        if rows>1:max_handle_step=float(np.abs(np.diff(handle_value,axis=0)).max(initial=0.))
        row_step=max(_row_series_step(series[-1]),_row_series_step(series[1]))
        # 5. Native bilinear resample of the mesh field, then the exact protection
        #    and envelope masks: a mesh cell straddling an anchor stays zero there.
        for start in range(0,shape[0],96):
            _cancel(cancel);stop=min(start+96,shape[0]);yy,xx=np.indices((stop-start,shape[1]),dtype=np.float64);yy+=start
            X=xx+left-origin[0];Y=yy+top-origin[1]
            qq=(h[0]*X+h[1]*Y-q0)/mesh_pitch;vv=(vertical[0]*X+vertical[1]*Y-v0)/mesh_pitch
            displacement[start:stop]=map_coordinates(lattice,[vv,qq],order=1,prefilter=False,mode='constant',cval=0.)
        displacement[protected]=0;displacement[~envelope]=0
        displacement[[0,-1],:]=0;displacement[:,[0,-1]]=0
    jacobian=_jacobian(displacement,h,cancel=cancel)
    unlimited=dict(jacobian)
    limited=1.
    if amount and PARAMETERS['strain_limited_amplitude']:
        # The mesh field is the L2-minimal-gradient field, but its peak gradient
        # still concentrates (measured 1.8x the naive ramp rate on the real 662
        # master), so a request can exceed what the checked window admits. The
        # delivered amplitude is the largest fraction that fits the window under
        # both the area ratio and the central-difference principal stretch; the
        # fraction is recorded, never hidden, and the window itself is untouched.
        for _ in range(4):
            fraction=_limited_fraction(jacobian,strain_min,strain_max)
            if fraction>=1.:break
            limited*=fraction
            displacement=(displacement*np.float32(fraction)).astype(np.float32)
            max_control*=fraction
            jacobian=_jacobian(displacement,h,cancel=cancel)
        limited=float(max(min(limited,1.),0.))
    field_row_change=float(np.abs(np.diff(displacement,axis=0)).max(initial=0.)) if shape[0]>1 else 0.
    stats={**jacobian,**mesh_info,'strain_limited_fraction':float(limited),
        'maximum_adjacent_face_row_change':float(row_step/mesh_pitch),
        'maximum_adjacent_face_column_change':float(max_handle_step/mesh_pitch),
        'maximum_adjacent_field_row_change':field_row_change,
        'row_change_metric':{'bound_key':'maximum_adjacent_face_row_change',
            'bound_quantity':'one side control change between adjacent mesh rows divided by the mesh pitch (px per px)',
            'per_column_key':'maximum_adjacent_face_column_change',
            'per_column_quantity':'the same difference read inside one lattice column; audit only, it also moves when a handle shifts by a cell',
            'field_key':'maximum_adjacent_field_row_change',
            'field_quantity':'delivered displacement change between adjacent native pixel rows (px per pixel row)'},
        'target_delta':float(target_delta if amount else 0.),
        'required_inner_ramp':float(inner_ramp_ratio*target_delta),'required_outer_ramp':float(outer_ramp_ratio*target_delta),
        'admitted_inner_band':float(info['inner_width']),'admitted_outer_band':float(info['outer_width']),
        'strain_bound':[strain_min,strain_max],
        'sampling_jacobian_bound':[1./strain_max,1./strain_min],
        'forward_strain_min':float(1./jacobian['jacobian_max']),
        'forward_strain_max':float(1./jacobian['jacobian_min']),
        'unlimited_forward_strain_min':float(1./unlimited['jacobian_max']),
        'unlimited_forward_strain_max':float(1./unlimited['jacobian_min']),
        'unlimited_principal_stretch_min':float(unlimited['principal_stretch_min']),
        'unlimited_principal_stretch_max':float(unlimited['principal_stretch_max']),
        'unlimited_principal_stretch_envelope_min':float(unlimited['principal_stretch_envelope_min']),
        'unlimited_principal_stretch_envelope_max':float(unlimited['principal_stretch_envelope_max']),
        'strain_metric':{'bound_keys':['forward_strain_min','forward_strain_max'],
            'bound_quantity':'area_ratio_reciprocal_1_over_det_of_sampling_jacobian',
            'principal_quantity':'principal_stretch_of_forward_deformation_gradient_inverse',
            'principal_discretisation':'central_difference',
            'envelope_quantity':'principal_stretch_over_four_one_sided_differences',
            'envelope_role':'conservative_audit_only_not_enforced'},
        'natural_strain_bound_met':bool(amount==0 or (
            jacobian['jacobian_min']>=1./strain_max-1e-6 and jacobian['jacobian_max']<=1./strain_min+1e-6
            and jacobian['principal_stretch_min']>=strain_min-1e-6
            and jacobian['principal_stretch_max']<=strain_max+1e-6)),
        'maximum_control_delta':max_control,'maximum_displacement':float(np.abs(displacement).max(initial=0)),
        'delivered_fraction_of_request':float(max_control/target_delta) if target_delta>0 else 0.,
        'modified_native_pixels':int(np.count_nonzero(displacement)),
        'invalid_scanline_reasons':dict(reasons),'control_counts':dict(counts),
        'whole_current_P_exact':True,'envelope_exact':True,'geometric_acceptance_only':True,'artistic_acceptance':False}
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
        raise ValueError('Face contour mapping violates the complete P or envelope-exact protection')
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
    strain_min=float(PARAMETERS['forward_strain_min']);strain_max=float(PARAMETERS['forward_strain_max'])
    if mapping['stats'].get('strain_bound')!=[strain_min,strain_max]:
        raise ValueError('Face contour map lost its checked forward-strain bound')
    # The verdict itself is derived in build_mapping from the very array that
    # line 382 above re-measures, so a forged field is already rejected there or
    # by the canonical-geometry comparison; no second, ordering-sensitive claim
    # check is needed here.
    if mapping['stats'].get('weight_operator')!=PARAMETERS['weight_operator']:
        raise ValueError('Face contour map lost its selected mesh weight operator')
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
    source=np.ascontiguousarray(master[y0:y1,x0:x1])
    source_G=None if type(G) is int else np.ascontiguousarray(np.asarray(G[y0:y1,x0:x1],np.float32))
    actual_change=False
    for row in range(0,bottom-top,96):
        _cancel(cancel);end=min(row+96,bottom-top);yy,xx=np.indices((end-row,right-left),dtype=np.float64);yy+=row
        map_y=(yy+top+h[1]*d[row:end]-y0).astype(np.float32);map_x=(xx+left+h[0]*d[row:end]-x0).astype(np.float32)
        active=changed[row:end];target=result[top+row:top+end,left:right]
        sampled=cv2.remap(source,map_x,map_y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        for channel in range(3):target[...,channel][active]=sampled[...,channel][active]
        if not np.array_equal(target[active],master[top+row:top+end,left:right][active]):actual_change=True
        if source_G is not None:
            sampled_G=cv2.remap(source_G,map_x,map_y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
            transported[top+row:top+end,left:right][active]=sampled_G[active]
    _cancel(cancel)
    if not actual_change:raise ValueError('Face contour sampling made no actual RGB change')
    protected=mapping['protected']|~changed
    if not np.array_equal(result[top:bottom,left:right][protected],master[top:bottom,left:right][protected]):
        raise ValueError('Face contour sampling changed protected original RGB')
    if type(G) is not int and not np.array_equal(transported[top:bottom,left:right][protected],G[top:bottom,left:right][protected]):
        raise ValueError('Face contour sampling changed protected provenance coverage')
    return result,transported
