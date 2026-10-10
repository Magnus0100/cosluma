"""Native source sole contours and local receiver attenuation; no RGB/reflection."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter, label, map_coordinates

RECIPE = 'native-source-shoe-owner-sole-receiver-falloff-v2'
MAX_ROI_PIXELS = 2_000_000


def _sample(values, box, canvas, order=1):
    l,t,r,b=box
    x=(np.arange(l,r,dtype=np.float64)+.5)*values.shape[1]/canvas[0]-.5
    y=(np.arange(t,b,dtype=np.float64)+.5)*values.shape[0]/canvas[1]-.5
    yy,xx=np.broadcast_arrays(y[:,None],x[None,:])
    return map_coordinates(values,[yy,xx],order=order,mode='grid-constant',cval=0,prefilter=False)


def build_sole_contact_falloff(contact, footwear, alpha, receiver, *, source_semantics,
                               source_canvas_size, mapping_supported=False):
    """Native matte defines contours; cached shoe cores define separate owners.

    EDTs stay in one bounded shoe ROI. Receiver weights and foreground alpha
    are never baked into attenuation. The consumer uses Q once and final over
    consumes alpha. Source normal priors do not establish metric contact.
    """
    contact,footwear,receiver=[np.asarray(x,np.float32) for x in (contact,footwear,receiver)]
    alpha=np.asarray(alpha,np.float32);semantics=np.asarray(source_semantics)
    canvas=tuple(map(int,source_canvas_size))
    if len({x.shape for x in (contact,footwear,receiver)})!=1 or contact.ndim!=2:
        raise ValueError('Contact observations must share their source proxy lattice')
    if alpha.shape!=(canvas[1],canvas[0]) or semantics.ndim!=2:
        raise ValueError('Contact needs native source alpha and cached source semantics')
    if not all(np.isfinite(x).all() for x in (contact,footwear,receiver,alpha,semantics)):
        raise ValueError('Nonfinite contact observation')
    ys,xs=np.nonzero(np.isin(semantics,[9,18]))
    # Qualify semantic cores before choosing a box: reflected/background shoe
    # labels cannot stretch the source-owned ROI or qualify another person.
    if len(xs):
        ay=(ys+.5)*canvas[1]/semantics.shape[0]-.5
        ax=(xs+.5)*canvas[0]/semantics.shape[1]-.5
        fy=(ys+.5)*footwear.shape[0]/semantics.shape[0]-.5
        fx=(xs+.5)*footwear.shape[1]/semantics.shape[1]-.5
        inside=(map_coordinates(alpha,[ay,ax],order=1,mode='grid-constant',cval=0,prefilter=False)>=.5)&(
            map_coordinates(footwear,[fy,fx],order=0,mode='grid-constant',cval=0,prefilter=False)>.5)
        ys,xs=ys[inside],xs[inside]
    pitch=max(canvas[0]/footwear.shape[1],canvas[1]/footwear.shape[0],
              canvas[0]/semantics.shape[1],canvas[1]/semantics.shape[0])
    margin=int(np.ceil(8*pitch))
    if len(xs):
        box=(max(0,int(np.floor(xs.min()*canvas[0]/semantics.shape[1]))-margin),
             max(0,int(np.floor(ys.min()*canvas[1]/semantics.shape[0]))-margin),
             min(canvas[0],int(np.ceil((xs.max()+1)*canvas[0]/semantics.shape[1]))+margin),
             min(canvas[1],int(np.ceil((ys.max()+1)*canvas[1]/semantics.shape[0]))+margin))
    else:
        box=(0,0,1,1)
    area=(box[2]-box[0])*(box[3]-box[1])
    if not len(xs) or area>MAX_ROI_PIXELS:
        reason='source_owned_shoe_core_unavailable' if not len(xs) else 'native_contact_roi_budget_exceeded'
        return np.zeros((1,1),np.float32),np.zeros((1,1),np.int16),dict(
            recipe=RECIPE,native_box=[0,0,1,1],native_roi_pixels=1,
            observed_roi_pixels=area,max_native_roi_pixels=MAX_ROI_PIXELS,native_canvas_size=list(canvas),
            qualified_for_automatic_contact=False,qualified_source_components=[],
            unsupported_source_components=[9,18],components=[],state=reason,
            mapping_supported=bool(mapping_supported),receiver_support_baked_into_attenuation=False,
            foreground_coverage_baked_into_attenuation=False,receiver_support_consumed_once=True,
            foreground_coverage_changed=False,no_rgb_consumed=True,complete_cast_shadow=False,
            floor_reflection=False,metric_geometry=False,local_module_unavailable=True)
    l,t,r,b=box
    a=np.clip(alpha[t:b,l:r],0,1);f=_sample(footwear,box,canvas,0)>.5
    q=np.clip(_sample(receiver,box,canvas),0,1);p=_sample(contact,box,canvas)>.05
    s=_sample(semantics,box,canvas,0)
    domain=f&(a>=.5)
    silhouette=(a>=.5)&~binary_erosion(a>=.5,structure=np.ones((3,3),bool),border_value=0)
    fields=np.zeros(a.shape,np.float32);owners=np.zeros(a.shape,np.int16)
    seeds=[(s==shoe_id)&domain for shoe_id in (9,18)]
    distances=[distance_transform_edt(~seed) if seed.any() else np.full(a.shape,np.inf) for seed in seeds]
    owner=np.where(distances[0]<distances[1],9,18)
    owner[np.minimum(distances[0],distances[1])==np.inf]=0
    components,_=label(domain,structure=np.ones((3,3),bool))
    admitted=np.unique(components[seeds[0]|seeds[1]])
    domain&=np.isin(components,admitted[admitted>0])
    minimum=int(np.ceil(8*pitch*pitch));records=[]
    for shoe_id,seed in zip((9,18),seeds):
        shoe=domain&(owner==shoe_id)
        record=dict(source_shoe_owner=shoe_id,source_semantic_seed_pixels=int(seed.sum()),
                    source_owned_pixels=int(shoe.sum()),qualified=False)
        records.append(record)
        if not mapping_supported:
            record['reason']='candidate_receiver_mapping_unproven';continue
        if seed.sum()<minimum or not shoe.any():
            record['reason']='source_shoe_semantic_core_insufficient';continue
        distance,nearest=distance_transform_edt(~shoe,return_indices=True)
        proposal=p&(q>.1)&(a<.5)&(owner==shoe_id)
        record['joint_support_seed_pixels']=int(proposal.sum())
        if proposal.sum()<minimum:
            record['reason']='observed_joint_receiver_contact_insufficient';continue
        sy,sx=np.nonzero(shoe);py,px=np.nonzero(proposal)
        centre=np.array([sx.mean(),sy.mean()]);direction=np.array([px.mean(),py.mean()])-centre
        length=float(np.linalg.norm(direction))
        if length<=pitch:
            record['reason']='observed_support_direction_unavailable';continue
        direction/=length;tangent=np.array([direction[1],-direction[0]])
        points=np.stack([sx,sy],-1)-centre;u=points@tangent;v=points@direction
        start=int(np.floor(u.min()));end=int(np.ceil(u.max()))
        frontier=np.full(end-start+1,-np.inf)
        np.maximum.at(frontier,np.clip(np.rint(u).astype(int)-start,0,len(frontier)-1),v)
        known=np.isfinite(frontier)
        frontier=np.interp(np.arange(len(frontier)),np.flatnonzero(known),frontier[known])
        anchors=np.zeros(a.shape,np.float32)
        anchors[nearest[0,proposal],nearest[1,proposal]]=1;anchors*=silhouette
        ay,ax=np.nonzero(anchors);ap=np.stack([ax,ay],-1)-centre
        av=ap@direction;au=ap@tangent;extent=float(np.quantile(distance[proposal],.95))
        eligible=(av>=np.interp(au,np.arange(start,end+1),frontier)-1.5)&(av>=v.max()-extent)
        anchors[ay[~eligible],ax[~eligible]]=0
        if np.count_nonzero(anchors)<max(4,int(pitch)) or extent<=0:
            record['reason']='native_floor_facing_sole_contour_insufficient';continue
        weight=gaussian_filter(anchors,max(.5,pitch/2),mode='constant',cval=0)
        weight/=max(float(weight.max()),1e-8);weight=weight[nearest[0],nearest[1]]
        anchor_height=(nearest[1]-centre[0])*direction[0]+(nearest[0]-centre[1])*direction[1]
        effective=distance+np.maximum(v.max()-anchor_height,0)
        local=(q>.1)&(a<.5)&(owner==shoe_id)&(distance<=extent)
        falloff=np.square(np.maximum(1-np.square(effective/max(extent,1e-8)),0))*weight*local
        if np.count_nonzero(falloff)<minimum:
            record['reason']='native_sole_receiver_corridor_insufficient';continue
        fields=np.maximum(fields,falloff.astype(np.float32));owners[falloff>0]=shoe_id
        ay,ax=np.nonzero(anchors)
        record.update(qualified=True,reason=None,native_sole_anchor_pixels=int(len(ax)),
            native_anchor_bounds=[int(ax.min()+l),int(ay.min()+t),int(ax.max()+l),int(ay.max()+t)],
            observed_proposal_extent_native_pixels=extent,falloff_support_pixels=int(np.count_nonzero(falloff)),
            observed_support_direction_xy=direction.tolist())
    qualified=[x['source_shoe_owner'] for x in records if x['qualified']]
    return fields,owners,dict(recipe=RECIPE,native_box=list(box),native_roi_pixels=area,
        max_native_roi_pixels=MAX_ROI_PIXELS,native_canvas_size=list(canvas),
        qualified_for_automatic_contact=bool(qualified),qualified_source_components=qualified,
        unsupported_source_components=[x['source_shoe_owner'] for x in records if not x['qualified']],
        components=records,state='qualified_native_source_sole_receiver' if qualified else 'native_sole_receiver_unsupported',
        mapping_supported=bool(mapping_supported),semantic_sampling_pitch_native=pitch,
        source_contour='native_alpha_half_coverage_floor_facing_boundary',
        source_owner='cached_left_right_shoe_core_within_source_footwear_component',
        receiver_support_baked_into_attenuation=False,foreground_coverage_baked_into_attenuation=False,
        receiver_support_consumed_once=True,foreground_coverage_changed=False,no_rgb_consumed=True,
        complete_cast_shadow=False,floor_reflection=False,metric_geometry=False)


@lru_cache(maxsize=4)
def _roi_values(path,modified):
    with np.load(path,allow_pickle=False) as resource:
        return np.asarray(resource['values'],np.float32)


def sample_native_contact(asset,box,output_size):
    """Sample a bounded native ROI at full/preview/ROI pixel centers."""
    if asset.get('coordinate_space')!='source_canvas_roi' or not asset.get('native_box'):
        raise ValueError('Current native ROI contact required; legacy contact is read-only')
    path=Path(asset['path']);values=_roi_values(str(path),path.stat().st_mtime_ns)
    native=asset['original_canvas_size'];roi=asset['native_box'];l,t,r,b=box
    x=(np.arange(l,r,dtype=np.float64)+.5)*native[0]/output_size[0]-.5-roi[0]
    y=(np.arange(t,b,dtype=np.float64)+.5)*native[1]/output_size[1]-.5-roi[1]
    yy,xx=np.broadcast_arrays(y[:,None],x[None,:])
    return map_coordinates(values,[yy,xx],order=1,mode='grid-constant',cval=0,prefilter=False).astype(np.float32)


def receiver_attenuation(contact,receiver,contact_strength,authored=None,authored_strength=0.):
    """A*Q once; max with artist attenuation once; alpha belongs to final over."""
    result=np.clip(contact,0,1)*np.clip(receiver,0,1)*float(np.clip(contact_strength,0,1))
    if authored is not None:
        result=np.maximum(result,np.clip(authored,0,1)*float(np.clip(authored_strength,0,1)))
    return result.astype(np.float32)
