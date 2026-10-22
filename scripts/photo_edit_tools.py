"""Reserved coordinator tools editing the current best, never source RGB replay.

Session reservation precedes every provider call. Local edits return complete
images and retain original best pixels outside their authorized coverage.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageCms, ImageOps

from asset_identity import sha256_file
from current_model_registry import atomic_json
from photo_editing import identity, validate_scope, validate_observed_design

SCHEMA='cosluma_photo_edit_tool.v1'
OPTIONS={'provider','model','api_base','api_url','api_key_env','size','timeout'}


def _asset(path):
    path=Path(path).resolve(strict=True)
    return {'path':str(path),'sha256':sha256_file(path)}


def _cancel(cancel):
    if cancel and cancel():raise InterruptedError('Editing tool cancelled; inspect the existing reserved attempt')


def _best_rgb(path):
    """Keep oriented RGB8 values exactly; do not colour-convert untouched pixels."""
    import io
    with Image.open(path) as image:
        if image.mode!='RGB' or image.format not in {'PNG','JPEG'}:
            raise ValueError('Current best requires an explicit RGB8 PNG/JPEG sRGB canvas')
        profile=image.info.get('icc_profile')
        if profile and 'srgb' not in ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(io.BytesIO(profile))).lower():
            raise ValueError('Current best must be converted to sRGB explicitly before a local editing tool')
        result=ImageOps.exif_transpose(image).copy()
        result.load()
        return result


def _typed_inputs(context,state,scope,design):
    from photo_appearance import validate_asset
    from photo_session_evidence import resolve_file,resolve_asset
    images=context.get('images');assets=context.get('appearance_assets')
    if (not isinstance(images,list) or not isinstance(assets,dict)
            or [row.get('role') for row in images] not in (['source','best'],['source','best','working'])
            or set(assets)!={row['role'] for row in images}):
        raise ValueError('Current tools require actual typed appearance inputs for every role')
    source=context.get('source_original')
    if source!=resolve_file(state,state['binding']['source']) or _asset(source['path'])!=source:
        raise ValueError('Original source bytes differ from the checked session')
    binding={'source_sha256':source['sha256'],'scope':scope,'design_id':design['design_id']}
    loaded={}
    for row in images:
        if _asset(row['path'])!={key:row[key] for key in ('path','sha256')}:
            raise ValueError('Current appearance preview bytes changed')
        current={'receipt':validate_asset(assets[row['role']],**binding)}
        if current['receipt']['files']['preview']['sha256']!=row['sha256']:
            raise ValueError('Preview does not represent the bound float appearance master')
        loaded[row['role']]=current
    if (assets['source']!=resolve_asset(state,state.get('source_appearance'))
            or loaded['source']['receipt']['kind']!='source_baseline'
            or any(value['receipt']['source_origin']!=loaded['source']['receipt']['source_origin']
                for value in loaded.values())):
        raise ValueError('Tools require the actual original source appearance and matching ancestry')
    return source,loaded


def _close_loaded(*items):
    closed=set()
    for item in items:
        if item is None:continue
        for key in ('master','edit_coverage','generated_coverage'):
            mapping=getattr(item.get(key),'_mmap',None)
            if mapping is not None and id(mapping) not in closed:
                mapping.close();closed.add(id(mapping))


def _reservation(context,config,kind):
    """Require the real reserved decision, scope, current float canvas and cost."""
    import photo_edit_session as session
    from photo_session_evidence import resolve_file,resolve_asset
    scope=validate_scope(config['scope'])
    if context.get('scope')!=scope:raise ValueError('Tool scope differs from the checked session')
    design=context.get('design') or {}
    payload={key:design.get(key) for key in ('summary','observation','changes','lighting','keep','uncertain')}
    if validate_observed_design(payload,scope)!=design:raise ValueError('Tool observed design changed')
    images=context.get('images')
    if not isinstance(images,list) or [row.get('role') for row in images]!=['source','best']:
        raise ValueError('Image editing needs the actual baseline preview and current canvas')
    state=session._read(Path(config['session_directory']))
    shaping=config.get('shaping_authorization')
    from photo_shaping import operations as permitted_operations,shaping_operations,validate_authorization
    if context.get('shaping_authorization')!=shaping or state['binding'].get('shaping_authorization')!=shaping:
        raise ValueError('Tool shaping permission differs from its checked session')
    if shaping is not None:
        validate_authorization(shaping,source_sha256=context['source_original']['sha256'],scope=scope,
            design_id=design['design_id'],base_appearance=context['shaping_base_appearance'],
            source_appearance=context['appearance_assets']['source'])
        if kind!='edit_region' or context['appearance_assets']['best']['asset_id']!=shaping['base_appearance']['asset_id']:
            raise ValueError('Shaping must edit the actual unshaped input with the regional tool')
    source,loaded=_typed_inputs(context,state,scope,design)
    best=images[1];decision=context.get('decision') or {}
    if (decision.get('action')!='tool' or decision.get('tool')!=kind
            or not isinstance(decision.get('operations'),list) or not decision['operations']
            or set(decision['operations'])-set(permitted_operations(scope,shaping))
            or not isinstance(decision.get('instruction'),str) or not decision['instruction'].strip()):
        raise ValueError('Decision does not authorize this editing tool')
    if shaping is not None and set(decision['operations'])!=set(shaping_operations(shaping)):
        raise ValueError('Shaping cannot expand, omit or reclassify selected parts')
    attempt=next((row for row in state.get('attempts',[]) if row.get('attempt_id')==context.get('attempt_id')),None)
    costs=(attempt or {}).get('costs') or {}
    stage=context.get('operation_stage')
    current=next((row for row in state.get('candidates',[]) if row.get('candidate_id')==session.current_canvas_id(state)),None)
    if (state.get('session_id')!=context.get('session_id') or state.get('status')!='active' or state.get('phase')!='tool'
            or state.get('scope')!=scope or state.get('design')!=design
            or state.get('user_intent')!=context.get('user_intent')
            or stage not in {'initial','improvement'} or not attempt
            or attempt.get('operation_stage')!=stage or attempt.get('status')!='reserved' or attempt.get('kind')!='tool'
            or attempt.get('context_sha256')!=identity(context) or attempt.get('context')!=context
            or state.get('pending_decision')!=decision or costs.get('image_submissions')!=1 or costs.get('tool_calls')!=1
            or kind=='edit_region' and costs.get('local_repairs')!=1
            or not current or resolve_file(state,current.get('image'))!={key:best[key] for key in ('path','sha256')}
            or resolve_asset(state,current.get('appearance_asset'))!=context['appearance_assets']['best']
            or state['binding'].get('appearance_required') is not True
            or stage=='initial' and state['binding'].get('editing_initialization')!=session.INITIALIZATION):
        raise ValueError('Image submission requires actual typed inputs and its persisted reserved decision')
    expected=dict(state['binding']['initial_usage'])
    for charged in state['attempts']:
        for key,amount in charged['costs'].items():expected[key]+=amount
    if expected!=state['used'] or any(state['used'][key]>state['binding']['budget'][key] for key in state['used']):
        raise ValueError('Persisted session usage differs from the checked reservation')
    prior_decision=next((row for row in reversed(state['attempts']) if row.get('kind')=='decide'
        and row.get('status')=='completed'),None)
    actual=(prior_decision or {}).get('response') or {}
    decision_images=[{'role':row['role'],**resolve_file(state,row)} for row in ((prior_decision or {}).get('context') or {}).get('images',[])
        if row.get('role') in {'source','best'}]
    tool_images=[{'role':row['role'],**resolve_file(state,row)} for row in images]
    if ({key:value for key,value in actual.items() if key not in {'provenance','observed_images'}}!=decision
            or decision_images!=tool_images
            or (actual.get('provenance') or {}).get('model_requests')!=1
            or (actual.get('provenance') or {}).get('generation_used') is not False):
        raise ValueError('Reserved image request lacks its actual completed model decision')
    if stage=='initial' and loaded['best']['receipt']['kind']!='source_baseline':
        raise ValueError('Initial model editing must start with the actual source-native baseline')
    review=current.get('review')
    if stage=='improvement' and (not review or review not in (context.get('feedback') or [])
            or not any(row.get('role')=='candidate' and row.get('sha256')==best['sha256']
                for row in review.get('observed_images',[]))):
        raise ValueError('Improvement feedback must retain its actual current-canvas observation')
    return scope,design,source,best,decision,attempt,loaded['best']


def _region(value,size):
    if (not isinstance(value,(list,tuple)) or len(value)!=4
            or any(type(v) not in {int,float} or not math.isfinite(v) for v in value)):
        raise ValueError('Edit region needs four finite normalized coordinates')
    x0,y0,x1,y1=value
    if not 0<=x0<x1<=1 or not 0<=y0<y1<=1:
        raise ValueError('Repair region lies outside the current best canvas')
    w,h=size;core=[math.floor(x0*w),math.floor(y0*h),math.ceil(x1*w),math.ceil(y1*h)]
    if core[2]-core[0]<4 or core[3]-core[1]<4 or core==[0,0,w,h]:
        raise ValueError('Observed local repair requires a nontrivial bounded region')
    pad=max(16,math.ceil(max(core[2]-core[0],core[3]-core[1])*.2))
    context=[max(0,core[0]-pad),max(0,core[1]-pad),min(w,core[2]+pad),min(h,core[3]+pad)]
    return core,context


def _fit_context(core,box,canvas,aspect):
    """Expand context to a fixed generator canvas without expanding edit rights."""
    width,height=canvas
    wanted_w=max(box[2]-box[0],math.ceil((box[3]-box[1])*aspect))
    wanted_h=max(box[3]-box[1],math.ceil(wanted_w/aspect))
    wanted_w=max(wanted_w,math.ceil(wanted_h*aspect))
    if wanted_w>width or wanted_h>height:
        raise ValueError('Observed region cannot fit this provider aspect without inventing context')
    x0=min(max(0,math.floor((core[0]+core[2]-wanted_w)/2)),width-wanted_w)
    y0=min(max(0,math.floor((core[1]+core[3]-wanted_h)/2)),height-wanted_h)
    return [x0,y0,x0+wanted_w,y0+wanted_h]


def _provider_aspect(resolved):
    """Only use an explicitly fixed latent canvas, never guess from model names."""
    if resolved.get('kind')!='comfyui':return None
    from comfyui_adapter import load_workflow
    document=load_workflow(Path(resolved['workflow_path']))
    graph=document.get('workflow',document)
    sizes={(node.get('inputs',{}).get('width'),node.get('inputs',{}).get('height'))
           for node in graph.values() if node.get('class_type') in
           {'EmptySD3LatentImage','EmptyLatentImage','EmptyFlux2LatentImage'}}
    explicit=[(w,h) for w,h in sizes if type(w) is int and type(h) is int and w>0 and h>0]
    if len(explicit)>1:raise ValueError('Ambiguous fixed image-generation canvas')
    return explicit[0][0]/explicit[0][1] if explicit else None


FUSION={'schema':'cosluma_photographic_join.v3','minimum_region_short_edge':256,
    'minimum_region_pixels':131072,'maximum_taper_px':24.,'minimum_taper_px':2.,
    # Restored to the reviewed 2px floor: the level is carried by the
    # registration, so the taper only has to cover the content transition and
    # goes back to 2px where the retained photograph has real contrast there,
    # widening towards 24px only where the scene is flat (what a matte would
    # supply). Measured on the three real products, widening this floor from 2
    # to 6 px moved 672's halo from 18 px to 13 px and 662's tonal excess from
    # -0.0085 to -0.0086, i.e. it bought nothing and it was this round's own
    # regression, so the floor is back where it was.
    'structure_sigma':2.,'structure_reference_quantile':.95,
    # Region-scale registration: the join used to be right only inside an
    # 8-36px membrane, so the region core kept whatever exposure and colour
    # temperature the provider chose (measured up to 3.7x and 231 mired away
    # from the photograph it replaced). One robust per-channel gain now takes
    # the provider's low frequency level and white point onto the photographed
    # level of the very core being replaced: the provider supplies appearance,
    # not a new exposure. The retained ring is measured as well and recorded,
    # so the receipt shows both sides of the cut.
    'registration_sigma_px':18.,'registration_ring_px':96.,
    # Constant over the whole authorized core, never a function of the taper.
    'registration_field':'constant_over_authorized_core',
    # Unchanged from the reviewed state: measured open (see the receipt's
    # clamped flag) 685 needs about 8x and clamps at -3 EV, which is accounted
    # for rather than papered over by widening the bound.
    'registration_gain_ev_bounds':(-3.,3.),
    'tone_depth_fraction':.012,
    'tone_depth_bounds':(8,36),'tone_bound':.35,'detail_sigma':1.6,'detail_fine_sigma':.6,
    'detail_window_px':64.,'noise_ratio_bounds':(.4,4.),
    # A provider patch can carry a third of the photograph's own detail energy.
    # The bound has to reach that far or the region stays visibly smoother than
    # the scene around it; the measured gains are recorded either way.
    'band_inner_px':8,'band_interior_px':64,
    # The provider's own residue is rescaled, never synthesised. Measured on
    # the real 662 patch (about 1/6 of the photograph's own detail energy) the
    # bound has to reach that far or the region stays visibly smoother than the
    # scene around it, and the join band then reads as a blur. Both gains are
    # recorded with their measured achievement, so a clamped case is visible.
    'detail_ratio_bounds':(.2,6.),
    # The join band is judged against the parent's own band, not against a
    # constant: content that is genuinely smoother than the ring must not force
    # texture to be invented.
    'join_detail_relative_limit':1.3,
    # One constant gain carries the region's mean level and white point onto
    # the photograph's, but the photograph's own level is not constant inside
    # the region. Measured on the real 685 product the pixels just inside the
    # cut sat +0.066 luma above the retained photograph, which is the seam the
    # audit reads (1.79 against its 1.6 limit) and the reason the membrane
    # saturated at its 0.35 bound. The relation is therefore solved jointly
    # across the boundary: the registration holds in the interior and the gain
    # moves onto the parent's own local relation inside the measured join band,
    # in both directions, so a darker environment can darken the region instead
    # of only ever brightening it. It is a per-channel exposure/white-point
    # gain, not a tone transplant, and it stays inside the reviewed EV bound.
    'boundary_light_field':'joint_region_band_subject_environment_two_way_gain'}
_LUMA=np.asarray([.2126,.7152,.0722],np.float32)


def _mask(size):
    w,h=size;feather=max(1,min(64,math.ceil(min(w,h)*.08)))
    x=np.minimum(np.arange(w),np.arange(w)[::-1])/feather
    y=np.minimum(np.arange(h),np.arange(h)[::-1])/feather
    value=np.clip(np.minimum(y[:,None],x[None,:]),0,1).astype(np.float32)
    return value*value*(3-2*value)


def _join_analysis_ready(size):
    """A photographic join needs enough native support to estimate its own field."""
    width,height=size
    return (min(width,height)>=FUSION['minimum_region_short_edge']
        and width*height>=FUSION['minimum_region_pixels'])


def _edge_distance(size):
    """Distance in pixels to the nearest edge of the authorized write."""
    w,h=size
    x=np.minimum(np.arange(w,dtype=np.float32),np.arange(w,dtype=np.float32)[::-1])+1
    y=np.minimum(np.arange(h,dtype=np.float32),np.arange(h,dtype=np.float32)[::-1])+1
    return np.minimum(x[None,:],y[:,None]).astype(np.float32)


def _structure_taper(prior):
    """Per-pixel join width: tight across measured structure, wider where it is flat.

    The former composite used one fixed short-edge fraction for every pixel, so a
    53 px straight ramp cut through hair, arms and props and cross-faded two
    contents. The taper now follows the retained content's own structure, which
    is what a matte would otherwise supply; registered provider alpha, when the
    provider returns it, still owns the join and is applied exactly.
    """
    from scipy.ndimage import gaussian_filter
    grey=np.asarray(prior,np.float32)@_LUMA
    smooth=gaussian_filter(grey,float(FUSION['structure_sigma']),mode='nearest')
    vertical,horizontal=np.gradient(smooth)
    strength=np.hypot(horizontal,vertical)
    reference=float(np.quantile(strength,float(FUSION['structure_reference_quantile'])))
    normalised=np.clip(strength/max(reference,1e-6),0.,1.)
    wide,narrow=float(FUSION['maximum_taper_px']),float(FUSION['minimum_taper_px'])
    return (wide+(narrow-wide)*normalised).astype(np.float32)


def _join_ramp(distance,taper):
    """Taper-only source weight of the join, straight from the core geometry."""
    value=np.clip(distance/np.maximum(taper,np.float32(1.)),0.,1.)
    return (value*value*(3.-2.*value)).astype(np.float32)


def _ownership_distance(ownership,size):
    """Distance in pixels to the boundary this join actually writes, whichever side owns it.

    The audit measures its seam, its halo and its join band against the
    delivered boundary, so a light solve has to be measured from the same
    place: where ownership hands the pixel back to the parent. The reserved
    rectangle stays the fallback geometry when there is no ownership to follow,
    and an ownership that covers (or excludes) the whole core has no boundary
    inside it at all, so the core geometry stands.
    """
    from scipy.ndimage import distance_transform_edt
    if ownership is None:return _edge_distance(size)
    owned=np.asarray(ownership,np.float32)>=np.float32(.5)
    if not owned.any() or owned.all():return _edge_distance(size)
    return np.ascontiguousarray(distance_transform_edt(~owned),dtype=np.float32)


def _support_low(values,support,sigma):
    """The low band of a field measured only where the field actually exists.

    An unpremultiplied provider patch is exactly zero where nothing is written,
    so smoothing it raw drags the band near the cut towards those zeros and
    asks for a gain the provider never needed. Normalising by the support keeps
    the measurement on the pixels that carry content.
    """
    from scipy.ndimage import gaussian_filter
    weight=np.asarray(support,np.float32)
    if not bool(weight.min()>np.float32(0.)):
        numerator=gaussian_filter(np.asarray(values,np.float32)*weight[...,None],
            (sigma,sigma,0),mode='nearest')
        denominator=gaussian_filter(weight,(sigma,sigma),mode='nearest')
        return numerator/np.maximum(denominator,np.float32(1e-6))[...,None]
    return gaussian_filter(np.asarray(values,np.float32),(sigma,sigma,0),mode='nearest')


def _boundary_light_gain(prior,patch,ownership,support_field,distance,scale):
    """Per-channel low-frequency gain solved jointly across the join boundary.

    A constant registration matches the region's mean level and white point onto
    the photographed ones, but the photograph's own level is not constant
    inside the region, and the retained scene around the cut owns the local
    light field. The solve therefore keeps the registration where the join has
    its own interior and moves the gain onto the parent's own local relation
    inside the measured join band, blending between them in log2 exposure so
    the field is continuous across the boundary instead of stepping on it.
    Both directions are solved: an environment darker than the source darkens
    the written pixels exactly as a brighter one lifts them, which the reviewed
    one-way response could not express. The support, the band, the interior and
    the EV bound are the reviewed join constants; nothing is widened here, and
    the field reads the parent and the provider content only, never the edit
    coverage, so the join stays linear in its own weight.
    """
    sigma=float(FUSION['registration_sigma_px'])
    low_ev,high_ev=FUSION['registration_gain_ev_bounds']
    support=np.asarray(support_field,bool)
    # The target across the cut is the photographed level as the retained side
    # carries it, measured on the parent itself, so the delivered pixels meet
    # what surrounds them; the provider side is measured on its own support so
    # the zeros it leaves outside the write cannot drag the band down.
    prior_low=_support_low(prior,np.ones_like(support),sigma)
    patch_low=_support_low(patch,support,sigma)
    local=np.log2(np.maximum(prior_low,np.float32(1e-6))/np.maximum(patch_low,np.float32(1e-6)))
    band=float(FUSION['band_inner_px']);interior=float(FUSION['band_interior_px'])
    reach=np.clip((interior-np.asarray(distance,np.float32))/max(interior-band,1.),0.,1.)
    weight=_smooth(reach).astype(np.float32)[...,None]
    base=np.log2(np.asarray(scale,np.float32)).reshape(1,1,3)
    field=base*(np.float32(1.)-weight)+local*weight
    outside=field<np.float32(low_ev)-np.float32(1e-6)
    above=field>np.float32(high_ev)+np.float32(1e-6)
    field=np.clip(field,np.float32(low_ev),np.float32(high_ev))
    inband=weight[...,0]>=np.float32(.5)
    record={'policy':FUSION['boundary_light_field'],'support_sigma_px':sigma,
        'band_px':int(FUSION['band_inner_px']),'interior_px':int(FUSION['band_interior_px']),
        'gain_ev_bounds':[low_ev,high_ev],
        'gain_ev_min_max':[round(float(field.min()),4),round(float(field.max()),4)],
        'registered_scale_ev':[round(float(v),4) for v in np.log2(np.asarray(scale,np.float32))],
        'band_gain_ev_median':([round(float(v),4) for v in np.median(field[inband],axis=0)]
            if inband.any() else None),
        'band_pixels':int(np.count_nonzero(inband)),
        'clamped_pixels':int(np.count_nonzero(np.any(outside|above,axis=-1))),
        'clamped_in_band_pixels':int(np.count_nonzero(np.any((outside|above)
            &(weight[...,0]>=np.float32(.5))[...,None],axis=-1))),
        'written_pixels':int(np.count_nonzero(support)),
        'darkened_pixels':int(np.count_nonzero(np.any(field<np.float32(1.)-np.float32(1e-6),axis=-1))),
        'directions':'brighten_and_darken','environment_side':'retained_parent_around_the_write',
        'reads_edit_coverage':False,'alpha_modified':False}
    return np.exp2(field).astype(np.float32),record


def _registration_scale(native,core,box,patch,prior,distance=None,*,weight=None):
    """Region-scale registration onto the exposure the photograph had there.

    The first moment of the patch's own core interior is compared with the
    first moment of the photographed master at the same place: their
    per-channel ratio is one gain that carries both the region's luminance and
    its colour temperature back onto the exposure and white point the
    photograph actually had there, and it is measured on the same statistic the
    retained ring is compared against. The local membrane then only has to
    close the remaining few pixels at the cut. The retained ring is measured as
    well and reported, so the receipt shows the region against both references.
    The estimate is taken over every written pixel of the authorized core, so
    the resulting field is one constant per channel over the whole region and
    does not vary with the taper: the taper then only has to carry the content
    transition, and the membrane only the residual the constant field leaves.
    A registered provider alpha restricts the estimate to the pixels it actually
    covers; the patch is expected unpremultiplied (see `_unpremultiply`).
    Only the core is written, and the gain is bounded, flagged and recorded.
    """
    x0,y0,x1,y1=core;bx0,by0,bx1,by1=box
    sigma=float(FUSION['registration_sigma_px']);pad=int(FUSION['registration_ring_px'])
    rx0=max(bx0,x0-pad);ry0=max(by0,y0-pad);rx1=min(bx1,x1+pad);ry1=min(by1,y1+pad)
    window=np.asarray(native[ry0:ry1,rx0:rx1],np.float32)
    ring=np.ones(window.shape[:2],bool)
    ring[y0-ry0:y1-ry0,x0-rx0:x1-rx0]=False
    inside=np.ones((y1-y0,x1-x0),bool)
    if weight is not None:
        inside=(np.asarray(weight,np.float32)>np.float32(0)).reshape(inside.shape)
    if not np.any(inside):
        inside=np.ones(inside.shape,bool)
    # The first moment of the region is what an exposure and white point mean,
    # and it is the statistic the retained ring is compared against, so the
    # registration is estimated on exactly that quantity.
    target_level=np.asarray(prior,np.float32)[inside].mean(0)
    region_level=np.asarray(patch,np.float32)[inside].mean(0)
    ring_level=target_level
    if ring.any():
        ring_level=window[ring].mean(0)
    low_ev,high_ev=FUSION['registration_gain_ev_bounds']
    wanted=target_level/np.maximum(region_level,np.float32(1e-6))
    scale=np.clip(wanted,np.float32(2.**low_ev),np.float32(2.**high_ev)).astype(np.float32)
    applied=(region_level*scale).astype(np.float32)
    residual=applied/np.maximum(target_level,np.float32(1e-6))
    return {'field':FUSION['registration_field'],
        'scale':[round(float(v),6) for v in scale],
        'target_level':[round(float(v),6) for v in target_level],
        'region_level':[round(float(v),6) for v in region_level],
        'ring_level':[round(float(v),6) for v in ring_level],

        'ring_pixels':int(np.count_nonzero(ring)),
        'clamped':bool(np.any(np.abs(wanted-scale)>np.float32(1e-6))),
        'residual_ev':[round(float(v),4) for v in np.log2(np.maximum(residual,1e-6))],
        'sigma_px':float(sigma)}


def _band_field(values,sigma,window,*,grey=None):
    """Local RMS of one band, as a per-pixel field instead of one global scalar."""
    from scipy.ndimage import gaussian_filter
    source=np.asarray(values,np.float32)
    if grey is None:
        band=source-gaussian_filter(source,sigma,mode='nearest')
    else:
        band=source
    luma=(band@_LUMA) if band.ndim==3 else band
    energy=gaussian_filter(luma*luma,window,mode='nearest')
    return band,np.sqrt(np.maximum(energy,0.)).astype(np.float32)


def _immerman_noise(grey):
    """Immerkaer's 3x3 noise estimate, the same operator the metric uses."""
    from scipy.ndimage import convolve
    kernel=np.asarray([[1.,-2.,1.],[-2.,4.,-2.],[1.,-2.,1.]],np.float32)/np.float32(6.)
    return np.abs(convolve(np.asarray(grey,np.float32),kernel,mode='nearest')).astype(np.float32)


def _match_texture(patch,prior):
    """Match delivered texture and grain energy to the retained photograph's.

    One global high-frequency gain left the region either too smooth or too
    noisy wherever the provider's content varied. Two local (64 px window)
    energy fields now do the work: the 0.6-1.6 px structure band against the
    parent's own band, and the sub-0.6 px fine band against the parent's own
    fine energy. Both gains are bounded and recorded, and the achieved band and
    Immerkaer noise ratios of the result are measured afterwards and recorded
    too: the estimator that is reported is never a different estimator from the
    one being corrected. No texel is synthesised, only the provider's own
    residue is rescaled.
    """
    from scipy.ndimage import gaussian_filter
    window=float(FUSION['detail_window_px'])
    sigma=float(FUSION['detail_sigma']);fine_sigma=float(FUSION['detail_fine_sigma'])
    low_bound,high_bound=FUSION['detail_ratio_bounds']
    noise_bound_low,noise_bound_high=FUSION['noise_ratio_bounds']
    patch_low=gaussian_filter(patch,(sigma,sigma,0),mode='nearest')
    prior_low=gaussian_filter(prior,(sigma,sigma,0),mode='nearest')
    patch_fine=gaussian_filter(patch,(fine_sigma,fine_sigma,0),mode='nearest')
    prior_fine=gaussian_filter(prior,(fine_sigma,fine_sigma,0),mode='nearest')
    patch_fine_band=(patch-patch_fine).astype(np.float32)
    prior_fine_band=(prior-prior_fine).astype(np.float32)
    patch_coarse=((patch-patch_low)-patch_fine_band).astype(np.float32)
    prior_coarse=((prior-prior_low)-prior_fine_band).astype(np.float32)
    _,patch_structure=_band_field(patch_coarse,sigma,window,grey=True)
    _,prior_structure=_band_field(prior_coarse,sigma,window,grey=True)
    _,patch_fine_energy=_band_field(patch_fine_band,sigma,window,grey=True)
    _,prior_fine_energy=_band_field(prior_fine_band,sigma,window,grey=True)
    structure_gain=np.clip(prior_structure/np.maximum(patch_structure,np.float32(1e-6)),
        low_bound,high_bound).astype(np.float32)
    fine_gain=np.clip(prior_fine_energy/np.maximum(patch_fine_energy,np.float32(1e-6)),
        noise_bound_low,noise_bound_high).astype(np.float32)
    structure_gain=gaussian_filter(structure_gain,window*.25,mode='nearest')
    fine_gain=gaussian_filter(fine_gain,window*.25,mode='nearest')
    gain=np.clip(fine_gain,noise_bound_low,noise_bound_high).astype(np.float32)
    matched=(patch_low+patch_coarse*structure_gain[...,None]
        +patch_fine_band*gain[...,None]).astype(np.float32)
    return matched,{'structure_gain_median':round(float(np.median(structure_gain)),6),
        'noise_gain_median':round(float(np.median(gain)),6),
        'structure_gain_bounds':[low_bound,high_bound],
        'noise_gain_bounds':[noise_bound_low,noise_bound_high]}


def _bounded_tone(patch,correction):
    """Reduce one scalar dose so the joined tone stays inside display SDR."""
    upper=np.float32(1.);dose=np.ones_like(patch)
    positive,negative=correction>0,correction<0
    np.divide(upper-patch,correction,out=dose,where=positive)
    np.divide(-patch,correction,out=dose,where=negative)
    return (correction*np.clip(dose.min(-1),0.,1.)[...,None]).astype(np.float32)


def _unpremultiply(associated,weight):
    """Provider colour as itself, so alpha cannot steer the measurements.

    A generated patch is composited on its own alpha, and production hands the
    join associated (premultiplied) colour. Estimating a level or a texture
    energy on that would divide every statistic by the matte: a soft hole in the
    provider's alpha would drag the region level towards black. The analysis
    therefore runs on the unpremultiplied content and alpha stays where it
    belongs, in the destination decay and the single premultiplication of the
    source; an alpha of zero writes nothing and keeps no content.
    """
    values=np.asarray(associated,np.float32)
    mask=np.asarray(weight,np.float32)
    safe=np.maximum(mask,np.float32(1e-6))
    return np.where(mask[...,None]>np.float32(0),
        values/safe[...,None],np.float32(0.)).astype(np.float32)


def _register_patch(patch,scale):
    """Apply the regional gain to the region's low band only.

    Scaling the whole patch would rescale its texture too, and a wrong exposure
    must not do that: the provider's detail is right at its own amplitude and
    only the level is wrong. The scale therefore multiplies the 1.6px low band
    while the high band keeps its amplitude; the region mean still lands on the
    photographed level because that band has zero mean. The gain is one
    per-channel value over the whole region when the boundary light solve is
    off, and one per-channel value per pixel when the solve is on: it is still
    an exposure/white-point gain on the low band, never a rescale of the
    provider's texture.
    """
    from scipy.ndimage import gaussian_filter
    sigma=float(FUSION['detail_sigma'])
    values=np.asarray(patch,np.float32)
    gain=np.asarray(scale,np.float32)
    if gain.ndim==1:
        gain=gain.reshape(1,1,3)
    if gain.ndim!=3 or gain.shape[-1]!=3 or gain.shape[0] not in (1,values.shape[0]) or gain.shape[1] not in (1,values.shape[1]):
        raise ValueError('Join registration gain must be one value per channel or one per pixel')
    low=gaussian_filter(values,(sigma,sigma,0),mode='nearest')
    high=values-low
    return np.clip(low*gain+high,0.,np.float32(1.)).astype(np.float32)


def _tone_depth(distance):
    """Recorded membrane depth in pixels, from the authorized core geometry."""
    depth=int(round(min(distance.shape)*float(FUSION['tone_depth_fraction'])))
    low,high=FUSION['tone_depth_bounds']
    return max(int(low),min(int(high),depth))


def _tone_correction(prior,patch,distance):
    """Membrane that closes the last few pixels between region and ring.

    The region scale is already registered onto the ring, so this term only
    has to remove what the patch's own content does near the cut. Its dose is
    driven by the distance to the authorized core boundary (never by a blended
    weight, which alpha would contaminate) and it reaches exactly zero inside
    the recorded depth, leaving the provider's own lighting there.
    """
    from scipy.ndimage import gaussian_filter
    depth=_tone_depth(distance)
    sigma=max(1.,depth*.5)
    residual=np.asarray(prior,np.float32)-np.asarray(patch,np.float32)
    lowpass=gaussian_filter(residual,(sigma,sigma,0),mode='nearest')
    dose=np.clip(1.-(distance-np.float32(1.))/np.float32(depth),0.,1.)
    bound=float(FUSION['tone_bound'])
    return _bounded_tone(patch,np.clip(lowpass,-bound,bound)*dose[...,None].astype(np.float32)),depth


def _join_detail(prior,delivered,distance):
    """Recomputable band/interior detail, the definition the audit measured.

    Band is the first 8 px inside the authorized core edge and interior is
    everything deeper than 64 px, both scored by the median absolute 1.6 px
    high-pass of the delivered luma. Reporting the same number for the
    retained parent makes the ratio self-checking inside the receipt.
    """
    from scipy.ndimage import gaussian_filter
    band=(distance>np.float32(1.))&(distance<=np.float32(1.+FUSION['band_inner_px']))
    interior=distance>np.float32(1.+FUSION['band_interior_px'])
    def ratio(values):
        if not (band.any() and interior.any()):return None
        grey=np.asarray(values,np.float32)@_LUMA
        highpass=grey-gaussian_filter(grey,float(FUSION['detail_sigma']),mode='nearest')
        reference=float(np.median(np.abs(highpass[interior])))
        return round(float(np.median(np.abs(highpass[band])))/max(reference,1e-9),6)
    measured=ratio(delivered);retained=ratio(prior)
    limit=float(FUSION['join_detail_relative_limit'])
    relative=(None if measured is None or not retained
        else round(float(measured)/float(retained),6))
    return {'band_px':int(FUSION['band_inner_px']),'interior_px':int(FUSION['band_interior_px']),
        'delivered':measured,'parent':retained,'relative_limit':limit,
        'delivered_over_parent':relative,
        'relative_ok':None if relative is None else bool(relative<=limit)}


# The round's own authorized operations, and what each one means for ownership.
# A scene operation authorizes the scene side of the core as content of this
# round, so it must be written and joined; only a person-only round may narrow
# the write to the subject. Relighting is a joint cross-boundary solve, so both
# sides have to be written for it to mean anything.
SCENE_OPERATIONS=('scene_edit','scene_cleanup','scene_composite','relight')
SUBJECT={'schema':'cosluma_subject_ownership.v1','role':'current_canvas_subject_matte',
    # The matte is the person silhouette of the canvas actually being edited,
    # inferred by this route's own BiRefNet silhouette (source_alpha), never
    # taken from the composite's own coverage: coverage is bounded edit
    # permission, this matte is who owns the boundary.
    'producer':'source_alpha.source_silhouette:birefnet_lite_onnx',
    'storage':'subject-matte.npy inside the owned tool attempt',
    'quantisation':'uint8 round trip of the silhouette score',
    # Ownership rule, in one place. Default: the authorized core is written as
    # reserved, scene included. Narrowing to the subject happens only for a
    # person-only round, and a core with no subject in it never narrows.
    'ownership_rule':'authorized content is never silently discarded; when in doubt, write',
    'default_write':'authorized_core_scene_included',
    'narrowed_write':'subject_generated_scene_retained',
    'scene_operations':list(SCENE_OPERATIONS),
    'person_only_operations':['portrait_retouch'],
    'operation_semantics':{
        'scene_edit':'authorized_scene_write',
        'scene_cleanup':'authorized_scene_write',
        'scene_composite':'authorized_scene_write',
        'relight':'authorized_joint_lighting_both_sides_written',
        'portrait_retouch':'person_only_may_narrow_to_the_subject',
        'detail_repair':'person_only_only_without_any_scene_operation_and_with_preserved_scene_policy'},
    'narrowing_requires':('no scene operation in the round and scene_policy == preserve'),
    # matte <= soft_low is the scene side of the core, matte >= .5 is the subject.
    'soft_low':.02,'soft_high':.98,'maximum_edge_width_px':8.,'minimum_subject_pixels':256}


def _smooth(value):
    value=np.clip(value,0.,1.)
    return (value*value*(3.-2.*value)).astype(np.float32)


def _subject_array(subject,size):
    """Validate one decoded subject matte that covers exactly this canvas.

    A matte is segmentation, not permission: it has to be a 2-D weight per
    canvas pixel that actually separates a subject from the scene. A uniform
    mask, an index image or the edit coverage cannot stand in for it.
    """
    if subject is None:return None
    value=np.asarray(subject)
    if value.ndim!=2 or tuple(value.shape)!=(size[1],size[0]):
        raise ValueError('Subject matte must hold one weight per current canvas pixel')
    if value.dtype==np.uint8:
        value=value.astype(np.float32)/np.float32(255.)
    elif value.dtype in (np.float16,np.float32,np.float64,np.bool_):
        value=value.astype(np.float32)
    else:
        raise ValueError('Subject matte must be a decoded segmentation, not indices or labels')
    if not np.isfinite(value).all() or float(value.min())<0. or float(value.max())>1.:
        raise ValueError('Subject matte weights must be finite and inside zero to one')
    inside=int(np.count_nonzero(value>=np.float32(.5)))
    outside=int(np.count_nonzero(value<np.float32(.5)))
    floor=int(SUBJECT['minimum_subject_pixels'])
    if inside<floor or outside<floor:
        raise ValueError('Subject matte must separate a real subject from a real scene')
    # A segmentation is not a box: the whole point of the matte is to replace
    # the reserved rectangle, so a filled rectangle is refused as evidence.
    subject=value>=np.float32(.5)
    rows=np.flatnonzero(np.any(subject,axis=1));columns=np.flatnonzero(np.any(subject,axis=0))
    rectangle=np.zeros_like(subject)
    rectangle[rows[0]:rows[-1]+1,columns[0]:columns[-1]+1]=True
    if np.array_equal(rectangle,subject):
        raise ValueError('Subject matte must be a segmentation, not the reserved rectangle')
    return np.ascontiguousarray(value,dtype=np.float32)


def _matte_soft(matte):
    return ((matte>np.float32(SUBJECT['soft_low']))&(matte<np.float32(SUBJECT['soft_high'])))


def _matte_edge_width(matte):
    """Measured width of the matte's own transition, in canvas pixels."""
    from scipy.ndimage import distance_transform_edt
    soft=_matte_soft(matte)
    if not soft.any():return float(FUSION['minimum_taper_px'])
    reach=distance_transform_edt(~soft)
    width=2.*float(np.median(reach[soft]))
    return float(np.clip(width,FUSION['minimum_taper_px'],float(SUBJECT['maximum_edge_width_px'])))


def _subject_ownership(matte):
    """The subject's ownership of each core pixel, saturated to exactly zero and one.

    The transition is the matte's own soft band, so the boundary between the
    generated subject and the retained scene is the person's boundary instead
    of a rectangle.
    """
    low,high=float(SUBJECT['soft_low']),float(SUBJECT['soft_high'])
    return _smooth((matte-np.float32(low))/np.float32(high-low))


def _matte_taper(matte,edge):
    """Join width: the matte's own edge where the subject boundary is, wider away from it."""
    from scipy.ndimage import distance_transform_edt
    wide=float(FUSION['maximum_taper_px'])
    narrow=float(np.clip(edge,float(FUSION['minimum_taper_px']),wide))
    soft=_matte_soft(matte)
    if not soft.any():return np.full(matte.shape,wide,np.float32)
    reach=distance_transform_edt(~soft)
    return (narrow+(wide-narrow)*_smooth(reach/np.float32(wide))).astype(np.float32)


def _ownership_gate(matte,edge):
    """One where the join may correct tone, zero across the subject boundary.

    The membrane estimates the parent's tone inside the cut. Across a real
    ownership change that estimate would paint the scene's tone onto the
    generated subject (or the reverse), which is exactly the rim it is supposed
    to close, so the dose is gated to same-owner pixels only.
    """
    from scipy.ndimage import distance_transform_edt
    soft=_matte_soft(matte)
    if not soft.any():return np.ones(matte.shape,np.float32)
    reach=distance_transform_edt(~soft)
    return _smooth(reach/np.float32(max(edge,1.)))


def _subject_record(core_matte,core,source,edge,taper,gate,subject_only):
    """Reviewable evidence of which boundary this join actually wrote.

    The write boundary is the reserved rectangle unless the round narrowed the
    write to the subject; the transition boundary is the subject's own edge
    whenever a real matte supplies it. Both are recorded, so the receipt cannot
    claim a subject boundary it did not write, or hide one it did.
    """
    from scipy.ndimage import distance_transform_edt
    soft=_matte_soft(matte=core_matte)
    band=(source>np.float32(0))&(source<np.float32(.999))
    near=soft if not soft.any() else distance_transform_edt(~soft)<=np.float32(1.)
    pixels=float(core_matte.size)
    edges=_edge_distance((core[2]-core[0],core[3]-core[1]))<=np.float32(2.)
    scene=core_matte<=np.float32(SUBJECT['soft_low'])
    scene_pixels=int(np.count_nonzero(scene))
    scene_written=int(np.count_nonzero(scene&(source>np.float32(0))))
    return {'schema':SUBJECT['schema'],'role':SUBJECT['role'],'available':True,
        'boundary_source':'subject_matte_edge' if subject_only else 'authorized_rectangle',
        'transition_boundary':'subject_matte_edge',
        'write_boundary':'subject_matte_edge' if subject_only else 'authorized_rectangle',
        'ownership':(SUBJECT['narrowed_write'] if subject_only else SUBJECT['default_write']),
        'subject_only_round':bool(subject_only),
        'scene_pixels':scene_pixels,'scene_pixels_written':scene_written,
        'scene_side_written_fraction':round(scene_written/max(scene_pixels,1),6),
        'edge_width_px':round(float(edge),3),
        'subject_fraction_of_core':round(float(np.count_nonzero(core_matte>=np.float32(.5)))/pixels,6),
        'silhouette_pixels_in_core':int(np.count_nonzero(soft)),
        'written_fraction_of_core':round(float(np.count_nonzero(source>0))/pixels,6),
        'retained_scene_fraction_of_core':round(float(np.count_nonzero(source<=0))/pixels,6),
        'taper_px':{'minimum':round(float(taper.min()),3),'median':round(float(np.median(taper)),3),
            'maximum':round(float(taper.max()),3)},
        'join_band_pixels':int(np.count_nonzero(band)),
        'join_band_on_silhouette_fraction':round(float(np.count_nonzero(band&near))
            /max(int(np.count_nonzero(band)),1),6),
        'membrane_gated_on_silhouette':bool(gate is not None),
        'core_edge_inside_the_subject_fraction':round(float(np.mean(core_matte[edges]>=np.float32(.5)))
            if edges.any() else 0.,6)}


def _recorded_provider_readings(native,patch,box,subject,core):
  """What the generated picture did with the subject it was meant to keep.

  Recorded beside the verdict and never part of it: the reading explains why a
  composite may be detail limited, and nothing here changes a decision. Returns
  None when there is no subject matte to compare against, or when the reading
  cannot be taken, so a composite that already succeeded never fails here.
  """
  if subject is None or core is None:
    return None
  try:
    import composite_quality
    from float_appearance import linear_to_srgb
    def srgb8(value):
      encoded=np.asarray(linear_to_srgb(np.asarray(value,np.float32)),np.float32)
      return np.uint8(np.clip(encoded,0.,1.)*255.+0.5)
    canvas=np.zeros_like(native)
    canvas[box[1]:box[3],box[0]:box[2]]=patch
    return composite_quality.provider_subject_diagnostics(
      srgb8(native),srgb8(canvas),np.asarray(subject,np.float32),core)
  except Exception:
    return None

def fuse_region(native,patch,alpha,*,core,box,subject=None,subject_only=False):
    """Compose one regional generated patch into the current canvas.

    The authorized core and its generation context are exactly the ones the
    reservation checked. Untouched pixels keep the parent's own bytes, and the
    returned coverage is the real written join weight, not a fetched permission.

    Ownership follows the round's authorized operations, not the mere presence
    of a subject in the core: by default the whole authorized core is written,
    scene side included, and the subject matte only shapes the transition and
    the boundary light solve. `subject_only=True` is the person-only round,
    where the write itself narrows to the subject and the scene side keeps the
    parent's bytes; a caller may only pass it when the round authorizes no
    scene operation and the scope preserves the scene (see `_scene_write_trigger`).
    """
    x0,y0,x1,y1=core
    local=patch[y0-box[1]:y1-box[1],x0-box[0]:x1-box[0]]
    registered=None if alpha is None else alpha[y0-box[1]:y1-box[1],x0-box[0]:x1-box[0]]
    # Every measurement below runs on the provider's own colour; alpha is only
    # ever applied once, in `source` and in the premultiplied term.
    content=local if registered is None else _unpremultiply(local,registered)
    prior=np.asarray(native[y0:y1,x0:x1],np.float32)
    result=np.array(native,np.float32,copy=True)
    view=result[y0:y1,x0:x1]
    canvas_subject=_subject_array(subject,(native.shape[1],native.shape[0]))
    core_subject=(None if canvas_subject is None
        else np.ascontiguousarray(canvas_subject[y0:y1,x0:x1]))
    edge=ownership=gate=None
    # Only the branches below that truly know the written boundary set these
    # keys; with no matte the join does not own the declaration, so the caller's
    # recorded reason for the reserved write stands instead of being overwritten
    # by a generic rectangle claim.
    subject_record={'schema':SUBJECT['schema'],'role':SUBJECT['role'],'available':False}
    if core_subject is not None:
        inside=int(np.count_nonzero(core_subject>=np.float32(.5)))
        subject_record.update(available=True,
            subject_fraction_of_core=round(inside/max(core_subject.size,1),6))
        if inside<int(SUBJECT['minimum_subject_pixels']):
            # The authorized core lies in the scene: there is no subject
            # boundary inside it to own, so the reserved write stands and the
            # receipt says so instead of silently clipping the edit away.
            subject_record.update(boundary_source='authorized_core_without_subject',
                write_boundary='authorized_rectangle',ownership='authorized_core_written_as_reserved',
                subject_only_round=False)
            core_subject=None
        else:
            edge=_matte_edge_width(core_subject)
            ownership=_subject_ownership(core_subject)
            gate=_ownership_gate(core_subject,edge)
            write_boundary='subject_matte_edge' if subject_only else 'authorized_rectangle'
            subject_record.update(boundary_source=write_boundary,transition_boundary='subject_matte_edge',
                write_boundary=write_boundary,
                ownership=(SUBJECT['narrowed_write'] if subject_only else SUBJECT['default_write']),
                subject_only_round=bool(subject_only))
    record={'schema':FUSION['schema'],'policy':dict(FUSION),'working_domain':'display_linear_sRGB',
        'alpha_modified':False,'coordinates_changed':False,'generated_texture_added':False,
        'provider_alpha_registered':registered is not None,'visual_acceptance':False}
    if not _join_analysis_ready((x1-x0,y1-y0)):
        weight=_mask((x1-x0,y1-y0))
        if ownership is not None:
            subject_record['transition_boundary']='subject_matte_edge_inside_the_bypass'
            if subject_only:
                weight=(weight*ownership).astype(np.float32)
                subject_record['boundary_source']='subject_matte_edge_inside_the_bypass'
                subject_record['write_boundary']='subject_matte_edge_inside_the_bypass'
        actual=weight if registered is None else (weight*registered).astype(np.float32)
        coverage=np.zeros(native.shape[:2],np.float32)
        coverage[y0:y1,x0:x1]=actual
        mixed=prior*(1-actual[...,None])+local*weight[...,None]
        view[actual>0]=mixed[actual>0]
        record.update(mode='bounded_rectangular_crossfade',
            bypass_reason='edit_region_below_photographic_join_support',subject=subject_record)
        return result,coverage,record
    from scipy.ndimage import gaussian_filter
    distance=_edge_distance((x1-x0,y1-y0))
    # The boundary that gives the generated subject back to the parent is the
    # subject's own edge whenever a real matte is available; the retained
    # structure is only the approximation used when there is none.
    taper=_structure_taper(prior) if ownership is None else _matte_taper(core_subject,edge)
    ramp=_join_ramp(distance,taper)
    # The write itself narrows to the subject only for a person-only round; the
    # matte still shapes the taper either way.
    narrowed=ownership is not None and subject_only
    written=(ramp*ownership).astype(np.float32) if narrowed else ramp
    # alpha belongs to the destination decay only: the source is the
    # premultiplied patch weighted by the taper, so semi-transparent provider
    # pixels compose as source-over instead of silently darkening the region.
    source=written if registered is None else (written*registered).astype(np.float32)
    # The constant field is estimated on the pixels this join actually writes,
    # so the subject's own level is measured against the photographed subject
    # instead of against a core that is mostly retained scene.
    analysis_weight=(written if registered is None
        else (written*(registered>np.float32(0))).astype(np.float32))
    registration=_registration_scale(native,core,box,content,prior,weight=analysis_weight)
    registration['unpremultiplied_estimate']=True
    # The level and white point are not constant inside the region, so the
    # constant registration alone leaves a rim exactly at the cut. The joint
    # solve below keeps it in the interior and carries the parent's own local
    # relation across the boundary band, in both directions, on the geometry
    # this join actually writes. It reads the parent and the provider content
    # only, so the join stays linear in its own coverage weight.
    # The light solve follows the boundary this join actually delivers: the
    # subject's edge only when the write narrowed to the subject, otherwise the
    # authorized core's own edge, which is where the delivered pixels meet the
    # retained photograph.
    boundary=_ownership_distance(ownership if narrowed else None,(x1-x0,y1-y0))
    boundary_light=None
    if ownership is None:
        gain=registration['scale']
    else:
        content_support=(np.ones_like(written,dtype=bool) if registered is None
            else (registered>np.float32(0.)))
        gain,boundary_light=_boundary_light_gain(prior,content,ownership,content_support,
            boundary,registration['scale'])
    registered_patch=_register_patch(content,gain)
    registration['clipped_pixels']=int(np.count_nonzero(np.any(registered_patch
        >=np.float32(1.),axis=-1)))
    matched,gains=_match_texture(registered_patch,prior)
    correction,depth=_tone_correction(prior,matched,distance)
    membrane_max=round(float(np.abs(correction).max(initial=0.)),6)
    if gate is not None:
        # Same-owner tone only: the membrane may not carry the scene's level
        # across the subject's boundary or the subject's across the scene's.
        correction=(correction*gate[...,None]).astype(np.float32)
    aligned=np.clip(matched+correction,0.,1.).astype(np.float32)
    premult=aligned if registered is None else (aligned*registered[...,None]).astype(np.float32)
    mixed=(prior*(1.-source[...,None])+written[...,None]*premult).astype(np.float32)
    coverage=np.zeros(native.shape[:2],np.float32)
    coverage[y0:y1,x0:x1]=source
    view[source>0]=mixed[source>0]
    interior=distance>np.float32(1.+float(FUSION['band_interior_px']))
    deep=distance>np.float32(1.+depth)
    def band_rms(values):
        grey=np.asarray(values,np.float32)@_LUMA
        band=grey-gaussian_filter(grey,float(FUSION['detail_sigma']),mode='nearest')
        return float(np.sqrt(np.mean(np.square(band[interior])))) if interior.any() else 0.
    def noise_mean(values):
        field=_immerman_noise(np.asarray(values,np.float32)@_LUMA)
        return float(np.mean(field[interior])) if interior.any() else 0.
    reference=band_rms(prior)
    record.update(mode='registered_photographic_join',
        registration=registration,
        registration_gain_ev=round(float(np.log2(np.max(registration['scale'])
            /np.min(registration['scale']))),6),
        detail=gains,
        achieved_high_frequency_ratio=round(band_rms(aligned)/max(reference,1e-9),6),
        achieved_noise_ratio=round(noise_mean(aligned)/max(noise_mean(prior),1e-9),6),
        join_detail=_join_detail(prior,mixed,distance),
        tone_correction_max=round(float(np.abs(correction).max(initial=0.)),6),
        tone_correction_max_before_ownership_gate=membrane_max,
        tone_correction_interior_max=round(float(np.abs(correction[deep]).max(initial=0.))
            if deep.any() else 0.,6),
        tone_correction_pixels=int(np.count_nonzero(np.any(correction!=0,axis=-1))),
        tone_depth_px=int(depth),
        boundary_pixels=int(np.count_nonzero((written>0)&(written<1))),
        boundary_light=boundary_light,
        joined_fraction=round(float(np.count_nonzero(source>0)/source.size),6),
        alpha_in_destination_only=True,
        analysis_domain='unpremultiplied_provider_colour',
        subject=(_subject_record(core_subject,core,source,edge,taper,gate,subject_only)
            if ownership is not None else subject_record),
        coverage_semantics='written_join_weight_inside_the_authorized_core')
    return result,coverage,record


def _resample(rgb,size):
    from float_appearance import srgb_to_linear
    values=srgb_to_linear(np.asarray(rgb,np.float32)/255)
    if tuple(rgb.size)==tuple(size):return values
    result=np.empty((size[1],size[0],3),np.float32)
    for channel in range(3):
        result[...,channel]=np.asarray(Image.fromarray(values[...,channel]).resize(tuple(size),Image.Resampling.BILINEAR))
    return result


def _generated_pixels(path,size):
    """Register opacity with associated linear RGB; transparent RGB has no weight."""
    from raw_support import open_rgb
    from float_appearance import srgb_to_linear
    rgb=open_rgb(path)
    if abs((rgb.width/rgb.height)/(size[0]/size[1])-1)>.02:
        raise ValueError('Generated patch framing needs explicit registration; refusing implicit distortion')
    with Image.open(path) as raster:
        oriented=ImageOps.exif_transpose(raster)
        alpha=np.asarray(oriented.convert('RGBA').getchannel('A'),np.float32)/255
    if alpha.shape!=(rgb.height,rgb.width):
        raise ValueError('Generated opacity and colour use different oriented coordinates')
    if np.all(alpha==1):return rgb,_resample(rgb,size),None
    associated=srgb_to_linear(np.asarray(rgb,np.float32)/255)*alpha[...,None]
    if tuple(rgb.size)!=tuple(size):
        resized=np.empty((size[1],size[0],3),np.float32)
        for channel in range(3):
            resized[...,channel]=np.asarray(Image.fromarray(associated[...,channel]).resize(tuple(size),Image.Resampling.BILINEAR))
        associated=resized
        alpha=np.asarray(Image.fromarray(alpha).resize(tuple(size),Image.Resampling.BILINEAR))
    return rgb,associated,alpha



def _scope_boundaries(scope,zh,*,local=False):
    if scope['target_setting']=='same':
        setting=('保留裁图实际可见的原场景室内外属性。' if local else '保持原场景的室内外属性。') if zh else (
            'Retain the original indoor/outdoor setting type visible in this crop.' if local else
            'Retain the original indoor/outdoor setting type.')
    else:
        setting=('获准的目标场景为'+('室内。' if scope['target_setting']=='indoor' else '室外。')) if zh else (
            'The authorized target setting is '+scope['target_setting']+'.')
    if scope['scene_policy']=='preserve':
        setting+=(' 保留现有场景，仅作获准精修，不重建场景。' if zh else
            ' Preserve the existing scene; make only the permitted retouch, without scene reconstruction.')
    elif scope['scene_policy']=='local_edit':
        setting+=(' 保留原场地可见主要结构，仅作获准局部场景修改。' if zh else
            ' Retain the visible main structure of the original location; scene changes are local.')
    if not scope['effects_allowed']:
        setting+=(' 不新增叙事特效。' if zh else ' Do not add narrative effects.')
    return setting


def _prompt(scope,decision,core_in_input,family,shaping_authorization=None):
    zh=scope['language']=='zh'
    if core_in_input is not None:
        # The input is a crop, not an identity reference for a new photograph.
        # Full-photo goals/keep-lists can describe objects outside this frame;
        # including them asks the image model to invent those missing objects.
        if zh:
            intro='仅编辑提供的局部照片，按当前裁图取景落实获准的局部编辑。'
            bounds='许可修改的核心区域以裁图归一化坐标[x0,y0,x1,y1]表示：'+json.dumps(core_in_input)+'。'
            preserve='保留输入裁图已有的物体截断、相对位置、尺度、视角与边界，以及未要求修改的内容。'
            output='输出保持当前裁图的取景范围，不延展画面，也不补全被裁切的对象；不添加文字、水印、边框或对比面板。'
        else:
            intro='Edit only the supplied cropped photograph, with its present framing, to make the permitted local edit.'
            bounds='Permitted edit core in normalized crop coordinates [x0,y0,x1,y1]: '+json.dumps(core_in_input)+'.'
            preserve='Retain the input crop\'s existing cut-off objects, relative positions, scale, viewpoint and boundaries, and content not requested to change.'
            output='Return the same crop framing without extending the view or completing cut-off objects. Do not add captions, watermarks, borders or comparison panels.'
        preserve+=' '+_scope_boundaries(scope,zh,local=True)
        if shaping_authorization is not None:
            from photo_shaping import part_limits
            preserve += (' '+part_limits(shaping_authorization)+
                ' Only explicitly selected contours are exempt from preserving current shape. '
                'Keep facial features, expression, pose, stature, limb length, clothing construction and props. '
                'Repair only the immediate newly exposed background inside the core; no scene redesign, skin beautification '
                'or independent lighting change is authorized. Do not stretch the crop or change its framing.')
        if family=='qwen_image_2_1':
            return ' '.join(str(part).replace('\n',' ') for part in (intro,decision['instruction'],bounds,preserve,output))
        return intro+'\n\nLOCAL EDIT\n'+decision['instruction']+'\n\nCROP BOUNDARY\n'+bounds+' '+preserve+'\n\nOUTPUT\n'+output
    if zh:
        intro='编辑提供的当前最佳完整照片，充分落实本轮修改指令。'
        limits='保留人物身份、妆造、表情、姿态、体型、服装、重要道具及未要求修改的内容。输出一张完整照片，不加文字、水印、边框或对比面板。'
    else:
        intro='Edit the supplied current best complete photograph to fully realize this edit instruction.'
        limits='Preserve identity, deliberate makeup, expression, pose, body shape, costume, important props and content not requested to change. Return one complete photograph without captions, watermarks, borders or comparison panels.'
    setting=_scope_boundaries(scope,zh)
    # The model already observed the design and review when selecting this
    # instruction. Repeating old goals/keep-lists here can lock the attribute
    # this edit must change (for example the failed illumination).
    if family=='qwen_image_2_1':
        return ' '.join(str(part).replace('\n',' ') for part in (intro,decision['instruction'],limits,setting) if part)
    return intro+'\n\nMAIN EDIT\n'+decision['instruction']+'\n\nPRESERVE\n'+limits+' '+setting


def _birefnet_subject_matte(root,canvas_path,image):
    """The route's own real subject silhouette of the canvas being edited.

    The desktop process must not mix ONNX runtimes, so this reuses the
    registered silhouette inference instead of importing a second runtime.
    """
    from source_alpha import source_silhouette
    silhouette,metadata=source_silhouette(Path(root),Path(canvas_path),image,backend='birefnet')
    keep=('onnxruntime','execution_provider','session_providers','input_dtype','preprocess',
        'postprocess','cache_hit','cache_key','source_sha256','oriented_size','candidate_used',
        'role','requested_backend')
    return np.asarray(silhouette,np.float32)/np.float32(255.),{'producer':SUBJECT['producer'],
        'role':SUBJECT['role'],'silhouette':{key:metadata[key] for key in keep if key in metadata}}


def _scene_write_trigger(scope,decision):
    """Does this round authorize the scene side of the core, and on what ground?

    Ownership is decided by what the round was authorized to do, never by the
    mere presence of a subject in the core. A scene operation - or a scope that
    does not preserve the scene - makes the scene side authorized content that
    must be written and joined; only a person-only round may narrow the write to
    the subject. Relighting is a joint cross-boundary solve, so it counts as a
    scene write: one-sided relighting is not the operation that was authorized.
    """
    operations=[str(name) for name in (decision.get('operations') or [])]
    scene=sorted(name for name in operations if name in SCENE_OPERATIONS)
    policy=str(scope.get('scene_policy'))
    if scene:return False,'authorized_scene_operation:'+','.join(scene)
    if policy!='preserve':return False,'scope_allows_scene_rebuild:'+policy
    return True,'person_only_round:'+','.join(sorted(operations))


JOINED_SUBJECT_KEYS=('boundary_source','write_boundary','transition_boundary','ownership',
    'subject_only_round','edge_width_px','written_fraction_of_core','retained_scene_fraction_of_core',
    'scene_pixels','scene_pixels_written','scene_side_written_fraction','subject_fraction_of_core')


def _joined_subject_record(subject_record,fusion):
    """The receipt's boundary fields, copied from the join that actually wrote them.

    One source of truth: the pre-join declaration says what the round intended,
    and this overwrites it with the boundary the join delivered, marking itself
    authoritative. A join that makes no boundary claim at all (a reserved
    rectangle below the photographic join support) leaves the declaration in
    place instead of being overwritten by a generic rectangle claim.
    """
    joined=(fusion or {}).get('subject') or {}
    if 'boundary_source' in joined:subject_record['authoritative']='join.subject'
    for key in JOINED_SUBJECT_KEYS:
        if key in joined:subject_record[key]=joined[key]
    return subject_record


def _current_canvas_subject(config,best,image):
    """Ownership from a real subject matte of the current canvas, or a recorded absence.

    A missing model, runtime or registration is recorded as missing; the edit
    coverage is never substituted for the matte. The caller keeps the reviewed
    rectangular join when there is no matte, so nothing is silently relabelled.
    """
    size=image.size
    record={'schema':SUBJECT['schema'],'role':SUBJECT['role'],'available':False,
        'canvas_size':[int(size[0]),int(size[1])],'current_canvas_sha256':best['sha256'],
        'matte_is_the_edit_coverage':False,
        # Declared before the join runs, and not the truth about the boundary
        # that gets written: the join's own record overwrites these and marks
        # itself authoritative, so a stale rectangle can never stand in for a
        # subject boundary the matte actually took over.
        'boundary_source':'authorized_rectangle','ownership':'authorized_core_written_as_reserved',
        'acquisition':{'available':False,'boundary_source':'authorized_rectangle',
            'ownership':'authorized_core_written_as_reserved'},
        'authoritative':'pre_join_declaration'}
    provider=config.get('subject_matte')
    if provider is False:
        record['reason']='subject_matte_disabled_by_configuration'
        return None,record
    try:
        _cancel(config.get('cancel'))
        if callable(provider):
            value,provenance=provider(root=Path(config['root']),canvas_path=Path(best['path']),
                image=image)
            provenance=dict(provenance or {})
            provenance.setdefault('producer','injected_subject_matte_provider')
        else:
            value,provenance=_birefnet_subject_matte(config['root'],best['path'],image)
        matte=_subject_array(value,size)
        pixels=float(max(matte.size,1))
        record.update(available=True,**provenance)
        record['subject_fraction_of_canvas']=round(
            float(np.count_nonzero(matte>=np.float32(.5)))/pixels,6)
        record['soft_band_fraction_of_canvas']=round(float(np.count_nonzero(_matte_soft(matte)))/pixels,6)
        return matte,record
    except (OSError,RuntimeError,ValueError,ImportError,TypeError,AttributeError,KeyError) as error:
        record['reason']=type(error).__name__+': '+str(error)[:400]
        return None,record
    finally:
        _cancel(config.get('cancel'))


def _execute(context,config,kind):
    import reference_provider
    from raw_support import deterministic_srgb_profile
    from photo_appearance import freeze_candidate,validate_asset,load_asset
    _cancel(config.get('cancel'))
    scope,design,source,best,decision,reservation,appearance=_reservation(context,config,kind)
    base=_best_rgb(best['path']);size=base.size
    parent=context['appearance_assets']['best']
    core=box=None
    if kind=='edit_region':core,box=_region(decision.get('region'),size)
    elif decision.get('region') is not None:raise ValueError('Joint editing does not accept a regional box')
    if core is None and scope['work_type']=='retouch':
        raise ValueError('Retouching requires a bounded regional edit instead of whole-image redrawing')
    output=Path(config['output']).resolve()/context['attempt_id'];output.mkdir(parents=True,exist_ok=True)
    attempt_path=output/'tool-attempt.json'
    binding={'session_id':context['session_id'],'attempt_id':context['attempt_id'],
        'context_sha256':identity(context),'source':source,'current_best':best,
        'current_appearance':parent,'scope_id':scope['scope_id'],'design_id':design['design_id'],
        'decision':decision,'operation_stage':context['operation_stage'],'reserved_costs':reservation['costs']}
    record={'schema':SCHEMA,'state':'reserved','binding':binding,'automatic_retry':False,
        'human_approved':False,'image_generation_submissions_upper_bound':1}
    with attempt_path.open('x',encoding='utf8') as stream:
        json.dump(record,stream,ensure_ascii=False,indent=2);stream.flush();os.fsync(stream.fileno())
    started=time.perf_counter();asset_binding={'source_sha256':source['sha256'],'scope':scope,'design_id':design['design_id']}
    appearance=loaded=None
    try:
        appearance=load_asset(parent,**asset_binding);native=appearance['master']
        shaping=config.get('shaping_authorization')
        if shaping is not None and 'face' in shaping['request'].get('parts',{}):
            raise ValueError('Face shaping uses its checked deterministic geometry tool')
        options=config.get('provider_options') or {}
        if not isinstance(options,dict) or set(options)-OPTIONS:raise ValueError('Unsupported checked provider options')
        resolved=reference_provider.provider_config(**{key:options[key] for key in
            ('provider','model','api_base','api_url','api_key_env') if key in options})
        if box is not None and (aspect:=_provider_aspect(resolved)) is not None:box=_fit_context(core,box,size,aspect)
        family='qwen_image_2_1' if resolved.get('kind')=='comfyui' and str(resolved.get('registry_id','')).startswith('qwen_image') else 'gpt_image'
        uploaded=base if box is None else base.crop(tuple(box))
        input_path=output/'current-best-input.png';uploaded.save(input_path,format='PNG',icc_profile=deterministic_srgb_profile())
        local_core=None if core is None else [(core[0]-box[0])/uploaded.width,(core[1]-box[1])/uploaded.height,
            (core[2]-box[0])/uploaded.width,(core[3]-box[1])/uploaded.height]
        # Ownership comes from the real subject matte of this exact canvas, and
        # only where a photographic join has the support to carry it.
        subject=subject_record=None
        if core is not None and _join_analysis_ready((core[2]-core[0],core[3]-core[1])):
            subject,subject_record=_current_canvas_subject(config,best,base)
            if subject is not None:
                # Store exactly the quantised weights the join uses, so the
                # recorded digest reproduces the composite bit for bit.
                matte_path=output/'subject-matte.npy'
                np.save(matte_path,np.uint8(np.clip(subject*np.float32(255.)+np.float32(.5),0.,255.)),
                    allow_pickle=False)
                subject=(np.load(matte_path,allow_pickle=False).astype(np.float32)/np.float32(255.))
                subject_record.update(file='subject-matte.npy',sha256=sha256_file(matte_path),
                    dtype='uint8',weights='silhouette_score_over_255')
        else:
            whole=core is None
            subject_record={'schema':SUBJECT['schema'],'role':SUBJECT['role'],'available':False,
                'boundary_source':('whole_canvas_replacement' if whole
                    else 'authorized_core_below_photographic_join_support'),
                'ownership':('authorized_canvas_written_as_reserved' if whole
                    else 'authorized_core_written_as_reserved'),
                'reason':('whole_canvas_composite_has_no_join_boundary' if whole
                    else 'edit_region_below_photographic_join_support'),
                'matte_is_the_edit_coverage':False}
        subject_only,trigger=_scene_write_trigger(scope,decision)
        subject_record.setdefault('acquisition',{})
        subject_record['acquisition'].update({'available':bool(subject_record.get('available')),
            'reason':subject_record.get('reason'),
            'boundary_source':subject_record.get('boundary_source'),
            'ownership':subject_record.get('ownership')})
        subject_record.setdefault('authoritative','pre_join_declaration')
        subject_record.update(scene_write_trigger=trigger,ownership_rule=SUBJECT['ownership_rule'],
            scene_policy=scope.get('scene_policy'),
            scene_operations_authorized=[name for name in (decision.get('operations') or [])
                if name in SCENE_OPERATIONS],
            subject_only_round=bool(subject_only))
        record['subject_matte']=subject_record
        prompt=_prompt(scope,decision,local_core,family,shaping);provider_dir=output/'provider'
        arguments=['--root',str(Path(config['root']).resolve()),'--image',str(input_path),
            '--source',source['path'],'--sample','edited','--out-dir',str(provider_dir),'--prompt',prompt,'--preserve-alpha']
        for key,value in options.items():
            if value is not None:arguments.extend(['--'+key.replace('_','-'),str(value)])
        record.update(state='submitting',generator_input=_asset(input_path),core_box=core,context_box=box,
            current_best_size=list(size),compiled_prompt={'family':family,'prompt':prompt,'sha256':identity(prompt)})
        atomic_json(attempt_path,record);_cancel(config.get('cancel'))
        for item in (source,best):
            if _asset(item['path'])!={key:item[key] for key in ('path','sha256')}:raise ValueError('Tool input changed before provider submission')
        validate_asset(parent,**asset_binding)
        if resolved.get('kind')=='comfyui':
            import workflow_timing
            from beautygrpo_prepare import release_idle_for_stage
            def handoff_cancel():
                _cancel(config.get('cancel'))
            with workflow_timing.span('model_handoff',model_request=False,http_request=False):
                record['portrait_generation_memory_transition']=release_idle_for_stage(
                    next_stage_cpu_bytes=0,next_stage='generation',next_stage_gpu=True,cancel=handoff_cancel)
            atomic_json(attempt_path,record)
            _cancel(config.get('cancel'))
        code=reference_provider.main(arguments)
        if code:raise RuntimeError('Image provider did not produce a registered candidate')
        generated_path=provider_dir/'edited.png';provider_receipt=provider_dir/'edited.provenance.json'
        record.update(state='response_received',generated=_asset(generated_path),provider_receipt=_asset(provider_receipt))
        atomic_json(attempt_path,record);_cancel(config.get('cancel'))
        generated,patch,alpha=_generated_pixels(generated_path,uploaded.size)
        generated_size=list(generated.size)
        for item in (source,best):
            if _asset(item['path'])!={key:item[key] for key in ('path','sha256')}:raise ValueError('Tool input changed during generation')
        validate_asset(parent,**asset_binding)
        fusion=None
        if core is None and alpha is None:
            result=patch;coverage=None;fraction=1.
        elif core is None:
            core=box=[0,0,*size];coverage=alpha;result=native.copy()
            mixed=native*(1-alpha[...,None])+patch
            result[coverage>0]=mixed[coverage>0]
            fraction=float(np.count_nonzero(coverage)/(size[0]*size[1]))
        else:
            result,coverage,fusion=fuse_region(native,patch,alpha,core=core,box=box,subject=subject,
                subject_only=subject_only)
            fraction=float(np.count_nonzero(coverage)/(size[0]*size[1]))
        # Single source of truth: the receipt's boundary fields are the ones the
        # join actually wrote, copied from the join's own record. The pre-join
        # declaration stays only under 'acquisition'.
        _joined_subject_record(subject_record,fusion)
        if coverage is not None and not np.any(coverage>0):
            raise ValueError('Generated image has no visible coverage inside its permitted edit')
        record.update(core_box=core,context_box=box)
        generation={'image':record['generated'],'provider_receipt':record['provider_receipt'],
            'context_box':box,'core_box':core}
        asset=freeze_candidate(output/'appearance',kind='generated_full' if core is None else 'mixed_regional',
            master=result,parent=parent,coverage=coverage,generation=generation,
            cancel=config.get('cancel'),**asset_binding)
        loaded=load_asset(asset,**asset_binding);destination=loaded['preview_path'];candidate=_asset(destination)
        if core is not None:
            actual=np.asarray(_best_rgb(destination));old=np.asarray(base)
            if not np.array_equal(actual[coverage==0],old[coverage==0]):
                raise ValueError('Untouched native preview pixels changed during regional encoding')
        provider_readings=_recorded_provider_readings(native,patch,box,subject,core)
        composition={'schema':SCHEMA,'binding':binding,'working_domain':'display_linear_sRGB',
            'output_encoding':'float32_master_with_matching_RGB8_sRGB_preview','core_box':core,'context_box':box,
            'coverage':loaded['receipt']['edit_coverage'],
            'coverage_semantics':'current_canvas_edit_coverage_not_source_subject_alpha','join':fusion,
            'subject_matte':subject_record,
            'edit_permission':'full_canvas' if kind=='joint_edit' else 'bounded_region',
            'composition_mode':'opaque_replacement' if coverage is None else 'linear_premultiplied_rgba_over_parent',
            'generated_opacity':'opaque' if alpha is None else 'registered_alpha',
            'resampling':'bilinear_linear_RGB' if alpha is None else 'bilinear_linear_associated_RGB_and_alpha',
            'appearance_asset':asset,'parent_appearance':parent,
            'outside_core_float_identical_to_current_best':core is not None,
            'outside_core_pixels_identical_to_current_best':core is not None,
            'zero_coverage_float_identical_to_current_best':coverage is not None,
            'source_subject_RGB_restored':False,'generated_context_size':generated_size,
            'complete_candidate':candidate,'effective_coverage_fraction':fraction,
            'provider_readings':provider_readings,
            'identity_verified':False,'geometric_registration_verified':False,
            'lighting_and_depth_of_field_verified':False,'needs_visual_review':True,'human_approved':False}
        composition_path=output/'composition.json';atomic_json(composition_path,composition)
        record.update(state='complete',complete_candidate=candidate,appearance_asset=asset,
            composition_receipt=_asset(composition_path),elapsed_seconds=time.perf_counter()-started)
        atomic_json(attempt_path,record);_cancel(config.get('cancel'))
        provenance={'schema':SCHEMA,'tool':kind,
            'source':source,'current_best':best,'parent_appearance':parent,'appearance_asset':asset,
            'full_candidate':{**candidate,'size':list(size)},'generated_image':{**record['generated'],'size':generated_size},
            'provider_receipt':record['provider_receipt'],'composition_receipt':record['composition_receipt'],
            'tool_attempt':_asset(attempt_path),'automatic_retry':False,'human_approved':False,'needs_visual_review':True}
        return {'image':str(destination),'appearance_asset':asset,'provenance':provenance,
            'resources':{'image_submissions':1,'local_repairs':int(kind=='edit_region'),'wall_seconds':time.perf_counter()-started}}
    except BaseException as error:
        record.update(state='interrupted' if isinstance(error,(InterruptedError,KeyboardInterrupt)) else 'needs_attention',
            error={'type':type(error).__name__,'message':str(error)[:1200]},elapsed_seconds=time.perf_counter()-started)
        atomic_json(attempt_path,record)
        raise
    finally:
        _close_loaded(appearance,loaded)


def joint_edit(context,config):
    return _execute(context,config,'joint_edit')


def edit_region(context,config):
    return _execute(context,config,'edit_region')


def make_tools(root,output,*,scope,session_directory,provider_options=None,cancel=None,
               shaping_authorization=None,shaping_base_appearance=None,subject_matte=None):
    from photo_edit_session import Tool
    scope=validate_scope(scope)
    config={'root':str(Path(root).resolve()),'output':str(Path(output).resolve()),'scope':scope,
        'session_directory':str(Path(session_directory).resolve()),'provider_options':copy.deepcopy(provider_options or {}),
        'cancel':cancel,'shaping_authorization':copy.deepcopy(shaping_authorization),
        'shaping_base_appearance':copy.deepcopy(shaping_base_appearance),
        # None selects the route's own real subject silhouette, False records
        # the absence, and a callable injects another reviewed segmentation.
        'subject_matte':subject_matte}
    producer=identity({'code':{name:sha256_file(Path(__file__).parent/name) for name in
        ('photo_edit_tools.py','photo_appearance.py','photo_session_evidence.py','photo_editing.py','reference_provider.py','qwen_grid_diagnostic.py','raw_support.py','float_appearance.py','photo_shaping.py','photo_shaping_geometry.py','source_alpha.py')},
        'scope_id':scope['scope_id'],'provider_options':config['provider_options'],
        'shaping_authorization':shaping_authorization})
    if shaping_authorization is not None:
        from photo_shaping import shaping_operations,shaping_tool
        if shaping_tool(shaping_authorization)!='edit_region':
            raise ValueError('Generative shaping admits body-only requests')
        operations=tuple(shaping_operations(shaping_authorization))
    else:
        operations=tuple(scope['allowed_operations'])
    tools=[Tool('edit_region',operations,{'image_submissions':1,'local_repairs':1},producer,
             lambda context:edit_region(context,config),requires_region=True)]
    if shaping_authorization is not None:
        return tools
    if scope['work_type']!='retouch':
        tools.insert(0,Tool('joint_edit',operations,{'image_submissions':1},producer,
            lambda context:joint_edit(context,config)))
    from photo_focus_integration import make_focus_tool
    tools.append(make_focus_tool(root,output,scope=scope,session_directory=session_directory,cancel=cancel))
    from photo_development import make_development_tool
    tools.append(make_development_tool(root,output,scope=scope,session_directory=session_directory,cancel=cancel))
    return tools
