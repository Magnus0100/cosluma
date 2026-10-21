"""Model-selected photographic focus/exposure/relative colour on current float.

No RAW re-decode, source RGB restoration, geometry operation or image model is
performed. The masks describe broad photographic settings, never subject alpha.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from asset_identity import sha256_file
from current_model_registry import atomic_json
from photo_editing import identity,validate_scope
import workflow_timing

TOOL='develop_photograph'
SCHEMA='cosluma_photographic_development.v1'
PARAMETERS={'adjustments':{'type':'photographic_adjustments','min_items':1,'max_items':4,
    'fields':{'region':'normalized_canvas_box','exposure_ev':{'minimum':-2.,'maximum':2.},
        'warmth_ev':{'minimum':-1.,'maximum':1.},'tint_ev':{'minimum':-1.,'maximum':1.},
        'feather':{'minimum':0.,'maximum':.5,'unit':'region_short_edge'},
        'focus_sigma':{'minimum':0.,'maximum':.02,'unit':'native_short_edge','optional':True}}}}
_LUMA=np.asarray([.2126,.7152,.0722],np.float32)


def producer_identity():
    return {name:sha256_file(Path(__file__).with_name(name)) for name in (
        'photo_development.py','photo_appearance.py','photo_edit_session.py','photo_session_evidence.py')}


def operator_for(parameters):
    from photo_edit_session import validate_parameters
    parameters=validate_parameters(parameters,PARAMETERS)
    return {'name':'photographic_exposure_balance.v2','working_domain':'display_linear_sRGB',
        'parameters':parameters,'field_composition':'ordered_soft_overrides',
        'processing_order':['gaussian_current_parent_and_soft_region_blend','exposure_and_relative_colour_balance'],
        'focus_method':'gaussian_of_all_current_parent_RGB_in_selected_regions',
        'focus_composition':'ordered_soft_overrides_from_same_immutable_parent',
        'focus_boundary':'reflect_at_canvas_edge','focus_radius':'ceil(4*sigma_pixels)',
        'focus_default_sigma':0.,'physical_depth_of_field':False,
        'colour_method':'relative_linear_srgb_luminance_preserving_balance',
        'exposure_method':'whitepoint_preserving_bounded_ev_gain',
        'gamut_mapping':'luminance_preserving_desaturation','physical_relighting':False,
        'raw_redecoded':False,'source_subject_RGB_restored':False,'geometry_resampled':False,
        'quality_improvement_claimed':False}


def _cancel(cancel):
    if cancel and cancel():raise InterruptedError('Photographic adjustment cancelled; current complete version retained')


def _mask_band(item,width,height,top,bottom):
    x0,y0,x1,y1=item['region']
    left,right=float(x0)*width,float(x1)*width
    upper,lower=float(y0)*height,float(y1)*height
    xx=np.arange(width,dtype=np.float32)[None,:]+np.float32(.5)
    yy=np.arange(top,bottom,dtype=np.float32)[:,None]+np.float32(.5)
    inside=(xx>=left)&(xx<right)&(yy>=upper)&(yy<lower)
    feather=float(item['feather'])*min(right-left,lower-upper)
    if feather==0:return inside.astype(np.float32)
    distance=np.minimum(np.minimum(xx-left,right-xx),np.minimum(yy-upper,lower-yy))
    fraction=np.clip(distance/feather,0,1).astype(np.float32)
    return np.where(inside,np.float32(.5)-np.float32(.5)*np.cos(np.float32(math.pi)*fraction),0).astype(np.float32)


def _focused_band(master,sigma,mask,top,bottom,*,cancel=None):
    """Filter only the selected row span/columns plus the complete Gaussian halo.

    Each region reads the same immutable parent. Reflect padding is used only at
    actual canvas edges; the retained ROI never reaches an artificial tile edge.
    """
    from scipy.ndimage import gaussian_filter
    columns=np.flatnonzero(np.any(mask>0,axis=0))
    if not len(columns):return None
    height,width=master.shape[:2];radius=math.ceil(4*sigma)
    left,right=int(columns[0]),int(columns[-1])+1
    halo_left=max(0,left-radius);halo_right=min(width,right+radius)
    halo_top=max(0,top-radius);halo_bottom=min(height,bottom+radius)
    _cancel(cancel)
    filtered=gaussian_filter(np.asarray(master[halo_top:halo_bottom,halo_left:halo_right]),
        sigma=(sigma,sigma),axes=(0,1),radius=(radius,radius),mode='reflect',output=np.float32)
    _cancel(cancel)
    return left,right,filtered[top-halo_top:bottom-halo_top,left-halo_left:right-halo_left]


def iter_development_bands(master,parameters,*,cancel=None,band_rows=128):
    """Yield (top, float RGB, permitted coverage, statistics) in bounded bands."""
    parameters=operator_for(parameters)['parameters']
    if (not isinstance(master,np.ndarray) or master.dtype!=np.float32 or master.ndim!=3
            or master.shape[2]!=3 or min(master.shape[:2])<1 or type(band_rows) is not int or not 1<=band_rows<=512):
        raise ValueError('Photographic development requires a native float32 RGB master and bounded bands')
    height,width=master.shape[:2]
    for top in range(0,height,band_rows):
        _cancel(cancel);bottom=min(height,top+band_rows);source=np.asarray(master[top:bottom])
        if not np.isfinite(source).all() or np.any(source<0) or np.any(source>1):
            raise ValueError('Photographic input must be finite normalized display-linear sRGB')
        fields=np.zeros((bottom-top,width,3),np.float32)
        coverage=np.zeros((bottom-top,width),np.float32)
        focused=source.copy();has_focus=False
        for item in parameters['adjustments']:
            mask=_mask_band(item,width,height,top,bottom)
            settings=np.asarray([item['exposure_ev'],item['warmth_ev'],item['tint_ev']],np.float32)
            fields*=1-mask[...,None];fields+=mask[...,None]*settings
            coverage=coverage+mask*(1-coverage)
            sigma=float(item.get('focus_sigma',0))*min(width,height)
            if sigma>0:
                filtered=_focused_band(master,sigma,mask,top,bottom,cancel=cancel)
                if filtered is not None:
                    left,right,blurred=filtered;weight=mask[:,left:right,None]
                    current=focused[:,left:right]
                    current[:]=current*(1-weight)+blurred*weight
                    has_focus=True
            elif has_focus:
                # A later zero setting overrides earlier focus, as it does EV.
                focused[:]=focused*(1-mask[...,None])+source*mask[...,None]
        active=np.any(fields!=0,axis=2)&(coverage>0)
        result=focused;stats={'changed_pixels':0,'covered_pixels':int(np.count_nonzero(coverage)),
            'unrecoverable_black_pixels':int(np.count_nonzero(np.all(focused==0,axis=2)&active)),
            'linear_gain_would_exceed_sdr_pixels':0,'gamut_compressed_pixels':0}
        if np.any(active):
            ev,warm,tint=(fields[...,i] for i in range(3))
            colour_ev=np.stack((warm+tint*.5,-tint,-warm+tint*.5),axis=-1)
            balanced=focused*np.exp2(colour_ev)
            luminance=np.sum(focused*_LUMA,axis=2)
            altered=np.sum(balanced*_LUMA,axis=2)
            scale=np.divide(luminance,altered,out=np.ones_like(luminance),where=altered>0)
            balanced*=scale[...,None]
            gain=np.exp2(ev)
            mapped=gain*luminance/(1+(gain-1)*luminance)
            exposure=np.divide(mapped,luminance,out=np.ones_like(luminance),where=luminance>0)
            balanced*=exposure[...,None]
            peak=np.max(balanced,axis=2)
            amount=np.divide(1-mapped,peak-mapped,out=np.ones_like(mapped),where=peak>mapped)
            amount=np.clip(amount,0,1)
            compressed=mapped[...,None]+(balanced-mapped[...,None])*amount[...,None]
            compressed=np.clip(compressed,0,1).astype(np.float32)
            result[active]=compressed[active]
            stats['linear_gain_would_exceed_sdr_pixels']=int(np.count_nonzero((gain*luminance>1)&active))
            stats['gamut_compressed_pixels']=int(np.count_nonzero((peak>1)&active))
        # Preserve the parent exactly outside authorized coverage, independently
        # of filtering and floating-point arithmetic in feathered regions.
        result[coverage==0]=source[coverage==0]
        stats['changed_pixels']=int(np.count_nonzero(np.any(result!=source,axis=2)))
        yield top,result,coverage,stats


def develop_float(master,parameters,*,cancel=None,band_rows=128):
    """Evaluate one combined setting field; no full-frame field per adjustment."""
    result=np.empty_like(master);coverage=np.empty(master.shape[:2],np.float32)
    stats={'native_pixels':int(master.shape[0]*master.shape[1]),'changed_pixels':0,'covered_pixels':0,
        'unrecoverable_black_pixels':0,'linear_gain_would_exceed_sdr_pixels':0,'gamut_compressed_pixels':0,
        'band_rows':band_rows,'physical_relighting_verified':False}
    for top,band,mask,counts in iter_development_bands(master,parameters,cancel=cancel,band_rows=band_rows):
        result[top:top+len(band)]=band;coverage[top:top+len(band)]=mask
        for key,value in counts.items():stats[key]+=value
    stats['no_change']=stats['changed_pixels']==0
    if np.all(coverage==1):coverage=1
    elif np.all(coverage==0):coverage=0
    return result,coverage,stats


def _reservation(context,config):
    import photo_edit_session as session
    from photo_edit_tools import _typed_inputs
    from photo_session_evidence import resolve_file,resolve_asset
    state=session._read(Path(config['session_directory']));scope=validate_scope(config['scope'])
    if context.get('scope')!=scope or context.get('design')!=state['design']:
        raise ValueError('Photographic operation differs from checked scope/design')
    images=context.get('images') or []
    if [row.get('role') for row in images]!=['source','best']:
        raise ValueError('Photographic development operates on the actual current canvas')
    source,_=_typed_inputs(context,state,scope,state['design']);best=images[1]
    row=next((r for r in state['candidates'] if r['candidate_id']==session.current_canvas_id(state)),None)
    attempt=next((r for r in state['attempts'] if r['attempt_id']==context.get('attempt_id')),None)
    decision=context.get('decision') or {};stage=context.get('operation_stage')
    descriptor=(state.get('runtime') or {}).get('tools',{}).get(TOOL) or {}
    costs=dict.fromkeys(session.COUNTERS,0);costs.update(tool_calls=1,improvement_steps=session.tool_improvement_cost(state))
    if (state['status']!='active' or state['phase']!='tool' or state['session_id']!=context.get('session_id')
            or state['user_intent']!=context.get('user_intent') or stage not in {'initial','improvement'}
            or state['pending_decision']!=decision or decision.get('action')!='tool' or decision.get('tool')!=TOOL
            or not decision.get('operations') or set(decision['operations'])-set(scope['allowed_operations'])
            or not attempt or attempt['kind']!='tool' or attempt['status']!='reserved'
            or attempt['context']!=context or attempt['context_sha256']!=identity(context) or attempt['costs']!=costs
            or descriptor.get('producer_identity')!=config['producer_identity']
            or descriptor.get('parameter_schema')!=PARAMETERS or not row
            or resolve_file(state,row['image'])!={k:best[k] for k in ('path','sha256')}
            or resolve_asset(state,row['appearance_asset'])!=context['appearance_assets']['best']):
        raise ValueError('Photographic tool requires its actual model decision, typed current input and zero-image reservation')
    parameters=session.validate_parameters(decision.get('parameters'),PARAMETERS)
    prior=next((r for r in reversed(state['attempts']) if r.get('kind')=='decide' and r.get('status')=='completed'),None) or {}
    response=prior.get('response') or {}
    if {k:v for k,v in response.items() if k not in {'provenance','observed_images'}}!=decision:
        raise ValueError('Photographic execution lacks the actual completed model selection')
    return state,source,best,row,parameters


@workflow_timing.timed('photographic_development',model_request=False,http_request=False)
def develop_photograph(context,config):
    from photo_appearance import load_asset,freeze_candidate,validate_asset
    from photo_edit_tools import _asset,_close_loaded
    _cancel(config.get('cancel'));state,source,best,row,parameters=_reservation(context,config)
    parent=context['appearance_assets']['best'];binding={'source_sha256':source['sha256'],
        'scope':config['scope'],'design_id':context['design']['design_id']}
    output=Path(config['output']).resolve()/context['attempt_id'];output.mkdir(parents=True,exist_ok=True)
    path=output/'development-attempt.json'
    record={'schema':SCHEMA,'state':'reserved','session_id':context['session_id'],'attempt_id':context['attempt_id'],
        'context_sha256':identity(context),'input_parent':parent,'decision':context['decision'],
        'image_generation_submissions':0,'automatic_retry':False,'human_approved':False}
    with path.open('x',encoding='utf8') as stream:
        json.dump(record,stream,ensure_ascii=False,indent=2);stream.flush();os.fsync(stream.fileno())
    started=time.perf_counter();loaded=finished=None
    try:
        loaded=load_asset(parent,**binding)
        values,coverage,stats=develop_float(loaded['master'],parameters,cancel=config.get('cancel'))
        operation={'operator':operator_for(parameters),'stats':stats,'producer':producer_identity()}
        _cancel(config.get('cancel'));validate_asset(parent,**binding)
        asset=freeze_candidate(output/'appearance',kind='photographic_adjustment',master=values,
            parent=parent,coverage=coverage,photographic_operation=operation,cancel=config.get('cancel'),**binding)
        finished=load_asset(asset,**binding);receipt=finished['receipt']
        candidate=_asset(finished['preview_path']);owned=receipt['photographic_operation']['operation_receipt']
        operation_receipt={'path':str(Path(asset['path'])/owned['file']),'sha256':owned['sha256']}
        _cancel(config.get('cancel'));validate_asset(parent,**binding);validate_asset(asset,**binding)
        record.update(state='complete',appearance_asset=asset,complete_candidate=candidate,
            operation_receipt=operation_receipt,elapsed_seconds=time.perf_counter()-started)
        atomic_json(path,record)
        return {'image':str(finished['preview_path']),'appearance_asset':asset,
            'provenance':{'schema':SCHEMA,'tool':TOOL,'source':source,'current_best':best,
                'parent_appearance':parent,'appearance_asset':asset,'operation_receipt':operation_receipt,
                'tool_attempt':_asset(path),'full_candidate':{**candidate,'size':receipt['canvas_size']},
                'provider':'local_photographic_development','model':TOOL,'generation_id':None,
                'image_generation_submissions':0,'automatic_retry':False,'human_approved':False,
                'physical_relighting_verified':False},
            'resources':{'image_submissions':0,'local_repairs':0,'wall_seconds':time.perf_counter()-started}}
    except BaseException as error:
        record.update(state='interrupted' if isinstance(error,(InterruptedError,KeyboardInterrupt)) else 'needs_attention',
            error={'type':type(error).__name__,'message':str(error)[:1200]});atomic_json(path,record)
        raise
    finally:_close_loaded(loaded,finished)


def make_development_tool(root,output,*,scope,session_directory,cancel=None):
    from photo_edit_session import Tool,current_canvas_id
    from photo_session_evidence import resolve_asset,resolve_file
    from photo_appearance import validate_asset
    scope=validate_scope(scope);producer=identity({'code':producer_identity(),'scope_id':scope['scope_id'],'parameters':PARAMETERS})
    config={'root':str(Path(root).resolve()),'output':str(Path(output).resolve()),'scope':scope,
        'session_directory':str(Path(session_directory).resolve()),'cancel':cancel,'producer_identity':producer}
    def input_check(state):
        if state.get('scope')!=scope:return False
        row=next((r for r in state.get('candidates',[]) if r['candidate_id']==current_canvas_id(state)),None)
        if not row or not row.get('appearance_asset'):return False
        asset=resolve_asset(state,row['appearance_asset']);receipt=validate_asset(asset,
            source_sha256=state['binding']['source']['sha256'],scope=scope,design_id=state['binding']['design_id'])
        return receipt['files']['preview']['sha256']==resolve_file(state,row['image'])['sha256']
    operations=tuple(item for item in ('portrait_retouch','relight') if item in scope['allowed_operations'])
    return Tool(TOOL,operations,{},producer,lambda context:develop_photograph(context,config),
        requires_region=False,parameter_schema=copy.deepcopy(PARAMETERS),input_check=input_check)
