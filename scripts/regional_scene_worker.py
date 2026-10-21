"""One budgeted native ROI image; edit permission is enforced only by the CPU."""
from __future__ import annotations

from runtime_resources import external_python_popen

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from contextlib import contextmanager

import numpy as np
from PIL import Image

from source_scene_enhancement import asset, background, digest, pixels_sha, write
from cache_store import exclusive_file_lock

SCHEMA = 'regional_scene_refinement.v1'
SIZE = 1024
HALO = 128
MAX_CORE = SIZE-2*HALO
ALIGNMENT = {'minimum_anchor_pixels':1024,'maximum_exposure_EV':2.,
             'exposure_scope':'returned_image_normalization_to_unchanged_halo_not_artistic_relighting',
             'maximum_chroma_EV':.10,'minimum_linear_signal':1e-5,
             'adaptive_floor_fraction':.1,'low_frequency_sigma':3.,
             'anchor_filter_guard_sigmas':4.,
             'maximum_halo_median_residual_EV':.45,'maximum_halo_q90_residual_EV':1.25,
             'minimum_structured_halo_correlation':.5,'structured_halo_minimum_std_EV':.1,
             'policy':'photometric_normalization_halo_gain_and_local_canvas_check.v3'}


def authorization(run, root=None):
    from creative_task import normalize_task_intent, validate_task_plan
    intent=normalize_task_intent(run.get('task_intent') or (run.get('options') or {}).get('task_intent'))
    regional=intent.get('regional_refinement') or {'maximum_submissions':0,'regions':[]}
    reuse=regional.get('reuse_generated_from_run_id')
    if not regional.get('maximum_submissions') and not reuse:return intent, regional, None
    checked=validate_task_plan(run.get('task_plan') or (run.get('options') or {}).get('task_plan'),intent)
    if reuse:
        from creative_task import revalidate_regional_candidate_reuse
        proof=revalidate_regional_candidate_reuse(root,checked)
        if (proof!=checked.get('regional_candidate_reuse_authorization') or
                proof['source_sha256']!=run['source']['sha256'] or proof['reference_sha256']!=run['reference']['sha256']
                or checked['image_budget'].get('regional_refinement')!=0):
            raise ValueError('Known regional candidate reuse requires its exact zero-submission proof')
        return intent,regional,checked
    resume=checked.get('regional_resume_authorization')
    if regional.get('resume_from_run_id'):
        if root is None or not isinstance(resume,dict):
            raise ValueError('Regional continuation requires its checked zero-submission proof')
        from creative_task import revalidate_regional_resume
        if revalidate_regional_resume(root,checked)!=resume:
            raise ValueError('Regional continuation proof changed before execution')
        if (resume['source_sha256']!=run['source']['sha256'] or
                resume['reference_sha256']!=run['reference']['sha256']):
            raise ValueError('Regional continuation source/reference differs from the failed task')
    elif intent['mode']=='existing_reference':
        origin=(run.get('generation_budget') or {}).get('origin_run_id') or run.get('run_id')
        if resume is not None or not run.get('run_id') or origin!=run['run_id']:
            raise ValueError('A new explicit regional task must own a new independent run origin')
        if checked['inputs'].get('reference',{}).get('sha256')!=run['reference']['sha256']:
            raise ValueError('Explicit regional task must bind its selected existing reference')
    if (intent['mode'] not in ('create_reference','existing_reference') or intent['constraints'].get('background_mode')!='enhance_source'
            or checked['image_budget'].get('regional_refinement')!=1 or len(regional['regions'])!=1
            or regional['maximum_submissions']!=1
            or checked['inputs']['source']['sha256']!=run['source']['sha256']):
        raise ValueError('Regional generation requires one checked source-bound explicit budget')
    return intent, regional, checked


def resume_failure(root, checked, previous, binding):
    """Called only while holding the original origin's reservation lock."""
    proof=checked.get('regional_resume_authorization')
    if not proof:return None
    from creative_task import revalidate_regional_resume
    if revalidate_regional_resume(root,checked)!=proof:
        raise ValueError('Regional resume authorization changed under reservation lock')
    old=previous.get('binding') or {}
    if (previous.get('schema')!=SCHEMA or previous.get('state')!='failed_no_submission'
            or previous.get('model_submissions')!=0 or previous.get('request_started')
            or previous.get('prompt_id') or old.get('checked_plan_id')!=proof['previous_plan_id']
            or old.get('direction_id')!=proof['previous_direction_id']
            or any(old.get(key)!=binding[key] for key in ('source','reference','core_box','native_size'))):
        raise ValueError('Only an unchanged proved zero-submission regional failure can resume')
    return copy.deepcopy(previous)


def geometry(plan, region):
    box=region['core_box'];w,h=plan['canvas']['source_oriented_size']
    if (not isinstance(box,list) or len(box)!=4 or any(type(v) is not int for v in box)
            or not 0<=box[0]<box[2]<=w or not 0<=box[1]<box[3]<=h
            or box[2]-box[0]>MAX_CORE or box[3]-box[1]>MAX_CORE):
        raise ValueError('One safe native ROI must fit the canvas and 768px core limit')
    if plan['mask_roles']['foreground_alpha']['sha256']!=plan['source_baseline']['assets']['foreground_alpha']['sha256']:
        raise ValueError('Regional generation requires the original native subject coverage')
    cx,cy=(box[0]+box[2])//2,(box[1]+box[3])//2
    context=[cx-SIZE//2,cy-SIZE//2,cx+SIZE//2,cy+SIZE//2]
    from design_reconstruction import sample_asset
    # Native guard is tested before downsampling, preserving every 1px exclusion.
    guard=32;guardbox=(max(0,box[0]-guard),max(0,box[1]-guard),min(w,box[2]+guard),min(h,box[3]+guard))
    alpha=sample_asset(plan['mask_roles']['foreground_alpha'],'values',guardbox,(w,h))
    if np.any(alpha!=0): raise ValueError('Regional core must be at least 32 native pixels from any source coverage')
    return box,context


def _sample_context(plan, context):
    from design_reconstruction import sample_asset
    w,h=plan['canvas']['source_oriented_size'];l,t,r,b=context
    bounds=(max(0,l),max(0,t),min(w,r),min(h,b))
    values=sample_asset(background(plan)['assets']['background'],'values',bounds,(w,h))
    validity=sample_asset(background(plan)['assets']['validity'],'values',bounds,(w,h))
    if np.any(validity!=1):
        from display_transform import display_linear
        from source_scene_enhancement import transform_for
        baseline=plan['source_baseline']['assets']
        alpha=sample_asset(plan['mask_roles']['foreground_alpha'],'values',bounds,(w,h))
        observed=(sample_asset(baseline['foreground_premultiplied'],'values',bounds,(w,h))
            +sample_asset(baseline['old_environment'],'values',bounds,(w,h))*(1-alpha[...,None]))
        transform=transform_for(plan)
        if transform is not None:observed=display_linear(observed,transform)
        values=values.copy();values[alpha!=0]=observed[alpha!=0]
    return np.pad(values,((max(0,-t),max(0,b-h)),(max(0,-l),max(0,r-w)),(0,0)),mode='edge')


def local_prompt(region,direction):
    effect=direction.get('main_effect') or {}
    instruction=(region.get('instruction') or
                 (effect.get('description') if effect.get('mode')=='user_described' else None))
    if not instruction or not instruction.strip():
        raise ValueError('Regional editing requires a concrete local instruction; global portrait prompts are not local tasks')
    l,t,r,b=region['core_box'];cx,cy=(l+r)//2,(t+b)//2
    left,top,right,bottom=l-cx+SIZE//2,t-cy+SIZE//2,r-cx+SIZE//2,b-cy+SIZE//2
    return ('Edit the ONE supplied image, the exact LOCAL 1024x1024 background crop. '
        f'Only pixels inside x=[{left},{right}), y=[{top},{bottom}) may change; '
        'these are editing coordinates, NOT a visible rectangle to draw. '
        'Keep the ENTIRE 1024x1024 original image at its exact pixel scale and perspective. '
        'Do not shrink, zoom, crop, add white/black borders, frames, panels, masks or a new canvas. '
        'Return the complete local photograph alone. Do not reconstruct a complete portrait, '
        'or add a person, face, body, hands, costume, bag, text or comparison panels. Preserve the existing local '
        'lattice, leaves, objects, defocus and background layout. Keep unedited halo content unchanged. '
        'Implement only this one local instruction, once inside the allowed coordinates; no extra objects or copies elsewhere. '
        'Any requested paper butterflies are small non-luminous paper objects, not neon or magic.\nLOCAL INSTRUCTION:\n'
        +instruction.strip())


def blend_patch(base, generated, permission):
    from scipy.ndimage import distance_transform_edt
    if (base.shape!=generated.shape or base.shape[:2]!=permission.shape
            or not np.isfinite(generated).all() or np.any((generated<0)|(generated>1))):
        raise ValueError('Regional candidate must be finite SDR on its exact model canvas')
    distance=distance_transform_edt(np.pad(permission,1))[1:-1,1:-1]
    x=np.clip((distance-1)/16,0,1).astype(np.float32)
    weight=x*x*(3-2*x);weight[~permission]=0
    result=base.copy();active=weight>0
    result[active]=base[active]+(generated[active]-base[active])*weight[active,None]
    if not np.array_equal(result[~permission],base[~permission]):raise ValueError('Regional permission escaped')
    return result,weight


def align_candidate(base, generated, permission, reliable):
    """Anchor one ROI to the global native plate; never fit against new effects."""
    from source_scene_enhancement import adjust
    from scipy.ndimage import gaussian_filter
    from scipy.ndimage import distance_transform_edt
    reliable=np.asarray(reliable,bool)
    if generated.shape!=base.shape or permission.shape!=base.shape[:2] or reliable.shape!=permission.shape:
        raise ValueError('Regional alignment grids differ')
    if not np.isfinite(generated).all() or np.any((generated<0)|(generated>1)):
        raise ValueError('Regional output is not finite display-linear SDR')
    halo=(~permission)&reliable
    # Every lowpass sample's full kernel footprint stays outside the editable
    # core and unknown/subject pixels. New effects cannot bias its anchors.
    distance=distance_transform_edt(np.pad(halo,1))[1:-1,1:-1]
    halo &= distance>ALIGNMENT['low_frequency_sigma']*ALIGNMENT['anchor_filter_guard_sigmas']
    if int(halo.sum())<ALIGNMENT['minimum_anchor_pixels']:
        raise ValueError('Regional halo lacks reliable unchanged background anchors')
    luma=np.asarray([.2126,.7152,.0722],np.float32)
    by=gaussian_filter(base@luma,ALIGNMENT['low_frequency_sigma'])
    gy=gaussian_filter(generated@luma,ALIGNMENT['low_frequency_sigma'])
    floor=max(ALIGNMENT['minimum_linear_signal'],float(np.quantile(by[halo],.1))*ALIGNMENT['adaptive_floor_fraction'])
    anchor=halo&(by>floor)&(gy>floor)
    if int(anchor.sum())<ALIGNMENT['minimum_anchor_pixels']:
        raise ValueError('Regional halo lacks reliable unchanged background anchors')
    ev=float(np.clip(np.median(np.log2(by[anchor]/gy[anchor])),
                    -ALIGNMENT['maximum_exposure_EV'],ALIGNMENT['maximum_exposure_EV']))
    chroma=np.zeros(3,np.float32);channel_samples=[]
    for channel in range(3):
        bc=gaussian_filter(base[...,channel],ALIGNMENT['low_frequency_sigma'])
        gc=gaussian_filter(generated[...,channel],ALIGNMENT['low_frequency_sigma'])
        usable=anchor&(bc>floor)&(gc>floor)
        channel_samples.append(int(usable.sum()))
        if usable.sum()>=ALIGNMENT['minimum_anchor_pixels']:
            chroma[channel]=float(np.median(np.log2(bc[usable]/gc[usable])))-ev
    chroma-=float(chroma@luma)
    chroma=np.clip(chroma,-ALIGNMENT['maximum_chroma_EV'],ALIGNMENT['maximum_chroma_EV'])
    gain=np.exp2(ev+chroma).astype(np.float32)
    aligned=adjust(generated,gain,np.ones(permission.shape,np.float32))
    ay=gaussian_filter(aligned@luma,ALIGNMENT['low_frequency_sigma'])
    residual=np.abs(np.log2((ay[anchor]+floor)/(by[anchor]+floor)))
    q50,q90=np.quantile(residual,[.5,.9]);expected=np.log2(by[anchor]+floor);actual=np.log2(ay[anchor]+floor)
    structured=float(expected.std())>=ALIGNMENT['structured_halo_minimum_std_EV']
    correlation=float(np.corrcoef(expected,actual)[0,1]) if structured and actual.std()>1e-6 else (0. if structured else None)
    incompatible=(q50>ALIGNMENT['maximum_halo_median_residual_EV'] or q90>ALIGNMENT['maximum_halo_q90_residual_EV']
            or structured and (not np.isfinite(correlation) or correlation<ALIGNMENT['minimum_structured_halo_correlation']))
    report={'profile':copy.deepcopy(ALIGNMENT),'anchor_pixels':int(anchor.sum()),'exposure_EV':ev,
        'chroma_EV':chroma.tolist(),'gain':gain.tolist(),'new_core_content_used_as_anchor':False,
        'adaptive_signal_floor':floor,'chroma_anchor_pixels':channel_samples,
        'halo_median_residual_EV':float(q50),'halo_q90_residual_EV':float(q90),
        'halo_structure_correlation':correlation if correlation is None or np.isfinite(correlation) else None,
        'halo_content_compatibility_checked':True,'halo_content_compatible':not bool(incompatible),
        'semantic_acceptance_verified':False,'geometry_corrected':False,'joint_multiROI_optimization':False}
    if incompatible:
        error=ValueError('Regional output does not preserve the local canvas halo; wrong composition/structure or excessive relighting')
        error.diagnostics=report
        raise error
    return aligned,report


def _context_reliability(plan,context):
    from design_reconstruction import sample_asset
    w,h=plan['canvas']['source_oriented_size'];l,t,r,b=context
    bounds=(max(0,l),max(0,t),min(w,r),min(h,b))
    alpha=sample_asset(plan['mask_roles']['foreground_alpha'],'values',bounds,(w,h))
    v=sample_asset(background(plan)['assets']['validity'],'values',bounds,(w,h))
    return np.pad((alpha==0)&(v==1),((max(0,-t),max(0,b-h)),(max(0,-l),max(0,r-w))),constant_values=False)


@contextmanager
def native_handoff(run,output):
    """Release only the managed idle provider, retaining the original failure."""
    try:
        yield
    finally:
        active_error=sys.exc_info()[0] is not None
        try:
            from start_qwen_image import release_for_run
            from runtime_memory import native_analysis_cpu_bytes
            receipt=release_for_run(run,next_stage_cpu_bytes=native_analysis_cpu_bytes(run),next_stage='native_analysis')
        except BaseException as error:
            receipt={'state':'handoff_failed','error_type':type(error).__name__,'error':str(error)}
            write(Path(output)/'native-handoff.json',receipt)
            if not active_error:raise
        else:write(Path(output)/'native-handoff.json',receipt)


def validate_saved(record, run, plan):
    from reference_appearance_stack import verify_asset
    verify_asset(record['receipt']);r=json.loads(Path(record['receipt']['path']).read_text('utf8'))
    if (r.get('schema')!=SCHEMA or r.get('state')!='completed'
            or r['binding']['source']!=run['source']['sha256'] or r['binding']['reference']!=run['reference']['sha256']
            or r['binding']['native_size']!=plan['canvas']['source_oriented_size']):
        raise ValueError('Saved regional result is not a completed matching source/reference/canvas')
    for key in ('patch','input_background','output_background','validity','raw_provenance','raw_candidate',
                'input_context','reliable_halo'):
        verify_asset(r[key])
    if r.get('conditioning_policy')=='single_native_ROI_CPU_permission.v3':
        for key in ('condition_manifest','permission_mask'):verify_asset(r[key])
        provenance=json.loads(Path(r['raw_provenance']['path']).read_text(encoding='utf8'))
        actual=provenance['provider_contract']['comfyui'];uploads=actual.get('condition_inputs') or []
        graph=actual['request']['prompt']
        encoders=[node['inputs'] for node in graph.values() if node['class_type']=='CosplayTextEncodeQwenImage21']
        if (len(uploads)!=1 or len(encoders)!=1 or
                [key for key in encoders[0] if key.startswith('images.')] != ['images.image_1']
                or r.get('permission_condition_transmitted') is not False or r.get('condition_image_count')!=1):
            raise ValueError('Saved regional generation did not use the declared single-image CPU-permission contract')
    geometry(plan,{'core_box':r['binding']['core_box']})
    from design_reconstruction import sample_asset
    from float_appearance import srgb_to_linear
    core=r['binding']['core_box'];cx,cy=r['binding']['context_box'][:2]
    base=sample_asset(r['input_background'],'values',core,tuple(r['binding']['native_size']))
    if pixels_sha(base)!=r['binding']['base_core_pixels_sha256']:raise ValueError('Regional receipt base pixel identity changed')
    with Image.open(r['raw_candidate']['path']) as im:
        if im.size!=(SIZE,SIZE):raise ValueError('Saved regional raw model canvas changed')
        raw=srgb_to_linear(np.asarray(im.convert('RGB'),np.float32)/255)
    with np.load(r['input_context']['path'],allow_pickle=False) as z:context=z['values'].copy()
    with np.load(r['reliable_halo']['path'],allow_pickle=False) as z:reliable=z['values'].astype(bool)
    if pixels_sha(context)!=r['binding']['context_pixels_sha256']:raise ValueError('Saved regional context changed')
    permission=np.zeros((SIZE,SIZE),bool);permission[core[1]-cy:core[3]-cy,core[0]-cx:core[2]-cx]=True
    if pixels_sha(permission)!=r['binding']['permission_pixels_sha256']:raise ValueError('Saved regional permission changed')
    aligned,alignment=align_candidate(context,raw,permission,reliable)
    if alignment!=r['alignment']:raise ValueError('Regional frozen alignment differs from its actual halo evidence')
    expected,_=blend_patch(context,aligned,permission)
    expected=expected[core[1]-cy:core[3]-cy,core[0]-cx:core[2]-cx]
    with np.load(r['patch']['path'],allow_pickle=False) as z:patch=z['values']
    if not np.array_equal(patch,expected):raise ValueError('Saved regional patch differs from its actual model output and permission blend')
    return r


def _apply(plan, core, pixels, output):
    from native_array_storage import NativeArrayRowWriter
    from design_reconstruction import sample_asset
    w,h=plan['canvas']['source_oriented_size'];writer=NativeArrayRowWriter(output,(h,w,3))
    try:
        for top in range(0,h,128):
            end=min(h,top+128);values=sample_asset(background(plan)['assets']['background'],'values',(0,top,w,end),(w,h)).copy()
            y0,y1=max(top,core[1]),min(end,core[3])
            if y0<y1:values[y0-top:y1-top,core[0]:core[2]]=pixels[y0-core[1]:y1-core[1]]
            writer.write(values.astype(np.float32))
        writer.publish()
    finally:writer.close()
    return asset(output,**{k:v for k,v in background(plan)['assets']['background'].items() if k not in {'path','sha256'}})


def _replay(saved, root, run, plan, output):
    from design_reconstruction import sample_asset
    r=validate_saved(saved,run,plan);core=r['binding']['core_box'];native=tuple(plan['canvas']['source_oriented_size'])
    base=sample_asset(background(plan)['assets']['background'],'values',core,native)
    if pixels_sha(base)!=r['binding']['base_core_pixels_sha256']:
        raise ValueError('Saved regional base changed; refusing to drop/reinvent the effect with zero budget')
    with np.load(r['patch']['path'],allow_pickle=False) as z:pixels=z['values'].copy()
    if pixels.shape!=base.shape or not np.isfinite(pixels).all():raise ValueError('Saved regional native patch is invalid')
    result=copy.deepcopy(plan);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    new= _apply(plan,core,pixels,output/'replayed-background.npz')
    background(result)['assets']['background']=new
    # New replay evidence binds its new transport without changing historical provenance.
    replay={**r,'input_background':copy.deepcopy(background(plan)['assets']['background']),
            'output_background':new,'saved_regional_parent':copy.deepcopy(saved['receipt']),
            'replay_state':'saved regional result reused','new_model_submissions':0}
    path=output/'regional-replay.json';write(path,replay)
    return result,{'state':'saved regional result reused','receipt':asset(path),'new_model_submissions':0}


def prepare(root,run,plan,output,*,saved=None,cancel=None):
    intent,regional,checked=authorization(run,root=root)
    if regional.get('reuse_generated_from_run_id'):
        return reuse_generated(root,run,plan,output,checked)
    if saved:
        previous=validate_saved(saved,run,plan)
        if (not regional.get('maximum_submissions') or
                (regional['regions'][0]['core_box']==previous['binding']['core_box']
                 and checked['creative_direction']['direction_id']==previous['binding']['direction_id'])):
            return _replay(saved,root,run,plan,output)
    if not regional.get('maximum_submissions'):return plan,None
    region=regional['regions'][0];geometry(plan,region)
    from beautygrpo_prepare import release_idle_for_stage
    handoff=release_idle_for_stage(next_stage_cpu_bytes=0,next_stage='generation',next_stage_gpu=True)
    if handoff.get('state') in {'busy_service_untouched','unowned_service_untouched'}:
        raise RuntimeError('Regional generation cannot enter while Beauty owns an unresolved request')
    from current_model_registry import load_registry
    from runtime_resources import code_path,isolated_python_environment
    output=Path(output)/digest({'run_id':run.get('run_id'),'regional':regional,
                              'plan_id':checked['plan_id']})[:20]
    output.mkdir(parents=True,exist_ok=True)
    request=output/'request.json';write(request,{'root':str(Path(root).resolve()),'run':run,'plan':plan,'handoff':handoff})
    python=load_registry(root,registry_path=run.get('options',{}).get('model_registry'))['python']
    with (output/'worker.log').open('wb') as log:
        process=external_python_popen([python,'-B','-X','utf8',str(code_path('regional_scene_worker.py')),
            '--request',str(request),'--output',str(output)],stdout=log,stderr=subprocess.STDOUT,
            env=isolated_python_environment(),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:
            while process.poll() is None:
                if cancel and cancel():process.terminate();raise InterruptedError('Regional generation cancelled; reservation retained')
                time.sleep(.2)
        finally:
            if process.poll() is None:process.terminate()
            process.wait(timeout=10)
    if process.returncode:raise RuntimeError('Regional preparation failed; no automatic retry; see '+str(output/'worker.log'))
    result=json.loads((output/'result.json').read_text('utf8'))
    return result['plan'],result['regional']


def reuse_generated(root,run,plan,output,checked):
    """CPU-only explicit reuse of a known returned image, never of an unknown request."""
    from creative_task import revalidate_regional_candidate_reuse
    from reference_appearance_stack import verify_asset
    from float_appearance import srgb_to_linear
    from native_array_storage import save_native_array
    proof=revalidate_regional_candidate_reuse(root,checked)
    if proof!=checked.get('regional_candidate_reuse_authorization'):
        raise ValueError('Known regional candidate proof changed before CPU reuse')
    regional=checked['intent']['regional_refinement'];region=regional['regions'][0]
    core,_=geometry(plan,region)
    packet=proof['assets']
    for value in packet.values():verify_asset(value)
    reservation=Path(root)/'outputs/regional-reservations'/proof['origin_run_id']/'regional-refinement.json'
    if __import__('asset_identity').sha256_file(reservation)!=proof['reservation_sha256']:
        raise ValueError('Known regional reservation changed after authorization')
    historical=json.loads(reservation.read_text(encoding='utf8'));binding=historical['binding']
    context=binding['context_box']
    if (proof.get('original_core_box',binding['core_box'])!=binding['core_box']
            or proof.get('requested_core_box',core)!=core
            or not context[0]<=core[0]<core[2]<=context[2]
            or not context[1]<=core[1]<core[3]<=context[3]):
        raise ValueError('Requested CPU permission must remain inside the original model context')
    base=_sample_context(plan,context)
    permission=np.zeros((SIZE,SIZE),bool);cx,cy=context[:2];l,t,r,b=core
    permission[t-cy:b-cy,l-cx:r-cx]=True
    original_core=binding['core_box'];ol,ot,orr,ob=original_core
    original_permission=np.zeros((SIZE,SIZE),bool)
    original_permission[ot-cy:ob-cy,ol-cx:orr-cx]=True
    if (binding['context_pixels_sha256']!=pixels_sha(base)
            or binding['base_core_pixels_sha256']!=pixels_sha(base[ot-cy:ob-cy,ol-cx:orr-cx])
            or binding['permission_pixels_sha256']!=pixels_sha(original_permission)):
        raise ValueError('Known regional pixels do not match the current source scene base/permission')
    with np.load(packet['input_context']['path'],allow_pickle=False) as z:actual_context=z['values']
    if not np.array_equal(actual_context,base):raise ValueError('Known regional input context differs from the current native base')
    with np.load(packet['reliable_halo']['path'],allow_pickle=False) as z:reliable=z['values'].astype(bool)
    if not np.array_equal(reliable,_context_reliability(plan,context)):
        raise ValueError('Known regional halo ownership changed')
    with Image.open(packet['raw_candidate']['path']) as image:
        if image.size!=(SIZE,SIZE):raise ValueError('Known regional model canvas differs')
        generated=srgb_to_linear(np.asarray(image.convert('RGB'),np.float32)/255)
    aligned,alignment=align_candidate(base,generated,permission,reliable)
    after,_=blend_patch(base,aligned,permission);pixels=after[t-cy:b-cy,l-cx:r-cx].copy()
    output=Path(output)/('known-'+proof['prompt_id']);output.mkdir(parents=True,exist_ok=True)
    patch_path=output/'native-patch.npz';save_native_array(patch_path,pixels)
    permission_path=output/'cpu-permission.png';Image.fromarray(np.uint8(permission)*255).save(permission_path)
    target=_apply(plan,core,pixels,output/'regional-background.npz');result=copy.deepcopy(plan)
    background(result)['assets']['background']=target
    applied_binding={**copy.deepcopy(binding),'core_box':core,
        'base_core_pixels_sha256':pixels_sha(base[t-cy:b-cy,l-cx:r-cx]),
        'permission_pixels_sha256':pixels_sha(permission),
        'checked_plan_id':checked['plan_id'],'direction_id':checked['creative_direction']['direction_id']}
    receipt={'schema':SCHEMA,'state':'completed','binding':applied_binding,'model_binding':copy.deepcopy(binding),
        'input_background':copy.deepcopy(background(plan)['assets']['background']),
        'output_background':target,'validity':copy.deepcopy(background(plan)['assets']['validity']),
        'patch':asset(patch_path),'raw_candidate':packet['raw_candidate'],'raw_provenance':packet['raw_provenance'],
        'input_context':packet['input_context'],'reliable_halo':packet['reliable_halo'],'alignment':alignment,
        'conditioning_policy':'single_native_ROI_CPU_permission.v3','condition_manifest':packet['condition_manifest'],
        'permission_mask':asset(permission_path),'original_model_permission_mask':packet['permission_mask'],
        'permission_condition_transmitted':False,'condition_image_count':1,
        'full_portrait_conditioned':False,'full_global_prompt_used':False,'mask_is_conditioning_only':False,
        'new_model_submissions':0,'prompt_id':proof['prompt_id'],'known_generated_reuse_authorization':copy.deepcopy(proof),
        'consumer_checked_plan_id':checked['plan_id'],'outside_permission_exact':True,
        'source_Fp_alpha_changed':False,'source_subject_geometry_changed':False,'source_geometry_scope':'source_subject_only',
        'background_geometry_preservation_verified':False,'geometry_review_required':True,
        'visual_acceptance':False,'quality_status':'needs_visual_review','replay_state':'known generated result reused'}
    path=output/'regional-receipt.json';write(path,receipt)
    record={'state':'known generated result reused','receipt':asset(path),'new_model_submissions':0}
    validate_saved(record,run,result)
    return result,record


def prepare_local(root,run,plan,output):
    """Isolated provider process: durable journal before/at the sole HTTP submit."""
    intent,regional,checked=authorization(run,root=root)
    if checked is None:raise ValueError('No positive checked regional budget')
    region=regional['regions'][0];core,context=geometry(plan,region)
    direction=checked['creative_direction'];prompt=local_prompt(region,direction)
    base=_sample_context(plan,context);l,t,r,b=core;cx,cy=context[:2]
    permission=np.zeros((SIZE,SIZE),bool);permission[t-cy:b-cy,l-cx:r-cx]=True
    binding={'source':run['source']['sha256'],'reference':run['reference']['sha256'],
        'native_size':plan['canvas']['source_oriented_size'],'core_box':core,'context_box':context,
        'base_core_pixels_sha256':pixels_sha(base[t-cy:b-cy,l-cx:r-cx]),
        'context_pixels_sha256':pixels_sha(base),'permission_pixels_sha256':pixels_sha(permission),
        'direction_id':direction['direction_id'],'prompt':prompt,'checked_plan_id':checked['plan_id'],
        'producer_code':{name:__import__('asset_identity').sha256_file(Path(__file__).with_name(name))
                         for name in ('regional_scene_worker.py','source_scene_enhancement.py','reference_provider.py','comfyui_adapter.py')}}
    origin=(run.get('generation_budget') or {}).get('origin_run_id') or run.get('run_id')
    resume_proof=checked.get('regional_resume_authorization')
    if resume_proof and origin!=resume_proof['origin_run_id']:
        raise ValueError('Regional continuation must consume its original origin reservation')
    if not isinstance(origin,str) or not origin or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in origin):
        raise ValueError('Regional reservation requires an explicit safe run origin')
    ledger=Path(root)/'outputs/regional-reservations'/origin/'regional-refinement.json';ledger.parent.mkdir(parents=True,exist_ok=True)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with exclusive_file_lock(ledger.with_suffix('.lock')):
        previous_failure=None
        if ledger.is_file():
            old=json.loads(ledger.read_text('utf8'))
            if resume_proof:
                previous_failure=resume_failure(root,checked,old,binding)
            elif old.get('state')=='completed' and old.get('binding')==binding:
                return _replay({'receipt':old['receipt']},root,run,plan,output/'cached')
            elif old.get('state') in {'reserved','submitted'}:
                old['state']='unknown';write(ledger,old)
            if previous_failure is None:
                raise RuntimeError('Regional budget already reserved/consumed/unknown; never resubmit automatically')
        elif resume_proof:
            raise ValueError('Regional continuation cannot create a replacement for a missing original reservation')
        state={'schema':SCHEMA,'state':'reserved','binding':binding,'maximum_submissions':1,
               'automatic_retry':False,'model_submissions':0}
        if previous_failure is not None:
            state.update(previous_failure=previous_failure,
                attempt_history=[*previous_failure.get('attempt_history',[]),
                    {k:v for k,v in previous_failure.items() if k not in ('attempt_history','previous_failure')}],
                resume_authorization=copy.deepcopy(resume_proof),consumer_run_id=run['run_id'])
        write(ledger,state)
        try:
            from float_appearance import linear_to_srgb,srgb_to_linear
            def image(values,path):Image.fromarray(np.uint8(np.clip(linear_to_srgb(values),0,1)*255+.5)).save(path)
            first=output/'roi.png';image(base,first)
            mask=output/'permission.png';Image.fromarray(np.uint8(permission)*255).save(mask)
            from native_array_storage import save_native_array
            context_path=output/'input-context.npz';save_native_array(context_path,base)
            reliable_path=output/'reliable-halo.npz';save_native_array(reliable_path,_context_reliability(plan,context).astype(np.float32))
            manifest=output/'conditions.json';write(manifest,{
                'inputs':[{'role':'source',**asset(first),'coordinate_space':'source_canvas_roi'}],
                'permission_mask':{**asset(mask),'transmitted_to_model':False,
                                   'authority':'CPU_native_core_composite_only'}})
            import reference_provider
            opts=run.get('options') or {};provider=opts.get('provider') or run.get('reference',{}).get('provider')
            if provider not in {'qwen_image_2_1','qwen_image_2_1_fast','qwen_image_2_1_uncensored','qwen_image_2_1_uncensored_fast'}:
                raise ValueError('Regional refinement requires the managed Qwen multi-reference route')
            original_request=reference_provider.comfyui_adapter.request_generation
            class Events(list):
                def append(self,event):
                    super().append(event)
                    if event.get('event')=='comfyui_submitted':
                        state.update(state='submitted',model_submissions=1,prompt_id=event['prompt_id']);write(ledger,state)
            def journal(*args,**kwargs):
                args=list(args);parent=args[5];events=Events()
                args[5]=events
                state['request_started']=True;write(ledger,state)
                try:return original_request(*args,**kwargs)
                finally:parent.extend(events)
            reference_provider.comfyui_adapter.request_generation=journal
            source=Path(run['source']['path']);source=source if source.is_absolute() else Path(root)/source
            call=['--root',str(root),'--image',str(first),'--source',str(source),'--sample','regional',
                '--out-dir',str(output/'provider'),'--prompt',prompt,'--provider',provider,
                '--inputs-json',str(manifest)]
            for flag,key in (('--model','model'),('--api-base','api_base'),('--api-url','api_url')):
                if opts.get(key):call.extend([flag,str(opts[key])])
            try:
                with native_handoff(run,output):
                    code=reference_provider.main(call)
                    if not code:
                        state.update(provider_returned=True,state='generated_returned',
                            raw_candidate=asset(output/'provider/regional.png'),
                            raw_provenance=asset(output/'provider/regional.provenance.json'))
                        write(ledger,state)
            finally:reference_provider.comfyui_adapter.request_generation=original_request
            if code:raise RuntimeError('Regional provider did not complete')
            candidate=output/'provider/regional.png';provenance=output/'provider/regional.provenance.json'
            p=json.loads(provenance.read_text('utf8'));events=p['provider_contract']['transport']['events']
            submitted=[e for e in events if e.get('event')=='comfyui_submitted']
            if len(submitted)!=1 or state.get('model_submissions')!=1:raise ValueError('Regional submission evidence is not exactly one')
            with Image.open(candidate) as im:
                if im.size!=(SIZE,SIZE):raise ValueError('Regional model changed its exact ROI canvas; no geometry inference')
                generated=srgb_to_linear(np.asarray(im.convert('RGB'),np.float32)/255)
            with np.load(reliable_path,allow_pickle=False) as z:reliable=z['values'].astype(bool)
            aligned,alignment=align_candidate(base,generated,permission,reliable)
            after,weight=blend_patch(base,aligned,permission)
            patch=after[t-cy:b-cy,l-cx:r-cx].copy()
            from native_array_storage import save_native_array
            patch_path=output/'native-patch.npz';save_native_array(patch_path,patch)
            target=_apply(plan,core,patch,output/'regional-background.npz');result=copy.deepcopy(plan)
            background(result)['assets']['background']=target
            receipt={'schema':SCHEMA,'state':'completed','binding':binding,'input_background':copy.deepcopy(background(plan)['assets']['background']),
                'output_background':target,'validity':copy.deepcopy(background(plan)['assets']['validity']),
                'patch':asset(patch_path),'raw_candidate':asset(candidate),'raw_provenance':asset(provenance),
                'input_context':asset(context_path),'reliable_halo':asset(reliable_path),'alignment':alignment,
                'conditioning_policy':'single_native_ROI_CPU_permission.v3',
                'condition_manifest':asset(manifest),'permission_mask':asset(mask),
                'permission_condition_transmitted':False,'condition_image_count':1,
                'full_portrait_conditioned':False,'full_global_prompt_used':False,
                'new_model_submissions':1,'prompt_id':submitted[0]['prompt_id'],'mask_is_conditioning_only':False,
                'outside_permission_exact':True,'source_Fp_alpha_changed':False,'source_geometry_changed':False,
                'source_geometry_scope':'source_subject_only','source_subject_geometry_changed':False,
                'background_geometry_preservation_verified':False,'geometry_review_required':True,
                'visual_acceptance':False,'quality_status':'needs_visual_review'}
            receipt_path=output/'regional-receipt.json';write(receipt_path,receipt)
            regional={'state':'completed','receipt':asset(receipt_path),'new_model_submissions':1}
            state.update(state='completed',receipt=regional['receipt']);write(ledger,state)
            return result,regional
        except BaseException as error:
            state.update(state=('generated_rejected' if state.get('provider_returned') else
                                'unknown' if state.get('request_started') else 'failed_no_submission'),
                         error_type=type(error).__name__,error=str(error))
            if getattr(error,'diagnostics',None):state['rejection_diagnostics']=error.diagnostics
            write(ledger,state)
            raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    request=json.loads(args.request.read_text('utf8'))
    plan,regional=prepare_local(Path(request['root']),request['run'],request['plan'],args.output)
    write(args.output/'result.json',{'plan':plan,'regional':regional})


if __name__=='__main__':main()
