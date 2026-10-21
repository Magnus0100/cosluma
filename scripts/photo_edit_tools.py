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


def _mask(size):
    w,h=size;feather=max(1,min(64,math.ceil(min(w,h)*.08)))
    x=np.minimum(np.arange(w),np.arange(w)[::-1])/feather
    y=np.minimum(np.arange(h),np.arange(h)[::-1])/feather
    value=np.clip(np.minimum(y[:,None],x[None,:]),0,1).astype(np.float32)
    return value*value*(3-2*value)


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
        if core is None and alpha is None:
            result=patch;coverage=None;fraction=1.
        elif core is None:
            core=box=[0,0,*size];coverage=alpha;result=native.copy()
            mixed=native*(1-alpha[...,None])+patch
            result[coverage>0]=mixed[coverage>0]
            fraction=float(np.count_nonzero(coverage)/(size[0]*size[1]))
        else:
            x0,y0,x1,y1=core
            local=patch[y0-box[1]:y1-box[1],x0-box[0]:x1-box[0]]
            mask=_mask((x1-x0,y1-y0))
            coverage=np.zeros(native.shape[:2],np.float32)
            actual=mask if alpha is None else mask*alpha[y0-box[1]:y1-box[1],x0-box[0]:x1-box[0]]
            coverage[y0:y1,x0:x1]=actual;result=native.copy()
            prior=native[y0:y1,x0:x1]
            mixed=prior*(1-actual[...,None])+local*mask[...,None]
            result[y0:y1,x0:x1][actual>0]=mixed[actual>0]
            fraction=float(np.count_nonzero(coverage)/(size[0]*size[1]))
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
        composition={'schema':SCHEMA,'binding':binding,'working_domain':'display_linear_sRGB',
            'output_encoding':'float32_master_with_matching_RGB8_sRGB_preview','core_box':core,'context_box':box,
            'coverage':loaded['receipt']['edit_coverage'],
            'coverage_semantics':'current_canvas_edit_coverage_not_source_subject_alpha',
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
               shaping_authorization=None,shaping_base_appearance=None):
    from photo_edit_session import Tool
    scope=validate_scope(scope)
    config={'root':str(Path(root).resolve()),'output':str(Path(output).resolve()),'scope':scope,
        'session_directory':str(Path(session_directory).resolve()),'provider_options':copy.deepcopy(provider_options or {}),
        'cancel':cancel,'shaping_authorization':copy.deepcopy(shaping_authorization),
        'shaping_base_appearance':copy.deepcopy(shaping_base_appearance)}
    producer=identity({'code':{name:sha256_file(Path(__file__).parent/name) for name in
        ('photo_edit_tools.py','photo_appearance.py','photo_session_evidence.py','photo_editing.py','reference_provider.py','qwen_grid_diagnostic.py','raw_support.py','float_appearance.py','photo_shaping.py','photo_shaping_geometry.py')},
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
