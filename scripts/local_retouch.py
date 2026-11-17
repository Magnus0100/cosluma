"""Explicit, masked local edits of a saved photograph with the selected engine."""
from __future__ import annotations

from datetime import datetime
import json
import math
import os
import shutil
from pathlib import Path
import time
import urllib.parse
import uuid

import cv2
import numpy as np
from PIL import Image

from image_io import (fit_generated, image_size, open_photo, open_rgb, save_image, save_jpeg,
    save_json, to_preview)
from model_clients import download, request, post_json, superresolve
from face_beauty import _upload


LOCAL_DEFAULTS = {'url':'http://127.0.0.1:8188',
    'model':'qwen_image_2.1_int8_convrot.safetensors','clip':'qwen3vl_8b_bf16.safetensors',
    'vae':'qwen_image_2.1_vae_bf16.safetensors','steps':40,'resolution':1024,'seed':42}
# Optional second backend for the same masked edit. Each engine keeps its own saved settings,
# so selecting one never rewrites the other's configured address or weight filenames.
# 2048 area is the ~4 MP canvas of the official MiniMax-H3-Image edit graph: the service pre-scales the
# picture being edited onto that canvas before conditioning. A saved 4096 is left exactly as stored.
H3_DEFAULTS = {'url':'http://127.0.0.1:8195',
    'model':'minimax_h3_image_pruned_int8_convrot.safetensors',
    'clip':'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors',
    'vae':'minimax_h3_image_vae_fp16.safetensors','steps':6,'resolution':2048,'seed':42}
LOCAL_KEYS = ('url','model','clip','vae','steps','resolution','seed')
LOCAL_ENGINES = {'qwen':{'label':'Qwen-Image-2.1','node':'CosLumaQwenLocalEdit','defaults':LOCAL_DEFAULTS},
    'h3':{'label':'MiniMax-H3-Image','node':'CosLumaH3ImageEdit','defaults':H3_DEFAULTS,
        'card':'https://huggingface.co/Comfy-Org/MiniMax-H3-Image/blob/892de7cb61df49753bf15145a711154803a355cb/README.md'}}
TASKS = {'hair_repair':('补发与露网修补',
        'Repair only the marked wig gaps, exposed wig net, broken hair sections or missing strands. '
        'Continue the adjacent hair color, strand thickness, texture and hairstyle. Do not add wind or redesign the hairstyle.'),
    'hair_shape':('发束造型',
        'Give the selected hair expressive, natural styling suited to the person and photograph, '
        'following the user\'s request and the original hair color and texture.'),
    'clothing':('衣物动态',
        'Give the selected costume fabric expressive, natural movement in the wind, '
        'suited to the person\'s pose and the photograph. Use the original fabric material and design.')}
# Positive descriptions of the finished appearance, used only when the user left the brief empty. The
# model redraws the whole photograph from noise, so naming a defect can put that defect into the result.
H3_TASKS = {'hair_repair':
        'Repair the hair into continuous, natural strands that match the adjacent color and texture.',
    'hair_shape':
        'Shape the hair into flowing, natural strands in the direction the user asks for.',
    'clothing':'Give the costume fabric expressive, natural movement in the wind, suited to the person\'s pose.'}


def h3_prompt(task, brief):
    """One instruction for the edited view: the final appearance the user wants.

    The graph conditions on pictures only, so the text names <Picture 1> and then says what the
    finished photograph should look like. The brief is passed through as entered - the user's own
    intent and any negations are never rewritten - and nothing describes the software's writeback,
    because the model redraws the whole photograph and the compositor is not its concern.
    """
    return 'Edit <Picture 1>. '+((brief or '').strip() or H3_TASKS[task])


def local_engine(config):
    """The explicitly selected engine and its parameters; a missing engine keeps Qwen."""
    config = dict(config or {})
    engine = config.get('engine') or 'qwen'
    if engine not in LOCAL_ENGINES:
        raise ValueError('请选择 Qwen-Image-2.1 或 MiniMax-H3-Image 作为局部修饰引擎')
    if engine == 'qwen':
        values = {key:config.get(key,LOCAL_DEFAULTS[key]) for key in LOCAL_KEYS}
    else:
        stored = config.get('h3') or {}
        values = {key:stored.get(key,H3_DEFAULTS[key]) for key in LOCAL_KEYS}
    return {'engine':engine,**values}


def _edit_box(mask):
    yy,xx = np.where(mask>0)
    if not len(xx):
        raise ValueError('请先涂出需要修改的范围')
    height,width = mask.shape
    padding = max((xx.max()-xx.min()+1)*.5,(yy.max()-yy.min()+1)*.5,min(width,height)*.12)
    return tuple(int(value) for value in (max(0,np.floor(xx.min()-padding)),
        max(0,np.floor(yy.min()-padding)),min(width,np.ceil(xx.max()+padding+1)),
        min(height,np.ceil(yy.max()+padding+1))))


def canvas_resolution(size, requested, sr_scale):
    """Choose one 32-aligned native canvas that a single SR pass can cover.

    Both image encoders round each aspect-preserving dimension to 32 pixels.
    Account for that rounding before generation, rather than enlarging a short return.
    The connection's resolution remains the minimum, and its saved settings are unchanged.
    """
    width,height = size
    ratio = math.sqrt(width/height)
    required = max((width/sr_scale+16)/ratio,(height/sr_scale+16)*ratio)
    return max(requested,math.ceil(required/32)*32)


def _blend_region(mask, size):
    """The pixels one blend can touch: the tight selection bbox plus one background halo.

    The single-pixel halo is needed because _gradient_writeback reads one pixel outside the
    selection as a fixed boundary value and distanceTransform needs the surrounding zeros to
    measure the selection's own thickness. Projected row/column flags avoid the full-size
    index array np.nonzero would build. An empty selection keeps the recorded crop so the
    existing "empty mask" error is still the one reported.
    """
    support = mask > 0
    rows = np.flatnonzero(support.any(axis=1))
    columns = np.flatnonzero(support.any(axis=0))
    if not rows.size or not columns.size:
        return 0,0,size[0],size[1]
    return (max(int(columns[0])-1,0),max(int(rows[0])-1,0),
        min(int(columns[-1])+2,size[0]),min(int(rows[-1])+2,size[1]))


def _detail_fit(detail_size, size):
    """Describe how the recorded model return maps onto the native crop.

    Equal aspect keeps the previous downsample exactly; a different aspect is
    centre-cropped instead of being stretched onto the crop.
    """
    width,height = int(detail_size[0]),int(detail_size[1])
    if width < size[0] or height < size[1]:
        raise ValueError('已记录的生成结果不足以覆盖原尺寸局部；已有输出保留')
    # Exact integer cross-multiplication: only a truly identical aspect keeps the
    # previous downsample, so a near-miss can never be recorded as exact.
    if width*size[1] == height*size[0]:
        return 'exact aspect; lanczos downsample to the recorded native crop',False
    return ('equal-aspect centre crop %dx%d to %dx%d; lanczos downsample, no non-uniform stretch'
        % (width,height,size[0],size[1])),True


def prepare_local_output(directory, sr_config, progress=None):
    """Cover the native crop with one configured SR pass, never RGB enlargement."""
    report = progress or (lambda message:None)
    directory = Path(directory)
    state = json.loads((directory/'session.json').read_text(encoding='utf-8'))
    x0,y0,x1,y1 = state['box']
    size = (x1-x0,y1-y0)
    source = directory/'model-output.png'
    native = image_size(source)
    if max(size[0]/native[0],size[1]/native[1])>1 and not sr_config:
        raise ValueError('局部模型原生输出不足以覆盖原尺寸修改区域，请在连接设置中配置超分模型；原始输出已保留。')
    returned,records = superresolve(source,size,sr_config,directory,report)
    dimensions = image_size(returned)
    if dimensions[0]<size[0] or dimensions[1]<size[1]:
        raise ValueError(f'一次超分返回 {dimensions[0]}×{dimensions[1]}，不足以覆盖局部 {size[0]}×{size[1]}；请缩小修改范围后主动生成。已有输出保留。')
    state.update(model_output_dimensions=list(native),detail_file=os.path.relpath(returned,directory),
        detail_dimensions=list(dimensions),writeback_dimensions=list(size),
        detail_fit=_detail_fit(dimensions,size)[0],blend_feather='selection thickness',
        superresolution={'model':sr_config['model'],'passes':records} if records else None)
    save_json(directory/'session.json',state)
    return state


def generate_local(base, root, mask, guide, task, brief, reference, config, progress=None, *, pixels=None,
        parent_version=None, sr_config=None, llm_config, head_config=None, protected=None, library_path=None):
    report = progress or (lambda message:None)
    if task not in TASKS:
        raise ValueError('请选择局部修饰任务')
    config = local_engine(config)
    engine = LOCAL_ENGINES[config['engine']]
    directory = Path(root)/('retouch-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
    directory.mkdir()
    state = {'status':'running','task':task,'task_label':TASKS[task][0],'brief':brief,'reference':reference,
        'source':str(Path(base).resolve()),'engine':config['engine'],
        'config':{key:config[key] for key in ('engine',)+LOCAL_KEYS},
        'parent_version':parent_version,
        'model_input_bit_depth':8,'model_output_bit_depth':8,'output_bit_depth':16}
    save_json(directory/'session.json',state)
    endpoint = config['url'].rstrip('/')
    try:
        report('读取已保存版本，准备处理画布与涂选范围')
        if pixels is None:
            pixels = open_photo(base)
        height,width = pixels.shape[:2]
        selected = cv2.resize(np.asarray(mask.convert('L')),(width,height),interpolation=cv2.INTER_NEAREST)
        protected = (protected or Image.new('L',mask.size)).convert('L').resize(
            (width,height),Image.Resampling.NEAREST)
        selected = selected.copy()
        selected[np.asarray(protected)>0] = 0
        tight = _edit_box(selected)  # the same check: an empty selection still errors here
        shape_edit = task in ('clothing','hair_shape')
        kind,label_name = ('hair','头发') if task=='hair_shape' else ('cloth','衣物')
        # Styling uses the full photograph; repairing a small wig defect retains its detail view.
        hair_detail = config['engine']=='h3' and task=='hair_repair'
        full_canvas = shape_edit
        box = (0,0,width,height) if full_canvas else tight
        x0,y0,x1,y1 = box
        requested_resolution = config['resolution']
        if sr_config:
            config['resolution'] = canvas_resolution((x1-x0,y1-y0),requested_resolution,sr_config.get('scale',4))
        state['config'].update(resolution=config['resolution'],requested_resolution=requested_resolution)
        # Keep the user's exact selected support. Feathering only happens
        # inside that support, so every outside float32 pixel is unchanged.
        cropped = selected[y0:y1,x0:x1]
        save_image(Image.fromarray(cropped),directory/'mask.png')
        save_image(Image.fromarray(cropped),directory/'generation-mask.png')
        save_image(protected.crop(box),directory/'protected.png')
        source = to_preview(pixels[y0:y1,x0:x1])
        save_image(source,directory/'input.png')
        # Detail edits retain full-photo context; only the primary view is written back.
        # Qwen also reads a mask reference. H3 receives photographs only.
        label = '<Picture {0}>' if config['engine']=='h3' else 'Image {0}'
        if config['engine']=='h3':
            roles = ['image1: selected detail crop at original pixels' if hair_detail
                else 'image1: full saved photograph']
        else:
            roles = ['image1: full saved photograph' if full_canvas else 'image1: selected crop']
        context = None
        if config['engine']=='h3':
            images = [source]
            prompt = h3_prompt(task,brief)
        else:
            model_mask = Image.fromarray(cropped).convert('RGB')
            roles.append('image2: white identifies the selected edit region in image1')
            prompt = (label.format(1)+' is the source photograph. '+label.format(2)+
                ' identifies the selected edit region in white. '+TASKS[task][1]+
                ' Return the edited view of '+label.format(1)+', with the same framing and no visible marks. '
                'User request: '+brief.strip())
            images = [source,model_mask]
        if not full_canvas:
            context = to_preview(pixels,max_edge=1536)
            save_image(context,directory/'context.png')
            images.append(context)
            roles.append('image'+str(len(images))+': full photograph, reference context only (maximum long edge 1536)')
            prompt += (' '+label.format(len(images))+' is the full photograph for context; '
                'return the edited detail view of '+label.format(1)+'.')
        if guide.getbbox():
            marked = source.convert('RGBA')
            drawing = guide.resize((width,height),Image.Resampling.NEAREST).crop(box)
            marked.alpha_composite(drawing)
            if config['engine']=='qwen' and not full_canvas:
                # Keep all user inputs within the node's four image slots: detail + mask +
                # context + optional appearance reference. The clean source is saved separately.
                images[0] = marked.convert('RGB')
                roles[0] += '; includes the user\'s colored edit marks'
                guide_index = 1
                save_image(images[0],directory/'marked-input.png')
            else:
                images.append(marked.convert('RGB'))
                guide_index = len(images)
                roles.append('image'+str(guide_index)+': user-colored edit marks on the same view')
            prompt += (' '+label.format(guide_index)+' shows the user\'s edit marks; '
                'apply the requested edit and remove the marks in the finished photograph.')
            save_image(drawing,directory/'guide.png')
        if reference:
            with open_rgb(reference) as image:
                image.thumbnail((1536,1536),Image.Resampling.LANCZOS)
                images.append(image.copy())
            save_image(images[-1],directory/'reference.png')
            roles.append('appearance reference only, not the person to render')
            prompt += (f' {label.format(len(images))} is an appearance reference only; '
                f'keep the person and scene from {label.format(1)}.')
        if shape_edit:
            prompt = f'Edit {label.format(1)}, the full photograph. '+TASKS[task][1]+' User request: '+brief.strip()
            if config['engine']=='qwen':
                noun = 'hair' if task=='hair_shape' else 'garment'
                roles[1] = f'image2: source {noun} locator only; white identifies the existing part, not its future shape or movement extent'
                prompt += f' Image 2 identifies the original {noun}; its new shape may extend beyond the white locator.'
            if guide.getbbox():
                prompt += f' {label.format(guide_index)} shows the user\'s shape marks; remove marks in the finished photograph.'
            if reference:
                prompt += f' {label.format(len(images))} is an appearance reference only.'
        from generation_prompts import (write_generation_prompt, write_prompt_candidate,
            review_prompt_candidate)
        from image_io import preview_data_url
        def prompt_images():
            attached = [(label.format(index)+': '+role,preview_data_url(image,1536))
                for index,(role,image) in enumerate(zip(roles,images),1)]
            if shape_edit and config['engine']=='h3':
                attached.append(('仅供编写与审查定位目标'+label_name+'的用户涂选，不发送给生图模型',
                    preview_data_url(Image.fromarray(cropped).convert('RGB'),1536)))
            return attached
        task_inputs = {'task':task,'user_request':brief.strip(),
            'template_prompt':prompt,'image_roles':roles}
        if shape_edit:
            from local_regions import part_regions, part_references, union_masks
            from prompt_library import ReferenceLibrary
            if not head_config:
                raise ValueError(TASKS[task][0]+'需要连接中已配置的 SAM3.1 分割服务')
            # Locate source parts only. The model decides their new shape in the full photograph;
            # the compositor derives writeback from the actual generated hair or garments.
            intent = Image.fromarray(cropped)
            save_image(intent,directory/'intent-mask.png')
            library = ReferenceLibrary(library_path)
            instruction = ('根据照片与涂选识别要编辑的原有'+label_name+'部件，查阅参考库中与本任务相关的质感、受光或风感描述。'
                '参考按本次任务取用，条目中服务于基础制作的不变要求不覆盖用户明确发起的局部造型。'
                'prompt 用简短英文表达用户想要的效果与适合的质感，具体形状由生图模型结合整图决定。'
                '只在用户明确指定时描述目标位置或形状；范围保护由软件负责。'
                '返回 JSON 结构为 {"prompt":"英文生图提示词","parts":[{'
                '"part":"原片部件",'
                '"segment_query":"具体英文部件名词，不带计数",'
                '"protected_objects":["原片实际头饰、前景道具或硬质配饰的简短英文类别名词"],'
                '"attachment_point":[0,0]}]}。'
                '每个独立目标部件分别列入 parts，单个部件也使用此数组。'
                'attachment_point 仅定位原片可见连接处，以整图宽高各 0 到 1000 表示，不规划新形状。'
                'protected_objects 填最多四类原片实际头饰、前景道具与硬质配饰；'
                '脸、耳与手臂由软件单独提取。')
            candidate = write_prompt_candidate(config['engine']+'_local',prompt_images(),
                llm_config,directory,task=task_inputs,instruction=instruction,progress=report,references=library)
            parts = candidate['parts']
            structure = {'parts':parts}
            state['edit_structure'] = structure
            save_json(directory/'edit-structure.json',structure)
            report('提取原'+label_name+'部件与前景保护区')
            references = part_references(intent,parts,head_config.get('input_max_edge',2048))
            source_parts = []
            objects = list(dict.fromkeys(obj for part in parts for obj in part['protected_objects']))
            for index,(part,part_intent) in enumerate(zip(parts,references)):
                extracted = part_regions(source,{**part,'protected_objects':objects},head_config,
                    directory,'source-'+str(index),reference_mask=part_intent,protect=index==0,task=task)
                source_parts.append(extracted[kind])
                if index==0:
                    foreground = extracted['protected']
            source_part = union_masks(source_parts)
            save_image(source_part,directory/('source-'+kind+'.png'))
            save_image(foreground,directory/'source-protected.png')
            save_image(foreground,directory/'source-preserved.png')
            source_part_view = source_part.resize((width,height),Image.Resampling.NEAREST).crop(box)
            source_part_views = [part_mask.resize((width,height),Image.Resampling.NEAREST).crop(box)
                for part_mask in source_parts]
            preserved_view = foreground.resize((width,height),Image.Resampling.NEAREST).crop(box)
            manual_view = protected.crop(box)
            save_image(source_part_view,directory/'generation-mask.png')
            if config['engine']=='qwen':
                images[1] = source_part_view.convert('RGB')
            candidate = review_prompt_candidate(config['engine']+'_local',candidate,prompt_images(),
                llm_config,directory,task={**task_inputs,'edit_structure':structure,
                    'selected_references':list(library.loaded.values())},
                instruction=instruction,progress=report)
        else:
            from prompt_library import ReferenceLibrary
            library = ReferenceLibrary(library_path)
            candidate = write_generation_prompt(config['engine']+'_local',prompt_images(),
                llm_config,directory,task=task_inputs,progress=report,references=library,
                instruction='补发只修复涂选缺口与露网，按需借鉴参考库中的发丝质感和受光，不加入造型任务。')
        prompt = candidate['prompt']
        names = [_upload(image,config) for image in images]
        # Only the pictures go to the model: no mask node, no sampler noise_mask and no grid tail on the
        # desktop path, so the edit is the plain full-context graph the released node documents.
        graph = {'8':{'class_type':engine['node'],'inputs':{
            **{key:config[key] for key in ('model','clip','vae','steps','resolution','seed')},
            'prompt':prompt,'filename_prefix':'CosLuma/local-'+uuid.uuid4().hex}}}
        for index,name in enumerate(names,1):
            node = str(10+index)
            graph[node] = {'class_type':'LoadImage','inputs':{'image':name}}
            graph['8']['inputs']['image'+str(index)] = [node,0]
        if config['engine']=='h3':
            method = (engine['label']+' native image edit on the selected detail crop, with the full photograph '
                'as a context reference; only the float32 masked writeback is limited to the selection'
                ) if hair_detail else (engine['label']+' full-context native image edit; only the float32 '
                'masked writeback is limited to the selection')
        else:
            method = engine['label']+' separate mask reference; software masked float32 writeback'
        state.update(box=list(box),source_dimensions=[width,height],prompt=prompt,
            input_scope='full_saved_photo' if full_canvas else 'detail_crop_with_full_photo_context',
            writeback_scope='painted_mask',model_inputs=roles,
            input_dimensions=list(source.size),draft_mask_dimensions=list(mask.size),
            mask_dimensions=[x1-x0,y1-y0],method=method,
            text_encoder_device='cuda:0',text_encoder_offload_device='cpu',text_encoder_dtype='bfloat16')
        state['generation_mask_file'] = str((directory/'generation-mask.png').resolve())
        if context is not None:
            state['context_dimensions'] = list(context.size)
        if config['engine']=='h3':
            # The recorded roles are the pictures actually sent; the selection is not one of them.
            state.update(input_scope='detail_crop_with_full_photo_context' if hair_detail else 'full_saved_photo',
                mask_usage='software float32 writeback only; the model receives no mask',
                mask_role='final_writeback',
                sampling={'steps':config['steps'],
                    'executed_denoise_steps':config['steps'],
                    'distilled_grid':'checkpoint PDD grid, %d intervals' % config['steps'],
                    'sampler_protection':'none'},
                model_inputs=roles)
            if context is not None:
                # the 8-bit context preview that was actually sent, never the full 6K pixels
                state['context_dimensions'] = list(context.size)
            # One fusion strategy for every new H3 session, whatever the task. Saved records without
            # this field keep their deterministic historical behaviour.
            state['blend_mode'] = 'poisson_gradient'
        if config['engine']=='qwen':
            # QwenImage21Cache keeps a lossless KV cache in CPU memory; the H3 engine has no such cache.
            state.update(cache_device='cpu',cache_dtype='lossless')
        save_json(directory/'session.json',state)
        save_json(directory/'workflow.json',graph)
        detail = str(config['steps'])+' 步'
        if config['engine']=='h3':
            detail = '完整 %d 段蒸馏网格 · 实际 %d 次去噪' % (config['steps'],config['steps'])
        report(engine['label']+' · '+TASKS[task][0]+' · '+detail+' · 面积分辨率 '+str(config['resolution']))
        submission = post_json(endpoint+'/prompt',{'prompt':graph,'client_id':'cosluma-'+uuid.uuid4().hex})
        save_json(directory/'submission.json',submission)
        identifier = submission['prompt_id']
        while True:
            history = json.loads(request(endpoint+'/history/'+urllib.parse.quote(identifier)))
            if identifier in history:
                result = history[identifier]
                save_json(directory/'history.json',result)
                status = result.get('status',{})
                if status.get('status_str')=='error':
                    error = '; '.join(row[1].get('exception_message','') for row in status.get('messages',[])
                        if row[0]=='execution_error')
                    raise RuntimeError(engine['label']+' 局部修饰失败：'+error)
                outputs = result.get('outputs',{}).get('8',{}).get('images')
                if outputs:
                    download(endpoint+'/view?'+urllib.parse.urlencode(outputs[0]),directory/'model-output.png')
                    state['model'] = result['outputs']['8']['model'][0]
                    break
                if status.get('completed'):
                    raise RuntimeError(engine['label']+' 局部修饰未返回图片；已保留服务记录')
            time.sleep(1)
        # Decode now; a corrupt model image must not be reported as ready.
        with open_rgb(directory/'model-output.png') as image:
            state['model_output_dimensions'] = list(image.size)
            if shape_edit:
                from local_regions import part_writeback
                report('提取生成图中的同一'+label_name+'部件，合并原位置与新轮廓')
                generated_parts = [part_regions(image,part,head_config,directory,
                    'generated-'+str(index),reference_mask=part_mask,task=task)[kind]
                    for index,(part,part_mask) in enumerate(zip(parts,source_part_views))]
                generated_part = union_masks(generated_parts)
                save_image(generated_part,directory/('generated-'+kind+'.png'))
                writeback = part_writeback(source_part_view,generated_part,
                    preserved_view,manual_view,(x1-x0,y1-y0))
                save_image(writeback,directory/'mask.png')
                region_key = 'hair_regions' if task=='hair_shape' else 'clothing_regions'
                state.update(writeback_scope='hair_union_minus_protected' if task=='hair_shape' else 'garment_union_minus_protected',
                    mask_role='old and generated part with boundary band, minus source foreground and user protection',
                    **{region_key:{name:str((directory/filename).resolve()) for name,filename in {
                        'source':'source-'+kind+'.png','generated':'generated-'+kind+'.png',
                        'foreground':'source-preserved.png','user_protection':'protected.png',
                        'intent':'intent-mask.png'}.items()}},
                    region_config={key:head_config[key] for key in ('url','model')})
        save_json(directory/'session.json',state)
        state = prepare_local_output(directory,sr_config,report)
        state['status'] = 'ready'
        save_json(directory/'session.json',state)
        return str(directory),pixels
    except Exception as error:
        state.update(status='failed',error=str(error))
        save_json(directory/'session.json',state)
        raise


def _gradient_writeback(source_crop, generated, mask):
    """Carry the model's gradients into the selection with the photograph as a fixed boundary.

    H3 edits are solved as one small Poisson problem over the selection: the guidance
    is the model's own gradient on inside-inside pixel edges only, the gradient guidance across the
    selection boundary is zero, and every original neighbour of a boundary pixel is a fixed Dirichlet
    value. A single global offset cannot do that, so carried detail cannot lift the whole selected
    area off the surrounding tone. Only the selection bbox plus one pixel halo is built, and
    neighbours outside the frame simply do not exist.
    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    support = mask>0
    rows = np.flatnonzero(support.any(axis=1))
    columns = np.flatnonzero(support.any(axis=0))
    if not rows.size or not columns.size:
        raise ValueError('局部蒙版为空，无法做梯度域融合')
    height,width = support.shape
    y0 = max(int(rows[0])-1,0); y1 = min(int(rows[-1])+2,height)
    x0 = max(int(columns[0])-1,0); x1 = min(int(columns[-1])+2,width)
    inside = support[y0:y1,x0:x1]
    original = np.ascontiguousarray(source_crop[y0:y1,x0:x1],np.float32)
    model = np.ascontiguousarray(generated[y0:y1,x0:x1],np.float32)
    small_height,small_width = inside.shape
    index = -np.ones(inside.shape,np.int64)
    index[inside] = np.arange(int(inside.sum()),dtype=np.int64)
    unknowns = int(inside.sum())
    ys,xs = np.nonzero(inside)
    degree = np.zeros(unknowns,np.float32)
    rhs = np.zeros((unknowns,3),np.float32)
    edges = []
    for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
        ny,nx = ys+dy,xs+dx
        exists = (ny>=0)&(ny<small_height)&(nx>=0)&(nx<small_width)
        here = np.flatnonzero(exists)
        qy,qx = ny[here],nx[here]
        linked = inside[qy,qx]
        # degree counts only real neighbours, so a selection on the frame edge is not over-constrained
        degree[here] += 1
        mine,theirs = here[linked],index[qy[linked],qx[linked]]
        edges.append((mine,theirs))
        rhs[mine] += model[ys[mine],xs[mine]]-model[qy[linked],qx[linked]]
        rhs[here[~linked]] += original[qy[~linked],qx[~linked]]
    internal = sum(len(mine) for mine,_ in edges)
    boundary = int(degree.sum())-internal
    if not boundary:
        # A selection with no outside neighbour leaves the Laplacian with a nullspace, so the model
        # result is returned as it is; the ordinary float mask with inward feather still writes it.
        return generated,{'solver':'not used','method':'gradient domain not applied',
            'status':'no outside boundary: the model result is returned unchanged',
            'unknowns':unknowns,'internal_directed_edges':int(internal),'boundary_edges':0,
            'bbox':[int(x0),int(y0),int(x1),int(y1)]}
    data = np.concatenate([degree]+[-np.ones(len(mine),np.float32) for mine,_ in edges])
    rows_index = np.concatenate([np.arange(unknowns,dtype=np.int64)]+[mine.astype(np.int64) for mine,_ in edges])
    columns_index = np.concatenate([np.arange(unknowns,dtype=np.int64)]
        +[theirs.astype(np.int64) for _,theirs in edges])
    matrix = sparse.coo_matrix((data,(rows_index,columns_index)),shape=(unknowns,unknowns)).tocsc()
    solved = np.empty((unknowns,3),np.float32)
    try:
        for channel in range(3):
            solved[:,channel] = spsolve(matrix,rhs[:,channel])
    except Exception as error:
        raise RuntimeError('梯度域融合求解失败，已保留原始输出与蒙版：'+str(error)) from error
    if not np.isfinite(solved).all():
        raise RuntimeError('梯度域融合求解失败，已保留原始输出与蒙版：解中出现非有限值')
    # The caller owns this float work array and reads it back only through the
    # selection, so the solved field is written in place instead of copying the
    # whole blend region a second time.
    generated[y0:y1,x0:x1][inside] = np.clip(solved,0.0,1.0)
    record = {'solver':'scipy.sparse.linalg.spsolve, float32, one system per RGB channel',
        'method':'model gradient guidance on inside-inside edges only',
        'boundary_guidance':'zero across the selection boundary; original neighbours are fixed Dirichlet values',
        'unknowns':unknowns,'internal_directed_edges':int(internal),'boundary_edges':boundary,
        'bbox':[int(x0),int(y0),int(x1),int(y1)]}
    return generated,record


def retouch_crop(pixels, directory):
    directory = Path(directory)
    state = json.loads((directory/'session.json').read_text(encoding='utf-8'))
    if state['status'] not in ('ready','complete'):
        raise ValueError('本次局部修饰尚未生成完整结果')
    x0,y0,x1,y1 = state['box']
    size = (x1-x0,y1-y0)
    if list(pixels.shape[1::-1])!=state['source_dimensions']:
        raise ValueError('所选版本尺寸与本次局部修饰记录不一致')
    with Image.open(directory/'mask.png') as image:
        mask = np.array(image,dtype=np.uint8)
    if mask.shape != size[::-1]:
        raise ValueError('局部蒙版与生成区域尺寸不一致')
    # Restrict the gradient-domain solve to the selected region and its boundary.
    region = (_blend_region(mask,size) if state.get('blend_mode')=='poisson_gradient'
        else (0,0,size[0],size[1]))
    rx0,ry0,rx1,ry1 = region
    local_mask = mask[ry0:ry1,rx0:rx1]
    with open_rgb(directory/state['detail_file']) as image:
        fit_description,centre_crop = _detail_fit(image.size,size)
        resized = fit_generated(image,size) if centre_crop else image.resize(size,Image.Resampling.LANCZOS)
        # The resample above still covers the whole recorded crop, so the sampled values are
        # the ones the full-crop path produced; only the float conversion is limited to the
        # blend region. Division by 255 keeps its exact float32 result.
        patch = resized.crop((rx0,ry0,rx1,ry1))
        del resized
        generated = np.empty((ry1-ry0,rx1-rx0,3),np.float32)
        np.divide(np.asarray(patch,np.float32),np.float32(255),out=generated)
        del patch
    distance = cv2.distanceTransform(np.uint8(local_mask>0),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    # Context padding must not dilute a small repair. Limit the inward edge
    # by the selection's own thickness, leaving an opaque interior.
    # feather keeps the recorded crop's own scale (min(size)), not the blend region's.
    feather = max(1,min(min(size)*.01,float(distance.max())*.25))
    edge = np.clip(distance/feather,0,1)
    alpha = local_mask.astype(np.float32)/255*(edge*edge*(3-2*edge))
    crop = pixels[y0:y1,x0:x1].copy()
    blend = crop[ry0:ry1,rx0:rx1]
    gradient_match = None
    if state.get('blend_mode')=='poisson_gradient':
        # H3 sessions solve the colour field as a gradient-domain problem.
        generated,gradient_match = _gradient_writeback(blend,generated,local_mask)
        if gradient_match and 'bbox' in gradient_match:
            left,top,right,bottom = gradient_match['bbox']
            gradient_match['bbox'] = [left+rx0,top+ry0,right+rx0,bottom+ry0]
    support = local_mask>0
    blend[support] += (generated[support]-blend[support])*alpha[support,None]
    blend[support] = np.clip(blend[support],0,1)
    state['blend'] = {'method':('float32 gradient-domain writeback; inward smoothstep feather'
            if gradient_match else 'float32 masked writeback; inward smoothstep feather'),
        'feather_pixels':float(feather),'resize':fit_description,
        'gradient_domain':gradient_match,'outside_mask':'unchanged source pixels'}
    # Sessions written before detail_fit existed still load; the recorded value
    # is derived from the actual model return instead of a stored assumption.
    state.setdefault('detail_fit',fit_description)
    return crop,state


def revise_writeback(directory, mask, *, protected):
    """Make a new mask revision of an existing generated canvas, without model calls."""
    previous = Path(directory)
    state = json.loads((previous/'session.json').read_text(encoding='utf-8'))
    width,height = state['source_dimensions']
    x0,y0,x1,y1 = state['box']
    # Preserve native mask detail in cells the user did not edit on the smaller canvas.
    native_mask = Image.new('L',(width,height))
    with Image.open(previous/'mask.png') as saved:
        native_mask.paste(saved.convert('L'),(x0,y0))
    mask = mask.convert('L')
    baseline = native_mask.resize(mask.size,Image.Resampling.NEAREST)
    changed = Image.fromarray(np.uint8(np.asarray(mask)!=np.asarray(baseline))*255)
    changed = np.asarray(changed.resize((width,height),Image.Resampling.NEAREST))>0
    selected = np.array(native_mask)
    painted = np.asarray(mask.resize((width,height),Image.Resampling.NEAREST))
    selected[changed] = painted[changed]
    protected = protected.convert('L').resize((width,height),Image.Resampling.NEAREST)
    selected[np.asarray(protected)>0] = 0
    for region_key in ('clothing_regions','hair_regions'):
        if region_key in state:
            # Display-size painting must not reopen the native foreground boundary through resampling.
            with Image.open(state[region_key]['foreground']) as foreground:
                selected[np.asarray(foreground.convert('L').resize((width,height),Image.Resampling.NEAREST))>127] = 0
    _edit_box(selected)
    if (selected[:y0].any() or selected[y1:].any() or
            selected[y0:y1,:x0].any() or selected[y0:y1,x1:].any()):
        raise ValueError('当前范围超出已有生成画布；请收回范围，或主动重新生成以覆盖新区域。')
    revised = previous.parent/('retouch-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
    revised.mkdir()
    state.update(status='running',generation_session=state.get('generation_session',str(previous.resolve())),
        writeback_revision_of=str(previous.resolve()),
        detail_file=str((previous/state['detail_file']).resolve()),
        draft_mask_dimensions=list(mask.size),writeback_scope='painted_mask',
        mask_role='manual revision; native mask preserved outside edited display cells')
    state['generation_mask_file'] = state.get('generation_mask_file',
        str(Path(state['generation_session'])/'mask.png'))
    save_json(revised/'session.json',state)
    try:
        save_image(Image.fromarray(selected[y0:y1,x0:x1]),revised/'mask.png')
        save_image(protected.crop((x0,y0,x1,y1)),revised/'protected.png')
        for region_key in ('clothing_regions','hair_regions'):
            if region_key in state:
                state[region_key]['user_protection'] = str((revised/'protected.png').resolve())
        shutil.copyfile(previous/'model-output.png',revised/'model-output.png')
        if (previous/'prompt-sources.json').is_file():
            shutil.copyfile(previous/'prompt-sources.json',revised/'prompt-sources.json')
        if (previous/'generation-prompt.json').is_file():
            record = json.loads((previous/'generation-prompt.json').read_text(encoding='utf-8'))
            save_json(revised/'generation-prompt.json',{**record,'reused_from':state['generation_session']})
        state['status'] = 'ready'
        save_json(revised/'session.json',state)
    except Exception as error:
        save_json(revised/'session.json',{**state,'status':'failed','error':str(error)})
        raise
    return str(revised)


def save_local(pixels, directory):
    directory = Path(directory)
    crop,state = retouch_crop(pixels,directory)
    x0,y0,x1,y1 = state['box']
    final = pixels.copy()
    final[y0:y1,x0:x1] = crop
    # The crop is fully copied into the saved frame; releasing it keeps the
    # 16-bit save and the JPEG preview off a second full-size float copy.
    del crop
    # Every explicit save gets its own version; saved photos are immutable.
    output = directory/('edit-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
    output.mkdir()
    result = {**state,'status':'complete','directory':str(directory.resolve()),
        'layer':{'kind':'masked_replacement','source':state['source'],
            'session':str(directory.resolve()),'canvas_box':state['box'],
            'mask':str((directory/'mask.png').resolve()),'blend_mode':state.get('blend_mode','float32_mask')},
        'files':{'final':save_image(final,output/'final.png'),'jpeg':save_jpeg(final,output/'final.jpg')}}
    save_json(output/'retouch.json',result)
    return result
