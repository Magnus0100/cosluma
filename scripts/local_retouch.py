"""Explicit, masked local edits of a saved photograph with the selected engine."""
from __future__ import annotations

from datetime import datetime
import json
import os
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
        'Shape only the marked hair strands according to the requested silhouette, direction and drawing. '
        'Keep the wig color, visible hairline, face, headwear and the rest of the hairstyle.'),
    'neck':('双下巴修饰',
        'Subtly refine only the marked submental and neck area to reduce the double chin. '
        'Keep believable anatomy, original skin texture and lighting. Preserve the face identity, facial outline, collar and jewelry.')}
# Positive descriptions of the finished appearance, used only when the user left the brief empty. The
# model redraws the whole photograph from noise, so naming a defect can put that defect into the result.
H3_TASKS = {'hair_repair':
        'Repair the hair into continuous, natural strands that match the adjacent color and texture.',
    'hair_shape':
        'Shape the hair into flowing, natural strands in the direction the user asks for.',
    'neck':'Gently smooth the neck skin for a natural portrait finish.'}


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
        parent_version=None, sr_config=None):
    report = progress or (lambda message:None)
    if task not in TASKS:
        raise ValueError('请选择补发、发束造型或双下巴修饰')
    config = local_engine(config)
    engine = LOCAL_ENGINES[config['engine']]
    directory = Path(root)/('retouch-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
    directory.mkdir()
    state = {'status':'running','task':task,'brief':brief,'reference':reference,
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
        tight = _edit_box(selected)  # the same check: an empty selection still errors here
        # H3-Image redraws what it is given. Skin is smooth enough for the 2K area canvas, so the neck
        # pass hands it the whole photograph; hair is far finer, so its primary picture is the detail
        # crop at original pixels and the full photograph follows as a separate context reference.
        hair_detail = config['engine']=='h3' and task in ('hair_repair','hair_shape')
        box = tight if hair_detail or config['engine']!='h3' else (0,0,width,height)
        x0,y0,x1,y1 = box
        # Keep the user's exact selected support. Feathering only happens
        # inside that support, so every outside float32 pixel is unchanged.
        cropped = selected[y0:y1,x0:x1]
        save_image(Image.fromarray(cropped),directory/'mask.png')
        source = to_preview(pixels[y0:y1,x0:x1])
        save_image(source,directory/'input.png')
        # The two engines take different inputs: the Qwen graph reads a mask picture, while the
        # MiniMax-H3-Image graph conditions on pictures only - the whole photograph for the neck, the
        # detail crop plus a full-photo context picture for hair. In the H3 engine the selection only
        # limits the writeback and is never sent to the model.
        label = '<Picture {0}>' if config['engine']=='h3' else 'Image {0}'
        if config['engine']=='h3':
            roles = ['image1: selected detail crop at original pixels' if hair_detail
                else 'image1: full saved photograph']
        else:
            roles = ['image1: selected crop']
        context = None
        if config['engine']=='h3':
            images = [source]
            prompt = h3_prompt(task,brief)
            if hair_detail:
                # 8-bit context picture for reference only: it is never pasted back, the writeback
                # still comes from the detail crop. A 1536 long edge (max_edge) keeps the native VAE
                # reference sane while the primary picture is pre-scaled onto the 2K area canvas.
                context = to_preview(pixels,max_edge=1536)
                save_image(context,directory/'context.png')
                images.append(context)
                roles.append('image2: the full photograph, scaled to a 1536 long-edge (max_edge) context reference')
                prompt += (' '+label.format(2)+' is the full photograph for context; '
                    'return the edited view of '+label.format(1)+'.')
        else:
            model_mask = Image.fromarray(cropped).convert('RGB')
            prompt = (label.format(1)+' is the source photograph. '+label.format(2)+
                ' is its separate edit mask: white is editable, '
                'black must stay unchanged. '+TASKS[task][1]+' Do not display the mask or any guide marks in the output. '
                'Return the same image framing and camera perspective. User request: '+brief.strip())
            images = [source,model_mask]
        if guide.getbbox():
            marked = source.convert('RGBA')
            drawing = guide.resize((width,height),Image.Resampling.NEAREST).crop(box)
            marked.alpha_composite(drawing)
            images.append(marked.convert('RGB'))
            roles.append('colored guide marks on the same picture: the requested strand shape')
            if config['engine']=='h3':
                prompt += (' '+label.format(len(images))+' draws the intended strand shape as colored guide marks; '
                    'follow that shape and remove the marks in the finished photograph.')
            else:
                prompt += (' '+label.format(3)+' shows the intended strand shape as colored guide marks on the same crop; '
                    'remove these marks in the finished photograph.')
            save_image(drawing,directory/'guide.png')
        if reference:
            with open_rgb(reference) as image:
                image.thumbnail((1536,1536),Image.Resampling.LANCZOS)
                images.append(image.copy())
            save_image(images[-1],directory/'reference.png')
            roles.append('style/strand reference only, not the person to render')
            prompt += (f' {label.format(len(images))} is a style/strand reference only; '
                f'keep the person and scene from {label.format(1)}.')
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
            input_dimensions=list(source.size),draft_mask_dimensions=list(mask.size),
            mask_dimensions=[x1-x0,y1-y0],method=method,
            text_encoder_device='cuda:0',text_encoder_offload_device='cpu',text_encoder_dtype='bfloat16')
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
        report(engine['label']+' · '+TASKS[task][0]+' · '+detail)
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
    # Only the gradient-domain blend is a selection-local solve. The historical median
    # offset reads unedited context up to min(size)*.03 around the selection, so it keeps
    # the whole recorded crop exactly as before.
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
    colour_match = None
    gradient_match = None
    if state.get('blend_mode')=='poisson_gradient':
        # New H3 sessions solve the colour field as a gradient-domain problem for every task; this
        # replaces the plain paste and the historical median offset, with the accepted equations.
        generated,gradient_match = _gradient_writeback(blend,generated,local_mask)
        if gradient_match and 'bbox' in gradient_match:
            left,top,right,bottom = gradient_match['bbox']
            gradient_match['bbox'] = [left+rx0,top+ry0,right+rx0,bottom+ry0]
    elif state['task']=='neck':
        # Historical H3 records and every Qwen session keep this estimate of the incidental RGB
        # shift from neighbouring unedited context; never write that context or change the raw.
        radius = min(size)*.03
        outside = np.uint8(local_mask==0)
        distance_outside = cv2.distanceTransform(outside,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        context = (outside>0)&(distance_outside<=radius)
        if np.any(context):
            offset = np.median((generated-blend)[context],axis=0).astype(np.float32)
            generated -= offset
            colour_match = {'method':'median RGB offset from unedited context',
                'radius_pixels':float(radius),'offset':offset.tolist()}
    support = local_mask>0
    blend[support] += (generated[support]-blend[support])*alpha[support,None]
    blend[support] = np.clip(blend[support],0,1)
    state['blend'] = {'method':('float32 gradient-domain writeback; inward smoothstep feather'
            if gradient_match else 'float32 masked writeback; inward smoothstep feather'),
        'feather_pixels':float(feather),'resize':fit_description,'colour_match':colour_match,
        'gradient_domain':gradient_match,'outside_mask':'unchanged source pixels'}
    # Sessions written before detail_fit existed still load; the recorded value
    # is derived from the actual model return instead of a stored assumption.
    state.setdefault('detail_fit',fit_description)
    return crop,state


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
        'files':{'final':save_image(final,output/'final.png'),'jpeg':save_jpeg(final,output/'final.jpg')}}
    save_json(output/'retouch.json',result)
    return result
