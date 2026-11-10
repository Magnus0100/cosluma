"""Explicit, masked Qwen-Image-2.1 edits of a saved photograph."""
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
from model_clients import request, post_json, superresolve
from face_beauty import _upload


LOCAL_DEFAULTS = {'url':'http://127.0.0.1:8188',
    'model':'qwen_image_2.1_int8_convrot.safetensors','clip':'qwen3vl_8b_bf16.safetensors',
    'vae':'qwen_image_2.1_vae_bf16.safetensors','steps':40,'resolution':1024,'seed':42}
TASKS = {'hair_repair':('补发与露网修补',
        'Repair only the marked wig gaps, exposed wig net, broken hair sections or missing strands. '
        'Continue the adjacent hair color, strand thickness, texture and hairstyle. Do not add wind or redesign the hairstyle.'),
    'hair_shape':('发束造型',
        'Shape only the marked hair strands according to the requested silhouette, direction and drawing. '
        'Keep the wig color, visible hairline, face, headwear and the rest of the hairstyle.'),
    'neck':('双下巴修饰',
        'Subtly refine only the marked submental and neck area to reduce the double chin. '
        'Keep believable anatomy, original skin texture and lighting. Preserve the face identity, facial outline, collar and jewelry.')}


def _edit_box(mask):
    yy,xx = np.where(mask>0)
    if not len(xx):
        raise ValueError('请先涂出需要修改的范围')
    height,width = mask.shape
    padding = max((xx.max()-xx.min()+1)*.5,(yy.max()-yy.min()+1)*.5,min(width,height)*.12)
    return tuple(int(value) for value in (max(0,np.floor(xx.min()-padding)),
        max(0,np.floor(yy.min()-padding)),min(width,np.ceil(xx.max()+padding+1)),
        min(height,np.ceil(yy.max()+padding+1))))


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
    config = {**LOCAL_DEFAULTS,**config}
    directory = Path(root)/('retouch-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
    directory.mkdir()
    state = {'status':'running','task':task,'brief':brief,'reference':reference,
        'source':str(Path(base).resolve()),'config':{key:config[key] for key in LOCAL_DEFAULTS},
        'parent_version':parent_version,
        'model_input_bit_depth':8,'model_output_bit_depth':8,'output_bit_depth':16}
    save_json(directory/'session.json',state)
    endpoint = config['url'].rstrip('/')
    try:
        report('读取已保存版本，准备蒙版内的局部与上下文')
        if pixels is None:
            pixels = open_photo(base)
        height,width = pixels.shape[:2]
        selected = cv2.resize(np.asarray(mask.convert('L')),(width,height),interpolation=cv2.INTER_NEAREST)
        box = _edit_box(selected)
        x0,y0,x1,y1 = box
        # Keep the user's exact selected support. Feathering only happens
        # inside that support, so every outside float32 pixel is unchanged.
        cropped = selected[y0:y1,x0:x1]
        save_image(Image.fromarray(cropped),directory/'mask.png')
        source = to_preview(pixels[y0:y1,x0:x1])
        save_image(source,directory/'input.png')
        model_mask = Image.fromarray(cropped).convert('RGB')
        prompt = ('Image 1 is the source photograph. Image 2 is its separate edit mask: white is editable, '
            'black must stay unchanged. '+TASKS[task][1]+' Do not display the mask or any guide marks in the output. '
            'Return the same image framing and camera perspective. User request: '+brief.strip())
        images = [source,model_mask]
        if guide.getbbox():
            marked = source.convert('RGBA')
            drawing = guide.resize((width,height),Image.Resampling.NEAREST).crop(box)
            marked.alpha_composite(drawing)
            images.append(marked.convert('RGB'))
            prompt += ' Image 3 shows the intended strand shape as colored guide marks on the same crop; remove these marks in the finished photograph.'
            save_image(drawing,directory/'guide.png')
        if reference:
            with open_rgb(reference) as image:
                image.thumbnail((1536,1536),Image.Resampling.LANCZOS)
                images.append(image.copy())
            save_image(images[-1],directory/'reference.png')
            prompt += f' Image {len(images)} is a style/strand reference only; keep the person and scene from Image 1.'
        names = [_upload(image,config) for image in images]
        graph = {'8':{'class_type':'CosLumaQwenLocalEdit','inputs':{
            **{key:config[key] for key in ('model','clip','vae','steps','resolution','seed')},
            'prompt':prompt,'filename_prefix':'CosLuma/local-'+uuid.uuid4().hex}}}
        for index,name in enumerate(names,1):
            node = str(10+index)
            graph[node] = {'class_type':'LoadImage','inputs':{'image':name}}
            graph['8']['inputs']['image'+str(index)] = [node,0]
        state.update(box=list(box),source_dimensions=[width,height],prompt=prompt,
            input_dimensions=list(source.size),draft_mask_dimensions=list(mask.size),
            mask_dimensions=[x1-x0,y1-y0],
            method='Qwen-Image-2.1 separate mask reference; software masked float32 writeback',
            cache_device='cpu',cache_dtype='lossless',text_encoder_device='cuda:0',text_encoder_offload_device='cpu',text_encoder_dtype='bfloat16')
        save_json(directory/'session.json',state)
        save_json(directory/'workflow.json',graph)
        report('Qwen-Image-2.1 · '+TASKS[task][0]+' · '+str(config['steps'])+' 步')
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
                    raise RuntimeError('Qwen 局部修饰失败：'+error)
                outputs = result.get('outputs',{}).get('8',{}).get('images')
                if outputs:
                    (directory/'model-output.png').write_bytes(request(endpoint+'/view?'+urllib.parse.urlencode(outputs[0])))
                    state['model'] = result['outputs']['8']['model'][0]
                    break
                if status.get('completed'):
                    raise RuntimeError('Qwen 局部修饰未返回图片；已保留服务记录')
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


def retouch_crop(pixels, directory):
    directory = Path(directory)
    state = json.loads((directory/'session.json').read_text(encoding='utf-8'))
    if state['status'] not in ('ready','complete'):
        raise ValueError('本次局部修饰尚未生成完整结果')
    x0,y0,x1,y1 = state['box']
    size = (x1-x0,y1-y0)
    if list(pixels.shape[1::-1])!=state['source_dimensions']:
        raise ValueError('所选版本尺寸与本次局部修饰记录不一致')
    with open_rgb(directory/state['detail_file']) as image:
        fit_description,centre_crop = _detail_fit(image.size,size)
        resized = fit_generated(image,size) if centre_crop else image.resize(size,Image.Resampling.LANCZOS)
        generated = np.asarray(resized,np.float32)/255
    with Image.open(directory/'mask.png') as image:
        mask = np.array(image,dtype=np.uint8)
    if mask.shape != size[::-1]:
        raise ValueError('局部蒙版与生成区域尺寸不一致')
    distance = cv2.distanceTransform(np.uint8(mask>0),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    # Context padding must not dilute a small repair. Limit the inward edge
    # by the selection's own thickness, leaving an opaque interior.
    feather = max(1,min(min(size)*.01,float(distance.max())*.25))
    edge = np.clip(distance/feather,0,1)
    alpha = mask.astype(np.float32)/255*(edge*edge*(3-2*edge))
    crop = pixels[y0:y1,x0:x1].copy()
    colour_match = None
    if state['task']=='neck':
        # Skin edits must inherit the original local skin tone. Estimate only
        # the incidental RGB shift from neighbouring unedited context; never
        # write that context or change the saved raw model output.
        radius = min(size)*.03
        outside = np.uint8(mask==0)
        distance_outside = cv2.distanceTransform(outside,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        context = (outside>0)&(distance_outside<=radius)
        if np.any(context):
            offset = np.median((generated-crop)[context],axis=0).astype(np.float32)
            generated -= offset
            colour_match = {'method':'median RGB offset from unedited context',
                'radius_pixels':float(radius),'offset':offset.tolist()}
    support = mask>0
    crop[support] += (generated[support]-crop[support])*alpha[support,None]
    crop[support] = np.clip(crop[support],0,1)
    state['blend'] = {'method':'float32 masked writeback; inward smoothstep feather',
        'feather_pixels':float(feather),'resize':fit_description,'colour_match':colour_match,
        'outside_mask':'unchanged source pixels'}
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
    # Every explicit save gets its own version; saved photos are immutable.
    output = directory/('edit-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
    output.mkdir()
    result = {**state,'status':'complete','directory':str(directory.resolve()),
        'files':{'final':save_image(final,output/'final.png'),'jpeg':save_jpeg(final,output/'final.jpg')}}
    save_json(output/'retouch.json',result)
    return result
