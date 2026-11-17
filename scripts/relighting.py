"""PIXLRelight light fields; camera RGB stays in the float compositor."""
from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import time
import urllib.parse
import uuid

import cv2
import numpy as np
from PIL import Image

from image_io import save_json, to_preview
from model_clients import download, multipart, post_json, request


RELIGHT_DEFAULTS = {'model_directory':'pixlrelight',
    'intrinsics_directory':'marigold-iid-lighting-v1-1', 'device':'cuda:0', 'seed':42}


def _upload(image, config):
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    body, content_type = multipart({'type':'input','overwrite':'false'}, 'image',
        'cosluma-light-'+uuid.uuid4().hex+'.png', buffer.getvalue())
    result = json.loads(request(config['url'].rstrip('/')+'/upload/image', data=body,
        headers={'Content-Type':content_type}))
    return '/'.join(part for part in (result.get('subfolder'),result['name']) if part)


def _align_reference(source, target, directory):
    """Register only the generated lighting reference; saved photo pixels are never warped."""
    with to_preview(source,1536) as image:
        original = np.asarray(image).copy()
    with to_preview(target) as image:
        reference = np.asarray(image.resize((original.shape[1],original.shape[0]),Image.Resampling.BICUBIC))
    sift = cv2.SIFT_create()
    first,a = sift.detectAndCompute(cv2.cvtColor(reference,cv2.COLOR_RGB2GRAY),None)
    second,b = sift.detectAndCompute(cv2.cvtColor(original,cv2.COLOR_RGB2GRAY),None)
    if a is None or b is None or len(b)<2:
        raise ValueError('光照参考与当前成片缺少可对应的纹理，无法对齐；两张图片已保留。')
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(a,b,k=2)
    matches = [m for m,n in pairs if m.distance<.75*n.distance]
    if len(matches)<3:
        raise ValueError('光照参考与当前成片的对应点不足，无法对齐；两张图片已保留。')
    transform,inliers = cv2.estimateAffinePartial2D(
        np.array([first[m.queryIdx].pt for m in matches],np.float32),
        np.array([second[m.trainIdx].pt for m in matches],np.float32),method=cv2.RANSAC)
    if transform is None:
        raise ValueError('无法估计光照参考的整体等比对齐；两张图片已保留。')
    aligned = cv2.warpAffine(reference,transform,(original.shape[1],original.shape[0]),
        flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
    from image_io import save_image
    save_image(Image.fromarray(aligned),Path(directory)/'lighting-reference-aligned.png')
    return aligned.astype(np.float32)/255, {'method':'SIFT similarity; generated reference only',
        'reference_to_source':transform.tolist(),'analysis_dimensions':[original.shape[1],original.shape[0]],
        'matches':len(matches),'inliers':int(inliers.sum())}


def transfer_lighting(source, target, alpha, config, directory):
    """Apply a light field to float32 RGB; alpha=None uses the complete saved frame."""
    if not config or not config.get('url'):
        raise ValueError('请先在连接设置中填写 PIXLRelight 重打光服务；未调用其他光照方法。')
    config = {**RELIGHT_DEFAULTS, **config}
    directory = Path(directory)
    alignment = None
    if alpha is None:
        target,alignment = _align_reference(source,target,directory)
    height,width = source.shape[:2] if alpha is None else alpha.shape
    scale = 512/max(width,height)
    size = tuple(max(16,round(value*scale/16)*16) for value in (width,height))
    left,top = (512-size[0])//2,(512-size[1])//2
    # Context outside the identity region belongs to the generated frame. This
    # preview is only a model input, never the carrier for final identity pixels.
    context = source if alpha is None else source*alpha[...,None]+target*(1-alpha[...,None])
    inputs = []
    for pixels in (context,target):
        image = to_preview(pixels).resize(size,Image.Resampling.BICUBIC)
        canvas = Image.new('RGB',(512,512),'white')
        canvas.paste(image,(left,top))
        inputs.append(_upload(canvas,config))
    graph = {
        '1':{'class_type':'LoadImage','inputs':{'image':inputs[0]}},
        '2':{'class_type':'LoadImage','inputs':{'image':inputs[1]}},
        '3':{'class_type':'CosLumaPIXLRelight','inputs':{'source':['1',0], 'target':['2',0],
            **{key:config[key] for key in RELIGHT_DEFAULTS},
            'filename_prefix':'CosLuma/relight-'+uuid.uuid4().hex}}}
    save_json(directory/'relight-workflow.json',graph)
    base = config['url'].rstrip('/')
    submission = post_json(base+'/prompt',{'prompt':graph,'client_id':'cosluma-'+uuid.uuid4().hex})
    save_json(directory/'relight-submission.json',submission)
    identifier = submission['prompt_id']
    while True:
        history = json.loads(request(base+'/history/'+urllib.parse.quote(identifier)))
        if identifier in history:
            result = history[identifier]
            save_json(directory/'relight-history.json',result)
            status = result.get('status',{})
            if status.get('status_str')=='error':
                error = '; '.join(row[1].get('exception_message','')
                    for row in status.get('messages',[]) if row[0]=='execution_error')
                raise RuntimeError('PIXLRelight 重打光失败：'+error)
            fields = result.get('outputs',{}).get('3',{}).get('fields')
            if fields:
                break
            if status.get('completed'):
                raise RuntimeError('PIXLRelight 未返回增益与偏置；已保留服务响应。')
        time.sleep(1)
    record = fields[0]
    path = directory/'relight-fields.npz'
    download(base+'/view?'+urllib.parse.urlencode(record['file']),path)
    with np.load(path,allow_pickle=False) as data:
        gain,bias = data['gain'],data['bias']
    if gain.shape!=(512,512,3) or bias.shape!=gain.shape or not (
            np.isfinite(gain).all() and np.isfinite(bias).all()):
        raise ValueError('PIXLRelight 返回的光照场尺寸或数值无效；原片和模型文件保留。')
    gain = cv2.resize(gain[top:top+size[1],left:left+size[0]],(width,height),interpolation=cv2.INTER_LINEAR)
    bias = cv2.resize(bias[top:top+size[1],left:left+size[0]],(width,height),interpolation=cv2.INTER_LINEAR)
    matched = source*gain.astype(np.float32)+bias.astype(np.float32)
    np.clip(matched,0,1,out=matched)
    return matched, {**record['model'],'method':'PIXLRelight RGB-reference affine modulation',
        'field_file':str(path.resolve()),'field_dimensions':[512,512],
        'content_box':[left,top,left+size[0],top+size[1]],
        'alignment':alignment,
        'guide':('aligned full-frame lighting reference from the complete edited photograph' if alpha is None
            else 'generated head lighting on the aligned canvas'),
        'source_detail':('saved float32 composite including all prior edits' if alpha is None
            else 'float32 camera RGB sampled once with the SIFT similarity transform'),
        'model_input_bit_depth':8,'field_bit_depth':32,'application_bit_depth':32}


def save_relight(pixels, brief, context, config, *, progress=None):
    """Explicit final lighting of a saved version, using its complete frame as the only reference."""
    from datetime import datetime
    from face_beauty import _run
    from generation_prompts import write_generation_prompt
    from image_io import open_photo, preview_data_url, save_image, save_jpeg
    from local_retouch import LOCAL_ENGINES, local_engine

    report = progress or (lambda message:None)
    directory = Path(context['directory'])/('relight-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
    directory.mkdir(parents=True)
    record = {'kind':'relight','status':'running','base':context['base'],
        'parent_version':context['parent_version'],'brief':brief,
        'created_at':datetime.now().astimezone().isoformat(timespec='seconds')}
    save_json(directory/'relight.json',record)
    try:
        engine = local_engine(config['local_edit'])
        lighting_config = config['relight']
        node = LOCAL_ENGINES[engine['engine']]
        record['engine'] = engine
        with to_preview(pixels,1536) as image:
            save_image(image,directory/'lighting-input.png')
            description = write_generation_prompt('relight',
                [('当前已保存的完整成片',preview_data_url(image,1536))],config['llm'],directory,
                task={'engine':engine['engine'],'user_request':brief,
                    'parent_version':context['parent_version']},progress=report)
            name = _upload(image,engine)
        graph = {'1':{'class_type':'LoadImage','inputs':{'image':name}},
            '8':{'class_type':node['node'],'inputs':{
                **{key:engine[key] for key in ('model','clip','vae','steps','resolution','seed')},
                'image1':['1',0],'prompt':description['prompt'],
                'filename_prefix':'CosLuma/final-light-'+uuid.uuid4().hex}}}
        report(node['label']+' · 生成整图光照参考')
        history = _run(graph,{'8':'lighting-reference.png'},engine,directory,'lighting-reference',label='整图光照参考')
        target = open_photo(directory/'lighting-reference.png')
        record.update(prompt=description['prompt'],
            lighting_reference_native_dimensions=[target.shape[1],target.shape[0]],
            generation=history['outputs']['8'].get('model'))
        report('PIXLRelight · 完整画面光照分析与浮点调制')
        final,lighting = transfer_lighting(pixels,target,None,lighting_config,directory)
        del target
        report('另存原尺寸 16 位 PNG 与 JPEG')
        save_image(final,directory/'final.png')
        save_jpeg(final,directory/'final.jpg')
        record.update(status='complete',lighting=lighting,
            files={'final':str(directory/'final.png'),'jpeg':str(directory/'final.jpg')},
            output_dimensions=[pixels.shape[1],pixels.shape[0]])
    except Exception as error:
        record.update(status='failed',error=str(error))
        save_json(directory/'relight.json',record)
        raise
    save_json(directory/'relight.json',record)
    return record
