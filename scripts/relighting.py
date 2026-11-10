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
from model_clients import multipart, post_json, request


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


def transfer_lighting(source, target, alpha, config, directory):
    """Estimate one gain/bias field and apply it to the once-sampled camera RGB."""
    if not config or not config.get('url'):
        raise ValueError('请先在连接设置中填写 PIXLRelight 重打光服务；未调用其他光照方法。')
    config = {**RELIGHT_DEFAULTS, **config}
    directory = Path(directory)
    height,width = alpha.shape
    scale = 512/max(width,height)
    size = tuple(max(16,round(value*scale/16)*16) for value in (width,height))
    left,top = (512-size[0])//2,(512-size[1])//2
    # Context outside the identity region belongs to the generated frame. This
    # preview is only a model input, never the carrier for final identity pixels.
    context = source*alpha[...,None]+target*(1-alpha[...,None])
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
    path.write_bytes(request(base+'/view?'+urllib.parse.urlencode(record['file'])))
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
        'guide':'generated head lighting on the aligned canvas',
        'source_detail':'float32 camera RGB sampled once with the SIFT similarity transform',
        'model_input_bit_depth':8,'field_bit_depth':32,'application_bit_depth':32}
