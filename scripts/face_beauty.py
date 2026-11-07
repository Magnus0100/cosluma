"""Optional, editable face retouching of a completed float photograph."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
import json
import os
from pathlib import Path
import time
import urllib.parse
import uuid

import cv2
import numpy as np
from PIL import Image
from scipy.interpolate import RBFInterpolator

from image_io import open_photo, open_rgb, save_image, save_jpeg, save_json, to_preview
from model_clients import multipart, post_json, request

BEAUTY_DEFAULTS = {'url':'http://127.0.0.1:8188', 'analysis_model':'facexformer.safetensors',
    'landmark_model':'face_landmarker.task', 'analysis_device':'cuda:0',
    'model':'FLUX.1-Kontext-dev', 'lora':'beautygrpo.safetensors',
    'inference_device':'cuda:0','memory':'group_offload'}

# MediaPipe's anatomical contours; eyelids and iris centres come from the
# dedicated 478-point landmarker, independently of skin segmentation.
JAW = [234,93,132,58,172,136,150,149,152,378,379,365,397,288,361,323,454]
EYES = ([33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246],
    [263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466])
BROWS = ([70,63,105,66,107,55,65,52,53,46], [300,293,334,296,336,285,295,282,283,276])
LIPS = [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185]
INNER_LIPS = [78,95,88,178,87,14,317,402,318,324,308,415,310,311,312,13,82,81,80,191]
NOSE = [168,6,197,195,5,4,1,2,98,97,327,326]


@dataclass(frozen=True)
class BeautyParameters:
    skin: float = 0.0
    slim: float = 0.0

    def __post_init__(self):
        if not 0 <= self.skin <= 1 or not 0 <= self.slim <= 1:
            raise ValueError('美肤与瘦脸强度须在 0 到 1 之间')


def _upload(image, config):
    stream = BytesIO()
    image.save(stream, format='PNG')
    body, content_type = multipart({'type':'input', 'overwrite':'false'}, 'image',
        'cosluma-beauty-'+uuid.uuid4().hex+'.png', stream.getvalue())
    result = json.loads(request(config['url'].rstrip('/')+'/upload/image', data=body,
        headers={'Content-Type':content_type}))
    return '/'.join(part for part in (result.get('subfolder'), result['name']) if part)


def _run(graph, outputs, config, directory, name):
    base = config['url'].rstrip('/')
    save_json(directory/(name+'-workflow.json'), graph)
    submission = post_json(base+'/prompt', {'prompt':graph, 'client_id':'cosluma-'+uuid.uuid4().hex})
    save_json(directory/(name+'-submission.json'), submission)
    identifier = submission['prompt_id']
    while True:
        history = json.loads(request(base+'/history/'+urllib.parse.quote(identifier)))
        if identifier in history:
            result = history[identifier]
            save_json(directory/(name+'-history.json'), result)
            status = result.get('status', {})
            if status.get('status_str') == 'error':
                details = '; '.join(row[1].get('exception_message','') for row in status.get('messages', [])
                    if row[0] == 'execution_error')
                raise RuntimeError('人脸美化服务失败：'+details)
            if all(result.get('outputs', {}).get(node, {}).get('images') for node in outputs):
                for node, filename in outputs.items():
                    image = result['outputs'][node]['images'][0]
                    (directory/filename).write_bytes(request(base+'/view?'+urllib.parse.urlencode(image)))
                return result
            if status.get('completed'):
                raise RuntimeError('人脸美化服务未返回所需图片；已保留服务记录')
        time.sleep(1)


class BeautySession:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.state = json.loads((self.directory/'session.json').read_text(encoding='utf-8'))
        self.box = tuple(self.state['box'])
        self.points = np.asarray(self.state['landmarks'], np.float32)
        with open_rgb(self.directory/'skin.png') as mask:
            self.mask = np.asarray(mask, np.float32)[..., 0]/255
        self.delta = None
        if self.state.get('skin_ready'):
            with open_rgb(self.directory/self.state['skin_file']) as image:
                retouched = np.asarray(image, np.float32)/255
            with open_rgb(self.directory/self.state.get('skin_input','face-input.png')) as image:
                original = np.asarray(image, np.float32)/255
            self.delta = retouched-original

    @property
    def base_path(self):
        return (self.directory/self.state['base']).resolve()


def prepare_face(base, region, root, config, progress=None):
    report = progress or (lambda message:None)
    directory = Path(root)/('beauty-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
    directory.mkdir(parents=True, exist_ok=False)
    report('读取原成片与主体头部')
    pixels = open_photo(base)
    height, width = pixels.shape[:2]
    x0, y0, x1, y1 = np.rint(region).astype(int)
    preview = to_preview(pixels[y0:y1, x0:x1], 1024)
    image = _upload(preview, config)
    report('定位眼睑与面部轮廓，分析皮肤范围')
    prefix = 'CosLuma/beauty-'+uuid.uuid4().hex
    graph = {
        '1':{'class_type':'LoadImage','inputs':{'image':image}},
        '2':{'class_type':'CosLumaFaceAnalysis','inputs':{'image':['1',0],
            'analysis_model':config['analysis_model'], 'landmark_model':config['landmark_model'],
            'device':config.get('analysis_device','cuda:0')}},
        '3':{'class_type':'SaveImage','inputs':{'images':['2',0],'filename_prefix':prefix+'-face'}},
        '4':{'class_type':'MaskToImage','inputs':{'mask':['2',1]}},
        '5':{'class_type':'SaveImage','inputs':{'images':['4',0],'filename_prefix':prefix+'-skin'}}}
    result = _run(graph, {'3':'analysis-face.png','5':'skin.png'}, config, directory, 'analysis')
    analysis = result['outputs']['2']['face'][0]
    scale = np.array([x1-x0, y1-y0])/np.array(preview.size)
    box = np.rint(np.array(analysis['box'])*np.tile(scale, 2)+[x0,y0,x0,y0]).astype(int)
    box = np.clip(box, [0,0,0,0], [width,height,width,height])
    # Model RGB comes from the native completed photograph, independently of
    # the smaller analysis image. Match the official 1024 bilinear input.
    left,top,right,bottom = box
    to_preview(pixels[top:bottom,left:right]).resize((1024,1024),Image.Resampling.BILINEAR).save(directory/'face-input.png')
    state = {'base':os.path.relpath(Path(base).resolve(), directory), 'box':box.tolist(),
        'landmarks':analysis['landmarks'], 'source_dimensions':[width,height], 'skin_ready':False,
        'face_input':'native completed RGB; official 1024 bilinear resize',
        'analysis':{'model':analysis['analysis_model'],'landmark_model':analysis['landmark_model'],
            'landmark_device':analysis['landmark_device'],'device':analysis['device']}}
    save_json(directory/'session.json', state)
    return BeautySession(directory), pixels


def prepare_skin(session, config, progress=None):
    report = progress or (lambda message:None)
    directory = session.directory/('skin-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
    directory.mkdir()
    report('从完整成片准备官方 1024 像素输入')
    pixels = open_photo(session.base_path)
    left,top,right,bottom = session.box
    image = to_preview(pixels[top:bottom,left:right]).resize((1024,1024),Image.Resampling.BILINEAR)
    del pixels
    image.save(directory/'input.png')
    filename = _upload(image, config)
    report('BeautyGRPO 官方 BF16 完整推理 · 28 步')
    graph = {
        '1':{'class_type':'LoadImage','inputs':{'image':filename}},
        '2':{'class_type':'CosLumaBeautyGRPO','inputs':{'image':['1',0],
            'model':config['model'],'lora':config['lora'],'device':config['inference_device'],
            'memory':config['memory'],'filename_prefix':'CosLuma/beauty-'+uuid.uuid4().hex}}}
    result = _run(graph, {'2':'skin-result.png'}, config, directory, 'skin')
    session.state.update(skin_ready=True, skin_file=os.path.relpath(directory/'skin-result.png',session.directory),
        skin_input=os.path.relpath(directory/'input.png',session.directory),
        skin=result['outputs']['2']['beauty'][0])
    save_json(session.directory/'session.json', session.state)
    return BeautySession(session.directory)


def _feature_mask(points, size, *, protect_nose=False):
    mask = np.ones(size[::-1],np.uint8)
    contours = (*BROWS,*EYES,LIPS,*([NOSE] if protect_nose else []))
    for indices in contours:
        polygon = np.rint(points[indices]*np.array(size)).astype(np.int32)
        cv2.fillConvexPoly(mask,cv2.convexHull(polygon),0)
    mask = cv2.erode(mask,np.ones((3,3),np.uint8))
    return cv2.GaussianBlur(mask.astype(np.float32),(0,0),max(1,min(size)*.003))*mask


def _slim_map(points, strength, size):
    """Inverse thin-plate spline; fixed features and a stationary outer border."""
    indices = list(dict.fromkeys(JAW+sum(EYES,[])+sum(BROWS,[])+LIPS+INNER_LIPS+NOSE+list(range(468,478))))
    mesh = np.asarray(points, np.float64)
    original = mesh[indices]
    target = original.copy()
    horizontal = mesh[473]-mesh[468]
    horizontal /= np.linalg.norm(horizontal)
    center = (mesh[6]+mesh[4])/2
    weights = np.array([0,.25,.5,.75,.95,1,1,.6,0,.6,1,1,.95,.75,.5,.25,0])
    displacement = ((original[:17]-center)@horizontal)*(.18*strength*weights)
    target[:17] -= displacement[:,None]*horizontal
    line = np.linspace(0,1,6)
    border = np.concatenate([np.column_stack((line,np.zeros(6))),np.column_stack((line,np.ones(6))),
        np.column_stack((np.zeros(4),line[1:-1])),np.column_stack((np.ones(4),line[1:-1]))])
    source = np.vstack((original, border))
    destination = np.vstack((target, border))
    inverse = RBFInterpolator(destination, source-destination, kernel='thin_plate_spline', smoothing=1e-5)
    grid_width, grid_height = min(256,size[0]), min(256,size[1])
    yy,xx = np.mgrid[0:1:complex(0,grid_height),0:1:complex(0,grid_width)]
    grid = np.stack((xx,yy),axis=-1)
    delta = inverse(grid.reshape(-1,2)).reshape(grid_height,grid_width,2).astype(np.float32)
    edge = np.minimum(np.minimum(xx,1-xx),np.minimum(yy,1-yy))/.08
    edge = np.clip(edge,0,1)
    edge = edge*edge*(3-2*edge)
    delta *= edge[...,None]
    delta = cv2.resize(delta,size,interpolation=cv2.INTER_LINEAR)
    delta *= _feature_mask(mesh,size,protect_nose=True)[...,None]
    yy,xx = np.mgrid[:size[1],:size[0]].astype(np.float32)
    return xx+delta[...,0]*(size[0]-1), yy+delta[...,1]*(size[1]-1)


def render_beauty(pixels, session, parameters):
    if parameters.skin == 0 and parameters.slim == 0:
        return pixels
    x0,y0,x1,y1 = session.box
    size = (x1-x0,y1-y0)
    crop = pixels[y0:y1,x0:x1].copy()
    if parameters.skin:
        if session.delta is None:
            raise ValueError('请先生成美肤基础')
        mask = cv2.resize(session.mask,size,interpolation=cv2.INTER_LINEAR)
        # Explicitly protect eyes, brows and lips when applying the learned
        # skin adjustment. The original float pixels remain the RGB carrier.
        mask *= _feature_mask(session.points,size)
        delta = cv2.resize(session.delta,size,interpolation=cv2.INTER_CUBIC)
        crop = np.clip(crop+delta*(mask*parameters.skin)[...,None],0,1)
    if parameters.slim:
        map_x,map_y = _slim_map(session.points,parameters.slim,size)
        crop = cv2.remap(crop,map_x,map_y,cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_REFLECT_101)
    output = pixels.copy()
    output[y0:y1,x0:x1] = np.clip(crop,0,1)
    return output


def save_beauty(pixels, session, parameters):
    directory = session.directory/('edit-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
    directory.mkdir()
    final = render_beauty(pixels,session,parameters)
    result = {'parameters':{'skin':parameters.skin,'slim':parameters.slim},
        'source':str(session.base_path),'source_dimensions':session.state['source_dimensions'],
        'session':os.path.relpath(session.directory,session.directory.parent), 'box':list(session.box),
        'analysis':session.state['analysis'],'skin':session.state.get('skin') if parameters.skin else None,
        'files':{'final':save_image(final,directory/'final.png'),'jpeg':save_jpeg(final,directory/'final.jpg')},
        'output_bit_depth':16, 'status':'complete'}
    save_json(directory/'beauty.json',result)
    latest = dict(result)
    latest['files'] = {key:os.path.relpath(value,session.directory.parent) for key,value in result['files'].items()}
    save_json(session.directory.parent/'beauty-latest.json',latest)
    return result
