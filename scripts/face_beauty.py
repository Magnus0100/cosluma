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

from image_io import open_photo, open_rgb, save_image, save_jpeg, save_json, to_preview
from model_clients import download, multipart, post_json, request

BEAUTY_DEFAULTS = {'url':'http://127.0.0.1:8188', 'analysis_model':'facexformer.safetensors',
    'landmark_model':'fgtbt.safetensors',
    'analysis_device':'cuda:0',
    'model':'FLUX.1-Kontext-dev', 'lora':'beautygrpo.safetensors',
    'inference_device':'cuda:0','memory':'bnb_nf4'}


def _topology(name):
    if name == 'wflw-98':
        return {'count':98, 'jaw':list(range(33)), 'eyes':[list(range(60,68)),list(range(68,76))],
            'brows':[list(range(33,42)),list(range(42,51))], 'lips':list(range(76,88)),
            'inner_lips':list(range(88,96)), 'nose':list(range(51,60)), 'pupils':[96,97]}
    raise ValueError('未知面部关键点拓扑：'+str(name))


@dataclass(frozen=True)
class BeautyParameters:
    skin: float = 0.0
    slim: float = 0.0
    protect_features: bool = True

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
                    download(base+'/view?'+urllib.parse.urlencode(image),directory/filename)
                return result
            if status.get('completed'):
                raise RuntimeError('人脸美化服务未返回所需图片；已保留服务记录')
        time.sleep(1)


class BeautySession:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.state = json.loads((self.directory/'session.json').read_text(encoding='utf-8'))
        self.box = tuple(self.state['box'])
        missing = [key for key in ('analysis_region','landmark_topology') if key not in self.state]
        if missing:
            # Historical mediapipe sessions are not migrated: only the current
            # FGTBT analysis can be restored, and the user re-frames it explicitly.
            raise ValueError('该美化会话不是当前 FGTBT 98 点分析结果（缺少 '
                +'、'.join(missing)+'），仅当前分析会话可恢复；请在美化页重新框选人脸并分析，'
                '已有成片与旧文件保持不变')
        self.analysis_region = tuple(self.state['analysis_region'])
        self.points = np.asarray(self.state['landmarks'], np.float32)
        self.topology = self.state['landmark_topology']
        if self.points.shape != (_topology(self.topology)['count'],2) or not np.isfinite(self.points).all():
            raise ValueError('美化会话的关键点与所记录拓扑不一致')
        with open_rgb(self.directory/'skin.png') as mask:
            self.mask = np.asarray(mask, np.float32)[..., 0]/255
        self.delta = None
        if self.state.get('skin_ready'):
            with open_rgb(self.directory/self.state['skin_file']) as image:
                retouched = np.asarray(image, np.float32)/255
            with open_rgb(self.directory/self.state['skin_input']) as image:
                original = np.asarray(image, np.float32)/255
            self.delta = retouched-original

    @property
    def base_path(self):
        return (self.directory/self.state['base']).resolve()


FACE_REGION_ERROR = '人脸范围须为四个有效像素坐标 [left,top,right,bottom]'


def face_region_values(region):
    """Validate the explicit face rectangle before any service call."""
    try:
        if isinstance(region, np.ndarray):
            values = region
        elif isinstance(region, (list, tuple)):
            # numpy promotes [False, 0, 20, 30] to an int array, hiding the bool;
            # inspect the declared element types before any conversion.
            if any(isinstance(item, (bool, np.bool_)) for item in region):
                raise ValueError(FACE_REGION_ERROR)
            if not all(isinstance(item, (int, float, np.integer, np.floating)) for item in region):
                raise ValueError(FACE_REGION_ERROR)
            values = np.asarray(region)
        else:
            raise ValueError(FACE_REGION_ERROR)
        if values.dtype == np.bool_ or values.dtype.kind not in 'iuf':
            raise ValueError(FACE_REGION_ERROR)
        values = values.astype(np.float64)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(FACE_REGION_ERROR) from None
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError(FACE_REGION_ERROR)
    return values


def prepare_face(base, region, root, config, progress=None):
    report = progress or (lambda message:None)
    report('读取原成片与选定人脸范围')
    pixels = open_photo(base)
    height, width = pixels.shape[:2]
    region = face_region_values(region)
    x0, y0, x1, y1 = np.rint(region).astype(int)
    if not (0<=x0<x1<=width and 0<=y0<y1<=height):
        raise ValueError('人脸范围须位于当前成片内，并包含有效面积')
    directory = Path(root)/('beauty-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
    directory.mkdir(parents=True, exist_ok=False)
    preview = to_preview(pixels[y0:y1, x0:x1], 1024)
    image = _upload(preview, config)
    report('定位眼睑与面部轮廓，分析皮肤范围')
    prefix = 'CosLuma/beauty-'+uuid.uuid4().hex
    graph = {
        '1':{'class_type':'LoadImage','inputs':{'image':image}},
        '2':{'class_type':'CosLumaFGTBTAnalysis','inputs':{'image':['1',0],
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
    # Rounding the native crop may change its width and height independently.
    # Carry points through preview pixels, then normalize to the actual crop.
    preview_box = np.asarray(analysis['box'],np.float32)
    preview_points = np.asarray(analysis['landmarks'],np.float32)*(preview_box[2:]-preview_box[:2])+preview_box[:2]
    native_points = preview_points*scale+[x0,y0]
    normalized_points = (native_points-box[:2])/(box[2:]-box[:2])
    state = {'base':os.path.relpath(Path(base).resolve(), directory), 'box':box.tolist(),
        'landmarks':normalized_points.tolist(),'landmark_topology':analysis['landmark_topology'],
        'source_dimensions':[width,height], 'skin_ready':False,'analysis_region':[int(x0),int(y0),int(x1),int(y1)],
        'face_input':'native completed RGB; official 1024 bilinear resize',
        'analysis':{'model':analysis['analysis_model'],'landmark_model':analysis['landmark_model'],
            'landmark_device':analysis['landmark_device'],'device':analysis['device'],
            'face_region_source':analysis['face_region_source'],'method':analysis['landmark_method']}}
    save_json(directory/'session.json', state)
    return BeautySession(directory), pixels


def prepare_skin(session, config, progress=None, *, pixels=None, llm_config):
    report = progress or (lambda message:None)
    directory = session.directory/('skin-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
    directory.mkdir()
    report('从完整成片准备官方 1024 像素输入')
    if pixels is None:
        pixels = open_photo(session.base_path)
    left,top,right,bottom = session.box
    image = to_preview(pixels[top:bottom,left:right]).resize((1024,1024),Image.Resampling.BILINEAR)
    del pixels
    image.save(directory/'input.png')
    from generation_prompts import write_generation_prompt
    from image_io import preview_data_url
    write_generation_prompt('beauty',[('用户主动生成美肤基础的人脸',preview_data_url(image))],
        llm_config,directory,task={'task':'user_requested_skin_base'},progress=report)
    filename = _upload(image, config)
    report('BeautyGRPO · NF4 底座 / 完整 LoRA · 28 步' if config['memory']=='bnb_nf4'
        else 'BeautyGRPO 官方 BF16 完整推理 · 28 步')
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


def _feature_mask(points, size, *, protect_nose=False, topology):
    mask = np.ones(size[::-1],np.uint8)
    shape = _topology(topology)
    contours = (*shape['brows'],*shape['eyes'],shape['lips'],*([shape['nose']] if protect_nose else []))
    for indices in contours:
        polygon = np.rint(points[indices]*np.array(size)).astype(np.int32)
        cv2.fillConvexPoly(mask,cv2.convexHull(polygon),0)
    mask = cv2.erode(mask,np.ones((3,3),np.uint8))
    return cv2.GaussianBlur(mask.astype(np.float32),(0,0),max(1,min(size)*.003))*mask


def _face_mask(points, size, *, topology):
    mask = np.zeros(size[::-1],np.uint8)
    _topology(topology)
    # WFLW has a jaw contour but no forehead points. Extend the upper brow
    # hull only for the optional beauty-difference mask, never for warping.
    brow = points[list(range(33,51))].copy()
    brow[:,1] -= np.ptp(points[:33,1])*.22
    contour = np.concatenate((points[:33],brow))
    polygon = cv2.convexHull(np.rint(contour*np.array(size)).astype(np.int32))
    cv2.fillPoly(mask,[polygon],1)
    return cv2.GaussianBlur(mask.astype(np.float32),(0,0),max(1,min(size)*.01))*mask


def _slim_map(points, strength, size, *, topology):
    """Inverse thin-plate spline; fixed features and a stationary outer border."""
    shape = _topology(topology)
    jaw = shape['jaw']
    indices = list(dict.fromkeys(jaw+sum(map(list,shape['eyes']),[])+sum(map(list,shape['brows']),[])+
        shape['lips']+shape['inner_lips']+shape['nose']+shape['pupils']))
    mesh = np.asarray(points, np.float64)
    original = mesh[indices]
    target = original.copy()
    horizontal = mesh[97]-mesh[96]
    center = (mesh[51]+mesh[54])/2
    # Zero at temples and central chin; strongest on the lower cheeks.
    t = np.linspace(0,1,len(jaw))
    weights = np.sin(2*np.pi*t)**2
    if np.linalg.norm(horizontal)<1e-6:
        raise ValueError('眼部关键点重合，无法确定瘦脸方向')
    horizontal /= np.linalg.norm(horizontal)
    displacement = ((original[:len(jaw)]-center)@horizontal)*(.18*strength*weights)
    target[:len(jaw)] -= displacement[:,None]*horizontal
    line = np.linspace(0,1,6)
    border = np.concatenate([np.column_stack((line,np.zeros(6))),np.column_stack((line,np.ones(6))),
        np.column_stack((np.zeros(4),line[1:-1])),np.column_stack((np.ones(4),line[1:-1]))])
    source = np.vstack((original, border))
    destination = np.vstack((target, border))
    from scipy.interpolate import RBFInterpolator
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
    delta *= _feature_mask(mesh,size,protect_nose=True,topology=topology)[...,None]
    yy,xx = np.mgrid[:size[1],:size[0]].astype(np.float32)
    return xx+delta[...,0]*(size[0]-1), yy+delta[...,1]*(size[1]-1)


def render_beauty_crop(pixels, session, parameters):
    """Render only the native face region, shared by preview and final saving."""
    x0,y0,x1,y1 = session.box
    size = (x1-x0,y1-y0)
    crop = pixels[y0:y1,x0:x1].copy()
    if parameters.skin:
        if session.delta is None:
            raise ValueError('请先生成美肤基础')
        if parameters.protect_features:
            mask = cv2.resize(session.mask,size,interpolation=cv2.INTER_LINEAR)
            mask *= _feature_mask(session.points,size,topology=session.topology)
        else:
            # The user's explicit choice allows the learned adjustment on
            # eyes, brows and lips, while keeping it within the face contour.
            mask = _face_mask(session.points,size,topology=session.topology)
        delta = cv2.resize(session.delta,size,interpolation=cv2.INTER_CUBIC)
        crop = np.clip(crop+delta*(mask*parameters.skin)[...,None],0,1)
    if parameters.slim:
        map_x,map_y = _slim_map(session.points,parameters.slim,size,topology=session.topology)
        crop = cv2.remap(crop,map_x,map_y,cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_REFLECT_101)
    return np.clip(crop,0,1)


def render_beauty(pixels, session, parameters):
    if parameters.skin == 0 and parameters.slim == 0:
        return pixels
    x0,y0,x1,y1 = session.box
    crop = render_beauty_crop(pixels,session,parameters)
    output = pixels.copy()
    output[y0:y1,x0:x1] = crop
    return output


def save_beauty(pixels, session, parameters):
    directory = session.directory/('edit-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
    directory.mkdir()
    final = render_beauty(pixels,session,parameters)
    result = {'parameters':{'skin':parameters.skin,'slim':parameters.slim,
            'protect_features':parameters.protect_features},
        'source':str(session.base_path),'source_dimensions':session.state['source_dimensions'],
        'session':os.path.relpath(session.directory,session.directory.parent), 'box':list(session.box),
        'analysis':session.state['analysis'],'landmark_topology':session.topology,
        'skin':session.state.get('skin') if parameters.skin else None,
        'files':{'final':save_image(final,directory/'final.png'),'jpeg':save_jpeg(final,directory/'final.jpg')},
        'output_bit_depth':16, 'status':'complete'}
    save_json(directory/'beauty.json',result)
    latest = dict(result)
    latest['files'] = {key:os.path.relpath(value,session.directory.parent) for key,value in result['files'].items()}
    save_json(session.directory.parent/'beauty-latest.json',latest)
    return result
