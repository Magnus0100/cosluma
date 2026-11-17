"""Explicit full-frame light and color edits on saved float32 photographs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import math
from pathlib import Path
import uuid

import numpy as np

from image_io import apply_tone_curve, open_photo, preview_data_url, save_image, save_jpeg, save_json, to_preview


TONE_PROPERTIES = {
    'exposure_ev':{'type':'number','minimum':-2,'maximum':2,'description':'曝光 EV；默认 0'},
    'shadows':{'type':'number','minimum':.05,'maximum':.4,'description':'输入明度 0.25 的输出；默认 0.25'},
    'midtones':{'type':'number','minimum':.4,'maximum':.6,'description':'输入明度 0.5 的输出；默认 0.5'},
    'highlights':{'type':'number','minimum':.6,'maximum':.95,'description':'输入明度 0.75 的输出；默认 0.75'},
    'warmth':{'type':'number','minimum':-1,'maximum':1,'description':'暖冷平衡；负值偏冷，正值偏暖，默认 0；不是开尔文色温'},
    'tint':{'type':'number','minimum':-1,'maximum':1,'description':'绿洋红平衡；负值偏绿，正值偏洋红，默认 0'},
    'saturation':{'type':'number','minimum':0,'maximum':2,'description':'线性光饱和度；0 黑白，1 原值，2 加强'}
}


@dataclass(frozen=True)
class ToneParameters:
    exposure_ev: float = 0.0
    shadows: float = .25
    midtones: float = .5
    highlights: float = .75
    warmth: float = 0.0
    tint: float = 0.0
    saturation: float = 1.0

    def __post_init__(self):
        for name, specification in TONE_PROPERTIES.items():
            low,high = specification['minimum'],specification['maximum']
            value = getattr(self,name)
            if not math.isfinite(value) or not low<=value<=high:
                raise ValueError(f'{name} 须在 {low} 至 {high} 之间')

    def record(self):
        return asdict(self)

    def points(self):
        return [[0,0],[.25,self.shadows],[.5,self.midtones],[.75,self.highlights],[1,1]]


def render_tone(pixels, parameters):
    """Linear-light EV, RGB balance and saturation, then the existing luminance curve."""
    if parameters==ToneParameters():
        return pixels
    result = np.empty_like(pixels)
    gain = np.float32(2**parameters.exposure_ev)
    balance = parameters.warmth!=0 or parameters.tint!=0
    color = balance or parameters.saturation!=1
    weights = np.array([.2126,.7152,.0722],np.float32)
    if balance:
        gains = np.exp2(np.array([.25*parameters.warmth+.125*parameters.tint,
            -.25*parameters.tint,-.25*parameters.warmth+.125*parameters.tint],np.float32))
        # Preserve neutral-gray luminance before clipping; controls are RGB balance, not CCT.
        gains /= gains @ weights
    for start in range(0,pixels.shape[0],256):
        tile = pixels[start:start+256]
        if parameters.exposure_ev==0 and not color:
            result[start:start+256] = tile
            continue
        linear = np.where(tile<=.04045,tile/np.float32(12.92),
            ((tile+np.float32(.055))/np.float32(1.055))**np.float32(2.4))
        linear *= gain
        if balance:
            linear *= gains
        if parameters.saturation!=1:
            luminance = (linear @ weights)[...,None]
            linear -= luminance
            linear *= np.float32(parameters.saturation)
            linear += luminance
        np.clip(linear,0,1,out=linear)
        result[start:start+256] = np.where(linear<=.0031308,linear*np.float32(12.92),
            np.float32(1.055)*linear**np.float32(1/2.4)-np.float32(.055))
    if (parameters.shadows,parameters.midtones,parameters.highlights)!=(.25,.5,.75):
        apply_tone_curve(result,parameters.points(),out=result)
    return result


def save_tone(pixels, parameters, context):
    directory = Path(context['directory'])/('tone-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
    directory.mkdir(parents=True)
    record = {'kind':'tone','status':'running','base':context['base'],
        'parent_version':context['parent_version'],'parameters':parameters.record(),
        'created_at':datetime.now().astimezone().isoformat(timespec='seconds')}
    save_json(directory/'tone.json',record)
    try:
        result = render_tone(pixels,parameters)
        save_image(result,directory/'final.png')
        save_jpeg(result,directory/'final.jpg')
        record.update(status='complete',files={'final':str(directory/'final.png'),'jpeg':str(directory/'final.jpg')},
            output_dimensions=[pixels.shape[1],pixels.shape[0]])
    except Exception as error:
        record.update(status='failed',error=str(error))
        save_json(directory/'tone.json',record)
        raise
    save_json(directory/'tone.json',record)
    return record


def save_tone_batch(targets, parameters, directory, *, progress=None):
    """Save the same explicit parameters once per frozen saved version, one photo at a time."""
    if not targets:
        raise ValueError('请先勾选要批量调整光色的交付照片')
    report = progress or (lambda message:None)
    directory = Path(directory)
    directory.mkdir(parents=True)
    record = {'kind':'tone_batch','status':'running','parameters':parameters.record(),
        'created_at':datetime.now().astimezone().isoformat(timespec='seconds'),
        'record':str(directory/'batch.json'),'items':[dict(target,status='pending') for target in targets]}
    save_json(directory/'batch.json',record)
    for number,item in enumerate(record['items'],1):
        report(f'正在另存光调 {number}/{len(targets)}：'+item['name'])
        pixels = None
        try:
            if 'error' in item:
                raise ValueError(item['error'])
            pixels = open_photo(item['base'])
            result = save_tone(pixels,parameters,item)
            item.update(status='complete',files=result['files'],
                output_dimensions=result['output_dimensions'],
                saved_record=str(Path(result['files']['final']).parent/'tone.json'))
        except Exception as error:
            item.update(status='failed',error=str(error))
        finally:
            del pixels
        save_json(directory/'batch.json',record)
    complete = sum(item['status']=='complete' for item in record['items'])
    record['status'] = 'complete' if complete==len(targets) else 'partial' if complete else 'failed'
    save_json(directory/'batch.json',record)
    return record


class ToneActions:
    """One registered parameter operation, bound to the editor's selected saved version."""
    def __init__(self):
        self.request = None

    def tools(self):
        if self.request is not None:
            return []
        return [{'type':'function','function':{'name':'adjust_tone',
            'description':'按用户明确要求设置整图曝光、明度曲线、暖冷、绿洋红与饱和度。参数是相对当前编辑底图的绝对值，非累加。'
                '只有用户明确要求保存或另存时 save 才为 true。询问和讨论不调用。',
            'parameters':{'type':'object','properties':{
                **TONE_PROPERTIES,
                'save':{'type':'boolean'}},
                'required':[*TONE_PROPERTIES,'save']}}},
            {'type':'function','function':{'name':'relight_photo',
                'description':'用户明确要求整图重打光并保存时调用。对当前保存版本生成光照参考，'
                    '纳入已有头发与衣物编辑，使用光照场调制原尺寸像素并另存。未保存的手动光调不参与。'
                    '仅曝光、暖冷等固定调整使用 adjust_tone。',
                'parameters':{'type':'object','properties':{'relight_brief':{'type':'string',
                    'description':'用户本条光照要求原文'}},'required':['relight_brief']}}}]

    def call(self, *, relight_brief=None, **tone):
        if relight_brief is not None:
            self.request = {'kind':'relight','brief':relight_brief}
            return {'status':'requested','operation':self.request,'message':'已登记重打光并另存，尚未执行。'}
        return self.adjust_tone(**tone)

    def adjust_tone(self, exposure_ev, shadows, midtones, highlights, save=False, *, warmth=0, tint=0, saturation=1):
        if not isinstance(save,bool):
            raise TypeError('save 须为布尔值')
        parameters = ToneParameters(exposure_ev,shadows,midtones,highlights,warmth,tint,saturation)
        self.request = {'parameters':parameters.record(),'save':save}
        return {'status':'requested','operation':self.request,
            'message':'参数已登记，本条回复保存后执行；图片尚未调整或保存。'}


def discuss_tone(pixels, parameters, conversation, config, directory, *, saved_pixels=None):
    from model_clients import _describe
    actions = ToneActions()
    directory = Path(directory)
    directory.mkdir(parents=True)
    with to_preview(pixels,1280) as image:
        images = [('当前参数的整图预览',preview_data_url(image,1280))]
    if saved_pixels is not None and parameters!=ToneParameters():
        with to_preview(saved_pixels,1280) as image:
            images.append(('当前保存版本；重打光的完整输入',preview_data_url(image,1280)))
    history = [{'role':row['role'],'content':row['content']+
        ('\n实际操作状态：'+row['operation_status'] if 'operation_status' in row else '')}
        for row in conversation]
    result = _describe('你正在已保存照片的光色调整页。'
        '明确修改请求通过 adjust_tone 设置完整参数；其他参数沿用当前值。'
        '固定光调整图统一处理。先在线性光中调整曝光、暖冷和绿洋红通道平衡、'
        '基于明度的饱和度，再截断超出色域的值，随后使用固定端点与三个控制点的平滑单调明度曲线。'
        '暖冷与绿洋红是 -1 至 1 的相对平衡，不是开尔文色温；正值分别偏暖、偏洋红；'
        '饱和度 0 为黑白，1 为原值。恢复初始值指曝光/暖冷/绿洋红 0、饱和度 1、曲线 0.25/0.5/0.75。'
        '用户明确要求整图重打光并保存时用 relight_photo，输入是当前已保存版本，'
        '包含已有局部编辑，不包含尚未保存的手动光调。重打光的光照参考生成另由模板编写与独立审查。'
        '只通过实际工具调用修改，返回 JSON 的 reply 字段作简洁回复。'
        '工具登记后说明即将执行，不声称已经完成；保存仅在本条用户明确要求时发起。',
        images,config,directory/'response.json',task={'parameters':parameters.record()},
        conversation=history,actions=actions,agent_role='CosLuma 的整图光色调整 Agent')
    return {'reply':result['reply'].strip(),'operation':actions.request}
