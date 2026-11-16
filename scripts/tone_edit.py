"""Explicit full-frame exposure and luminance edits on saved float32 photographs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import math
from pathlib import Path
import uuid

import numpy as np

from image_io import apply_tone_curve, preview_data_url, save_image, save_jpeg, save_json, to_preview


@dataclass(frozen=True)
class ToneParameters:
    exposure_ev: float = 0.0
    shadows: float = .25
    midtones: float = .5
    highlights: float = .75

    def __post_init__(self):
        for name, low, high in (('exposure_ev',-2,2),('shadows',.05,.4),
                ('midtones',.4,.6),('highlights',.6,.95)):
            value = getattr(self,name)
            if not math.isfinite(value) or not low<=value<=high:
                raise ValueError(f'{name} 须在 {low} 至 {high} 之间')

    def record(self):
        return asdict(self)

    def points(self):
        return [[0,0],[.25,self.shadows],[.5,self.midtones],[.75,self.highlights],[1,1]]


def render_tone(pixels, parameters):
    """Linear-light EV followed by the existing display-luminance curve; no resampling."""
    if parameters==ToneParameters():
        return pixels
    result = np.empty_like(pixels)
    gain = np.float32(2**parameters.exposure_ev)
    for start in range(0,pixels.shape[0],256):
        tile = pixels[start:start+256]
        if parameters.exposure_ev==0:
            result[start:start+256] = tile
            continue
        linear = np.where(tile<=.04045,tile/np.float32(12.92),
            ((tile+np.float32(.055))/np.float32(1.055))**np.float32(2.4))
        linear *= gain
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


class ToneActions:
    """One registered parameter operation, bound to the editor's selected saved version."""
    def __init__(self):
        self.request = None

    def tools(self):
        if self.request is not None:
            return []
        return [{'type':'function','function':{'name':'adjust_tone',
            'description':'按用户明确要求设置整图曝光与明度曲线。参数是相对当前编辑底图的绝对值，非累加。'
                '只有用户明确要求保存或另存时 save 才为 true。询问和讨论不调用。',
            'parameters':{'type':'object','properties':{
                'exposure_ev':{'type':'number','minimum':-2,'maximum':2},
                'shadows':{'type':'number','minimum':.05,'maximum':.4,'description':'输入明度 0.25 的输出；默认 0.25'},
                'midtones':{'type':'number','minimum':.4,'maximum':.6,'description':'输入明度 0.5 的输出；默认 0.5'},
                'highlights':{'type':'number','minimum':.6,'maximum':.95,'description':'输入明度 0.75 的输出；默认 0.75'},
                'save':{'type':'boolean'}},
                'required':['exposure_ev','shadows','midtones','highlights','save']}}}]

    def call(self, exposure_ev, shadows, midtones, highlights, save=False):
        if not isinstance(save,bool):
            raise TypeError('save 须为布尔值')
        parameters = ToneParameters(exposure_ev,shadows,midtones,highlights)
        self.request = {'parameters':parameters.record(),'save':save}
        return {'status':'requested','operation':self.request,
            'message':'参数已登记，本条回复保存后执行；图片尚未调整或保存。'}


def discuss_tone(pixels, parameters, conversation, config, directory):
    from model_clients import _describe
    actions = ToneActions()
    directory = Path(directory)
    directory.mkdir(parents=True)
    with to_preview(pixels,1280) as image:
        images = [('当前参数的整图预览',preview_data_url(image,1280))]
    history = [{'role':row['role'],'content':row['content']+
        ('\n实际操作状态：'+row['operation_status'] if 'operation_status' in row else '')}
        for row in conversation]
    result = _describe('你正在已保存照片的曝光与明度曲线编辑页。'
        '明确修改请求通过 adjust_tone 设置完整参数；其他参数沿用当前值。'
        '整图统一调整，不修脸、不生成、不局部重打光。曝光先在线性光中调整并截断超出范围的高光，'
        '随后使用固定端点与三个控制点的平滑单调明度曲线。数值归零指曝光 0、曲线 0.25/0.5/0.75。'
        '只通过实际工具调用修改，返回 JSON 的 reply 字段作简洁回复。'
        '工具登记后说明即将执行，不声称已经完成；保存仅在本条用户明确要求时发起。',
        images,config,directory/'response.json',task={'parameters':parameters.record()},
        conversation=history,actions=actions,agent_role='CosLuma 的整图光调节 Agent')
    return {'reply':result['reply'].strip(),'operation':actions.request}
