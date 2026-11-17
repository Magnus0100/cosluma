"""Optional face-beauty discussion using the editor's existing operations."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from face_beauty import BeautyParameters, render_beauty_crop
from image_io import preview_data_url, to_preview


class BeautyActions:
    def __init__(self, context):
        self.context = context
        self.request = None

    def tools(self):
        if self.request is not None or self.context['session'] is None:
            return []
        return [{'type':'function','function':{'name':'request_face_beauty',
            'description':'按明确要求调整已有美化参数、生成美肤基础或另存。adjust 复用固定算法；'
                'generate_skin 才调用既有美肤生成；save 另存当前参数结果。不能改变人脸范围、模型或提示词。',
            'parameters':{'type':'object','properties':{
                'action':{'type':'string','enum':['adjust','generate_skin','save']},
                'parameters':{'type':['object','null'],'properties':{
                    'skin':{'type':'number','minimum':0,'maximum':1},
                    'slim':{'type':'number','minimum':0,'maximum':1},
                    'protect_features':{'type':'boolean'}},
                    'required':['skin','slim','protect_features'],
                    'description':'adjust 填完整参数，其他值沿用当前；强度 0–1，对应滑杆 0–100。生成或另存可填 null 沿用当前值。'},
                'save':{'type':'boolean','description':'adjust 或 generate_skin 且本条明确要求另存时才为 true。'}},
                'required':['action','parameters','save']}}}]

    def call(self, action, parameters=None, save=False):
        if action not in ('adjust','generate_skin','save'):
            raise ValueError('未知人脸美化操作：'+action)
        if self.context['session'] is None:
            raise ValueError('请先手动选择人脸并分析')
        values = BeautyParameters(**(self.context['parameters'] if action=='save' or parameters is None else parameters))
        # The manual controls and recorded operations use the same hundredth steps.
        values = BeautyParameters(round(values.skin*100)/100,round(values.slim*100)/100,values.protect_features)
        if values.skin and not self.context['skin_ready'] and action!='generate_skin':
            raise ValueError('尚无美肤基础，请明确生成美肤基础，或将美肤强度设为零')
        self.request = {'action':action,'parameters':asdict(values),'save':action=='save' or save}
        return {'status':'requested','operation':self.request,
            'message':'操作已登记，由编辑页执行；尚未生成、调整或另存。'}


def discuss_beauty(pixels, session, context, conversation, config, directory):
    from model_clients import _describe
    actions = BeautyActions(context)
    images = []
    if session is not None:
        left,top,right,bottom = session.box
        if context['view_mode'] in ('source','compare'):
            with to_preview(pixels[top:bottom,left:right],1280) as image:
                images.append(('发送时美化前的人脸局部',preview_data_url(image,1280)))
        if context['view_mode'] in ('output','compare'):
            crop = render_beauty_crop(pixels,session,BeautyParameters(**context['preview_parameters']))
            with to_preview(crop,1280) as image:
                images.append(('发送时美化后的人脸局部',preview_data_url(image,1280)))
            del crop
    task = dict(context,attached_images=[label for label,_ in images])
    history = []
    for row in conversation:
        text = row['content']
        if row.get('context'):
            text += '\n发送时资料：'+json.dumps(row['context'],ensure_ascii=False)
        if row.get('operation_status'):
            text += '\n实际执行状态：'+row['operation_status']
        history.append({'role':row['role'],'content':text})
    directory = Path(directory)
    directory.mkdir(parents=True)
    result = _describe('你在用户主动打开的可选人脸美化页。只通过 request_face_beauty 操作。'
        '范围由用户手动选择，未分析时请先选择人脸并分析，不自行分析或找脸。'
        '附图只包含发送时查看模式的人脸局部，不能声称看到未附区域或原始模型输出。'
        'parameters 是当前滑杆值，preview_parameters 是上次成功显示的值。'
        '明确调整强度或五官保护用 adjust，提供完整参数，未提到的沿用当前值；'
        '强度 0–1 对应界面 0–100，全部归零指美肤和瘦脸都为零，保留五官保护设置。'
        '美肤和瘦脸复用现有基础和固定算法，不会重新生成。'
        '只有明确要求生成或重新生成美肤基础才用 generate_skin，它复用既有模板编写、独立审查和官方 BeautyGRPO 指令；'
        '不能调整模型、提示词或人脸范围。没有美肤基础时不自动生成，应说明需要主动生成。'
        '生成不自行改变强度，用户明确指定参数才设置，其他值沿用当前。'
        '只有本条明确要求另存才 save=true；单独另存当前结果用 action=save 且 parameters=null。'
        '询问、比较、建议和满意表态只回复，不调用工具。另存不等于采用交付版本。'
        '工具登记后说明即将执行，不声称已经完成。返回 JSON 的 reply 字段。',
        images,config,directory/'response.json',task=task,conversation=history,actions=actions,
        agent_role='CosLuma 的可选人脸美化 Agent')
    return {'reply':result['reply'].strip(),'operation':actions.request}
