"""Discuss a frozen saved-photo view and request its existing editing operations."""
from __future__ import annotations

import json
from pathlib import Path

from tone_edit import ToneParameters


class DeliveryActions:
    def __init__(self, context):
        self.context = context
        self.versions = {version['id']:version for version in context['versions']}
        self.request = None

    def tools(self):
        if self.request is not None:
            return []
        return [{'type':'function','function':{'name':'request_saved_photo',
            'description':'按本条明确要求调光、查看比较版本或采用已保存版本为交付。'
                'tone 参数相对指定已保存版本，保存不等于采用为交付。compare 只改左侧查看对象；'
                'adopt 只改交付版本并保留勾选。普通询问和建议不调用。',
            'parameters':{'type':'object','properties':{
                'action':{'type':'string','enum':['tone','compare','adopt']},
                'version_id':{'type':['string','null'],'enum':[*self.versions,None],
                    'description':'任务资料中的完整版本 ID；tone 为 null 时使用发送时交付版本；compare 为 null 时查看原片。'},
                'parameters':{'type':['object','null'],'properties':{
                    'exposure_ev':{'type':'number','minimum':-2,'maximum':2},
                    'shadows':{'type':'number','minimum':.05,'maximum':.4},
                    'midtones':{'type':'number','minimum':.4,'maximum':.6},
                    'highlights':{'type':'number','minimum':.6,'maximum':.95}},
                    'required':['exposure_ev','shadows','midtones','highlights'],
                    'description':'tone 填完整参数，默认值 0 / 0.25 / 0.5 / 0.75；其他操作为 null。'},
                'save':{'type':'boolean','description':'tone 且本条明确要求另存时才为 true。'}},
                'required':['action','version_id','parameters','save']}}}]

    def call(self, action, version_id=None, parameters=None, save=False):
        if action not in ('tone','compare','adopt'):
            raise ValueError('未知成片操作：'+action)
        if not isinstance(save,bool):
            raise TypeError('save 须为布尔值')
        if action=='tone' and version_id is None:
            version_id = self.context['selected_version']
        if version_id is not None:
            self.versions[version_id]
        elif action=='adopt':
            raise ValueError('采用交付版本需要明确指定已保存版本')
        self.request = {'action':action,'version_id':version_id}
        if action=='tone':
            self.request.update(parameters=ToneParameters(**parameters).record(),save=save)
        return {'status':'requested','operation':self.request,
            'message':'操作已登记，回复保存后由桌面执行，尚未修改图片或交付选择。'}


def discuss_delivery(context, conversation, config, directory):
    from model_clients import _describe, discussion_preview
    actions = DeliveryActions(context)
    images = [(image['label'],discussion_preview(image['path'])) for image in context['images']]
    history = []
    for row in conversation:
        content = row['content']
        if row.get('context'):
            content += '\n本条发送时查看资料：'+json.dumps(row['context'],ensure_ascii=False)
        if row.get('operation_status'):
            content += '\n实际操作状态：'+row['operation_status']
        history.append({'role':row['role'],'content':content})
    task = {key:value for key,value in context.items() if key!='images'}
    task['attached_images'] = [image['label'] for image in context['images']]
    directory = Path(directory)
    directory.mkdir(parents=True)
    result = _describe('你在项目作品与交付页，围绕当前照片讨论已有成片。'
        '任务资料冻结发送时的照片、交付版本、左侧比较对象与实际查看模式；历史不随浏览改变。'
        '只对本次附图声称看到了细节，其他版本仅有文字资料。'
        '用户本条明确要求时用 request_saved_photo：compare 查看比较对象，adopt 明确采用已保存版本，'
        'tone 调整指定已保存 PNG 的整图曝光与明度曲线。询问、建议和讨论只回复。'
        '只提供当前照片的版本 ID；“左边”按 comparison_version，“右边”按 selected_version 理解。'
        '左侧为原片时不可采用或调光，请用户先选一个已保存版本。'
        'tone 参数相对所选已保存版本，默认 0 EV 和 0.25/0.5/0.75 曲线；曝光先在线性光中计算并截断高光，'
        '随后用固定黑白端点的单调明度曲线。其他参数保持默认，除非本条有明确要求。'
        '用户要局部变化、生成或单独修脸时说明应使用对应编辑入口。'
        '只有本条明确要求保存才 save=true；另存不采用交付，采用只接受已有版本。'
        '工具登记后回复即将执行，不声称已完成；返回 JSON 的 reply 字段。',
        images,config,directory/'response.json',task=task,conversation=history,actions=actions,
        agent_role='CosLuma 的成片摄影 Agent')
    return {'reply':result['reply'].strip(),'operation':actions.request}
