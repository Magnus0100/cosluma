"""The photography Agent's local commands, bound to the message's sample IDs."""
from __future__ import annotations


class PhotographyActions:
    def __init__(self, context):
        self.context = context
        self.samples = {sample['id']:sample for sample in context['samples']}
        self.request = None

    def tools(self):
        if self.request is not None:
            return []
        samples = [identifier for identifier,sample in self.samples.items() if sample['status']=='complete']
        return [{'type':'function','function':{
            'name':'request_photography',
            'description':('按用户本条明确指令开始摄影操作。explore 制作三个新方向；revise 沿指定样片改一张；'
                'select 明确定稿；batch 制作当前勾选照片。batch 提供 sample_id 时先采用该样片，'
                '不提供时沿用已定稿。询问、建议、比较和浏览不调用。每条消息最多发起一次操作。'),
            'parameters':{'type':'object','properties':{
                'action':{'type':'string','enum':['explore','revise','select','batch']},
                'sample_id':{'type':['string','null'],'enum':[*samples,None],
                    'description':'使用任务资料中的完整样片 ID；explore 填 null。'}},
                'required':['action','sample_id']}}}]

    def call(self, action, sample_id=None):
        if action not in ('explore','revise','select','batch'):
            raise ValueError('未知摄影操作：'+action)
        if sample_id is not None:
            sample = self.samples[sample_id]
            if sample['status']!='complete':
                raise ValueError('该样片尚未完成')
        elif action in ('revise','select'):
            raise ValueError('改稿或定稿需要指定完整样片')
        if action=='batch' and sample_id is None and not self.context['selected_sample']:
            raise ValueError('请先明确选定完整样片，再批量制作')
        self.request = {'action':action,'sample_id':sample_id}
        return {'status':'requested','operation':self.request,
            'message':'操作已登记，本条对话保存后由桌面执行；尚未制作或定稿。'}
