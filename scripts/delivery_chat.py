"""Discuss a frozen saved-photo view and request its existing editing operations."""
from __future__ import annotations

import json
from pathlib import Path

from creative_edits import normalise_edits
from local_retouch import TASKS
from tone_edit import TONE_PROPERTIES, ToneParameters


def tone_target_summary(targets):
    return [{'name':item['name'],'version':item['parent_version']['label']} if 'parent_version' in item
        else {'name':item['name'],'error':item['error']} for item in targets]


class DeliveryActions:
    def __init__(self, context, user_text):
        self.context = context
        self.user_text = user_text
        self.versions = {version['id']:version for version in context['versions']}
        self.batches = {batch['id']:batch for batch in context['tone_batches']}
        self.request = None

    def tools(self):
        if self.request is not None:
            return []
        return [{'type':'function','function':{'name':'request_saved_photo',
            'description':'按本条明确要求调整光色、打开局部修饰、比较、采用或生成一张新稿。local 只打开对应局部任务并保存本条原文，不自动生成。'
                'tone 参数相对指定已保存版本，保存不等于采用为交付。compare 只改左侧查看对象；'
                'tone_batch 将完整参数用于发送时交付勾选的各自版本并另存；不改勾选或采用版本。'
                'adopt 只改交付版本并保留勾选。revise 沿所选版本所属的基础成片整图改稿，'
                '使用本条用户原文，只制作一张且不采用为交付。普通询问和建议不调用。',
            'parameters':{'type':'object','properties':{
                'action':{'type':'string','enum':['tone','tone_batch','compare','adopt','revise','local']},
                'version_id':{'type':['string','null'],'enum':[*self.versions,None],
                    'description':'任务资料中的完整版本 ID；tone / revise / local 为 null 时使用发送时交付版本；compare 为 null 时查看原片，tone_batch 填 null。'},
                'parameters':{'type':['object','null'],'properties':TONE_PROPERTIES,
                    'required':list(TONE_PROPERTIES),
                    'description':'tone / tone_batch 填完整参数；曝光/暖冷/色调默认 0，曲线 0.25/0.5/0.75，饱和度 1；其他操作为 null。'},
                'save':{'type':'boolean','description':'tone 且本条明确要求另存时才为 true；tone_batch 明确批量另存须为 true；revise 本身就是明确发起新稿。'},
                'local_task':{'type':['string','null'],'enum':[*TASKS,None],
                    'description':'local 操作指定修饰任务；衣物微风使用 clothing。其他操作填 null。'},
                'revision_edits':{'type':['object','null'],'properties':{
                    'hair':{'type':['boolean','null']},'relight':{'type':['boolean','null']}},
                    'required':['hair','relight'],
                    'description':'仅 revise 使用；明确要求开启/关闭修发或人物重打光才填布尔值，null 沿用基础版设置。其他操作填 null。'}},
                'required':['action','version_id','parameters','save','revision_edits','local_task']}}},
            {'type':'function','function':{'name':'complete_delivery',
                'description':'用户明确要求时采用已记录的一轮批量光色成功结果、导出现有勾选版本，或采用该轮并导出。'
                    'adopt_batch 保留勾选；export 复制发送时勾选版本；adopt_batch_and_export 只导出该轮成功采用且发送时勾选的照片。'
                    '导出使用界面已设置的目标父目录，不能任意指定路径，不生成或重新编码。',
                'parameters':{'type':'object','properties':{
                    'action':{'type':'string','enum':['adopt_batch','export','adopt_batch_and_export']},
                    'batch_id':{'type':['string','null'],
                        'enum':[item['id'] for item in self.context['tone_batches']]+[None],
                        'description':'采用时填写本次资料中实际批量记录 ID；单独导出填 null。'},
                    'formats':{'type':['array','null'],'items':{'type':'string','enum':['JPEG','PNG']},
                        'description':'本条明确指定的导出格式；null 沿用发送时界面设置。仅采用时填 null。'}},
                    'required':['action','batch_id','formats']}}}]

    def call(self, action, version_id=None, parameters=None, save=False, revision_edits=None, *, batch_id=None, formats=None, local_task=None):
        if action in ('adopt_batch','export','adopt_batch_and_export'):
            if action!='export':
                self.batches[batch_id]
            if action!='adopt_batch':
                if not self.context['export']['parent']:
                    raise ValueError('请先在交付页设置交付位置，再通过对话导出')
                formats = self.context['export']['formats'] if formats is None else formats
                if not formats or any(kind not in ('JPEG','PNG') for kind in formats):
                    raise ValueError('请选择 JPEG 或 PNG 导出格式')
            self.request = {'action':action,'batch_id':batch_id,'formats':formats}
            return {'status':'requested','operation':self.request,'message':'已登记交付操作，回复保存后执行；尚未采用或导出。'}
        if action not in ('tone','tone_batch','compare','adopt','revise','local'):
            raise ValueError('未知成片操作：'+action)
        if not isinstance(save,bool):
            raise TypeError('save 须为布尔值')
        if action in ('tone','revise','local') and version_id is None:
            version_id = self.context['selected_version']
        if version_id is not None:
            self.versions[version_id]
        elif action=='adopt':
            raise ValueError('采用交付版本需要明确指定已保存版本')
        self.request = {'action':action,'version_id':version_id}
        if action=='tone_batch':
            if not save:
                raise ValueError('批量光色只在本条明确要求另存时执行')
            targets = self.context['tone_targets']
            if not targets:
                raise ValueError('发送时没有勾选交付照片，请先手动勾选')
            self.request.update(version_id=None,parameters=ToneParameters(**parameters).record(),save=True)
        elif action=='tone':
            self.request.update(parameters=ToneParameters(**parameters).record(),save=save)
        elif action=='local':
            TASKS[local_task]
            self.request.update(local_task=local_task,local_brief=self.user_text)
        elif action=='revise':
            version = self.versions[version_id]
            basis = self.versions[version['revision_basis']]
            edits = dict(basis['edits'])
            for key in ('hair','relight'):
                value = (revision_edits or {}).get(key)
                if value is not None:
                    edits[key] = value
            self.request.update(version_id=basis['id'],requested_version_id=version_id,
                revision_brief=self.user_text,edits=normalise_edits(edits))
        return {'status':'requested','operation':self.request,
            'message':'操作已登记，回复保存后由桌面执行，尚未修改图片或交付选择。'}


def discuss_delivery(context, conversation, config, directory):
    from model_clients import _describe, discussion_preview
    actions = DeliveryActions(context,conversation[-1]['content'])
    images = [(image['label'],discussion_preview(image['path'])) for image in context['images']]
    history = []
    for row in conversation:
        content = row['content']
        if row.get('context'):
            content += '\n本条发送时查看资料：'+json.dumps(row['context'],ensure_ascii=False)
        if row.get('operation_status'):
            content += '\n实际操作状态：'+row['operation_status']
        history.append({'role':row['role'],'content':content})
    task = {key:value for key,value in context.items() if key not in ('images','tone_targets','export','tone_batches')}
    task['tone_targets'] = tone_target_summary(context['tone_targets'])
    task['export'] = {'parent':context['export']['parent'],'formats':context['export']['formats'],
        'photos':[{'name':Path(item['source']).name,'version':item['version']} for item in context['export']['choices'].values() if item['included']]}
    task['tone_batches'] = [{key:value for key,value in item.items() if key!='targets'} for item in context['tone_batches']]
    task['attached_images'] = [image['label'] for image in context['images']]
    directory = Path(directory)
    directory.mkdir(parents=True)
    result = _describe('你在项目作品与交付页，围绕当前照片讨论已有成片。'
        '任务资料冻结发送时的照片、交付版本、左侧比较对象与实际查看模式；历史不随浏览改变。'
        '只对本次附图声称看到了细节，其他版本仅有文字资料。'
        '用户本条明确要求时用 request_saved_photo：compare 查看比较对象，adopt 明确采用已保存版本，'
        'tone 调整指定已保存 PNG 的整图曝光、明度曲线与色彩，revise 沿指定版本改一张新稿。'
        '询问、建议和讨论只回复。'
        '本条明确要求把同一组光色参数批量应用并另存给交付勾选照片时，用 tone_batch；'
        '范围固定为 tone_targets，不依赖文件名搜索或当前浏览；version_id=null，save=true。'
        '只请求试调或讨论整组效果时不批量另存，可先用 tone 在单张预览。'
        '批量在各自已选保存版本上使用同一组绝对参数，不自动匹配每张光色，不自动采用交付。'
        '明确要求采用一轮批量光色或导出时，调用 complete_delivery；tone_batches 是已记录的可用轮次，最新一轮排在最前。'
        '采用并导出使用 adopt_batch_and_export，仅导出该轮成功采用且发送时勾选的照片；失败项保留错误，不拿旧版替代。'
        '单独 export 复制发送时已勾选的交付版本；目标位置由界面设置，格式未明确时沿用界面格式。'
        '没有交付位置时请用户先设置；不从消息中直接接受任意路径。询问、比较、满意等表态不算明确采用或导出。'
        '只提供当前照片的版本 ID；“左边”按 comparison_version，“右边”按 selected_version 理解。'
        '左侧为原片时不可采用、调光或作为改稿版本，请用户先选一个已保存版本。'
        'tone 参数相对所选已保存版本，曝光/暖冷/色调默认 0，曲线 0.25/0.5/0.75，饱和度 1；'
        '先在线性光中计算曝光、RGB 平衡与明度饱和度，再截断出色域值，随后用固定黑白端点的单调明度曲线。'
        '暖冷与绿洋红为 -1 至 1 的相对平衡，不是开尔文色温；正值分别偏暖、偏洋红；饱和度 0 黑白、1 原值。'
        '其他参数保持默认，除非本条有明确要求。'
        'revise 复用原片身份与 revision_basis 基础版的方向、参考和该分支已采纳要求；'
        '美化、局部修饰和光调版本只指向所属基础版，其后期像素不继承到整图改稿。'
        '新要求直接使用本条用户原文。修发与重打光默认沿用基础版 edits，只有明确要求才修改开关。'
        '用户明确要求衣物动态、补发或发束造型时使用 local，指定 local_task；基于指定保存版本打开局部编辑，将本条原文存为要求，用户涂选范围后主动生成。衣物动态不走 revise。'
        '改稿会重新生成整幅画面，未指定区域也可能变化，新稿另存后用于比较，不自动采用交付。'
        '单独修脸使用人脸美化入口；单纯曝光、色彩和曲线调整使用固定算法 tone，不走生图。'
        '只有本条明确要求保存才 save=true；另存不采用交付，采用只接受已有版本。'
        '工具登记后回复即将执行，不声称已完成；返回 JSON 的 reply 字段。',
        images,config,directory/'response.json',task=task,conversation=history,actions=actions,
        agent_role='CosLuma 的成片摄影 Agent')
    return {'reply':result['reply'].strip(),'operation':actions.request}
