"""Bound local-edit discussion to one task, painted selection and saved base."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from image_io import preview_data_url, to_preview


class LocalActions:
    def __init__(self, context, user_text):
        self.context = context
        self.user_text = user_text
        self.actions = ['draft','generate']
        if context['result']:
            self.actions += ['recompose']+([] if context.get('writeback_dirty') else ['save'])
        self.request = None

    def tools(self):
        if self.request is not None:
            return []
        return [{'type':'function','function':{'name':'request_local_edit',
            'description':'操作当前局部修饰任务：draft 只修改具体要求；generate 按当前任务生成，发束造型与衣物按新旧轮廓提取；'
                'recompose 用当前涂选范围重新合成已有生成输出，不调用模型；save 另存当前已有结果。'
                '不能改范围、任务、参考图或引擎，询问和建议不调用。',
            'parameters':{'type':'object','properties':{
                'action':{'type':'string','enum':self.actions},
                'append_request':{'type':'boolean',
                    'description':'用户明确新增或修改具体要求时为 true，将本条用户原文追加为最新要求；只要求按已有要求生成或另存时为 false。draft 固定为 true。'},
                'save':{'type':'boolean',
                    'description':'generate / recompose 且用户本条明确要求处理后另存时才为 true；其他情况先预览。'}},
                'required':['action','append_request','save']}}}]

    def call(self, action, append_request=False, save=False):
        if action not in self.actions:
            raise ValueError('当前不可执行的局部修饰操作：'+action)
        if not isinstance(append_request,bool) or not isinstance(save,bool):
            raise TypeError('append_request 与 save 须为布尔值')
        append_request = action=='draft' or (action=='generate' and append_request)
        brief = '\n'.join(value for value in (self.context['brief'].rstrip(),self.user_text) if value)
        self.request = {'action':action,'brief':brief if append_request else None,
            'save':save if action in ('generate','recompose') else False}
        return {'status':'requested','operation':self.request,
            'message':'操作已登记，回复保存后由编辑页执行，尚未生成或另存。'}


def discuss_local(pixels, mask, guide, context, conversation, config, directory, protected=None, prepared=None):
    from local_retouch import retouch_crop
    from model_clients import _describe, discussion_preview
    actions = LocalActions(context,conversation[-1]['content'])
    images = []
    mode = context['view_mode']
    task = dict(context)
    if mode=='paint':
        with to_preview(pixels,max(mask.size)) as original, original.resize(mask.size,Image.Resampling.LANCZOS).convert('RGBA') as marked:
            # Marked inputs are attached only when the user is viewing the painting canvas.
            with Image.new('RGBA',mask.size,(236,92,145,0)) as overlay:
                overlay.putalpha(mask.point(lambda value:round(value*.36)))
                marked.alpha_composite(overlay)
            if protected is not None:
                with Image.new('RGBA',mask.size,(78,141,236,0)) as overlay:
                    overlay.putalpha(protected.point(lambda value:round(value*.44)))
                    marked.alpha_composite(overlay)
            marked.alpha_composite(guide)
            with marked.convert('RGB') as rgb:
                images.append(('本次保存底图；粉色涂选、蓝色保护、青色形状标记',preview_data_url(rgb,1280)))
    else:
        with to_preview(pixels,1280) as original:
            if mode in ('source','compare'):
                images.append(('本次保存底图',preview_data_url(original,1280)))
            if mode in ('output','compare'):
                if context['result']:
                    crop,state = prepared if prepared is not None else retouch_crop(pixels,context['result'])
                    width,height = pixels.shape[1],pixels.shape[0]
                    x0,y0,x1,y1 = state['box']
                    box = tuple(round(value*scale) for value,scale in zip(
                        (x0,y0,x1,y1),(original.width/width,original.height/height)*2))
                    with original.copy() as output, to_preview(crop,1280) as detail:
                        with detail.resize((box[2]-box[0],box[3]-box[1]),Image.Resampling.LANCZOS) as patch:
                            output.paste(patch,box[:2])
                        images.append(('当前修饰结果，完整画幅',preview_data_url(output,1280)))
                    del crop
                    task['result_crop_box'] = state['box']
                else:
                    images.append(('尚无修饰结果，当前显示保存底图',preview_data_url(original,1280)))
    if context['reference']:
        images.append(('用户已选择的修饰参考图',discussion_preview(context['reference'])))
    task['attached_images'] = [label for label,_ in images]
    history = []
    for row in conversation:
        text = row['content']
        if row.get('context'):
            text += '\n本条发送时任务：'+json.dumps(row['context'],ensure_ascii=False)
        if row.get('operation_status'):
            text += '\n实际操作状态：'+row['operation_status']
        history.append({'role':row['role'],'content':text})
    directory = Path(directory)
    directory.mkdir(parents=True)
    result = _describe('你在已保存照片的局部修饰页。任务资料和附图冻结发送时的底图、当前任务、'
        '具体要求、涂选范围、形状标记、参考图与查看模式。只讨论本任务，只对实际附图描述细节。'
        'generation_dirty 表示草稿要求已改，当前附图仍是上次生成结果；不声称新要求已经生效。'
        '用户修改要求或说先记下时用 draft，只保存草稿；明确要求生成时用 generate；'
        '明确要求另存当前已有结果时用 save。询问、建议和比较只回复，不自动执行。'
        '用户已调整涂选范围且明确要求应用到已有生成结果时用 recompose；它只重新合成，复用原始模型输出与已有超分，不生图。'
        '用户明确新增或修改要求时 append_request=true，将本条用户原文追加到具体要求，后者覆盖冲突；'
        '只要求按已有要求生成或另存时为 false。draft 只追加原文，不生成；不代写或扩大用户要求。'
        '范围、形状标记、任务选择、参考图和引擎由用户在控件中设置，工具不能修改。'
        '没有涂选范围时请用户先涂范围，不能自行找区域或改为整图改稿。'
        '生成复用任务模板编写与独立提示词审查，结果默认先预览；'
        '本条明确要求生成并另存才把 save 设为 true，不要求额外审批。'
        '模型会重新生成输入视图；补发仅写回涂选范围，发束造型和衣物按涂选识别原部件，'
        '由生图决定新轮廓，合并实际新旧范围后排除原片前景与用户保护区。最终写回范围外保持底图像素；'
        '不宣称模型原生像素锁定，不将保存等同于采用为交付。'
        '工具登记后回复即将执行，不声称已经完成。返回 JSON 的 reply 字段。',
        images,config,directory/'response.json',task=task,conversation=history,actions=actions,
        agent_role='CosLuma 的局部修饰摄影 Agent')
    return {'reply':result['reply'].strip(),'operation':actions.request}
