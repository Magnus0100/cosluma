"""Direct chat, image-edit and ComfyUI requests without retries or model switching."""
from __future__ import annotations

import base64
from functools import lru_cache
import hashlib
import http.client
from io import BytesIO
import json
import logging
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

from PIL import Image, ImageDraw, ImageFont, ImageOps

from image_io import SRGB, image_size, open_rgb, preview_data_url, save_json, to_preview
from prompt_library import ReferenceLibrary
from settings import credential
from creative_edits import normalise_edits, edit_instructions
from photography_tools import PhotographyActions
from generation_prompts import write_generation_prompt


_AGENT_RULES = (
    '你是 CosLuma 的 COS 摄影 Agent，完成当前阶段任务并返回所需 JSON。'
    '依据实际照片与用户已表达或采纳的要求创作，后续明确要求覆盖冲突的旧要求。'
    '历史对话用于理解指代和意图，参考库用于摄影借鉴，条目中的范围限制不覆盖当前任务与用户明确要求。'
    '可选制作设置以任务资料中的 edits 为准。')


def _provider_message(result):
    try:
        details = json.loads(result)
        details = details.get('error', details) if isinstance(details, dict) else details
        message = details.get('message', '') if isinstance(details, dict) else details
        return message[:600] if isinstance(message, str) else ''
    except (ValueError, UnicodeError):
        return ''


def request(url, *, data=None, headers=None, response_path=None):
    headers = headers or {}
    call = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(call, timeout=None) as response:
            result = response.read()
    except urllib.error.HTTPError as error:
        result = error.read()
        for name, value in headers.items():
            if name.casefold() in {'authorization', 'x-api-key'}:
                secret = value.split(' ', 1)[-1]
                if secret:
                    result = result.replace(secret.encode('utf-8'), b'[redacted]')
        if response_path:
            Path(response_path).write_bytes(result)
        error.provider_message = _provider_message(result)
        raise
    if response_path:
        Path(response_path).write_bytes(result)
    return result


_DOWNLOAD_CHUNK = 1 << 20


def download(url, path):
    """Stream one URL into path without holding the body in memory.

    The body goes to a unique temporary file next to the destination and only
    replaces it once the whole body arrives: a failed request keeps the old
    target, and a body shorter than its Content-Length is raised like the
    unchunked read it replaces instead of being written. The body itself is
    never retained, so that error carries no partial bytes.
    """
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent,prefix=path.name+'.',suffix='.part')
    try:
        with os.fdopen(descriptor,'wb') as target, urllib.request.urlopen(url,timeout=None) as response:
            expected = getattr(response,'length',None)
            shutil.copyfileobj(response,target,_DOWNLOAD_CHUNK)
            written = target.tell()
        if expected is not None and written != expected:
            raise http.client.IncompleteRead(b'',expected)
        os.replace(temporary,path)
    except BaseException as failure:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        if isinstance(failure, urllib.error.HTTPError):
            failure.provider_message = _provider_message(failure.read())
        raise
    return path


def post_json(url, payload, *, key=None, response_path=None):
    headers = {'Content-Type': 'application/json'}
    if key:
        headers['Authorization'] = 'Bearer ' + key
    data = request(url, data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
        headers=headers, response_path=response_path)
    return json.loads(data)


def multipart(fields, name, filename, data, content_type='image/png', *, additional_files=()):
    boundary = 'cosluma-' + uuid.uuid4().hex
    parts = []
    for key, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode('utf-8'))
    for field, file, pixels, mime in [(name,filename,data,content_type),*additional_files]:
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; filename="{file}"\r\nContent-Type: {mime}\r\n\r\n'.encode('utf-8'))
        parts.extend([pixels,b'\r\n'])
    parts.append(f'--{boundary}--\r\n'.encode('ascii'))
    return b''.join(parts), 'multipart/form-data; boundary=' + boundary


def _describe(instruction, images, config, response_path, *, task, conversation=(), references=None,
        actions=None, agent_role='CosLuma 的 COS 摄影 Agent'):
    task = dict(task)
    if references is not None:
        instruction += (
            '\n按需查阅摄影参考：根据原片与用户要求搜索相关分类，读取少量条目后作答，也可直接回答。'
            '工具依次提供一次搜索、一次读取；reference_catalogue 是分类导航，最终引用实际读到的条目。')
        task['reference_catalogue'] = references.overview()
    content = [{'type':'text','text':'本次任务资料：\n'+json.dumps(task,ensure_ascii=False)}]
    for label,url in images:
        content.extend([{'type':'text','text':label},
            {'type':'image_url','image_url':{'url':url}}])
    rules = _AGENT_RULES.replace('CosLuma 的 COS 摄影 Agent',agent_role)
    messages = [{'role':'system','content':rules+'\n当前阶段要求：\n'+instruction}]
    if conversation:
        messages.extend(conversation[:-1])
        content.insert(0,{'type':'text','text':conversation[-1]['content']})
    messages.append({'role':'user','content':content})
    # Keep the actual text/attachment order without duplicating image payloads or credentials.
    input_messages = []
    for row in messages:
        value = row['content']
        if isinstance(value,list):
            value = [({'type':'image_url', 'sha256':hashlib.sha256(
                part['image_url']['url'].encode('utf-8')).hexdigest(),
                'detail':part['image_url'].get('detail','auto')}
                if part.get('type')=='image_url' else dict(part)) for part in value]
        input_messages.append({'role':row['role'],'content':value})
    trace = {'model':config['model'],'agent_role':agent_role,'input_messages':input_messages,'calls':[]}
    response_path = Path(response_path)
    trace_path = response_path.with_name(response_path.stem+'-calls.json')
    started = time.perf_counter()
    operation = 'request'
    failure = None
    try:
        reference_phase = 0
        for phase in range(1 + (2 if references is not None else 0) + (1 if actions is not None else 0)):
            payload = {'model':config['model'], 'stream':False, 'messages':messages}
            tools = references.tools(reference_phase) if references is not None else []
            if actions is not None:
                tools += actions.tools()
                if actions.request is not None:
                    tools = []
            if tools:
                payload.update(tools=tools, tool_choice='auto', parallel_tool_calls=False)
            entry = {'phase':phase+1,'offered_tools':[tool['function']['name'] for tool in tools],
                'response_file':str(response_path)}
            trace['calls'].append(entry)
            operation = 'request'
            call_started = time.perf_counter()
            try:
                response = post_json(config['url'], payload, key=credential(config), response_path=response_path)
            finally:
                entry['seconds'] = round(time.perf_counter()-call_started,4)
            operation = 'response'
            choice = response['choices'][0]
            entry.update(finish_reason=choice.get('finish_reason'),usage=response.get('usage'))
            message = choice['message']
            calls = message.get('tool_calls')
            if not calls:
                break
            if len(calls)!=1 or calls[0]['function']['name'] not in [tool['function']['name'] for tool in tools]:
                raise ValueError('Agent 每步只能调用一个当前提供的工具；原始响应已保存')
            lookup_path = response_path.with_name(response_path.stem+f'-lookup-{phase+1}.json')
            save_json(lookup_path,response)
            entry['response_file'] = str(lookup_path)
            call = calls[0]
            arguments = json.loads(call['function']['arguments'])
            operation = call['function']['name']
            if actions is not None and operation in [tool['function']['name'] for tool in actions.tools()]:
                result = actions.call(**arguments)
                entry['requested_operation'] = actions.request
            else:
                result = references.call(operation, arguments)
                reference_phase += 1
            messages.append({'role':'assistant', 'content':message.get('content'), 'tool_calls':calls})
            messages.append({'role':'tool', 'tool_call_id':call['id'], 'content':json.dumps(result,ensure_ascii=False)})
            if references is not None:
                save_json(response_path.with_name(response_path.stem+'-references.json'), references.lookups)
        operation = 'output_json'
        content = message.get('content')
        if isinstance(content,list):
            content = '\n'.join(row['text'] for row in content if isinstance(row,dict)
                and row.get('type')=='text' and isinstance(row.get('text'),str))
        content = (content or '').strip()
        if content.startswith('```') and '\n' in content:
            content = content.split('\n',1)[1].rsplit('```',1)[0].strip()
        result = json.loads(content)
        trace['outcome'] = 'returned_json'
        return result
    except Exception as error:
        failure = error
        trace.update(outcome='failed',error={'operation':operation,'type':type(error).__name__})
        if operation=='output_json':
            trace['error'].update(finish_reason=choice.get('finish_reason'),refusal=bool(message.get('refusal')))
        if isinstance(error,urllib.error.HTTPError):
            trace['error']['http_status'] = error.code
        raise
    finally:
        trace['elapsed_seconds'] = round(time.perf_counter()-started,4)
        try:
            save_json(trace_path,trace)
        except OSError as error:
            if failure is None:
                logging.getLogger(__name__).warning('模型调用记录未保存：%s：%s',trace_path,error)
            else:
                failure.add_note('模型调用记录保存失败：'+str(error))


def _selection_sheets(sources, directory, report):
    """Decode each original once into numbered, uncropped sRGB contact sheets."""
    images = []
    font = ImageFont.load_default(size=24)
    for start in range(0,len(sources),12):
        page = start//12+1
        with Image.new('RGB',(2560,1536),'#eef1f4') as sheet:
            draw = ImageDraw.Draw(sheet)
            for cell,source in enumerate(sources[start:start+12]):
                number = start+cell+1
                report(f'准备选片预览 {number}/{len(sources)}')
                x,y = cell%4*640,cell//4*512
                draw.rectangle((x,y,x+639,y+31),fill='#202934')
                draw.text((x+12,y+16),str(number),font=font,fill='white',anchor='lm')
                with open_rgb(source,max_edge=640) as image:
                    thumbnail = ImageOps.contain(image,(640,480),method=Image.Resampling.LANCZOS)
                    sheet.paste(thumbnail,(x+(640-thumbnail.width)//2,y+32+(480-thumbnail.height)//2))
                    thumbnail.close()
            stream = BytesIO()
            sheet.save(stream,format='JPEG',quality=93,subsampling=0,icc_profile=SRGB)
            pixels = stream.getvalue()
        (directory/f'selection-sheet-{page}.jpg').write_bytes(pixels)
        label = f'选片总览第 {page} 页，编号 {start+1}–{min(start+12,len(sources))}'
        images.append((label,'data:image/jpeg;base64,'+base64.b64encode(pixels).decode('ascii')))
    return images


def _selection_result(result, sources):
    """Resolve a complete, unambiguous numbered response back to import order."""
    if len(result['photos'])!=len(sources):
        raise ValueError(f'选片响应须逐张覆盖全部 {len(sources)} 张原片；原始响应已保存')
    photos = {}
    for photo in result['photos']:
        number = photo['id']
        if type(number) is not int or not 1<=number<=len(sources) or number in photos:
            raise ValueError('选片编号须逐张对应原片，不得重复或超出范围；原始响应已保存')
        if type(photo['selected']) is not bool or not isinstance(photo['reason'],str):
            raise ValueError('选片记录的 selected 须为布尔值、reason 须为文本；原始响应已保存')
        photos[number] = photo
    representative = result['representative_id']
    if any(photo['selected'] for photo in photos.values()):
        if type(representative) is not int or representative not in photos or not photos[representative]['selected']:
            raise ValueError('代表原片必须引用一张已勾选的照片；原始响应已保存')
    elif representative is not None:
        raise ValueError('没有勾选照片时，代表原片须为 null；原始响应已保存')
    return {'summary':result['summary'],
        'photos':[{'source':source,'selected':photos[index]['selected'],'reason':photos[index]['reason']}
            for index,source in enumerate(sources,1)],
        'representative':sources[representative-1] if representative is not None else None}


def select_photos(sources, character, brief, config, directory, *, selection_brief='', progress=None):
    """Suggest a selection from every imported photo using one vision request."""
    sources = [str(source) for source in sources]
    if not sources:
        raise ValueError('没有可供选片的原片')
    directory = Path(directory)
    report = progress or (lambda message:None)
    images = _selection_sheets(sources,directory,report)
    files = [{'id':index,'filename':Path(source).name} for index,source in enumerate(sources,1)]
    instruction = (
        '你是帮助用户整理 COS 摄影原片的选片 Agent，只提出保留建议，不制作或修改照片。'
        '总览图覆盖本次全部原片，按数字编号对应下方文件名表，未编号的空白格不是照片。'
        '请跨页比较全组，而非每页独立选片。结合选片要求，比较表情、构图、姿态、服装与道具可见性，'
        '识别连拍中表达近似的重复画面，并保留有价值的近景、中景、远景与动作多样性。'
        '不评分，不设分数阈值或固定保留比例；建议保留与不保留都要给出简短、具体的画面理由。'
        '角色与补充要求可为空；未知角色设定不编造，只依据用户说明和照片中可见内容。'
        '这些是按朝向整理的 8 位 sRGB 缩略总览，不能据此断言原像素锐度、精确对焦或细微清晰度。'
        '返回 JSON，包含且只有 summary、photos、representative_id。summary 是本组取舍的简短中文总结；'
        f'photos 按编号逐张返回全部 {len(sources)} 张，每张恰好一条，不能遗漏、重复或自行增加编号。'
        '每条包含且只有 id（原片整数编号）、selected（JSON 布尔值）、reason（中文理由）。'
        'representative_id 是一张 selected 为 true、适合作为后续创作试样代表的原片编号；'
        '优先考虑妆造、服装与道具表达清楚、能代表本组的照片。全部不选时 representative_id 必须为 null。'
        '任务资料中的 photos 是编号与文件名表，character、initial_brief、selection_brief 是用户输入；'
        '选片要求留空时按本组画面表现与多样性给出建议。')
    report('Agent 正在选片')
    result = _describe(instruction,images,config,directory/'selection-response.json',
        task={'character':character,'initial_brief':brief,'selection_brief':selection_brief,'photos':files})
    selection = _selection_result(result,sources)
    save_json(directory/'selection.json',{'character':character,'brief':brief,'selection_brief':selection_brief,
        'sources':sources,**selection})
    return selection


def _representatives(sources, *, required=()):
    selected = list(dict.fromkeys(required))
    remaining = [source for source in sources if source not in selected]
    count = min(6-len(selected),len(remaining))
    return selected + [remaining[round(index*(len(remaining)-1)/max(1,count-1))] for index in range(count)]


@lru_cache(maxsize=16)
def _discussion_preview(path, modified_ns):
    """Reuse display data for unchanged files; the modification time is part of the key."""
    with open_rgb(path,max_edge=1280) as image:
        return preview_data_url(image,1280)


def discussion_preview(path):
    path = Path(path)
    return _discussion_preview(str(path),path.stat().st_mtime_ns)


def _discussion_samples(context, conversation):
    """Keep stable IDs for all candidates available during successful discussions."""
    samples = {sample['id']:sample for sample in context['samples']}
    for row in conversation:
        if row['role']!='user' or row.get('status')=='failed' or 'context' not in row:
            continue
        previous = row['context']
        for sample in previous['samples']:
            if sample['id'] not in samples:
                samples[sample['id']] = sample
    return [dict(sample,selected=sample['id']==context['selected_sample']) for sample in samples.values()]


def _discussion_messages(conversation):
    """Project local records to provider messages, keeping each turn's visual referents."""
    messages = []
    for row in conversation:
        if row.get('status')=='failed':
            continue
        content = row['content']
        if row.get('direction_reverted'):
            content = '此条创作要求更新已由用户撤回，仅保留作讨论记录，不沿用其新增要求：\n'+content
        if row['role']=='user' and 'context' in row:
            context = row['context']
            anchors = {key:context[key] for key in ('label','viewed_source','viewed_sample',
                'view_mode','next_sample_source','selected_sample')}
            if 'viewed_samples' in context:
                anchors['viewed_samples'] = context['viewed_samples']
            anchors['comparison_sample'] = context.get('comparison_sample')
            anchors['edits'] = normalise_edits(context.get('edits'))
            anchors['samples'] = [{key:sample[key] for key in ('id','label','source','status')} for sample in context['samples']]
            content = json.dumps({'user_message':content,'discussion_context':anchors},ensure_ascii=False)
        if row['role']=='assistant' and row.get('operation') is not None:
            content = json.dumps({'reply':content,'operation':row['operation'],
                'operation_status':row.get('operation_status','not_dispatched'),
                'operation_error':row.get('operation_error')},ensure_ascii=False)
        messages.append({'role':row['role'],'content':content})
    return messages


def _visible_sample_ids(context):
    mode = context['view_mode']
    ids = set(context.get('viewed_samples',[])) if mode=='samples' else set()
    if mode in ('output','compare') and context['viewed_sample'] is not None:
        ids.add(context['viewed_sample'])
    if mode in ('source','compare') and context.get('comparison_sample') is not None:
        ids.add(context['comparison_sample'])
    return ids


def _discussion_image_ids(context, conversation):
    ids = {sample['id'] for sample in context['samples'] if sample['id']!=context.get('comparison_sample')}
    ids.update(_visible_sample_ids(context))
    if context['selected_sample'] is not None:
        ids.add(context['selected_sample'])
    for row in reversed(conversation):
        if row['role']=='user' and row.get('status')!='failed' and 'context' in row:
            ids.update(_visible_sample_ids(row['context']))
            break
    return ids


def _sample_images(samples, image_ids):
    return [('真实样片：'+sample['label']+'；样片 ID：'+sample['id'],discussion_preview(sample['image']))
        for sample in samples if sample['status']=='complete' and sample['id'] in image_ids]


def _discussion_details(context, samples, image_ids):
    details = {key:value for key,value in context.items() if key!='samples'}
    details['samples'] = [dict({key:value for key,value in sample.items() if key!='image'},
        has_image=sample['status']=='complete' and sample['id'] in image_ids) for sample in samples]
    return details


def user_requirement_text(*sources):
    """Merge user wording in order, keeping the latest occurrence of repeated text."""
    text = []
    for source in sources:
        if source is None:
            continue
        values = [source] if isinstance(source, str) else list(source)
        for value in values:
            if isinstance(value, str) and value.strip():
                value = value.strip()
                if value in text:
                    text.remove(value)
                text.append(value)
    return text


def _typed_briefs(edits):
    """The two briefs the user typed, used only while the user switched them on."""
    edits = normalise_edits(edits)
    return [edits[key] for enabled, key in (('hair', 'hair_brief'), ('relight', 'lighting_brief')) if edits[enabled]]


def discussion_user_requirements(discussion, edits):
    """Replay adopted wording; a direct full edit replaces earlier requirements."""
    quotes = []
    previous = None
    for row in ([] if discussion is None else discussion['messages']):
        if row.get('role') == 'user':
            previous = row
            if (row.get('status') != 'failed' and not row.get('direction_reverted')
                    and row.get('direction_action') == 'edit'
                    and isinstance(row.get('direction_update'), str)):
                quotes = [row['direction_update']]
            continue
        evidence = row.get('direction_evidence')
        content = (previous or {}).get('content')
        if (row.get('role') != 'assistant' or row.get('status') == 'failed'
                or (previous or {}).get('status') == 'failed' or row.get('direction_action') is not None
                or not row.get('direction_changed') or row.get('direction_reverted')
                or not isinstance(evidence, str) or not isinstance(content, str) or evidence not in content):
            continue
        if row.get('direction_update') == '':
            quotes = []
        else:
            quotes.append(evidence)
    return user_requirement_text(quotes, _typed_briefs(edits))


def discuss_style(sources, character, brief, library_path, config, directory, conversation, *, direction='', context=None, edits=None):
    """Discuss photographs and request one explicit operation through local tools."""
    images = []
    selected = _representatives(sources) if context is None else context['sources']
    for source in selected:
        images.append(('原片：'+Path(source).name+'；素材 ID：'+str(source),discussion_preview(source)))
    samples = [] if context is None else _discussion_samples(context,conversation)
    image_ids = set() if context is None else _discussion_image_ids(context,conversation[:-1])
    images.extend(_sample_images(samples,image_ids))
    instruction = (
        '你是与用户共同创作 COS 摄影后期的摄影 Agent，现在处于讨论与试样阶段。'
        '观察原片的妆造、道具、构图与受光，结合已有样片，用自然中文讨论具体的摄影选择及理由。'
        '角色信息以用户说明、可靠常识和可见素材为依据，需要关键信息时简短提问。'
        '身份、脸型、机位与姿态保留，实物服装和道具默认保留，具体修补按用户要求处理。'
        '每条历史消息的 discussion_context 冻结当时查看对象，“这张”“第二张”按该条消息的轮次和稳定 ID 理解。'
        'viewed_source 是原片，viewed_sample 是当前稿，comparison_sample 是改稿前样片；view_mode 说明实际查看方式。'
        'samples 模式下 viewed_samples 是同时可见样片，selected_sample 是独立保存的定稿。'
        'has_image 标记本次实际附图，其他历史样片可依据已有文字资料讨论。'
        '创作建议由当前素材形成；修发与重打光按 edits 状态讨论，特效按单张构图决定。'
        '用户本条明确要求试样、改稿、采用样片或制作整组时，调用 request_photography。'
        '明确说先做一张首稿时使用 make_first_draft，明确想看多个方向才使用 explore。'
        '操作只在回复保存后执行，reply 说明即将开始，不声称操作已完成。普通讨论、建议、比较与浏览不调用。'
        '改稿和定稿使用上下文的完整样片 ID；明确采用并制作可用 batch 和该 sample_id。'
        '只说开始批量时沿用 selected_sample；未定稿且用户未指定时先询问选哪张，不自行替用户定稿。'
        '返回 JSON：reply、direction_update、direction_evidence。reply 为自然对话。'
        'current_creative_direction 是现有完整要求；只有本条用户明确增加、修改、取消要求或采纳建议时才更新。'
        '更新时 direction_update 写完整新要求，保留未取消的内容；空字符串表示明确清空。'
        '单纯“采用这张”“开始制作”等操作指令不改写视觉要求，也不将操作步骤写入 direction_update。'
        'direction_evidence 逐字引用本条用户表达修改或采纳的完整短句。普通询问、比较与浏览时两者均为 null。'
        '要求只写已采纳的视觉意图；样片引用写清轮次与名称，物件取舍按用户实际表达。'
        +edit_instructions(edits))
    actions = PhotographyActions(context) if context is not None else None
    result = _describe(instruction,images,config,directory/('chat-'+uuid.uuid4().hex[:8]+'-response.json'),
        task={'character':character,'initial_brief':brief,'current_creative_direction':direction,
            'discussion_context':_discussion_details(context,samples,image_ids) if context is not None else None,
            'edits':normalise_edits(edits)},
        conversation=_discussion_messages(conversation), references=ReferenceLibrary(library_path),actions=actions)
    update = result['direction_update']
    evidence = result['direction_evidence']
    if (update is None and evidence is not None) or (update is not None and
            (not isinstance(evidence,str) or not evidence.strip() or evidence not in conversation[-1]['content'])):
        raise ValueError('创作要求更新没有引用本条用户原文的明确依据；原始响应已保存，已有要求保持')
    return {'reply':result['reply'].strip(),'direction_update':update.strip() if update is not None else None,
        'direction_evidence':evidence,'operation':actions.request if actions is not None else None}


def plan_photos(sources, character, brief, library_path, config, directory, *, option_count=None, creative_direction='', discussion=None, edits=None):
    """Design directions for the supplied character and photos, using library references."""
    edits = normalise_edits(edits)
    references = ReferenceLibrary(library_path)
    selected = (_representatives(sources) if discussion is None else
        _representatives(discussion['context']['sources'],required=(str(sources[0]),)))
    images = []
    for index,source in enumerate(selected):
        with open_rgb(source) as image:
            label = ('本次试样原片：' if index==0 else '讨论对照原片：') if discussion is not None else '原片：'
            images.append((label+Path(source).name+'；素材 ID：'+str(source),preview_data_url(image,1280)))
    if option_count is None:
        task = ('直接定一个有角色氛围、精致、真实且可执行的整组摄影方向，不反问、不列候选。'
            '返回 JSON，只有 direction 和 reference_ids。')
    elif option_count == 1:
        task = (
            '像接到这一组原片的专业后期师，直接给出你推荐制作的一张首稿，不要求客户先选风格。'
            '从本次试样原片指出一个已经成立、值得保留的画面特点，以及怎样通过一项明确的处理让它更好。'
            '要求留空时，从原片已有的妆造、构图和现场光出发。让光照在肤色上留下现场本来的冷暖；'
            '把处理落到看得见的地方及想达到的效果。发现原片已经成立的地方，直接沿用。'
            '返回 JSON，只有 options，值为一个对象组成的数组，包含 title、summary、direction、reference_ids。'
            'title 按这张图的处理建议命名；summary 用一到两句向客户交代本次要做的改动。'
            'direction 是同一份建议给修图师的执行说明，写清主处理和必要的配合，范围与 summary 一致。')
    else:
        task = (f'为用户设计 {option_count} 个可供真实样片比较的整组摄影方向。'
            '方向须在用户允许的修改范围内有可辨认的视觉差异，不只改名称。'
            '允许换景时可探索场景叙事；用户保留原场景或只要求调色时，'
            '通过明暗分配、色彩关系、光质和情绪形成差异，不为凑三个方向越过用户允许的修改范围。'
            '每个方向都适合这组角色与服装，按本次是否启用重打光决定能否改变原片光向，不为凑差异让人物与环境曝光失配。'
            f'返回 JSON，只有 options，值为恰好 {option_count} 个对象。每个对象包含 title、summary、direction、reference_ids。'
            'title 根据本次设计命名，不能使用固定风格菜单；'
            'summary 用一到两句话说明画面、光感、氛围，以及为什么适合当前角色和原片；不包含模型或算法术语。')
    samples = []
    inputs = {'character':character,'initial_brief':brief,'current_creative_direction':creative_direction,'edits':edits}
    if discussion is not None:
        samples = _discussion_samples(discussion['context'],discussion['messages'])
        image_ids = _discussion_image_ids(discussion['context'],discussion['messages'])
        images.extend(_sample_images(samples,image_ids))
        task += (
            '每个方向还必须包含 sample_reference_id，取值为 null 或下方一张已完成真实样片的完整 ID。'
            '用户明确沿用样片时填写对应 ID，浏览或比较时填 null。样片只作风格参考，身份与构图来自本张原片。'
            'successful_discussion 的消息上下文记录当时轮次、ID 与查看方式；has_image 表示本次附图。'
            'comparison_sample 为改稿前样片，viewed_sample 为当前稿，viewed_source 为原片。'
            '按当前要求与明确沿用的场景范围设计。')
        inputs.update(discussion_context=_discussion_details(discussion['context'],samples,image_ids),
            successful_discussion=_discussion_messages(discussion['messages']))
    instruction = (
        '你负责 COS 摄影后期的方向设计。观察代表原片的实物妆造、道具、构图与受光，'
        '结合角色、用户要求和实际参考形成摄影意图、方向与名称。' + task +
        'direction 写给单张修图师：说明想让观众先看到什么，哪些可见区域需要怎样处理，处理后应是什么样。'
        '只写这组照片实际要用的决定；换景、重打光等复杂创作可以写足必要细节。'
        'reference_ids 只列本次实际读过且用到的条目 ID。'
        'current_creative_direction 覆盖 initial_brief 中的冲突。依据角色与原片设计，默认沿用原片妆造、衣装和道具。'
        '修肤、磨皮或单独调整肤色属于用户点名才做的精修；光色方案可以照顾人物受光。'
        '未启用修发或重打光时，分别沿用原发型或现场光向；开启时按对应要求设计。')
    name = 'style-options' if option_count is not None else 'art-direction'
    description = _describe(instruction,images,config,directory/(name+'-response.json'),task=inputs,references=references)
    descriptions = description['options'] if option_count is not None else [description]
    if not isinstance(descriptions, list) or (option_count is not None and len(descriptions) != option_count):
        raise ValueError('提示词服务没有返回请求数量的摄影方向；原始响应已保存')
    user_requirements = None if discussion is None else discussion_user_requirements(discussion, edits)
    plans = []
    for item in descriptions:
        plan = {'direction':item['direction'].strip(),'references':references.selected(item['reference_ids']),
            'representative_sources':[str(source) for source in selected],'creative_direction':creative_direction,
            'edits':dict(edits)}
        if discussion is not None:
            plan['user_requirements'] = user_requirements
            reference_id = item['sample_reference_id']
            plan['sample_reference_id'] = reference_id
            if reference_id is not None:
                reference = next((sample for sample in samples if sample['id']==reference_id and sample['status']=='complete'),None)
                if reference is None:
                    raise ValueError('摄影方向引用了未提供的完整样片；原始响应已保存')
                plan['style_reference'] = reference['image']
        if option_count is not None:
            for key in ('title', 'summary'):
                plan[key] = item[key].strip()
        plans.append(plan)
    result = plans if option_count is not None else plans[0]
    save_json(directory/(name+'.json'),result)
    return result


def write_prompt(source, character, brief, library, config, directory, *, direction='', style_reference=None,
        creative_direction='', revision_brief='', revision_requirements=None, edits=None, user_requirements=None,
        revision_user_text='', generate=True, progress=None):
    edits = normalise_edits(edits)
    verbatim = user_requirement_text(brief, _typed_briefs(edits), user_requirements,
        [value for value in (revision_requirements or []) if isinstance(value, str)], revision_user_text)
    schema = ('返回一个 JSON 对象，只有 prompt、head_box 和 headwear 三个字段。' if edits['hair'] else
        '返回一个 JSON 对象，只有 prompt 和 head_box 两个字段。')
    instruction = (
        schema+'head_box 是本张原片主体整头的边界框 [左,上,右,下]，坐标归一化到 0 到 1000。'
        '范围完整包住主体的脸、假发和头饰并留少量空间；镜中正脸为作品主体时定位该头部。'
        'group_direction 是摄影建议，current_creative_direction 是用户采纳的当前要求。'
        'revision_requirements 是父版本分支已采纳的要求，latest_revision_brief 是本次修改；沿用其中未被替换的要求。'
        '有风格样片时，在 prompt 说明第一张决定人物和构图，第二张只提供本次需要沿用的摄影风格。')
    if edits['hair']:
        instruction += (
            '按 edits.hair_brief 修发；留空时修补露网、缺口和杂发，沿用原有发型。'
            '有明确造型要求时写出目标，让生图结合原发束与整图决定未指定的细节。')
    else:
        instruction += '沿用原片整头与发型。'
    if edits['relight']:
        instruction += '按 edits.lighting_brief 或摄影方向重建人物和环境的统一受光，同一面部几何用于光照迁移。'
    else:
        instruction += '沿用现场受光方向，可协调曝光与色温。'
    if edits['hair']:
        instruction += (
            '\nheadwear 是本张原片实际可见、须保留的实体头饰英文名称数组，每项为具体物体名，无头饰时为空数组。'
            'user_requirements 是已采纳的用户原话；group_direction 与 current_creative_direction 是归纳方向。'
            '宽大透光头纱由生成画面保留衣装外观与受光，只有用户原话明确要求保留原片头纱时才列入 headwear。')
    save_json(directory / 'prompt-sources.json', library)
    images = [('本张原片，头部定位只依据此图',preview_data_url(source))]
    if style_reference is not None:
        with open_rgb(style_reference) as image:
            images.append(('用户选定的真实风格样片，仅参考摄影风格',preview_data_url(image,1280)))
    task = {'character':character,'initial_brief':brief,'current_creative_direction':creative_direction,
            'group_direction':direction,'revision_requirements':revision_requirements or [],
            'latest_revision_brief':revision_brief,'user_requirements':verbatim,
            'selected_references':[{key:item[key] for key in ('title','text')} for item in library],'edits':edits}
    if generate:
        description = write_generation_prompt('scene',images,config,directory,
            task=task,instruction=instruction,progress=progress)
    else:
        instruction += '复用已有生成图，按原片定位；未提供方向时 prompt 留空。'
        description = _describe(instruction,images,config,directory/'llm-response.json',task=task)
    box = description['head_box']
    left,top,right,bottom = box
    if not (all(type(value) in (int,float) and math.isfinite(value) for value in box)
            and 0<=left<right<=1000 and 0<=top<bottom<=1000):
        raise ValueError('头部范围须满足 0<=左<右<=1000 与 0<=上<下<=1000（按原片归一化坐标）；原始响应已保存')
    if edits['hair']:
        description['headwear'] = list(dict.fromkeys(name.strip() for name in description['headwear']))
    return description


def edit_image(source, prompt, config, directory, *, name='generated', style_reference=None):
    # Send an oriented sRGB JPEG of the full frame for scene design. The
    # final compositing uses the full-resolution source pixels.
    stream = BytesIO()
    to_preview(source, config.get('input_max_edge',1536)).save(stream, format='JPEG', quality=95, subsampling=0, icc_profile=SRGB)
    image_bytes = stream.getvalue()
    (directory / (name + '-input.jpg')).write_bytes(image_bytes)
    fields = {**config.get('parameters', {}), 'model': config['model'], 'prompt': prompt, 'n': 1}
    references = []
    if style_reference is not None:
        with open_rgb(style_reference) as reference:
            reference.thumbnail((1280,1280))
            buffer = BytesIO()
            reference.save(buffer,format='JPEG',quality=95,subsampling=0,icc_profile=SRGB)
        pixels = buffer.getvalue()
        (directory/(name+'-style-reference.jpg')).write_bytes(pixels)
        references.append(('image[]','selected-style.jpg',pixels,'image/jpeg'))
    body, content_type = multipart(fields, 'image[]', 'original.jpg', image_bytes, 'image/jpeg',additional_files=references)
    headers = {'Content-Type': content_type}
    key = credential(config)
    if key:
        headers['Authorization'] = 'Bearer ' + key
    response_bytes = request(config['url'], data=body, headers=headers,
        response_path=directory / (name + '-response.json'))
    response = json.loads(response_bytes)
    del response_bytes, body, image_bytes
    result = response['data'][0]
    encoded = result.pop('b64_json',None)
    inline = bool(encoded)
    pixels = None
    path = directory / (name + '-native.png')
    if inline:
        pixels = base64.b64decode(encoded)
        del encoded
        path.write_bytes(pixels)
    else:
        download(result['url'],path)
    with open_rgb(path) as image:
        image.load()
    if inline:
        response['cosluma_saved_image'] = {'file':path.name,'bytes':len(pixels),
            'extracted_from':'data[0].b64_json'}
        save_json(directory/(name+'-response.json'),response)
    return path


def comfy_upscale(source, config, directory):
    base = config['url'].rstrip('/')
    stream = BytesIO()
    with open_rgb(source) as image:
        image.save(stream, format='PNG')
    body, content_type = multipart({'type': 'input', 'overwrite': 'false'}, 'image',
        'cosluma-' + uuid.uuid4().hex + '.png', stream.getvalue())
    uploaded = json.loads(request(base + '/upload/image', data=body, headers={'Content-Type': content_type}))
    filename = '/'.join(part for part in (uploaded.get('subfolder'), uploaded['name']) if part)
    graph = {
        '1': {'class_type': 'LoadImage', 'inputs': {'image': filename}},
        '2': {'class_type': 'UpscaleModelLoader', 'inputs': {'model_name': config['model']}},
        '3': {'class_type': 'ImageUpscaleWithModel', 'inputs': {'upscale_model': ['2', 0], 'image': ['1', 0]}},
        '4': {'class_type': 'SaveImage', 'inputs': {'images': ['3', 0], 'filename_prefix': 'CosLuma/' + uuid.uuid4().hex}}}
    save_json(directory / 'sr-workflow.json', graph)
    queued = post_json(base + '/prompt', {'prompt': graph, 'client_id': 'cosluma-' + uuid.uuid4().hex})
    save_json(directory / 'sr-submission.json', queued)
    prompt_id = queued['prompt_id']
    while True:
        history = json.loads(request(base + '/history/' + urllib.parse.quote(prompt_id)))
        if prompt_id in history:
            result = history[prompt_id]
            save_json(directory / 'sr-history.json', result)
            if result.get('status', {}).get('status_str') == 'error':
                errors = [row[1].get('exception_message','')
                    for row in result['status'].get('messages',[]) if row[0]=='execution_error']
                details = '; '.join(errors)
                raise RuntimeError('ComfyUI SR failed'+(': '+details if details else ''))
            output = result.get('outputs', {}).get('4', {}).get('images')
            if output:
                path = directory / 'generated-sr.png'
                download(base + '/view?' + urllib.parse.urlencode(output[0]),path)
                return path
            if result.get('status', {}).get('completed'):
                raise RuntimeError('ComfyUI finished without an SR image; see the saved history file')
        time.sleep(1)


def superresolve(source, target, config, directory, progress):
    """Run the configured model once, then let the caller crop or downsample."""
    width,height = image_size(source)
    if max(target[0]/width,target[1]/height) <= 1:
        return Path(source), []
    progress('模型超分 1/1')
    result = comfy_upscale(source, config, directory)
    records = [{'input': str(source), 'model_return': str(result),
        'dimensions': list(image_size(result))}]
    return result, records


def extract_head_mask(image, config, directory, *, edit_hair=False, headwear=()):
    """Extract the whole head and, for authorised hair edits, the protected identity."""
    base = config['url'].rstrip('/')
    stream = BytesIO()
    image.save(stream,format='PNG')
    body,content_type = multipart({'type':'input','overwrite':'false'},'image',
        'cosluma-head-'+uuid.uuid4().hex+'.png',stream.getvalue())
    uploaded = json.loads(request(base+'/upload/image',data=body,headers={'Content-Type':content_type}))
    filename = '/'.join(part for part in (uploaded.get('subfolder'),uploaded['name']) if part)
    graph = {
        '1':{'class_type':'LoadImage','inputs':{'image':filename}},
        '2':{'class_type':'CheckpointLoaderSimple','inputs':{'ckpt_name':config['model']}},
        '3':{'class_type':'CLIPTextEncode','inputs':{'clip':['2',1],
            'text':'head:1, hair:1, headdress:1'}},
        '4':{'class_type':'SAM3_Detect','inputs':{'model':['2',0],'image':['1',0],
            'conditioning':['3',0],'threshold':.5,'refine_iterations':1,'individual_masks':False}},
        '5':{'class_type':'MaskToImage','inputs':{'mask':['4',0]}},
        '6':{'class_type':'SaveImage','inputs':{'images':['5',0],
            'filename_prefix':'CosLuma/head-'+uuid.uuid4().hex}}}
    if edit_hair:
        graph.update({
            '7':{'class_type':'CLIPTextEncode','inputs':{'clip':['2',1],
                'text':'face:1, ear:2'}},
            '8':{'class_type':'SAM3_Detect','inputs':{'model':['2',0],'image':['1',0],
                'conditioning':['7',0],'threshold':.5,'refine_iterations':1,'individual_masks':False}},
            '9':{'class_type':'MaskToImage','inputs':{'mask':['8',0]}},
            '10':{'class_type':'SaveImage','inputs':{'images':['9',0],
                'filename_prefix':'CosLuma/face-'+uuid.uuid4().hex}}})
        if headwear:
            graph.update({
                '15':{'class_type':'CLIPTextEncode','inputs':{'clip':['2',1],
                    'text':', '.join(name+':4' for name in headwear)}},
                '16':{'class_type':'SAM3_Detect','inputs':{'model':['2',0],'image':['1',0],
                    'conditioning':['15',0],'threshold':.5,'refine_iterations':1,'individual_masks':False}},
                '17':{'class_type':'MaskToImage','inputs':{'mask':['16',0]}},
                '18':{'class_type':'SaveImage','inputs':{'images':['17',0],
                    'filename_prefix':'CosLuma/headwear-'+uuid.uuid4().hex}}})
    save_json(directory/'head-workflow.json',graph)
    queued = post_json(base+'/prompt',{'prompt':graph,'client_id':'cosluma-'+uuid.uuid4().hex})
    prompt_id = queued['prompt_id']
    while True:
        history = json.loads(request(base+'/history/'+urllib.parse.quote(prompt_id)))
        if prompt_id in history:
            result = history[prompt_id]
            save_json(directory/'head-history.json',result)
            if result.get('status',{}).get('status_str')=='error':
                errors = [row[1].get('exception_message','') for row in result['status'].get('messages',[])
                    if row[0]=='execution_error']
                raise RuntimeError('Head extraction failed: '+'; '.join(errors))
            outputs = result.get('outputs',{})
            masks = {'head':('6','head-mask-return.png')}
            if edit_hair:
                masks['face'] = ('10','face-mask-return.png')
                if headwear:
                    masks['headwear'] = ('18','headwear-mask-return.png')
            if all(outputs.get(node,{}).get('images') for node,_ in masks.values()):
                returned = {}
                for name,(node,filename) in masks.items():
                    path = directory/filename
                    download(base+'/view?'+urllib.parse.urlencode(outputs[node]['images'][0]),path)
                    returned[name] = open_rgb(path)
                return returned
            if result.get('status',{}).get('completed'):
                missing = [name for name,(node,_) in masks.items() if not outputs.get(node,{}).get('images')]
                raise RuntimeError('SAM3.1 未返回所需蒙版：'+', '.join(missing))
        time.sleep(1)
