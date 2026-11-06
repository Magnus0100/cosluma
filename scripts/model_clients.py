"""Direct chat, image-edit and ComfyUI requests without retries or model switching."""
from __future__ import annotations

import base64
from io import BytesIO
import json
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

from image_io import SRGB, open_rgb, preview_data_url, save_json, to_preview
from prompt_library import catalogue, load_library
from settings import credential


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
        try:
            details = json.loads(result)
            details = details.get('error', details) if isinstance(details, dict) else details
            message = details.get('message', '') if isinstance(details, dict) else details
            error.provider_message = message[:600] if isinstance(message, str) else ''
        except (ValueError, UnicodeError):
            error.provider_message = ''
        raise
    if response_path:
        Path(response_path).write_bytes(result)
    return result


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


def _describe(instruction, images, config, response_path):
    content = [{'type':'text','text':instruction}]
    for label,url in images:
        content.extend([{'type':'text','text':label},
            {'type':'image_url','image_url':{'url':url}}])
    response = post_json(config['url'], {'model':config['model'],'stream':False,
        'messages':[{'role':'user','content':content}]}, key=credential(config),response_path=response_path)
    content = response['choices'][0]['message']['content']
    if isinstance(content,list):
        content = '\n'.join(row['text'] for row in content if row.get('type')=='text')
    if not isinstance(content,str) or not content.strip():
        raise ValueError('The prompt writer returned no description; its response was saved')
    content = content.strip()
    if content.startswith('```'):
        content = content.split('\n',1)[1].rsplit('```',1)[0].strip()
    return json.loads(content)


def plan_photos(sources, character, brief, library_path, config, directory, *, option_count=None):
    """Design directions for the supplied character and photos, using library references."""
    entries = load_library(library_path)
    count = min(6,len(sources))
    selected = [sources[round(index*(len(sources)-1)/max(1,count-1))] for index in range(count)]
    images = []
    for source in selected:
        with open_rgb(source) as image:
            images.append((Path(source).name,preview_data_url(image,1280)))
    if option_count is None:
        task = ('直接定一个有角色氛围、精致、真实且可执行的整组摄影方向，不反问、不列候选。'
            '返回 JSON，只有 direction 和 reference_ids。')
    else:
        task = (f'为用户设计 {option_count} 个可供真实样片比较的整组摄影方向。'
            '方向应在场景叙事、材质、配色或情绪上有可辨认的差异，不能只是给同一场景换滤镜或改名称。'
            '每个方向都适合这组角色、服装与原片受光，不为凑差异选择无法自然融合的极暗夜景。'
            f'返回 JSON，只有 options，值为恰好 {option_count} 个对象。每个对象包含 title、summary、direction、reference_ids。'
            'title 根据本次设计命名，不能使用固定风格菜单；'
            'summary 用一到两句话说明画面、光感、氛围，以及为什么适合当前角色和原片；不包含模型或算法术语。')
    instruction = (
        '你负责 COS 摄影后期的审美决策。先结合本次用户填写的角色、补充要求和这组代表原片，'
        '观察实物服装、道具、构图与受光，再为当前素材设计摄影方向。'
        '方向与名称均在看过这些输入之后形成，不预先指定场景或固定几类风格。' + task +
        '成片应有明确的摄影创作意图，不能只清除几个杂物、改成灰蓝色就结束。'
        'direction 用约 300 到 600 字说明具体场景、时间氛围、色彩、曝光层次、自然光来源、背景取舍和特效原则；'
        'reference_ids 从下方目录选取少量最相关的条目编号，通常 2 到 4 条，不需要特效时不要为了用素材而选特效。'
        '用户补充要求优先。没有补充要求时，由你依据角色的可靠常识、原片场地、实物妆造和受光决定；'
        '角色常识用于环境与情绪，不用于改变原片实物服装，不确定的角色设定不要编造。'
        '分清原拍摄场地与适合角色的创作场景：保留机位、尺度、人物支撑和接触几何，'
        '创作场景的语义和具体景物应依据当前角色与素材确定，不必保留缺少主题的原背景。'
        '未知角色只依据用户输入与照片可见信息设计，不把推测当成角色设定。'
        '场景必须在实拍透视、尺度、材质和景深上成立，不能用泛泛的梦幻背景或游戏概念图敷衍。'
        '先判断照片里的主光软硬、明暗和色温，再选择相容的场景与时间；除非用户明确要求，'
        '不把普通白天人像强改成极暗夜景，不引入原片没有的强烈条纹光或遮脸树影。'
        '后续会把原片整头（含脸、头发和头饰）整体等比融合回画面，保留生成的身体与环境；'
        '这不是空间重打光，方向必须让原片头部、颈部、身体、近处景物的曝光和受光自然连续。'
        '保留光向与软硬不等于保留原片的欠曝、灰雾或色偏；允许整体调整头部曝光和色温，'
        '让人物与环境共同达到有层次、明净且有真实光感的完成影调。'
        '明确统一场景语义、材质、配色和光质；每张的光线方向适应各自原片，不强行固定同一画面左右。'
        '清理路人、设备、包、垃圾、杂乱标识和破坏主题的现代建筑，保留合理接触、遮挡与必要道具。'
        '近景、俯拍与远景各按真实机位表达同一场景，不能每张硬加地平线、月亮或天空。'
        '人像保持自然有血色的肤色、脸部体积和清楚眼神，皮肤与衣物有真实细节，黑纱有深度、白纱不过曝；'
        '通透来自清楚的明暗层次，不来自蒙白、全图染青、磨皮或毫无来由地提亮人物。'
        'COS 精致感优先来自光色、干净背景与材质；特效按单张手势、视线、情绪、留白判断，允许完全不加。'
        'direction 只确定特效的适用理由与视觉语言，不替所有照片统一决定全部添加或全部禁用。'
        '目录仅供场景、布光和色彩灵感，是参考素材而不是候选风格菜单，也不限制可设计的场景范围；'
        '先确定适合当前角色与原片的创作意图，再选能帮助实现它的具体参考，不把参考标题直接凑成候选。'
        '条目里其他人物、换装、广告版式和固定画幅均不适用。'
        f'\n角色：{character}\n补充要求：{brief or "无，由你决定"}\n提示词目录：\n'+catalogue(entries))
    name = 'style-options' if option_count is not None else 'art-direction'
    description = _describe(instruction,images,config,directory/(name+'-response.json'))
    descriptions = description['options'] if option_count is not None else [description]
    if not isinstance(descriptions, list) or (option_count is not None and len(descriptions) != option_count):
        raise ValueError('提示词服务没有返回请求数量的摄影方向；原始响应已保存')
    plans = []
    for item in descriptions:
        direction = item['direction']
        if not isinstance(direction,str) or not direction.strip():
            raise ValueError('The prompt writer returned no photographic direction')
        indices = list(dict.fromkeys(item['reference_ids']))
        if any(type(index) is not int or index<0 or index>=len(entries) for index in indices):
            raise ValueError('The prompt writer selected an unknown library reference')
        plan = {'direction':direction,'references':[entries[index] for index in indices],
            'representative_sources':[str(source) for source in selected]}
        if option_count is not None:
            for key in ('title', 'summary'):
                if not isinstance(item.get(key), str) or not item[key].strip():
                    raise ValueError('摄影方向缺少名称或画面说明；原始响应已保存')
                plan[key] = item[key].strip()
        plans.append(plan)
    result = plans if option_count is not None else plans[0]
    save_json(directory/(name+'.json'),result)
    return result


def write_prompt(source, character, brief, library, config, directory, *, direction='', style_reference=None):
    instruction = (
        '你为没有修图灵感的用户设计 COS 摄影后期。观察原片，结合用户给出的角色、要求和下面的提示词库，'
        '返回一个 JSON 对象，只有 prompt 和 head_box 两个字段。prompt 是可交给图像编辑模型的完整提示词。'
        'head_box 是原片中需要保留的一个主体整头的边界框 [左,上,右,下]，坐标归一化到 0 到 1000。'
        '边界框完整包住脸、假发和头饰，保留少量周围空间；不选择水面倒影、屏幕或照片中的次要人像。'
        '当镜中清晰正脸是作品主体时选择该头部，不选择前景背对镜头的后脑。'
        '提供了整组方向时，本张只据构图、姿态和原片光向细化，不另起一套场景或影调。'
        '提供用户选定的真实样片时，以它的场景语义、材质、配色、曝光层次和光质为视觉参照；'
        '本张仍按第一张本张原片的构图、姿态与受光细化，不照搬样片机位或人物动作。'
        '生图模型也会收到原片作为第一张输入、风格样片作为第二张输入；有样片时在 prompt 中明确这两个职责，'
        '第二张仅参考摄影风格，不复制其中人物、姿态、脸型或构图。'
        'head_box 只定位本张原片的主体，不能从风格样片取坐标；样片不是身份回贴素材。'
        '复用已有生成图且未提供方向时，只确定原片主体头部，不重新设计场景。'
        '近景保持真实景深，俯拍不硬加地平线；倒影、镜像、必要的前景虚焦人物和接触关系均完整保留。'
        '提示词库只提供场景、光色、材质和特效灵感；其中其他角色的妆造、换脸塑形、换衣、'
        '重新构图、固定画幅、模型参数和运行流程均不适用于本次任务。'
        '后续将原片整头（脸、假发和头饰）整体等比对齐后融合回生成画面；身体、实物服装和环境保留生成结果。'
        '必须保持原片的人物位置、大小、姿态、脸型骨相、完整轮廓、实物服装结构和画幅，'
        '统一设计人物与环境的主光、环境反光、接触阴影和景深，使人物自然处于同一场景。'
        '整头形状、位置、大小和朝向必须贴合原片，场景主光方向应与原片头部受光相容；'
        '人物身体、颈部与环境的亮度、色温和阴影深度必须协调；'
        '不能把白天曝光的人物直接放进暗夜背景。主光要有可解释的来源，近处地面和道具承接同一光照。'
        '允许头部与整体统一调整曝光和色温；保留原片光向、脸部体积和真实细节，不照搬原片欠曝、灰雾与色偏。'
        '肤色干净且有血色、眼神清楚，通透依靠明暗层次与材质细节，不等于人物始终高亮，也不以整图染色代替光感。'
        '清理背景中与作品无关的路人、包和器材、垃圾、抢眼标识及零碎干扰，修复被清理区域的纹理与线条，'
        '按用户要求保留或移除具体景物，维护透视、景深与接触关系，不把保留实景理解为照搬杂乱背景。'
        '现代建筑、幕墙、游乐设施、密集树岸等若与已选场景冲突，主动清理并用该场景真实结构补全，'
        '不要等待用户逐件指出。原有自然环境若适合主题，保留其机位、结构和合理细节。'
        '特效依据单张照片的手势、视线、留白、景深和情绪决定，允许完全不加；'
        '有明确视觉动机时才克制使用，亮度和位置服务于人物，不用固定数量或全组统一开关。'
        '完成一张人物与环境受光协调的精致摄影成片，生成头部的光色将作为原片整头融合参考。'
        '不要换人、改衣、移动人物或强改拍摄视角，不要输出多个候选、对比图、文字或水印。'
        'prompt 是直接编辑照片的简洁指令，不向图像模型解释后续程序、抠图和融合算法。'
        f'\n角色：{character}\n用户要求：{brief}\n整组摄影方向：{direction}\n选定参考：\n'
        + '\n\n'.join(item['title'] + '\n' + item['text'] for item in library))
    save_json(directory / 'prompt-sources.json', library)
    images = [('本张原片，头部定位只依据此图',preview_data_url(source))]
    if style_reference is not None:
        with open_rgb(style_reference) as image:
            images.append(('用户选定的真实风格样片，仅参考摄影风格',preview_data_url(image,1280)))
    description = _describe(instruction,images,config,directory/'llm-response.json')
    if not isinstance(description['prompt'],str) or len(description['head_box'])!=4:
        raise ValueError('The LLM did not return an edit prompt and one head box')
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
    result = response['data'][0]
    pixels = base64.b64decode(result['b64_json']) if result.get('b64_json') else request(result['url'])
    path = directory / (name + '-native.png')
    path.write_bytes(pixels)
    with open_rgb(path) as image:
        image.load()
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
                path.write_bytes(request(base + '/view?' + urllib.parse.urlencode(output[0])))
                return path
            if result.get('status', {}).get('completed'):
                raise RuntimeError('ComfyUI finished without an SR image; see the saved history file')
        time.sleep(1)


def superresolve(source, target, config, directory, progress):
    """Run the configured model once, then let the caller crop or downsample."""
    with open_rgb(source) as image:
        ratio = max(target[0] / image.width, target[1] / image.height)
        if ratio <= 1:
            return Path(source), []
    progress('模型超分 1/1')
    result = comfy_upscale(source, config, directory)
    with open_rgb(result) as image:
        records = [{'input': str(source), 'model_return': str(result),
            'dimensions': list(image.size)}]
    return result, records


def extract_head_mask(image, config, directory):
    """Run the configured SAM model once on the original head crop."""
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
            images = result.get('outputs',{}).get('6',{}).get('images')
            if images:
                path = directory/'head-mask-return.png'
                path.write_bytes(request(base+'/view?'+urllib.parse.urlencode(images[0])))
                return open_rgb(path)
            if result.get('status',{}).get('completed'):
                raise RuntimeError('Head extraction finished without a mask')
        time.sleep(1)
