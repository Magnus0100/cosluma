"""One finishing chain shared by desktop, CLI and HTTP."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from collections import Counter
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import shutil
import uuid

from image_io import apply_tone_curve, fit_generated, open_photo, open_rgb, save_image, save_jpeg, save_json
from model_clients import edit_image, plan_photos, superresolve, write_prompt
from head_composite import restore_head
from creative_edits import normalise_edits, edits_match
from delivery_store import result_path


def _requested_size(source_size, config, explicit):
    if explicit is not None:
        return tuple(explicit)
    edge = config.get('output_long_edge', 6000)
    if edge:
        return tuple(max(1, round(value*edge/max(source_size))) for value in source_size)
    return tuple(source_size)


def _covered_size(requested, available):
    factor = min(1.0, available[0]/requested[0], available[1]/requested[1])
    return tuple(max(1,min(available[index],round(value*factor))) for index,value in enumerate(requested))


def finish_photo(source, character, root, config, *, brief='', size=None, generated=None,
        superresolved=None, art_direction=None, edits=None, progress=None):
    report = progress or (lambda message: None)
    if art_direction is not None:
        chosen_edits = normalise_edits(art_direction.get('edits'))
        if edits is not None and not edits_match(edits,chosen_edits):
            raise ValueError('修发或人物重打光设置与摄影方向不同，请按当前设置重新试样定稿')
        edits = chosen_edits
    else:
        edits = normalise_edits(edits)
    source = Path(source).expanduser().resolve()
    directory = Path(root).expanduser().resolve() / (source.stem+'-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
    directory.mkdir(parents=True,exist_ok=False)
    record = {'source':str(source),'character':character,'brief':brief,'directory':str(directory),
        'status':'running','stage':'decode','files':{},'sr':[],'edits':edits}
    if art_direction is not None and 'revision_of' in art_direction:
        record.update(revision_of=art_direction['revision_of'],revision_brief=art_direction.get('revision_brief',''),
            revision_requirements=deepcopy(art_direction.get('revision_requirements',[])))
    def stage(name):
        record['stage'] = name
        save_json(directory/'run.json',record)
        report(name)
    try:
        stage('读取原片')
        source_image = open_photo(source)
        source_size = (source_image.shape[1],source_image.shape[0])
        record['source_dimensions'] = list(source_size)
        requested = _requested_size(source_size,config,size)
        record['requested_dimensions'] = list(requested)
        if generated is None and art_direction is None:
            stage('设计场景与光色方向')
            art_direction = plan_photos([source],character,brief,config['prompt_library'],config['llm'],directory,edits=edits)
        if art_direction is not None:
            save_json(directory/'art-direction.json',art_direction)
            record['direction'] = art_direction['direction']
        stage('编写提示词与确定主体头部')
        library = art_direction['references'] if art_direction is not None else []
        description = write_prompt(source_image,character,brief,library,config['llm'],directory,
            direction=art_direction['direction'] if art_direction is not None else '',
            style_reference=art_direction.get('style_reference') if art_direction is not None else None,
            creative_direction=art_direction.get('creative_direction','') if art_direction is not None else '',
            revision_brief=art_direction.get('revision_brief','') if art_direction is not None else '',
            revision_requirements=art_direction.get('revision_requirements',[]) if art_direction is not None else [],edits=edits)
        prompt = description['prompt']
        record['head_box'] = description['head_box']
        if edits['hair']:
            record['headwear'] = description['headwear']
        prompt_path = directory/'prompt.txt'
        prompt_path.write_text(prompt,encoding='utf-8')
        record['files']['prompt'] = str(prompt_path)
        if generated is None:
            stage('生成画面')
            generated = edit_image(source_image,prompt,config['image'],directory,
                style_reference=art_direction.get('style_reference') if art_direction is not None else None)
        else:
            generated = Path(generated).expanduser().resolve()
            record['existing_generated'] = str(generated)
        record['files']['generated_native'] = str(generated)
        with open_rgb(generated) as native:
            record['native_dimensions'] = list(native.size)
        stage('匹配生成画面尺寸')
        if superresolved is None:
            returned,record['sr'] = superresolve(generated,requested,config.get('sr'),directory,stage)
        else:
            returned = Path(superresolved).expanduser().resolve()
            stage('复用已保存的模型超分结果')
        stage('整理输出画幅')
        with open_rgb(returned) as image:
            if superresolved is not None:
                record['sr'] = [{'input':str(generated),'model_return':str(returned),
                    'dimensions':list(image.size),'reused':True}]
            record['model_return_dimensions'] = list(image.size)
            output_size = _covered_size(requested,image.size)
            frame = fit_generated(image,output_size)
        record['output_dimensions'] = list(output_size)
        if output_size != requested:
            record['output_adjustment'] = 'single-pass SR coverage'
            report(f'请求 {requested[0]}×{requested[1]}，实际输出 {output_size[0]}×{output_size[1]}')
        stage('准备原片整头')
        final,record['head'] = restore_head(source_image,frame,record['head_box'],config,directory,stage,
            edits=edits,headwear=record.get('headwear',()))
        del source_image, frame
        record['head_blending'] = record['head']['method']
        if config.get('finish_curve'):
            stage('调整整图明度层次')
            final = apply_tone_curve(final, config['finish_curve'])
            record['finish_curve'] = config['finish_curve']
        stage('合成保存')
        record['files']['final'] = save_image(final,directory/'final.png')
        record['files']['jpeg'] = save_jpeg(final,directory/'final.jpg')
        record['output_bit_depth'] = 16
        record.update(status='complete',stage='完成')
        save_json(directory/'run.json',record)
        report('完成')
        return record
    except Exception as error:
        from urllib.error import HTTPError,URLError
        if isinstance(error,HTTPError):
            message = 'Model service returned HTTP '+str(error.code)
            if getattr(error,'provider_message',''):
                message += ': '+error.provider_message
        elif isinstance(error,URLError):
            message = 'Model service connection failed: '+str(error.reason)
        else:
            message = str(error)
        record.update(status='failed',error={'type':type(error).__name__,'message':message})
        save_json(directory/'run.json',record)
        failure = RuntimeError(f'{record["stage"]}: {message}. Run: {directory}')
        failure.directory = str(directory)
        failure.stage = record['stage']
        raise failure from None


def _failed_photo(source, error):
    result = {'source':str(source),'status':'failed','error':str(error)}
    if getattr(error, 'directory', None):
        result['directory'] = str(error.directory)
        result['stage'] = getattr(error, 'stage', '')
    return result


def beautify_completed(run_path, config, *, skin=0.0, slim=0.0, protect_features=True, session=None, progress=None):
    """Explicit post-production edit; the completed production chain is untouched."""
    from face_beauty import BeautyParameters, BeautySession, prepare_face, prepare_skin, save_beauty
    parameters = BeautyParameters(skin=skin,slim=slim,protect_features=protect_features)
    if not (skin or slim):
        raise ValueError('美肤与瘦脸均为零，原成片无需另存')
    path = Path(run_path).expanduser().resolve()
    run = json.loads(path.read_text(encoding='utf-8'))
    if run['status']!='complete':
        raise ValueError('请提供已完成成片的 run.json')
    base = Path(result_path(path.parent.parent,run,run['files']['final'])) if run.get('batch_directory') else (
        path.parent/Path(run['files']['final']).name)
    if session is None:
        prepared,pixels = prepare_face(base,run['head']['region'],path.parent,config['beauty'],progress)
    else:
        prepared = BeautySession(session)
        if prepared.base_path!=base.resolve():
            raise ValueError('美化会话对应另一张成片，请使用该成片的会话')
        pixels = open_photo(base)
    if skin and prepared.delta is None:
        prepared = prepare_skin(prepared,config['beauty'],progress,pixels=pixels)
    result = save_beauty(pixels,prepared,parameters)
    save_json(path.parent/'beauty-session.json',{'session':prepared.directory.name,
        'parameters':{'skin':skin,'slim':slim,'protect_features':protect_features}})
    return result


def _relocate_sample(value, original, target):
    if isinstance(value,dict):
        return {key:_relocate_sample(item,original,target) for key,item in value.items()}
    if isinstance(value,list):
        return [_relocate_sample(item,original,target) for item in value]
    if isinstance(value,str) and (value==str(original) or value.startswith(str(original)+'\\') or value.startswith(str(original)+'/')):
        return str(target)+value[len(str(original)):]
    return value


def _reuse_style_sample(selected, directory):
    original = Path(selected['result']['directory'])
    target = directory/(original.name+'-selected')
    target.mkdir()
    result = deepcopy(selected['result'])
    for key in ('final','jpeg'):
        photo = Path(result['files'][key])
        copied = target/photo.name
        shutil.copyfile(photo,copied)
        result['files'][key] = str(copied)
    result.update(directory=str(target),style_sample_reused=True,sample_origin=str(original),batch_directory=str(directory))
    save_json(target/'run.json',result)
    return result


def make_style_samples(sources, sample_source, character, root, config, *, brief='', size=None,
        progress=None, planned=None, sample_progress=None, started=None, creative_direction='', discussion=None, edits=None,
        base_style=None):
    """Render three directions, or explicitly revise one complete sample, through finish_photo."""
    report = progress or (lambda message: None)
    edits = normalise_edits(edits)
    sources = sorted(dict.fromkeys(Path(source).expanduser().resolve() for source in sources))
    sample_source = Path(sample_source).expanduser().resolve()
    if sample_source not in sources:
        raise ValueError('样片原片须从当前照片队列中选择')
    directory = Path(root).expanduser().resolve()/('styles-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
    directory.mkdir(parents=True,exist_ok=False)
    state = {'status':'planning','sample_source':str(sample_source),'sources':[str(source) for source in sources],
        'character':character,'brief':brief,'creative_direction':creative_direction,'edits':edits,'options':{},
        'requested_size':list(size) if size is not None else None,'output_long_edge':config.get('output_long_edge',6000)}
    if base_style is not None:
        state.update(kind='revision',revision_of=base_style['selection_file'],option_count=1,
            revision_brief=creative_direction)
    if discussion is not None:
        state['discussion'] = discussion
    save_json(directory/'styles.json',state)
    if started is not None:
        started(str(directory))
    report('沿用完整样片，准备单张改稿' if base_style is not None else '设计三个不同的摄影方向')
    try:
        if base_style is not None:
            if base_style['status']!='complete' or base_style['result']['status']!='complete':
                raise ValueError('请从已完成的真实样片开始改稿')
            plan = deepcopy(base_style['art_direction'])
            plan.pop('revision_requirements',None)
            plan.update(title=base_style['title'].removesuffix(' · 改稿')+' · 改稿',
                summary='沿用这张样片的摄影方向，按本轮讨论与设置修改。',
                creative_direction=creative_direction,edits=dict(edits),
                style_reference=base_style['result']['files']['jpeg'],
                sample_reference_id=base_style['selection_file'],revision_of=base_style['selection_file'],
                revision_brief=creative_direction)
            plans = [plan]
            save_json(directory/'style-options.json',plans)
        else:
            represented = [sample_source] + [source for source in sources if source != sample_source]
            plans = plan_photos(represented,character,brief,config['prompt_library'],config['llm'],directory,
                option_count=3,creative_direction=creative_direction,discussion=discussion,edits=edits)
    except Exception as error:
        message = getattr(error,'provider_message','') or str(error)
        state.update(status='failed',error=message)
        save_json(directory/'styles.json',state)
        action = '准备样片改稿' if base_style is not None else '设计风格样片'
        raise RuntimeError(f'{action}：{message}。目录：{directory}') from None
    state.update(status='rendering')
    save_json(directory/'styles.json',state)
    if planned is not None:
        planned(plans,str(directory))
    def produce(index, plan):
        def stage(message):
            if sample_progress is not None:
                sample_progress(index,message)
        option = {'index':index,'title':plan['title'],'summary':plan['summary'],'art_direction':plan,
            'sample_source':str(sample_source),'styles_directory':str(directory),
            'selection_file':str(directory/f'style-{index+1}.json'),
            'requested_size':list(size) if size is not None else None,'output_long_edge':config.get('output_long_edge',6000)}
        if base_style is not None:
            option.update(kind='revision',revision_of=plan['revision_of'],revision_brief=plan['revision_brief'])
        try:
            result = finish_photo(sample_source,character,directory,config,brief=brief,size=size,
                art_direction=plan,progress=stage)
        except Exception as error:
            result = _failed_photo(sample_source,error)
        option.update(status=result['status'],result=result)
        if result['status'] != 'complete':
            option['error'] = result['error']
        save_json(option['selection_file'],option)
        return option
    with ThreadPoolExecutor(max_workers=len(plans)) as pool:
        pending = {pool.submit(produce,index,plan) for index,plan in enumerate(plans)}
        for future in as_completed(pending):
            pending.remove(future)
            option = future.result()
            state['options'][str(option['index'])] = option
            save_json(directory/'styles.json',state)
            yield option
    state['status'] = 'complete' if all(option['status']=='complete' for option in state['options'].values()) else 'unfinished'
    save_json(directory/'styles.json',state)


def load_style(path):
    """Load an explicitly chosen sample, including a moved styles directory."""
    path = Path(path).expanduser().resolve()
    option = json.loads(path.read_text(encoding='utf-8-sig'))
    if option.get('status') != 'complete' or option.get('result',{}).get('status') != 'complete':
        raise ValueError('所选风格样片没有保存完整成片；请选择已完成的方案')
    previous = Path(option['styles_directory'])
    def locate(value):
        value = Path(value)
        try:
            return str(path.parent/value.relative_to(previous))
        except ValueError:
            return str(value if value.is_absolute() else path.parent/value)
    result = deepcopy(option['result'])
    result['directory'] = locate(option['result']['directory'])
    result['files'] = {key:locate(value) for key,value in option['result']['files'].items()}
    if 'sr' in result:
        result['sr'] = _relocate_sample(result['sr'],previous,path.parent)
    option['result'] = result
    for key in ('final', 'jpeg'):
        if not Path(result['files'][key]).is_file():
            raise FileNotFoundError(f'所选样片的成片文件不存在：{result["files"][key]}')
    option.update(styles_directory=str(path.parent),selection_file=str(path))
    return option


def _organise_outputs(result, directory, name):
    """Keep one completed file per format; both records point at the organised image."""
    result['exports'] = {}
    for kind,key,suffix in (('PNG','final','.png'),('JPEG','jpeg','.jpg')):
        target = directory/kind/(name+suffix)
        target.parent.mkdir(exist_ok=True)
        Path(result['files'][key]).rename(target)
        result['files'][key] = str(target)
        result['exports'][kind] = str(target)
        save_json(Path(result['directory'])/'run.json',result)


def revise_photo(source, character, root, config, *, art_direction, reference, revision_brief,
        revision_of, brief='', size=None, progress=None, batch_started=None):
    """Explicitly make one new version without changing a finalized group direction."""
    report = progress or (lambda message:None)
    source = Path(source).expanduser().resolve()
    plan = deepcopy(art_direction)
    requirements = deepcopy(plan.get('revision_requirements',[]))
    requirements.append(revision_brief)
    plan.update(style_reference=str(Path(reference).expanduser().resolve()),
        revision_brief=revision_brief,revision_of=revision_of,revision_requirements=requirements)
    directory = Path(root).expanduser().resolve()/('batch-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
    directory.mkdir(parents=True,exist_ok=False)
    state = {'status':'running','stage':'制作单张改稿','kind':'revision','total':1,'sources':[str(source)],
        'character':character,'brief':brief,'direction':plan['direction'],'edits':normalise_edits(plan.get('edits')),
        'revision_of':revision_of,'revision_brief':revision_brief,'revision_requirements':requirements,'items':{}}
    save_json(directory/'art-direction.json',plan)
    save_json(directory/'batch.json',state)
    if batch_started is not None:
        batch_started(str(directory))
    try:
        result = finish_photo(source,character,directory,config,brief=brief,size=size,
            art_direction=plan,progress=report)
    except Exception as error:
        result = _failed_photo(source,error)
    result.update(batch_directory=str(directory),revision_of=revision_of,revision_brief=revision_brief,
        revision_requirements=requirements)
    if result['status']=='complete':
        try:
            _organise_outputs(result,directory,source.stem)
        except OSError as error:
            result.update(status='failed',stage='整理成片',
                error=f'整理成片：{error}。已合成文件与整理记录保留在 {directory}')
    state['items'][str(source)] = result
    complete = result['status']=='complete'
    state.update(status='complete' if complete else 'unfinished',stage='完成' if complete else result.get('stage','制作单张改稿'))
    save_json(directory/'batch.json',state)
    return result


def finish_batch(sources, character, root, config, *, brief='', size=None, workers=None,
        progress=None, photo_progress=None, batch_started=None, style=None, edits=None):
    """Run independent photos through finish_photo; each photo is attempted once."""
    report = progress or (lambda message: None)
    sources = sorted(dict.fromkeys(Path(source).expanduser().resolve() for source in sources))
    if not sources:
        raise ValueError('No photographs were selected')
    selected = load_style(style) if style is not None else None
    if selected is not None:
        chosen_edits = normalise_edits(selected['art_direction'].get('edits'))
        if edits is not None and not edits_match(edits,chosen_edits):
            raise ValueError('修发或人物重打光设置与定稿样片不同，请按当前设置重新试样定稿')
        edits = chosen_edits
    else:
        edits = normalise_edits(edits)
    if selected is not None and (selected['result']['character'] != character or selected['result']['brief'] != brief):
        raise ValueError('角色或补充要求与所选样片不同，请按当前要求生成样片后选择')
    names = Counter(source.stem.casefold() for source in sources)
    directory = Path(root).expanduser().resolve()/('batch-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
    directory.mkdir(parents=True,exist_ok=False)
    state = {'status':'running','stage':'设计整组场景与光色方向','total':len(sources),'sources':[str(source) for source in sources],
        'character':character,'brief':brief,'edits':edits,'items':{}}
    save_json(directory/'batch.json',state)
    if batch_started is not None:
        batch_started(str(directory))
    if selected is None:
        report('设计整组场景与光色方向')
        try:
            art_direction = plan_photos(sources,character,brief,config['prompt_library'],config['llm'],directory,edits=edits)
        except Exception as error:
            message = getattr(error,'provider_message','') or str(error)
            state.update(status='failed',error=message)
            save_json(directory/'batch.json',state)
            raise RuntimeError(f'设计整组场景与光色方向：{message}。目录：{directory}') from None
    else:
        report('沿用选中的风格与样片')
        try:
            reference = directory/'selected-style.jpg'
            shutil.copyfile(selected['result']['files']['jpeg'],reference)
            art_direction = deepcopy(selected['art_direction'])
            art_direction['style_reference'] = str(reference)
            art_direction.pop('revision_of',None)
            art_direction.pop('revision_brief',None)
            art_direction.pop('revision_requirements',None)
            save_json(directory/'art-direction.json',art_direction)
        except Exception as error:
            state.update(status='failed',stage='准备选中风格',error=str(error))
            save_json(directory/'batch.json',state)
            raise RuntimeError(f'准备选中风格：{error}。目录：{directory}') from None
        state['selected_style'] = {key:selected[key] for key in ('title','summary','selection_file','sample_source')}
    state.update(stage='制作照片',direction=art_direction['direction'])
    save_json(directory/'batch.json',state)
    workers = config.get('workers', 4) if workers is None else workers
    def produce(source):
        try:
            def report_photo(message):
                if photo_progress is not None:
                    photo_progress(str(source), message)
                report(f'{Path(source).name}：{message}')
            return finish_photo(source, character, directory, config, brief=brief, size=size,art_direction=art_direction,
                progress=report_photo)
        except Exception as error:
            # Return the error text so a Future does not retain a traceback
            # and the full-resolution image buffers in the failed call.
            return _failed_photo(source,error)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {}
        sample_source = Path(selected['result']['source']).resolve() if selected is not None else None
        for source in sources:
            if source == sample_source:
                future = Future()
                try:
                    result = _reuse_style_sample(selected,directory)
                except Exception as error:
                    result = {'source':str(source),'status':'failed','stage':'整理选中样片',
                        'error':f'整理选中样片：{error}。完整样片仍保留在 {selected["result"]["directory"]}',
                        'directory':selected['result']['directory']}
                future.set_result(result)
                if photo_progress is not None:
                    photo_progress(str(source),'复用选中的完整样片')
            else:
                future = pool.submit(produce, source)
            pending[future] = source
        for future in as_completed(pending):
            source = pending.pop(future)
            result = future.result()
            result['batch_directory'] = str(directory)
            if result['status']=='complete':
                name = source.stem
                if names[name.casefold()]>1:
                    name = Path(result['directory']).name
                try:
                    _organise_outputs(result,directory,name)
                except OSError as error:
                    result.update(status='failed',stage='整理成片',
                        error=f'整理成片：{error}。已合成文件与整理记录保留在 {directory}')
            state['items'][str(source)] = result
            save_json(directory/'batch.json',state)
            yield result
    complete = all(row['status']=='complete' for row in state['items'].values())
    state.update(status='complete' if complete else 'unfinished',stage='完成' if complete else '未全部完成')
    save_json(directory/'batch.json',state)
