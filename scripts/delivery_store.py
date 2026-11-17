"""Saved project photographs, explicit versions and unchanged delivery copies."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import uuid

from creative_edits import edits_description, edits_label, normalise_edits
from image_io import save_json


def source_key(source):
    return os.path.normcase(str(Path(source).expanduser().resolve()))


def result_path(directory, result, value):
    """Resolve recorded files against the batch's current, explicitly opened location."""
    if not value:
        return None
    path = Path(value)
    directory = Path(directory)
    if not path.is_absolute():
        return str(directory/path)
    previous = result.get('batch_directory')
    if previous:
        try:
            return str(directory/path.relative_to(Path(previous)))
        except ValueError:
            pass
    return str(path)


def project_photos(batches):
    """Group completed results by original, retaining every explicitly saved version."""
    photos = {}
    for number, directory in enumerate(batches,1):
        directory = Path(directory)
        try:
            state = json.loads((directory/'batch.json').read_text(encoding='utf-8'))
            for source in state.get('sources',list(state['items'])):
                result = state['items'].get(source)
                if result is None or result['status']!='complete':
                    continue
                key = source_key(source)
                photo = photos.setdefault(key,{'key':key,'source':source,'name':Path(source).name,'versions':[]})
                work = result_path(directory,result,result['directory'])
                exports = result.get('exports',{})
                files = {kind:result_path(directory,result,exports[kind] if kind in exports else result['files'][field])
                    for kind,field in (('PNG','final'),('JPEG','jpeg'))}
                edits = normalise_edits(result.get('edits'))
                treatment = edits_label(edits)
                revision_brief = result.get('revision_brief','')
                description = edits_description(edits)
                if result.get('revision_of'):
                    requirements = result.get('revision_requirements',[])
                    description += ('\n本张已采纳要求（后者覆盖冲突）：\n'+'\n'.join(requirements)
                        if requirements else '\n本次改稿要求：'+revision_brief)
                revision_label = ' · 改稿' if result.get('revision_of') else ''
                common = {'batch':str(directory),'work':work,'dimensions':result['output_dimensions'],
                    'base':result_path(directory,result,result['files']['final']),'edits':edits,
                    'description':description,'direction':result.get('direction',''),
                    'revision_of':result.get('revision_of'),'revision_brief':revision_brief,
                    'revision_requirements':result.get('revision_requirements',[])}
                base = {'id':files['PNG'],'kind':'base','label':f'制作 {number}{revision_label} · {treatment} · 基础成片',
                    'short_label':f'制作 {number}{revision_label} · {treatment}','files':files,**common}
                if not any(version['id']==base['id'] for version in photo['versions']):
                    photo['versions'].append(base)
                for edit_number,path in enumerate(sorted(Path(work).glob('beauty-*/edit-*/beauty.json')),1):
                    if any(version['id']==str(path) for version in photo['versions']):
                        continue
                    saved = json.loads(path.read_text(encoding='utf-8'))
                    if saved['status']!='complete':
                        continue
                    parameters = saved['parameters']
                    label = f'制作 {number}{revision_label} · {treatment} · 美化 {edit_number}（美肤 {round(parameters["skin"]*100)} / 瘦脸 {round(parameters["slim"]*100)}）'
                    beauty_files = {}
                    for kind,field in (('PNG','final'),('JPEG','jpeg')):
                        recorded = Path(saved['files'][field])
                        beauty_files[kind] = str(path.parent/(recorded.name if recorded.is_absolute() else recorded))
                    parent = saved.get('parent_version') or {}
                    beauty_common = {**common,'description':parent.get('description',common['description'])+
                        f'\n人脸美化：美肤 {parameters["skin"]:g} / 瘦脸 {parameters["slim"]:g}'}
                    for key in ('retouches','tone_parameters'):
                        if key in parent:
                            beauty_common[key] = deepcopy(parent[key])
                    photo['versions'].append({'id':str(path),'kind':'beauty','label':label,
                        'short_label':f'制作 {number}{revision_label} · {treatment} · 美化 {edit_number}','parameters':parameters,
                        'files':beauty_files,'parent_version':parent,**beauty_common})
                from local_retouch import TASKS
                for edit_number,path in enumerate(sorted(Path(work).glob('retouch-*/edit-*/retouch.json')),1):
                    if any(version['id']==str(path) for version in photo['versions']):
                        continue
                    saved = json.loads(path.read_text(encoding='utf-8'))
                    if saved['status']!='complete':
                        continue
                    treatment_name = saved.get('task_label') or TASKS.get(saved['task'],(saved['task'],))[0]
                    files = {kind:str(path.parent/Path(saved['files'][field]).name)
                        for kind,field in (('PNG','final'),('JPEG','jpeg'))}
                    parent = saved.get('parent_version') or {}
                    retouches = parent.get('retouches',[])+[{'task':saved['task'],
                        'task_label':treatment_name,'brief':saved.get('brief','')}]
                    local_common = {**common,'description':parent.get('description',common['description'])+'\n局部修饰：'+treatment_name+
                        ('\n具体要求：'+saved['brief'] if saved.get('brief') else '')}
                    for key in ('parameters','tone_parameters'):
                        if key in parent:
                            local_common[key] = deepcopy(parent[key])
                    inherited = ' · 美化' if 'parameters' in local_common else ''
                    previous = ' · '+ '、'.join(item.get('task_label') or TASKS.get(item['task'],(item['task'],))[0]
                        for item in parent.get('retouches',[])) if parent.get('retouches') else ''
                    local_label = (f'制作 {number}{revision_label} · {treatment}{inherited}{previous} · '
                        f'局部修饰 {edit_number} · {treatment_name}')
                    photo['versions'].append({'id':str(path),'kind':'retouch',
                        'label':local_label,'short_label':local_label,
                        'task':saved['task'],'retouches':retouches,'parent_version':parent,'files':files,**local_common})
                for edit_number,path in enumerate(sorted(Path(work).glob('tone-*/tone.json')),1):
                    if any(version['id']==str(path) for version in photo['versions']):
                        continue
                    saved = json.loads(path.read_text(encoding='utf-8'))
                    if saved['status']!='complete':
                        continue
                    parent = saved['parent_version']
                    parameters = saved['parameters']
                    label = f'制作 {number}{revision_label} · {treatment} · 光调 {edit_number}（曝光 {parameters["exposure_ev"]:+g} EV）'
                    tone_common = {**common,'description':parent.get('description',description)+
                        f'\n整图光调：曝光 {parameters["exposure_ev"]:+g} EV；曲线输出 '
                        f'{parameters["shadows"]:g} / {parameters["midtones"]:g} / {parameters["highlights"]:g}'}
                    if any(name in parameters for name in ('warmth','tint','saturation')):
                        tone_common['description'] += (f'；暖冷 {parameters.get("warmth",0):+g}；'
                            f'绿洋红 {parameters.get("tint",0):+g}；饱和度 {parameters.get("saturation",1):g}')
                    for key in ('parameters','retouches'):
                        if key in parent:
                            tone_common[key] = deepcopy(parent[key])
                    photo['versions'].append({'id':str(path),'kind':'tone','label':label,'short_label':label,
                        'tone_parameters':parameters,'parent_version':parent,
                        'files':{kind:str(path.parent/Path(saved['files'][field]).name)
                            for kind,field in (('PNG','final'),('JPEG','jpeg'))},**tone_common})
        except (OSError,ValueError,KeyError,TypeError) as error:
            raise ValueError(f'读取项目成片失败：{directory}\n{error}') from error
    return list(photos.values())


def retouch_parent(version):
    return {key:deepcopy(version[key]) for key in ('id','kind','label','description','parameters','retouches','tone_parameters')
        if key in version}


def saved_editor_context(photo, version):
    return {'name':photo['name'],'base':version['files']['PNG'],
        'directory':version['work'],'batch_directory':version['batch'],
        'parent_version':retouch_parent(version)}


def beauty_editor_context(photo, version):
    context = saved_editor_context(photo,version)
    context['edit_id'] = version['id']
    if version['kind']=='beauty':
        record = Path(version['id'])
        saved = json.loads(record.read_text(encoding='utf-8'))
        context.update(base=saved['source'],session=str(record.parent.parent),
            beauty_parameters=saved['parameters'],parent_version=saved.get('parent_version'))
    return context


def revision_context(photo, version):
    """Read the explicitly selected base version's own production direction."""
    if version['kind']!='base':
        raise ValueError('请选择基础成片作为改稿依据；美化在新稿完成后另做')
    work = Path(version['work'])
    run = json.loads((work/'run.json').read_text(encoding='utf-8'))
    direction = json.loads((Path(version['batch'])/'art-direction.json').read_text(encoding='utf-8'))
    direction['edits'] = deepcopy(version['edits'])
    return {'source':photo['source'],'character':run['character'],'brief':run['brief'],
        'art_direction':direction,'reference':version['files']['JPEG'],
        'revision_of':version['id'],'size':tuple(version['dimensions'])}


def selected_version(photo, choice):
    version_id = choice['version']
    version = next((version for version in photo['versions'] if version['id']==version_id),None)
    if version is None:
        raise ValueError('此前选择的版本不在当前记录中，请重新选择：'+photo['name'])
    return version


def photo_choices(photos, saved):
    """New originals start with a base preview and stay outside the delivery selection."""
    choices = {key:dict(value) for key,value in saved.items()}
    for photo in photos:
        if photo['key'] not in choices:
            base = next(version for version in reversed(photo['versions']) if version['kind']=='base')
            choices[photo['key']] = {'source':photo['source'],'version':base['id'],'included':False}
    return choices


def saved_tone_batch(path):
    """Read actual saved outputs from one recorded round, including its failed photos."""
    path = Path(path)
    saved = json.loads(path.read_text(encoding='utf-8'))
    targets = []
    for item in saved['items']:
        target = {'photo_key':item['photo_key'],'name':item['name']}
        if item['status']=='complete':
            target['version_id'] = item['saved_record']
        else:
            target['error'] = item.get('error','该轮照片尚未完成')
        targets.append(target)
    return {'id':str(path),'created_at':saved['created_at'],'status':saved['status'],
        'parameters':saved['parameters'],'targets':targets,
        'photos':[{'name':item['name'],'status':item['status']} for item in saved['items']]}


def adopt_delivery_versions(photos, choices, targets):
    """Explicitly adopt known saved versions, retaining every delivery checkbox."""
    records = {photo['key']:photo for photo in photos}
    items = []
    for target in targets:
        item = dict(target)
        try:
            if 'error' in target:
                raise ValueError(target['error'])
            photo = records[target['photo_key']]
            version = selected_version(photo,{'version':target['version_id']})
            for path in version['files'].values():
                if not Path(path).is_file():
                    raise FileNotFoundError('所选版本文件不存在：'+path)
            choice = choices[photo['key']]
            item.update(status='complete',previous_version=choice['version'],label=version['label'])
            choice['version'] = version['id']
        except (OSError,ValueError,KeyError) as error:
            item.update(status='failed',error=str(error))
        items.append(item)
    complete = sum(item['status']=='complete' for item in items)
    return {'status':'complete' if complete==len(items) else 'partial' if complete else 'failed','items':items}


def export_delivery(photos, choices, parent, name, formats, *, progress=None, adoption=None):
    """Copy the explicitly selected saved versions; preserve every image byte."""
    report = progress or (lambda message:None)
    selected = [dict(choice,key=key) for key,choice in choices.items() if choice['included']]
    if not selected:
        raise ValueError('请先勾选要交付的作品')
    if not formats or any(kind not in ('JPEG','PNG') for kind in formats):
        raise ValueError('请选择 JPEG 或 16 位 PNG')
    title = re.sub(r'[<>:"/\\|?*\x00-\x1f]','_',name).strip(' .')
    directory = Path(parent)/(title+'_交付_'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
    directory.mkdir(parents=True,exist_ok=False)
    for kind in formats:
        (directory/kind).mkdir()
    records = {photo['key']:photo for photo in photos}
    names = {Path(choice['source']).stem.casefold() for choice in selected}
    used_names = set()
    result = {'directory':str(directory),'formats':list(formats),'items':[],
        'status':'running','manifest':str(directory/'delivery.json'),
        'created_at':datetime.now().astimezone().isoformat(timespec='seconds'),
        'selection':selected}
    if adoption is not None:
        result['adoption'] = deepcopy(adoption)
    save_json(directory/'delivery.json',result)
    for number,choice in enumerate(selected,1):
        stem = Path(choice['source']).stem
        candidate = stem
        suffix = 1
        while candidate.casefold() in used_names or (candidate!=stem and candidate.casefold() in names):
            suffix += 1
            candidate = f'{stem}_{suffix}'
        used_names.add(candidate.casefold())
        report(f'正在导出 {number}/{len(selected)}：'+Path(choice['source']).name)
        item = {'source':choice['source'],'version':choice['version'],'files':{},'status':'complete'}
        try:
            photo = records.get(choice['key'])
            if photo is None:
                raise ValueError('交付选择引用的原片不在当前项目成片记录中：'+choice.get('source',choice['key'])+
                    '；请检查对应批次记录是否已被移动或删除')
            version = selected_version(photo,choice)
            item['label'] = version['label']
            item['saved_files'] = dict(version['files'])
            item['dimensions'] = version['dimensions']
            for kind in formats:
                target = directory/kind/(candidate+('.jpg' if kind=='JPEG' else '.png'))
                temporary = target.with_name(target.name+'.tmp')
                try:
                    shutil.copy2(version['files'][kind],temporary)
                    os.replace(temporary,target)
                finally:
                    temporary.unlink(missing_ok=True)
                item['files'][kind] = str(target)
        except (OSError,ValueError,KeyError) as error:
            item.update(status='failed',error=str(error))
        result['items'].append(item)
        save_json(directory/'delivery.json',result)
    complete = sum(item['status']=='complete' for item in result['items'])
    result['status'] = 'complete' if complete==len(selected) else 'partial' if complete else 'failed'
    if result['status']=='complete' and adoption is not None and adoption['status']!='complete':
        result['status'] = 'partial'
    save_json(directory/'delivery.json',result)
    return result
