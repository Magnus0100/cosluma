"""Owned content locators for immutable historical session observations.

Resolved paths are IO locations, never replacement historical observations.
Contexts, responses, receipts and their identities remain byte/content bound.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4

from asset_identity import sha256_file
from photo_editing import identity

SCHEMA='cosluma_photo_session_evidence.v1'


def _sha(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def _owned(root,name):
    value=Path(name)
    if value.is_absolute() or not value.parts or '..' in value.parts:
        raise ValueError('Session evidence must use owned relative locators')
    result=(root/value).resolve()
    if not result.is_relative_to(root.resolve()):raise ValueError('Session evidence escaped its owned directory')
    return result


def _index(state):
    locator=state.get('evidence')
    if locator is None:return None,None
    path=Path(locator['path']).resolve(strict=True)
    if sha256_file(path)!=locator.get('sha256'):raise ValueError('Owned session evidence index changed')
    document=json.loads(path.read_text(encoding='utf8'))
    body={k:v for k,v in document.items() if k!='evidence_id'}
    binding=state['binding']
    if (document.get('schema')!=SCHEMA or document.get('evidence_id')!=identity(body)
            or document.get('producer_sha256')!=sha256_file(__file__)
            or document.get('evidence_id')!=locator.get('evidence_id')
            or document.get('session_id')!=state['session_id']
            or document.get('source_sha256')!=binding['source']['sha256']
            or document.get('scope_id')!=binding['scope_id'] or document.get('design_id')!=binding['design_id']):
        raise ValueError('Owned evidence belongs to another immutable session')
    return path.parent,document


def resolve_file(state,record):
    """Return verified actual IO location without modifying a historical record."""
    if not isinstance(record,dict) or not record.get('path') or not _sha(record.get('sha256')):
        raise ValueError('Evidence file needs its actual historical SHA and locator')
    root,index=_index(state)
    match=(index or {}).get('files',{}).get(record['sha256'])
    path=_owned(root,match['file']) if match else Path(record['path']).resolve(strict=True)
    if not path.is_file() or sha256_file(path)!=record['sha256']:
        raise ValueError('Session evidence bytes changed')
    return {'path':str(path),'sha256':record['sha256']}


def resolve_asset(state,descriptor):
    """Relocate a typed group using its unchanged receipt and asset identities."""
    if not isinstance(descriptor,dict):raise ValueError('Typed appearance locator required')
    root,index=_index(state)
    key=identity({k:descriptor.get(k) for k in ('asset_id','receipt_sha256')})
    match=(index or {}).get('assets',{}).get(key)
    if match is None:return copy.deepcopy(descriptor)
    result={**descriptor,'path':str(_owned(root,match['directory']))}
    if sha256_file(Path(result['path'])/'receipt.json')!=descriptor['receipt_sha256']:
        raise ValueError('Owned typed appearance receipt changed')
    return result


def read_json(state,record):
    return json.loads(Path(resolve_file(state,record)['path']).read_text(encoding='utf8'))


def adjacent_file(state,record,name):
    """Locate a frozen adjacent mapping by its original path, never by scanning."""
    original=str((Path(record['path']).parent/name).resolve())
    _root,index=_index(state)
    matches=[(sha,row) for sha,row in (index or {}).get('files',{}).items()
             if original in row.get('original_paths',[])]
    if len(matches)>1:raise ValueError('Historical adjacent evidence has ambiguous identities')
    if matches:return resolve_file(state,{'path':original,'sha256':matches[0][0]})
    return {'path':original,'sha256':sha256_file(original)}


def freeze_session_evidence(source_directory,destination,*,state,target_state=None,include_source=True):
    """Atomically own the declared file/typed-asset closure of one saved session.

    No directory scan, mutable ledger copy, model call or historical metadata
    rewrite occurs. All copied JSON remains byte-for-byte identical.
    """
    from appearance_assets import materialize_asset,validate_asset,declared_files
    source_directory=Path(source_directory).resolve();destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError('Session evidence is immutable')
    source_session=source_directory/'session.json'
    original_bytes=source_session.read_bytes()
    if json.loads(original_bytes)!=state:raise ValueError('Session changed before its evidence was frozen')
    destination.parent.mkdir(parents=True,exist_ok=True)
    staging=destination.with_name('.'+destination.name+'.partial-'+uuid4().hex);staging.mkdir()
    if type(include_source) is not bool:raise ValueError('Source inclusion must be explicit')
    files={};assets={};visited=set()
    visual_shas={row['image']['sha256'] for row in state.get('candidates',[])}
    for attempt in state.get('attempts',[]):
        visual_shas.update(row['sha256'] for row in attempt.get('context',{}).get('images',[]))
    source_sha=state['binding']['source']['sha256']
    target=target_state or state
    if any(target['binding'][key]!=state['binding'][key] for key in ('scope_id','design_id')) or target['binding']['source']['sha256']!=state['binding']['source']['sha256']:
        raise ValueError('Owned session evidence cannot transfer across source/design/scope')
    binding={'source_sha256':state['binding']['source']['sha256'],'scope':state['scope'],
             'design_id':state['binding']['design_id']}
    def add_file(record,*,opaque=False):
        sha=record['sha256'];original=str(Path(record['path']).resolve())
        if not include_source and sha==source_sha and sha not in visual_shas:return
        if sha in files:
            if original not in files[sha]['original_paths']:files[sha]['original_paths'].append(original)
            return
        actual=Path(resolve_file(state,record)['path'])
        suffix=Path(record['path']).suffix.lower()
        if len(suffix)>16 or any(c not in '.abcdefghijklmnopqrstuvwxyz0123456789_-' for c in suffix):suffix=''
        relative='files/'+sha+suffix;target=staging/relative;target.parent.mkdir(exist_ok=True)
        shutil.copyfile(actual,target)
        if sha256_file(target)!=sha:raise ValueError('Evidence changed during freezing')
        files[sha]={'file':relative,'sha256':sha,'original_paths':[original]}
        if actual.suffix.lower()=='.json' and not opaque:
            content=json.loads(actual.read_text(encoding='utf8'))
            # Registered provider evidence is opaque. It may describe removed
            # pre-registration quality probes, connection configuration or
            # model weights; none is a replay input. The typed group already
            # owns the real registered image and this exact receipt.
            provider_receipt=(isinstance(content,dict) and content.get('status')=='reference_candidate'
                and isinstance(content.get('asset'),dict) and isinstance(content.get('provider_contract'),dict))
            if not provider_receipt:walk(content)
        if Path(record['path']).name=='review-board.png':
            adjacent=adjacent_file(state,record,'review-board.json');add_file(adjacent)
    def walk(value):
        if isinstance(value,dict):
            if {'path','asset_id','receipt_sha256'}<=set(value):
                key=identity({k:value[k] for k in ('asset_id','receipt_sha256')})
                if key not in assets:
                    actual=resolve_asset(state,value);receipt=validate_asset(actual,**binding)
                    relative='assets/'+value['asset_id'];target=staging/relative
                    materialize_asset(actual,target,**binding)
                    assets[key]={'directory':relative,'asset_id':value['asset_id'],
                                 'receipt_sha256':value['receipt_sha256']}
                    # Follow the current type's explicit owned closure, including
                    # geometric support/map and its unchanged native U group.
                    # Historical paths inside receipts are never a file scan.
                    for name,digest in declared_files(actual,**binding).items():
                        path=Path(actual['path'])/name
                        if path.suffix.lower()=='.json':add_file({'path':str(path),'sha256':digest},opaque=True)
            if value.get('path') and _sha(value.get('sha256')):
                token=(str(value['path']),value['sha256'])
                if token not in visited:visited.add(token);add_file(value)
            for key,item in value.items():
                if key!='evidence':walk(item)
        elif isinstance(value,list):
            for item in value:walk(item)
    try:
        walk(state)
        document={'schema':SCHEMA,'producer_sha256':sha256_file(__file__),
            'session_id':target['session_id'],'source_session_id':state['session_id'],
            'source_sha256':binding['source_sha256'],'scope_id':state['binding']['scope_id'],
            'design_id':binding['design_id'],'source_session_sha256':sha256_file(source_session),
            'history':[{'attempt_id':row['attempt_id'],'context_sha256':row['context_sha256'],
                        'response_sha256':identity(row.get('response'))} for row in state['attempts']],
            'files':files,'assets':assets}
        document['evidence_id']=identity(document)
        (staging/'index.json').write_text(json.dumps(document,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        if source_session.read_bytes()!=original_bytes:raise ValueError('Session changed while its evidence was frozen')
        os.rename(staging,destination)
    except BaseException:
        shutil.rmtree(staging,ignore_errors=True);raise
    path=destination/'index.json'
    return {'path':str(path),'sha256':sha256_file(path),'evidence_id':document['evidence_id']}
