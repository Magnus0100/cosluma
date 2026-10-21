#!/usr/bin/env python3
"""Portable frozen editable revisions, without model runtimes or credentials."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from datetime import datetime, timezone
from asset_identity import sha256_file

FORMAT = 'cosplay.portable-project.v3'
ASSET = 'cosplay-asset://'
CONTEXT = 'cosplay-workspace://'
APPEARANCE = 'cosplay-appearance://'
SESSION = 'cosplay-session://'
_DIAGNOSTIC_KEYS = {'design_provenance', 'provenance', 'identity', 'scene_identity',
                    'supersedes', 'manifest', 'event_log', 'active_delivery', 'delivery'}
_RUNTIME_SUFFIXES = {'.pt', '.pth', '.onnx', '.safetensors', '.exe', '.dll', '.py', '.task'}


def is_diagnostic_locator(value):
    """Frozen code/model identities are metadata, not editable pixel assets."""
    return Path(str(value)).suffix.lower() in _RUNTIME_SUFFIXES


def _json(value):
    return (json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode('utf-8')


def _hash(path):
    # All content identities share the stat-bound cache. Extraction still
    # performs a full SHA-256 on each newly written asset; later validation of
    # the same immutable bytes reuses that verified identity instead of reading
    # every large native plane a second time during import.
    return sha256_file(path)


def _appearance_files(receipt, receipt_sha):
    """The typed receipt lists owned files; historical provider paths do not."""
    from appearance_assets import declared_files_from_receipt
    return declared_files_from_receipt(receipt,receipt_sha)


def _unpack_appearance(blob, directory, asset_id, receipt_sha):
    """Extract an already hash-verified stored group without rewriting receipts."""
    if not re.fullmatch('[a-f0-9]{64}',str(asset_id)):
        raise ValueError('Invalid portable appearance identity')
    directory.mkdir(parents=True,exist_ok=False)
    hashes={}
    with zipfile.ZipFile(blob) as packed:
        members=packed.infolist()
        if (len({item.filename for item in members})!=len(members)
                or any(item.compress_type!=zipfile.ZIP_STORED or item.file_size!=item.compress_size for item in members)
                or sum(item.file_size for item in members)>blob.stat().st_size):
            raise ValueError('Appearance group must contain bounded stored files')
        receipt_member=next((item for item in members if item.filename=='receipt.json'),None)
        if receipt_member is None or receipt_member.file_size>1024*1024:
            raise ValueError('Appearance group requires a bounded receipt')
        data=packed.read('receipt.json')
        if hashlib.sha256(data).hexdigest()!=receipt_sha:
            raise ValueError('Portable appearance receipt changed')
        receipt=json.loads(data)
        if not isinstance(receipt,dict) or receipt.get('asset_id')!=asset_id:
            raise ValueError('Portable appearance receipt belongs to another asset')
        files=_appearance_files(receipt,receipt_sha)
        if set(files)!={item.filename for item in members}:
            raise ValueError('Portable appearance contains missing or unowned files')
        for name,digest in files.items():
            if (not isinstance(name,str) or not name or '\\' in name or any(c in name for c in ':<>"|?*') or Path(name).is_absolute()
                    or any(part in {'','.','..'} or part.endswith((' ','.')) for part in name.split('/'))):
                raise ValueError('Invalid owned appearance member')
            if Path(name).suffix.lower() not in {'.npy','.npz','.png','.json'}:
                raise ValueError('Unexpected file in immutable appearance group')
            target=directory/name
            if not target.resolve().is_relative_to(directory.resolve()):raise ValueError('Appearance member escaped its group')
            target.parent.mkdir(parents=True,exist_ok=True)
            with packed.open(name) as reader,target.open('xb') as writer:
                shutil.copyfileobj(reader,writer)
            if _hash(target)!=digest:raise ValueError('Portable appearance file changed: '+name)
            hashes[str(target.resolve())]=digest
    return hashes


def validate_portrait_beauty(run, *, verify_assets=True):
    """Check frozen source identity without a model or absolute-path binding."""
    control = ((run.get('options') or {}).get('appearance_revision') or {}).get('portrait_beauty')
    if not control:
        return None
    if not isinstance(control, dict) or set(control) - {'result', 'strength', 'disabled_faces'}:
        raise ValueError('Beauty revision requires result, strength and disabled_faces')
    strength = control.get('strength', 1.)
    disabled = control.get('disabled_faces', [])
    if (isinstance(strength, bool) or not isinstance(strength, (int, float))
            or not math.isfinite(strength) or not 0 <= strength <= 1
            or not isinstance(disabled, list) or any(not isinstance(value, str) for value in disabled)
            or not isinstance(control.get('result'), str) or not control['result']):
        raise ValueError('Invalid frozen beauty controls')
    if not verify_assets:
        return None  # Archive locators are resolved only after extraction.
    if not Path(control['result']).is_absolute():
        raise ValueError('Frozen beauty results require an absolute JSON path')
    from portrait_beauty_layer import load_result
    from display_transform import source_transform_for_run
    from native_source_baseline import retouch_identity
    result, path = load_result(control['result'])
    plan = ((run.get('options') or {}).get('appearance_revision') or {}).get('design_reconstruction_plan') or {}
    baseline = plan.get('source_baseline') or (run.get('options') or {}).get('source_only_plan') or {}
    native_size = baseline.get('native_size') or (plan.get('canvas') or {}).get('source_oriented_size')
    retouch = retouch_identity(run)
    stable = {key: retouch[key] for key in ('delta_sha256', 'source_sha256', 'working_domain')} if retouch else None
    if (result['source_sha256'] != (run.get('source') or {}).get('sha256')
            or result['source_display'] != source_transform_for_run(run)
            or result['native_size'] != native_size or result.get('native_source_retouch') != stable
            or set(disabled) - {face['id'] for face in result['faces']}):
        raise ValueError('Frozen beauty layer differs from its source, display, retouch or faces')
    for face in result['faces']:
        for key in ('pixels', 'model_output'):
            asset = face.get(key)
            if not asset:
                continue
            target = Path(asset['path'])
            target = target if target.is_absolute() else path.parent / target
            if _hash(target) != asset['sha256']:
                raise ValueError('Frozen beauty asset changed: '+str(target))
    return result, path


def _validate_current_project(run, *, verify_beauty_assets=True):
    from design_reconstruction import render_mode_for_run,plan_for_run,validate_current_native_plan
    from display_transform import plan_for_run as display_plan_for_run
    from raw_support import RGB_DECODE_VERSION
    typed=(run.get('options') or {}).get('photo_appearance')
    mode=(run.get('task_intent') or (run.get('options') or {}).get('task_intent') or {}).get('mode')
    if mode=='create_reference' and typed is None:
        raise ValueError('Unsupported creation format; a current typed appearance is required')
    if typed is not None:
        from photo_editing import validate_scope
        scope=validate_scope((run.get('task_plan') or {}).get('editing_scope'))
        baseline=run.get('initial_source_appearance')
        shaping=(run.get('options') or {}).get('shaping_state')
        shaping_base=(shaping or {}).get('base_appearance')
        for asset in ((typed,baseline,shaping_base) if shaping is not None else (typed,baseline)):
            if (not isinstance(asset,dict) or set(asset)!={'path','asset_id','receipt_sha256'}
                    or not isinstance(asset['path'],str)
                    or any(not re.fullmatch('[a-f0-9]{64}',str(asset[key])) for key in ('asset_id','receipt_sha256'))):
                raise ValueError('Portable native appearance needs complete content-bound assets')
        if verify_beauty_assets:
            from generated_appearance_render import validate_run
            from appearance_assets import validate_asset
            scope,design,_,_=validate_run(Path(run.get('root') or '.'),run)
            binding={'source_sha256':run['source']['sha256'],'scope':scope,
                'design_id':design['observed_design']['design_id']}
            selected=validate_asset(typed,**binding);original=validate_asset(baseline,**binding)
            if (original['kind']!='source_baseline' or selected['source_origin']!=original['source_origin']
                    or selected['canvas_size']!=original['canvas_size']):
                raise ValueError('Portable final does not share its actual source baseline')
            if shaping is not None:
                from photo_shaping import validate_state
                validate_state(shaping,current=typed,**binding)
        return False
    render_mode_for_run(run)
    if not (run.get('source') or {}).get('sha256'):
        raise ValueError('Current editable projects require a bound original source')
    plan=plan_for_run(run)
    if plan:
        validate_current_native_plan(plan)
        if (plan.get('source_sha256')!=run['source']['sha256']
                or plan.get('candidate_sha256')!=(run.get('reference') or {}).get('sha256')):
            raise ValueError('Current project source/reference bindings differ')
    else:
        local=(run.get('options') or {}).get('source_only_plan') or {}
        if (run.get('reference') or local.get('decode_version')!=RGB_DECODE_VERSION
                or local.get('reference_assets_used') is not False
                or len(local.get('native_size') or [])!=2):
            raise ValueError('Current source-only project plan is missing or contaminated by a reference')
    display_plan_for_run(run)
    validate_portrait_beauty(run, verify_assets=verify_beauty_assets)
    return bool(plan)


def export_project(root, run_id, archive, *, include_raw=True, include_design=True):
    from single_source_workflow import load_run
    from native_source_baseline import enabled
    from design_reconstruction import plan_for_run
    root, archive = Path(root).resolve(), Path(archive).resolve()
    manifest, run = load_run(root, run_id)
    _validate_current_project(run)
    run = copy.deepcopy(run)
    for key in ('export_receipts', 'reopen_evidence', 'portable_import'):
        run.pop(key, None)
    if archive.exists():
        raise FileExistsError(f'Project package already exists: {archive}')
    typed=bool((run.get('options') or {}).get('photo_appearance'))
    if not include_raw and not typed and not enabled(plan_for_run(run)):
        raise ValueError('Omitting the source requires a saved paired native baseline')
    source = Path(run['source']['path'])
    source = (source if source.is_absolute() else root/source).resolve()
    reference = run.get('reference') or {}
    design = Path(reference['path']).resolve() if reference.get('path') else None
    if design and not Path(reference['path']).is_absolute():
        design = (root/reference['path']).resolve()
    omitted = {source} if not include_raw else set()
    if not include_design and design:
        omitted.add(design)
    assets, path_names, json_values = {}, {}, {}
    photo_groups,photo_directories={},{}
    session_groups,session_paths,opaque_names={},{},set()
    staging_root=root/'outputs/cache';staging_root.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='portable-appearance-',dir=staging_root) as group_staging:

        def register_photo_groups(value, diagnostic=False):
            if isinstance(value,dict):
                if value.get('schema')=='cosluma_shaping_authorization.v1':return
                if not diagnostic and set(value)=={'path','asset_id','receipt_sha256'}:
                    from appearance_assets import validate_asset,declared_files
                    scope=run['task_plan']['editing_scope']
                    design=(run.get('generation_design') or (run.get('options') or {}).get('generation_design'))['observed_design']['design_id']
                    receipt=validate_asset(value,source_sha256=run['source']['sha256'],scope=scope,design_id=design)
                    directory=Path(value['path']).resolve()
                    if directory in photo_directories:return
                    prior=photo_groups.get(value['asset_id'])
                    if prior and prior['receipt_sha256']!=value['receipt_sha256']:
                        raise ValueError('Conflicting byte identities for one appearance asset')
                    owned=declared_files(value,source_sha256=run['source']['sha256'],scope=scope,design_id=design)
                    blob=Path(group_staging)/(value['asset_id']+'.zip')
                    if not blob.exists():
                        with zipfile.ZipFile(blob,'x',compression=zipfile.ZIP_STORED,allowZip64=True) as packed:
                            for name,digest in sorted(owned.items()):
                                member=directory/name
                                if _hash(member)!=digest:raise ValueError('Appearance changed while packaging: '+name)
                                packed.write(member,name)
                        validate_asset(value,source_sha256=run['source']['sha256'],scope=scope,design_id=design)
                    digest=_hash(blob);name='assets/'+digest+'.zip'
                    assets.setdefault(name,{'path':blob,'original_sha256':digest})
                    photo_groups[value['asset_id']]={'archive_asset':name,'receipt_sha256':value['receipt_sha256']}
                    photo_directories[directory]={'asset_id':value['asset_id'],'files':owned}
                    return
                for key,item in value.items():register_photo_groups(item,diagnostic or key in _DIAGNOSTIC_KEYS)
            elif isinstance(value,list):
                for item in value:register_photo_groups(item,diagnostic)

        register_photo_groups(run)
        if (run.get('options') or {}).get('shaping_state') is not None:
            register_photo_groups(run['options']['shaping_state']['base_appearance'])

        def opaque_file(path):
            path=Path(path).resolve();digest=_hash(path)
            name='assets/'+digest+path.suffix.lower()
            assets.setdefault(name,{'path':path,'original_sha256':digest})
            path_names[path]=name
            opaque_names.add(name)
            return name

        def register_sessions(value):
            if isinstance(value,dict):
                if value.get('schema')=='cosluma_shaping_authorization.v1':return
                summary=value.get('photo_edit_session')
                if isinstance(summary,dict) and summary.get('path') and summary.get('session_id'):
                    from photo_edit_session import snapshot
                    from photo_session_evidence import freeze_session_evidence
                    path=Path(summary['path']);path=(path if path.is_absolute() else root/path).resolve()
                    if path not in session_paths:
                        state=snapshot(path.parent)
                        sid=state['session_id']
                        if sid!=summary['session_id']:raise ValueError('Portable session summary identity differs')
                        directory=Path(group_staging)/('session-'+sid);directory.mkdir()
                        locator=freeze_session_evidence(path.parent,directory/'evidence',state=state,include_source=include_raw)
                        state=copy.deepcopy(state);state['evidence']=locator
                        target=directory/'session.json';target.write_bytes(_json(state))
                        index=json.loads(Path(locator['path']).read_text('utf8'))
                        files={'evidence/index.json':opaque_file(locator['path'])}
                        for record in index['files'].values():
                            relative='evidence/'+record['file']
                            files[relative]=opaque_file(directory/relative)
                        appearance_assets={}
                        for record in index['assets'].values():
                            relative='evidence/'+record['directory']
                            descriptor={key:record[key] for key in ('asset_id','receipt_sha256')}
                            descriptor['path']=str(directory/relative)
                            register_photo_groups(descriptor)
                            appearance_assets[relative]=record['asset_id']
                        session_groups[sid]={'session_asset':opaque_file(target),'files':files,
                            'appearance_assets':appearance_assets,'evidence':{**locator,'path':'evidence/index.json'}}
                        session_paths[path]=sid
                for key,item in value.items():
                    if key not in _DIAGNOSTIC_KEYS and key!='independent_evaluation':register_sessions(item)
            elif isinstance(value,list):
                for item in value:register_sessions(item)

        register_sessions(run)

        def resolve(value, directory):
            if not isinstance(value, str) or '\n' in value or len(value)>1000 or '://' in value:
                return None
            path = Path(value)
            if not path.is_absolute() and '/' not in value and '\\' not in value and path.suffix.lower() not in {'.json','.jsonl','.npz','.npy','.png','.jpg','.jpeg','.tif','.tiff','.arw','.dng','.nef','.cr2','.cr3','.raf','.rw2','.txt','.pt','.pth','.onnx','.safetensors','.exe','.dll'}:
                return None
            if path.is_absolute():
                return path.resolve()
            local, workspace = directory/path, root/path
            if local.exists():
                return local.resolve()
            if workspace.exists():
                return workspace.resolve()
            if value.replace('\\','/').startswith(('outputs/','data/','.cache/','scripts/','docs/')):
                return workspace.resolve()
            if path.suffix.lower() in {'.json','.jsonl','.npz','.npy','.png','.jpg','.jpeg','.tif','.tiff','.arw','.dng','.nef','.cr2','.cr3','.raf','.rw2','.txt','.pt','.pth','.onnx','.safetensors','.exe','.dll'}:
                return local.resolve()
            return None

        def collect(value, directory, diagnostic=False):
            if isinstance(value, dict):
                if value.get('schema')=='cosluma_shaping_authorization.v1':return
                # Disabled layers remain part of the editable scene.
                for key, item in value.items():
                    collect(item, directory, diagnostic or key in _DIAGNOSTIC_KEYS)
            elif isinstance(value, list):
                for item in value:
                    collect(item, directory, diagnostic)
            elif isinstance(value, str) and not diagnostic:
                path = resolve(value, directory)
                if path and path.is_file() and path not in omitted and not is_diagnostic_locator(path):
                    add(path)

        def add(path):
            path = path.resolve()
            if path in session_paths:return
            if any(directory in path.parents for directory in photo_directories):return
            if path in path_names or path == manifest.resolve() or path in omitted or is_diagnostic_locator(path):
                return
            value=None
            if path.suffix.lower()=='.json':
                value=json.loads(path.read_text('utf-8'))
                if isinstance(value,dict) and value.get('schema') in {'cosluma_photo_appearance.v1','cosluma_geometric_appearance.v1'}:
                    if path.name!='receipt.json':
                        # Session evidence archives receipt bytes by content
                        # hash. Their directory is an opaque evidence bucket,
                        # never the original typed group's IO location.
                        opaque_file(path)
                        return
                    register_photo_groups({'path':str(path.parent),'asset_id':value.get('asset_id'),'receipt_sha256':_hash(path)})
                    return
                register_photo_groups(value)
            digest = _hash(path)
            name = 'assets/'+digest+path.suffix.lower()
            path_names[path] = name
            assets.setdefault(name, {'path': path, 'original_sha256': digest})
            if path.suffix.lower()=='.json':
                json_values[name] = value
                collect(value, path.parent)

        # Include owned outputs, review and log; discover external live dependencies
        # recursively from settings. Never traverse prior version manifests.
        for path in manifest.parent.rglob('receipt.json'):
            if path.is_file():
                value=json.loads(path.read_text('utf8'))
                if isinstance(value,dict) and value.get('schema') in {'cosluma_photo_appearance.v1','cosluma_geometric_appearance.v1'}:
                    register_photo_groups({'path':str(path.parent),'asset_id':value.get('asset_id'),'receipt_sha256':_hash(path)})
        for path in manifest.parent.rglob('*'):
            if path.is_file() and not is_diagnostic_locator(path) and path not in omitted:
                add(path)
        collect(run.get('options') or {}, root)
        collect(run.get('artifacts') or {}, root)
        if include_raw:
            add(source)
        if include_design and design:
            add(design)
        if reference.get('provenance'):
            sidecar=resolve(reference['provenance'], root)
            if sidecar and sidecar.is_file():
                add(sidecar)

        def portable(value, directory):
            if isinstance(value, dict):
                if value.get('schema')=='cosluma_shaping_authorization.v1':return copy.deepcopy(value)
                return {key:copy.deepcopy(item) if key=='independent_evaluation' else portable(item, directory)
                    for key,item in value.items()}
            if isinstance(value, list):
                return [portable(item,directory) for item in value]
            if isinstance(value,str):
                path=resolve(value,directory)
                if path in session_paths:return SESSION+session_paths[path]
                if path in photo_directories:
                    return APPEARANCE+photo_directories[path]['asset_id']
                if path:
                    for directory,group in sorted(photo_directories.items(),key=lambda item:len(item[0].parts),reverse=True):
                        if path.is_relative_to(directory):
                            relative=path.relative_to(directory).as_posix()
                            if relative in group['files']:return APPEARANCE+group['asset_id']+'/'+relative
                if path in path_names:
                    return ASSET+path_names[path]
                if path:
                    try:
                        return CONTEXT+path.relative_to(root).as_posix()
                    except ValueError:
                        if path.is_absolute() and Path(value).is_absolute():
                            return 'cosplay-external://'+path.name
                if value in {str(root),root.as_posix()}:
                    return CONTEXT
            return value

        archive.parent.mkdir(parents=True,exist_ok=True)
        package={'format':FORMAT,'created_at':datetime.now(timezone.utc).isoformat(),
                 'run_id':run_id,'include_raw':include_raw,'include_design':include_design,
                 'scope':'selected editable revision; historical ancestry is metadata, not snapshots',
                 'model_runtimes_included':False,'assets':{},'photo_appearance_groups':photo_groups,
                 'photo_session_groups':session_groups}
        try:
            with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_STORED,allowZip64=True) as bundle:
                for name,entry in assets.items():
                    path=entry['path']
                    if name in json_values and name not in opaque_names:
                        data=_json(portable(json_values[name],path.parent))
                        bundle.writestr(name,data)
                        digest=hashlib.sha256(data).hexdigest(); size=len(data)
                    else:
                        bundle.write(path,name)
                        digest=entry['original_sha256']; size=path.stat().st_size
                    package['assets'][name]={'sha256':digest,'bytes':size,'name':path.name}
                    if name in opaque_names:package['assets'][name]['opaque_session_evidence']=True
                bundle.writestr('run.json',_json(portable(run,root)))
                bundle.writestr('project.json',_json(package))
        except BaseException:
            archive.unlink(missing_ok=True)
            raise
        return {'ok':True,'archive':str(archive),'bytes':archive.stat().st_size,
                'asset_count':len(assets),'run_id':run_id,'format':FORMAT,
                'include_raw':include_raw,'include_design':include_design}


def import_project(root, archive, *, run_id=None):
    root, archive=Path(root).resolve(),Path(archive).resolve()
    with zipfile.ZipFile(archive) as bundle:
        package=json.loads(bundle.read('project.json'))
        if package.get('format')!=FORMAT:
            raise ValueError('Unsupported portable project format')
        assets=package.get('assets');groups=package.get('photo_appearance_groups',{})
        sessions=package.get('photo_session_groups',{})
        if not isinstance(assets,dict) or not isinstance(groups,dict) or not isinstance(sessions,dict):
            raise ValueError('Portable asset and appearance indexes must be objects')
        members=bundle.infolist()
        if (len({item.filename for item in members})!=len(members)
                or {item.filename for item in members}!={'project.json','run.json',*assets}):
            raise ValueError('Portable archive contains duplicate or undeclared members')
        _validate_current_project(json.loads(bundle.read('run.json')), verify_beauty_assets=False)
        identifier=run_id or package['run_id']
        if (not isinstance(identifier,str) or not identifier or identifier in {'.','..'}
                or identifier.endswith(('.', ' ')) or any(ord(char)<32 or char in '<>:"/\\|?*' for char in identifier)):
            raise ValueError('Invalid imported run ID')
        directory=root/'outputs/runs'/identifier
        if directory.resolve().parent != (root/'outputs/runs').resolve():
            raise ValueError('Imported run must stay inside the run store')
        if directory.exists():
            raise FileExistsError(f'Imported version already exists: {directory}')
        directory.mkdir(parents=True)
        owned=directory/'inputs/portable'
        owned.mkdir(parents=True)
        destinations={}
        photo_destinations={}
        session_destinations={}
        verified_hashes={}
        created_project_sha=None
        project_created=False
        try:
            for name,entry in package['assets'].items():
                if not re.fullmatch(r'assets/[0-9a-f]{64}(?:\.[a-z0-9]+)?', name):
                    raise ValueError('Invalid project asset entry')
                if (not isinstance(entry,dict) or not re.fullmatch('[a-f0-9]{64}',str(entry.get('sha256')))
                        or type(entry.get('bytes')) is not int or entry['bytes']<0
                        or bundle.getinfo(name).file_size!=entry['bytes'] or is_diagnostic_locator(name)):
                    raise ValueError('Invalid portable asset identity, size or runtime file')
                target=owned/Path(name).name
                with bundle.open(name) as source,target.open('wb') as destination:
                    shutil.copyfileobj(source,destination,4*1024*1024)
                if _hash(target)!=entry['sha256']:
                    raise ValueError(f'Project asset checksum mismatch: {name}')
                destinations[ASSET+name]=str(target.resolve())
                verified_hashes[str(target.resolve())]=entry['sha256']

            for asset_id,group in (package.get('photo_appearance_groups') or {}).items():
                if not isinstance(group,dict) or set(group)!={'archive_asset','receipt_sha256'}:
                    raise ValueError('Invalid portable appearance group')
                name=group.get('archive_asset')
                blob=destinations.get(ASSET+str(name))
                if not blob or Path(blob).suffix!='.zip':raise ValueError('Portable appearance group has no verified blob')
                target=owned/'appearances'/asset_id
                hashes=_unpack_appearance(Path(blob),target,asset_id,group.get('receipt_sha256'))
                photo_destinations[APPEARANCE+asset_id]=str(target.resolve())
                for path,digest in hashes.items():
                    verified_hashes[path]=digest
                    photo_destinations[APPEARANCE+asset_id+'/'+Path(path).relative_to(target).as_posix()]=path
                # Only the transport wrapper is removed; actual immutable
                # masters and receipts now live in their owned relative layout.
                verified_hashes.pop(str(Path(blob).resolve()),None)
                Path(blob).unlink()

            def owned_member(base,name):
                if (not isinstance(name,str) or not name or '\\' in name
                        or any(char in name for char in ':<>"|?*') or Path(name).is_absolute()
                        or any(not part or part in {'.','..'} or part.endswith((' ','.')) for part in name.split('/'))):
                    raise ValueError('Invalid owned session member')
                target=(base/name).resolve()
                if not target.is_relative_to(base.resolve()):raise ValueError('Session member escaped its group')
                return target

            for sid,group in sessions.items():
                if not re.fullmatch('[a-f0-9]{64}',str(sid)) or not isinstance(group,dict):
                    raise ValueError('Invalid portable session identity')
                if any(not isinstance(group.get(key),dict) for key in ('files','appearance_assets','evidence')):
                    raise ValueError('Portable session requires declared owned evidence')
                session_source=destinations.get(ASSET+str(group.get('session_asset')))
                if (not session_source or package['assets'][group['session_asset']].get('opaque_session_evidence') is not True):
                    raise ValueError('Missing immutable session asset')
                state=json.loads(Path(session_source).read_text('utf8'))
                from photo_editing import identity
                if state.get('session_id')!=sid or identity(state.get('binding'))!=sid:
                    raise ValueError('Portable session binding changed')
                if any(identity(row.get('context'))!=row.get('context_sha256') for row in state.get('attempts',[])):
                    raise ValueError('Portable historical context changed')
                base=owned/'sessions'/sid;base.mkdir(parents=True)
                for name,asset_name in group.get('files',{}).items():
                    target=owned_member(base,name)
                    source_path=destinations.get(ASSET+str(asset_name))
                    if not source_path or package['assets'][asset_name].get('opaque_session_evidence') is not True:
                        raise ValueError('Session member is not immutable evidence')
                    target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source_path,target)
                    verified_hashes[str(target)]=_hash(target)
                for name,asset_id in group.get('appearance_assets',{}).items():
                    source_path=photo_destinations.get(APPEARANCE+str(asset_id))
                    if not source_path:raise ValueError('Missing owned session appearance')
                    target=owned_member(base,name);target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copytree(source_path,target)
                locator=group.get('evidence') or {}
                index=owned_member(base,locator.get('path'))
                state['evidence']={**locator,'path':str(index)}
                from photo_session_evidence import resolve_file
                resolve_file(state,state['candidates'][0]['image'])
                target=base/'session.json';target.write_bytes(_json(state))
                verified_hashes[str(target)]=_hash(target)
                session_destinations[SESSION+sid]=str(target)

            def relocated(value):
                if isinstance(value,dict):
                    if value.get('schema')=='cosluma_shaping_authorization.v1':return copy.deepcopy(value)
                    result={key:copy.deepcopy(item) if key=='independent_evaluation' else relocated(item)
                        for key,item in value.items()}
                    path=result.get('path')
                    if path and Path(str(path)).suffix.lower()=='.json' and Path(str(path)).is_file() and 'sha256' in result:
                        result['sha256']=_hash(Path(path))
                        if 'bytes' in result:
                            result['bytes']=Path(path).stat().st_size
                    return result
                if isinstance(value,list):
                    return [relocated(item) for item in value]
                if isinstance(value,str):
                    if value.startswith(SESSION):
                        if value not in session_destinations:raise ValueError('Unowned portable session locator')
                        return session_destinations[value]
                    if value.startswith(APPEARANCE):
                        if value not in photo_destinations:raise ValueError('Unowned portable appearance member')
                        return photo_destinations[value]
                    if value.startswith(ASSET):
                        if value not in destinations:raise ValueError('Unowned portable asset locator')
                        return destinations[value]
                    if value.startswith(CONTEXT):
                        relative=value.removeprefix(CONTEXT)
                        if ('\\' in relative or any(char in relative for char in ':<>"|?*') or Path(relative).is_absolute()
                                or any(part=='..' or part.endswith((' ','.')) and part!='.' for part in Path(relative).parts)):
                            raise ValueError('Portable workspace locator escapes its workspace')
                        target=(root/relative).resolve()
                        if not target.is_relative_to(root):raise ValueError('Portable workspace locator escapes its workspace')
                        return str(target)
                    if value.startswith('cosplay-external://'):
                        name=value.removeprefix('cosplay-external://')
                        if (not name or name in {'.','..'} or name.endswith((' ','.')) or Path(name).name!=name
                                or any(char in name for char in ':\\/<>"|?*')):
                            raise ValueError('Invalid unbundled external locator')
                        return str(directory/'unbundled'/name)
                return value

            # Rewrite content-bound JSON dependencies before their parents.
            # Approval graphs can be deeper than two levels; a fixed number
            # of passes leaves valid portable evidence with stale checksums.
            original_json={name:json.loads(bundle.read(name.removeprefix(ASSET)))
                for name,target in destinations.items() if Path(target).suffix=='.json'
                and package['assets'][name.removeprefix(ASSET)].get('opaque_session_evidence') is not True}
            written,visiting=set(),set()
            def hash_dependencies(value):
                if isinstance(value,dict):
                    if value.get('schema')=='cosluma_shaping_authorization.v1':return
                    dependency=value.get('path')
                    if 'sha256' in value and isinstance(dependency,str) and dependency in original_json:
                        rewrite_json(dependency)
                    for item in value.values():
                        hash_dependencies(item)
                elif isinstance(value,list):
                    for item in value:
                        hash_dependencies(item)
            def rewrite_json(name):
                if name in written:
                    return
                if name in visiting:
                    raise ValueError('Portable JSON checksum dependencies contain a cycle')
                visiting.add(name)
                original=original_json[name]
                hash_dependencies(original)
                path=Path(destinations[name]);path.write_bytes(_json(relocated(original)))
                verified_hashes[str(path.resolve())]=_hash(path)
                visiting.remove(name);written.add(name)
            for name in original_json:
                rewrite_json(name)
            run=relocated(json.loads(bundle.read('run.json')))
            for key in ('export_receipts', 'reopen_evidence', 'portable_import'):
                run.pop(key, None)
            native=_validate_current_project(run)
            original_run_id=run['run_id']
            original_render_identity={key:run.get('render',{}).get(key) for key in ('run_id','revision')}
            plan=((run.get('options') or {}).get('appearance_revision') or {}).get('design_reconstruction_plan') or {}
            run['run_id']=identifier; run['root']=str(root)
            run['manifest']=(directory/'manifest.json').relative_to(root).as_posix()
            run['event_log']=(directory/'logs/run.log.jsonl').relative_to(root).as_posix()
            run['portable_import']={'format':FORMAT,'original_run_id':package['run_id'],
                                    'include_raw':package['include_raw'],'include_design':package['include_design'],
                                    'original_render_identity':original_render_identity}
            # Rebind only saved facts that actually belong to this imported
            # version. Stale inherited renders and ancestry remain unchanged.
            if identifier!=original_run_id:
                render=run.get('render') or {}
                if render.get('run_id')==original_run_id and render.get('revision')==run.get('revision'):
                    render['run_id']=identifier
                metadata=(run.get('auto_draft') or {}).get('metadata') or {}
                if metadata.get('run_id')==original_run_id and metadata.get('revision')==run.get('revision'):
                    metadata['run_id']=identifier
                continuation=run.get('continuation_edit') or {}
                if continuation.get('rendered_run_id')==original_run_id and continuation.get('rendered_revision')==run.get('revision'):
                    continuation['rendered_run_id']=identifier

            complete_inputs=(package.get('include_raw') is True and
                (not run.get('reference') or package.get('include_design') is True))
            (directory/'manifest.json').write_bytes(_json(run))
            (directory/'portable-project.json').write_bytes(_json(package))
            if complete_inputs:
                from project_cli import create_native_project_if_missing, manifest_path
                from session_draft import binding_for_run
                if not (run.get('source') or {}).get('sha256'):
                    raise ValueError('Editable imports require a source identity')
                if native:
                    if not (run.get('reference') or {}).get('sha256'):
                        raise ValueError('Native editable imports require a design identity')
                    if (plan.get('source_sha256')!=run['source']['sha256']
                            or plan.get('candidate_sha256')!=run['reference']['sha256']):
                        raise ValueError('Native editable import does not match its source/design bindings')
                project_created=create_native_project_if_missing(root,default_sample=str(run['sample']))
                if project_created:
                    created_project_sha=_hash(manifest_path(root))
                if (run.get('options') or {}).get('photo_appearance'):
                    from appearance_assets import validate_asset,declared_files
                    design=run['generation_design']['observed_design']['design_id']
                    scope=run['task_plan']['editing_scope']
                    binding={'assets':[run['source'],run['reference']]}
                    appearances=[run['options']['photo_appearance'],run['initial_source_appearance']]
                    if run['options'].get('shaping_state') is not None:
                        appearances.append(run['options']['shaping_state']['base_appearance'])
                    for asset in appearances:
                        receipt=validate_asset(asset,source_sha256=run['source']['sha256'],scope=scope,design_id=design)
                        binding['assets'].extend({'path':str(Path(asset['path'])/name),'sha256':digest}
                            for name,digest in declared_files(asset,source_sha256=run['source']['sha256'],scope=scope,design_id=design).items())
                else:
                    binding=binding_for_run(root,run)
                # Extraction already verified every byte. Reuse those hashes
                # instead of rereading GB-sized native planes for the binding.
                for asset in binding['assets']:
                    if verified_hashes.get(str(Path(asset['path']).resolve()))!=asset['sha256']:
                        raise ValueError('Imported editable asset is missing or has a different identity: '+asset['path'])
            # Extraction verified every asset; record this workspace and saved
            # revision separately, never as a claim of process-resume testing.
            from artifact_receipts import file_stamp, write_receipt
            saved = {}
            for key in ('auto_draft', 'render_master', 'render_scene_master'):
                value = (run.get('artifacts') or {}).get(key)
                if value and str(Path(value).resolve()) in verified_hashes:
                    target = Path(value).resolve()
                    saved[key] = {'path': str(target), 'sha256': verified_hashes[str(target)], 'stamp': file_stamp(target)}
            if len(saved) == 3:
                write_receipt(root, run, {'schema_version': 'portable_import_receipt.v1',
                    'asset_checksums_verified': True, 'asset_count': len(verified_hashes),
                    'saved_render_artifacts': saved, 'created_at': datetime.now(timezone.utc).isoformat()})
            import workbench_store
            workbench_store.record_recent(root,sample=str(run['sample']),run_id=identifier,source=run['source']['path'])
        except BaseException as error:
            # Only this newly created run is removed; the workspace is retained.
            # A failed semantic NPY check may retain a read-only mmap in its
            # traceback. Release exited validation frames before Windows delete;
            # preserve the exception and traceback locations for the caller.
            import traceback
            traceback.clear_frames(error.__traceback__)
            shutil.rmtree(directory)
            project_path=root/'project.json'
            if created_project_sha and project_path.is_file() and _hash(project_path)==created_project_sha:
                project_path.unlink()
            raise
    return {'ok':True,'run_id':identifier,'manifest':str(directory/'manifest.json'),
            'asset_count':len(destinations),'run':run,'active_project_pointer_changed':False,
            'project_identity_created':project_created}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    commands=parser.add_subparsers(dest='command',required=True)
    export=commands.add_parser('export'); export.add_argument('run_id'); export.add_argument('archive',type=Path)
    export.add_argument('--omit-raw',action='store_true'); export.add_argument('--omit-design',action='store_true')
    load=commands.add_parser('import'); load.add_argument('archive',type=Path); load.add_argument('--run-id')
    args=parser.parse_args(argv)
    result=(export_project(args.root,args.run_id,args.archive,include_raw=not args.omit_raw,include_design=not args.omit_design)
            if args.command=='export' else import_project(args.root,args.archive,run_id=args.run_id))
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
