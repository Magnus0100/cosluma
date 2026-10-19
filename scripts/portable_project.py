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
import zipfile
from pathlib import Path
from datetime import datetime, timezone
from asset_identity import sha256_file

FORMAT = 'cosplay.portable-project.v2'
ASSET = 'cosplay-asset://'
CONTEXT = 'cosplay-workspace://'
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
    if not include_raw and not enabled(plan_for_run(run)):
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
        if path in path_names or path == manifest.resolve() or path in omitted or is_diagnostic_locator(path):
            return
        digest = _hash(path)
        name = 'assets/'+digest+path.suffix.lower()
        path_names[path] = name
        assets.setdefault(name, {'path': path, 'original_sha256': digest})
        if path.suffix.lower()=='.json':
            value = json.loads(path.read_text('utf-8'))
            json_values[name] = value
            collect(value, path.parent)

    # Include owned outputs, review and log; discover external live dependencies
    # recursively from settings. Never traverse prior version manifests.
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
            return {key: portable(item, directory) for key,item in value.items()}
        if isinstance(value, list):
            return [portable(item,directory) for item in value]
        if isinstance(value,str):
            path=resolve(value,directory)
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
             'model_runtimes_included':False,'assets':{}}
    try:
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_STORED,allowZip64=True) as bundle:
            for name,entry in assets.items():
                path=entry['path']
                if name in json_values:
                    data=_json(portable(json_values[name],path.parent))
                    bundle.writestr(name,data)
                    digest=hashlib.sha256(data).hexdigest(); size=len(data)
                else:
                    bundle.write(path,name)
                    digest=entry['original_sha256']; size=path.stat().st_size
                package['assets'][name]={'sha256':digest,'bytes':size,'name':path.name}
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
        _validate_current_project(json.loads(bundle.read('run.json')), verify_beauty_assets=False)
        identifier=run_id or package['run_id']
        if not identifier or '/' in identifier or '\\' in identifier or identifier in {'.','..'}:
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
        verified_hashes={}
        created_project_sha=None
        project_created=False
        try:
            for name,entry in package['assets'].items():
                if not re.fullmatch(r'assets/[0-9a-f]{64}(?:\.[a-z0-9]+)?', name):
                    raise ValueError('Invalid project asset entry')
                target=owned/Path(name).name
                with bundle.open(name) as source,target.open('wb') as destination:
                    shutil.copyfileobj(source,destination,4*1024*1024)
                if _hash(target)!=entry['sha256']:
                    raise ValueError(f'Project asset checksum mismatch: {name}')
                destinations[ASSET+name]=str(target.resolve())
                verified_hashes[str(target.resolve())]=entry['sha256']

            def relocated(value):
                if isinstance(value,dict):
                    result={key:relocated(item) for key,item in value.items()}
                    path=result.get('path')
                    if path and Path(str(path)).suffix.lower()=='.json' and Path(str(path)).is_file() and 'sha256' in result:
                        result['sha256']=_hash(Path(path))
                        if 'bytes' in result:
                            result['bytes']=Path(path).stat().st_size
                    return result
                if isinstance(value,list):
                    return [relocated(item) for item in value]
                if isinstance(value,str):
                    if value.startswith(ASSET):
                        return destinations[value]
                    if value.startswith(CONTEXT):
                        return str(root/value.removeprefix(CONTEXT))
                    if value.startswith('cosplay-external://'):
                        return str(directory/'unbundled'/value.removeprefix('cosplay-external://'))
                return value

            # Rewrite content-bound JSON dependencies before their parents.
            # Approval graphs can be deeper than two levels; a fixed number
            # of passes leaves valid portable evidence with stale checksums.
            original_json={name:json.loads(bundle.read(name.removeprefix(ASSET)))
                for name,target in destinations.items() if Path(target).suffix=='.json'}
            written,visiting=set(),set()
            def hash_dependencies(value):
                if isinstance(value,dict):
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
        except BaseException:
            # Only this newly created run is removed; the workspace is retained.
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
