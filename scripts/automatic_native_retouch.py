"""One durable source-only retouch attempt for an authorized new preparation.

Saved renders only consume frozen assets; this module is never a render hook.
All recognition/producer dependencies run in the selected model interpreter.
"""
from __future__ import annotations

from runtime_resources import external_python_popen

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from asset_identity import sha256_file

SCHEMA='automatic_native_retouch.v1'
BINDING='automatic_native_retouch_binding.v1'
MAXIMUM=8


def read(path): return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path): return sha256_file(path)
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8');temp.replace(path)
def resolve(root,value):
    path=Path(value);return (path if path.is_absolute() else Path(root)/path).resolve()


def _owned_layer(root,run,layer_path):
    """Copy the causal graph and rehash relocated JSON from dependencies up.

    Camera RAW remains the root input. All crop, API raster, receipt, prompt,
    semantic and delta assets become version-owned, without lossy re-encoding.
    """
    root=Path(root).resolve();source=resolve(root,run['source']['path'])
    directory=root/'outputs/runs'/run['run_id']/'inputs/native-retouch'
    directory.mkdir(parents=True,exist_ok=True)
    targets={};visiting=set()
    def copy_file(path):
        path=Path(path).resolve()
        if path==source: return source
        if path in targets: return targets[path]
        name=hashlib.sha256(str(path).encode()).hexdigest()[:16]+'-'+path.name
        target=directory/name;targets[path]=target
        if path.suffix.lower()=='.json' and path not in visiting:
            visiting.add(path);value=relocate(read(path),path.parent);visiting.remove(path)
            write(target,value)
        else:
            shutil.copy2(path,target)
        return target
    def relocate(value,parent):
        if isinstance(value,list): return [relocate(item,parent) for item in value]
        if not isinstance(value,dict): return value
        result={}
        for key,item in value.items():
            path_key=key in {'path','source','manifest'} or key.endswith(('_path','_file'))
            if path_key and isinstance(item,str) and '://' not in item:
                path=Path(item);path=path if path.is_absolute() else parent/path
                if path.is_file() and path.suffix.lower() not in {'.py','.exe','.dll','.pt','.pth','.safetensors'}:
                    result[key]=str(copy_file(path));continue
            result[key]=relocate(item,parent)
        if isinstance(result.get('path'),str) and 'sha256' in result and Path(result['path']).is_file():
            result['sha256']=sha(result['path'])
            if 'bytes' in result:result['bytes']=Path(result['path']).stat().st_size
        return result
    return copy_file(layer_path)


def own_binding(root,run):
    """Re-own the frozen retouch closure for a child; no API or pixel algorithm."""
    run=copy.deepcopy(run);record=(run.get('options') or {}).get('native_source_retouch')
    if not record:return run
    layer_path=resolve(root,record['layer']['path'])
    if sha(layer_path)!=record['layer']['sha256']:raise ValueError('Retouch layer changed before child ownership')
    owned=_owned_layer(root,run,layer_path)
    record['layer']={'path':str(owned),'sha256':sha(owned)}
    record['owner_run_id']=run['run_id']
    plan=((run.get('options') or {}).get('appearance_revision') or {}).get('design_reconstruction_plan')
    if plan and 'source_baseline' in plan:
        plan['source_baseline']['native_source_retouch']=copy.deepcopy(record)
    return run




def _bound_face_protection(face,proxy,semantic,output):
    """Rebind byte-identical cached evidence paths without inventing landmarks."""
    proxy=Path(proxy).resolve();identity=face.get('identity',{})
    if identity.get('input')!=sha(proxy) or identity.get('labels')!=sha(semantic['labels_path']):
        raise ValueError('Cached face protection belongs to different source or semantic pixels')
    value=copy.deepcopy(face);original=Path(face['field_path']).parent/'result.json'
    value['input_path']=str(proxy)
    value['binding_revalidation']={'original_evidence':{'path':str(original),'sha256':sha(original)},
        'same_input_sha256':sha(proxy),'same_labels_sha256':sha(semantic['labels_path']),
        'landmarks_recomputed':False}
    target=Path(output)/'face-protection-bound-source.json';write(target,value)
    return target




def _validate_beauty_source_before_preparation(root,run):
    """Check a frozen source lock before a fresh reference baseline exists."""
    options=run.get('options') or {};revision=options.get('appearance_revision') or {}
    from portrait_beauty_layer import validate_source_for_run,load_result,identity_for_run
    # An existing plan must pass its full contract, including a malformed or
    # empty plan. Only the genuinely new task has no baseline to validate yet.
    if 'design_reconstruction_plan' in revision or 'source_only_plan' in options:
        validate_source_for_run(run,root=root)
        return
    control=revision['portrait_beauty']
    if not isinstance(control,dict) or set(control)-{'result','strength','disabled_faces'}:
        raise ValueError('Beauty revision requires result, strength and disabled_faces')
    result,_=load_result(control['result']);source=run.get('source') or {}
    from raw_support import RGB_DECODE_VERSION,is_raw
    precision=result.get('source_precision') or {}
    if (not isinstance(precision,dict) or source.get('decode_version')!=RGB_DECODE_VERSION
            or precision.get('decode_version')!=RGB_DECODE_VERSION
            or result['source_sha256']!=source.get('sha256')):
        raise ValueError('Beauty preparation must bind the current source and decoder')
    if result['native_size']!=source.get('oriented_size'):
        raise ValueError('Beauty layer belongs to a different native source canvas')
    expected='raw16_linear' if is_raw(Path(source['path'])) else 'raster8_srgb'
    if precision.get('source_precision')!=expected:
        raise ValueError('Beauty preparation has a different source precision')
    from display_transform import validate_source_transform
    validate_source_transform(result['source_display'],expected)
    from native_source_baseline import retouch_identity
    retouch=retouch_identity(run,root=root)
    stable={key:retouch[key] for key in ('delta_sha256','source_sha256','working_domain')} if retouch else None
    if stable!=result.get('native_source_retouch'):
        raise ValueError('Source retouch changed after beauty preparation; prepare beauty again')
    identity_for_run(run)
    # The reference producer will create the actual baseline. Its Beauty
    # replay retains validate_source_for_run; no placeholder plan is invented.


def prepare_once(root,run,args=None,*,source_analysis_path=None,cancel=None,reuse_layer_path=None,authorized=True,reuse_only=False):
    """ABI: return {run,status,layer_record}; invoke only at new/explicit work.

    ``authorized`` is explicit integration context, not inferred from a saved
    render. Cancellation/error reserves the attempt and retains the original
    draft; interrupted processes never repeat recognition on reopening.
    """
    root=Path(root).resolve();retained=copy.deepcopy(run)
    if not authorized:return {'run':retained,'status':'not_authorized_for_new_preparation','layer_record':None}
    if ((retained.get('options') or {}).get('appearance_revision') or {}).get('portrait_beauty'):
        _validate_beauty_source_before_preparation(root,retained)
        # BeautyGRPO and its reference fit already consume this source. A
        # completion child must replay that input instead of adding a new
        # upstream retouch and invalidating the generated face layer.
        retained['automatic_native_retouch'] = {
            'schema': SCHEMA, 'status': 'source_locked_by_portrait_beauty',
            'source_sha256': run['source']['sha256'], 'generation_submissions': 0,
            'upstream_source_changed': False}
        return {'run':retained,'status':'source_locked_by_portrait_beauty',
                'layer_record':(retained.get('options') or {}).get('native_source_retouch')}
    if (retained.get('options') or {}).get('native_source_retouch'):
        from native_source_baseline import retouch_identity
        retouch_identity(retained,root=root)
        return {'run':retained,'status':'frozen_source_retouch_reused','layer_record':retained['options']['native_source_retouch']}
    directory=root/'outputs/runs'/run['run_id']/'automatic-native-retouch';directory.mkdir(parents=True,exist_ok=True)
    marker=directory/'attempt.json'
    if marker.exists():
        record=read(marker);result=directory/'result.json'
        if result.is_file() and record.get('status')=='frozen_source_stage_ready':
            binding=read(result)['layer_record'];retained.setdefault('options',{})['native_source_retouch']=binding
            from native_source_baseline import retouch_identity
            retouch_identity(retained,root=root)
            return {'run':retained,'status':'reopened_frozen_source_stage','layer_record':binding}
        retained['automatic_native_retouch']=record
        return {'run':retained,'status':'already_attempted','layer_record':None}
    record={'schema':SCHEMA,'status':'reserved','attempt':1,'owner_run_id':run['run_id'],
        'source_sha256':run['source']['sha256'],'maximum_candidates':MAXIMUM,
        'generation_submissions':0,'original_draft_retained':True,'automatic_visual_acceptance':False}
    with marker.open('x',encoding='utf8') as stream:json.dump(record,stream,ensure_ascii=False,indent=2)
    write(directory/'input-run.json',run)
    def cancelled():
        return bool((cancel and cancel()) or (getattr(args,'cancel_request_path',None) and Path(args.cancel_request_path).exists()))
    try:
        if cancelled():raise InterruptedError('Native source preparation cancelled')
        from local_photo_analysis import local_runtime_paths
        runtime=local_runtime_paths(root,registry_path=(run.get('options') or {}).get('model_registry'))
        command=[str(runtime['python']),'-X','utf8',str(Path(__file__).resolve()),'--worker','--root',str(root),
            '--run',str(directory/'input-run.json'),'--output',str(directory)]
        for flag,value in [('--analysis',source_analysis_path),('--reuse-layer',reuse_layer_path),
            ('--provider',getattr(args,'provider',None)),('--vision-model',getattr(args,'vision_model',None)),
            ('--api-base',getattr(args,'api_base',None)),('--api-url',getattr(args,'api_url',None)),
            ('--api-key-env',getattr(args,'api_key_env',None)),('--cancel-request-path',getattr(args,'cancel_request_path',None))]:
            if value:command += [flag,str(value)]
        if reuse_only:command+=['--reuse-only']
        with (directory/'worker.log').open('w',encoding='utf8') as log:
            process=external_python_popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONIOENCODING':'utf8'},
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            try:
                while process.poll() is None:
                    if cancelled():
                        process.terminate();process.wait(timeout=5);raise InterruptedError('Native source preparation cancelled')
                    time.sleep(.25)
            finally:
                if process.poll() is None:process.kill();process.wait(timeout=5)
        if cancelled():raise InterruptedError('Native source preparation cancelled')
        if process.returncode:raise RuntimeError('Native retouch worker unavailable; see '+str(directory/'worker.log'))
        result=read(directory/'result.json')
        retained.setdefault('options',{})['native_source_retouch']=result['layer_record']
        record.update(status='frozen_source_stage_ready',original_draft_retained=result['layer_record']['approved_patches']==0,
            approved_patches=result['layer_record']['approved_patches'],reused_actual_evidence=result['reused_actual_evidence'])
        write(marker,record)
        retained['automatic_native_retouch']=record
        return {'run':retained,'status':record['status'],'layer_record':result['layer_record']}
    except Exception as exc:
        record.update(status='cancelled_draft_retained' if isinstance(exc,InterruptedError) else 'unavailable_draft_retained',
            error_type=type(exc).__name__,error=str(exc)[:500],original_draft_retained=True)
        write(marker,record);retained['automatic_native_retouch']=record
        return {'run':retained,'status':record['status'],'layer_record':None}


def worker(root,run_path,output,*,analysis_path=None,reuse_layer=None,provider=None,vision_model=None,
           api_base=None,api_url=None,api_key_env=None,cancel_request_path=None,reuse_only=False):
    import native_retouch_producer as producer
    from raw_support import open_linear
    root,output=Path(root).resolve(),Path(output).resolve();run=read(run_path)
    def check_cancel():
        if cancel_request_path and Path(cancel_request_path).exists():
            raise InterruptedError('Native source preparation cancelled')
    check_cancel()
    source=resolve(root,run['source']['path'])
    if sha(source)!=run['source']['sha256']:raise ValueError('Automatic retouch source differs from the original root')
    linear,precision=open_linear(source);native=list(linear.shape[1::-1]);del linear
    expected_domain='scene_linear_sRGB' if precision['source_precision']=='raw16_linear' else 'display_linear_sRGB'
    candidates=[Path(reuse_layer)] if reuse_layer else []
    matched=[]
    for path in candidates:
        try:
            layer=read(path)
            if layer.get('source',{}).get('sha256')!=run['source']['sha256']:continue
            checked=producer.validate_frozen_retouch_layer(layer,source_sha256=run['source']['sha256'],native_size=native)
            donors=read(layer['donor_manifest']['path']);proposals=read(donors['proposals']['path'])
            if checked['working_domain']!=expected_domain:
                if reuse_layer:raise ValueError('Explicit retouch evidence belongs to a different RAW/display domain')
                continue
            if proposals['producer_sha256']!=sha(producer.__file__):
                raise ValueError('Explicit retouch evidence was produced by different code; old evidence is not upgraded')
            # Reuse requires actual recognition for every proposed candidate,
            # including rejects/unknowns. Missing files are never "zero fix" QA.
            for candidate in proposals['candidates']:
                producer.validate_vision_receipt(read(candidate['request']['approval_path']),candidate['request'],candidate['target_crop'])
            matched.append(path)
        except (ValueError,OSError,KeyError):
            if reuse_layer:raise
    reused=bool(matched)
    if reused:layer_path=matched[0]
    else:
        if reuse_only:raise ValueError('No exact actual retouch evidence available; reuse-only mode forbids fresh model/API work')
        check_cancel()
        from local_photo_analysis import prepare_local_appearance
        analysis=read(analysis_path) if analysis_path else prepare_local_appearance(root,source,
            source_hash=run['source']['sha256'],registry_path=(run.get('options') or {}).get('model_registry'))
        analysis_path=Path(analysis_path) if analysis_path else Path(analysis['analysis_directory'])/'analysis.json'
        face_path=None
        try:
            from local_face_protection import prepare_face_protection
            face=prepare_face_protection(root,Path(analysis['analysis_directory'])/'source-srgb.png',analysis['semantic_result'])
            face_path=_bound_face_protection(face,Path(analysis['analysis_directory'])/'source-srgb.png',analysis['semantic_result'],output)
        except (OSError,RuntimeError,ValueError,KeyError):pass
        proposals=producer.prepare_proposals(source,analysis_path,output/'producer',face_protection_path=face_path,maximum=MAXIMUM)
        import vision_transport
        def reviews(requests):
            for request in requests:
                check_cancel()
                payload,provenance=vision_transport.request_vision_json(Path(request['source']),Path(request['out_dir']),
                    Path(request['prompt_path']).read_text(encoding='utf8'),provider=provider,
                    vision_model=vision_model or 'gpt-6.1-sol',api_base=api_base,api_url=api_url,api_key_env=api_key_env)
                write(request['approval_path'],{'response':payload,'provenance':provenance})
                check_cancel()
        reviews(proposals['vision_requests'])
        donors=producer.prepare_donors(output/'producer/proposals.json')
        reviews(donors['vision_requests'])
        producer.freeze_approved_retouch(output/'producer/donors.json')
        layer_path=output/'producer/frozen/layer.json'
    check_cancel()
    owned=_owned_layer(root,run,layer_path);layer=read(owned)
    checked=producer.validate_frozen_retouch_layer(layer,source_sha256=run['source']['sha256'],native_size=native)
    binding={'schema':BINDING,'owner_run_id':run['run_id'],'source_sha256':run['source']['sha256'],
        'native_size':native,'working_domain':checked['working_domain'],'layer':{'path':str(owned),'sha256':sha(owned)},
        'approved_patches':len(layer['patches']),'execution_stage':'before_display_alpha_decomposition_and_reference_TBS',
        'frozen_baseline_replay_consumptions':0,'producer_sha256':sha(producer.__file__),
        'automatic_visual_acceptance':False}
    result={'schema':SCHEMA,'layer_record':binding,'reused_actual_evidence':reused,'api_requests_sent':0 if reused else
        len(proposals['vision_requests'])+len(donors['vision_requests'])}
    write(output/'result.json',result)
    print(json.dumps({'status':'frozen_source_stage_ready','approved_patches':binding['approved_patches'],
        'reused_actual_evidence':reused,'api_requests_sent':result['api_requests_sent']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',action='store_true')
    for name in ('root','run','output'):parser.add_argument('--'+name,required=True)
    for name in ('analysis','reuse-layer','provider','vision-model','api-base','api-url','api-key-env','cancel-request-path'):
        parser.add_argument('--'+name)
    parser.add_argument('--reuse-only',action='store_true')
    a=parser.parse_args();worker(a.root,a.run,a.output,analysis_path=a.analysis,reuse_layer=a.reuse_layer,provider=a.provider,
        vision_model=a.vision_model,api_base=a.api_base,api_url=a.api_url,api_key_env=a.api_key_env,
        cancel_request_path=a.cancel_request_path,reuse_only=a.reuse_only)
