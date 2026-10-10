"""Read-only real saved native ROI, TIFF16 and portable delivery verification.

Uses frozen assets on CPU. No image/text APIs or model processes are allowed.
Large portable packaging is explicit and done once per selected saved version.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import tifffile

from automatic_native_retouch import read,sha,write
from display_transform import encode_display,plan_for_run as display_plan
from native_detail import read_scene_crop
from native_roi_render import render_native_roi
from single_source_workflow import load_run


def strip_crop(path,box):
    left,top,right,bottom=box
    with tifffile.TiffFile(path) as file:
        page=file.pages[0]
        if page.is_tiled or page.planarconfig!=1 or len(page.shape)!=3 or page.shape[2]!=3:
            raise ValueError('Verification requires the stored RGB strip master')
        result=np.empty((bottom-top,right-left,3),page.dtype)
        for index in range(top//page.rowsperstrip,(bottom-1)//page.rowsperstrip+1):
            file.filehandle.seek(page.dataoffsets[index])
            decoded,location,shape=page.decode(file.filehandle.read(page.databytecounts[index]),index)
            row=location[2];start,stop=max(top,row),min(bottom,row+shape[1])
            result[start-top:stop-top]=decoded[0,start-row:stop-row,left:right,:]
        return result


def boxes_for_run(run,edge=192):
    from cosplay_native_review import _source_labels
    plan=run['options']['appearance_revision']['design_reconstruction_plan']
    width,height=plan['canvas']['source_oriented_size'];result=[]
    def box(name,center,basis):
        x,y=center;left=int(np.clip(round(x-edge/2),0,width-edge));top=int(np.clip(round(y-edge/2),0,height-edge))
        result.append({'name':name,'box':[left,top,left+edge,top+edge],'sampling_basis':basis})
    try:
        labels,identity=_source_labels(run,plan)
        ph,pw=labels.shape
        for name,classes in [('face',(3,)),('left_hand',(6,)),('right_hand',(15,)),
                             ('cloth',(1,13,23)),('left_shoe',(9,)),('right_shoe',(18,))]:
            yy,xx=np.nonzero(np.isin(labels,classes))
            if not len(xx):continue
            nearest=np.argmin((xx-np.median(xx))**2+(yy-np.median(yy))**2)
            box(name,((float(xx[nearest])+.5)*width/pw,(float(yy[nearest])+.5)*height/ph),
                {'method':'nearest actual owner pixel to semantic median','classes':list(classes),
                 'frozen_labels_sha256':identity['sha256']})
    except (ValueError,KeyError,OSError) as exc:
        box('subject_center',(width*.5,height*.5),{'method':'native grid fallback','reason':str(exc)})
    box('upper_left_background',(width*.15,height*.25),{'method':'fixed native grid'})
    box('lower_right_floor',(width*.8,height*.93),{'method':'fixed native grid'})
    return result


def json_records(value):
    if isinstance(value,dict):
        if isinstance(value.get('path'),str) and value.get('sha256'):
            yield value
        for item in value.values():yield from json_records(item)
    elif isinstance(value,list):
        for item in value:yield from json_records(item)


def owned_closure(run,owned):
    """Check the frozen child graph without scanning mutable final QA files.

    Historical paths and model/code identities remain diagnostic metadata. Their
    references are reported separately; this check covers owned hash-bound files
    reachable from the saved rendering plan and retouch binding.
    """
    from portable_project import is_diagnostic_locator
    owned=Path(owned).resolve();pending=[run['options']['appearance_revision']['design_reconstruction_plan'],
        run['options'].get('native_source_retouch') or {}]
    checked,visited_json,external={},set(),set();records=0
    while pending:
        value=pending.pop()
        for record in json_records(value):
            target=Path(record['path'])
            if not target.is_absolute():target=owned/target
            target=target.resolve()
            if not target.is_relative_to(owned):
                external.add(str(target));continue
            if is_diagnostic_locator(target):
                external.add(str(target));continue
            if not target.is_file():raise AssertionError('Missing owned dependency: '+str(target))
            records+=1
            if str(target) not in checked:checked[str(target)]=sha(target)
            digest=checked[str(target)]
            if digest!=record['sha256']:raise AssertionError('Owned JSON record has stale SHA: '+str(target))
            if 'bytes' in record and record['bytes']!=target.stat().st_size:
                raise AssertionError('Owned JSON record has stale byte count: '+str(target))
            if target.suffix.lower()=='.json' and target not in visited_json:
                visited_json.add(target);pending.append(read(target))
    return {'status':'passed','scope':'owned frozen plan/retouch hash-bound dependency graph; mutable QA excluded',
        'hash_bound_records_checked':records,'unique_owned_assets_checked':len(checked),
        'owned_JSON_files_traversed':len(visited_json),'asset_sha256':checked,
        'external_history_or_runtime_paths_outside_owned_scope':len(external)}


def frozen_render_identity(run):
    value={'source':run['source'],'reference':run['reference'],
        'plan':run['options']['appearance_revision']['design_reconstruction_plan'],
        'native_source_retouch':run['options'].get('native_source_retouch'),
        'masters':{key:run['artifacts'][key] for key in ('auto_draft','render_master','render_scene_master')}}
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def verify(root,run_id,output,*,portable=False,allow_manifest_updates=False):
    import single_source_workflow as workflow
    import system_post_completion as system
    import export_artifact
    import portable_project
    import vision_transport
    import reference_provider
    import subprocess
    root,output=Path(root).resolve(),Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    manifest,run=load_run(root,run_id);native=run['options']['appearance_revision']['design_reconstruction_plan']['canvas']['source_oriented_size']
    paths=[root/'project.json',Path(__file__).resolve().parents[1]/'project.json',
        Path(run['source']['path']),Path(run['reference']['path'])]
    if not allow_manifest_updates:paths.append(manifest)
    declared_root=Path(run.get('root') or root)
    paths.append(declared_root/'project.json')
    for parent_root in {root,declared_root}:
        parent_manifest=parent_root/workflow.RUNS_DIR/str(run.get('parent_run_id'))/'manifest.json'
        if parent_manifest.is_file():
            if not allow_manifest_updates:paths.append(parent_manifest)
            parent=read(parent_manifest)
            paths.extend(Path(parent['artifacts'][key]) for key in
                ('auto_draft','render_master','render_scene_master') if parent.get('artifacts',{}).get(key))
    paths.extend(Path(value) for key,value in run['artifacts'].items() if key in ('auto_draft','render_master','render_scene_master'))
    preserved={str(path.resolve()):sha(path) for path in paths if path.is_file()}
    receipt={'schema':'saved_native_delivery_verification.v1','root':str(root),'run_id':run_id,
        'revision':run['revision'],'source_sha256':run['source']['sha256'],'reference_sha256':run['reference']['sha256'],
        'native_size':native,'verification_scope':'frozen CPU pixel replay and storage/portability; no visual-quality approval',
        'external_api_requests':0,'gpu_used':False,'automatic_visual_acceptance':False,'preserved_inputs':preserved,
        'allow_manifest_metadata_updates':allow_manifest_updates,'frozen_render_identity':frozen_render_identity(run),'stages':{}}
    started=time.perf_counter()
    def persist():write(output/'verification.json',receipt)
    persist()
    def unavailable(*args,**kwargs):raise AssertionError('Saved verification must not call API or spawn a model process')
    try:
        with patch.object(vision_transport,'request_vision_json',side_effect=unavailable), \
             patch.object(reference_provider,'main',side_effect=unavailable), \
             patch.object(subprocess,'Popen',side_effect=unavailable):
            samples=[]
            closure_failure=None
            try:receipt['stages']['owned_dependency_closure']=owned_closure(run,manifest.parent)
            except (AssertionError,ValueError,OSError) as exc:
                closure_failure=exc
                receipt['stages']['owned_dependency_closure']={'status':'failed','error_type':type(exc).__name__,'error':str(exc)}
            persist()
            if run['options'].get('native_source_retouch'):
                from native_retouch_producer import validate_frozen_retouch_layer
                binding=run['options']['native_source_retouch']
                receipt['stages']['frozen_retouch']={'status':'passed',**validate_frozen_retouch_layer(
                    read(binding['layer']['path']),source_sha256=run['source']['sha256'],native_size=native)};persist()
            for item in boxes_for_run(run):
                roi,execution=render_native_roi(run,item['box'])
                frozen,precision=read_scene_crop(run['artifacts']['render_scene_master'],item['box'])
                # Different bounded array layouts can change final float32
                # accumulation by a few ULPs. Delivery quantization below must
                # still be exact; do not call tolerance an exact float replay.
                max_difference=float(np.max(np.abs(roi-frozen)))
                np.testing.assert_allclose(roi,frozen,rtol=4*np.finfo(np.float32).eps,atol=1e-8)
                expected16=np.asarray(encode_display(roi,display_plan(run))*65535.0+.5,dtype=np.uint16)
                stored16=strip_crop(run['artifacts']['render_master'],item['box'])
                np.testing.assert_array_equal(expected16,stored16)
                preview=np.asarray(Image.open(run['artifacts']['auto_draft']).crop(item['box']).convert('RGB'))
                expected8=np.asarray(encode_display(roi,display_plan(run))*255.0+.5,dtype=np.uint8)
                np.testing.assert_array_equal(preview,expected8)
                samples.append({**item,'native_ROI_float32_exact':bool(np.array_equal(roi,frozen)),
                    'native_ROI_float32_max_difference':max_difference,'native_ROI_float32_within_float32_tolerance':True,
                    'float32_relative_tolerance':float(4*np.finfo(np.float32).eps),'float32_absolute_tolerance':1e-8,
                    'stored_TIFF16_quantization_exact':True,
                    'stored_PNG8_quantization_exact':True,'linear_crop_sha256':hashlib.sha256(roi.tobytes()).hexdigest(),
                    'uint16_crop_sha256':hashlib.sha256(stored16.tobytes()).hexdigest(),
                    'bounded_native_ROI':execution['bounded_roi_renderer'],'full_canvas_rendered':execution['full_canvas_rendered'],
                    'working_domain':precision['working_domain']})
                receipt['stages']['native_samples']={'status':'verifying','samples':samples};persist()
            receipt['stages']['native_samples'].update(status='passed',sample_count=len(samples));persist()
            exported=export_artifact.export_source_run(root,run_id,directory=str(output/'tiff16'),format='TIFF16',artifact='draft')
            assert sha(exported['path'])==sha(run['artifacts']['render_master'])
            with tifffile.TiffFile(exported['path']) as file:
                page=file.pages[0];assert page.dtype==np.uint16 and page.shape==tuple(native[::-1])+ (3,) and page.bitspersample==16
                assert 34675 in page.tags
            assert exported['metadata']['approved'] is False and exported['metadata']['export_rerendered'] is False
            receipt['stages']['tiff16_export']={'status':'passed','actual_native_RGB_uint16':True,'ICC_present':True,
                'exact_saved_master_copy':True,'new_render_or_decode':False,**exported};persist()
            _,reopened=load_run(root,run_id)
            ordinary=SimpleNamespace(generate=False,complete_post=False,cosplay_correction_budget=0)
            assert system.complete_once(root,reopened,ordinary)==reopened
            receipt['stages']['saved_reopen']={'status':'passed','network_requests':0,'model_processes':0};persist()
            from native_detail import saved_native_detail
            from live_native_detail import live_native_detail
            box=samples[0]['box'];width,height=native
            body={'center':[(box[0]+box[2])/2/width,(box[1]+box[3])/2/height],'edge':box[2]-box[0]}
            saved=saved_native_detail(root,run_id,body);live=live_native_detail(root,run_id,body)
            with Image.open(saved['preview_path']) as saved_image,Image.open(live['preview_path']) as live_image:
                np.testing.assert_array_equal(np.asarray(saved_image),np.asarray(live_image))
            public={'schema':'public_saved_native_reopen.v1','run_id':run_id,'ok':True,'RGB_exact':True,
                'network_requests':0,'model_processes':0,'saved':saved,'live':live}
            write(output/'public-native-reopen.json',public)
            receipt['stages']['public_native_reopen']={'status':'passed','saved_live_RGB_exact':True,
                'network_requests':0,'model_processes':0,'receipt':str(output/'public-native-reopen.json')};persist()
            if portable:
                archive=output/(run_id+'.cosplay')
                if archive.exists():
                    import zipfile
                    with zipfile.ZipFile(archive) as package:
                        saved=read_package=json.loads(package.read('project.json'))
                        if saved['run_id']!=run_id:raise ValueError('Existing package belongs to another saved run')
                    package_result={'archive':str(archive),'archive_reused':True,'bytes':archive.stat().st_size}
                else:package_result=portable_project.export_project(root,run_id,archive)
                receipt['stages']['portable']={'status':'exported','export':package_result};persist()
                import zipfile
                with zipfile.ZipFile(archive) as package:
                    runtime_entries=[name for name in package.namelist() if portable_project.is_diagnostic_locator(name)]
                assert not runtime_entries,'Portable package includes code/model runtime bytes: '+str(runtime_entries)
                receipt['stages']['portable']['runtime_code_model_entry_count']=0;persist()
                moved=output/'imported-workspace';identifier=run_id+'-portable-verified'
                imported_manifest=moved/workflow.RUNS_DIR/identifier/'manifest.json'
                if imported_manifest.exists():_,imported=load_run(moved,identifier)
                else:imported=portable_project.import_project(moved,archive,run_id=identifier)['run']
                for key in ('auto_draft','render_master','render_scene_master'):
                    assert sha(imported['artifacts'][key])==sha(run['artifacts'][key])
                if (imported.get('options') or {}).get('native_source_retouch'):
                    from native_retouch_producer import validate_frozen_retouch_layer
                    binding=imported['options']['native_source_retouch']
                    validate_frozen_retouch_layer(read(binding['layer']['path']),source_sha256=run['source']['sha256'],native_size=native)
                checked=0
                for path in (moved/workflow.RUNS_DIR/identifier/'inputs/portable').glob('*.json'):
                    for record in json_records(read(path)):
                        target=Path(record['path'])
                        if target.is_file() and not portable_project.is_diagnostic_locator(target):
                            assert sha(target)==record['sha256'],'Relocated JSON record has stale SHA: '+str(target)
                            checked+=1
                imported_roi,_=render_native_roi(imported,samples[0]['box'])
                original_roi,_=render_native_roi(run,samples[0]['box']);np.testing.assert_array_equal(imported_roi,original_roi)
                assert system.complete_once(moved,imported,ordinary)==imported
                receipt['stages']['portable'].update(status='passed',imported_root=str(moved),imported_run_id=identifier,
                    all_saved_masters_byte_exact=True,complete_JSON_records_hash_checked=checked,
                    imported_native_ROI_exact=True,imported_saved_reopen_requests=0)
                persist()
            if closure_failure is not None:raise closure_failure
        receipt.update(ok=True,status='passed')
    except Exception as exc:
        receipt.update(ok=False,status='failed',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        receipt['elapsed_seconds']=round(time.perf_counter()-started,3)
        receipt['preserved_inputs_unchanged']=all(Path(path).is_file() and sha(path)==digest for path,digest in preserved.items())
        _,latest=load_run(root,run_id)
        receipt['frozen_render_identity_unchanged']=frozen_render_identity(latest)==receipt['frozen_render_identity']
        if not receipt['preserved_inputs_unchanged'] or not receipt['frozen_render_identity_unchanged']:
            receipt.update(ok=False,status='failed',error_type='AssertionError',
                error='A protected input, master, project or frozen rendering plan changed during verification')
        persist()
        if not receipt['preserved_inputs_unchanged']:raise AssertionError('Verification changed a protected root/source/parent/run/master')
        if not receipt['frozen_render_identity_unchanged']:raise AssertionError('Frozen render plan changed during verification')
    print(json.dumps({'ok':receipt['ok'],'run_id':run_id,'native_samples':len(samples),'portable':portable,
        'preserved_inputs_unchanged':True,'receipt':str(output/'verification.json'),'elapsed_seconds':receipt['elapsed_seconds']}))
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--run',required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--portable',action='store_true')
    parser.add_argument('--allow-manifest-updates',action='store_true',help='Allow QA metadata updates while guarding the frozen plan and parent/child pixels')
    args=parser.parse_args();verify(args.root,args.run,args.output,portable=args.portable,allow_manifest_updates=args.allow_manifest_updates)
