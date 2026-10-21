"""Export current saved native masters without rendering or format upgrades."""
from __future__ import annotations
import hashlib
import io
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import numpy as np
import tifffile
from PIL import Image, ImageCms
from asset_identity import sha256_file


def _hash(path):
    return sha256_file(path)


def _master_header(path, *, floating, size):
    if not path.is_file():
        raise FileNotFoundError("Save a current native master before exporting: "+str(path))
    with tifffile.TiffFile(path) as actual:
        page=actual.pages[0]
        if len(actual.pages)!=1 or page.shape!=(size[1],size[0],3) or page.photometric!=2 or page.planarconfig!=1:
            raise ValueError("Saved master must be native-size contiguous RGB")
        if page.dtype!=(np.dtype('float32') if floating else np.dtype('uint16')):
            raise ValueError("Saved master has the wrong actual precision")
        metadata=(actual.shaped_metadata or ({},))[0] or {}
        embedded=34675 in page.tags
        if floating:
            if metadata.get('working_domain')!='display_linear_sRGB' or metadata.get('display_transform_applied') is not True:
                raise ValueError("Only the current display-linear float32 master can be exported")
            if embedded:
                raise ValueError("Linear master cannot carry a nonlinear sRGB ICC profile")
        else:
            if not embedded:
                raise ValueError("The uint16 display master must embed its sRGB ICC profile")
            profile=ImageCms.ImageCmsProfile(io.BytesIO(page.tags[34675].value))
            if 'srgb' not in ImageCms.getProfileDescription(profile).casefold():
                raise ValueError("The uint16 display master must use sRGB")
        return {'actual_bit_depth':32 if floating else 16,'icc_profile_embedded':embedded}


def export_source_run(root: Path, run_id: str, *, directory: str, format: str="PNG", artifact: str="draft") -> dict:
    import single_source_workflow as workflow
    from design_reconstruction import render_mode_for_run,plan_for_run,validate_current_native_plan
    from display_transform import plan_for_run as display_plan_for_run
    root=Path(root).resolve();manifest_path,run=workflow.load_run(root,run_id)
    saved_run = run
    continuation=run.get('continuation_edit')
    if isinstance(continuation,dict) and continuation.get('render_state')!='saved':
        raise RuntimeError("Save continuation edits before exporting")
    if not directory.strip():raise ValueError("Choose an export directory")
    destination=Path(directory).expanduser().resolve()
    for protected in (root/'data',root/'outputs/runs',root/'outputs/final_delivery'):
        if destination==protected.resolve() or protected.resolve() in destination.parents:
            raise ValueError("Choose an export directory outside source/run/delivery storage")
    format={'JPG':'JPEG','TIF':'TIFF'}.get(format.upper(),format.upper())
    extensions={'JPEG':'.jpg','PNG':'.png','TIFF':'.tif','TIFF16':'.tif','TIFF32_LINEAR':'.tif'}
    if format not in extensions:raise ValueError("Supported export formats: JPEG, PNG, TIFF, TIFF16, TIFF32_LINEAR")
    def resolve(value):
        if not value:raise ValueError("Current saved artifact is missing")
        path=Path(str(value));return path if path.is_absolute() else root/path
    if artifact=='delivery':
        delivery,record=workflow._delivery_record(root,str(run['sample']),run_id)
        run=json.loads((delivery/'manifest.json').read_text('utf8'))
        image_path=delivery/'auto_draft.png';master_path=delivery/'render_master.tif';scene_path=delivery/'render_scene.tif'
        review_path=delivery/'visual_review.json';approved=True
        approval={'reviewer':record.get('reviewer'),'reviewed_at':record.get('reviewed_at')}
    elif artifact=='draft':
        outputs=run.get('artifacts') or {}
        image_path=resolve(outputs.get('auto_draft'));master_path=resolve(outputs.get('render_master'));scene_path=resolve(outputs.get('render_scene_master'))
        review_path=resolve(outputs.get('visual_review') or manifest_path.parent/'visual_review.json')
        approved=False;approval=None
    else:raise ValueError("artifact must be draft or delivery")
    scoped=(run.get('task_plan') or {}).get('editing_scope')
    typed=(run.get('options') or {}).get('photo_appearance')
    if typed:
        from generated_appearance_render import validate_run,saved_display_transform
        from appearance_assets import validate_asset
        scope,design,_,_=validate_run(root,run)
        receipt=validate_asset(typed,source_sha256=run['source']['sha256'],scope=scope,
            design_id=design['observed_design']['design_id'])
        size=tuple(receipt['canvas_size'])
        render=run.get('render') or {}
        if render.get('photo_appearance')!=typed:
            raise ValueError('Saved render belongs to a different appearance master')
        saved=render.get('saved_artifact_sha256') or {}
        for key,path in (('auto_draft',image_path),('render_master',master_path),('render_scene_master',scene_path)):
            if not saved.get(key) or _hash(path)!=saved[key]:
                raise ValueError('Saved typed render bytes changed: '+key)
        if _hash(image_path)!=receipt['files']['preview']['sha256']:
            raise ValueError('Saved preview differs from the selected native master')
        display=saved_display_transform(root,run)
    elif scoped:
        raise ValueError('A current scoped result requires its native appearance asset')
    else:
        render_mode_for_run(run)
        plan=plan_for_run(run)
        if plan:
            validate_current_native_plan(plan);size=tuple(map(int,plan['canvas']['source_oriented_size']))
        else:
            source_only=run.get('options',{}).get('source_only_plan') or {}
            size=tuple(map(int,source_only.get('native_size') or []))
            if len(size)!=2:raise ValueError("Current source-only native canvas is missing")
        display=display_plan_for_run(run)
    render=run.get('render') or {};precision=render.get('render_precision') or {}
    if (render.get('operator',{}).get('working_domain')!='display_linear_sRGB'
            or precision.get('working_domain')!='display_linear_sRGB'
            or precision.get('working_precision')!='float32_linear'
            or precision.get('master_encoding')!='uint16_srgb'):
        raise ValueError("Export requires a saved current display-linear float32 render")
    master_header=_master_header(master_path,floating=False,size=size)
    scene_header=_master_header(scene_path,floating=True,size=size)
    if not image_path.is_file():raise FileNotFoundError("Current saved preview is missing")
    with Image.open(image_path) as preview:
        if preview.size!=size or preview.mode!='RGB' or not preview.info.get('icc_profile'):
            raise ValueError("Current saved preview must be native RGB with sRGB ICC")
        profile=ImageCms.ImageCmsProfile(io.BytesIO(preview.info['icc_profile']))
        if 'srgb' not in ImageCms.getProfileDescription(profile).casefold():
            raise ValueError('Current saved preview must use the current sRGB colour space')
        preview_format=preview.format
        if format=='PNG' and preview_format=='PNG':
            preview.verify()
    label='approved' if approved else 'draft';destination.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    target=destination/f"{run['sample']}-r{run.get('revision',1)}-{label}-{stamp}-{uuid4().hex[:8]}{extensions[format]}"
    floating=format=='TIFF32_LINEAR';high_bit=format in {'TIFF16','TIFF32_LINEAR'}
    copied_bytes = False
    if high_bit:
        render_path=scene_path if floating else master_path
        render_sha256 = _hash(render_path)
        shutil.copy2(render_path,target)
        if target.stat().st_size != render_path.stat().st_size:
            raise IOError('Exact master copy did not preserve byte size')
        export_sha256 = _hash(target)
        if export_sha256 != render_sha256:
            raise IOError('Exact master copy did not preserve content')
        copied_bytes = True
        header=scene_header if floating else master_header
        precision_origin='stored_display_linear_float32_master' if floating else 'stored_uint16_srgb_master'
        mode='RGB_float32' if floating else 'RGB_uint16'
    else:
        render_path=image_path
        if format=='PNG' and preview_format=='PNG':
            # The saved native preview already has the validated ICC/profile.
            # Copy its exact bytes rather than decoding and compressing again.
            render_sha256 = _hash(image_path)
            shutil.copy2(image_path,target)
            if target.stat().st_size != image_path.stat().st_size:
                raise IOError('Exact preview copy did not preserve byte size')
            export_sha256 = _hash(target)
            if export_sha256 != render_sha256:
                raise IOError('Exact preview copy did not preserve content')
            copied_bytes = True
        else:
            with Image.open(image_path) as original:
                kwargs={'icc_profile':original.info['icc_profile']}
                if format=='JPEG':kwargs.update(quality=95,subsampling=0)
                elif format=='TIFF':kwargs.update(compression='tiff_lzw')
                original.save(target,format=format,**kwargs)
        header={'actual_bit_depth':8,'icc_profile_embedded':True};mode='RGB'
        precision_origin='rgb8_preview_from_current_master'
    metadata={
        'schema_version':'source_export.v3','run_id':run_id,'revision':run.get('revision',1),'sample':run['sample'],
        'artifact':artifact,'status':label,'approved':approved,'approval':approval,'created_at':datetime.now(timezone.utc).isoformat(),
        'format':format,'width':size[0],'height':size[1],'mode':mode,**header,
        'color_space':'linear RGB (sRGB primaries)' if floating else 'sRGB',
        'transfer_function':'linear' if floating else 'sRGB',
        'icc_profile_origin':None if floating else 'stored current sRGB master or derived preview',
        'source_sha256':run['source']['sha256'],'render_path':str(render_path),
        'render_sha256':render_sha256 if copied_bytes else _hash(render_path),
        'export_path':str(target),'export_sha256':export_sha256 if copied_bytes else _hash(target),'review_path':str(review_path),
        'review_sha256':_hash(review_path) if review_path.is_file() else None,
        'review_basis':('stored_working_master_with_display_transform' if floating else 'stored_float_master_and_derived_preview') if approved else None,
        'display_transform':display,'software_version':run.get('software_version'),'camera_raw_master':False,
        'precision_origin':precision_origin,'render_precision':precision,'high_bit_depth_master':high_bit,'high_bit_depth_render':high_bit,
        'working_domain':'display_linear_sRGB','working_linear_master':floating,'scene_linear_master':False,
        'appearance_precision':'display_linear_float32','export_rerendered':False}
    sidecar=target.with_suffix(target.suffix+'.json');sidecar.write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n','utf8')
    from artifact_receipts import file_stamp, write_receipt
    render_artifact = 'render_scene_master' if floating else 'render_master' if high_bit else 'auto_draft'
    saved_value = (saved_run.get('artifacts') or {}).get(render_artifact)
    saved_path = resolve(saved_value) if saved_value else None
    saved_binding = {}
    if saved_path is not None and saved_path.is_file():
        saved_hash = metadata['render_sha256'] if saved_path.resolve() == render_path.resolve() else _hash(saved_path)
        saved_binding = {'path': str(saved_path.resolve()), 'sha256': saved_hash, 'stamp': file_stamp(saved_path)}
    receipt = {
        'schema_version': 'source_export_receipt.v2',
        'status': 'succeeded',
        'artifact': artifact,
        'format': format,
        'render_artifact': render_artifact,
        'saved_render': saved_binding,
        'export': {'path': str(target), 'sha256': metadata['export_sha256'], 'stamp': file_stamp(target)},
        'sidecar': {'path': str(sidecar), 'sha256': _hash(sidecar), 'stamp': file_stamp(sidecar)},
        'actual_bit_depth': metadata['actual_bit_depth'],
        'color_space': metadata['color_space'],
        'created_at': metadata['created_at'],
    }
    write_receipt(root, saved_run, receipt)
    return {'ok':True,'path':str(target),'metadata_path':str(sidecar),'metadata':metadata}
