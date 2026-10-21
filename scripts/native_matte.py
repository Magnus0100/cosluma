"""Frozen native matte editing with a coherent source-pair worker cache."""

from runtime_resources import external_python_run
from pathlib import Path
import copy,hashlib,json,os,subprocess,time
import numpy as np
from PIL import Image,ImageDraw
from native_matte_dependencies import refresh_matte_dependencies
from asset_identity import sha256_file

SOLVER='native-matte-cf-ml-paired-v2-measured-dependency-invalidation'


def _json(path):return json.loads(Path(path).read_text('utf8'))


def _reuse_solver_cache(directory,identity):
    """Keep completed solver work when only preview orchestration changed."""
    for request_path in directory.parent.glob('*/request.json'):
        previous_directory=request_path.parent
        marker=previous_directory/'result.json'
        if previous_directory==directory or not marker.is_file():continue
        previous=_json(request_path).get('identity',{})
        if previous!=identity:continue
        result=_json(marker)
        for asset in result['assets'].values():
            source=Path(asset['path']);target=directory/source.name
            if not target.exists():os.link(source,target)
            asset['path']=str(target.resolve())
        target=directory/'roi_pair.npz'
        if not target.exists():os.link(previous_directory/'roi_pair.npz',target)
        request=_json(request_path);request.update(directory=str(directory),identity=identity)
        (directory/'request.json').write_text(json.dumps(request,ensure_ascii=False,indent=2),'utf8')
        (directory/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf8')
        return True
    return False


def _editing_plan(parent_plan,body):
    """Replay draft strokes from durable parent assets; copy only UI settings."""
    plan=copy.deepcopy(parent_plan)
    requested=(body.get('appearance_revision') or {}).get('design_reconstruction_plan') or {}
    rows={str(layer.get('id')):layer for layer in requested.get('layers',[])}
    for layer in plan.get('layers',[]):
        setting=rows.get(str(layer.get('id')), {})
        for key in ('enabled','strength','params','blend_mode','label'):
            if key in setting:layer[key]=copy.deepcopy(setting[key])
    if 'enabled' in requested:plan['enabled']=bool(requested['enabled'])
    return plan


def _strokes(body,canvas):
    strokes=copy.deepcopy(body.get('strokes') or [])
    for stroke in strokes:
        if stroke.get('mode') not in {'foreground','background'}:
            raise ValueError('Matte stroke mode must be foreground or background')
        if not stroke.get('points') or any(len(p)!=2 or not all(0<=float(c)<=1 for c in p) for p in stroke['points']):
            raise ValueError('Matte stroke points must be normalized source coordinates')
        if float(stroke.get('radius_px',0))<=0:
            raise ValueError('Matte stroke radius must be positive')
        if list(stroke.get('native_canvas_size') or canvas)!=list(canvas):
            raise ValueError('Matte strokes must use this photo native canvas')
        stroke['native_canvas_size']=list(canvas)
    return strokes


def _prepare(root,run_id,body):
    from single_source_workflow import load_run
    from design_reconstruction import plan_for_run,has_native_foreground,validate_current_native_plan,render_mode_for_run
    from local_photo_analysis import local_runtime_paths
    root=Path(root).resolve();manifest,run=load_run(root,run_id);parent=plan_for_run(run)
    render_mode_for_run(run);validate_current_native_plan(parent)
    canvas=tuple(parent['canvas']['source_oriented_size']);strokes=_strokes(body,canvas);plan=_editing_plan(parent,body)
    if not strokes:return root,manifest,run,plan,None,None
    from runtime_resources import code_path
    identity={'solver':SOLVER,'parent_run_id':run_id,'parent_manifest_sha256':sha256_file(manifest),
              'source_sha256':run['source']['sha256'],'strokes':strokes,'source_canvas':list(canvas),
              'code':{name:sha256_file(code_path(name)) for name in ('native_matte_worker.py','local_colour_brush.py')},
              'pymatting':'1.1.16 closed-form alpha + ML F/B; linear source; finite annotation support'}
    key=hashlib.sha256(json.dumps(identity,sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:24]
    directory=root/'outputs/cache/native-matte'/run['source']['sha256'][:16]/key;directory.mkdir(parents=True,exist_ok=True)
    result_path=directory/'result.json';cache_hit=result_path.is_file() or _reuse_solver_cache(directory,identity)
    if not cache_hit:
        request={'root':str(root),'directory':str(directory),'parent_plan':parent,'strokes':strokes,'identity':identity}
        request_path=directory/'request.json';request_path.write_text(json.dumps(request,ensure_ascii=False,indent=2),'utf8')
        runtime=local_runtime_paths(root)['python']
        with (directory/'worker.log').open('w',encoding='utf8') as log:
            result=external_python_run([str(runtime),'-X','utf8',str(code_path('native_matte_worker.py')),'--request',str(request_path)],cwd=root,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('Native matte solver failed; details: '+str(directory/'worker.log'))
    result=_json(result_path);result['cache_hit']=cache_hit;result['cache_key']=key;result['cache_directory']=str(directory)
    bg=next(layer for layer in plan['layers'] if layer['kind']=='background_design')['assets']
    original=parent['source_baseline']['assets']
    plan.setdefault('source_baseline',{})['assets']=copy.deepcopy(original)
    new_assets=result['assets']
    bg['foreground_premultiplied']=copy.deepcopy(new_assets['foreground_premultiplied']);bg['old_environment']=copy.deepcopy(new_assets['old_environment'])
    for key_name in ('foreground_alpha','edge_applicability','edited_edge_light_applicability','matte_weak_added_material'):
        plan['mask_roles'][key_name]=copy.deepcopy(new_assets[key_name])
    plan['mask_roles']['edited_edge_light_applicability'].update(representation='retained_original_pair_coverage',role='automatic_appearance_support_only',not_alpha=True,not_confidence=True)
    plan['mask_roles']['matte_weak_added_material'].update(representation='weak_added_material_observation',not_confidence=True)
    plan['native_matte_revision']={'schema_version':'native_matte_revision.v1','cache_key':result['cache_key'],'solver':SOLVER,'source_sha256':run['source']['sha256'],
        'parent_run_id':run_id,'crop_box':result['crop_box'],'strokes':strokes,'requires_edge_light_refit':False,
        'material_support_kind':'local_linear_FB_fit_not_confidence','illumination_support_kind':'retained_original_pair_coverage',
        'prior_revision':copy.deepcopy(parent.get('native_matte_revision'))}
    plan['native_matte_revision']['dependency_refresh']=refresh_matte_dependencies(parent,plan,result,directory,run)
    return root,manifest,run,plan,result,directory


def _native_comparison(directory,plan,result):
    """Native pair/coverage inspection. Shared whole preview shows manual edits."""
    from raw_support import linear_preview
    from design_reconstruction import sample_asset
    with np.load(directory/'roi_pair.npz') as data:d={key:data[key] for key in data.files}
    box=tuple(result['crop_box']);size=tuple(result['native_size']);assets=next(layer for layer in plan['layers'] if layer['kind']=='background_design')['assets']
    target=sample_asset(assets['background'],'values',box,size);validity=sample_asset(assets['validity'],'values',box,size) if 'validity' in assets else np.ones(d['alpha_before'].shape,np.float32)
    bg_layer=next(layer for layer in plan['layers'] if layer['kind']=='background_design');strength=float(np.clip(bg_layer.get('strength',1),0,1)) if plan.get('enabled') and bg_layer.get('enabled') else 0
    target_weight=strength
    old_background=d['background_before']*(1-strength*validity[...,None])+target_weight*target
    new_background=d['background_after']*(1-strength*validity[...,None])+target_weight*target
    old=d['fp_before']+(1-d['alpha_before'][...,None])*old_background
    after=d['fp_after']+(1-d['alpha_after'][...,None])*new_background
    pictures=[('DURABLE PARENT PAIR',old),('EDITED MATTE / PAIRED RGB',after)]
    h,w=old.shape[:2];sheet=Image.new('RGB',(w*2,h+28),'#191b24');draw=ImageDraw.Draw(sheet)
    for index,(name,values) in enumerate(pictures):sheet.paste(linear_preview(values),(index*w,28));draw.text((index*w+8,7),name,fill='white')
    path=directory/'native-pair-comparison.png';sheet.save(path)
    return str(path.resolve())


def preview_matte(root,run_id,body,*,preview_cache=None):
    from source_preview import SourcePreviewCache
    started=time.perf_counter();root,manifest,run,plan,result,directory=_prepare(root,run_id,body)
    edited=copy.deepcopy(body);edited.setdefault('appearance_revision',{})['design_reconstruction_plan']=plan
    preview=(preview_cache or SourcePreviewCache(root)).preview(run_id,run,manifest,edited)
    native=_native_comparison(directory,plan,result) if result else None
    return {'ok':True,'run_id':run_id,'request_id':body.get('request_id'),'cache_key':result['cache_key'] if result else None,
        'cache_hit':result['cache_hit'] if result else True,'preview_path':preview['preview_path'],
        'preview_size':preview['preview_size'],'source_size':preview['source_size'],'native_preview_path':native,
        'crop_box':result['crop_box'] if result else None,'native_preview_scope':'paired matte/source RGB with design background; manual edits included in whole preview',
        'patched_plan':plan,'raw_decoded':False,'gpt_calls':0,'source_reference':'paired_native_baseline',
        'applicability':{'new_edge_light_refit_required':bool(result),'added_without_material_support_present':result['added_without_material_support_present'] if result else False,
            'illumination_support_kind':'unchanged measured source pair only; changed coverage retains source light',
            'dependency_refresh':plan.get('native_matte_revision',{}).get('dependency_refresh'),
            'message':'Changed source pairs invalidate old automatic light support. Newly added coverage keeps source light until a new source-bound measurement is prepared.'},
        'elapsed_seconds':round(time.perf_counter()-started,3)}


def adjust_matte(root,run_id,body):
    from single_source_workflow import adjust_source_run
    root,manifest,run,plan,result,directory=_prepare(root,run_id,body)
    revision=copy.deepcopy(body.get('appearance_revision') or {});revision['design_reconstruction_plan']=plan
    fields=('manual_layers','denoise_strength')
    kwargs={key:copy.deepcopy(body[key]) for key in fields if key in body}
    kwargs['appearance_revision']=revision
    child=adjust_source_run(root,run_id,**kwargs)
    return {'ok':True,'run':child,'cache_key':result['cache_key'] if result else None,'matte_solver_cache_hit':result['cache_hit'] if result else True,
            'source_reference':'paired_native_baseline','raw_decoded':False,'gpt_calls':0}
