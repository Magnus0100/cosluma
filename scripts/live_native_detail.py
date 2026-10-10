"""Current native detail: bounded pixel computation, explicit nonlocal fallback."""
from pathlib import Path
from runtime_resources import code_path
import hashlib
import json
import threading
import time

from PIL import Image

_render_lock = threading.Lock()


def live_native_detail(root, run_id, body):
    from single_source_workflow import load_run
    from source_preview import _edited_run, _renderer_identity, _prepare_receiver_edit
    from design_reconstruction import plan_for_run, has_native_foreground
    from native_source_baseline import source_for_plan, enabled
    from float_appearance import render_source_float
    from display_transform import plan_for_run as display_plan_for_run
    from raw_support import linear_preview
    from native_roi_render import render_native_roi,ROIUnavailable

    started=time.perf_counter();root=Path(root).resolve()
    manifest,run=load_run(root,run_id)
    edited=_prepare_receiver_edit(root,_edited_run(run,body));plan=plan_for_run(edited)
    if not has_native_foreground(plan) or not enabled(plan):
        raise ValueError('Live native detail requires an immutable paired native project')
    width,height=map(int,plan['canvas']['source_oriented_size'])
    center=body.get('center',[.5,.5]);edge=min(2048,max(128,int(body.get('edge',768))))
    cw,ch=min(edge,width),min(edge,height)
    left=max(0,min(width-cw,round(float(center[0])*width-cw/2)))
    top=max(0,min(height-ch,round(float(center[1])*height-ch/2)))
    box=(left,top,left+cw,top+ch);display=display_plan_for_run(edited)
    identity={'run_id':run_id,'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
              'options':edited['options'],'crop_box':box,'display':display,
              'renderer':_renderer_identity(),
              'live_renderer':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'roi_renderer':hashlib.sha256(code_path('native_roi_render.py').read_bytes()).hexdigest()}
    key=hashlib.sha256(json.dumps(identity,sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:20]
    directory=root/'outputs/cache/live-native-detail'/run_id;directory.mkdir(parents=True,exist_ok=True)
    target=directory/(key+'.png');receipt_path=directory/(key+'.json');hit=False;execution=None
    with _render_lock:
        hit=target.is_file() and receipt_path.is_file()
        if hit:
            stored=json.loads(receipt_path.read_text(encoding='utf-8'))
            execution=stored['execution'];strategy=stored['strategy'];fallback=stored.get('fallback')
        if not hit:
            fallback=None
            try:
                crop,execution=render_native_roi(edited,box)
                strategy='source_coordinate_roi_with_native_filter_halo'
            except ROIUnavailable as exc:
                fallback=str(exc);source=source_for_plan(plan,(width,height))
                scene,operator=render_source_float(source,edited)
                crop=scene[top:top+ch,left:left+cw].copy()
                execution=operator.get('design_reconstruction_plan')
                strategy='production_full_native_in_memory_then_crop'
                del scene,source
            image=linear_preview(crop,display_transform=display)
            image.save(target,icc_profile=image.info.get('icc_profile'))
            receipt_path.write_text(json.dumps({'execution':execution,'strategy':strategy,'fallback':fallback},
                ensure_ascii=False,indent=2),encoding='utf-8')
    bounded=strategy=='source_coordinate_roi_with_native_filter_halo'
    return {'ok':True,'run_id':run_id,'revision':run['revision'],'request_id':body.get('request_id'),
            'preview_path':str(target.resolve()),'crop_box':list(box),'crop_size':[cw,ch],
            'native_size':[width,height],'source_size':[width,height],
            'working_precision':'float32_linear','source_reference':'paired_native_baseline',
            'working_domain':'display_linear_sRGB',
            'saved_version_only':False,'unsaved_edits_included':True,'resampled':False,
            'raw_decoded':False,'gpt_calls':0,'whole_native_save':False,'no_fullsave':True,
            'full_canvas_rendered':not hit and not bounded,'bounded_roi_renderer':bounded,
            'render_scope':'source_native_roi_plus_halo' if bounded else 'full_native_canvas_then_source_coordinate_crop',
            'render_strategy':strategy,'roi_fallback_reason':fallback,
            'donor_and_halo_policy':'native source-coordinate finite halo' if bounded else 'full native donor/context domain',
            'asset_io_policy':'NPZ assets may decompress full planes; bounded refers to pixel computation and RGB buffers',
            'native_matte_revision':plan.get('native_matte_revision'),
            'display_transform':display,'design_execution':execution,'cache_hit':hit,
            'elapsed_seconds':round(time.perf_counter()-started,3)}
