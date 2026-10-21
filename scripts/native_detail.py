"""Read a saved working master's native crop without decoding its canvas."""
from pathlib import Path
import hashlib
import json
import time

import numpy as np


def read_scene_crop(path, box):
    import tifffile
    left, top, right, bottom = box
    with tifffile.TiffFile(path) as file:
        page = file.pages[0]
        metadata = (file.shaped_metadata or ({},))[0] or {}
        if metadata.get('working_domain')!='display_linear_sRGB':
            raise ValueError('Native detail accepts only the current display-linear master')
        height, width, channels = page.shape
        if page.is_tiled or page.planarconfig != 1 or channels != 3 or page.dtype != np.float32:
            raise ValueError('Native detail requires the saved RGB float32 strip working master')
        result = np.empty((bottom-top, right-left, 3), np.float32)
        first, last = top//page.rowsperstrip, (bottom-1)//page.rowsperstrip
        for index in range(first, last+1):
            file.filehandle.seek(page.dataoffsets[index])
            data = file.filehandle.read(page.databytecounts[index])
            decoded, location, shape = page.decode(data, index)
            row = location[2]
            y0, y1 = max(top, row), min(bottom, row+shape[1])
            result[y0-top:y1-top] = decoded[0, y0-row:y1-row, left:right, :]
        return result, {'decoded_strips':last-first+1, 'total_strips':len(page.dataoffsets),
                        'native_size':[width,height], 'working_precision':'float32_linear',
                        'working_domain':metadata.get('working_domain'),
                        'full_canvas_decoded':False}


def saved_native_detail(root, run_id, body):
    from single_source_workflow import load_run
    from display_transform import plan_for_run
    from raw_support import linear_preview
    import tifffile
    started = time.perf_counter()
    root = Path(root)
    _, run = load_run(root, run_id)
    value = (run.get('artifacts') or {}).get('render_scene_master')
    if not value:
        raise ValueError('Save a high-resolution version before inspecting native details')
    path = Path(value)
    if not path.is_absolute():
        path = root/path
    if (run.get('options') or {}).get('photo_appearance'):
        from asset_identity import sha256_file
        expected=((run.get('render') or {}).get('saved_artifact_sha256') or {}).get('render_scene_master')
        if not expected or sha256_file(path)!=expected:
            raise ValueError('Saved native master changed after the selected appearance was rendered')
    with tifffile.TiffFile(path) as file:
        height,width = file.pages[0].shape[:2]
    center = body.get('center', [.5,.5])
    edge = min(2048, max(128, int(body.get('edge', 768))))
    cw,ch = min(edge,width),min(edge,height)
    left = max(0,min(width-cw,round(float(center[0])*width-cw/2)))
    top = max(0,min(height-ch,round(float(center[1])*height-ch/2)))
    box = (left,top,left+cw,top+ch)
    if (run.get('options') or {}).get('photo_appearance'):
        from generated_appearance_render import saved_display_transform
        display=saved_display_transform(root,run)
    else:
        display=plan_for_run(run,root=root)
    if ((run.get('render') or {}).get('operator') or {}).get('working_domain')!='display_linear_sRGB':
        raise ValueError('Saved version does not declare a current display-linear render')
    working_domain='display_linear_sRGB'
    identity = {'scene':str(path),'mtime':path.stat().st_mtime_ns,'size':path.stat().st_size,
                'box':box,'display':display}
    key = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
    directory = root/'outputs/cache/native-detail'/run_id
    directory.mkdir(parents=True,exist_ok=True)
    target = directory/(key+'.png')
    values, precision = read_scene_crop(path,box)
    if precision['working_domain']!=working_domain:
        raise ValueError('Saved master and render domains differ')
    image = linear_preview(values,display_transform=display)
    image.save(target,icc_profile=image.info['icc_profile'])
    result = {'ok':True,'run_id':run_id,'revision':run['revision'],'request_id':body.get('request_id'),
            'preview_path':str(target.resolve()),'crop_box':list(box),'crop_size':[cw,ch],
            **precision,'working_domain':working_domain,'display_transform':display,'saved_version_only':True,
            'unsaved_edits_included':False,'resampled':False,'raw_decoded':False,'gpt_calls':0,
            'elapsed_seconds':round(time.perf_counter()-started,3)}
    if body.get('include_source') is True:
        if (run.get('options') or {}).get('photo_appearance'):
            from appearance_assets import load_asset
            scope=(run.get('task_plan') or {})['editing_scope']
            design=run.get('generation_design') or (run.get('options') or {}).get('generation_design') or {}
            source_asset=run.get('initial_source_appearance')
            baseline=load_asset(source_asset,source_sha256=run['source']['sha256'],scope=scope,
                design_id=design['observed_design']['design_id'])
            try:
                if (baseline['receipt']['kind']!='source_baseline'
                        or baseline['receipt']['canvas_size']!=[width,height]):
                    raise ValueError('Saved original appearance is not the matching native source baseline')
                source=np.array(baseline['master'][top:top+ch,left:left+cw],copy=True)
            finally:
                for array in baseline.values():
                    if isinstance(array,np.memmap) and getattr(array,'_mmap',None) is not None:
                        array._mmap.close()
            source_key=hashlib.sha256(json.dumps({'box':box,'asset_id':source_asset['asset_id'],
                'display':display},sort_keys=True).encode()).hexdigest()[:20]
            source_target=directory/(source_key+'-source.png')
            source_image=linear_preview(source,display_transform=display)
            source_image.save(source_target,icc_profile=source_image.info['icc_profile'])
            result.update(source_preview_path=str(source_target.resolve()),
                source_comparison={'basis':'saved_source_native_float_baseline','asset_id':source_asset['asset_id'],
                    'crop_box':list(box),'native_size':[width,height],'resampled':False,'raw_decoded':False},
                elapsed_seconds=round(time.perf_counter()-started,3))
            return result
        plan = (((run.get('options') or {}).get('appearance_revision') or {})
                .get('design_reconstruction_plan') or {})
        baseline = plan.get('source_baseline') or {}
        assets = baseline.get('assets') or {}
        if (baseline.get('schema') == 'appearance_source_baseline.v1'
                and baseline.get('working_domain') == 'display_linear_sRGB'
                and all(name in assets for name in ('foreground_premultiplied', 'foreground_alpha', 'old_environment'))):
            from design_reconstruction import sample_asset
            from asset_identity import sha256_file
            for asset in assets.values():
                if isinstance(asset, dict) and asset.get('path') and asset.get('sha256'):
                    source_path = Path(asset['path'])
                    source_path = source_path if source_path.is_absolute() else root/source_path
                    if sha256_file(source_path) != asset['sha256']:
                        raise ValueError('Saved source comparison baseline changed')
            source = sample_asset(assets['foreground_premultiplied'], 'values', box, (width,height))
            alpha = sample_asset(assets['foreground_alpha'], 'values', box, (width,height))
            background = sample_asset(assets['old_environment'], 'values', box, (width,height))
            source = source + (1-alpha[...,None])*background
            source_key = hashlib.sha256(json.dumps({'box':box, 'assets':assets,
                'display':display},sort_keys=True).encode()).hexdigest()[:20]
            source_target = directory/(source_key+'-source.png')
            source_image = linear_preview(source, display_transform=display)
            source_image.save(source_target, icc_profile=source_image.info['icc_profile'])
            result.update(source_preview_path=str(source_target.resolve()),
                source_comparison={'basis':'saved_pre_beauty_native_baseline',
                    'crop_box':list(box),'native_size':[width,height],
                    'resampled':False,'raw_decoded':False,
                    'native_source_retouch_included':bool(baseline.get('native_source_retouch'))})
        else:
            result['source_comparison'] = {'unavailable_reason':
                '当前保存版本没有可核验的美化前原生配对基线。'}
        result['elapsed_seconds'] = round(time.perf_counter()-started,3)
    return result
