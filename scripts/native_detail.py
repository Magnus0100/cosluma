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
    with tifffile.TiffFile(path) as file:
        height,width = file.pages[0].shape[:2]
    center = body.get('center', [.5,.5])
    edge = min(2048, max(128, int(body.get('edge', 768))))
    cw,ch = min(edge,width),min(edge,height)
    left = max(0,min(width-cw,round(float(center[0])*width-cw/2)))
    top = max(0,min(height-ch,round(float(center[1])*height-ch/2)))
    box = (left,top,left+cw,top+ch)
    display = plan_for_run(run)
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
    return {'ok':True,'run_id':run_id,'revision':run['revision'],'request_id':body.get('request_id'),
            'preview_path':str(target.resolve()),'crop_box':list(box),'crop_size':[cw,ch],
            **precision,'working_domain':working_domain,'display_transform':display,'saved_version_only':True,
            'unsaved_edits_included':False,'resampled':False,'raw_decoded':False,'gpt_calls':0,
            'elapsed_seconds':round(time.perf_counter()-started,3)}
