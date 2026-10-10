"""Export real saved images independently through batch HTTP; preserve parents."""
import json
from pathlib import Path
import tempfile
import threading
import urllib.error
import urllib.request

import numpy as np
from PIL import Image

from backend_server import Backend, LocalHTTPServer, make_handler
from float_appearance import srgb_to_linear
from native_source_baseline import prepare_source_only
from raw_support import RGB_DECODE_VERSION, linear_preview, save_render_master, save_scene_master
from single_source_workflow import write_json


def saved(root, identifier):
    directory = root / 'outputs/runs' / identifier
    directory.mkdir(parents=True)
    source = directory / 'source.png'
    Image.new('RGB', (18, 24), '#657585').save(source)
    import hashlib
    run = {'kind': 'single_input_run', 'input_mode': 'single_source', 'run_id': identifier,
        'sample': identifier, 'revision': 1, 'status': 'needs_visual_review',
        'source': {'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                   'format': 'PNG', 'icc_present': False},
        'options': {'render_mode': 'native_source_design', 'preserve_project_selection': True},
        'render': {'run_id': identifier, 'revision': 1}}
    with Image.open(source) as image:
        values = srgb_to_linear(np.asarray(image, np.float32) / 255)
    pixels, plan = prepare_source_only(values, run,
        {'decode_version': RGB_DECODE_VERSION, 'source_precision': 'raster8_srgb'})
    run['options']['source_only_plan'] = plan
    output = directory / 'auto_draft.png'
    preview = linear_preview(pixels, display_transform={'method': 'clip_sdr'})
    preview.save(output, icc_profile=preview.info['icc_profile'])
    master, scene = directory / 'render_master.tif', directory / 'render_scene.tif'
    save_render_master(pixels, master, display_transform={'method': 'clip_sdr'})
    save_scene_master(pixels, scene, working_domain='display_linear_sRGB')
    run['artifacts'] = {'auto_draft': str(output), 'render_master': str(master), 'render_scene_master': str(scene)}
    run['render'].update(operator={'working_domain': 'display_linear_sRGB'}, render_precision={
        'working_domain': 'display_linear_sRGB', 'working_precision': 'float32_linear', 'master_encoding': 'uint16_srgb'})
    write_json(directory / 'manifest.json', run)
    return directory / 'manifest.json'


def main():
    with tempfile.TemporaryDirectory(prefix='cosplay-batch-export-') as folder:
        root = Path(folder).resolve()
        pointer = root / 'project.json'
        pointer.write_text('{"selected":"preserved"}')
        protected = [pointer, saved(root, 'first'), saved(root, 'last')]
        before = {p: p.read_bytes() for p in protected}
        backend = Backend(root)
        server = LocalHTTPServer(('127.0.0.1', 0), make_handler(backend))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

        def post(path, body):
            request = urllib.request.Request(f'http://127.0.0.1:{server.server_port}'+path,
                data=json.dumps(body).encode('utf8'), headers={'Content-Type': 'application/json'})
            with opener.open(request, timeout=5) as response:
                return json.load(response)

        try:
            result = post('/api/source-runs/export-batch', {'items': [{'run_id': 'first'}, None,
                {'run_id': 'missing'}, {'run_id': 'last'}], 'directory': str(root / 'exports'), 'format': 'PNG'})
            assert result['batch']['summary'] == {'succeeded': 2, 'failed': 2}
            assert Path(result['receipt']).is_file() and len(list((root / 'exports').glob('*.png'))) == 2
            for row in result['batch']['items']:
                if row['status'] == 'succeeded':
                    assert not row['result']['metadata']['approved']
                    assert row['result']['metadata']['export_rerendered'] is False
            assert all(p.read_bytes() == data for p, data in before.items())
            for route in ('/api/batch', '/api/batch/local-reference', '/api/batch/retired/retry-local-item'):
                try:
                    post(route, {})
                except urllib.error.HTTPError as error:
                    assert error.code == 404
                else:
                    raise AssertionError('Retired unplanned batch execution route remains')
        finally:
            server.shutdown()
            server.server_close()
            thread.join(5)
    print(json.dumps({'ok': True, 'real_saved_exports': 2, 'partial_failure_isolated': True,
        'parents_and_selection_preserved': True, 'retired_batch_routes_removed': True}))


if __name__ == '__main__':
    main()
