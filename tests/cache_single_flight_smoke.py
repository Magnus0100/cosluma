"""Two real processes share cache misses; GPU producers use explicit CPU fixtures."""
from __future__ import annotations

from contextlib import ExitStack
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image


def counted(root):
    with (root/'producer-calls.txt').open('a', encoding='utf8') as stream:
        stream.write(str(os.getpid())+'\n')
    time.sleep(.25)


def foreground(root, request):
    import source_foreground_cache as fg
    import reference_stage_worker
    identity = {'source_sha256': fg.file_identity(root/'source.png'),
                'worker_sha256': 'CPU-fixture', 'parameters': fg.PARAMETERS}
    def producer(python, unused_root, argv, directory, *args, **kwargs):
        counted(root)
        assets = {}
        for key in fg.ASSET_KEYS:
            path = directory/(key+'.npz')
            np.savez(path, values=np.zeros((2, 3), np.float32))
            assets[key] = str(path)
        descriptor = {'schema_version': fg.PARAMETERS['fb_diagnostics'],
            'source_sha256': identity['source_sha256'], 'producer_sha256': identity['worker_sha256'],
            'coordinate_space': 'oriented_source_pixels', 'canvas_size': [3, 2],
            'renderer_consumed': False, 'support_expanded': False,
            'path': assets['fb_diagnostics_path'], 'sha256': fg.file_identity(assets['fb_diagnostics_path']),
            'paired_asset_sha256': {key: fg.file_identity(assets[key]) for key in fg.ASSET_KEYS[:-1]}}
        result = {**assets, 'schema_version': 'source_foreground.v1',
            'native_values_encoding': fg.PARAMETERS['native_values_encoding'],
            'analysis_size': [3, 2], 'fb_diagnostics': descriptor, 'provenance': {}}
        (directory/'result.json').write_text(json.dumps(result), encoding='utf8')
        return {}
    with patch.object(fg, '_identity', return_value=identity), \
            patch.object(fg, 'local_runtime_paths', return_value={'python': 'CPU-fixture', 'vitmatte_model_root': 'unused',
                'registry_path': root/'registry.json'}), \
            patch.object(reference_stage_worker, 'execute_stage', side_effect=producer):
        return fg.prepare_source_foreground(root, root/'source.png', saliency_path=root/'source.png')['cache_hit']


def scene(root, request):
    import local_scene_analysis as local
    paths = {'worker': Path(local.__file__), 'models': root/'model',
             'profile': 'sam31_static_detector_v1', 'registry': root/'registry.json'}
    def producer(unused_root, unused_paths, body):
        counted(root)
        result = {'reflection_matches': [], 'contact': {},
            'background_mapping': {'candidate_to_source_proxy': np.eye(3).tolist(),
                                   'method': 'CPU-fixture', 'supported': True},
            'provenance': {}}
        (Path(body['output_dir'])/'scene.json').write_text(json.dumps(result), encoding='utf8')
    with patch.object(local, 'scene_runtime_paths', return_value=paths), \
            patch.object(local, 'local_scene_ready', return_value=True), \
            patch.object(local, 'scene_model_identity', return_value={'content_sha256': 'CPU-fixture'}), \
            patch.object(local, '_scene_service_request', side_effect=producer):
        return local.prepare_local_scene(root, root/'source.png', root/'source.png')['cache_hit']


def regions(root, request):
    import beautygrpo_prepare as beauty
    record = {'model_input': str(root/'source.png'), 'model_input_sha256': beauty.sha(root/'source.png'),
              'native_size': [12, 10]}
    config = {'face_registry': str(root/'registry.json'), 'regions_python': 'CPU-fixture',
              'regions_model_root': str(root/'model')}
    def producer(argv, **kwargs):
        counted(root)
        output = Path(argv[argv.index('--output')+1])
        values = output/'regions.npz'
        np.savez(values, values=np.zeros((10, 12), np.float32))
        beauty.write_json(output/'result.json', {'schema_version': 'portrait_regions.v1',
            'source_sha256': record['model_input_sha256'], 'source_native_size': [12, 10],
            'worker_sha256': 'CPU-fixture', 'regions': [{'npz_path': str(values)}]})
        return SimpleNamespace(returncode=0)
    output = root/request
    output.mkdir()
    with patch.dict(os.environ, {'COSPLAY_BEAUTY_CACHE_ROOT': str(root/'cache')}), \
            patch('current_model_registry.load_registry', return_value={'face_model_path': str(root/'source.png')}), \
            patch.object(beauty.subprocess, 'run', side_effect=producer):
        result = beauty._run_regions(config, record, root, output/'regions', 8)
        cached = json.loads(result.read_text(encoding='utf8'))
        assert Path(cached['regions'][0]['npz_path']).is_file()
        return cached['cache']['cache_hit']


def generated(root, request):
    import beautygrpo_prepare as beauty
    import beautygrpo_worker as worker
    run = {'source': {'path': str(root/'source.png'), 'sha256': beauty.sha(root/'source.png')}, 'options': {}}
    config = {'base_manifest_sha256': 'CPU-fixture', 'offload': 'sequential_cpu_offload'}
    def binding(record, *args):
        return {'source_sha256': record['source_sha256'], 'model_input_sha256': record['model_input_sha256'],
                'source_display': record['source_display'], 'model_identity': worker.model_identity(config)}
    def parser(config, record, unused_root, output, max_faces):
        output.mkdir()
        beauty.write_json(output/'result.json', {'fixture': 'CPU regions'})
        return output/'result.json'
    def producer(input_path, regions, config, output, **kwargs):
        counted(root)
        output.mkdir()
        pixels = output/'generated.png'
        Image.new('RGB', (8, 8), (210, 60, 100)).save(pixels)
        weights = output/'weight.npy'
        np.save(weights, np.ones((8, 8), np.float32), allow_pickle=False)
        face = {'id': 'face-01', 'native_box': [2, 1, 10, 9],
                'generated_native_roi': str(pixels), 'generated_native_roi_sha256': beauty.sha(pixels),
                'composition_weight': str(weights), 'composition_weight_sha256': beauty.sha(weights),
                'model_output': str(pixels)}
        result = {'inference_completed': True, 'faces': [face],
            'source': {'sha256': beauty.sha(input_path), 'native_size': [12, 10]},
            'regions': {'path': str(regions)}, 'model_identity': worker.model_identity(config),
            'base_verified': {}, 'lora_verified': {}, 'versions': {}}
        beauty.write_json(output/'result.json', result)
        return result
    with ExitStack() as stack:
        stack.enter_context(patch.object(beauty, 'installation_path', return_value=root/'unused.json'))
        stack.enter_context(patch.object(beauty, 'load_installation', return_value=config))
        stack.enter_context(patch.object(beauty, '_cache_binding', side_effect=binding))
        stack.enter_context(patch.object(beauty, '_run_regions', side_effect=parser))
        stack.enter_context(patch.object(worker, 'run', side_effect=producer))
        result = beauty.prepare(run, root, root/request)
    assert Path(result['layer']['faces'][0]['pixels']['path']).is_relative_to(root/request)
    return result['cache_hit']


def contender(kind, root, request, ready, start, results):
    # Imports in spawned processes are intentionally real; only expensive
    # inference/installed-model identities above are controlled fixtures.
    ready.set()
    assert start.wait(20)
    hit = globals()[kind](Path(root), request)
    results.put(hit)


def hold_lock(path, ready):
    from cache_store import exclusive_file_lock
    with exclusive_file_lock(path):
        ready.set()
        time.sleep(30)


def main():
    context = multiprocessing.get_context('spawn')
    with tempfile.TemporaryDirectory(prefix='cache-single-flight-') as temporary:
        root = Path(temporary)
        for kind in ('foreground', 'scene', 'regions', 'generated'):
            folder = root/kind
            folder.mkdir()
            Image.new('RGB', (12, 10), (20, 50, 80)).save(folder/'source.png')
            (folder/'registry.json').write_text('{}', encoding='utf8')
            start, results = context.Event(), context.Queue()
            ready = [context.Event(), context.Event()]
            children = [context.Process(target=contender, args=(kind, str(folder), 'request-'+str(i), ready[i], start, results))
                        for i in range(2)]
            try:
                for child in children: child.start()
                assert all(event.wait(20) for event in ready)
                start.set()
                for child in children:
                    child.join(30)
                    assert child.exitcode == 0, (kind, child.exitcode)
                assert sorted([results.get(timeout=2), results.get(timeout=2)]) == [False, True], kind
                assert len((folder/'producer-calls.txt').read_text().splitlines()) == 1, kind
            finally:
                for child in children:
                    if child.is_alive(): child.terminate(); child.join(5)
                results.close(); results.join_thread()
        # The kernel releases an abandoned lock; its marker file stays stable.
        ready = context.Event()
        lock_path = root/'abandoned.lock'
        child = context.Process(target=hold_lock, args=(lock_path, ready))
        child.start()
        try:
            assert ready.wait(10)
            child.terminate(); child.join(5)
            from cache_store import exclusive_file_lock
            with exclusive_file_lock(lock_path, timeout=2):
                assert lock_path.is_file()
                try:
                    with exclusive_file_lock(lock_path, timeout=.5, cancel=lambda: True):
                        raise AssertionError('A cancelled waiter acquired the occupied lock')
                except InterruptedError:
                    pass
        finally:
            if child.is_alive(): child.terminate(); child.join(5)
    print(json.dumps({'ok': True, 'two_process_single_producer': ['foreground', 'scene', 'regions', 'generated'],
                      'crash_lock_recovery': True, 'gpu_used': False, 'model_inferences': 0,
                      'scope': 'real OS locking and cache validation; CPU inference fixtures'}))


if __name__ == '__main__':
    main()
