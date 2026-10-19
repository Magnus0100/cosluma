"""CPU contracts for current profile, tensor representation and route isolation."""
from __future__ import annotations

import json
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from unittest.mock import patch

import numpy as np

import local_scene_analysis as scene
import sam31_model_profile as profiles
from sam3_scene_worker import instances


class CpuTensor:
    def __init__(self, array):
        self.array = np.asarray(array)
    def detach(self): return self
    def cpu(self): return self
    def float(self): return CpuTensor(self.array.astype(np.float32))
    def numpy(self): return self.array


def rejected(call):
    try:
        call()
    except (ValueError, RuntimeError, OSError):
        return
    raise AssertionError('Invalid SAM3 candidate contract was accepted')


def output(masks, boxes, scores):
    return {key: CpuTensor(value) for key, value in [('masks', masks), ('boxes', boxes), ('scores', scores)]}


def service_startup_contracts(root):
    paths = {'python': Path(sys.executable), 'worker': root/'worker.py',
             'models': root/'models', 'profile': profiles.IMAGE_PROFILE}
    poison = {'PYTHONPATH': str(root/'foreign-packages'), 'PYTHONHOME': 'foreign-home',
              'PYTHONUSERBASE': 'foreign-user'}
    with patch.dict(os.environ, poison):
        with patch.object(scene.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)) as run:
            scene._run_scene_worker_once(['python', 'worker.py'], root/'fallback.log')
        environment = run.call_args.kwargs['env']
        assert not set(poison) & set(environment)
        assert environment['PYTHONNOUSERSITE'] == '1'
        assert environment['HF_HUB_OFFLINE'] == '1'
        with patch.object(scene.subprocess, 'Popen', side_effect=OSError('cannot launch selected runtime')):
            try:
                scene._start_scene_service(root, paths)
            except scene._SceneServiceUnavailable as error:
                assert isinstance(error.__cause__, OSError)
                assert 'cannot launch selected runtime' in str(error)
            else:
                raise AssertionError('Launch failure was ignored')

    class StalledProcess:
        def __init__(self):
            self.done = threading.Event()
            self.stdout = self
            self.killed = False
        def readline(self):
            self.done.wait(5)
            return ''
        def poll(self): return -9 if self.killed else None
        def kill(self):
            self.killed = True
            self.done.set()
        def wait(self, timeout=None): return -9
    child = StalledProcess()
    started = time.monotonic()
    with patch.object(scene.subprocess, 'Popen', return_value=child):
        try:
            scene._start_scene_service(root, paths, timeout_seconds=.02)
        except scene._SceneServiceUnavailable as error:
            assert isinstance(error.__cause__, TimeoutError)
        else:
            raise AssertionError('A stalled ready pipe was accepted')
    assert child.killed and time.monotonic()-started < 2
    assert scene._SCENE_SERVICE is None
    ready = type('ReadyProcess', (), {'stdout': io.StringIO('{"ready": true}\n')})()
    assert json.loads(scene._read_scene_ready(ready, 1))['ready']


def main():
    assert profiles.specification()['hair_alpha'] is False
    assert profiles.specification()['profile'] == 'sam31_static_detector_v1'
    assert profiles.specification()['detector_tensor_count'] == 1166
    rejected(lambda: profiles.specification('sam3_static_image_v1'))
    rejected(lambda: profiles.specification('sam3_1_multiplex_video_candidate_v1'))
    rejected(lambda: profiles.specification('latest_unknown'))
    returned = profiles.specification()
    returned['preprocess']['resolution'] = 16
    assert profiles.specification()['preprocess']['resolution'] == 1008
    token = object()
    state = profiles.image_state({'detector.a': token, 'tracker.x': object()}, {'a'})
    assert state == {'a': token}
    rejected(lambda: profiles.image_state({'tracker.a': token}, {'a'}))
    rejected(lambda: profiles.image_state({'detector.a': token, 'detector.extra': token}, {'a'}))
    masks = np.ones((2, 1, 3, 5), bool)
    found = instances(output(masks, [[-2, 0, 4, 3], [1, 0, 5, 3]], [.6, .8]), size=(5, 3), concept='bag')
    assert len(found) == 2 and found[0][0]['score'] > found[1][0]['score']
    assert found[1][0]['box'] == [0., 0., 4., 3.] and found[0][1].shape == (3, 5)
    rejected(lambda: instances(output(masks, [[0,0,5,3]]*2, [.6,.8]), size=(3,5), concept='bag'))
    rejected(lambda: instances(output(masks.astype(np.float32), [[0,0,5,3]]*2, [.6,.8]), size=(5,3), concept='bag'))
    rejected(lambda: instances(output(masks, [[0,0,5,3]]*2, [.6,np.nan]), size=(5,3), concept='bag'))
    rejected(lambda: instances(output(masks, [[0,0,5,3]]*2, [.6,1.2]), size=(5,3), concept='bag'))
    assert instances(output(masks, [[2,0,1,3]]*2, [.6,.8]), size=(5,3), concept='bag') == []
    assert instances(output(np.zeros_like(masks), [[0,0,5,3]]*2, [.6,.8]), size=(5,3), concept='bag') == []
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        service_startup_contracts(root)
        from current_model_registry import load_registry
        config = {key: str(root/key) for key in ('python', 'scene_python', 'semantics_checkpoint',
            'semantics_repo_dir', 'vitmatte_model_root', 'dinov3_model_root', 'scene_model_root', 'birefnet_model_path', 'lama_checkpoint', 'background_sr_checkpoint', 'face_model_path', 'face_python', 'onnx_runtime_root')}
        config.update(schema='cosplay_local_models.current.v1', ready=True,
            semantics_profile='sapiens2_seg_0.4b', matting_profile='vitmatte_small_distinctions',
            dinov3_profile='dinov3_vitl16', scene_profile=profiles.IMAGE_PROFILE)
        registry = root/'current.json'
        registry.write_text(json.dumps(config))
        paths = scene.scene_runtime_paths(root, registry_path=registry)
        assert paths['profile'] == profiles.IMAGE_PROFILE and paths['worker'].name == 'sam3_scene_worker.py'
        rejected(lambda: scene.scene_runtime_paths(root, registry_path=root/'missing.json'))
        assert scene.local_scene_ready(root, registry_path=registry) is False
        for rejected_profile in ('grounding_dino_sam2_legacy', 'sam3_image_official_candidate_v1',
                                  'sam3_1_multiplex_video_candidate_v1', 'sam3_static_image_v1'):
            registry.write_text(json.dumps({**config, 'scene_profile': rejected_profile}))
            rejected(lambda: scene.scene_runtime_paths(root, registry_path=registry))
        registry.write_text(json.dumps(config))
        with patch.dict('os.environ', {'COSPLAY_MODEL_REGISTRY': str(registry)}):
            assert scene.scene_runtime_paths(root)['profile'] == profiles.IMAGE_PROFILE
        assert load_registry(root, registry_path=registry)['dinov3_profile'] == 'dinov3_vitl16'
    print(json.dumps({'ok': True, 'checks': 27, 'gpu_used': False, 'models_inferred': 0,
                      'worker_environment_isolated': True, 'startup_timeout_bounded': True,
                      'launch_error_preserved': True,
                      'scope': 'CPU profile, shape, representation and explicit routing contracts'}))


if __name__ == '__main__':
    main()
