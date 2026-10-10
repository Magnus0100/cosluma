"""CPU contracts for current profile, tensor representation and route isolation."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
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
                      'scope': 'CPU profile, shape, representation and explicit routing contracts'}))


if __name__ == '__main__':
    main()
