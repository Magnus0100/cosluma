"""Current parser registry and real source F/B reconstruction contracts."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
from current_model_registry import load_registry, PROFILES, PATH_KEYS, SCHEMA
from local_photo_analysis import local_runtime_paths, analysis_identity, local_models_ready
from local_semantics_worker import CLASS_NAMES
from sapiens_model_profiles import profile, verified_checkpoint
from source_foreground import estimate_source_background
from source_foreground_cache import _identity


def rejected(call):
    try: call()
    except (ValueError, RuntimeError, OSError): return
    raise AssertionError('Removed or invalid model contract accepted')


def main():
    repo = Path(__file__).resolve().parents[1]/'.cache/local-models/sapiens2/repo'
    specification = profile('sapiens2_seg_0.4b', 'segmentation')
    config = repo/specification['config']
    assignments = {node.targets[0].id: node.value for node in ast.parse(config.read_text(encoding='utf8')).body
        if isinstance(node, ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0], ast.Name)}
    assert ast.literal_eval(assignments['num_classes']) == len(CLASS_NAMES) == 29
    for old in ('sapiens2_seg_1b', 'sapiens2_matting_1b'):
        rejected(lambda: profile(old, 'segmentation'))
    rejected(lambda: profile('sapiens2_seg_0.4b', 'matting'))
    with tempfile.TemporaryDirectory(prefix='cosplay_current_models_') as temporary:
        root = Path(temporary)
        current = {key: str(root/key) for key in PATH_KEYS}
        current.update(PROFILES, schema=SCHEMA, ready=True, semantics_repo_dir=str(repo))
        registry = root/'current.json'; registry.write_text(json.dumps(current))
        paths = local_runtime_paths(root, registry_path=registry)
        assert paths['semantics_profile']=='sapiens2_seg_0.4b'
        assert paths['matting_profile']=='vitmatte_small_distinctions'
        assert not local_models_ready(root, registry_path=registry)
        rejected(lambda: local_runtime_paths(root, registry_path=root/'missing.json'))
        with patch.dict('os.environ', {'COSPLAY_MODEL_REGISTRY': str(registry)}):
            assert local_runtime_paths(root)['registry_path']==registry
        for key in PROFILES:
            registry.write_text(json.dumps({**current, key:'old-candidate'}))
            rejected(lambda: load_registry(root, registry_path=registry))
        registry.write_text(json.dumps(current))
        corrupt=root/'corrupt.safetensors';corrupt.write_bytes(b'not-official-checkpoint')
        rejected(lambda: verified_checkpoint(corrupt,'sapiens2_seg_0.4b','segmentation'))
        def identity(path):
            return specification['checkpoint_sha256'] if Path(path)==paths['checkpoint'] else 'bound-'+str(path)
        with patch('local_photo_analysis.file_identity', side_effect=identity):
            observed = analysis_identity(root, registry_path=registry)
        assert observed['model_profile']=='sapiens2_seg_0.4b'
        with patch('source_foreground_cache.file_identity', side_effect=lambda path:'bound-'+str(path)):
            matte = _identity(root, root/'source', root/'saliency', None, paths)
        assert matte['matting_profile']=='vitmatte_small_distinctions'
        bad={**paths,'matting_profile':'sapiens2_matting_1b'}
        with patch('source_foreground_cache.file_identity', side_effect=lambda path:'bound-'+str(path)):
            rejected(lambda: _identity(root,root/'source',root/'saliency',None,bad))
    alpha=np.zeros((24,32),np.float32);alpha[5:19,6:26]=1
    alpha[4,6:26]=.35;alpha[19,6:26]=.65;alpha[6:18,5]=.25;alpha[6:18,26]=.75
    trimap=np.full(alpha.shape,128,np.uint8);trimap[alpha==0]=0;trimap[alpha==1]=255
    # Exercise the real CPU F/B estimator with high-frequency original detail.
    yy,xx=np.mgrid[:24,:32]
    detail=(.4+.1*((xx+yy)%2))[...,None]*np.array([.8,1.,.7])
    bg=np.broadcast_to(np.array([.11,.16,.19]),detail.shape)
    source=(detail*alpha[...,None]+bg*(1-alpha[...,None])).astype(np.float32)
    original_source=source.copy();original_alpha=alpha.copy()
    old,fp,support,tiles,scale,diagnostics=estimate_source_background(source,alpha,trimap,tile=32,overlap=8,halo=4)
    assert tiles>0 and np.isfinite(fp).all() and np.isfinite(old).all()
    np.testing.assert_allclose(fp+(1-alpha[...,None])*old,source,atol=2e-7,rtol=0)
    np.testing.assert_array_equal(fp[alpha==1],source[alpha==1])
    assert not fp[alpha==0].any()
    np.testing.assert_array_equal(source,original_source);np.testing.assert_array_equal(alpha,original_alpha)
    assert diagnostics['canvas_size'].tolist()==[32,24]
    print('OK: current registry/profile only; pinned 29-class parser; wrong task and weight refusal; real CPU source F/B reconstruction and opaque high-frequency detail preserved; no GPU inference')


if __name__=='__main__':main()
