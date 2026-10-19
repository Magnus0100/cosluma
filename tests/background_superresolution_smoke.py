"""CUDA regression: tiled inference agrees with full inference and preserves support."""
from pathlib import Path
import json
import sys
import argparse
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from background_superresolution_worker import (SRVGGNetCompact, WEIGHT_SHA256,
    digest, enhance_tiled, maps, sample, _native_carriers)
from current_model_registry import load_registry


def cpu_smoke():
    # The deterministic stand-in exercises sampling, support and fallback only;
    # no weight files, model kernels or CUDA context are loaded.
    class StandIn:
        calls = 0
        def __call__(self, value):
            self.calls += 1
            return torch.nn.functional.interpolate(value, scale_factor=4, mode='nearest')
    model = StandIn()
    rng = np.random.default_rng(20261012)
    image = rng.random((71, 83, 3), dtype=np.float32)
    target = (332, 284)
    support = np.full((71, 83), .4, np.float32)
    support[:, :40] = 0
    original, amount = _native_carriers(image, support, target)
    expected = sample(image, maps((0, 0, *target), (83, 71), target))
    assert np.array_equal(original, expected)
    assert np.array_equal(amount, sample(support, maps((0, 0, *target), (83, 71), target)))
    zero = np.empty_like(original)
    mixed = np.empty_like(original)
    with patch.object(torch.Tensor, 'cuda', lambda self: self), \
            patch.object(torch.Tensor, 'half', lambda self: self), \
            patch('torch.cuda.mem_get_info', side_effect=AssertionError('CPU smoke touched CUDA')):
        stats = {}
        enhance_tiled(image, target, np.zeros_like(support), zero, model, tile=32,
                      execution='host', stats=stats)
        assert np.array_equal(zero, expected) and not stats['device_staging']
        assert stats['model_calls'] == model.calls == 0 and stats['zero_support_tiles_skipped'] == 9
        with patch.object(torch.Tensor, 'cuda', side_effect=AssertionError('Zero support transferred model input')):
            enhance_tiled(image, target, np.zeros_like(support), zero, model, tile=32, execution='host')
        mixed_stats = {}
        enhance_tiled(image, target, support, mixed, model, tile=32, execution='host', stats=mixed_stats)
        assert mixed_stats['model_calls'] == model.calls == 6 and mixed_stats['zero_support_tiles_skipped'] == 3
        assert np.isfinite(mixed).all()
        assert np.array_equal(mixed[amount == 0], zero[amount == 0])
        # Compute the unsuppressed stand-in once with the same halo and native
        # sampling, then apply the established support equation independently.
        all_detail = np.empty_like(mixed)
        enhance_tiled(image, target, np.ones_like(support), all_detail, model, tile=32, execution='host')
        np.testing.assert_array_equal(mixed, original+amount[...,None]*(all_detail-original))
        native = (317, 263)
        native_zero = np.empty((263, 317, 3), np.float32)
        native_mixed = np.empty_like(native_zero)
        enhance_tiled(image, native, np.zeros_like(support), native_zero, model, tile=32, execution='host')
        enhance_tiled(image, native, support, native_mixed, model, tile=32, execution='host')
        native_amount = sample(support, maps((0, 0, *native), (83, 71), native))
        assert np.array_equal(native_mixed[native_amount == 0], native_zero[native_amount == 0])
        host256 = np.empty_like(original)
        enhance_tiled(image, target, support, host256, model, tile=256, execution='host')
        class FailOnce:
            calls = 0
            def __call__(self, value):
                self.calls += 1
                if self.calls == 1:
                    raise torch.cuda.OutOfMemoryError('Synthetic activation OOM')
                return model(value)
        fallback = np.empty_like(original)
        with patch('torch.cuda.empty_cache'):
            enhance_tiled(image, target, support, fallback, FailOnce(), tile=256,
                          execution='host', stats=stats)
        assert stats['tile'] == 128 and stats['oom_fallback']
        assert stats['model_calls'] == 2 and stats['model_calls_before_oom'] == 1
        assert np.array_equal(host256, fallback)
        # A support boundary inside the last interpolation footprint still
        # requests detail; source-tile zero checks alone would lose this dose.
        boundary_support = np.zeros((71, 83), np.float32)
        boundary_support[:, 32:] = .4
        boundary_stats = {}
        enhance_tiled(image, native, boundary_support, native_mixed, model, tile=32,
                      execution='host', stats=boundary_stats)
        assert boundary_stats['zero_support_tiles_skipped'] == 0 and boundary_stats['model_calls'] == 9
        # A supported tile must retain its model failure contract.
        class Failed:
            def __call__(self, value):
                raise RuntimeError('Supported inference failed')
        try:
            enhance_tiled(image, native, support, native_mixed, Failed(), tile=32, execution='host')
        except RuntimeError as error:
            assert str(error) == 'Supported inference failed'
        else:
            raise AssertionError('Supported model failure was swallowed')
    print(json.dumps({'ok': True, 'cpu_only': True, 'source_carrier_and_zero_support_exact': True,
        'noninteger_scene_sampling': True, 'host_oom_tile_reduction': True,
        'host_zero_support_model_calls_skipped': True, 'interpolation_boundary_support_preserved': True,
        'model_attempt_counts_include_oom': True, 'supported_model_failure_propagates': True,
        'real_model_inferences': 0, 'cuda_contexts': 0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-only', action='store_true')
    if parser.parse_args().cpu_only:
        return cpu_smoke()
    assert torch.cuda.is_available(), 'This regression requires the configured local CUDA runtime'
    weight = Path(load_registry()['background_sr_checkpoint'])
    assert digest(weight) == WEIGHT_SHA256
    model = SRVGGNetCompact().cuda().half().eval()
    state = torch.load(weight, map_location='cpu', weights_only=True)
    model.load_state_dict(state.get('params_ema', state.get('params', state)), strict=True)
    rng = np.random.default_rng(20261011)
    image = rng.random((71, 83, 3), dtype=np.float32)
    target = (332, 284)
    with torch.inference_mode():
        tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).unsqueeze(0).cuda().half()
        expected = model(tensor).squeeze(0).float().permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    tiled = np.empty_like(expected)
    enhance_tiled(image, target, np.ones(image.shape[:2], np.float32), tiled, model, tile=32)
    error = float(np.max(np.abs(tiled-expected)))
    assert error <= .003, ('Tiled inference introduces a seam', error)
    baseline = sample(image, maps((0, 0, *target), (83, 71), target))
    zero = np.empty_like(expected)
    zero_stats={}
    enhance_tiled(image, target, np.zeros(image.shape[:2], np.float32), zero, model, tile=32,stats=zero_stats)
    assert np.array_equal(zero, baseline), 'Suppressed SR regions changed'
    assert zero_stats['device_staging'] and zero_stats['model_calls']==0
    mixed = np.empty_like(expected)
    enhance_tiled(image, target, np.full(image.shape[:2], .4, np.float32), mixed, model, tile=32)
    assert np.allclose(mixed, baseline+.4*(tiled-baseline), atol=2e-7)
    assert np.isfinite(mixed).all()
    # Non-integer native scaling exercises installed OpenCV sampling precision.
    native=(317,263);mask=np.full(image.shape[:2],.4,np.float32);mask[:,:12]=0
    host=np.empty((263,317,3),np.float32);device=np.empty_like(host)
    enhance_tiled(image,native,mask,host,model,tile=32,execution='host')
    enhance_tiled(image,native,mask,device,model,tile=32)
    sampling_error=float(np.max(np.abs(host-device)))
    assert sampling_error<=.003,('Device sampling changed the established recipe',sampling_error)
    native_mask=sample(mask,maps((0,0,*native),(83,71),native))
    assert np.array_equal(host[native_mask==0],device[native_mask==0])
    low_memory={}
    with patch('torch.cuda.mem_get_info',return_value=(2**20,16*2**30)):
        enhance_tiled(image,native,mask,device,model,tile=32,stats=low_memory)
    assert not low_memory['device_staging'] and np.array_equal(host,device)
    class FailOnce:
        def __init__(self):self.calls=0
        def __call__(self,tensor):
            self.calls+=1
            if self.calls==1:raise torch.cuda.OutOfMemoryError('Controlled activation OOM')
            return model(tensor)
    oom_stats={}
    with patch('torch.cuda.mem_get_info',return_value=(16*2**30,16*2**30)):
        enhance_tiled(image,native,mask,device,FailOnce(),tile=32,stats=oom_stats)
    assert oom_stats['oom_fallback'] and not oom_stats['device_staging']
    assert np.array_equal(host,device), 'OOM fallback did not replay the complete canvas'
    # A host inference OOM halves the same tile loop, bounded at 128 pixels.
    host256=np.empty_like(host)
    enhance_tiled(image,native,mask,host256,model,tile=256,execution='host')
    host_oom={}
    enhance_tiled(image,native,mask,device,FailOnce(),tile=256,execution='host',stats=host_oom)
    assert host_oom['tile']==128 and host_oom['oom_fallback'] and np.array_equal(host256,device)
    print(json.dumps({'ok': True, 'tiled_full_max_error': error,
                      'noninteger_device_host_max_error':sampling_error,
                      'low_memory_and_activation_oom_fallback':True,'host_oom_tile_reduction':True,
                      'zero_support_unchanged': True, 'scene_support_strength_replay': True,
                      'current_layered_sr_preserved':True}))


if __name__ == '__main__':
    main()
