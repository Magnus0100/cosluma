"""Strict source Alpha intermediate cache: identity, bytes and real counts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

import source_foreground as foreground
import source_foreground_cache as foreground_cache


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    with tempfile.TemporaryDirectory(prefix='source_alpha_cache_') as raw:
        root = Path(raw)
        source = root/'source.raw'; source.write_bytes(b'bound source bytes')
        model = root/'vitmatte'; model.mkdir()
        for name in ('model.safetensors', 'config.json', 'preprocessor_config.json'):
            (model/name).write_bytes(('model '+name).encode())
        display = np.zeros((6, 8, 3), np.uint8)
        display[..., 0] = np.arange(8, dtype=np.uint8)
        trimap = np.full((6, 8), 128, np.uint8)
        trimap[0] = 0
        trimap[-1] = 255
        with patch.object(foreground, '_alpha_runtime_identity',
                          return_value={'runtime': 'CPU fixture; no model inference'}):
            identity = foreground.alpha_refinement_identity(display, trimap, model,
                source=source, precision={'source_precision': 'raw16_linear',
                                      'decode_version': 'fixture-decode',
                                      'native_source_retouch': {'schema': 'retouch.v1', 'sha256': 'r'*64}},
                native_size=(8, 6), device='cpu')
        semantics_a = root/'diagnostic-a.npz'; semantics_a.write_bytes(b'a')
        semantics_b = root/'diagnostic-b.npz'; semantics_b.write_bytes(b'b')
        saliency = root/'saliency.png'; saliency.write_bytes(b'saliency')
        runtime = {'vitmatte_model_root': model, 'matting_profile': 'vitmatte_small_distinctions'}
        with patch.object(foreground_cache, 'file_identity', side_effect=lambda path: _sha(path)):
            core_a = foreground_cache._identity(root, source, saliency, semantics_a, runtime)
            core_b = foreground_cache._identity(root, source, saliency, semantics_b, runtime)
        assert core_a == core_b, 'diagnostic-only semantics must not invalidate Alpha/F-B cache'
        calls = []

        def fake_refine(values, mask, model_dir, **kwargs):
            calls.append(dict(kwargs))
            result = np.full(mask.shape, .5, np.float32)
            result[mask == 0] = 0
            result[mask == 255] = 1
            execution = kwargs.pop('_execution', None)
            if execution is not None:
                execution['model_inference_batches'] = 2
            return result, 4

        cache = root/'alpha-cache'
        with patch.object(foreground, 'refine_alpha', side_effect=fake_refine):
            first, first_tiles, first_receipt = foreground.refine_alpha_cached(
                display, trimap, model, cache, identity=identity, device='cpu')
            second, second_tiles, second_receipt = foreground.refine_alpha_cached(
                display, trimap, model, cache, identity=identity, device='cpu')
        assert len(calls) == 1, 'published Alpha must avoid a second model execution'
        assert first_tiles == 4 and second_tiles == 0
        assert first_receipt['cache_hit'] is False and second_receipt['cache_hit'] is True
        np.testing.assert_array_equal(first, second)
        receipt = json.loads(Path(second_receipt['receipt_path']).read_text(encoding='utf8'))
        assert receipt['model_inference_batches'] == 2 and receipt['tiles'] == 4
        assert _sha(Path(second_receipt['receipt_path']).parent/'alpha.npz') == receipt['alpha_sha256']

        # Trimap bytes are part of the key; a changed unknown/known decision
        # must never reuse the old learned coverage.
        changed = trimap.copy(); changed[1, 1] = 0
        changed_identity = dict(identity, trimap_sha256=foreground._alpha_array_sha(changed))
        with patch.object(foreground, 'refine_alpha', side_effect=fake_refine):
            _, _, changed_receipt = foreground.refine_alpha_cached(
                display, changed, model, cache, identity=changed_identity, device='cpu')
        assert changed_receipt['cache_hit'] is False and len(calls) == 2

        # A damaged array or receipt is a cache miss, never an old-pixel reuse.
        asset = Path(second_receipt['receipt_path']).parent/'alpha.npz'
        asset.write_bytes(b'corrupt')
        with patch.object(foreground, 'refine_alpha', side_effect=fake_refine):
            _, _, damaged_receipt = foreground.refine_alpha_cached(
                display, trimap, model, cache, identity=identity, device='cpu')
        assert damaged_receipt['cache_hit'] is False and len(calls) == 3
        assert damaged_receipt['model_inference_batches_this_attempt'] == 2
    print('OK: Alpha cache binds source/trimap/model/code/runtime, reuses only verified bytes, and records actual tiles/batches')


if __name__ == '__main__':
    main()
