"""Small I/O lifecycle checks; no RAW, model, GPU, or large payloads."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tifffile

import raw_support
import source_foreground_cache


def main() -> int:
    with tempfile.TemporaryDirectory(prefix='cosplay-io-lifecycle-') as folder:
        root = Path(folder)
        # TIFF metadata is read from the mature header even though Pillow mode
        # is RGB; the current non-RAW decode contract remains raster8_srgb.
        source = root / 'source.tif'
        values = np.arange(8 * 12 * 3, dtype=np.uint16).reshape(8, 12, 3) + 25600
        tifffile.imwrite(source, values, photometric='rgb', metadata=None)
        with patch.object(raw_support.ImageOps, 'exif_transpose', side_effect=AssertionError('metadata path decoded pixels')):
            assert raw_support.dimensions(source) == (12, 8)
            info = raw_support.storage_info(source)
        assert info['bits_per_channel'] == 16
        assert info['bit_depth_source'] == 'tiff_bits_per_sample'
        assert raw_support.open_linear(source)[1]['source_precision'] == 'raster8_srgb'

        # A truncated completed-bundle receipt is a miss. A mocked current
        # producer then publishes a valid result through the coordinator's
        # atomic replace path; no model/worker runtime is started here.
        saliency = root / 'saliency.png'; saliency.write_bytes(b'saliency')
        identity = {'source_sha256': 'a' * 64,
                    'parameters': {'native_values_encoding': 'fixture'},
                    'worker_sha256': 'worker'}
        cache_root = root / 'cache'
        key = __import__('hashlib').sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
        result_path = cache_root / ('a' * 16) / key / 'result.json'
        result_path.parent.mkdir(parents=True)
        result_path.write_text('{', encoding='utf8')
        calls = []

        def fake_execute(*args, **kwargs):
            calls.append(True)
            (Path(args[3]) / 'result.json').write_text(json.dumps({'provenance': {}}), encoding='utf8')
            return {}

        with patch.object(source_foreground_cache, 'local_runtime_paths', return_value={'python': 'fixture-python', 'vitmatte_model_root': root}), \
             patch.object(source_foreground_cache, '_identity', return_value=identity), \
             patch.object(source_foreground_cache, 'cache_dir', return_value=cache_root), \
             patch.object(source_foreground_cache, '_valid_assets', return_value=True), \
             patch('reference_stage_worker.execute_stage', side_effect=fake_execute):
            result = source_foreground_cache.prepare_source_foreground(
                root, source, source_hash=identity['source_sha256'], saliency_path=saliency)
        assert calls and result['cache_hit'] is False
        assert json.loads(result_path.read_text(encoding='utf8'))['cache_hit'] is False
        assert not list(result_path.parent.glob('result.json.tmp-*'))
    print(json.dumps({'ok': True, 'tiff_header_bits': 16,
                      'raster_decode_contract': 'raster8_srgb',
                      'malformed_bundle_receipt_recovered': True,
                      'worker_or_model_loaded': False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
