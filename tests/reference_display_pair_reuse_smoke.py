"""CPU proof of immutable pre-Beauty F/B reuse and invalidation."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

import native_source_baseline as source
from native_array_storage import save_native_array


def main():
    with TemporaryDirectory(prefix='reference-display-pair-') as temporary:
        root = Path(temporary)
        run = {'source': {'sha256': 's'*64}, 'options': {'appearance_revision': {
            'design_reconstruction_plan': {'source_baseline': {}, 'mask_roles': {}}}}}
        precision = {'source_precision': 'raster8_srgb', 'decode_version': 'fixture-decode'}
        transform = {'schema': 'fixture-display', 'domain': 'display_linear_sRGB'}
        alpha = np.tile(np.array([0, .2, .5, 1, 1], np.float32), (4, 1))
        pixels = np.arange(60, dtype=np.float32).reshape(4, 5, 3)/64
        estimates = []

        def estimate(values, coverage, trimap):
            estimates.append(1)
            background = values.copy()
            background[trimap == 255] = 0
            background[trimap == 128] *= .25
            foreground = values-background*(1-coverage[..., None])
            support = (trimap == 128).astype(np.float32)*.8
            support[0, 1] = np.float32(-0.)
            return background, foreground, support, 2, 1., {}

        def identity(*, current_run=run, current_precision=precision, current_transform=transform,
                     alpha_sha='a'*64):
            return source.display_pair_identity(current_run, precision=current_precision,
                transform=current_transform, alpha_sha256=alpha_sha, native_size=(5, 4))

        with patch.dict(os.environ, {'COSPLAY_REFERENCE_SOURCE_CACHE': str(root/'cache')}), \
                patch('source_foreground.estimate_source_background', side_effect=estimate):
            original_identity = identity()
            cold = source.prepare_display_pair(pixels, alpha, run, original_identity)
            assert len(estimates) == 1 and cold[-1] is None
            expected = [array.copy() for array in cold[:3]]
            assets = {}
            for name, values, representation in (
                    ('old_environment', cold[0], 'straight_RGB'),
                    ('foreground_premultiplied', cold[1], 'premultiplied_RGB'),
                    ('edge_applicability', cold[2], 'edge_confidence')):
                path = root/(name+'.npz')
                save_native_array(path, values)
                assets[name] = {'path': str(path), 'sha256': source.sha256_file(path),
                    'size': [5, 4], 'coordinate_space': 'source_canvas',
                    'working_space': 'display_linear_sRGB', 'representation': representation}
            record = source.publish_display_pair_assets(original_identity, assets,
                tiles=cold[3], reconstruction_error=cold[4])
            warm = source.prepare_display_pair(pixels, alpha, run, original_identity)
            assert len(estimates) == 1 and warm[-1] is not None
            for a, b in zip(expected, warm[:3]):
                np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))
            warm[1][:] = 0  # Delta-buffer ownership cannot change the shared NPZ.
            replay = source.prepare_display_pair(pixels, alpha, run, original_identity)
            np.testing.assert_array_equal(expected[1].view(np.uint32), replay[1].view(np.uint32))

            # Freeze never consumes old-B pixels on a verified reuse. Keep
            # its complete archive, but do not materialize that RGB buffer.
            original_load = source.np.load
            decoded = []
            def observe_load(path, *args, **kwargs):
                decoded.append(Path(path))
                return original_load(path, *args, **kwargs)
            with patch.object(source.np, 'load', side_effect=observe_load):
                lazy = source.prepare_display_pair_for_freeze(pixels, alpha, run, original_identity)
            assert isinstance(lazy[0], source.VerifiedNativeArrayAsset)
            assert Path(assets['old_environment']['path']) not in decoded
            assert decoded == [Path(assets['foreground_premultiplied']['path']),
                               Path(assets['edge_applicability']['path'])]
            lazy[0].validate_asset(assets['old_environment'])
            assert lazy[0].payload_bytes == expected[0].nbytes and lazy[0].shape == expected[0].shape
            np.testing.assert_array_equal(lazy[1].view(np.uint32), expected[1].view(np.uint32))
            np.testing.assert_array_equal(lazy[2].view(np.uint32), expected[2].view(np.uint32))
            assert lazy[3:5] == replay[3:5] and len(estimates) == 1

            beautiful = copy.deepcopy(run)
            beautiful['options']['appearance_revision']['portrait_beauty'] = {'strength': .2, 'result': 'changed'}
            assert identity(current_run=beautiful) == original_identity
            assert source.load_display_pair_assets(beautiful, original_identity) is not None
            changed_alpha = identity(alpha_sha='b'*64)
            assert source.prepare_display_pair(pixels, alpha, run, changed_alpha)[-1] is None
            assert len(estimates) == 2
            with patch.object(source, 'retouch_identity', return_value={'delta_sha256': 'retouched'}):
                changed_retouch = identity()
                assert source.prepare_display_pair(pixels, alpha, run, changed_retouch)[-1] is None
                assert len(estimates) == 3
            for changed in (identity(current_precision={**precision, 'decode_version': 'new-decoder'}),
                            identity(current_transform={**transform, 'new-transform': True}),
                            identity(current_run={**run, 'source': {'sha256': 't'*64}})):
                assert source.load_display_pair_assets(run, changed) is None
            with patch.object(source, '_appearance_estimator_identity', return_value={'producer': 'changed'}):
                assert source.load_display_pair_assets(run, identity()) is None

            # Relocated saved plans keep no duplicate live cache locators.
            plan = run['options']['appearance_revision']['design_reconstruction_plan']
            plan['source_baseline'] = {'schema': 'appearance_source_baseline.v1',
                'working_domain': 'display_linear_sRGB', 'source_display_applied_once': True,
                'appearance_pair': {key: value for key, value in record.items() if key != 'assets'},
                'assets': {key: assets[key] for key in ('foreground_premultiplied', 'old_environment')}}
            plan['mask_roles']['edge_applicability'] = assets['edge_applicability']
            with patch.object(source, '_reference_source_cache_dir', return_value=None):
                saved = source.prepare_display_pair(pixels, alpha, run, original_identity)
                assert saved[-1] is not None and len(estimates) == 3
                for a, b in zip(expected, saved[:3]):
                    np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))
                saved_lazy = source.prepare_display_pair_for_freeze(pixels, alpha, run, original_identity)
                assert isinstance(saved_lazy[0], source.VerifiedNativeArrayAsset)
                saved_lazy[0].validate_asset(assets['old_environment'])
                plan['mask_roles']['edge_applicability'] = {**assets['edge_applicability'], 'sha256': 'wrong'}
                assert source.load_display_pair_assets(run, original_identity) is None
            assets['foreground_premultiplied']['path'] = str(root/'missing.npz')
            source.publish_display_pair_assets(original_identity, assets, tiles=2, reconstruction_error=0.)
            assert source.load_display_pair_assets(run, original_identity) is None
    print(json.dumps({'status': 'passed', 'cold_miss': True, 'warm_hit': True,
        'saved_plan_hit': True, 'beauty_independent': True, 'alpha_retouch_invalidation': True,
        'source_decode_display_estimator_invalidation': True, 'payload_bit_exact': True,
        'missing_or_changed_assets_miss': True, 'model_requests': 0, 'gpu_requests': 0}))


if __name__ == '__main__':
    main()
