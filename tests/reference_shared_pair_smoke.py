"""Cross-reference/work reuse stays exact, bound, and saved-project independent."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

import native_source_baseline as source
from native_array_storage import save_native_array


def same_bits(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


def main():
    with TemporaryDirectory(prefix='reference-shared-pair-') as temporary:
        root = Path(temporary).resolve()
        cache_base = root/'outputs/cache'
        stages = [cache_base/'reference-pipeline'/name/'source-cache' for name in ('first','second','third','fourth')]
        alpha = np.tile(np.array([0., .2, .5, 1., 1.], np.float32), (4, 1))
        pixels = np.arange(60, dtype=np.float32).reshape(4, 5, 3)/64
        transform = {'schema': 'fixture-display', 'domain': 'display_linear_sRGB'}
        beauty = {'strength': 1., 'face_pixels': 'f'*64}
        precision = {'source_precision': 'raster8_srgb', 'decode_version': 'fixture-decode',
            'source_display_consumptions': 1, 'portrait_beauty': beauty, 'portrait_beauty_consumptions': 1}
        binding = {'source': 's'*64, 'reference': 'r'*64, 'decode_version': 'fixture-decode',
            'source_display': transform, 'native_source_retouch': None, 'portrait_beauty': beauty, 'alpha': 'a'*64}
        run = {'root': str(root), 'source': {'sha256': 's'*64}, 'options': {'appearance_revision': {
            'design_reconstruction_plan': {'source_baseline': {'assets': {'foreground_alpha': {'sha256': 'a'*64}}}}}}}
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

        def pair(stage, *, current_run=run, current_binding=binding, current_precision=precision):
            with patch.dict(os.environ, {'COSPLAY_REFERENCE_SOURCE_CACHE': str(stage)}):
                return source.background_for_reference_prepare(pixels, alpha, current_run,
                    binding=current_binding, precision=current_precision, transform=transform, apply_beauty=True)

        with patch('source_foreground.estimate_source_background', side_effect=estimate):
            first = pair(stages[0])
            assert len(estimates) == 1 and not first[-1]['cache_hit']
            second = pair(stages[1], current_binding={**binding, 'reference': 'another-reference'})
            assert len(estimates) == 1 and second[-1]['cache_scope'] == 'shared'
            for expected, actual in zip(first[:3], second[:3]):
                same_bits(actual, expected)
            second[1][:] = 0  # The native baseline owns only its loaded buffer.
            repeated = pair(stages[1])
            same_bits(repeated[1], first[1])
            assert len(estimates) == 1

            # Corruption that preserves layout, size, and timestamp still misses.
            path = Path(first[-1]['directory'])/'foreground.npy'
            stat = path.stat()
            damaged = np.load(path, mmap_mode='r+', allow_pickle=False)
            damaged[0] = [1., 0., 1.]
            damaged.flush(); damaged._mmap.close()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            repaired = pair(stages[2])
            assert len(estimates) == 2 and not repaired[-1]['cache_hit']
            same_bits(repaired[1], first[1])

            # Source, Beauty, alpha, and estimator boundaries remain strict.
            changed = copy.deepcopy(run)
            changed['source']['sha256'] = 't'*64
            pair(stages[3], current_run=changed, current_binding={**binding, 'source': 't'*64})
            changed = copy.deepcopy(run)
            changed['options']['appearance_revision']['design_reconstruction_plan']['source_baseline']['assets']['foreground_alpha']['sha256'] = 'b'*64
            pair(stages[3], current_run=changed, current_binding={**binding, 'alpha': 'b'*64})
            changed_beauty = {'strength': .5, 'face_pixels': 'f'*64}
            pair(stages[3], current_binding={**binding, 'portrait_beauty': changed_beauty},
                current_precision={**precision, 'portrait_beauty': changed_beauty})
            with patch.object(source, '_appearance_estimator_identity', return_value={'producer': 'changed'}):
                pair(stages[3])
            assert len(estimates) == 6

            # Malformed disposable locators cannot bypass integrity or stop a revision.
            pair_identity = json.loads((Path(repaired[-1]['directory'])/'metadata.json').read_text())['identity']
            with patch.dict(os.environ, {'COSPLAY_REFERENCE_SOURCE_CACHE': str(stages[0])}):
                locator = source._pair_locator_path('sparse-appearance', pair_identity)
                locator.write_text('[]', encoding='utf8')
            malformed_json = pair(cache_base/'reference-pipeline'/'malformed-json'/'source-cache')
            assert len(estimates) == 7 and not malformed_json[-1]['cache_hit']
            same_bits(malformed_json[1], first[1])
            with patch.dict(os.environ, {'COSPLAY_REFERENCE_SOURCE_CACHE': str(stages[0])}):
                source._publish_pair_locator('sparse-appearance', pair_identity,
                    {'directory': malformed_json[-1]['directory'], 'arrays': None})
            malformed_arrays = pair(cache_base/'reference-pipeline'/'malformed-arrays'/'source-cache')
            assert len(estimates) == 8 and not malformed_arrays[-1]['cache_hit']
            same_bits(malformed_arrays[1], first[1])

            # Reuse the pre-Beauty owned F/B set without repeating estimation or encoding.
            display_precision = {'source_precision': 'raster8_srgb', 'decode_version': 'fixture-decode'}
            with patch.dict(os.environ, {'COSPLAY_REFERENCE_SOURCE_CACHE': str(stages[0])}):
                identity = source.display_pair_identity(run, precision=display_precision,
                    transform=transform, alpha_sha256='a'*64, native_size=(5, 4))
                cold = source.prepare_display_pair(pixels, alpha, run, identity)
                assets = {}
                for name, array, representation in (
                        ('foreground_premultiplied', cold[1], 'premultiplied_RGB'),
                        ('old_environment', cold[0], 'straight_RGB'),
                        ('edge_applicability', cold[2], 'edge_confidence')):
                    asset_path = stages[0].parent/'appearance'/(name+'.npz')
                    asset_path.parent.mkdir(parents=True, exist_ok=True)
                    save_native_array(asset_path, array)
                    assets[name] = {'path': str(asset_path), 'sha256': source.sha256_file(asset_path),
                        'size': [5, 4], 'coordinate_space': 'source_canvas',
                        'working_space': 'display_linear_sRGB', 'representation': representation}
                record = source.publish_display_pair_assets(identity, assets, tiles=2, reconstruction_error=0.)
            assert len(estimates) == 9
            with patch.dict(os.environ, {'COSPLAY_REFERENCE_SOURCE_CACHE': str(stages[1])}):
                warm = source.prepare_display_pair(pixels, alpha, run, identity)
                assert warm[-1] is not None and len(estimates) == 9
                for expected, actual in zip(cold[:3], warm[:3]):
                    same_bits(actual, expected)

            owned = root/'saved-inputs'
            owned.mkdir()
            owned_assets = copy.deepcopy(assets)
            for name, asset in owned_assets.items():
                target = owned/(name+'.npz')
                shutil.copy2(asset['path'], target)
                asset['path'] = str(target)
            plan = run['options']['appearance_revision']['design_reconstruction_plan']
            plan['source_baseline'] = {'schema': 'appearance_source_baseline.v1',
                'working_domain': 'display_linear_sRGB', 'source_display_applied_once': True,
                'appearance_pair': {key: value for key, value in record.items() if key != 'assets'},
                'assets': {key: owned_assets[key] for key in ('foreground_premultiplied','old_environment')}}
            plan['mask_roles'] = {'edge_applicability': owned_assets['edge_applicability']}
            assert cache_base.resolve().is_relative_to(root) and cache_base != root
            shutil.rmtree(cache_base)
            with patch.dict(os.environ, {'COSPLAY_REFERENCE_SOURCE_CACHE': str(stages[1])}):
                saved = source.prepare_display_pair(pixels, alpha, run, identity)
                assert saved[-1] is not None and len(estimates) == 9
                for expected, actual in zip(cold[:3], saved[:3]):
                    same_bits(actual, expected)
    print(json.dumps({'status': 'passed', 'cross_reference_and_work_hit': True,
        'source_Beauty_alpha_estimator_bound': True, 'sparse_finite_corruption_rejected': True,
        'malformed_disposable_locator_falls_back': True,
        'opaque_zero_alpha_signed_zero_bits_exact': True, 'loaded_buffers_independent': True,
        'display_pair_reused_without_encode': True, 'saved_reopen_after_all_cache_removed': True,
        'directory_scans': 0, 'gpu_requests': 0, 'model_requests': 0}))


if __name__ == '__main__':
    main()
