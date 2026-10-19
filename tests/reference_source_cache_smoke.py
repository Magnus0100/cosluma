"""Offline proof that reference stages share one bound source decode."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from PIL import Image

import native_source_baseline as source
import reference_pipeline
from reference_appearance_prepare import _matching_donor_source
from raw_support import RGB_DECODE_VERSION
from display_transform import source_transform_for_precision


def main():
    with TemporaryDirectory(prefix='reference-source-cache-') as temporary:
        root = Path(temporary)
        image_path = root / 'source.png'
        Image.fromarray(np.full((4, 5, 3), [80, 110, 140], dtype=np.uint8), 'RGB').save(image_path)
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        transform = source_transform_for_precision('raster8_srgb')
        run = {'source': {'path': str(image_path), 'sha256': digest},
               'options': {'appearance_revision': {'design_reconstruction_plan': {
                   'schema_version': 'design_reconstruction_plan.v1',
                   'working_domain': 'display_linear_sRGB',
                   'source_baseline': {'schema': 'appearance_source_baseline.v1',
                       'working_domain': 'display_linear_sRGB', 'source_sha256': digest,
                       'source_precision': 'raster8_srgb', 'decode_version': RGB_DECODE_VERSION,
                       'assets': {'foreground_alpha': {'sha256': 'a' * 64}},
                       'source_display': transform, 'source_display_applied_once': True}}}}}
        cache = root / 'source-cache'
        calls = []
        beauty_identity = {'schema': 'portrait_beauty_layer.v1', 'source_sha256': digest,
                           'native_size': [5, 4], 'working_domain': 'display_linear_sRGB',
                           'source_display': transform, 'strength': 1., 'disabled_faces': [],
                           'faces': [{'id': 'face-1', 'native_box': [0, 0, 5, 4],
                                      'pixels_sha256': 'f' * 64}]}
        original = source.open_source_for_run

        def counted(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        old = os.environ.get('COSPLAY_REFERENCE_SOURCE_CACHE')
        os.environ['COSPLAY_REFERENCE_SOURCE_CACHE'] = str(cache)
        try:
            run_with_beauty = json.loads(json.dumps(run))
            run_with_beauty['options']['appearance_revision']['portrait_beauty'] = {'result': 'fixture'}
            binding = {'source': digest, 'decode_version': RGB_DECODE_VERSION,
                       'alpha': 'a' * 64,
                       'source_display': transform, 'native_source_retouch': None,
                       'portrait_beauty': beauty_identity}
            with patch.object(source, 'open_source_for_run', side_effect=counted), \
                    patch('portrait_beauty_layer.identity_for_run', return_value=beauty_identity), \
                    patch('portrait_beauty_layer.apply_for_run', side_effect=lambda values, _run: (values + .1, {'applied': True})):
                display_first, _, _ = source.source_for_reference_prepare(run, apply_beauty=False)
                beauty, beauty_precision, _ = source.source_for_reference_prepare(run_with_beauty, binding=binding, apply_beauty=True)
                beauty_reuse, _, _ = source.source_for_reference_prepare(run_with_beauty, binding=binding, apply_beauty=True)
                display_reuse, display_precision, _ = source.source_for_reference_prepare(run_with_beauty, binding=binding, apply_beauty=False)
            display_before = display_reuse.copy()
            display_cache_before = (cache/'source-display.npy').read_bytes()
            with patch.object(source, 'open_source_for_run', side_effect=AssertionError('Donors decoded source again')), \
                    patch('portrait_beauty_layer.identity_for_run', return_value=beauty_identity), \
                    patch('portrait_beauty_layer.apply_for_run', side_effect=AssertionError('Donors replayed cached Beauty')):
                donor_source, donor_execution = _matching_donor_source(display_reuse, run_with_beauty,
                    binding=binding, precision=display_precision, transform=transform)
                assert isinstance(donor_source, np.memmap) and not donor_source.flags.writeable
                np.testing.assert_array_equal(donor_source.view(np.uint32), beauty.view(np.uint32))
                assert donor_execution['source_endpoint_cache_hit'] and not donor_execution['beauty_replay_performed']
                assert donor_execution['applied'] and donor_execution['enabled']
                donor_source._mmap.close()
                try:
                    _matching_donor_source(display_reuse, run_with_beauty,
                        binding={**binding, 'source': 'changed'}, precision=display_precision, transform=transform)
                except ValueError:
                    pass
                else:
                    raise AssertionError('Cached donor source accepted changed fitting binding')
            np.testing.assert_array_equal(display_reuse.view(np.uint32), display_before.view(np.uint32))
            assert (cache/'source-display.npy').read_bytes() == display_cache_before
            # A direct freeze or incomplete Beauty endpoint keeps its existing
            # replay, using the already decoded immutable display source.
            beauty_path = cache/'source-beauty.npy'
            hidden = cache/'hidden-beauty.npy'
            beauty_path.rename(hidden)
            try:
                with patch.object(source, 'open_source_for_run', side_effect=AssertionError('Donor cache miss decoded source')), \
                        patch('portrait_beauty_layer.identity_for_run', return_value=beauty_identity), \
                        patch('portrait_beauty_layer.apply_for_run', side_effect=lambda values, _run:
                            (values + .1, {'enabled': True, 'applied': True})) as replay:
                    donor_source, donor_execution = _matching_donor_source(display_reuse, run_with_beauty,
                        binding=binding, precision=display_precision, transform=transform)
                    np.testing.assert_array_equal(donor_source.view(np.uint32), beauty.view(np.uint32))
                    assert replay.call_count == 1 and donor_execution['beauty_replay_performed']
                    assert not donor_execution['source_endpoint_cache_hit']
                with patch('portrait_beauty_layer.apply_for_run', side_effect=RuntimeError('Frozen Beauty failed')):
                    try:
                        _matching_donor_source(display_reuse, run_with_beauty,
                            binding=binding, precision=display_precision, transform=transform)
                    except RuntimeError as error:
                        assert str(error) == 'Frozen Beauty failed'
                    else:
                        raise AssertionError('Donor fallback swallowed Beauty replay failure')
            finally:
                hidden.rename(beauty_path)
            alpha = np.tile(np.array([0, .2, .5, 1, 1], np.float32), (4, 1))
            estimates = []

            def estimate(values, coverage, trimap):
                estimates.append(1)
                background = values.copy()
                background[trimap == 255] = 0
                background[trimap == 128] *= .25
                foreground = values-background*(1-coverage[..., None])
                support = (trimap == 128).astype(np.float32)*.8
                support[0, 1] = np.float32(-0.0)
                return background, foreground, support, 2, 1., {}

            def pair(values, precision, *, beautiful, current_run=run_with_beauty, current_binding=binding):
                return source.background_for_reference_prepare(values, alpha, current_run,
                    binding=current_binding, precision=precision, transform=transform,
                    apply_beauty=beautiful)

            with patch('source_foreground.estimate_source_background', side_effect=estimate):
                first = pair(beauty, beauty_precision, beautiful=True)
                originals = [array.copy() for array in first[:3]]
                first[1][:] = 0  # native_baseline owns and modifies its foreground.
                second = pair(beauty, beauty_precision, beautiful=True)
                assert len(estimates) == 1 and second[-1]['cache_hit']
                for expected, actual in zip(originals, second[:3]):
                    np.testing.assert_array_equal(expected.view(np.uint32), actual.view(np.uint32))
                second[1][:] = 1
                third = pair(beauty, beauty_precision, beautiful=True)
                np.testing.assert_array_equal(originals[1].view(np.uint32), third[1].view(np.uint32))
                display_pair = pair(display_reuse, display_precision, beautiful=False)
                assert len(estimates) == 2 and not display_pair[-1]['cache_hit']
                assert display_pair[-1]['appearance_variant'] == 'display'
                assert first[-1]['directory'] != display_pair[-1]['directory']
                assert not np.array_equal(originals[1], display_pair[1])
                changed = json.loads(json.dumps(run_with_beauty))
                changed['options']['appearance_revision']['design_reconstruction_plan']['source_baseline']['assets']['foreground_alpha']['sha256'] = 'b' * 64
                changed_pair = pair(beauty, beauty_precision, beautiful=True,
                    current_run=changed, current_binding={**binding, 'alpha': 'b' * 64})
                assert len(estimates) == 3 and not changed_pair[-1]['cache_hit']
                try:
                    pair(beauty, beauty_precision, beautiful=False)
                except ValueError:
                    pass
                else:
                    raise AssertionError('Beauty pixels must not enter an original-display F/B cache')
        finally:
            if old is None:
                os.environ.pop('COSPLAY_REFERENCE_SOURCE_CACHE', None)
            else:
                os.environ['COSPLAY_REFERENCE_SOURCE_CACHE'] = old
        assert len(calls) == 1, f'expected one source decode, got {len(calls)}'
        np.testing.assert_array_equal(display_first, display_reuse)
        np.testing.assert_array_equal(beauty, beauty_reuse)
        assert not np.array_equal(display_reuse, beauty)
        metadata = json.loads((cache / 'metadata.json').read_text(encoding='utf8'))
        assert metadata['schema'] == 'reference_source_cache.v1'
        assert metadata['variants']['display']['identity']['source_sha256'] == digest
        assert sorted(metadata['variants']) == ['beauty', 'display']
        probe = root / 'worker-env.txt'
        code = ('from pathlib import Path; import os; '
                f'Path(r"{probe}").write_text(os.environ["COSPLAY_REFERENCE_SOURCE_CACHE"])')
        reference_pipeline._process(Path.cwd(), ['-c', code], root / 'worker',
                                    stage='cache-env', source_cache=cache)
        assert probe.read_text(encoding='utf8') == str(cache.resolve())
        print(json.dumps({'schema': 'reference_source_cache_smoke.v1', 'status': 'passed',
                          'source_decode_calls': len(calls), 'variants': sorted(metadata['variants']),
                          'appearance_pair_estimates': len(estimates),
                          'appearance_pair_reuse_exact': True,
                          'appearance_pair_reuse_bitwise_exact': True,
                          'appearance_pair_variants_separate': True,
                          'freeze_donors_reuse_readonly_beauty_endpoint': True,
                          'immutable_display_pair_unchanged': True,
                          'donor_cache_miss_replays_without_decode': True,
                          'model_requests': 0, 'image_generation_requests': 0, 'gpu_requests': 0}))


if __name__ == '__main__':
    main()
