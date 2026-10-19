"""CPU exactness and corruption fallback for disposable native T+B reuse."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
from unittest.mock import patch

import numpy as np

import reference_colour_fit as matching
import reference_pre_scene_cache as cache
import reference_material_palette as palette_math
from colour_space import from_lab
import render_reference_colour_composite as renderer
from render_reference_colour_composite import match_foreground
from reference_appearance_prepare import _freeze_matching_native


def same_bits(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert actual.tobytes(order='C') == expected.tobytes(order='C')


def main():
    rng = np.random.default_rng(9031)
    source = rng.uniform(.01, .88, (12, 257, 3)).astype(np.float32)
    rgb = lambda lab: from_lab(np.asarray(lab, np.float32)).astype(np.float32)
    source[:, 132:200] = rgb([.55, .02, .03])
    source[:, 200:245] = [.1, .3, .12]
    alpha = np.ones(source.shape[:2], np.float32)
    alpha[:, 246:251] = [.99, .96, .94, .7, .2]
    alpha[:, 251:256] = 0
    alpha[:, 256] = .98  # The final one-column block must keep the tile kernel.
    fp = source*alpha[..., None]
    params = matching.fit([])
    params['operators'] = {name: {'matrix': [[1.04, .015], [-.02, .98]],
        'bias': [.006, -.004], 'exposure_ev': .22} for name in ('face', 'clothing', 'hair')}
    params['operators']['clothing']['tone_curve'] = {'source_knots': [0., .25, .5, .75, 1.],
        'target_knots': [0., .23, .53, .78, 1.]}
    params['conditional_chroma_residual'] = {'hair': {'knots': [-.1, 0., .1],
        'nodes': [[.005, -.004], [.001, .002], [-.004, .006]], 'observed_range': [-.09, .09],
        'source_channel': 1, 'support_margin': .05}}
    params['local_chroma_residual'] = {'face': {'bounds_normalized': [[.05, .04], [.8, .85]],
        'nodes': [[.006, -.005], [-.004, .002], [.001, .004], [-.003, -.006]],
        'support_margin_normalized': [.04, .05],
        'support_hull_equations': [[-1., 0., .05], [1., 0., -.8], [0., -1., .04], [0., 1., -.85]],
        'strength': .75}}
    from current_stack_fixture import bind_local_domain
    bind_local_domain(params['local_chroma_residual']['face'], (257,12))
    params['protection_policy'] = {'mode': 'split_lightness_chroma'}
    params['authored_chroma_adjustments'] = [{'region': 'face', 'chroma_shift': [.007, -.002],
        'source_a_ramp': [-.004, -.02], 'stroke': {'points': [[.7, .5], [.9, .5]],
            'native_canvas_size': [257, 12], 'radius_px': 24, 'hardness': .6, 'opacity': .8}}]
    training = np.broadcast_to(rgb([.55, .02, .03]), (96, 128, 3)).copy()
    target = np.broadcast_to(rgb([.55, .035, .015]), training.shape).copy()
    labels = np.ones(training.shape[:2], np.uint8)
    palette = palette_math.fit_palette(training, target, labels, labels, np.ones(labels.shape), matching.fit([]))
    assert palette['regions']['clothing']['status'] == 'supported'
    params['automatic_material_palette'] = palette
    transform = {'schema': 'fixture-display', 'domain': 'display_linear_sRGB'}
    params['binding'] = {'source': 's'*64, 'reference': 'r'*64, 'alpha': 'a'*64,
        'source_display': transform, 'portrait_beauty': None, 'source_weights_sha256': 'w'*64}
    lw = np.zeros((*source.shape[:2], len(matching.REGIONS)), np.float32)
    cw = np.zeros_like(lw)
    for index, name in enumerate(matching.REGIONS):
        if name in params['operators']:
            lw[..., index] = .25
            cw[..., index] = .2
    face_index = list(matching.REGIONS).index('face')
    lw[:, 200:245] = 0
    cw[:, 200:245] = 0
    lw[:, 200:245, face_index] = 1
    cw[:, 200:245, face_index] = 1
    colour = np.zeros((*source.shape[:2], 6), np.float32)
    colour[..., :2] = [.001, -.0015]
    colour[..., 2] = .75
    scene = np.empty((*alpha.shape, 4), np.float32)
    scene[..., :3] = [.18, .002, -.003]
    scene[..., 3] = .75
    donors = np.empty((*alpha.shape, 5), np.float32)
    donors[..., 0] = 1.1
    donors[..., 1:4] = [.003, -.002, .001]
    donors[..., 4] = .9
    edge = np.full(alpha.shape, .6, np.float32)
    native = (257, 12)
    proxy = (257, 12)
    box = (0, 0, 257, 12)
    receipt = {'asset': {'sha256': 'c'*64}, 'hair_correction_guard': None}
    kwargs = dict(origin=(0, 0), canvas_size=native, reference_blend=colour,
                  chroma_weights=cw, _trusted=True)
    with TemporaryDirectory(prefix='reference-pre-scene-') as temporary:
        root = Path(temporary)
        endpoint = root/'source-display.npy'
        np.save(endpoint, source, allow_pickle=False)
        identity = {'source_sha256': params['binding']['source'], 'source_display': transform,
            'portrait_beauty': None, 'native_size': list(native), 'native_source_retouch': None,
            'decode_version': 'fixture-decode'}
        (root/'metadata.json').write_text(json.dumps({'variants': {'display': {
            'identity': identity, 'file_bytes': endpoint.stat().st_size}}}), encoding='utf8')
        with patch.dict(os.environ, {'COSPLAY_REFERENCE_SOURCE_CACHE': str(root)}):
            assert cache.open_reader(params, receipt, native, proxy) is None
            writer = cache.open_writer(params, receipt, native, proxy)
            assert writer is not None
            baseline = match_foreground(fp, alpha, lw, donors, edge, params,
                pre_scene_sink=lambda values: writer.write(box, values), **kwargs)
            assert writer.finish()['written'] is True
            original = match_foreground(fp, alpha, lw, donors, edge, params, scene_balance=scene, **kwargs)
            for field in ('automatic_material_palette', 'authored_chroma_adjustments'):
                without = copy.deepcopy(params)
                without.pop(field)
                comparison = match_foreground(fp, alpha, lw, donors, edge, without, scene_balance=scene, **kwargs)
                assert original.tobytes() != comparison.tobytes(), field+' must exercise nonzero work'
            reader = cache.open_reader(params, receipt, native, proxy)
            assert reader is not None
            before_stats = dict(reader.stats)
            independent, counts = reader._read_band(box, fp, alpha)
            assert reader.stats == before_stats and counts['hit_bands'] == 1
            prepared, valid = reader.read(box, fp, alpha)
            same_bits(independent[0], prepared)
            np.testing.assert_array_equal(independent[1], valid)
            assert valid.any() and not valid[alpha != 1].any()
            actual = match_foreground(fp, alpha, lw, donors, edge, params, scene_balance=scene,
                pre_scene_target=prepared, pre_scene_valid=valid, **kwargs)
            same_bits(actual, original)
            with patch.object(renderer, '_PRE_SCENE_MAX_ADJACENT_BLOCKS', 1):
                unmerged = match_foreground(fp, alpha, lw, donors, edge, params, scene_balance=scene,
                    pre_scene_target=prepared, pre_scene_valid=valid, **kwargs)
            same_bits(unmerged, actual)
            same_bits(actual[alpha == 0], np.zeros_like(actual[alpha == 0]))

            changed = fp.copy()
            changed[5, 133, 0] = np.nextafter(changed[5, 133, 0], np.float32(1))
            expected = match_foreground(changed, alpha, lw, donors, edge, params, scene_balance=scene, **kwargs)
            prepared, valid = reader.read(box, changed, alpha)
            assert not valid[5, 133] and reader.stats['endpoint_mismatch_pixels'] == 1
            actual = match_foreground(changed, alpha, lw, donors, edge, params, scene_balance=scene,
                pre_scene_target=prepared, pre_scene_valid=valid, **kwargs)
            same_bits(actual, expected)
            with patch.object(renderer, '_PRE_SCENE_MAX_ADJACENT_BLOCKS', 1):
                unmerged = match_foreground(changed, alpha, lw, donors, edge, params, scene_balance=scene,
                    pre_scene_target=prepared, pre_scene_valid=valid, **kwargs)
            same_bits(unmerged, actual)
            reader.close()

            # Exercise ordered delta/gamut/stats publication across several
            # bands and a tail, using the existing native matching kernel.
            context = (root/'parallel-bands', endpoint, {'native_size': list(native)})
            band_writer = cache.PreSceneWriter(context)
            tile_source = renderer.matching_tiles
            for _, _, tile_box, region in tile_source(alpha, tile_rows=5):
                band_writer.write(tile_box, independent[0][region])
            assert band_writer.finish()['written']
            strong_donors = donors.copy(); strong_donors[..., 0] = 2.6
            plain = np.concatenate((lw, strong_donors, cw), axis=-1)
            channels = (lw.shape[-1], donors.shape[-1], cw.shape[-1])
            outputs = []
            working_before = fp.copy()
            for workers in (1, 2):
                reader = cache.PreSceneReader(context)
                merge = reader._merge_stats
                order = []
                def commit(counts):
                    assert threading.current_thread() is threading.main_thread()
                    order.append(dict(counts)); merge(counts)
                delta = np.empty_like(fp)
                with patch.object(renderer, 'matching_tiles', side_effect=lambda a:
                        tile_source(a, tile_rows=5)), patch.object(reader, '_merge_stats', side_effect=commit):
                    result = _freeze_matching_native(fp, delta, alpha, edge, params,
                        colour, scene, None, None, None, proxy, native, plain, channels,
                        has_chroma=True, pre_scene_reader=reader, workers=workers)
                assert reader.target is reader.endpoint is None
                assert len(order) == 3 and result[3]['hit_bands'] == 3
                assert result[2]['requested_outside_sdr_pixels'] > 0
                assert result[:2] == (True, True)
                outputs.append((delta, result))
            same_bits(outputs[0][0], outputs[1][0])
            assert outputs[0][1] == outputs[1][1]
            same_bits(fp, working_before)
            same_bits((fp+outputs[1][0])[alpha == 0], np.zeros_like(fp[alpha == 0]))
            for failing in ('worker', 'consumer'):
                reader = cache.PreSceneReader(context)
                delta = np.full_like(fp, .1234); kept = delta.copy()
                target = (patch.object(renderer, 'match_foreground', side_effect=RuntimeError('worker failed'))
                          if failing == 'worker' else
                          patch.object(reader, '_merge_stats', side_effect=RuntimeError('consumer failed')))
                with patch.object(renderer, 'matching_tiles', side_effect=lambda a:
                        tile_source(a, tile_rows=5)), target:
                    try:
                        _freeze_matching_native(fp, delta, alpha, edge, params,
                            colour, scene, None, None, None, proxy, native, plain, channels,
                            has_chroma=True, pre_scene_reader=reader, workers=2)
                    except RuntimeError as error:assert str(error) == failing+' failed'
                    else:raise AssertionError('Native band failure was lost')
                assert reader.target is reader.endpoint is None
                same_bits(delta, kept)

            for key in ('source', 'alpha', 'portrait_beauty', 'source_weights_sha256'):
                changed_params = copy.deepcopy(params)
                changed_params['binding'][key] = 'changed'
                assert cache.open_reader(changed_params, receipt, native, proxy) is None
            changed_params = copy.deepcopy(params)
            changed_params['operators']['face']['bias'][0] += .001
            assert cache.open_reader(changed_params, receipt, native, proxy) is None
            assert cache.open_reader(params, {'asset': {'sha256': 'different'}}, native, proxy) is None
            assert cache.open_reader(params, receipt, native, (128, 12)) is None
            with patch.object(cache, 'sha256_file', return_value='changed-code'):
                assert cache.open_reader(params, receipt, native, proxy) is None

            # Preserve size and timestamp to exercise CRC, not only stat checks.
            target = writer.directory/'target.npy'
            stat = target.stat()
            damaged = np.load(target, mmap_mode='r+', allow_pickle=False)
            damaged[0, 5] = [1, 0, 1]
            damaged.flush()
            damaged._mmap.close()
            os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            reader = cache.open_reader(params, receipt, native, proxy)
            assert reader is not None
            independent, counts = reader._read_band(box, fp, alpha)
            assert independent is None and counts['corrupt_bands'] == counts['miss_bands'] == 1
            assert reader.stats['corrupt_bands'] == reader.stats['miss_bands'] == 0
            assert reader.read(box, fp, alpha) is None
            assert reader.stats['corrupt_bands'] == 1
            reader.close()
            with target.open('wb') as file:
                file.write(b'broken-npy')
            assert cache.open_reader(params, receipt, native, proxy) is None
            # A disposable cache write failure cannot fail valid matching.
            writer = cache.open_writer(params, receipt, native, proxy)
            with patch.object(cache, '_crc', side_effect=ValueError('bad temporary storage')):
                actual = match_foreground(fp, alpha, lw, donors, edge, params,
                    pre_scene_sink=lambda values: writer.write(box, values), **kwargs)
            same_bits(actual, baseline)
            assert writer.finish()['written'] is False
    print(json.dumps({'status': 'passed', 'T_colour_scene_edge_bit_exact': True,
        'tone_conditional_local_fields': True, 'thin_tail_kernel_preserved': True,
        'palette_and_authored_edits_nonzero_bit_exact': True,
        'bounded_adjacency_merge_CPU_bit_exact': True,
        'changed_endpoint_recomputed': True, 'all_binding_changes_miss': True,
        'same_stat_finite_corruption_rejected': True, 'truncation_safe': True,
        'cache_failure_keeps_pipeline': True, 'parallel_delta_gamut_stats_bit_exact': True,
        'parallel_exception_closes_readers': True, 'model_requests': 0, 'gpu_requests': 0}))


if __name__ == '__main__':
    main()
