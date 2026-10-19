"""CPU bounds and global-coordinate checks for fixed native miss merging."""
from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np

import render_reference_colour_composite as renderer


def main():
    height, width = 128, 769
    source = np.arange(height*width*3, dtype=np.float32).reshape(height, width, 3)/300000.
    weights = np.zeros((height, width, 2), np.float32)
    weights[:, :400, 0] = .25; weights[:, 400:, 1] = .5
    chroma = weights*.75
    colour = np.ones((height, width, 6), np.float32)*.003
    origin, canvas = (37, 911), (4688, 7028)
    calls = []

    def pointwise(values, region_weights, parameters, *, origin, canvas_size,
                  reference_blend, chroma_weights, _trusted):
        assert canvas_size == canvas and _trusted
        first = origin[0]-37
        last = first+values.shape[1]
        np.testing.assert_array_equal(values, source[:, first:last])
        np.testing.assert_array_equal(region_weights, weights[:, first:last])
        np.testing.assert_array_equal(chroma_weights, chroma[:, first:last])
        np.testing.assert_array_equal(reference_blend, colour[:, first:last])
        xx = (np.arange(first, last, dtype=np.float32)+37+.5)/canvas[0]
        yy = (np.arange(height, dtype=np.float32)+911+.5)/canvas[1]
        output = values.copy()
        output[..., 0] += xx
        output[..., 1] += yy[:, None]
        output[..., 2] += region_weights.sum(-1)+chroma_weights.sum(-1)+reference_blend[..., 0]
        calls.append((first, last, values.shape[0]*values.shape[1]))
        return output

    # Misses form a two-block run, a separating valid block, another run and
    # a final one-column miss. Sparse endpoint mismatches share this same mask.
    valid = np.zeros((height, width), bool)
    valid[:, 256:384] = True
    valid[:, 640:768] = True
    valid[7, 401] = True
    prepared = np.full_like(source, .123456)
    kept_source, kept_prepared, kept_valid = source.copy(), prepared.copy(), valid.copy()
    arguments = dict(origin=origin, canvas_size=canvas, reference_blend=colour,
                     chroma_weights=chroma, prepared=prepared, valid=valid, _trusted=True)
    with patch.object(renderer, '_match_straight', pointwise):
        with patch.object(renderer, '_PRE_SCENE_MAX_ADJACENT_BLOCKS', 1):
            old = renderer._reuse_pre_scene(source, weights, {}, **arguments)
        baseline_calls = list(calls); calls.clear()
        new = renderer._reuse_pre_scene(source, weights, {}, **arguments)
    assert old.tobytes() == new.tobytes()
    assert new[valid].tobytes() == prepared[valid].tobytes()
    assert baseline_calls == [(0, 128, 16384), (128, 256, 16384), (384, 512, 16384),
                              (512, 640, 16384), (767, 769, 256)]
    assert calls == [(0, 256, 32768), (384, 640, 32768), (767, 769, 256)]
    assert max(right-left for left, right, _ in calls) <= 256
    assert np.array_equal(source, kept_source) and np.array_equal(prepared, kept_prepared)
    assert np.array_equal(valid, kept_valid)
    assert list(renderer._pre_scene_recompute_ranges(np.ones((3, 17), bool))) == []
    assert list(renderer._pre_scene_recompute_ranges(np.zeros((3, 1), bool))) == [(0, 1)]
    assert renderer._PRE_SCENE_MAX_ADJACENT_BLOCKS == 2
    print(json.dumps({'schema': 'pre_scene_adjacency_smoke.v1', 'passed': True,
                      'single_fixed_two_block_path': True, 'calls_before': len(baseline_calls), 'calls_after': len(calls),
                      'same_global_origin_canvas_owner_and_field_slices': True,
                      'valid_pixels_bit_exact': True, 'masked_writes_bit_exact': True,
                      'maximum_range_columns': 256, 'GPU_eligibility_pixels': [32768, 32768],
                      'thin_tail_preserved': True, 'input_unchanged': True, 'GPU_used': False}))


if __name__ == '__main__':
    main()
