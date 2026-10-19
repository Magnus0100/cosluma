"""Real CRC/endpoint masks preserve owner, visibility and global B sampling."""
from __future__ import annotations

import json
import os
from contextlib import ExitStack
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

import reference_pre_scene_cache as cache
import render_reference_colour_composite as renderer


def reject(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError('Invalid pre-scene B sampling contract was accepted')


def main():
    rng = np.random.default_rng(76251)
    native, proxy = (280, 40), (31, 27)
    box = (11, 7, 268, 22)
    shape = (15, 257)
    source = rng.uniform(.1, .7, (native[1], native[0], 3)).astype(np.float32)
    alpha = np.ones(shape, np.float32)
    alpha[:, 5] = .8; alpha[:, 256] = .98
    working = source[box[1]:box[3], box[0]:box[2]]*alpha[..., None]
    colour = rng.uniform(-.003, .003, (proxy[1], proxy[0], 6)).astype(np.float32)
    colour[..., 2] = .8; colour[..., 5] = .7
    owners = np.ones(colour.shape[:2], np.uint16)
    owners[:, 9:18] = 4; owners[:, 18:23] = 1004
    support = rng.uniform(.1, .9, shape).astype(np.float32)
    grid = renderer._sample_grid(box, proxy, native)
    plan = renderer._sample_plan(grid, colour.shape[:2])
    kwargs = {'owners': owners, 'grid': grid, 'sample_plan': plan,
        'visible_face_support': support}
    full = renderer.sample_reference_field(colour, box, proxy, native, **kwargs)
    with tempfile.TemporaryDirectory(prefix='pre-scene-sampling-') as temporary, ExitStack() as cleanup:
        root = Path(temporary)
        endpoint = root/'source.npy'; np.save(endpoint, source, allow_pickle=False)
        context = (root/'cache', endpoint, {'native_size': list(native)})
        writer = cache.PreSceneWriter(context)
        writer.write(box, source[box[1]:box[3], box[0]:box[2]].copy())
        assert writer.finish()['written']
        reader = cache.PreSceneReader(context)
        cleanup.callback(reader.close)
        prepared, valid = reader.read(box, working, alpha)
        assert list(renderer._pre_scene_recompute_ranges(valid)) == [(0, 128), (255, 257)]
        assert not valid[alpha != 1].any()
        sampled = renderer.sample_pre_scene_reference_field(colour, box, proxy, native,
            pre_scene_valid=valid, **kwargs)
        for first, right in renderer._pre_scene_recompute_ranges(valid):
            assert sampled[:, first:right].tobytes() == full[:, first:right].tobytes()
        assert sampled[~valid].tobytes() == full[~valid].tobytes()
        # The same unplanned and owner-free callers keep their original
        # coordinate and OpenCV sampling behaviour as well.
        for field in (colour, colour.astype(np.float64)):
            for domain in (None, owners):
                extra = {'visible_face_support': support} if domain is not None else {}
                original = renderer.sample_reference_field(field, box, proxy, native,
                    owners=domain, **extra)
                actual = renderer.sample_pre_scene_reference_field(field, box, proxy, native,
                    pre_scene_valid=valid, owners=domain, **extra)
                assert actual.dtype == original.dtype
                assert original[~valid].tobytes() == actual[~valid].tobytes()
        changed = working.copy()
        changed[3, 80, 0] = np.nextafter(changed[3, 80, 0], np.float32(1))
        _, changed_valid = reader.read(box, changed, alpha)
        assert not changed_valid[3, 80] and reader.stats['endpoint_mismatch_pixels'] == 1
        actual = renderer.sample_pre_scene_reference_field(colour, box, proxy, native,
            pre_scene_valid=changed_valid, **kwargs)
        assert actual[~changed_valid].tobytes() == full[~changed_valid].tobytes()
        changed_middle = working.copy()
        changed_middle[3, 180, 0] = np.nextafter(changed_middle[3, 180, 0], np.float32(1))
        assert reader.read(box, changed_middle, alpha) is None
        reader.close()
        target = context[0]/'target.npy'; stat = target.stat()
        damaged = np.load(target, mmap_mode='r+', allow_pickle=False)
        damaged[box[1], box[0]] = [1, 0, 1]
        damaged.flush(); damaged._mmap.close()
        os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        reader = cache.PreSceneReader(context)
        cleanup.callback(reader.close)
        assert reader.read(box, working, alpha) is None and reader.stats['corrupt_bands'] == 1
        fallback = renderer.sample_pre_scene_reference_field(colour, box, proxy, native,
            pre_scene_valid=None, **kwargs)
        assert fallback.tobytes() == full.tobytes()
        reader.close()
    all_valid = np.ones(shape, bool)
    with patch.object(renderer, 'sample_reference_field', side_effect=AssertionError('Cached B was sampled')):
        assert not renderer.sample_pre_scene_reference_field(colour, box, proxy, native,
            pre_scene_valid=all_valid, **kwargs).any()
    for mask in (all_valid.astype(np.float32), all_valid[:, :-1], np.ones((1, 2, 3), bool)):
        reject(lambda mask=mask: renderer.sample_pre_scene_reference_field(colour, box, proxy, native,
            pre_scene_valid=mask, **kwargs))
    for invalid in ({**kwargs, 'owners': owners[:-1]},
                    {**kwargs, 'owners': owners.astype(np.float32)},
                    {**kwargs, 'owners': None},
                    {**kwargs, 'visible_face_support': np.full(shape, np.nan, np.float32)},
                    {**kwargs, 'visible_face_support': np.full(shape, 1.1, np.float32)},
                    {**kwargs, 'grid': tuple(value[:, :-1] for value in grid)},
                    {**kwargs, 'sample_plan': {**plan, 'values_shape': (1, 1)}}):
        reject(lambda invalid=invalid: renderer.sample_pre_scene_reference_field(colour, box, proxy, native,
            pre_scene_valid=all_valid, **invalid))
    print(json.dumps({'schema': 'reference_pre_scene_sampling_smoke.v1', 'passed': True,
        'actual_CRC_opaque_endpoint_masks': True, 'source_global_origin_owner_visibility_bits': True,
        'one_column_tail_neighbour_preserved': True, 'endpoint_change_and_CRC_failure': True,
        'fully_cached_invalid_metadata_or_support_rejected': True, 'models_or_GPU_used': False}))


if __name__ == '__main__':
    main()
