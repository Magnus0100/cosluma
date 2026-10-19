"""Exact proxy copying retains the original owner, visibility and grid contract."""
from __future__ import annotations
import json
from unittest.mock import patch
import numpy as np
import render_reference_colour_composite as renderer


def slow(values, box, proxy_size, canvas_size, **kwargs):
    with patch.object(renderer, '_sample_identity_grid', return_value=None):
        return renderer.sample_maps(values, box, proxy_size, canvas_size, **kwargs)


def observed(function):
    try:
        values = function()
        return ('values', values.dtype.str, values.shape, values.tobytes())
    except (ValueError, IndexError) as error:
        return ('error', type(error).__name__, str(error))


def main():
    rng = np.random.default_rng(67201)
    size = (31, 27)
    fields = rng.normal(0, .025, (size[1], size[0], 6)).astype(np.float32)
    fields[2, 3] = np.float32(-0.)
    owners = np.ones(fields.shape[:2], np.int32)
    owners[:, 5:11] = 4; owners[:, 11:17] = 1004
    cases = 0
    for dtype in (np.float32, np.float64):
        for box in ((0, 0, 31, 27), (3, 4, 29, 21)):
            values = fields.astype(dtype)
            shape = box[3]-box[1], box[2]-box[0]
            grid = renderer._sample_grid(box, size, size)
            plan = renderer._sample_plan(grid, values.shape[:2])
            support = rng.uniform(0, 1, shape).astype(np.float32)
            for use_plan in (False, True):
                kwargs = {'owners': owners, 'grid': grid if use_plan else None,
                    'sample_plan': plan if use_plan else None, 'hair_face_support': support}
                expected = slow(values, box, size, size, **kwargs)
                with patch.object(renderer, 'sample_scoped_maps', side_effect=AssertionError('Identity used GPU dispatch')):
                    actual = renderer.sample_maps(values, box, size, size, **kwargs)
                assert expected.tobytes() == actual.tobytes()
                cases += 1
    box = (3, 4, 29, 21)
    grid = renderer._sample_grid(box, size, size)
    plan = renderer._sample_plan(grid, fields.shape[:2])
    variants = [
        {'grid': (grid[0]+np.float32(.25), grid[1])},
        {'grid': (grid[0], grid[1]+np.float32(.25))},
        {'sample_plan': {**plan, 'values_shape': (2, 2)}},
        {'sample_plan': {**plan, 'weights': (plan['weights'][0]*.5, *plan['weights'][1:])}},
        {'sample_plan': {**plan, 'indices': (*plan['indices'][:3], np.full(grid[0].shape, 9999, np.int32))}},
        {'sample_plan': {**plan, 'owner_indices': plan['owner_indices'].astype(np.float32)}},
        {'owners': owners.astype(np.float32)}, {'owners': owners[:-1]},
        {'hair_face_support': np.full((17, 26), np.nan, np.float32)},
        {'hair_face_support': np.full((17, 26), 1.1, np.float32)},
        {'hair_face_support': np.ones((3, 4), np.float32)},
    ]
    for extra in variants:
        kwargs = {'owners': owners, 'grid': grid, 'sample_plan': plan, **extra}
        assert observed(lambda: slow(fields, box, size, size, **kwargs)) == observed(
            lambda: renderer.sample_maps(fields, box, size, size, **kwargs))
        assert renderer._sample_identity_grid(fields, box, size, size,
            kwargs['owners'], kwargs['grid'], kwargs.get('hair_face_support'), kwargs['sample_plan']) is None
        cases += 1
    canvas = (62, 54)
    assert observed(lambda: slow(fields, box, size, canvas, owners=owners)) == observed(
        lambda: renderer.sample_maps(fields, box, size, canvas, owners=owners))
    assert renderer._sample_identity_grid(fields, box, size, canvas, owners, None, None, None) is None
    print(json.dumps({'passed': True, 'cases': cases+1,
        'integer_original_grid_and_all_indices_inbounds_proved': True,
        'nonzero_ROI_owner_visibility_signedzero_widerdtype_exact': True,
        'wrong_grid_size_owner_support_and_plan_use_original_contract': True,
        'identity_CPU_before_GPU_dispatch': True, 'models_GPU_used': False}))


if __name__ == '__main__':
    main()
