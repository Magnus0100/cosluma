"""Actual CUDA field ownership, bounded chroma and finite output checks."""
import json
import sys
from unittest.mock import patch
import numpy as np
from native_cuda_fields import cuda_field_scope, eligible, execution, local_colour, palette
from reference_material_palette import SETTINGS


def main():
    with patch.dict(sys.modules, {'torch': None}):
        with cuda_field_scope():
            assert not eligible((17, 31))
            assert execution()['device'] is None
    shape = (192, 256)
    assert not eligible(shape)
    field = {'bounds_normalized': [[.2, .2], [.8, .8]],
        'nodes': [[.01, 0]]*4, 'support_margin_normalized': [.02, .02],
        'support_hull_equations': [[-1, 0, .2], [1, 0, -.8], [0, -1, .2], [0, 1, -.8]],
        'strength': 1.}
    from current_stack_fixture import bind_local_domain
    mask = np.zeros(shape, bool); mask[39:154, 52:205] = True
    bind_local_domain(field, (256,192), mask=mask)
    values = np.full((*shape, 3), .3, np.float32)
    weights = np.zeros((*shape, 1), np.float32)
    weights[:, :128] = 1
    model = {'center': [.6, 0, 0], 'scale': [1, 1, 1], 'centers': [[0, 0, 0]],
        'accepted_groups': [True], 'source_support_low': [0, -1, -1],
        'source_support_high': [1, 1, 1], 'nodes': [[.01, 0]]}
    with cuda_field_scope():
        colour = local_colour(field, shape, (0, 0), (256, 192))
        assert np.isfinite(colour).all() and colour.shape == (*shape, 2)
        assert np.max(np.abs(colour[:, :40])) == 0 and colour[96, 128, 0] > .009
        result = palette(values, values, weights, np.ones(np.prod(shape), bool),
            [(0, {'model': model})], 1., SETTINGS)
        assert np.isfinite(result).all() and result.min() >= 0 and result.max() <= 1
        assert np.array_equal(result[:, 128:], values[:, 128:])
        assert np.any(result[:, :128] != values[:, :128])
        from colour_space import to_lab
        assert np.max(np.abs(to_lab(result)[..., 0]-to_lab(values)[..., 0])) < 1e-4
        from reference_appearance_controls import corner_targets, blend_delta
        alpha = np.ones(shape, np.float32)
        alpha[:, 192:] = 0
        base = values*alpha[..., None]
        target = result*alpha[..., None]
        corners, doses, supported = corner_targets(base, target, alpha)
        for value in corners.values():
            assert np.isfinite(value).all() and value.min() >= 0
            assert np.all(value <= alpha[..., None])
            assert np.array_equal(value[~supported], base[~supported])
        full = target-base
        assert np.array_equal(blend_delta(full, corners['colour_delta']-base,
            corners['lightness_delta']-base, 1, 1), full)
        facts = execution()
        assert facts['palette_calls'] == 1 and facts['local_field_calls'] == 1
        assert facts['control_calls'] == 1
    assert not eligible(shape)
    print(json.dumps({'ok': True, 'source_ownership': True, 'lightness_preserved': True,
        'finite_SDR': True, 'CUDA_equality_gate': False, 'execution': facts}))


if __name__ == '__main__':
    main()
