"""Local T controls use observed source components, not an anchor-hull dose."""
from __future__ import annotations

import argparse
import copy
import json
from unittest.mock import patch

import numpy as np

import reference_colour_fit as fit


def main(cuda=False):
    shape = (64, 80)
    labels = np.zeros(shape, np.int32)
    labels[4:60, 3:36] = 3
    labels[4:60, 45:78] = 3  # another face/same-material island, no anchors
    guide = np.full((*shape, 3), [.5, .01, -.01], np.float32)
    alpha = (labels > 0).astype(np.float32)
    anchors = []
    for y in range(20, 45, 5):
        for x in range(10, 31, 4):
            anchors.append({'region': 'face', 'xy': [x, y], 'source': [.5, 0, 0],
                'reference': [.5, .03*((x-10)/10-1), .015*((y-20)/10-1)]})
    parameters = fit.fit([])
    parameters['operators'] = {'face': {'matrix': [[1., 0], [0, 1.]], 'bias': [0, 0], 'exposure_ev': 0.}}
    fitted = fit.fit_local_residual(anchors, parameters, (80, 64), source_labels=labels,
        source_lab=guide, source_alpha=alpha)
    field = fitted['local_chroma_residual']['face']
    assert field['application_policy'] == 'source_material_continuation.v1'
    assert field['application_domain']['anchor_components'] == 1
    assert field['application_domain']['source_pixels'] == int(np.count_nonzero(labels[:, :40]))
    # A constant low-frequency trend must remain constant above/beyond the
    # anchor hull on this source component. A disconnected same-owner face
    # still has exactly zero dose.
    constant = copy.deepcopy(field)
    constant['nodes'] = [[.018, .012]]*4
    constant['support_margin_normalized'] = [.025, .025]
    actual = fit._local_colour(constant, shape, (0, 0), (80, 64))
    np.testing.assert_allclose(actual[4:60, 3:36], np.broadcast_to([.018, .012], (56, 33, 2)), atol=1e-8)
    assert not actual[:, 45:].any()
    assert not actual[labels == 0].any()
    full = fit._local_colour(field, (256, 320), (0, 0), (320, 256))
    cropped = fit._local_colour(field, (117, 161), (13, 21), (320, 256))
    np.testing.assert_array_equal(cropped, full[21:138, 13:174])
    broken = copy.deepcopy(field); broken['application_domain']['sha256'] = 'not-the-content'
    try:
        fit._local_colour(broken, shape, (0, 0), (80, 64))
    except ValueError:
        pass
    else:
        raise AssertionError('Corrupt source-component support was accepted')
    unsupported = fit.fit_local_residual(anchors, parameters, (80, 64), source_labels=labels,
        source_lab=guide, source_alpha=np.zeros(shape, np.float32))
    assert not unsupported['local_chroma_residual']
    if cuda:
        from native_cuda_fields import cuda_field_scope, execution
        import torch
        assert torch.cuda.is_available(), 'CUDA was requested but is unavailable'
        with cuda_field_scope():
            gpu = fit._local_colour(field, (256, 320), (0, 0), (320, 256))
            assert execution()['local_field_calls'] == 1
        np.testing.assert_allclose(gpu, full, atol=2e-8, rtol=2e-6)
        assert np.array_equal(gpu == 0, full == 0)
    print(json.dumps({'ok': True, 'anchor_hull_not_application_boundary': True,
        'unobserved_same_owner_island_exact_zero': True, 'native_crop_exact': True,
        'domain_bytes_verified': True, 'only_source_component_policy_supported': True,
        'CUDA_compared': bool(cuda)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--cuda', action='store_true')
    main(parser.parse_args().cuda)
