"""Check request-local Lab reuse and zero-material weight contracts."""
from __future__ import annotations

import copy
import dataclasses
from unittest.mock import patch

import numpy as np

import reference_colour_fit as matching
from reference_material_palette import fit_palette


def main():
    rng = np.random.default_rng(20261014)
    source = rng.uniform(.12, .58, (120, 120, 3)).astype(np.float32)
    reference = rng.uniform(.14, .62, (120, 120, 3)).astype(np.float32)
    labels = np.full(source.shape[:2], 3, np.uint8)
    alpha = np.ones(labels.shape, np.float32)
    protection = np.zeros(labels.shape, np.float32)
    anchors, _ = matching.collect_anchors(source, source, labels, labels, alpha,
                                          protection, protection, alpha)
    params = matching.fit(anchors)
    source_obs = matching.prepare_lab_observation(source)
    reference_obs = matching.prepare_lab_observation(reference)
    identity_obs = matching.prepare_lab_observation(source)

    # Reusing the exact request-local bindings retains the old results.
    cached_anchors, _ = matching.collect_anchors(
        source, source, labels, labels, alpha, protection, protection, alpha,
        source_observation=source_obs, reference_observation=identity_obs)
    assert cached_anchors == matching.collect_anchors(
        source, source, labels, labels, alpha, protection, protection, alpha)[0]
    uncached = matching.collect_material_distributions(
        source, reference, labels, labels, alpha, protection, protection)
    cached = matching.collect_material_distributions(
        source, reference, labels, labels, alpha, protection, protection,
        source_observation=source_obs, reference_observation=reference_obs)
    assert cached == uncached

    # Guarded weights may use the bound source RGB without a second transform;
    # a same-shaped Lab array for a different RGB object is rejected.
    guarded = copy.deepcopy(params)
    guarded['operators']['face']['source_material_guard'] = {
        'minimum_L': .12, 'transition_L': .02}
    uncached_weights = matching.region_weights(labels, protection, guarded, source_values=source)
    cached_weights = matching.region_weights(labels, protection, guarded,
                                             source_values=source,
                                             source_observation=source_obs)
    assert np.array_equal(cached_weights, uncached_weights)
    implicit_source_weights = matching.region_weights(labels, protection, guarded,
                                                      source_observation=source_obs)
    assert np.array_equal(implicit_source_weights, uncached_weights)
    uncached_apply = matching.apply(source, uncached_weights, guarded)
    cached_apply = matching.apply(source, cached_weights, guarded,
                                  values_observation=source_obs)
    assert np.array_equal(cached_apply, uncached_apply)
    other = source.copy()
    try:
        matching.region_weights(labels, protection, guarded, source_values=other,
                                source_observation=source_obs)
    except ValueError:
        pass
    else:
        raise AssertionError('Lab cache must reject a same-shaped alias/copy with different identity')
    writable_lab = source_obs.lab.copy()
    writable = matching.LabObservation(source, writable_lab)
    try:
        matching.region_weights(labels, protection, guarded, source_values=source,
                                source_observation=writable)
    except ValueError:
        pass
    else:
        raise AssertionError('Writable cached Lab must be rejected')
    forged = dataclasses.replace(source_obs, lab=source_obs.lab.copy())
    forged.lab.setflags(write=False)
    try:
        matching.region_weights(labels, protection, guarded, source_values=source,
                                source_observation=forged)
    except ValueError:
        pass
    else:
        raise AssertionError('Replacing a cached Lab payload must be rejected')
    try:
        matching.region_weights(labels, protection, guarded, source_values=source,
                                source_observation=matching.prepare_lab_observation(source[:1]))
    except ValueError:
        pass
    else:
        raise AssertionError('Mismatched Lab shape must be rejected')

    # Only channels already forced to zero may skip Gaussian. Valid palette
    # support and normalized stack semantics remain unchanged.
    palette_params = copy.deepcopy(matching.fit([]))
    palette_params['automatic_material_palette'] = {
        'schema': 'reference_material_palette.v1', 'working_domain': 'display_linear_sRGB',
        'region_order': list(matching.REGIONS), 'settings': __import__('reference_material_palette').SETTINGS,
        'changes_lightness': False, 'geometry_warp': False, 'reference_pixels_in_output': False,
        'regions': {name: {'status': 'identity'} for name in matching.REGIONS}}
    calls = []
    original_gaussian = matching.gaussian_filter
    def observed(values, *args, **kwargs):
        calls.append((values.shape, int(np.count_nonzero(values))))
        return original_gaussian(values, *args, **kwargs)
    with patch.object(matching, 'gaussian_filter', observed):
        zero_weights = matching.region_weights(labels, protection, palette_params,
                                               source_observation=source_obs)
    assert len(calls) == 0, 'All identity materials should skip zero Gaussian planes'
    assert zero_weights.dtype == np.float32 and not zero_weights.any()
    assert np.all(zero_weights.sum(-1) <= 1 + 1e-6)

    # An active material still takes the mature SciPy path.
    active = copy.deepcopy(palette_params)
    active['operators']['face'] = {'matrix': np.eye(2).tolist(), 'bias': [0, 0], 'exposure_ev': 0}
    calls.clear()
    with patch.object(matching, 'gaussian_filter', observed):
        active_weights = matching.region_weights(labels, protection, active,
                                                 source_observation=source_obs)
    assert calls and all(nonzero > 0 for _, nonzero in calls)
    assert active_weights.dtype == np.float32 and np.all(active_weights.sum(-1) <= 1 + 1e-6)

    # Palette producer accepts the same source/native-reference bindings and
    # preserves its JSON-safe result; no cross-request state is involved.
    uncached_palette = fit_palette(source, reference, labels, labels, alpha, params)
    cached_palette = fit_palette(source, reference, labels, labels, alpha, params,
                                 source_observation=source_obs,
                                 reference_observation=reference_obs)
    assert cached_palette == uncached_palette
    palette_params = copy.deepcopy(params)
    palette_params['automatic_material_palette'] = cached_palette
    palette_weights = matching.region_weights(labels, protection, palette_params,
                                              source_values=source,
                                              source_observation=source_obs)
    baseline_palette_apply = matching.apply(source, palette_weights, palette_params,
                                            chroma_weights=palette_weights)
    cached_palette_apply = matching.apply(source, palette_weights, palette_params,
                                          chroma_weights=palette_weights,
                                          values_observation=source_obs)
    assert np.array_equal(cached_palette_apply, baseline_palette_apply)
    print('PASS: request-local Lab reuse; identity/shape/writable negatives; zero Gaussian only for forced-zero materials; active Gaussian; float32 normalized weights; palette parity')


if __name__ == '__main__':
    main()
