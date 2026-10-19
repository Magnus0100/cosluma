"""Adversarial offline checks; passing does not grant artistic acceptance."""
from __future__ import annotations

import json
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from unittest.mock import patch

import reference_boundary_light_transfer as producer
from reference_boundary_light_transfer import prepare_boundary_light


def fixture(owner=4):
    h, w = 144, 160
    y, x = np.indices((h, w), dtype=np.float32)
    alpha = np.zeros((h, w), np.float32)
    alpha[25:120, 35:120] = 1
    labels = np.zeros((h, w), np.uint8)
    labels[alpha > 0] = owner
    source = alpha[..., None] * np.asarray([.02, .015, .012], np.float32)
    # Every boundary pixel has invalid DINO support. Only interior anchors
    # register the images; boundary measurements use a separate domain.
    valid = (alpha == 1) & (x >= 50) & (x < 105) & (y >= 40) & (y < 105)
    maps = {"ref_x": x, "ref_y": y, "valid": valid, "confidence": valid.astype(np.float32)}
    reference = source.copy()
    reference[(x >= 115) & (alpha > 0)] += [.30, .12, .04]
    return source, alpha, reference, labels, maps, x, y


def main():
    # The native channel decomposition must retain SciPy's pass order and
    # float32 rounding, including reflected image edges and strided input.
    rng = np.random.default_rng(20261013)
    native_rgb = rng.random((1001, 2000, 3), dtype=np.float32)[:, ::2]
    native_domain = rng.random(native_rgb.shape[:2]) > .3
    for sigma, supplied_weight in ((.625, False), (5.5, True)):
        weight = gaussian_filter(native_domain.astype(np.float32), sigma)
        expected = gaussian_filter(np.where(native_domain[..., None], native_rgb, 0),
                                   (sigma, sigma, 0))/np.maximum(weight[..., None], 1e-6)
        actual = producer._low(native_rgb, native_domain, sigma,
                               weight=weight if supplied_weight else None)
        assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32))
    del native_rgb, native_domain, weight, expected, actual

    source, alpha, reference, labels, maps, x, y = fixture()
    def run(p=source, a=alpha, ref=reference, rl=labels, **kw):
        return prepare_boundary_light(p, a, ref, alpha, labels, rl, maps, **kw)
    positive = run()
    assert positive["support"].any()
    assert not positive["support"][x < 105].any()  # bounded 3px illumination envelope, not a strand donor
    assert np.array_equal(positive["alpha"], alpha)
    assert np.all(positive["delta_p"] >= 0)
    assert not np.any(positive["delta_p"][alpha == 0])

    # An adversarial reference checker has no low-frequency lighting contrast.
    # Rectifying its positive strands before filtering would create a donor.
    checker = np.zeros_like(reference); checker[alpha > 0] = .2
    checker[alpha > 0] += (((x.astype(int)+y.astype(int))%2)*2-1)[alpha > 0, None]*.1
    assert not run(ref=checker)['support'].any()
    assert positive['receipt']['maximum_applied_exposure_EV'] <= 1.5+1e-6
    assert positive['receipt']['reference_texture_imported'] is False

    assert not run(ref=source)["support"].any()
    assert not run(p=reference)["support"].any()  # already consumed highlight
    uniform = np.where(alpha[..., None] > 0, source + [.20, .10, .04], 0).astype(np.float32)
    assert not run(ref=uniform)["support"].any()
    # A bright background pillar with unchanged reference foreground supplies
    # no person-owned highlight evidence, irrespective of its shape/brightness.
    pillar = source.copy(); pillar[(alpha == 0) & (x > 125)] = [.8, .5, .2]
    assert not run(ref=pillar)["support"].any()
    assert not run(rl=np.zeros_like(labels))["support"].any()

    # Registered reference geometry can pass while providing no highlight.
    # Keep its zero increment and complete refusal receipt, without filtering
    # the native source at the reference-to-source scale.
    with patch.object(producer, '_observe', wraps=producer._observe) as observe:
        empty = run(ref=np.zeros_like(reference))
    assert observe.call_count == 1
    assert empty['receipt']['components'][0]['registration']['state'] == 'qualified'
    assert empty['receipt']['components'][0]['registered_contour_fraction'] >= .7
    assert empty['receipt']['state'] == 'evidence_limited'
    assert not empty['support'].any() and not empty['reference_response'].any()
    assert np.array_equal(empty['after_p'], source)
    assert np.array_equal(empty['alpha'], alpha)

    protected = np.zeros_like(alpha); protected[:, 114:121] = 1
    valid = np.ones_like(alpha); valid[80:95] = 0
    guarded = run(validity=valid, identity_protection=protected)
    assert not guarded["support"][protected > .5].any()
    assert not guarded["support"][valid < .8].any()

    # Warm existing gold trim is not a missing light response.
    gold = reference.copy()
    assert not run(p=gold, ref=gold)["support"].any()
    # Missing material anchors and badly shifted registration are explicit
    # abstentions rather than permission to paint a full contour.
    shifted = {**maps, "ref_x": maps["ref_x"] + 45}
    assert not prepare_boundary_light(source, alpha, reference, alpha, labels, labels, shifted)["support"].any()

    # Partial-alpha foreground is an appearance increment, never an alpha
    # replacement. This check does not certify unknown F/B photometry.
    partial_alpha = alpha.copy(); partial_alpha[40:100, 119] = .35
    partial_source = source * partial_alpha[..., None]
    partial = run(p=partial_source, a=partial_alpha)
    assert np.array_equal(partial["alpha"], partial_alpha)
    assert np.all(partial["after_p"] <= partial_alpha[..., None] + 1e-6)

    print(json.dumps({"passed": True, "checks": 15,
        "positive_pixels": int(positive["support"].sum()),
        "boundary_correspondence_separate_from_interior_valid": True,
        "visual_acceptance": False, "production_integrated": False,
        "reference_checker_does_not_supply_light_or_texture": True,
        "zero_reference_skips_native_response_filter": True,
        "native_parallel_filter_float32_bits_exact": True,
        "actual_model_inferences": 0, "actual_network_requests": 0,
        "visual_acceptance": False}))


if __name__ == "__main__":
    main()
