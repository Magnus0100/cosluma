"""Adversarial offline checks; passing does not grant artistic acceptance."""
from __future__ import annotations

import json
import numpy as np
from scipy.ndimage import distance_transform_edt
from unittest.mock import patch

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
    source, alpha, reference, labels, maps, x, y = fixture()
    def run(p=source, a=alpha, ref=reference, rl=labels, **kw):
        return prepare_boundary_light(p, a, ref, alpha, labels, rl, maps, **kw)
    positive = run()
    assert positive["support"].any()
    assert not positive["support"][x < 110].any()
    assert np.array_equal(positive["alpha"], alpha)
    assert np.all(positive["delta_p"] >= 0)
    assert not np.any(positive["delta_p"][alpha == 0])

    # The 3x3 native median suppresses one-pixel source sparkle before the
    # detail ratio is formed. Keep this as an algorithmic counterexample, not
    # an artistic acceptance claim.
    noisy = source.copy(); noisy[70, 117] += [.03, .01, .002]
    filtered = run(p=noisy)
    with patch('reference_boundary_light_transfer.median_filter', side_effect=lambda values, size: values.copy()):
        unfiltered = run(p=noisy)
    assert filtered['support'][70, 117] and unfiltered['support'][70, 117]
    assert float(np.linalg.norm(filtered['delta_p'][70, 117])) < float(np.linalg.norm(unfiltered['delta_p'][70, 117]))
    assert float(np.linalg.norm(unfiltered['delta_p'][70, 117])) > 1e-3

    assert not run(ref=source)["support"].any()
    assert not run(p=reference)["support"].any()  # already consumed highlight
    uniform = np.where(alpha[..., None] > 0, source + [.20, .10, .04], 0).astype(np.float32)
    assert not run(ref=uniform)["support"].any()
    # A bright background pillar with unchanged reference foreground supplies
    # no person-owned highlight evidence, irrespective of its shape/brightness.
    pillar = source.copy(); pillar[(alpha == 0) & (x > 125)] = [.8, .5, .2]
    assert not run(ref=pillar)["support"].any()
    assert not run(rl=np.zeros_like(labels))["support"].any()

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

    print(json.dumps({"passed": True, "checks": 14,
        "positive_pixels": int(positive["support"].sum()),
        "boundary_correspondence_separate_from_interior_valid": True,
        "visual_acceptance": False, "production_integrated": False,
        "median_3x3_sparkle_counterexample": True,
        "actual_model_inferences": 0, "actual_network_requests": 0,
        "visual_acceptance": False}))


if __name__ == "__main__":
    main()
