"""Continuous hair observation/application boundaries and bounded RGB dose."""
from __future__ import annotations

import json
import numpy as np

from reference_boundary_light_transfer import (
    _sample_observation, _source_support, _bounded_dose, _observe, prepare_boundary_light, LUMA)
from reference_boundary_light_transfer_smoke import fixture


def main():
    # The real failure crossed y=.5 of a nearest reference-owner cell despite
    # both source pixels being opaque hair. Associated sampling must have no
    # second nearest gate and must consume qualification only once.
    confidence = np.asarray([[1., 1.], [0., 0.]], np.float32)
    associated = confidence[..., None]*np.asarray([.42, .18, .04], np.float32)
    yy = np.asarray([[.49999, .50001, .99999, 1.]], np.float32)
    xx = np.full_like(yy, .25)
    estimate, weight = _sample_observation(associated, confidence, xx, yy)
    actual = estimate*weight[..., None]
    assert np.max(np.abs(actual[0, 0]-actual[0, 1])) < 1e-5
    assert np.allclose(actual, np.asarray([.42, .18, .04])*weight[..., None], atol=2e-7)
    assert np.all(actual[0, -1] == 0) and np.max(actual[0, -2]) < 1e-5

    labels = np.full((40, 64), 4, np.uint8); labels[:, 40:] = 23
    labels[15:25, 20] = 23; labels[5, 50] = 4  # 1px exclusion and isolated evidence-free island
    alpha = np.ones(labels.shape, np.float32)
    valid = np.ones_like(alpha); protection = np.zeros_like(alpha)
    support = _source_support(labels, alpha, valid, protection, 7.)
    assert not np.any(support[labels != 4]) and support[5, 50] == 0
    assert np.all(support[:, 39:] == 0) and not np.any(support[:, 19:22][15:25])
    # Soft protection uses the same owner, with zero dose at/above .5. The
    # validity zero at .8 is likewise approached continuously from inside.
    for threshold, name in ((.5, 'protection'), (.8, 'validity')):
        values = []
        for amount in (threshold-1e-5, threshold, threshold+1e-5):
            p = np.full_like(alpha, amount) if name == 'protection' else protection
            v = np.full_like(alpha, amount) if name == 'validity' else valid
            values.append(_source_support(labels, alpha, v, p, 7.)[20, 8])
        assert np.max(values) < 1e-6

    # A common RGB multiplier preserves the correction's colour direction.
    # Sweep black/bright pixels, partial coverage, and arbitrarily strong light.
    rng = np.random.default_rng(27762775)
    a = rng.uniform(0, 1, (40, 60)).astype(np.float32)
    p = rng.uniform(0, 1, (40, 60, 3)).astype(np.float32)*a[..., None]
    p[0] = 0; p[1, :, 0] = a[1]
    dose = rng.uniform(0, 8, p.shape).astype(np.float32)
    bounded = _bounded_dose(p, a, dose)
    assert np.isfinite(bounded).all() and np.all(bounded >= 0)
    assert np.all(p+bounded <= a[..., None]+1e-7)
    assert np.all(bounded[0] == 0) and np.all(bounded[1] == 0)
    ev = np.log2(np.divide((p+bounded)@LUMA, p@LUMA, out=np.ones_like(a), where=(p@LUMA)>0))
    assert ev.max() <= 1.5+1e-6
    factors = bounded/dose
    assert np.max(factors.max(-1)-factors.min(-1)) < 1e-7
    near = _bounded_dose(p, a, dose+1e-5)
    assert np.max(np.abs(near-bounded)) < 2e-5
    saturated = np.asarray([[[1., .1, .1]]], np.float32)
    for red in (0., 1e-12, 1e-8):
        corner = _bounded_dose(saturated, np.ones((1, 1), np.float32),
            np.asarray([[[red, .2, .2]]], np.float32))
        assert np.all(corner == 0), 'Saturated zero-dose channel must not create a 0->epsilon jump'

    # Low-frequency lighting survives while a reference checker does not get
    # rectified into apparent positive lighting or imported native texture.
    source, alpha, reference, owners, maps, x, y = fixture()
    reference_owners = owners.copy(); reference_owners[75:90, 110:120] = 23
    actual_deltas = []
    for shift in (.49999, .50001):
        result = prepare_boundary_light(source, alpha, reference, alpha, owners, reference_owners,
            {**maps, 'ref_y': maps['ref_y']+shift})
        assert result['receipt']['components'][0]['registered_contour_fraction'] >= .7
        actual_deltas.append(result['delta_p'][74, 116])
    assert np.min([np.linalg.norm(value) for value in actual_deltas]) > .001
    assert np.max(np.abs(actual_deltas[0]-actual_deltas[1])) < 1e-5, actual_deltas
    flat = np.where(alpha[..., None]>0, .2, 0).astype(np.float32)*np.ones_like(source)
    check = flat.copy(); check[alpha>0] += (((x.astype(int)+y.astype(int))%2)*2-1)[alpha>0, None]*.1
    ca, cw, _ = _observe(check, alpha, owners, 1.)
    fa, fw, _ = _observe(flat, alpha, owners, 1.)
    # Interior comparison avoids a checker's truncated boundary quadrature.
    interior = (x>45)&(x<110)&(y>35)&(y<110)&(cw>0)
    assert np.max(np.abs(ca[interior]-fa[interior])) < 2e-4
    assert np.array_equal(cw, fw)
    # An existing-source measurement is independent from the boundary
    # application's depth support. It must not vanish at confidence==0.
    response, confidence, _ = _observe(reference, alpha, owners, 1., associated=False)
    associated, _, _ = _observe(reference, alpha, owners, 1.)
    assert np.allclose(associated, response*confidence[..., None], atol=1e-7)
    assert np.isfinite(response).all()
    print(json.dumps({'ok': True, 'associated_owner_crossing_continuous': True,
        'source_garment_identity_and_1px_island_fail_closed': True,
        'reference_checker_not_light_or_texture': True, 'common_RGB_EV_and_gamut_bounded': True,
        'maximum_EV': float(ev.max()), 'model_inferences': 0, 'visual_acceptance': False}))


if __name__ == '__main__':
    main()
