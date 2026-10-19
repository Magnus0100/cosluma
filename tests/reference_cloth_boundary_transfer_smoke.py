"""Offline algorithm counterexamples; these do not grant artistic acceptance."""
from __future__ import annotations
import json
import sys
from unittest.mock import patch
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation
import reference_cloth_boundary_transfer as producer
from reference_cloth_boundary_transfer import prepare_cloth_boundary_light


def fixture():
    h, w = 192, 192; y, x = np.indices((h, w), dtype=np.float32)
    a = np.zeros((h, w), np.float32); a[24:168, 24:168] = 1
    labels = np.where(a > 0, 23, 0).astype(np.uint8)
    src = a[..., None]*np.asarray([.03, .025, .02], np.float32)
    src *= (1+.05*np.cos(y/2))[..., None]
    valid = (a == 1)&(x >= 50)&(x < 145)&(y >= 50)&(y < 145)
    maps = {'ref_x': x, 'ref_y': y, 'valid': valid, 'confidence': valid.astype(np.float32)}
    ref = src.copy(); ref[(x >= 150)&(a == 1)] += [.12, .05, .015]
    return src, a, ref, labels, maps, x, y


def slanted_counterexample():
    # A 192px categorical reference with a sloping boundary is observed on a
    # 512px source whose alpha is smooth at native scale. Strict nearest-owner
    # and fully-opaque gates make visible steps unless dose fades inside them.
    rh = rw = 192; nh = nw = 512
    y, x = np.indices((rh, rw), dtype=np.float32); edge = 133+.23*(y-24)
    ra = ((x >= 24)&(y >= 24)&(y < 168)&(x < edge)).astype(np.float32)
    rl = np.where(ra > 0, 23, 0).astype(np.uint8)
    ref = ra[..., None]*np.asarray([.03, .025, .02], np.float32)
    ref[(ra > 0)&(edge-x < 18)] += [.12, .05, .015]
    ny, nx = np.indices((nh, nw), dtype=np.float32)
    rx, ry = (nx+.5)*rw/nw-.5, (ny+.5)*rh/nh-.5
    native_edge = 133+.23*(ry-24)
    a = (np.clip((native_edge-rx)*nw/rw+.5, 0, 1)
         *((rx >= 24)&(ry >= 24)&(ry < 168))).astype(np.float32)
    sl = producer._sample(rl, rx, ry, True).astype(np.uint8)
    src = np.broadcast_to(np.asarray([.03, .025, .02], np.float32), (nh, nw, 3)).copy()
    p = src*a[..., None]
    valid = (sl == 23)&(distance_transform_edt(a >= .5)*rw/nw > 30)
    maps = {'ref_x': rx, 'ref_y': ry, 'valid': valid, 'confidence': valid.astype(np.float32)}
    args = (p, a, ref, ra, sl, rl, maps)
    smooth = prepare_cloth_boundary_light(*args, source_rgb=src)
    # Restore the previous categorical dose gate for a controlled ablation.
    # Source colour is constant, so the material taper does not affect this
    # comparison. Every strict source/reference ownership gate stays active.
    original = producer._inward_weight
    producer._inward_weight = lambda distance, width: np.ones_like(distance, np.float32)
    try: hard = prepare_cloth_boundary_light(*args, source_rgb=src)
    finally: producer._inward_weight = original
    hard_luma, smooth_luma = hard['delta_p']@producer.LUMA, smooth['delta_p']@producer.LUMA
    border = hard['support']&binary_dilation(~hard['support'])
    selected = border&(hard_luma > .02)
    interior = (hard_luma > .02)&~binary_dilation(border, iterations=8)
    assert selected.sum() > 100 and interior.any()
    ratios = np.quantile(smooth_luma[selected]/hard_luma[selected], [.1, .5, .9, 1])
    retained = float(np.median(smooth_luma[interior]/hard_luma[interior]))
    assert ratios[1] < .1 and ratios[2] < .15 and ratios[3] < .25
    assert retained >= .9
    assert not smooth['support'][sl != 23].any()
    assert not smooth['support'][a < .95].any()
    assert np.array_equal(smooth['alpha'], a)
    return {'native_size': [nw, nh], 'reference_size': [rw, rh],
            'hard_boundary_samples': int(selected.sum()),
            'boundary_dose_ratio_q10_q50_q90_max': ratios.tolist(),
            'interior_warm_surface_dose_retained_median_ratio': retained,
            'source_owner_and_alpha_domain_unchanged': True}


def main():
    # A fractional lattice's rectangle prefilter may reject only windows
    # with fewer than 32 possible domain samples; weights/robust median still
    # make the final qualification. Include clipped frame edges and holes.
    rng=np.random.default_rng(377)
    domain=rng.random((67,83))>.75
    xs=np.linspace(0,82,11);ys=np.linspace(0,66,9);radius=6.3
    possible=producer._window_sample_candidates(domain,xs,ys,radius)
    for j,yy in enumerate(ys):
        for i,xx in enumerate(xs):
            left,top=max(0,int(np.floor(xx-radius))),max(0,int(np.floor(yy-radius)))
            right,bottom=min(83,int(np.ceil(xx+radius))+1),min(67,int(np.ceil(yy+radius))+1)
            assert possible[j,i] == (np.count_nonzero(domain[top:bottom,left:right])>=32)
    size=512
    yy,xx=np.indices((size,size),dtype=np.float32)
    own=np.zeros((size,size),bool);own[32:-32,32:-32]=True
    depth=distance_transform_edt(own);edge=own&(depth<=24);core=own&(depth>=24)
    affinity=(.8+.1*np.cos(xx/71)*np.cos(yy/83))*own
    paired=rng.uniform(-.03,.03,(size,size,3)).astype(np.float32)
    paired[(xx>size-65)&own]+=[.12,.05,.015]
    serial=producer._field(paired,affinity,edge,core,1.,workers=1)
    parallel=producer._field(paired,affinity,edge,core,1.,workers=2)
    for a,b in zip(serial[:5],parallel[:5]):
        assert np.array_equal(a.view(np.uint32),b.view(np.uint32))
    assert serial[-1] == parallel[-1]
    src, a, ref, labels, maps, x, y = fixture(); checks = []
    def run(p=src, alpha=a, reference=ref, ra=a, sl=labels, rl=labels,
            source_rgb=src, registration=maps, **kw):
        return prepare_cloth_boundary_light(p, alpha, reference, ra, sl, rl, registration,
                                            source_rgb=source_rgb, **kw)
    def empty(name, **kw):
        result = run(**kw)
        assert not result['support'].any(), name
        assert result['receipt']['state'] == 'evidence_limited', name
        checks.append(name)
    positive = run(); assert positive['support'].any()
    assert positive['receipt']['state'] == 'prepared'
    assert positive['receipt']['changed_pixels'] == int(np.any(positive['delta_p'] != 0, -1).sum())
    assert positive['delta_p'].dtype == np.float32
    # Low-frequency coefficients may spread into the nearby shoulder surface;
    # the unlit opposite half must not acquire a complete boundary rim.
    assert not positive['support'][x < 96].any()
    assert np.array_equal(positive['alpha'], a)
    assert np.all(positive['delta_p'] >= 0) and np.all(positive['after_p'] <= a[..., None]+2e-5)
    checks.append('positive_local_warm_cloth_coefficients_and_exact_float32_receipt')
    empty('no_reference_foreground_light', reference=src)
    empty('already_consumed_light', p=ref)
    for name, colour in [('uniform_warm_colour_offset', [.08, .04, .01]),
                         ('uniform_brightness_offset', [.06, .06, .06])]:
        uniform = src.copy(); uniform[a == 1] += colour
        empty(name, reference=uniform)
    pillar = src.copy(); pillar[(a == 0)&(x > 170)] = [.8, .5, .2]
    empty('bright_background_pillar_is_not_cloth_evidence', reference=pillar)
    empty('independent_reference_semantics_required', rl=np.zeros_like(labels))
    empty('other_source_owner_is_excluded', sl=np.where(a > 0, 13, 0).astype(np.uint8))
    empty('missing_valid_registration_anchors', registration={**maps, 'valid': np.zeros_like(a, bool)})
    empty('wrong_affine_geometry', registration={**maps, 'ref_x': x+40})
    shifted_alpha = np.roll(a, -40, axis=1)
    shifted_labels = np.where(shifted_alpha > 0, 23, 0).astype(np.uint8)
    empty('wrong_independent_reference_silhouette', ra=shifted_alpha, rl=shifted_labels)
    protected = np.zeros_like(a); protected[(x >= 150)&(y < 100)] = 1
    validity = np.ones_like(a); validity[(x >= 150)&(y >= 100)] = 0
    guarded = run(validity=validity, identity_protection=protected)
    assert not guarded['support'][protected >= .5].any()
    assert not guarded['support'][validity < .8].any()
    checks.append('native_validity_and_identity_protection_are_delivery_guards')
    gold = src.copy(); trim = (x >= 160)&(x < 164)&(a == 1)
    gold[trim] = [.4, .22, .04]; goldref = gold.copy()
    goldref[(x >= 150)&(a == 1)] += [.12, .05, .015]
    golden = run(p=gold, source_rgb=gold, reference=goldref)
    assert golden['support'].any() and not golden['support'][trim].any()
    empty('existing_gold_trim_is_not_missing_light', p=gold, source_rgb=gold, reference=gold)
    checks.append('bright_intrinsic_gold_trim_does_not_receive_black_cloth_dose')
    partial = a.copy(); partial[(x == 167)&(a == 1)] = .35
    partial_result = run(p=src*partial[..., None], alpha=partial)
    assert np.array_equal(partial_result['alpha'], partial)
    assert not partial_result['support'][partial < .95].any()
    checks.append('partial_source_alpha_is_unsupported_and_exact_zero')
    rpartial = a.copy(); rpartial[(x >= 150)&(a == 1)] = .35
    empty('partial_reference_foreground_does_not_supply_coefficients', ra=rpartial,
          reference_fp=ref*rpartial[..., None], reference_fb_support=np.ones_like(a))
    opaque_fp = ref*a[..., None]
    observed = run(reference_fp=opaque_fp, reference_fb_support=np.zeros_like(a))
    assert observed['support'].any()
    checks.append('opaque_reference_fp_does_not_need_partial_FB_support')
    thin = src.copy(); thin[(x == 160)&(a == 1)] += [.3, .15, .04]
    empty('thin_reference_invented_line_is_not_a_coarse_colour_donor', reference=thin)
    # Native delivery uses a finer grid than the fixed reference observation.
    # High-frequency source-only weave must survive even though coefficients
    # and material clustering are fitted at reference sampling scale.
    factor = 3
    expand = lambda z: np.repeat(np.repeat(z, factor, axis=0), factor, axis=1)
    na, nl, ns = expand(a), expand(labels), expand(src)
    ny, nx = np.indices(na.shape, dtype=np.float32)
    ns *= np.where((ny.astype(np.int32)%2) == 0, 1.06, .94)[..., None]
    nvalid = expand(maps['valid'])
    nmaps = {'ref_x': (nx+.5)/factor-.5, 'ref_y': (ny+.5)/factor-.5,
             'valid': nvalid, 'confidence': nvalid.astype(np.float32)}
    empty_bands = []
    empty_gate = producer._empty_coefficient_band
    def record_empty(response, confidence):
        empty = empty_gate(response, confidence)
        empty_bands.append(bool(empty))
        return empty
    with patch.object(producer, '_empty_coefficient_band', side_effect=record_empty):
        native = run(p=ns, alpha=na, sl=nl, source_rgb=ns, registration=nmaps)
    with patch.object(producer, '_empty_coefficient_band', return_value=False):
        unskipped = run(p=ns, alpha=na, sl=nl, source_rgb=ns, registration=nmaps)
    assert any(empty_bands) and not all(empty_bands)
    for key in ('delta_p', 'after_p', 'alpha'):
        assert np.array_equal(native[key].view(np.uint32), unskipped[key].view(np.uint32))
    assert np.array_equal(native['support'], unskipped['support'])
    assert native['receipt'] == unskipped['receipt']
    assert not empty_gate(np.full((2, 3, 3), np.nan), np.zeros((2, 3)))
    assert not empty_gate(np.zeros((2, 3, 3)), np.full((2, 3), np.nan))
    checks.append('exact_empty_bands_skip_native_work_without_changing_float_bits_or_receipt')
    checks.append('empty_multiplier_does_not_hide_nonfinite_coefficients')
    assert native['support'].any() and np.array_equal(native['alpha'], na)
    assert native['receipt']['changed_pixels'] == int(np.any(native['delta_p'] != 0, -1).sum())
    assert max(native['receipt']['components'][0]['analysis_size']) < max(na.shape)
    checks.append('native_grid_is_rendered_after_reference_scale_coefficient_analysis')
    dose = native['delta_p'][180:360, 465:480, 0]
    pairs = (dose[::2] > 0)&(dose[1::2] > 0)
    assert pairs.any() and np.median(dose[::2][pairs]/dose[1::2][pairs]) > 1.06
    checks.append('native_source_only_texture_phase_modulates_coarse_reference_light')
    # Legacy Fp/alpha defects elsewhere are preserved, never clamped. The
    # anomaly deliberately affects blue while red has room for a new dose;
    # qualification must abstain for the entire RGB pixel.
    legacy = src.copy(); legacy_a = a.copy()
    legacy_a[80, 160] = .975; legacy[80, 160, 2] = .985
    legacy[90, 160, 2] = 1.01
    legacy_a[12, 12] = .1; legacy[12, 12] = [.12, .01, .01]
    legacy_labels = labels.copy(); legacy_labels[12, 12] = 13
    old = run(p=legacy, alpha=legacy_a, sl=legacy_labels)
    bad = np.any(legacy > legacy_a[..., None]+2e-5, -1)
    assert old['support'].any() and bad.sum() == 3
    assert np.all(old['delta_p'][bad] == 0)
    assert np.array_equal(old['after_p'][bad], legacy[bad])
    assert np.array_equal(old['alpha'], legacy_a)
    assert np.all(old['after_p'][old['support']] <= legacy_a[old['support'], None]+2e-5)
    assert old['receipt']['preexisting_unassociated_pixel_count'] == 3
    assert old['receipt']['checks']['preexisting_unassociated_pixels_unchanged']
    checks.append('unrelated_legacy_endpoint_anomalies_are_preserved_without_global_rejection')
    checks.append('one_bad_legacy_channel_disqualifies_whole_RGB_pixel')
    slanted = slanted_counterexample()
    checks.append('512_native_192_reference_sloping_categorical_boundary_stairs_attenuated')
    checks.append('interior_warm_surface_is_retained_after_boundary_taper')
    assert 'cv2' not in sys.modules
    checks.append('desktop_import_does_not_require_cv2')
    print(json.dumps({'passed': True, 'checks': checks, 'check_count': len(checks),
        'positive_pixels': int(positive['support'].sum()), 'slanted_boundary': slanted, 'visual_acceptance': False,
        'actual_model_inferences': 0, 'actual_network_requests': 0}, indent=2))


if __name__ == '__main__': main()
