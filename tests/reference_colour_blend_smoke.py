"""Check colour-only fusion, source-coordinate crops and edge propagation."""
import numpy as np

from reference_colour_blend import normalized_colour, apply_colour_layer
from reference_colour_fit import SCHEMA, SETTINGS, REGIONS, from_lab, to_lab
from render_reference_colour_composite import sample_maps, donor_fields, match_foreground


def main():
    mask = np.zeros((30, 20), np.float32); mask[5:25, 4:16] = 1
    colour = np.full((30, 20, 2), [.01, -.02], np.float32)
    colour[mask == 0] = [.9, .8]
    low, support = normalized_colour(colour, mask, 4)
    assert np.max(abs(low[support > .1]-[.01, -.02])) < 1e-6, 'Background colour must not leak through normalized material filtering'
    source = np.broadcast_to(from_lab(np.array([.45, .025, .03], np.float32)), (90, 60, 3)).copy()
    field = np.zeros((30, 20, 3), np.float32)
    field[5:25, 4:16] = [.004, -.008, .4]
    full = sample_maps(field, (0, 0, 60, 90), (20, 30), (60, 90))
    matched = apply_colour_layer(source, full)
    assert np.array_equal(apply_colour_layer(source, full, strength=0), source)
    assert np.array_equal(matched[:10], source[:10]), 'Outside selection must remain exact'
    assert np.max(abs(to_lab(matched)[..., 0]-to_lab(source)[..., 0])) < 2e-6, 'Spatial reference colour must retain native source lightness'
    box = (11, 17, 47, 72)
    crop_field = sample_maps(field, box, (20, 30), (60, 90))
    assert np.array_equal(crop_field, full[17:72, 11:47])
    assert np.array_equal(apply_colour_layer(source[17:72, 11:47], crop_field), matched[17:72, 11:47]), 'Native ROI cannot refilter or renormalize the reference layer'
    split = np.zeros((16,16,3),np.float32); split[:,:8] = [.04,0,1]
    owners = np.ones((16,16),np.int32);owners[:,8:] = 10
    separated = sample_maps(split,(0,0,64,64),(16,16),(64,64),owners=owners)
    assert not separated[:,32:,:2].any(), 'Native field interpolation cannot cross the observed material boundary'
    assert np.allclose(separated[:,:32,0],.04)
    piece = sample_maps(split,(19,7,49,52),(16,16),(64,64),owners=owners)
    assert np.array_equal(piece,separated[7:52,19:49])
    white = np.ones((2,3,3),np.float32)
    tint = np.full_like(white,[.03,.02,1.])
    bounded = apply_colour_layer(white,tint)
    assert np.max(abs(to_lab(bounded)[...,0]-to_lab(white)[...,0])) < 2e-6, 'Gamut must shrink the correction instead of clipping white'
    assert np.max(abs(bounded-white)) < 1e-5
    # Deep shadow channels expose lightness loss from a fixed RGB tolerance
    # followed by clipping. Cover multiple magnitudes and channel orderings.
    levels = np.geomspace(1e-5, .05, 32).astype(np.float32)
    dark = np.stack([levels, levels*.015, levels*.7], axis=-1)[None]
    dark = np.concatenate([dark, np.roll(dark, 1, axis=-1), np.zeros_like(dark)], axis=0)
    dark_field = np.broadcast_to(np.array([.028, -.018, 1], np.float32), dark.shape).copy()
    shifted = apply_colour_layer(dark, dark_field)
    assert np.max(abs(to_lab(shifted)[...,0]-to_lab(dark)[...,0])) < 4e-6
    assert np.array_equal(shifted[2], dark[2]), 'Zero-dose gamut fallback preserves original black exactly'
    assert shifted.min() >= 0 and shifted.max() <= 1
    parameters = {'schema': SCHEMA, 'working_domain': 'display_linear_sRGB', 'settings': SETTINGS,
        'operators': {'face': {'matrix': np.eye(2).tolist(), 'bias': [0, 0], 'exposure_ev': 0}}}
    proxy = source[:30, :20].copy()
    weights = np.zeros((30, 20, len(REGIONS)), np.float32); weights[..., 0] = 1
    alpha = np.ones((30, 20), np.float32)
    reference = np.broadcast_to(np.array([.004, -.008, .4], np.float32), proxy.shape).copy()
    donors = donor_fields(proxy, alpha, weights, parameters, reference)
    expected = apply_colour_layer(proxy, reference)
    low_alpha = alpha*.05; fp = proxy*low_alpha[..., None]
    edge = match_foreground(fp, low_alpha, weights, donors, alpha, parameters,
        origin=(0, 0), canvas_size=(20, 30), reference_blend=reference)
    assert np.max(abs(edge-low_alpha[..., None]*expected)) < 2e-6, 'Low-alpha edges must receive the final fused donor colour without division'
    assert np.array_equal(alpha, np.ones_like(alpha)) and np.array_equal(low_alpha, np.full_like(alpha, .05))
    empty = match_foreground(fp*0, alpha*0, weights, donors, alpha, parameters,
        origin=(0, 0), canvas_size=(20, 30), reference_blend=reference)
    assert not empty.any(), 'Reference colour cannot create foreground at zero alpha'
    print('PASS: masked lowpass isolation, zero/selection identity, source lightness, native ROI parity, fused edge donors, alpha preservation')


if __name__ == '__main__':
    main()
