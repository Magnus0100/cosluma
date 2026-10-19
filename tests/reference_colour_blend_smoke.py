"""Check colour-only fusion, source-coordinate crops and edge propagation."""
import numpy as np
from unittest.mock import patch

from reference_colour_blend import apply_colour_layer, fuse_reference_colour
from reference_colour_fit import SCHEMA, SETTINGS, REGIONS, from_lab, to_lab
from render_reference_colour_composite import sample_maps, donor_fields, match_foreground, _sample_grid, _sample_plan


def main():
    source = np.broadcast_to(from_lab(np.array([.45, .025, .03], np.float32)), (90, 60, 3)).copy()
    field = np.zeros((30, 20, 3), np.float32)
    field[5:25, 4:16] = [.004, -.008, .4]
    full = sample_maps(field, (0, 0, 60, 90), (20, 30), (60, 90))
    matched = apply_colour_layer(source, full)
    assert np.array_equal(apply_colour_layer(source, full, strength=0), source)
    assert np.array_equal(matched[:10], source[:10]), 'Outside selection must remain exact'
    # No correction means exact RGB replay, even for signed zero, while the
    # nonzero-strength operator retains its original float32 output contract.
    untouched = np.full((11, 17, 3), np.float32(-0.0))
    untouched[4, 8] = [.12, .24, .36]
    neutral = np.zeros_like(untouched); neutral[..., 2] = 1
    with patch('reference_colour_blend.to_lab', side_effect=AssertionError('zero-field Lab conversion')):
        unchanged = apply_colour_layer(untouched, neutral)
        double_input = apply_colour_layer(untouched.astype(np.float64), neutral)
    np.testing.assert_array_equal(unchanged.view(np.uint32), untouched.view(np.uint32))
    assert double_input.dtype == np.float32
    sparse = neutral.copy(); sparse[4, 8, :2] = [.004, -.008]
    conversions = []
    def observed_lab(values):
        conversions.append(values.shape)
        return to_lab(values)
    with patch('reference_colour_blend.to_lab', side_effect=observed_lab):
        sparse_result = apply_colour_layer(untouched, sparse)
    assert conversions == [(1, 3)]
    np.testing.assert_array_equal(sparse_result[0].view(np.uint32), untouched[0].view(np.uint32))
    invalid_neutral = neutral.copy(); invalid_neutral[0, 0, 2] = 2
    try:apply_colour_layer(untouched, invalid_neutral)
    except ValueError:pass
    else:raise AssertionError('Zero correction bypassed field validation')
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
    shared_grid = _sample_grid((19,7,49,52),(16,16),(64,64))
    shared_plan = _sample_plan(shared_grid, split.shape[:2])
    assert np.array_equal(piece, sample_maps(split,(19,7,49,52),(16,16),(64,64),
        owners=owners,grid=shared_grid,sample_plan=shared_plan)), 'Shared owner sampling must remain bitwise exact'
    # Hair misclassification cannot turn a visible forehead into a hair target.
    # The support consumes amplitude once after owner sampling; QA coverage is
    # still evidence and the neighbouring face operator remains untouched.
    hair_owners=np.full((16,16),4,np.int32);hair_owners[:,8:]=3
    raw_field=np.full((16,16,3),[.02,-.01,.6],np.float32)
    visible=np.full((64,64),.5,np.float32);visible[16:32,:32]=1
    vetoed=sample_maps(raw_field,(0,0,64,64),(16,16),(64,64),owners=hair_owners,hair_face_support=visible)
    assert np.allclose(vetoed[:16,:32,:2],[.01,-.005],atol=1e-8)
    assert not vetoed[16:32,:32,:2].any()
    assert np.allclose(vetoed[:,32:,:2],[.02,-.01],atol=1e-8)
    np.testing.assert_allclose(vetoed[...,2],.6,atol=1e-7)
    assert np.array_equal(sample_maps(raw_field,(19,7,49,52),(16,16),(64,64),owners=hair_owners,
        hair_face_support=visible[7:52,19:49]),vetoed[7:52,19:49])
    assert np.array_equal(sample_maps(raw_field,(0,0,64,64),(16,16),(64,64),owners=hair_owners,
        hair_face_support=np.zeros_like(visible)),sample_maps(raw_field,(0,0,64,64),(16,16),(64,64),owners=hair_owners))
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
    reference_v2 = np.concatenate((reference, np.zeros_like(reference)), axis=-1)
    donors = donor_fields(proxy, alpha, weights, parameters, reference_v2)
    expected = apply_colour_layer(proxy, reference)
    low_alpha = alpha*.05; fp = proxy*low_alpha[..., None]
    edge = match_foreground(fp, low_alpha, weights, donors, alpha, parameters,
        origin=(0, 0), canvas_size=(20, 30), reference_blend=reference_v2)
    assert np.max(abs(edge-low_alpha[..., None]*expected)) < 2e-6, 'Low-alpha edges must receive the final fused donor colour without division'
    assert np.array_equal(alpha, np.ones_like(alpha)) and np.array_equal(low_alpha, np.full_like(alpha, .05))
    empty = match_foreground(fp*0, alpha*0, weights, donors, alpha, parameters,
        origin=(0, 0), canvas_size=(20, 30), reference_blend=reference_v2)
    assert not empty.any(), 'Reference colour cannot create foreground at zero alpha'
    # The generated person has its own detail and coloured contacts/makeup.
    # Cumulative T+B contamination must not become a new feature colour.
    beauty_lab = np.full((45, 60, 3), [.48, .026, .027], np.float32)
    beauty_lab[..., 0] += .004*np.sin(np.arange(60)[None]*1.7)
    beauty_lab[8:15, 8:18, 1:] = [.013, .061]   # brown iris
    beauty_lab[8:15, 35:45, 1:] = [-.018, -.045] # blue contact
    beauty_lab[30:38, 20:40, 1:] = [.09, .028]   # lipstick
    beauty_pixels = from_lab(beauty_lab).astype(np.float32)
    final_lab = beauty_lab.copy(); final_lab[..., 0] += .04
    final_lab[..., 1:] += [.012, -.022]  # spurious green/blue feature shift
    contaminated = from_lab(final_lab).astype(np.float32)
    protected = np.zeros((45, 60, 6), np.float32)
    protected[4:40, 4:50, 5] = 1
    fixed = fuse_reference_colour(contaminated, protected, source_values=beauty_pixels)
    active = protected[..., 5] > 0
    lab_fixed = to_lab(fixed)
    assert np.max(abs(lab_fixed[..., 0]-to_lab(contaminated)[..., 0])) < 2e-6, 'Final lightness/detail must be retained'
    assert np.max(abs(lab_fixed[active, 1:]-beauty_lab[active, 1:])) < 2e-6, 'Generated contacts/lipstick must retain their chroma'
    assert np.array_equal(fixed[~active], contaminated[~active]), 'The accepted effect outside features must be exact'
    protected[..., 3:5] = [.012, .008]
    protected[4:40, 4:50, 5] = np.linspace(0, 1, 46)[None]
    fixed = fuse_reference_colour(contaminated, protected, source_values=beauty_pixels)
    expected_ab = final_lab[..., 1:] + protected[..., 5, None]*(
        beauty_lab[..., 1:]+beauty_lab[..., :1]*protected[..., 3:5]-final_lab[..., 1:])
    assert np.max(abs(to_lab(fixed)[..., 1:]-expected_ab)) < 2e-6, 'Only smooth supported ambient can tint a feature'
    piece = fuse_reference_colour(contaminated[6:39, 9:52], protected[6:39, 9:52],
                                  source_values=beauty_pixels[6:39, 9:52])
    assert np.array_equal(piece, fixed[6:39, 9:52]), 'Feature guard must agree at native ROI and full frame'
    assert np.array_equal(fuse_reference_colour(contaminated, protected, source_values=beauty_pixels, strength=0), contaminated)
    white_guard = np.full((2, 3, 6), 0., np.float32); white_guard[..., 3:5] = [.07, 0]; white_guard[..., 5] = 1
    guarded_white = fuse_reference_colour(white, white_guard, source_values=white)
    assert guarded_white.min() >= 0 and guarded_white.max() <= 1
    assert np.max(abs(to_lab(guarded_white)[..., 0]-to_lab(white)[..., 0])) < 2e-6
    print('PASS: zero/selection identity and signed-zero bits without Lab, sparse-only conversion and float32 output, strict zero-field validation, source lightness, native ROI parity, fused edge donors, alpha preservation')


if __name__ == '__main__':
    main()
