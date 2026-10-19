"""Contract smoke for the independent bounded scene balance pixel operator."""
from __future__ import annotations

import json, tempfile
from pathlib import Path
import numpy as np
from unittest.mock import patch

from reference_colour_fit import to_lab
from reference_scene_balance import (apply_scene_balance, resize_sdr_baseline,
    PREPARATION_LIMITS, PREPARATION_POLICY, REGION_LIMITS, SDR_AREA_ROUNDOFF,
    _receipt_limits, read_skin_protection, _master,
    extend_mouth_lightness)


def main():
    import cv2
    with tempfile.TemporaryDirectory(prefix='cosplay_scene_protection_') as folder:
        import tifffile
        from float_appearance import srgb_to_linear
        master_path = Path(folder) / '中文 路径' / '场景母版.tif'
        master_path.parent.mkdir()
        encoded = np.zeros((5, 7, 3), dtype=np.uint16)
        encoded[:] = [12000, 32000, 58000]
        tifffile.imwrite(master_path, encoded, photometric='rgb')
        np.testing.assert_allclose(_master(master_path, (7, 5)), srgb_to_linear(encoded.astype(np.float32)/65535), atol=1e-7)
        tifffile.imwrite(master_path, encoded.astype(np.uint8), photometric='rgb')
        try:
            _master(master_path, (7, 5))
        except ValueError:
            pass
        else:
            raise AssertionError('Unicode-path RGB8 TIFF was accepted as uint16')
        path=Path(folder)/'mask.npz'
        mask=np.arange(35,dtype=np.float32).reshape(5,7)/34
        expected=cv2.resize(mask,(14,10),interpolation=cv2.INTER_LINEAR)
        for role in ('protection','values'):
            np.savez_compressed(path,**{role:mask})
            np.testing.assert_array_equal(read_skin_protection(path,(14,10)),expected)
        np.savez_compressed(path,protection=mask,values=np.full((5,7),float('nan'),np.float32))
        np.testing.assert_array_equal(read_skin_protection(path,(14,10)),expected)
        for invalid in (np.ones((5,7,1),np.float32),np.ones((0,7),np.float32),
                        np.array([[float('nan')]],np.float32),np.array([[1.01]],np.float32),
                        np.array([[-.01]],np.float32)):
            np.savez_compressed(path,protection=invalid)
            try:
                read_skin_protection(path,(14,10))
                raise AssertionError('Malformed native protection mask was accepted')
            except ValueError: pass
        np.savez_compressed(path,other=mask)
        try:
            read_skin_protection(path,(14,10))
            raise AssertionError('Missing mask key was accepted')
        except ValueError: pass
    new_limits={'limits':PREPARATION_LIMITS,'region_limits':REGION_LIMITS,'preparation_policy':PREPARATION_POLICY}
    assert _receipt_limits(new_limits)==PREPARATION_LIMITS
    for invalid in ({'limits':PREPARATION_LIMITS}, {**new_limits,'limits':{**PREPARATION_LIMITS,'maximum_ev':2}},
                    {**new_limits,'region_limits':{}}):
        try:
            _receipt_limits(invalid)
            raise AssertionError('Undeclared preparation caps were accepted')
        except ValueError: pass
    # A real non-integer INTER_AREA reduction of saturated white creates
    # endpoint overshoot on the same OpenCV path used by 60 MP preparation.
    white = np.ones((99,66,3),np.float32)
    raw_area = cv2.resize(white,(14,21),interpolation=cv2.INTER_AREA)
    area, area_receipt = resize_sdr_baseline(white,(14,21))
    assert raw_area.max() <= 1+SDR_AREA_ROUNDOFF
    assert np.array_equal(area,np.ones_like(area)), 'white endpoint roundoff must normalize to one'
    assert area_receipt['corrected_channels'] == int((raw_area>1).sum())
    assert area_receipt['maximum_excursion'] <= SDR_AREA_ROUNDOFF
    # The normalization boundary never accepts source HDR or NaN, nor a
    # materially out-of-range resampling result. Pixel validation stays strict.
    for bad in (1+1e-4, float('nan'), float('inf'), -.001):
        invalid = white.copy(); invalid[0,0,0] = bad
        try:
            resize_sdr_baseline(invalid,(14,21))
            raise AssertionError('invalid source was normalized instead of rejected')
        except ValueError:
            pass
        invalid_area = np.ones_like(area); invalid_area[0,0,0] = bad
        with patch.object(cv2,'resize',return_value=invalid_area):
            try:
                resize_sdr_baseline(white,(14,21))
                raise AssertionError('invalid resampling result was normalized instead of rejected')
            except ValueError:
                pass
    over = area.copy(); over[0,0,0] = np.nextafter(np.float32(1),np.float32(2))
    normalized = apply_scene_balance(over,np.zeros((*over.shape[:2],4),np.float32))
    assert normalized[0,0,0] == 1, 'bounded SDR endpoint roundoff must normalize'
    invalid_scene = area.copy(); invalid_scene[0,0,0] = 1 + 1e-3
    try:
        apply_scene_balance(invalid_scene,np.zeros((*invalid_scene.shape[:2],4),np.float32))
        raise AssertionError('scene pixel operator must reject material SDR overflow')
    except ValueError:
        pass
    y, x = np.mgrid[:23, :31]
    rgb = np.stack([.04+.005*x, .05+.004*x, .06+.003*x], -1).astype(np.float32)
    rgb += (.004*np.sin(x*1.7)*np.cos(y*.9))[..., None]
    field = np.zeros((*rgb.shape[:2], 4), np.float32)
    field[3:20,4:28,0] = .25
    field[3:20,4:28,3] = .8
    zero = apply_scene_balance(rgb, field, strength=0)
    assert np.array_equal(zero, rgb), 'zero strength must be exact'
    assert np.array_equal(apply_scene_balance(rgb, np.zeros_like(field)), rgb), 'zero field must be exact'
    result = apply_scene_balance(rgb, field)
    outside = field[...,3] == 0
    assert np.array_equal(result[outside], rgb[outside]), 'outside field must be exact'
    assert np.isfinite(result).all() and result.min() >= 0 and result.max() <= 1
    before, after = to_lab(rgb), to_lab(result)
    chroma_error = float(np.max(np.abs(before[...,1:]-after[...,1:])))
    assert chroma_error < 4e-7, ('zero ab field must retain current chroma', chroma_error)
    expected_gain = 2**(.25/3)
    mask = field[...,0] != 0
    assert np.max(abs(after[...,0][mask]/before[...,0][mask]-expected_gain)) < 3e-6
    # Constant midtone EV retains native log-L detail contrast exactly to float precision.
    detail_before = np.diff(np.log(before[8,6:25,0]))
    detail_after = np.diff(np.log(after[8,6:25,0]))
    detail_error = float(np.max(abs(detail_before-detail_after)))
    assert detail_error < 2e-6, ('midtone detail contrast changed', detail_error)
    box = (slice(4,17),slice(7,26))
    assert np.array_equal(result[box], apply_scene_balance(rgb[box],field[box])), 'ROI must equal full image'
    coverage_only = field.copy()
    coverage_only[...,3] *= .1
    assert np.array_equal(result,apply_scene_balance(rgb,coverage_only)), 'coverage must not be consumed twice'
    # A known planar skin illumination must continue through differently
    # coloured lips/teeth without a categorical EV seam or any added pigment.
    from local_face_protection import mouth_lightness_domain
    yy, xx = np.mgrid[:41, :61]
    material = np.full((41,61), 3, np.int32)
    material[13:30,16:45] = 25
    material[20:24,26:35] = 26
    regions = [{'region': 'lips', 'polygon': [[16,13],[44,13],[44,29],[16,29]],
                'inner_polygon': [[26,20],[34,20],[34,23],[26,23]]}]
    opaque = np.ones(material.shape, np.float32)
    domain = mouth_lightness_domain(material.shape, regions, material)
    plane = (.05+.002*xx+.001*yy).astype(np.float32)
    mouth_field = np.zeros((*material.shape,4), np.float32)
    mouth_field[...,0] = plane; mouth_field[...,3] = .9
    mouth_field[domain] = 0
    continued, applied, report = extend_mouth_lightness(mouth_field,material,opaque,regions)
    assert np.array_equal(applied, domain) and report['status'] == 'prepared'
    np.testing.assert_allclose(continued[...,0],plane,atol=2e-8,rtol=0)
    assert np.array_equal(continued[~domain],mouth_field[~domain])
    assert not continued[...,1:3].any(), 'mouth illumination must not transfer pigment'
    assert not mouth_field[domain].any(), 'input field must remain immutable'
    # Unsupported background light cannot supply a face illumination donor.
    negative_labels = material.copy(); negative_labels[~domain] = 0
    background = mouth_field.copy(); background[~domain,0] = .75
    rejected, rejected_domain, rejection = extend_mouth_lightness(
        background,negative_labels,opaque,regions)
    assert not rejected_domain.any() and rejection['status'] == 'rejected'
    assert np.array_equal(rejected,background)
    from render_reference_colour_composite import sample_maps
    hair_owners=np.full(rgb.shape[:2],4,np.int32);hair_owners[:,20:]=3
    visible=np.full(rgb.shape[:2],.5,np.float32)
    raw_scene=np.zeros_like(field);raw_scene[...,0]=.5;raw_scene[...,3]=.8
    guarded=sample_maps(raw_scene,(0,0,31,23),(31,23),(31,23),owners=hair_owners,hair_face_support=visible)
    assert np.array_equal(guarded[:,:20,0],np.full((23,20),.25,np.float32))
    assert np.array_equal(guarded[:,20:,0],raw_scene[:,20:,0])
    assert np.array_equal(guarded[...,3],raw_scene[...,3]), 'QA coverage must not be used as visible-face dose'
    guarded_result=apply_scene_balance(rgb,guarded)
    expected_scene=raw_scene.copy();expected_scene[:,:20,0]=.25
    assert np.array_equal(guarded_result,apply_scene_balance(rgb,expected_scene))
    # A monotone highlight endpoint retains separation instead of clipping to a white plateau.
    gray = np.array([.72,.80,.88,.94,.98,.995,1.],np.float32)
    highlights = np.repeat(gray[None,:,None],3,-1)
    high_field = np.zeros((1,len(gray),4),np.float32)
    high_field[...,0],high_field[...,3] = .4,1
    high_result = apply_scene_balance(highlights,high_field)
    assert np.isfinite(high_result).all() and high_result.min() >= 0 and high_result.max() <= 1
    assert np.all(np.diff(high_result[0,:,0]) > 1e-5), 'highlights were flattened or reversed'
    # New preparation caps use the same SDR pixel operator and endpoint curve.
    expanded_field=high_field.copy();expanded_field[...,0]=.75
    expanded=apply_scene_balance(highlights,expanded_field)
    assert expanded.min()>=0 and expanded.max()<=1 and np.all(np.diff(expanded[0,:,0])>1e-5)
    assert np.array_equal(high_result,apply_scene_balance(highlights,high_field))
    negative_field=field.copy();negative_field[...,0]*=-.8
    reduced=apply_scene_balance(rgb,negative_field)
    reduced_lab=to_lab(reduced)
    assert np.max(abs(np.diff(np.log(reduced_lab[8,6:25,0]))-detail_before))<2e-6
    # Strong bounded tint on saturated/highlight pixels must reduce dose, not clip channels.
    saturated = np.array([[[1.,.02,.01],[.01,.95,.02],[.04,.06,.99],[1.,1.,1.],[.999,.997,.995]]],np.float32)
    tint = np.zeros((1,5,4),np.float32)
    tint[...,0],tint[...,1],tint[...,2],tint[...,3] = .4,.018,-.015,1
    bounded = apply_scene_balance(saturated,tint)
    assert np.isfinite(bounded).all() and bounded.min() >= 0 and bounded.max() <= 1
    assert np.linalg.norm(tint[...,1:3],axis=-1).max() <= PREPARATION_LIMITS['maximum_ab']
    assert np.array_equal(bounded[:,2:4],apply_scene_balance(saturated[:,2:4],tint[:,2:4])), 'gamut ROI differs'
    # One gamut conflict must not make every legal pixel enter the fixed
    # 20-step search. Legal proposals retain their original computed RGB bits.
    import reference_scene_balance as scene
    mixed = np.full((16, 32, 3), .18, np.float32); mixed[0, :5] = saturated[0]
    mixed_field = np.broadcast_to(tint[0, 0], (16, 32, 4)).copy()
    mixed_lab = scene.to_lab(mixed.reshape(-1, 3))
    initial_lab = mixed_lab.copy()
    initial_lab[:, 0] = scene._limited_lightness(mixed_lab[:, 0], np.full(512, .4, np.float32))
    initial_lab[:, 1:] += np.array([.018, -.015], np.float32)
    initial = scene.from_lab(initial_lab)
    conflicts = (initial.min(-1) < 0) | (initial.max(-1) > 1) | ~np.isfinite(initial).all(-1)
    assert 0 < conflicts.sum() < 512
    proposal_sizes = []
    original_from_lab = scene.from_lab
    def observed_from_lab(values):
        proposal_sizes.append(len(values))
        return original_from_lab(values)
    with patch.object(scene, 'from_lab', side_effect=observed_from_lab):
        mixed_result = apply_scene_balance(mixed, mixed_field)
    assert proposal_sizes == [512] + [int(conflicts.sum())]*21
    np.testing.assert_array_equal(mixed_result.reshape(-1, 3)[~conflicts].view(np.uint32), initial[~conflicts].view(np.uint32))
    from reference_colour_fit import SCHEMA, REGIONS
    from render_reference_colour_composite import donor_fields, match_foreground
    source = np.full((4,5,3), .18, np.float32)
    baseline = np.full_like(source, .22)
    weights = np.zeros((4,5,len(REGIONS)), np.float32)
    weights[...,list(REGIONS).index('hair')] = 1
    params = {'schema': SCHEMA, 'working_domain': 'display_linear_sRGB',
              'settings': {'maximum_exposure_ev': 1},
              'operators': {'hair': {'exposure_ev': 0, 'matrix': [[1,0],[0,1]], 'bias': [0,0]}}}
    edge_field = np.zeros((4,5,4), np.float32)
    edge_field[...,0], edge_field[...,3] = .25, 1
    donors = donor_fields(source, np.ones((4,5),np.float32), weights, params,
                          scene_balance=edge_field, matched_baseline=baseline)
    edge_alpha = np.full((4,5), .05, np.float32)
    edge_alpha[0,0] = 0
    fp = source*edge_alpha[...,None]
    edge_result = match_foreground(fp,edge_alpha,weights,donors,np.ones_like(edge_alpha),params,
                                   origin=(0,0),canvas_size=(5,4),scene_balance=edge_field)
    edge_expected = apply_scene_balance(baseline,edge_field)*edge_alpha[...,None]
    edge_error = float(np.max(abs(edge_result-edge_expected)))
    assert edge_error < 1e-7, ('low alpha target must include S and exactly one alpha',edge_error)
    assert np.array_equal(edge_result[0,0],np.zeros(3,np.float32)), 'alpha zero must be zero'
    print(json.dumps({'smoke': 'reference_scene_balance', 'passed': True,
                      'protection_and_values_mask_layouts':True,'malformed_protection_masks_rejected':True,
                      'current_caps_explicit_and_invalid_caps_rejected':True,
                      'expanded_highlight_range_monotone_SDR':True,'negative_EV_detail_retained':True,
                      'area_endpoint_roundoff_normalized': area_receipt,
                      'HDR_NaN_inf_rejected': True, 'pixel_operator_SDR_validation_strict':True,
                      'zero_strength_exact': True, 'outside_field_exact': True,
                      'chroma_error_with_zero_ab': chroma_error, 'midtone_log_detail_error': detail_error,
                      'ROI_exact': True, 'coverage_not_reapplied': True,
                      'highlight_separation': np.diff(high_result[0,:,0]).tolist(),
                      'gamut_finite_and_bounded': True, 'low_alpha_final_scene_target_error': edge_error,
                      'gamut_search_only_initial_conflicts': True, 'legal_proposal_RGB_bits_preserved': True,
                      'low_alpha_no_alpha_squared': True, 'alpha_zero_exact': True}))


if __name__ == '__main__':
    main()
