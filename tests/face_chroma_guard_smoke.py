"""CPU evidence for clean-skin ambient estimates independent of eye/lip colour."""
import json
from unittest.mock import patch

import numpy as np

from local_face_protection import (LIP_INNER_INDICES, LIP_OUTER_INDICES, prepare_feature_chroma_guard,
                                  prepare_feature_materials, prepare_strict_lip_support,
                                  source_feature_regions)


def fixture():
    height, width = 112, 144
    labels = np.zeros((height, width), np.uint8)
    labels[16:96, 32:112] = 3
    alpha = (labels != 0).astype(np.float32)
    beauty = np.full((height, width, 3), [.45, .025, .03], np.float32)
    protection = np.zeros((height, width), np.float32)
    # Distinct native feature colours are deliberately far from skin: their
    # colour cannot become an ambient donor or be classified as contamination.
    features = ((np.s_[36:44, 47:55], 2, [.17, .025, .018]),
                (np.s_[36:44, 85:93], 3, [.23, -.015, -.065]),
                (np.s_[70:78, 65:83], 24, [.32, .105, .035]))
    for box, label, colour in features:
        labels[box] = label
        beauty[box] = colour
        protection[box] = 1
    # Existing worker feather, represented by a non-binary mask in this test.
    protection[34:36, 45:57] = .5
    protection[68:80, 63:85] = np.maximum(protection[68:80, 63:85], .5)
    expected_ratio = np.array([.018, -.012], np.float32)
    fused = beauty.copy()
    fused[..., 1:] += beauty[..., :1] * expected_ratio
    return beauty, fused, labels, alpha, protection, features, expected_ratio


def main():
    # Cached full landmarks retain the Cupid concavity and a distinct opening.
    # The old worker hull deliberately disagrees, so consuming it fails here.
    outer = np.array([[24, 36], [22, 32], [24, 28], [28, 26], [34, 26], [38, 30],
                      [42, 26], [48, 26], [54, 28], [58, 32], [60, 36], [58, 40],
                      [54, 44], [48, 47], [43, 48], [38, 49], [33, 48], [28, 46],
                      [25, 43], [23, 39]], np.float32)
    inner = np.array([[27, 37], [28, 36], [30, 35], [33, 35], [36, 35], [39, 35],
                      [42, 35], [46, 35], [49, 35], [52, 36], [55, 37], [52, 38],
                      [49, 39], [46, 39], [42, 39], [39, 39], [36, 39], [33, 39],
                      [30, 39], [28, 38]], np.float32)
    points = np.zeros((478, 2), np.float32)
    points[list(LIP_OUTER_INDICES)], points[list(LIP_INNER_INDICES)] = outer, inner
    cached_face = {'size': [96, 72], 'landmarks': points.tolist(), 'protected_regions': [
        {'region': 'lips', 'polygon': [[18, 20], [64, 20], [64, 56], [18, 56]]},
        {'region': 'left_eye', 'polygon': [[25, 10], [30, 10], [30, 14], [25, 14]]}]}
    topology = source_feature_regions(cached_face, [96, 72])
    lip_region = next(region for region in topology if region['region'] == 'lips')
    assert lip_region['landmark_indices'] == list(LIP_OUTER_INDICES)
    np.testing.assert_array_equal(lip_region['polygon'], outer)
    np.testing.assert_array_equal(lip_region['inner_polygon'], inner)
    scaled = source_feature_regions(cached_face, [192, 144])
    np.testing.assert_array_equal(scaled[-1]['polygon'], outer*2)
    semantic = np.full((72, 96), 3, np.uint8)
    semantic[20:56, 18:64] = 24  # Includes both real pigment and wrong skin labels.
    semantic[10:15, 25:31] = 2
    semantic[37, 39] = 3  # The opening is often mislabeled as skin by segmentation.
    semantic[44, 40] = 26  # Even inside the contour, teeth are not lip pigment.
    original_semantic = semantic.copy()
    expanded = np.zeros(semantic.shape, np.float32)
    expanded[20:56, 18:64] = 1
    expanded[8:17, 23:33] = .7
    refined, bounded, material_report = prepare_feature_materials(semantic, expanded, topology)
    assert np.array_equal(semantic, original_semantic), 'Raw source semantics must remain immutable'
    assert refined[28, 38] == refined[22, 38] == refined[53, 38] == 3, 'Cupid gap and outside lips are skin'
    assert refined[44, 38] == 24 and refined[37, 39] == 26, 'Lip pigment and opening need distinct materials'
    assert bounded[22, 38] == bounded[28, 38] == bounded[53, 38] == 0
    assert bounded[44, 38] == 1, 'Structural protection must remain within the real mouth'
    assert bounded[12, 27] == expanded[12, 27], 'Existing eye protection must remain'
    strict_topology, collar_topology, topology_report = prepare_strict_lip_support(
        semantic.shape, topology, refined, np.ones(semantic.shape, np.float32))
    assert strict_topology[28, 38] == strict_topology[37, 39] == strict_topology[44, 40] == 0
    assert strict_topology[44, 38] > .9
    assert not collar_topology[37, 39], 'The opening cannot become a rejected skin collar'
    assert material_report['pixel_count_space'] == 'source_proxy'
    assert material_report['appearance_semantic_lip_to_face_pixels'] > 0
    assert topology_report['inner_mouth_colour_support'] is False
    missing_topology = source_feature_regions({**cached_face, 'landmarks': []}, [96, 72])
    assert not any(region['region'] == 'lips' for region in missing_topology), 'Missing landmarks cannot use the old hull'
    unchanged, original_protection, unsupported = prepare_feature_materials(semantic, expanded, missing_topology)
    assert np.array_equal(unchanged, semantic) and np.array_equal(original_protection, expanded)
    assert unsupported['status'] == 'unsupported'
    # A zero-area canonical opening is a legal closed mouth, not a failed face.
    closed_points = points.copy()
    closed_points[list(LIP_INNER_INDICES), 1] = 37
    closed_topology = source_feature_regions({**cached_face, 'landmarks': closed_points.tolist()}, [96, 72])
    assert closed_topology[-1]['inner_mouth_state'] == 'closed'
    closed_labels, _, closed_report = prepare_feature_materials(semantic, expanded, closed_topology)
    closed_support, _, closed_support_report = prepare_strict_lip_support(
        semantic.shape, closed_topology, closed_labels, np.ones(semantic.shape, np.float32))
    assert closed_report['status'] == 'prepared' and closed_report['inner_mouth_pixels'] == 0
    assert closed_report['inner_mouth_states'] == ['closed'] and closed_support[37, 39] == 1
    assert closed_labels[28, 38] == 3 and closed_labels[37, 39] == 3
    assert closed_support_report['regions'][0]['inner_mouth_excluded'] is False
    json.dumps(material_report, allow_nan=False)

    beauty, fused, labels, alpha, protection, features, expected = fixture()
    regions = [{'region': 'lips', 'polygon': [[65, 70], [82, 70], [82, 77], [65, 77]]},
               {'region': 'left_eye', 'polygon': [[47, 36], [54, 36], [54, 43], [47, 43]]},
               {'region': 'right_eye', 'polygon': [[85, 36], [92, 36], [92, 43], [85, 43]]}]
    beauty_before = beauty.copy()
    guard, report = prepare_feature_chroma_guard(beauty, fused, labels, alpha, protection,
                                                feature_regions=regions)
    assert report['status'] == 'prepared'
    assert report['roi'] == [32, 16, 112, 96]
    protected = guard[..., 2] > 0
    np.testing.assert_allclose(guard[protected, :2], np.broadcast_to(expected,
                               (np.count_nonzero(protected), 2)), atol=2e-7)
    assert not guard[guard[..., 2] == 0].any()
    assert np.array_equal(guard[:60, :, 2], protection[:60]), 'Lip tightening cannot change eye coverage'
    lip_support, reject_collar, lip_report = prepare_strict_lip_support(labels.shape, regions, labels, alpha)
    assert np.array_equal(guard[60:, :, 2], lip_support[60:])
    assert guard[66, 74, 2] == guard[81, 74, 2] == guard[73, 61, 2] == guard[73, 86, 2] == 0
    assert guard[73, 74, 2] > .9 and guard[70, 74, 2] == 0
    assert np.any(reject_collar) and not guard[reject_collar].any(), 'Rejection collar is never a colour dose'
    assert not lip_support[:70].any() and not lip_support[78:].any()
    assert not lip_support[:, :65].any() and not lip_support[:, 83:].any()
    assert not guard[labels == 0].any(), 'The lip hull cannot extend ownership into the background'
    assert np.array_equal(beauty, beauty_before), 'Producer cannot recolour Beauty feature/detail pixels'
    for box, _, colour in features:
        np.testing.assert_allclose(beauty[box], np.broadcast_to(colour, beauty[box].shape), atol=1e-7)

    polluted = fused.copy()
    polluted[features[0][0]][..., 1:] += [-.11, .07]
    polluted[features[2][0]][..., 1:] += [.025, -.12]
    polluted[68:70, 70:78, 1:] += [-.11, .09]
    polluted[72:76, 83:85, 1:] += [.08, -.12]
    # A smaller unprotected colour blot must not drive the robust skin estimate.
    polluted[21:25, 40:44, 1:] += [.16, -.14]
    repaired, contamination_report = prepare_feature_chroma_guard(beauty, polluted, labels, alpha, protection,
                                                                  feature_regions=regions)
    np.testing.assert_allclose(repaired, guard, atol=2e-7)
    assert contamination_report['rejected_donor_pixels'] == 16

    # Normalized blur retains a broad directional environment shift; a single
    # global median would erase this intentional variation.
    yy, xx = np.indices(labels.shape, dtype=np.float32)
    ratio = np.stack((.01 + .0002*(xx-32), -.008 + .0001*(yy-16)), axis=-1)
    directional = beauty.copy()
    directional[..., 1:] += beauty[..., :1]*ratio
    direction_guard, _ = prepare_feature_chroma_guard(beauty, directional, labels, alpha, protection,
                                                     feature_regions=regions)
    left = direction_guard[39, 51, 0]
    right = direction_guard[39, 89, 0]
    assert right-left > .006, 'Smooth directional lighting must survive ambient estimation'

    # Unsupported or translucent pixels cannot supply donors or receive guard.
    invalid_labels = labels.copy()
    invalid_alpha = alpha.copy()
    bad_protection = protection.copy()
    invalid_labels[36:44, 47:55] = 0
    invalid_alpha[70:78, 65:83] = .8
    invalid_alpha[71:73, 70:78] = .8
    invalid_labels[73:76, 78:82] = 4
    bad_protection[4:12, 4:12] = 1
    bright_background = polluted.copy()
    bright_background[labels == 0, 1:] = [.7, -.9]
    gated, _ = prepare_feature_chroma_guard(beauty, bright_background, invalid_labels, invalid_alpha, bad_protection,
                                          feature_regions=regions)
    assert not gated[4:12, 4:12].any() and not gated[36:44, 47:55].any()
    assert not gated[70:78, 65:83].any()
    assert not gated[71:73, 70:78].any() and not gated[73:76, 78:82].any()
    np.testing.assert_allclose(gated[39, 89, :2], expected, atol=2e-7)

    no_donor_labels = labels.copy()
    no_donor_labels[labels == 3] = 2
    fallback, rejected = prepare_feature_chroma_guard(beauty, fused, no_donor_labels, alpha, protection,
                                                      feature_regions=regions)
    assert rejected['status'] == 'rejected'
    assert rejected['reason'] == 'insufficient_clean_opaque_skin_donors'
    assert not fallback[..., :2].any()
    assert np.array_equal(fallback[..., 2], guard[..., 2]), 'Missing donors cannot expand colour protection'
    no_landmarks, missing = prepare_feature_chroma_guard(beauty, fused, labels, alpha,
                                                        np.ones_like(protection), feature_regions=[])
    assert not no_landmarks.any() and missing['reason'] == 'no_source_feature_landmarks'

    # Wrong semantic lips outside the canonical geometry are a negative sample.
    # A supported appearance colour cannot turn them into a legal lip target.
    spilled_labels = labels.copy()
    spilled_labels[68:70, 70:78] = 24
    spilled_labels[78:81, 70:78] = 25
    strict, _, rejection = prepare_strict_lip_support(labels.shape, regions, spilled_labels, alpha)
    assert not strict[68:70, 70:78].any() and not strict[78:81, 70:78].any()
    assert rejection['source_lip_labels_outside_support'] >= 40
    assert rejection['outside_outer_colour_support'] is False
    assert lip_report['collar_used_for_colour_dose'] is False

    # A large source lip supports a measurable feather entirely inside its edge.
    large_regions = [{'region': 'lips', 'polygon': [[40, 40], [100, 40], [100, 65], [40, 65]]}]
    large, _, _ = prepare_strict_lip_support(labels.shape, large_regions, labels, alpha)
    assert large[40, 74] == large[41, 74] == large[42, 74] == 0
    assert 0 < large[43, 74] < 1 and large[44, 74] == 1
    assert not large[:40].any() and not large[66:].any()

    # The diagonal is x+y=100. Raster EDT overstates its first inside pixel's
    # depth; exact perpendicular distance must keep that pixel colour-free.
    diagonal_regions = [{'region': 'lips', 'polygon': [[40, 60], [70, 30], [100, 60], [70, 90]]}]
    diagonal, _, diagonal_report = prepare_strict_lip_support(labels.shape, diagonal_regions, labels, alpha)
    assert diagonal[46, 55] == diagonal[47, 55] == diagonal[48, 55] == 0
    expected_dose = ((55+49-100)/np.sqrt(2)-2.13)/1.2
    expected_dose = expected_dose**2*(3-2*expected_dose)
    assert abs(float(diagonal[49, 55])-expected_dose) < 1e-6
    assert diagonal_report['regions'][0]['inward_safety_proxy_px'] == 2.13

    # A same-owner face field has no nearest-owner exclusion at the hull edge.
    # At 10x scaling, native (313,313) lies beyond x+y=625, while its bilinear
    # grid (.85,.85) relative to (30,30) could borrow that inward corner. This
    # valid corner has 1.768 proxy distance: a 1.5 band was insufficient.
    from render_reference_colour_composite import sample_reference_field
    corner_labels = np.full((64, 64), 3, np.uint8)
    corner_regions = [{'region': 'lips', 'polygon': [[0, 0], [62.5, 0], [0, 62.5]]}]
    corner, _, _ = prepare_strict_lip_support(corner_labels.shape, corner_regions,
                                             corner_labels, np.ones((64, 64), np.float32))
    assert corner[30, 30] == 0 and corner[10, 10] == 1
    corner_field = np.zeros((64, 64, 6), np.float32)
    corner_field[..., 5] = corner
    assert 313+313 > 62.5*10, 'The sampled pixel must be outside the true native hull'
    corner_sample = sample_reference_field(corner_field, (313, 313, 314, 314), (64, 64),
                                           (640, 640), owners=corner_labels)
    assert corner_sample[0, 0, 5] == 0, 'Same-owner bilinear interpolation cannot reintroduce lip spill'

    bounded = beauty.copy()
    bounded[..., 1:] += beauty[..., :1]*[.3, -.4]
    limited, limited_report = prepare_feature_chroma_guard(beauty, bounded, labels, alpha, protection,
                                                          feature_regions=regions)
    assert np.linalg.norm(limited[..., :2], axis=-1).max() <= .080001
    assert .07999 < limited_report['actual_maximum_ambient_ab_per_L'] <= .080001

    # Inspect actual filter inputs: no full proxy filter is allowed.
    from scipy.ndimage import gaussian_filter
    shapes = []
    def observed_filter(values, *args, **kwargs):
        shapes.append(values.shape)
        return gaussian_filter(values, *args, **kwargs)
    with patch('scipy.ndimage.gaussian_filter', side_effect=observed_filter):
        prepare_feature_chroma_guard(beauty, fused, labels, alpha, protection, feature_regions=regions)
    assert shapes == [(80, 80), (80, 80, 2)]
    json.dumps(report, allow_nan=False)
    print(json.dumps({'ok': True, 'scope': 'CPU proxy guard and native field sampling only; no models or saved render',
                      'checks': ['normal_feature_colours_not_donors', 'green_blue_pollution_isolated',
                                 'ordered_cupid_and_inner_mouth_topology', 'appearance_materials_without_raw_changes',
                                 'mouth_L_protection_does_not_cover_skin', 'missing_topology_keeps_original_policy',
                                 'closed_mouth_has_no_opening_without_hull_fallback',
                                 'strict_lip_geometry_no_outward_dose', 'same_owner_bilinear_spill_rejected',
                                 'directional_ambient_preserved', 'opaque_ownership_required',
                                 'missing_donor_fallback_report', 'bounded_ambient', 'ROI_only_filters']}))


if __name__ == '__main__':
    main()
