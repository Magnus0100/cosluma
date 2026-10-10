"""CPU regression for qualified native detail, colour basis and exact boundaries."""
import copy
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy.ndimage import gaussian_filter

from native_receiver_preservation import (CONFIG, LUMA, array, prepare_native_receiver_preservation,
    qualified_support, sha, transfer_receiver_bands)


def main():
    y, x = np.mgrid[:180, :240]
    source_y = .24+.035*np.sin(x*1.4)+.018*np.sin(y*1.1)+.04*(x > 120)
    source = np.repeat(source_y[..., None], 3, axis=2).astype(np.float32)
    candidate = np.empty_like(source); candidate[:] = [.18, .09, .06]
    allowed = np.zeros((180, 240), bool); allowed[25:155, 25:215] = True
    receiver = np.ones_like(allowed)
    after, weight, checks = transfer_receiver_bands(source, candidate, allowed, receiver, sigma=8., feather=4.)
    assert checks['changed_pixels'] > 10_000 and checks['outside_support_exact']
    np.testing.assert_array_equal(after[~allowed], candidate[~allowed])
    actual_y = after @ LUMA
    core = np.s_[50:130, 50:190]
    before_high = ((candidate @ LUMA)-gaussian_filter(candidate @ LUMA, 3))[core]
    after_high = (actual_y-gaussian_filter(actual_y, 3))[core]
    assert after_high.std() > max(1e-4, 100*before_high.std())
    source_high = (source_y-gaussian_filter(source_y, 3))[core]
    assert np.corrcoef(source_high.ravel(), after_high.ravel())[0, 1] > .99
    # The dark native contour stays at its source coordinate, without importing
    # the source's grey RGB or its broad brightness difference.
    chroma = after[allowed]/after[allowed, 1:2]
    np.testing.assert_allclose(chroma, np.tile([2., 1., 2/3], (allowed.sum(), 1)), atol=2e-6)
    assert abs(float(actual_y[core].mean())-float((candidate @ LUMA)[core].mean())) < .002
    assert abs(float(actual_y[core].mean())-float(source_y[core].mean())) > .1
    # Foreground cannot enter the source lowpass or become a bright halo.
    receiver[70:90, 100:140] = False
    source2 = source.copy(); source2[~receiver] = [1., 0., 1.]
    a, _, _ = transfer_receiver_bands(source, candidate, allowed, receiver, sigma=8., feather=4.)
    b, _, _ = transfer_receiver_bands(source2, candidate, allowed, receiver, sigma=8., feather=4.)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(b[~receiver], candidate[~receiver])
    zero, _, zero_report = transfer_receiver_bands(source, candidate, np.zeros_like(allowed), receiver, sigma=8., feather=4.)
    np.testing.assert_array_equal(zero, candidate); assert zero_report['state'] == 'unknown'
    bright = candidate.copy(); bright[:] = [.999, .8, .7]
    gamut, _, _ = transfer_receiver_bands(source, bright, allowed, receiver, sigma=8., feather=4.)
    assert np.isfinite(gamut).all() and gamut.min() >= 0 and gamut.max() <= 1

    from design_scene_assets import prepare_contact_receiver_assets
    from local_floor_pollution import PARAMS
    native, proxy = (72, 80), (36, 40)
    alpha = np.zeros(native[::-1], np.float32); alpha[30:50, 24:40] = 1
    alpha[50:52, 24:40] = .25
    footwear = np.zeros(proxy[::-1], np.float32); footwear[15:25, 12:20] = 1
    proposal = np.zeros_like(footwear); proposal[25:30, 12:20] = 1
    floor = np.zeros_like(footwear); floor[25:] = 1
    labels = np.zeros_like(footwear, np.uint8); labels[15:25, 12:20] = 9
    fragment = np.zeros_like(footwear); fragment[26:39, 8:28] = 1
    points = [[10, 28], [14, 28], [18, 28], [25, 28], [10, 32], [18, 32], [25, 32], [10, 37], [25, 37]]
    with tempfile.TemporaryDirectory(prefix='cosplay_native_receiver_') as folder:
        root = Path(folder)
        def asset(name, values, key='values', **extra):
            path = root/(name+'.npz'); np.savez_compressed(path, **{key: values})
            return {'path': str(path), 'sha256': sha(path), 'size': list(values.shape[:2][::-1]),
                'coordinate_space': 'source_canvas', 'working_space': 'display_linear_sRGB', **extra}
        a = asset('alpha', alpha)
        scene = {'source': {'oriented_size': list(native), 'sha256': 'a'*64},
            'candidate': {'oriented_size': list(native), 'sha256': 'b'*64},
            'background_mapping': {'supported': True, 'candidate_original_to_source_original': np.eye(3).tolist()},
            'contact': {'source': {'supported': True, 'contact_support': asset('proposal', proposal),
                'footwear': asset('shoe', footwear), 'support_surface': asset('floor', floor)},
                'candidate': {'support_surface': asset('candidate-floor', floor)}}}
        contact, _ = prepare_contact_receiver_assets(scene, proxy, root/'contact',
            source_foreground_alpha=a, source_semantics=asset('labels', labels, key='labels', coordinate_space='source_proxy'))
        source_fragment = asset('source-fragment', fragment, coordinate_space='source_scene_proxy')
        protection = asset('protection', fragment, coordinate_space='candidate_scene_proxy',
            original_canvas_size=list(native), role='paired_observed_floor_reflection_protection')
        yy, xx = np.mgrid[:80, :72]
        old = np.repeat((.25+.05*np.sin(xx*1.5))[..., None], 3, axis=2).astype(np.float32)
        candidate_native = np.empty_like(old); candidate_native[:] = [.2, .1, .075]
        olda = asset('old', old)
        target = asset('target', candidate_native, representation='sampling_validity_associated_RGB')
        valid = asset('validity', np.ones_like(alpha))
        contract = {'schema': 'observed_floor_reflection_ownership.v1', 'qualified': True, 'mapping_supported': True,
            'protected_pixels': int(fragment.sum()), 'components': [{'candidate_component': 0, 'qualified': True,
                'source_inlier_points': copy.deepcopy(points), 'candidate_inlier_points': copy.deepcopy(points), 'pixels': int(fragment.sum()),
                'verified_unique_inliers': 9, 'candidate_overlap': 1., 'source_overlap': 1.}],
            'observation_bindings': {'qualification_params': PARAMS}, 'binding': {'source_sha256': 'a'*64,
                'candidate_sha256': 'b'*64, 'source_detected_fragment_sha256': source_fragment['sha256'],
                'protection_sha256': protection['sha256']}}
        plan = {'source_sha256': 'a'*64, 'candidate_sha256': 'b'*64,
            'canvas': {'source_oriented_size': list(native), 'source_proxy_size': list(proxy),
                'candidate_mapping': {'scene_registration_supported': True,
                    'candidate_original_to_final_original': np.eye(3).tolist()}},
            'mask_roles': {'foreground_alpha': a, 'candidate_reflection_protection': protection,
                'source_observed_floor_fragments': source_fragment}, 'layers': [contact,
                {'kind': 'background_design', 'assets': {'background': target, 'validity': valid}}],
            'source_baseline': {'working_domain': 'display_linear_sRGB', 'assets': {'old_environment': olda, 'foreground_alpha': a}},
            'provenance': {'candidate_floor_contract': {'reflection_ownership': contract}}}
        run = {'source': {'sha256': 'a'*64}, 'reference': {'sha256': 'b'*64}}
        original_plan = copy.deepcopy(plan)
        result = prepare_native_receiver_preservation(run, plan, root/'prepared')
        assert result['receiver_preservation']['state'] == 'prepared' and plan == original_plan
        bg = result['design_reconstruction_plan']['layers'][-1]
        actual = array(bg['assets']['background'])
        np.testing.assert_array_equal(actual[alpha > 0], candidate_native[alpha > 0])
        assert (actual != candidate_native).any()
        assert bg['assets']['validity'] == valid and result['design_reconstruction_plan']['layers'][0] == contact
        observed_partial = alpha.copy(); observed_partial[58:63, 20:30] = .5
        differing = copy.deepcopy(plan)
        differing['source_baseline']['assets']['foreground_alpha'] = asset('observed-partial', observed_partial)
        guarded = prepare_native_receiver_preservation(run, differing, root/'observed-alpha')
        guarded_bg = array(guarded['design_reconstruction_plan']['layers'][-1]['assets']['background'])
        np.testing.assert_array_equal(guarded_bg[observed_partial > 0], candidate_native[observed_partial > 0])
        repeated = prepare_native_receiver_preservation(run, result['design_reconstruction_plan'], root/'repeat')
        assert repeated['cache_hit'] and repeated['design_reconstruction_plan'] == result['design_reconstruction_plan']
        unknown = copy.deepcopy(plan); unknown['provenance']['candidate_floor_contract']['reflection_ownership']['qualified'] = False
        unchanged = prepare_native_receiver_preservation(run, unknown, root/'unknown')
        assert unchanged['receiver_preservation']['state'] == 'unknown' and unchanged['design_reconstruction_plan'] == unknown
        unknown = copy.deepcopy(plan); unknown['layers'] = [plan['layers'][-1]]
        assert qualified_support(unknown, run)[0] is None
        def reject(changed):
            try: prepare_native_receiver_preservation(run, changed, root/'invalid')
            except ValueError: return
            raise AssertionError('Tampered ownership must not be consumed')
        for key in ('source_sha256', 'candidate_sha256', 'protection_sha256', 'source_detected_fragment_sha256'):
            changed = copy.deepcopy(plan)
            changed['provenance']['candidate_floor_contract']['reflection_ownership']['binding'][key] = 'd'*64
            reject(changed)
        changed = copy.deepcopy(plan)
        changed['provenance']['candidate_floor_contract']['reflection_ownership']['components'][0]['source_inlier_points'][0][0] += 10
        reject(changed)
        changed = copy.deepcopy(plan); changed['mask_roles']['candidate_reflection_protection']['sha256'] = 'e'*64; reject(changed)
        changed = copy.deepcopy(result['design_reconstruction_plan'])
        changed['layers'][-1]['assets']['native_receiver_preservation']['sha256'] = 'f'*64; reject(changed)
        assert not (root/'unknown'/'native-receiver-background.npz').exists()
    print('OK: native luminance/contour bands restored; candidate chroma and broad tone retained; foreground excluded from kernels; zero/unqualified exact; gamut bounded; frozen assets, hashes, distributed geometry, receiver and idempotence verified')


if __name__ == '__main__':
    main()
