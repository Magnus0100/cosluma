"""CPU proof/registration, source-guide fitting and frozen native replay checks."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

import registered_reference_style as style


def frozen(root, coefficients, native):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    coefficients, _ = style.bound_jacobians(np.asarray(coefficients, np.float32))
    path = root/'coefficients.npz'; np.savez_compressed(path, values=coefficients)
    asset = style._asset(path)
    identity = {'native_size': list(native), 'source_sha256': 'a'*64,
        'reference_sha256': 'b'*64, 'baseline_master_sha256': 'c'*64}
    receipt = {'schema': style.SCHEMA, 'profile': style.PROFILE, 'identity': identity,
        'reference_RGB_composited': False, 'model_submissions': 0,
        'coefficients_sha256': asset['sha256']}
    contract = root/'contract.json'; style._write(contract, receipt)
    return {'kind': style.KIND, 'enabled': True, 'strength': 1.,
        'params': {'schema': style.SCHEMA, 'native_size': list(native), 'identity_sha256': style._digest(identity)},
        'assets': {'coefficients': asset, 'contract': style._asset(contract)}}


def generated_reference_fixture(root, *, mode='create_reference'):
    """Synthetic transport evidence for CPU contract tests; no model ran."""
    from creative_direction import reference_contract_for, verify_source_conditioned_reference
    root = Path(root)
    source, reference = root/'source.png', root/'design.png'
    Image.new('RGB', (80, 120), '#3a5268').save(source)
    Image.new('RGB', (80, 120), '#304e74').save(reference)
    source_asset, reference_asset = style._asset(source), style._asset(reference)
    prompt, prompt_id = 'synthetic CPU fixture design', 'fixture-source-only-prompt'
    graph = {'1': {'class_type': 'LoadImage', 'inputs': {'image': 'fixture-source.png'}},
        '2': {'class_type': 'CosplayTextEncodeQwenImage21',
              'inputs': {'prompt': prompt, 'images.image_1': ['1', 0]}}}
    condition = {**source_asset, 'role': 'source', 'upload': {'name': 'fixture-source.png'}}
    provenance = {'status': 'reference_candidate', 'provider': 'qwen_image_2_1_uncensored_fast', 'source': source_asset,
        'asset': reference_asset, 'prompt': prompt, 'generation_id': prompt_id,
        'provider_contract': {'request': {'prompt': prompt, 'api_input': source_asset,
                'input_sha256': source_asset['sha256']}, 'response': {'id': prompt_id},
            'comfyui': {'prompt_id': prompt_id, 'execution_status': {'completed': True, 'status_str': 'success'},
                       'condition_inputs': [condition], 'request': {'prompt': graph}},
            'transport': {'events': [{'event': 'comfyui_submitted', 'prompt_id': prompt_id, 'request': {'prompt': graph}},
                                     {'event': 'comfyui_completed', 'prompt_id': prompt_id}]}}}
    provenance_path = root/'design.provenance.json'; style._write(provenance_path, provenance)
    reference_asset['provenance'] = str(provenance_path)
    intent = {'mode': mode, 'constraints': {'background_mode': 'enhance_source', 'reference_role': 'source_conditioned_design'}}
    inputs = {'source': source_asset}
    if mode == 'existing_reference':
        inputs['reference'] = {**reference_asset, 'source_conditioned_binding':
            verify_source_conditioned_reference(root, source_asset['sha256'], reference_asset)}
    plan = {'ready': True, 'intent': intent, 'inputs': inputs,
        'reference_contract': reference_contract_for(intent, inputs)}
    saved_provenance = json.loads(provenance_path.read_text(encoding='utf8'))
    return {'source': source_asset, 'reference': reference_asset, 'task_intent': intent, 'task_plan': plan}, provenance_path, saved_provenance


class StyleTests(unittest.TestCase):
    def test_consumer_import_never_requires_worker_only_opencv(self):
        import builtins
        import importlib
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name == 'cv2' or name.startswith('cv2.'):
                raise AssertionError('Desktop consumer attempted OpenCV import')
            return original(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=guarded): importlib.reload(style)

    def test_only_exact_checked_same_photo_role_and_sha_authorizes(self):
        from creative_direction import reference_contract_for
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            source = root/'source.png'; reference = root/'reference.png'
            Image.new('RGB', (40, 50), '#235171').save(source)
            Image.new('RGB', (40, 50), '#525971').save(reference)
            inputs = {'source': style._asset(source), 'reference': style._asset(reference)}
            intent = {'mode': 'existing_reference', 'constraints': {
                'background_mode': 'enhance_source', 'reference_role': 'same_photo_finish'}}
            descriptor = reference_contract_for(intent, inputs)
            run = {**inputs, 'task_intent': intent,
                'task_plan': {'ready': True, 'intent': intent, 'inputs': inputs, 'reference_contract': descriptor}}
            self.assertEqual(style.checked_contract(run), descriptor)
            for change in ('role', 'source', 'reference', 'ready'):
                bad = copy.deepcopy(run)
                if change == 'role': bad['task_plan']['reference_contract']['role'] = 'style_reference'
                elif change == 'ready': bad['task_plan']['ready'] = False
                else: bad['task_plan']['inputs'][change]['sha256'] = 'e'*64
                with self.subTest(change=change), self.assertRaises(ValueError): style.checked_contract(bad)
            Image.new('RGB', (40, 50), '#435171').save(reference)
            with self.assertRaises(ValueError): style.checked_contract(run)

    def test_generated_role_binds_real_source_only_provenance_after_generation(self):
        with tempfile.TemporaryDirectory() as t:
            run, _, _ = generated_reference_fixture(t)
            original_plan = copy.deepcopy(run['task_plan'])
            self.assertIsNone(original_plan['reference_contract']['reference_sha256'])
            self.assertNotIn('reference', original_plan['inputs'])
            with self.assertRaises(ValueError): style.checked_contract(run)
            effective = style.checked_contract(run, t)
            self.assertEqual(effective['role'], 'source_conditioned_design')
            self.assertEqual(effective['reference_sha256'], run['reference']['sha256'])
            self.assertEqual(effective['reference_binding'], 'generated_source_verified')
            self.assertEqual(effective['provider_prompt_id'], 'fixture-source-only-prompt')
            self.assertFalse(effective['reference_rgb_compositing_allowed'])
            self.assertEqual(run['task_plan'], original_plan)
            bad = copy.deepcopy(run)
            bad['task_plan']['inputs']['reference'] = bad['reference']
            with self.assertRaises(ValueError): style.checked_contract(bad, t)

    def test_generated_existing_continuation_and_actual_wrong_condition_are_revalidated(self):
        with tempfile.TemporaryDirectory() as t:
            run, path, provenance = generated_reference_fixture(t, mode='existing_reference')
            self.assertEqual(style.checked_contract(run, t), run['task_plan']['reference_contract'])
            for defect in ('manual_image_condition', 'second_condition', 'wrong_source', 'wrong_output'):
                bad = copy.deepcopy(provenance)
                comfy = bad['provider_contract']['comfyui']
                if defect == 'manual_image_condition': comfy['condition_inputs'][0]['role'] = 'reference'
                elif defect == 'second_condition':
                    comfy['condition_inputs'].append(copy.deepcopy(comfy['condition_inputs'][0]))
                    comfy['request']['prompt']['2']['inputs']['images.image_2'] = ['1', 0]
                elif defect == 'wrong_source': bad['source']['sha256'] = 'e'*64
                else: bad['asset']['sha256'] = 'e'*64
                style._write(path, bad)
                with self.subTest(defect=defect), self.assertRaises(ValueError): style.checked_contract(run, t)
            style._write(path, provenance)
            self.assertEqual(style.checked_contract(run, t)['reference_sha256'], run['reference']['sha256'])

    def test_registration_rejects_nearby_shot_cluster_and_reflection(self):
        rng = np.random.default_rng(3)
        source = rng.uniform([40, 40], [950, 1450], (300, 2))
        reference = source + rng.normal(0, .3, source.shape)
        inliers = np.arange(300) < 200
        report = style.registration_quality(source, reference, inliers, np.eye(3), (1000, 1500), (1000, 1500), coverage=.82)
        self.assertEqual(report['inliers'], 200)
        self.assertGreater(report['spatial']['source']['hull_fraction'], .7)
        for count in (62, 71, 100, 160):
            with self.subTest(count=count), self.assertRaises(ValueError):
                style.registration_quality(source, reference, np.arange(300) < count, np.eye(3), (1000, 1500), (1000, 1500), coverage=.82)
        cluster = source*.08
        with self.assertRaises(ValueError):
            style.registration_quality(cluster, cluster, np.ones(300, bool), np.eye(3), (1000, 1500), (1000, 1500), coverage=.82)
        reflection = np.array([[-1., 0, 1000], [0, 1., 0], [0, 0, 1.]])
        reflected = source.copy(); reflected[:, 0] = 1000-source[:, 0]
        with self.assertRaises(ValueError):
            style.registration_quality(source, reflected, np.ones(300, bool), reflection, (1000, 1500), (1000, 1500), coverage=.82)
        with self.assertRaises(ValueError):
            style.registration_quality(source, reference, inliers, np.eye(3), (1000, 1500), (1000, 1500), coverage=.2)

    def test_guided_field_corrects_narrow_colour_edges_better_than_gaussian(self):
        size = 256; yy, xx = np.indices((size, size), dtype=np.float32)
        lab = np.zeros((size, size, 3), np.float32)
        lab[..., 0] = .55+.03*yy/size + ((xx+yy)%2)*.002
        lab[..., 1] = np.where((xx > 94)&(xx < 110), -.045, np.where((xx > 110)&(xx < 126), .04, 0))
        lab[..., 2] = .015
        target_lab = lab.copy(); target_lab[..., 1] = 0
        source = style._from_lab(lab); target = style._from_lab(target_lab)
        coefficients, report = style.fit_coefficients(source, target, np.ones((size, size), np.float32))
        A = coefficients[..., :9].reshape((size, size, 3, 3))
        q = lab + style._affine_delta(lab, A, coefficients[..., 9:])
        gaussian = lab + gaussian_filter(target_lab-lab, (12*size/1600, 12*size/1600, 0))
        region = np.s_[20:-20, 90:130]
        guided_error = np.mean(abs(q[region][..., 1]-target_lab[region][..., 1]))
        gaussian_error = np.mean(abs(gaussian[region][..., 1]-target_lab[region][..., 1]))
        self.assertLess(guided_error, gaussian_error*.75)
        self.assertFalse(report['semantic_material_isolation'])
        self.assertFalse(report['reference_high_frequency_RGB_used_as_output'])

    def test_target_checker_is_not_composited_and_crop_validity_is_not_a_gate(self):
        size = 192; yy, xx = np.indices((size, size))
        source = np.full((size, size, 3), .18, np.float32)
        target = source.copy(); target += np.where((xx+yy)%2, .04, -.04)[..., None].astype(np.float32)
        validity = np.ones((size, size), np.float32); validity[:, :48] = 0
        coefficients, _ = style.fit_coefficients(source, target, validity)
        # No source guide variation: A is zero. A target checker cannot turn
        # into output texture through pointwise reference RGB copying.
        self.assertLess(float(abs(coefficients[..., :9]).max()), .001)
        b = coefficients[..., 9:]
        self.assertLess(float(np.std(b[20:-20, 64:-20, 0])), .0002)
        # A uniform observed shift extends continuously outside the crop;
        # valid=0 is lack of evidence, not zero correction/application dose.
        target[:] = .23
        coefficients, _ = style.fit_coefficients(source, target, validity)
        self.assertGreater(float(coefficients[:, :24, 9].mean()), .02)
        self.assertLess(float(np.max(abs(np.diff(coefficients[..., 9], axis=1)))), .001)

    def test_same_observed_pixels_with_missing_crop_are_exact_identity(self):
        rng = np.random.default_rng(51)
        source = rng.uniform(.05, .75, (128, 192, 3)).astype(np.float32)
        validity = np.ones(source.shape[:2], np.float32); validity[:, :42] = 0
        target = source.copy(); target[:, :42] = 0
        c, _ = style.fit_coefficients(source, target, validity)
        self.assertTrue(np.array_equal(c, np.zeros_like(c)))
        with tempfile.TemporaryDirectory() as t:
            layer = frozen(t, c, (192, 128))
            self.assertTrue(np.array_equal(style.apply(source, layer, (0, 0, 192, 128), (192, 128)), source))

    def test_convex_jacobian_bounds_hold_between_nodes_without_killing_valid_gain(self):
        rng = np.random.default_rng(4)
        bad = rng.normal(0, 4, (7, 11, 12)).astype(np.float32)
        bounded, report = style.bound_jacobians(bad)
        sampled = style._sample(bounded, (0, 0, 101, 99), (101, 99))
        J = np.eye(3)+sampled[..., :9].reshape((99, 101, 3, 3))
        sv = np.linalg.svd(J, compute_uv=False)
        self.assertGreater(float(np.linalg.det(J).min()), 0)
        self.assertGreaterEqual(float(sv.min()), .1-3e-6)
        self.assertLessEqual(float(sv.max()), 3.+3e-6)
        self.assertLessEqual(float((sv[..., 0]/sv[..., -1]).max()), 30.)
        self.assertGreater(report['attenuated_nodes'], 0)
        valid = np.zeros((2, 2, 12), np.float32)
        valid[..., :9] = (np.eye(3, dtype=np.float32)*1.2).reshape(9)
        valid[..., 9:] = .02
        result, report = style.bound_jacobians(valid)
        self.assertTrue(np.array_equal(result, valid))
        self.assertEqual(report['dose_minimum'], 1.)

    def test_full_roi_and_thin_windows_are_float32_bitexact_and_portable(self):
        rng = np.random.default_rng(5); native = (289, 173)
        source = rng.uniform(.06, .65, (native[1], native[0], 3)).astype(np.float32)
        coefficients = rng.uniform(-.015, .015, (17, 23, 12)).astype(np.float32)
        with tempfile.TemporaryDirectory() as t:
            layer = frozen(Path(t)/'owned', coefficients, native)
            full, report = style.apply_with_report(source, layer, (0, 0, *native), native)
            for box in ((0, 0, 1, 173), (288, 0, 289, 173), (13, 71, 256, 170), (0, 127, 289, 130)):
                l, top, r, b = box
                roi = style.apply(source[top:b, l:r], layer, box, native)
                self.assertTrue(np.array_equal(roi, full[top:b, l:r]), box)
            self.assertFalse(report['source_texture_bitexact'])
            self.assertFalse(report['reference_RGB_composited'])
            self.assertTrue(np.array_equal(style.apply(source, layer, (0, 0, *native), native, strength=0), source))
            # Frozen replay requires only the two owned assets, no source,
            # target-temp PNG, baseline TIFF or code checkout input files.
            imported = Path(t)/'imported'; imported.mkdir()
            for value in layer['assets'].values():
                old = Path(value['path']); new = imported/old.name; new.write_bytes(old.read_bytes()); value['path'] = str(new)
            self.assertTrue(np.array_equal(style.apply(source, layer, (0, 0, *native), native), full))
            (imported/'coefficients.npz').write_bytes(b'corrupt')
            with self.assertRaises(ValueError): style.apply(source, layer, (0, 0, *native), native)

    def test_gamut_common_dose_is_bounded_continuous_and_preserves_rgb_ray(self):
        source = np.array([[[.95, .2, .05], [0., 1., .2], [.5, .5, .5]]], np.float32)
        mapped = np.array([[[2., -.3, .4], [-1., 2., .8], [.6, .7, .8]]], np.float32)
        result, dose = style._ray_gamut(source, mapped, 1.)
        self.assertGreaterEqual(float(result.min()), 0)
        self.assertLessEqual(float(result.max()), 1)
        self.assertTrue(np.array_equal(result[0, 1], source[0, 1]))
        np.testing.assert_allclose(result, source+(mapped-source)*dose[..., None], atol=1e-7)
        samples = np.linspace(.85, 1.15, 401, dtype=np.float32)
        src = np.full((1, 401, 3), .5, np.float32)
        candidate = src.copy(); candidate[..., 0] = samples
        result, _ = style._ray_gamut(src, candidate, 1.)
        self.assertLess(float(np.max(abs(np.diff(result[0, :, 0])))), .001)
        self.assertGreater(float(np.min(np.diff(result[0, :, 0]))), -1e-7)

    def test_nonfinite_coefficients_and_wrong_native_canvas_fail_closed(self):
        with tempfile.TemporaryDirectory() as t:
            source = np.full((20, 30, 3), .2, np.float32)
            c = np.zeros((3, 4, 12), np.float32)
            layer = frozen(t, c, (30, 20))
            with self.assertRaises(ValueError): style.apply(source, layer, (0, 0, 30, 20), (31, 20))
            c[0, 0, 0] = np.nan
            with self.assertRaises(ValueError): style.bound_jacobians(c)
            invalid = source.copy(); invalid[0, 0] = -1
            with self.assertRaises(ValueError): style.fit_coefficients(invalid, source, np.ones((20, 30), np.float32))

    def test_preview_is_explicit_and_uses_identical_normalized_pixel_centres(self):
        c = np.zeros((7, 11, 12), np.float32)
        c[..., 9] = np.linspace(-.01, .02, 11, dtype=np.float32)[None]
        with tempfile.TemporaryDirectory() as t:
            layer = frozen(t, c, (90, 60))
            source = np.full((20, 30, 3), .24, np.float32)
            with self.assertRaises(ValueError): style.apply(source, layer, (0, 0, 30, 20), (30, 20))
            preview = style.apply_with_report(source, layer, (0, 0, 30, 20), (30, 20), preview=True)[0]
            native = style.apply(np.full((60, 90, 3), .24, np.float32), layer, (0, 0, 90, 60), (90, 60))
            # A 3:1 scale has exactly coincident centres at native 1,4,7,...
            self.assertTrue(np.array_equal(preview, native[1::3, 1::3]))
            with self.assertRaises(ValueError): style.apply_with_report(source[:10], layer, (0, 0, 30, 10), (30, 10), preview=True)


if __name__ == '__main__':
    unittest.main()
