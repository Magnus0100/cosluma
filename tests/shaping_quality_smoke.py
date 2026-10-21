"""Objective shaping acceptance measurements; no detector, model or run writes."""
from __future__ import annotations

import copy
import unittest

import numpy as np
from scipy.ndimage import gaussian_filter

import face_contour_deform as deform
import shaping_quality as quality
from face_contour_deform_smoke import detections, parent


def textured_parent():
    """A real photograph carries detail; a bare ramp hides sub-pixel resampling."""
    rgb, G = parent()
    yy, xx = np.indices(rgb.shape[:2], dtype=np.float32)
    pattern = (.06 * np.sin(xx * .9) * np.cos(yy * .7))[..., None]
    return np.clip(rgb + pattern, 0., 1.).astype(np.float32), G


class ShapingQualityTests(unittest.TestCase):
    def mapping_and_pixels(self, amount=.3):
        rgb, G = textured_parent()
        mapping = deform.build_mapping(detections(), [512, 512], amount)
        shaped, _ = deform.sample_mapping(rgb, G, mapping)
        return mapping, rgb, shaped

    def test_report_measures_its_own_pixels_and_keeps_acceptance_separate(self):
        mapping, rgb, shaped = self.mapping_and_pixels()
        report = quality.validate(quality.measure(mapping, rgb, shaped))
        items = report["items"]
        self.assertEqual(items["untouched_regions"]["changed_outside_support"], 0)
        self.assertGreater(items["untouched_regions"]["changed_pixels"], 0)
        self.assertGreater(items["target_attainment"]["delivered_narrowing"], 0.)
        self.assertLessEqual(items["target_attainment"]["delivered_to_requested"], 1.0000001)
        self.assertTrue(items["tonal_neutrality"]["ok"])
        self.assertEqual(items["tonal_neutrality"]["maximum_delta_e_outside_support"], 0.)
        self.assertGreater(items["texture_retention"]["in_band_high_frequency_ratio"], 0.)
        self.assertFalse(report["artistic_acceptance"])
        self.assertFalse(report["human_acceptance"])
        self.assertEqual(report["model_inferences"], 0)
        self.assertEqual(report["verdict"] == "pass", not report["failed_items"])

    def test_zero_amount_is_an_exact_neutral_pass(self):
        mapping, rgb, _ = self.mapping_and_pixels(0)
        report = quality.measure(mapping, rgb, rgb)
        self.assertEqual(report["verdict"], "pass")
        self.assertEqual(report["failed_items"], [])
        self.assertEqual(report["items"]["untouched_regions"]["changed_pixels"], 0)

    def test_blurred_band_fails_texture_retention(self):
        mapping, rgb, shaped = self.mapping_and_pixels()
        left, top, right, bottom = mapping["box"]
        support = mapping["edit_support"]
        damaged = shaped.copy()
        window = damaged[top:bottom, left:right]
        window[support] = gaussian_filter(window, (1.6, 1.6, 0))[support]
        report = quality.measure(mapping, rgb, damaged)
        self.assertFalse(report["items"]["texture_retention"]["ok"])
        self.assertIn("texture_retention", report["failed_items"])

    def test_change_outside_the_declared_support_fails_untouched_regions(self):
        mapping, rgb, shaped = self.mapping_and_pixels()
        damaged = shaped.copy()
        damaged[0, 0] = np.array([1., 0., 0.], np.float32)
        report = quality.measure(mapping, rgb, damaged)
        self.assertGreater(report["items"]["untouched_regions"]["changed_outside_support"], 0)
        self.assertIn("untouched_regions", report["failed_items"])

    def test_delivered_narrowing_grows_with_the_requested_amount(self):
        strong_mapping, strong_rgb, strong_shaped = self.mapping_and_pixels(.3)
        weak_mapping, weak_rgb, weak_shaped = self.mapping_and_pixels(.15)
        strong = quality.measure(strong_mapping, strong_rgb, strong_shaped)
        weak = quality.measure(weak_mapping, weak_rgb, weak_shaped)
        self.assertGreater(strong["items"]["target_attainment"]["delivered_narrowing"],
            weak["items"]["target_attainment"]["delivered_narrowing"])

    def test_forged_or_overclaiming_reports_are_rejected(self):
        mapping, rgb, shaped = self.mapping_and_pixels()
        report = quality.measure(mapping, rgb, shaped)
        forged = copy.deepcopy(report)
        forged["items"]["texture_retention"]["ok"] = not forged["items"]["texture_retention"]["ok"]
        with self.assertRaises(ValueError):
            quality.validate(forged)
        claimed = copy.deepcopy(report)
        claimed["human_acceptance"] = True
        with self.assertRaises(ValueError):
            quality.validate(claimed)
        partial = copy.deepcopy(report)
        partial["items"] = {"target_attainment": report["items"]["target_attainment"]}
        with self.assertRaises(ValueError):
            quality.validate(partial)
        tampered = copy.deepcopy(report)
        tampered["verdict"] = "pass" if report["verdict"] == "fail" else "fail"
        with self.assertRaises(ValueError):
            quality.validate(tampered)

    def test_non_mapping_inputs_are_refused(self):
        mapping, rgb, shaped = self.mapping_and_pixels()
        with self.assertRaises(ValueError):
            quality.measure({"schema": "other"}, rgb, shaped)
        with self.assertRaises(ValueError):
            quality.measure(mapping, rgb.astype(np.uint8), shaped)
        with self.assertRaises(ValueError):
            quality.measure(mapping, rgb, shaped[:100])


if __name__ == "__main__":
    unittest.main()
