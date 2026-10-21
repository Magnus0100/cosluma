"""Visible grid evidence must not depend on invisible RGB; no model requests."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from qwen_grid_diagnostic import grid_metrics, periodic_grid_evidence
from reference_provider import qwen_quality_gate


class AlphaQualityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='qwen-alpha-quality-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        yy, xx = np.indices((256, 256))
        self.grid = np.repeat(np.uint8(((xx + yy) % 8 < 3) * 240 + 8)[..., None], 3, axis=2)
        self.alpha = np.zeros((256, 256), np.uint8)
        self.alpha[72:184, 72:184] = 255

    def save(self, name, values, mode=None):
        path = self.root / (name + '.png')
        Image.fromarray(values, mode=mode).save(path)
        return path

    def test_zero_alpha_rgb_cannot_change_visible_grid_decision(self):
        clean = np.full_like(self.grid, 128)
        hidden = np.where(self.alpha[..., None] == 0, self.grid, clean)
        first = self.save('clean', np.dstack((clean, self.alpha)))
        second = self.save('hidden-grid', np.dstack((hidden, self.alpha)))
        before = (first.read_bytes(), second.read_bytes())
        qualities = [qwen_quality_gate(path) for path in (first, second)]
        self.assertTrue(all(quality['passed'] for quality in qualities))
        self.assertFalse(any(quality['grid_detected'] for quality in qualities))
        self.assertEqual(qualities[0]['metrics']['visible_projections'], qualities[1]['metrics']['visible_projections'])
        self.assertEqual(before, (first.read_bytes(), second.read_bytes()))

    def test_opaque_rgb_and_rgba_retain_the_same_original_scores(self):
        rgb = self.save('rgb', self.grid)
        rgba = self.save('opaque-rgba', np.dstack((self.grid, np.full(self.alpha.shape, 255, np.uint8))))
        first, second = grid_metrics(rgb), grid_metrics(rgba)
        for key in ('mean_rgb8', 'std_rgb8', 'highpass_std', 'diag_lag', 'diag_lag_neighbors',
                    'anti_diag_lag', 'diag_peak_lag_4_16', 'evaluation_domain'):
            self.assertEqual(first[key], second[key], key)
        self.assertEqual(first['evaluation_domain'], 'opaque_encoded_luminance')
        self.assertFalse(qwen_quality_gate(rgb)['passed'])
        self.assertFalse(qwen_quality_gate(rgba)['passed'])

    def test_visible_grid_is_still_rejected_with_partial_opacity(self):
        image = self.save('half-opacity-grid', np.dstack((self.grid, np.full(self.alpha.shape, 128, np.uint8))))
        quality = qwen_quality_gate(image)
        self.assertFalse(quality['passed'])
        self.assertTrue(quality['periodicity']['corroborated'])
        self.assertEqual(quality['threshold'], .02)
        self.assertEqual(quality['metrics']['evaluation_domain'], 'associated_linear_rgb_over_neutral_backgrounds')
        input_quality = qwen_quality_gate(image, context='source_input')
        self.assertTrue(input_quality['passed'])
        self.assertTrue(input_quality['grid_detected'])

    def test_all_transparent_grid_has_no_visible_periodicity(self):
        image = self.save('transparent-grid', np.dstack((self.grid, np.zeros_like(self.alpha))))
        quality = qwen_quality_gate(image)
        self.assertTrue(quality['passed'])  # Grid diagnostic only, not edit fulfillment.
        self.assertFalse(quality['grid_detected'])
        self.assertEqual(quality['metrics']['opacity']['zero_pixels'], self.alpha.size)

    def test_palette_transparency_has_the_same_visible_evidence(self):
        indices = np.zeros_like(self.alpha)
        indices[self.alpha == 255] = 1
        yy, xx = np.indices(indices.shape)
        indices[(self.alpha == 0) & ((xx + yy) % 8 < 3)] = 2
        image = Image.fromarray(indices, mode='P')
        palette = [128, 128, 128, 128, 128, 128, 255, 0, 255] + [0] * (768 - 9)
        image.putpalette(palette)
        image.info['transparency'] = bytes([0, 255, 0] + [255] * 253)
        path = self.root / 'palette.png'
        image.save(path)
        rgba = self.save('straight', np.dstack((np.full_like(self.grid, 128), self.alpha)))
        self.assertEqual(grid_metrics(path)['visible_projections'], grid_metrics(rgba)['visible_projections'])

    def test_peaks_from_different_backgrounds_cannot_create_a_grid(self):
        self.assertFalse(periodic_grid_evidence(
            {'diag_lag': {'4': 1., '8': 1., '16': 1.}}, threshold=.02)['corroborated'])
        def evidence(eight, sixteen):
            return {'diag_lag': {'4': 0., '8': eight, '16': sixteen},
                'diag_lag_neighbors': {str(lag): 0. for lag in (3, 5, 7, 9, 15, 17)},
                'diag_peak_lag_4_16': max(eight, sixteen)}
        metrics = {'visible_projections': {'over_black': evidence(.9, .001),
                                          'over_white': evidence(.001, .8)}}
        result = periodic_grid_evidence(metrics, threshold=.02)
        self.assertFalse(result['corroborated'])
        metrics['visible_projections']['over_white'] = evidence(.08, .07)
        result = periodic_grid_evidence(metrics, threshold=.02)
        self.assertTrue(result['corroborated'])
        self.assertEqual(result['primary_projection'], 'over_white')

    def test_gate_reports_the_projection_with_actual_periodic_evidence(self):
        def metrics(eight, sixteen):
            return {'diag_lag': {'4': 0., '8': eight, '16': sixteen},
                'diag_lag_neighbors': {str(lag): 0. for lag in (3, 5, 7, 9, 15, 17)},
                'diag_peak_lag_4_16': max(eight, sixteen)}
        black, white = metrics(.9, .001), metrics(.08, .07)
        document = {**black, 'primary_projection': 'over_black',
                    'visible_projections': {'over_black': black, 'over_white': white}}
        with patch('qwen_grid_diagnostic.grid_metrics', return_value=document):
            quality = qwen_quality_gate(self.root / 'diagnostic-boundary.png')
        self.assertFalse(quality['passed'])
        self.assertEqual(quality['periodicity']['primary_projection'], 'over_white')
        self.assertEqual(quality['metrics']['primary_projection'], 'over_white')
        self.assertEqual(quality['diag_peak_lag_4_16'], .08)


if __name__ == '__main__':
    unittest.main()
