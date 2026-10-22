"""Objective composite acceptance measurements; no model, run or approval writes."""
from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter

import composite_quality as quality
import photo_edit_tools
import photo_editing
from raw_support import deterministic_srgb_profile

SHAPE = (384, 512)
BOX = (150, 100, 380, 320)
PARAMETERS = quality.PARAMETERS


def photograph(shape=SHAPE, seed=7):
    """A textured synthetic photograph: a blend that loses detail must show it."""
    rng = np.random.default_rng(seed)
    height, width = shape
    yy, xx = np.indices((height, width), np.float32)
    tone = .34 + .22 * (yy / height) + .05 * np.sin(xx / 70.) * np.cos(yy / 110.)
    blobs = np.zeros((height, width), np.float32)
    for cx, cy, sigma, gain in ((120., 110., 60., .10), (330., 250., 80., -.09), (430., 110., 40., .07)):
        blobs += gain * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)))
    texture = .018 * np.sin(xx * 1.7) * np.cos(yy * 1.3) + .01 * np.sin(xx * .31 + yy * .23)
    grain = rng.normal(0., .008, (height, width)).astype(np.float32)
    base = tone + blobs + texture + grain
    warm = .035 * np.sin(xx / 150.) + .02 * np.cos(yy / 90.)
    return np.clip(np.stack((base + warm, base, base - warm * .8), -1), 0., 1.).astype(np.float32)


def feather(box=BOX, width=12, shape=SHAPE):
    """A smooth write weight that reaches zero on the declared box boundary."""
    height, canvas = shape
    left, top, right, bottom = box
    yy, xx = np.indices((bottom - top, right - left), np.float32)
    edge = np.minimum(np.minimum(xx, (right - left - 1) - xx),
        np.minimum(yy, (bottom - top - 1) - yy))
    ramp = np.clip(edge / float(width), 0., 1.)
    coverage = np.zeros((height, canvas), np.float32)
    coverage[top:bottom, left:right] = ramp * ramp * (3. - 2. * ramp)
    return coverage


def box_mask(box=BOX, shape=SHAPE):
    coverage = np.zeros(shape, np.float32)
    coverage[box[1]:box[3], box[0]:box[2]] = 1.
    return coverage


def blend(parent, patch, coverage):
    return (parent * (1. - coverage[..., None]) + patch * coverage[..., None]).astype(np.float32)


def pixel_digest(values):
    body = np.ascontiguousarray(values, np.float32)
    prefix = f'{body.shape[0]}x{body.shape[1]}x{body.shape[2]}:'.encode()
    return hashlib.sha256(prefix + body.tobytes()).hexdigest()


class CompositeQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent = photograph()
        yy, xx = np.indices(cls.parent.shape[:2], np.float32)
        cls.patch = np.clip(cls.parent * 1.02 + .004, 0., 1.).astype(np.float32)
        cls.coverage = feather()
        cls.composite = blend(cls.parent, cls.patch, cls.coverage)
        cls.hard_coverage = box_mask()
        cls.hard = cls.parent.copy()
        cls.hard[BOX[1]:BOX[3], BOX[0]:BOX[2]] = np.clip(
            cls.patch[BOX[1]:BOX[3], BOX[0]:BOX[2]] + .05, 0., 1.).astype(np.float32)
        blemish = .15 * np.exp(-(((xx - 250.) ** 2 + (yy - 190.) ** 2) / (2 * 6. ** 2)))
        cls.retouch_parent = np.clip(cls.parent - blemish[..., None], 0., 1.).astype(np.float32)
        cls.retouch_box = (200, 140, 300, 240)
        cls.retouch_coverage = feather(box=cls.retouch_box, width=6)
        cls.retouch = blend(cls.retouch_parent, cls.patch, cls.retouch_coverage)
        warm = np.clip(cls.parent * np.array([1.05, 1., .94], np.float32), 0., 1.).astype(np.float32)
        cls.warm_coverage = feather(width=3)
        cls.warm = blend(cls.parent, warm, cls.warm_coverage)
        cls.outside_change = cls.composite.copy()
        cls.outside_change[:, 0:30] = np.clip(cls.outside_change[:, 0:30] + 2 / 255., 0., 1.)
        selector = cls.coverage[..., None] >= .5
        blurred = np.where(selector, gaussian_filter(cls.composite, (2., 2., 0.)), cls.composite)
        cls.blurred = blurred.astype(np.float32)
        rng = np.random.default_rng(11)
        cls.noisy = np.clip(cls.composite
            + rng.normal(0., .03, cls.composite.shape).astype(np.float32) * selector, 0., 1.).astype(np.float32)
        inside = distance_transform_edt(cls.coverage >= .5)
        cls.ring = np.exp(-((inside - 6.5) ** 2) / (2 * 1.8 ** 2)) * (inside >= 1)
        cls.full = np.clip(cls.parent * 1.03 + .005, 0., 1.).astype(np.float32)
        cls.full_blurred = gaussian_filter(cls.parent, (2., 2., 0.)).astype(np.float32)

    def rim(self, amplitude):
        return np.clip(self.composite + amplitude * self.ring[..., None], 0., 1.).astype(np.float32)

    def measure(self, result, parent=None, **options):
        options.setdefault('work_type', 'partial_composite')
        if options['work_type'] == 'partial_composite' or options['work_type'] == 'retouch':
            options.setdefault('coverage', self.coverage if parent is None else self.retouch_coverage)
            options.setdefault('region', list(BOX) if parent is None else list(self.retouch_box))
        return quality.measure(self.parent if parent is None else parent, result, **options)

    @staticmethod
    def save(path, values):
        Image.fromarray(np.uint8(np.round(values * 255.))).save(
            path, icc_profile=deterministic_srgb_profile())

    def test_relative_contrast_and_chroma_are_reported_without_touching_the_verdict(self):
        report = quality.validate(self.measure(self.composite, label='fixture'))
        detail = report['items']['generated_detail']
        tonality = report['items']['cross_boundary_tonality']
        # The judged reading keeps exactly its old threshold formula; the new readings ride beside it.
        self.assertEqual(detail['ok'], bool(
            quality.PARAMETERS['detail_ratio_min'] <= detail['high_frequency_ratio']
            <= quality.PARAMETERS['detail_ratio_max']
            and detail['noise_ratio'] <= quality.PARAMETERS['noise_ratio_max']))
        self.assertGreater(detail['generated_level'], 0.)
        self.assertGreater(detail['parent_generated_level'], 0.)
        self.assertAlmostEqual(detail['generated_relative_detail'],
            detail['generated_high_frequency'] / detail['generated_level'], places=12)
        self.assertAlmostEqual(detail['parent_relative_detail'],
            detail['parent_region_high_frequency'] / detail['parent_generated_level'], places=12)
        self.assertAlmostEqual(detail['relative_detail_ratio'],
            detail['generated_relative_detail'] / detail['parent_relative_detail'], places=9)
        self.assertIn('only high_frequency_ratio and noise_ratio', detail['relative_detail_note'])
        # Chroma runs the same step machinery and is explicitly never a gate.
        self.assertAlmostEqual(tonality['chroma_excess'],
            tonality['chroma_step'] - tonality['chroma_content_step'], places=12)
        self.assertEqual(tonality['chroma_excess_role'], 'advisory_not_gating_no_calibrated_bound')
        self.assertEqual(tonality['ok'], bool(abs(tonality['luma_excess']) <= quality.PARAMETERS['tonality_step_max']
            and abs(tonality['warmth_excess']) <= quality.PARAMETERS['warmth_step_max']))

    def test_work_types_follow_the_editing_scope(self):
        self.assertEqual(quality.WORK_TYPES, tuple(photo_editing.WORK_TYPES))
        self.assertEqual(set(quality.WORK_TYPES), {'retouch', 'partial_composite', 'full_composite'})

    def test_clean_partial_composite_passes_without_claiming_acceptance(self):
        report = quality.validate(self.measure(self.composite, label='fixture'))
        self.assertEqual(report['verdict'], 'pass')
        self.assertEqual(report['failed_items'], [])
        self.assertEqual(report['not_applicable_items'], [])
        self.assertEqual(set(report['items']), {'untouched_regions', 'boundary_seam',
            'cross_boundary_tonality', 'hue_step_excess', 'generated_detail', 'boundary_halo'})
        self.assertEqual(report['items']['untouched_regions']['changed_outside_region'], 0)
        self.assertEqual(report['items']['untouched_regions']['changed_under_zero_coverage'], 0)
        self.assertEqual(report['items']['generated_detail']['basis'], 'untouched_original_pixels')
        self.assertGreater(report['items']['untouched_regions']['changed_pixels'], 0)
        self.assertGreater(report['items']['generated_detail']['generated_fraction_of_canvas'], 0.)
        self.assertEqual(report['items']['boundary_halo']['halo_width'], 0)
        self.assertEqual(report['label'], 'fixture')
        self.assertEqual(report['canvas_size'], [SHAPE[1], SHAPE[0]])
        self.assertEqual(report['region'], list(BOX))
        self.assertFalse(report['artistic_acceptance'])
        self.assertFalse(report['human_acceptance'])
        self.assertEqual(report['model_inferences'], 0)
        self.assertEqual(len(report['report_sha256']), 64)
        self.assertEqual(report['coverage']['sha256'], pixel_digest(self.coverage[..., None]))

    def test_retouch_path_measures_the_repaired_region(self):
        report = quality.validate(self.measure(self.retouch, parent=self.retouch_parent,
            work_type='retouch'))
        self.assertEqual(report['work_type'], 'retouch')
        self.assertEqual(report['verdict'], 'pass')
        self.assertEqual(report['failed_items'], [])

    def test_hard_join_fails_seam_and_tonality(self):
        report = quality.validate(quality.measure(self.parent, self.hard,
            work_type='partial_composite', coverage=self.hard_coverage, region=list(BOX)))
        self.assertIn('boundary_seam', report['failed_items'])
        self.assertIn('cross_boundary_tonality', report['failed_items'])
        self.assertGreater(report['items']['boundary_seam']['gradient_ratio'],
            PARAMETERS['seam_gradient_ratio_max'])
        self.assertGreater(abs(report['items']['cross_boundary_tonality']['luma_excess']),
            PARAMETERS['tonality_step_max'])

    def test_colour_temperature_break_fails_tonality_and_the_hue_axis(self):
        report = quality.validate(quality.measure(self.parent, self.warm, work_type='partial_composite',
            coverage=self.warm_coverage, region=list(BOX)))
        # A colour-temperature break rotates the colour relation at the cut, so the
        # hue-type item is expected to report it too; neither item replaces the other.
        self.assertEqual(report['failed_items'], ['cross_boundary_tonality'])
        self.assertEqual(report['provisional_items'], ['hue_step_excess'])
        self.assertNotIn('hue_step_excess', report['failed_items'])
        self.assertGreater(abs(report['items']['cross_boundary_tonality']['warmth_excess']),
            PARAMETERS['warmth_step_max'])
        self.assertGreater(report['items']['hue_step_excess']['excess_dE76'],
            PARAMETERS['hue_step_excess_max_dE76'])
        self.assertEqual(report['items']['hue_step_excess']['baseline'],
            'parent_canvas_own_step_across_the_same_shells')
        self.assertTrue(report['items']['boundary_seam']['ok'])

    def test_change_outside_the_target_region_fails_untouched_regions(self):
        report = quality.validate(self.measure(self.outside_change))
        self.assertEqual(report['failed_items'], ['untouched_regions'])
        self.assertGreater(report['items']['untouched_regions']['changed_outside_region'], 0)
        self.assertGreater(report['items']['untouched_regions']['changed_under_zero_coverage'], 0)

    def test_blurred_generated_content_fails_detail(self):
        report = quality.validate(self.measure(self.blurred))
        self.assertEqual(report['failed_items'], ['generated_detail'])
        self.assertLess(report['items']['generated_detail']['high_frequency_ratio'],
            PARAMETERS['detail_ratio_min'])
        self.assertGreater(report['items']['generated_detail']['original_noise'], 0.)

    def test_added_grain_in_generated_content_fails_detail(self):
        report = quality.validate(self.measure(self.noisy))
        # Known sensitivity, recorded rather than tuned away: grain moves the
        # colour of the first written shell, so this fixture crosses the
        # provisional hue bound as well. Two real products cannot separate that
        # reading from a real hue rotation.
        self.assertEqual(report['failed_items'], ['generated_detail'])
        self.assertGreater(report['items']['generated_detail']['noise_ratio'],
            PARAMETERS['noise_ratio_max'])
        self.assertGreater(report['items']['hue_step_excess']['excess_dE76'],
            PARAMETERS['hue_step_excess_max_dE76'])

    def test_hue_step_is_measured_against_the_photograph_and_never_replaces_tonality(self):
        report = quality.validate(self.measure(self.composite, label='fixture'))
        hue = report['items']['hue_step_excess']
        tonality = report['items']['cross_boundary_tonality']
        # Both axes stand side by side; the luminance/warmth verdict keeps exactly its own formula.
        self.assertEqual(tonality['ok'], bool(
            abs(tonality['luma_excess']) <= PARAMETERS['tonality_step_max']
            and abs(tonality['warmth_excess']) <= PARAMETERS['warmth_step_max']))
        self.assertEqual(hue['ok'], bool(
            hue['excess_dE76'] <= PARAMETERS['hue_step_excess_max_dE76']))
        self.assertEqual(hue['baseline'], 'parent_canvas_own_step_across_the_same_shells')
        self.assertEqual(hue['judged_axis'], 'signed_CIELAB_a_star_and_b_star_vector')
        self.assertAlmostEqual(hue['photo_chroma_step'],
            float(np.hypot(hue['photo_step_da'], hue['photo_step_db'])), places=5)
        self.assertAlmostEqual(hue['delivered_chroma_step'],
            float(np.hypot(hue['delivered_step_da'], hue['delivered_step_db'])), places=5)
        self.assertAlmostEqual(hue['excess_dE76'],
            float(np.hypot(hue['delivered_step_da'] - hue['photo_step_da'],
                hue['delivered_step_db'] - hue['photo_step_db'])), places=5)

    def test_hue_step_baseline_is_the_photographs_own_step(self):
        # The same delivered canvas over two different photographs has to read
        # differently, because the baseline is the photograph's own step and not a constant.
        tinted = self.parent.copy()
        tinted[BOX[1]:BOX[3], BOX[0]:BOX[2]] = np.clip(
            tinted[BOX[1]:BOX[3], BOX[0]:BOX[2]] * np.array([1., .82, 1.12], np.float32), 0., 1.)
        near = quality.validate(quality.measure(self.parent, self.composite,
            work_type='partial_composite', coverage=self.coverage,
            region=list(BOX)))['items']['hue_step_excess']
        far = quality.validate(quality.measure(tinted, self.composite,
            work_type='partial_composite', coverage=self.coverage,
            region=list(BOX)))['items']['hue_step_excess']
        self.assertNotAlmostEqual(near['photo_chroma_step'], far['photo_chroma_step'], places=3)
        self.assertAlmostEqual(near['delivered_chroma_step'], far['delivered_chroma_step'], places=9)
        self.assertNotAlmostEqual(near['excess_dE76'], far['excess_dE76'], places=3)

    def test_hue_step_records_its_provisional_calibration(self):
        report = quality.validate(self.measure(self.composite))
        evidence = report['items']['hue_step_excess']['threshold_evidence']
        self.assertEqual(evidence, quality.HUE_STEP_EVIDENCE)
        self.assertEqual(evidence['samples'], 2)
        self.assertEqual(evidence['status'], 'provisional_insufficient_sample')
        self.assertGreater(evidence['required_samples'], evidence['samples'])
        self.assertEqual(len(evidence['measured_excess_dE76']), 2)
        self.assertIn('unverified', evidence['note'])
        # The calibration cannot be rewritten inside a report.
        forged = copy.deepcopy(report)
        forged['items']['hue_step_excess']['threshold_evidence']['samples'] = 40
        with self.assertRaises(ValueError):
            quality.validate(forged)

    def test_hue_step_rejects_a_reading_that_does_not_match_its_own_vectors(self):
        report = quality.validate(self.measure(self.composite))
        forged = copy.deepcopy(report)
        item = forged['items']['hue_step_excess']
        item['excess_dE76'] = 0.
        item['ok'] = True
        forged['failed_items'] = [name for name in quality.GATING_ITEMS if not forged['items'][name]['ok']]
        forged['verdict'] = 'fail' if forged['failed_items'] else 'pass'
        forged['report_sha256'] = quality._digest(forged)
        with self.assertRaises(ValueError):
            quality.validate(forged)

    def test_bright_and_dark_rims_report_a_halo_width(self):
        for amplitude in (.03, -.03):
            with self.subTest(amplitude=amplitude):
                report = quality.validate(self.measure(self.rim(amplitude)))
                self.assertEqual(report['failed_items'], ['boundary_halo'])
                self.assertGreater(report['items']['boundary_halo']['halo_width'],
                    PARAMETERS['halo_width_max'])
                self.assertGreater(report['items']['boundary_halo']['visible_excess'],
                    PARAMETERS['halo_visible'])
                self.assertTrue(report['items']['cross_boundary_tonality']['ok'])
                self.assertTrue(report['items']['boundary_seam']['ok'])

    def test_whole_canvas_composite_measures_against_the_parent(self):
        report = quality.validate(quality.measure(self.parent, self.full,
            work_type='full_composite', coverage={'uniform': 1}))
        self.assertEqual(report['verdict'], 'pass')
        self.assertEqual(report['not_applicable_items'],
            ['boundary_seam', 'cross_boundary_tonality', 'hue_step_excess', 'boundary_halo'])
        self.assertEqual(report['region'], None)
        self.assertTrue(report['items']['untouched_regions']['scope_covers_canvas'])
        self.assertEqual(report['items']['generated_detail']['basis'], 'immediate_parent_pixels')
        self.assertEqual(report['coverage'], {'uniform': 1})

    def test_whole_canvas_join_with_a_preserved_border_is_measured(self):
        whole = feather(box=(0, 0, SHAPE[1], SHAPE[0]), width=10)
        report = quality.validate(quality.measure(self.parent, blend(self.parent, self.patch, whole),
            work_type='full_composite', coverage=whole, region=(0, 0, SHAPE[1], SHAPE[0])))
        self.assertEqual(report['verdict'], 'pass')
        self.assertEqual(report['not_applicable_items'], [])
        self.assertTrue(report['items']['untouched_regions']['scope_covers_canvas'])
        self.assertGreater(report['items']['boundary_seam']['parent_gradient'], 0.)

    def test_whole_canvas_blur_fails_detail(self):
        report = quality.validate(quality.measure(self.parent, self.full_blurred,
            work_type='full_composite', coverage=1))
        self.assertEqual(report['failed_items'], ['generated_detail'])
        self.assertEqual(report['not_applicable_items'],
            ['boundary_seam', 'cross_boundary_tonality', 'hue_step_excess', 'boundary_halo'])

    def test_report_digest_and_item_verdicts_reject_forgery(self):
        report = quality.validate(self.measure(self.composite))
        forged = copy.deepcopy(report)
        forged['items']['boundary_halo']['ok'] = not forged['items']['boundary_halo']['ok']
        with self.assertRaises(ValueError):
            quality.validate(forged)
        claimed = copy.deepcopy(report)
        claimed['human_acceptance'] = True
        with self.assertRaises(ValueError):
            quality.validate(claimed)
        inferred = copy.deepcopy(report)
        inferred['model_inferences'] = 1
        with self.assertRaises(ValueError):
            quality.validate(inferred)
        partial = copy.deepcopy(report)
        del partial['items']['boundary_seam']
        with self.assertRaises(ValueError):
            quality.validate(partial)
        dropped = copy.deepcopy(report)
        dropped['items']['untouched_regions'].pop('changed_under_zero_coverage')
        with self.assertRaises(ValueError):
            quality.validate(dropped)
        flipped = copy.deepcopy(report)
        flipped['verdict'] = 'fail' if report['verdict'] == 'pass' else 'pass'
        with self.assertRaises(ValueError):
            quality.validate(flipped)
        loosened = copy.deepcopy(report)
        loosened['parameters']['halo_width_max'] = 40
        with self.assertRaises(ValueError):
            quality.validate(loosened)
        added = copy.deepcopy(report)
        added['approved'] = True
        with self.assertRaises(ValueError):
            quality.validate(added)
        tampered = copy.deepcopy(report)
        tampered['report_sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            quality.validate(tampered)

    def test_report_binds_the_pixels_it_measured(self):
        first = self.measure(self.composite)
        second = self.measure(self.hard)
        self.assertNotEqual(first['report_sha256'], second['report_sha256'])
        self.assertEqual(first['inputs']['parent_sha256'], second['inputs']['parent_sha256'])
        self.assertNotEqual(first['inputs']['result_sha256'], second['inputs']['result_sha256'])
        delivered = quality.measure(np.uint8(np.round(self.parent * 255.)),
            np.uint8(np.round(self.composite * 255.)), work_type='partial_composite',
            coverage=self.coverage, region=list(BOX))
        quality.validate(delivered)
        self.assertEqual(delivered['verdict'], 'pass')
        self.assertNotEqual(delivered['inputs']['parent_sha256'], first['inputs']['parent_sha256'])

    def test_invalid_composites_are_refused(self):
        def refused(*args, **options):
            with self.assertRaises(ValueError):
                quality.measure(*args, **options)
        refused(self.parent, self.composite, work_type='scene')
        refused(self.parent, self.composite.astype(np.float64), work_type='partial_composite',
            coverage=self.coverage, region=list(BOX))
        refused(self.parent, self.composite[:-4], work_type='partial_composite',
            coverage=self.coverage, region=list(BOX))
        refused(self.parent, self.composite, work_type='partial_composite', coverage=self.coverage)
        refused(self.parent, self.composite, work_type='partial_composite',
            coverage=np.zeros(SHAPE, np.float32), region=list(BOX))
        refused(self.parent, self.composite, work_type='partial_composite',
            coverage=self.coverage, region=(0, 0, 4, 4))
        refused(self.parent, self.composite, work_type='partial_composite',
            coverage=self.coverage, region=(-10, 0, 100, 100))
        refused(self.parent, self.composite, work_type='partial_composite',
            coverage=self.coverage.transpose(), region=list(BOX))
        refused(self.parent, self.composite, work_type='full_composite',
            coverage=self.coverage, region=list(BOX))
        refused(self.parent, self.parent, work_type='partial_composite',
            coverage=self.coverage, region=list(BOX))
        refused(self.parent, self.composite * 2., work_type='partial_composite',
            coverage=self.coverage, region=list(BOX))
        refused(self.parent, self.composite, work_type='partial_composite',
            coverage=self.coverage, region=list(BOX), label='x' * 300)
        leaked = self.coverage.copy()
        leaked[5, 5] = 1.
        refused(self.parent, self.composite, work_type='partial_composite',
            coverage=leaked, region=list(BOX))
        broken = self.composite.copy()
        broken[0, 0, 0] = np.nan
        refused(self.parent, broken, work_type='partial_composite',
            coverage=self.coverage, region=list(BOX))

    def test_cli_measures_finished_files_and_signals_the_verdict(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            parent_path, result_path = directory / 'parent.png', directory / 'delivered.png'
            self.save(parent_path, self.parent)
            self.save(result_path, self.composite)
            coverage_path = directory / 'edit-coverage.npy'
            np.save(coverage_path, self.coverage)
            report_path = directory / 'quality.json'
            arguments = ['--parent', str(parent_path), '--result', str(result_path),
                '--coverage', str(coverage_path), '--work-type', 'partial_composite',
                '--region', ','.join(str(value) for value in BOX), '--label', 'fixture',
                '--out', str(report_path)]
            with redirect_stdout(StringIO()) as printed:
                self.assertEqual(quality.main(arguments), 0)
            self.assertTrue(json.loads(printed.getvalue()))
            report = quality.validate(json.loads(report_path.read_text(encoding='utf8')))
            self.assertEqual(report['verdict'], 'pass')
            self.assertEqual(report['label'], 'fixture')
            mask_path = directory / 'edit-coverage.png'
            Image.fromarray(np.uint8(np.round(self.coverage * 255.))).save(mask_path)
            arguments[arguments.index(str(coverage_path))] = str(mask_path)
            with redirect_stdout(StringIO()):
                self.assertEqual(quality.main(arguments), 0)
            self.save(directory / 'hard.png', self.hard)
            failing = list(arguments)
            failing[failing.index(str(result_path))] = str(directory / 'hard.png')
            with redirect_stdout(StringIO()):
                self.assertEqual(quality.main(failing), 1)
            invalid = list(arguments)
            invalid[invalid.index(','.join(str(value) for value in BOX))] = '0,0,4,4'
            with redirect_stdout(StringIO()):
                self.assertEqual(quality.main(invalid), 2)
            self.save(directory / 'full.png', self.full)
            with redirect_stdout(StringIO()):
                self.assertEqual(quality.main(['--parent', str(parent_path), '--result',
                    str(directory / 'full.png'), '--work-type', 'full_composite']), 0)
            with redirect_stdout(StringIO()):
                self.assertEqual(quality.main(['--parent', str(directory / 'missing.png'),
                    '--result', str(result_path), '--work-type', 'full_composite']), 2)

    def test_measures_a_composite_encoded_by_the_production_path(self):
        from float_appearance import srgb_to_linear
        from raw_support import linear_preview
        box = (140, 90, 380, 300)
        left, top, right, bottom = box
        linear = np.asarray(srgb_to_linear(self.parent), np.float32)
        mask = photo_edit_tools._mask((right - left, bottom - top))
        coverage = np.zeros(linear.shape[:2], np.float32)
        coverage[top:bottom, left:right] = mask
        prior = linear[top:bottom, left:right]

        def composed(patch):
            master = linear.copy()
            mixed = prior * (1. - mask[..., None]) + patch * mask[..., None]
            master[top:bottom, left:right] = np.where((mask > 0.)[..., None], mixed, prior)
            return master

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)

            def write(values, name):
                image = linear_preview(values)
                path = directory / name
                image.save(path, format='PNG', icc_profile=image.info['icc_profile'])
                return path

            parent_path = write(linear, 'parent.png')
            coverage_path = directory / 'edit-coverage.npy'
            np.save(coverage_path, coverage)
            delivered = write(composed(np.clip(prior * 1.02 + .01, 0., 1.)), 'delivered.png')
            report = quality.validate(quality.measure(quality.load_canvas(parent_path),
                quality.load_canvas(delivered), work_type='partial_composite',
                coverage=quality.load_coverage(coverage_path), region=list(box),
                label='production-regional'))
            self.assertEqual(report['verdict'], 'pass')
            self.assertEqual(report['failed_items'], [])
            self.assertEqual(report['items']['untouched_regions']['changed_outside_region'], 0)
            self.assertEqual(report['items']['untouched_regions']['changed_under_zero_coverage'], 0)
            self.assertGreater(report['items']['boundary_seam']['parent_gradient'], 0.)
            softened = write(composed(gaussian_filter(prior, (2., 2., 0.))), 'softened.png')
            softened_report = quality.validate(quality.measure(quality.load_canvas(parent_path),
                quality.load_canvas(softened), work_type='partial_composite',
                coverage=quality.load_coverage(coverage_path), region=list(box)))
            self.assertEqual(softened_report['failed_items'], ['generated_detail'])


class ProductionJoinTests(unittest.TestCase):
    """The production regional join, measured with the same acceptance items."""

    SHAPE = (768, 1024)
    CORE = (176, 96, 976, 720)

    def setUp(self):
        self.parent = photograph(self.SHAPE, seed=23)
        self.left, self.top, self.right, self.bottom = self.CORE
        self.patch = self.parent.copy()

    def _region(self, values):
        return values[self.top:self.bottom, self.left:self.right]

    def _replace(self, local):
        patch = self.parent.copy()
        patch[self.top:self.bottom, self.left:self.right] = local
        return patch

    def legacy(self, patch):
        """The former production composition: one fixed rectangular cross-fade."""
        mask = photo_edit_tools._mask((self.right-self.left, self.bottom-self.top))
        prior = self._region(self.parent)
        mixed = prior*(1-mask[..., None])+self._region(patch)*mask[..., None]
        result = self.parent.copy()
        result[self.top:self.bottom, self.left:self.right] = mixed
        coverage = np.zeros(self.SHAPE, np.float32)
        coverage[self.top:self.bottom, self.left:self.right] = mask
        return result, coverage

    def joined(self, patch, alpha=None, subject=None, subject_only=False):
        return photo_edit_tools.fuse_region(self.parent, patch, alpha, core=list(self.CORE),
            box=[0, 0, self.SHAPE[1], self.SHAPE[0]], subject=subject, subject_only=subject_only)

    def matte(self, centre=(576., 400.), radii=(220., 160.), softness=0.04, shape=None):
        """A silhouette-shaped subject weight with a real transition around it."""
        shape = shape or self.SHAPE
        yy, xx = np.indices(shape, np.float32)
        inside = 1.-(((xx-centre[0])/radii[0])**2+((yy-centre[1])/radii[1])**2)
        return np.clip(inside/np.float32(softness), 0., 1.).astype(np.float32)

    def core_mask(self):
        mask = np.zeros(self.SHAPE, bool)
        mask[self.top:self.bottom, self.left:self.right] = True
        return mask

    @staticmethod
    def _luma(values):
        return values @ np.asarray([.2126, .7152, .0722], np.float32)

    def _join_tone_error(self, result, coverage=None):
        """Low-frequency tone error in the pixels just inside the cut."""
        offset = gaussian_filter(self._luma(result)-self._luma(self.parent), 2., mode='nearest')
        near = photo_edit_tools._edge_distance((self.right-self.left,
            self.bottom-self.top)) <= 3.
        region = np.zeros(self.SHAPE, bool)
        region[self.top:self.bottom, self.left:self.right] = near
        return float(np.median(np.abs(offset[region])))

    def _region_ring_level(self, result):
        """The audit region/ring low-frequency level, as log luma."""
        left, top, right, bottom = self.CORE
        region = np.zeros(self.SHAPE, bool)
        region[top:bottom, left:right] = True
        around = np.zeros(self.SHAPE, bool)
        around[top-40:bottom+40, left-40:right+40] = True
        ring = around & ~region
        grey = self._luma(result)
        return float(np.log(max(float(grey[region].mean()), 1e-9)
            /max(float(grey[ring].mean()), 1e-9)))

    def _band_texture(self, result, coverage):
        """Delivered detail energy in the join band against the patch interior."""
        grey = self._luma(result)
        highpass = grey-gaussian_filter(grey, 1.6, mode='nearest')
        region = np.zeros(self.SHAPE, bool)
        region[self.top:self.bottom, self.left:self.right] = True
        band = region & (coverage > .05) & (coverage < .95)
        far = photo_edit_tools._edge_distance((self.right-self.left, self.bottom-self.top))
        interior = np.zeros(self.SHAPE, bool)
        interior[self.top:self.bottom, self.left:self.right] = far > 96
        interior &= coverage >= .99
        return float(np.std(highpass[band])/max(np.std(highpass[interior]), 1e-9))

    def test_production_join_records_its_policy_and_keeps_untouched_bytes(self):
        local = np.clip(self._region(self.parent)*np.asarray([.97, .99, 1.05], np.float32), 0., 1.)
        result, coverage, record = self.joined(self._replace(local.astype(np.float32)))
        self.assertEqual(record['mode'], 'registered_photographic_join')
        self.assertEqual(record['working_domain'], 'display_linear_sRGB')
        self.assertFalse(record['alpha_modified'])
        self.assertFalse(record['coordinates_changed'])
        self.assertFalse(record['generated_texture_added'])
        self.assertFalse(record['visual_acceptance'])
        self.assertTrue(record['alpha_in_destination_only'])
        self.assertLessEqual(record['tone_correction_max'], photo_edit_tools.FUSION['tone_bound'])
        self.assertFalse(record['registration']['clamped'])
        self.assertGreater(float(record['boundary_pixels']), 0)
        self.assertEqual(record['policy'], photo_edit_tools.FUSION)
        # Region-scale registration is recorded with the ring it was measured on.
        self.assertIsNotNone(record['registration'])
        self.assertGreater(record['registration']['ring_pixels'], 0)
        self.assertEqual(len(record['registration']['scale']), 3)
        low_ev, high_ev = photo_edit_tools.FUSION['registration_gain_ev_bounds']
        for value in record['registration']['scale']:
            self.assertGreaterEqual(value, 2.**low_ev-1e-6)
            self.assertLessEqual(value, 2.**high_ev+1e-6)
        # Texture matching reports its measured gains and its achieved ratios.
        self.assertGreater(record['detail']['structure_gain_median'], 0)
        self.assertGreater(record['detail']['noise_gain_median'], 0)
        self.assertEqual(record['detail']['structure_gain_bounds'],
            list(photo_edit_tools.FUSION['detail_ratio_bounds']))
        self.assertGreater(record['achieved_high_frequency_ratio'], 0)
        self.assertGreater(record['achieved_noise_ratio'], 0)
        detail = record['join_detail']
        self.assertEqual(detail['band_px'], photo_edit_tools.FUSION['band_inner_px'])
        self.assertEqual(detail['interior_px'], photo_edit_tools.FUSION['band_interior_px'])
        self.assertIsNotNone(detail['delivered'])
        self.assertIsNotNone(detail['parent'])
        # The membrane is geometry-driven, so it is exactly zero in the interior
        # (a patch that already matches the photograph needs none at all).
        self.assertGreaterEqual(record['tone_correction_max'], 0)
        self.assertLessEqual(record['tone_correction_max'], photo_edit_tools.FUSION['tone_bound'])
        self.assertEqual(record['tone_correction_interior_max'], 0)
        self.assertGreater(coverage.max(), .999)
        self.assertTrue(np.any((coverage > 0) & (coverage < 1)))
        self.assertTrue(np.array_equal(result[coverage == 0], self.parent[coverage == 0]))

    def test_the_join_registers_the_region_scale_onto_the_retained_ring(self):
        factors = ([.55, .70, .85], [.40, .50, .62])
        delivered = []
        for factor in factors:
            local = np.clip(self._region(self.parent)*np.asarray(factor, np.float32), 0., 1.)
            patch = self._replace(local.astype(np.float32))
            result, coverage, record = self.joined(patch)
            level = self._region_ring_level(result)
            delivered.append((level, record))
            former = self._region_ring_level(self.legacy(patch)[0])
            # The former composition kept the provider's own exposure: the
            # region/ring level moved with the patch it was asked to inset.
            self.assertGreater(abs(former), abs(level)+.15)
            # The cut itself still carries the parent's tone.
            self.assertLess(self._join_tone_error(result), 2/255.)
            self.assertEqual(record['tone_correction_interior_max'], 0)
        # Registration makes the delivered level independent of that exposure.
        self.assertLess(abs(delivered[0][0]-delivered[1][0]), .01)
        self.assertGreater(max(delivered[1][1]['registration']['scale']),
            max(delivered[0][1]['registration']['scale']))

    def test_the_membrane_still_closes_what_one_gain_cannot(self):
        """A lighting gradient inside the patch needs the local membrane."""
        patch = self._graded_patch()
        result, coverage, record = self.joined(patch)
        self.assertGreater(record['tone_correction_pixels'], 0)
        self.assertGreater(record['tone_correction_max'], 0.)
        self.assertLessEqual(record['tone_correction_max'], photo_edit_tools.FUSION['tone_bound'])
        # ... and it still stops exactly at the recorded depth.
        self.assertEqual(record['tone_correction_interior_max'], 0)
        self.assertLess(self._join_tone_error(result), 2/255.)

    def test_the_join_band_carries_one_texture_not_two_ghosts(self):
        # A provider patch that is genuinely smoother than the photograph: the
        # former 50px cross-fade averaged two textures in the band, so the band
        # no longer looked like either of them.
        local = gaussian_filter(self._region(self.parent), (.6, .6, 0.)).astype(np.float32)
        patch = self._replace(local)
        result, coverage, record = self.joined(patch)
        former_result, former_coverage = self.legacy(patch)
        restored = self._band_texture(result, coverage)
        former = self._band_texture(former_result, former_coverage)
        self.assertLess(abs(restored-1.), .10)
        self.assertTrue(former < .9 or former > 1.25)
        # The same number the audit uses, recomputed by the join itself.
        self.assertLess(abs(float(record['join_detail']['delivered'])-1.), .35)

    def register(self, detail_sigma):
        smooth = gaussian_filter(self._region(self.parent),
            (detail_sigma, detail_sigma, 0.)).astype(np.float32)
        patch = self._replace(smooth)
        result, coverage, record = self.joined(patch)
        former_result, former_coverage = self.legacy(patch)
        after = quality.measure(self.parent, result, work_type='partial_composite',
            coverage=coverage, region=list(self.CORE))['items']['generated_detail']
        before = quality.measure(self.parent, former_result, work_type='partial_composite',
            coverage=former_coverage, region=list(self.CORE))['items']['generated_detail']
        return before, after, record

    def test_generated_detail_is_registered_to_the_parent_photograph(self):
        before, after, record = self.register(.6)
        self.assertFalse(before['ok'])
        self.assertTrue(after['ok'])
        self.assertGreater(after['high_frequency_ratio'], before['high_frequency_ratio'])
        self.assertGreaterEqual(float(record['detail']['structure_gain_median']), 1.)
        self.assertLessEqual(float(record['detail']['structure_gain_median']),
            photo_edit_tools.FUSION['detail_ratio_bounds'][1])
        self.assertGreater(float(record['achieved_high_frequency_ratio']), 0.)

    def test_detail_registration_is_bounded_and_never_invents_texture(self):
        before, after, record = self.register(1.2)
        self.assertFalse(before['ok'])
        self.assertGreater(after['high_frequency_ratio'], 2.5*before['high_frequency_ratio'])
        # The gain stays inside the recorded bound and the join never claims to
        # have painted texture that the provider did not produce.
        low_bound, high_bound = photo_edit_tools.FUSION['detail_ratio_bounds']
        self.assertGreaterEqual(float(record['detail']['structure_gain_median']), 3.)
        self.assertLessEqual(float(record['detail']['structure_gain_median']), high_bound)
        self.assertGreaterEqual(float(record['detail']['noise_gain_median']), 1.)
        self.assertFalse(record['generated_texture_added'])
        self.assertGreater(float(record['achieved_high_frequency_ratio']), .25)
        self.assertLess(float(record['achieved_high_frequency_ratio']), 1.8)

    def test_small_edits_keep_the_bounded_former_join_with_a_recorded_reason(self):
        core = (24, 32, 72, 96)
        result, coverage, record = photo_edit_tools.fuse_region(self.parent, self.patch, None,
            core=list(core), box=[0, 0, self.SHAPE[1], self.SHAPE[0]])
        self.assertEqual(record['mode'], 'bounded_rectangular_crossfade')
        self.assertEqual(record['bypass_reason'], 'edit_region_below_photographic_join_support')
        self.assertFalse(photo_edit_tools._join_analysis_ready((48, 64)))
        self.assertTrue(photo_edit_tools._join_analysis_ready((self.right-self.left,
            self.bottom-self.top)))
        mask = photo_edit_tools._mask((48, 64))
        mixed = self.parent[32:96, 24:72]*(1-mask[..., None])+self.patch[32:96, 24:72]*mask[..., None]
        expected = self.parent.copy();expected[32:96, 24:72] = mixed
        self.assertTrue(np.array_equal(result, expected))
        expected_coverage = np.zeros(self.SHAPE, np.float32)
        expected_coverage[32:96, 24:72] = mask
        self.assertTrue(np.array_equal(coverage, expected_coverage))

    def test_registered_alpha_owns_the_join_and_is_never_modified(self):
        yy, xx = np.indices(self.SHAPE, np.float32)
        alpha = np.clip(1.5-(((xx-576.)/300.)**2+((yy-480.)/260.)**2), 0., 1.).astype(np.float32)
        kept = alpha.copy()
        result, coverage, record = self.joined(self.patch, alpha=alpha)
        self.assertTrue(record['provider_alpha_registered'])
        self.assertTrue(np.array_equal(alpha, kept))
        self.assertEqual(int(np.count_nonzero(coverage[alpha == 0])), 0)
        self.assertLessEqual(float(coverage.max()), 1.)
        self.assertTrue(np.array_equal(result[coverage == 0], self.parent[coverage == 0]))

    def _tinted_patch(self):
        """A provider patch that really differs from the parent after registration."""
        return self._replace(np.roll(self._region(self.parent), 5, axis=1))

    def _graded_patch(self):
        """A patch whose own lighting gradient no single gain can remove."""
        ramp = 1.+np.float32(.45)*np.linspace(0., 1., self.right-self.left,
            dtype=np.float32)[None, :, None]
        return self._replace(np.clip(self._region(self.parent)*ramp, 0., 1.).astype(np.float32))

    def test_alpha_decays_only_the_destination_so_the_join_is_linear_in_coverage(self):
        """Premultiplied source-over: a soft matte scales the join, not its analysis."""
        patch = self._tinted_patch()
        covered = self.joined(patch, alpha=np.ones(self.SHAPE, np.float32))[0]
        # Production hands the join associated colour, so a half matte is the
        # same patch premultiplied by half, not the full patch beside a weight.
        weight = np.full(self.SHAPE, .5, np.float32)
        half = self.joined((patch*weight[..., None]).astype(np.float32), alpha=weight)[0]
        left, top, right, bottom = self.CORE
        reference = self.parent[left:right, top:bottom]
        full_delta = covered[left:right, top:bottom]-reference
        half_delta = half[left:right, top:bottom]-reference
        self.assertGreater(float(np.abs(full_delta).max()), 1e-3)
        self.assertTrue(np.allclose(half_delta, full_delta*np.float32(.5), atol=1e-6))

    def test_soft_matte_leaves_the_region_interior_free_of_the_membrane(self):
        patch = self._tinted_patch()
        yy, xx = np.indices(self.SHAPE, np.float32)
        alpha = np.clip(1.5-(((xx-576.)/300.)**2+((yy-480.)/260.)**2), 0., 1.).astype(np.float32)
        result, coverage, record = self.joined(patch, alpha=alpha)
        # The membrane's dose comes from the core geometry, so a soft matte can
        # never drag it into the interior; deep inside the coverage is the matte.
        self.assertEqual(record['tone_correction_interior_max'], 0)
        left, top, right, bottom = self.CORE
        far = photo_edit_tools._edge_distance((right-left, bottom-top))
        taper = photo_edit_tools._structure_taper(self._region(self.parent))
        ramp = photo_edit_tools._join_ramp(far, taper)
        # Coverage is the taper ramp times the registered matte, nothing else.
        expected = np.zeros(self.SHAPE, np.float32)
        expected[top:bottom, left:right] = ramp*alpha[top:bottom, left:right]
        self.assertTrue(np.allclose(coverage, expected, atol=1e-6))
        deep = far > taper+1.
        self.assertTrue(deep.any())
        self.assertTrue(np.array_equal(coverage[top:bottom, left:right][deep],
            alpha[top:bottom, left:right][deep]))
        self.assertTrue(np.array_equal(result[alpha == 0], self.parent[alpha == 0]))


    def test_spatial_matte_cannot_steer_the_registration_or_the_level(self):
        """A varying matte may not reach the analysis: same gain, same level."""
        patch = self._tinted_patch()
        yy, xx = np.indices(self.SHAPE, np.float32)
        alpha = np.clip(1.35-(((xx-576.)/460.)**2+((yy-480.)/420.)**2), 0., 1.).astype(np.float32)
        self.assertGreater(int(np.count_nonzero((alpha > .05) & (alpha < .95))), 1000)
        flat = self.joined(patch, alpha=np.ones(self.SHAPE, np.float32))
        weight = np.full(self.SHAPE, .6, np.float32)
        even = self.joined((patch*weight[..., None]).astype(np.float32), alpha=weight)
        soft = self.joined((patch*alpha[..., None]).astype(np.float32), alpha=alpha)
        # Registration and texture matching run on unpremultiplied content, so
        # the same patch registers identically whatever matte carries it: an
        # even 0.6 matte gives exactly the full-coverage gain, and a spatially
        # varying one differs only where it covers fewer pixels (alpha == 0 is
        # excluded from the moment, as it writes nothing).
        self.assertEqual(flat[2]['registration']['scale'], even[2]['registration']['scale'])
        self.assertEqual(flat[2]['detail'], even[2]['detail'])
        self.assertTrue(np.allclose(flat[2]['registration']['scale'],
            soft[2]['registration']['scale'], atol=2e-3))
        self.assertEqual(flat[2]['analysis_domain'], 'unpremultiplied_provider_colour')
        self.assertTrue(soft[2]['registration']['unpremultiplied_estimate'])
        # Where the matte is closed and the taper complete the delivered level
        # is the photographed level: the provider supplies appearance, not a
        # new exposure, and the soft matte may not drag it towards black.
        left, top, right, bottom = self.CORE
        far = photo_edit_tools._edge_distance((right-left, bottom-top))
        closed = ((far > photo_edit_tools.FUSION['maximum_taper_px'])
            & (alpha[top:bottom, left:right] >= np.float32(.999)))
        self.assertGreater(int(np.count_nonzero(closed)), 10000)
        delivered = self._luma(soft[0][top:bottom, left:right])[closed]
        prior = self._luma(self.parent[top:bottom, left:right])[closed]
        level = float(delivered.mean()/max(prior.mean(), 1e-9)-1.)
        self.assertLessEqual(abs(level), .01)
        # Level, not content: the low-frequency difference must stay within 1%.
        low = gaussian_filter(self._luma(soft[0][top:bottom, left:right])
            -self._luma(self.parent[top:bottom, left:right]), 8., mode='nearest')
        self.assertLessEqual(float(np.median(np.abs(low[closed]))), .01)

    def test_the_subject_matte_owns_the_boundary_instead_of_the_rectangle(self):
        """A person-only round: the write itself ends where the person ends."""
        matte = self.matte()
        kept = matte.copy()
        tinted = np.clip(self._region(self.parent)*np.asarray([1.6, 1.5, 1.4], np.float32),
            0., 1.).astype(np.float32)
        result, coverage, record = self.joined(self._replace(tinted), subject=matte, subject_only=True)
        self.assertTrue(np.array_equal(matte, kept))
        subject = record['subject']
        self.assertEqual(subject['boundary_source'], 'subject_matte_edge')
        self.assertEqual(subject['write_boundary'], 'subject_matte_edge')
        self.assertEqual(subject['ownership'], 'subject_generated_scene_retained')
        self.assertTrue(subject['subject_only_round'])
        self.assertEqual(record['mode'], 'registered_photographic_join')
        self.assertEqual(record['analysis_domain'], 'unpremultiplied_provider_colour')
        self.assertTrue(record['alpha_in_destination_only'])
        self.assertEqual(record['tone_correction_interior_max'], 0)
        self.assertTrue(subject['membrane_gated_on_silhouette'])
        scene = matte <= np.float32(photo_edit_tools.SUBJECT['soft_low'])
        self.assertGreater(int(np.count_nonzero(scene & self.core_mask())), 10000)
        # The retained scene keeps the parent's own bytes: the rectangle's
        # generated shell inside the core is exactly what made the halo.
        self.assertEqual(int(np.count_nonzero(coverage[scene])), 0)
        self.assertTrue(np.array_equal(result[scene], self.parent[scene]))
        # The subject is written with the provider's own content.
        deep = (matte >= np.float32(.999)) & self.core_mask()
        self.assertGreater(int(np.count_nonzero(deep)), 10000)
        self.assertGreaterEqual(float(coverage[deep].min()), .999)
        self.assertGreater(float(np.abs(result[deep]-self.parent[deep]).max()), 1e-3)
        # Coverage is still bounded edit permission: nothing outside the core.
        self.assertEqual(int(np.count_nonzero(coverage[~self.core_mask()])), 0)
        # The transition sits on the silhouette, not on the box.
        self.assertGreater(subject['join_band_on_silhouette_fraction'], .5)
        self.assertGreater(subject['written_fraction_of_core'], .05)
        self.assertLess(subject['written_fraction_of_core'], .95)
        self.assertAlmostEqual(subject['written_fraction_of_core']
            +subject['retained_scene_fraction_of_core'], 1., places=4)
        self.assertGreaterEqual(subject['edge_width_px'],
            float(photo_edit_tools.FUSION['minimum_taper_px']))
        self.assertLessEqual(subject['edge_width_px'],
            float(photo_edit_tools.SUBJECT['maximum_edge_width_px']))
        self.assertLess(subject['taper_px']['minimum'], subject['taper_px']['maximum'])

    def test_the_edit_coverage_cannot_stand_in_for_a_subject_matte(self):
        """Permission is not segmentation: the reserved box and the coverage are refused."""
        for name, value in (('whole canvas', np.ones(self.SHAPE, np.float32)),
                ('reserved box', box_mask(self.CORE, self.SHAPE)),
                ('labels', np.zeros(self.SHAPE, np.int32)),
                ('wrong canvas', np.ones((64, 64), np.float32)),
                ('out of range', np.full(self.SHAPE, 1.5, np.float32)),
                ('empty', np.zeros(self.SHAPE, np.float32))):
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    self.joined(self.patch, subject=value)

    def test_an_authorized_core_without_the_subject_keeps_the_reserved_write(self):
        """A core in the scene has no subject boundary to own, and says so."""
        far = self.matte(centre=(80., 80.), radii=(60., 60.))
        self.assertEqual(int(np.count_nonzero((far >= np.float32(.5)) & self.core_mask())), 0)
        plain = self.joined(self.patch)
        clipped = self.joined(self.patch, subject=far)
        self.assertEqual(clipped[2]['subject']['boundary_source'], 'authorized_core_without_subject')
        self.assertEqual(clipped[2]['subject']['ownership'], 'authorized_core_written_as_reserved')
        self.assertTrue(np.array_equal(clipped[0], plain[0]))
        self.assertTrue(np.array_equal(clipped[1], plain[1]))

    def test_the_rectangle_shell_is_gone_once_the_subject_owns_the_boundary(self):
        """The shell the audit measures is the generated band inside the box edge."""
        from float_appearance import linear_to_srgb
        local = np.clip(self._region(self.parent)*np.asarray([.55, .60, .50], np.float32),
            0., 1.).astype(np.float32)
        patch = self._replace(local)
        matte = self.matte()
        far = photo_edit_tools._edge_distance((self.right-self.left, self.bottom-self.top))
        shell = (far >= 1.) & (far <= 16.)
        prior_smooth = gaussian_filter(self._luma(self.parent), 3., mode='nearest')

        def shell_step(values):
            delivered = gaussian_filter(self._luma(values), 3., mode='nearest')
            return float(np.median(np.abs(delivered[self.top:self.bottom, self.left:self.right][shell]
                - prior_smooth[self.top:self.bottom, self.left:self.right][shell])))

        def halo(values, coverage):
            encode = lambda data: np.uint8(np.clip(np.asarray(linear_to_srgb(
                np.asarray(data, np.float32)), np.float32), 0., 1.)*255.+.5)
            report = quality.validate(quality.measure(encode(self.parent), encode(values),
                work_type='partial_composite', coverage=coverage, region=list(self.CORE),
                label='subject-boundary'))
            return report['items']['boundary_halo']

        before = self.joined(patch)
        after = self.joined(patch, subject=matte, subject_only=True)
        # Before, the band inside the cut is the provider's own darkened scene:
        # one gain cannot tell the scene from the subject across that cut, so
        # this is the low-frequency shell the audit measures on real cases.
        band = slice(self.top, self.bottom), slice(self.left, self.right)
        self.assertGreater(float(before[1][band][shell].min()), 0.)
        self.assertGreater(float(np.abs(before[0][band][shell]
            -self.parent[band][shell]).max()), 1e-3)
        # After, the scene inside the reserved box that the subject does not own
        # keeps the parent's own bytes, so there is no shell to measure at all.
        self.assertEqual(float(after[1][band][shell].max()), 0.)
        self.assertEqual(shell_step(after[0]), 0.)
        former, current = halo(before[0], before[1]), halo(after[0], after[1])
        self.assertTrue(former['applicable'])
        self.assertLessEqual(current['halo_width'], former['halo_width'])
        self.assertLess(current['halo_width'], PARAMETERS['halo_width_max']+1)
        self.assertLess(current['visible_excess'], PARAMETERS['halo_visible'])

    def test_an_authorized_scene_round_writes_the_scene_side_of_the_core(self):
        """A scene operation authorizes the scene side: it is written, not handed back."""
        matte = self.matte()
        tinted = np.clip(self._region(self.parent)*np.asarray([1.6, 1.5, 1.4], np.float32),
            0., 1.).astype(np.float32)
        patch = self._replace(tinted)
        result, coverage, record = self.joined(patch, subject=matte)
        subject = record['subject']
        # The default is the authorized core written as reserved; the matte
        # shapes the transition and the light solve, it does not discard content.
        self.assertEqual(subject['boundary_source'], 'authorized_rectangle')
        self.assertEqual(subject['write_boundary'], 'authorized_rectangle')
        self.assertEqual(subject['transition_boundary'], 'subject_matte_edge')
        self.assertEqual(subject['ownership'], photo_edit_tools.SUBJECT['default_write'])
        self.assertFalse(subject['subject_only_round'])
        scene = matte <= np.float32(photo_edit_tools.SUBJECT['soft_low'])
        self.assertGreater(int(np.count_nonzero(scene & self.core_mask())), 10000)
        self.assertGreaterEqual(subject['scene_side_written_fraction'], .99)
        self.assertGreater(subject['written_fraction_of_core'], .99)
        # The scene side inside the core is delivered content, and it is the
        # provider's, not the parent's bytes.
        far = photo_edit_tools._edge_distance((self.right-self.left, self.bottom-self.top))
        interior = np.zeros(self.SHAPE, bool)
        interior[self.top:self.bottom, self.left:self.right] = (far
            > np.float32(photo_edit_tools.FUSION['maximum_taper_px']+2.))
        deep = scene & self.core_mask() & interior
        self.assertGreater(int(np.count_nonzero(deep)), 10000)
        self.assertGreaterEqual(float(coverage[deep].min()), .999)
        self.assertGreater(float(np.abs(result[deep]-self.parent[deep]).max()), 1e-3)
        # The subject is written too, and the transition still follows the matte.
        owned = (matte >= np.float32(.999)) & self.core_mask() & interior
        self.assertGreater(int(np.count_nonzero(owned)), 10000)
        self.assertGreaterEqual(float(coverage[owned].min()), .999)
        self.assertLess(subject['taper_px']['minimum'], subject['taper_px']['maximum'])
        # Coverage is still bounded edit permission: nothing outside the core.
        self.assertEqual(int(np.count_nonzero(coverage[~self.core_mask()])), 0)
        self.assertTrue(np.array_equal(result[coverage == 0], self.parent[coverage == 0]))

    def test_only_the_person_only_round_narrows_the_write_to_the_subject(self):
        """Same matte, same patch: the round's authorization decides the scene side."""
        matte = self.matte()
        tinted = np.clip(self._region(self.parent)*np.asarray([1.6, 1.5, 1.4], np.float32),
            0., 1.).astype(np.float32)
        patch = self._replace(tinted)
        scene = matte <= np.float32(photo_edit_tools.SUBJECT['soft_low'])
        authorized = self.joined(patch, subject=matte)
        narrowed = self.joined(patch, subject=matte, subject_only=True)
        self.assertGreater(int(np.count_nonzero(authorized[1][scene])), 10000)
        self.assertEqual(int(np.count_nonzero(narrowed[1][scene])), 0)
        self.assertGreater(float(np.abs(authorized[0][scene]-self.parent[scene]).max()), 1e-3)
        self.assertTrue(np.array_equal(narrowed[0][scene], self.parent[scene]))
        self.assertGreater(authorized[2]['subject']['written_fraction_of_core'],
            narrowed[2]['subject']['written_fraction_of_core'])
        self.assertEqual(narrowed[2]['subject']['scene_pixels_written'], 0)
        self.assertGreaterEqual(authorized[2]['subject']['scene_pixels_written'],
            int(.99*authorized[2]['subject']['scene_pixels']))
        # The matte is not the permission: both arms stay inside the authorized core.
        for _, coverage, _ in (authorized, narrowed):
            self.assertEqual(int(np.count_nonzero(coverage[~self.core_mask()])), 0)

    def test_the_receipt_takes_its_boundary_from_the_join_that_wrote_it(self):
        """A stale rectangle declaration cannot stand in for the written boundary."""
        matte = self.matte()
        tinted = np.clip(self._region(self.parent)*np.asarray([1.6, 1.5, 1.4], np.float32),
            0., 1.).astype(np.float32)
        patch = self._replace(tinted)

        def declaration():
            return {'schema': photo_edit_tools.SUBJECT['schema'], 'available': True,
                'boundary_source': 'authorized_rectangle',
                'ownership': 'authorized_core_written_as_reserved',
                'authoritative': 'pre_join_declaration',
                'acquisition': {'boundary_source': 'authorized_rectangle'}}

        for subject_only, boundary, ownership in (
                (False, 'authorized_rectangle', photo_edit_tools.SUBJECT['default_write']),
                (True, 'subject_matte_edge', photo_edit_tools.SUBJECT['narrowed_write'])):
            with self.subTest(subject_only=subject_only):
                record = self.joined(patch, subject=matte, subject_only=subject_only)[2]
                receipt = photo_edit_tools._joined_subject_record(declaration(), record)
                self.assertEqual(receipt['boundary_source'], boundary)
                self.assertEqual(receipt['write_boundary'], boundary)
                self.assertEqual(receipt['transition_boundary'], 'subject_matte_edge')
                self.assertEqual(receipt['ownership'], ownership)
                self.assertEqual(receipt['subject_only_round'], subject_only)
                self.assertEqual(receipt['authoritative'], 'join.subject')
                self.assertEqual(receipt['boundary_source'],
                    record['subject']['boundary_source'])
                # What the round declared before the join still survives.
                self.assertEqual(receipt['acquisition']['boundary_source'], 'authorized_rectangle')
        # A join with no boundary claim of its own leaves the declaration alone.
        plain = self.joined(patch)[2]
        receipt = photo_edit_tools._joined_subject_record({'boundary_source':
            'authorized_core_below_photographic_join_support',
            'authoritative': 'pre_join_declaration'}, plain)
        self.assertEqual(receipt['boundary_source'],
            'authorized_core_below_photographic_join_support')
        self.assertEqual(receipt['authoritative'], 'pre_join_declaration')

    def test_the_matte_never_reaches_the_provider_alpha_contract(self):
        """The matte decides ownership only; registered alpha still owns the join."""
        matte = self.matte()
        yy, xx = np.indices(self.SHAPE, np.float32)
        alpha = np.clip(1.5-(((xx-576.)/300.)**2+((yy-480.)/260.)**2), 0., 1.).astype(np.float32)
        kept = alpha.copy()
        result, coverage, record = self.joined(self.patch, alpha=alpha, subject=matte)
        self.assertTrue(np.array_equal(alpha, kept))
        self.assertTrue(record['provider_alpha_registered'])
        self.assertTrue(record['alpha_in_destination_only'])
        self.assertTrue(np.array_equal(result[coverage == 0], self.parent[coverage == 0]))
        self.assertEqual(int(np.count_nonzero(coverage[alpha == 0])), 0)
        self.assertLessEqual(float(coverage.max()), 1.)



if __name__ == '__main__':
    unittest.main()
