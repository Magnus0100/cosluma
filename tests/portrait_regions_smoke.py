"""Coordinate/semantic contracts for the portrait parsing worker."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

WORKER = Path(__file__).resolve().parents[1] / "scripts/portrait_regions_worker.py"
spec = importlib.util.spec_from_file_location("portrait_regions_worker", WORKER)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class PortraitRegionContracts(unittest.TestCase):
    @staticmethod
    def landmarks(rotation=0):
        points = np.zeros((478, 2), np.float64)
        angle = np.linspace(-np.pi / 2, 3 * np.pi / 2, len(worker.FACE_OVAL), endpoint=False)
        points[list(worker.FACE_OVAL)] = np.stack((160 + 100 * np.cos(angle), 160 + 140 * np.sin(angle)), axis=1)
        points[33] = points[133] = [120, 120]
        points[362] = points[263] = [200, 120]
        points[2] = [160, 180]
        matrix = np.array([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]])
        return (points - [160, 160]) @ matrix.T + [160, 160]

    def test_segface_class_order_is_not_bisenet_order(self):
        self.assertEqual(worker.CLASS_NAMES[1:4], ("neck", "skin", "cloth"))
        self.assertEqual(tuple(worker.CLASS_NAMES[i] for i in worker.GROUPS["eyes"]), ("l_eye", "r_eye"))
        self.assertEqual(tuple(worker.CLASS_NAMES[i] for i in worker.GROUPS["lips"]), ("l_lip", "u_lip"))
        self.assertEqual(worker.CLASS_NAMES[14], "hair")
        self.assertFalse(set(worker.GROUPS["skin_weight"]) & set(worker.GROUPS["features_gate"]))

    def test_boundary_crop_keeps_aspect_ratio_and_inverse_mapping(self):
        transform = worker.letterbox_transform(240, 480)
        self.assertEqual(transform["resized_size"], [256, 512])
        self.assertEqual(transform["padding_left_top"], [128, 0])
        field = np.full((512, 512), 18, np.uint8)
        field[:, 128:384] = 2
        restored = worker._restore_field(field, transform, nearest=True)
        self.assertEqual(restored.shape, (480, 240))
        self.assertTrue((restored == 2).all(), "Padding must never become native semantic labels")

    def test_two_faces_retain_separate_native_boxes(self):
        first = [[100, 200], [200, 200], [200, 350], [100, 350]]
        second = [[650, 200], [750, 200], [750, 350], [650, 350]]
        a = worker.context_box(first, 1000, 800)
        b = worker.context_box(second, 1000, 800)
        self.assertLessEqual(a[2], b[0])
        self.assertTrue(0 <= a[0] < a[2] <= 1000)
        self.assertTrue(0 <= b[1] < b[3] <= 800)
        self.assertRaises(ValueError, worker.context_box, [], 1000, 800)

    def test_tiled_detection_covers_whole_source_and_deduplicates(self):
        canvas = np.zeros((700, 468), np.uint8)
        for l, t, r, b in worker.detection_tiles(468, 700):
            canvas[t:b, l:r] = 1
        self.assertTrue(canvas.all())
        a = {"landmarks": [[50, 50], [100, 50], [100, 100], [50, 100]]}
        duplicate = {"landmarks": [[52, 52], [102, 52], [102, 102], [52, 102]]}
        distant = {"landmarks": [[250, 50], [300, 50], [300, 100], [250, 100]]}
        self.assertTrue(worker._same_face(a, duplicate))
        self.assertFalse(worker._same_face(a, distant))

    def test_no_face_fallback_or_missing_weight_substitution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertRaises(FileNotFoundError, worker.run, root / "source.png", root,
                              root / "missing.pt", root / "output", root / "registry.json", "cpu")
            self.assertFalse((root / "output").exists())

    def test_missing_mouth_keeps_upper_skin_and_excludes_props_outside_oval(self):
        labels = np.full((320, 320), 2, np.uint8)
        fields, metadata = worker.qualify_support(np.ones_like(labels, np.float32), labels,
                                                 self.landmarks(), [0, 0, 320, 320])
        self.assertEqual(metadata["status"], "visible_upper_face_only")
        self.assertGreater(fields["skin_weight"][80, 160], .99)
        self.assertEqual(fields["skin_weight"][220, 160], 0)
        self.assertEqual(fields["skin_weight"][80, 5], 0)
        self.assertGreater(metadata["qualified_support_sum"], 0)

    def test_single_visible_eye_does_not_reject_profile(self):
        labels = np.full((320, 320), 2, np.uint8)
        labels[118:125, 118:125] = 8
        labels[230:235, 155:165] = 12
        fields, metadata = worker.qualify_support(np.ones_like(labels, np.float32), labels,
                                                 self.landmarks(), [0, 0, 320, 320])
        self.assertEqual(metadata["status"], "full_visible_face")
        self.assertEqual(metadata["eye_pixels"]["r_eye"], 0)
        self.assertGreater(fields["skin_weight"][250, 160], .99)

    def test_lower_face_cut_follows_tilt_and_missing_geometry_has_no_fake_support(self):
        points = self.landmarks(np.pi / 5)
        labels = np.full((320, 320), 2, np.uint8)
        fields, metadata = worker.qualify_support(np.ones_like(labels, np.float32), labels, points, [0, 0, 320, 320])
        down = np.array(metadata["lower_face_cut"]["toward_chin_unit_vector"])
        eye_center = (points[33] + points[133] + points[362] + points[263]) * .25
        upper, lower = eye_center - down * 35, eye_center + down * 35
        self.assertGreater(fields["skin_weight"][int(upper[1]), int(upper[0])], .99)
        self.assertEqual(fields["skin_weight"][int(lower[1]), int(lower[0])], 0)
        missing, status = worker.qualify_support(np.ones_like(labels, np.float32), labels, [], [0, 0, 320, 320])
        self.assertEqual(status["status"], "unavailable")
        self.assertFalse(missing["skin_weight"].any())

    def test_small_nonzero_probability_cannot_create_material_or_feature_ownership(self):
        labels = np.full((320, 320), 3, np.uint8)
        labels[60:100, 120:200] = 2
        labels[230:235, 155:165] = 12
        # Softmax skin probability is never truly zero, including on clothing.
        soft_skin = np.full(labels.shape, .0001, np.float32)
        soft_skin[labels == 2] = .9
        fields, metadata = worker.qualify_support(soft_skin, labels, self.landmarks(), [0, 0, 320, 320])
        self.assertFalse(fields["skin_weight"][labels == 3].any())
        self.assertFalse(fields["skin_weight"][labels == 12].any())
        self.assertGreater(fields["skin_weight"][80, 160], .8)
        self.assertEqual(fields["features_gate"][80, 160], 0)
        self.assertEqual(fields["material_skin_support"][59, 160], 0)

    def test_region_npz_uses_standard_numpy_archive_contract(self):
        """The fast writer remains readable by the worker's existing np.load paths."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "regions.npz"
            labels = np.arange(12, dtype=np.uint8).reshape(3, 4)
            support = np.linspace(0, 1, 12, dtype=np.float16).reshape(3, 4)
            worker.save_npz(path, labels=labels, support=support)
            with np.load(path, allow_pickle=False) as loaded:
                np.testing.assert_array_equal(loaded["labels"], labels)
                np.testing.assert_array_equal(loaded["support"], support)


if __name__ == "__main__":
    unittest.main()
