import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from services.detector.detector import Detector  # noqa: E402


SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "sample_images")


class TestDetector(unittest.TestCase):
    def setUp(self):
        self.detector = Detector()

    def test_raises_on_empty_image(self):
        with self.assertRaises(ValueError):
            self.detector.detect(np.zeros((0, 0, 3), dtype=np.uint8))

    def test_raises_on_none_image(self):
        with self.assertRaises(ValueError):
            self.detector.detect(None)

    def test_detects_products_on_sample_shelf(self):
        path = os.path.join(SAMPLE_DIR, "sample_shelf_1.jpg")
        img = cv2.imread(path)
        self.assertIsNotNone(img, "sample_shelf_1.jpg must exist and be readable")

        detections, meta = self.detector.detect(img)

        self.assertGreater(len(detections), 5, "should find a meaningful number of products on a busy shelf")
        self.assertEqual(meta["num_detections"], len(detections))
        self.assertIn(meta["method"], ("shelf-grid", "contour-fallback"))
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            self.assertLess(x1, x2)
            self.assertLess(y1, y2)
            self.assertGreaterEqual(x1, 0)
            self.assertGreaterEqual(y1, 0)
            self.assertLessEqual(x2, img.shape[1])
            self.assertLessEqual(y2, img.shape[0])
            self.assertTrue(0.0 <= d["confidence"] <= 1.0)

    def test_no_detections_on_blank_image(self):
        path = os.path.join(SAMPLE_DIR, "sample_empty_shelf.jpg")
        img = cv2.imread(path)
        self.assertIsNotNone(img)
        detections, meta = self.detector.detect(img)
        self.assertEqual(len(detections), 0)

    def test_detection_ids_are_sequential(self):
        path = os.path.join(SAMPLE_DIR, "sample_shelf_1.jpg")
        img = cv2.imread(path)
        detections, _ = self.detector.detect(img)
        ids = [d["detection_id"] for d in detections]
        self.assertEqual(ids, list(range(len(detections))))

    def test_handles_very_large_image_via_resize(self):
        # 4000x4000 synthetic image should not error or take unreasonably long
        big = np.random.randint(0, 255, (4000, 4000, 3), dtype=np.uint8)
        detections, meta = self.detector.detect(big)
        self.assertEqual(meta["image_size"], {"width": 4000, "height": 4000})

    def test_nms_removes_duplicate_boxes(self):
        boxes = [(0, 0, 100, 100, 0.9), (5, 5, 100, 100, 0.5), (300, 300, 400, 400, 0.8)]
        kept = Detector._non_max_suppression(boxes, iou_threshold=0.3)
        self.assertEqual(len(kept), 2)


if __name__ == "__main__":
    unittest.main()
