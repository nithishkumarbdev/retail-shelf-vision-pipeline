import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from services.detector.detector import Detector  # noqa: E402
from services.grouping.grouping import ProductGrouper  # noqa: E402

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "sample_images")


class TestProductGrouper(unittest.TestCase):
    def setUp(self):
        self.grouper = ProductGrouper()

    def test_empty_detections(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        grouped, summary = self.grouper.group(img, [])
        self.assertEqual(grouped, [])
        self.assertEqual(summary, [])

    def test_single_detection_gets_group_zero(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = [{"detection_id": 0, "bbox": [10, 10, 50, 50], "confidence": 0.9}]
        grouped, summary = self.grouper.group(img, detections)
        self.assertEqual(grouped[0]["group_id"], 0)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["size"], 1)

    def test_visually_similar_boxes_cluster_together(self):
        # Build an image with two clearly different colored blocks, each repeated 3x
        img = np.zeros((100, 300, 3), dtype=np.uint8)
        img[:, 0:50] = (0, 200, 0)      # green block A
        img[:, 100:150] = (0, 205, 5)   # green block B (near-identical to A)
        img[:, 200:250] = (200, 0, 0)   # blue block C (visually distinct)

        detections = [
            {"detection_id": 0, "bbox": [0, 0, 50, 100], "confidence": 0.9},
            {"detection_id": 1, "bbox": [100, 0, 150, 100], "confidence": 0.9},
            {"detection_id": 2, "bbox": [200, 0, 250, 100], "confidence": 0.9},
        ]
        grouped, summary = self.grouper.group(img, detections)
        gid_map = {d["detection_id"]: d["group_id"] for d in grouped}
        self.assertEqual(gid_map[0], gid_map[1], "near-identical green blocks should share a group")
        self.assertNotEqual(gid_map[0], gid_map[2], "visually distinct blue block should be a different group")

    def test_group_ids_cover_all_detections(self):
        path = os.path.join(SAMPLE_DIR, "sample_shelf_1.jpg")
        img = cv2.imread(path)
        detections, _ = Detector().detect(img)
        grouped, summary = self.grouper.group(img, detections)

        self.assertEqual(len(grouped), len(detections))
        all_member_ids = sorted(sum([g["member_detection_ids"] for g in summary], []))
        self.assertEqual(all_member_ids, sorted(d["detection_id"] for d in detections))


if __name__ == "__main__":
    unittest.main()
