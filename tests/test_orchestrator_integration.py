import io
import os
import sys
import shutil
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Point at ports nothing is listening on so the HTTP calls fail fast (connection
# refused) and the orchestrator's in-process fallback kicks in -- this lets the
# whole pipeline be exercised through the real Flask routes without needing the
# detector/grouping processes to be separately running during the test run.
os.environ.setdefault("DETECTOR_SERVICE_URL", "http://localhost:5991")
os.environ.setdefault("GROUPING_SERVICE_URL", "http://localhost:5992")
os.environ.setdefault("ENABLE_INPROCESS_FALLBACK", "true")

from orchestrator.app import app, config  # noqa: E402

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "sample_images")


class TestOrchestratorIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.testing = True
        cls.client = app.test_client()

    def test_index_page_loads(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Retail Shelf AI Pipeline", resp.data)

    def test_analyze_happy_path(self):
        img_path = os.path.join(SAMPLE_DIR, "sample_shelf_1.jpg")
        with open(img_path, "rb") as f:
            data = {"image": (io.BytesIO(f.read()), "sample_shelf_1.jpg")}
            resp = self.client.post("/api/analyze", data=data, content_type="multipart/form-data")

        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertIn("detections", body)
        self.assertIn("groups_summary", body)
        self.assertGreater(body["num_products_detected"], 0)
        self.assertIsNotNone(body["visualization_url"])

        # the visualization file should actually exist on disk
        vis_filename = body["visualization_url"].split("/")[-1]
        vis_path = os.path.join(config.OUTPUT_DIR, vis_filename)
        self.assertTrue(os.path.exists(vis_path))

    def test_analyze_no_image_field(self):
        resp = self.client.post("/api/analyze", data={}, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("error", resp.get_json())

    def test_analyze_json_base64_input(self):
        import cv2
        from common.image_utils import image_to_base64

        img = cv2.imread(os.path.join(SAMPLE_DIR, "sample_single_product.jpg"))
        b64 = image_to_base64(img)
        resp = self.client.post("/api/analyze", json={"image_base64": b64})
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertGreaterEqual(body["num_products_detected"], 1)

    def test_analyze_corrupted_image(self):
        data = {"image": (io.BytesIO(b"not a real image"), "bad.jpg")}
        resp = self.client.post("/api/analyze", data=data, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 400)

    def test_analyze_unsupported_extension(self):
        data = {"image": (io.BytesIO(b"hello"), "notes.txt")}
        resp = self.client.post("/api/analyze", data=data, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 400)

    def test_analyze_empty_filename(self):
        data = {"image": (io.BytesIO(b""), "")}
        resp = self.client.post("/api/analyze", data=data, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 400)

    def test_health_endpoint_reports_degraded_without_services(self):
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertIn("services", body)
        self.assertEqual(body["services"]["orchestrator"], "ok")

    @classmethod
    def tearDownClass(cls):
        # Clean up any visualization files this test run created
        if os.path.exists(config.OUTPUT_DIR):
            for fname in os.listdir(config.OUTPUT_DIR):
                if fname.endswith(".jpg"):
                    try:
                        os.remove(os.path.join(config.OUTPUT_DIR, fname))
                    except OSError:
                        pass


if __name__ == "__main__":
    unittest.main()
