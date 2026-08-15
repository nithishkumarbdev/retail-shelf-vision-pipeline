"""
Detector microservice.

POST /detect
    body (JSON): {"image_base64": "<base64 string, no data: prefix required>"}
    response 200: {
        "detections": [{"detection_id": 0, "bbox": [x1,y1,x2,y2], "confidence": 0.81}, ...],
        "meta": {"method": "shelf-grid", "num_detections": 12,
                 "processing_time_ms": 42.1, "image_size": {"width":.., "height":..}}
    }
    response 400: {"error": "..."}   (bad/missing/corrupt input)
    response 500: {"error": "..."}   (unexpected detector failure)

GET /health -> {"status": "ok", "service": "detector"}
"""
import logging
import os
import sys

from flask import Flask, request, jsonify

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from common.image_utils import base64_to_image, ImageDecodeError  # noqa: E402
from services.detector.detector import Detector  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [detector] %(levelname)s %(message)s")
logger = logging.getLogger("detector-service")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024

detector = Detector()


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "Payload too large. Max 15MB."}), 413


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "detector"}), 200


@app.route("/detect", methods=["POST"])
def detect():
    payload = request.get_json(silent=True)
    if not payload or "image_base64" not in payload:
        return jsonify({"error": "Request body must be JSON with an 'image_base64' field"}), 400

    try:
        image = base64_to_image(payload["image_base64"])
    except ImageDecodeError as exc:
        logger.warning("Image decode failed: %s", exc)
        return jsonify({"error": f"Invalid image payload: {exc}"}), 400

    try:
        detections, meta = detector.detect(image)
    except ValueError as exc:
        logger.warning("Detector rejected input: %s", exc)
        return jsonify({"error": str(exc)}), 400
    except Exception:
        logger.exception("Unexpected detector failure")
        return jsonify({"error": "Internal detector failure"}), 500

    return jsonify({"detections": detections, "meta": meta}), 200


if __name__ == "__main__":
    port = int(os.environ.get("DETECTOR_PORT", "5001"))
    app.run(host="0.0.0.0", port=port, threaded=True)
