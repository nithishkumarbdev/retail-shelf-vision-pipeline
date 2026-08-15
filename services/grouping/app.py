"""
Grouping microservice.

POST /group
    body (JSON): {
        "image_base64": "<base64 string>",
        "detections": [{"detection_id": 0, "bbox": [x1,y1,x2,y2], "confidence": 0.81}, ...]
    }
    response 200: {
        "detections": [{..., "group_id": 2}, ...],
        "groups_summary": [{"group_id": 0, "member_detection_ids": [0,3,7], "size": 3}, ...]
    }
    response 400: {"error": "..."}
    response 500: {"error": "..."}

GET /health -> {"status": "ok", "service": "grouping"}
"""
import logging
import os
import sys

from flask import Flask, request, jsonify

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from common.image_utils import base64_to_image, ImageDecodeError  # noqa: E402
from services.grouping.grouping import ProductGrouper  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [grouping] %(levelname)s %(message)s")
logger = logging.getLogger("grouping-service")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024

grouper = ProductGrouper()


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "Payload too large. Max 15MB."}), 413


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "grouping"}), 200


@app.route("/group", methods=["POST"])
def group():
    payload = request.get_json(silent=True)
    if not payload or "image_base64" not in payload or "detections" not in payload:
        return jsonify({"error": "Request body must be JSON with 'image_base64' and 'detections' fields"}), 400

    try:
        image = base64_to_image(payload["image_base64"])
    except ImageDecodeError as exc:
        logger.warning("Image decode failed: %s", exc)
        return jsonify({"error": f"Invalid image payload: {exc}"}), 400

    detections = payload["detections"]
    if not isinstance(detections, list):
        return jsonify({"error": "'detections' must be a list"}), 400

    for i, d in enumerate(detections):
        if "bbox" not in d or len(d["bbox"]) != 4:
            return jsonify({"error": f"detections[{i}] missing a valid 'bbox'"}), 400
        d.setdefault("detection_id", i)

    try:
        grouped_detections, groups_summary = grouper.group(image, detections)
    except Exception:
        logger.exception("Unexpected grouping failure")
        return jsonify({"error": "Internal grouping failure"}), 500

    return jsonify({"detections": grouped_detections, "groups_summary": groups_summary}), 200


if __name__ == "__main__":
    port = int(os.environ.get("GROUPING_PORT", "5002"))
    app.run(host="0.0.0.0", port=port, threaded=True)
