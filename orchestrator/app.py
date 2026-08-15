"""
Orchestrator: the "Flask Webserver" block from the diagram.

Flow (matches the PDF's numbered diagram):
    1. browser uploads image -> orchestrator
    2. orchestrator -> detector microservice (image travels as base64 JSON, no file paths)
    3. orchestrator -> grouping microservice (image + detections)
    4. orchestrator draws the color-coded visualization, saves it to disk, and
       returns the JSON response (+ visualization URL) to the browser

Endpoints
---------
GET  /                      minimal web upload UI
POST /api/analyze           multipart/form-data 'image' file  OR  JSON {"image_base64": "..."}
                             -> full pipeline JSON response (see docs/api_spec.md)
GET  /api/health             aggregate health of orchestrator + both microservices
GET  /static/outputs/<file>  saved visualization images (served by Flask's static handler)
"""
import base64
import logging
import os
import sys
import time
import uuid

import cv2
import numpy as np
import requests
from flask import Flask, request, jsonify, render_template, url_for

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.image_utils import (  # noqa: E402
    base64_to_image,
    image_to_base64,
    file_storage_to_base64,
    ImageDecodeError,
    allowed_filename,
)
from orchestrator import config  # noqa: E402
from services.detector.detector import Detector  # noqa: E402
from services.grouping.grouping import ProductGrouper  # noqa: E402

os.makedirs(config.OUTPUT_DIR, exist_ok=True)
os.makedirs(config.UPLOAD_DIR, exist_ok=True)
os.makedirs(config.LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [orchestrator] %(levelname)s %(message)s",
    handlers=[logging.FileHandler(config.LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger("orchestrator")

app = Flask(__name__, static_folder=os.path.join(config.BASE_DIR, "static"), template_folder="../templates")
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

# In-process fallback instances (used only if the HTTP microservice calls fail
# and ENABLE_INPROCESS_FALLBACK=true; see config.py)
_fallback_detector = Detector()
_fallback_grouper = ProductGrouper()

# A palette of visually distinct BGR colors, cycled per group_id for box color-coding
_PALETTE = [
    (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180),
    (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
    (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200),
    (128, 0, 0), (170, 255, 195), (128, 128, 0), (0, 0, 128),
]


def _color_for_group(group_id: int):
    return _PALETTE[group_id % len(_PALETTE)]


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "Uploaded file exceeds the 15MB size limit."}), 413


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    status = {"orchestrator": "ok"}
    for name, base_url in (("detector", config.DETECTOR_SERVICE_URL), ("grouping", config.GROUPING_SERVICE_URL)):
        try:
            r = requests.get(f"{base_url}/health", timeout=3)
            status[name] = "ok" if r.status_code == 200 else f"unhealthy ({r.status_code})"
        except requests.RequestException as exc:
            status[name] = f"unreachable ({exc.__class__.__name__})"
    overall_ok = all(v == "ok" for v in status.values())
    return jsonify({"status": "ok" if overall_ok else "degraded", "services": status}), 200


@app.route("/api/analyze", methods=["POST"])
def analyze():
    request_id = str(uuid.uuid4())
    t_start = time.time()
    logger.info("[%s] New analyze request", request_id)

    # ---- 1. Get the image as base64 (no hardcoded file paths anywhere) ----
    try:
        image_b64 = _extract_image_b64_from_request()
    except BadRequestError as exc:
        logger.warning("[%s] Bad request: %s", request_id, exc)
        return jsonify({"error": str(exc)}), 400

    try:
        image = base64_to_image(image_b64)
    except ImageDecodeError as exc:
        logger.warning("[%s] Image decode failed: %s", request_id, exc)
        return jsonify({"error": f"Invalid or corrupted image: {exc}"}), 400

    if image.shape[0] < 10 or image.shape[1] < 10:
        return jsonify({"error": "Image is too small to process"}), 400

    # ---- 2. Detection stage ----
    try:
        detections, detector_meta = _call_detector(image, image_b64, request_id)
    except PipelineStageError as exc:
        logger.error("[%s] Detector stage failed: %s", request_id, exc)
        return jsonify({"error": f"Detection stage failed: {exc}"}), 502

    if len(detections) == 0:
        elapsed = round((time.time() - t_start) * 1000, 2)
        logger.info("[%s] No products detected", request_id)
        return jsonify({
            "request_id": request_id,
            "detections": [],
            "groups_summary": [],
            "num_products_detected": 0,
            "num_groups": 0,
            "visualization_url": None,
            "detector_meta": detector_meta,
            "total_processing_time_ms": elapsed,
            "warning": "No products were detected in this image.",
        }), 200

    # ---- 3. Grouping stage ----
    try:
        grouped_detections, groups_summary = _call_grouper(image, image_b64, detections, request_id)
    except PipelineStageError as exc:
        logger.error("[%s] Grouping stage failed: %s", request_id, exc)
        return jsonify({"error": f"Grouping stage failed: {exc}"}), 502

    # ---- 4. Visualization ----
    vis_filename = f"{request_id}.jpg"
    vis_path = os.path.join(config.OUTPUT_DIR, vis_filename)
    try:
        _draw_and_save_visualization(image, grouped_detections, vis_path)
    except Exception:
        logger.exception("[%s] Visualization failed", request_id)
        return jsonify({"error": "Failed to render visualization"}), 500

    elapsed = round((time.time() - t_start) * 1000, 2)
    response = {
        "request_id": request_id,
        "detections": grouped_detections,
        "groups_summary": groups_summary,
        "num_products_detected": len(grouped_detections),
        "num_groups": len(groups_summary),
        "visualization_url": url_for("static", filename=f"outputs/{vis_filename}", _external=False),
        "detector_meta": detector_meta,
        "total_processing_time_ms": elapsed,
    }
    logger.info(
        "[%s] Done: %d detections, %d groups, %.1f ms",
        request_id, len(grouped_detections), len(groups_summary), elapsed,
    )
    return jsonify(response), 200


# -------------------------------------------------------------------------- #
# Helpers
# -------------------------------------------------------------------------- #
class BadRequestError(Exception):
    pass


class PipelineStageError(Exception):
    pass


def _extract_image_b64_from_request() -> str:
    if "image" in request.files:
        f = request.files["image"]
        if f.filename == "":
            raise BadRequestError("No file selected")
        if not allowed_filename(f.filename):
            raise BadRequestError(f"Unsupported file format: {f.filename}")
        return file_storage_to_base64(f)

    payload = request.get_json(silent=True)
    if payload and "image_base64" in payload:
        return payload["image_base64"]

    raise BadRequestError(
        "No image provided. Send multipart/form-data with an 'image' file field, "
        "or JSON with an 'image_base64' field."
    )


def _call_detector(image: np.ndarray, image_b64: str, request_id: str):
    try:
        resp = requests.post(
            f"{config.DETECTOR_SERVICE_URL}/detect",
            json={"image_base64": image_b64},
            timeout=config.SERVICE_TIMEOUT_SECONDS,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data["detections"], data["meta"]
        raise PipelineStageError(f"detector service returned HTTP {resp.status_code}: {resp.text[:200]}")
    except requests.RequestException as exc:
        logger.warning("[%s] Detector HTTP call failed (%s)", request_id, exc.__class__.__name__)
        if not config.ENABLE_INPROCESS_FALLBACK:
            raise PipelineStageError(f"detector service unreachable: {exc}") from exc
        logger.info("[%s] Falling back to in-process detector", request_id)
        try:
            detections, meta = _fallback_detector.detect(image)
            meta["fallback_mode"] = True
            return detections, meta
        except Exception as exc2:
            raise PipelineStageError(f"in-process detector fallback also failed: {exc2}") from exc2


def _call_grouper(image: np.ndarray, image_b64: str, detections: list, request_id: str):
    try:
        resp = requests.post(
            f"{config.GROUPING_SERVICE_URL}/group",
            json={"image_base64": image_b64, "detections": detections},
            timeout=config.SERVICE_TIMEOUT_SECONDS,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data["detections"], data["groups_summary"]
        raise PipelineStageError(f"grouping service returned HTTP {resp.status_code}: {resp.text[:200]}")
    except requests.RequestException as exc:
        logger.warning("[%s] Grouping HTTP call failed (%s)", request_id, exc.__class__.__name__)
        if not config.ENABLE_INPROCESS_FALLBACK:
            raise PipelineStageError(f"grouping service unreachable: {exc}") from exc
        logger.info("[%s] Falling back to in-process grouper", request_id)
        try:
            return _fallback_grouper.group(image, detections)
        except Exception as exc2:
            raise PipelineStageError(f"in-process grouping fallback also failed: {exc2}") from exc2


def _draw_and_save_visualization(image: np.ndarray, grouped_detections: list, out_path: str):
    vis = image.copy()
    thickness = max(2, round(min(vis.shape[:2]) / 400))
    font_scale = max(0.4, min(vis.shape[:2]) / 1200)

    for d in grouped_detections:
        x1, y1, x2, y2 = d["bbox"]
        gid = d.get("group_id", 0)
        color = _color_for_group(gid)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        label = f"G{gid}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        label_y = max(y1, th + 4)
        cv2.rectangle(vis, (x1, label_y - th - 4), (x1 + tw + 4, label_y), color, -1)
        cv2.putText(vis, label, (x1 + 2, label_y - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    ok = cv2.imwrite(out_path, vis)
    if not ok:
        raise IOError(f"cv2.imwrite failed for {out_path}")


if __name__ == "__main__":
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG, threaded=True)
