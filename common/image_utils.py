"""
common/image_utils.py

Shared helpers for encoding/decoding images as base64 so that images are
passed as DATA over HTTP requests between microservices, never as file paths
"""
import base64
import io
import numpy as np
from PIL import Image


class ImageDecodeError(Exception):
    """Raised when an incoming image payload cannot be decoded."""


def image_to_base64(image_bgr: np.ndarray, fmt: str = "JPEG", quality: int = 90) -> str:
    """Encode an OpenCV BGR numpy image to a base64 string (no data: prefix)."""
    import cv2
    rgb = image_bgr[:, :, ::-1]
    pil_img = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def base64_to_image(b64_string: str) -> np.ndarray:
    """Decode a base64 string (optionally with a data: URL prefix) to an OpenCV BGR numpy image."""
    if not b64_string or not isinstance(b64_string, str):
        raise ImageDecodeError("Empty or invalid image payload")

    if "," in b64_string[:60] and b64_string.strip().startswith("data:"):
        b64_string = b64_string.split(",", 1)[1]

    try:
        raw = base64.b64decode(b64_string, validate=False)
    except Exception as exc:
        raise ImageDecodeError(f"Could not base64-decode image payload: {exc}") from exc

    if len(raw) == 0:
        raise ImageDecodeError("Decoded image payload is empty")

    try:
        pil_img = Image.open(io.BytesIO(raw))
        pil_img.load()
    except Exception as exc:
        raise ImageDecodeError(f"Payload is not a valid/supported image: {exc}") from exc

    pil_img = pil_img.convert("RGB")
    arr = np.array(pil_img)
    bgr = arr[:, :, ::-1].copy()
    return bgr


def file_storage_to_base64(file_storage) -> str:
    """Convert a Flask/Werkzeug FileStorage upload directly to base64 (in-memory, no disk write required)."""
    raw = file_storage.read()
    if not raw:
        raise ImageDecodeError("Uploaded file is empty")
    return base64.b64encode(raw).decode("utf-8")


MAX_IMAGE_BYTES = 15 * 1024 * 1024  # 15 MB safety cap for "very large input" handling
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}


def allowed_filename(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
