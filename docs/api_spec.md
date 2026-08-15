# API Specification

All endpoints accept/return `application/json` unless noted (the analyze
endpoint also accepts `multipart/form-data`). All images are transmitted as
**base64-encoded bytes inside JSON**, or as an in-memory multipart file
upload — never as a filesystem path.

---

## Orchestrator (`http://localhost:5000` by default)

### `POST /api/analyze`

The main pipeline entry point.

**Request — option A (multipart, used by the web UI):**
```
Content-Type: multipart/form-data
field "image": <binary file>, filename must end in .jpg/.jpeg/.png/.bmp/.webp
```

**Request — option B (JSON):**
```json
{
  "image_base64": "<base64-encoded image bytes, data: prefix optional>"
}
```

**Response — 200 OK:**
```json
{
  "request_id": "2c8e9f05-d7bf-4c6c-818e-0707ace19a90",
  "detections": [
    {
      "detection_id": 0,
      "bbox": [32, 30, 101, 164],
      "confidence": 0.59,
      "group_id": 2
    }
  ],
  "groups_summary": [
    {"group_id": 0, "member_detection_ids": [0, 8, 9], "size": 3},
    {"group_id": 2, "member_detection_ids": [1, 2, 3], "size": 15}
  ],
  "num_products_detected": 33,
  "num_groups": 6,
  "visualization_url": "/static/outputs/2c8e9f05-d7bf-4c6c-818e-0707ace19a90.jpg",
  "detector_meta": {
    "method": "shelf-grid",
    "num_detections": 33,
    "processing_time_ms": 61.4,
    "image_size": {"width": 668, "height": 740}
  },
  "total_processing_time_ms": 78.2
}
```

**Response — 200 OK, no products found:**
```json
{
  "request_id": "...",
  "detections": [],
  "groups_summary": [],
  "num_products_detected": 0,
  "num_groups": 0,
  "visualization_url": null,
  "detector_meta": {"...": "..."},
  "total_processing_time_ms": 12.3,
  "warning": "No products were detected in this image."
}
```

**Error responses:**
| Status | Cause | Body |
|---|---|---|
| 400 | no image field / empty filename / unsupported extension | `{"error": "..."}` |
| 400 | image bytes are not decodable | `{"error": "Invalid or corrupted image: ..."}` |
| 413 | request body > 15MB | `{"error": "Uploaded file exceeds the 15MB size limit."}` |
| 502 | detector or grouping stage failed and fallback is disabled/also failed | `{"error": "Detection stage failed: ..."}` or `{"error": "Grouping stage failed: ..."}` |
| 500 | visualization rendering failed | `{"error": "Failed to render visualization"}` |

### `GET /api/health`
```json
{
  "status": "ok",
  "services": {"orchestrator": "ok", "detector": "ok", "grouping": "ok"}
}
```
`status` is `"degraded"` if any dependency is unreachable/unhealthy; the
orchestrator still serves requests in this state via in-process fallback
(unless `ENABLE_INPROCESS_FALLBACK=false`).

### `GET /`
Serves the HTML upload UI (`templates/index.html`). Not a JSON endpoint.

### `GET /static/outputs/<filename>`
Serves a previously generated visualization image (standard Flask static
file handler).

---

## Detector microservice (`http://localhost:5001` by default)

### `POST /detect`
**Request:**
```json
{"image_base64": "<base64 bytes>"}
```
**Response — 200 OK:**
```json
{
  "detections": [
    {"detection_id": 0, "bbox": [32, 30, 101, 164], "confidence": 0.59}
  ],
  "meta": {
    "method": "shelf-grid",
    "num_detections": 33,
    "processing_time_ms": 61.4,
    "image_size": {"width": 668, "height": 740}
  }
}
```
**Errors:** 400 (missing/invalid field, undecodable image), 500 (unexpected
internal failure).

### `GET /health`
```json
{"status": "ok", "service": "detector"}
```

---

## Grouping microservice (`http://localhost:5002` by default)

### `POST /group`
**Request:**
```json
{
  "image_base64": "<base64 bytes>",
  "detections": [
    {"detection_id": 0, "bbox": [32, 30, 101, 164], "confidence": 0.59}
  ]
}
```
**Response — 200 OK:**
```json
{
  "detections": [
    {"detection_id": 0, "bbox": [32, 30, 101, 164], "confidence": 0.59, "group_id": 2}
  ],
  "groups_summary": [
    {"group_id": 2, "member_detection_ids": [0, 1, 4], "size": 3}
  ]
}
```
**Errors:** 400 (missing fields, malformed `detections`, undecodable image),
500 (unexpected internal failure).

### `GET /health`
```json
{"status": "ok", "service": "grouping"}
```

---

## Field reference

| Field | Type | Meaning |
|---|---|---|
| `bbox` | `[int, int, int, int]` | `[x1, y1, x2, y2]` in original-image pixel coordinates, top-left origin |
| `confidence` | `float 0.0–1.0` | Heuristic detection-quality score (see `docs/evaluation_report.md` — this is **not** a calibrated probability the way a trained classifier's softmax output would be) |
| `group_id` | `int` | Unique id per visually-similar cluster (proxy for "brand group"); stable within one response, **not** stable/comparable across separate requests |
| `detection_id` | `int` | Index of the detection within this response's `detections` list |
| `request_id` | `uuid4 string` | Correlates one browser request across orchestrator + both service logs |
