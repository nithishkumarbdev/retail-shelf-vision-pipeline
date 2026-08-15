# Architecture

## 1. Pipeline overview

```
                  ┌─────────────────────────┐
 Browser  ──(1)──▶│   Orchestrator (Flask)   │
  upload           │  - web UI (/)            │
  image            │  - POST /api/analyze     │
                    │  - GET  /api/health      │
                    └───────┬─────────┬────────┘
                            │(2)      │(3)
                  image_b64 │         │ image_b64 + detections
                            ▼         ▼
                 ┌────────────────┐ ┌───────────────────┐
                 │ Detector service│ │ Grouping service   │
                 │ POST /detect    │ │ POST /group         │
                 │ (Flask, :5001)  │ │ (Flask, :5002)      │
                 └────────────────┘ └───────────────────┘
                            │(2 resp)          │(3 resp)
                            ▼                  ▼
                  detections[]         grouped detections[] + groups_summary
                            \                /
                             \              /
                              ▼            ▼
                       Orchestrator draws visualization,
                       saves to static/outputs/<id>.jpg,
                       returns JSON  ──(4)──▶ Browser
```

This matches four-step diagram: (1) browser uploads to
Flask, (2) Flask → detector, (3) Flask → grouping, (4) Flask → browser with
the JSON (+ visualization).

## 2. Stage-by-stage

### Stage: Data input (orchestrator, `POST /api/analyze`)
- **Input**: `multipart/form-data` with an `image` file field, or JSON
  `{"image_base64": "..."}`.
- **Output**: an in-memory OpenCV BGR `numpy.ndarray`.
- **Technology**: Flask request parsing, Pillow/OpenCV for decode.
- **Files**: `orchestrator/app.py::_extract_image_b64_from_request`,
  `common/image_utils.py::base64_to_image`.
- **Error handling**: missing field → 400; empty filename → 400; unsupported
  extension → 400; corrupt/undecodable bytes → 400 (`ImageDecodeError`);
  payload over 15MB → 413 (`MAX_CONTENT_LENGTH`).

### Stage: Data validation
- Image dimensions checked (`>= 10x10`); empty arrays rejected. Handled
  inline in `orchestrator/app.py::analyze` and `Detector.detect()`.

### Stage: Preprocessing (detector service)
- Resize cap (longest side ≤ 1400px) for latency control on very large
  inputs; grayscale conversion; Gaussian blur to denoise before edge
  detection.
- **Files**: `services/detector/detector.py::Detector._resize_if_needed`.

### Stage: Detection (detector service, `POST /detect`)
- Classical CV: gradient-projection row/column segmentation + contour
  fallback + non-max suppression. See `docs/evaluation_report.md` for how
  and why, and the README's "Upgrading to deep models" section for the
  drop-in replacement path.
- **Output**: `[{"bbox": [x1,y1,x2,y2], "confidence": float}, ...]`.
- **Error handling**: invalid JSON body → 400; bad image → 400; unexpected
  exception → 500 with generic message (no internal trace leaked to client;
  full trace goes to the service log).

### Stage: Grouping (grouping service, `POST /group`)
- Per-detection color histogram (HSV) + texture (LBP) feature vector →
  standardized → agglomerative clustering with a distance threshold set from
  the percentile of pairwise distances (adaptive per image, no fixed "number
  of brands" assumption).
- **Output**: same detections with an added `group_id`, plus a
  `groups_summary` listing each group's members.

### Stage: Post-processing / visualization (orchestrator)
- Boxes drawn on a copy of the original image, color-coded by `group_id`
  (16-color palette, cycled), with a small `G<id>` label per box.
- Saved to `static/outputs/<request_id>.jpg` — never overwrites, one file
  per request, served via Flask's static file handler at
  `/static/outputs/<file>`.

### Stage: Result display
- JSON returned to the browser; the bundled web UI (`templates/index.html`)
  also renders the original + annotated image side by side and a stats
  summary (products detected, groups, latency).

### Stage: Logging & monitoring
- Each service logs to both stdout and (orchestrator only) a rotating-free
  file at `logs/pipeline.log`, with a per-request UUID (`request_id`)
  threaded through every log line so a single request can be traced across
  all three services' logs.
- `GET /api/health` on the orchestrator aggregates its own status plus a
  live health check of both microservices — suitable for a container
  orchestrator's liveness/readiness probe or a simple uptime dashboard.

### Stage: Feedback / retraining
- Out of scope for this classical-CV pipeline (nothing is "trained"). If
  upgraded to a deep detector, the natural extension point is: log
  low-confidence detections (already captured per-box in the response) to a
  review queue, and periodically retrain/fine-tune on operator-corrected
  labels.

## 3. Technology stack

| Layer            | Choice                          | Why |
|-------------------|----------------------------------|-----|
| Web framework      | Flask 3                    |
| Image processing   | OpenCV (headless), Pillow        | No GPU/deep-learning deps needed; available offline |
| Detection          | Custom classical CV (gradient projection + contours) | No internet access to fetch pretrained weights; genuinely low-compute |
| Grouping           | scikit-learn (AgglomerativeClustering), scikit-image (LBP) | Established, well-tested, CPU-only |
| Inter-service transport | HTTP + JSON (base64 images) | Matches "no hardcoded file paths" requirement; each service independently deployable/scalable |
| WSGI (Docker)      | gunicorn                          | Production-appropriate; Flask dev server used only for local/demo (`run_local.sh`) |
| Containerization   | Docker + Docker Compose           | One command to bring up all three services |

## 4. Scaling architecture

Each of the three services is a stateless Flask process (no shared
in-memory state, no session affinity needed), so scaling is horizontal:

- Run N replicas of `detector` and M replicas of `grouping` behind a load
  balancer (e.g. an nginx/Traefik reverse proxy, or a cloud load balancer in
  front of a Docker Swarm/Kubernetes Service). `docker-compose.yml` can be
  extended with `deploy.replicas` under Swarm, or ported to k8s Deployments
  + Services with minimal changes since the services only need a DNS name +
  port.
- The orchestrator only needs `DETECTOR_SERVICE_URL` / `GROUPING_SERVICE_URL`
  to point at the load balancer's address — it never talks to a specific
  instance.
- Because images travel as request bodies (not shared-disk file paths),
  there is no shared filesystem requirement between services, which is what
  makes this horizontally scalable in the first place (a hardcoded-path
  design would break the moment detector and orchestrator run on different
  hosts).
- The detector's resize cap (1400px longest side) bounds worst-case latency
  and memory per request regardless of upload size, which helps predictable
  autoscaling.
- For very high throughput, `services/detector/app.py` and
  `services/grouping/app.py` can be run with more gunicorn workers/threads
  (`--workers`, `--threads` in each Dockerfile's CMD), since both stages are
  CPU-bound and release the GIL during OpenCV/NumPy calls.

## 5. Security, validation, and privacy considerations

- **Input validation**: file extension allow-list, 15MB request size cap
  (`MAX_CONTENT_LENGTH`), image decode wrapped in try/except with a specific
  `ImageDecodeError` rather than letting raw exceptions surface.
- **No secrets required**: the default pipeline uses no external paid AI
  API, so there's no API key to leak. `.env.example` documents where one
  *would* go (`EXTERNAL_DETECTION_API_KEY`) if the detector is upgraded to a
  hosted service, with the standard guidance: never commit `.env`, always
  read keys from environment variables.
- **No persistent storage of uploads**: uploaded images are processed
  entirely in memory and are not written to `static/uploads/`; only the
  annotated *output* visualization is persiste privacy requirements demand it,
  `static/outputs/` can be purged on a schedule (a cron job or a
  `/api/admin/purge` endpoint) — not implemented here since not requested.
- **Error messages**: 5xx responses return a generic message to the client;
  full stack traces are logged server-side only (`logger.exception(...)`),
  avoiding internal path/implementation disclosure.
- **CORS**: not enabled — the bundled UI is same-origin. If a separate
  frontend origin needs to call `/api/analyze`, add `flask-cors` and
  restrict `Access-Control-Allow-Origin` to known origins.

## 6. Deployment architecture

```
                    ┌───────────────────────────┐
                    │      Reverse proxy /        │
                    │   load balancer (optional)  │
                    └──────────────┬──────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │   orchestrator (N replicas)     │
                    └──────┬─────────────────┬────────┘
                            │                 │
                 ┌──────────▼───────┐ ┌───────▼──────────┐
                 │ detector (N reps) │ │ grouping (N reps) │
                 └───────────────────┘ └────────────────────┘
```

`docker-compose.yml` implements the single-host version of this (one
container per service, internal Docker network, orchestrator reaching
`detector`/`grouping` by service name). For multi-host, the same three
images can be deployed to any container orchestrator (Kubernetes, ECS,
Swarm) unchanged — the only environment-specific piece is the two service
URL env vars on the orchestrator.
