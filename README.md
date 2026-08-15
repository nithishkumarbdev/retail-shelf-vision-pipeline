# Retail Shelf AI Pipeline — Product Detection & Brand Grouping

A Flask-based AI pipeline that takes a retail shelf image, detects individual
products, groups them by visual/brand similarity, draws a color-coded
visualization, and returns everything as JSON. Built as three independently
runnable microservices (orchestrator + detector + grouping), .built as three independently runnable microservices for modularity

```
browser --(1) image--> [orchestrator/Flask] --(2)--> [detector service]
                              |
                              +-------(3)--> [grouping service]
                              |
browser <--(4) JSON + viz----+
```

## Important note on the AI models used

**The environment this project was built in has no internet access** (no
`pip install`, no downloading pretrained weights). So instead of a deep
detector (YOLO/etc.) trained on SKU-110K, this project uses:

- **Detector**: a classical computer-vision algorithm (edge/gradient
  projection to find shelf rows and product columns, refined with contour
  analysis). Zero GPU, no training, no external weights required.
- **Grouping**: color-histogram + texture (LBP) feature clustering
  (agglomerative clustering with an adaptive distance threshold), used as a
  visual-similarity proxy for "same brand."

Both are genuinely implemented and tested. They
are also both intentionally swappable: see "Upgrading to deep models" below
for the exact change needed once you have internet/GPU access.

## Project structure

```
retail-shelf-vision-pipeline/
├── orchestrator/         # Flask webserver: web UI + /api/analyze + pipeline glue
│   ├── app.py
│   └── config.py
├── services/
│   ├── detector/          # Detector microservice (own Flask app + Dockerfile)
│   │   ├── app.py
│   │   ├── detector.py
│   │   └── Dockerfile
│   └── grouping/          # Grouping microservice (own Flask app + Dockerfile)
│       ├── app.py
│       ├── grouping.py
│       └── Dockerfile
├── common/
│   └── image_utils.py     # base64 <-> image helpers (no file-path passing)
├── templates/index.html   # minimal upload UI
├── static/
│   ├── uploads/            # (unused for persistence — images are processed in memory)
│   └── outputs/            # saved visualization images, served at /static/outputs/<file>
├── sample_images/          # synthetic test shelf images + generator script
├── tests/                  # unittest suite (25 tests)
├── docs/                   
├── requirements.txt
├── .env.example
├── docker-compose.yml
├── run_local.sh            # run all 3 services locally, no Docker
└── run_tests.sh
```

## Quick start (local, no Docker)

Requires Python 3.10+.

```bash
cd retail-shelf-vision-pipeline
python3 -m venv venv && source venv/bin/activate    # optional but recommended
pip install -r requirements.txt

# generates sample_images/*.jpg used by the tests and for manual trying-out
python3 sample_images/generate_sample.py

./run_local.sh
```

Then open **http://localhost:5000** in a browser, upload an image, and click
"Analyze image." Or call the API directly:

```bash
curl -X POST -F "image=@sample_images/sample_shelf_1.jpg" \
  http://localhost:5000/api/analyze
```

Stop everything with `Ctrl+C` (the script stops all three processes).

## Quick start (Docker)

```bash
docker compose up --build
```

Same URLs as above once containers are healthy (`docker compose ps`).

## Running tests

```bash
./run_tests.sh
# or directly:
python3 -m unittest discover -s tests -v
```

25 tests covering the detector, the grouper, image encode/decode, and the
orchestrator's HTTP endpoints (happy path + failure cases), run against the
actual generated sample images.

## API

See `docs/api_spec.md` for full request/response schemas for every block
(orchestrator, detector, grouping). Quick summary:

- `POST /api/analyze` — multipart `image` file **or** JSON `{"image_base64": ...}`
  → JSON with `detections` (bbox + confidence + group_id per product),
  `groups_summary`, and `visualization_url`.
- `GET /api/health` — aggregate health of orchestrator + both microservices.
- Each microservice also exposes its own `POST /detect` / `POST /group` and
  `GET /health` for independent testing/scaling.

1. **No hardcoded file paths for client-server communication.** Images are
   always sent as base64-encoded JSON payloads (or in-memory multipart
   uploads) between the browser, the orchestrator, and both microservices —
   never as filesystem paths. See `common/image_utils.py`.
2. **Visualizations are saved to files.** Every successful `/api/analyze`
   call writes a color-coded annotated image to `static/outputs/<request_id>.jpg`
   and returns its URL in the response.
3. **Microservices, independently scalable.** `detector` and `grouping` are
   separate Flask processes with their own Dockerfiles/ports; the
   orchestrator calls them over HTTP and can be pointed at multiple replicas
   behind a load balancer. If a downstream service is unreachable, the orchestrator
   transparently falls back to running that stage in-process rather than
   failing the request outright (`ENABLE_INPROCESS_FALLBACK`, on by default)
   — this was verified by killing the detector process mid-run.

## Upgrading to deep models later

The detector and grouper are deliberately isolated behind a stable
interface, so upgrading doesn't touch the Flask layer, the JSON contract, or
the visualization code:

- **Detector** (`services/detector/detector.py`, `Detector.detect()`):
  replace the body with a YOLOv8 (or similar) inference call, e.g. fine-tuned
  on SKU-110K or a grocery-detection dataset. Return the same
  `{"bbox": [x1,y1,x2,y2], "confidence": float}` list.
  ```bash
  pip install ultralytics
  # then load a checkpoint, e.g. YOLO("best.pt"), and map results.boxes to the schema above
  ```
- **Grouping** (`services/grouping/grouping.py`, `ProductGrouper._extract_features()`):
  replace the color/texture feature vector with embeddings from a pretrained
  vision model (e.g. CLIP's image encoder) before clustering, or swap
  clustering for a trained brand classifier if you have labeled data.


