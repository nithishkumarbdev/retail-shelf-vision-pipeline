import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DETECTOR_SERVICE_URL = os.environ.get("DETECTOR_SERVICE_URL", "http://localhost:5001")
GROUPING_SERVICE_URL = os.environ.get("GROUPING_SERVICE_URL", "http://localhost:5002")

# If a downstream microservice call fails (connection error / timeout), fall back to
# running that stage's logic in-process so the pipeline stays available. This is a
# resilience feature, not a way to skip building real microservices -- the primary
# path always goes over HTTP.
ENABLE_INPROCESS_FALLBACK = os.environ.get("ENABLE_INPROCESS_FALLBACK", "true").lower() == "true"

SERVICE_TIMEOUT_SECONDS = float(os.environ.get("SERVICE_TIMEOUT_SECONDS", "10"))

OUTPUT_DIR = os.path.join(BASE_DIR, "static", "outputs")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")

MAX_CONTENT_LENGTH = 15 * 1024 * 1024  # 15 MB request cap

LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "pipeline.log")

PORT = int(os.environ.get("ORCHESTRATOR_PORT", "5000"))
HOST = os.environ.get("ORCHESTRATOR_HOST", "0.0.0.0")
DEBUG = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
