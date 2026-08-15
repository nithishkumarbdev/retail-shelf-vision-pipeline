#!/usr/bin/env bash
# Starts the detector, grouping, and orchestrator services locally (no Docker).
# Usage: ./run_local.sh
set -euo pipefail
cd "$(dirname "$0")"

if [ -f .env ]; then
  set -a; source .env; set +a
fi

export DETECTOR_PORT="${DETECTOR_PORT:-5001}"
export GROUPING_PORT="${GROUPING_PORT:-5002}"
export DETECTOR_SERVICE_URL="${DETECTOR_SERVICE_URL:-http://localhost:5001}"
export GROUPING_SERVICE_URL="${GROUPING_SERVICE_URL:-http://localhost:5002}"
export ORCHESTRATOR_PORT="${ORCHESTRATOR_PORT:-5000}"

mkdir -p logs static/uploads static/outputs

echo "Starting detector service on :$DETECTOR_PORT ..."
python3 services/detector/app.py > logs/detector_stdout.log 2>&1 &
DETECTOR_PID=$!

echo "Starting grouping service on :$GROUPING_PORT ..."
python3 services/grouping/app.py > logs/grouping_stdout.log 2>&1 &
GROUPING_PID=$!

echo "Starting orchestrator on :$ORCHESTRATOR_PORT ..."
python3 orchestrator/app.py > logs/orchestrator_stdout.log 2>&1 &
ORCHESTRATOR_PID=$!

cleanup() {
  echo ""
  echo "Stopping services..."
  kill "$DETECTOR_PID" "$GROUPING_PID" "$ORCHESTRATOR_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

sleep 2
echo ""
echo "All services started."
echo "  Web UI:            http://localhost:$ORCHESTRATOR_PORT"
echo "  Orchestrator API:  http://localhost:$ORCHESTRATOR_PORT/api/analyze"
echo "  Detector service:  http://localhost:$DETECTOR_PORT/health"
echo "  Grouping service:  http://localhost:$GROUPING_PORT/health"
echo ""
echo "Press Ctrl+C to stop all services."
echo "Logs: logs/detector_stdout.log, logs/grouping_stdout.log, logs/orchestrator_stdout.log"
echo ""

wait
