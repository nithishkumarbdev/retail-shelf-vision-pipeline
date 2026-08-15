#!/usr/bin/env bash
# Runs the full automated test suite.
# Usage: ./run_tests.sh
set -euo pipefail
cd "$(dirname "$0")"

export DETECTOR_SERVICE_URL="http://localhost:5991"   # deliberately unreachable -> exercises fallback path
export GROUPING_SERVICE_URL="http://localhost:5992"

python3 -m unittest discover -s tests -v
