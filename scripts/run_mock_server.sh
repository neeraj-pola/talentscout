#!/bin/bash
# Launches the mock external sources API on port 9417.
# Usage: ./scripts/run_mock_server.sh
set -e
cd "$(dirname "$0")/.."
exec uvicorn mock_sources_api.server:app --port 9417 --reload