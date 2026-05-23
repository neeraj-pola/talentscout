#!/bin/bash
# Launches the TalentScout API on port 8000.
# Usage: ./scripts/run_api.sh
set -e
cd "$(dirname "$0")/.."
exec uvicorn app.api.server:app --port 8000 --reload