#!/bin/bash
# Launches the TalentScout Streamlit UI on port 8501.
# Usage: ./scripts/run_ui.sh
set -e
cd "$(dirname "$0")/.."
exec streamlit run ui/app.py --server.port 8501