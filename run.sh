#!/usr/bin/env bash
# run.sh — single-command launcher for TalentScout.
#
# What this script does:
#   1. Creates a Python venv in .venv/ if it doesn't exist.
#   2. Installs requirements.txt if not already installed.
#   3. Checks for .env (and OPENAI_API_KEY inside it). Stops with instructions
#      if missing.
#   4. Starts mock candidate-source server (port 9417) in the background.
#   5. Waits for the mock server to be ready.
#   6. Starts the FastAPI orchestrator (port 8000) in the background.
#   7. Waits for the API to be ready.
#   8. Starts the Streamlit UI (port 8501) in the foreground.
#
# Ctrl+C kills all three processes cleanly.

set -e

# ────────────────────────────────────────────────────────────────────
# Colors for readability
# ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # no color

info()    { echo -e "${BLUE}[i]${NC} $1"; }
ok()      { echo -e "${GREEN}[✓]${NC} $1"; }
warn()    { echo -e "${YELLOW}[!]${NC} $1"; }
error()   { echo -e "${RED}[✗]${NC} $1"; }

# ────────────────────────────────────────────────────────────────────
# Paths and ports
# ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"
REQUIREMENTS="requirements.txt"
ENV_FILE=".env"
ENV_EXAMPLE=".env.example"

MOCK_PORT=9417
API_PORT=8000
UI_PORT=8501

# PID file so Ctrl+C handler can find and kill the background processes
PID_DIR=".run_pids"
mkdir -p "$PID_DIR"

# ────────────────────────────────────────────────────────────────────
# Cleanup on exit (Ctrl+C or normal termination)
# ────────────────────────────────────────────────────────────────────
cleanup() {
    echo ""
    info "Shutting down..."

    # Kill any process whose PID we recorded
    for pidfile in "$PID_DIR"/*.pid; do
        [ -e "$pidfile" ] || continue
        pid=$(cat "$pidfile")
        name=$(basename "$pidfile" .pid)
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            # Give it 2 seconds to exit gracefully
            sleep 0.5
            # If still alive, force-kill
            if kill -0 "$pid" 2>/dev/null; then
                kill -9 "$pid" 2>/dev/null || true
            fi
            ok "Stopped $name (pid $pid)"
        fi
        rm -f "$pidfile"
    done

    # Belt-and-suspenders — kill anything still bound to our ports
    for port in $MOCK_PORT $API_PORT $UI_PORT; do
        if command -v lsof >/dev/null 2>&1; then
            pid=$(lsof -ti :$port 2>/dev/null || true)
            if [ -n "$pid" ]; then
                kill -9 $pid 2>/dev/null || true
            fi
        fi
    done

    rmdir "$PID_DIR" 2>/dev/null || true
    info "Goodbye."
    exit 0
}
trap cleanup INT TERM EXIT

# ────────────────────────────────────────────────────────────────────
# 1. Find a compatible Python (3.11, 3.12, or 3.13)
# ────────────────────────────────────────────────────────────────────
# We need 3.11+ for the type-union syntax used in the code, but the
# scientific-Python ecosystem (torch, chromadb, sentence-transformers)
# is generally a few months behind on supporting new Python releases.
# As of mid-2026, 3.14 wheels for several deps are missing or unstable.
# So we cap at 3.13 and try a few common interpreter names if the
# default `python3` is outside the supported range.

PYTHON_CMD=""
SUPPORTED_MINORS=(13 12 11)   # tried in this order if `python3` itself is unusable

check_python_version() {
    # Returns 0 if version is in [3.11, 3.13]; 1 otherwise.
    local cmd=$1
    if ! command -v "$cmd" >/dev/null 2>&1; then
        return 1
    fi
    local version
    version=$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null) || return 1
    local major=${version%.*}
    local minor=${version#*.}
    if [ "$major" -eq 3 ] && [ "$minor" -ge 11 ] && [ "$minor" -le 13 ]; then
        return 0
    fi
    return 1
}

# Try the default `python3` first
if check_python_version python3; then
    PYTHON_CMD=python3
else
    # Default is unusable — try specific version names
    for minor in "${SUPPORTED_MINORS[@]}"; do
        if check_python_version "python3.$minor"; then
            PYTHON_CMD="python3.$minor"
            break
        fi
    done
fi

if [ -z "$PYTHON_CMD" ]; then
    default_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "not installed")
    echo ""
    error "No compatible Python interpreter found."
    error "TalentScout requires Python 3.11, 3.12, or 3.13."
    error "Your default python3 is: $default_version"
    echo ""
    echo "To fix this on macOS (Homebrew):"
    echo "  brew install python@3.13"
    echo "  Then re-run ./run.sh"
    echo ""
    echo "On Linux (apt):"
    echo "  sudo apt install python3.13 python3.13-venv"
    echo ""
    echo "Or use pyenv to install and pin a version:"
    echo "  pyenv install 3.13.0"
    echo "  pyenv local 3.13.0"
    echo ""
    # Don't run cleanup on this exit (no processes started yet)
    trap - INT TERM EXIT
    rmdir "$PID_DIR" 2>/dev/null || true
    exit 1
fi

PY_VERSION=$("$PYTHON_CMD" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')
ok "Python $PY_VERSION ($PYTHON_CMD)"

# ────────────────────────────────────────────────────────────────────
# 2. Create venv if missing
# ────────────────────────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    info "Creating virtual environment in $VENV_DIR/ ..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
    ok "Virtual environment created"
fi

# ────────────────────────────────────────────────────────────────────
# 3. Activate venv
# ────────────────────────────────────────────────────────────────────
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
ok "Activated venv"

# ────────────────────────────────────────────────────────────────────
# 4. Install requirements if needed
# ────────────────────────────────────────────────────────────────────
# Use a marker file so we don't re-install every run. The marker also tracks
# the hash of requirements.txt so changes trigger a re-install.
INSTALL_MARKER="$VENV_DIR/.requirements_installed"
REQ_HASH=$(shasum -a 256 "$REQUIREMENTS" 2>/dev/null | cut -d' ' -f1 || md5sum "$REQUIREMENTS" 2>/dev/null | cut -d' ' -f1)

if [ ! -f "$INSTALL_MARKER" ] || [ "$(cat "$INSTALL_MARKER" 2>/dev/null)" != "$REQ_HASH" ]; then
    info "Installing dependencies (this may take a few minutes the first time)..."
    pip install --upgrade pip --quiet
    pip install -r "$REQUIREMENTS" --quiet
    echo "$REQ_HASH" > "$INSTALL_MARKER"
    ok "Dependencies installed"
else
    ok "Dependencies already installed"
fi

# ────────────────────────────────────────────────────────────────────
# 5. Check for .env
# ────────────────────────────────────────────────────────────────────
if [ ! -f "$ENV_FILE" ]; then
    if [ -f "$ENV_EXAMPLE" ]; then
        cp "$ENV_EXAMPLE" "$ENV_FILE"
        echo ""
        warn "No .env file found. Created one from .env.example."
        warn "Edit .env and add your OPENAI_API_KEY, then re-run ./run.sh"
        echo ""
        # Don't run cleanup on this exit (no processes started yet)
        trap - INT TERM EXIT
        rmdir "$PID_DIR" 2>/dev/null || true
        exit 1
    else
        error ".env and .env.example both missing. Cannot continue."
        exit 1
    fi
fi

# Source the .env so the subprocesses inherit it
set -a
# shellcheck source=/dev/null
source "$ENV_FILE"
set +a

# Verify OPENAI_API_KEY is set (not empty, not the placeholder)
if [ -z "${OPENAI_API_KEY:-}" ] || [ "$OPENAI_API_KEY" = "sk-your-key-here" ]; then
    error "OPENAI_API_KEY is not set in .env"
    error "Edit .env and add a valid OpenAI API key, then re-run ./run.sh"
    trap - INT TERM EXIT
    rmdir "$PID_DIR" 2>/dev/null || true
    exit 1
fi
ok "OPENAI_API_KEY found in .env"

# ────────────────────────────────────────────────────────────────────
# 6. Check ports are free
# ────────────────────────────────────────────────────────────────────
check_port_free() {
    local port=$1
    local name=$2
    if command -v lsof >/dev/null 2>&1; then
        if lsof -ti :$port >/dev/null 2>&1; then
            error "Port $port is already in use (needed for $name)."
            error "Kill the process using it, or change the port in run.sh"
            exit 1
        fi
    fi
}
check_port_free $MOCK_PORT "mock server"
check_port_free $API_PORT "API"
check_port_free $UI_PORT "UI"

# ────────────────────────────────────────────────────────────────────
# 7. Start mock candidate-source server (port 9417)
# ────────────────────────────────────────────────────────────────────
info "Starting mock candidate-source server on port $MOCK_PORT..."
python -m uvicorn mock_sources_api.server:app \
    --host 0.0.0.0 --port $MOCK_PORT \
    > "$PID_DIR/mock_server.log" 2>&1 &
echo $! > "$PID_DIR/mock_server.pid"

# Wait for mock server to be ready (poll the health endpoint up to 15s)
for i in $(seq 1 30); do
    if curl -s "http://localhost:$MOCK_PORT/health" > /dev/null 2>&1; then
        ok "Mock server ready (pid $(cat "$PID_DIR/mock_server.pid"))"
        break
    fi
    sleep 0.5
    if [ $i -eq 30 ]; then
        error "Mock server failed to start within 15s. See $PID_DIR/mock_server.log"
        exit 1
    fi
done

# ────────────────────────────────────────────────────────────────────
# 8. Start FastAPI orchestrator (port 8000)
# ────────────────────────────────────────────────────────────────────
info "Starting FastAPI orchestrator on port $API_PORT..."
python -m uvicorn app.api.server:app \
    --host 0.0.0.0 --port $API_PORT \
    > "$PID_DIR/api.log" 2>&1 &
echo $! > "$PID_DIR/api.pid"

# Wait for API to be ready
for i in $(seq 1 60); do
    if curl -s "http://localhost:$API_PORT/health" > /dev/null 2>&1; then
        ok "API ready (pid $(cat "$PID_DIR/api.pid"))"
        break
    fi
    sleep 0.5
    if [ $i -eq 60 ]; then
        error "API failed to start within 30s. See $PID_DIR/api.log"
        exit 1
    fi
done

# ────────────────────────────────────────────────────────────────────
# 9. Start Streamlit UI (port 8501) in the foreground
# ────────────────────────────────────────────────────────────────────
echo ""
ok "All services up:"
echo "    • Mock server:  http://localhost:$MOCK_PORT"
echo "    • API:          http://localhost:$API_PORT"
echo "    • UI:           http://localhost:$UI_PORT  ← OPEN THIS"
echo ""
info "Press Ctrl+C to stop everything."
echo ""

# Streamlit runs in foreground so the user sees its output
streamlit run ui/app.py \
    --server.port $UI_PORT \
    --server.headless true \
    --browser.gatherUsageStats false

# If streamlit exits on its own, trigger cleanup
cleanup
