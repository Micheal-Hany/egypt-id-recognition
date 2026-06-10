#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#  EgyptID — REST API launcher
#  Usage:
#    ./run_api.sh              # default: dev mode, port 8000
#    ./run_api.sh --prod       # production mode (gunicorn, 1 worker)
#    ./run_api.sh --port 9000  # custom port
# ══════════════════════════════════════════════════════════════════════════════

set -e

# ── Defaults ──────────────────────────────────────────────────────────────────
HOST="0.0.0.0"
PORT=8000
MODE="dev"          # dev | prod

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --prod)   MODE="prod";  shift ;;
        --dev)    MODE="dev";   shift ;;
        --port)   PORT="$2";    shift 2 ;;
        --host)   HOST="$2";    shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--prod] [--dev] [--port PORT] [--host HOST]"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo "  ███████╗ ██████╗ ██╗   ██╗██████╗ ████████╗██╗██████╗ "
echo "  ██╔════╝██╔════╝ ╚██╗ ██╔╝██╔══██╗╚══██╔══╝██║██╔══██╗"
echo "  █████╗  ██║  ███╗ ╚████╔╝ ██████╔╝   ██║   ██║██║  ██║"
echo "  ██╔══╝  ██║   ██║  ╚██╔╝  ██╔═══╝    ██║   ██║██║  ██║"
echo "  ███████╗╚██████╔╝   ██║   ██║        ██║   ██║██████╔╝"
echo "  ╚══════╝ ╚═════╝    ╚═╝   ╚═╝        ╚═╝   ╚═╝╚═════╝ "
echo ""
echo "  Egyptian National ID Recognition — REST API"
echo "  ─────────────────────────────────────────────"
echo "  Mode : $MODE"
echo "  Host : $HOST"
echo "  Port : $PORT"
echo ""

# ── Locate project root (directory containing this script) ───────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Activate virtualenv if present ───────────────────────────────────────────
if [[ -f ".venv/bin/activate" ]]; then
    echo "  [✓] Activating virtualenv (.venv)"
    source ".venv/bin/activate"
elif [[ -f "venv/bin/activate" ]]; then
    echo "  [✓] Activating virtualenv (venv)"
    source "venv/bin/activate"
else
    echo "  [i] No virtualenv found — using system Python"
fi

# ── Check Python ──────────────────────────────────────────────────────────────
PYTHON=$(command -v python3 || command -v python)
if [[ -z "$PYTHON" ]]; then
    echo "  [✗] Python not found. Please install Python 3.9+"
    exit 1
fi
echo "  [✓] Python: $($PYTHON --version)"

# ── Check required packages ───────────────────────────────────────────────────
echo "  [i] Checking dependencies..."

check_pkg() {
    $PYTHON -c "import $1" 2>/dev/null && \
        echo "  [✓] $1" || \
        { echo "  [✗] $1 not installed — run: pip install $2"; MISSING=1; }
}

MISSING=0
check_pkg fastapi      "fastapi"
check_pkg uvicorn      "uvicorn[standard]"
check_pkg multipart    "python-multipart"
check_pkg cv2          "opencv-python-headless"
check_pkg easyocr      "easyocr"
check_pkg PIL          "Pillow"
check_pkg numpy        "numpy"

if [[ $MISSING -eq 1 ]]; then
    echo ""
    echo "  [✗] Some packages are missing."
    echo "      Run:  pip install -r requirements.txt"
    echo "            pip install fastapi uvicorn[standard] python-multipart"
    exit 1
fi

# ── Check api.py exists ───────────────────────────────────────────────────────
if [[ ! -f "api.py" ]]; then
    echo "  [✗] api.py not found in $SCRIPT_DIR"
    exit 1
fi

# ── YOLO model warnings (non-fatal) ───────────────────────────────────────────
echo ""
[[ -f "detect_id_card.pt" ]] && \
    echo "  [✓] detect_id_card.pt found" || \
    echo "  [⚠] detect_id_card.pt not found — will use OpenCV edge detection fallback"

[[ -f "detect_id.pt" ]] && \
    echo "  [✓] detect_id.pt found" || \
    echo "  [⚠] detect_id.pt not found — field detection disabled"

# ── Launch ────────────────────────────────────────────────────────────────────
echo ""
echo "  ─────────────────────────────────────────────"

if [[ "$MODE" == "prod" ]]; then
    # ── Production: gunicorn + uvicorn worker (single worker — EasyOCR singleton)
    if ! command -v gunicorn &>/dev/null; then
        echo "  [✗] gunicorn not found. Install with: pip install gunicorn"
        exit 1
    fi
    echo "  [▶] Starting production server (gunicorn)..."
    echo "      API  →  http://$HOST:$PORT"
    echo "      Docs →  http://$HOST:$PORT/docs"
    echo ""
    exec gunicorn api:app \
        --worker-class uvicorn.workers.UvicornWorker \
        --workers 1 \
        --bind "$HOST:$PORT" \
        --timeout 120 \
        --access-logfile - \
        --error-logfile -
else
    # ── Development: uvicorn with hot-reload
    echo "  [▶] Starting development server (uvicorn --reload)..."
    echo "      API  →  http://localhost:$PORT"
    echo "      Docs →  http://localhost:$PORT/docs"
    echo "      Stop →  Ctrl+C"
    echo ""
    exec uvicorn api:app \
        --host "$HOST" \
        --port "$PORT" \
        --reload
fi
