#!/usr/bin/env bash
# launch_server.sh — Set up and start the FL server on an OCI Always Free ARM instance.
#
# This script:
#   1. Installs Docker (if not present)
#   2. Clones the repository (if not present)
#   3. Builds the lightweight server Docker image (no GPU needed)
#   4. Starts the FL server container + MLflow UI
#
# Prerequisites:
#   - OCI VM.Standard.A1.Flex (Always Free) with Oracle Linux 8 or Ubuntu 22.04 ARM
#   - Security list allows TCP 8080 (from VCN CIDR) and TCP 5000 (from your IP)
#
# Usage:
#   bash scripts/oci/launch_server.sh                         # defaults: 50 rounds, min 3 clients
#   bash scripts/oci/launch_server.sh --rounds 200            # 200 rounds
#   bash scripts/oci/launch_server.sh --min_clients 6         # wait for 6 nodes before starting
#   bash scripts/oci/launch_server.sh --repo <git-url>        # specify repo URL
#
# Environment variables:
#   REPO_URL          Git repository URL (if not passed via --repo)
#   WORK_DIR          Working directory (default: ~/vision_fl)
#   FL_PORT           Flower gRPC port (default: 8080)
#   MLFLOW_PORT       MLflow UI port (default: 5000)

set -euo pipefail

# ── Parse arguments ───────────────────────────────────────────────────────────
ROUNDS=50
MIN_CLIENTS=3
REPO_URL="${REPO_URL:-https://github.com/TheGitSlender/Voice_FL.git}"
BRANCH="${BRANCH:-deploy/oci-dynamic}"
WORK_DIR="${WORK_DIR:-$HOME/voicefl}"
FL_PORT="${FL_PORT:-8080}"
MLFLOW_PORT="${MLFLOW_PORT:-5000}"
CONFIG="configs/l2arctic_lora_poc.yaml"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rounds)       ROUNDS="$2"; shift 2 ;;
        --min_clients)  MIN_CLIENTS="$2"; shift 2 ;;
        --repo)         REPO_URL="$2"; shift 2 ;;
        --branch)       BRANCH="$2"; shift 2 ;;
        --config)       CONFIG="$2"; shift 2 ;;
        --work_dir)     WORK_DIR="$2"; shift 2 ;;
        -h|--help)
            grep '^#' "$0" | head -25 | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

log()  { printf '\033[36m[server-setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[server-setup] WARNING:\033[0m %s\n' "$*" >&2; }
err()  { printf '\033[31m[server-setup] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# ── 1. Install Docker if missing ─────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
    log "Installing Docker..."
    if command -v dnf &>/dev/null; then
        sudo dnf install -y docker
    elif command -v apt-get &>/dev/null; then
        sudo apt-get update && sudo apt-get install -y docker.io
    else
        err "Unsupported package manager. Install Docker manually."
    fi
    sudo systemctl enable --now docker
    sudo usermod -aG docker "$USER"
    log "Docker installed. You may need to log out and back in for group changes."
fi

# ── 2. Clone repository if missing ───────────────────────────────────────────
if [[ ! -d "$WORK_DIR" ]]; then
    if [[ -z "$REPO_URL" ]]; then
        err "Repository not found at $WORK_DIR and no --repo URL provided."
    fi
    log "Cloning repository to $WORK_DIR..."
    git clone -b "$BRANCH" "$REPO_URL" "$WORK_DIR"
fi

cd "$WORK_DIR"
log "Working directory: $(pwd)"

# ── 3. Install Python dependencies for MLflow + analysis ─────────────────────
log "Installing Python dependencies for MLflow and analysis..."
pip3 install --quiet mlflow matplotlib numpy PyYAML 2>/dev/null || \
    python3 -m pip install --quiet mlflow matplotlib numpy PyYAML

# ── 4. Build server Docker image ─────────────────────────────────────────────
log "Building server Docker image (voicefl-lora-server)..."
docker build -t voicefl-lora-server -f docker/Dockerfile.server .

# ── 5. Stop any existing server container ─────────────────────────────────────
if docker ps -a --format '{{.Names}}' | grep -q '^fl-server$'; then
    log "Stopping existing fl-server container..."
    docker stop fl-server 2>/dev/null || true
    docker rm fl-server 2>/dev/null || true
fi

# ── 6. Create checkpoint and MLflow directories ──────────────────────────────
mkdir -p checkpoints/federated checkpoints/centralized mlruns

# ── 7. Start the FL server ────────────────────────────────────────────────────
log "Starting FL server (rounds=$ROUNDS, min_clients=$MIN_CLIENTS, port=$FL_PORT)..."
docker run -d --name fl-server \
    -p "$FL_PORT:8080" \
    -v "$(pwd)/checkpoints:/app/checkpoints" \
    -v "$(pwd)/mlruns:/app/mlruns" \
    -v "$(pwd)/configs:/app/configs" \
    -e PYTHONUNBUFFERED=1 \
    -e MLFLOW_TRACKING_URI=file:///app/mlruns \
    --restart unless-stopped \
    voicefl-lora-server \
    python -u federated/server_lora.py \
        --config "$CONFIG" \
        --rounds "$ROUNDS" \
        --min_clients "$MIN_CLIENTS"

# ── 8. Start MLflow UI ───────────────────────────────────────────────────────
log "Starting MLflow UI on port $MLFLOW_PORT..."
# Kill any existing MLflow process
pkill -f "mlflow.*$MLFLOW_PORT" 2>/dev/null || true
nohup mlflow ui \
    --backend-store-uri "$(pwd)/mlruns" \
    --host 0.0.0.0 \
    --port "$MLFLOW_PORT" \
    > /tmp/mlflow.log 2>&1 &

# ── 9. Print connection info ─────────────────────────────────────────────────
echo ""
log "═══════════════════════════════════════════════════════"
log "  FL Server is running!"
log "═══════════════════════════════════════════════════════"
echo ""

# Get private IP for node connections (within the VCN)
PRIVATE_IP=$(ip -4 addr show | grep 'inet 10\.' | head -1 | awk '{print $2}' | cut -d/ -f1 || echo "<PRIVATE_IP>")
PUBLIC_IP=$(curl -s --connect-timeout 3 ifconfig.me 2>/dev/null || echo "<PUBLIC_IP>")

log "  gRPC endpoint:    $PRIVATE_IP:$FL_PORT (VCN internal)"
log "  MLflow UI:        http://$PUBLIC_IP:$MLFLOW_PORT"
echo ""
log "  Connect nodes with:"
log "    FL_SERVER_ADDRESS=$PRIVATE_IP:$FL_PORT bash scripts/oci/launch_node.sh --speakers RRBI,TNI"
echo ""
log "  Server logs:      docker logs -f fl-server"
log "  Stop server:      docker stop fl-server"
echo ""
