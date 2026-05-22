#!/usr/bin/env bash
# launch_node.sh — Set up and start FL worker node(s) on an OCI GPU instance.
#
# This script:
#   1. Installs Docker + NVIDIA Container Toolkit (if not present)
#   2. Clones the repository (if not present)
#   3. Installs Python deps + pre-caches HuggingFace models
#   4. Prepares speaker data (if not already prepared)
#   5. Builds the node Docker image
#   6. Starts one FL client container per speaker
#
# Prerequisites:
#   - OCI VM.GPU.A10.1 (or any NVIDIA GPU instance) with Oracle Linux 8
#   - The FL server must be running and reachable
#
# Usage:
#   bash scripts/oci/launch_node.sh --server 10.0.0.5:8080 --speakers RRBI,TNI
#   bash scripts/oci/launch_node.sh --server 10.0.0.5:8080 --speakers BWC,LXC,YKWK,HJK
#   bash scripts/oci/launch_node.sh --server 10.0.0.5:8080 --speakers RRBI  # single node
#
# Environment variables:
#   FL_SERVER_ADDRESS   Server address (alternative to --server)
#   REPO_URL            Git repository URL (if not passed via --repo)
#   WORK_DIR            Working directory (default: ~/vision_fl)
#   HF_CACHE            HuggingFace cache directory (default: ~/.cache/huggingface)

set -euo pipefail

# ── Parse arguments ───────────────────────────────────────────────────────────
SERVER="${FL_SERVER_ADDRESS:-}"
SPEAKERS=""
REPO_URL="${REPO_URL:-https://github.com/TheGitSlender/Voice_FL.git}"
BRANCH="${BRANCH:-deploy/oci-dynamic}"
WORK_DIR="${WORK_DIR:-$HOME/voicefl}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
CONFIG="configs/l2arctic_lora_poc.yaml"
NODES_DIR="data/l2arctic_nodes"
SKIP_SETUP=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --server)       SERVER="$2"; shift 2 ;;
        --speakers)     SPEAKERS="$2"; shift 2 ;;
        --repo)         REPO_URL="$2"; shift 2 ;;
        --branch)       BRANCH="$2"; shift 2 ;;
        --config)       CONFIG="$2"; shift 2 ;;
        --work_dir)     WORK_DIR="$2"; shift 2 ;;
        --hf_cache)     HF_CACHE="$2"; shift 2 ;;
        --skip_setup)   SKIP_SETUP=true; shift ;;
        -h|--help)
            grep '^#' "$0" | head -25 | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

[[ -z "$SERVER" ]] && { echo "ERROR: --server <IP:PORT> or FL_SERVER_ADDRESS required"; exit 1; }
[[ -z "$SPEAKERS" ]] && { echo "ERROR: --speakers <SPEAKER1,SPEAKER2,...> required"; exit 1; }

# Split comma-separated speakers into array
IFS=',' read -ra SPEAKER_LIST <<< "$SPEAKERS"

log()  { printf '\033[36m[node-setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[node-setup] WARNING:\033[0m %s\n' "$*" >&2; }
err()  { printf '\033[31m[node-setup] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

if ! $SKIP_SETUP; then

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
fi

# ── 2. Install NVIDIA Container Toolkit if missing ───────────────────────────
if ! docker info 2>/dev/null | grep -qi nvidia; then
    log "Installing NVIDIA Container Toolkit..."
    if command -v dnf &>/dev/null; then
        distribution=$(. /etc/os-release && echo "$ID$VERSION_ID")
        curl -s -L "https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.repo" \
            | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo > /dev/null
        sudo dnf install -y nvidia-container-toolkit
    elif command -v apt-get &>/dev/null; then
        distribution=$(. /etc/os-release && echo "$ID$VERSION_ID")
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
            | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -s -L "https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list" \
            | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
            | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
        sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
    fi
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
fi

# Verify GPU access in Docker
log "Verifying GPU access in Docker..."
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1 \
    || err "GPU not accessible inside Docker. Check NVIDIA drivers and Container Toolkit."
log "GPU access verified ✓"

# ── 3. Clone repository if missing ───────────────────────────────────────────
if [[ ! -d "$WORK_DIR" ]]; then
    if [[ -z "$REPO_URL" ]]; then
        err "Repository not found at $WORK_DIR and no --repo URL provided."
    fi
    log "Cloning repository to $WORK_DIR..."
    git clone -b "$BRANCH" "$REPO_URL" "$WORK_DIR"
fi

cd "$WORK_DIR"

# ── 4. Install Python deps + pre-cache HuggingFace models ───────────────────
log "Running environment setup..."
export HF_CACHE
bash scripts/setup_env.sh

# ── 5. Prepare L2-ARCTIC data if missing ─────────────────────────────────────
MISSING=()
for spk in "${SPEAKER_LIST[@]}"; do
    if [[ ! -f "$NODES_DIR/$spk/features.pt" ]]; then
        MISSING+=("$spk")
    fi
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
    log "Preparing L2-ARCTIC data for: ${MISSING[*]}"
    python data/prepare_l2arctic_lora.py --split meta_train
fi

# ── 6. Build node Docker image ───────────────────────────────────────────────
log "Building node Docker image (voicefl-lora-node)..."
docker build -t voicefl-lora-node -f docker/Dockerfile.node .

fi  # end SKIP_SETUP

cd "${WORK_DIR}"

# ── 7. Launch FL client containers ───────────────────────────────────────────
log "Starting ${#SPEAKER_LIST[@]} FL node(s) connecting to $SERVER..."

for spk in "${SPEAKER_LIST[@]}"; do
    CONTAINER_NAME="node-${spk}"

    # Skip if container already running
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        warn "Container $CONTAINER_NAME already running — skipping"
        continue
    fi

    # Remove stopped container with same name
    docker rm "$CONTAINER_NAME" 2>/dev/null || true

    DATA_DIR="$(pwd)/$NODES_DIR/$spk"
    if [[ ! -f "$DATA_DIR/features.pt" ]]; then
        warn "Data not found for $spk at $DATA_DIR — skipping"
        continue
    fi

    log "  Starting $CONTAINER_NAME (speaker=$spk)..."
    docker run -d --gpus all \
        --name "$CONTAINER_NAME" \
        -v "$DATA_DIR:/data:ro" \
        -v "$HF_CACHE:/root/.cache/huggingface:ro" \
        -e CUDA_VISIBLE_DEVICES=0 \
        -e FL_SERVER_ADDRESS="$SERVER" \
        -e SPEAKER_ID="$spk" \
        -e HF_HUB_OFFLINE=1 \
        -e TRANSFORMERS_OFFLINE=1 \
        -e PYTHONUNBUFFERED=1 \
        voicefl-lora-node \
        python -u federated/run_node_lora.py \
            --config "$CONFIG" \
            --speaker_id "$spk" \
            --data_dir /data \
            --server_address "$SERVER"
done

# ── 8. Print status ──────────────────────────────────────────────────────────
echo ""
log "═══════════════════════════════════════════════════════"
log "  ${#SPEAKER_LIST[@]} FL node(s) started!"
log "═══════════════════════════════════════════════════════"
echo ""
log "  Server: $SERVER"
log "  Speakers: ${SPEAKER_LIST[*]}"
echo ""
log "  View logs:   docker logs -f node-${SPEAKER_LIST[0]}"
log "  Stop all:    docker stop ${SPEAKER_LIST[*]/#/node-}"
log "  Add more:    bash scripts/oci/launch_node.sh --server $SERVER --speakers BWC,LXC --skip_setup"
echo ""
