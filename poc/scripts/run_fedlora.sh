#!/usr/bin/env bash
# run_fedlora.sh — Launch the FedLoRA-MAML federated training stack (L2-ARCTIC).
#
# Usage:
#   ./scripts/run_fedlora.sh            # preflight + launch (images already built)
#   ./scripts/run_fedlora.sh check      # preflight checks only
#   ./scripts/run_fedlora.sh build      # build Docker images only
#   ./scripts/run_fedlora.sh launch     # launch stack (images already built)
#   ./scripts/run_fedlora.sh status     # show container health + recent server output
#   ./scripts/run_fedlora.sh logs       # tail server logs
#   ./scripts/run_fedlora.sh logs RRBI  # tail a specific node (bare speaker ID)
#   ./scripts/run_fedlora.sh down       # stop and remove containers
#   ./scripts/run_fedlora.sh mlflow     # start MLflow UI on :5000
#
# Environment variables:
#   HF_CACHE             HuggingFace cache dir (default: ~/.cache/huggingface)
#   CUDA_VISIBLE_DEVICES GPU index (default: 0)

set -euo pipefail
cd "$(dirname "$0")/.."  # always run from poc/

COMPOSE_FILE="docker/docker-compose.l2arctic.yml"
NODES_DIR="data/l2arctic_nodes"
MLRUNS_DIR="mlruns"
CKPT_DIR="checkpoints/federated"

export HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mapfile -t REQUIRED_SPEAKERS < <(
    python -c "import json; d=json.load(open('data/l2arctic_split.json')); print('\n'.join(d['meta_train']))"
)

# ── Helpers ───────────────────────────────────────────────────────────────────
log()  { printf '\033[36m[run_fedlora]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[run_fedlora] WARNING:\033[0m %s\n' "$*" >&2; }
err()  { printf '\033[31m[run_fedlora] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

compose() { docker compose -f "$COMPOSE_FILE" "$@"; }

# ── check ─────────────────────────────────────────────────────────────────────
cmd_check() {
    log "Running preflight checks..."

    command -v docker &>/dev/null || err "docker not found in PATH"

    [[ -d "$HF_CACHE" ]] || err "HF_CACHE not found: $HF_CACHE  (set: export HF_CACHE=/path/to/cache)"

    local model_dir="$HF_CACHE/hub/models--facebook--wav2vec2-base-960h"
    if [[ ! -d "$model_dir" ]]; then
        warn "wav2vec2-base-960h not in HF cache: $HF_CACHE"
        warn "Containers run with HF_HUB_OFFLINE=1 and will fail without it."
        warn "Pre-cache: python -c \"from transformers import Wav2Vec2ForCTC; Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h')\""
        read -r -p "Continue anyway? [y/N] " reply
        [[ "$reply" == [yY] ]] || exit 1
    fi

    [[ -f "configs/l2arctic_lora_poc.yaml" ]] || err "Config not found: configs/l2arctic_lora_poc.yaml"

    local missing=()
    for spk in "${REQUIRED_SPEAKERS[@]}"; do
        [[ -f "$NODES_DIR/$spk/features.pt" ]] || missing+=("$spk")
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        err "Missing node data for: ${missing[*]}
  Run: python data/prepare_l2arctic_lora.py --split meta_train"
    fi

    local latest
    latest="$(ls "$CKPT_DIR"/theta_star_lora_round_*.pt 2>/dev/null | sort | tail -1 || true)"
    if [[ -n "$latest" ]]; then
        log "Resume checkpoint: $(basename "$latest")"
    else
        log "No existing checkpoint — starting from fresh LoRA init"
    fi

    mkdir -p "$CKPT_DIR" "$MLRUNS_DIR"
    log "Preflight OK  GPU=$CUDA_VISIBLE_DEVICES  HF_CACHE=$HF_CACHE"
}

# ── build ─────────────────────────────────────────────────────────────────────
cmd_build() {
    log "Building Docker images (voicefl-lora-server, voicefl-lora-node)..."
    compose build
    log "Images built."
}

# ── launch ────────────────────────────────────────────────────────────────────
cmd_launch() {
    log "Stopping any existing stack..."
    compose down --remove-orphans 2>/dev/null || true

    log "Starting L2-ARCTIC federated stack — 1 server + 12 nodes (6 accent groups)..."
    compose up -d

    echo ""
    log "Stack is up. Commands:"
    log "  Status:       ./scripts/run_fedlora.sh status"
    log "  Server logs:  ./scripts/run_fedlora.sh logs"
    log "  Node logs:    ./scripts/run_fedlora.sh logs RRBI"
    log "  Stop:         ./scripts/run_fedlora.sh down"
    log "  MLflow UI:    ./scripts/run_fedlora.sh mlflow"
    echo ""
    log "Attaching to server logs (Ctrl-C detaches; stack keeps running)..."
    compose logs -f server
}

# ── status ────────────────────────────────────────────────────────────────────
cmd_status() {
    log "Container status:"
    compose ps
    echo ""
    log "Recent server output:"
    compose logs server --tail=20
}

# ── logs ──────────────────────────────────────────────────────────────────────
cmd_logs() {
    local target="${1:-server}"
    # Accept bare speaker ID (RRBI) or full service name (node_RRBI)
    if [[ "$target" != "server" && "$target" != node_* ]]; then
        target="node_${target}"
    fi
    log "Tailing logs for: $target  (Ctrl-C to detach)"
    compose logs -f "$target"
}

# ── down ──────────────────────────────────────────────────────────────────────
cmd_down() {
    log "Stopping stack..."
    compose down --remove-orphans
    log "Stack stopped."
}

# ── mlflow ────────────────────────────────────────────────────────────────────
cmd_mlflow() {
    command -v mlflow &>/dev/null || err "mlflow not in PATH. Install: pip install mlflow"
    local abs_mlruns
    abs_mlruns="$(pwd)/$MLRUNS_DIR"
    log "Starting MLflow UI  backend=$abs_mlruns"
    log "Open: http://localhost:5000"
    mlflow ui --backend-store-uri "$abs_mlruns" --host 0.0.0.0 --port 5000
}

# ── main ──────────────────────────────────────────────────────────────────────
CMD="${1:-all}"
shift 2>/dev/null || true

case "$CMD" in
    check)   cmd_check ;;
    build)   cmd_build ;;
    launch)  cmd_launch ;;
    status)  cmd_status ;;
    logs)    cmd_logs "${1:-server}" ;;
    down)    cmd_down ;;
    mlflow)  cmd_mlflow ;;
    all)     cmd_check && cmd_launch ;;
    -h|--help|help)
        grep '^#' "$0" | head -20 | sed 's/^# \?//'
        ;;
    *)
        echo "Unknown command: $CMD"
        echo "Usage: $0 {check|build|launch|status|logs|down|mlflow|all}"
        exit 1
        ;;
esac
