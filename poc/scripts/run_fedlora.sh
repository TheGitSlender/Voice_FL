#!/usr/bin/env bash
# run_fedlora.sh — Launch the FedLoRA-MAML federated training stack.
#
# Usage:
#   ./scripts/run_fedlora.sh            # preflight + build + launch (default)
#   ./scripts/run_fedlora.sh check      # preflight checks only
#   ./scripts/run_fedlora.sh build      # build Docker images only
#   ./scripts/run_fedlora.sh launch     # launch stack (images already built)
#   ./scripts/run_fedlora.sh logs       # tail server logs of a running stack
#   ./scripts/run_fedlora.sh down       # stop the stack
#   ./scripts/run_fedlora.sh mlflow     # start MLflow UI (host-side)
#
# Environment variables:
#   HF_CACHE             HuggingFace cache dir (default: ~/.cache/huggingface)
#   CUDA_VISIBLE_DEVICES GPU index (default: 0)

set -euo pipefail
cd "$(dirname "$0")/.."  # always run from poc/

COMPOSE_FILE="docker/docker-compose.vctk.yml"
NODES_DIR="data/vctk_nodes"
MLRUNS_DIR="mlruns"
CKPT_DIR="checkpoints/federated"

export HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

REQUIRED_SPEAKERS=(p225 p226 p227 p228 p229 p230 p231 p232 p233 p234 p236 p237)

# ── Helpers ───────────────────────────────────────────────────────────────────
log()  { printf '\033[36m[run_fedlora]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[run_fedlora] WARNING:\033[0m %s\n' "$*" >&2; }
err()  { printf '\033[31m[run_fedlora] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

compose() { docker compose -f "$COMPOSE_FILE" "$@"; }

# ── check ─────────────────────────────────────────────────────────────────────
cmd_check() {
    log "Running preflight checks..."

    command -v docker &>/dev/null || err "docker not found in PATH"

    [[ -d "$HF_CACHE" ]] || err "HF_CACHE not found: $HF_CACHE  (set: export HF_CACHE=/path)"

    local model_dir="$HF_CACHE/hub/models--facebook--wav2vec2-base-960h"
    if [[ ! -d "$model_dir" ]]; then
        warn "wav2vec2-base-960h not found in $HF_CACHE"
        warn "Containers use HF_HUB_OFFLINE=1 and will fail without a cached model."
        warn "Pre-cache: python -c \"from transformers import Wav2Vec2ForCTC; Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h')\""
        read -r -p "Continue anyway? [y/N] " reply
        [[ "$reply" == [yY] ]] || exit 1
    fi

    [[ -f "configs/vctk_lora_poc.yaml" ]] || err "Config not found: configs/vctk_lora_poc.yaml"

    local missing=()
    for spk in "${REQUIRED_SPEAKERS[@]}"; do
        [[ -f "$NODES_DIR/$spk/features.pt" ]] || missing+=("$spk")
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        err "Missing VCTK node data for: ${missing[*]}\n  Run: python data/prepare_vctk_lora.py"
    fi

    # Existing checkpoint? Show resume status.
    local latest
    latest="$(ls "$CKPT_DIR"/theta_star_lora_round_*.pt 2>/dev/null | sort | tail -1 || true)"
    if [[ -n "$latest" ]]; then
        log "Resume checkpoint found: $(basename "$latest")"
    elif [[ -f "$CKPT_DIR/theta_star_lora.pt" ]]; then
        log "Final checkpoint found (will resume from it)"
    else
        log "No existing checkpoint — starting from fresh LoRA init"
    fi

    mkdir -p "$CKPT_DIR" "$MLRUNS_DIR"

    log "Preflight OK  GPU=$CUDA_VISIBLE_DEVICES  HF_CACHE=$HF_CACHE"
}

# ── build ─────────────────────────────────────────────────────────────────────
cmd_build() {
    cmd_check
    log "Building Docker images (voicefl-lora-server, voicefl-lora-node)..."
    compose build
    log "Images built."
}

# ── launch ────────────────────────────────────────────────────────────────────
cmd_launch() {
    log "Stopping any existing stack..."
    compose down --remove-orphans 2>/dev/null || true

    log "Starting federated stack (1 server + 12 nodes)..."
    compose up -d

    echo ""
    log "Stack is up. Useful commands:"
    log "  All logs:    docker compose -f $COMPOSE_FILE logs -f"
    log "  Node logs:   docker compose -f $COMPOSE_FILE logs -f node_p225"
    log "  Stop:        $0 down"
    log "  MLflow UI:   $0 mlflow"
    echo ""
    log "Attaching to server logs (Ctrl-C detaches; stack keeps running)..."
    compose logs -f server
}

# ── logs ──────────────────────────────────────────────────────────────────────
cmd_logs() {
    local service="${1:-server}"
    log "Tailing logs for: $service"
    compose logs -f "$service"
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
    logs)    cmd_logs "${1:-server}" ;;
    down)    cmd_down ;;
    mlflow)  cmd_mlflow ;;
    all)     cmd_check && cmd_build && cmd_launch ;;
    -h|--help|help)
        grep '^#' "$0" | head -20 | sed 's/^# \?//'
        ;;
    *)
        echo "Unknown command: $CMD"
        echo "Usage: $0 {check|build|launch|logs|down|mlflow|all}"
        exit 1
        ;;
esac
