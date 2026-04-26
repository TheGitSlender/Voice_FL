#!/usr/bin/env bash
# Phase 1 (FOMAML-ANIL) only. For Phase 2 (FedLoRA-MAML), use run_fedlora.sh.
# Run the full 20-round federated PoC.
# Run from the repo root: ./scripts/run_poc.sh
#
# For a quick smoke test (5 rounds):
#   NUM_ROUNDS=5 ./scripts/run_poc.sh

set -euo pipefail
cd "$(dirname "$0")/.."

NUM_ROUNDS="${NUM_ROUNDS:-20}"
ENV_FILE=".env.nodes"

if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: .env.nodes not found. Run ./scripts/build.sh first."
  exit 1
fi

# Load node hashes
set -a
source "$ENV_FILE"
set +a

echo "=== VoiceFL-MAML PoC — Federated Run ($NUM_ROUNDS rounds) ==="
echo ""

# Pre-flight checks
echo "[PRE-FLIGHT]"
for i in 1 2 3 4 5; do
  var="NODE_HASH_$i"
  hash="${!var}"
  node_dir="data/nodes/$hash"
  if [ ! -d "$node_dir" ]; then
    echo "ERROR: Node directory not found: $node_dir"
    exit 1
  fi
  feat="$node_dir/features.pt"
  if [ ! -f "$feat" ]; then
    echo "ERROR: features.pt not found: $feat"
    exit 1
  fi
  echo "  node$i: $hash (OK)"
done
echo ""

# Sovereignty pre-check
echo "[SOVEREIGNTY PRE-CHECK]"
if find data/nodes -name "*.pkl" | grep -q .; then
  echo "FAIL: .pkl files found — run python data/features.py"
  exit 1
fi
if grep -r "speaker_id" data/nodes/ 2>/dev/null | grep -q .; then
  echo "FAIL: speaker_id found in data/nodes/"
  exit 1
fi
echo "  OK — no .pkl, no speaker_id"
echo ""

# Run Docker stack
echo "[FEDERATED TRAINING] Starting $NUM_ROUNDS rounds..."
export NUM_ROUNDS
docker-compose -f docker/docker-compose.yml up --abort-on-container-exit

echo ""
echo "[EVALUATION]"
python evaluation/eval_poc.py --config configs/poc.yaml
