#!/usr/bin/env bash
# Build Docker images and generate resolved docker-compose with real node hashes.
# Run from the repo root: ./scripts/build.sh

set -euo pipefail
cd "$(dirname "$0")/.."

SELECTION_FILE="data/speaker_selection.json"

if [ ! -f "$SELECTION_FILE" ]; then
  echo "ERROR: $SELECTION_FILE not found. Run ./scripts/prepare_data.sh first."
  exit 1
fi

echo "=== Reading node hashes from $SELECTION_FILE ==="
mapfile -t HASHES < <(python -c "
import json
d = json.load(open('$SELECTION_FILE'))
for h in list(d.keys()):
    print(h)
")

N="${#HASHES[@]}"
if [ "$N" -ne 5 ]; then
  echo "ERROR: Expected 5 nodes, found $N."
  exit 1
fi

echo "Nodes:"
for i in "${!HASHES[@]}"; do
  echo "  node$((i+1)): ${HASHES[$i]}"
done
echo ""

# Export hashes as environment variables
export NODE_HASH_1="${HASHES[0]}"
export NODE_HASH_2="${HASHES[1]}"
export NODE_HASH_3="${HASHES[2]}"
export NODE_HASH_4="${HASHES[3]}"
export NODE_HASH_5="${HASHES[4]}"

# Create checkpoint directories
for i in 1 2 3 4 5; do
  mkdir -p "checkpoints/nodes/node${i}"
done
mkdir -p checkpoints/federated
mkdir -p checkpoints/centralized

echo "=== Building Docker images ==="
docker-compose -f docker/docker-compose.yml build

echo ""
echo "=== Build complete ==="
docker images | grep voicefl

echo ""
echo "Node hashes exported:"
echo "  NODE_HASH_1=$NODE_HASH_1"
echo "  NODE_HASH_2=$NODE_HASH_2"
echo "  NODE_HASH_3=$NODE_HASH_3"
echo "  NODE_HASH_4=$NODE_HASH_4"
echo "  NODE_HASH_5=$NODE_HASH_5"

# Save env file for run_poc.sh
cat > .env.nodes << EOF
NODE_HASH_1=$NODE_HASH_1
NODE_HASH_2=$NODE_HASH_2
NODE_HASH_3=$NODE_HASH_3
NODE_HASH_4=$NODE_HASH_4
NODE_HASH_5=$NODE_HASH_5
EOF
echo ""
echo "Saved: .env.nodes"
echo "Next: ./scripts/run_poc.sh"
