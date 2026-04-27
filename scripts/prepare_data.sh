#!/usr/bin/env bash
# Prepare data pipeline (Steps 3–4)
# Run from the repo root: ./scripts/prepare_data.sh

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== VoiceFL-MAML PoC — Data Preparation ==="
echo ""

# Step 0: Install pinned requirements
# This is mandatory — the voicefl env may have incompatible versions pre-installed.
# Specifically: transformers>=4.39 calls torch.utils._pytree.register_pytree_node
# which was removed in torch 2.5+. Pin to <4.39 to avoid this.
echo "[STEP 0] Installing pinned requirements..."
pip install -q -r requirements.txt
echo "  Done."
echo ""

# Step 1: Environment check
echo "[STEP 1] Checking environment..."
python -c "
import torch, transformers, datasets, flwr, jiwer
import importlib.metadata
print(f'  torch:        {torch.__version__}')
print(f'  transformers: {transformers.__version__}')
print(f'  flwr:         {flwr.__version__}')
print(f'  jiwer:        {importlib.metadata.version(\"jiwer\")}')
print('  OK — all packages importable')
"
echo ""

# Step 2: Download and speaker selection
echo "[STEP 2] Speaker selection..."
python data/download.py
echo ""

# Step 3: PII masking
echo "[STEP 3] PII masking..."
python data/pii_masking.py
echo ""

# Step 4: Feature extraction
echo "[STEP 4] Feature extraction..."
python data/features.py
echo ""

# Step 5: Task sampler smoke test
echo "[STEP 5] Task sampler smoke test..."
python data/task_sampler.py
echo ""

echo "=== Data pipeline complete ==="
echo ""
echo "Node directories:"
ls -la data/nodes/
echo ""
echo "Next: python maml/meta_train.py --config configs/poc.yaml"
