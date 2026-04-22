#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}

echo "=== VoiceFL-MAML environment setup ==="

echo ""
echo "--- Python dependencies ---"
$PYTHON -m pip install --upgrade pip
$PYTHON -m pip install -r requirements.txt

echo ""
echo "--- Verifying core imports ---"
$PYTHON -c "import torch, transformers, flwr, higher, mlflow; print('  torch:', torch.__version__); print('  cuda:', torch.cuda.is_available())"

echo ""
echo "--- VCTK data preparation ---"
echo "  Run to download and prepare VCTK speaker nodes:"
echo "    python data/prepare_vctk_lora.py"
echo "  Meta-test data (run ONCE at the end only):"
echo "    python data/prepare_vctk_lora.py --splits meta_test"

echo ""
echo "--- Training ---"
echo "  Centralized gate (required before federated):"
echo "    python -u maml/meta_train.py --config configs/vctk_lora_poc.yaml"
echo ""
echo "  Docker training (Lightning AI / 48GB GPU):"
echo "    docker compose -f docker/docker-compose.train.yml up --build"
echo ""
echo "  MLflow UI:"
echo "    mlflow ui --backend-store-uri mlruns"

echo ""
echo "Setup complete."
