#!/usr/bin/env bash
# setup_env.sh — Install Python dependencies for VoiceFL-MAML.
#
# Handles environments where torch >= 2.4.0 is already installed (e.g. Lightning AI
# studios that ship with torch 2.8.x). In that case torch/torchaudio version pins
# in requirements.txt are skipped to avoid an unintended downgrade.
#
# Usage:
#   ./scripts/setup_env.sh

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python3}

echo "=== VoiceFL-MAML environment setup ==="
echo ""

# ── 1. Detect installed torch version ─────────────────────────────────────────
TORCH_VERSION=$($PYTHON -c "import torch; print(torch.__version__)" 2>/dev/null || echo "none")
echo "Detected torch: $TORCH_VERSION"

# Extract major.minor as integers for comparison
TORCH_MAJOR=$(echo "$TORCH_VERSION" | cut -d. -f1 | tr -d '[:alpha:]')
TORCH_MINOR=$(echo "$TORCH_VERSION" | cut -d. -f2 | tr -d '[:alpha:]+-')
TORCH_MAJOR=${TORCH_MAJOR:-0}
TORCH_MINOR=${TORCH_MINOR:-0}

# requirements.txt pins torch<2.4.0; if the installed version is >= 2.4 skip those lines
SKIP_TORCH=false
if [[ "$TORCH_MAJOR" -gt 2 ]] || [[ "$TORCH_MAJOR" -eq 2 && "$TORCH_MINOR" -ge 4 ]]; then
    SKIP_TORCH=true
fi

# ── 2. Install dependencies ────────────────────────────────────────────────────
echo ""
echo "--- Installing Python dependencies ---"
$PYTHON -m pip install --upgrade pip -q

if $SKIP_TORCH; then
    echo "  torch $TORCH_VERSION >= 2.4 detected — skipping torch/torchaudio pins to avoid downgrade."
    # Filter out torch and torchaudio lines, install everything else
    grep -v "^torch" requirements.txt | $PYTHON -m pip install -q -r /dev/stdin
    # Install torchaudio without a version pin; pip will pick one compatible with installed torch
    $PYTHON -m pip install -q torchaudio
else
    echo "  Installing from requirements.txt (pinned versions)."
    $PYTHON -m pip install -q -r requirements.txt
fi

echo "  Done."

# ── 3. Verify core imports ─────────────────────────────────────────────────────
echo ""
echo "--- Verifying imports ---"
$PYTHON -c "
import sys

checks = [
    ('torch',         lambda m: m.__version__),
    ('torchaudio',    lambda m: m.__version__),
    ('transformers',  lambda m: m.__version__),
    ('datasets',      lambda m: m.__version__),
    ('flwr',          lambda m: m.__version__),
    ('mlflow',        lambda m: m.__version__),
    ('jiwer',         lambda m: m.__version__),
    ('higher',        lambda m: m.__version__),
]

ok = True
for name, ver_fn in checks:
    try:
        import importlib
        m = importlib.import_module(name)
        print(f'  OK    {name:<14} {ver_fn(m)}')
    except Exception as e:
        print(f'  FAIL  {name:<14} {e}', file=sys.stderr)
        ok = False

import torch
print(f'\n  CUDA available: {torch.cuda.is_available()}')
if not ok:
    sys.exit(1)
"

# ── 4. Pre-cache HuggingFace models ───────────────────────────────────────────
echo ""
echo "--- Pre-caching HuggingFace models ---"
echo "  wav2vec2-base-960h (~380 MB) + wav2vec2-base-100h (~380 MB). Instant on repeat runs."
$PYTHON -c "
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
print('  Downloading wav2vec2-base-960h processor...')
Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
print('  Downloading wav2vec2-base-960h model...')
Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h', attn_implementation='eager')
print('  Downloading wav2vec2-base-100h model (Phase 2 backbone)...')
Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-100h', attn_implementation='eager')
print('  Done.')
"

# ── 5. Next steps ──────────────────────────────────────────────────────────────
echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo ""
echo "  STEP 6 — Prepare VCTK meta-test data (downloads ~11 GB VCTK on first run):"
echo "    python data/prepare_vctk_lora.py --splits meta_test --out_dir data/vctk_test_nodes"
echo "    Gate: 4 dirs under data/vctk_test_nodes/, each with features.pt + labels.txt"
echo ""
echo "  STEP 8 — Final evaluation (GPU recommended; CPU works but takes ~90 min):"
echo "    python evaluation/eval_lora.py"
echo "    python evaluation/eval_lora.py --device cpu   # force CPU"
echo "    Gate: evaluation/results/fedlora_maml_vctk.json"
