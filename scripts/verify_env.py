"""
scripts/verify_env.py — Environment validation for VoiceFL-MAML

Checks all required packages for the Per-FedAvg MAML pipeline.

Note: Opacus is intentionally NOT required. It is incompatible with
the `higher` library used for second-order MAML (track_higher_grads=True
wraps model in a functional form that Opacus cannot hook into). DP is
implemented manually in privacy/dp_meta.py.

Note: learn2learn cannot be installed on Python 3.13+ due to a removed
C header (longintrepr.h). FOMAML is implemented natively in maml/engine.py
using PyTorch autograd with create_graph=False (first-order approximation).

Run:
    python scripts/verify_env.py
"""

import sys

print(f"Python:       {sys.version.split()[0]}")

import torch
print(f"PyTorch:      {torch.__version__}")
print(f"CUDA:         {torch.cuda.is_available()} — {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print(f"VRAM free:    {free / 1e9:.1f} GB / {total / 1e9:.1f} GB")

import higher
_higher_ver = getattr(higher, "__version__", "installed")
print(f"higher:       {_higher_ver}")

try:
    from autodp import rdp_acct
    import autodp
    _autodp_ver = getattr(autodp, "__version__", "installed")
    print(f"autodp:       {_autodp_ver}  (RDP accountant available)")
except ImportError as e:
    print(f"autodp:       MISSING — {e}")

import flwr
print(f"Flower:       {flwr.__version__}")

import transformers
print(f"Transformers: {transformers.__version__}")

import datasets
print(f"Datasets:     {datasets.__version__}")

import librosa
print(f"Librosa:      {librosa.__version__}")

import mlflow
print(f"MLflow:       {mlflow.__version__}")

import sklearn
print(f"Sklearn:      {sklearn.__version__}")

import jiwer
_jiwer_ver = getattr(jiwer, "__version__", "installed")
print(f"jiwer:        {_jiwer_ver}")

import soundfile
print(f"soundfile:    {soundfile.__version__}")

import yaml
print(f"PyYAML:       {yaml.__version__}")

print()
print("Intentionally absent (incompatible with higher):")
try:
    import opacus
    print(f"  Opacus {opacus.__version__} — WARNING: installed but should not be used with higher")
except ImportError:
    print("  Opacus — not installed (correct)")

print()
print("learn2learn status:")
try:
    import learn2learn
    print(f"  learn2learn {learn2learn.__version__} — available (native FOMAML in engine.py used instead)")
except ImportError:
    print("  learn2learn — not installed (Python 3.13 incompatibility; native FOMAML used in engine.py)")

print()
print("All critical dependencies verified — Phase 0 environment ready.")
