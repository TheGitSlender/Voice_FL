import sys
import torch
import flwr
import opacus
import transformers
import datasets
import librosa
import mlflow
import sklearn

print(f"Python:       {sys.version.split()[0]}")
print(f"PyTorch:      {torch.__version__}")
print(f"CUDA:         {torch.cuda.is_available()} — {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print(f"VRAM free:    {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB / {torch.cuda.mem_get_info()[1] / 1e9:.1f} GB")
print(f"Flower:       {flwr.__version__}")
print(f"Opacus:       {opacus.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"Datasets:     {datasets.__version__}")
print(f"Librosa:      {librosa.__version__}")
print(f"MLflow:       {mlflow.__version__}")
print(f"Sklearn:      {sklearn.__version__}")
print()
print("All good — Phase 0 complete.")
