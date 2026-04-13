"""
Feature extraction pipeline.

Reads raw_clips.pkl per node, normalizes audio, saves float32 tensors.
Deletes raw_clips.pkl after saving features.pt (Invariant I1).

Output per node:
  data/nodes/{node_hash}/features.pt  — list of 1D float32 tensors (T_samples,)
  data/nodes/{node_hash}/labels.txt   — one uppercase transcription per line

Invariant I1: no .pkl files under data/nodes/ after this script completes.
"""

import os
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent
NODES_DIR = DATA_DIR / "nodes"


def normalize_audio(audio) -> "torch.Tensor":
    """Normalize float32 numpy array to [-1, 1] and return as 1D torch tensor."""
    import torch
    import numpy as np
    arr = np.array(audio, dtype=np.float32)
    max_val = np.abs(arr).max()
    normalized = arr / (max_val + 1e-8)
    return torch.tensor(normalized, dtype=torch.float32)


def process_node(node_dir: Path) -> None:
    import pickle
    import torch

    pkl_path = node_dir / "raw_clips.pkl"
    features_path = node_dir / "features.pt"
    labels_path = node_dir / "labels.txt"

    if not pkl_path.exists():
        if features_path.exists():
            print(f"  {node_dir.name}: already processed, skipping")
            return
        print(f"  {node_dir.name}: ERROR — no raw_clips.pkl and no features.pt")
        sys.exit(1)

    print(f"  {node_dir.name}: loading clips...")
    with open(pkl_path, "rb") as f:
        clips = pickle.load(f)

    tensors = []
    labels = []
    for clip in clips:
        t = normalize_audio(clip["audio"])
        tensors.append(t)
        labels.append(clip["text"])

    torch.save(tensors, features_path)
    labels_path.write_text("\n".join(labels))
    print(f"  {node_dir.name}: saved {len(tensors)} tensors → features.pt, labels.txt")

    # Delete raw_clips.pkl (Invariant I1)
    pkl_path.unlink()
    assert not pkl_path.exists(), f"INVARIANT VIOLATION I1: {pkl_path} still exists"
    print(f"  {node_dir.name}: deleted raw_clips.pkl [I1 OK]")


def validate_features(node_dir: Path) -> None:
    """Sanity-check that features.pt is a list of 1D float32 tensors."""
    import torch
    features_path = node_dir / "features.pt"
    tensors = torch.load(features_path, weights_only=True)
    assert isinstance(tensors, list), "features.pt must be a list"
    for i, t in enumerate(tensors[:5]):
        assert t.dtype == torch.float32, f"tensor {i} dtype {t.dtype} != float32"
        assert t.dim() == 1, f"tensor {i} is not 1D"
        assert t.abs().max() <= 1.0 + 1e-5, f"tensor {i} not in [-1,1]"
    print(f"  {node_dir.name}: validation OK ({len(tensors)} tensors)")


def main():
    try:
        import torch
        import numpy as np
    except ImportError:
        print("ERROR: required packages not installed.")
        sys.exit(1)

    node_dirs = sorted([d for d in NODES_DIR.iterdir() if d.is_dir()])
    if not node_dirs:
        print("ERROR: no node directories found. Run: python data/pii_masking.py")
        sys.exit(1)

    print(f"Processing {len(node_dirs)} nodes...")
    for node_dir in node_dirs:
        process_node(node_dir)

    print("\nValidating feature tensors...")
    for node_dir in node_dirs:
        validate_features(node_dir)

    # Invariant I1 final check
    pkl_files = list(NODES_DIR.rglob("*.pkl"))
    if pkl_files:
        print(f"INVARIANT VIOLATION I1: found .pkl files: {pkl_files}")
        sys.exit(1)
    print("\nI1 check: PASSED — no .pkl files in data/nodes/")
    print("Run: python data/task_sampler.py (or proceed to model step)")


if __name__ == "__main__":
    main()
