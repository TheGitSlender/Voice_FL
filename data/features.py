"""
Feature extraction pipeline.

Reads raw_clips.pkl per node, optionally resamples audio (if a 'sr' field is
present and != TARGET_SR), normalizes to [-1, 1], and saves float32 tensors.
Deletes raw_clips.pkl after saving features.pt (Invariant I1).

For VCTK data, prepare_vctk.py already resamples from 22050Hz to 16kHz before
saving raw_clips.pkl, so no resampling is needed here. The 'sr' field check is
a safety net for any future data source.

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
TARGET_SR = 16_000

def resample_if_needed(audio, source_sr: int, target_sr: int = TARGET_SR):
    """Resample audio from source_sr to target_sr if they differ.

    Args:
        audio: 1D numpy float32 array
        source_sr: sample rate of the input audio
        target_sr: desired output sample rate (default 16kHz)

    Returns:
        numpy float32 array at target_sr
    """
    import numpy as np
    arr = np.array(audio, dtype=np.float32)
    if source_sr == target_sr:
        return arr
    import torchaudio
    import torch
    resampler = torchaudio.transforms.Resample(orig_freq=source_sr, new_freq=target_sr)
    t = torch.tensor(arr).unsqueeze(0)
    return resampler(t).squeeze(0).numpy().astype(np.float32)

def normalize_audio(audio, source_sr: int = TARGET_SR) -> "torch.Tensor":
    """Resample if needed, normalize float32 numpy array to [-1, 1], return 1D tensor."""
    import torch
    import numpy as np
    arr = resample_if_needed(audio, source_sr)
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
                                                                                          
        source_sr = clip.get("sr", TARGET_SR)
        t = normalize_audio(clip["audio"], source_sr=source_sr)
        tensors.append(t)
        labels.append(clip["text"])

    torch.save(tensors, features_path)
    labels_path.write_text("\n".join(labels))
    print(f"  {node_dir.name}: saved {len(tensors)} tensors → features.pt, labels.txt")

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
    import argparse
    try:
        import torch
        import numpy as np              
    except ImportError:
        print("ERROR: required packages not installed.")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes_dir", default=None,
                        help="Override nodes directory (default: data/nodes/)")
    args = parser.parse_args()

    nodes_dir = Path(args.nodes_dir) if args.nodes_dir else NODES_DIR

    node_dirs = sorted([d for d in nodes_dir.iterdir() if d.is_dir()])
    if not node_dirs:
        print(f"ERROR: no node directories found in {nodes_dir}")
        print("Run: python data/prepare_vctk.py")
        sys.exit(1)

    print(f"Processing {len(node_dirs)} nodes in {nodes_dir}...")
    for node_dir in node_dirs:
        process_node(node_dir)

    print("\nValidating feature tensors...")
    for node_dir in node_dirs:
        validate_features(node_dir)

    pkl_files = list(nodes_dir.rglob("*.pkl"))
    if pkl_files:
        print(f"INVARIANT VIOLATION I1: found .pkl files: {pkl_files}")
        sys.exit(1)
    print(f"\nI1 check: PASSED — no .pkl files in {nodes_dir}")
    print("Next: python data/insights/run_insights.py  OR  python maml/meta_train.py")

if __name__ == "__main__":
    main()
