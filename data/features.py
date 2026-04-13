"""
data/features.py — S2b: Raw Waveform Normalization (Wav2Vec2-compatible)

Converts per-node raw audio clips (raw_clips.pkl) to normalized float32
waveform tensors and permanently deletes the raw waveforms.

Wav2Vec2 takes raw waveforms directly — NOT log-mel spectrograms.
The Wav2Vec2Processor handles any required preprocessing internally during
model forward passes.

Feature format:
  dtype      = torch.float32
  shape      = (T_samples,)  1D, where T_samples = clip_duration_s * 16000
  values     = [-1, 1]  (peak-normalized per clip)
  sample_rate = 16000 Hz (LibriSpeech native — no resampling needed)

Output per node:
  data/nodes/node_XXX/features.pt  — list of (T_i,) float32 tensors
  data/nodes/node_XXX/labels.txt   — UPPERCASE transcriptions, one per line

Deletes:
  data/nodes/node_XXX/raw_clips.pkl  — MUST be deleted; verified before next node

Privacy note (I1): after this script completes, no waveform data remains
on disk anywhere under data/nodes/. The model will compute its own internal
representations (CNN feature extractor + transformer encoder) from these
normalized tensors at training time.

Run:
    python data/features.py
Requires:
    data/nodes/node_XXX/raw_clips.pkl  (produced by pii_masking.py)
    data/cleaning_config.json
"""

import json
import os
import pickle
from typing import List

import numpy as np
import torch
from tqdm import tqdm

SAMPLE_RATE = 16000
NODES_DIR = "data/nodes"
CONFIG_FILE = "data/cleaning_config.json"


def load_cleaning_config() -> dict:
    if not os.path.exists(CONFIG_FILE):
        return {"normalize_audio": True}
    with open(CONFIG_FILE) as f:
        return json.load(f)


def normalize_waveform(audio: np.ndarray) -> np.ndarray:
    """Peak-normalize audio to [-1, 1]. Returns float32."""
    audio = audio.astype(np.float32)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / (peak + 1e-8)
    return audio


def process_node(node_id: str, normalize: bool) -> dict:
    """
    Load raw_clips.pkl, save raw waveform tensors, delete pkl.

    Each clip is saved as a 1D (T_samples,) float32 tensor.
    Labels are saved as UPPERCASE text (Wav2Vec2 trained on uppercase).
    raw_clips.pkl is permanently deleted after saving — verified with assert.

    Returns per-node statistics dict.
    """
    node_dir = os.path.join(NODES_DIR, node_id)
    pkl_path = os.path.join(node_dir, "raw_clips.pkl")
    feat_path = os.path.join(node_dir, "features.pt")
    label_path = os.path.join(node_dir, "labels.txt")

    if not os.path.exists(pkl_path):
        raise FileNotFoundError(
            f"{pkl_path} not found. Run: python data/pii_masking.py first."
        )

    with open(pkl_path, "rb") as f:
        clips = pickle.load(f)

    features: List[torch.Tensor] = []
    labels: List[str] = []
    duration_values: List[float] = []

    for clip in tqdm(clips, desc=f"{node_id}", leave=False):
        audio: np.ndarray = clip["audio"]
        text: str = clip["text"]

        if normalize:
            audio = normalize_waveform(audio)
        else:
            audio = audio.astype(np.float32)

        # Wav2Vec2 expects raw waveform as 1D float32 tensor at 16kHz
        tensor = torch.tensor(audio, dtype=torch.float32)  # shape: (T_samples,)
        features.append(tensor)
        labels.append(text.upper())  # Wav2Vec2 trained on uppercase
        duration_values.append(len(audio) / SAMPLE_RATE)

    # Save variable-length list — do NOT pad or stack
    torch.save(features, feat_path)

    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(labels))

    # I1: Delete raw waveform file — non-negotiable privacy step
    os.remove(pkl_path)
    assert not os.path.exists(pkl_path), f"BUG: failed to delete {pkl_path}"

    return {
        "node_id": node_id,
        "n_clips": len(clips),
        "dur_min_s": round(min(duration_values), 2),
        "dur_max_s": round(max(duration_values), 2),
        "dur_mean_s": round(sum(duration_values) / len(duration_values), 2),
    }


def verify_node(node_id: str) -> None:
    """Verify saved tensors are 1D float32 and pkl is gone."""
    node_dir = os.path.join(NODES_DIR, node_id)
    feat_path = os.path.join(node_dir, "features.pt")
    label_path = os.path.join(node_dir, "labels.txt")
    pkl_path = os.path.join(node_dir, "raw_clips.pkl")

    features = torch.load(feat_path, weights_only=True)
    assert isinstance(features, list), "features.pt must be a list"
    assert len(features) > 0, "Empty features list"

    for i, t in enumerate(features):
        assert isinstance(t, torch.Tensor), f"clip {i}: not a tensor"
        assert t.dtype == torch.float32, f"clip {i}: wrong dtype {t.dtype}"
        assert t.ndim == 1, f"clip {i}: wrong ndim {t.ndim} (expected 1D)"
        assert t.shape[0] > 0, f"clip {i}: empty waveform"

    with open(label_path, encoding="utf-8") as f:
        label_lines = [ln for ln in f.read().strip().split("\n") if ln]
    assert len(label_lines) == len(features), (
        f"Label count {len(label_lines)} != feature count {len(features)}"
    )

    assert not os.path.exists(pkl_path), (
        f"BUG: raw_clips.pkl still exists at {pkl_path}"
    )


def final_verification(node_ids: List[str]) -> None:
    print("=" * 60)
    print("Final verification — no .pkl files remain")
    print("=" * 60)

    for root, _, files in os.walk(NODES_DIR):
        for fname in files:
            if fname.endswith(".pkl"):
                raise RuntimeError(
                    f"BUG: .pkl file found: {os.path.join(root, fname)}"
                )

    total_clips = 0
    total_duration_s = 0.0

    print(f"{'Node':<12} {'Clips':>7} {'Dur min':>9} {'Dur max':>9} {'Dur mean':>10}")
    print("-" * 52)

    for node_id in sorted(node_ids):
        node_dir = os.path.join(NODES_DIR, node_id)
        feat_path = os.path.join(node_dir, "features.pt")
        meta_path = os.path.join(node_dir, "metadata.json")

        features = torch.load(feat_path, weights_only=True)
        durations = [t.shape[0] / SAMPLE_RATE for t in features]

        with open(meta_path) as f:
            meta = json.load(f)

        total_clips += len(features)
        total_duration_s += meta.get("total_duration_s", sum(durations))

        print(
            f"{node_id:<12} {len(features):>7} "
            f"{min(durations):>8.1f}s "
            f"{max(durations):>8.1f}s "
            f"{sum(durations)/len(durations):>9.1f}s"
        )

    print()
    print(f"No .pkl files remain under {NODES_DIR}/  ✓")
    print(f"Feature format: raw float32 waveform, variable length, 16kHz  ✓")
    print(f"Total clips: {total_clips:,}")
    print(f"Total audio: {total_duration_s / 3600:.2f} hours")


def main() -> None:
    print()
    print("VoiceFL-MAML — Phase 1 / S2b: Raw Waveform Normalization")
    print("(Wav2Vec2 takes raw audio — no mel spectrograms)")
    print()

    cfg = load_cleaning_config()
    normalize = cfg.get("normalize_audio", True)
    print(f"normalize_audio: {normalize}")
    print()

    if not os.path.isdir(NODES_DIR):
        raise RuntimeError(
            f"{NODES_DIR} not found. Run: python data/pii_masking.py first."
        )

    node_dirs = sorted([
        d for d in os.listdir(NODES_DIR)
        if os.path.isdir(os.path.join(NODES_DIR, d))
        and os.path.exists(os.path.join(NODES_DIR, d, "raw_clips.pkl"))
    ])

    if not node_dirs:
        raise RuntimeError(
            f"No raw_clips.pkl files found in {NODES_DIR}. "
            "Run: python data/pii_masking.py first."
        )

    print(f"Found {len(node_dirs)} nodes to process: {', '.join(node_dirs)}")
    print()
    print("=" * 60)
    print("Processing nodes (load → normalize → save 1D tensors → delete pkl)")
    print("=" * 60)

    stats = []
    for node_id in node_dirs:
        node_stats = process_node(node_id, normalize=normalize)
        verify_node(node_id)
        stats.append(node_stats)
        print(
            f"  {node_id}: {node_stats['n_clips']} clips, "
            f"dur ∈ [{node_stats['dur_min_s']}, {node_stats['dur_max_s']}]s, "
            f"pkl deleted ✓"
        )

    print()
    final_verification(node_dirs)
    print()
    print("=" * 60)
    print("Feature extraction complete. Raw audio fully deleted.")
    print("Next step: python data/partition.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
