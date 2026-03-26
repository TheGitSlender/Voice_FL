"""
data/features.py — S2b: Log-Mel Feature Extraction

Converts per-node raw audio clips (raw_clips.pkl) to log-mel filterbank
features and permanently deletes the raw waveforms.

This is the critical privacy step: after this file runs, no waveform data
remains on disk anywhere under data/nodes/.

Feature format (fixed for entire project):
  n_mels     = 80
  n_fft      = 400   (25ms window at 16 kHz)
  hop_length = 160   (10ms hop at 16 kHz)
  f_min      = 0.0
  f_max      = 8000.0
  dtype      = torch.float32
  shape      = (T, 80)  where T varies by clip duration

Output per node:
  data/nodes/node_XXX/features.pt  — list of (T_i, 80) float32 tensors
  data/nodes/node_XXX/labels.txt   — transcriptions, one per line

Deletes:
  data/nodes/node_XXX/raw_clips.pkl  — MUST be deleted; verified before next node

Run:
    python data/features.py
Requires:
    data/nodes/node_XXX/raw_clips.pkl  (produced by pii_masking.py)
    data/cleaning_config.json
"""

import json
import math
import os
import pickle

import librosa
import numpy as np
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Feature extraction parameters — FIXED, do not change
# ---------------------------------------------------------------------------
N_MELS      = 80
N_FFT       = 400    # 25ms at 16kHz
HOP_LENGTH  = 160    # 10ms at 16kHz
SAMPLE_RATE = 16000
F_MIN       = 0.0
F_MAX       = 8000.0

NODES_DIR   = "data/nodes"
CONFIG_FILE = "data/cleaning_config.json"


# ---------------------------------------------------------------------------
# Load cleaning config
# ---------------------------------------------------------------------------
def load_cleaning_config() -> dict:
    if not os.path.exists(CONFIG_FILE):
        return {"normalize_audio": True}
    with open(CONFIG_FILE) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Step 2 — Feature extraction per clip
# ---------------------------------------------------------------------------
def extract_log_mel(audio: np.ndarray, normalize: bool) -> torch.Tensor:
    """
    Extract log-mel filterbank features from a raw audio array.

    Args:
        audio:     float32 numpy array, expected at 16 kHz
        normalize: if True, normalize audio to [-1, 1] before extraction

    Returns:
        torch.float32 tensor of shape (T, 80)
    """
    audio = audio.astype(np.float32)

    # Resample if needed (LibriSpeech is always 16kHz, but be safe)
    # librosa.resample is only called if sampling rate differs — the
    # raw_clips.pkl only stores arrays, not sampling rate. We trust the
    # dataset guarantee: LibriSpeech is uniformly 16 kHz.

    if normalize:
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / (max_val + 1e-8)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        fmin=F_MIN,
        fmax=F_MAX,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)  # shape: (80, T)
    log_mel = log_mel.T                              # shape: (T, 80) — time-first
    return torch.tensor(log_mel, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Step 1+3+4+5 — Process one node
# ---------------------------------------------------------------------------
def process_node(node_id: str, normalize: bool) -> dict:
    """
    Load raw_clips.pkl, extract features, save outputs, delete pkl.
    Returns per-node statistics dict.
    """
    node_dir   = os.path.join(NODES_DIR, node_id)
    pkl_path   = os.path.join(node_dir, "raw_clips.pkl")
    feat_path  = os.path.join(node_dir, "features.pt")
    label_path = os.path.join(node_dir, "labels.txt")

    if not os.path.exists(pkl_path):
        raise FileNotFoundError(
            f"{pkl_path} not found. Run: python data/pii_masking.py first."
        )

    # Load raw clips
    with open(pkl_path, "rb") as f:
        clips = pickle.load(f)

    n_clips   = len(clips)
    features  = []
    labels    = []
    T_values  = []

    for clip in tqdm(clips, desc=f"{node_id}", leave=False):
        audio = clip["audio"]
        text  = clip["text"]

        tensor = extract_log_mel(audio, normalize=normalize)
        features.append(tensor)
        labels.append(text)
        T_values.append(tensor.shape[0])

    # Step 3 — Save features.pt (list of variable-length tensors — do NOT pad/stack)
    torch.save(features, feat_path)

    # Save labels.txt (one transcription per line, same order as features)
    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(labels))

    # Step 4 — Delete raw waveform file (NON-NEGOTIABLE)
    os.remove(pkl_path)
    assert not os.path.exists(pkl_path), f"BUG: failed to delete {pkl_path}"

    return {
        "node_id":  node_id,
        "n_clips":  n_clips,
        "T_min":    min(T_values),
        "T_max":    max(T_values),
        "T_mean":   sum(T_values) / len(T_values),
    }


# ---------------------------------------------------------------------------
# Step 5 — Verify one node's output
# ---------------------------------------------------------------------------
def verify_node(node_id: str) -> None:
    node_dir   = os.path.join(NODES_DIR, node_id)
    feat_path  = os.path.join(node_dir, "features.pt")
    label_path = os.path.join(node_dir, "labels.txt")
    pkl_path   = os.path.join(node_dir, "raw_clips.pkl")

    features = torch.load(feat_path, weights_only=True)
    assert isinstance(features, list),   "features.pt must be a list"
    assert len(features) > 0,            "Empty features list"

    for i, t in enumerate(features):
        assert isinstance(t, torch.Tensor),     f"clip {i}: not a tensor"
        assert t.dtype == torch.float32,         f"clip {i}: wrong dtype {t.dtype}"
        assert t.ndim == 2,                      f"clip {i}: wrong ndim {t.ndim}"
        assert t.shape[1] == N_MELS,             f"clip {i}: wrong mel bins {t.shape[1]}"

    with open(label_path, encoding="utf-8") as f:
        label_lines = f.read().strip().split("\n")
    assert len(label_lines) == len(features), (
        f"Label count {len(label_lines)} != feature count {len(features)}"
    )

    assert not os.path.exists(pkl_path), f"BUG: raw_clips.pkl still exists at {pkl_path}"


# ---------------------------------------------------------------------------
# Step 6 — Final verification: no pkl files remain
# ---------------------------------------------------------------------------
def final_verification(node_ids: list[str]) -> tuple[int, float]:
    print("=" * 60)
    print("STEP 6 — Final verification")
    print("=" * 60)

    # Check no .pkl files anywhere under data/nodes/
    for root, _, files in os.walk(NODES_DIR):
        for fname in files:
            if fname.endswith(".pkl"):
                raise RuntimeError(
                    f"BUG: .pkl file found after extraction: {os.path.join(root, fname)}"
                )

    total_clips    = 0
    total_duration = 0.0

    print(f"{'Node':<12} {'Clips':>7} {'T min':>7} {'T max':>7} {'T mean':>8}")
    print("-" * 45)

    for node_id in sorted(node_ids):
        node_dir   = os.path.join(NODES_DIR, node_id)
        feat_path  = os.path.join(node_dir, "features.pt")
        meta_path  = os.path.join(node_dir, "metadata.json")

        features = torch.load(feat_path, weights_only=True)
        T_values = [t.shape[0] for t in features]

        with open(meta_path) as f:
            meta = json.load(f)

        total_clips    += len(features)
        total_duration += meta.get("total_duration_s", 0.0)

        print(
            f"{node_id:<12} {len(features):>7} "
            f"{min(T_values):>7} {max(T_values):>7} "
            f"{sum(T_values)/len(T_values):>8.1f}"
        )

    print()
    print(f"No .pkl files remain under {NODES_DIR}/  ✓")
    print(f"Total clips processed: {total_clips:,}")
    print(f"Total audio processed: {total_duration/3600:.2f} hours")
    return total_clips, total_duration


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print()
    print("VoiceFL — Phase 1 / S2b: Log-Mel Feature Extraction")
    print()

    cfg       = load_cleaning_config()
    normalize = cfg.get("normalize_audio", True)
    print(f"normalize_audio: {normalize}")
    print()

    # Discover nodes by looking for raw_clips.pkl
    if not os.path.isdir(NODES_DIR):
        raise RuntimeError(f"{NODES_DIR} not found. Run: python data/pii_masking.py first.")

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
    print("STEP 1–5 — Processing nodes (load → extract → save → delete pkl)")
    print("=" * 60)

    stats = []
    for node_id in node_dirs:
        node_stats = process_node(node_id, normalize=normalize)
        verify_node(node_id)
        stats.append(node_stats)
        print(
            f"  {node_id}: {node_stats['n_clips']} clips, "
            f"T ∈ [{node_stats['T_min']}, {node_stats['T_max']}], pkl deleted ✓"
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
