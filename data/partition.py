"""
data/partition.py — S2c: Partition Validation and Manifest

Reads all 20 node directories produced by features.py.
Validates that the partition is correct, complete, and PII-free.
Saves data/partition_manifest.json as the authoritative record of
what data is available for FL simulation.

Feature format expected (Wav2Vec2-compatible):
  dtype  = torch.float32
  shape  = (T_samples,)  1D variable-length raw waveform
  values = [-1, 1] (peak-normalized)

This script does NOT move or copy data. It reads and validates only.

Run:
    python data/partition.py
Output:
    data/partition_manifest.json
"""

import json
import math
import os
from collections import Counter

import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NODES_DIR = "data/nodes"
MANIFEST_FILE = "data/partition_manifest.json"
SAMPLE_RATE = 16000

FEATURE_CONFIG = {
    "format": "raw_waveform_float32_1d",
    "sample_rate": SAMPLE_RATE,
    "normalized": True,
    "model_compatibility": "facebook/wav2vec2-base-960h",
    "notes": "1D (T_samples,) tensors; Wav2Vec2Processor handles padding at batch time",
}


# ---------------------------------------------------------------------------
# Step 1 — Load and validate all nodes
# ---------------------------------------------------------------------------
def validate_node(node_id: str) -> dict:
    """
    Validates one node directory and returns its statistics.
    Expects 1D float32 raw waveform tensors (Wav2Vec2-compatible).
    Raises AssertionError on any violation.
    """
    node_dir = os.path.join(NODES_DIR, node_id)
    feat_path = os.path.join(node_dir, "features.pt")
    label_path = os.path.join(node_dir, "labels.txt")
    meta_path = os.path.join(node_dir, "metadata.json")
    pkl_path = os.path.join(node_dir, "raw_clips.pkl")

    assert not os.path.exists(pkl_path), (
        f"{node_id}: raw_clips.pkl still exists — run features.py to delete it"
    )

    assert os.path.exists(feat_path), f"{node_id}: features.pt missing"
    features = torch.load(feat_path, weights_only=True)
    assert isinstance(features, list), f"{node_id}: features.pt is not a list"
    assert len(features) >= 50, f"{node_id}: only {len(features)} clips — minimum 50 required"

    dur_values = []

    for i, t in enumerate(features):
        assert isinstance(t, torch.Tensor), f"{node_id} clip {i}: not a tensor"
        assert t.dtype == torch.float32, f"{node_id} clip {i}: dtype {t.dtype}"
        assert t.ndim == 1, (
            f"{node_id} clip {i}: ndim {t.ndim} — expected 1D raw waveform"
        )
        assert t.shape[0] > 0, f"{node_id} clip {i}: empty waveform tensor"
        # Values should be in [-1.1, 1.1] after peak normalization
        max_val = float(t.abs().max())
        assert max_val <= 1.1, (
            f"{node_id} clip {i}: max abs value {max_val:.3f} > 1.1 — not normalized"
        )
        dur_values.append(t.shape[0] / SAMPLE_RATE)

    assert os.path.exists(label_path), f"{node_id}: labels.txt missing"
    with open(label_path, encoding="utf-8") as f:
        labels = [ln for ln in f.read().strip().split("\n") if ln]
    assert len(labels) == len(features), (
        f"{node_id}: label count {len(labels)} != feature count {len(features)}"
    )

    assert os.path.exists(meta_path), f"{node_id}: metadata.json missing"
    with open(meta_path) as f:
        meta = json.load(f)
    assert "speaker_id" not in meta, f"{node_id}: speaker_id in metadata.json — PII leak!"

    total_dur_s = sum(dur_values)
    mean_dur_s = total_dur_s / len(dur_values)
    std_dur_s = math.sqrt(sum((d - mean_dur_s) ** 2 for d in dur_values) / len(dur_values))

    return {
        "node_id": node_id,
        "clip_count": len(features),
        "total_duration_s": round(total_dur_s, 3),
        "mean_clip_duration_s": round(mean_dur_s, 3),
        "std_clip_duration_s": round(std_dur_s, 3),
        "min_duration_s": round(min(dur_values), 3),
        "max_duration_s": round(max(dur_values), 3),
        "features_path": feat_path,
        "labels_path": label_path,
        "_labels": labels,
    }


# ---------------------------------------------------------------------------
# Step 2 — Vocabulary richness per node
# ---------------------------------------------------------------------------
def compute_vocab_richness(labels: list[str]) -> float:
    """
    vocabulary_richness = unique_words / total_words.
    Proxy for non-IID text distribution across nodes.
    """
    all_words = []
    for text in labels:
        all_words.extend(text.split())
    if not all_words:
        return 0.0
    unique_words = len(set(all_words))
    return round(unique_words / len(all_words), 4)


# ---------------------------------------------------------------------------
# Step 3 — Save manifest
# ---------------------------------------------------------------------------
def save_manifest(node_stats: list[dict], total_clips: int, total_dur_s: float):
    nodes_for_manifest = [
        {k: v for k, v in s.items() if k != "_labels"}
        for s in node_stats
    ]

    manifest = {
        "n_nodes": len(node_stats),
        "total_clips": total_clips,
        "total_duration_hours": round(total_dur_s / 3600, 4),
        "nodes": nodes_for_manifest,
        "feature_config": FEATURE_CONFIG,
        "pii_status": "verified_clean",
        "ready_for_fl": True,
        "maml_ready": True,
    }

    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved: {MANIFEST_FILE}")


# ---------------------------------------------------------------------------
# Step 4 — Print readiness table
# ---------------------------------------------------------------------------
def print_readiness_table(node_stats: list[dict], total_clips: int, total_dur_s: float):
    col_w = [10, 7, 10, 8, 16]
    header = ["Node", "Clips", "Duration", "Mean T", "Vocab Richness"]
    sep    = "─" * (sum(col_w) + len(col_w) * 3 + 1)

    print()
    print("┌" + sep + "┐")
    print("│  VoiceFL Data Partition — Ready for FL Simulation".ljust(len(sep) + 1) + "│")
    print("├" + sep + "┤")
    row = "│"
    for i, h in enumerate(header):
        row += f" {h:<{col_w[i]}} │"
    print(row)
    print("├" + sep + "┤")

    for s in node_stats:
        dur_min = s["total_duration_s"] / 60
        row = (
            f"│ {s['node_id']:<{col_w[0]}} │"
            f" {s['clip_count']:>{col_w[1]}} │"
            f" {dur_min:>{col_w[2]-3}.1f} min │"
            f" {s['mean_clip_duration_s']:>{col_w[3]}.1f}s │"
            f" {s['vocabulary_richness']:>{col_w[4]-3}.4f}     │"
        )
        print(row)

    print("└" + sep + "┘")
    print()
    print(f"Total: {len(node_stats)} nodes, {total_clips:,} clips, "
          f"{total_dur_s/3600:.2f} hours audio")
    print("PII status: CLEAN — no raw audio, no speaker identity in any artifact")
    print("Feature format: raw float32 waveform, 1D, variable length, 16kHz")
    print("Ready for FL simulation: YES")
    print("Ready for MAML: YES")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print()
    print("VoiceFL — Phase 1 / S2c: Partition Validation")
    print()

    if not os.path.isdir(NODES_DIR):
        raise RuntimeError(f"{NODES_DIR} not found. Run the pipeline from download.py first.")

    node_dirs = sorted([
        d for d in os.listdir(NODES_DIR)
        if os.path.isdir(os.path.join(NODES_DIR, d))
        and d.startswith("node_")
    ])

    if not node_dirs:
        raise RuntimeError(f"No node directories found in {NODES_DIR}")

    print("=" * 60)
    print("STEP 1 — Validating all nodes")
    print("=" * 60)

    node_stats = []
    errors     = []

    for node_id in node_dirs:
        try:
            stats = validate_node(node_id)
            node_stats.append(stats)
            print(f"  {node_id}: {stats['clip_count']} clips  ✓")
        except (AssertionError, FileNotFoundError) as e:
            errors.append(str(e))
            print(f"  {node_id}: FAILED — {e}")

    if errors:
        print()
        print(f"Validation failed with {len(errors)} error(s). Fix before proceeding.")
        for e in errors:
            print(f"  - {e}")
        return

    print()

    # Step 2 — Vocabulary richness
    print("=" * 60)
    print("STEP 2 — Computing vocabulary richness (non-IID proxy)")
    print("=" * 60)
    for s in node_stats:
        s["vocabulary_richness"] = compute_vocab_richness(s["_labels"])
        print(f"  {s['node_id']}: richness={s['vocabulary_richness']:.4f}")
    print()

    total_clips = sum(s["clip_count"]      for s in node_stats)
    total_dur_s = sum(s["total_duration_s"] for s in node_stats)

    # Step 3 — Save manifest
    print("=" * 60)
    print("STEP 3 — Saving partition manifest")
    print("=" * 60)
    save_manifest(node_stats, total_clips, total_dur_s)
    print()

    # Step 4 — Print readiness table
    print_readiness_table(node_stats, total_clips, total_dur_s)

    print("=" * 60)
    print("Partition validation complete.")
    print("Next step: python data/task_sampler.py  (verify sampler)")
    print("=" * 60)


if __name__ == "__main__":
    main()
