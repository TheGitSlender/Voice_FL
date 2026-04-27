"""
Download LibriSpeech train-clean-100 and select 5 speakers with low heterogeneity.

Selection criteria (priority order):
  1. Between 100 and 130 clips per speaker
  2. duration_std between 1.0s and 2.0s
  3. Similar data quantities (low inter-speaker variance)
  4. No speaker with fewer than 80 clips

Output: data/speaker_selection.json
"""

import json
import os
import hashlib
import secrets
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

DATA_DIR = Path(__file__).parent
SALT_FILE = DATA_DIR / "salt.txt"
SELECTION_FILE = DATA_DIR / "speaker_selection.json"

TARGET_SPEAKERS = 5
MIN_CLIPS = 80
PREFERRED_MIN_CLIPS = 100
PREFERRED_MAX_CLIPS = 130
PREFERRED_DUR_STD_MIN = 1.0
PREFERRED_DUR_STD_MAX = 2.0
SAMPLE_RATE = 16000

def load_salt() -> str:
    if SALT_FILE.exists():
        return SALT_FILE.read_text().strip()
    salt = secrets.token_hex(32)
    SALT_FILE.write_text(salt)
    return salt

def hash_speaker(speaker_id: int, salt: str) -> str:
    raw = f"{salt}:{speaker_id}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]

def compute_speaker_stats(dataset) -> dict:
    """Compute per-speaker clip count and duration statistics."""
    stats: dict[int, list[float]] = defaultdict(list)
    print("Computing per-speaker statistics...")
    for i, item in enumerate(dataset):
        if i % 5000 == 0:
            print(f"  processed {i}/{len(dataset)}")
        spk = item["speaker_id"]
        n_samples = len(item["audio"]["array"])
        duration_s = n_samples / SAMPLE_RATE
        stats[spk].append(duration_s)
    return stats

def select_speakers(stats: dict) -> list[dict]:
    """Select 5 speakers meeting low-heterogeneity criteria."""
    candidates = []
    for spk_id, durations in stats.items():
        n = len(durations)
        if n < MIN_CLIPS:
            continue
        std = float(np.std(durations))
        mean = float(np.mean(durations))
        candidates.append({
            "speaker_id": spk_id,
            "clip_count": n,
            "duration_mean": mean,
            "duration_std": std,
        })

    tier1 = [
        c for c in candidates
        if PREFERRED_MIN_CLIPS <= c["clip_count"] <= PREFERRED_MAX_CLIPS
        and PREFERRED_DUR_STD_MIN <= c["duration_std"] <= PREFERRED_DUR_STD_MAX
    ]
    print(f"Tier-1 candidates (preferred range): {len(tier1)}")

    pool = tier1 if len(tier1) >= TARGET_SPEAKERS else candidates
    if len(pool) < TARGET_SPEAKERS:
        raise RuntimeError(
            f"Not enough speakers meeting criteria. Found {len(pool)}, need {TARGET_SPEAKERS}."
        )

    pool.sort(key=lambda c: (c["duration_std"], abs(c["clip_count"] - 115)))
    selected = pool[:TARGET_SPEAKERS]
    return selected

def main():
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not installed. Run: pip install datasets")
        sys.exit(1)

    print("Loading LibriSpeech train-clean-100 (this may take a while)...")
    ds = load_dataset(
        "openslr/librispeech_asr",
        "clean",
        split="train.100",
        trust_remote_code=True,
    )
    print(f"Dataset loaded: {len(ds)} clips")

    stats = compute_speaker_stats(ds)
    print(f"Total unique speakers: {len(stats)}")

    selected = select_speakers(stats)
    salt = load_salt()

    selection = {}
    for entry in selected:
        spk_id = entry["speaker_id"]
        node_hash = hash_speaker(spk_id, salt)
        selection[node_hash] = {
            "clip_count": entry["clip_count"],
            "duration_mean": round(entry["duration_mean"], 3),
            "duration_std": round(entry["duration_std"], 3),
        }
        print(
            f"  node={node_hash}  clips={entry['clip_count']}  "
            f"mean={entry['duration_mean']:.2f}s  std={entry['duration_std']:.2f}s"
        )

    SELECTION_FILE.write_text(json.dumps(selection, indent=2))
    print(f"\nSaved {len(selection)} speakers to {SELECTION_FILE}")
    print("Run: python data/pii_masking.py")

if __name__ == "__main__":
    main()
