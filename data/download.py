"""
data/download.py — S1: Data Source Discovery

Loads LibriSpeech train.clean.100 from HuggingFace, analyzes the speaker
distribution, selects 20 diverse speakers via stratified sampling, assigns
anonymous node IDs, and saves data/speaker_selection.json.

Raw audio is NOT saved — HuggingFace caches the dataset locally after first run.

Run:
    python data/download.py
Output:
    data/speaker_selection.json
    data/salt.txt  (gitignored — do not commit)
"""

import hashlib
import io
import json
import os
import uuid
from collections import defaultdict

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_REPO   = "openslr/librispeech_asr"
DATASET_CONFIG = "clean"
DATASET_SPLIT  = "train.100"
N_NODES        = 20
MIN_CLIPS      = 50
SALT_FILE      = "data/salt.txt"
OUTPUT_FILE    = "data/speaker_selection.json"


# ---------------------------------------------------------------------------
# Audio helpers — bypass torchcodec entirely
# ---------------------------------------------------------------------------
def get_duration(audio_dict: dict) -> float:
    """Read only the header to get clip duration. No full decode."""
    if audio_dict.get("bytes"):
        info = sf.info(io.BytesIO(audio_dict["bytes"]))
    else:
        info = sf.info(audio_dict["path"])
    return info.frames / info.samplerate


# ---------------------------------------------------------------------------
# Step 1 — Load dataset
# ---------------------------------------------------------------------------
def load_librispeech():
    print("=" * 60)
    print("STEP 1 — Loading LibriSpeech train.clean.100")
    print("=" * 60)
    print(f"Source: {DATASET_REPO}  config={DATASET_CONFIG}  split={DATASET_SPLIT}")
    print("First run downloads ~6 GB and caches locally. Subsequent runs are instant.")
    print()

    ds = load_dataset(DATASET_REPO, DATASET_CONFIG, split=DATASET_SPLIT, cache_dir="data")

    # Disable automatic audio decoding — we read duration via soundfile header only.
    # This bypasses torchcodec entirely.
    ds = ds.cast_column("audio", Audio(decode=False))

    print(f"Total clips loaded: {len(ds):,}")
    print(f"Columns:            {ds.column_names}")
    print()
    return ds


# ---------------------------------------------------------------------------
# Step 2 — Analyze speaker distribution
# ---------------------------------------------------------------------------
def analyze_speakers(ds) -> list[dict]:
    print("=" * 60)
    print("STEP 2 — Analyzing speaker distribution")
    print("=" * 60)

    speaker_clips: dict[int, list[float]] = defaultdict(list)

    for row in tqdm(ds, total=len(ds), desc="Grouping by speaker"):
        dur = get_duration(row["audio"])
        speaker_clips[row["speaker_id"]].append(dur)

    speaker_stats = []
    for sid, durs in speaker_clips.items():
        arr = np.array(durs)
        speaker_stats.append({
            "speaker_id":       sid,
            "clip_count":       len(arr),
            "total_duration_s": float(arr.sum()),
            "mean_duration_s":  float(arr.mean()),
            "duration_std":     float(arr.std()),
        })

    all_clips = [s["clip_count"] for s in speaker_stats]
    print(f"Total speakers:         {len(speaker_stats)}")
    print(f"Min clips per speaker:  {min(all_clips)}")
    print(f"Max clips per speaker:  {max(all_clips)}")
    print(f"Mean clips per speaker: {sum(all_clips)/len(all_clips):.1f}")
    print()
    return speaker_stats


# ---------------------------------------------------------------------------
# Step 3 — Rank by duration_std
# ---------------------------------------------------------------------------
def rank_speakers(speaker_stats: list[dict]) -> list[dict]:
    print("=" * 60)
    print("STEP 3 — Ranking speakers by duration_std (heterogeneity proxy)")
    print("=" * 60)
    print("duration_std = std of clip durations per speaker.")
    print("High variance → more varied speech patterns → better FL diversity.")
    print()

    eligible = [s for s in speaker_stats if s["clip_count"] >= MIN_CLIPS]
    print(f"Speakers with >= {MIN_CLIPS} clips: {len(eligible)} / {len(speaker_stats)}")

    eligible.sort(key=lambda x: x["duration_std"])
    for rank, s in enumerate(eligible):
        s["diversity_rank"] = rank + 1

    std_values = [s["duration_std"] for s in eligible]
    print(f"duration_std range: [{min(std_values):.3f}s, {max(std_values):.3f}s]")
    print()
    return eligible


# ---------------------------------------------------------------------------
# Step 4 — Stratified speaker selection
# ---------------------------------------------------------------------------
def load_or_create_salt() -> str:
    os.makedirs("data", exist_ok=True)
    if os.path.exists(SALT_FILE):
        with open(SALT_FILE) as f:
            return f.read().strip()
    salt = str(uuid.uuid4())
    with open(SALT_FILE, "w") as f:
        f.write(salt)
    print(f"Generated new salt → {SALT_FILE}  (gitignored, do not commit)")
    return salt


def anon_hash(speaker_id: int, salt: str) -> str:
    raw = f"{speaker_id}:{salt}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def select_speakers(eligible: list[dict], salt: str) -> list[dict]:
    print("=" * 60)
    print("STEP 4 — Stratified speaker selection")
    print("=" * 60)
    print(f"Target: {N_NODES} speakers from {len(eligible)} eligible.")
    print("Strategy: divide rank-sorted speakers into 20 equal buckets,")
    print("          pick the median speaker from each bucket.")
    print()

    n = len(eligible)
    bucket_size = n / N_NODES
    selected = []

    for i in range(N_NODES):
        start  = int(i * bucket_size)
        end    = int((i + 1) * bucket_size)
        bucket = eligible[start:end]
        if not bucket:
            raise RuntimeError(f"Empty bucket at index {i} — not enough eligible speakers.")
        mid = bucket[len(bucket) // 2]
        node_id = f"node_{i + 1:03d}"
        selected.append({
            "speaker_id":       mid["speaker_id"],
            "node_id":          node_id,
            "anon_hash":        anon_hash(mid["speaker_id"], salt),
            "clip_count":       mid["clip_count"],
            "total_duration_s": mid["total_duration_s"],
            "mean_duration_s":  mid["mean_duration_s"],
            "duration_std":     mid["duration_std"],
            "diversity_rank":   mid["diversity_rank"],
        })

    assert len({s["speaker_id"] for s in selected}) == N_NODES, "Duplicate speakers in selection"
    return selected


# ---------------------------------------------------------------------------
# Step 5 — Save and print summary
# ---------------------------------------------------------------------------
def save_selection(selected: list[dict], total_speakers: int) -> None:
    payload = {
        "selected_speakers":        selected,
        "total_speakers_available": total_speakers,
        "selection_strategy":       "stratified_by_duration_std",
        "min_clips_threshold":      MIN_CLIPS,
        "dataset":                  f"{DATASET_REPO} {DATASET_SPLIT}",
    }
    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {OUTPUT_FILE}")


def print_summary(selected: list[dict]) -> None:
    print()
    print("=" * 60)
    print("STEP 5 — Selected speaker summary")
    print("=" * 60)
    header = f"{'Node':<12} {'Speaker':>10} {'Clips':>7} {'Total(min)':>12} {'Std(s)':>8} {'Rank':>6}"
    print(header)
    print("-" * len(header))
    for s in selected:
        print(
            f"{s['node_id']:<12} "
            f"{s['speaker_id']:>10} "
            f"{s['clip_count']:>7} "
            f"{s['total_duration_s']/60:>12.1f} "
            f"{s['duration_std']:>8.3f} "
            f"{s['diversity_rank']:>6}"
        )
    print()
    total_clips = sum(s["clip_count"] for s in selected)
    total_hrs   = sum(s["total_duration_s"] for s in selected) / 3600
    print(f"Total clips in selection: {total_clips:,}")
    print(f"Total audio in selection: {total_hrs:.2f} hours")
    print()
    print(f"Output: {OUTPUT_FILE}")
    print("Next step: python data/pii_masking.py")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print()
    print("VoiceFL — Phase 1 / S1: Data Source Discovery")
    print()

    ds            = load_librispeech()
    speaker_stats = analyze_speakers(ds)
    eligible      = rank_speakers(speaker_stats)
    salt          = load_or_create_salt()
    selected      = select_speakers(eligible, salt)
    save_selection(selected, total_speakers=len(speaker_stats))
    print_summary(selected)


if __name__ == "__main__":
    main()
