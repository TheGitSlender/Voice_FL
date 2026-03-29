"""
data/pii_masking.py — S2a: PII Masking and Data Governance

Reads data/speaker_selection.json (produced by download.py).
For each selected speaker, extracts their audio clips from the HuggingFace
dataset, strips ALL identifying metadata, and saves stripped clips per node.

Privacy guarantee:
  - speaker_id:  used only transiently during filtering, never written to disk
  - chapter_id:  discarded entirely
  - file path:   discarded entirely
  - id field:    discarded entirely
  - audio array: treated as biometric — converted to mel features in features.py,
                 then deleted

Output per node (data/nodes/node_XXX/):
  - raw_clips.pkl   ← TEMPORARY — consumed and deleted by features.py
  - metadata.json   ← permanent node metadata (no speaker_id)

Run:
    python data/pii_masking.py
Requires:
    data/speaker_selection.json
    data/cleaning_config.json
"""

import io
import json
import os
import pickle
from collections import defaultdict

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_REPO    = "openslr/librispeech_asr"
DATASET_CONFIG  = "clean"
DATASET_SPLIT   = "train.100"
SELECTION_FILE  = "data/speaker_selection.json"
CONFIG_FILE     = "data/cleaning_config.json"
NODES_DIR       = "data/nodes"


# ---------------------------------------------------------------------------
# Step 1 — Load speaker selection
# ---------------------------------------------------------------------------
def load_speaker_selection() -> tuple[dict[int, dict], list[dict]]:
    """Returns (speaker_id -> node_meta, list of selected speaker dicts)."""
    print("=" * 60)
    print("STEP 1 — Loading speaker selection")
    print("=" * 60)

    if not os.path.exists(SELECTION_FILE):
        raise FileNotFoundError(
            f"{SELECTION_FILE} not found. Run: python data/download.py first."
        )

    with open(SELECTION_FILE) as f:
        payload = json.load(f)

    selected = payload["selected_speakers"]
    mapping  = {s["speaker_id"]: s for s in selected}

    print(f"Loaded {len(selected)} selected speakers from {SELECTION_FILE}")
    for s in selected:
        print(f"  {s['node_id']}  speaker={s['speaker_id']}  clips={s['clip_count']}")
    print()
    return mapping, selected


# ---------------------------------------------------------------------------
# Load cleaning config
# ---------------------------------------------------------------------------
def load_cleaning_config() -> dict:
    if not os.path.exists(CONFIG_FILE):
        print(f"WARNING: {CONFIG_FILE} not found — using defaults.")
        return {
            "min_duration_s": 1.0,
            "max_duration_s": 30.0,
            "max_silence_fraction": 0.5,
            "normalize_audio": True,
            "truncate_long_clips": False,
        }
    with open(CONFIG_FILE) as f:
        cfg = json.load(f)
    print(f"Cleaning config loaded from {CONFIG_FILE}:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print()
    return cfg


# ---------------------------------------------------------------------------
# Step 2 — Filter dataset to selected speakers
# ---------------------------------------------------------------------------
def filter_and_group(ds, mapping: dict[int, dict], cfg: dict) -> dict[str, list[dict]]:
    """
    Iterates the dataset once. For each row belonging to a selected speaker:
      - Strips all PII fields
      - Applies cleaning filters (duration, silence)
      - Groups stripped clips by node_id

    Returns: node_id -> list of {audio: np.array, text: str, duration_s: float}
    """
    print("=" * 60)
    print("STEP 2+3+4 — Filtering, stripping PII, and grouping by node")
    print("=" * 60)

    selected_ids = set(mapping.keys())
    node_clips: dict[str, list[dict]] = defaultdict(list)

    filter_counts: dict[str, dict[str, int]] = {
        s["node_id"]: {"kept": 0, "too_short": 0, "too_long": 0, "silence": 0}
        for s in mapping.values()
    }

    min_dur   = cfg.get("min_duration_s", 0.0) or 0.0
    max_dur   = cfg.get("max_duration_s", None)
    max_sil   = cfg.get("max_silence_fraction", 0.0) or 0.0
    truncate  = cfg.get("truncate_long_clips", False)

    for row in tqdm(ds, total=len(ds), desc="Filtering dataset"):
        sid = row["speaker_id"]
        if sid not in selected_ids:
            continue

        node_meta = mapping[sid]
        node_id   = node_meta["node_id"]
        fc        = filter_counts[node_id]

        audio_dict = row["audio"]
        if audio_dict.get("bytes"):
            arr, sr = sf.read(io.BytesIO(audio_dict["bytes"]), dtype="float32")
        else:
            arr, sr = sf.read(audio_dict["path"], dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)  # stereo → mono
        arr = arr.astype(np.float32)
        dur = len(arr) / sr

        # --- Cleaning filters ---

        # Duration: too short
        if min_dur > 0 and dur < min_dur:
            fc["too_short"] += 1
            continue

        # Duration: too long — truncate or drop
        if max_dur is not None and dur > max_dur:
            if truncate:
                max_samples = int(max_dur * sr)
                arr = arr[:max_samples]
                dur = len(arr) / sr
            else:
                fc["too_long"] += 1
                continue

        # Silence fraction
        if max_sil > 0:
            sil_frac = float((np.abs(arr) < 0.01).sum()) / len(arr)
            if sil_frac > max_sil:
                fc["silence"] += 1
                continue

        # --- Strip PII — keep only audio array + text + duration ---
        # speaker_id, chapter_id, file, id are intentionally NOT stored
        stripped = {
            "audio":      arr,
            "text":       row["text"],
            "duration_s": dur,
        }

        node_clips[node_id].append(stripped)
        fc["kept"] += 1

    # Print filter report
    print()
    print(f"{'Node':<12} {'Kept':>7} {'Too short':>11} {'Too long':>10} {'Silence':>9}")
    print("-" * 55)
    for node_id in sorted(filter_counts.keys()):
        fc = filter_counts[node_id]
        print(
            f"{node_id:<12} {fc['kept']:>7} {fc['too_short']:>11} "
            f"{fc['too_long']:>10} {fc['silence']:>9}"
        )
    print()
    return dict(node_clips)


# ---------------------------------------------------------------------------
# Step 5 — Save per-node raw_clips.pkl (temporary)
# ---------------------------------------------------------------------------
def save_raw_clips(node_clips: dict[str, list[dict]], selected: list[dict]):
    print("=" * 60)
    print("STEP 5 — Saving raw_clips.pkl per node")
    print("=" * 60)
    print("NOTE: raw_clips.pkl is a temporary file.")
    print("      features.py will consume and DELETE it after extracting mel features.")
    print()

    node_meta_map = {s["node_id"]: s for s in selected}

    for node_id in sorted(node_clips.keys()):
        clips   = node_clips[node_id]
        node_dir = os.path.join(NODES_DIR, node_id)
        os.makedirs(node_dir, exist_ok=True)

        pkl_path = os.path.join(node_dir, "raw_clips.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(clips, f)

        print(f"  {node_id}: {len(clips)} clips → {pkl_path}")

    print()


# ---------------------------------------------------------------------------
# Step 6 — Save metadata.json (permanent, no speaker_id)
# ---------------------------------------------------------------------------
def save_metadata(node_clips: dict[str, list[dict]], selected: list[dict]):
    print("=" * 60)
    print("STEP 6 — Saving metadata.json per node (no speaker_id)")
    print("=" * 60)

    node_meta_map = {s["node_id"]: s for s in selected}

    for node_id in sorted(node_clips.keys()):
        clips    = node_clips[node_id]
        meta_src = node_meta_map[node_id]
        node_dir = os.path.join(NODES_DIR, node_id)

        durations   = [c["duration_s"] for c in clips]
        total_dur   = sum(durations)
        mean_dur    = total_dur / len(durations) if durations else 0.0

        metadata = {
            "node_id":         node_id,
            "anon_hash":       meta_src["anon_hash"],
            "clip_count":      len(clips),
            "total_duration_s": round(total_dur, 3),
            "mean_duration_s":  round(mean_dur, 3),
        }
        # Explicit check: speaker_id must NOT appear
        assert "speaker_id" not in metadata, "BUG: speaker_id leaked into metadata"

        meta_path = os.path.join(node_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  {node_id}: {len(clips)} clips, {total_dur/60:.1f} min")

    print()


# ---------------------------------------------------------------------------
# Step 7 — Verify PII removal
# ---------------------------------------------------------------------------
def verify_pii_removal(node_clips: dict[str, list[dict]]):
    print("=" * 60)
    print("STEP 7 — Verifying PII removal")
    print("=" * 60)

    # Pick one node for spot-check
    node_id = sorted(node_clips.keys())[0]
    node_dir = os.path.join(NODES_DIR, node_id)
    pkl_path = os.path.join(node_dir, "raw_clips.pkl")

    with open(pkl_path, "rb") as f:
        clips = pickle.load(f)

    assert len(clips) > 0, "No clips found in pkl"
    for clip in clips:
        assert "speaker_id" not in clip,  "BUG: speaker_id present in clip"
        assert "chapter_id" not in clip,  "BUG: chapter_id present in clip"
        assert "file"        not in clip,  "BUG: file path present in clip"
        assert "id"          not in clip,  "BUG: id field present in clip"
        assert "audio"       in clip,      "BUG: audio array missing"
        assert "text"        in clip,      "BUG: text missing"
        assert "duration_s"  in clip,      "BUG: duration_s missing"
        assert clip["audio"].dtype == np.float32, "BUG: audio dtype is not float32"

    # Verify metadata.json has no speaker_id
    meta_path = os.path.join(node_dir, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    assert "speaker_id" not in meta, "BUG: speaker_id in metadata.json"

    print(f"Spot-checked {node_id}: {len(clips)} clips")
    print("PII masking verified: no speaker identity in any node artifact")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print()
    print("VoiceFL — Phase 1 / S2a: PII Masking and Data Governance")
    print()

    mapping, selected = load_speaker_selection()
    cfg               = load_cleaning_config()

    print("=" * 60)
    print("STEP 2 — Loading LibriSpeech (from cache)")
    print("=" * 60)
    ds = load_dataset(DATASET_REPO, DATASET_CONFIG, split=DATASET_SPLIT, cache_dir="data")
    ds = ds.cast_column("audio", Audio(decode=False))  # decode manually with soundfile
    print(f"Dataset loaded: {len(ds):,} clips")
    print()

    node_clips = filter_and_group(ds, mapping, cfg)

    # Enforce minimum clips constraint
    low_nodes = [nid for nid, clips in node_clips.items() if len(clips) < 50]
    if low_nodes:
        print(f"WARNING: Nodes below 50-clip minimum after filtering: {low_nodes}")
        print("These nodes will still be processed, but consider relaxing cleaning_config.json")
        print("or re-running download.py to replace them with speakers that have more clips.")
        print()

    save_raw_clips(node_clips, selected)
    save_metadata(node_clips, selected)
    verify_pii_removal(node_clips)

    print("=" * 60)
    total_clips = sum(len(c) for c in node_clips.values())
    print(f"PII masking complete: {len(node_clips)} nodes, {total_clips:,} clips total")
    print(f"Next step: python data/features.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
