"""
LibriSpeech eval-split data preparation.

Reads meta_split.json, pulls clips from the CACHED LibriSpeech test-clean
and dev-clean (validation) splits, strips all PII, and saves raw_clips.pkl
per node. No download required — data is already on disk.

Audio is already at 16kHz — no resampling needed.

Output per training node: data/nodes/{hash}/raw_clips.pkl
Output per test node:     data/test_nodes/{hash}/raw_clips.pkl

Format: list of {"audio": np.ndarray (16kHz float32), "text": str}
No speaker_id or any identity field (Invariant I2).

Usage:
    python data/prepare_librispeech.py                      # meta-train + meta-val
    python data/prepare_librispeech.py --splits meta_test   # final eval prep only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path

import io

import numpy as np
import soundfile as sf

DATA_DIR = Path(__file__).parent
NODES_DIR = DATA_DIR / "nodes"
TEST_NODES_DIR = DATA_DIR / "test_nodes"
META_SPLIT_FILE = DATA_DIR / "meta_split.json"
SALT_FILE = DATA_DIR / "salt.txt"

LIBRISPEECH_CACHE = Path.home() / ".cache/huggingface/datasets/openslr___librispeech_asr"
SPLITS_TO_LOAD = ["test", "validation"]

def load_meta_split() -> dict:
    if not META_SPLIT_FILE.exists():
        print("ERROR: data/meta_split.json not found. Run: python data/meta_split.py")
        sys.exit(1)
    return json.loads(META_SPLIT_FILE.read_text())

def load_salt() -> str:
    if not SALT_FILE.exists():
        print("ERROR: data/salt.txt not found. Run: python data/meta_split.py")
        sys.exit(1)
    return SALT_FILE.read_text().strip()

def hash_speaker(speaker_id: int | str, salt: str) -> str:
    raw = f"{salt}:{speaker_id}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]

def find_arrow_files() -> dict[str, Path]:
    found: dict[str, Path] = {}
    for version_dir in LIBRISPEECH_CACHE.glob("clean/*/*/"):
        for split in SPLITS_TO_LOAD:
            f = version_dir / f"librispeech_asr-{split}.arrow"
            if f.exists() and split not in found:
                found[split] = f
    return found

def load_clips_from_arrow(
    arrow_path: Path,
    target_speaker_ids: set[int],
) -> dict[int, list[dict]]:
    """Extract clips for target speakers from an arrow IPC stream.

    Returns: {speaker_id: [{"audio": np.ndarray, "text": str}, ...]}
    """
    import pyarrow as pa

    node_clips: dict[int, list] = {spk: [] for spk in target_speaker_ids}

    with open(arrow_path, "rb") as fh:
        reader = pa.ipc.open_stream(fh)
                          
        all_batches = []
        try:
            while True:
                all_batches.append(reader.read_next_batch())
        except (StopIteration, pa.lib.ArrowInvalid):
            pass                 

    if not all_batches:
        return node_clips

    schema_names = all_batches[0].schema.names
    has_audio = "audio" in schema_names

    for batch in all_batches:
        spk_col = batch.column("speaker_id").to_pylist()
        text_col = batch.column("text").to_pylist()
        audio_col = batch.column("audio").to_pylist() if has_audio else None

        for idx, spk in enumerate(spk_col):
            spk_int = int(spk)
            if spk_int not in target_speaker_ids:
                continue

            text = str(text_col[idx]).upper().strip()
            if not text:
                continue

            if audio_col is None:
                continue

            audio_entry = audio_col[idx]
            if isinstance(audio_entry, dict) and "bytes" in audio_entry:
                arr, _ = sf.read(io.BytesIO(audio_entry["bytes"]), dtype="float32")
            elif isinstance(audio_entry, dict) and "array" in audio_entry:
                arr = np.array(audio_entry["array"], dtype=np.float32)
            else:
                arr = np.array(audio_entry, dtype=np.float32)
            node_clips[spk_int].append({"audio": arr, "text": text})

    return node_clips

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["meta_train", "meta_val", "meta_test"],
        default=["meta_train", "meta_val"],
    )
    args = parser.parse_args()

    is_test_only = args.splits == ["meta_test"]
    if "meta_test" in args.splits and not is_test_only:
        print("ERROR: Do not mix meta_test with other splits.")
        print("Run meta-test preparation separately:")
        print("  python data/prepare_librispeech.py --splits meta_test")
        sys.exit(1)

    meta_split = load_meta_split()
    salt = load_salt()

    target_hashes: set[str] = set()
    for split_name in args.splits:
        target_hashes.update(meta_split[split_name])

    print(f"Preparing splits: {args.splits} ({len(target_hashes)} nodes)")

    hash_to_spk: dict[str, int] = {}
    for split_name in ["meta_train", "meta_val", "meta_test"]:
        for h in meta_split[split_name]:
            if h in target_hashes:
                                                                     
                for stat_hash in meta_split.get("speaker_stats", {}):
                    if stat_hash == h:
                                                                               
                        break

    print("Resolving node hashes to speaker IDs...")
    arrow_files = find_arrow_files()
    if not arrow_files:
        print(f"ERROR: No LibriSpeech arrow files found in {LIBRISPEECH_CACHE}")
        sys.exit(1)

    all_speaker_ids: set[int] = set()
    import pyarrow as pa
    for split, path in arrow_files.items():
        with open(path, "rb") as fh:
            reader = pa.ipc.open_stream(fh)
            for batch in reader:
                for spk in batch.column("speaker_id").to_pylist():
                    all_speaker_ids.add(int(spk))

    for spk_id in all_speaker_ids:
        h = hash_speaker(spk_id, salt)
        if h in target_hashes:
            hash_to_spk[h] = spk_id

    if len(hash_to_spk) < len(target_hashes):
        missing = target_hashes - set(hash_to_spk.keys())
        print(f"ERROR: Could not resolve {len(missing)} hashes: {missing}")
        sys.exit(1)

    target_speaker_ids = set(hash_to_spk.values())
    spk_to_hash = {v: k for k, v in hash_to_spk.items()}

    print(f"Resolved {len(hash_to_spk)} speakers: {sorted(target_speaker_ids)}")

    all_clips: dict[int, list] = {spk: [] for spk in target_speaker_ids}

    for split, path in arrow_files.items():
        print(f"Loading {split} clips from {path.name}...")
        clips = load_clips_from_arrow(path, target_speaker_ids)
        for spk, clip_list in clips.items():
            all_clips[spk].extend(clip_list)
            if clip_list:
                print(f"  speaker {spk}: {len(clip_list)} clips from {split}")

    out_dir = TEST_NODES_DIR if is_test_only else NODES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    for spk_id, clips in all_clips.items():
        node_hash = spk_to_hash[spk_id]
        node_dir = out_dir / node_hash
        node_dir.mkdir(parents=True, exist_ok=True)
        pkl_path = node_dir / "raw_clips.pkl"
        with open(pkl_path, "wb") as fh:
            pickle.dump(clips, fh)
        print(f"  {node_hash}: {len(clips)} clips → {pkl_path}")

    result = subprocess.run(
        ["grep", "-r", "speaker_id", str(out_dir)],
        capture_output=True, text=True,
    )
    if result.stdout.strip():
        print("INVARIANT VIOLATION I2: speaker_id found in node artifacts")
        sys.exit(1)
    print("\nI2 check: PASSED — no speaker_id in node artifacts")
    print(f"Nodes saved to: {out_dir}")
    print("Next: python data/features.py")

if __name__ == "__main__":
    main()
