"""
PII masking pipeline.

Reads speaker_selection.json, pulls each speaker's clips from HuggingFace,
saves raw_clips.pkl per node WITHOUT any speaker identity fields.

Output per node: data/nodes/{node_hash}/raw_clips.pkl
  Format: list of {"audio": np.ndarray, "text": str}
  No speaker_id, no chapter_id, no id field.

Invariant I2: grep -r "speaker_id" data/nodes/ must return nothing.
"""

import json
import os
import pickle
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent
NODES_DIR = DATA_DIR / "nodes"
SELECTION_FILE = DATA_DIR / "speaker_selection.json"
SALT_FILE = DATA_DIR / "salt.txt"
SAMPLE_RATE = 16000


def load_selection() -> dict:
    if not SELECTION_FILE.exists():
        print("ERROR: speaker_selection.json not found. Run: python data/download.py")
        sys.exit(1)
    return json.loads(SELECTION_FILE.read_text())


def load_salt() -> str:
    if not SALT_FILE.exists():
        print("ERROR: salt.txt not found. Run: python data/download.py")
        sys.exit(1)
    return SALT_FILE.read_text().strip()


def build_speaker_id_to_hash(ds, salt: str, selection: dict) -> dict:
    """Scan the dataset to find which speaker IDs map to our selected node hashes."""
    import hashlib
    seen = set()
    result = {}
    for item in ds:
        spk = item["speaker_id"]
        if spk in seen:
            continue
        seen.add(spk)
        raw = f"{salt}:{spk}".encode()
        h = hashlib.sha256(raw).hexdigest()[:16]
        if h in selection:
            result[spk] = h
        if len(result) == len(selection):
            break
    return result


def main():
    try:
        from datasets import load_dataset
        import numpy as np
    except ImportError:
        print("ERROR: required packages not installed. Run: pip install datasets numpy")
        sys.exit(1)

    selection = load_selection()
    salt = load_salt()

    print("Loading LibriSpeech train-clean-100...")
    ds = load_dataset(
        "openslr/librispeech_asr",
        "clean",
        split="train.100",
        trust_remote_code=True,
    )
    print(f"Loaded {len(ds)} clips. Mapping speaker IDs to node hashes...")

    spk_to_hash = build_speaker_id_to_hash(ds, salt, selection)
    print(f"Resolved {len(spk_to_hash)} speaker IDs.")

    if len(spk_to_hash) != len(selection):
        missing = set(selection.keys()) - set(spk_to_hash.values())
        print(f"ERROR: Could not resolve all speakers. Missing node hashes: {missing}")
        sys.exit(1)

    # Collect clips per node; strip all PII fields
    node_clips: dict[str, list] = {h: [] for h in selection}
    print("Collecting clips (stripping PII)...")
    for i, item in enumerate(ds):
        spk = item["speaker_id"]
        if spk not in spk_to_hash:
            continue
        node_hash = spk_to_hash[spk]
        # Only keep audio array and normalized text — no identity fields
        clean = {
            "audio": item["audio"]["array"].astype("float32"),
            "text": item["text"].upper().strip(),
        }
        node_clips[node_hash].append(clean)
        if i % 5000 == 0:
            print(f"  processed {i}/{len(ds)}")

    NODES_DIR.mkdir(parents=True, exist_ok=True)
    for node_hash, clips in node_clips.items():
        node_dir = NODES_DIR / node_hash
        node_dir.mkdir(parents=True, exist_ok=True)
        pkl_path = node_dir / "raw_clips.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(clips, f)
        print(f"  {node_hash}: {len(clips)} clips → {pkl_path}")

    # Final PII check
    import subprocess
    result = subprocess.run(
        ["grep", "-r", "speaker_id", str(NODES_DIR)],
        capture_output=True, text=True
    )
    if result.stdout.strip():
        print("INVARIANT VIOLATION I2: speaker_id found in data/nodes/")
        sys.exit(1)
    print("\nI2 check: PASSED — no speaker_id in data/nodes/")
    print("Run: python data/features.py")


if __name__ == "__main__":
    main()
