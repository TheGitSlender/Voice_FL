"""
VCTK data preparation pipeline (download + PII masking in one step).

Loads VCTK from HuggingFace, applies meta_split.json speaker selection,
resamples from 22050Hz to 16kHz, strips all PII, and saves raw_clips.pkl
per node directory.

By default only meta-train and meta-val speakers are processed.
Meta-test speakers MUST NOT be loaded during training — use --splits meta_test
only when running the final evaluation.

Output per node:
  data/nodes/{node_hash}/raw_clips.pkl          (meta-train + meta-val)
  data/test_nodes/{node_hash}/raw_clips.pkl     (meta-test, --splits meta_test only)

Format of each pkl: list of {"audio": np.ndarray (16kHz float32), "text": str}
No speaker_id or any identity field (Invariant I2).

Usage:
    python data/prepare_vctk.py                         # meta-train + meta-val
    python data/prepare_vctk.py --splits meta_test      # final eval prep only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent
NODES_DIR = DATA_DIR / "nodes"
TEST_NODES_DIR = DATA_DIR / "test_nodes"
META_SPLIT_FILE = DATA_DIR / "meta_split.json"
SALT_FILE = DATA_DIR / "salt.txt"

TARGET_SR = 16_000
VCTK_NATIVE_SR = 22_050

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

def build_resampler(source_sr: int, target_sr: int):
    import torchaudio
    return torchaudio.transforms.Resample(orig_freq=source_sr, new_freq=target_sr)

def resample_clip(audio_array, item_sr: int, target_sr: int, resampler) -> "np.ndarray":
    import torch
    import numpy as np

    if item_sr == target_sr:
        return np.array(audio_array, dtype=np.float32)

    t = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)
    resampled = resampler(t).squeeze(0).numpy()
    return resampled.astype(np.float32)

def load_vctk():
    from datasets import load_dataset

    for path in ("speechbrain/vctk", "vctk"):
        try:
            print(f"  Trying '{path}' ...")
            ds = load_dataset(path, split="train", trust_remote_code=True)
            print(f"  Loaded {len(ds)} clips from '{path}'")
            return ds
        except Exception as exc:
            print(f"  Failed: {exc}")

    raise RuntimeError(
        "Could not load VCTK. Ensure datasets is installed and you are logged in:\n"
        "  pip install datasets && huggingface-cli login"
    )

def detect_speaker_field(ds) -> str:
    for field in ("speaker_id", "speaker", "spk_id", "speaker_name"):
        if field in ds.features:
            return field
    raise RuntimeError(
        f"No speaker field found. Columns: {list(ds.features.keys())}"
    )

def detect_text_field(item: dict) -> str | None:
    for field in ("text", "sentence", "transcription", "normalized_text"):
        if field in item and item[field]:
            return field
    return None

def map_hashes_to_speaker_ids(
    ds,
    salt: str,
    target_hashes: set[str],
    speaker_field: str,
) -> dict[str, str]:
    """Scan dataset to resolve target hashes → original speaker IDs."""
    seen: set[str] = set()
    hash_to_spk: dict[str, str] = {}

    for item in ds:
        spk = str(item[speaker_field])
        if spk in seen:
            continue
        seen.add(spk)
        h = hashlib.sha256(f"{salt}:{spk}".encode()).hexdigest()[:16]
        if h in target_hashes:
            hash_to_spk[h] = spk
        if len(hash_to_spk) == len(target_hashes):
            break

    return hash_to_spk

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["meta_train", "meta_val", "meta_test"],
        default=["meta_train", "meta_val"],
        help="Which splits to prepare (default: meta_train meta_val)",
    )
    args = parser.parse_args()

    is_test_only = args.splits == ["meta_test"]
    has_test = "meta_test" in args.splits

    if has_test and not is_test_only:
        print("ERROR: Do not mix meta_test with other splits.")
        print("Run meta-test preparation separately and only for final evaluation:")
        print("  python data/prepare_vctk.py --splits meta_test")
        sys.exit(1)

    try:
        import numpy as np              
        import torchaudio              
        from datasets import load_dataset              
    except ImportError as exc:
        print(f"ERROR: Missing dependency: {exc}")
        print("Run: pip install datasets numpy torchaudio")
        sys.exit(1)

    meta_split = load_meta_split()
    salt = load_salt()

    target_hashes: set[str] = set()
    for split_name in args.splits:
        target_hashes.update(meta_split[split_name])

    print(f"Preparing splits: {args.splits}")
    print(f"Target nodes: {len(target_hashes)}")

    print("\nLoading VCTK dataset...")
    ds = load_vctk()

    speaker_field = detect_speaker_field(ds)
    print(f"Speaker field: '{speaker_field}'")

    print("Mapping node hashes to speaker IDs...")
    hash_to_spk = map_hashes_to_speaker_ids(ds, salt, target_hashes, speaker_field)

    if len(hash_to_spk) < len(target_hashes):
        missing = target_hashes - set(hash_to_spk.keys())
        print(f"ERROR: Could not resolve {len(missing)} hashes: {missing}")
        sys.exit(1)

    spk_to_hash = {v: k for k, v in hash_to_spk.items()}

    source_sr: int = ds[0]["audio"].get("sampling_rate", VCTK_NATIVE_SR)
    print(f"Source sample rate: {source_sr}Hz → Target: {TARGET_SR}Hz")
    resampler = build_resampler(source_sr, TARGET_SR)

    out_dir = TEST_NODES_DIR if is_test_only else NODES_DIR

    node_clips: dict[str, list] = {h: [] for h in target_hashes}
    target_speakers = set(spk_to_hash.keys())

    print("Collecting and resampling clips...")
    for i, item in enumerate(ds):
        spk = str(item[speaker_field])
        if spk not in target_speakers:
            continue

        node_hash = spk_to_hash[spk]
        item_sr: int = item["audio"].get("sampling_rate", source_sr)
        audio = resample_clip(item["audio"]["array"], item_sr, TARGET_SR, resampler)

        text_field = detect_text_field(item)
        if text_field is None:
            continue
        text = str(item[text_field]).upper().strip()
        if not text:
            continue

        node_clips[node_hash].append({"audio": audio, "text": text})

        if i % 5000 == 0 and i:
            print(f"  processed {i}/{len(ds)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    for node_hash, clips in node_clips.items():
        node_dir = out_dir / node_hash
        node_dir.mkdir(parents=True, exist_ok=True)
        pkl_path = node_dir / "raw_clips.pkl"
        with open(pkl_path, "wb") as fh:
            pickle.dump(clips, fh)
        print(f"  {node_hash}: {len(clips)} clips → {pkl_path}")

    result = subprocess.run(
        ["grep", "-r", "speaker_id", str(out_dir)],
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        print("INVARIANT VIOLATION I2: speaker_id found in node artifacts")
        sys.exit(1)
    print("\nI2 check: PASSED — no speaker_id in node artifacts")

    if is_test_only:
        print(f"Meta-test nodes saved to: {out_dir}")
        print("Next: python evaluation/eval_poc.py")
    else:
        print(f"Training nodes saved to: {out_dir}")
        print("Next: python data/features.py")

if __name__ == "__main__":
    main()
