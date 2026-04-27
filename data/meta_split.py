"""
Speaker selection and meta-split definition.

Uses LibriSpeech test-clean and dev-clean (validation) splits.
These speakers were NEVER used in wav2vec2-base-960h fine-tuning (train-960h).
The lm_head has not been adapted to their speech patterns.
The k=0 WER gap is real — no perturbation protocol needed.

Scientific validity:
  wav2vec2-base-960h fine-tuned on: train-100 + train-360 + train-other-500
  NOT fine-tuned on: test-clean, dev-clean (these splits are our evaluation pool)

Speaker pool: 40 test-clean + 40 dev-clean = 80 speakers, 32–108 clips each.
MIN_CLIPS=40 leaves 76 eligible. We pick the top 20 by clip count.

Split (12 meta-train / 4 meta-val / 4 meta-test):
  meta-train: 12 speakers used in federation (highest clip counts)
  meta-val:   4 speakers used for hyperparameter search only
  meta-test:  4 speakers with LOWEST clip counts (harder test, locked first)

Output: data/meta_split.json

CRITICAL: Run once before any training. Commit to git immediately.
         Do not re-run after training has started.

Usage:
    python data/meta_split.py
    python data/meta_split.py --dry_run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import sys
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).parent
SALT_FILE = DATA_DIR / "salt.txt"
META_SPLIT_FILE = DATA_DIR / "meta_split.json"

TOTAL_SPEAKERS = 20
META_TRAIN_N = 12
META_VAL_N = 4
META_TEST_N = 4
MIN_CLIPS = 40

LIBRISPEECH_CACHE = Path.home() / ".cache/huggingface/datasets/openslr___librispeech_asr"
SPLITS_TO_USE = ["test", "validation"]                      

def load_salt() -> str:
    if SALT_FILE.exists():
        return SALT_FILE.read_text().strip()
    salt = secrets.token_hex(32)
    SALT_FILE.write_text(salt)
    return salt

def hash_speaker(speaker_id: int | str, salt: str) -> str:
    raw = f"{salt}:{speaker_id}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]

def find_arrow_files() -> dict[str, Path]:
    """Locate cached LibriSpeech clean test + validation arrow files."""
    found: dict[str, Path] = {}
    for version_dir in LIBRISPEECH_CACHE.glob("clean/*/*/"):
        for split in SPLITS_TO_USE:
            f = version_dir / f"librispeech_asr-{split}.arrow"
            if f.exists() and split not in found:
                found[split] = f
    return found

def count_clips_per_speaker(arrow_path: Path) -> Counter:
    """Read speaker_id column from arrow IPC stream, count clips per speaker."""
    import pyarrow as pa

    counts: Counter = Counter()
    with open(arrow_path, "rb") as fh:
        reader = pa.ipc.open_stream(fh)
        for batch in reader:
            for spk in batch.column("speaker_id").to_pylist():
                counts[int(spk)] += 1
    return counts

def select_speakers(
    combined_counts: Counter,
) -> list[tuple[int, int]]:                            
    """Return top TOTAL_SPEAKERS by clip count, all with >= MIN_CLIPS."""
    eligible = [(spk, cnt) for spk, cnt in combined_counts.items() if cnt >= MIN_CLIPS]
    eligible.sort(key=lambda x: (-x[1], x[0]))
    if len(eligible) < TOTAL_SPEAKERS:
        raise RuntimeError(
            f"Only {len(eligible)} speakers have >= {MIN_CLIPS} clips "
            f"(need {TOTAL_SPEAKERS})."
        )
    return eligible[:TOTAL_SPEAKERS]

def make_split(
    ordered: list[tuple[int, int]],
) -> tuple[list[int], list[int], list[int]]:
    """Assign 12/4/4 split.

    Ordered by clip count descending.
    meta-test = last 4 (fewest clips = hardest test = most honest evaluation).
    meta-val  = next 4.
    meta-train = first 12.
    """
    ids = [spk for spk, _ in ordered]
    meta_test = ids[-META_TEST_N:]
    remaining = ids[:-META_TEST_N]
    meta_val = remaining[-META_VAL_N:]
    meta_train = remaining[:-META_VAL_N]

    assert len(meta_train) == META_TRAIN_N
    assert len(meta_val) == META_VAL_N
    assert len(meta_test) == META_TEST_N
    assert not (set(meta_train) & set(meta_val))
    assert not (set(meta_train) & set(meta_test))
    assert not (set(meta_val) & set(meta_test))
    return meta_train, meta_val, meta_test

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if META_SPLIT_FILE.exists() and not args.dry_run:
        print(f"WARNING: {META_SPLIT_FILE} already exists.")
        print("Re-running overwrites locked meta-test speakers and invalidates all training.")
        answer = input("Continue? [yes/no]: ").strip().lower()
        if answer != "yes":
            print("Aborted.")
            sys.exit(0)

    print("Locating cached LibriSpeech test + validation arrow files...")
    arrow_files = find_arrow_files()
    if not arrow_files:
        print("ERROR: No LibriSpeech clean test/validation arrow files found.")
        print(f"Expected in: {LIBRISPEECH_CACHE}")
        print("Run: from datasets import load_dataset; load_dataset('openslr/librispeech_asr', 'clean', split='test')")
        sys.exit(1)

    for split, path in arrow_files.items():
        print(f"  Found {split}: {path.name}")

    combined: Counter = Counter()
    per_split_counts: dict[str, Counter] = {}
    for split, path in arrow_files.items():
        print(f"Counting clips in {split}...")
        counts = count_clips_per_speaker(path)
        per_split_counts[split] = counts
        combined.update(counts)
        print(f"  {len(counts)} speakers, {sum(counts.values())} clips")

    ordered = select_speakers(combined)
    print(f"\nSelected {len(ordered)} speakers (min {MIN_CLIPS} clips each):")
    for spk, cnt in ordered:
        print(f"  speaker_id={spk}  clips={cnt}")

    meta_train, meta_val, meta_test = make_split(ordered)
    salt = load_salt()

    def hashed(ids: list[int]) -> list[str]:
        return [hash_speaker(i, salt) for i in ids]

    spk_source: dict[int, str] = {}
    for split, counts in per_split_counts.items():
        for spk in counts:
            spk_source[spk] = split

    clip_counts_dict = {spk: cnt for spk, cnt in ordered}

    split_doc = {
        "meta_train": hashed(meta_train),
        "meta_val": hashed(meta_val),
        "meta_test": hashed(meta_test),
        "dataset": "librispeech_asr_clean",
        "splits_used": sorted(arrow_files.keys()),
        "total_speakers": TOTAL_SPEAKERS,
        "split": f"{META_TRAIN_N}/{META_VAL_N}/{META_TEST_N}",
        "split_rationale": "meta-test locked before any training decisions; fewest clips = hardest speakers",
        "min_clips_threshold": MIN_CLIPS,
        "speaker_stats": {
            hash_speaker(spk, salt): {
                "clip_count": clip_counts_dict[spk],
                "source_split": spk_source.get(spk, "unknown"),
            }
            for spk, _ in ordered
        },
    }

    if args.dry_run:
        print("\n[DRY RUN — not saving]\n")
        print(json.dumps(split_doc, indent=2))
        return

    META_SPLIT_FILE.write_text(json.dumps(split_doc, indent=2))
    print(f"\nSaved: {META_SPLIT_FILE}")
    print(f"\nmeta-test speakers locked (hashed): {split_doc['meta_test']}")
    print("Commit NOW before any training:")
    print("  git add data/meta_split.json data/salt.txt")
    print("  git commit -m 'chore: lock meta-test speakers (librispeech test+val)'")
    print("\nNext: python data/prepare_librispeech.py")

if __name__ == "__main__":
    main()
