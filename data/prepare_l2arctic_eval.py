"""
Lock evaluation and support clip indices for L2-ARCTIC meta-test speakers.

Saves indices (not audio) to data/l2arctic_eval_clips.json.
Must be run ONCE before any training begins, then committed to git.

For each meta-test speaker:
  - 50 eval utterances  (randomly selected, seed=42, NEVER used in training)
  - 20 support utterances (next 20 after eval, for inner-loop adaptation)
  Support ∩ eval = ∅ by construction.

Usage:
    python data/prepare_l2arctic_eval.py
    git add data/l2arctic_eval_clips.json data/l2arctic_split.json
    git commit -m "chore: lock L2-ARCTIC eval and support clip indices"

Output: data/l2arctic_eval_clips.json
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

SPLIT_FILE = Path("data/l2arctic_split.json")
L2ARCTIC_DIR = Path("data/l2arctic")
OUTPUT_FILE = Path("data/l2arctic_eval_clips.json")

EVAL_N = 50
SUPPORT_N = 20
SEED = 42

def get_utterance_ids(speaker_dir: Path) -> list[str]:
    """Return sorted utterance IDs for a speaker directory."""
    wav_dir = speaker_dir / "wav"
    if not wav_dir.exists():
        return []
    return sorted(p.stem for p in wav_dir.glob("*.wav"))

def main() -> None:
    split = json.loads(SPLIT_FILE.read_text())
    meta_test_speakers = list(split["meta_test"].keys())

    clips: dict = {}
    rng = random.Random(SEED)

    for speaker_id in meta_test_speakers:
        l1 = split["meta_test"][speaker_id]
        speaker_dir = L2ARCTIC_DIR / speaker_id
        all_utts = get_utterance_ids(speaker_dir)

        if not all_utts:
            log.warning(
                f"{speaker_id}: no WAVs found in {speaker_dir}/wav — "
                f"run download_l2arctic.py first. Saving empty entry."
            )
            clips[speaker_id] = {
                "l1": l1,
                "eval_indices": [],
                "support_indices": [],
                "note": "populated after download_l2arctic.py runs",
            }
            continue

        total = len(all_utts)
        if total < EVAL_N + SUPPORT_N:
            log.warning(
                f"{speaker_id}: only {total} utterances, need {EVAL_N + SUPPORT_N}. "
                f"Reducing support set."
            )
            eval_n = min(EVAL_N, total)
            support_n = min(SUPPORT_N, total - eval_n)
        else:
            eval_n = EVAL_N
            support_n = SUPPORT_N

        eval_indices = sorted(rng.sample(range(total), eval_n))
        remaining = [i for i in range(total) if i not in set(eval_indices)]
        support_indices = sorted(rng.sample(remaining, support_n))

        assert not set(eval_indices) & set(support_indices), "Overlap bug"

        clips[speaker_id] = {
            "l1": l1,
            "eval_indices": eval_indices,
            "eval_utterances": [all_utts[i] for i in eval_indices],
            "support_indices": support_indices,
            "support_utterances": [all_utts[i] for i in support_indices],
            "total_available": total,
        }
        log.info(
            f"  {speaker_id} ({l1}): "
            f"eval={eval_n}, support={support_n}, total={total}"
        )

    OUTPUT_FILE.write_text(json.dumps(clips, indent=2))
    log.info(f"Saved to {OUTPUT_FILE}")
    log.info("Commit this file before running any training.")

if __name__ == "__main__":
    main()
