"""
VCTK data preparation for FedLoRA-MAML (Phase 2).

Downloads VCTK from HuggingFace, selects 20 speakers, resamples to 16kHz,
normalizes audio, and saves features.pt + labels.txt per node.

Speaker split is locked in data/vctk_split.json on first run. Commit this file
before running any training (same protocol as l2arctic_split.json).

Output:
  data/vctk_nodes/{speaker_id}/features.pt   list of 1D float32 tensors (16kHz, [-1,1])
  data/vctk_nodes/{speaker_id}/labels.txt    one uppercase transcription per line
  data/vctk_split.json                       locked speaker split

Usage:
    python data/prepare_vctk_lora.py                    # meta-train + meta-val
    python data/prepare_vctk_lora.py --splits meta_test # meta-test only (ONCE at the end)
    python data/prepare_vctk_lora.py --min_clips 80     # minimum clips per speaker
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

VCTK_NODES_DIR = ROOT / "data" / "vctk_nodes"
SPLIT_FILE = ROOT / "data" / "vctk_split.json"

TARGET_SR = 16_000
MIN_CLIPS_DEFAULT = 100

_DEFAULT_SPLIT = {
    "meta_train": [
        "p225", "p226", "p227", "p228", "p229",
        "p230", "p231", "p232", "p233", "p234",
        "p236", "p237",
    ],
    "meta_val": ["p238", "p239", "p240", "p241"],
    "meta_test": ["p243", "p244", "p245", "p246"],
    "note": (
        "meta_test locked before any training. "
        "wav2vec2-base-960h was fine-tuned on American English (LibriSpeech). "
        "VCTK British/Scottish/Irish speakers are genuinely out-of-domain."
    ),
}

def load_or_create_split() -> dict:
    if SPLIT_FILE.exists():
        split = json.loads(SPLIT_FILE.read_text())
        print(f"Loaded existing split from {SPLIT_FILE}")
        return split
    SPLIT_FILE.write_text(json.dumps(_DEFAULT_SPLIT, indent=2))
    print(f"Created new split → {SPLIT_FILE}")
    print("  COMMIT THIS FILE before running any training.")
    return _DEFAULT_SPLIT

def _try_load_vctk():
    """Try multiple HuggingFace paths to load VCTK."""
    from datasets import load_dataset

    candidates = [
        ("speechbrain/vctk", {"split": "train", "trust_remote_code": True}),
        ("vctk", {"split": "train", "trust_remote_code": True}),
    ]
    for path, kwargs in candidates:
        try:
            print(f"  Trying '{path}' ...")
            ds = load_dataset(path, **kwargs)
            print(f"  Loaded {len(ds)} clips from '{path}'")
            return ds
        except Exception as exc:
            print(f"  Failed: {exc}")

    raise RuntimeError(
        "Cannot load VCTK from HuggingFace.\n"
        "  pip install datasets && huggingface-cli login\n"
        "  or: python data/prepare_vctk_lora.py --local_dir /path/to/vctk"
    )

def _detect_fields(ds) -> tuple[str, str]:
    """Return (speaker_field, text_field) from the dataset schema."""
    features = ds.features if hasattr(ds, "features") else ds[0].keys()

    speaker_field = None
    for f in ("speaker_id", "speaker", "spk_id", "speaker_name"):
        if f in features:
            speaker_field = f
            break
    if speaker_field is None:
        raise RuntimeError(f"No speaker field found. Columns: {list(features)}")

    text_field = None
    for f in ("text", "sentence", "transcription", "normalized_text"):
        if f in features:
            text_field = f
            break
    if text_field is None:
        raise RuntimeError(f"No text field found. Columns: {list(features)}")

    return speaker_field, text_field

def _resample(audio_array, source_sr: int) -> torch.Tensor:
    """Resample to 16kHz, normalize to [-1, 1], return 1D float32 tensor."""
    import numpy as np
    arr = np.array(audio_array, dtype=np.float32)

    if source_sr != TARGET_SR:
        try:
            import torchaudio
            resampler = torchaudio.transforms.Resample(orig_freq=source_sr, new_freq=TARGET_SR)
            arr = resampler(torch.tensor(arr).unsqueeze(0)).squeeze(0).numpy().astype(np.float32)
        except OSError:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(TARGET_SR, source_sr)
            arr = resample_poly(arr, TARGET_SR // g, source_sr // g).astype(np.float32)

    max_val = np.abs(arr).max()
    normalized = arr / (max_val + 1e-8)
    return torch.tensor(normalized, dtype=torch.float32)

def collect_clips(ds, speaker_ids: set[str], speaker_field: str, text_field: str) -> dict[str, list]:
    """Collect (audio_tensor, text) pairs per speaker."""
    import numpy as np

    node_data: dict[str, list] = {s: [] for s in speaker_ids}
    first_sr_map: dict[str, int] = {}

    for item in ds:
        spk = str(item[speaker_field])
        if spk not in speaker_ids:
            continue

        audio_dict = item["audio"]
        sr = int(audio_dict.get("sampling_rate", TARGET_SR))
        arr = audio_dict["array"]

        text = str(item.get(text_field, "")).upper().strip()
        if not text:
            continue

        tensor = _resample(arr, sr)
        if len(tensor) < 3200:                      
            continue

        node_data[spk].append((tensor, text))
        first_sr_map.setdefault(spk, sr)

    return node_data

def save_node(speaker_id: str, clips: list[tuple[torch.Tensor, str]], out_dir: Path) -> None:
    node_dir = out_dir / speaker_id
    node_dir.mkdir(parents=True, exist_ok=True)

    features = [t for t, _ in clips]
    labels = [text for _, text in clips]

    torch.save(features, node_dir / "features.pt")
    (node_dir / "labels.txt").write_text("\n".join(labels) + "\n")
    print(f"  {speaker_id}: {len(clips)} clips → {node_dir}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare VCTK nodes for FedLoRA-MAML")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["meta_train", "meta_val", "meta_test"],
        default=["meta_train", "meta_val"],
    )
    parser.add_argument(
        "--min_clips", type=int, default=MIN_CLIPS_DEFAULT,
        help="Minimum clips per speaker (default: 100)",
    )
    parser.add_argument(
        "--out_dir", type=str, default=str(VCTK_NODES_DIR),
        help="Output node directory (default: data/vctk_nodes/)",
    )
    args = parser.parse_args()

    if "meta_test" in args.splits and len(args.splits) > 1:
        print("ERROR: run meta_test separately — never mix with train/val splits.")
        sys.exit(1)

    split = load_or_create_split()

    target_speakers: set[str] = set()
    for s in args.splits:
        target_speakers.update(split.get(s, []))

    out_dir = Path(args.out_dir)
    print(f"\nTarget speakers ({len(target_speakers)}): {sorted(target_speakers)}")
    print(f"Output: {out_dir}")

    print("\nLoading VCTK from HuggingFace...")
    ds = _try_load_vctk()

    speaker_field, text_field = _detect_fields(ds)
    print(f"Fields → speaker: '{speaker_field}', text: '{text_field}'")

    print("\nCollecting and resampling clips (this may take a few minutes)...")
    node_data = collect_clips(ds, target_speakers, speaker_field, text_field)

    print("\nSaving node directories...")
    ok_count = 0
    for speaker_id in sorted(target_speakers):
        clips = node_data[speaker_id]
        if len(clips) < args.min_clips:
            print(
                f"  WARNING: {speaker_id} has only {len(clips)} clips "
                f"(< {args.min_clips}). Saving anyway."
            )
        save_node(speaker_id, clips, out_dir)
        ok_count += 1

    print(f"\nDone: {ok_count}/{len(target_speakers)} speaker nodes saved to {out_dir}")

    if "meta_test" not in args.splits:
        print("\nNext steps:")
        print("  1. git add data/vctk_split.json && git commit -m 'chore: lock vctk speaker split'")
        print("  2. python maml/meta_train.py --config configs/vctk_lora_poc.yaml --rounds 20")
    else:
        print("\nMeta-test nodes saved. Run final evaluation:")
        print("  python evaluation/eval_lora.py --config configs/vctk_lora_poc.yaml")

if __name__ == "__main__":
    main()
