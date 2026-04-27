"""
L2-ARCTIC data preparation for FedLoRA-MAML.

Produces per-speaker node directories:
  features.pt   list of 1D float32 tensors (16 kHz, [-1, 1])
  labels.txt    one uppercase transcription per line

Data source (tried in order):
  1. Local processed corpus — data/l2arctic/{SPEAKER_ID}/wav/*.wav + transcript.txt
     Produced by:  python data/download_l2arctic.py --data_dir <corpus_root>
  2. HuggingFace  nguyenvulebinh/l2arctic  (public subset, ~1000 utts/speaker)

Usage:
    python data/prepare_l2arctic_lora.py --split meta_train
    python data/prepare_l2arctic_lora.py --split meta_test --output_dir data/l2arctic_test_nodes
    python data/prepare_l2arctic_lora.py --split all

Gates (run after each call):
    find data/l2arctic_nodes data/l2arctic_test_nodes -name "*.pkl"   # must be empty
    grep -r "speaker_id" data/l2arctic_nodes data/l2arctic_test_nodes # must be empty
"""

from __future__ import annotations

import argparse
import json
import sys
from math import gcd
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SPLIT_FILE    = ROOT / "data" / "l2arctic_split.json"
LOCAL_DIR     = ROOT / "data" / "l2arctic"
NODES_DIR     = ROOT / "data" / "l2arctic_nodes"
TEST_DIR      = ROOT / "data" / "l2arctic_test_nodes"
TARGET_SR     = 16_000


# ── Audio helpers ─────────────────────────────────────────────────────────────

def _resample(arr: np.ndarray, source_sr: int) -> torch.Tensor:
    """Resample to 16 kHz, normalize to [-1, 1], return 1D float32 tensor."""
    if source_sr != TARGET_SR:
        try:
            import torchaudio
            t = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
            arr = torchaudio.transforms.Resample(
                orig_freq=source_sr, new_freq=TARGET_SR
            )(t).squeeze(0).numpy().astype(np.float32)
        except OSError:
            from scipy.signal import resample_poly
            g = gcd(TARGET_SR, source_sr)
            arr = resample_poly(arr, TARGET_SR // g, source_sr // g).astype(np.float32)

    arr = arr.astype(np.float32)
    max_val = np.abs(arr).max()
    return torch.tensor(arr / (max_val + 1e-8), dtype=torch.float32)


# ── Local corpus loader ───────────────────────────────────────────────────────

def _load_local(speaker_id: str) -> list[tuple[torch.Tensor, str]] | None:
    """
    Load clips from pre-processed local corpus (output of download_l2arctic.py).

    Returns list of (audio_tensor, text) or None if speaker not found locally.
    """
    speaker_dir = LOCAL_DIR / speaker_id
    wav_dir     = speaker_dir / "wav"
    transcript  = speaker_dir / "transcript.txt"

    if not wav_dir.exists():
        return None

    texts: dict[str, str] = {}
    if transcript.exists():
        for line in transcript.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if "\t" in line:
                utt_id, text = line.split("\t", 1)
                texts[utt_id.strip()] = text.strip().upper()

    try:
        import torchaudio as _ta
        _use_torchaudio = True
    except OSError:
        _use_torchaudio = False

    clips: list[tuple[torch.Tensor, str]] = []
    for wav_path in sorted(wav_dir.glob("*.wav")):
        utt_id = wav_path.stem
        text = texts.get(utt_id, "")
        if not text:
            continue
        try:
            if _use_torchaudio:
                waveform, sr = _ta.load(str(wav_path))
                arr = waveform.mean(0).numpy().astype(np.float32)
            else:
                from scipy.io import wavfile
                sr, arr = wavfile.read(str(wav_path))
                if arr.dtype != np.float32:
                    arr = arr.astype(np.float32) / np.iinfo(arr.dtype).max
                if arr.ndim == 2:
                    arr = arr.mean(axis=1)
            tensor = _resample(arr, int(sr))
            clips.append((tensor, text))
        except Exception as exc:
            print(f"  warning: skipping {wav_path.name}: {exc}")

    return clips if clips else None


# ── HuggingFace loader ────────────────────────────────────────────────────────

def _detect_fields(features: dict) -> tuple[str, str]:
    """Auto-detect speaker and text column names from a HuggingFace dataset."""
    speaker_field = None
    for f in ("speaker_id", "speaker", "spk_id", "accent"):
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


_HF_REPO = "chikingsley/l2-arctic-manual-v5.0-16k"


def _load_hf(speaker_ids: set[str]) -> dict[str, list[tuple[torch.Tensor, str]]]:
    """
    Load clips from chikingsley/l2-arctic-manual-v5.0-16k (all splits combined).

    Audio is already 16 kHz float64 normalized — just cast to float32.
    Uses all splits (train/validation/test/suitcase) to maximize clips per speaker.
    """
    from datasets import load_dataset

    print(f"Loading {_HF_REPO} from HuggingFace (all splits) …")
    ds_dict = load_dataset(_HF_REPO, trust_remote_code=True)

    id_map = {s.upper(): s for s in speaker_ids}
    results: dict[str, list[tuple[torch.Tensor, str]]] = {s: [] for s in speaker_ids}

    for split_name, split_ds in ds_dict.items():
        for item in split_ds:
            raw_spk = str(item["speaker_id"]).upper()
            spk = id_map.get(raw_spk)
            if spk is None:
                continue
            audio_dict = item["audio"]
            arr = np.array(audio_dict["array"], dtype=np.float32)
            text = str(item["transcript"]).upper().strip()
            if not text:
                continue
            # Audio is already 16 kHz normalized; just ensure [-1,1] and float32
            max_val = np.abs(arr).max()
            tensor = torch.tensor(arr / (max_val + 1e-8), dtype=torch.float32)
            results[spk].append((tensor, text))

    loaded = {k: v for k, v in results.items() if v}
    missing = speaker_ids - set(loaded.keys())
    if missing:
        print(f"  WARNING: speakers not found in dataset: {sorted(missing)}")

    return results

    speaker_field, text_field = _detect_fields(ds.features)
    print(f"  Fields → speaker: '{speaker_field}', text: '{text_field}'")

    # Build a case-insensitive mapping to handle ID casing differences
    id_map = {s.upper(): s for s in speaker_ids}

    results: dict[str, list[tuple[torch.Tensor, str]]] = {s: [] for s in speaker_ids}
    skipped = 0

    for item in ds:
        raw_spk = str(item[speaker_field]).upper()
        spk = id_map.get(raw_spk)
        if spk is None:
            continue

        audio_dict = item["audio"]
        sr  = int(audio_dict.get("sampling_rate", TARGET_SR))
        arr = np.array(audio_dict["array"], dtype=np.float32)
        text = str(item[text_field]).upper().strip()
        if not text:
            skipped += 1
            continue

        tensor = _resample(arr, sr)
        results[spk].append((tensor, text))

    if skipped:
        print(f"  Skipped {skipped} clips with empty transcriptions.")

    return results


# ── Node writer ───────────────────────────────────────────────────────────────

def write_node(
    speaker_id: str,
    clips: list[tuple[torch.Tensor, str]],
    output_dir: Path,
) -> None:
    """Write features.pt + labels.txt for one speaker into output_dir/speaker_id/."""
    node_dir = output_dir / speaker_id
    node_dir.mkdir(parents=True, exist_ok=True)

    tensors = [c[0] for c in clips]
    labels  = [c[1] for c in clips]

    torch.save(tensors, node_dir / "features.pt")
    (node_dir / "labels.txt").write_text(
        "\n".join(labels) + "\n", encoding="utf-8"
    )

    # Invariant checks
    assert not list(node_dir.glob("*.pkl")), "pkl files must not exist"
    assert not (node_dir / "speaker_id").exists(), "speaker_id file must not exist"

    print(f"  {speaker_id}: {len(clips)} clips → {node_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare L2-ARCTIC node data for FedLoRA-MAML")
    p.add_argument(
        "--split",
        choices=["meta_train", "meta_val", "meta_test", "all"],
        default="meta_train",
        help="Which split(s) to prepare (default: meta_train)",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Override output directory (default: nodes_dir or test_nodes_dir based on split)",
    )
    p.add_argument(
        "--min_clips",
        type=int,
        default=50,
        help="Warn if a speaker has fewer clips than this (default: 50)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    split_data = json.loads(SPLIT_FILE.read_text())

    # Determine which speakers to prepare and where to write them
    tasks: list[tuple[str, str, Path]] = []  # (speaker_id, l1, output_dir)

    splits_to_run = (
        ["meta_train", "meta_val", "meta_test"]
        if args.split == "all"
        else [args.split]
    )

    for split_name in splits_to_run:
        speakers = split_data[split_name]
        if isinstance(speakers, dict):
            speakers = speakers  # {id: l1}
        else:
            speakers = {s: "?" for s in speakers}

        if args.output_dir is not None:
            out = args.output_dir
        elif split_name == "meta_test":
            out = TEST_DIR
        else:
            out = NODES_DIR

        for spk_id, l1 in speakers.items():
            tasks.append((spk_id, l1, out))

    print(f"Preparing {len(tasks)} speakers …")

    # Try local first; collect which speakers need HuggingFace fallback
    hf_needed: list[tuple[str, str, Path]] = []
    local_done: set[str] = set()

    for spk_id, l1, out in tasks:
        clips = _load_local(spk_id)
        if clips is not None:
            if len(clips) < args.min_clips:
                print(f"  warning: {spk_id} only {len(clips)} clips (min={args.min_clips})")
            print(f"[local] {spk_id} ({l1}): {len(clips)} clips")
            write_node(spk_id, clips, out)
            local_done.add(spk_id)
        else:
            hf_needed.append((spk_id, l1, out))

    if hf_needed:
        hf_ids = {t[0] for t in hf_needed}
        print(f"\n[hf] Fetching {len(hf_ids)} speakers from HuggingFace …")
        hf_clips = _load_hf(hf_ids)

        for spk_id, l1, out in hf_needed:
            clips = hf_clips.get(spk_id, [])
            if not clips:
                print(f"  ERROR: {spk_id} not found locally or on HuggingFace")
                continue
            if len(clips) < args.min_clips:
                print(f"  warning: {spk_id} only {len(clips)} clips (min={args.min_clips})")
            print(f"[hf]    {spk_id} ({l1}): {len(clips)} clips")
            write_node(spk_id, clips, out)

    print("\nDone. Run gate checks:")
    print('  find data/l2arctic_nodes data/l2arctic_test_nodes -name "*.pkl"')
    print('  grep -r "speaker_id" data/l2arctic_nodes data/l2arctic_test_nodes')


if __name__ == "__main__":
    main()
