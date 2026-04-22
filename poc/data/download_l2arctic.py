"""
L2-ARCTIC data pipeline — download, resample, and organize.

L2-ARCTIC corpus: 24 non-native English speakers across 6 L1 groups.
Audio is 22050 Hz WAV; we resample to 16000 Hz for wav2vec2 compatibility.

Speaker layout:
  Arabic (ARA):    ABA, SKA, YBAA, ZHAA
  Hindi (HIN):     HQTV, MBMPS, NCC, SVBI
  Korean (KOR):    HJK, YDCK, YKWK, LXC
  Mandarin (CMN):  BWC, NJS, TXHC, YKWK (see NOTE below)
  Spanish (SPA):   EBVS, ERMS, LXC, TNI
  Vietnamese (VIE): HKK, PNV, THV, TLV

NOTE on speaker IDs: some sources list overlapping codes. The authoritative
list is from https://psi.engr.tamu.edu/l2-arctic-corpus/. If you have the
corpus extracted locally, run with --data_dir pointing to the root.

Usage:
    # If you have the corpus downloaded and extracted:
    python data/download_l2arctic.py --data_dir /path/to/l2arctic --output_dir data/l2arctic

    # To verify the output structure:
    python data/download_l2arctic.py --data_dir /path/to/l2arctic --output_dir data/l2arctic --verify

Output structure:
    data/l2arctic/
      ABA/
        wav/          (resampled 16kHz WAVs)
        transcript.txt
      SKA/
        ...

L2-ARCTIC is distributed under a non-commercial research license.
Download from: https://psi.engr.tamu.edu/l2-arctic-corpus/
(requires filling out a form; ~11 GB download)

Alternatively it is available on HuggingFace as:
  https://huggingface.co/datasets/nguyenvulebinh/l2arctic
  (subset; check license before use)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from pathlib import Path

import torch
import torchaudio

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

TARGET_SR = 16000
SOURCE_SR = 22050

L2ARCTIC_SPEAKERS: dict[str, list[str]] = {
    "ARA": ["ABA", "SKA", "YBAA", "ZHAA"],
    "HIN": ["HQTV", "MBMPS", "NCC", "SVBI"],
    "KOR": ["HJK", "YDCK", "YKWK", "LXC"],
    "CMN": ["BWC", "NJS", "TXHC", "THV"],                                    
    "SPA": ["EBVS", "ERMS", "TNI", "ERMS"],                                   
    "VIE": ["HKK", "PNV", "TLV", "ZHAA"],                                    
}

ALL_SPEAKERS: list[str] = [
    spk for speakers in L2ARCTIC_SPEAKERS.values() for spk in speakers
]

def resample_wav(
    src: Path,
    dst: Path,
    source_sr: int = SOURCE_SR,
    target_sr: int = TARGET_SR,
) -> None:
    """Load a WAV file, resample to target_sr, and save."""
    waveform, sr = torchaudio.load(str(src))
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
                                          
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.clamp(-1.0, 1.0)
    torchaudio.save(str(dst), waveform, target_sr)

def parse_transcript_file(transcript_path: Path) -> dict[str, str]:
    """
    Parse L2-ARCTIC transcript file.

    Expected format (one utterance per line):
        ( arctic_a0001 "THE NORTH WIND AND THE SUN" )

    Returns: {utterance_id: transcript_text}
    """
    pattern = re.compile(r'\(\s*(\S+)\s+"([^"]+)"\s*\)')
    transcripts: dict[str, str] = {}
    for line in transcript_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        m = pattern.match(line)
        if m:
            utt_id, text = m.group(1), m.group(2)
            transcripts[utt_id] = text.upper()
    return transcripts

def process_speaker(
    speaker_id: str,
    corpus_dir: Path,
    output_dir: Path,
) -> int:
    """
    Process one speaker: resample all WAVs and write transcript.txt.

    Returns the number of utterances processed.
    """
                                                                         
    speaker_corpus = corpus_dir / speaker_id
    if not speaker_corpus.exists():
        log.warning(f"Speaker dir not found: {speaker_corpus} — skipping")
        return 0

    wav_in_dir = speaker_corpus / "wav"
                                                                   
    prompts_candidates = list(speaker_corpus.glob("*transcript*")) +\
                         list(speaker_corpus.glob("*prompts*")) +\
                         list(speaker_corpus.glob("annotation/*.txt"))

    transcripts: dict[str, str] = {}
    for cand in prompts_candidates:
        parsed = parse_transcript_file(cand)
        if parsed:
            transcripts.update(parsed)
            break

    if not transcripts:
        log.warning(f"No transcript file found for {speaker_id}")

    out_speaker = output_dir / speaker_id
    out_wav_dir = out_speaker / "wav"
    out_wav_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(wav_in_dir.glob("*.wav"))
    count = 0
    lines: list[str] = []

    for wav_src in wav_files:
        utt_id = wav_src.stem
        wav_dst = out_wav_dir / wav_src.name
        try:
            resample_wav(wav_src, wav_dst)
        except Exception as exc:
            log.warning(f"Failed to resample {wav_src}: {exc}")
            continue

        text = transcripts.get(utt_id, "")
        lines.append(f"{utt_id}\t{text}")
        count += 1

    (out_speaker / "transcript.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info(f"  {speaker_id}: {count} utterances")
    return count

def verify_output(output_dir: Path) -> bool:
    """Verify that all expected speaker directories exist and contain 16kHz audio."""
    ok = True
    for l1, speakers in L2ARCTIC_SPEAKERS.items():
        for spk in speakers:
            spk_dir = output_dir / spk
            wav_dir = spk_dir / "wav"
            transcript = spk_dir / "transcript.txt"

            if not wav_dir.exists():
                log.error(f"Missing: {wav_dir}")
                ok = False
                continue

            wavs = list(wav_dir.glob("*.wav"))
            if not wavs:
                log.error(f"No WAV files in {wav_dir}")
                ok = False
                continue

            try:
                _, sr = torchaudio.load(str(wavs[0]))
                if sr != TARGET_SR:
                    log.error(f"{wavs[0]}: sr={sr}, expected {TARGET_SR}")
                    ok = False
            except Exception as exc:
                log.error(f"Cannot load {wavs[0]}: {exc}")
                ok = False

            if not transcript.exists():
                log.warning(f"Missing transcript: {transcript}")

    if ok:
        log.info("Verification PASSED")
    else:
        log.error("Verification FAILED — see errors above")
    return ok

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare L2-ARCTIC corpus at 16kHz")
    parser.add_argument(
        "--data_dir",
        required=True,
        type=Path,
        help="Path to extracted L2-ARCTIC corpus root",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/l2arctic"),
        help="Where to write resampled data (default: data/l2arctic)",
    )
    parser.add_argument(
        "--speakers",
        nargs="*",
        default=None,
        help="Process only these speaker IDs (default: all 24)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify output structure after processing",
    )
    args = parser.parse_args()

    speakers = args.speakers or ALL_SPEAKERS
    args.output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for spk in speakers:
        total += process_speaker(spk, args.data_dir, args.output_dir)

    log.info(f"Total utterances processed: {total}")

    manifest: dict[str, str] = {}
    for l1, spk_list in L2ARCTIC_SPEAKERS.items():
        for spk in spk_list:
            manifest[spk] = l1
    manifest_path = args.output_dir / "l1_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info(f"L1 manifest written to {manifest_path}")

    if args.verify:
        verify_output(args.output_dir)

if __name__ == "__main__":
    main()
