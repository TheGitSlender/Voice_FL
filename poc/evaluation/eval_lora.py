"""
FedLoRA-MAML final evaluation on L2-ARCTIC meta-test speakers.

Run ONCE after all training is complete. Evaluating meta-test before
training finishes invalidates the experiment.

Baselines (all three required for interpretable results):
  BASELINE 1 — pretrained wav2vec2, k=0 (no adaptation)
  BASELINE 2 — LoRA from scratch, k=5 (same steps as FedLoRA, no meta-training)
  BASELINE 3 — FedLoRA-MAML θ*, k=0 (zero-shot meta-init quality)
  PRIMARY    — FedLoRA-MAML θ*, k=5 (meta-init + adaptation)

Data efficiency curve:
  For each meta-test speaker: vary support set size in {5, 10, 20, 50} utts.
  At each size: k=5 inner steps, evaluate on 50 eval clips.

Statistical reporting:
  Bootstrap 95% CI (1000 resamples) for every WER estimate.
  Report per-speaker — do not aggregate only.
  50 eval clips is a small sample; CIs will be wide. Report them anyway.

Output: evaluation/results/fedlora_maml_l2arctic.json

Usage:
    python evaluation/eval_lora.py
    python evaluation/eval_lora.py --federated_ckpt path/to/theta_star_lora.pt
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.lora_wav2vec2 import LoRAWav2Vec2, load_processor

RESULTS_DIR = ROOT / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SR = 16000
DEFAULT_CKPT = ROOT / "checkpoints" / "lora_federated" / "theta_star_lora.pt"
EVAL_CLIPS_FILE = ROOT / "data" / "l2arctic_eval_clips.json"
SPLIT_FILE = ROOT / "data" / "l2arctic_split.json"
L2ARCTIC_DIR = ROOT / "data" / "l2arctic"

def load_audio(wav_path: Path) -> torch.Tensor:
    """Load a WAV file, return 1D float32 tensor at 16kHz."""
    waveform, sr = torchaudio.load(str(wav_path))
    if sr != TARGET_SR:
        waveform = torchaudio.transforms.Resample(sr, TARGET_SR)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0)

def load_speaker_clips(
    speaker_id: str,
    clip_spec: dict,
    l2arctic_dir: Path,
) -> tuple[list[torch.Tensor], list[str], list[torch.Tensor], list[str]]:
    """
    Load eval and support audio + transcripts for one meta-test speaker.

    Returns:
        (eval_audio, eval_texts, support_audio, support_texts)
    """
    spk_dir = l2arctic_dir / speaker_id
    wav_dir = spk_dir / "wav"
    transcript_file = spk_dir / "transcript.txt"

    transcripts: dict[str, str] = {}
    if transcript_file.exists():
        for line in transcript_file.read_text(encoding="utf-8").splitlines():
            if "\t" in line:
                utt_id, text = line.split("\t", 1)
                transcripts[utt_id] = text.strip()

    def _load_utts(utt_ids: list[str]) -> tuple[list[torch.Tensor], list[str]]:
        audios, texts = [], []
        for uid in utt_ids:
            wav_path = wav_dir / f"{uid}.wav"
            if not wav_path.exists():
                continue
            audios.append(load_audio(wav_path))
            texts.append(transcripts.get(uid, ""))
        return audios, texts

    eval_audio, eval_texts = _load_utts(clip_spec.get("eval_utterances", []))
    support_audio, support_texts = _load_utts(clip_spec.get("support_utterances", []))
    return eval_audio, eval_texts, support_audio, support_texts

def _encode_audio(
    audio: torch.Tensor, processor, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    arr = audio.float().numpy()
    iv = processor(arr, sampling_rate=16000, return_tensors="pt", padding=False)
    return iv.input_values.to(device=device, dtype=dtype)

def decode(logits: torch.Tensor, processor) -> str:
    return processor.batch_decode(torch.argmax(logits, dim=-1))[0]

def compute_wer(hypotheses: list[str], references: list[str]) -> float:
    from jiwer import wer as _wer
    return float(_wer(references, hypotheses))

def bootstrap_wer(
    hypotheses: list[str],
    references: list[str],
    n: int = 1000,
) -> tuple[float, float, float]:
    """Bootstrap 95% CI. Returns (point_estimate, lower, upper)."""
    from jiwer import wer as _wer

    rng = np.random.default_rng(42)
    n_items = len(hypotheses)
    point = float(_wer(references, hypotheses))

    if n_items == 0:
        return point, point, point

    samples = []
    for _ in range(n):
        idx = rng.integers(0, n_items, size=n_items)
        h = [hypotheses[i] for i in idx]
        r = [references[i] for i in idx]
        samples.append(float(_wer(r, h)))

    arr = np.array(samples)
    return point, float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

def evaluate_no_adapt(
    model: LoRAWav2Vec2,
    processor,
    eval_audio: list[torch.Tensor],
    eval_texts: list[str],
    device: torch.device,
) -> tuple[list[str], list[str]]:
    """Run inference with zero adaptation steps."""
    dtype = next(model.model.parameters()).dtype
    model.model.eval()
    hyps: list[str] = []
    with torch.no_grad():
        for audio in eval_audio:
            iv = _encode_audio(audio, processor, device, dtype)
            out = model(iv)
            hyps.append(decode(out.logits, processor))
    return hyps, eval_texts

def evaluate_with_adapt(
    model: LoRAWav2Vec2,
    processor,
    support_audio: list[torch.Tensor],
    support_texts: list[str],
    eval_audio: list[torch.Tensor],
    eval_texts: list[str],
    k: int,
    inner_lr: float,
    device: torch.device,
    support_limit: int | None = None,
) -> tuple[list[str], list[str]]:
    """k inner steps on support set (trimmed to support_limit if given), then evaluate."""
    from ctc.differentiable_ctc import ctc_loss_differentiable
    from maml.engine import _encode_audio as _enc_audio, _encode_labels

    adapted = copy.deepcopy(model)
    dtype = next(adapted.model.parameters()).dtype
    inner_params = adapted.get_outer_loop_params()

    sup_audio = support_audio[:support_limit] if support_limit else support_audio
    sup_texts = support_texts[:support_limit] if support_limit else support_texts

    adapted.model.train()
    for _step in range(k):
        grads_acc = [torch.zeros_like(p) for p in inner_params]
        for audio, text in zip(sup_audio, sup_texts):
            iv = _enc_audio(audio, processor, device, dtype)
            out = adapted(iv)
            lbl = _encode_labels(text, processor, device)
            T = out.logits.shape[1]
            loss = ctc_loss_differentiable(
                out.logits,
                lbl,
                torch.tensor([T], device=device),
                torch.tensor([(lbl[0] != -100).sum().item()], device=device),
                blank=0,
            )
            clip_grads = torch.autograd.grad(
                loss, inner_params, allow_unused=True, create_graph=False
            )
            for acc, g in zip(grads_acc, clip_grads):
                if g is not None:
                    acc.add_(g)

        n = max(len(sup_audio), 1)
        for p, acc in zip(inner_params, grads_acc):
            p.data = p.data - inner_lr * acc / n

    adapted.model.eval()
    hyps: list[str] = []
    with torch.no_grad():
        for audio in eval_audio:
            iv = _enc_audio(audio, processor, device, dtype)
            out = adapted(iv)
            hyps.append(decode(out.logits, processor))

    del adapted
    return hyps, eval_texts

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FedLoRA-MAML evaluation on L2-ARCTIC")
    p.add_argument("--federated_ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--inner_lr", type=float, default=5e-4)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--bootstrap_n", type=int, default=1000)
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=float, default=16.0)
    return p.parse_args()

def load_lora_checkpoint(model: LoRAWav2Vec2, ckpt_path: Path) -> None:
    """Load LoRA + lm_head weights from checkpoint."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                                                                      
    trainable_keys = {n for n, p in model.model.named_parameters() if p.requires_grad}
    if isinstance(sd, dict):
                                                         
        filtered = {k: v for k, v in sd.items() if k in trainable_keys}
        if filtered:
            model.model.load_state_dict(filtered, strict=False)
        else:
            model.model.load_state_dict(sd, strict=False)

def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    eval_clips = json.loads(EVAL_CLIPS_FILE.read_text()) if EVAL_CLIPS_FILE.exists() else {}
    split = json.loads(SPLIT_FILE.read_text())
    meta_test_speakers = list(split["meta_test"].keys())

    print("=" * 60)
    print("FedLoRA-MAML Evaluation — L2-ARCTIC Meta-Test")
    print("=" * 60)
    print(f"Speakers: {meta_test_speakers}")
    print(f"Device: {device} | k={args.k} | inner_lr={args.inner_lr}")
    print()

    processor = load_processor()
    results: dict = {
        "meta_test_speakers": meta_test_speakers,
        "per_speaker": {},
        "communication": {
            "params_per_round": 319_488,
            "bytes_per_round": 319_488 * 4,                    
            "vs_full_perfedavg_bytes": 94_371_712 * 4,
            "reduction_factor": round(94_371_712 / 319_488),
        },
    }

    for speaker_id in meta_test_speakers:
        l1 = split["meta_test"][speaker_id]
        clip_spec = eval_clips.get(speaker_id, {})

        eval_audio, eval_texts, support_audio, support_texts = load_speaker_clips(
            speaker_id, clip_spec, L2ARCTIC_DIR
        )

        if not eval_audio:
            print(f"  {speaker_id}: no audio found — skipping (run download_l2arctic.py first)")
            results["per_speaker"][speaker_id] = {"l1": l1, "error": "no audio"}
            continue

        print(f"  [{speaker_id} ({l1})] eval={len(eval_audio)}, support={len(support_audio)}")
        spk: dict = {"l1": l1}

        print(f"    Baseline 1: pretrained k=0...")
        base_model = LoRAWav2Vec2(device=str(device), r=args.lora_rank, alpha=args.lora_alpha)
        hyps, refs = evaluate_no_adapt(base_model, processor, eval_audio, eval_texts, device)
        pt, lo, hi = bootstrap_wer(hyps, refs, n=args.bootstrap_n)
        spk["WER_pretrained_k0"] = round(pt, 4)
        spk["WER_pretrained_k0_ci"] = [round(lo, 4), round(hi, 4)]
        print(f"      WER={pt:.4f} [{lo:.4f}, {hi:.4f}]")
        del base_model

        print(f"    Baseline 2: LoRA scratch k={args.k}...")
        scratch_model = LoRAWav2Vec2(device=str(device), r=args.lora_rank, alpha=args.lora_alpha)
        hyps, refs = evaluate_with_adapt(
            scratch_model, processor, support_audio, support_texts,
            eval_audio, eval_texts, args.k, args.inner_lr, device,
        )
        pt, lo, hi = bootstrap_wer(hyps, refs, n=args.bootstrap_n)
        spk[f"WER_scratch_k{args.k}"] = round(pt, 4)
        spk[f"WER_scratch_k{args.k}_ci"] = [round(lo, 4), round(hi, 4)]
        print(f"      WER={pt:.4f} [{lo:.4f}, {hi:.4f}]")
        del scratch_model

        if args.federated_ckpt.exists():
                                                     
            print(f"    Baseline 3: FedLoRA θ* k=0...")
            fed_model = LoRAWav2Vec2(device=str(device), r=args.lora_rank, alpha=args.lora_alpha)
            load_lora_checkpoint(fed_model, args.federated_ckpt)
            hyps, refs = evaluate_no_adapt(fed_model, processor, eval_audio, eval_texts, device)
            pt, lo, hi = bootstrap_wer(hyps, refs, n=args.bootstrap_n)
            spk["WER_fedlora_k0"] = round(pt, 4)
            spk["WER_fedlora_k0_ci"] = [round(lo, 4), round(hi, 4)]
            print(f"      WER={pt:.4f} [{lo:.4f}, {hi:.4f}]")

            print(f"    Primary: FedLoRA θ* k={args.k}...")
            hyps, refs = evaluate_with_adapt(
                fed_model, processor, support_audio, support_texts,
                eval_audio, eval_texts, args.k, args.inner_lr, device,
            )
            pt, lo, hi = bootstrap_wer(hyps, refs, n=args.bootstrap_n)
            spk[f"WER_fedlora_k{args.k}"] = round(pt, 4)
            spk[f"WER_fedlora_k{args.k}_ci"] = [round(lo, 4), round(hi, 4)]
            print(f"      WER={pt:.4f} [{lo:.4f}, {hi:.4f}]")

            print(f"    Data efficiency curve...")
            efficiency: dict[str, float] = {}
            for n_support in [5, 10, 20, 50]:
                if n_support > len(support_audio):
                    break
                eff_model = LoRAWav2Vec2(
                    device=str(device), r=args.lora_rank, alpha=args.lora_alpha
                )
                load_lora_checkpoint(eff_model, args.federated_ckpt)
                hyps, refs = evaluate_with_adapt(
                    eff_model, processor, support_audio, support_texts,
                    eval_audio, eval_texts, args.k, args.inner_lr, device,
                    support_limit=n_support,
                )
                wer, _, _ = bootstrap_wer(hyps, refs, n=args.bootstrap_n)
                efficiency[f"{n_support}_utts"] = round(wer, 4)
                print(f"      {n_support} utts: WER={wer:.4f}")
                del eff_model

            spk["data_efficiency"] = efficiency
            del fed_model
        else:
            print(f"    WARNING: ckpt not found ({args.federated_ckpt}) — skipping FedLoRA baselines")
            spk["WER_fedlora_k0"] = None
            spk[f"WER_fedlora_k{args.k}"] = None

        results["per_speaker"][speaker_id] = spk

    out_path = RESULTS_DIR / "fedlora_maml_l2arctic.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for spk_id, spk_data in results["per_speaker"].items():
        if "error" in spk_data:
            continue
        k = args.k
        pre_k0 = spk_data.get("WER_pretrained_k0", "N/A")
        fed_kn = spk_data.get(f"WER_fedlora_k{k}", "N/A")
        scr_kn = spk_data.get(f"WER_scratch_k{k}", "N/A")
        print(
            f"  {spk_id:6} ({spk_data['l1']:2}) "
            f"pretrained_k0={pre_k0:.4f}  "
            f"scratch_k{k}={scr_kn:.4f}  "
            f"fedlora_k{k}={fed_kn if fed_kn is None else f'{fed_kn:.4f}'}"
        )

if __name__ == "__main__":
    main()
