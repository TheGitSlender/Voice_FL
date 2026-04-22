"""
Pre-training data insights.

Run ALL four insight analyses before any MAML training begins.
Results inform dataset choice, hyperparameter ranges, and evaluation framing.

Insights generated:
  1. VCTK vs LibriSpeech domain shift — how hard is VCTK for pretrained wav2vec2?
  2. Per-speaker WER variance — which speakers are hardest for the pretrained model?
  3. Adaptation curve without MAML — does simple fine-tuning saturate quickly?
  4. Support set size sensitivity — how many clips does a new speaker need?

Output files:
  data/insights/domain_shift.json
  data/insights/speaker_variance.json
  data/insights/pretrained_adaptation_curve.json
  data/insights/support_size_sensitivity.json

Usage:
    python data/insights/run_insights.py --device cpu
    python data/insights/run_insights.py --device cuda --skip 3 4  (skip slow insights)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

INSIGHTS_DIR = Path(__file__).parent
INSIGHTS_DIR.mkdir(parents=True, exist_ok=True)

EVAL_CLIPS = 20                                                  
SUPPORT_CLIPS = 8                                
K_VALUES = [1, 3, 5, 10]
K_SIZES = [2, 4, 8, 16]

def compute_wer_no_adaptation(model, processor, audio_clips, texts, device) -> float:
    """WER with no adaptation and no perturbation."""
    from jiwer import wer as _wer
    import torch
    import copy

    m = copy.deepcopy(model)
    m.model.eval()
    dtype = next(m.model.parameters()).dtype
    hypotheses = []
    with torch.no_grad():
        for audio in audio_clips:
            arr = audio.float().numpy()
            iv = processor(arr, sampling_rate=16000, return_tensors="pt",
                           padding=False).input_values.to(device=device, dtype=dtype)
            out = m.model(input_values=iv)
            pred_ids = torch.argmax(out.logits, dim=-1)
            hypotheses.append(processor.batch_decode(pred_ids)[0])

    return float(_wer(texts, hypotheses))

def compute_wer_adapted(
    model,
    processor,
    support_audio,
    support_texts,
    query_audio,
    query_texts,
    k: int,
    inner_lr: float,
    device,
) -> float:
    """WER after k adaptation steps on support, evaluated on query."""
    from jiwer import wer as _wer
    from maml.engine import _accumulate_grads_over_clips, _encode_audio
    import torch
    import copy

    m = copy.deepcopy(model)
    m.model.train()
    dtype = next(m.model.parameters()).dtype
    lm_params = list(m.model.lm_head.parameters())

    for _ in range(k):
        grads, _ = _accumulate_grads_over_clips(
            m.model, support_audio, support_texts, lm_params, processor, device, dtype
        )
        for p, g in zip(lm_params, grads):
            p.data = p.data - inner_lr * g
        del grads

    m.model.eval()
    hypotheses = []
    with torch.no_grad():
        for audio in query_audio:
            arr = audio.float().numpy()
            iv = processor(arr, sampling_rate=16000, return_tensors="pt",
                           padding=False).input_values.to(device=device, dtype=dtype)
            out = m.model(input_values=iv)
            pred_ids = torch.argmax(out.logits, dim=-1)
            hypotheses.append(processor.batch_decode(pred_ids)[0])

    return float(_wer(query_texts, hypotheses))

def run_insight_1_domain_shift(model, processor, meta_val_dirs: list[Path], device) -> dict:
    """Insight 1: VCTK vs LibriSpeech WER for pretrained model (no adaptation)."""
    print("\n[Insight 1: Domain shift VCTK vs LibriSpeech]")
    import torch

    vctk_wers = []
    for node_dir in meta_val_dirs[:2]:
        tensors = torch.load(node_dir / "features.pt", weights_only=True)
        texts = (node_dir / "labels.txt").read_text().splitlines()
        clips = tensors[:10]
        clip_texts = texts[:10]
        if len(clips) < 2:
            continue
        wer = compute_wer_no_adaptation(model, processor, clips, clip_texts, device)
        vctk_wers.append(wer)
        print(f"  {node_dir.name[:8]}: VCTK WER = {wer:.4f}")

    mean_vctk = float(np.mean(vctk_wers)) if vctk_wers else None

    libri_wer = None
    try:
        from datasets import load_dataset
        print("  Loading LibriSpeech test-clean (10 clips)...")
        libri = load_dataset("openslr/librispeech_asr", "clean",
                             split="test", trust_remote_code=True)
        import torchaudio
        resampler = None
        libri_clips = []
        libri_texts = []
        for item in libri.select(range(10)):
            sr = item["audio"].get("sampling_rate", 16000)
            arr = item["audio"]["array"].astype("float32")
            if sr != 16000:
                if resampler is None:
                    resampler = torchaudio.transforms.Resample(sr, 16000)
                import torch as _torch
                arr = resampler(_torch.tensor(arr).unsqueeze(0)).squeeze(0).numpy()
            max_val = abs(arr).max()
            arr = arr / (max_val + 1e-8)
            libri_clips.append(torch.tensor(arr))
            libri_texts.append(item["text"].upper().strip())
        libri_wer = compute_wer_no_adaptation(model, processor, libri_clips, libri_texts, device)
        print(f"  LibriSpeech test-clean WER (k=0): {libri_wer:.4f}")
    except Exception as exc:
        print(f"  LibriSpeech comparison skipped ({exc})")

    result = {
        "vctk_mean_wer_k0": mean_vctk,
        "librispeech_wer_k0": libri_wer,
        "per_node_vctk_wer": vctk_wers,
        "finding": (
            "Large domain shift — personalization strongly motivated"
            if (mean_vctk or 0) > 0.15
            else "Moderate or low domain shift — investigate whether personalization is necessary"
        ),
    }

    if mean_vctk is not None and mean_vctk < 0.05:
        result["warning"] = (
            "VCTK WER < 5% with pretrained model (no adaptation). "
            "Personalization may have little room to improve. "
            "Document this finding and reassess whether the PoC claim is achievable."
        )

    out = INSIGHTS_DIR / "domain_shift.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"  Saved: {out}")
    return result

def run_insight_2_speaker_variance(
    model, processor, all_node_dirs: list[Path], device
) -> dict:
    """Insight 2: Per-speaker WER distribution for pretrained model."""
    print("\n[Insight 2: Speaker WER variance]")
    import torch

    speaker_wers = {}
    for node_dir in all_node_dirs:
        tensors = torch.load(node_dir / "features.pt", weights_only=True)
        texts = (node_dir / "labels.txt").read_text().splitlines()
        clips = tensors[:20]
        clip_texts = texts[:20]
        if len(clips) < 2:
            continue
        wer = compute_wer_no_adaptation(model, processor, clips, clip_texts, device)
        speaker_wers[node_dir.name] = float(wer)
        print(f"  {node_dir.name[:8]}: WER = {wer:.4f}")

    wer_values = list(speaker_wers.values())
    result = {
        "per_speaker_wer": speaker_wers,
        "mean_wer": float(np.mean(wer_values)),
        "std_wer": float(np.std(wer_values)),
        "min_wer": float(np.min(wer_values)),
        "max_wer": float(np.max(wer_values)),
        "finding": (
            f"WER range [{np.min(wer_values):.3f}, {np.max(wer_values):.3f}] — "
            f"{'high' if np.std(wer_values) > 0.1 else 'low'} variance across speakers"
        ),
    }

    out = INSIGHTS_DIR / "speaker_variance.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"  Saved: {out}")
    return result

def run_insight_3_pretrained_adaptation_curve(
    model, processor, meta_val_dirs: list[Path], inner_lr: float, device
) -> dict:
    """Insight 3: Fine-tuning adaptation curve (no MAML) on meta-val speakers."""
    print("\n[Insight 3: Pretrained model adaptation curve (no MAML)]")
    import torch

    all_curves = {}
    for node_dir in meta_val_dirs:
        tensors = torch.load(node_dir / "features.pt", weights_only=True)
        texts = (node_dir / "labels.txt").read_text().splitlines()

        if len(tensors) < SUPPORT_CLIPS + EVAL_CLIPS:
            print(f"  {node_dir.name[:8]}: not enough clips, skipping")
            continue

        support = tensors[:SUPPORT_CLIPS]
        support_texts = texts[:SUPPORT_CLIPS]
        query = tensors[SUPPORT_CLIPS: SUPPORT_CLIPS + EVAL_CLIPS]
        query_texts = texts[SUPPORT_CLIPS: SUPPORT_CLIPS + EVAL_CLIPS]

        curve = {}
                      
        wer0 = compute_wer_no_adaptation(model, processor, query, query_texts, device)
        curve[0] = wer0

        for k in K_VALUES:
            wer_k = compute_wer_adapted(
                model, processor, support, support_texts,
                query, query_texts, k, inner_lr, device,
            )
            curve[k] = wer_k

        all_curves[node_dir.name] = {str(k): v for k, v in curve.items()}
        print(f"  {node_dir.name[:8]}: k=0 → {wer0:.4f}, " +
              "  ".join(f"k={k} → {curve[k]:.4f}" for k in K_VALUES))

    saturation_k = None
    if all_curves:
        wers_by_k: dict[int, list] = {0: [], **{k: [] for k in K_VALUES}}
        for curve in all_curves.values():
            for k in [0] + K_VALUES:
                wers_by_k[k].append(float(curve[str(k)]))
        mean_by_k = {k: float(np.mean(vs)) for k, vs in wers_by_k.items()}

        prev = mean_by_k[0]
        for k in K_VALUES:
            curr = mean_by_k[k]
            rel_improve = (prev - curr) / (prev + 1e-8)
            if rel_improve < 0.01:
                saturation_k = k
                break
            prev = curr

    result = {
        "per_speaker_curves": all_curves,
        "inner_lr_used": inner_lr,
        "saturation_k": saturation_k,
        "finding": (
            f"Adaptation saturates around k={saturation_k} without MAML"
            if saturation_k else "No clear saturation observed in k range tested"
        ),
        "implication": (
            "MAML's primary benefit is faster early adaptation (lower WER at small k), "
            "not a higher ceiling" if saturation_k and saturation_k <= 5
            else "Fine-tuning still improving at k=10 — MAML may raise the ceiling"
        ),
    }

    out = INSIGHTS_DIR / "pretrained_adaptation_curve.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"  Saved: {out}")
    return result

def run_insight_4_support_size_sensitivity(
    model, processor, meta_val_dirs: list[Path], inner_lr: float, device
) -> dict:
    """Insight 4: How many support clips does a new user need?

    Compares WER at k=5 across different support set sizes K=[2,4,8,16].
    Run AFTER meta-training (the model arg should be the trained θ* if available,
    or pretrained model as a lower bound).
    """
    print("\n[Insight 4: Support set size sensitivity]")
    import torch

    all_results = {}
    for node_dir in meta_val_dirs:
        tensors = torch.load(node_dir / "features.pt", weights_only=True)
        texts = (node_dir / "labels.txt").read_text().splitlines()

        max_support_needed = max(K_SIZES)
        if len(tensors) < max_support_needed + EVAL_CLIPS:
            print(f"  {node_dir.name[:8]}: not enough clips, skipping")
            continue

        query = tensors[max_support_needed: max_support_needed + EVAL_CLIPS]
        query_texts = texts[max_support_needed: max_support_needed + EVAL_CLIPS]

        size_results = {}
        for k_size in K_SIZES:
            support = tensors[:k_size]
            support_texts = texts[:k_size]
            wer = compute_wer_adapted(
                model, processor, support, support_texts,
                query, query_texts, k=5, inner_lr=inner_lr, device=device,
            )
            size_results[k_size] = float(wer)

        all_results[node_dir.name] = size_results
        print(f"  {node_dir.name[:8]}: " +
              "  ".join(f"K={ks} → {size_results[ks]:.4f}" for ks in K_SIZES))

    result = {
        "per_speaker": all_results,
        "k_values_tested": K_SIZES,
        "inner_steps": 5,
        "inner_lr_used": inner_lr,
        "finding": "See per-speaker table — WER at K=2 vs K=8 shows data efficiency",
    }

    out = INSIGHTS_DIR / "support_size_sensitivity.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"  Saved: {out}")
    return result

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--inner_lr", type=float, default=1e-4)
    parser.add_argument("--skip", nargs="+", type=int, default=[],
                        help="Insight numbers to skip (e.g. --skip 3 4)")
    parser.add_argument("--meta_split", default=None,
                        help="Path to meta_split.json (default: data/meta_split.json)")
    args = parser.parse_args()

    import torch
    from models.wav2vec2_maml import Wav2Vec2MAML, load_processor

    device = torch.device(args.device)

    meta_split_path = Path(args.meta_split) if args.meta_split else ROOT / "data" / "meta_split.json"
    nodes_dir = ROOT / "data" / "nodes"

    meta_val_hashes: set[str] = set()
    if meta_split_path.exists():
        meta_split = json.loads(meta_split_path.read_text())
        meta_val_hashes = set(meta_split.get("meta_val", []))
        train_hashes = set(meta_split.get("meta_train", []))
        all_training_hashes = meta_val_hashes | train_hashes
        print(f"Loaded meta_split: {len(meta_val_hashes)} meta-val speakers")
    else:
        print("WARNING: meta_split.json not found — using all nodes in data/nodes/")
        all_training_hashes = None

    all_node_dirs = sorted(d for d in nodes_dir.iterdir() if d.is_dir()
                           if (d / "features.pt").exists())

    if all_training_hashes is not None:
        meta_val_dirs = [d for d in all_node_dirs if d.name in meta_val_hashes]
    else:
        meta_val_dirs = all_node_dirs

    if not meta_val_dirs:
        print(f"ERROR: No meta-val node directories found under {nodes_dir}")
        print("Run: python data/prepare_vctk.py && python data/features.py")
        sys.exit(1)

    print(f"Meta-val nodes: {len(meta_val_dirs)}")
    print(f"All training nodes: {len(all_node_dirs)}")
    print(f"Device: {device}\n")

    print("Loading pretrained wav2vec2-base-960h (no meta-training)...")
    model = Wav2Vec2MAML(device=device)
    processor = load_processor()

    if 1 not in args.skip:
        run_insight_1_domain_shift(model, processor, meta_val_dirs, device)

    if 2 not in args.skip:
        run_insight_2_speaker_variance(model, processor, all_node_dirs, device)

    if 3 not in args.skip:
        run_insight_3_pretrained_adaptation_curve(
            model, processor, meta_val_dirs, args.inner_lr, device
        )

    if 4 not in args.skip:
        run_insight_4_support_size_sensitivity(
            model, processor, meta_val_dirs, args.inner_lr, device
        )

    print("\n=== Insights complete ===")
    print(f"Output files in: {INSIGHTS_DIR}")
    print("\nReview domain_shift.json before proceeding.")
    print("If VCTK WER < 5% with pretrained model, document this finding and reassess.")

if __name__ == "__main__":
    main()
