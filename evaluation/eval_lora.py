"""
FedLoRA-MAML final evaluation on L2-ARCTIC meta-test speakers.

Run ONCE after all training is complete. Evaluating meta-test before
training finishes invalidates the experiment.

Baselines (all required for interpretable results):
  Baseline A  (ablation): wav2vec2-base-960h, k=0 and k=5
              English CTC fine-tuned. Strong upper-bound reference.
  Baseline B  (ablation): wav2vec2-base (SSL-only), k=0 and k=5
              No CTC fine-tuning. Expected blank collapse (WER≈1.0).
  Baseline C  (PRIMARY): wav2vec2-base-100h, k=0 and k=5
              Same backbone FedLoRA-MAML was trained from, but WITHOUT
              any federation. This is the fair comparison: does 20 rounds
              of federated meta-training beat just using the 100h model
              directly?
  Federated:  FedLoRA θ* (trained from wav2vec2-base-100h), k=0 and k=5
              PRIMARY result — meta-init from 100h baseline.

Primary comparison: WER_federated_k5 vs WER_100h_k5
  (same starting point; only difference is federated meta-training)
Ablation A:   WER_federated_k5 vs WER_960h_k5
  (meta-from-100h vs strong 960h English CTC baseline)

Reporting:
  Per-speaker AND per-accent-group (L1) breakdown.
  Bootstrap 95% CI (1000 resamples) for every WER estimate.
  20 eval clips is a small sample; CIs will be wide. Report them anyway.

Data format:
  Reads from data/l2arctic_test_nodes/{speaker_id}/features.pt (list of 1D float32
  tensors, 16 kHz, [-1, 1]) and labels.txt (one uppercase transcription per line).
  Clips are shuffled with a fixed seed; first 20 → eval, next 50 → support.

Output: evaluation/results/fedlora_maml_l2arctic.json

Usage:
    python evaluation/eval_lora.py
    python evaluation/eval_lora.py --federated_ckpt checkpoints/federated/theta_star_lora_round_0040.pt
    python evaluation/eval_lora.py --device cpu   # if no GPU
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.lora_wav2vec2 import LoRAWav2Vec2, load_processor

RESULTS_DIR = ROOT / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CKPT    = ROOT / "checkpoints" / "federated" / "theta_star_lora_round_0040.pt"
SPLIT_FILE      = ROOT / "data" / "l2arctic_split.json"
VCTK_TEST_DIR   = ROOT / "data" / "l2arctic_test_nodes"

N_SUPPORT = 50   # support clips for inner-loop adaptation
EVAL_SEED = 0    # fixed seed for eval/support split — never change after first run
_SAMPLE_RATE = 16_000


# ── Data loading ──────────────────────────────────────────────────────────────

def load_speaker_clips(
    speaker_id: str,
    node_dir: Path,
) -> tuple[list[torch.Tensor], list[str], list[torch.Tensor], list[str]]:
    """
    Load eval and support clips for one VCTK meta-test speaker.

    Reads features.pt (list of 1D float32 tensors) + labels.txt.
    Shuffles with EVAL_SEED, takes:
      - first N_SUPPORT indices → support set
      - all remaining indices  → eval set  (typically ~100 clips)
    The two sets never overlap.

    Returns: (eval_audio, eval_texts, support_audio, support_texts)
    """
    node_path = node_dir / speaker_id
    features: list[torch.Tensor] = torch.load(
        node_path / "features.pt", map_location="cpu", weights_only=False
    )
    labels = (node_path / "labels.txt").read_text(encoding="utf-8").strip().splitlines()

    n_clips = min(len(features), len(labels))
    rng = np.random.default_rng(EVAL_SEED)
    indices = rng.permutation(n_clips)

    support_idx = indices[:N_SUPPORT]
    eval_idx    = indices[N_SUPPORT:]

    eval_audio   = [features[i] for i in eval_idx]
    eval_texts   = [labels[i].upper().strip() for i in eval_idx]
    sup_audio    = [features[i] for i in support_idx]
    sup_texts    = [labels[i].upper().strip() for i in support_idx]

    return eval_audio, eval_texts, sup_audio, sup_texts


# ── Inference helpers ─────────────────────────────────────────────────────────

def _encode_audio(
    audio: torch.Tensor,
    processor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    arr = audio.float().numpy()
    iv = processor(arr, sampling_rate=_SAMPLE_RATE, return_tensors="pt", padding=False)
    return iv.input_values.to(device=device, dtype=dtype)


def decode(logits: torch.Tensor, processor) -> str:
    return processor.batch_decode(torch.argmax(logits, dim=-1))[0]


def bootstrap_wer(
    hypotheses: list[str],
    references: list[str],
    n: int = 1000,
) -> tuple[float, float, float]:
    """Returns (point_estimate, lower_95, upper_95)."""
    from jiwer import wer as _wer
    rng = np.random.default_rng(42)
    n_items = len(hypotheses)
    point = float(_wer(references, hypotheses))
    if n_items == 0:
        return point, point, point
    samples = []
    for _ in range(n):
        idx = rng.integers(0, n_items, size=n_items)
        samples.append(float(_wer([references[i] for i in idx], [hypotheses[i] for i in idx])))
    arr = np.array(samples)
    return point, float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


# ── Evaluation modes ──────────────────────────────────────────────────────────

def evaluate_no_adapt(
    model: LoRAWav2Vec2,
    processor,
    eval_audio: list[torch.Tensor],
    eval_texts: list[str],
    device: torch.device,
) -> tuple[list[str], list[str]]:
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
    """Run k inner-loop steps on support set, then evaluate on eval set.

    Uses the same normalized custom CTC loss as training (ctc_loss_differentiable
    divided by label count) so that inner_lr=5e-3 has the same effective scale
    as during MAML meta-training.
    """
    from maml.engine import _encode_audio as _enc_audio, _encode_labels
    from ctc.differentiable_ctc import ctc_loss_differentiable

    adapted = copy.deepcopy(model)
    dtype = next(adapted.model.parameters()).dtype
    inner_params = adapted.get_outer_loop_params()

    sup_audio = support_audio[:support_limit] if support_limit else support_audio
    sup_texts = support_texts[:support_limit] if support_limit else support_texts

    adapted.model.eval()
    for _step in range(k):
        grads_acc = [torch.zeros_like(p) for p in inner_params]
        for audio, text in zip(sup_audio, sup_texts):
            iv = _enc_audio(audio, processor, device, dtype)
            lbl = _encode_labels(text, processor, device)
            S = int((lbl[0] != -100).sum().item())
            if S == 0:
                continue
            out = adapted.model(input_values=iv)
            logits = out.logits
            T_frames = logits.shape[1]
            loss = ctc_loss_differentiable(
                logits, lbl,
                torch.tensor([T_frames], device=device),
                torch.tensor([S], device=device),
                blank=0,
            ) / max(S, 1)
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


# ── Checkpoint loading ────────────────────────────────────────────────────────

def load_lora_checkpoint(model: LoRAWav2Vec2, ckpt_path: Path) -> None:
    """Load LoRA + lm_head weights from a federated checkpoint."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    trainable_keys = {n for n, p in model.model.named_parameters() if p.requires_grad}
    if isinstance(sd, dict):
        filtered = {k: v for k, v in sd.items() if k in trainable_keys}
        model.model.load_state_dict(filtered if filtered else sd, strict=False)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FedLoRA-MAML evaluation on L2-ARCTIC meta-test")
    p.add_argument("--federated_ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--test_nodes_dir", type=Path, default=VCTK_TEST_DIR)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--inner_lr", type=float, default=5e-3)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--bootstrap_n", type=int, default=1000)
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=float, default=16.0)
    # Ablation baselines
    p.add_argument("--base_model_960h", default="facebook/wav2vec2-base-960h",
                   help="Baseline A: strong English CTC baseline (960h)")
    p.add_argument("--base_model_ssl", default="facebook/wav2vec2-base",
                   help="Baseline B: SSL-only, no CTC fine-tuning")
    p.add_argument("--base_model_100h", default="facebook/wav2vec2-base-100h",
                   help="Baseline C / Fed starting point: CTC fine-tuned on 100h")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    split = json.loads(SPLIT_FILE.read_text())
    raw = split["meta_test"]
    # raw is {speaker_id: l1_code} for L2-ARCTIC
    speaker_l1: dict[str, str] = raw if isinstance(raw, dict) else {s: "?" for s in raw}
    meta_test_speakers: list[str] = list(speaker_l1.keys())

    # Accent label expansions for readable output
    L1_NAMES = {"hi": "Hindi", "vi": "Vietnamese", "zh": "Mandarin",
                "ko": "Korean", "es": "Spanish", "ar": "Arabic"}

    print("=" * 60)
    print("FedLoRA-MAML Evaluation — L2-ARCTIC Meta-Test")
    print("=" * 60)
    for spk, l1 in speaker_l1.items():
        print(f"  {spk:<8}  L1: {l1}  ({L1_NAMES.get(l1, l1)})")
    print(f"\nCheckpoint   : {args.federated_ckpt}")
    print(f"Test nodes   : {args.test_nodes_dir}")
    print(f"Device       : {device}  |  k={args.k}  |  inner_lr={args.inner_lr}")
    print(f"Eval clips   : all non-support clips per speaker  |  Support pool: {N_SUPPORT}")
    print()

    # Pre-flight: meta-test nodes must exist before we start
    missing = [
        s for s in meta_test_speakers
        if not (args.test_nodes_dir / s / "features.pt").exists()
    ]
    if missing:
        print(f"ERROR: meta-test nodes not found for: {missing}")
        print(f"  Run first:")
        print(f"    python data/prepare_l2arctic_lora.py --split meta_test --output_dir {args.test_nodes_dir}")
        sys.exit(1)

    if not args.federated_ckpt.exists():
        print(f"WARNING: federated checkpoint not found: {args.federated_ckpt}")
        print("  Federated baselines will be skipped.")

    processor = load_processor()
    results: dict = {
        "meta_test_speakers": meta_test_speakers,
        "eval_config": {
            "k": args.k,
            "inner_lr": args.inner_lr,
            "n_eval_clips": "all_non_support",
            "n_support_clips": N_SUPPORT,
            "eval_seed": EVAL_SEED,
            "federated_ckpt": str(args.federated_ckpt),
        },
        "communication_efficiency": {
            "params_per_round": 319_488,
            "bytes_per_round": 319_488 * 4,
            "vs_full_perfedavg_bytes": 94_371_712 * 4,
            "reduction_factor": round(94_371_712 / 319_488),
        },
        "per_speaker": {},
    }

    for speaker_id in meta_test_speakers:
        print(f"\n{'─' * 50}")
        print(f"Speaker: {speaker_id}")

        eval_audio, eval_texts, support_audio, support_texts = load_speaker_clips(
            speaker_id, args.test_nodes_dir
        )
        print(f"  Loaded: {len(eval_audio)} eval clips, {len(support_audio)} support clips")
        spk: dict = {}
        k = args.k

        # ── Baseline A: wav2vec2-base-960h (English CTC fine-tuned) ───────────
        print(f"  [A1] WER_960h_k0   (Baseline A — CTC fine-tuned, no adapt)...")
        m = LoRAWav2Vec2(device=str(device), r=args.lora_rank, alpha=args.lora_alpha,
                         model_name=args.base_model_960h)
        hyps, refs = evaluate_no_adapt(m, processor, eval_audio, eval_texts, device)
        pt, lo, hi = bootstrap_wer(hyps, refs, n=args.bootstrap_n)
        spk["WER_960h_k0"]    = round(pt, 4)
        spk["WER_960h_k0_ci"] = [round(lo, 4), round(hi, 4)]
        print(f"       WER={pt:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
        del m

        print(f"  [A2] WER_960h_k{k}  (Baseline A — CTC fine-tuned, k={k} adapt)...")
        m = LoRAWav2Vec2(device=str(device), r=args.lora_rank, alpha=args.lora_alpha,
                         model_name=args.base_model_960h)
        hyps, refs = evaluate_with_adapt(m, processor, support_audio, support_texts,
                                         eval_audio, eval_texts, k, args.inner_lr, device)
        pt, lo, hi = bootstrap_wer(hyps, refs, n=args.bootstrap_n)
        spk[f"WER_960h_k{k}"]    = round(pt, 4)
        spk[f"WER_960h_k{k}_ci"] = [round(lo, 4), round(hi, 4)]
        print(f"       WER={pt:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
        del m

        # ── Baseline B: wav2vec2-base SSL-only (ablation — expected collapse) ───
        print(f"  [B1] WER_base_k0   (Baseline B — SSL-only, no adapt)...")
        m = LoRAWav2Vec2(device=str(device), r=args.lora_rank, alpha=args.lora_alpha,
                         model_name=args.base_model_ssl)
        hyps, refs = evaluate_no_adapt(m, processor, eval_audio, eval_texts, device)
        pt, lo, hi = bootstrap_wer(hyps, refs, n=args.bootstrap_n)
        spk["WER_base_k0"]    = round(pt, 4)
        spk["WER_base_k0_ci"] = [round(lo, 4), round(hi, 4)]
        print(f"       WER={pt:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
        del m

        print(f"  [B2] WER_base_k{k}  (Baseline B — SSL-only, k={k} adapt)...")
        m = LoRAWav2Vec2(device=str(device), r=args.lora_rank, alpha=args.lora_alpha,
                         model_name=args.base_model_ssl)
        hyps, refs = evaluate_with_adapt(m, processor, support_audio, support_texts,
                                         eval_audio, eval_texts, k, args.inner_lr, device)
        pt, lo, hi = bootstrap_wer(hyps, refs, n=args.bootstrap_n)
        spk[f"WER_base_k{k}"]    = round(pt, 4)
        spk[f"WER_base_k{k}_ci"] = [round(lo, 4), round(hi, 4)]
        print(f"       WER={pt:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
        del m

        # ── Baseline C: wav2vec2-base-100h (PRIMARY comparison to federated) ──
        # Same backbone as FedLoRA-MAML training, zero federation.
        # The fair test: does federated meta-training beat the 100h starting point?
        print(f"  [C1] WER_100h_k0   (Baseline C — 100h CTC, no adapt)...")
        m = LoRAWav2Vec2(device=str(device), r=args.lora_rank, alpha=args.lora_alpha,
                         model_name=args.base_model_100h)
        hyps, refs = evaluate_no_adapt(m, processor, eval_audio, eval_texts, device)
        pt, lo, hi = bootstrap_wer(hyps, refs, n=args.bootstrap_n)
        spk["WER_100h_k0"]    = round(pt, 4)
        spk["WER_100h_k0_ci"] = [round(lo, 4), round(hi, 4)]
        print(f"       WER={pt:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
        del m

        print(f"  [C2] WER_100h_k{k}  (Baseline C — 100h CTC, k={k} adapt)...")
        m = LoRAWav2Vec2(device=str(device), r=args.lora_rank, alpha=args.lora_alpha,
                         model_name=args.base_model_100h)
        hyps, refs = evaluate_with_adapt(m, processor, support_audio, support_texts,
                                         eval_audio, eval_texts, k, args.inner_lr, device)
        pt, lo, hi = bootstrap_wer(hyps, refs, n=args.bootstrap_n)
        spk[f"WER_100h_k{k}"]    = round(pt, 4)
        spk[f"WER_100h_k{k}_ci"] = [round(lo, 4), round(hi, 4)]
        print(f"       WER={pt:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
        del m

        # ── Federated: θ* trained from wav2vec2-base-100h ────────────────────
        if args.federated_ckpt.exists():
            fed_model = LoRAWav2Vec2(device=str(device), r=args.lora_rank, alpha=args.lora_alpha,
                                     model_name=args.base_model_100h)
            load_lora_checkpoint(fed_model, args.federated_ckpt)

            print(f"  [F1] WER_federated_k0  (θ* zero-shot)...")
            hyps, refs = evaluate_no_adapt(fed_model, processor, eval_audio, eval_texts, device)
            pt, lo, hi = bootstrap_wer(hyps, refs, n=args.bootstrap_n)
            spk["WER_federated_k0"]    = round(pt, 4)
            spk["WER_federated_k0_ci"] = [round(lo, 4), round(hi, 4)]
            print(f"       WER={pt:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

            print(f"  [F2] WER_federated_k{k}  (θ* + adaptation, PRIMARY)...")
            hyps, refs = evaluate_with_adapt(
                fed_model, processor, support_audio, support_texts,
                eval_audio, eval_texts, k, args.inner_lr, device,
            )
            pt, lo, hi = bootstrap_wer(hyps, refs, n=args.bootstrap_n)
            spk[f"WER_federated_k{k}"]    = round(pt, 4)
            spk[f"WER_federated_k{k}_ci"] = [round(lo, 4), round(hi, 4)]
            print(f"       WER={pt:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

            print(f"  [eff] Data efficiency curve (support sizes: 5, 10, 20, 50)...")
            efficiency: dict[str, float] = {}
            for n_sup in [5, 10, 20, 50]:
                if n_sup > len(support_audio):
                    break
                eff_model = LoRAWav2Vec2(device=str(device), r=args.lora_rank,
                                         alpha=args.lora_alpha, model_name=args.base_model_100h)
                load_lora_checkpoint(eff_model, args.federated_ckpt)
                hyps, refs = evaluate_with_adapt(
                    eff_model, processor, support_audio, support_texts,
                    eval_audio, eval_texts, k, args.inner_lr, device, support_limit=n_sup,
                )
                wer_pt, _, _ = bootstrap_wer(hyps, refs, n=args.bootstrap_n)
                efficiency[f"{n_sup}_utts"] = round(wer_pt, 4)
                print(f"       {n_sup:2d} utts → WER={wer_pt:.4f}")
                del eff_model

            spk["data_efficiency"] = efficiency
            del fed_model
        else:
            print(f"  [F1/F2] Federated baselines SKIPPED — checkpoint not found")
            spk["WER_federated_k0"]         = None
            spk[f"WER_federated_k{k}"]      = None

        spk["l1"] = speaker_l1[speaker_id]
        results["per_speaker"][speaker_id] = spk

    # ── Per-accent aggregation ────────────────────────────────────────────────
    # Each accent has exactly 1 test speaker, so this is a relabeling + grouping
    # structure that scales cleanly if more test speakers are added later.
    k = args.k
    accent_groups: dict[str, dict] = {}
    for spk_id, spk_data in results["per_speaker"].items():
        l1 = spk_data["l1"]
        if l1 not in accent_groups:
            accent_groups[l1] = {"speakers": [], "l1_name": L1_NAMES.get(l1, l1)}
        grp = accent_groups[l1]
        grp["speakers"].append(spk_id)
        # Aggregate WER metrics: mean across speakers in group
        for key in [f"WER_100h_k0", f"WER_100h_k{k}",
                    f"WER_federated_k0", f"WER_federated_k{k}"]:
            val = spk_data.get(key)
            if val is not None:
                grp.setdefault(f"{key}_vals", []).append(val)
                grp[key] = round(float(np.mean(grp[f"{key}_vals"])), 4)

    results["per_accent"] = accent_groups

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = RESULTS_DIR / "fedlora_maml_l2arctic.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out_path}")

    def _fmt(v): return f"{v:.4f}" if v is not None else "  N/A "

    # ── Per-speaker summary ───────────────────────────────────────────────────
    print("\n" + "=" * 84)
    print(f"Per-Speaker Results  (k={k})")
    print("=" * 84)
    print(f"  {'Spk':<6} {'L1':<4} {'A:960h_k'+str(k):>10} {'C:100h_k0':>10} "
          f"{'C:100h_k'+str(k):>10} {'Fed_k0':>8} {'Fed_k'+str(k):>8}  sig?")
    print(f"  {'─'*6} {'─'*4} {'─'*10} {'─'*10} {'─'*10} {'─'*8} {'─'*8}  {'─'*4}")

    sig_count = 0
    for spk_id, spk_data in results["per_speaker"].items():
        l1        = spk_data.get("l1", "?")
        a_kn      = spk_data.get(f"WER_960h_k{k}")
        c_k0      = spk_data.get("WER_100h_k0")
        c_kn      = spk_data.get(f"WER_100h_k{k}")
        fed_k0    = spk_data.get("WER_federated_k0")
        fed_kn    = spk_data.get(f"WER_federated_k{k}")
        c_kn_ci   = spk_data.get(f"WER_100h_k{k}_ci", [None, None])
        fed_kn_ci = spk_data.get(f"WER_federated_k{k}_ci", [None, None])

        # Primary significance test: federated vs 100h (same training backbone)
        significant = (
            fed_kn is not None and c_kn is not None
            and fed_kn_ci[1] is not None and c_kn_ci[0] is not None
            and fed_kn_ci[1] < c_kn_ci[0]
        )
        if significant:
            sig_count += 1

        sig_str = " YES✓" if significant else "  no "
        print(f"  {spk_id:<6} {l1:<4} {_fmt(a_kn):>10} {_fmt(c_k0):>10} "
              f"{_fmt(c_kn):>10} {_fmt(fed_k0):>8} {_fmt(fed_kn):>8}  {sig_str}")

    # ── Per-accent summary ────────────────────────────────────────────────────
    print("\n" + "=" * 84)
    print(f"Per-Accent Breakdown  (k={k})")
    print("=" * 84)
    print(f"  {'L1':<4} {'Accent':<12} {'Spk':<6} {'A:960h':>8} {'C:100h':>8} "
          f"{'Fed':>8}  {'Δ Fed-100h':>11}  {'Δ Fed-960h':>11}")
    print(f"  {'─'*4} {'─'*12} {'─'*6} {'─'*8} {'─'*8} {'─'*8}  {'─'*11}  {'─'*11}")

    def _d(v): return (f"{v:+.4f}" + (" ▼" if v < 0 else " ▲")) if v is not None else "   N/A "
    for l1 in ["hi", "vi", "zh", "ko", "es", "ar"]:
        if l1 not in accent_groups:
            continue
        grp = accent_groups[l1]
        for spk_id in grp["speakers"]:
            spk_data = results["per_speaker"][spk_id]
            a_kn   = spk_data.get(f"WER_960h_k{k}")
            c_kn   = spk_data.get(f"WER_100h_k{k}")
            fed_kn = spk_data.get(f"WER_federated_k{k}")
            d_100h = round(fed_kn - c_kn, 4)  if (fed_kn is not None and c_kn is not None) else None
            d_960h = round(fed_kn - a_kn, 4)  if (fed_kn is not None and a_kn is not None) else None
            print(f"  {l1:<4} {grp['l1_name']:<12} {spk_id:<6} "
                  f"{_fmt(a_kn):>8} {_fmt(c_kn):>8} {_fmt(fed_kn):>8}  "
                  f"{_d(d_100h):>13}  {_d(d_960h):>13}")

    n_spk = len(meta_test_speakers)
    print(f"\n  Primary (Fed vs 100h baseline): {sig_count}/{n_spk} speakers improved (non-overlapping 95% CI)")
    if sig_count >= 2:
        print(f"\n  CLAIM SUPPORTED: FedLoRA-MAML θ* improves over direct 100h fine-tuning "
              f"on {sig_count}/{n_spk} meta-test speakers.")
    else:
        print(f"\n  Claim NOT supported at threshold (need ≥2/{n_spk}). "
              "Document as null result per evaluation integrity protocol.")


if __name__ == "__main__":
    main()
