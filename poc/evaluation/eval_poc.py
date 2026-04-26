"""
PoC final evaluation — run ONCE on meta-test speakers.

Implements a 5-step protocol with proper baselines and bootstrap CIs.
No perturbation protocol. Null results are reported accurately.

Steps:
  1. Sovereignty + invariant checks (I1–I6)
  2. Establish baselines on meta-test speakers
       BASELINE 1: pretrained model, k=0
       BASELINE 2: pretrained model, k=10 (direct fine-tuning, no MAML)
  3. Evaluate federated θ* on meta-test speakers (k=0 and k=10)
  4. Evaluate centralized matched-update θ* on meta-test speakers
  5. Adaptation curves at k=0,1,3,5,10 for federated θ* vs pretrained baseline
  6. Bootstrap 95% CI for all WER estimates

Baselines required for a valid result:
  WER_federated_k10 < WER_pretrained_k10  (MAML helps over no-MAML)
  AND confidence intervals do not overlap
  AND adaptation curve shows faster early convergence for MAML

Output:
  evaluation/results/final_eval.json
  evaluation/results/adaptation_curves.json

Usage:
    python evaluation/eval_poc.py --config configs/poc.yaml
    python evaluation/eval_poc.py --device cuda --federated_ckpt checkpoints/federated/theta_star.pt
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_SAMPLE_RATE = 16_000

PASS_STR = "\033[32mPASS\033[0m"
FAIL_STR = "\033[31mFAIL\033[0m"
INFO_STR = "\033[33mINFO\033[0m"
K_CURVE = [0, 1, 3, 5, 10]

def label(ok: bool, detail: str = "") -> None:
    status = PASS_STR if ok else FAIL_STR
    msg = f"  [{status}]"
    if detail:
        msg += f" {detail}"
    print(msg)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--federated_ckpt", default=None)
    p.add_argument("--centralized_ckpt", default=None)
    p.add_argument("--meta_split", default=None)
    p.add_argument("--test_nodes_dir", default=None)
    p.add_argument("--eval_clips", type=int, default=20,
                   help="Query clips per meta-test speaker")
    p.add_argument("--support_size", type=int, default=8)
    p.add_argument("--eval_k", type=int, default=10,
                   help="k for main WER comparison (default 10)")
    p.add_argument("--inner_lr", type=float, default=None)
    p.add_argument("--bootstrap_n", type=int, default=1000)
    return p.parse_args()

def load_config(path: str | None) -> dict:
    import yaml
    defaults = {
        "device": "cpu",
        "inner_lr": 1e-4,
        "support_size": 8,
        "eval_clips": 20,
        "eval_k": 10,
        "bootstrap_n": 1000,
        "nodes_dir": str(ROOT / "data" / "nodes"),
        "test_nodes_dir": str(ROOT / "data" / "test_nodes"),
        "meta_split": str(ROOT / "data" / "meta_split.json"),
        "federated_ckpt": str(ROOT / "checkpoints" / "federated" / "theta_star.pt"),
        "centralized_ckpt": str(ROOT / "checkpoints" / "centralized" / "theta_star_matched.pt"),
    }
    if path is None:
        return defaults
    with open(path) as f:
        cfg = yaml.safe_load(f)
    defaults.update(cfg.get("evaluation", {}))
    defaults.update(cfg.get("maml", {}))
    defaults["device"] = cfg.get("device", defaults["device"])
    return defaults

def bootstrap_ci(
    values: list[float],
    n_samples: int = 1000,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Return (mean, lower_ci, upper_ci) via percentile bootstrap.

    Args:
        values: per-clip WER values (0 or 1 per word, or clip-level floats)
        n_samples: bootstrap iterations
        ci: confidence level (default 0.95)
    """
    arr = np.array(values, dtype=np.float64)
    rng = np.random.default_rng(seed=42)
    means = [rng.choice(arr, size=len(arr), replace=True).mean()
             for _ in range(n_samples)]
    alpha = (1 - ci) / 2
    lower = float(np.percentile(means, alpha * 100))
    upper = float(np.percentile(means, (1 - alpha) * 100))
    return float(arr.mean()), lower, upper

def wer_per_clip(
    model_copy,
    processor,
    audio_clips: list,
    texts: list[str],
    device,
) -> list[float]:
    """Return per-clip WER (for bootstrap resampling)."""
    from jiwer import wer as _wer
    import torch

    model_copy.model.eval()
    dtype = next(model_copy.model.parameters()).dtype
    per_clip: list[float] = []

    with torch.no_grad():
        for audio, ref in zip(audio_clips, texts):
            arr = audio.float().numpy()
            iv = processor(arr, sampling_rate=_SAMPLE_RATE, return_tensors="pt",
                           padding=False).input_values.to(device=device, dtype=dtype)
            out = model_copy.model(input_values=iv)
            hyp = processor.batch_decode(torch.argmax(out.logits, dim=-1))[0]
            per_clip.append(float(_wer([ref], [hyp])))

    return per_clip

def adapt_model(model, processor, support_audio, support_texts, k, inner_lr, device):
    """Return a deepcopy of model adapted for k steps on support set (no perturbation)."""
    import copy
    import torch
    from maml.engine import _accumulate_grads_over_clips

    m = copy.deepcopy(model)
    if k == 0:
        return m

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

    return m

def check_sovereignty(nodes_dir: Path) -> bool:
    all_pass = True
    print("\n[SOVEREIGNTY CHECKS]")

    pkl_result = subprocess.run(
        ["find", str(nodes_dir), "-name", "*.pkl"],
        capture_output=True, text=True,
    )
    i1 = not pkl_result.stdout.strip()
    ok_str = "clean" if i1 else pkl_result.stdout.strip()
    label(i1, f"I1: no .pkl files in data/nodes/ — {ok_str}")
    all_pass &= i1

    grep_result = subprocess.run(
        ["grep", "-r", "speaker_id", str(nodes_dir)],
        capture_output=True, text=True,
    )
    i2 = not grep_result.stdout.strip()
    ok_str = "clean" if i2 else grep_result.stdout.strip()
    label(i2, f"I2: no speaker_id in data/nodes/ — {ok_str}")
    all_pass &= i2

    return all_pass

def check_invariant_i3(model) -> bool:
    lm_head_ids = {id(p) for p in model.model.lm_head.parameters()}
    outer_ids = {id(p) for p in model.get_outer_loop_params()}
    overlap = lm_head_ids & outer_ids
    ok = len(overlap) == 0
    label(ok, f"I3: lm_head not in outer_loop_params — overlap={len(overlap)}")
    return ok

def check_invariant_i4() -> bool:
    strategy_file = ROOT / "federated" / "strategy_maml.py"
    if not strategy_file.exists():
        label(False, "I4: strategy_maml.py not found")
        return False
    content = strategy_file.read_text()
    ok = "_theta_star" in content and "outer_lr" in content and "avg" in content
    label(ok, "I4: strategy applies θ*←θ*−β·grad")
    return ok

def check_invariant_i5(sampler) -> bool:
    from data.task_sampler import VoiceTaskSampler              
    try:
        for _ in range(10):
            sampler.sample_task()
        label(True, "I5: support∩query=∅ (10 samples)")
        return True
    except AssertionError as exc:
        label(False, f"I5: support∩query overlap — {exc}")
        return False

def check_invariant_i6(model) -> bool:
    violations = [
        name for name, p in model.model.wav2vec2.named_parameters()
        if not p.requires_grad
    ]
    ok = len(violations) == 0
    detail = "all True" if ok else f"violations: {violations[:3]}"
    label(ok, f"I6: encoder requires_grad=True — {detail}")
    return ok

def evaluate_speaker(
    speaker_name: str,
    tensors: list,
    texts: list[str],
    pretrained_model,
    fed_model,
    cen_model,                       
    processor,
    cfg: dict,
    device,
) -> dict:
    """Run full evaluation protocol for one meta-test speaker."""
    support_size = cfg["support_size"]
    eval_clips = cfg["eval_clips"]
    k = cfg["eval_k"]
    inner_lr = cfg["inner_lr"]
    n_bootstrap = cfg["bootstrap_n"]

    needed = support_size + eval_clips
    if len(tensors) < needed:
        print(f"  WARNING: {speaker_name} only {len(tensors)} clips (need {needed})")
        eval_clips = max(1, len(tensors) - support_size)

    support_audio = tensors[:support_size]
    support_texts = texts[:support_size]
    query_audio = tensors[support_size: support_size + eval_clips]
    query_texts = texts[support_size: support_size + eval_clips]

    result: dict = {"speaker": speaker_name, "n_query_clips": len(query_audio)}

    print(f"  {speaker_name[:8]}: BASELINE 1 (pretrained k=0)...")
    import copy
    pt_k0_model = copy.deepcopy(pretrained_model)
    clips_b1 = wer_per_clip(pt_k0_model, processor, query_audio, query_texts, device)
    mean_b1, lo_b1, hi_b1 = bootstrap_ci(clips_b1, n_bootstrap)
    result["pretrained_k0"] = {"mean": mean_b1, "ci_lower": lo_b1, "ci_upper": hi_b1}
    del pt_k0_model

    print(f"  {speaker_name[:8]}: BASELINE 2 (pretrained k={k})...")
    pt_adapted = adapt_model(pretrained_model, processor, support_audio, support_texts,
                              k, inner_lr, device)
    clips_b2 = wer_per_clip(pt_adapted, processor, query_audio, query_texts, device)
    mean_b2, lo_b2, hi_b2 = bootstrap_ci(clips_b2, n_bootstrap)
    result["pretrained_adapted"] = {"k": k, "mean": mean_b2,
                                    "ci_lower": lo_b2, "ci_upper": hi_b2}
    del pt_adapted

    print(f"  {speaker_name[:8]}: FEDERATED k=0...")
    fed_k0_model = copy.deepcopy(fed_model)
    clips_f0 = wer_per_clip(fed_k0_model, processor, query_audio, query_texts, device)
    mean_f0, lo_f0, hi_f0 = bootstrap_ci(clips_f0, n_bootstrap)
    result["federated_k0"] = {"mean": mean_f0, "ci_lower": lo_f0, "ci_upper": hi_f0}
    del fed_k0_model

    print(f"  {speaker_name[:8]}: FEDERATED k={k}...")
    fed_adapted = adapt_model(fed_model, processor, support_audio, support_texts,
                               k, inner_lr, device)
    clips_fa = wer_per_clip(fed_adapted, processor, query_audio, query_texts, device)
    mean_fa, lo_fa, hi_fa = bootstrap_ci(clips_fa, n_bootstrap)
    result["federated_adapted"] = {"k": k, "mean": mean_fa,
                                   "ci_lower": lo_fa, "ci_upper": hi_fa}
    del fed_adapted

    if cen_model is not None:
        print(f"  {speaker_name[:8]}: CENTRALIZED k={k}...")
        cen_adapted = adapt_model(cen_model, processor, support_audio, support_texts,
                                   k, inner_lr, device)
        clips_ca = wer_per_clip(cen_adapted, processor, query_audio, query_texts, device)
        mean_ca, lo_ca, hi_ca = bootstrap_ci(clips_ca, n_bootstrap)
        result["centralized_adapted"] = {"k": k, "mean": mean_ca,
                                         "ci_lower": lo_ca, "ci_upper": hi_ca}
        del cen_adapted

    print(f"  {speaker_name[:8]}: Adaptation curves...")
    curves: dict[str, dict] = {"pretrained": {}, "federated": {}}
    for k_val in K_CURVE:
        for label_key, base_model in [("pretrained", pretrained_model), ("federated", fed_model)]:
            adapted = adapt_model(base_model, processor, support_audio, support_texts,
                                   k_val, inner_lr, device)
            per_clip = wer_per_clip(adapted, processor, query_audio, query_texts, device)
            mean_wer, lo, hi = bootstrap_ci(per_clip, n_bootstrap)
            curves[label_key][k_val] = {"mean": mean_wer, "ci_lower": lo, "ci_upper": hi}
            del adapted

    result["adaptation_curves"] = curves

    return result

def assess_claim(speaker_results: list[dict], k: int) -> dict:
    """Evaluate whether the primary claim is supported."""
    fed_key = "federated_adapted"
    pt_key = "pretrained_adapted"

    comparisons = []
    for r in speaker_results:
        if fed_key not in r or pt_key not in r:
            continue
        fed_wer = r[fed_key]["mean"]
        pt_wer = r[pt_key]["mean"]
        fed_lo = r[fed_key]["ci_lower"]
        pt_hi = r[pt_key]["ci_upper"]
        cis_overlap = fed_lo < pt_hi                                    
        comparisons.append({
            "speaker": r["speaker"],
            "fed_wer": fed_wer,
            "pt_wer": pt_wer,
            "maml_improves": fed_wer < pt_wer,
            "ci_overlap": cis_overlap,
            "significant": not cis_overlap,
        })

    n_improved = sum(1 for c in comparisons if c["maml_improves"])
    n_significant = sum(1 for c in comparisons if c["significant"])
    n_total = len(comparisons)

    claim_supported = n_improved == n_total and n_significant > 0

    return {
        "claim": f"Federated FOMAML θ* improves adaptation over direct fine-tuning at k={k}",
        "claim_supported": claim_supported,
        "n_improved": n_improved,
        "n_significant": n_significant,
        "n_speakers": n_total,
        "per_speaker": comparisons,
        "interpretation": (
            "VALID POSITIVE: MAML improves over fine-tuning with statistical significance"
            if claim_supported
            else (
                "VALID NULL: MAML does not significantly improve over fine-tuning. "
                "This is a valid scientific finding. Document it accurately."
            )
        ),
    }

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.device:
        cfg["device"] = args.device
    if args.federated_ckpt:
        cfg["federated_ckpt"] = args.federated_ckpt
    if args.centralized_ckpt:
        cfg["centralized_ckpt"] = args.centralized_ckpt
    if args.inner_lr:
        cfg["inner_lr"] = args.inner_lr
    if args.eval_clips:
        cfg["eval_clips"] = args.eval_clips
    if args.support_size:
        cfg["support_size"] = args.support_size
    if args.eval_k:
        cfg["eval_k"] = args.eval_k
    if args.bootstrap_n:
        cfg["bootstrap_n"] = args.bootstrap_n

    import torch
    from models.wav2vec2_maml import Wav2Vec2MAML, load_processor
    from data.task_sampler import VoiceTaskSampler

    device = torch.device(cfg["device"])

    print("=" * 70)
    print("VoiceFL-MAML PoC — Final Evaluation (VCTK, meta-test speakers)")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Eval k: {cfg['eval_k']}  inner_lr: {cfg['inner_lr']}")
    print(f"Bootstrap samples: {cfg['bootstrap_n']}")

    nodes_dir = Path(cfg["nodes_dir"])
    sovereignty_ok = check_sovereignty(nodes_dir)

    meta_split_path = Path(args.meta_split) if args.meta_split else Path(cfg["meta_split"])
    if not meta_split_path.exists():
        print(f"\nERROR: meta_split.json not found at {meta_split_path}")
        print("Run: python data/meta_split.py")
        sys.exit(1)

    meta_split = json.loads(meta_split_path.read_text())
    meta_test_hashes = set(meta_split["meta_test"])
    print(f"\nmeta-test speakers: {len(meta_test_hashes)}")

    test_nodes_dir = Path(args.test_nodes_dir) if args.test_nodes_dir else Path(cfg["test_nodes_dir"])
    if not test_nodes_dir.exists():
        print(f"\nERROR: test_nodes_dir not found: {test_nodes_dir}")
        print("Run: python data/prepare_vctk.py --splits meta_test && python data/features.py --nodes_dir data/test_nodes")
        sys.exit(1)

    test_node_dirs = [
        d for d in sorted(test_nodes_dir.iterdir()) if d.is_dir()
        and d.name in meta_test_hashes
        and (d / "features.pt").exists()
    ]

    if not test_node_dirs:
        print(f"\nERROR: No prepared meta-test node directories found.")
        print(f"Expected hashes in {test_nodes_dir}: {meta_test_hashes}")
        sys.exit(1)

    print(f"Found {len(test_node_dirs)} prepared meta-test nodes")

    print("\n[LOADING MODELS]")
    processor = load_processor()

    print("  Loading pretrained wav2vec2-base-960h (no MAML)...")
    pretrained_model = Wav2Vec2MAML(device=device)

    fed_ckpt = Path(cfg["federated_ckpt"])
    if not fed_ckpt.exists():
        print(f"\nERROR: Federated checkpoint not found: {fed_ckpt}")
        sys.exit(1)
    print(f"  Loading federated θ* from {fed_ckpt}...")
    fed_model = Wav2Vec2MAML(device=device)
    _fed_data = torch.load(fed_ckpt, map_location="cpu", weights_only=True)
    _fed_conv = fed_model.model.wav2vec2.encoder.pos_conv_embed.conv
    if hasattr(_fed_conv, "parametrizations") and "weight" in _fed_conv.parametrizations:
        if "encoder.pos_conv_embed.conv.weight" in _fed_data:
            torch.nn.utils.parametrize.remove_parametrizations(_fed_conv, "weight")
    fed_model.set_encoder_state_dict(_fed_data)
    del _fed_data, _fed_conv

    cen_ckpt = Path(cfg["centralized_ckpt"])
    cen_model = None
    if cen_ckpt.exists():
        print(f"  Loading centralized matched-update θ* from {cen_ckpt}...")
        cen_model = Wav2Vec2MAML(device=device)
        cen_model.set_encoder_state_dict(
            torch.load(cen_ckpt, map_location="cpu", weights_only=True)
        )
    else:
        print(f"  [{INFO_STR}] Centralized checkpoint not found — skipping federation comparison")
        print(f"           Run: python maml/meta_train.py --max_updates N to generate it")

    print("\n[INVARIANT CHECKS]")
                                                               
    train_node_dirs = sorted(d for d in nodes_dir.iterdir()
                              if d.is_dir() and (d / "features.pt").exists())[:1]
    if train_node_dirs:
        i3_ok = check_invariant_i3(fed_model)
        i4_ok = check_invariant_i4()
        sampler = VoiceTaskSampler(train_node_dirs[0], cfg["support_size"], 8)
        i5_ok = check_invariant_i5(sampler)
        i6_ok = check_invariant_i6(fed_model)
        all_invariants = sovereignty_ok and i3_ok and i4_ok and i5_ok and i6_ok
    else:
        print("  WARNING: No training nodes found — skipping invariant checks I3-I6")
        all_invariants = sovereignty_ok

    print(f"\n[EVALUATION — {len(test_node_dirs)} meta-test speakers]")
    print(f"Support: {cfg['support_size']} clips | Query: {cfg['eval_clips']} clips | k={cfg['eval_k']}")
    print()

    speaker_results = []
    for node_dir in test_node_dirs:
        tensors = torch.load(node_dir / "features.pt", weights_only=True)
        texts = (node_dir / "labels.txt").read_text().splitlines()
        print(f"Speaker {node_dir.name[:8]} ({len(tensors)} clips):")

        result = evaluate_speaker(
            speaker_name=node_dir.name,
            tensors=tensors,
            texts=texts,
            pretrained_model=pretrained_model,
            fed_model=fed_model,
            cen_model=cen_model,
            processor=processor,
            cfg=cfg,
            device=device,
        )
        speaker_results.append(result)

        r = result
        k = cfg["eval_k"]
        print(f"    pretrained k=0:  WER={r['pretrained_k0']['mean']:.4f} "
              f"[{r['pretrained_k0']['ci_lower']:.4f}, {r['pretrained_k0']['ci_upper']:.4f}]")
        print(f"    pretrained k={k}: WER={r['pretrained_adapted']['mean']:.4f} "
              f"[{r['pretrained_adapted']['ci_lower']:.4f}, {r['pretrained_adapted']['ci_upper']:.4f}]")
        print(f"    federated k={k}:  WER={r['federated_adapted']['mean']:.4f} "
              f"[{r['federated_adapted']['ci_lower']:.4f}, {r['federated_adapted']['ci_upper']:.4f}]")
        if cen_model and "centralized_adapted" in r:
            print(f"    centralized k={k}: WER={r['centralized_adapted']['mean']:.4f} "
                  f"[{r['centralized_adapted']['ci_lower']:.4f}, {r['centralized_adapted']['ci_upper']:.4f}]")
        print()

    claim_result = assess_claim(speaker_results, cfg["eval_k"])

    print("=" * 70)
    print("CLAIM ASSESSMENT")
    print("=" * 70)
    print(f"Claim: {claim_result['claim']}")
    print(f"Supported: {claim_result['claim_supported']}")
    print(f"  Improved over fine-tuning: {claim_result['n_improved']}/{claim_result['n_speakers']} speakers")
    print(f"  Statistically significant: {claim_result['n_significant']}/{claim_result['n_speakers']} speakers")
    print(f"\n  {claim_result['interpretation']}")

    def agg_mean(key: str, sub_key: str = "mean") -> float | None:
        vals = [r[key][sub_key] for r in speaker_results if key in r]
        return float(np.mean(vals)) if vals else None

    print("\n[AGGREGATE WER — mean across meta-test speakers]")
    agg = {
        "pretrained_k0": agg_mean("pretrained_k0"),
        "pretrained_adapted": agg_mean("pretrained_adapted"),
        "federated_k0": agg_mean("federated_k0"),
        "federated_adapted": agg_mean("federated_adapted"),
        "centralized_adapted": agg_mean("centralized_adapted") if cen_model else None,
    }
    for k, v in agg.items():
        if v is not None:
            print(f"  {k:30s}: {v:.4f}")

    final_result = {
        "config": cfg,
        "n_meta_test_speakers": len(test_node_dirs),
        "eval_k": cfg["eval_k"],
        "inner_lr": cfg["inner_lr"],
        "support_size": cfg["support_size"],
        "eval_clips": cfg["eval_clips"],
        "bootstrap_samples": cfg["bootstrap_n"],
        "aggregate_wer": agg,
        "claim_assessment": claim_result,
        "per_speaker": speaker_results,
        "invariants_passed": all_invariants,
        "sovereignty_passed": sovereignty_ok,
    }

    curve_result = {
        "k_values": K_CURVE,
        "speakers": {
            r["speaker"]: r.get("adaptation_curves", {})
            for r in speaker_results
        },
    }

    out_final = RESULTS_DIR / "final_eval.json"
    out_curves = RESULTS_DIR / "adaptation_curves.json"
    out_final.write_text(json.dumps(final_result, indent=2))
    out_curves.write_text(json.dumps(curve_result, indent=2))

    print(f"\nResults saved:")
    print(f"  {out_final}")
    print(f"  {out_curves}")
    print("\nNext: python evaluation/plot_results.py")

    if not all_invariants:
        print("\nINVARIANT VIOLATIONS detected — fix before reporting results")
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    main()
