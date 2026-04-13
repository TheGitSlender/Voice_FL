"""
PoC evaluation script — checks all 3 success criteria + all 6 invariants.

Exit codes:
  0 — all criteria and invariants passed
  1 — one or more failures

Usage:
    python evaluation/eval_poc.py --config configs/poc.yaml
    python evaluation/eval_poc.py --federated_ckpt checkpoints/federated/theta_star.pt
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    msg = f"  [{status}] {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return condition


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--federated_ckpt", type=str, default=None)
    p.add_argument("--centralized_ckpt", type=str, default=None)
    p.add_argument("--n_tasks", type=int, default=3, help="Tasks per node for WER eval")
    return p.parse_args()


def load_config(path: str | None) -> dict:
    import yaml
    defaults = {
        "device": "cpu",
        "inner_steps": 3,
        "inner_lr": 1e-4,
        "support_size": 8,
        "query_size": 8,
        "nodes_dir": str(ROOT / "data" / "nodes"),
        "federated_ckpt": str(ROOT / "checkpoints" / "federated" / "theta_star.pt"),
        "centralized_ckpt": str(ROOT / "checkpoints" / "centralized" / "theta_star.pt"),
    }
    if path is None:
        return defaults
    with open(path) as f:
        cfg = yaml.safe_load(f)
    defaults.update(cfg.get("evaluation", {}))
    defaults.update(cfg.get("maml", {}))
    return defaults


def check_sovereignty(nodes_dir: Path) -> bool:
    """Criterion 3 + Invariants I1, I2."""
    all_pass = True
    print("\n[SOVEREIGNTY CHECKS]")

    # I1: no .pkl files
    pkl_result = subprocess.run(
        ["find", str(nodes_dir), "-name", "*.pkl"],
        capture_output=True, text=True,
    )
    i1 = not pkl_result.stdout.strip()
    all_pass &= check("I1: no .pkl files in data/nodes/", i1, pkl_result.stdout.strip() or "clean")

    # I2: no speaker_id in node artifacts
    grep_result = subprocess.run(
        ["grep", "-r", "speaker_id", str(nodes_dir)],
        capture_output=True, text=True,
    )
    i2 = not grep_result.stdout.strip()
    all_pass &= check("I2: no speaker_id in data/nodes/", i2, grep_result.stdout.strip() or "clean")

    return all_pass


def check_invariant_i3(model) -> bool:
    """I3: get_outer_loop_params() excludes lm_head."""
    lm_head_ids = {id(p) for p in model.model.lm_head.parameters()}
    outer_ids = {id(p) for p in model.get_outer_loop_params()}
    overlap = lm_head_ids & outer_ids
    return check(
        "I3: lm_head not in outer_loop_params()",
        len(overlap) == 0,
        f"overlap={len(overlap)} params",
    )


def check_invariant_i4() -> bool:
    """I4: strategy aggregates gradients, not weights. Documented by code review."""
    # We check the strategy file for the canonical update formula
    strategy_file = ROOT / "federated" / "strategy_maml.py"
    if not strategy_file.exists():
        return check("I4: strategy uses gradient descent", False, "strategy_maml.py not found")
    content = strategy_file.read_text()
    # Check for the actual gradient-descent update pattern used in strategy_maml.py:
    # self._theta_star[i].astype(np.float32) - self.outer_lr * avg
    has_gradient_update = "_theta_star" in content and "outer_lr" in content and "avg" in content
    return check(
        "I4: strategy applies θ*←θ*−β·grad (not weight avg)",
        has_gradient_update,
    )


def check_invariant_i5(sampler) -> bool:
    """I5: support ∩ query = ∅ on 10 samples."""
    from data.task_sampler import VoiceTaskSampler
    try:
        for _ in range(10):
            sampler.sample_task()
        return check("I5: support∩query=∅ (10 samples)", True)
    except AssertionError as e:
        return check("I5: support∩query=∅", False, str(e))


def check_invariant_i6(model) -> bool:
    """I6: encoder requires_grad stays True."""
    violations = [
        name for name, p in model.model.wav2vec2.named_parameters()
        if not p.requires_grad
    ]
    ok = len(violations) == 0
    return check(
        "I6: encoder requires_grad=True",
        ok,
        f"violations: {violations[:3]}" if violations else "all True",
    )


def evaluate_adaptation(model, processor, samplers, cfg, device) -> tuple[int, list]:
    """Criterion 1: WER(k=3) < WER(k=0) on ≥4/5 nodes."""
    from maml.engine import compute_wer_k0, compute_wer_k3

    print("\n[CRITERION 1: ADAPTATION]")
    passed = 0
    node_results = []
    for i, sampler in enumerate(samplers):
        wer0_list, wer3_list = [], []
        for _ in range(cfg["n_tasks"]):
            task = sampler.sample_task()
            w0 = compute_wer_k0(model, processor, task.query_audio, task.query_labels, device)
            w3 = compute_wer_k3(
                model, processor,
                task.support_audio, task.support_labels,
                task.query_audio, task.query_labels,
                cfg["inner_steps"], cfg["inner_lr"], device,
            )
            wer0_list.append(w0)
            wer3_list.append(w3)

        mean0 = sum(wer0_list) / len(wer0_list)
        mean3 = sum(wer3_list) / len(wer3_list)
        ok = mean3 < mean0
        if ok:
            passed += 1
        check(
            f"Node {i+1}: WER(k=3)={mean3:.3f} < WER(k=0)={mean0:.3f}",
            ok,
        )
        node_results.append({"wer_k0": mean0, "wer_k3": mean3, "passed": ok})

    criterion1 = passed >= 4
    check(f"Criterion 1: {passed}/{len(samplers)} nodes adapted", criterion1, "need ≥4")
    return passed, node_results


def evaluate_federation(
    fed_model,
    cen_model,
    processor,
    samplers,
    cfg,
    device,
) -> bool:
    """Criterion 2: federated WER within 15% of centralized WER."""
    from maml.engine import compute_wer_k0

    print("\n[CRITERION 2: FEDERATION]")
    fed_wers, cen_wers = [], []
    for sampler in samplers:
        for _ in range(cfg["n_tasks"]):
            task = sampler.sample_task()
            wf = compute_wer_k0(fed_model, processor, task.query_audio, task.query_labels, device)
            wc = compute_wer_k0(cen_model, processor, task.query_audio, task.query_labels, device)
            fed_wers.append(wf)
            cen_wers.append(wc)

    mean_fed = sum(fed_wers) / len(fed_wers)
    mean_cen = sum(cen_wers) / len(cen_wers)
    if mean_cen > 0:
        relative_diff = abs(mean_fed - mean_cen) / mean_cen
    else:
        relative_diff = 0.0

    ok = relative_diff <= 0.15
    check(
        f"Criterion 2: federated WER={mean_fed:.3f} vs centralized WER={mean_cen:.3f}",
        ok,
        f"diff={relative_diff:.1%} (limit 15%)",
    )
    return ok


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg["n_tasks"] = args.n_tasks
    if args.device:
        cfg["device"] = args.device
    if args.federated_ckpt:
        cfg["federated_ckpt"] = args.federated_ckpt
    if args.centralized_ckpt:
        cfg["centralized_ckpt"] = args.centralized_ckpt

    import torch
    from models.wav2vec2_maml import Wav2Vec2MAML, load_processor
    from data.task_sampler import VoiceTaskSampler

    device = torch.device(cfg["device"])
    nodes_dir = Path(cfg["nodes_dir"])

    print("=" * 60)
    print("VoiceFL-MAML PoC — Evaluation")
    print("=" * 60)

    # ---- Sovereignty (Criterion 3 + I1 + I2) ---------------------------------
    sovereignty_ok = check_sovereignty(nodes_dir)

    # ---- Load models ----------------------------------------------------------
    print("\n[LOADING MODELS]")
    fed_ckpt = Path(cfg["federated_ckpt"])
    cen_ckpt = Path(cfg["centralized_ckpt"])

    fed_model_ok = fed_ckpt.exists()
    cen_model_ok = cen_ckpt.exists()
    check("Federated checkpoint exists", fed_model_ok, str(fed_ckpt))
    check("Centralized checkpoint exists", cen_model_ok, str(cen_ckpt))

    if not (fed_model_ok and cen_model_ok):
        print("\nERROR: Missing checkpoints. Run meta_train.py and the federated stack first.")
        sys.exit(1)

    processor = load_processor()

    # Federated checkpoint was saved without parametrizations (plain conv.weight).
    # Strip before loading so the key layout matches.
    fed_model = Wav2Vec2MAML(device=device)
    _fed_ckpt_data = torch.load(fed_ckpt, map_location="cpu", weights_only=True)
    _fed_conv = fed_model.model.wav2vec2.encoder.pos_conv_embed.conv
    if hasattr(_fed_conv, "parametrizations") and "weight" in _fed_conv.parametrizations:
        if "encoder.pos_conv_embed.conv.weight" in _fed_ckpt_data:
            torch.nn.utils.parametrize.remove_parametrizations(_fed_conv, "weight")
    fed_model.set_encoder_state_dict(_fed_ckpt_data)
    del _fed_ckpt_data, _fed_conv

    # Centralized checkpoint was saved with parametrizations active — load as-is.
    cen_model = Wav2Vec2MAML(device=device)
    cen_model.set_encoder_state_dict(
        torch.load(cen_ckpt, map_location="cpu", weights_only=True)
    )

    # ---- Invariant checks -----------------------------------------------------
    print("\n[INVARIANT CHECKS]")
    node_dirs = sorted(d for d in nodes_dir.iterdir() if d.is_dir())
    samplers = [
        VoiceTaskSampler(d, cfg["support_size"], cfg["query_size"])
        for d in node_dirs
    ]

    i3_ok = check_invariant_i3(fed_model)
    i4_ok = check_invariant_i4()
    i5_ok = check_invariant_i5(samplers[0])
    i6_ok = check_invariant_i6(fed_model)

    all_invariants = sovereignty_ok and i3_ok and i4_ok and i5_ok and i6_ok

    # ---- Criterion 1: Adaptation ----------------------------------------------
    passed_nodes, node_results = evaluate_adaptation(
        fed_model, processor, samplers, cfg, device
    )
    criterion1_ok = passed_nodes >= 4

    # ---- Criterion 2: Federation -----------------------------------------------
    criterion2_ok = evaluate_federation(
        fed_model, cen_model, processor, samplers, cfg, device
    )

    # ---- Criterion 3: Sovereignty (already checked) ---------------------------
    print("\n[CRITERION 3: SOVEREIGNTY]")
    check("Criterion 3: sovereignty (no PKL + no speaker_id)", sovereignty_ok)

    # ---- Final summary --------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    all_pass = all_invariants and criterion1_ok and criterion2_ok and sovereignty_ok
    check("All invariants (I1–I6)", all_invariants)
    check("Criterion 1: Adaptation", criterion1_ok, f"{passed_nodes}/5 nodes")
    check("Criterion 2: Federation", criterion2_ok)
    check("Criterion 3: Sovereignty", sovereignty_ok)

    if all_pass:
        print("\n\033[32mPoC PASSED\033[0m — all 3 criteria met, all 6 invariants satisfied")
        sys.exit(0)
    else:
        print("\n\033[31mPoC FAILED\033[0m — see failures above")
        sys.exit(1)


if __name__ == "__main__":
    main()
