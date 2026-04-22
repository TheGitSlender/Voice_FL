"""
Centralized MAML training.

Runs FOMAML (or lora_maml) over all meta-train nodes (from meta_split.json)
without Docker/FL. Gate evaluation uses meta-val speakers (never meta-test).
Perturbation protocol removed — speakers are genuinely unseen.

Usage:
    python maml/meta_train.py --config configs/poc.yaml
    python maml/meta_train.py --rounds 5 --device cuda
    python maml/meta_train.py --mode lora_maml --config configs/lora_poc.yaml
    python maml/meta_train.py --max_updates N  # matched-update centralized baseline

Checkpoint: checkpoints/centralized/theta_star.pt (fomaml)
            checkpoints/centralized/theta_star_lora.pt (lora_maml)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

def parse_args():
    p = argparse.ArgumentParser(description="Centralized MAML training")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--outer_lr", type=float, default=None)
    p.add_argument("--inner_steps", type=int, default=None)
    p.add_argument("--max_updates", type=int, default=None,
                   help="Stop after N total gradient updates (for matched-update baseline)")
    p.add_argument("--output_name", type=str, default="theta_star",
                   help="Output checkpoint filename stem (default: theta_star)")
    p.add_argument("--clip_norm", type=float, default=None,
                   help="Gradient clip norm (default: auto from config or 20.0)")
    p.add_argument("--init_ckpt", type=str, default=None,
                   help="Load encoder state from this checkpoint before training")
    p.add_argument("--eval_only", action="store_true",
                   help="Skip training; load saved theta_star.pt and run gate eval only")
    p.add_argument("--mode", type=str, default=None,
                   help="MAML mode: fomaml (default) or lora_maml")
    return p.parse_args()

def load_config(path: str | None) -> dict:
    import yaml
    defaults = {
        "rounds": 20,
        "device": "cpu",
        "outer_lr": 2e-4,
        "inner_lr": 1e-4,
        "inner_steps": 3,
        "support_size": 8,
        "query_size": 8,
        "mode": "fomaml",
        "checkpoint_dir": str(ROOT / "checkpoints" / "centralized"),
        "nodes_dir": str(ROOT / "data" / "nodes"),
        "meta_split": str(ROOT / "data" / "meta_split.json"),
    }
    if path is None:
        return defaults
    with open(path) as f:
        cfg = yaml.safe_load(f)
    maml_cfg = cfg.get("maml", {})
                                                                        
    if "k" in maml_cfg and "inner_steps" not in maml_cfg:
        maml_cfg["inner_steps"] = maml_cfg.pop("k")
    defaults.update(maml_cfg)
    if "hardware" in cfg:
        defaults["device"] = cfg["hardware"].get("device", defaults["device"])
    else:
        defaults["device"] = cfg.get("device", defaults["device"])
    if "data" in cfg:
        data_cfg = cfg["data"]
        if "meta_split" in data_cfg:
            defaults["meta_split"] = data_cfg["meta_split"]
        if "split_file" in data_cfg:
            defaults["meta_split"] = data_cfg["split_file"]
        if "nodes_dir" in data_cfg:
            defaults["nodes_dir"] = data_cfg["nodes_dir"]
    return defaults

def _compute_wer_lora_at_k(
    model,
    processor,
    support_audio: list,
    support_labels: list[str],
    query_audio: list,
    query_labels: list[str],
    k: int,
    inner_lr: float,
    device,
    max_audio_samples: int | None = 48000,
) -> float:
    """WER for LoRAWav2Vec2 at k inner adaptation steps.

    Uses nn.CTCLoss (via model.model(..., labels=...)) for FOMAML-style eval.
    Truncates audio to max_audio_samples to match training conditions; skips clips
    where T_frames < 2*S_labels - 1 after truncation.
    """
    import copy
    import torch
    from jiwer import wer as _wer
    from maml.engine import _encode_audio, _encode_labels

    WAV_STRIDE = 320

    def _clip(audio: torch.Tensor, n_labels: int) -> torch.Tensor | None:
        a = audio[:max_audio_samples] if (max_audio_samples and len(audio) > max_audio_samples) else audio
        if len(a) // WAV_STRIDE < 2 * n_labels - 1:
            return None
        return a

    model_copy = copy.deepcopy(model)
    dtype = next(model_copy.model.parameters()).dtype
    model_copy.model.eval()

    if k > 0:
        trainable_params = [p for p in model_copy.model.parameters() if p.requires_grad]

        for _ in range(k):
            losses = []
            for audio, text in zip(support_audio, support_labels):
                label_ids = _encode_labels(text, processor, device)
                clipped = _clip(audio, label_ids.shape[-1])
                if clipped is None:
                    continue
                input_values = _encode_audio(clipped, processor, device, dtype)
                out = model_copy.model(input_values=input_values, labels=label_ids)
                if not torch.isfinite(out.loss):
                    continue
                losses.append(out.loss)
            if not losses:
                continue
            total_loss = torch.stack(losses).mean()
            grads = torch.autograd.grad(total_loss, trainable_params, allow_unused=True)
            with torch.no_grad():
                for p, g in zip(trainable_params, grads):
                    if g is not None:
                        p.data = p.data - inner_lr * g
            del total_loss, losses, grads

    model_copy.model.eval()
    hypotheses, references = [], []
    with torch.no_grad():
        for audio, text in zip(query_audio, query_labels):
            clipped = _clip(audio, 1)                         
            if clipped is None:
                continue
            input_values = _encode_audio(clipped, processor, device, dtype)
            out = model_copy.model(input_values=input_values)
            pred_ids = torch.argmax(out.logits, dim=-1)
            hyp = processor.batch_decode(pred_ids)[0]
            hypotheses.append(hyp)
            references.append(text)

    del model_copy
    if not hypotheses:
        return 1.0
    return float(_wer(references, hypotheses))

def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.rounds is not None:
        cfg["rounds"] = args.rounds
    if args.device is not None:
        cfg["device"] = args.device
    if args.outer_lr is not None:
        cfg["outer_lr"] = args.outer_lr
    if args.inner_steps is not None:
        cfg["inner_steps"] = args.inner_steps
    eval_only = args.eval_only
    max_updates = args.max_updates
    output_name = args.output_name
    clip_norm = args.clip_norm if args.clip_norm is not None else 20.0
    init_ckpt = args.init_ckpt
    if args.mode is not None:
        cfg["mode"] = args.mode

    import torch
    from data.task_sampler import VoiceTaskSampler
    from maml.engine import MAMLEngine, compute_wer_at_k, _encode_audio, _encode_labels
    from maml.tracking import Tracker

    mode = cfg.get("mode", "fomaml")
    device = torch.device(cfg["device"])
    print(f"Device: {device}  Mode: {mode}", flush=True)
    print(f"Rounds: {cfg['rounds']}  outer_lr: {cfg['outer_lr']}  inner_steps: {cfg['inner_steps']}", flush=True)
    if max_updates:
        print(f"Max updates: {max_updates} (matched-update mode)")

    experiment_name = cfg.get("experiment_name",
                              cfg.get("logging", {}).get("experiment_name", f"maml_{mode}") if isinstance(cfg.get("logging"), dict) else f"maml_{mode}")
    run_name = f"{output_name}_r{cfg['rounds']}_k{cfg['inner_steps']}"
    tracker = Tracker(experiment_name=experiment_name, run_name=run_name)

    nodes_dir = Path(cfg["nodes_dir"])

    meta_split_path = Path(cfg["meta_split"])
    meta_train_hashes: set[str] | None = None
    meta_val_hashes: set[str] | None = None
    if meta_split_path.exists():
        import json
        meta_split = json.loads(meta_split_path.read_text())
        meta_train_hashes = set(meta_split.get("meta_train", []))
        meta_val_hashes = set(meta_split.get("meta_val", []))
        print(f"meta_split loaded: {len(meta_train_hashes)} meta-train speakers")
    else:
        print(f"WARNING: meta_split.json not found at {meta_split_path}")
        print("Using all nodes in nodes_dir (ensure meta-test speakers are not present)")

    all_node_dirs = sorted(d for d in nodes_dir.iterdir() if d.is_dir()
                           if (d / "features.pt").exists())

    if meta_train_hashes is not None:
        node_dirs = [d for d in all_node_dirs if d.name in meta_train_hashes]
        val_dirs = [d for d in all_node_dirs if d.name in (meta_val_hashes or set())]
    else:
        node_dirs = all_node_dirs
        val_dirs = []

    if not node_dirs:
        print("ERROR: No meta-train node directories found.")
        print(f"  nodes_dir: {nodes_dir}")
        print("  Run: python data/prepare_vctk.py && python data/features.py")
        sys.exit(1)
    print(f"Meta-train nodes: {[d.name[:8] for d in node_dirs]}")
    if val_dirs:
        print(f"Meta-val nodes  : {[d.name[:8] for d in val_dirs]}")

    if mode == "lora_maml":
        from models.lora_wav2vec2 import LoRAWav2Vec2
        from models.lora_wav2vec2 import load_processor
        print("Loading LoRAWav2Vec2...")
        model = LoRAWav2Vec2(device=device)
        print(f"  Trainable: {model.count_trainable():,}  Frozen: {model.count_frozen():,}")
    else:
        from models.wav2vec2_maml import Wav2Vec2MAML, load_processor
        print("Loading Wav2Vec2MAML...")
        model = Wav2Vec2MAML(device=device)

    processor = load_processor()

    samplers = [
        VoiceTaskSampler(d, cfg["support_size"], cfg["query_size"])
        for d in node_dirs
    ]

    tracker.log_params({
        "mode": mode,
        "rounds": cfg["rounds"],
        "inner_steps": cfg["inner_steps"],
        "inner_lr": cfg["inner_lr"],
        "outer_lr": cfg["outer_lr"],
        "outer_optimizer": cfg.get("outer_optimizer", "sgd"),
        "lr_schedule": cfg.get("lr_schedule", "constant"),
        "lr_warmup_rounds": cfg.get("lr_warmup_rounds", 0),
        "tasks_per_node": cfg.get("tasks_per_node", 1),
        "support_size": cfg["support_size"],
        "query_size": cfg["query_size"],
        "max_audio_samples": str(cfg.get("max_audio_samples", "none")),
        "clip_norm": clip_norm,
        "n_meta_train_nodes": len(node_dirs),
        "n_meta_val_nodes": len(val_dirs),
        "device": str(device),
        "output_name": output_name,
    })

    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    round_metrics: list[dict] = []

    final_path = ckpt_dir / f"{output_name}.pt"

    def _save_ckpt(path: Path) -> None:
        if mode == "lora_maml":
            state = {n: p.data.clone()
                     for n, p in model.model.named_parameters() if p.requires_grad}
            torch.save(state, path)
        else:
            torch.save(model.get_encoder_state_dict(), path)

    def _load_ckpt(path: Path) -> None:
        if mode == "lora_maml":
            state = torch.load(path, map_location=device, weights_only=True)
            with torch.no_grad():
                for n, p in model.model.named_parameters():
                    if n in state:
                        p.data.copy_(state[n])
        else:
            model.set_encoder_state_dict(torch.load(path, map_location=device, weights_only=True))

    if init_ckpt:
        ckpt_path_init = Path(init_ckpt)
        if not ckpt_path_init.exists():
            print(f"ERROR: --init_ckpt path does not exist: {init_ckpt}")
            sys.exit(1)
        print(f"Loading init checkpoint: {init_ckpt}")
        _load_ckpt(ckpt_path_init)

    if eval_only:
        if not final_path.exists():
            print(f"ERROR: --eval_only requires {final_path}")
            sys.exit(1)
        print(f"Loading saved θ* from {final_path}")
        _load_ckpt(final_path)
        metrics_path = ckpt_dir / "training_metrics.json"
        if metrics_path.exists():
            import json as _json
            round_metrics = _json.loads(metrics_path.read_text())
        print("Skipping training (--eval_only)")
    else:
        import math

        max_audio_samples = cfg.get("max_audio_samples", None)
        tasks_per_node = cfg.get("tasks_per_node", 1)
        engine = MAMLEngine(model, processor, cfg["inner_steps"], cfg["inner_lr"], device,
                            mode=mode, max_audio_samples=max_audio_samples)
        outer_lr = cfg["outer_lr"]
        outer_params = model.get_outer_loop_params()

        use_adamw = cfg.get("outer_optimizer", "sgd").lower() == "adamw"
        weight_decay = cfg.get("outer_weight_decay", 1e-4)
        warmup_rounds = cfg.get("lr_warmup_rounds", 0)
        use_cosine = cfg.get("lr_schedule", "constant").lower() == "cosine"
        total_rounds = cfg["rounds"]

        if use_adamw:
            optimizer = torch.optim.AdamW(
                outer_params, lr=outer_lr, weight_decay=weight_decay, betas=(0.9, 0.999)
            )
            print(f"Outer optimizer: AdamW(lr={outer_lr}, wd={weight_decay})")
        else:
            optimizer = None
            print(f"Outer optimizer: SGD(lr={outer_lr})")

        def _current_lr(rnd: int) -> float:
            if rnd <= warmup_rounds:
                return outer_lr * rnd / max(warmup_rounds, 1)
            if not use_cosine:
                return outer_lr
            progress = (rnd - warmup_rounds) / max(total_rounds - warmup_rounds, 1)
            return 1e-6 + 0.5 * (outer_lr - 1e-6) * (1 + math.cos(math.pi * progress))

        total_updates = 0
        max_upd = max_updates

        print(f"\nStarting centralized {mode.upper()} ({total_rounds} rounds, "
              f"{len(samplers)} nodes × {tasks_per_node} tasks = "
              f"{len(samplers) * tasks_per_node} updates/round, clip_norm={clip_norm})")
        if max_audio_samples:
            print(f"  Clip cap: {max_audio_samples} samples ({max_audio_samples/16000:.1f}s)")
        else:
            print("  Clip cap: none (full-length clips)")

        for rnd in range(1, total_rounds + 1):
            round_grads = [torch.zeros_like(p) for p in outer_params]
            round_query_losses: list[float] = []

            for sampler in samplers:
                for _ in range(tasks_per_node):
                    if max_upd is not None and total_updates >= max_upd:
                        break
                    task = sampler.sample_task()
                    node_grads, query_loss_val = engine.compute_meta_gradient(
                        task, return_query_loss=True
                    )
                    for acc, g in zip(round_grads, node_grads):
                        acc.add_(g)
                    round_query_losses.append(query_loss_val)
                    total_updates += 1
                    del node_grads

            if not round_query_losses:
                print(f"  Reached max_updates={max_upd} after round {rnd - 1}")
                break

            n_tasks = len(round_query_losses)
            avg_grads = [g / n_tasks for g in round_grads]

            total_norm = sum(g.norm() ** 2 for g in avg_grads) ** 0.5
            clip_coef = clip_norm / (total_norm + 1e-6)
            if clip_coef < 1.0:
                for g in avg_grads:
                    g.mul_(clip_coef)

            lr_now = _current_lr(rnd)
            if use_adamw:
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_now
                optimizer.zero_grad()
                for p, g in zip(outer_params, avg_grads):
                    p.grad = g.clone()
                optimizer.step()
            else:
                with torch.no_grad():
                    for p, g in zip(outer_params, avg_grads):
                        p.data.sub_(lr_now * g)

            avg_loss = sum(round_query_losses) / len(round_query_losses)
            grad_norm_val = float(total_norm)
            round_metrics.append({
                "round": rnd,
                "avg_query_loss": avg_loss,
                "grad_norm": grad_norm_val,
                "total_updates": total_updates,
                "lr": lr_now,
            })
            tracker.log_metrics({
                "train/loss": avg_loss,
                "train/grad_norm": grad_norm_val,
                "train/total_updates": float(total_updates),
                "train/lr": lr_now,
            }, step=rnd)

            if rnd % 10 == 0 or rnd == 1:
                ckpt_path = ckpt_dir / f"{output_name}_round_{rnd:04d}.pt"
                _save_ckpt(ckpt_path)
                print(f"  Round {rnd:3d}: loss={avg_loss:.4f}  grad_norm={grad_norm_val:.3f}"
                      f"  lr={lr_now:.2e}  updates={total_updates}  → {ckpt_path.name}",
                      flush=True)

            if max_upd is not None and total_updates >= max_upd:
                print(f"  Reached max_updates={max_upd}")
                break

        _save_ckpt(final_path)
        print(f"\nFinal θ* saved: {final_path}  (total_updates={total_updates})", flush=True)

    EVAL_INNER_STEPS = cfg["inner_steps"]
    EVAL_INNER_LR = cfg["inner_lr"]

    eval_support = cfg.get("eval_support_size", cfg["support_size"])
    eval_query = cfg.get("eval_query_size", cfg["query_size"])
    eval_dirs = val_dirs if val_dirs else node_dirs[:4]
    eval_samplers = [
        VoiceTaskSampler(d, eval_support, eval_query)
        for d in eval_dirs
    ]

    print(
        f"\nGate evaluation (meta-val, no perturbation):\n"
        f"  speakers: {[d.name[:8] for d in eval_dirs]}\n"
        f"  adaptation: {EVAL_INNER_STEPS} steps @ lr={EVAL_INNER_LR}\n"
    )

    gate_results = {}
    passed = 0
    for i, (node_dir, sampler) in enumerate(zip(eval_dirs, eval_samplers)):
        task = sampler.sample_task()

        max_audio_samples_eval = cfg.get("max_audio_samples",
                                          48000 if mode == "lora_maml" else None)
        if mode == "lora_maml":
            wer0 = _compute_wer_lora_at_k(
                model, processor,
                task.support_audio, task.support_labels,
                task.query_audio, task.query_labels,
                k=0, inner_lr=EVAL_INNER_LR, device=device,
                max_audio_samples=max_audio_samples_eval,
            )
            wer_k = _compute_wer_lora_at_k(
                model, processor,
                task.support_audio, task.support_labels,
                task.query_audio, task.query_labels,
                k=EVAL_INNER_STEPS, inner_lr=EVAL_INNER_LR, device=device,
                max_audio_samples=max_audio_samples_eval,
            )
        else:
            wer0 = compute_wer_at_k(
                model, processor,
                task.support_audio, task.support_labels,
                task.query_audio, task.query_labels,
                k=0, inner_lr=EVAL_INNER_LR, device=device,
            )
            wer_k = compute_wer_at_k(
                model, processor,
                task.support_audio, task.support_labels,
                task.query_audio, task.query_labels,
                k=EVAL_INNER_STEPS, inner_lr=EVAL_INNER_LR, device=device,
            )

        ok = wer_k < wer0
        if ok:
            passed += 1
        flag = "PASS" if ok else "FAIL"
        print(f"  {node_dir.name[:8]}: WER(k=0)={wer0:.4f}  WER(k={EVAL_INNER_STEPS})={wer_k:.4f}  [{flag}]")
        gate_results[node_dir.name] = {
            "wer_k0": wer0,
            "wer_k_adapted": wer_k,
            "eval_inner_steps": EVAL_INNER_STEPS,
            "passed": ok,
        }

    metrics_path = ckpt_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps(round_metrics, indent=2))

    gate_path = ckpt_dir / "gate_eval_results.json"
    gate_path.write_text(json.dumps(gate_results, indent=2))
    print(f"\nGate results saved: {gate_path}")
    print(f"Training metrics saved: {metrics_path}")

    tracker.log_gate_results(gate_results, EVAL_INNER_STEPS)
    tracker.log_training_summary(round_metrics)
    tracker.log_artifact(final_path)
    tracker.end()

    n_eval = len(eval_dirs)
    threshold = max(1, n_eval - 1)                                  
    print(f"\nGate: {passed}/{n_eval} meta-val speakers improved (need ≥{threshold})")
    if passed >= threshold:
        print("GATE PASSED — proceed to federated training (Step 6)")
    else:
        print("GATE FAILED — WER(k=K) ≥ WER(k=0) on too many meta-val speakers")
        print("Check: gradient norms, inner_lr, number of rounds")
        sys.exit(1)

if __name__ == "__main__":
    main()
