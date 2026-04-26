"""
Flower server entry point for FedLoRA-MAML (Phase 2).

Loads initial θ* (LoRA + lm_head params), then runs FL rounds using
PerFedAvgStrategy. Saves θ* checkpoint after training.

Checkpoint priority:
  1. checkpoints/federated/theta_star_lora.pt  — resume a previous federated run
  2. checkpoints/centralized/theta_star.pt     — seed from centralized gate
  3. Fresh pretrained weights (LoRAWav2Vec2 defaults)

The frozen backbone (~94.5M params) is never transmitted. Only the ~320K
LoRA + lm_head params travel over the wire.

Usage:
    python federated/server_lora.py --config configs/vctk_lora_poc.yaml
    python federated/server_lora.py --config configs/vctk_lora_poc.yaml --rounds 50
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

ROOT = Path(__file__).parent.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--outer_lr", type=float, default=None)
    return p.parse_args()


def load_config(path: str | None) -> dict:
    from maml import _load_yaml

    defaults = {
        "rounds": 200,
        "outer_lr": 5e-4,
        "min_available_clients": 12,
        "fraction_fit": 0.5,
        "checkpoint_dir": str(ROOT / "checkpoints" / "federated"),
        "port": 8080,
        "experiment_name": "fedlora_maml_vctk",
    }
    if path is None:
        return defaults
    cfg = _load_yaml(path)

    maml_cfg = cfg.get("maml", {})
    if "k" in maml_cfg and "rounds" not in maml_cfg:
        pass  # k is inner_steps, not rounds
    if "rounds" in maml_cfg:
        defaults["rounds"] = maml_cfg["rounds"]
    if "outer_lr" in maml_cfg:
        defaults["outer_lr"] = maml_cfg["outer_lr"]

    fed_cfg = cfg.get("federated", {})
    if "num_nodes" in fed_cfg:
        defaults["min_available_clients"] = fed_cfg["num_nodes"]
    if "cohort_fraction" in fed_cfg:
        defaults["fraction_fit"] = fed_cfg["cohort_fraction"]
    if "num_rounds" in fed_cfg and "rounds" not in maml_cfg:
        defaults["rounds"] = fed_cfg["num_rounds"]

    log_cfg = cfg.get("logging", {})
    if "experiment_name" in log_cfg:
        defaults["experiment_name"] = log_cfg["experiment_name"]

    return defaults


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.rounds is not None:
        cfg["rounds"] = args.rounds
    if args.port is not None:
        cfg["port"] = args.port
    if args.outer_lr is not None:
        cfg["outer_lr"] = args.outer_lr

    import torch
    import flwr as fl
    from models.lora_wav2vec2 import LoRAWav2Vec2, load_processor
    from federated.strategy_maml import PerFedAvgStrategy
    from flwr.common import ndarrays_to_parameters

    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    final_ckpt = ckpt_dir / "theta_star_lora.pt"
    centralized_ckpt_candidates = [
        ROOT / "checkpoints" / "centralized" / "theta_star.pt",
        ROOT / "checkpoints" / "centralized" / "theta_star_lora.pt",
    ]

    print("Initializing LoRAWav2Vec2 for θ* (LoRA + lm_head params)...", flush=True)
    model = LoRAWav2Vec2(device="cpu")
    print(
        f"  Trainable: {model.count_trainable():,} | Frozen: {model.count_frozen():,}",
        flush=True,
    )

    # Order matches get_outer_loop_params() — both iterate model.parameters().
    trainable_named = [
        (n, p) for n, p in model.model.named_parameters() if p.requires_grad
    ]
    param_names = [n for n, _ in trainable_named]

    round_ckpts = sorted(ckpt_dir.glob("theta_star_lora_round_*.pt"))
    resume_ckpt = round_ckpts[-1] if round_ckpts else (final_ckpt if final_ckpt.exists() else None)
    # Round offset keeps global round numbers consistent across session restarts.
    round_offset = int(round_ckpts[-1].stem.split("_")[-1]) if round_ckpts else 0

    if resume_ckpt is not None:
        print(f"  Resuming from {resume_ckpt.name} (round offset={round_offset})", flush=True)
        ckpt = torch.load(resume_ckpt, map_location="cpu", weights_only=True)
        with torch.no_grad():
            for n, p in trainable_named:
                if n in ckpt:
                    p.data.copy_(ckpt[n])
    else:
        centralized_ckpt = next(
            (c for c in centralized_ckpt_candidates if c.exists()), None
        )
        if centralized_ckpt is not None:
            print(f"  Seeding from centralized checkpoint: {centralized_ckpt}", flush=True)
            ckpt = torch.load(centralized_ckpt, map_location="cpu", weights_only=True)
            with torch.no_grad():
                for n, p in trainable_named:
                    if n in ckpt:
                        p.data.copy_(ckpt[n])
        else:
            print("  No checkpoint found — using fresh LoRA init (B=0, A~N(0,0.02))", flush=True)

    initial_ndarrays = [
        p.detach().float().cpu().numpy() for _, p in trainable_named
    ]
    initial_params = ndarrays_to_parameters(initial_ndarrays)
    del initial_ndarrays, trainable_named

    del model
    gc.collect()

    from maml import _load_yaml
    maml_cfg = _load_yaml(args.config).get("maml", {}) if args.config else {}

    strategy = PerFedAvgStrategy(
        initial_parameters=initial_params,
        outer_lr=cfg["outer_lr"],
        min_available_clients=cfg["min_available_clients"],
        fraction_fit=cfg["fraction_fit"],
        checkpoint_dir=str(ckpt_dir),
        checkpoint_every=10,
        param_names=param_names,
        experiment_name=cfg["experiment_name"],
        round_offset=round_offset,
        weight_decay=maml_cfg.get("outer_weight_decay", 1e-4),
        total_rounds=cfg["rounds"],
        warmup_rounds=maml_cfg.get("lr_warmup_rounds", 20),
    )

    cohort_size = max(1, int(cfg["fraction_fit"] * cfg["min_available_clients"]))
    address = f"0.0.0.0:{cfg['port']}"
    print(
        f"Starting Flower server on {address}  "
        f"rounds={cfg['rounds']}  "
        f"cohort={cohort_size}/{cfg['min_available_clients']}  "
        f"outer_lr={cfg['outer_lr']}",
        flush=True,
    )

    history = fl.server.start_server(
        server_address=address,
        config=fl.server.ServerConfig(num_rounds=cfg["rounds"]),
        strategy=strategy,
        grpc_max_message_length=64 * 1024 * 1024,  # 64 MB — LoRA is ~640 KB/client
    )

    # Reconstruct state dict from final theta_star arrays and param_names
    final_arrays = strategy._theta_star
    state_dict = {
        name: torch.tensor(arr.astype("float32"))
        for name, arr in zip(param_names, final_arrays)
    }
    torch.save(state_dict, final_ckpt)
    print(f"\nFinal θ* saved: {final_ckpt}", flush=True)
    strategy.close()
    return history


if __name__ == "__main__":
    main()
