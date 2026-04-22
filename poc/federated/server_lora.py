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
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--outer_lr", type=float, default=None)
    return p.parse_args()


def load_config(path: str | None) -> dict:
    import yaml

    defaults = {
        "rounds": 200,
        "outer_lr": 5e-4,
        "min_available_clients": 12,
        "fraction_fit": 0.5,
        "checkpoint_dir": str(ROOT / "checkpoints" / "federated"),
        "port": 8080,
    }
    if path is None:
        return defaults
    with open(path) as f:
        cfg = yaml.safe_load(f)

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

    # Collect (name, param) pairs for trainable params — order matches
    # get_outer_loop_params() since both iterate model.parameters() in
    # PyTorch registration order.
    trainable_named = [
        (n, p) for n, p in model.model.named_parameters() if p.requires_grad
    ]
    param_names = [n for n, _ in trainable_named]

    if final_ckpt.exists():
        print(f"  Resuming from {final_ckpt}", flush=True)
        ckpt = torch.load(final_ckpt, map_location="cpu", weights_only=True)
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

    # Extract initial numpy arrays (float16 for wire efficiency)
    initial_ndarrays = [
        p.detach().half().cpu().numpy() for _, p in trainable_named
    ]
    initial_params = ndarrays_to_parameters(initial_ndarrays)
    del initial_ndarrays, trainable_named

    del model
    gc.collect()

    strategy = PerFedAvgStrategy(
        initial_parameters=initial_params,
        outer_lr=cfg["outer_lr"],
        min_available_clients=cfg["min_available_clients"],
        fraction_fit=cfg["fraction_fit"],
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
    return history


if __name__ == "__main__":
    main()
