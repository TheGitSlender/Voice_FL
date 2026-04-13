"""
Flower server entry point.

Loads initial θ* (or initializes fresh), then runs FL rounds using
PerFedAvgStrategy. Saves θ* checkpoint after each round.

Usage (direct):
    python federated/server_maml.py --config configs/poc.yaml

Usage (Docker):
    Launched by docker-compose.yml as the 'server' service.
"""

from __future__ import annotations

import argparse
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
        "rounds": 20,
        "outer_lr": 2e-4,
        "min_available_clients": 5,
        "fraction_fit": 1.0,
        "checkpoint_dir": str(ROOT / "checkpoints" / "federated"),
        "port": 8080,
    }
    if path is None:
        return defaults
    with open(path) as f:
        cfg = yaml.safe_load(f)
    defaults.update(cfg.get("federated", {}))
    defaults.update(cfg.get("maml", {}))
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
    from models.wav2vec2_maml import Wav2Vec2MAML, load_processor
    from federated.strategy_maml import PerFedAvgStrategy
    from flwr.common import ndarrays_to_parameters

    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Load or initialize θ*
    final_ckpt = ckpt_dir / "theta_star.pt"
    centralized_ckpt = ROOT / "checkpoints" / "centralized" / "theta_star.pt"

    print("Initializing server model (θ*)...")
    model = Wav2Vec2MAML(device="cpu")
    processor = load_processor()

    # Load checkpoint FIRST (while parametrizations are still active), then
    # strip them.  Centralized checkpoints are saved with parametrization keys
    # (original0/original1); the federated checkpoint is saved without (plain
    # weight).  We handle both by loading before stripping.
    if final_ckpt.exists():
        print(f"  Resuming from {final_ckpt}")
        ckpt = torch.load(final_ckpt, map_location="cpu", weights_only=True)
        # Federated checkpoint may already have plain `weight` key — if so,
        # temporarily strip parametrizations before loading.
        conv = model.model.wav2vec2.encoder.pos_conv_embed.conv
        if hasattr(conv, "parametrizations") and "weight" in conv.parametrizations:
            if "encoder.pos_conv_embed.conv.weight" in ckpt:
                torch.nn.utils.parametrize.remove_parametrizations(conv, "weight")
        model.set_encoder_state_dict(ckpt)
    elif centralized_ckpt.exists():
        print(f"  Seeding from centralized checkpoint: {centralized_ckpt}")
        # Centralized checkpoint has parametrization keys — load with them active.
        model.set_encoder_state_dict(
            torch.load(centralized_ckpt, map_location="cpu", weights_only=True)
        )
    else:
        print("  Using fresh pretrained weights")

    # Now strip parametrizations so outer_loop_params layout matches clients
    # (clients call to_bf16() which removes parametrizations internally).
    # 211 params (original0 + original1 + …) → 210 params (plain weight + …)
    conv = model.model.wav2vec2.encoder.pos_conv_embed.conv
    if hasattr(conv, "parametrizations") and "weight" in conv.parametrizations:
        torch.nn.utils.parametrize.remove_parametrizations(conv, "weight")

    import gc

    # Use float16 for wire transfer — halves message size (~189 MB vs 378 MB)
    # so 5 concurrent sends in Round 3 use ~945 MB instead of ~1.9 GB.
    initial_ndarrays = [
        p.detach().half().cpu().numpy()
        for p in model.get_outer_loop_params()
    ]
    initial_params = ndarrays_to_parameters(initial_ndarrays)
    del initial_ndarrays

    # Free the model object — it is not needed during FL rounds.
    # The strategy holds θ* as numpy arrays; the model is only needed at the
    # end to reconstruct the final state dict for saving.
    param_names = [name for name, _ in model.model.wav2vec2.named_parameters()]
    del model
    gc.collect()

    strategy = PerFedAvgStrategy(
        initial_parameters=initial_params,
        outer_lr=cfg["outer_lr"],
        min_available_clients=cfg["min_available_clients"],
        fraction_fit=cfg["fraction_fit"],
    )

    address = f"0.0.0.0:{cfg['port']}"
    print(f"Starting Flower server on {address}  rounds={cfg['rounds']}")

    history = fl.server.start_server(
        server_address=address,
        config=fl.server.ServerConfig(num_rounds=cfg["rounds"]),
        strategy=strategy,
        grpc_max_message_length=512 * 1024 * 1024,  # 512 MB for large model params
    )

    # Save final θ* — upcast float16 wire arrays back to float32 for the checkpoint
    final_arrays = strategy._theta_star
    state_dict = {
        name: torch.tensor(arr.astype("float32"))
        for name, arr in zip(param_names, final_arrays)
    }
    torch.save(state_dict, final_ckpt)
    print(f"\nFinal θ* saved: {final_ckpt}")
    return history


if __name__ == "__main__":
    main()
