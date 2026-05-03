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
        "rounds": 20,
        "outer_lr": 2e-4,
        "min_available_clients": 5,
        "fraction_fit": 1.0,
        "checkpoint_dir": str(ROOT / "checkpoints" / "federated"),
        "port": 8080,
    }
    if path is None:
        return defaults
    cfg = _load_yaml(path)
    defaults.update(cfg.get("federated", {}))
    defaults.update(cfg.get("maml", {}))
    defaults["_aggregation"] = cfg.get("aggregation", {})
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

    final_ckpt = ckpt_dir / "theta_star.pt"
    centralized_ckpt = ROOT / "checkpoints" / "centralized" / "theta_star.pt"

    print("Initializing server model (θ*)...")
    model = Wav2Vec2MAML(device="cpu")
    processor = load_processor()

    if final_ckpt.exists():
        print(f"  Resuming from {final_ckpt}")
        ckpt = torch.load(final_ckpt, map_location="cpu", weights_only=True)
                                                                           
        conv = model.model.wav2vec2.encoder.pos_conv_embed.conv
        if hasattr(conv, "parametrizations") and "weight" in conv.parametrizations:
            if "encoder.pos_conv_embed.conv.weight" in ckpt:
                torch.nn.utils.parametrize.remove_parametrizations(conv, "weight")
        model.set_encoder_state_dict(ckpt)
    elif centralized_ckpt.exists():
        print(f"  Seeding from centralized checkpoint: {centralized_ckpt}")
                                                                                  
        model.set_encoder_state_dict(
            torch.load(centralized_ckpt, map_location="cpu", weights_only=True)
        )
    else:
        print("  Using fresh pretrained weights")

    conv = model.model.wav2vec2.encoder.pos_conv_embed.conv
    if hasattr(conv, "parametrizations") and "weight" in conv.parametrizations:
        torch.nn.utils.parametrize.remove_parametrizations(conv, "weight")

    import gc

    initial_ndarrays = [
        p.detach().half().cpu().numpy()
        for p in model.get_outer_loop_params()
    ]
    initial_params = ndarrays_to_parameters(initial_ndarrays)
    del initial_ndarrays

    param_names = [name for name, _ in model.model.wav2vec2.named_parameters()]
    del model
    gc.collect()

    agg_cfg = cfg.get("_aggregation", {})
    strategy = PerFedAvgStrategy(
        initial_parameters=initial_params,
        outer_lr=cfg["outer_lr"],
        min_available_clients=cfg["min_available_clients"],
        fraction_fit=cfg["fraction_fit"],
        robust_method=agg_cfg.get("robust_method", "mean"),
        trim_ratio=agg_cfg.get("trim_ratio", 0.1),
        n_byzantine=agg_cfg.get("n_byzantine", 1),
        norm_filter_multiplier=agg_cfg.get("norm_filter_multiplier", 2.0),
        secure_agg=bool(agg_cfg.get("secure_agg", False)),
        secagg_mask_scale=agg_cfg.get("secagg_mask_scale", 0.01),
    )

    address = f"0.0.0.0:{cfg['port']}"
    print(f"Starting Flower server on {address}  rounds={cfg['rounds']}")

    history = fl.server.start_server(
        server_address=address,
        config=fl.server.ServerConfig(num_rounds=cfg["rounds"]),
        strategy=strategy,
        grpc_max_message_length=512 * 1024 * 1024,                                 
    )

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
