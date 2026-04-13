"""
federated/simulation_maml.py — Flower VCE simulation for Per-FedAvg MAML

Entry point for local FL simulation using Flower's Virtual Client Engine (VCE).
VCE runs all clients sequentially on the same machine — no gRPC.
All 20 nodes share the same GPU via time-sliced execution.

Usage:
    python federated/simulation_maml.py
    python federated/simulation_maml.py --config configs/experiment.yaml

Hardware:
  dev.yaml:        RTX 4070 Super, FOMAML, no DP
  experiment.yaml: Lightning AI A100, full MAML, DP enabled
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

import mlflow
import torch
import yaml


def run_simulation(config_path: str = "configs/dev.yaml") -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    import flwr as fl
    from transformers import Wav2Vec2Processor

    from data.task_sampler import VoiceTaskSampler
    from federated.client_maml import MAMLClient
    from federated.strategy_maml import PerFedAvgStrategy
    from maml.engine import MAMLConfig, MAMLEngine
    from models.wav2vec2_maml import Wav2Vec2MAML
    from privacy.dp_meta import DPConfig
    from privacy.rdp_accountant import RDPAccountant

    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    if cfg["mlflow"].get("tracking_uri"):
        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])

    with mlflow.start_run() as run:
        mlflow.log_params({
            "maml_mode": cfg["maml"]["mode"],
            "k": cfg["maml"]["k"],
            "inner_lr": cfg["maml"]["inner_lr"],
            "outer_lr": cfg["maml"]["outer_lr"],
            "num_rounds": cfg["federated"]["num_rounds"],
            "cohort_fraction": cfg["federated"]["cohort_fraction"],
            "dp_enabled": cfg["privacy"]["enabled"],
            "epsilon_target": cfg["privacy"]["epsilon"] if cfg["privacy"]["enabled"] else "off",
            "adaptation_mode": cfg["maml"]["adaptation_mode"],
            "config_path": config_path,
        })

        device = cfg["hardware"]["device"]

        # Load processor once — shared across all simulated clients (memory efficiency)
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

        node_dirs = sorted(Path(cfg["data"]["node_dir"]).glob("node_*"))
        if not node_dirs:
            print(f"No node directories found in {cfg['data']['node_dir']}")
            sys.exit(1)

        print(f"Found {len(node_dirs)} nodes for simulation")

        # Pre-load all task samplers — avoids repeated disk I/O per round
        task_samplers: Dict[str, VoiceTaskSampler] = {}
        for d in node_dirs:
            node_idx = d.name.split("_")[1]
            task_samplers[node_idx] = VoiceTaskSampler(
                str(d),
                K=cfg["maml"]["support_size"],
                Q=cfg["maml"]["query_size"],
                processor=processor,
                device=device,
            )

        def client_fn(cid: str) -> MAMLClient:
            """Factory: creates one MAMLClient per simulated node."""
            model = Wav2Vec2MAML(
                model_name=cfg["model"]["name"],
                mode=cfg["maml"]["adaptation_mode"],
            ).to(device)

            maml_config = MAMLConfig(
                mode=cfg["maml"]["mode"],
                k=cfg["maml"]["k"],
                inner_lr=cfg["maml"]["inner_lr"],
                outer_lr=cfg["maml"]["outer_lr"],
                support_size=cfg["maml"]["support_size"],
                query_size=cfg["maml"]["query_size"],
                adaptation_mode=cfg["maml"]["adaptation_mode"],
                use_bf16=cfg["hardware"].get("use_bf16", True),
            )
            engine = MAMLEngine(model, maml_config)

            dp_config = DPConfig(
                epsilon=cfg["privacy"]["epsilon"],
                delta=cfg["privacy"]["delta"],
                C=cfg["privacy"].get("C", 1.0),
                enabled=cfg["privacy"]["enabled"],
                sample_rate=cfg["federated"]["cohort_fraction"],
            )

            accountant = RDPAccountant(
                target_epsilon=cfg["privacy"]["epsilon"],
                target_delta=cfg["privacy"]["delta"],
            )

            return MAMLClient(
                node_id=f"node_{cid}",
                model=model,
                engine=engine,
                task_sampler=task_samplers[cid],
                dp_config=dp_config,
                accountant=accountant,
            )

        # BAE (optional)
        bae = None
        if cfg.get("bae", {}).get("enabled", False):
            from security.bae_maml import BAEConfig, BehavioralAnalysisEngine
            bae = BehavioralAnalysisEngine(BAEConfig())

        strategy = PerFedAvgStrategy(
            outer_lr=cfg["maml"]["outer_lr"],
            bae=bae,
            mlflow_run=run,
            fraction_fit=cfg["federated"]["cohort_fraction"],
            fraction_evaluate=1.0,
            min_fit_clients=max(2, int(len(node_dirs) * cfg["federated"]["cohort_fraction"])),
            min_evaluate_clients=min(4, len(node_dirs)),
            min_available_clients=len(node_dirs),
        )

        # Patch strategy to capture initial params from first client
        _orig_aggregate_fit = strategy.aggregate_fit

        def _aggregate_fit_with_init(server_round, results, failures):
            if strategy.current_params is None and results:
                from flwr.common import parameters_to_ndarrays
                strategy.current_params = parameters_to_ndarrays(
                    results[0][1].parameters
                )
            return _orig_aggregate_fit(server_round, results, failures)

        strategy.aggregate_fit = _aggregate_fit_with_init

        gpu_per_client = cfg["hardware"].get("gpu_per_client", 0.5)
        client_resources: Dict = {"num_cpus": 1}
        if torch.cuda.is_available():
            client_resources["num_gpus"] = gpu_per_client

        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=len(node_dirs),
            config=fl.server.ServerConfig(num_rounds=cfg["federated"]["num_rounds"]),
            strategy=strategy,
            client_resources=client_resources,
        )

        print(f"\nSimulation complete. MLflow run: {run.info.run_id}")
        print(f"View results: mlflow ui --backend-store-uri {cfg['mlflow'].get('tracking_uri', './mlruns')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dev.yaml")
    args = parser.parse_args()
    run_simulation(args.config)
