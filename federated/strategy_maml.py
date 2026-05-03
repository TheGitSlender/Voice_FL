"""
PerFedAvg aggregation strategy for Flower.

On each round:
  1. Collect meta-gradients from all selected clients (encoded as parameters)
  2. Compute weighted average: avg_grad = Σ(n_i * g_i) / Σ(n_i)
  3. Apply outer update: θ* ← θ* − β · avg_grad
  4. Return updated θ* as the new global parameters

This is NOT standard FedAvg weight averaging (Invariant I4).
The server never sees node data, speaker IDs, or lm_head parameters.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import flwr as fl
from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

from maml.tracking import Tracker
from federated.aggregation import RobustAggregator
from federated.secure_agg import SecureAggregator

# Metric keys that clients may send in FitRes.metrics
_CLIENT_METRIC_KEYS = (
    "query_loss",
    "grad_norm",
    "clip_coef",
    "inner_loss_init",
    "inner_loss_final",
)


class PerFedAvgStrategy(fl.server.strategy.Strategy):
    """
    Per-FedAvg: server maintains θ* and applies gradient descent each round.

    Args:
        initial_parameters:    encoder/LoRA weights as Flower Parameters
        outer_lr:              β — server-side learning rate (default 2e-4)
        min_available_clients: minimum clients before server starts
        fraction_fit:          fraction of available clients to use per round
        checkpoint_dir:        if set, save θ* every `checkpoint_every` rounds
        checkpoint_every:      rounds between checkpoints (default 10)
        param_names:           parameter names for checkpoint state_dict keys
        experiment_name:       MLflow experiment name
        tracking_uri:          MLflow tracking URI (overridden by MLFLOW_TRACKING_URI env)
    """

    def __init__(
        self,
        initial_parameters: Parameters,
        outer_lr: float = 2e-4,
        min_available_clients: int = 5,
        fraction_fit: float = 1.0,
        checkpoint_dir: Optional[str] = None,
        checkpoint_every: int = 10,
        param_names: Optional[List[str]] = None,
        experiment_name: str = "fedlora_maml_vctk",
        tracking_uri: str = "mlruns",
        round_offset: int = 0,
        # AdamW outer optimizer hyperparameters
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_eps: float = 1e-8,
        weight_decay: float = 1e-4,
        # Cosine LR schedule with linear warmup
        total_rounds: int = 200,
        warmup_rounds: int = 20,
        # Byzantine-robust aggregation
        robust_method: str = "mean",
        trim_ratio: float = 0.1,
        n_byzantine: int = 1,
        norm_filter_multiplier: float = 2.0,
        # Secure aggregation (pairwise masking)
        secure_agg: bool = False,
        secagg_mask_scale: float = 0.01,
        secagg_base_seed: int = 0,
    ):
        super().__init__()
        self.base_outer_lr = outer_lr
        self.outer_lr = outer_lr  # updated each round by schedule
        self.fraction_fit = fraction_fit
        self.min_available_clients = min_available_clients
        self._checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self._checkpoint_every = checkpoint_every
        self._param_names = param_names

        self._theta_star: list[np.ndarray] = parameters_to_ndarrays(initial_parameters)
        self._round_offset = round_offset

        # AdamW moment buffers.
        self._adam_m: list[np.ndarray] = [np.zeros_like(t, dtype=np.float32) for t in self._theta_star]
        self._adam_v: list[np.ndarray] = [np.zeros_like(t, dtype=np.float32) for t in self._theta_star]
        self._adam_t: int = 0
        self._adam_beta1 = adam_beta1
        self._adam_beta2 = adam_beta2
        self._adam_eps = adam_eps
        self._weight_decay = weight_decay

        # LR schedule
        self._total_rounds = total_rounds
        self._warmup_rounds = warmup_rounds

        # Robust aggregation
        self._aggregator = RobustAggregator()
        self._robust_method = robust_method
        self._trim_ratio = trim_ratio
        self._n_byzantine = n_byzantine
        self._norm_filter_multiplier = norm_filter_multiplier

        # Secure aggregation
        self._secure_agg = secure_agg
        self._secagg_mask_scale = secagg_mask_scale
        self._secagg_base_seed = secagg_base_seed

        self._tracker = Tracker(
            experiment_name=experiment_name,
            run_name="federated_run",
            tracking_uri=tracking_uri,
        )
        self._tracker.log_params({
            "outer_lr": outer_lr,
            "adam_beta1": adam_beta1,
            "adam_beta2": adam_beta2,
            "weight_decay": weight_decay,
            "warmup_rounds": warmup_rounds,
            "total_rounds": total_rounds,
            "min_available_clients": min_available_clients,
            "fraction_fit": fraction_fit,
            "checkpoint_every": checkpoint_every,
            "n_params": len(self._theta_star),
            "robust_method": robust_method,
            "secure_agg": int(secure_agg),
        })

    def _schedule_lr(self, global_round: int) -> float:
        """Cosine decay with linear warmup."""
        if global_round <= self._warmup_rounds:
            return self.base_outer_lr * global_round / max(self._warmup_rounds, 1)
        progress = (global_round - self._warmup_rounds) / max(
            self._total_rounds - self._warmup_rounds, 1
        )
        return self.base_outer_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        return ndarrays_to_parameters(self._theta_star)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        # Block until min_available_clients have connected, then sample fraction_fit of them.
        num_sample = max(1, int(self.fraction_fit * self.min_available_clients))
        clients = client_manager.sample(
            num_clients=num_sample,
            min_num_clients=self.min_available_clients,
        )
        base_config = {"round": server_round + self._round_offset}

        if not self._secure_agg:
            fit_ins = FitIns(parameters=parameters, config=base_config)
            return [(c, fit_ins) for c in clients]

        # SecAgg: each client gets a unique FitIns with its pairwise mask seeds.
        per_client_configs = SecureAggregator.make_round_configs(
            round_num=server_round,
            n_clients=len(clients),
            mask_scale=self._secagg_mask_scale,
            base_seed=self._secagg_base_seed,
        )
        return [
            (c, FitIns(parameters=parameters, config={**base_config, **per_client_configs[i]}))
            for i, c in enumerate(clients)
        ]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        import gc

        global_round = server_round + self._round_offset

        if not results:
            print(f"  [server] Round {global_round}: no results received")
            return None, {}

        if failures:
            print(f"  [server] Round {global_round}: {len(failures)} client failures")

        n_params = len(self._theta_star)

        # Accumulators for weighted-average client diagnostics
        metric_sums: dict[str, float] = {k: 0.0 for k in _CLIENT_METRIC_KEYS}
        metric_weights: dict[str, float] = {k: 0.0 for k in _CLIENT_METRIC_KEYS}

        all_grads: list[list[np.ndarray]] = []
        all_weights: list[float] = []

        for _client, fit_res in results:
            grads = parameters_to_ndarrays(fit_res.parameters)
            w = float(fit_res.num_examples)
            all_grads.append([g.astype(np.float32) for g in grads])
            all_weights.append(w)

            for k in _CLIENT_METRIC_KEYS:
                raw = fit_res.metrics.get(k)
                if raw is not None:
                    try:
                        fv = float(raw)
                        if not math.isnan(fv):
                            metric_sums[k] += fv * w
                            metric_weights[k] += w
                    except (TypeError, ValueError):
                        pass

            del grads, fit_res
            gc.collect()

        total_weight = sum(all_weights)

        # Byzantine-robust aggregation — returns normalized List[np.ndarray]
        avg_grads, audit = self._aggregator.aggregate(
            all_grads,
            all_weights,
            method=self._robust_method,
            trim_ratio=self._trim_ratio,
            n_byzantine=self._n_byzantine,
            norm_filter_multiplier=self._norm_filter_multiplier,
        )
        del all_grads
        gc.collect()

        # AdamW outer update with cosine LR schedule
        self.outer_lr = self._schedule_lr(global_round)
        self._adam_t += 1
        t = self._adam_t
        b1, b2, eps = self._adam_beta1, self._adam_beta2, self._adam_eps

        for i in range(n_params):
            g = avg_grads[i]
            # Bias-corrected Adam moments
            self._adam_m[i] = b1 * self._adam_m[i] + (1.0 - b1) * g
            self._adam_v[i] = b2 * self._adam_v[i] + (1.0 - b2) * g * g
            m_hat = self._adam_m[i] / (1.0 - b1 ** t)
            v_hat = self._adam_v[i] / (1.0 - b2 ** t)
            theta = self._theta_star[i].astype(np.float32)
            # AdamW: weight decay applied to parameter, not gradient
            self._theta_star[i] = (
                theta * (1.0 - self.outer_lr * self._weight_decay)
                - self.outer_lr * m_hat / (np.sqrt(v_hat) + eps)
            )

        del avg_grads
        gc.collect()

        avg_m: dict[str, float] = {
            k: metric_sums[k] / metric_weights[k] if metric_weights[k] > 0 else float("nan")
            for k in _CLIENT_METRIC_KEYS
        }

        print(
            f"  [server] Round {global_round}: aggregated {audit['n_clients_used']}/{len(results)} clients"
            f" [{self._robust_method}] | "
            f"β={self.outer_lr:.2e} | total_examples={total_weight} | "
            f"avg_query_loss={avg_m['query_loss']:.4f} | "
            f"avg_grad_norm={avg_m['grad_norm']:.4f} | "
            f"inner {avg_m['inner_loss_init']:.4f}→{avg_m['inner_loss_final']:.4f}"
        )

        self._tracker.log_metrics(
            {
                "server/n_clients": float(len(results)),
                "server/total_examples": float(total_weight),
                "server/outer_lr": self.outer_lr,
                "server/rob_n_used": float(audit["n_clients_used"]),
                "server/rob_n_dropped": float(audit["n_clients_dropped"]),
                "client/avg_query_loss": avg_m["query_loss"],
                "client/avg_grad_norm": avg_m["grad_norm"],
                "client/avg_clip_coef": avg_m["clip_coef"],
                "client/avg_inner_loss_init": avg_m["inner_loss_init"],
                "client/avg_inner_loss_final": avg_m["inner_loss_final"],
            },
            step=global_round,
        )

        if (
            self._checkpoint_dir is not None
            and self._param_names is not None
            and global_round % self._checkpoint_every == 0
        ):
            self._save_checkpoint(global_round)

        updated_params = ndarrays_to_parameters(self._theta_star)
        return updated_params, {"round": server_round}

    def _save_checkpoint(self, server_round: int) -> None:
        import torch
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self._checkpoint_dir / f"theta_star_lora_round_{server_round:04d}.pt"
        state_dict = {
            name: torch.tensor(arr.astype("float32"))
            for name, arr in zip(self._param_names, self._theta_star)
        }
        torch.save(state_dict, path)
        print(f"  [server] Checkpoint saved: {path.name}", flush=True)
        self._tracker.log_artifact(path)

    def configure_evaluate(self, server_round, parameters, client_manager):
        """Evaluation handled externally by eval_poc.py."""
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

    def evaluate(self, server_round, parameters):
        return None

    def close(self) -> None:
        """End the MLflow run. Call after fl.server.start_server() returns."""
        self._tracker.end()
