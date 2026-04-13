"""
federated/strategy_maml.py — Per-FedAvg aggregation strategy

Implements the Per-FedAvg server-side aggregation rule:
  θ* ← θ* − β · weighted_avg(∇L_meta_clients)

Standard FedAvg:  θ_new = weighted_avg(θ_clients)     [weight averaging]
Per-FedAvg:       θ* ← θ* − β · avg(meta_grads)       [gradient descent on meta-objective]

Clients return meta-gradients shaped like the encoder parameters.
This strategy treats them as gradients and applies outer lr β — NOT as weights.

BAE screening (optional): meta-gradients are screened before averaging.
Anomalous gradients (Byzantine, poisoning, free-rider) get zero weight.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import flwr as fl
from flwr.common import (
    Parameters,
    FitRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.strategy import FedAvg


class PerFedAvgStrategy(FedAvg):
    """
    Per-FedAvg: aggregate meta-gradients and apply outer learning rate β.

    The server maintains current_params (encoder weights θ*) and updates
    them each round via gradient descent on the averaged meta-gradient.

    current_params is initialized from the first client's get_parameters()
    call (Flower's default behavior when initialize_parameters returns None).
    """

    def __init__(
        self,
        outer_lr: float = 2e-4,
        bae=None,
        mlflow_run=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.outer_lr = outer_lr
        self.bae = bae
        self.mlflow_run = mlflow_run
        self.current_params: Optional[List[np.ndarray]] = None

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple],
        failures,
    ) -> Tuple[Optional[Parameters], Dict]:
        """
        Aggregate meta-gradients and update θ*.

        1. Extract meta-gradients and sample counts from results
        2. BAE screening (if enabled) — adjusts per-node weights
        3. Weighted average of verified meta-gradients
        4. Apply: θ* ← θ* − β · avg_gradient
        5. Update self.current_params and return as Parameters
        """
        if not results:
            return None, {}

        gradients: List[List[np.ndarray]] = []
        weights: List[int] = []
        metrics_per_node: Dict = {}

        for client_proxy, fit_res in results:
            grads = parameters_to_ndarrays(fit_res.parameters)
            gradients.append(grads)
            weights.append(fit_res.num_examples)
            metrics_per_node[client_proxy.cid] = fit_res.metrics

        # BAE screening: adjusts weights for anomalous nodes
        if self.bae is not None:
            grad_dict = {
                results[i][0].cid: gradients[i]
                for i in range(len(results))
            }
            node_weights = self.bae.screen_updates(grad_dict, server_round)
            adjusted_weights = [
                weights[i] * node_weights.get(results[i][0].cid, 1.0)
                for i in range(len(results))
            ]
        else:
            adjusted_weights = weights

        total_weight = sum(adjusted_weights)
        if total_weight == 0:
            if self.current_params is not None:
                return ndarrays_to_parameters(self.current_params), {}
            return None, {}

        # Weighted average of meta-gradients
        avg_gradient = [
            sum(w * g[i] for w, g in zip(adjusted_weights, gradients)) / total_weight
            for i in range(len(gradients[0]))
        ]

        if self.current_params is None:
            raise RuntimeError(
                "current_params not initialized. Flower should have called "
                "get_parameters on a client before aggregate_fit."
            )

        # Per-FedAvg update: θ* ← θ* − β · avg_gradient
        updated_params = [
            p - self.outer_lr * g
            for p, g in zip(self.current_params, avg_gradient)
        ]
        self.current_params = updated_params

        # MLflow logging
        if self.mlflow_run is not None:
            import mlflow

            active_nodes = len([w for w in adjusted_weights if w > 0])
            losses = [m.get("query_loss", 0) for m in metrics_per_node.values()]
            epsilons = [m.get("epsilon", 0) for m in metrics_per_node.values() if "epsilon" in m]

            metrics_to_log = {
                "train/query_loss": float(np.mean(losses)),
                "train/active_nodes": active_nodes,
            }
            if epsilons:
                metrics_to_log["train/epsilon_avg"] = float(np.mean(epsilons))
            mlflow.log_metrics(metrics_to_log, step=server_round)

        return ndarrays_to_parameters(updated_params), {
            "avg_query_loss": float(np.mean(
                [m.get("query_loss", 0) for m in metrics_per_node.values()]
            )),
            "active_nodes": len([w for w in adjusted_weights if w > 0]),
        }

    def aggregate_evaluate(self, server_round: int, results, failures):
        """Aggregate per-node WER results."""
        if not results:
            return None, {}

        wers = [fit_res.metrics.get("wer", 1.0) for _, fit_res in results]
        gains = [fit_res.metrics.get("adaptation_gain", 0.0) for _, fit_res in results]

        aggregated = {
            "eval/mean_wer": float(np.mean(wers)),
            "eval/mean_adaptation_gain": float(np.mean(gains)),
            "eval/nodes_with_positive_gain": int(sum(1 for g in gains if g > 0)),
        }

        if self.mlflow_run is not None:
            import mlflow
            mlflow.log_metrics(aggregated, step=server_round)

        return float(np.mean(wers)), aggregated

    def initialize_parameters(self, client_manager):
        """
        Return None so Flower requests initial parameters from a client.
        The client's get_parameters() will return encoder weights,
        which become self.current_params on the first aggregate_fit call.
        """
        return None

    def on_fit_config_fn(self, server_round: int) -> Dict:
        return {"server_round": server_round}

    def on_evaluate_config_fn(self, server_round: int) -> Dict:
        return {"server_round": server_round}
