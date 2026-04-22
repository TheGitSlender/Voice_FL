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

import sys
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

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

class PerFedAvgStrategy(fl.server.strategy.Strategy):
    """
    Per-FedAvg: server maintains θ* and applies gradient descent each round.

    Args:
        initial_parameters: encoder weights as Flower Parameters
        outer_lr:           β — server-side learning rate (default 2e-4)
        min_available_clients: minimum clients before server starts
        fraction_fit:       fraction of available clients to use per round
    """

    def __init__(
        self,
        initial_parameters: Parameters,
        outer_lr: float = 2e-4,
        min_available_clients: int = 5,
        fraction_fit: float = 1.0,
    ):
        super().__init__()
        self.outer_lr = outer_lr
        self.fraction_fit = fraction_fit
        self.min_available_clients = min_available_clients

        self._theta_star: list[np.ndarray] = parameters_to_ndarrays(initial_parameters)

    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        return ndarrays_to_parameters(self._theta_star)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[ClientProxy, FitIns]]:
                                                                                  
        clients = client_manager.sample(
            num_clients=self.min_available_clients,
            min_num_clients=self.min_available_clients,
        )
        fit_ins = FitIns(parameters=parameters, config={"round": server_round})
        return [(c, fit_ins) for c in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        import gc

        if not results:
            print(f"  [server] Round {server_round}: no results received")
            return None, {}

        if failures:
            print(f"  [server] Round {server_round}: {len(failures)} client failures")

        n_params = len(self._theta_star)
        acc_grads = [np.zeros(t.shape, dtype=np.float32) for t in self._theta_star]
        total_weight = 0

        for _client, fit_res in results:
            grads = parameters_to_ndarrays(fit_res.parameters)
            w = fit_res.num_examples
            total_weight += w
            for i, g in enumerate(grads):
                                                                              
                acc_grads[i] += w * g.astype(np.float32)
                                                           
            del grads, fit_res
            gc.collect()

        for i in range(n_params):
            avg = acc_grads[i].astype(np.float32) / total_weight
            self._theta_star[i] = (
                self._theta_star[i].astype(np.float32) - self.outer_lr * avg
            ).astype(np.float16)

        del acc_grads
        gc.collect()

        print(
            f"  [server] Round {server_round}: aggregated {len(results)} clients | "
            f"β={self.outer_lr} | total_examples={total_weight}"
        )

        updated_params = ndarrays_to_parameters(self._theta_star)
        return updated_params, {"round": server_round}

    def configure_evaluate(self, server_round, parameters, client_manager):
        """Evaluation handled externally by eval_poc.py."""
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

    def evaluate(self, server_round, parameters):
        return None
