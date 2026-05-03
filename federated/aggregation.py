"""Byzantine-robust gradient aggregation for PerFedAvg.

All methods share the same interface:
  input:  List[List[np.ndarray]]  — one gradient list per client
  output: (List[np.ndarray], dict) — aggregated gradient + audit info

The audit dict is logged to MLflow as server/rob_* metrics.
The AdamW update in strategy_maml.py is unchanged — it sees the same
normalized gradient vector regardless of which method produced it.

Scalability: every method is correct for arbitrary N. The current
cohort of 3 uses 'mean' (the default). Switch the config key to enable
robustness without any code change.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


_VALID_METHODS = frozenset(
    ("mean", "trimmed_mean", "coordinate_median", "krum", "norm_filter")
)


class RobustAggregator:
    """Pluggable Byzantine-robust aggregation strategies."""

    def aggregate(
        self,
        grads: List[List[np.ndarray]],
        weights: List[float],
        method: str = "mean",
        trim_ratio: float = 0.1,
        n_byzantine: int = 1,
        norm_filter_multiplier: float = 2.0,
    ) -> Tuple[List[np.ndarray], Dict]:
        """Aggregate per-client gradient lists into one gradient list.

        Args:
            grads:                  One list of parameter arrays per client.
            weights:                num_examples per client (used for weighted mean).
            method:                 Aggregation strategy.
            trim_ratio:             For trimmed_mean — fraction to drop from each tail.
            n_byzantine:            For krum — assumed max Byzantine clients.
            norm_filter_multiplier: For norm_filter — threshold = median_norm × this.

        Returns:
            (avg_grads, audit) where avg_grads has the same shape as grads[0]
            and audit is a dict with n_clients_in / n_clients_used / n_clients_dropped.
        """
        if method not in _VALID_METHODS:
            raise ValueError(
                f"Unknown aggregation method {method!r}. "
                f"Choose from {sorted(_VALID_METHODS)}."
            )

        n = len(grads)
        if n == 0:
            raise ValueError("No client gradients to aggregate.")

        audit: Dict = {
            "method": method,
            "n_clients_in": n,
            "n_clients_used": n,
            "n_clients_dropped": 0,
        }

        if n == 1:
            return [g.copy() for g in grads[0]], audit

        if method == "mean":
            return self._weighted_mean(grads, weights), audit

        if method == "trimmed_mean":
            result, used = self._trimmed_mean(grads, trim_ratio)
            audit["n_clients_used"] = used
            audit["n_clients_dropped"] = n - used
            return result, audit

        if method == "coordinate_median":
            return self._coordinate_median(grads), audit

        if method == "krum":
            result, selected = self._krum(grads, n_byzantine)
            audit["n_clients_used"] = 1
            audit["n_clients_dropped"] = n - 1
            audit["krum_selected_idx"] = selected
            return result, audit

        # norm_filter
        result, used = self._norm_filter(grads, weights, norm_filter_multiplier)
        audit["n_clients_used"] = used
        audit["n_clients_dropped"] = n - used
        return result, audit

    # ------------------------------------------------------------------
    # Internal strategies
    # ------------------------------------------------------------------

    @staticmethod
    def _weighted_mean(
        grads: List[List[np.ndarray]], weights: List[float]
    ) -> List[np.ndarray]:
        total = sum(weights)
        result = [np.zeros_like(grads[0][i], dtype=np.float32) for i in range(len(grads[0]))]
        for client_grads, w in zip(grads, weights):
            share = w / total
            for i, g in enumerate(client_grads):
                result[i] += share * g.astype(np.float32)
        return result

    @staticmethod
    def _trimmed_mean(
        grads: List[List[np.ndarray]], trim_ratio: float
    ) -> Tuple[List[np.ndarray], int]:
        n = len(grads)
        k = int(np.floor(trim_ratio * n))
        # Never trim so aggressively that nothing survives
        if 2 * k >= n:
            k = max(0, (n - 1) // 2)
        n_used = n - 2 * k

        result = []
        for p in range(len(grads[0])):
            stacked = np.stack(
                [g[p].astype(np.float32) for g in grads], axis=0
            )  # (n, *param_shape)
            if k > 0:
                # Sort along client axis, slice the middle n_used rows
                sorted_stacked = np.sort(stacked, axis=0)
                trimmed = sorted_stacked[k : n - k]
            else:
                trimmed = stacked
            result.append(trimmed.mean(axis=0))
        return result, n_used

    @staticmethod
    def _coordinate_median(grads: List[List[np.ndarray]]) -> List[np.ndarray]:
        result = []
        for p in range(len(grads[0])):
            stacked = np.stack(
                [g[p].astype(np.float32) for g in grads], axis=0
            )
            result.append(np.median(stacked, axis=0).astype(np.float32))
        return result

    @staticmethod
    def _krum(
        grads: List[List[np.ndarray]], n_byzantine: int
    ) -> Tuple[List[np.ndarray], int]:
        n = len(grads)
        # Each client selects the n_neighbors closest others and sums their distances
        n_neighbors = max(1, n - n_byzantine - 1)

        # Flatten each client's gradient for distance computation
        flat = [
            np.concatenate([g.flatten().astype(np.float32) for g in client_grads])
            for client_grads in grads
        ]

        scores = np.zeros(n)
        for i in range(n):
            dists = sorted(
                float(np.dot(flat[i] - flat[j], flat[i] - flat[j]))
                for j in range(n)
                if j != i
            )
            scores[i] = sum(dists[:n_neighbors])

        selected = int(np.argmin(scores))
        return [g.copy().astype(np.float32) for g in grads[selected]], selected

    @staticmethod
    def _norm_filter(
        grads: List[List[np.ndarray]],
        weights: List[float],
        multiplier: float,
    ) -> Tuple[List[np.ndarray], int]:
        norms = [
            float(
                np.linalg.norm(
                    np.concatenate([g.flatten().astype(np.float32) for g in client_grads])
                )
            )
            for client_grads in grads
        ]
        threshold = float(np.median(norms)) * multiplier
        keep = [i for i, norm in enumerate(norms) if norm <= threshold] or list(range(len(grads)))

        kept_grads = [grads[i] for i in keep]
        kept_weights = [weights[i] for i in keep]
        return RobustAggregator._weighted_mean(kept_grads, kept_weights), len(keep)
