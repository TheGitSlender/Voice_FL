"""Pairwise-masking secure aggregation for gradient privacy.

Protocol (Bonawitz et al. 2017, simplified):
  For each pair (i, j) with i < j in the round's sampled cohort:
    - Derive a shared mask r_ij from pair_seed = round_seed XOR pair_index.
    - Client i adds  r_ij to its gradient.
    - Client j subtracts r_ij from its gradient.
  Server aggregates normally; masks cancel: Σ masked_grads == Σ plain_grads.

Security model: Honest-but-curious server.

Limitation of this prototype: the round_seed is chosen by the server and
distributed to clients via FitIns.config.  A server that retains all round
seeds can recompute every mask and recover individual gradients.

TODO (production upgrade): Replace the server-distributed round_seed with
Diffie-Hellman key agreement between client pairs so the server never learns
the pair seeds.  Flower 2.x ships a SecAgg plugin that implements this full
protocol.  The client-side apply_masks() and the aggregation math below are
unchanged — only the seed-distribution mechanism needs to change.

Usage:
    # Server side — in configure_fit()
    configs = SecureAggregator.make_round_configs(round_num, n_clients)
    # configs[i] is injected into FitIns.config for client i

    # Client side — in fit() after computing grad_arrays
    if int(ins.config.get("secagg_enabled", 0)):
        grad_arrays = SecureAggregator.apply_masks(grad_arrays, **secagg_kwargs)
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


class SecureAggregator:
    """Stateless pairwise-mask secure aggregation helpers."""

    @staticmethod
    def make_round_configs(
        round_num: int,
        n_clients: int,
        mask_scale: float = 0.01,
        base_seed: int = 0,
    ) -> List[Dict]:
        """Return per-client config dicts for injection into FitIns.config.

        Each dict contains the five secagg_* scalar keys that client_lora.fit()
        reads.  The round_seed is mixed from base_seed and round_num so that
        different rounds produce different masks.
        """
        # Knuth multiplicative hash to spread round_num across seed space
        round_seed = int((base_seed ^ (round_num * 2654435761)) & 0xFFFFFFFF)
        return [
            {
                "secagg_enabled": 1,
                "secagg_round_num": round_num,
                "secagg_round_seed": round_seed,
                "secagg_client_idx": idx,
                "secagg_cohort_size": n_clients,
                "secagg_mask_scale": mask_scale,
            }
            for idx in range(n_clients)
        ]

    @staticmethod
    def apply_masks(
        grad_arrays: List[np.ndarray],
        round_num: int,
        client_idx: int,
        cohort_size: int,
        round_seed: int,
        mask_scale: float,
    ) -> List[np.ndarray]:
        """Return new gradient arrays with pairwise cancelling masks applied.

        For every partner j != client_idx:
          - If client_idx < j: add the pair mask (client_idx is the "lower" peer).
          - If client_idx > j: subtract the pair mask.
        Both sides use the same pair_seed, so the masks cancel on summation.

        Does NOT mutate the input arrays.
        """
        masked = [arr.copy() for arr in grad_arrays]
        for j in range(cohort_size):
            if j == client_idx:
                continue
            low = min(client_idx, j)
            high = max(client_idx, j)
            pair_idx = low * cohort_size + high
            pair_seed = int((round_seed ^ pair_idx) & 0xFFFFFFFF)
            rng = np.random.default_rng(pair_seed)
            sign = 1 if client_idx < j else -1
            for i, arr in enumerate(grad_arrays):
                mask = rng.standard_normal(arr.shape).astype(np.float32) * mask_scale
                masked[i] = masked[i] + sign * mask
        return masked

    @staticmethod
    def verify_cancellation(
        masked_per_client: List[List[np.ndarray]],
        plain_per_client: List[List[np.ndarray]],
        atol: float = 1e-5,
    ) -> bool:
        """Test utility: assert element-wise Σ masked == Σ plain.

        Uses float64 accumulation to avoid float32 cancellation error.
        """
        n_params = len(plain_per_client[0])
        for p in range(n_params):
            plain_sum = sum(c[p].astype(np.float64) for c in plain_per_client)
            masked_sum = sum(c[p].astype(np.float64) for c in masked_per_client)
            if not np.allclose(plain_sum, masked_sum, atol=atol):
                return False
        return True
