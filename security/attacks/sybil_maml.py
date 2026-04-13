"""
security/attacks/sybil_maml.py — Sybil attack simulator

A Sybil attacker controls multiple fake identities. Each fake node sends
a near-copy of the real gradient with small perturbations, amplifying
the malicious gradient's influence in the weighted average.
"""

from typing import List

import numpy as np


def sybil_gradients(
    real_gradient: List[np.ndarray],
    n_fake: int = 3,
    noise: float = 0.01,
    seed: int = 0,
) -> List[List[np.ndarray]]:
    """
    Generate n_fake near-copies of the real gradient.

    Each fake node submits a slightly perturbed version of the real gradient
    to avoid triggering norm-based anomaly detection while amplifying
    the real gradient's effect by a factor of n_fake.

    Args:
      real_gradient: the original gradient to amplify
      n_fake: number of Sybil identities to generate
      noise: standard deviation of perturbation (small = stealthier)
      seed: base random seed

    Returns:
      list of n_fake gradient lists, each shaped like real_gradient
    """
    result = []
    for i in range(n_fake):
        rng = np.random.RandomState(seed + i)
        fake = [
            g + rng.randn(*g.shape).astype(np.float32) * noise
            for g in real_gradient
        ]
        result.append(fake)
    return result
