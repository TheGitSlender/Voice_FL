"""
security/attacks/poisoning_maml.py — Gradient poisoning attack simulator

Takes a clean meta-gradient and adds scaled noise to corrupt it.
A poisoning attack is harder to detect than Byzantine because the
gradient is partially correlated with the honest update.
"""

from typing import List

import numpy as np


def poisoned_gradient(
    clean_gradient: List[np.ndarray],
    poison_scale: float = 5.0,
    seed: int = 42,
) -> List[np.ndarray]:
    """
    Add scaled Gaussian noise to a clean meta-gradient.

    Args:
      clean_gradient: the honest meta-gradient (list of numpy arrays)
      poison_scale: noise magnitude relative to gradient scale
      seed: random seed

    Returns:
      poisoned gradient with same shapes as clean_gradient
    """
    rng = np.random.RandomState(seed)
    return [
        g + rng.randn(*g.shape).astype(np.float32) * poison_scale
        for g in clean_gradient
    ]
