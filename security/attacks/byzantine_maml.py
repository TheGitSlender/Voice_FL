"""
security/attacks/byzantine_maml.py — Byzantine gradient attack simulator

Produces a random gradient with the same shapes as the model's outer params.
A Byzantine node sends a gradient uncorrelated with any honest update.
"""

from typing import List

import numpy as np


def byzantine_gradient(
    param_shapes: List[tuple],
    scale: float = 1.0,
    seed: int = None,
) -> List[np.ndarray]:
    """
    Generate a random Byzantine gradient.

    Args:
      param_shapes: list of parameter shapes from model.get_outer_loop_params()
      scale: magnitude of the random gradient
      seed: optional random seed for reproducibility

    Returns:
      list of numpy arrays with same shapes as outer-loop params
    """
    rng = np.random.RandomState(seed)
    return [rng.randn(*s).astype(np.float32) * scale for s in param_shapes]
