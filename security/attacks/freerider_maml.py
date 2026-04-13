"""
security/attacks/freerider_maml.py — Free-rider gradient attack simulator

A free-rider contributes nothing: sends all-zero gradients.
Benefits from the global model without paying the communication or compute cost.
"""

from typing import List

import numpy as np


def freerider_gradient(param_shapes: List[tuple]) -> List[np.ndarray]:
    """
    Generate a zero (free-rider) gradient.

    Args:
      param_shapes: list of parameter shapes from model.get_outer_loop_params()

    Returns:
      list of zero numpy arrays with same shapes as outer-loop params
    """
    return [np.zeros(s, dtype=np.float32) for s in param_shapes]
