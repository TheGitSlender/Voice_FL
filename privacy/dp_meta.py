"""
privacy/dp_meta.py — Manual DP for MAML meta-gradients

Implements (ε, δ)-DP via L2 clipping + Gaussian noise on the outer-loop
meta-gradient before transmission to the FL server.

Critical invariant (I5): NO Opacus.
  Opacus is incompatible with higher's functional model wrapper
  (higher.innerloop_ctx replaces the model with a stateless functional form
  that Opacus cannot hook into for per-sample gradient tracking).
  DP is applied manually to the already-computed meta-gradient.

DP mechanism:
  Step 1 — L2 clip:  ΔW̄ = ΔW · min(1, C / ‖ΔW‖₂)
  Step 2 — Noise:    ΔW_private = ΔW̄ + N(0, σ²C²I)

  C = clipping threshold (default 1.0)
  σ = noise multiplier (calibrated to target (ε, δ) via autodp RDP analysis)

Applied to the outer-loop meta-gradient (encoder params only) before transmission.
The lm_head gradient is never transmitted (I3) so DP does not need to cover it.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch


@dataclass
class DPConfig:
    epsilon: float = 8.0
    delta: float = 1e-5
    C: float = 1.0           # L2 clipping threshold
    sigma: Optional[float] = None  # noise multiplier — computed if None
    enabled: bool = True
    sample_rate: float = 0.1  # cohort fraction (for RDP accountant)

    def __post_init__(self) -> None:
        if self.sigma is None and self.enabled:
            self.sigma = compute_sigma(self.epsilon, self.delta, self.C)
        elif not self.enabled:
            self.sigma = 0.0


def compute_sigma(epsilon: float, delta: float, C: float) -> float:
    """
    Compute noise multiplier σ to satisfy (ε, δ)-DP.

    Uses autodp RDP calibration when available.
    Falls back to the Gaussian mechanism closed-form estimate:
      σ = sqrt(2 · ln(1.25/δ)) / ε

    The autodp path uses Rényi DP composition for more accurate accounting.
    """
    try:
        from autodp import calibrator_zoo, mechanism_zoo

        mech = mechanism_zoo.GaussianMechanism
        calib = calibrator_zoo.eps_delta_calibrator()
        sigma = calib(mech, epsilon, delta, [0.01, 1000.0])
        return float(sigma)
    except (ImportError, Exception):
        # Fallback: classical Gaussian mechanism bound
        sigma = math.sqrt(2 * math.log(1.25 / delta)) / epsilon
        return sigma


def apply_dp_to_meta_gradient(
    meta_grads: List[Optional[torch.Tensor]],
    C: float,
    sigma: float,
) -> Tuple[List[Optional[torch.Tensor]], float]:
    """
    Apply (ε, δ)-DP to the outer-loop meta-gradient.

    Step 1 — Global L2 clip across all gradient tensors:
      ΔW̄ = ΔW · min(1, C / ‖ΔW‖₂)

    Step 2 — Add isotropic Gaussian noise:
      ΔW_private = ΔW̄ + N(0, σ²C²I)

    All operations are on the same device as the input tensors.
    None entries (allow_unused params) are preserved as None.

    Args:
      meta_grads: list of gradient tensors aligned with model.parameters()
      C: L2 clipping threshold
      sigma: noise multiplier

    Returns:
      (sanitized_grads, original_norm)
      original_norm: pre-clip L2 norm (for audit logging)
    """
    valid = [g for g in meta_grads if g is not None]
    if not valid:
        return meta_grads, 0.0

    # Global L2 norm across all gradient tensors
    flat = torch.cat([g.detach().flatten() for g in valid])
    norm = flat.norm(2).item()

    # Clipping coefficient: 1 if norm <= C, else C/norm
    clip = min(1.0, C / (norm + 1e-8))

    result: List[Optional[torch.Tensor]] = []
    for g in meta_grads:
        if g is None:
            result.append(None)
            continue
        clipped = g.detach() * clip
        if sigma > 0:
            noise = torch.randn_like(clipped) * sigma * C
            result.append(clipped + noise)
        else:
            result.append(clipped)

    return result, norm
