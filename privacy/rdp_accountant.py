"""
privacy/rdp_accountant.py — Rényi DP accountant

Tracks cumulative privacy budget across FL rounds.
Halts training when target epsilon is exhausted.

Uses autodp's analytical RDP accountant (anaRDPacct) when available.
Falls back to a linear per-step estimate (conservative) when autodp is absent.

Each FL round with DP-SGD noise contributes to the total epsilon via
RDP composition. The accountant converts the accumulated RDP to (ε, δ)-DP
for reporting.
"""

import math


class RDPAccountant:
    """
    Rényi DP accountant for tracking cumulative privacy budget.

    Call step() once per FL round (once per client fit() call).
    Call get_epsilon() to check current spend.
    Call is_exhausted() before each round to enforce the budget cap.
    """

    def __init__(self, target_epsilon: float, target_delta: float) -> None:
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.steps = 0
        self._history: list = []  # list of (noise_multiplier, sample_rate)

        try:
            from autodp import rdp_acct, rdp_bank

            self.acct = rdp_acct.anaRDPacct()
            self._rdp_bank = rdp_bank
            self._use_autodp = True
        except (ImportError, Exception):
            self._use_autodp = False

    def step(self, noise_multiplier: float, sample_rate: float) -> None:
        """Register one FL round of DP-SGD noise application."""
        self._history.append((noise_multiplier, sample_rate))
        self.steps += 1

        if self._use_autodp:
            # compose_subsampled_mechanism expects func(alpha) → RDP epsilon
            # RDP_gaussian(params, alpha) — wrap as a single-arg callable
            gaussian_func = lambda alpha: self._rdp_bank.RDP_gaussian(
                {"sigma": noise_multiplier}, alpha
            )
            self.acct.compose_subsampled_mechanism(gaussian_func, sample_rate)

    def get_epsilon(self) -> float:
        """Current epsilon spend under (ε, δ)-DP."""
        if not self._history:
            return 0.0
        if self._use_autodp:
            return float(self.acct.get_eps(self.target_delta))
        # Fallback: linear estimate per step (conservative)
        sigma, _ = self._history[-1]
        eps_per_step = math.sqrt(2 * math.log(1.25 / self.target_delta)) / sigma
        return eps_per_step * self.steps

    def is_exhausted(self) -> bool:
        """True when the target epsilon budget has been spent."""
        return self.get_epsilon() >= self.target_epsilon

    def remaining(self) -> float:
        """Remaining epsilon budget."""
        return max(0.0, self.target_epsilon - self.get_epsilon())

    def summary(self) -> str:
        """Human-readable privacy budget summary."""
        return (
            f"Steps: {self.steps} | "
            f"ε spent: {self.get_epsilon():.4f} / {self.target_epsilon} | "
            f"Remaining: {self.remaining():.4f} | "
            f"δ: {self.target_delta}"
        )
