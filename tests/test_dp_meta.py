"""
tests/test_dp_meta.py — DP meta-gradient unit tests

Verifies:
  1. Clipped gradient L2 norm ≤ C (always, 1000 trials)
  2. Noised gradient ≠ clipped gradient
  3. RDP accountant epsilon strictly increases each step
  4. is_exhausted() triggers at correct epsilon
  5. sigma > 0 from compute_sigma()

Run:
    pytest tests/test_dp_meta.py -v
"""

import pytest
import torch
import numpy as np

from privacy.dp_meta import DPConfig, apply_dp_to_meta_gradient, compute_sigma
from privacy.rdp_accountant import RDPAccountant


class TestApplyDPToMetaGradient:
    """Tests for the clip + Gaussian noise DP mechanism."""

    def _make_grads(self, n: int = 10, dim: int = 100):
        """Create random gradient tensors."""
        return [torch.randn(dim) for _ in range(n)]

    def test_clipped_norm_leq_C_1000_trials(self) -> None:
        """Critical: clipped gradient norm must NEVER exceed C."""
        C = 1.0
        sigma = 0.0  # no noise — isolate clipping behavior

        for _ in range(1000):
            grads = self._make_grads(n=5, dim=50)
            sanitized, norm = apply_dp_to_meta_gradient(grads, C, sigma)

            valid = [g for g in sanitized if g is not None]
            flat = torch.cat([g.flatten() for g in valid])
            clipped_norm = float(flat.norm(2).item())

            assert clipped_norm <= C + 1e-5, (
                f"Clipped norm {clipped_norm:.6f} > C={C} — clipping invariant violated"
            )

    def test_clipped_norm_leq_C_large_gradients(self) -> None:
        """Large gradients must be clipped down to C."""
        C = 1.0
        # Create a gradient with norm >> C
        grads = [torch.ones(1000) * 100.0]  # norm = 100*sqrt(1000) >> 1
        sanitized, norm = apply_dp_to_meta_gradient(grads, C, sigma=0.0)

        flat = torch.cat([g.flatten() for g in sanitized if g is not None])
        clipped_norm = float(flat.norm(2).item())
        assert clipped_norm <= C + 1e-5

    def test_noised_differs_from_clipped(self) -> None:
        """Adding noise must change the gradient."""
        C = 1.0
        sigma = 1.0
        grads = [torch.randn(100)]

        clipped, _ = apply_dp_to_meta_gradient(grads, C, sigma=0.0)
        noised, _ = apply_dp_to_meta_gradient(grads, C, sigma=sigma)

        assert not torch.allclose(clipped[0], noised[0]), (
            "Noised gradient is identical to clipped — noise not applied"
        )

    def test_none_grads_preserved(self) -> None:
        """None entries (allow_unused params) must pass through unchanged."""
        grads = [torch.randn(10), None, torch.randn(10), None]
        sanitized, _ = apply_dp_to_meta_gradient(grads, C=1.0, sigma=0.5)

        assert sanitized[1] is None
        assert sanitized[3] is None

    def test_returns_norm(self) -> None:
        """apply_dp must return the pre-clip norm for audit logging."""
        grads = [torch.randn(100)]
        _, norm = apply_dp_to_meta_gradient(grads, C=1.0, sigma=0.0)
        assert isinstance(norm, float)
        assert norm >= 0.0

    def test_empty_grads_returns_zero_norm(self) -> None:
        """All-None gradient list should return 0.0 norm."""
        grads = [None, None, None]
        sanitized, norm = apply_dp_to_meta_gradient(grads, C=1.0, sigma=1.0)
        assert norm == 0.0
        assert all(g is None for g in sanitized)

    def test_sigma_zero_no_noise(self) -> None:
        """sigma=0 means no noise — clipped gradient unchanged."""
        C = 0.5
        grads = [torch.randn(50)]
        sanitized, _ = apply_dp_to_meta_gradient(grads, C, sigma=0.0)

        flat = torch.cat([g.flatten() for g in sanitized if g is not None])
        norm = float(flat.norm(2).item())
        assert norm <= C + 1e-5


class TestComputeSigma:
    def test_sigma_positive(self) -> None:
        sigma = compute_sigma(epsilon=8.0, delta=1e-5, C=1.0)
        assert sigma > 0.0, f"sigma should be positive, got {sigma}"

    def test_larger_epsilon_smaller_sigma(self) -> None:
        """Higher epsilon (less privacy) → lower noise multiplier."""
        sigma_tight = compute_sigma(epsilon=2.0, delta=1e-5, C=1.0)
        sigma_loose = compute_sigma(epsilon=8.0, delta=1e-5, C=1.0)
        assert sigma_tight > sigma_loose, (
            f"Expected sigma(ε=2) > sigma(ε=8), got {sigma_tight:.4f} vs {sigma_loose:.4f}"
        )


class TestRDPAccountant:
    def test_epsilon_increases_per_step(self) -> None:
        acct = RDPAccountant(target_epsilon=8.0, target_delta=1e-5)
        epsilons = [acct.get_epsilon()]
        for _ in range(5):
            acct.step(noise_multiplier=1.0, sample_rate=0.1)
            epsilons.append(acct.get_epsilon())

        for i in range(1, len(epsilons)):
            assert epsilons[i] >= epsilons[i - 1], (
                f"Epsilon decreased at step {i}: {epsilons[i-1]:.4f} → {epsilons[i]:.4f}"
            )

    def test_epsilon_strictly_increases(self) -> None:
        acct = RDPAccountant(target_epsilon=8.0, target_delta=1e-5)
        acct.step(noise_multiplier=1.0, sample_rate=0.1)
        eps1 = acct.get_epsilon()
        acct.step(noise_multiplier=1.0, sample_rate=0.1)
        eps2 = acct.get_epsilon()
        assert eps2 > eps1, f"Epsilon did not increase: {eps1:.4f} → {eps2:.4f}"

    def test_is_exhausted_triggers_at_target(self) -> None:
        """is_exhausted() must return True when accumulated ε >= target."""
        target = 1.0
        acct = RDPAccountant(target_epsilon=target, target_delta=1e-5)

        # Run until exhausted
        max_steps = 10000
        for _ in range(max_steps):
            if acct.is_exhausted():
                break
            acct.step(noise_multiplier=0.5, sample_rate=0.5)  # aggressive — exhausts fast
        else:
            pytest.fail("RDP accountant never reached target epsilon")

        assert acct.get_epsilon() >= target
        assert acct.is_exhausted()

    def test_remaining_decreases(self) -> None:
        acct = RDPAccountant(target_epsilon=8.0, target_delta=1e-5)
        remaining_before = acct.remaining()
        acct.step(noise_multiplier=1.0, sample_rate=0.1)
        remaining_after = acct.remaining()
        assert remaining_after <= remaining_before

    def test_zero_steps_zero_epsilon(self) -> None:
        acct = RDPAccountant(target_epsilon=8.0, target_delta=1e-5)
        assert acct.get_epsilon() == 0.0

    def test_summary_string(self) -> None:
        acct = RDPAccountant(target_epsilon=8.0, target_delta=1e-5)
        acct.step(noise_multiplier=1.0, sample_rate=0.1)
        summary = acct.summary()
        assert "Steps" in summary
        assert "ε spent" in summary
