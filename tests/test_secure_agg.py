"""Tests for pairwise-mask secure aggregation."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from federated.secure_agg import SecureAggregator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_plain_grads(n_clients: int, shapes: list, seed: int = 0):
    rng = np.random.default_rng(seed)
    return [
        [rng.standard_normal(s).astype(np.float32) for s in shapes]
        for _ in range(n_clients)
    ]


def _mask_all_clients(plain_grads, round_num: int, mask_scale: float = 0.01, base_seed: int = 0):
    n = len(plain_grads)
    configs = SecureAggregator.make_round_configs(round_num, n, mask_scale, base_seed)
    masked = []
    for i, (grads, cfg) in enumerate(zip(plain_grads, configs)):
        m = SecureAggregator.apply_masks(
            grads,
            round_num=cfg["secagg_round_num"],
            client_idx=cfg["secagg_client_idx"],
            cohort_size=cfg["secagg_cohort_size"],
            round_seed=cfg["secagg_round_seed"],
            mask_scale=cfg["secagg_mask_scale"],
        )
        masked.append(m)
    return masked


# ---------------------------------------------------------------------------
# Core invariant: masks cancel on summation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_clients", [2, 3, 12])
def test_masks_cancel(n_clients):
    """Σ masked_grads == Σ plain_grads for n=2, 3, and 12 clients."""
    shapes = [(8,), (4, 4), (16,)]
    plain = _make_plain_grads(n_clients, shapes, seed=n_clients)
    masked = _mask_all_clients(plain, round_num=1, mask_scale=0.05)
    assert SecureAggregator.verify_cancellation(masked, plain), \
        f"Mask cancellation failed for n_clients={n_clients}"


def test_masks_cancel_n3_explicit():
    """3-client cancellation with explicit sum check (current cohort size)."""
    shapes = [(10,)]
    plain = _make_plain_grads(3, shapes, seed=42)
    masked = _mask_all_clients(plain, round_num=5)

    plain_sum = sum(c[0].astype(np.float64) for c in plain)
    masked_sum = sum(c[0].astype(np.float64) for c in masked)
    np.testing.assert_allclose(plain_sum, masked_sum, atol=1e-5)


# ---------------------------------------------------------------------------
# Edge case: scale=0 → masks are zero → masked == plain
# ---------------------------------------------------------------------------

def test_mask_scale_zero():
    shapes = [(6,), (3, 2)]
    plain = _make_plain_grads(4, shapes, seed=1)
    masked = _mask_all_clients(plain, round_num=1, mask_scale=0.0)
    for client_plain, client_masked in zip(plain, masked):
        for p, m in zip(client_plain, client_masked):
            np.testing.assert_array_equal(p, m)


# ---------------------------------------------------------------------------
# Privacy property: no individual masked gradient equals the plain gradient
# (when mask_scale is large enough relative to the gradient)
# ---------------------------------------------------------------------------

def test_server_cannot_recover_individual_gradients():
    """With nonzero mask_scale, no masked gradient equals its plain counterpart."""
    shapes = [(32,)]
    plain = _make_plain_grads(3, shapes, seed=77)
    masked = _mask_all_clients(plain, round_num=3, mask_scale=1.0)
    for i, (p_grads, m_grads) in enumerate(zip(plain, masked)):
        for p, m in zip(p_grads, m_grads):
            assert not np.allclose(p, m), \
                f"Client {i} masked gradient equals plain gradient — mask had no effect"


# ---------------------------------------------------------------------------
# Round independence: different rounds produce different masks
# ---------------------------------------------------------------------------

def test_round_seed_independence():
    shapes = [(8,)]
    plain = _make_plain_grads(2, shapes, seed=0)
    masked_r1 = _mask_all_clients(plain, round_num=1)
    masked_r2 = _mask_all_clients(plain, round_num=2)
    # The two rounds produce different masked gradients
    for m1, m2 in zip(masked_r1, masked_r2):
        for a1, a2 in zip(m1, m2):
            assert not np.allclose(a1, a2), "Masks should differ between rounds"


# ---------------------------------------------------------------------------
# make_round_configs returns correct keys and values
# ---------------------------------------------------------------------------

def test_make_round_configs_keys():
    n = 5
    configs = SecureAggregator.make_round_configs(round_num=7, n_clients=n, mask_scale=0.02)
    required = {
        "secagg_enabled", "secagg_round_num", "secagg_round_seed",
        "secagg_client_idx", "secagg_cohort_size", "secagg_mask_scale",
    }
    assert len(configs) == n
    for i, cfg in enumerate(configs):
        assert required == set(cfg.keys()), f"Missing keys in config {i}"
        assert cfg["secagg_client_idx"] == i
        assert cfg["secagg_cohort_size"] == n
        assert cfg["secagg_enabled"] == 1
        assert cfg["secagg_mask_scale"] == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# apply_masks is a pure function (does not mutate input)
# ---------------------------------------------------------------------------

def test_apply_masks_pure():
    shapes = [(4, 4)]
    plain = _make_plain_grads(3, shapes, seed=55)
    original_copies = [arr.copy() for arr in plain[0]]

    configs = SecureAggregator.make_round_configs(round_num=1, n_clients=3)
    _ = SecureAggregator.apply_masks(
        plain[0],
        round_num=configs[0]["secagg_round_num"],
        client_idx=configs[0]["secagg_client_idx"],
        cohort_size=configs[0]["secagg_cohort_size"],
        round_seed=configs[0]["secagg_round_seed"],
        mask_scale=configs[0]["secagg_mask_scale"],
    )

    for original, after in zip(original_copies, plain[0]):
        np.testing.assert_array_equal(original, after, err_msg="apply_masks mutated the input")


# ---------------------------------------------------------------------------
# verify_cancellation correctly detects non-cancellation
# ---------------------------------------------------------------------------

def test_verify_cancellation_detects_failure():
    shapes = [(6,)]
    plain = _make_plain_grads(2, shapes)
    # Corrupt one masked gradient so masks no longer cancel
    masked = _mask_all_clients(plain, round_num=1)
    masked[0][0] += 99.0  # deliberate corruption
    result = SecureAggregator.verify_cancellation(masked, plain)
    assert result is False
