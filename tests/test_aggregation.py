"""Tests for Byzantine-robust gradient aggregation."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from federated.aggregation import RobustAggregator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_grads(n_clients: int, n_params: int, param_shape: tuple, seed: int = 0):
    """Return n_clients honest gradient lists, each with n_params arrays."""
    rng = np.random.default_rng(seed)
    return [
        [rng.standard_normal(param_shape).astype(np.float32) for _ in range(n_params)]
        for _ in range(n_clients)
    ]


def _uniform_weights(n: int, val: float = 10.0) -> list:
    return [val] * n


def _reference_weighted_mean(grads, weights):
    total = sum(weights)
    result = [np.zeros_like(grads[0][i], dtype=np.float32) for i in range(len(grads[0]))]
    for cg, w in zip(grads, weights):
        for i, g in enumerate(cg):
            result[i] += (w / total) * g.astype(np.float32)
    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def aggregator():
    return RobustAggregator()


# ---------------------------------------------------------------------------
# Backward compatibility: mean == old weighted accumulation
# ---------------------------------------------------------------------------

def test_mean_matches_weighted_average(aggregator):
    grads = _make_grads(5, 3, (4, 4))
    weights = [3.0, 7.0, 5.0, 2.0, 8.0]
    result, audit = aggregator.aggregate(grads, weights, method="mean")
    expected = _reference_weighted_mean(grads, weights)
    for r, e in zip(result, expected):
        np.testing.assert_allclose(r, e, rtol=1e-5)
    assert audit["n_clients_used"] == 5
    assert audit["n_clients_dropped"] == 0


# ---------------------------------------------------------------------------
# Degenerate: single client passes through unchanged
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["mean", "trimmed_mean", "coordinate_median", "krum", "norm_filter"])
def test_degenerate_n1(aggregator, method):
    grads = _make_grads(1, 2, (3,))
    result, audit = aggregator.aggregate(grads, [1.0], method=method)
    for r, g in zip(result, grads[0]):
        np.testing.assert_array_equal(r, g)
    assert audit["n_clients_in"] == 1


# ---------------------------------------------------------------------------
# Trimmed mean removes outlier gradient
# ---------------------------------------------------------------------------

def test_trimmed_mean_removes_outlier(aggregator):
    rng = np.random.default_rng(42)
    shape = (10,)
    # 4 honest clients with small gradients
    honest = [[rng.standard_normal(shape).astype(np.float32)] for _ in range(4)]
    # 1 Byzantine with a 100× gradient
    byzantine = [[(rng.standard_normal(shape) * 100).astype(np.float32)]]

    grads = honest + byzantine
    weights = _uniform_weights(5)

    mean_result, _ = aggregator.aggregate(grads, weights, method="mean")
    trim_result, audit = aggregator.aggregate(grads, weights, method="trimmed_mean", trim_ratio=0.2)

    # Trimmed mean should be much closer to honest mean than plain mean
    honest_mean = np.mean([g[0] for g in honest], axis=0)
    mean_err = float(np.linalg.norm(mean_result[0] - honest_mean))
    trim_err = float(np.linalg.norm(trim_result[0] - honest_mean))
    assert trim_err < mean_err, "trimmed_mean should outperform mean against outlier"
    assert audit["n_clients_dropped"] >= 1


# ---------------------------------------------------------------------------
# Coordinate median is robust to 1 Byzantine in N=5
# ---------------------------------------------------------------------------

def test_coordinate_median_robust_to_poisoning(aggregator):
    rng = np.random.default_rng(7)
    shape = (20,)
    honest = [[rng.standard_normal(shape).astype(np.float32)] for _ in range(4)]
    byzantine = [[(rng.standard_normal(shape) * 200).astype(np.float32)]]
    grads = honest + byzantine
    weights = _uniform_weights(5)

    mean_result, _ = aggregator.aggregate(grads, weights, method="mean")
    med_result, _ = aggregator.aggregate(grads, weights, method="coordinate_median")

    honest_mean = np.mean([g[0] for g in honest], axis=0)
    mean_err = float(np.linalg.norm(mean_result[0] - honest_mean))
    med_err = float(np.linalg.norm(med_result[0] - honest_mean))
    assert med_err < mean_err


# ---------------------------------------------------------------------------
# Krum selects an honest client's gradient (N=5, f=1)
# ---------------------------------------------------------------------------

def test_krum_selects_honest_client(aggregator):
    rng = np.random.default_rng(13)
    shape = (16,)
    # 4 honest clients clustered near zero
    honest = [
        [(rng.standard_normal(shape) * 0.1).astype(np.float32)]
        for _ in range(4)
    ]
    # 1 Byzantine far away
    byzantine = [[(rng.standard_normal(shape) * 100 + 500).astype(np.float32)]]
    grads = honest + byzantine  # byzantine is index 4
    weights = _uniform_weights(5)

    result, audit = aggregator.aggregate(grads, weights, method="krum", n_byzantine=1)
    selected = audit["krum_selected_idx"]
    # Krum must not select the Byzantine client
    assert selected != 4, f"Krum selected the Byzantine client (index 4)"
    assert audit["n_clients_used"] == 1


def test_krum_n3_f1(aggregator):
    """Minimum viable krum: N=3, f=1 — selects 1 of 3."""
    rng = np.random.default_rng(99)
    shape = (8,)
    honest = [[(rng.standard_normal(shape) * 0.1).astype(np.float32)] for _ in range(2)]
    byzantine = [[(np.ones(shape) * 1000).astype(np.float32)]]
    grads = honest + byzantine
    weights = _uniform_weights(3)

    result, audit = aggregator.aggregate(grads, weights, method="krum", n_byzantine=1)
    assert audit["krum_selected_idx"] in (0, 1), "Should select one of the honest clients"


# ---------------------------------------------------------------------------
# Norm filter drops high-norm client
# ---------------------------------------------------------------------------

def test_norm_filter_drops_high_norm(aggregator):
    rng = np.random.default_rng(21)
    shape = (32,)
    honest = [[rng.standard_normal(shape).astype(np.float32)] for _ in range(4)]
    # Byzantine has norm ~50× the honest clients
    big = [(rng.standard_normal(shape) * 50).astype(np.float32)]
    grads = honest + [big]
    weights = _uniform_weights(5)

    result, audit = aggregator.aggregate(
        grads, weights, method="norm_filter", norm_filter_multiplier=2.0
    )
    assert audit["n_clients_dropped"] >= 1
    honest_mean = _reference_weighted_mean(honest, _uniform_weights(4))
    filtered_err = float(np.linalg.norm(result[0] - honest_mean[0]))
    plain_result, _ = aggregator.aggregate(grads, weights, method="mean")
    plain_err = float(np.linalg.norm(plain_result[0] - honest_mean[0]))
    assert filtered_err < plain_err


# ---------------------------------------------------------------------------
# Invalid method raises ValueError
# ---------------------------------------------------------------------------

def test_invalid_method_raises(aggregator):
    grads = _make_grads(3, 1, (4,))
    with pytest.raises(ValueError, match="Unknown aggregation method"):
        aggregator.aggregate(grads, _uniform_weights(3), method="bad_method")


# ---------------------------------------------------------------------------
# Audit dict always contains required keys
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["mean", "trimmed_mean", "coordinate_median", "krum", "norm_filter"])
def test_audit_keys_present(aggregator, method):
    grads = _make_grads(4, 2, (5,))
    _, audit = aggregator.aggregate(grads, _uniform_weights(4), method=method)
    for key in ("method", "n_clients_in", "n_clients_used", "n_clients_dropped"):
        assert key in audit, f"Missing audit key {key!r} for method {method!r}"
    assert audit["n_clients_used"] + audit["n_clients_dropped"] == audit["n_clients_in"]
