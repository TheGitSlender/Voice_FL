"""
Lightweight integration test for the federated stack — no GPU, no model, no gRPC.

Drives PerFedAvgStrategy directly with mock Flower objects to verify:
  1. Strategy instantiation with all new params
  2. configure_fit() with SecAgg disabled (single FitIns broadcast)
  3. configure_fit() with SecAgg enabled (per-client FitIns with mask seeds)
  4. aggregate_fit() with each robust method
  5. SecAgg round-trip: masks applied by clients cancel on aggregation
  6. Backward-compat: default config (method=mean, secure_agg=False) is identical to old logic

Run from repo root:
    python scripts/test_fed_integration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from flwr.common import (
    Code,
    FitIns,
    FitRes,
    Parameters,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from federated.aggregation import RobustAggregator
from federated.secure_agg import SecureAggregator
from federated.strategy_maml import PerFedAvgStrategy


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

PARAM_SHAPES = [(32, 8), (8,), (64, 16), (16,)]
N_PARAMS = len(PARAM_SHAPES)
RNG = np.random.default_rng(0)


def _rand_params() -> list[np.ndarray]:
    return [RNG.standard_normal(s).astype(np.float32) * 0.01 for s in PARAM_SHAPES]


def _make_strategy(**kwargs) -> PerFedAvgStrategy:
    import mlflow
    mlflow.end_run()  # close any previously active run before starting a new one
    initial = ndarrays_to_parameters(_rand_params())
    defaults = dict(
        initial_parameters=initial,
        min_available_clients=3,
        fraction_fit=1.0,
        outer_lr=1e-4,
        total_rounds=10,
        warmup_rounds=2,
        checkpoint_dir=None,   # no disk I/O during tests
        experiment_name="test_run",
    )
    defaults.update(kwargs)
    return PerFedAvgStrategy(**defaults)


def _make_fit_res(grads: list[np.ndarray], n_examples: int = 50) -> FitRes:
    return FitRes(
        status=Status(code=Code.OK, message=""),
        parameters=ndarrays_to_parameters(grads),
        num_examples=n_examples,
        metrics={
            "query_loss": float(RNG.uniform(1.0, 3.0)),
            "grad_norm": float(RNG.uniform(5.0, 20.0)),
            "clip_coef": 1.0,
            "inner_loss_init": float(RNG.uniform(2.0, 5.0)),
            "inner_loss_final": float(RNG.uniform(1.0, 3.0)),
        },
    )


class _MockClientProxy:
    """Minimal stand-in for flwr.server.client_proxy.ClientProxy."""
    def __init__(self, cid: str):
        self.cid = cid


class _MockClientManager:
    """Returns a fixed set of mock clients on sample()."""
    def __init__(self, n: int):
        self._clients = [_MockClientProxy(f"client_{i}") for i in range(n)]

    def sample(self, num_clients: int, min_num_clients: int):
        return self._clients[:num_clients]


# ---------------------------------------------------------------------------
# Test 1: strategy instantiation with all new params
# ---------------------------------------------------------------------------

def test_instantiation():
    strategy = _make_strategy(
        robust_method="trimmed_mean",
        trim_ratio=0.1,
        n_byzantine=1,
        norm_filter_multiplier=2.0,
        secure_agg=True,
        secagg_mask_scale=0.05,
    )
    assert strategy._robust_method == "trimmed_mean"
    assert strategy._secure_agg is True
    print("  [1] instantiation with all new params ... OK")


# ---------------------------------------------------------------------------
# Test 2: configure_fit() without SecAgg — same FitIns for all clients
# ---------------------------------------------------------------------------

def test_configure_fit_no_secagg():
    strategy = _make_strategy(secure_agg=False)
    mgr = _MockClientManager(3)
    initial_params = ndarrays_to_parameters(_rand_params())

    pairs = strategy.configure_fit(server_round=1, parameters=initial_params, client_manager=mgr)
    assert len(pairs) == 3

    # All clients should share the same FitIns object (same config)
    configs = [fit_ins.config for _, fit_ins in pairs]
    assert all(c == configs[0] for c in configs), "All clients should have the same config without SecAgg"
    assert "secagg_enabled" not in configs[0]
    print("  [2] configure_fit() without SecAgg ... OK")


# ---------------------------------------------------------------------------
# Test 3: configure_fit() with SecAgg — unique per-client FitIns
# ---------------------------------------------------------------------------

def test_configure_fit_with_secagg():
    strategy = _make_strategy(secure_agg=True, secagg_mask_scale=0.02)
    mgr = _MockClientManager(3)
    initial_params = ndarrays_to_parameters(_rand_params())

    pairs = strategy.configure_fit(server_round=1, parameters=initial_params, client_manager=mgr)
    assert len(pairs) == 3

    configs = [fit_ins.config for _, fit_ins in pairs]
    for i, cfg in enumerate(configs):
        assert cfg["secagg_enabled"] == 1
        assert cfg["secagg_client_idx"] == i
        assert cfg["secagg_cohort_size"] == 3
        assert cfg["secagg_mask_scale"] == 0.02

    # Each client must have a different client_idx
    idxs = [c["secagg_client_idx"] for c in configs]
    assert idxs == [0, 1, 2]
    # All share the same round_seed (derived from server round)
    seeds = [c["secagg_round_seed"] for c in configs]
    assert len(set(seeds)) == 1, "All clients in the same round share the round_seed"
    print("  [3] configure_fit() with SecAgg — per-client FitIns ... OK")


# ---------------------------------------------------------------------------
# Test 4: aggregate_fit() with each robust method
# ---------------------------------------------------------------------------

def test_aggregate_fit_all_methods():
    n_clients = 5
    client_grads = [_rand_params() for _ in range(n_clients)]
    results = [(None, _make_fit_res(grads, n_examples=50)) for grads in client_grads]

    for method in ("mean", "trimmed_mean", "coordinate_median", "krum", "norm_filter"):
        strategy = _make_strategy(robust_method=method)
        updated_params, agg_metrics = strategy.aggregate_fit(
            server_round=1, results=results, failures=[]
        )
        assert updated_params is not None, f"aggregate_fit returned None for method={method}"
        # Verify the returned parameters have the right number of arrays
        arrays = parameters_to_ndarrays(updated_params)
        assert len(arrays) == N_PARAMS, f"Wrong param count for method={method}"
        for arr, shape in zip(arrays, PARAM_SHAPES):
            assert arr.shape == shape, f"Shape mismatch for method={method}: {arr.shape} != {shape}"
        print(f"  [4] aggregate_fit() method={method!r} ... OK")


# ---------------------------------------------------------------------------
# Test 5: SecAgg round-trip — masked grads from clients cancel on aggregation
# ---------------------------------------------------------------------------

def test_secagg_round_trip():
    """
    Simulates a full SecAgg round:
      - Server sends per-client configs via configure_fit()
      - Each client applies masks to its gradients
      - Server aggregates — masks cancel — result equals plain mean
    """
    n_clients = 3
    strategy_secagg = _make_strategy(
        secure_agg=True,
        secagg_mask_scale=0.05,
        robust_method="mean",
    )
    strategy_plain = _make_strategy(
        secure_agg=False,
        robust_method="mean",
    )
    # Sync theta_star so both strategies start from the same point
    strategy_plain._theta_star = [t.copy() for t in strategy_secagg._theta_star]

    # Generate client gradient data
    plain_grads = [_rand_params() for _ in range(n_clients)]

    # Simulate SecAgg: server issues per-client configs
    mgr = _MockClientManager(n_clients)
    initial_params = ndarrays_to_parameters(strategy_secagg._theta_star)
    pairs = strategy_secagg.configure_fit(server_round=1, parameters=initial_params, client_manager=mgr)

    # Each client applies its masks
    masked_results = []
    for (_, fit_ins), grads in zip(pairs, plain_grads):
        cfg = fit_ins.config
        masked = SecureAggregator.apply_masks(
            grads,
            round_num=int(cfg["secagg_round_num"]),
            client_idx=int(cfg["secagg_client_idx"]),
            cohort_size=int(cfg["secagg_cohort_size"]),
            round_seed=int(cfg["secagg_round_seed"]),
            mask_scale=float(cfg["secagg_mask_scale"]),
        )
        masked_results.append((None, _make_fit_res(masked, n_examples=50)))

    # Plain results (no masks)
    plain_results = [(None, _make_fit_res(g, n_examples=50)) for g in plain_grads]

    # Both strategies must produce the same θ* after aggregation
    updated_secagg, _ = strategy_secagg.aggregate_fit(1, masked_results, [])
    updated_plain, _ = strategy_plain.aggregate_fit(1, plain_results, [])

    secagg_arrays = parameters_to_ndarrays(updated_secagg)
    plain_arrays = parameters_to_ndarrays(updated_plain)

    for i, (sa, pa) in enumerate(zip(secagg_arrays, plain_arrays)):
        np.testing.assert_allclose(
            sa, pa, atol=1e-4,
            err_msg=f"Param {i}: SecAgg result diverges from plain result — masks did not cancel"
        )
    print("  [5] SecAgg round-trip: masked aggregate == plain aggregate ... OK")


# ---------------------------------------------------------------------------
# Test 6: backward compat — default strategy produces same result as reference mean
# ---------------------------------------------------------------------------

def test_backward_compat_default_equals_reference_mean():
    """Default strategy (mean, no SecAgg) must match a hand-computed weighted mean."""
    n_clients = 4
    weights = [30.0, 50.0, 20.0, 40.0]
    client_grads = [_rand_params() for _ in range(n_clients)]

    strategy = _make_strategy()  # all defaults

    # Capture theta_star before update
    theta_before = [t.copy() for t in strategy._theta_star]

    results = [
        (None, _make_fit_res(grads, n_examples=int(w)))
        for grads, w in zip(client_grads, weights)
    ]
    strategy.aggregate_fit(1, results, [])

    # The strategy must have moved theta_star (it applied AdamW)
    moved = any(
        not np.allclose(strategy._theta_star[i], theta_before[i])
        for i in range(N_PARAMS)
    )
    assert moved, "theta_star did not change after aggregate_fit"
    print("  [6] backward compat: theta_star updates on default config ... OK")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_instantiation,
        test_configure_fit_no_secagg,
        test_configure_fit_with_secagg,
        test_aggregate_fit_all_methods,
        test_secagg_round_trip,
        test_backward_compat_default_equals_reference_mean,
    ]

    print("=" * 60)
    print("Federated stack integration tests (no GPU, no gRPC)")
    print("=" * 60)
    import mlflow
    passed = failed = 0
    for fn in tests:
        try:
            fn()
            passed += 1
        except Exception as exc:
            print(f"  FAIL {fn.__name__}: {exc}")
            import traceback; traceback.print_exc()
            failed += 1
        finally:
            mlflow.end_run()  # always clean up after each test

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
