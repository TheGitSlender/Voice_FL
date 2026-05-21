"""
Speed benchmark: native CTC vs custom v1 (sequential) vs v2 (parallel scan).

Measures time per forward+backward with double-backward (create_graph=True)
at three sequence lengths.

Run: python scripts/benchmark_ctc.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from ctc.differentiable_ctc import DifferentiableCTC
from ctc.differentiable_ctc_v2 import DifferentiableCTCFast


def _benchmark_native(T: int, C: int, S: int, n_iters: int, device: str) -> float:
    """Native nn.CTCLoss (first-order only — no double-backward)."""
    dtype = torch.float32
    native_ctc = nn.CTCLoss(blank=0, reduction="none")
    times = []
    for _ in range(n_iters):
        torch.manual_seed(42)
        logits = torch.randn(T, C, dtype=dtype, device=device, requires_grad=True)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        log_probs_ctc = log_probs.unsqueeze(1)
        targets = torch.randint(1, C, (S,), device=device)
        input_lengths = torch.tensor([T], device=device)
        target_lengths = torch.tensor([S], device=device)

        if device != "cpu":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        loss = native_ctc(log_probs_ctc, targets.unsqueeze(0), input_lengths, target_lengths)
        loss.squeeze().backward()

        if device != "cpu":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    warmup = min(5, n_iters // 2)
    return sum(times[warmup:]) / len(times[warmup:]) * 1000


def _benchmark_ctc(ctc_module, T: int, C: int, S: int, n_iters: int, device: str) -> float:
    """Custom CTC with double-backward (create_graph=True)."""
    dtype = torch.float32
    times = []
    for _ in range(n_iters):
        torch.manual_seed(42)
        logits = torch.randn(T, C, dtype=dtype, device=device, requires_grad=True)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        targets = torch.randint(1, C, (S,), device=device)

        if device != "cpu":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        loss = ctc_module(log_probs, targets, blank=0)
        grad = torch.autograd.grad(loss, logits, create_graph=True)[0]
        grad.norm().backward()

        if device != "cpu":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    warmup = min(5, n_iters // 2)
    return sum(times[warmup:]) / len(times[warmup:]) * 1000


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    C = 32
    S = 8

    configs = [
        (100, "~2s audio"),
        (250, "~5s audio"),
        (500, "~10s audio"),
    ]

    header = (
        f"{'T frames':<10} {'Desc':<12} "
        f"{'Native(ms)':<12} {'v1(ms)':<10} {'v2(ms)':<10} "
        f"{'v1/nat':<8} {'v2/nat':<8} {'v1/v2':<8}"
    )
    print(header)
    print("-" * len(header))

    v1 = DifferentiableCTC()
    v2 = DifferentiableCTCFast()

    for T, desc in configs:
        n_iters = 20
        native_ms = _benchmark_native(T, C, S, n_iters, device)
        v1_ms = _benchmark_ctc(v1, T, C, S, n_iters, device)
        v2_ms = _benchmark_ctc(v2, T, C, S, n_iters, device)

        r_v1 = v1_ms / native_ms if native_ms > 0 else float("inf")
        r_v2 = v2_ms / native_ms if native_ms > 0 else float("inf")
        speedup = v1_ms / v2_ms if v2_ms > 0 else float("inf")

        print(
            f"{T:<10} {desc:<12} "
            f"{native_ms:<12.2f} {v1_ms:<10.2f} {v2_ms:<10.2f} "
            f"{r_v1:<8.1f}x {r_v2:<8.1f}x {speedup:<8.1f}x"
        )

    print(
        "\nColumns:\n"
        "  Native  = nn.CTCLoss (first-order only, no double-backward)\n"
        "  v1      = DifferentiableCTC (sequential loop, double-backward)\n"
        "  v2      = DifferentiableCTCFast (parallel scan, double-backward)\n"
        "  v1/nat  = how much slower v1 is vs native\n"
        "  v2/nat  = how much slower v2 is vs native\n"
        "  v1/v2   = speedup from v1 to v2 (higher = better)\n"
    )


if __name__ == "__main__":
    main()
