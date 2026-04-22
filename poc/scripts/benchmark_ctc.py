"""
Speed benchmark: custom differentiable CTC vs PyTorch native CTC.

Measures time per inner loop step at three sequence lengths.
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

def _benchmark(
    T: int, C: int, S: int, n_iters: int = 20, device: str = "cpu"
) -> tuple[float, float]:
    """Returns (native_ms, custom_ms) per iteration."""
    dtype = torch.float32
    torch.manual_seed(42)
    targets = torch.randint(1, C, (S,), device=device)

    native_ctc = nn.CTCLoss(blank=0, reduction="none")
    times_native = []
    for _ in range(n_iters):
        logits = torch.randn(T, C, dtype=dtype, device=device, requires_grad=True)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        log_probs_ctc = log_probs.unsqueeze(1)             
        input_lengths = torch.tensor([T], device=device)
        target_lengths = torch.tensor([S], device=device)

        if device != "cpu":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        loss = native_ctc(log_probs_ctc, targets.unsqueeze(0), input_lengths, target_lengths)
        loss.squeeze().backward()

        if device != "cpu":
            torch.cuda.synchronize()
        times_native.append(time.perf_counter() - t0)

    ctc = DifferentiableCTC()
    times_custom = []
    for _ in range(n_iters):
        logits = torch.randn(T, C, dtype=dtype, device=device, requires_grad=True)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        if device != "cpu":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        loss = ctc(log_probs, targets, blank=0)
        grad = torch.autograd.grad(loss, logits, create_graph=True)[0]
        grad.norm().backward()

        if device != "cpu":
            torch.cuda.synchronize()
        times_custom.append(time.perf_counter() - t0)

    warmup = 5
    native_ms = sum(times_native[warmup:]) / len(times_native[warmup:]) * 1000
    custom_ms = sum(times_custom[warmup:]) / len(times_custom[warmup:]) * 1000

    return native_ms, custom_ms

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

    print(f"{'T frames':<12} {'Description':<14} {'Native (ms)':<14} {'Custom+d2 (ms)':<16} {'Ratio':<8}")
    print("-" * 64)

    for T, desc in configs:
        native_ms, custom_ms = _benchmark(T, C, S, n_iters=20, device=device)
        ratio = custom_ms / native_ms if native_ms > 0 else float("inf")
        print(f"{T:<12} {desc:<14} {native_ms:<14.2f} {custom_ms:<16.2f} {ratio:<8.1f}x")

    print("\nExpected ratio: 3-10x (custom with double-backward vs native first-order only)")

if __name__ == "__main__":
    main()
