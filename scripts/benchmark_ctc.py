"""
Speed benchmark: native CTC vs v1 vs v2-sequential vs v2-scan.

Measures time per forward+backward with double-backward (create_graph=True).
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
    native_ctc = nn.CTCLoss(blank=0, reduction="none")
    times = []
    for _ in range(n_iters):
        torch.manual_seed(42)
        logits = torch.randn(T, C, dtype=torch.float32, device=device, requires_grad=True)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1).unsqueeze(1)
        targets = torch.randint(1, C, (S,), device=device)
        il = torch.tensor([T], device=device)
        tl = torch.tensor([S], device=device)
        if device != "cpu":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = native_ctc(log_probs, targets.unsqueeze(0), il, tl)
        loss.squeeze().backward()
        if device != "cpu":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    warmup = min(5, n_iters // 2)
    return sum(times[warmup:]) / len(times[warmup:]) * 1000


def _benchmark_ctc(ctc_module, T: int, C: int, S: int, n_iters: int, device: str) -> float:
    """Custom CTC with double-backward (create_graph=True)."""
    times = []
    for _ in range(n_iters):
        torch.manual_seed(42)
        logits = torch.randn(T, C, dtype=torch.float32, device=device, requires_grad=True)
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


def _measure_vram(ctc_module, T: int, C: int, S: int, device: str) -> float:
    """Measure VRAM delta (MB) for one forward+double-backward with create_graph=True."""
    if device == "cpu":
        return 0.0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()
    torch.manual_seed(42)
    logits = torch.randn(T, C, dtype=torch.float32, device=device, requires_grad=True)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    targets = torch.randint(1, C, (S,), device=device)
    loss = ctc_module(log_probs, targets, blank=0)
    grad = torch.autograd.grad(loss, logits, create_graph=True)[0]
    grad.norm().backward()
    mem_after = torch.cuda.max_memory_allocated()
    del loss, grad, logits, log_probs
    torch.cuda.empty_cache()
    return (mem_after - mem_before) / 1024 / 1024


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    C = 32
    S = 8
    n_iters = 20

    configs = [
        (100, "~2s audio"),
        (250, "~5s audio"),
        (500, "~10s audio"),
    ]

    v1 = DifferentiableCTC()
    v2_seq = DifferentiableCTCFast(scan=False)
    v2_scan = DifferentiableCTCFast(scan=True)

    header = (
        f"{'T':<6} {'Desc':<12} "
        f"{'Native':<10} {'v1':<10} {'v2-seq':<10} {'v2-scan':<10} "
        f"{'v1/v2s':<8} {'v1/scan':<8}"
    )
    print(header)
    print("-" * len(header))

    for T, desc in configs:
        nat = _benchmark_native(T, C, S, n_iters, device)
        t_v1 = _benchmark_ctc(v1, T, C, S, n_iters, device)
        t_seq = _benchmark_ctc(v2_seq, T, C, S, n_iters, device)
        t_scan = _benchmark_ctc(v2_scan, T, C, S, n_iters, device)

        sp_seq = t_v1 / t_seq if t_seq > 0 else float("inf")
        sp_scan = t_v1 / t_scan if t_scan > 0 else float("inf")

        print(
            f"{T:<6} {desc:<12} "
            f"{nat:<10.2f} {t_v1:<10.2f} {t_seq:<10.2f} {t_scan:<10.2f} "
            f"{sp_seq:<8.1f}x {sp_scan:<8.1f}x"
        )

    if device != "cpu":
        T_mem = 250
        print(f"\n--- VRAM usage (single call, T={T_mem}, S={S}) ---")
        for name, mod in [("v1", v1), ("v2-seq", v2_seq), ("v2-scan", v2_scan)]:
            vram = _measure_vram(mod, T_mem, C, S, device)
            print(f"  {name:<10} {vram:>8.1f} MB")

    print(
        "\nColumns:\n"
        "  Native   = nn.CTCLoss (first-order only)\n"
        "  v1       = DifferentiableCTC (sequential, custom logsumexp)\n"
        "  v2-seq   = DifferentiableCTCFast(scan=False) — optimized sequential [DEFAULT]\n"
        "  v2-scan  = DifferentiableCTCFast(scan=True)  — parallel scan (high VRAM)\n"
        "  v1/v2s   = speedup of v2-seq over v1\n"
        "  v1/scan  = speedup of v2-scan over v1\n"
    )


if __name__ == "__main__":
    main()
