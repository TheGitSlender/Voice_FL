"""Cross-validation tests: DifferentiableCTCFast (v2) vs DifferentiableCTC (v1).

Ensures the parallel-scan implementation produces identical results to the
sequential reference implementation across all edge cases.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ctc.differentiable_ctc import DifferentiableCTC, ctc_loss_differentiable
from ctc.differentiable_ctc_v2 import (
    DifferentiableCTCFast,
    _build_transition_mask,
    _build_transition_matrices,
    _get_neg_inf,
    _log_semiring_bmm,
    _log_semiring_matvec,
    _parallel_reduce,
    ctc_loss_differentiable_fast,
)
from tests.conftest import BLANK, VOCAB_SIZE


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_data(T: int, S: int, dtype=torch.float64, device="cpu"):
    torch.manual_seed(42)
    logits = torch.randn(T, VOCAB_SIZE, dtype=dtype, device=device, requires_grad=True)
    targets = torch.randint(1, VOCAB_SIZE, (S,), device=device)
    return logits, targets


def _both_losses(logits, targets, blank=BLANK):
    """Compute loss with both v1 and v2 from the same logits (independent grad)."""
    lp1 = F.log_softmax(logits.detach().clone().requires_grad_(True), dim=-1)
    lp2 = F.log_softmax(logits.detach().clone().requires_grad_(True), dim=-1)
    loss1 = DifferentiableCTC()(lp1, targets, blank=blank)
    loss2 = DifferentiableCTCFast()(lp2, targets, blank=blank)
    return loss1, loss2, lp1, lp2


# ── Core correctness ────────────────────────────────────────────────────────

class TestV2MatchesV1:
    """Loss and gradient values from v2 must match v1."""

    @pytest.mark.parametrize("T,S", [
        (20, 5), (49, 11), (100, 20), (10, 1), (5, 2), (1, 1),
    ])
    def test_loss_matches(self, T: int, S: int) -> None:
        logits, targets = _make_data(T, S)
        loss1, loss2, _, _ = _both_losses(logits, targets)
        rel_err = abs(loss1.item() - loss2.item()) / (abs(loss1.item()) + 1e-12)
        assert rel_err < 1e-4, f"Loss mismatch: v1={loss1.item():.8f} v2={loss2.item():.8f} rel={rel_err:.2e}"

    @pytest.mark.parametrize("T,S", [(20, 5), (49, 11), (10, 1)])
    def test_gradient_matches(self, T: int, S: int) -> None:
        logits, targets = _make_data(T, S)
        loss1, loss2, lp1, lp2 = _both_losses(logits, targets)
        loss1.backward()
        loss2.backward()
        max_rel = ((lp1.grad - lp2.grad).abs() / (lp1.grad.abs() + 1e-10)).max()
        assert max_rel < 1e-3, f"Grad max rel err: {max_rel:.2e}"

    def test_repeated_labels(self) -> None:
        """All-same labels: hardest CTC case (mandatory blanks)."""
        T = 10
        targets = torch.tensor([17, 17, 17, 17])
        torch.manual_seed(0)
        logits = torch.randn(T, VOCAB_SIZE, dtype=torch.float64, requires_grad=True)
        loss1, loss2, _, _ = _both_losses(logits, targets)
        rel_err = abs(loss1.item() - loss2.item()) / (abs(loss1.item()) + 1e-12)
        assert rel_err < 1e-4

    def test_single_frame_single_label(self) -> None:
        """T=1, S=1: absolute minimum."""
        logits, targets = _make_data(1, 1)
        loss1, loss2, _, _ = _both_losses(logits, targets)
        rel_err = abs(loss1.item() - loss2.item()) / (abs(loss1.item()) + 1e-12)
        assert rel_err < 1e-4


# ── Double-backward correctness ─────────────────────────────────────────────

class TestDoubleBackward:
    """create_graph=True works and produces correct second-order gradients."""

    def test_double_backward_succeeds(self) -> None:
        T, S = 15, 4
        logits, targets = _make_data(T, S)
        log_probs = F.log_softmax(logits, dim=-1)
        loss = DifferentiableCTCFast()(log_probs, targets, blank=BLANK)
        grad = torch.autograd.grad(loss, logits, create_graph=True)[0]
        grad_of_grad = torch.autograd.grad(grad.norm(), logits)[0]
        assert torch.isfinite(grad_of_grad).all()
        assert grad_of_grad.abs().sum().item() > 0

    def test_second_order_matches_v1(self) -> None:
        """Second-order gradients from v2 match v1."""
        T, S = 15, 4
        logits, targets = _make_data(T, S)

        lp1 = F.log_softmax(logits.detach().clone().requires_grad_(True), dim=-1)
        loss1 = DifferentiableCTC()(lp1, targets, blank=BLANK)
        g1 = torch.autograd.grad(loss1, lp1, create_graph=True)[0]
        g1_of_g1 = torch.autograd.grad(g1.norm(), lp1)[0]

        lp2 = F.log_softmax(logits.detach().clone().requires_grad_(True), dim=-1)
        loss2 = DifferentiableCTCFast()(lp2, targets, blank=BLANK)
        g2 = torch.autograd.grad(loss2, lp2, create_graph=True)[0]
        g2_of_g2 = torch.autograd.grad(g2.norm(), lp2)[0]

        # Second-order grads should be close but may not be identical
        # due to different computation order (scan vs loop)
        cos_sim = F.cosine_similarity(
            g1_of_g1.flatten().unsqueeze(0), g2_of_g2.flatten().unsqueeze(0)
        ).item()
        assert cos_sim > 0.99, f"Second-order cosine sim too low: {cos_sim:.4f}"

    def test_gradcheck(self) -> None:
        """torch.autograd.gradcheck passes for v2 in float64."""
        T, C, S = 8, 6, 3
        torch.manual_seed(42)
        logits = torch.randn(T, C, dtype=torch.float64, requires_grad=True)
        targets = torch.randint(1, C, (S,))

        def fn(lp):
            return DifferentiableCTCFast()(F.log_softmax(lp, dim=-1), targets, blank=0)

        assert torch.autograd.gradcheck(fn, (logits,), eps=1e-5, atol=1e-4)

    def test_gradgradcheck(self) -> None:
        """torch.autograd.gradgradcheck passes for v2 in float64."""
        T, C, S = 8, 6, 3
        torch.manual_seed(42)
        logits = torch.randn(T, C, dtype=torch.float64, requires_grad=True)
        targets = torch.randint(1, C, (S,))

        def fn(lp):
            return DifferentiableCTCFast()(F.log_softmax(lp, dim=-1), targets, blank=0)

        assert torch.autograd.gradgradcheck(fn, (logits,), eps=1e-5, atol=1e-4)

    def test_hessian_diagonal(self) -> None:
        """Autograd Hessian diagonal matches finite difference within 5%."""
        T, C, S = 10, 8, 3
        eps = 1e-3
        torch.manual_seed(42)
        logits_base = torch.randn(T, C, dtype=torch.float64)
        targets = torch.randint(1, C, (S,))
        ctc = DifferentiableCTCFast()

        logits_ag = logits_base.clone().requires_grad_(True)
        loss = ctc(F.log_softmax(logits_ag, dim=-1), targets, blank=0)
        grad = torch.autograd.grad(loss, logits_ag, create_graph=True)[0].flatten()

        for i in range(5):
            g2 = torch.autograd.grad(grad[i], logits_ag, retain_graph=True)[0].flatten()[i]

            flat = logits_base.flatten()
            fp, fm = flat.clone(), flat.clone()
            fp[i] += eps
            fm[i] -= eps
            lp_p = fp.reshape(T, C).requires_grad_(True)
            lp_m = fm.reshape(T, C).requires_grad_(True)
            gp = torch.autograd.grad(ctc(F.log_softmax(lp_p, dim=-1), targets), lp_p)[0].flatten()[i]
            gm = torch.autograd.grad(ctc(F.log_softmax(lp_m, dim=-1), targets), lp_m)[0].flatten()[i]
            fd = (gp - gm) / (2 * eps)

            denom = abs(fd.item()) if abs(fd.item()) > 1e-10 else 1e-10
            assert abs(g2.item() - fd.item()) / denom < 0.05, f"Hessian[{i}] mismatch"


# ── Batch interface ──────────────────────────────────────────────────────────

class TestBatchInterface:

    def test_batch_matches_v1(self) -> None:
        """Batch ctc_loss_differentiable_fast matches v1."""
        B, T, S = 2, 25, 4
        torch.manual_seed(0)
        logits = torch.randn(B, T, VOCAB_SIZE, dtype=torch.float64, requires_grad=True)
        labels = torch.randint(1, VOCAB_SIZE, (B, S))

        logits_v1 = logits.detach().clone().requires_grad_(True)
        loss_v1 = ctc_loss_differentiable(
            logits_v1, labels,
            torch.tensor([T, T]), torch.tensor([S, S]), blank=BLANK,
        )
        logits_v2 = logits.detach().clone().requires_grad_(True)
        loss_v2 = ctc_loss_differentiable_fast(
            logits_v2, labels,
            torch.tensor([T, T]), torch.tensor([S, S]), blank=BLANK,
        )

        rel_err = abs(loss_v1.item() - loss_v2.item()) / (abs(loss_v1.item()) + 1e-12)
        assert rel_err < 1e-4, f"Batch loss mismatch: {rel_err:.2e}"

    def test_padded_labels(self) -> None:
        """Labels padded with -100 are correctly stripped."""
        T = 30
        torch.manual_seed(0)
        logits = torch.randn(1, T, VOCAB_SIZE, dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([[5, 10, 15, -100, -100]])
        loss = ctc_loss_differentiable_fast(
            logits, labels, torch.tensor([T]), torch.tensor([3]), blank=BLANK,
        )
        assert torch.isfinite(loss)
        loss.backward()
        assert logits.grad is not None

    def test_mixed_lengths(self) -> None:
        B = 3
        T_max, S_max = 50, 10
        torch.manual_seed(0)
        logits = torch.randn(B, T_max, VOCAB_SIZE, dtype=torch.float64, requires_grad=True)
        input_lengths = torch.tensor([50, 30, 40])
        target_lengths = torch.tensor([10, 5, 8])
        labels = torch.full((B, S_max), -100, dtype=torch.long)
        for b in range(B):
            labels[b, :target_lengths[b]] = torch.randint(1, VOCAB_SIZE, (target_lengths[b],))

        loss = ctc_loss_differentiable_fast(logits, labels, input_lengths, target_lengths, blank=BLANK)
        assert torch.isfinite(loss)
        loss.backward()
        assert logits.grad is not None

    def test_batch_double_backward(self) -> None:
        B, T, S = 2, 25, 4
        torch.manual_seed(0)
        logits = torch.randn(B, T, VOCAB_SIZE, dtype=torch.float64, requires_grad=True)
        labels = torch.randint(1, VOCAB_SIZE, (B, S))
        loss = ctc_loss_differentiable_fast(
            logits, labels, torch.tensor([T, T]), torch.tensor([S, S]), blank=BLANK,
        )
        grad = torch.autograd.grad(loss, logits, create_graph=True)[0]
        g2 = torch.autograd.grad(grad.norm(), logits)[0]
        assert torch.isfinite(g2).all()

    def test_all_empty_targets_raises(self) -> None:
        logits = torch.randn(2, 20, VOCAB_SIZE, requires_grad=True)
        labels = torch.full((2, 5), -100, dtype=torch.long)
        with pytest.raises(ValueError, match="empty targets"):
            ctc_loss_differentiable_fast(
                logits, labels, torch.tensor([20, 20]), torch.tensor([0, 0]), blank=BLANK,
            )


# ── Numerical stability ─────────────────────────────────────────────────────

class TestNumericalStability:

    def test_float32_no_nan(self) -> None:
        T, S = 49, 8
        torch.manual_seed(0)
        logits = torch.randn(T, VOCAB_SIZE, dtype=torch.float32, requires_grad=True)
        targets = torch.randint(1, VOCAB_SIZE, (S,))
        log_probs = F.log_softmax(logits, dim=-1)
        loss = DifferentiableCTCFast()(log_probs, targets, blank=BLANK)
        assert torch.isfinite(loss)
        grad = torch.autograd.grad(loss, logits, create_graph=True)[0]
        assert torch.isfinite(grad).all()
        g2 = torch.autograd.grad(grad.norm(), logits)[0]
        assert torch.isfinite(g2).all()

    def test_extreme_logits(self) -> None:
        T, S = 20, 3
        targets = torch.randint(1, VOCAB_SIZE, (S,))
        for scale in [0.01, 1.0, 10.0, 100.0]:
            torch.manual_seed(0)
            logits = scale * torch.randn(T, VOCAB_SIZE, dtype=torch.float32, requires_grad=True)
            log_probs = F.log_softmax(logits, dim=-1)
            loss = DifferentiableCTCFast()(log_probs, targets, blank=BLANK)
            assert torch.isfinite(loss), f"Loss NaN/Inf at scale={scale}"
            grad = torch.autograd.grad(loss, logits, create_graph=True)[0]
            assert torch.isfinite(grad).all(), f"Grad NaN at scale={scale}"

    def test_uniform_logits(self) -> None:
        T, S = 20, 3
        targets = torch.randint(1, VOCAB_SIZE, (S,))
        logits = torch.zeros(T, VOCAB_SIZE, dtype=torch.float64, requires_grad=True)
        loss = DifferentiableCTCFast()(F.log_softmax(logits, dim=-1), targets, blank=BLANK)
        assert torch.isfinite(loss)
        grad = torch.autograd.grad(loss, logits, create_graph=True)[0]
        assert torch.isfinite(grad).all()
        assert grad.abs().sum() > 0


# ── Internal components ──────────────────────────────────────────────────────

class TestInternals:

    def test_log_semiring_identity(self) -> None:
        """Multiplying by identity in log-semiring preserves the matrix."""
        S = 7
        NEG_INF = _get_neg_inf(torch.float64)
        torch.manual_seed(0)
        M = torch.randn(1, S, S, dtype=torch.float64)
        I = torch.full((1, S, S), NEG_INF, dtype=torch.float64)
        idx = torch.arange(S)
        I[:, idx, idx] = 0.0

        IM = _log_semiring_bmm(I, M)
        MI = _log_semiring_bmm(M, I)

        assert torch.allclose(IM, M, atol=1e-10), "I ⊗ M ≠ M"
        assert torch.allclose(MI, M, atol=1e-10), "M ⊗ I ≠ M"

    def test_log_semiring_bmm_associative(self) -> None:
        """(A ⊗ B) ⊗ C = A ⊗ (B ⊗ C)."""
        S = 5
        torch.manual_seed(42)
        A = torch.randn(1, S, S, dtype=torch.float64)
        B = torch.randn(1, S, S, dtype=torch.float64)
        C = torch.randn(1, S, S, dtype=torch.float64)

        left = _log_semiring_bmm(_log_semiring_bmm(A, B), C)
        right = _log_semiring_bmm(A, _log_semiring_bmm(B, C))

        assert torch.allclose(left, right, atol=1e-8), "Log-semiring matmul is not associative"

    def test_transition_mask_structure(self) -> None:
        """Transition mask has correct sparsity for simple targets."""
        targets = torch.tensor([5, 10, 5])  # has a repeat in extended form
        S_ext = 7  # 2*3+1
        targets_ext = torch.full((S_ext,), 0, dtype=torch.long)
        targets_ext[1::2] = targets
        skip_invalid = torch.zeros(S_ext, dtype=torch.bool)
        if S_ext >= 3:
            skip_invalid[2:] = targets_ext[2:] == targets_ext[:-2]

        mask = _build_transition_mask(skip_invalid, S_ext, "cpu")

        # Diagonal should be all True (stay)
        for s in range(S_ext):
            assert mask[s, s].item(), f"Stay at ({s},{s}) should be True"

        # Sub-diagonal should be True for s >= 1 (advance)
        for s in range(1, S_ext):
            assert mask[s, s - 1].item(), f"Advance at ({s},{s-1}) should be True"

        # Check total nnz is reasonable (approximately 3 per row minus invalids)
        nnz = mask.sum().item()
        assert nnz <= 3 * S_ext  # at most 3 transitions per state

    def test_parallel_reduce_single_matrix(self) -> None:
        """Reduce of a single matrix returns that matrix."""
        S = 5
        torch.manual_seed(0)
        M = torch.randn(1, S, S, dtype=torch.float64)
        NEG_INF = _get_neg_inf(torch.float64)
        result = _parallel_reduce(M, NEG_INF)
        assert torch.allclose(result, M[0], atol=1e-10)


# ── CUDA tests ───────────────────────────────────────────────────────────────

class TestCUDA:

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_all_tensors_on_device(self) -> None:
        T, S = 20, 4
        device = "cuda"
        torch.manual_seed(0)
        logits = torch.randn(T, VOCAB_SIZE, dtype=torch.float32, device=device,
                             requires_grad=True)
        targets = torch.randint(1, VOCAB_SIZE, (S,), device=device)
        log_probs = F.log_softmax(logits, dim=-1)
        loss = DifferentiableCTCFast()(log_probs, targets, blank=BLANK)
        assert loss.device.type == "cuda"
        grad = torch.autograd.grad(loss, logits, create_graph=True)[0]
        assert grad.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_matches_cpu(self) -> None:
        T, S = 30, 6
        torch.manual_seed(0)
        logits_cpu = torch.randn(T, VOCAB_SIZE, dtype=torch.float64, requires_grad=True)
        targets = torch.randint(1, VOCAB_SIZE, (S,))

        lp_cpu = F.log_softmax(logits_cpu, dim=-1)
        loss_cpu = DifferentiableCTCFast()(lp_cpu, targets, blank=BLANK)

        logits_gpu = logits_cpu.detach().clone().cuda().requires_grad_(True)
        lp_gpu = F.log_softmax(logits_gpu, dim=-1)
        loss_gpu = DifferentiableCTCFast()(lp_gpu, targets.cuda(), blank=BLANK)

        rel_err = abs(loss_cpu.item() - loss_gpu.item()) / (abs(loss_cpu.item()) + 1e-12)
        assert rel_err < 1e-6, f"CPU/GPU mismatch: {rel_err:.2e}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_graph_freed_after_backward(self) -> None:
        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated()

        for _ in range(5):
            logits = torch.randn(49, VOCAB_SIZE, dtype=torch.float32, device="cuda",
                                 requires_grad=True)
            targets = torch.randint(1, VOCAB_SIZE, (5,), device="cuda")
            log_probs = F.log_softmax(logits, dim=-1)
            loss = DifferentiableCTCFast()(log_probs, targets, blank=BLANK)
            grad = torch.autograd.grad(loss, logits, create_graph=True)[0]
            g2 = torch.autograd.grad(grad.norm(), logits)[0]
            del loss, grad, g2, logits, log_probs

        torch.cuda.empty_cache()
        mem_after = torch.cuda.memory_allocated()

        leak = mem_after - mem_before
        assert leak < 10 * 1024 * 1024, f"Possible memory leak: {leak / 1024 / 1024:.1f} MB"


# ── Sequence length edge cases ───────────────────────────────────────────────

class TestEdgeCases:

    def test_too_few_frames_raises(self) -> None:
        targets = torch.tensor([11, 5, 15, 15, 8])
        logits = torch.randn(5, VOCAB_SIZE, dtype=torch.float64, requires_grad=True)
        with pytest.raises(ValueError, match="Sequence too short"):
            DifferentiableCTCFast()(F.log_softmax(logits, dim=-1), targets, blank=BLANK)

    def test_minimum_frames_exact(self) -> None:
        targets = torch.tensor([11, 5, 15, 15, 8])
        T = 6  # S=5, 1 repeat → min_T=6
        torch.manual_seed(0)
        logits = torch.randn(T, VOCAB_SIZE, dtype=torch.float64, requires_grad=True)
        loss = DifferentiableCTCFast()(F.log_softmax(logits, dim=-1), targets, blank=BLANK)
        assert torch.isfinite(loss)

    def test_T_equals_2(self) -> None:
        """T=2, S=1: one transition matrix, no scan needed."""
        targets = torch.tensor([5])
        torch.manual_seed(0)
        logits = torch.randn(2, VOCAB_SIZE, dtype=torch.float64, requires_grad=True)
        loss = DifferentiableCTCFast()(F.log_softmax(logits, dim=-1), targets, blank=BLANK)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(logits.grad).all()

    def test_long_sequence(self) -> None:
        """T=200 (typical wav2vec2 output for 4s audio)."""
        T, S = 200, 10
        torch.manual_seed(0)
        logits = torch.randn(T, VOCAB_SIZE, dtype=torch.float32, requires_grad=True)
        targets = torch.randint(1, VOCAB_SIZE, (S,))
        log_probs = F.log_softmax(logits, dim=-1)
        loss = DifferentiableCTCFast()(log_probs, targets, blank=BLANK)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(logits.grad).all()
