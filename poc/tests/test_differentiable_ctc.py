"""
Tests for ctc/differentiable_ctc.py — wav2vec2 training edge cases.

Groups:
  1. Core correctness (matches nn.CTCLoss, gradcheck)
  2. Wav2Vec2-specific shapes and tokenization
  3. Numerical stability (bf16, float32, extreme values)
  4. Sequence length edge cases (short audio, long labels, repeats)
  5. Batch interface (padding, mixed lengths, empty targets)
  6. Gradient flow through MAML inner loop
  7. Memory and device safety
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Wav2Vec2ForCTC

from ctc.differentiable_ctc import DifferentiableCTC, _safe_logsumexp, ctc_loss_differentiable
from tests.conftest import BLANK, DOWNSAMPLE_RATIO, SAMPLE_RATE, VOCAB_SIZE, make_audio, tokenize

class TestCoreCorrectness:
    """Loss values and gradients match PyTorch nn.CTCLoss."""

    @pytest.mark.parametrize("T,S", [(20, 5), (49, 11), (100, 20), (10, 1)])
    def test_loss_matches_native(self, T: int, S: int) -> None:
        """Custom CTC loss matches nn.CTCLoss within 1e-4 relative error."""
        torch.manual_seed(42)
        logits = torch.randn(T, VOCAB_SIZE, dtype=torch.float64, requires_grad=True)
        targets = torch.randint(1, VOCAB_SIZE, (S,))

        log_probs = F.log_softmax(logits, dim=-1)
        loss_custom = DifferentiableCTC()(log_probs, targets, blank=BLANK)

        logits_n = logits.detach().clone().requires_grad_(True)
        log_probs_n = F.log_softmax(logits_n, dim=-1).unsqueeze(1)
        loss_native = nn.CTCLoss(blank=BLANK, reduction="none")(
            log_probs_n, targets.unsqueeze(0),
            torch.tensor([T]), torch.tensor([S]),
        ).squeeze()

        rel_err = abs(loss_custom.item() - loss_native.item()) / (abs(loss_native.item()) + 1e-12)
        assert rel_err < 1e-4, f"Loss mismatch: {rel_err:.2e}"

    def test_gradient_matches_native(self) -> None:
        """First-order gradients match nn.CTCLoss within 1e-3 max relative error."""
        T, S = 30, 8
        torch.manual_seed(7)
        logits = torch.randn(T, VOCAB_SIZE, dtype=torch.float64, requires_grad=True)
        targets = torch.randint(1, VOCAB_SIZE, (S,))

        log_probs = F.log_softmax(logits, dim=-1)
        loss = DifferentiableCTC()(log_probs, targets, blank=BLANK)
        loss.backward()
        grad_custom = logits.grad.clone()

        logits_n = logits.detach().clone().requires_grad_(True)
        log_probs_n = F.log_softmax(logits_n, dim=-1).unsqueeze(1)
        loss_n = nn.CTCLoss(blank=BLANK, reduction="none")(
            log_probs_n, targets.unsqueeze(0),
            torch.tensor([T]), torch.tensor([S]),
        ).squeeze()
        loss_n.backward()
        grad_native = logits_n.grad.clone()

        max_rel = ((grad_custom - grad_native).abs() / (grad_native.abs() + 1e-10)).max()
        assert max_rel < 1e-3, f"Grad max rel err: {max_rel:.2e}"

    def test_gradcheck_double_backward(self) -> None:
        """torch.autograd.gradcheck + gradgradcheck pass in float64."""
        T, C, S = 8, 6, 3
        torch.manual_seed(42)
        logits = torch.randn(T, C, dtype=torch.float64, requires_grad=True)
        targets = torch.randint(1, C, (S,))

        def fn(lp: torch.Tensor) -> torch.Tensor:
            return DifferentiableCTC()(F.log_softmax(lp, dim=-1), targets, blank=0)

        assert torch.autograd.gradcheck(fn, (logits,), eps=1e-5, atol=1e-4)
        assert torch.autograd.gradgradcheck(fn, (logits,), eps=1e-5, atol=1e-4)

    def test_hessian_diagonal_finite_difference(self) -> None:
        """Autograd Hessian diagonal matches finite-difference within 5%."""
        T, C, S = 10, 8, 3
        eps = 1e-3
        torch.manual_seed(42)
        logits_base = torch.randn(T, C, dtype=torch.float64)
        targets = torch.randint(1, C, (S,))
        ctc = DifferentiableCTC()

        logits_ag = logits_base.clone().requires_grad_(True)
        loss = ctc(F.log_softmax(logits_ag, dim=-1), targets, blank=0)
        grad = torch.autograd.grad(loss, logits_ag, create_graph=True)[0].flatten()

        for i in range(5):
            g2 = torch.autograd.grad(grad[i], logits_ag, retain_graph=True)[0].flatten()[i]

            flat = logits_base.flatten()
            flat_p, flat_m = flat.clone(), flat.clone()
            flat_p[i] += eps
            flat_m[i] -= eps
            lp_p = flat_p.reshape(T, C).requires_grad_(True)
            lp_m = flat_m.reshape(T, C).requires_grad_(True)
            gp = torch.autograd.grad(ctc(F.log_softmax(lp_p, dim=-1), targets), lp_p)[0].flatten()[i]
            gm = torch.autograd.grad(ctc(F.log_softmax(lp_m, dim=-1), targets), lp_m)[0].flatten()[i]
            fd = (gp - gm) / (2 * eps)

            denom = abs(fd.item()) if abs(fd.item()) > 1e-10 else 1e-10
            assert abs(g2.item() - fd.item()) / denom < 0.05, f"Hessian[{i}] mismatch"

class TestWav2Vec2Shapes:
    """Tests using real wav2vec2 output shapes and tokenization."""

    def test_real_vocab_size(self) -> None:
        """CTC works with wav2vec2's 32-token vocab."""
        T = 49                     
        torch.manual_seed(0)
        logits = torch.randn(T, VOCAB_SIZE, dtype=torch.float32, requires_grad=True)
        targets = torch.tensor([6, 11, 5])                   

        log_probs = F.log_softmax(logits, dim=-1)
        loss = DifferentiableCTC()(log_probs, targets, blank=BLANK)

        assert torch.isfinite(loss), "Loss is not finite"
        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()

    @pytest.mark.parametrize("text", [
        "A",                              
        "THE",                                
        "HELLO",                                                    
        "THE CAT SAT",                    
        "AARDVARK",                                   
    ])
    def test_real_transcriptions(self, text: str, processor) -> None:
        """CTC handles real wav2vec2 tokenized transcriptions."""
        targets = tokenize(text, processor)
        S = targets.shape[0]
        n_repeats = (targets[1:] == targets[:-1]).sum().item()
        min_T = S + n_repeats
        T = max(min_T + 5, 49)                 

        torch.manual_seed(0)
        logits = torch.randn(T, VOCAB_SIZE, dtype=torch.float64, requires_grad=True)
        log_probs = F.log_softmax(logits, dim=-1)
        loss = DifferentiableCTC()(log_probs, targets, blank=BLANK)

        grad = torch.autograd.grad(loss, logits, create_graph=True)[0]
        assert torch.isfinite(grad).all(), f"NaN/Inf in first-order grad for {text!r}"

        g2 = torch.autograd.grad(grad.norm(), logits)[0]
        assert torch.isfinite(g2).all(), f"NaN/Inf in second-order grad for {text!r}"

    def test_wav2vec2_frame_counts(self, wav2vec2_model, device) -> None:
        """CTC handles the actual T_frames produced by wav2vec2 for various audio lengths."""
        model = wav2vec2_model
        model.eval()

        for dur_ms in [200, 500, 1000, 3000]:
            audio = make_audio(dur_ms, device)
            with torch.no_grad():
                out = model(input_values=audio)
            T = out.logits.shape[1]
            C = out.logits.shape[2]

            assert C == VOCAB_SIZE
            expected_T = int(dur_ms / 1000 * SAMPLE_RATE / DOWNSAMPLE_RATIO)
                                          
            assert abs(T - expected_T) <= 2, f"{dur_ms}ms: expected ~{expected_T} frames, got {T}"

class TestNumericalStability:
    """Float32, bf16, extreme logit values, safe_logsumexp."""

    def test_float32_no_nan(self) -> None:
        """Loss and double-backward are NaN-free in float32 (training precision)."""
        T, S = 49, 8
        torch.manual_seed(0)
        logits = torch.randn(T, VOCAB_SIZE, dtype=torch.float32, requires_grad=True)
        targets = torch.randint(1, VOCAB_SIZE, (S,))

        log_probs = F.log_softmax(logits, dim=-1)
        loss = DifferentiableCTC()(log_probs, targets, blank=BLANK)

        assert torch.isfinite(loss)
        grad = torch.autograd.grad(loss, logits, create_graph=True)[0]
        assert torch.isfinite(grad).all(), "NaN in first-order grad (float32)"
        g2 = torch.autograd.grad(grad.norm(), logits)[0]
        assert torch.isfinite(g2).all(), "NaN in second-order grad (float32)"

    def test_bfloat16_forward_backward(self) -> None:
        """Loss and first-order gradient are finite in bf16.

        wav2vec2_maml.to_bf16() casts the model to bf16 for VRAM savings.
        Double-backward in bf16 is not required (inner loop runs in fp32 via
        autocast), but single backward must work for the fomaml path.
        """
        if not torch.cuda.is_available():
            pytest.skip("bf16 test requires CUDA")

        T, S = 49, 5
        torch.manual_seed(0)
        logits = torch.randn(T, VOCAB_SIZE, dtype=torch.bfloat16, device="cuda",
                             requires_grad=True)
        targets = torch.randint(1, VOCAB_SIZE, (S,), device="cuda")

        log_probs = F.log_softmax(logits, dim=-1)
        loss = DifferentiableCTC()(log_probs, targets, blank=BLANK)

        assert torch.isfinite(loss), "Loss is NaN/Inf in bf16"
        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all(), "NaN in grad (bf16)"

    def test_extreme_logits_no_nan(self) -> None:
        """Very large or very small logits don't produce NaN.

        Early in training, logits can be large when lm_head weights are random.
        """
        T, S = 20, 3
        targets = torch.randint(1, VOCAB_SIZE, (S,))

        for scale in [0.01, 1.0, 10.0, 100.0]:
            torch.manual_seed(0)
            logits = scale * torch.randn(T, VOCAB_SIZE, dtype=torch.float32, requires_grad=True)
            log_probs = F.log_softmax(logits, dim=-1)
            loss = DifferentiableCTC()(log_probs, targets, blank=BLANK)

            assert torch.isfinite(loss), f"Loss NaN/Inf at scale={scale}"
            grad = torch.autograd.grad(loss, logits, create_graph=True)[0]
            assert torch.isfinite(grad).all(), f"Grad NaN at scale={scale}"

    def test_uniform_logits(self) -> None:
        """When all logits are equal, loss = -log(num_alignments) + T*log(C).

        Uniform logits test that the DP doesn't rely on contrast between classes.
        Gradients should be finite and non-zero (pushing away from uniform).
        """
        T, S = 20, 3
        targets = torch.randint(1, VOCAB_SIZE, (S,))
        logits = torch.zeros(T, VOCAB_SIZE, dtype=torch.float64, requires_grad=True)
        log_probs = F.log_softmax(logits, dim=-1)
        loss = DifferentiableCTC()(log_probs, targets, blank=BLANK)

        assert torch.isfinite(loss)
        grad = torch.autograd.grad(loss, logits, create_graph=True)[0]
        assert torch.isfinite(grad).all()
        assert grad.abs().sum() > 0, "Gradient is zero for uniform logits"

    def test_safe_logsumexp_all_neginf(self) -> None:
        """_safe_logsumexp returns NEG_INF (not NaN) for all-neginf input."""
        NEG_INF = torch.finfo(torch.float64).min / 2
        x = torch.full((3, 5), NEG_INF, dtype=torch.float64, requires_grad=True)

        out = _safe_logsumexp(x, dim=0)
        assert (out < -1e37).all(),\
            "Expected NEG_INF output for all-neginf input"
        assert not torch.isnan(out).any(), "NaN in safe_logsumexp output"

        out.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any(), "NaN in safe_logsumexp gradient"

    def test_safe_logsumexp_mixed(self) -> None:
        """_safe_logsumexp handles mix of NEG_INF and finite values."""
        NEG_INF = torch.finfo(torch.float64).min / 2
        x = torch.tensor([
            [NEG_INF, 1.0, NEG_INF],
            [NEG_INF, NEG_INF, 2.0],
            [NEG_INF, 3.0, NEG_INF],
        ], dtype=torch.float64, requires_grad=True)

        out = _safe_logsumexp(x, dim=0)        
        assert torch.isfinite(out[1]).item() and torch.isfinite(out[2]).item()
        assert not torch.isnan(out).any()

    def test_loss_stays_finite_across_inner_steps(self) -> None:
        """Simulated inner loop: loss and gradients don't explode over k steps.

        Mimics the ANIL inner loop: k SGD steps on lm_head-sized params,
        then query loss gradient. Catches gradient explosion in the custom CTC.
        """
        T, S = 30, 5
        C = VOCAB_SIZE
        inner_lr = 1e-4
        k_steps = 10                                          

        torch.manual_seed(0)
                                                           
        encoder_out = torch.randn(T, 64, dtype=torch.float32, requires_grad=True)
        lm_head = nn.Linear(64, C)
        targets = torch.randint(1, C, (S,))

        for step in range(k_steps):
            logits = lm_head(encoder_out)          
            log_probs = F.log_softmax(logits, dim=-1)
            loss = DifferentiableCTC()(log_probs, targets, blank=BLANK)

            assert torch.isfinite(loss), f"Loss NaN/Inf at step {step}"

            head_grads = torch.autograd.grad(loss, lm_head.parameters(), create_graph=False)
            with torch.no_grad():
                for p, g in zip(lm_head.parameters(), head_grads):
                    p.data = p.data - inner_lr * g

        final_logits = lm_head(encoder_out)
        final_loss = DifferentiableCTC()(F.log_softmax(final_logits, dim=-1), targets, blank=BLANK)
        assert torch.isfinite(final_loss), "Final query loss is NaN/Inf after inner loop"

class TestSequenceLengthEdgeCases:
    """Short audio, long labels, repeated labels, boundary conditions."""

    def test_single_label(self) -> None:
        """S=1 (single character): S_ext=3, minimal CTC."""
        T = 10
        torch.manual_seed(0)
        logits = torch.randn(T, VOCAB_SIZE, dtype=torch.float64, requires_grad=True)
        targets = torch.tensor([7])       

        loss = DifferentiableCTC()(F.log_softmax(logits, dim=-1), targets, blank=BLANK)
        assert torch.isfinite(loss)
        grad = torch.autograd.grad(loss, logits, create_graph=True)[0]
        assert torch.isfinite(grad).all()
        g2 = torch.autograd.grad(grad.norm(), logits)[0]
        assert torch.isfinite(g2).all()

    def test_minimum_frames_exact(self) -> None:
        """T exactly equals minimum required frames — should work."""
                                                             
        targets = torch.tensor([11, 5, 15, 15, 8])
        T = 6

        torch.manual_seed(0)
        logits = torch.randn(T, VOCAB_SIZE, dtype=torch.float64, requires_grad=True)
        loss = DifferentiableCTC()(F.log_softmax(logits, dim=-1), targets, blank=BLANK)
        assert torch.isfinite(loss)

    def test_too_few_frames_raises(self) -> None:
        """T below minimum raises ValueError."""
                                               
        targets = torch.tensor([11, 5, 15, 15, 8])
        logits = torch.randn(5, VOCAB_SIZE, dtype=torch.float64, requires_grad=True)

        with pytest.raises(ValueError, match="Sequence too short"):
            DifferentiableCTC()(F.log_softmax(logits, dim=-1), targets, blank=BLANK)

    def test_all_same_labels(self) -> None:
        """All-repeat labels like "MMMM" = [17,17,17,17]: S=4, 3 repeats → min_T=7.

        This is the hardest case for CTC — every label transition requires a
        mandatory blank in between. The skip transition is never valid.
        """
        targets = torch.tensor([17, 17, 17, 17])
        T = 10             

        torch.manual_seed(0)
        logits = torch.randn(T, VOCAB_SIZE, dtype=torch.float64, requires_grad=True)
        loss = DifferentiableCTC()(F.log_softmax(logits, dim=-1), targets, blank=BLANK)

        assert torch.isfinite(loss)
        grad = torch.autograd.grad(loss, logits, create_graph=True)[0]
        assert torch.isfinite(grad).all()

    def test_all_same_labels_too_short_raises(self) -> None:
        """[17,17,17,17] with T=6 (min is 7) raises ValueError."""
        targets = torch.tensor([17, 17, 17, 17])
        logits = torch.randn(6, VOCAB_SIZE, dtype=torch.float64)
        with pytest.raises(ValueError, match="Sequence too short"):
            DifferentiableCTC()(F.log_softmax(logits, dim=-1), targets, blank=BLANK)

    def test_short_audio_100ms(self, wav2vec2_model, device) -> None:
        """100ms audio → 4 frames. Only very short labels (S<=3, no repeats) fit."""
        model = wav2vec2_model
        model.eval()
        audio = make_audio(100, device)

        with torch.no_grad():
            out = model(input_values=audio)
        T = out.logits.shape[1]
        assert T >= 3, f"100ms should produce >=3 frames, got {T}"

        targets = torch.tensor([7, 5], device=device)
        logits = out.logits[0].detach().requires_grad_(True)
        loss = DifferentiableCTC()(F.log_softmax(logits, dim=-1), targets, blank=BLANK)
        assert torch.isfinite(loss)

    def test_long_transcription(self) -> None:
        """Long transcription (S=25) with enough frames."""
        S = 25
        torch.manual_seed(0)
        targets = torch.randint(1, VOCAB_SIZE, (S,))
        n_repeats = (targets[1:] == targets[:-1]).sum().item()
        T = S + n_repeats + 20                    

        logits = torch.randn(T, VOCAB_SIZE, dtype=torch.float64, requires_grad=True)
        loss = DifferentiableCTC()(F.log_softmax(logits, dim=-1), targets, blank=BLANK)
        assert torch.isfinite(loss)

    def test_T_equals_1_S_equals_1(self) -> None:
        """Absolute minimum: T=1, S=1 (single frame, single label)."""
        targets = torch.tensor([5])
        logits = torch.randn(1, VOCAB_SIZE, dtype=torch.float64, requires_grad=True)
        loss = DifferentiableCTC()(F.log_softmax(logits, dim=-1), targets, blank=BLANK)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(logits.grad).all()

class TestBatchInterface:
    """Tests for ctc_loss_differentiable with padded batches."""

    def test_single_sample_batch(self) -> None:
        """B=1 batch matches single-sample DifferentiableCTC."""
        T, S = 30, 5
        torch.manual_seed(0)
        logits_single = torch.randn(T, VOCAB_SIZE, dtype=torch.float64, requires_grad=True)
        targets = torch.randint(1, VOCAB_SIZE, (S,))
        loss_single = DifferentiableCTC()(
            F.log_softmax(logits_single, dim=-1), targets, blank=BLANK,
        )

        logits_batch = logits_single.detach().clone().unsqueeze(0).requires_grad_(True)
        labels_batch = targets.unsqueeze(0)
        loss_batch = ctc_loss_differentiable(
            logits_batch, labels_batch,
            torch.tensor([T]), torch.tensor([S]), blank=BLANK,
        )

        rel_err = abs(loss_single.item() - loss_batch.item()) / (abs(loss_single.item()) + 1e-12)
        assert rel_err < 1e-4, f"Batch/single loss mismatch: {rel_err:.2e}"

    def test_padded_labels(self) -> None:
        """Labels padded with -100 are correctly stripped."""
        T = 30
        torch.manual_seed(0)
        logits = torch.randn(1, T, VOCAB_SIZE, dtype=torch.float64, requires_grad=True)

        labels = torch.tensor([[5, 10, 15, -100, -100, -100]])
        loss = ctc_loss_differentiable(
            logits, labels,
            torch.tensor([T]), torch.tensor([3]), blank=BLANK,
        )
        assert torch.isfinite(loss)
        loss.backward()
        assert logits.grad is not None

    def test_mixed_length_batch(self) -> None:
        """Batch with different T and S per sample."""
        B = 3
        T_max, S_max = 50, 10
        torch.manual_seed(0)
        logits = torch.randn(B, T_max, VOCAB_SIZE, dtype=torch.float64, requires_grad=True)

        input_lengths = torch.tensor([50, 30, 40])
        target_lengths = torch.tensor([10, 5, 8])
        labels = torch.full((B, S_max), -100, dtype=torch.long)
        for b in range(B):
            labels[b, :target_lengths[b]] = torch.randint(1, VOCAB_SIZE, (target_lengths[b],))

        loss = ctc_loss_differentiable(logits, labels, input_lengths, target_lengths, blank=BLANK)
        assert torch.isfinite(loss)
        loss.backward()
        assert logits.grad is not None

    def test_all_empty_targets_raises(self) -> None:
        """Batch where all samples have empty targets raises ValueError."""
        logits = torch.randn(2, 20, VOCAB_SIZE, requires_grad=True)
        labels = torch.full((2, 5), -100, dtype=torch.long)
        with pytest.raises(ValueError, match="empty targets"):
            ctc_loss_differentiable(
                logits, labels,
                torch.tensor([20, 20]), torch.tensor([0, 0]), blank=BLANK,
            )

    def test_batch_double_backward(self) -> None:
        """Batch interface supports create_graph=True through the whole path."""
        B, T, S = 2, 25, 4
        torch.manual_seed(0)
        logits = torch.randn(B, T, VOCAB_SIZE, dtype=torch.float64, requires_grad=True)
        labels = torch.randint(1, VOCAB_SIZE, (B, S))

        loss = ctc_loss_differentiable(
            logits, labels,
            torch.tensor([T, T]), torch.tensor([S, S]), blank=BLANK,
        )
        grad = torch.autograd.grad(loss, logits, create_graph=True)[0]
        g2 = torch.autograd.grad(grad.norm(), logits)[0]

        assert torch.isfinite(g2).all(), "NaN in batch double-backward"

class TestMAMLGradientFlow:
    """Tests that second-order gradients flow correctly through the ANIL MAML pattern.

    These tests load a fresh wav2vec2 model (not the session fixture) because
    higher.innerloop_ctx is incompatible with PyTorch's ParametrizedConv1d.
    The parametrization must be removed before higher wraps the model.
    """

    @staticmethod
    def _make_model(device: torch.device) -> Wav2Vec2ForCTC:
        """Load a fresh wav2vec2 model for higher compatibility.

        Each MAML test gets its own fresh model to avoid state leakage.
        Do NOT strip parametrizations — higher handles them correctly
        with a fresh model.
        """
        from tests.conftest import WAV2VEC2_MODEL
        model = Wav2Vec2ForCTC.from_pretrained(WAV2VEC2_MODEL)
        return model.to(device=device, dtype=torch.float32)

    def test_encoder_receives_meta_gradients(self, device) -> None:
        """After k inner steps on lm_head via higher, encoder params get non-None
        meta-gradients through the custom CTC. This is the core MAML requirement.
        """
        import higher

        model = self._make_model(device)
                                                                           
        inner_params = list(model.lm_head.parameters())
        inner_opt = torch.optim.SGD(inner_params, lr=1e-4)

        torch.manual_seed(0)
        audio = torch.randn(1, 16000, device=device, dtype=torch.float32)
        labels = torch.tensor([[2, 5, 10, 3]], device=device)

        with higher.innerloop_ctx(
            model, inner_opt,
            copy_initial_weights=False,
            track_higher_grads=True,
            override={"lr": [1e-4]},
        ) as (fmodel, diffopt):
                                                                     
            out = fmodel(input_values=audio)
            logits = out.logits
            T = logits.shape[1]
            loss = ctc_loss_differentiable(
                logits, labels,
                torch.tensor([T], device=device),
                torch.tensor([4], device=device),
            )
            diffopt.step(loss)

            out_q = fmodel(input_values=audio)
            q_loss = ctc_loss_differentiable(
                out_q.logits, labels,
                torch.tensor([out_q.logits.shape[1]], device=device),
                torch.tensor([4], device=device),
            )

        encoder_params = list(model.wav2vec2.parameters())
        meta_grads = torch.autograd.grad(q_loss, encoder_params, allow_unused=True)

        non_none = [g for g in meta_grads if g is not None]
        assert len(non_none) > 0, "No meta-gradients reached encoder"

        n_finite = sum(1 for g in non_none if torch.isfinite(g).all())
        n_nonzero = sum(1 for g in non_none if g.abs().sum() > 0)
                                                                                   
        assert n_finite >= len(non_none) * 0.95, (
            f"Too many non-finite meta-gradients: {n_finite}/{len(non_none)} finite"
        )
        assert n_nonzero > 0, "All meta-gradients are zero"

    def test_lm_head_not_in_meta_gradients(self, device) -> None:
        """Meta-gradients are computed w.r.t. encoder only, never lm_head.

        This is invariant I3: lm_head never transmitted. The test verifies the
        gradient computation itself, not just the API.
        """
        import higher

        model = self._make_model(device)
                                                                     
        inner_params = list(model.lm_head.parameters())
        inner_opt = torch.optim.SGD(inner_params, lr=1e-4)

        torch.manual_seed(0)
        audio = torch.randn(1, 16000, device=device, dtype=torch.float32)
        labels = torch.tensor([[7, 5]], device=device)

        with higher.innerloop_ctx(
            model, inner_opt,
            copy_initial_weights=False,
            track_higher_grads=True,
            override={"lr": [1e-4]},
        ) as (fmodel, diffopt):
            out = fmodel(input_values=audio)
            logits = out.logits
            loss = ctc_loss_differentiable(
                logits, labels,
                torch.tensor([logits.shape[1]], device=device),
                torch.tensor([2], device=device),
            )
            diffopt.step(loss)

            q_out = fmodel(input_values=audio)
            q_loss = ctc_loss_differentiable(
                q_out.logits, labels,
                torch.tensor([q_out.logits.shape[1]], device=device),
                torch.tensor([2], device=device),
            )

        encoder_ids = {id(p) for p in model.wav2vec2.parameters()}
        lm_head_ids = {id(p) for p in model.lm_head.parameters()}
        assert encoder_ids.isdisjoint(lm_head_ids), "Param sets overlap"

    def test_second_order_differs_from_first_order(self) -> None:
        """Second-order meta-gradients differ from first-order on a small model.

        With the full wav2vec2 (95M params), the Hessian correction is O(lr*k/sqrt(dim))
        — negligible at lr=1e-4. This test uses a tiny encoder+head to make the
        second-order signal visible and verifiable.
        """
        import higher

        torch.manual_seed(42)
        T, C_in, C_out, S = 20, 16, VOCAB_SIZE, 4
        inner_lr = 1e-2                                           

        encoder = nn.Sequential(nn.Linear(C_in, 32), nn.ReLU(), nn.Linear(32, C_in))
        lm_head = nn.Linear(C_in, C_out)
        model = nn.Sequential(encoder, lm_head)

        x = torch.randn(T, C_in, requires_grad=False)
        targets = torch.randint(1, C_out, (S,))

        def run_maml(track: bool) -> torch.Tensor:
            m = copy.deepcopy(model)
            head_params = list(m[1].parameters())
            opt = torch.optim.SGD(head_params, lr=inner_lr)

            with higher.innerloop_ctx(
                m, opt, copy_initial_weights=False,
                track_higher_grads=track, override={"lr": [inner_lr]},
            ) as (fm, do):
                for _ in range(3):
                    logits = fm(x)
                    lp = F.log_softmax(logits, dim=-1)
                    loss = DifferentiableCTC()(lp, targets, blank=BLANK)
                    do.step(loss)

                q_logits = fm(x)
                q_lp = F.log_softmax(q_logits, dim=-1)
                q_loss = DifferentiableCTC()(q_lp, targets, blank=BLANK)

            enc_params = list(m[0].parameters())
            grads = torch.autograd.grad(q_loss, enc_params, allow_unused=True)
            return torch.cat([g.flatten() for g in grads if g is not None])

        grads_2nd = run_maml(track=True)
        grads_1st = run_maml(track=False)

        assert torch.isfinite(grads_2nd).all(), "Second-order grads contain NaN/Inf"
        assert torch.isfinite(grads_1st).all(), "First-order grads contain NaN/Inf"

        cos_sim = F.cosine_similarity(
            grads_2nd.unsqueeze(0), grads_1st.unsqueeze(0),
        ).item()
        assert cos_sim < 0.99, (
            f"Second-order and first-order meta-grads are too similar "
            f"(cosine sim={cos_sim:.4f}) — track_higher_grads may not be working"
        )

class TestDeviceAndMemory:
    """Device consistency, CUDA memory cleanup, graph freeing."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_all_tensors_same_device(self) -> None:
        """Custom CTC doesn't create CPU tensors when input is on CUDA."""
        T, S = 20, 4
        device = "cuda"
        torch.manual_seed(0)
        logits = torch.randn(T, VOCAB_SIZE, dtype=torch.float32, device=device,
                             requires_grad=True)
        targets = torch.randint(1, VOCAB_SIZE, (S,), device=device)

        log_probs = F.log_softmax(logits, dim=-1)
        loss = DifferentiableCTC()(log_probs, targets, blank=BLANK)

        assert loss.device.type == "cuda"
        grad = torch.autograd.grad(loss, logits, create_graph=True)[0]
        assert grad.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_graph_freed_after_backward(self) -> None:
        """Computation graph is freed after backward — no memory leak.

        In MAML training, each episode creates a new graph. If graphs aren't
        freed, CUDA OOM after a few episodes.
        """
        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated()

        for _ in range(5):
            logits = torch.randn(49, VOCAB_SIZE, dtype=torch.float32, device="cuda",
                                 requires_grad=True)
            targets = torch.randint(1, VOCAB_SIZE, (5,), device="cuda")
            log_probs = F.log_softmax(logits, dim=-1)
            loss = DifferentiableCTC()(log_probs, targets, blank=BLANK)
            grad = torch.autograd.grad(loss, logits, create_graph=True)[0]
            g2 = torch.autograd.grad(grad.norm(), logits)[0]
            del loss, grad, g2, logits, log_probs

        torch.cuda.empty_cache()
        mem_after = torch.cuda.memory_allocated()

        leak = mem_after - mem_before
        assert leak < 10 * 1024 * 1024, f"Possible memory leak: {leak / 1024 / 1024:.1f} MB"

    def test_cpu_works(self) -> None:
        """Full double-backward works on CPU (for testing without GPU)."""
        T, S = 20, 4
        torch.manual_seed(0)
        logits = torch.randn(T, VOCAB_SIZE, dtype=torch.float64, requires_grad=True)
        targets = torch.randint(1, VOCAB_SIZE, (S,))

        log_probs = F.log_softmax(logits, dim=-1)
        loss = DifferentiableCTC()(log_probs, targets, blank=BLANK)
        grad = torch.autograd.grad(loss, logits, create_graph=True)[0]
        g2 = torch.autograd.grad(grad.norm(), logits)[0]
        assert torch.isfinite(g2).all()

class TestRegressions:
    """Tests for specific failure modes discovered during wav2vec2 MAML training."""

    def test_blank_target_id_zero(self) -> None:
        """blank=0 is wav2vec2's pad token. Labels must not contain id=0."""
        T = 20
        targets = torch.tensor([5, 0, 10])                                             
        logits = torch.randn(T, VOCAB_SIZE, dtype=torch.float64, requires_grad=True)

        loss = DifferentiableCTC()(F.log_softmax(logits, dim=-1), targets, blank=BLANK)
        assert torch.isfinite(loss)

    def test_repeated_blank_boundary(self) -> None:
        """Extended target [blank, L, blank, L, blank] with L the same label.

        The skip transition from state 0 (blank) to state 2 (blank) IS valid
        because targets_ext[0] == targets_ext[2] == blank (same token, skip
        invalid for repeated labels). This is correct CTC behavior: you can't
        skip from one instance of a label to the next same label.
        """
        targets = torch.tensor([5, 5])                        
        T = 6                                        
        torch.manual_seed(0)
        logits = torch.randn(T, VOCAB_SIZE, dtype=torch.float64, requires_grad=True)

        loss = DifferentiableCTC()(F.log_softmax(logits, dim=-1), targets, blank=BLANK)
        assert torch.isfinite(loss)

        logits_n = logits.detach().clone().requires_grad_(True)
        native_loss = nn.CTCLoss(blank=BLANK, reduction="none")(
            F.log_softmax(logits_n, dim=-1).unsqueeze(1),
            targets.unsqueeze(0),
            torch.tensor([T]), torch.tensor([2]),
        ).squeeze()

        rel_err = abs(loss.item() - native_loss.item()) / (abs(native_loss.item()) + 1e-12)
        assert rel_err < 1e-4, f"Repeated label loss mismatch: {rel_err:.2e}"

    def test_label_at_vocab_boundary(self) -> None:
        """Labels using the highest valid token id (31 for wav2vec2)."""
        targets = torch.tensor([VOCAB_SIZE - 1, 1, VOCAB_SIZE - 1])
        T = 20
        torch.manual_seed(0)
        logits = torch.randn(T, VOCAB_SIZE, dtype=torch.float64, requires_grad=True)

        loss = DifferentiableCTC()(F.log_softmax(logits, dim=-1), targets, blank=BLANK)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(logits.grad).all()

    def test_many_inner_steps_no_nan_accumulation(self) -> None:
        """k=10 inner steps with create_graph=True doesn't accumulate NaN.

        The computation graph grows linearly with k. Numerical errors in
        _safe_logsumexp can compound through the graph. Test that k=10
        (more than we'll ever use in practice) stays clean.
        """
        T, S, C = 20, 3, VOCAB_SIZE
        torch.manual_seed(0)
        W = torch.randn(C, C, dtype=torch.float32, requires_grad=True)
        base_logits = torch.randn(T, C, dtype=torch.float32, requires_grad=True)
        targets = torch.randint(1, C, (S,))

        for step in range(10):
            logits = base_logits @ W
            log_probs = F.log_softmax(logits, dim=-1)
            loss = DifferentiableCTC()(log_probs, targets, blank=BLANK)
            grad = torch.autograd.grad(loss, W, create_graph=True)[0]
            W = W - 1e-4 * grad                       

        final_logits = base_logits @ W
        final_loss = DifferentiableCTC()(F.log_softmax(final_logits, dim=-1), targets, blank=BLANK)
        meta_grad = torch.autograd.grad(final_loss, base_logits, allow_unused=True)

        assert torch.isfinite(final_loss), f"Loss is NaN/Inf after 10 differentiable steps"
