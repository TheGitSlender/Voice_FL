"""
Verification suite for the custom differentiable CTC loss.

Tests 1-4: CPU, float64, numerical accuracy.
Test 5: GPU (or CPU fallback), full MAML inner loop integration.

Run: python scripts/verify_differentiable_ctc.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from ctc.differentiable_ctc import DifferentiableCTC, ctc_loss_differentiable

def _make_test_data(
    T: int, C: int, S: int, dtype: torch.dtype = torch.float64, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create random logits and targets for testing."""
    torch.manual_seed(42)
    logits = torch.randn(T, C, dtype=dtype, device=device, requires_grad=True)
                                                
    targets = torch.randint(1, C, (S,), device=device)
    return logits, targets

def test_0_gradcheck() -> None:
    """Test 0: torch.autograd.gradcheck — provably correct double-backward."""
    print("Test 0 — gradcheck (float64, eps=1e-5)...")

    T, C, S = 8, 6, 3
    torch.manual_seed(42)
    logits = torch.randn(T, C, dtype=torch.float64, requires_grad=True)
    targets = torch.randint(1, C, (S,))

    def ctc_fn(lp: torch.Tensor) -> torch.Tensor:
        log_probs = torch.nn.functional.log_softmax(lp, dim=-1)
        ctc = DifferentiableCTC()
        return ctc(log_probs, targets, blank=0)

    passed = torch.autograd.gradcheck(ctc_fn, (logits,), eps=1e-5, atol=1e-4)
    assert passed, "gradcheck failed"

    passed2 = torch.autograd.gradgradcheck(ctc_fn, (logits,), eps=1e-5, atol=1e-4)
    assert passed2, "gradgradcheck failed"

    print("  gradcheck:     PASSED")
    print("  gradgradcheck: PASSED\n")

def test_1_first_order_matches_pytorch() -> None:
    """Test 1: Loss and first-order gradients match nn.CTCLoss."""
    print("Test 1 — First-order gradient matches PyTorch CTC...")

    T, C, S = 20, 32, 5
    dtype = torch.float64

    logits_custom, targets = _make_test_data(T, C, S, dtype=dtype)
    log_probs_custom = torch.nn.functional.log_softmax(logits_custom, dim=-1)
    ctc = DifferentiableCTC()
    loss_custom = ctc(log_probs_custom, targets, blank=0)
    loss_custom.backward()
    grad_custom = logits_custom.grad.clone()

    logits_native = logits_custom.detach().clone().requires_grad_(True)
    log_probs_native = torch.nn.functional.log_softmax(logits_native, dim=-1)
                                        
    log_probs_native_ctc = log_probs_native.unsqueeze(1)             
    input_lengths = torch.tensor([T])
    target_lengths = torch.tensor([S])
    native_ctc = nn.CTCLoss(blank=0, reduction="none")
    loss_native = native_ctc(
        log_probs_native_ctc, targets.unsqueeze(0), input_lengths, target_lengths
    )
    loss_native = loss_native.squeeze()
    loss_native.backward()
    grad_native = logits_native.grad.clone()

    loss_rel_err = abs(loss_custom.item() - loss_native.item()) / (
        abs(loss_native.item()) + 1e-12
    )
    grad_max_rel_err = (
        (grad_custom - grad_native).abs() / (grad_native.abs() + 1e-10)
    ).max().item()

    print(f"  Custom loss:  {loss_custom.item():.6f}")
    print(f"  Native loss:  {loss_native.item():.6f}")
    print(f"  Loss rel err: {loss_rel_err:.2e}")
    print(f"  Grad max rel err: {grad_max_rel_err:.2e}")

    assert loss_rel_err < 1e-4, f"Loss mismatch: rel err {loss_rel_err:.2e}"
    assert grad_max_rel_err < 1e-3, f"Grad mismatch: max rel err {grad_max_rel_err:.2e}"
    print("  PASSED\n")

def test_2_double_backward_succeeds() -> None:
    """Test 2: Double-backward succeeds without RuntimeError."""
    print("Test 2 — Double-backward succeeds...")

    T, C, S = 15, 16, 4
    logits, targets = _make_test_data(T, C, S, dtype=torch.float64)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    ctc = DifferentiableCTC()
    loss = ctc(log_probs, targets, blank=0)

    grad = torch.autograd.grad(loss, logits, create_graph=True)[0]

    grad_norm = grad.norm()
    grad_of_grad = torch.autograd.grad(grad_norm, logits)[0]

    assert torch.isfinite(grad_of_grad).all(), "Second-order gradient contains non-finite values"
    assert grad_of_grad.abs().sum().item() > 0, "Second-order gradient is all zeros"

    print(f"  grad_of_grad norm: {grad_of_grad.norm().item():.6f}")
    print("  PASSED\n")

def test_3_second_order_differs_from_first() -> None:
    """Test 3: Second-order gradient encodes different information than first-order."""
    print("Test 3 — Second-order differs from first-order...")

    T, C, S = 15, 16, 4
    logits, targets = _make_test_data(T, C, S, dtype=torch.float64)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    ctc = DifferentiableCTC()
    loss = ctc(log_probs, targets, blank=0)

    grad_first = torch.autograd.grad(loss, logits, create_graph=True)[0]

    grad_norm = grad_first.norm()
    grad_second = torch.autograd.grad(grad_norm, logits)[0]

    cos_sim = torch.nn.functional.cosine_similarity(
        grad_first.detach().flatten().unsqueeze(0),
        grad_second.detach().flatten().unsqueeze(0),
    ).item()

    print(f"  Cosine similarity: {cos_sim:.4f}")
    assert cos_sim < 0.99, f"Cosine sim too high ({cos_sim:.4f}) — second-order not propagating"
    print("  PASSED\n")

def test_4_hessian_matches_finite_difference() -> None:
    """Test 4: Autograd Hessian diagonal matches finite difference (definitive correctness check)."""
    print("Test 4 — Hessian matches finite difference within 5%...")

    T, C, S = 10, 8, 3
    eps = 1e-3
    n_check = 5                                              

    logits_base, targets = _make_test_data(T, C, S, dtype=torch.float64)

    logits_ag = logits_base.detach().clone().requires_grad_(True)
    log_probs_ag = torch.nn.functional.log_softmax(logits_ag, dim=-1)
    ctc = DifferentiableCTC()
    loss_ag = ctc(log_probs_ag, targets, blank=0)

    grad_ag = torch.autograd.grad(loss_ag, logits_ag, create_graph=True)[0]
    grad_flat = grad_ag.flatten()

    hessian_diag_ag = []
    for i in range(n_check):
        g2 = torch.autograd.grad(grad_flat[i], logits_ag, retain_graph=True)[0]
        hessian_diag_ag.append(g2.flatten()[i].item())

    hessian_diag_fd = []
    logits_flat = logits_base.detach().flatten()

    for i in range(n_check):
                        
        logits_plus = logits_flat.clone()
        logits_plus[i] += eps
        lp_plus = logits_plus.reshape(T, C).requires_grad_(True)
        log_probs_plus = torch.nn.functional.log_softmax(lp_plus, dim=-1)
        loss_plus = ctc(log_probs_plus, targets, blank=0)
        grad_plus = torch.autograd.grad(loss_plus, lp_plus)[0].flatten()[i].item()

        logits_minus = logits_flat.clone()
        logits_minus[i] -= eps
        lp_minus = logits_minus.reshape(T, C).requires_grad_(True)
        log_probs_minus = torch.nn.functional.log_softmax(lp_minus, dim=-1)
        loss_minus = ctc(log_probs_minus, targets, blank=0)
        grad_minus = torch.autograd.grad(loss_minus, lp_minus)[0].flatten()[i].item()

        hessian_diag_fd.append((grad_plus - grad_minus) / (2 * eps))

    print(f"  {'Element':<8} {'Autograd':>12} {'Finite Diff':>12} {'Rel Err':>10}")
    all_ok = True
    for i in range(n_check):
        ag_val = hessian_diag_ag[i]
        fd_val = hessian_diag_fd[i]
        denom = abs(fd_val) if abs(fd_val) > 1e-10 else 1e-10
        rel_err = abs(ag_val - fd_val) / denom
        status = "OK" if rel_err < 0.05 else "FAIL"
        if rel_err >= 0.05:
            all_ok = False
        print(f"  {i:<8} {ag_val:>12.6f} {fd_val:>12.6f} {rel_err:>9.2e} {status}")

    assert all_ok, "Hessian diagonal mismatch > 5% — algorithm is incorrect"
    print("  PASSED\n")

def test_5_full_maml_integration() -> None:
    """Test 5: Full MAML inner loop with higher and custom CTC."""
    print("Test 5 — Full MAML inner loop integration...")

    import higher
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    print(f"  Device: {device}")

    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    model = model.to(device=device, dtype=dtype)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    inner_params = list(model.lm_head.parameters())
    inner_opt = torch.optim.SGD(inner_params, lr=1e-4)

    torch.manual_seed(0)
    dummy_audio = torch.randn(1, 16000, device=device, dtype=dtype)

    dummy_labels = torch.tensor([[2, 5, 10, 3]], device=device)                       

    k_steps = 3

    with higher.innerloop_ctx(
        model,
        inner_opt,
        copy_initial_weights=False,
        track_higher_grads=True,
        override={"lr": [1e-4]},
    ) as (fmodel, diffopt):
        for _step in range(k_steps):
            out = fmodel(input_values=dummy_audio)
            logits = out.logits                    
            T_frames = logits.shape[1]

            input_lengths = torch.tensor([T_frames], device=device)
            target_lengths = torch.tensor([4], device=device)

            loss = ctc_loss_differentiable(
                logits, dummy_labels, input_lengths, target_lengths, blank=0
            )
            diffopt.step(loss)

        out_q = fmodel(input_values=dummy_audio)
        logits_q = out_q.logits
        T_q = logits_q.shape[1]
        query_loss = ctc_loss_differentiable(
            logits_q,
            dummy_labels,
            torch.tensor([T_q], device=device),
            torch.tensor([4], device=device),
            blank=0,
        )

    encoder_params = list(model.wav2vec2.parameters())
    meta_grads = torch.autograd.grad(
        query_loss, encoder_params, allow_unused=True
    )

    n_non_none = sum(1 for g in meta_grads if g is not None)
    n_finite = sum(1 for g in meta_grads if g is not None and torch.isfinite(g).all())
    n_nonzero = sum(1 for g in meta_grads if g is not None and g.abs().sum().item() > 0)

    print(f"  Encoder params: {len(encoder_params)}")
    print(f"  Meta-grads non-None: {n_non_none}")
    print(f"  Meta-grads finite:   {n_finite}")
    print(f"  Meta-grads nonzero:  {n_nonzero}")

    assert n_non_none > 0, "All meta-gradients are None"
    assert n_finite == n_non_none, "Some meta-gradients are non-finite"
    assert n_nonzero > 0, "All meta-gradients are zero"
    print("  PASSED\n")

def main() -> None:
    print("=" * 60)
    print("Differentiable CTC Verification Suite")
    print("=" * 60 + "\n")

    passed = 0
    failed = 0

    for test_fn in [
        test_0_gradcheck,
        test_1_first_order_matches_pytorch,
        test_2_double_backward_succeeds,
        test_3_second_order_differs_from_first,
        test_4_hessian_matches_finite_difference,
        test_5_full_maml_integration,
    ]:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}\n")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
