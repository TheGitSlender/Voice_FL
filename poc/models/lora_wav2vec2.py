"""
LoRA-wrapped Wav2Vec2ForCTC for FedLoRA-MAML.

FedLoRA-MAML achieves TRUE second-order MAML on CTC. Two conditions make this
possible:
  1. ctc/differentiable_ctc.py — CTC loss in pure PyTorch ops, supports
     create_graph=True. Captures both LoRA curvature and CTC alignment curvature.
  2. LoRA restricts the inner loop to ~320K parameters — feasible Hessian at k=5.

This is NOT ANIL. Both encoder (via LoRA) and lm_head adapt in the inner loop.
The frozen backbone provides universal acoustic representations. The LoRA adapters
capture speaker-specific acoustic patterns.

Why eager attention (attn_implementation="eager"):
  F.scaled_dot_product_attention's CPU backend calls
  aten::_scaled_dot_product_flash_attention_for_cpu, which has no registered
  second derivative. Eager mode uses explicit bmm-based attention that autograd
  can differentiate through twice.

Why layers 6–11 only:
  Layers 0–5 encode speaker-independent acoustic-phonetic features (formants,
  pitch structure). Layers 6–11 encode linguistic context and speaker-specific
  prosodic patterns. Adapting upper layers captures speaker variation without
  disrupting universal acoustic representations.

Why manual LoRA (not peft):
  peft's LoraModel wraps modules with hooks and module traversal that is
  incompatible with higher.innerloop_ctx. Manual LoRALinear keeps frozen weights
  as plain frozen parameters and LoRA matrices as standard nn.Parameters — higher
  can patch these cleanly.

Parameter groups:
  lm_head:          ~25K  trainable — local adaptation, also aggregated by FL
  LoRA A/B:        ~295K  trainable — aggregated by FL server
  frozen backbone: ~94.5M frozen   — never transmitted, never trained

get_outer_loop_params() = get_lora_params() = all trainable = LoRA + lm_head.
No ANIL split — both are transmitted and meta-gradient targets.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

MODEL_NAME = "facebook/wav2vec2-base-960h"

LORA_TARGET_LAYERS: list[int] = list(range(6, 12))
LORA_TARGET_PROJECTIONS: list[str] = ["q_proj", "k_proj", "v_proj", "out_proj"]
LORA_R: int = 8
LORA_ALPHA: float = 16.0

class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with a LoRA delta.

    Forward: y = W * x + bias + (B @ A) * x * scaling

    W and bias are frozen references to the original linear's tensors.
    A is initialized N(0, 0.02); B is zero-initialized → delta = 0 at start.
    """

    def __init__(self, linear: nn.Linear, r: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        d_out, d_in = linear.weight.shape

        self.weight = linear.weight
        self.bias = linear.bias
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

        self.scaling = alpha / r

        self.lora_A = nn.Parameter(torch.randn(r, d_in) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            F.linear(x, self.weight, self.bias)
            + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        )

    def merge(self) -> nn.Linear:
        """Return a plain nn.Linear with LoRA delta merged into the weight."""
        merged_weight = self.weight + (self.lora_B @ self.lora_A) * self.scaling
        linear = nn.Linear(
            self.lora_A.shape[1], self.lora_B.shape[0],
            bias=self.bias is not None,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        linear.weight = nn.Parameter(merged_weight)
        if self.bias is not None:
            linear.bias = nn.Parameter(self.bias.clone())
        return linear

class LoRAWav2Vec2(nn.Module):
    """
    Wav2Vec2ForCTC with LoRA injected into transformer layers 6–11.

    Second-order MAML is now possible because:
      1. Inner loop trains only ~320K LoRA + lm_head params (manageable graph)
      2. CTC loss is fully differentiable via ctc/differentiable_ctc.py
         (both LoRA curvature AND CTC alignment curvature are captured)

    This is TRUE second-order MAML — not a first-order approximation.

    Usage:
        model = LoRAWav2Vec2(device="cuda")
        # All trainable params: LoRA A/B + lm_head
        params = model.get_lora_params()
        # Forward — loss computed externally via ctc_loss_differentiable
        out = model(input_values=audio)
        logits = out.logits
    """

    def __init__(
        self,
        device: str | torch.device = "cpu",
        r: int = LORA_R,
        alpha: float = LORA_ALPHA,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.r = r
        self.alpha = alpha

        self.model: Wav2Vec2ForCTC = Wav2Vec2ForCTC.from_pretrained(
            MODEL_NAME, attn_implementation="eager"
        )

        for param in self.model.parameters():
            param.requires_grad_(False)

        self._inject_lora()

        for param in self.model.lm_head.parameters():
            param.requires_grad_(True)

        self.model = self.model.to(self.device)

    def _inject_lora(self) -> None:
        layers = self.model.wav2vec2.encoder.layers
        for idx in LORA_TARGET_LAYERS:
            attn = layers[idx].attention
            for proj_name in LORA_TARGET_PROJECTIONS:
                original = getattr(attn, proj_name)
                if not isinstance(original, nn.Linear):
                    raise TypeError(
                        f"Expected nn.Linear at layer {idx}.attention.{proj_name}, "
                        f"got {type(original).__name__}"
                    )
                setattr(attn, proj_name, LoRALinear(original, r=self.r, alpha=self.alpha))

    def get_lora_params(self) -> list[nn.Parameter]:
        """All trainable params: LoRA A/B matrices + lm_head. Used for inner + outer loop."""
        return [p for p in self.model.parameters() if p.requires_grad]

    def get_outer_loop_params(self) -> list[nn.Parameter]:
        """
        Same as get_lora_params().

        In FedLoRA-MAML, LoRA params AND lm_head adapt in the inner loop AND
        are aggregated by the FL server. No ANIL split — both are meta-gradient
        targets transmitted to the FL server.
        """
        return [p for p in self.model.parameters() if p.requires_grad]

    def merge_lora(self) -> None:
        """
        Merge LoRA deltas into frozen backbone weights.

        W_final = W_frozen + B @ A * scaling

        Call after personalization is complete. After merging, inference has
        zero overhead vs the base model. The LoRALinear modules are replaced
        with plain nn.Linear modules.
        """
        layers = self.model.wav2vec2.encoder.layers
        for idx in LORA_TARGET_LAYERS:
            attn = layers[idx].attention
            for proj_name in LORA_TARGET_PROJECTIONS:
                module = getattr(attn, proj_name)
                if isinstance(module, LoRALinear):
                    setattr(attn, proj_name, module.merge())

    def forward(
        self,
        input_values: torch.Tensor,
    ):
        """
        Forward pass — no labels argument.

        Loss is computed externally via ctc_loss_differentiable() to enable
        second-order MAML through the CTC computation. Passing labels here
        would use nn.CTCLoss internally, which blocks double-backward.
        """
        return self.model(input_values=input_values)

    def count_trainable(self) -> int:
        return sum(p.numel() for p in self.get_lora_params())

    def count_frozen(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if not p.requires_grad)

def load_processor() -> Wav2Vec2Processor:
    return Wav2Vec2Processor.from_pretrained(MODEL_NAME)

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    from ctc.differentiable_ctc import ctc_loss_differentiable

    print("LoRAWav2Vec2 self-test...")
    model = LoRAWav2Vec2(device="cpu")

    trainable = model.count_trainable()
    assert 300_000 < trainable < 400_000, (
        f"Expected ~320K trainable params, got {trainable}"
    )
    print(f"  Trainable params: {trainable:,}")
    print(f"  Frozen params:    {model.count_frozen():,}")

    for n, p in model.model.named_parameters():
        is_lora = "lora" in n.lower() or "lm_head" in n
        assert p.requires_grad == is_lora, (
            f"Param {n}: requires_grad={p.requires_grad}, expected {is_lora}"
        )
    print("  Param freeze/unfreeze: OK")

    dummy = torch.randn(1, 16000)
    out = model(dummy)
    assert hasattr(out, "logits"), "Output has no .logits attribute"
    assert out.logits.shape[-1] == 32, f"Expected vocab_size=32, got {out.logits.shape[-1]}"
    print(f"  Forward pass: logits shape {out.logits.shape}")

    labels = torch.randint(1, 32, (1, 5))
    input_lengths = torch.tensor([out.logits.shape[1]])
    target_lengths = torch.tensor([5])

    loss = ctc_loss_differentiable(out.logits, labels, input_lengths, target_lengths)
    lora_params = model.get_lora_params()

    grad1 = torch.autograd.grad(
        loss, lora_params, create_graph=True, allow_unused=True
    )
    grad1_nn = [g for g in grad1 if g is not None]
    assert len(grad1_nn) > 0, "No first-order gradients for LoRA params"

    grad2 = torch.autograd.grad(
        sum(g.norm() for g in grad1_nn),
        lora_params,
        allow_unused=True,
    )
    grad2_nn = [g for g in grad2 if g is not None]
    assert len(grad2_nn) > 0, "No second-order gradients — double-backward failed"
    print("  Second-order MAML through LoRA + CTC: OK")

    model.merge_lora()
    out2 = model(dummy)
    assert out2.logits.shape == out.logits.shape, "merge_lora changed output shape"
    print("  merge_lora: OK")

    print("All self-tests PASSED.")
