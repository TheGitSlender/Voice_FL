"""
models/wav2vec2_maml.py — Wav2Vec2 wrapper for MAML personalization

Architecture: facebook/wav2vec2-base-960h
  - Encoder-only transformer + CTC head
  - Input: raw waveform float32 at 16kHz
  - Output: logits (batch, T_frames, vocab_size=32)
  - Loss: CTC (single-pass — clean MAML inner loop)

ANIL split (Asymmetric Inner Loop):
  Inner loop (adapts):       model.lm_head   (~25K params, linear 768→32)
  Outer loop (FL aggregate): model.wav2vec2  (~94M params, transformer encoder)

Why Wav2Vec2 over Whisper (architectural decision):
  Whisper-small (seq2seq):           Wav2Vec2-base-960h (CTC):
    encoder → decoder autoregressive   encoder → lm_head single pass
    inner loop: complex gradient flow  inner loop: clean, direct gradient flow
    WER test-clean: ~9%                WER test-clean: ~3.4%
    input: log-mel (80, T)             input: raw waveform (T_samples,)

  Wav2Vec2 CTC computes in a single forward pass. The MAML inner loop
  gradient is cleaner, faster, and easier to debug.

Critical invariant (I7): encoder requires_grad stays True.
  Do NOT set wav2vec2 encoder to requires_grad=False.
  Only exclude encoder from the INNER LOOP OPTIMIZER, not from grad computation.
  Second-order MAML gradients must flow through lm_head → encoder.
  Setting requires_grad=False on the encoder would prevent the meta-gradient
  from flowing through the encoder, breaking the outer loop update entirely.

Critical invariant (I3): lm_head is NEVER transmitted over FL.
  get_parameters() and set_parameters() operate on encoder params only.
  The lm_head stays local — it is the personalization component.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


class Wav2Vec2MAML(nn.Module):
    """
    Wav2Vec2-base-960h wrapped for Per-FedAvg MAML personalization.

    ANIL mode (default and recommended):
      Inner loop adapts:   model.lm_head  (linear 768→32, ~25K params)
      FL aggregates:       model.wav2vec2 (encoder, ~94M params)

    The encoder learns acoustic representations shared across all speakers.
    The lm_head adapts rapidly to each speaker's specific voice patterns.
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        mode: str = "anil",
    ) -> None:
        super().__init__()
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.mode = mode
        # I7: Do NOT set encoder requires_grad=False here.
        # The encoder participates in grad computation for second-order MAML.
        # Only lm_head is passed to the inner loop optimizer.

    def forward(
        self,
        input_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        input_values: (batch, T_samples) raw waveform, float32, 16kHz
        labels: (batch, N_tokens) tokenized transcriptions, -100 for padding
        Returns HuggingFace CausalLMOutput with .loss (CTC) and .logits.
        """
        return self.model(input_values=input_values, labels=labels)

    def get_inner_loop_params(self) -> List[nn.Parameter]:
        """
        Parameters for the INNER LOOP optimizer (personalization target).

        ANIL: only lm_head (~25K params — very fast inner loop).
        The inner loop only updates these. The encoder is not updated
        during inner loop steps but remains in the computation graph
        for second-order gradient flow (I7).
        """
        if self.mode == "anil":
            return list(self.model.lm_head.parameters())
        # Full mode: all parameters adapt in inner loop
        return [p for p in self.model.parameters() if p.requires_grad]

    def get_outer_loop_params(self) -> List[nn.Parameter]:
        """
        Parameters aggregated by FL (global backbone).

        ANIL: wav2vec2 encoder (all params except lm_head, ~94M).
        These are shared across all users and updated via meta-gradients.
        The lm_head is intentionally excluded (I3) — it stays local.
        """
        if self.mode == "anil":
            return [
                p for name, p in self.model.named_parameters()
                if not name.startswith("lm_head")
            ]
        return [p for p in self.model.parameters() if p.requires_grad]

    def get_named_outer_params(self) -> List[Tuple[str, nn.Parameter]]:
        """Named outer parameters — used for Flower serialization."""
        if self.mode == "anil":
            return [
                (name, p) for name, p in self.model.named_parameters()
                if not name.startswith("lm_head")
            ]
        return list(self.model.named_parameters())

    def decode(self, logits: torch.Tensor) -> List[str]:
        """Greedy CTC decode: argmax over vocab at each time step."""
        pred_ids = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(pred_ids)

    def num_params(self) -> Dict[str, int]:
        """Parameter count breakdown: total, encoder, lm_head."""
        total = sum(p.numel() for p in self.model.parameters())
        encoder = sum(
            p.numel() for name, p in self.model.named_parameters()
            if not name.startswith("lm_head")
        )
        head = sum(p.numel() for p in self.model.lm_head.parameters())
        return {"total": total, "encoder": encoder, "lm_head": head}


if __name__ == "__main__":
    """Quick self-test: verify ANIL split is correct."""
    model = Wav2Vec2MAML(mode="anil")
    counts = model.num_params()
    print(f"Total params:   {counts['total']:,}")
    print(f"Encoder params: {counts['encoder']:,}  (outer loop / FL)")
    print(f"lm_head params: {counts['lm_head']:,}  (inner loop / personalization)")

    assert counts["total"] == counts["encoder"] + counts["lm_head"], (
        "BUG: encoder + lm_head != total — parameter overlap detected"
    )
    print("Parameter split verified: total == encoder + lm_head  ✓")

    # Verify encoder requires_grad is True (I7)
    for name, p in model.model.named_parameters():
        if not name.startswith("lm_head"):
            assert p.requires_grad, f"BUG: encoder param {name} has requires_grad=False"
    print("Encoder requires_grad=True verified  ✓ (I7)")

    # Verify lm_head is not in outer loop params (I3)
    outer_names = {name for name, _ in model.get_named_outer_params()}
    assert not any(n.startswith("lm_head") for n in outer_names), (
        "BUG: lm_head found in outer loop params — violates I3"
    )
    print("lm_head excluded from outer loop  ✓ (I3)")
