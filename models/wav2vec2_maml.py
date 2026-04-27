"""
Wav2Vec2MAML — thin wrapper around Wav2Vec2ForCTC for ANIL meta-learning.

ANIL split:
  Inner loop (local, never transmitted): lm_head  (~25K params)
  Outer loop (aggregated by FL server):  wav2vec2 encoder (~94.5M params)

Critical invariant:
  I3: lm_head is never returned from get_outer_loop_params()
  I6: encoder.requires_grad stays True throughout

Usage:
  model = Wav2Vec2MAML()
  inner_params = model.get_inner_loop_params()   # lm_head only
  outer_params = model.get_outer_loop_params()   # encoder only
"""

from __future__ import annotations

from typing import Iterator

import torch
import torch.nn as nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

MODEL_NAME = "facebook/wav2vec2-base-960h"

class Wav2Vec2MAML(nn.Module):
    def __init__(self, device: str | torch.device = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
        self.model = self.model.to(self.device)

        self._verify_invariant_i6()

    def _verify_invariant_i6(self) -> None:
        for name, param in self.model.wav2vec2.named_parameters():
            assert param.requires_grad, (
                f"INVARIANT VIOLATION I6: {name}.requires_grad is False"
            )

    def get_inner_loop_params(self) -> list[nn.Parameter]:
        """Return lm_head parameters only (inner loop / local adaptation)."""
        return list(self.model.lm_head.parameters())

    def get_outer_loop_params(self) -> list[nn.Parameter]:
        """Return wav2vec2 encoder parameters only (outer loop / FL aggregation).

        Invariant I3: lm_head parameters are NOT included.
        """
        params = list(self.model.wav2vec2.parameters())
                                            
        lm_head_ids = {id(p) for p in self.model.lm_head.parameters()}
        for p in params:
            assert id(p) not in lm_head_ids, "INVARIANT VIOLATION I3: lm_head in outer params"
        return params

    def verify_param_partition(self) -> None:
        """Assert inner + outer = total, with zero overlap."""
        inner = {id(p) for p in self.get_inner_loop_params()}
        outer = {id(p) for p in self.get_outer_loop_params()}
        total = {id(p) for p in self.model.parameters()}

        overlap = inner & outer
        union = inner | outer

        assert not overlap, f"Inner/outer param overlap: {len(overlap)} params"
        assert union == total, (
            f"Param partition incomplete: "
            f"union={len(union)}, total={len(total)}, "
            f"gap={total - union}"
        )

    def forward(
        self,
        input_values: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ):
        """Forward pass. Returns Wav2Vec2CTC output (includes loss if labels given)."""
        return self.model(
            input_values=input_values,
            labels=labels,
            attention_mask=attention_mask,
        )

    def get_encoder_state_dict(self) -> dict:
        """Return a copy of the encoder state dict (for serialization)."""
        return {k: v.clone() for k, v in self.model.wav2vec2.state_dict().items()}

    def set_encoder_state_dict(self, state_dict: dict) -> None:
        """Load encoder weights from a state dict (from FL server)."""
        self.model.wav2vec2.load_state_dict(state_dict)
        self._verify_invariant_i6()

    def strip_weight_parametrizations(self) -> "Wav2Vec2MAML":
        """Remove pos_conv_embed weight parametrization so parameter count matches
        the server's 210-param layout (server calls this internally too).
        Must be called on CPU clients that do not call to_bf16().
        """
        conv = self.model.wav2vec2.encoder.pos_conv_embed.conv
        if hasattr(conv, "parametrizations") and "weight" in conv.parametrizations:
            torch.nn.utils.parametrize.remove_parametrizations(conv, "weight")
        return self

    def to_bf16(self) -> "Wav2Vec2MAML":
        """Cast model to BF16 for VRAM efficiency.

        PyTorch 2.1's weight_norm kernel does not support BF16 on
        pos_conv_embed.conv (which uses the new parametrize API).
        Remove the parametrization first so the weight becomes a plain
        BF16 parameter; the mathematical effect is identical.
        """
        conv = self.model.wav2vec2.encoder.pos_conv_embed.conv
        if hasattr(conv, "parametrizations") and "weight" in conv.parametrizations:
            torch.nn.utils.parametrize.remove_parametrizations(conv, "weight")
        self.model = self.model.to(torch.bfloat16)
        return self

def load_processor() -> "Wav2Vec2Processor":
    return Wav2Vec2Processor.from_pretrained(MODEL_NAME)

def _smoke_test() -> None:
    print("Wav2Vec2MAML smoke test...")
    model = Wav2Vec2MAML(device="cpu")

    inner = model.get_inner_loop_params()
    outer = model.get_outer_loop_params()
    print(f"  inner (lm_head): {sum(p.numel() for p in inner):,} params")
    print(f"  outer (encoder): {sum(p.numel() for p in outer):,} params")

    model.verify_param_partition()
    print("  param partition: OK")

    dummy = torch.zeros(1, 16000)
    out = model(dummy)
    print(f"  logits shape: {out.logits.shape}")
    print("Smoke test PASSED")

if __name__ == "__main__":
    _smoke_test()
