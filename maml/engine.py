"""
maml/engine.py — MAML engine for Wav2Vec2 personalization

Implements three MAML variants for Wav2Vec2 CTC:
  "full"   — Second-order MAML via `higher` (A100 recommended)
  "fomaml" — First-order MAML (RTX 4070 Super, native PyTorch)
  "reptile" — Reptile (meta-grad = θ_init - θ_adapted)

Public interface for all three modes:
  compute_meta_gradient(support_audio, support_labels, query_audio, query_labels)
  → (meta_grads, query_loss_value)

meta_grads is a list aligned with model.model.parameters().
After DP sanitization in the client, outer-loop gradients are transmitted to the server.

Note on learn2learn: cannot be installed on Python 3.13+ (missing longintrepr.h
in CPython internals). FOMAML is implemented natively using PyTorch autograd
with create_graph=False, which is mathematically equivalent to learn2learn's
MAML(first_order=True). The external API in this file is identical to what
the build spec requires.

ANIL inner loop note (I7): the inner optimizer receives only lm_head parameters.
The encoder (wav2vec2) is NOT in the inner optimizer but remains in the
computation graph. For second-order MAML, higher.innerloop_ctx with
track_higher_grads=True allows gradients to flow through the inner steps
back into the encoder. For FOMAML, we differentiate at the adapted weights
without flowing through the adaptation steps (first-order approximation).
"""

import copy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import higher
import torch

from models.wav2vec2_maml import Wav2Vec2MAML


@dataclass
class MAMLConfig:
    mode: str = "fomaml"          # "full" | "fomaml" | "reptile"
    k: int = 3                    # inner loop steps
    inner_lr: float = 1e-4        # α — inner loop learning rate
    outer_lr: float = 2e-4        # β — outer loop (used in centralized mode)
    support_size: int = 8
    query_size: int = 8
    adaptation_mode: str = "anil"
    use_bf16: bool = True
    gradient_checkpointing: bool = True


class MAMLEngine:
    """
    Wav2Vec2 MAML engine. Unified interface across full/fomaml/reptile modes.

    All three variants expose the same compute_meta_gradient() interface.
    Switch mode via MAMLConfig — no other code needs to change.

    The meta-gradient returned is ∂L_query/∂θ* — the outer-loop update signal.
    It has the same length and shapes as list(model.model.parameters()).
    After DP sanitization, the outer-loop portion is transmitted to the server.
    """

    def __init__(self, model: Wav2Vec2MAML, config: MAMLConfig) -> None:
        self.model = model
        self.config = config

    def compute_meta_gradient(
        self,
        support_audio: torch.Tensor,
        support_labels: torch.Tensor,
        query_audio: torch.Tensor,
        query_labels: torch.Tensor,
    ) -> Tuple[List[Optional[torch.Tensor]], float]:
        """
        Single task MAML step.

        Returns:
          meta_grads: list aligned with model.model.parameters()
                      None entries for unused params (allow_unused=True)
          query_loss: float — outer loop loss value for logging
        """
        if self.config.mode == "full":
            return self._full_maml(support_audio, support_labels, query_audio, query_labels)
        elif self.config.mode == "fomaml":
            return self._fomaml(support_audio, support_labels, query_audio, query_labels)
        elif self.config.mode == "reptile":
            return self._reptile(support_audio, support_labels, query_audio, query_labels)
        else:
            raise ValueError(f"Unknown MAML mode: {self.config.mode}")

    def _full_maml(
        self,
        support_audio: torch.Tensor,
        support_labels: torch.Tensor,
        query_audio: torch.Tensor,
        query_labels: torch.Tensor,
    ) -> Tuple[List[Optional[torch.Tensor]], float]:
        """
        Full second-order MAML via `higher`.

        track_higher_grads=True: backpropagation flows THROUGH the inner loop
        (Hessian term included). This is the exact Per-FedAvg gradient.

        Inner optimizer: lm_head only (ANIL — I7).
        Outer gradient: flows through lm_head adaptation back into wav2vec2 encoder.

        Requires ~80GB VRAM for Wav2Vec2 with K=3. Use A100.
        """
        inner_opt = torch.optim.SGD(
            self.model.get_inner_loop_params(),
            lr=self.config.inner_lr,
        )

        autocast_ctx = torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=self.config.use_bf16 and torch.cuda.is_available(),
        )

        with higher.innerloop_ctx(
            self.model.model,
            inner_opt,
            copy_initial_weights=False,
            track_higher_grads=True,  # second-order: Hessian term included
        ) as (fmodel, diffopt):

            for _ in range(self.config.k):
                with autocast_ctx:
                    out = fmodel(input_values=support_audio, labels=support_labels)
                diffopt.step(out.loss)  # differentiable lm_head update

            with autocast_ctx:
                query_out = fmodel(input_values=query_audio, labels=query_labels)

        # d(query_loss)/d(θ*) — flows through k inner steps (second-order)
        meta_grads = torch.autograd.grad(
            query_out.loss,
            self.model.model.parameters(),
            allow_unused=True,
            retain_graph=False,
        )
        return list(meta_grads), query_out.loss.item()

    def _fomaml(
        self,
        support_audio: torch.Tensor,
        support_labels: torch.Tensor,
        query_audio: torch.Tensor,
        query_labels: torch.Tensor,
    ) -> Tuple[List[Optional[torch.Tensor]], float]:
        """
        First-order MAML (FOMAML) — native PyTorch implementation.

        Equivalent to learn2learn MAML(first_order=True) but without
        the Cython dependency that breaks on Python 3.13+.

        Approximation: gradient at θ_i' treated as if it came from θ*
        (no backprop through inner loop — no Hessian term).

        Steps:
          1. Save initial model state
          2. Run k inner gradient steps on lm_head using support set
          3. Evaluate on query set from adapted state
          4. Compute gradient at adapted params (create_graph=False → FOMAML)
          5. Restore original model state
        """
        # Save initial state for restoration
        init_state = copy.deepcopy(self.model.model.state_dict())

        # Inner loop: k SGD steps on lm_head only (ANIL)
        inner_opt = torch.optim.SGD(
            self.model.get_inner_loop_params(),
            lr=self.config.inner_lr,
        )
        for _ in range(self.config.k):
            inner_opt.zero_grad()
            out = self.model(input_values=support_audio, labels=support_labels)
            out.loss.backward()
            inner_opt.step()

        # Evaluate at adapted params — FOMAML: no graph through inner loop
        self.model.model.zero_grad()
        autocast_ctx = torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=self.config.use_bf16 and torch.cuda.is_available(),
        )
        with autocast_ctx:
            query_out = self.model(input_values=query_audio, labels=query_labels)

        # create_graph=False: first-order approximation (no Hessian)
        meta_grads = torch.autograd.grad(
            query_out.loss,
            self.model.model.parameters(),
            allow_unused=True,
            retain_graph=False,
            create_graph=False,
        )
        query_loss_val = query_out.loss.item()

        # Restore initial params (do not permanently update model)
        self.model.model.load_state_dict(init_state)

        return list(meta_grads), query_loss_val

    def _reptile(
        self,
        support_audio: torch.Tensor,
        support_labels: torch.Tensor,
        query_audio: torch.Tensor,
        query_labels: torch.Tensor,
    ) -> Tuple[List[Optional[torch.Tensor]], float]:
        """
        Reptile: meta-gradient = (θ_init - θ_adapted) / k.

        Runs k SGD steps on the full model, then computes the difference
        between initial and adapted weights as a pseudo-gradient.
        Restores initial params before returning.

        The query set is used only for loss reporting (no gradient computation).
        """
        # Save initial params
        init_params = [p.data.clone() for p in self.model.model.parameters()]

        inner_opt = torch.optim.SGD(
            self.model.model.parameters(),
            lr=self.config.inner_lr,
        )
        for _ in range(self.config.k):
            inner_opt.zero_grad()
            out = self.model(input_values=support_audio, labels=support_labels)
            out.loss.backward()
            inner_opt.step()

        # Meta-gradient = direction from adapted back to init
        meta_grads = [
            (init - p.data)
            for init, p in zip(init_params, self.model.model.parameters())
        ]

        # Restore initial params
        for p, init in zip(self.model.model.parameters(), init_params):
            p.data.copy_(init)

        with torch.no_grad():
            query_out = self.model(input_values=query_audio, labels=query_labels)

        return meta_grads, query_out.loss.item()

    def adapt(
        self,
        audio: torch.Tensor,
        labels: torch.Tensor,
        k: Optional[int] = None,
    ) -> "Wav2Vec2ForCTC":
        """
        Personalize for a new user at inference time.

        Returns an adapted model copy (does NOT modify self.model).
        Runs k SGD steps on lm_head using the provided support clips.

        k defaults to config.k.
        """
        from transformers import Wav2Vec2ForCTC

        k = k or self.config.k
        adapted = copy.deepcopy(self.model.model)
        adapted = adapted.to(audio.device)

        opt = torch.optim.SGD(
            adapted.lm_head.parameters(),
            lr=self.config.inner_lr,
        )
        for _ in range(k):
            opt.zero_grad()
            out = adapted(input_values=audio, labels=labels)
            out.loss.backward()
            opt.step()

        return adapted
