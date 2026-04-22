"""
Flower client for FedLoRA-MAML (Phase 2).

Each round:
  1. Receive LoRA + lm_head weights (θ*) from server via set_parameters()
  2. Run `tasks_per_node` lora_maml episodes, average meta-gradients
  3. Return averaged gradients (LoRA + lm_head shapes) to server

What this does NOT do:
  - Does NOT transmit the frozen backbone (~94.5M params never leave the node)
  - Does NOT return weight updates — returns gradients (Invariant I4)
  - Does NOT expose speaker identity (Invariants I1, I2)

Parameter order is established by LoRAWav2Vec2.get_outer_loop_params(), which
is a deterministic traversal of model.parameters(). Server and client always
call get_outer_loop_params() on identically-constructed models, so shapes and
order are consistent across the wire.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import flwr as fl
from flwr.common import (
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Parameters,
    Status,
    Code,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.lora_wav2vec2 import LoRAWav2Vec2, load_processor
from data.task_sampler import VoiceTaskSampler
from maml.engine import MAMLEngine

_MAX_GRAD_NORM = 20.0


class MAMLClientLora(fl.client.Client):
    """
    Flower client implementing PerFedAvg for FedLoRA-MAML.

    Each call to fit():
      - Loads LoRA + lm_head weights (θ*) from server
      - Runs tasks_per_node independent lora_maml episodes
      - Averages meta-gradients across episodes
      - Returns gradient-clipped average (same shapes as LoRA + lm_head params)

    higher.innerloop_ctx(copy_initial_weights=False) does NOT modify the
    original model's .data — after each episode the model still holds θ*.
    Running multiple tasks sequentially from the same θ* is therefore safe.
    """

    def __init__(
        self,
        node_dir: str | Path,
        device: str | torch.device = "cpu",
        inner_steps: int = 5,
        inner_lr: float = 1e-4,
        support_size: int = 20,
        query_size: int = 30,
        tasks_per_node: int = 4,
        max_audio_samples: int | None = None,
    ):
        self.device = torch.device(device)
        self.model = LoRAWav2Vec2(device=self.device)
        self.processor = load_processor()
        self.sampler = VoiceTaskSampler(node_dir, support_size, query_size)
        self.engine = MAMLEngine(
            self.model,
            self.processor,
            inner_steps=inner_steps,
            inner_lr=inner_lr,
            device=self.device,
            mode="lora_maml",
            max_audio_samples=max_audio_samples,
        )
        self.tasks_per_node = tasks_per_node

        print(
            f"[client] clips={self.sampler.num_clips} | "
            f"inner_steps={inner_steps} | inner_lr={inner_lr} | "
            f"tasks/round={tasks_per_node} | max_samples={max_audio_samples}",
            flush=True,
        )

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """Return LoRA + lm_head weights as float16 (halves wire size)."""
        ndarrays = [
            p.detach().half().cpu().numpy()
            for p in self.model.get_outer_loop_params()
        ]
        return GetParametersRes(
            status=Status(code=Code.OK, message=""),
            parameters=ndarrays_to_parameters(ndarrays),
        )

    def set_parameters(self, parameters: Parameters) -> None:
        """Load LoRA + lm_head weights from server. Frozen backbone untouched."""
        ndarrays = parameters_to_ndarrays(parameters)
        outer_params = self.model.get_outer_loop_params()
        if len(ndarrays) != len(outer_params):
            raise ValueError(
                f"Parameter count mismatch: received {len(ndarrays)}, "
                f"expected {len(outer_params)}"
            )
        with torch.no_grad():
            for p, arr in zip(outer_params, ndarrays):
                # arr may be float16 (from wire); cast to the param's dtype (float32)
                p.copy_(torch.tensor(arr, dtype=p.dtype, device=self.device))

    def fit(self, ins: FitIns) -> FitRes:
        """
        Run tasks_per_node lora_maml episodes and return averaged meta-gradients.

        higher.innerloop_ctx does not mutate original model params, so all tasks
        start from the same θ* that was loaded by set_parameters().

        Returns gradients (not weight updates) — Invariant I4.
        """
        self.set_parameters(ins.parameters)

        acc_grads: list[torch.Tensor] | None = None

        for _ in range(self.tasks_per_node):
            task = self.sampler.sample_task()
            task_grads = self.engine.compute_meta_gradient(task)

            if acc_grads is None:
                acc_grads = [g.clone() for g in task_grads]
            else:
                for i, g in enumerate(task_grads):
                    acc_grads[i].add_(g)

            del task_grads

        avg_grads = [g.div_(self.tasks_per_node) for g in acc_grads]  # type: ignore[union-attr]

        # Global gradient norm clip
        total_norm = sum(g.float().norm() ** 2 for g in avg_grads) ** 0.5
        clip_coef = _MAX_GRAD_NORM / (total_norm.item() + 1e-6)
        if clip_coef < 1.0:
            for g in avg_grads:
                g.mul_(clip_coef)

        grad_arrays = [g.half().cpu().numpy() for g in avg_grads]
        num_examples = self.tasks_per_node * (
            self.sampler.support_size + self.sampler.query_size
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return FitRes(
            status=Status(code=Code.OK, message=""),
            parameters=ndarrays_to_parameters(grad_arrays),
            num_examples=num_examples,
            metrics={},
        )

    def evaluate(self, ins):
        from flwr.common import EvaluateRes
        return EvaluateRes(
            status=Status(code=Code.OK, message=""),
            loss=0.0,
            num_examples=0,
            metrics={},
        )
