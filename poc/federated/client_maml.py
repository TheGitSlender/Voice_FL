"""Flower client for FOMAML-ANIL. Returns encoder meta-gradients per round."""

from __future__ import annotations

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

from models.wav2vec2_maml import Wav2Vec2MAML, load_processor
from data.task_sampler import VoiceTaskSampler
from maml.engine import MAMLEngine

class MAMLClient(fl.client.Client):
    """Flower client implementing PerFedAvg (FOMAML-ANIL)."""

    def __init__(
        self,
        node_dir: str | Path,
        device: str | torch.device = "cpu",
        inner_steps: int = 3,
        inner_lr: float = 1e-4,
        support_size: int = 8,
        query_size: int = 8,
        max_grad_norm: float = 10.0,
    ):
        self.device = torch.device(device)
        self.model = Wav2Vec2MAML(device=self.device)
        if self.device.type == "cuda":
            self.model.to_bf16()
        else:
                                                                             
            self.model.strip_weight_parametrizations()

        self.processor = load_processor()
        self.sampler = VoiceTaskSampler(node_dir, support_size, query_size)
        self.engine = MAMLEngine(
            self.model, self.processor, inner_steps, inner_lr, self.device
        )
        self._max_grad_norm = max_grad_norm

        print(f"[client] Node dir: {Path(node_dir).name[:16]} | clips: {self.sampler.num_clips}")

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """Return encoder weights as float16 (lm_head excluded — Invariant I3)."""
        ndarrays = [
            p.detach().half().cpu().numpy()
            for p in self.model.get_outer_loop_params()
        ]
        return GetParametersRes(
            status=Status(code=Code.OK, message=""),
            parameters=ndarrays_to_parameters(ndarrays),
        )

    def set_parameters(self, parameters: Parameters) -> None:
        """Load encoder weights from server. lm_head untouched (Invariant I3)."""
        ndarrays = parameters_to_ndarrays(parameters)
        outer_params = self.model.get_outer_loop_params()
        if len(ndarrays) != len(outer_params):
            raise ValueError(
                f"Parameter count mismatch: received {len(ndarrays)}, "
                f"expected {len(outer_params)}"
            )
        with torch.no_grad():
            for p, arr in zip(outer_params, ndarrays):
                p.copy_(torch.tensor(arr, dtype=p.dtype, device=self.device))

    def fit(self, ins: FitIns) -> FitRes:
        """
        Run one FOMAML episode.

        Returns meta-gradients encoded as parameters (Invariant I4:
        these are gradients, not weight updates).
        """
        self.set_parameters(ins.parameters)

        task = self.sampler.sample_task()
        meta_grads = self.engine.compute_meta_gradient(task)

        total_norm = sum(g.float().norm() ** 2 for g in meta_grads) ** 0.5
        clip_coef = self._max_grad_norm / (total_norm.item() + 1e-6)
        if clip_coef < 1.0:
            meta_grads = [g * clip_coef for g in meta_grads]

        grad_arrays = [g.half().cpu().numpy() for g in meta_grads]

        num_examples = len(task.support_audio) + len(task.query_audio)
        return FitRes(
            status=Status(code=Code.OK, message=""),
            parameters=ndarrays_to_parameters(grad_arrays),
            num_examples=num_examples,
            metrics={},
        )

    def evaluate(self, ins):
        """Not used in PoC — evaluation is done centrally via eval_poc.py."""
        from flwr.common import EvaluateRes
        return EvaluateRes(
            status=Status(code=Code.OK, message=""),
            loss=0.0,
            num_examples=0,
            metrics={},
        )
