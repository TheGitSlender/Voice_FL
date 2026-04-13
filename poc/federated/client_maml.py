"""
Flower client for FOMAML-ANIL federated learning.

What this client does each round:
  1. Receive θ* (encoder weights) from server via set_parameters()
  2. Load them into local model
  3. Run compute_meta_gradient() on a fresh task
  4. Return meta-gradients (encoded as numpy arrays) to server

What this client does NOT do:
  - It does NOT return updated weights (Invariant I4)
  - It does NOT include lm_head in get_parameters() (Invariant I3)
  - It does NOT transmit any audio data or speaker identity (Invariants I1, I2)
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

from models.wav2vec2_maml import Wav2Vec2MAML, load_processor
from data.task_sampler import VoiceTaskSampler
from maml.engine import MAMLEngine


class MAMLClient(fl.client.Client):
    """
    Flower client implementing PerFedAvg protocol.

    Each call to fit():
      - Loads θ* from server
      - Runs one FOMAML episode (support + query)
      - Returns encoder meta-gradients (same shape as encoder weights)
    """

    def __init__(
        self,
        node_dir: str | Path,
        device: str | torch.device = "cpu",
        inner_steps: int = 3,
        inner_lr: float = 1e-4,
        support_size: int = 8,
        query_size: int = 8,
    ):
        self.device = torch.device(device)
        self.model = Wav2Vec2MAML(device=self.device)
        if self.device.type == "cuda":
            self.model.to_bf16()

        self.processor = load_processor()
        self.sampler = VoiceTaskSampler(node_dir, support_size, query_size)
        self.engine = MAMLEngine(
            self.model, self.processor, inner_steps, inner_lr, self.device
        )

        print(f"[client] Node dir: {Path(node_dir).name[:16]} | clips: {self.sampler.num_clips}")

    # ------------------------------------------------------------------
    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """Return encoder weights (lm_head EXCLUDED — Invariant I3).

        Uses float16 for wire transfer to halve message size (~189 MB vs 378 MB),
        reducing peak gRPC buffer pressure when 5 clients are served concurrently.
        """
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

        # Clip gradient norm to float16-safe range before wire transfer.
        # BF16 range (~3.4e38) >> float16 range (~65504); values above float16
        # max would silently become Inf.  A norm cap of 10.0 keeps individual
        # parameter gradients well within float16 representable range.
        total_norm = sum(g.float().norm() ** 2 for g in meta_grads) ** 0.5
        max_norm = 10.0
        clip_coef = max_norm / (total_norm.item() + 1e-6)
        if clip_coef < 1.0:
            meta_grads = [g * clip_coef for g in meta_grads]

        # Encode gradients as float16 numpy arrays — halves message size
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
