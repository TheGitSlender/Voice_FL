"""Flower client for FedLoRA-MAML. Returns meta-gradients and per-round diagnostics."""

from __future__ import annotations

import math

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

from models.lora_wav2vec2 import LoRAWav2Vec2, load_processor
from data.task_sampler import VoiceTaskSampler
from maml.engine import MAMLEngine


class MAMLClientLora(fl.client.Client):
    """Flower client implementing PerFedAvg for FedLoRA-MAML."""

    def __init__(
        self,
        node_dir: str | Path,
        device: str | torch.device = "cpu",
        inner_steps: int = 5,
        inner_lr: float = 1e-4,
        support_size: int = 8,
        query_size: int = 16,
        tasks_per_node: int = 4,
        max_audio_samples: int | None = None,
        speaker_id: str = "",
        max_grad_norm: float = 50.0,
    ):
        self.device = torch.device(device)
        self._tag = f"[node/{speaker_id}]" if speaker_id else "[node]"
        self._inner_steps = inner_steps
        self._inner_lr = inner_lr
        self._max_audio_samples = max_audio_samples

        # Model stays on CPU between rounds; moved to GPU only during fit()
        # so all 12 containers coexist without exhausting VRAM.
        self.model = LoRAWav2Vec2(device="cpu")
        self.processor = load_processor()
        self.sampler = VoiceTaskSampler(node_dir, support_size, query_size)
        self.engine = MAMLEngine(
            self.model,
            self.processor,
            inner_steps=inner_steps,
            inner_lr=inner_lr,
            device=torch.device("cpu"),  # updated to GPU at fit() time
            mode="lora_maml",
            max_audio_samples=max_audio_samples,
        )
        self.tasks_per_node = tasks_per_node
        self._max_grad_norm = max_grad_norm

        print(
            f"{self._tag} clips={self.sampler.num_clips} | "
            f"inner_steps={inner_steps} | inner_lr={inner_lr} | "
            f"tasks/round={tasks_per_node} | max_samples={max_audio_samples}",
            flush=True,
        )

    def _move_to_gpu(self) -> None:
        """Move model and engine to GPU before a fit() call."""
        self.model.model.to(self.device)
        self.engine.device = self.device

    def _move_to_cpu(self) -> None:
        """Return model to CPU and free GPU cache after a fit() call."""
        self.model.model.to("cpu")
        self.engine.device = torch.device("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """Return LoRA + lm_head weights as float32."""
        ndarrays = [
            p.detach().float().cpu().numpy()
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
                # p.device reflects wherever the model currently lives (cpu or gpu)
                p.copy_(torch.tensor(arr, dtype=p.dtype, device=p.device))

    def fit(self, ins: FitIns) -> FitRes:
        """
        Run tasks_per_node lora_maml episodes and return averaged meta-gradients.

        Logs per-task inner-loop convergence and outer query loss to stdout.
        Returns diagnostic metrics in FitRes.metrics for server-side MLflow logging.
        """
        self._move_to_gpu()
        self.set_parameters(ins.parameters)
        server_round = int(ins.config.get("round", 0))

        acc_grads: list[torch.Tensor] | None = None
        query_losses: list[float] = []
        inner_losses_per_task: list[list[float]] = []

        for task_idx in range(self.tasks_per_node):
            task = self.sampler.sample_task()
            result = self.engine.compute_meta_gradient(
                task, return_query_loss=True, return_inner_losses=True
            )
            task_grads, query_loss, inner_losses = result

            # NaN guard — surface early so the run can be aborted
            if math.isnan(query_loss):
                print(
                    f"{self._tag} round={server_round} task={task_idx} "
                    f"WARNING: NaN query_loss — possible exploding gradient",
                    flush=True,
                )
            nan_steps = [i for i, l in enumerate(inner_losses) if math.isnan(l)]
            if nan_steps:
                print(
                    f"{self._tag} round={server_round} task={task_idx} "
                    f"WARNING: NaN inner loss at steps {nan_steps}",
                    flush=True,
                )

            query_losses.append(query_loss)
            inner_losses_per_task.append(inner_losses)

            inner_str = " ".join(
                f"k{i}={l:.3f}" if not math.isnan(l) else f"k{i}=NaN"
                for i, l in enumerate(inner_losses)
            )
            print(
                f"{self._tag} round={server_round} "
                f"task={task_idx + 1}/{self.tasks_per_node} "
                f"query_loss={query_loss:.4f} | inner: {inner_str}",
                flush=True,
            )

            if acc_grads is None:
                acc_grads = [g.clone() for g in task_grads]
            else:
                for i, g in enumerate(task_grads):
                    acc_grads[i].add_(g)
            del task_grads

        avg_grads = [g.div_(self.tasks_per_node) for g in acc_grads]  # type: ignore[union-attr]

        # Global gradient norm clip
        total_norm_sq = sum(g.float().norm() ** 2 for g in avg_grads)
        total_norm = float(total_norm_sq ** 0.5)
        clip_coef = min(1.0, self._max_grad_norm / (total_norm + 1e-6))
        if clip_coef < 1.0:
            for g in avg_grads:
                g.mul_(clip_coef)

        # Aggregate diagnostics across tasks
        valid_query = [l for l in query_losses if not math.isnan(l)]
        avg_query_loss = sum(valid_query) / len(valid_query) if valid_query else float("nan")

        # Per-step mean across tasks (for inner convergence curve)
        n_steps = max((len(ls) for ls in inner_losses_per_task if ls), default=0)
        inner_loss_init = float("nan")
        inner_loss_final = float("nan")
        if n_steps > 0:
            inits = [ls[0] for ls in inner_losses_per_task if ls and not math.isnan(ls[0])]
            finals = [ls[-1] for ls in inner_losses_per_task if ls and not math.isnan(ls[-1])]
            if inits:
                inner_loss_init = sum(inits) / len(inits)
            if finals:
                inner_loss_final = sum(finals) / len(finals)

        print(
            f"{self._tag} round={server_round} SUMMARY "
            f"avg_query_loss={avg_query_loss:.4f} "
            f"grad_norm={total_norm:.4f} "
            f"clip_coef={clip_coef:.4f} "
            f"inner_init={inner_loss_init:.4f} inner_final={inner_loss_final:.4f}",
            flush=True,
        )

        grad_arrays = [g.float().cpu().numpy() for g in avg_grads]
        num_examples = self.tasks_per_node * (
            self.sampler.support_size + self.sampler.query_size
        )

        self._move_to_cpu()

        return FitRes(
            status=Status(code=Code.OK, message=""),
            parameters=ndarrays_to_parameters(grad_arrays),
            num_examples=num_examples,
            metrics={
                "query_loss": avg_query_loss,
                "grad_norm": total_norm,
                "clip_coef": clip_coef,
                "inner_loss_init": inner_loss_init,
                "inner_loss_final": inner_loss_final,
            },
        )

    def evaluate(self, ins):
        from flwr.common import EvaluateRes
        return EvaluateRes(
            status=Status(code=Code.OK, message=""),
            loss=0.0,
            num_examples=0,
            metrics={},
        )
