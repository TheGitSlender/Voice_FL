"""
federated/client_maml.py — Flower client for Per-FedAvg MAML

Implements fl.client.NumPyClient for federated MAML personalization.

Critical invariants:
  I3: get_parameters() returns ENCODER params only — lm_head never transmitted
  I4: fit() returns META-GRADIENTS (∇L_meta_private) — not weight updates
  I5: DP applied manually via privacy/dp_meta.py — no Opacus

The FL server (PerFedAvgStrategy) treats the returned arrays as gradients
and applies the outer learning rate β:
  θ* ← θ* − β · weighted_avg(meta_grads_across_cohort)

This is NOT standard FedAvg weight averaging. The "parameters" returned
by fit() are gradient tensors, not model weights.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import flwr as fl

from data.task_sampler import VoiceTaskSampler
from maml.engine import MAMLEngine
from maml.meta_eval import evaluate_adaptation_at_k
from models.wav2vec2_maml import Wav2Vec2MAML
from privacy.dp_meta import DPConfig, apply_dp_to_meta_gradient
from privacy.rdp_accountant import RDPAccountant


class MAMLClient(fl.client.NumPyClient):
    """
    Flower client implementing Per-FedAvg for Wav2Vec2 personalization.

    fit() computes the MAML meta-gradient (outer loop gradient ∇L_meta),
    applies DP sanitization (I5), and returns the sanitized gradient as
    numpy arrays. The server applies these as gradient updates to θ*.

    evaluate() runs k-step adaptation on local data and reports WER.
    This is the personalized evaluation — not the global model quality.
    """

    def __init__(
        self,
        node_id: str,
        model: Wav2Vec2MAML,
        engine: MAMLEngine,
        task_sampler: VoiceTaskSampler,
        dp_config: DPConfig,
        accountant: RDPAccountant,
    ) -> None:
        self.node_id = node_id
        self.model = model
        self.engine = engine
        self.sampler = task_sampler
        self.dp = dp_config
        self.accountant = accountant

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """
        Return encoder (outer loop) parameters as numpy arrays.

        I3: lm_head is intentionally excluded. It stays local to the node
        and is the personalization component.
        """
        return [
            p.data.cpu().numpy()
            for p in self.model.get_outer_loop_params()
        ]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set encoder parameters from server broadcast.

        Only outer loop (encoder) params are updated. lm_head is not
        touched — it retains its locally-adapted state.
        """
        for p, val in zip(self.model.get_outer_loop_params(), parameters):
            p.data = torch.tensor(val, dtype=p.dtype, device=p.device)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Run MAML inner + outer loop, return DP-sanitized meta-gradient.

        I4: Returns meta-gradients (∇L_meta_private) as 'parameters'.
            These have the same shape as the encoder parameters.
            The server's PerFedAvgStrategy applies outer_lr β:
              θ* ← θ* − β · weighted_avg(meta_grads)

        Workflow:
          1. Set encoder params from server broadcast
          2. Check DP budget — skip round if exhausted
          3. Sample task (support + query split, I6)
          4. Compute meta-gradient via MAML engine
          5. Extract outer-loop (encoder) portion
          6. Apply DP: clip + Gaussian noise (I5)
          7. Update RDP accountant
          8. Return sanitized gradient as numpy + metrics
        """
        self.set_parameters(parameters)

        if self.accountant.is_exhausted():
            print(f"[{self.node_id}] DP budget exhausted — skipping round")
            return self.get_parameters({}), 0, {"dp_exhausted": True}

        support_a, support_l, query_a, query_l = self.sampler.sample_task()

        meta_grads, query_loss = self.engine.compute_meta_gradient(
            support_a, support_l, query_a, query_l
        )

        outer_grads = self._extract_outer_grads(meta_grads)

        if self.dp.enabled:
            sanitized, grad_norm = apply_dp_to_meta_gradient(
                outer_grads, self.dp.C, self.dp.sigma
            )
            self.accountant.step(
                noise_multiplier=self.dp.sigma,
                sample_rate=self.dp.sample_rate,
            )
        else:
            sanitized = [g.detach() if g is not None else None for g in outer_grads]
            grad_norm = float(
                torch.cat([
                    g.flatten() for g in outer_grads if g is not None
                ]).norm(2).item()
            ) if any(g is not None for g in outer_grads) else 0.0

        grad_numpy = [
            g.cpu().numpy() if g is not None
            else np.zeros(p.shape, dtype=np.float32)
            for g, p in zip(sanitized, self.model.get_outer_loop_params())
        ]

        n_samples = len(support_a) + len(query_a)

        metrics: Dict = {
            "query_loss": float(query_loss),
            "grad_norm": float(grad_norm),
            "node_id": self.node_id,
        }
        if self.dp.enabled:
            metrics["epsilon"] = float(self.accountant.get_epsilon())

        return grad_numpy, n_samples, metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[float, int, Dict]:
        """
        k-step adaptation evaluation. Returns WER after personalization.

        Evaluates both θ* directly (k=0) and adapted model (k=config.k)
        to measure adaptation gain per round.
        """
        self.set_parameters(parameters)

        k = self.engine.config.k
        wer_results = evaluate_adaptation_at_k(
            self.model, self.engine, self.sampler,
            k_values=[0, k],
        )

        wer_0 = wer_results["k=0"]
        wer_k = wer_results[f"k={k}"]

        return wer_k, self.sampler.total_clips, {
            "wer": wer_k,
            "wer_0shot": wer_0,
            "adaptation_gain": round(wer_0 - wer_k, 4),
            "node_id": self.node_id,
        }

    def _extract_outer_grads(
        self,
        all_grads: List[Optional[torch.Tensor]],
    ) -> List[Optional[torch.Tensor]]:
        """
        all_grads is aligned with model.model.parameters() (ALL params).
        Extract only the outer loop (encoder) parameter gradients.

        Uses id(p) matching to correctly identify encoder params,
        since model.get_outer_loop_params() returns the same objects.
        """
        all_params = list(self.model.model.parameters())
        outer_param_ids = {id(p) for p in self.model.get_outer_loop_params()}

        return [
            g for p, g in zip(all_params, all_grads)
            if id(p) in outer_param_ids
        ]
