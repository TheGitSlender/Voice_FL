"""
maml/meta_train.py — Centralized MAML meta-training

Validates that the MAML algorithm works before adding FL complexity.
All 20 node datasets are available locally in centralized mode.

Expected result after 50 epochs:
  WER(k=3) < WER(k=0) on at least 12/20 held-out speakers

If this gate fails, something is wrong with the MAML implementation
before federating. Debug here, not in the FL simulation.

Run:
    python maml/meta_train.py
"""

import random
from typing import List, Optional

import mlflow
import torch

from data.task_sampler import VoiceTaskSampler
from maml.engine import MAMLConfig, MAMLEngine
from models.wav2vec2_maml import Wav2Vec2MAML


def run_centralized_meta_training(
    model: Wav2Vec2MAML,
    engine: MAMLEngine,
    task_samplers: List[VoiceTaskSampler],
    n_meta_epochs: int = 50,
    tasks_per_epoch: int = 16,
    outer_optimizer: Optional[torch.optim.Optimizer] = None,
    mlflow_run=None,
) -> List[float]:
    """
    Centralized MAML meta-training (no FL, no DP).

    All 20 node datasets are available locally — used to validate that
    MAML inner loop actually personalizes before adding federated complexity.

    Each epoch:
      1. Sample tasks_per_epoch task samplers (random subset of nodes)
      2. Compute meta-gradient per task via engine.compute_meta_gradient()
      3. Accumulate meta-gradients and average across tasks
      4. Apply outer optimizer step

    Returns: list of per-epoch losses for tracking.
    """
    if outer_optimizer is None:
        outer_optimizer = torch.optim.AdamW(
            model.model.parameters(),
            lr=engine.config.outer_lr,
            weight_decay=0.01,
        )

    epoch_losses: List[float] = []

    for epoch in range(n_meta_epochs):
        outer_optimizer.zero_grad()
        epoch_loss = 0.0

        n_tasks = min(tasks_per_epoch, len(task_samplers))
        sampled = random.sample(task_samplers, n_tasks)

        for sampler in sampled:
            support_a, support_l, query_a, query_l = sampler.sample_task()
            meta_grads, q_loss = engine.compute_meta_gradient(
                support_a, support_l, query_a, query_l
            )
            epoch_loss += q_loss

            # Accumulate meta-gradients into .grad attribute
            for p, g in zip(model.model.parameters(), meta_grads):
                if g is None:
                    continue
                if p.grad is None:
                    p.grad = g.clone()
                else:
                    p.grad = p.grad + g.clone()

        # Average over tasks
        for p in model.model.parameters():
            if p.grad is not None:
                p.grad = p.grad / n_tasks

        outer_optimizer.step()

        avg_loss = epoch_loss / n_tasks
        epoch_losses.append(avg_loss)

        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d}/{n_meta_epochs} | meta_loss: {avg_loss:.4f}")
            if mlflow_run is not None:
                mlflow.log_metric("meta_train_loss", avg_loss, step=epoch)

    return epoch_losses


if __name__ == "__main__":
    """
    Quick 5-epoch centralized run to verify MAML works.
    Gate: WER(k=3) < WER(k=0) on >=12/20 nodes.
    """
    import os
    from pathlib import Path

    from transformers import Wav2Vec2Processor

    from maml.meta_eval import evaluate_adaptation_at_k

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    node_dirs = sorted(Path("data/nodes").glob("node_*"))
    if not node_dirs:
        print("No node dirs found. Run: python data/features.py first.")
        exit(1)

    task_samplers = [
        VoiceTaskSampler(str(d), K=4, Q=4, processor=processor, device=device)
        for d in node_dirs
    ]

    model = Wav2Vec2MAML(mode="anil").to(device)
    config = MAMLConfig(mode="fomaml", k=3, inner_lr=1e-4, outer_lr=2e-4, use_bf16=(device == "cuda"))
    engine = MAMLEngine(model, config)

    print("Running 5 epochs of centralized FOMAML...")
    losses = run_centralized_meta_training(
        model, engine, task_samplers,
        n_meta_epochs=5,
        tasks_per_epoch=4,
    )

    print("\nEvaluating adaptation at k=0 and k=3...")
    gains = 0
    for sampler in task_samplers[:5]:  # quick check on 5 nodes
        wers = evaluate_adaptation_at_k(model, engine, sampler, k_values=[0, 3])
        gain = wers["k=0"] - wers["k=3"]
        gains += 1 if gain > 0 else 0
        print(f"  {sampler.node_dir}: WER k=0={wers['k=0']:.3f} k=3={wers['k=3']:.3f} gain={gain:.3f}")

    print(f"\nNodes with WER(k=3) < WER(k=0): {gains}/5")
    print("Gate (full run): need >=12/20 nodes to show positive adaptation gain")
