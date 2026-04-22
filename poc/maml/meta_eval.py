"""
Evaluation helpers for WER computation and adaptation comparison.

Used by both meta_train.py (gate check) and evaluation/eval_poc.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

def evaluate_node(
    model,
    processor,
    sampler,
    inner_steps: int = 3,
    inner_lr: float = 1e-4,
    device=None,
    n_tasks: int = 5,
) -> dict:
    """
    Run n_tasks episodes for a single node.
    Returns mean WER at k=0 and k=3 across episodes.
    """
    import torch
    from maml.engine import compute_wer_k0, compute_wer_k3

    if device is None:
        device = torch.device("cpu")

    wer0_list = []
    wer3_list = []
    for _ in range(n_tasks):
        task = sampler.sample_task()
        w0 = compute_wer_k0(model, processor, task.query_audio, task.query_labels, device)
        w3 = compute_wer_k3(
            model, processor,
            task.support_audio, task.support_labels,
            task.query_audio, task.query_labels,
            inner_steps, inner_lr, device,
        )
        wer0_list.append(w0)
        wer3_list.append(w3)

    return {
        "wer_k0_mean": sum(wer0_list) / len(wer0_list),
        "wer_k3_mean": sum(wer3_list) / len(wer3_list),
        "wer_k0_list": wer0_list,
        "wer_k3_list": wer3_list,
        "adaptation_passed": sum(wer3_list) / len(wer3_list) < sum(wer0_list) / len(wer0_list),
    }

def load_theta_star(model, checkpoint_path: str | Path) -> None:
    """Load encoder state dict from a checkpoint into model."""
    import torch
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.set_encoder_state_dict(state_dict)
