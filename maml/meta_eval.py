"""
maml/meta_eval.py — MAML adaptation evaluation

Measures WER at each k adaptation step count (k=0,1,3,5,10).
The adaptation curve WER(k) should decrease as k increases.
A flat or rising curve means the inner loop is not personalizing.

WER is computed using jiwer after greedy CTC decoding.
"""

from typing import Dict, List

import jiwer
import torch

from data.task_sampler import VoiceTaskSampler
from maml.engine import MAMLEngine
from models.wav2vec2_maml import Wav2Vec2MAML


def evaluate_adaptation_at_k(
    model: Wav2Vec2MAML,
    engine: MAMLEngine,
    task_sampler: VoiceTaskSampler,
    k_values: List[int] = None,
) -> Dict[str, float]:
    """
    Measure WER at each k adaptation step count.

    For k=0: evaluate θ* directly (no adaptation).
    For k>0: adapt k steps on a fresh support set, then evaluate.

    Each call uses a fresh sample — results vary across calls.
    For stable estimates, average over multiple calls.

    Returns: {"k=0": 0.14, "k=1": 0.09, "k=3": 0.07, ...}
    """
    if k_values is None:
        k_values = [0, 1, 3, 5, 10]

    results: Dict[str, float] = {}

    for k in k_values:
        if k == 0:
            eval_audio, eval_labels = task_sampler.sample_eval_batch()
            wer = _compute_wer(model.model, model.processor, eval_audio, eval_labels)
        else:
            support_a, support_l, _, _ = task_sampler.sample_task()
            adapted_model = engine.adapt(support_a, support_l, k=k)
            eval_audio, eval_labels = task_sampler.sample_eval_batch()
            wer = _compute_wer(adapted_model, model.processor, eval_audio, eval_labels)

        results[f"k={k}"] = round(wer, 4)

    return results


def _compute_wer(
    model,
    processor,
    audio: torch.Tensor,
    label_ids: torch.Tensor,
) -> float:
    """Decode predictions and compute WER via jiwer."""
    model.eval()
    with torch.no_grad():
        logits = model(input_values=audio).logits

    pred_ids = torch.argmax(logits, dim=-1)
    predictions = processor.batch_decode(pred_ids)

    # Decode ground truth — replace -100 padding with pad_token_id before decoding
    label_ids_clean = label_ids.clone()
    label_ids_clean[label_ids_clean == -100] = processor.tokenizer.pad_token_id
    references = processor.batch_decode(label_ids_clean, group_tokens=False)

    return jiwer.wer(references, predictions)
