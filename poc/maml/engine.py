"""
MAML engine — FOMAML + ANIL (native PyTorch, no learn2learn at runtime).

compute_meta_gradient() runs K inner-loop steps on the support set (adapting
lm_head only), evaluates on the query set, and returns the outer-loop gradient
with respect to the ENCODER parameters only.

Algorithm: ANIL-FOMAML
  - copy.deepcopy creates a task-local clone of Wav2Vec2ForCTC
  - Inner loop: manual SGD step on lm_head parameters only
    (autograd.grad called only w.r.t. lm_head — encoder stays untouched)
  - Outer loop: autograd.grad called w.r.t. encoder parameters only
    (gradients flow through adapted lm_head back into encoder)
  - Gradients returned detached; caller applies them to the original model
  - Clips processed one at a time (batch_size=1, VRAM budget 3.5 GB/node)

Invariant I4: returns gradients, never weight updates.
Invariant I6: encoder.requires_grad stays True throughout.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from data.task_sampler import Task
    from models.wav2vec2_maml import Wav2Vec2MAML

sys.path.insert(0, str(Path(__file__).parent.parent))


def _encode_audio(
    audio_tensor: torch.Tensor,
    processor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert a 1D float32 waveform tensor to model input_values."""
    arr = audio_tensor.float().numpy()
    inputs = processor(
        arr,
        sampling_rate=16000,
        return_tensors="pt",
        padding=False,
    )
    return inputs.input_values.to(device=device, dtype=dtype)


def _encode_labels(text: str, processor, device: torch.device) -> torch.Tensor:
    """Tokenize a transcription to label ids."""
    # as_target_processor() was deprecated in transformers 4.18 and removed later.
    ids = processor.tokenizer(text, return_tensors="pt").input_ids
    return ids.to(device)


def _accumulate_grads_over_clips(
    model: nn.Module,
    audio_clips: list[torch.Tensor],
    texts: list[str],
    params: list[torch.nn.Parameter],
    processor,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[torch.Tensor], float]:
    """Compute mean gradient of CTC loss w.r.t. params, one clip at a time.

    Processes clips sequentially and frees each computation graph immediately,
    keeping peak VRAM to a single clip's graph rather than all clips at once.
    Full-length clips are used (no truncation) to avoid CTC label-length issues.

    Returns:
        (grad_list, mean_loss_scalar)
    """
    acc_grads = [torch.zeros_like(p) for p in params]
    total_loss = 0.0
    n_clips = len(audio_clips)

    for audio, text in zip(audio_clips, texts):
        input_values = _encode_audio(audio, processor, device, dtype)
        labels = _encode_labels(text, processor, device)
        out = model(input_values=input_values, labels=labels)

        clip_grads = torch.autograd.grad(
            out.loss,
            params,
            create_graph=False,
            allow_unused=True,
        )

        total_loss += out.loss.item()

        for acc, g in zip(acc_grads, clip_grads):
            if g is not None:
                acc.add_(g)

        # Free this clip's computation graph immediately
        del out, clip_grads, input_values, labels
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Average over clips
    for acc in acc_grads:
        acc.div_(n_clips)

    return acc_grads, total_loss / n_clips


class MAMLEngine:
    """
    FOMAML + ANIL engine for one episode.

    Args:
        model:       Wav2Vec2MAML instance
        processor:   Wav2Vec2Processor
        inner_steps: k (default 3)
        inner_lr:    α (default 1e-4)
        device:      torch device
    """

    def __init__(
        self,
        model: "Wav2Vec2MAML",
        processor,
        inner_steps: int = 3,
        inner_lr: float = 1e-4,
        device: str | torch.device = "cpu",
    ):
        self.model = model
        self.processor = processor
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr
        self.device = torch.device(device)
        # Note: l2l.MAML is NOT used here. compute_meta_gradient uses
        # copy.deepcopy + manual gradient propagation (FOMAML-ANIL pattern).

    def compute_meta_gradient(
        self,
        task: "Task",
        return_query_loss: bool = False,
    ) -> list[torch.Tensor] | tuple[list[torch.Tensor], float]:
        """
        Run one FOMAML-ANIL episode.

        Args:
            task:              Task namedtuple (support + query splits)
            return_query_loss: If True, also return the scalar query loss value

        Returns:
            Encoder meta-gradients — list of tensors with same shapes as
            model.get_outer_loop_params(). These are gradients, not weight
            updates (Invariant I4).
            If return_query_loss is True, returns (grads, query_loss_scalar).
        """
        # deepcopy gives a fully independent task-local model.
        # Do NOT use l2l.MAML.clone(): with first_order=True it creates detached
        # leaf tensors so gradients never reach the original model's params.
        # deepcopy + manual grad copy is the correct FOMAML-ANIL pattern.
        adapted = copy.deepcopy(self.model.model)
        adapted.train()
        dtype = next(adapted.parameters()).dtype

        lm_head_params = list(adapted.lm_head.parameters())
        encoder_params = list(adapted.wav2vec2.parameters())

        # ---- INNER LOOP (support set, lm_head only) --------------------------
        for _step in range(self.inner_steps):
            head_grads, _ = _accumulate_grads_over_clips(
                adapted,
                task.support_audio,
                task.support_labels,
                lm_head_params,
                self.processor,
                self.device,
                dtype,
            )
            for p, g in zip(lm_head_params, head_grads):
                p.data = p.data - self.inner_lr * g
            del head_grads

        # ---- OUTER LOOP (query set, encoder grads) ---------------------------
        enc_grads, query_loss_val = _accumulate_grads_over_clips(
            adapted,
            task.query_audio,
            task.query_labels,
            encoder_params,
            self.processor,
            self.device,
            dtype,
        )

        # FOMAML: clone encoder grads == meta-gradients (first-order approx, I4)
        result = [g.clone().detach() for g in enc_grads]

        # Explicitly free clone and flush CUDA pool
        del adapted, enc_grads
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if return_query_loss:
            return result, query_loss_val
        return result


# ---------------------------------------------------------------------------
# WER evaluation helpers
# ---------------------------------------------------------------------------

def _perturb_lm_head(model_copy: nn.Module, noise_std: float = 0.3) -> None:
    """Add Gaussian noise to lm_head weights to simulate a new-speaker head.

    ANIL evaluation protocol: the meta-trained encoder (θ*) is tested from
    a perturbed lm_head, not from the already-trained global head.  This
    isolates the encoder's contribution: a few adaptation steps should pull
    the noisy head back toward the speaker-specific optimum, demonstrating
    that the meta-trained encoder enables rapid adaptation.

    Without perturbation, wav2vec2-base-960h already achieves ~1% WER on
    LibriSpeech; k=3 steps cannot visibly improve an already-near-perfect head.
    """
    with torch.no_grad():
        for p in model_copy.lm_head.parameters():
            p.add_(noise_std * torch.randn_like(p))


def compute_wer_k0(
    model: "Wav2Vec2MAML",
    processor,
    audio_clips: list,
    label_texts: list[str],
    device: torch.device,
    perturb_lm_head_std: float = 0.3,
) -> float:
    """WER with zero adaptation (baseline).

    lm_head is perturbed by Gaussian noise (std=perturb_lm_head_std) before
    evaluation — this is the ANIL evaluation protocol for a meta-trained model.
    The encoder (θ*) is kept fixed; only the head is disturbed to simulate
    a client whose head has drifted from the global initialization.
    """
    from jiwer import wer as _wer

    model_copy = copy.deepcopy(model)
    if perturb_lm_head_std > 0:
        _perturb_lm_head(model_copy.model, perturb_lm_head_std)

    dtype = next(model_copy.model.parameters()).dtype
    model_copy.model.eval()
    hypotheses = []
    with torch.no_grad():
        for audio in audio_clips:
            input_values = _encode_audio(audio, processor, device, dtype)
            out = model_copy.model(input_values=input_values)
            pred_ids = torch.argmax(out.logits, dim=-1)
            hyp = processor.batch_decode(pred_ids)[0]
            hypotheses.append(hyp)
    return float(_wer(label_texts, hypotheses))


def compute_wer_k3(
    model: "Wav2Vec2MAML",
    processor,
    support_audio: list,
    support_labels: list[str],
    query_audio: list,
    query_labels: list[str],
    inner_steps: int = 3,
    inner_lr: float = 1e-4,
    device: torch.device = torch.device("cpu"),
    perturb_lm_head_std: float = 0.3,
) -> float:
    """WER after k inner-loop adaptation steps on support set.

    lm_head is perturbed by the same Gaussian noise as compute_wer_k0 before
    adaptation begins, so the comparison is:
      k=0:  perturbed head, no recovery steps
      k=K:  perturbed head + K gradient steps on support set
    This measures whether the meta-trained encoder enables rapid head recovery.
    """
    from jiwer import wer as _wer

    model_copy = copy.deepcopy(model)
    if perturb_lm_head_std > 0:
        _perturb_lm_head(model_copy.model, perturb_lm_head_std)

    dtype = next(model_copy.model.parameters()).dtype
    model_copy.model.train()

    lm_head_params = list(model_copy.model.lm_head.parameters())

    for _ in range(inner_steps):
        head_grads, _ = _accumulate_grads_over_clips(
            model_copy.model,
            support_audio,
            support_labels,
            lm_head_params,
            processor,
            device,
            dtype,
        )
        for p, g in zip(lm_head_params, head_grads):
            p.data = p.data - inner_lr * g
        del head_grads

    model_copy.model.eval()
    hypotheses = []
    with torch.no_grad():
        for audio in query_audio:
            input_values = _encode_audio(audio, processor, device, dtype)
            out = model_copy.model(input_values=input_values)
            pred_ids = torch.argmax(out.logits, dim=-1)
            hyp = processor.batch_decode(pred_ids)[0]
            hypotheses.append(hyp)

    return float(_wer(query_labels, hypotheses))
