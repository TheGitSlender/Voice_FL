"""
MAML engine — FOMAML + ANIL (native PyTorch, no learn2learn at runtime).

compute_meta_gradient() runs K inner-loop steps on the support set (adapting
lm_head only), evaluates on the query set, and returns the outer-loop gradient
with respect to the ENCODER parameters only.

Modes:
  - "fomaml" (default): first-order approximation, no computation graph retained.
    Uses copy.deepcopy + manual SGD. Runs on RTX 4070 Super.
  - "second_order_ctc": true second-order MAML via higher.innerloop_ctx +
    custom differentiable CTC (ctc/differentiable_ctc.py). Requires higher>=0.2.1.
    The custom CTC replaces nn.CTCLoss with pure PyTorch ops that support
    double-backward (logsumexp, gather, cat, where). ~3-10x slower than FOMAML.

Algorithm: ANIL
  - Inner loop: adapt lm_head only (support set)
  - Outer loop: encoder meta-gradients (query set)
  - Gradients returned detached; caller applies them to the original model
  - Clips processed one at a time (batch_size=1, VRAM budget 3.5 GB/node)

Invariant I4: returns gradients, never weight updates.
Invariant I6: encoder.requires_grad stays True throughout.
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from data.task_sampler import Task
    from models.wav2vec2_maml import Wav2Vec2MAML

sys.path.insert(0, str(Path(__file__).parent.parent))

@dataclass
class MAMLConfig:
    """Configuration for MAMLEngine."""

    mode: str = "fomaml"
    k: int = 3
    inner_lr: float = 1e-4
    outer_lr: float = 2e-4
    support_size: int = 8
    query_size: int = 8
                        
    lora_rank: int = 8
    lora_alpha: float = 16.0

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

        del out, clip_grads, input_values, labels
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for acc in acc_grads:
        acc.div_(n_clips)

    return acc_grads, total_loss / n_clips

_VALID_MODES = ("fomaml", "second_order_ctc", "lora_maml")

class MAMLEngine:
    """
    MAML + ANIL engine for one episode.

    Supports three modes:
      - "fomaml"          first-order approximation (default, fast, RTX 4070 Super).
                          model must be Wav2Vec2MAML.
      - "second_order_ctc" true second-order via higher + custom differentiable CTC.
                          model must be Wav2Vec2MAML.
      - "lora_maml"       true second-order MAML over LoRA weights only (~295K params).
                          model must be LoRAWav2Vec2. Inner loop adapts LoRA + lm_head;
                          outer loop meta-gradients target LoRA only (for FL).

    Args:
        model:       Wav2Vec2MAML or LoRAWav2Vec2 instance
        processor:   Wav2Vec2Processor
        inner_steps: k (default 3)
        inner_lr:    α (default 1e-4)
        device:      torch device
        mode:        one of "fomaml", "second_order_ctc", "lora_maml"
    """

    def __init__(
        self,
        model,
        processor,
        inner_steps: int = 3,
        inner_lr: float = 1e-4,
        device: str | torch.device = "cpu",
        mode: str = "fomaml",
        max_audio_samples: int | None = None,
    ):
        if mode not in _VALID_MODES:
            raise ValueError(f"Unknown mode: {mode!r}. Use one of {_VALID_MODES}.")
        self.model = model
        self.processor = processor
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr
        self.device = torch.device(device)
        self.mode = mode
                                                                                 
        self.max_audio_samples = max_audio_samples

    def compute_meta_gradient(
        self,
        task: "Task",
        return_query_loss: bool = False,
        return_inner_losses: bool = False,
    ):
        """
        Run one MAML-ANIL episode.

        Args:
            task:               Task namedtuple (support + query splits)
            return_query_loss:  If True, include scalar query loss in return value
            return_inner_losses: If True (lora_maml only), include list[float] of
                                 per-step mean support losses in return value

        Returns (lora_maml mode, both flags True):
            (grads, query_loss, inner_losses)
        Returns (any mode, return_query_loss only):
            (grads, query_loss)
        Returns (default):
            grads
        """
        if self.mode == "second_order_ctc":
            return self._compute_meta_gradient_second_order(task, return_query_loss)
        if self.mode == "lora_maml":
            return self._compute_meta_gradient_lora_maml(
                task, return_query_loss, return_inner_losses
            )
        return self._compute_meta_gradient_fomaml(task, return_query_loss)

    def _compute_meta_gradient_fomaml(
        self,
        task: "Task",
        return_query_loss: bool = False,
    ) -> list[torch.Tensor] | tuple[list[torch.Tensor], float]:
        """FOMAML path: first-order approximation via deepcopy + manual SGD."""
        adapted = copy.deepcopy(self.model.model)
        adapted.train()
        dtype = next(adapted.parameters()).dtype

        lm_head_params = list(adapted.lm_head.parameters())
        encoder_params = list(adapted.wav2vec2.parameters())

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

        enc_grads, query_loss_val = _accumulate_grads_over_clips(
            adapted,
            task.query_audio,
            task.query_labels,
            encoder_params,
            self.processor,
            self.device,
            dtype,
        )

        result = [g.clone().detach() for g in enc_grads]

        del adapted, enc_grads
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if return_query_loss:
            return result, query_loss_val
        return result

    def _compute_meta_gradient_second_order(
        self,
        task: "Task",
        return_query_loss: bool = False,
    ) -> list[torch.Tensor] | tuple[list[torch.Tensor], float]:
        """Second-order MAML via higher + custom differentiable CTC.

        Uses higher.innerloop_ctx with track_higher_grads=True to retain the
        computation graph through inner-loop updates. The custom CTC loss
        (ctc.differentiable_ctc) provides the double-backward support that
        nn.CTCLoss lacks.

        Clips are processed one at a time (VRAM budget), with losses
        accumulated via the differentiable CTC's batch interface.
        """
        import higher

        from ctc.differentiable_ctc import ctc_loss_differentiable

        dtype = next(self.model.model.parameters()).dtype
                                               
        inner_params = list(self.model.model.lm_head.parameters())
        inner_opt = torch.optim.SGD(inner_params, lr=self.inner_lr)

        with higher.innerloop_ctx(
            self.model.model,
            inner_opt,
            copy_initial_weights=False,
            track_higher_grads=True,
            override={"lr": [self.inner_lr]},
        ) as (fmodel, diffopt):
                                                
            for _step in range(self.inner_steps):
                step_losses = []
                for audio, text in zip(task.support_audio, task.support_labels):
                    input_values = _encode_audio(audio, self.processor, self.device, dtype)
                    labels = _encode_labels(text, self.processor, self.device)

                    out = fmodel(input_values=input_values)
                    logits = out.logits                    
                    T_frames = logits.shape[1]

                    input_lengths = torch.tensor([T_frames], device=self.device)
                    target_lengths = torch.tensor(
                        [(labels[0] != -100).sum().item()], device=self.device
                    )

                    loss_clip = ctc_loss_differentiable(
                        logits, labels, input_lengths, target_lengths, blank=0
                    )
                    step_losses.append(loss_clip)

                step_loss = torch.stack(step_losses).mean()
                diffopt.step(step_loss)

            query_losses = []
            for audio, text in zip(task.query_audio, task.query_labels):
                input_values = _encode_audio(audio, self.processor, self.device, dtype)
                labels = _encode_labels(text, self.processor, self.device)

                out = fmodel(input_values=input_values)
                logits = out.logits
                T_frames = logits.shape[1]

                input_lengths = torch.tensor([T_frames], device=self.device)
                target_lengths = torch.tensor(
                    [(labels[0] != -100).sum().item()], device=self.device
                )

                loss_clip = ctc_loss_differentiable(
                    logits, labels, input_lengths, target_lengths, blank=0
                )
                query_losses.append(loss_clip)

            query_loss = torch.stack(query_losses).mean()

        encoder_params = list(self.model.model.wav2vec2.parameters())
        meta_grads_raw = torch.autograd.grad(
            query_loss, encoder_params, allow_unused=True
        )

        result = []
        for g in meta_grads_raw:
            if g is not None:
                result.append(g.clone().detach())
            else:
                result.append(torch.zeros(1, device=self.device))

        query_loss_val = query_loss.item()

        del meta_grads_raw
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if return_query_loss:
            return result, query_loss_val
        return result

    def _compute_meta_gradient_lora_maml(
        self,
        task: "Task",
        return_query_loss: bool = False,
        return_inner_losses: bool = False,
    ):
        """True second-order MAML over LoRA + lm_head (FedLoRA-MAML).

        Uses higher.innerloop_ctx with track_higher_grads=True to retain the
        computation graph through inner-loop steps. ~320K trainable params are
        in the inner loop (LoRA A/B + lm_head), making full second-order MAML
        feasible on an RTX 4070 Super.

        This is TRUE second-order MAML — not a first-order approximation.
        The Hessian captures:
          (a) LoRA subspace curvature — how adaptation changes gradient landscape
          (b) CTC alignment curvature — how alignment paths shift during adaptation
        Both terms flow through ctc/differentiable_ctc.py.

        No ANIL split: both LoRA and lm_head are outer-loop meta-gradient targets
        (aggregated by FL server). self.model.get_outer_loop_params() returns all
        trainable params.

        NOTE: do NOT call model.train() — wav2vec2's dropout + higher's functional
        patching produces NaN logits. Gradients flow correctly in eval mode.

        Requires higher>=0.2.1.
        """
        import higher

        from ctc.differentiable_ctc import ctc_loss_differentiable

        dtype = next(self.model.model.parameters()).dtype

        WAV_STRIDE = 320

        def _clip_audio(a: torch.Tensor, S: int) -> torch.Tensor | None:
            """Truncate to max_audio_samples, but skip if T < 2*S−1 after truncation.

            CTC requires T ≥ 2*S−1 (blank between every label). If the clip would
            be too short after truncation, return None to signal skip.
            """
            clipped = a[: self.max_audio_samples] if (
                self.max_audio_samples is not None and len(a) > self.max_audio_samples
            ) else a
            T_est = len(clipped) // WAV_STRIDE
            if T_est < 2 * S - 1:
                return None
            return clipped

        inner_params = self.model.get_outer_loop_params()
        inner_opt = torch.optim.SGD(inner_params, lr=self.inner_lr)
        inner_step_losses: list[float] = []

        with higher.innerloop_ctx(
            self.model.model,
            inner_opt,
            copy_initial_weights=False,
            track_higher_grads=True,
            override={"lr": [self.inner_lr]},
        ) as (fmodel, diffopt):

            for _step in range(self.inner_steps):
                step_losses = []
                for audio, text in zip(task.support_audio, task.support_labels):
                    labels = _encode_labels(text, self.processor, self.device)
                    S = int((labels[0] != -100).sum().item())
                    clipped = _clip_audio(audio, S)
                    if clipped is None:
                        del labels
                        continue

                    input_values = _encode_audio(clipped, self.processor, self.device, dtype)
                    out = fmodel(input_values=input_values)
                    logits = out.logits
                    T_frames = logits.shape[1]

                    loss_clip = ctc_loss_differentiable(
                        logits,
                        labels,
                        torch.tensor([T_frames], device=self.device),
                        torch.tensor([S], device=self.device),
                        blank=0,
                    )
                    step_losses.append(loss_clip)

                if not step_losses:
                    inner_step_losses.append(float("nan"))
                    continue
                step_mean = torch.stack(step_losses).mean()
                inner_step_losses.append(step_mean.item())
                diffopt.step(step_mean)

            query_losses = []
            for audio, text in zip(task.query_audio, task.query_labels):
                labels = _encode_labels(text, self.processor, self.device)
                S = int((labels[0] != -100).sum().item())
                clipped = _clip_audio(audio, S)
                if clipped is None:
                    del labels
                    continue

                input_values = _encode_audio(clipped, self.processor, self.device, dtype)
                out = fmodel(input_values=input_values)
                logits = out.logits
                T_frames = logits.shape[1]

                loss_clip = ctc_loss_differentiable(
                    logits,
                    labels,
                    torch.tensor([T_frames], device=self.device),
                    torch.tensor([S], device=self.device),
                    blank=0,
                )
                query_losses.append(loss_clip)

            if not query_losses:
                outer_params_early = self.model.get_outer_loop_params()
                result_zero = [torch.zeros_like(p) for p in outer_params_early]
                if return_query_loss and return_inner_losses:
                    return result_zero, 0.0, inner_step_losses
                if return_query_loss:
                    return result_zero, 0.0
                if return_inner_losses:
                    return result_zero, inner_step_losses
                return result_zero

            query_loss = torch.stack(query_losses).mean()

        outer_params = self.model.get_outer_loop_params()
        meta_grads_raw = torch.autograd.grad(
            query_loss, outer_params, allow_unused=True
        )

        result: list[torch.Tensor] = []
        for p, g in zip(outer_params, meta_grads_raw):
            if g is not None:
                result.append(g.clone().detach())
            else:
                result.append(torch.zeros_like(p))

        query_loss_val = query_loss.item()

        del meta_grads_raw
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if return_query_loss and return_inner_losses:
            return result, query_loss_val, inner_step_losses
        if return_query_loss:
            return result, query_loss_val
        if return_inner_losses:
            return result, inner_step_losses
        return result

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
    perturb_lm_head_std: float = 0.0,
) -> float:
    """WER with zero adaptation steps (no perturbation by default).

    perturb_lm_head_std=0.0 is the correct default for evaluating on genuinely
    unseen speakers (VCTK). The pretrained model has NOT seen these speakers,
    so the k=0 WER gap is real without needing artificial noise injection.

    Set perturb_lm_head_std > 0 only if you explicitly need the legacy ANIL
    noise-recovery protocol (not recommended for valid meta-learning evaluation).
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
    perturb_lm_head_std: float = 0.0,
) -> float:
    """WER after k inner-loop adaptation steps on support set (no perturbation by default).

    The name 'k3' is historical; pass any inner_steps value.
    With perturb_lm_head_std=0.0 (default), this evaluates genuine adaptation
    to an unseen speaker from the VCTK dataset.
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

def compute_wer_at_k(
    model: "Wav2Vec2MAML",
    processor,
    support_audio: list,
    support_labels: list[str],
    query_audio: list,
    query_labels: list[str],
    k: int,
    inner_lr: float = 1e-4,
    device: torch.device = torch.device("cpu"),
) -> float:
    """WER at exactly k adaptation steps, no perturbation.

    Convenience wrapper over compute_wer_k0 (k=0) and compute_wer_k3 (k>0).
    Use this for adaptation curve generation on genuinely unseen VCTK speakers.
    """
    if k == 0:
        return compute_wer_k0(model, processor, query_audio, query_labels, device,
                               perturb_lm_head_std=0.0)
    return compute_wer_k3(model, processor, support_audio, support_labels,
                           query_audio, query_labels, k, inner_lr, device,
                           perturb_lm_head_std=0.0)
