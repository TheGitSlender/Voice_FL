"""
CTC loss in pure PyTorch — supports create_graph=True (double-backward).

PyTorch's nn.CTCLoss calls aten::_ctc_loss_backward, a hand-written CUDA
kernel not registered for second derivatives. This implementation replaces
it entirely with native differentiable ops: logsumexp, gather, cat, where.

Algorithm: standard CTC forward pass in log-space with extended targets
(blank insertion), vectorized over S_ext at each time step.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

def _safe_logsumexp(x: torch.Tensor, dim: int) -> torch.Tensor:
    """logsumexp that returns NEG_INF (not NaN) when all inputs are NEG_INF.

    Standard logsumexp computes max + log(sum(exp(x - max))). When all inputs
    are -inf, max=-inf and x-max=NaN, producing NaN gradients during
    double-backward. This version detects all-neginf slices and returns a
    clean NEG_INF with no NaN in the gradient.
    """
    NEG_INF = torch.finfo(x.dtype).min / 2
    max_val = x.detach().amax(dim=dim, keepdim=True)
    all_neginf = max_val.squeeze(dim) < -1e37
    safe_max = max_val.clamp(min=-1e38)
    exp_shifted = torch.exp(x - safe_max)
    sum_exp = exp_shifted.sum(dim=dim)
    out = safe_max.squeeze(dim) + torch.log(sum_exp.clamp(min=1e-38))
    return torch.where(all_neginf, torch.full_like(out, NEG_INF), out)

class DifferentiableCTC(nn.Module):
    """
    CTC loss for a single sample — supports create_graph=True.

    Input:
      log_probs:  (T, C) float tensor — log probabilities per frame
      targets:    (S,)   long tensor  — target label ids (not blank-expanded)
      blank:      int    — blank token index (default 0)
    Output:
      loss:       scalar — negative log likelihood, supports double-backward
    """

    def forward(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        blank: int = 0,
    ) -> torch.Tensor:
        T, _C = log_probs.shape
        S = targets.shape[0]
        S_ext = 2 * S + 1
        device = log_probs.device
        dtype = log_probs.dtype
        NEG_INF = torch.finfo(dtype).min / 2

        if S > 0:
            n_repeats = (targets[1:] == targets[:-1]).sum().item()
        else:
            n_repeats = 0
        min_T = S + n_repeats
        if T < min_T:
            raise ValueError(
                f"Sequence too short: T={T} frames cannot encode S={S} labels "
                f"(minimum {min_T} frames required). "
                f"Use longer utterances or smaller S."
            )

        targets_ext = torch.full(
            (S_ext,), blank, dtype=torch.long, device=device
        )
        targets_ext[1::2] = targets                           

        skip_invalid = torch.zeros(S_ext, dtype=torch.bool, device=device)
        if S_ext >= 3:
            skip_invalid[2:] = targets_ext[2:] == targets_ext[:-2]

        all_emissions = log_probs[:, targets_ext]              

        e0 = all_emissions[0]            
        state0 = e0[0:1]                         
        if S_ext > 1:
            state1 = e0[1:2]                               
            rest = torch.full((S_ext - 2,), NEG_INF, dtype=dtype, device=device)
            alpha_t = torch.cat([state0, state1, rest])
        else:
            alpha_t = state0

        alphas = [alpha_t]

        for t in range(1, T):
            prev = alphas[t - 1]                             

            stay = prev

            advance = torch.cat([prev.new_full((1,), NEG_INF), prev[:-1]])

            if S_ext >= 3:
                skip_raw = torch.cat([prev.new_full((2,), NEG_INF), prev[:-2]])
                skip = skip_raw.masked_fill(skip_invalid, NEG_INF)
            else:
                skip = prev.new_full((S_ext,), NEG_INF)

            combined = torch.stack([stay, advance, skip], dim=0)              
            alpha_new = _safe_logsumexp(combined, dim=0) + all_emissions[t]

            alphas.append(alpha_new)

        alpha_final = alphas[-1]

        if S_ext >= 2:
            final_states = torch.stack([alpha_final[-2], alpha_final[-1]])
        else:
            final_states = alpha_final[-1:]

        log_likelihood = _safe_logsumexp(final_states, dim=0)
        return -log_likelihood

def ctc_loss_differentiable(
    logits: torch.Tensor,
    labels: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int = 0,
) -> torch.Tensor:
    """
    Batch CTC loss supporting double-backward.

    logits:         (B, T, C) — raw logits from wav2vec2 (before log_softmax)
    labels:         (B, S)    — target ids, -100 for padding
    input_lengths:  (B,)      — actual frame counts
    target_lengths: (B,)      — actual target lengths
    blank:          int       — blank token index

    Returns mean loss over batch. Processes samples sequentially (B is small).
    """
    log_probs = F.log_softmax(logits, dim=-1)             
    ctc = DifferentiableCTC()
    losses = []

    for b in range(logits.shape[0]):
        T = input_lengths[b].item()
        S = target_lengths[b].item()

        lp = log_probs[b, :T, :]          
        tgt = labels[b]

        tgt_clean = tgt[tgt != -100][:S]

        if tgt_clean.shape[0] == 0:
            continue

        loss_b = ctc(lp, tgt_clean, blank=blank)
        losses.append(loss_b)

    if not losses:
        raise ValueError(
            "All samples in batch have empty targets. "
            "Cannot compute CTC loss — check data pipeline."
        )

    return torch.stack(losses).mean()
