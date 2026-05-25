"""GPU-optimized CTC loss — supports create_graph=True (second-order MAML).

Two execution modes:
  1. Sequential (default): Optimized sequential DP loop, same O(T × S_ext)
     memory as v1 but ~2–5× faster (native logsumexp, fewer allocations).
     Safe for MAML with create_graph=True — negligible VRAM overhead.

  2. Parallel scan (scan=True): Log-semiring parallel prefix scan that
     reduces the loop from O(T) to O(log₂ T). ~14–75× faster but uses
     O(T × S_ext³) memory with create_graph=True. Only recommended when
     VRAM is abundant or S_ext is small.

Reference implementation: ctc/differentiable_ctc.py (v1, sequential loop).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_neg_inf(dtype: torch.dtype) -> float:
    """Return a large negative value that won't overflow during computation.

    Uses a finite sentinel so that torch.logsumexp works correctly without
    producing NaN when all inputs are NEG_INF (unlike torch.finfo.min which
    causes -inf - (-inf) = NaN in the exp step).
    """
    if dtype == torch.float64:
        return -1e100
    return -1e20  # float32, float16, bfloat16


# ---------------------------------------------------------------------------
# Log-semiring operations (used by parallel scan mode)
# ---------------------------------------------------------------------------

def _log_semiring_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batched matrix multiply in the (log, logaddexp) semiring.

    a: (N, S, S), b: (N, S, S)
    Returns (N, S, S) where C[n, i, j] = logsumexp_k(a[n, i, k] + b[n, k, j])
    """
    return torch.logsumexp(a.unsqueeze(-1) + b.unsqueeze(-3), dim=-2)


def _log_semiring_matvec(mat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Matrix-vector product in the (log, logaddexp) semiring.

    mat: (S, S), vec: (S,)
    Returns (S,) where result[i] = logsumexp_j(mat[i, j] + vec[j])
    """
    return torch.logsumexp(mat + vec.unsqueeze(0), dim=1)


# ---------------------------------------------------------------------------
# Transition matrix construction (used by parallel scan mode)
# ---------------------------------------------------------------------------

def _build_transition_mask(
    skip_invalid: torch.Tensor,
    S_ext: int,
    device: torch.device,
) -> torch.Tensor:
    """Build (S_ext, S_ext) bool mask for valid CTC transitions.

    mask[s, j] = True means a valid transition from state j to state s:
      - Stay:    j = s     (always valid)
      - Advance: j = s - 1 (valid for s >= 1)
      - Skip:    j = s - 2 (valid for s >= 2 and not skip_invalid[s])
    """
    mask = torch.zeros(S_ext, S_ext, dtype=torch.bool, device=device)

    idx = torch.arange(S_ext, device=device)
    mask[idx, idx] = True  # stay

    if S_ext >= 2:
        adv = torch.arange(1, S_ext, device=device)
        mask[adv, adv - 1] = True  # advance

    if S_ext >= 3:
        skip_src = torch.arange(2, S_ext, device=device)
        valid = ~skip_invalid[2:]
        s_valid = skip_src[valid]
        if s_valid.numel() > 0:
            mask[s_valid, s_valid - 2] = True  # skip

    return mask


def _build_transition_matrices(
    emissions: torch.Tensor,
    transition_mask: torch.Tensor,
    neg_inf: float,
) -> torch.Tensor:
    """Build (T, S_ext, S_ext) transition matrices from emissions.

    emissions:       (T, S_ext) — log_probs indexed by targets_ext
    transition_mask: (S_ext, S_ext) — boolean mask for valid transitions
    neg_inf:         value for impossible transitions

    Returns (T, S_ext, S_ext) where mats[t, s, j] = emissions[t, s] if
    transition_mask[s, j] else neg_inf.
    """
    neg_inf_val = torch.tensor(neg_inf, dtype=emissions.dtype, device=emissions.device)
    return torch.where(
        transition_mask.unsqueeze(0),
        emissions.unsqueeze(-1),
        neg_inf_val,
    )


# ---------------------------------------------------------------------------
# Parallel reduce (used by scan mode)
# ---------------------------------------------------------------------------

def _make_identity(n: int, S_ext: int, dtype: torch.dtype, device: torch.device,
                   neg_inf: float) -> torch.Tensor:
    """Create n log-semiring identity matrices (S_ext, S_ext)."""
    eye = torch.full((n, S_ext, S_ext), neg_inf, dtype=dtype, device=device)
    idx = torch.arange(S_ext, device=device)
    eye[:, idx, idx] = 0.0
    return eye


def _parallel_reduce(mats: torch.Tensor, neg_inf: float) -> torch.Tensor:
    """Compute the product M_{n-1} ⊗ ... ⊗ M_0 via parallel reduce.

    mats: (N, S, S) — N matrices to multiply left-to-right
    Returns (S, S) — the total product

    Uses O(log₂ N) sequential steps of batched log-semiring matmul.
    """
    n = mats.shape[0]
    S_ext = mats.shape[1]

    # Pad to next power of 2
    next_pow2 = 1
    while next_pow2 < n:
        next_pow2 *= 2

    if next_pow2 > n:
        pad = _make_identity(next_pow2 - n, S_ext, mats.dtype, mats.device, neg_inf)
        mats = torch.cat([mats, pad], dim=0)

    while mats.shape[0] > 1:
        left = mats[0::2]    # even indices (earlier matrices)
        right = mats[1::2]   # odd indices (later matrices)
        mats = _log_semiring_bmm(right, left)

    return mats[0]


# ---------------------------------------------------------------------------
# CTC loss — single sample
# ---------------------------------------------------------------------------

class DifferentiableCTCFast(nn.Module):
    """CTC loss supporting create_graph=True — drop-in replacement for v1.

    Two modes:
      scan=False (default): Optimized sequential DP. ~2–5× faster than v1,
          memory O(T × S_ext) — safe for MAML with create_graph=True.
      scan=True: Log-semiring parallel prefix scan. ~14–75× faster than v1,
          but O(T × S_ext³) memory with create_graph=True. Use only when
          VRAM is plentiful or S_ext is small.

    Args:
      scan: If True, use parallel scan (high speed, high VRAM).
            If False (default), use optimized sequential (moderate speed, low VRAM).
    """

    def __init__(self, scan: bool = False):
        super().__init__()
        self.scan = scan

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
        NEG_INF = _get_neg_inf(dtype)

        # --- Validate minimum sequence length ---
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

        # --- Build extended targets: [blank, label_0, blank, label_1, ..., blank] ---
        targets_ext = torch.full(
            (S_ext,), blank, dtype=torch.long, device=device
        )
        targets_ext[1::2] = targets

        # --- Skip transition invalidity ---
        skip_invalid = torch.zeros(S_ext, dtype=torch.bool, device=device)
        if S_ext >= 3:
            skip_invalid[2:] = targets_ext[2:] == targets_ext[:-2]

        # --- Index emissions by extended targets: (T, S_ext) ---
        all_emissions = log_probs[:, targets_ext]

        # --- Build initial alpha_0 from t=0 emissions ---
        alpha = torch.full((S_ext,), NEG_INF, dtype=dtype, device=device)
        alpha[0] = all_emissions[0, 0]
        if S_ext > 1:
            alpha[1] = all_emissions[0, 1]

        # --- T=1: no transitions needed ---
        if T == 1:
            if S_ext >= 2:
                final_states = torch.stack([alpha[-2], alpha[-1]])
            else:
                final_states = alpha[-1:]
            return -torch.logsumexp(final_states, dim=0)

        # --- Dispatch to scan or sequential ---
        if self.scan:
            alpha_final = self._forward_scan(
                alpha, all_emissions, skip_invalid, S_ext, T, device, dtype, NEG_INF,
            )
        else:
            alpha_final = self._forward_sequential(
                alpha, all_emissions, skip_invalid, S_ext, T, NEG_INF,
            )

        # --- Extract log-likelihood from final states ---
        if S_ext >= 2:
            final_states = torch.stack([alpha_final[-2], alpha_final[-1]])
        else:
            final_states = alpha_final[-1:]

        log_likelihood = torch.logsumexp(final_states, dim=0)
        return -log_likelihood

    # -- Sequential mode (default) ------------------------------------------

    @staticmethod
    def _forward_sequential(
        alpha: torch.Tensor,
        all_emissions: torch.Tensor,
        skip_invalid: torch.Tensor,
        S_ext: int,
        T: int,
        NEG_INF: float,
    ) -> torch.Tensor:
        """Optimized sequential DP — O(T × S_ext) memory.

        Key improvements over v1:
          - Finite NEG_INF → native torch.logsumexp (no custom safe version)
          - F.pad + slicing for shifts (fewer allocations than torch.cat)
          - No .detach(), .clamp(), .new_full() per step
        """
        for t in range(1, T):
            # Pad alpha on the left with 2 NEG_INF values, then slice
            # to get stay / advance / skip views efficiently
            alpha_padded = F.pad(alpha, (2, 0), value=NEG_INF)  # (S_ext + 2,)

            stay = alpha_padded[2:]       # alpha[0..S_ext-1]  (same as alpha)
            advance = alpha_padded[1:-1]  # [NEG_INF, alpha[0..S_ext-2]]
            skip = alpha_padded[:-2]      # [NEG_INF, NEG_INF, alpha[0..S_ext-3]]

            if S_ext >= 3:
                skip = skip.masked_fill(skip_invalid, NEG_INF)

            # Combine sources and compute new alpha via logsumexp + emission
            combined = torch.stack([stay, advance, skip], dim=0)  # (3, S_ext)
            alpha = torch.logsumexp(combined, dim=0) + all_emissions[t]

        return alpha

    # -- Parallel scan mode -------------------------------------------------

    @staticmethod
    def _forward_scan(
        alpha_0: torch.Tensor,
        all_emissions: torch.Tensor,
        skip_invalid: torch.Tensor,
        S_ext: int,
        T: int,
        device: torch.device,
        dtype: torch.dtype,
        NEG_INF: float,
    ) -> torch.Tensor:
        """Parallel prefix scan — O(log₂ T) steps, O(T × S_ext³) memory."""
        transition_mask = _build_transition_mask(skip_invalid, S_ext, device)
        mats = _build_transition_matrices(
            all_emissions[1:], transition_mask, NEG_INF
        )
        P = _parallel_reduce(mats, NEG_INF)
        return _log_semiring_matvec(P, alpha_0)


# ---------------------------------------------------------------------------
# Batch interface — drop-in replacement for ctc_loss_differentiable
# ---------------------------------------------------------------------------

def ctc_loss_differentiable_fast(
    logits: torch.Tensor,
    labels: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int = 0,
    scan: bool = False,
) -> torch.Tensor:
    """Batch CTC loss supporting double-backward (fast version).

    Drop-in replacement for ctc_loss_differentiable with identical API.

    logits:         (B, T, C) — raw logits from wav2vec2 (before log_softmax)
    labels:         (B, S)    — target ids, -100 for padding
    input_lengths:  (B,)      — actual frame counts
    target_lengths: (B,)      — actual target lengths
    blank:          int       — blank token index
    scan:           bool      — use parallel scan (faster but more VRAM)

    Returns mean loss over batch.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    ctc = DifferentiableCTCFast(scan=scan)
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
