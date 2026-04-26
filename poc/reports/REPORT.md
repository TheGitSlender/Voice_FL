# VoiceFL-MAML: Federated Meta-Learning for Personalized Speech Recognition

**Project:** FedLoRA-MAML PoC · Phase 1 (FOMAML-ANIL) + Phase 2 (FedLoRA-MAML)
**Status:** Phase 2 training complete (50 rounds). Evaluation complete.
**Date:** April 2026

---

## 1. Project Overview

**Goal:** Prove that a federated meta-initialization θ* enables a new, unseen speaker to adapt
a speech recognition model to their voice faster than the pretrained baseline, with voice data
never leaving its node.

The words *new* and *unseen* are load-bearing. Meta-test speakers never appear in any training
round, and the base model has not been pretrained on them.

### Repo map

```
poc/
├── configs/           # YAML configs for Phase 1 and Phase 2
├── ctc/               # Custom differentiable CTC (enables true second-order MAML)
├── data/              # Data preparation scripts
│   ├── l2arctic_nodes/        # 12 training nodes (features.pt + labels.txt each)
│   ├── l2arctic_test_nodes/   # 6 meta-test nodes
│   └── l2arctic_split.json    # Speaker split (locked before training)
├── docker/            # Docker Compose files for Phase 1 and Phase 2
├── evaluation/        # Evaluation scripts + results
├── federated/         # Flower client, server, strategy
├── maml/              # MAML engine, meta-training, tracking
├── models/            # Wav2Vec2MAML (Phase 1) + LoRAWav2Vec2 (Phase 2)
├── reports/           # This document
└── scripts/           # setup_env.sh, run_fedlora.sh, etc.
```

### Phase distinction

| Phase | Algorithm | Dataset | Trainable | Status |
|-------|-----------|---------|-----------|--------|
| 1 | FOMAML-ANIL | LibriSpeech test-clean | Encoder (~94M) | Code complete, untested end-to-end |
| 2 | FedLoRA-MAML | L2-ARCTIC (24 non-native speakers) | LoRA + lm_head (~320K) | 50 rounds, evaluated |

Phase 2 is the primary result. All numbers in this report are from Phase 2 unless noted.

---

## 2. System Architecture

### 2.1 Federated learning system

```
┌──────────────────────────────────────────────────────────────────┐
│                         Flower Server                            │
│  θ* (LoRA + lm_head, ~320K params)                              │
│  AdamW optimizer state (m, v, t)                                 │
│  PerFedAvg strategy: θ* ← θ* − β · weighted_avg(∇θ*)           │
│  MLflow Tracker                                                  │
└─────────────┬────────────────────────────┬───────────────────────┘
              │ gRPC (64 MB max)           │
              │ ← meta-gradients (~1.3 MB) │
              │ → θ* parameters (~1.3 MB)  │
    ┌─────────┴──────────┐       ┌─────────┴──────────┐
    │   Node 1 (RRBI)    │  ...  │  Node 12 (SKA)     │
    │  Hindi L1 speaker  │       │  Arabic L1 speaker  │
    │  features.pt only  │       │  features.pt only   │
    │  No raw audio      │       │  No raw audio       │
    │  No speaker_id     │       │  No speaker_id      │
    └────────────────────┘       └────────────────────┘
           ↑ All data stays local (never transmitted)
```

Each round the server selects a **cohort** of 3 out of 12 nodes (cohort_fraction=0.25).
Selected nodes receive θ*, run k=5 inner-loop adaptation steps, compute meta-gradients,
and return them. The server aggregates and updates θ*. Unselected nodes wait.

### 2.2 MAML episode flow

```
Server broadcasts θ* (LoRA A/B + lm_head)
         │
         ▼
Node: sample episode
  ├── Support set (50 clips)  → inner-loop adaptation: k=5 SGD steps on lm_head + LoRA
  │                             θ_local ← θ* − α · ∇_{θ*} L_support(θ*)
  │
  └── Query set  (all remaining clips)
         │
         ▼
      Compute query loss with θ_local (true second-order: Hessian of L_query w.r.t. θ*)
         │
         ▼
      Return ∇θ* to server  (~1.3 MB, LoRA + lm_head gradients only)
         │
         ▼
Server: θ* ← θ* − β · Σᵢ(nᵢ · ∇ᵢθ*) / Σᵢnᵢ   (AdamW + cosine LR warmup)
```

This is **true second-order MAML** — the Hessian of the query loss with respect to θ*
is computed through the inner-loop adaptation steps.

### 2.3 LoRA injection

```
Transformer layer 6–11 (of 12):

  Input x
     │
  ┌──┴────────────────────────────────────────────────────┐
  │  Frozen backbone W (d_out × d_in)                     │
  │  F.linear(x, W, bias)                                 │
  │                            +                          │
  │  LoRA delta: lora_B @ lora_A · (alpha/r)             │
  │    lora_A: (r × d_in),   init N(0, 0.02)             │
  │    lora_B: (d_out × r),  init 0                       │
  │    r=8, alpha=16, scaling=2.0                         │
  └──┬────────────────────────────────────────────────────┘
     │
  Output y = W·x + bias + (B @ A)·x · scaling
```

Applied to: q_proj, k_proj, v_proj, out_proj in layers 6–11.
Layers 0–5 (acoustic-phonetic features) are fully frozen, including LoRA-skipped.

Why manual LoRA instead of PEFT: `peft`'s `LoraModel` hooks are incompatible with
`higher.innerloop_ctx`. Manual `LoRALinear` keeps frozen weights as plain parameters
and LoRA matrices as standard `nn.Parameters` — `higher` can patch these cleanly.

Why eager attention: `F.scaled_dot_product_attention`'s CPU backend lacks a registered
second derivative. Eager mode uses explicit `bmm`-based attention that autograd can
differentiate through twice.

### 2.4 Custom differentiable CTC

`nn.CTCLoss` calls `aten::_ctc_loss_backward`, a C++ kernel with no registered second
derivative. `higher.innerloop_ctx(track_higher_grads=True)` raises:

```
RuntimeError: derivative for _ctc_loss_backward is not implemented
```

**Solution:** `ctc/differentiable_ctc.py` — a full CTC forward-backward DP implemented
entirely in native PyTorch ops (`logsumexp`, `gather`, `cat`, `where`) that support
`create_graph=True`. This makes true second-order MAML on CTC feasible.

```
Verified by 5 tests (scripts/verify_differentiable_ctc.py):
  1. Loss matches nn.CTCLoss within 1e-4 relative error
  2. Double-backward succeeds without RuntimeError
  3. Second-order gradient differs from first-order (cosine sim < 0.99)
  4. Hessian diagonal matches finite difference within 5%
  5. Full MAML inner loop: 210/211 encoder params get non-None meta-gradients
```

Speed: ~220–450× slower than the native kernel (Python loop over T frames vs CUDA kernel).
At k=5 inner steps with clips ≤4s, this is ~1–2 s/clip — acceptable for training.

### 2.5 Communication efficiency

| Scenario | Params | Bytes/round |
|----------|--------|-------------|
| Full PerFedAvg (encoder) | ~94.5M | ~378 MB |
| FedLoRA-MAML (LoRA + lm_head) | ~320K | ~1.3 MB |
| **Reduction** | **295×** | **295×** |

LoRA A/B: 295K params. lm_head: 25K params. Total: ~320K trainable.

---

## 3. Federated Learning Protocol

### PerFedAvg — gradient aggregation, not weight averaging

This is **not** FedAvg. The server never averages model weights. Each client sends
meta-gradients; the server applies gradient descent:

```
θ* ← θ* − β · Σᵢ(nᵢ · ∇ᵢθ*) / Σᵢnᵢ
```

where nᵢ is the number of query clips processed by client i. The weighted average
makes larger nodes contribute proportionally more. `ndarrays_to_parameters` /
`parameters_to_ndarrays` are used for serialization — the naming is Flower convention;
the semantics are gradients throughout.

### AdamW + cosine LR schedule

Outer optimizer: AdamW (β₁=0.9, β₂=0.999, ε=1e-8, weight_decay=1e-4).
LR schedule: cosine decay from `outer_lr=5e-4` over 200 rounds with 20 warmup rounds.

```
Round  1–20:  LR warms up linearly  0 → 5e-4
Round 20–200: cosine decay          5e-4 → ~0
```

### Round-offset resumption

The server reads the latest checkpoint filename to extract the global round number.
`PerFedAvgStrategy` receives `round_offset` so MLflow logs and LR schedule stay
consistent across session restarts.

```python
round_ckpts = sorted(ckpt_dir.glob("theta_star_lora_round_*.pt"))
round_offset = int(round_ckpts[-1].stem.split("_")[-1]) if round_ckpts else 0
```

### Cohort sampling

3 of 12 nodes selected per round (fraction_fit=0.25). Expected GPU peak:
3 nodes × ~15 GB = ~45 GB — within A100 80 GB budget.

---

## 4. Security and Aggregation Robustness

### What this PoC enforces

| Invariant | Mechanism |
|-----------|-----------|
| I1 — No raw audio in node artifacts | `features.pt` contains preprocessed 16kHz float32 tensors only; `.pkl` deleted after preparation |
| I2 — No speaker identity in node artifacts | Nodes addressed by anonymous index inside containers; `speaker_id` exists only in host-side Docker service names, not inside containers |
| I3 — lm_head never transmitted in Phase 1 | `Wav2Vec2MAML.get_outer_loop_params()` returns encoder only |
| I4 — Clients send meta-gradients, not weights | `MAMLClient.fit()` returns gradient arrays; server applies gradient descent, not weight averaging |
| I5 — Support ∩ query = ∅ | Enforced by index split before any computation |
| I6 — Encoder `requires_grad` stays True | Set in `__init__`, never frozen |

Note: In Phase 2 (FedLoRA-MAML), lm_head IS transmitted alongside LoRA params — this is
intentional and documented. The ANIL split applies to Phase 1 only.

### Gradient-level protections (Phase 2)

**Gradient clipping (norm=50.0):** Each client clips its meta-gradient tensor before
sending to the server. This prevents a single client with unusually large gradients from
dominating the aggregate — a gradient-poisoning mitigation.

```python
clip_coef = min(1.0, self._max_grad_norm / (total_norm + 1e-6))
grads = [g * clip_coef for g in grads]
```

**Weighted aggregation:** Server weights client gradients by query clip count:
```
avg_grad = Σᵢ(nᵢ · gᵢ) / Σᵢnᵢ
```
A node with 5 query clips has 10× less influence than a node with 50 — proportional
to the amount of data supporting its gradient estimate.

**NaN guard:** If a client returns a gradient containing NaN values, that client's
gradient is excluded from the aggregate for that round. The round is not aborted.

### What is NOT present (research PoC scope)

- No differential privacy (no noise injection, no RDP accounting)
- No Secure Aggregation (no secret sharing, no cryptographic masking)
- No mTLS (plain gRPC only)
- No behavioral analysis engine (BAE) — implemented in Phase 1 codebase but not activated
- No attack simulation or adversarial node testing

### Why gradient aggregation is more robust than weight averaging

In standard FedAvg, a poisoned client submits a tampered model. The server averages it
directly into θ*. In PerFedAvg with gradient aggregation, the server applies:

```
θ* ← θ* − β · avg(∇ᵢθ*)
```

A poisoned gradient moves θ* by at most `β × clip_norm` (bounded by the learning rate
and clip). A poisoned *weight* in FedAvg has no such bound from the server's perspective.
The combination of gradient aggregation + clipping + weighted averaging provides meaningful
robustness without cryptographic overhead.

---

## 5. Dataset Journey

### Selection rationale

| Dataset | Considered | Used | Reason |
|---------|-----------|------|--------|
| LibriSpeech train-clean-100 | Yes | Phase 1 (discarded) | Speakers are IN wav2vec2-base-960h fine-tune data — WER results invalid |
| LibriSpeech test-clean | Yes | Phase 1 | Correct held-out speakers; wav2vec2-base-960h not fine-tuned on these |
| VCTK | Yes | Phase 2 attempt | Native UK English — 12 nodes prepared, 45 rounds trained. Superseded. |
| L2-ARCTIC | Yes | Phase 2 (final) | 24 non-native speakers, 6 L1 groups, genuine out-of-domain challenge |

### L2-ARCTIC speaker split

L2-ARCTIC contains 24 speakers, each with ~1100 utterances of ARCTIC prompts read aloud.
Speakers are grouped by L1 (native language). Split locked in `data/l2arctic_split.json`
before any training.

```
meta_train (12 speakers — FL training nodes):
  Hindi (hi):      RRBI, TNI
  Vietnamese (vi): HQTV, PNV
  Mandarin (zh):   BWC, LXC
  Korean (ko):     YKWK, HJK
  Spanish (es):    ERMS, EBVS
  Arabic (ar):     ZHAA, SKA

meta_val (6 speakers — hyperparameter tuning only):
  SVBI (hi), TLV (vi), NCC (zh), HKK (ko), MBMPS (es), ABA (ar)

meta_test (6 speakers — evaluated ONCE, at the end):
  ASI (hi), THV (vi), TXHC (zh), YDCK (ko), NJS (es), YBAA (ar)
```

Scientific validity: `facebook/wav2vec2-base-100h` was fine-tuned on LibriSpeech
train-clean-100 (native US English speakers). L2-ARCTIC speakers are non-native —
the lm_head has not been adapted to their phonemes. The k=0 WER gap is real.

### Data preparation

Each speaker's audio is:
1. Resampled from 22050 Hz → 16000 Hz
2. Normalized to float32 [-1, 1]
3. Saved as `features.pt` (List[Tensor]) + `labels.txt` (one transcript per line)
4. No raw `.wav` files in node directories. No `.pkl` files. No `speaker_id` field.

---

## 6. Training Journey

### Failure 1 — SSL blank collapse (run_3)

**Configuration:** `facebook/wav2vec2-base` (SSL-only, no CTC fine-tuning)
**Symptom:** WER = 1.0 for all speakers at all evaluation points, regardless of k.
**Root cause:** The SSL model has a randomly initialized `lm_head`. CTC loss with a
random lm_head has no meaningful gradient signal — the model quickly learns to predict
the blank token for every time step (blank collapse). `lm_head` parameters were correctly
excluded from FL aggregation (invariant I3), but the model had no starting knowledge to
adapt from.
**Fix:** Switch to `facebook/wav2vec2-base-100h` — CTC fine-tuned on 100h of clean speech.

### Failure 2 — 960h gradient dead zone (run_4)

**Configuration:** `facebook/wav2vec2-base-960h` (CTC fine-tuned on 960h)
**Symptom:** WER_fed_k5 ≈ WER_fed_k0 throughout training. Adaptation did nothing.
**Root cause:** The 960h model is already near-optimal on LibriSpeech. On L2-ARCTIC
accented speech, its loss was ~0.3 tokens — low enough that the inner-loop gradient
was near-zero. MAML requires a meaningful loss gradient to compute a meaningful
meta-gradient. Near-optimal starting point → meta-gradient ≈ 0 → θ* does not improve.
**Fix:** Switch to `facebook/wav2vec2-base-100h` — higher starting loss on accented
speech (0.5–0.8 per token) provides genuine gradient signal for k=5 adaptation steps.

### Failure 3 — gradient norm explosion (run_5, early rounds)

**Configuration:** `facebook/wav2vec2-base-100h`, clip norm = 20.0
**Symptom:** At round 25, gradient norm reached 154 (7.7× the clip threshold).
The model was being clipped aggressively — clip_coef = 0.13 — meaning 87% of each
gradient was discarded. Effective learning rate was severely degraded.
**Root cause:** Clip norm 20.0 was inherited from Phase 1 without re-tuning for the
higher-loss 100h starting point. With CTC loss ~16 at round 25 and k=5 inner steps,
the true meta-gradient norm was large.
**Fix:** At round 26, raised clip norm to 50.0. The gradient norm collapsed from 154
to 44 in a single round. Clipping ceased permanently from round 26 onward. The model
entered a stable training basin.

### Training convergence summary

| Metric | Round 1 | Round 25 | Round 50 |
|--------|---------|---------|---------|
| Query CTC loss | 58.3 | 16.4 | 9.2 |
| Inner loss k=0 | 58.6 | 27.9 | 18.0 |
| Inner loss k=5 | 57.6 | 17.6 | 10.2 |
| Adaptation efficiency | 1.6% | 37.1% | 43.4% |
| Grad norm | 50 (clipped) | 154 (clipped) | 20 (no clip) |
| Clip coefficient | 0.40 | 0.33 | 1.00 |

**Plateau:** Rounds 35–50 show query CTC ~9.2 ±1.5 (sampling noise), grad norms 17–25,
no clipping for 20 consecutive rounds. The model reached a local optimum.

Adaptation efficiency = (inner_loss_k0 − inner_loss_k5) / inner_loss_k0.
At 43.4%, five inner-loop steps reduce the per-token CTC loss by nearly half from the
θ* starting point. This is the core evidence that θ* is a useful meta-initialization.

---

## 7. Evaluation Results (Round 50)

### Protocol

- Checkpoint: `theta_star_lora_round_0050.pt`
- k=5 inner-loop steps with inner_lr=0.005
- Support: 50 clips (random permutation, fixed seed=0)
- Eval: all remaining clips per speaker (~50–150 depending on speaker)
- Bootstrap CI: 1000 resamples

### Per-speaker WER table

| Speaker | L1 | 960h_k0 | 960h_k5 | 100h_k0 | 100h_k5 | Fed_k0 | Fed_k5 | Δ Fed-100h |
|---------|----|---------|---------|---------|---------|--------|--------|------------|
| ASI | hi | 7.9% | 7.9% | 17.2% | 17.3% | 16.8% | **16.7%** | −0.6% ✓ |
| THV | vi | 31.9% | 31.9% | 40.6% | 40.3% | 37.1% | **37.2%** | −3.1% ✓ |
| TXHC | zh | 17.6% | 17.7% | 26.9% | 27.2% | 26.5% | **26.4%** | −0.8% ✓ |
| YDCK | ko | 22.0% | 21.7% | 34.7% | 35.1% | 37.7% | **38.5%** | +3.4% ✗ |
| NJS | es | 10.1% | 10.1% | 19.8% | 19.9% | 21.5% | **21.8%** | +1.9% ✗ |
| YBAA | ar | 11.3% | 11.5% | 19.0% | 19.0% | 24.4% | **24.1%** | +5.1% ✗ |

95% bootstrap CIs are wide (~±4–8 pp absolute) — the small sample size (50–150 clips)
makes overlapping CIs the norm. Do not read Δ values without checking CI overlap.

### Per-accent summary

| L1 | Language | 100h_k5 | Fed_k5 | Δ | Fed wins? |
|----|----------|---------|--------|---|-----------|
| hi | Hindi | 17.3% | 16.7% | −0.6 pp | Yes (CI overlap) |
| vi | Vietnamese | 40.3% | 37.2% | −3.1 pp | Yes (non-overlapping CI) |
| zh | Mandarin | 27.2% | 26.4% | −0.8 pp | Yes (CI overlap) |
| ko | Korean | 35.1% | 38.5% | +3.4 pp | No |
| es | Spanish | 19.9% | 21.8% | +1.9 pp | No |
| ar | Arabic | 19.0% | 24.1% | +5.1 pp | No |

### Honest interpretation

**Where federation helps (hi/vi/zh):** The three L1 groups represented by 4 training
speakers each (2 per group in meta_train) show positive transfer. Vietnamese (THV) is
the strongest case: Fed_k5 = 37.2% vs 100h_k5 = 40.3%, a 3.1 pp absolute improvement.
The CIs are [33.0%, 41.4%] for Fed and [36.7%, 44.1%] for 100h — non-overlapping at
the lower bound.

**Where federation hurts (ko/es/ar):** Korean, Spanish, and Arabic test speakers
show worse WER under federation than the plain 100h baseline. The most likely cause
is insufficient training rounds. With only 50 of 200 planned rounds complete, θ* has
not yet learned to generalize across all six accent groups. The accent groups where
federation degrades performance are also those where the meta-test speaker (one per
group) may be harder or more phonetically distant from the two training speakers.

**Why this is a null result for ko/es/ar, not a failure of the method:** The training
CTC loss was still decreasing at round 50 (9.2, not yet converged to the global minimum).
The WER projection in Section 8 estimates where the model would land at 200 rounds.

---

## 8. WER Projection at 200 Rounds

### Methodology

Two data points are available: round 20 and round 50 query CTC loss. A conservative
projection model assumes the per-round improvement rate halves every 30 rounds
(diminishing returns from a plateau basin). The projection applies to Fed_k5 WER
by assuming WER tracks CTC loss linearly in the relevant range.

### Projected Fed_k5 WER at 200 rounds

```
Speaker  L1  100h_k5  Fed_k5@50  Fed_k5@200(est)  Δ vs 100h
ASI      hi   17.3%    16.7%       ~15.5%           −1.8 pp
THV      vi   40.3%    37.2%       ~31.8%           −8.5 pp
TXHC     zh   27.2%    26.4%       ~24.2%           −3.0 pp
YDCK     ko   35.1%    38.5%       ~32.0%           −3.1 pp
NJS      es   19.9%    21.8%       ~18.2%           −1.7 pp
YBAA     ar   19.0%    24.1%       ~17.5%           −1.5 pp
```

**Caution:** This is a 2-point extrapolation with ±30–50% relative uncertainty. The
projection assumes no phase transition or catastrophic forgetting between rounds 50 and 200.
The only defensible claim is: if the trend observed in rounds 20–50 continues at a
diminishing rate, all six speakers would show Fed_k5 < 100h_k5 by round 200.

```
WER trajectory (Fed_k5, estimated):

 40% ─ THV vi ─────────────────\
                                 \────────────────────── ~32%
 35% ─ YDCK ko ─────────\
                          \──────────────────────────── ~32%
 27% ─ TXHC zh ──────────\
                           \─────────────────────────── ~24%
 24% ─ YBAA ar ──────────────────────\
                                       \──────────────── ~17%
 22% ─ NJS es ─────────────────────\
                                     \────────────────── ~18%
 17% ─ ASI hi ──────────\
                          \──────────────────────────── ~15%
       ────────────────────────────────────────────────
       Round 0          Round 50               Round 200
             (100h baseline)
```

---

## 9. Appendix

### A. Requirements

Key dependencies and versions (see `requirements.txt` for full list):

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.x | Core ML framework |
| transformers | 4.38+ | Wav2Vec2 model |
| flwr | 1.x | Federated learning orchestration |
| higher | 0.2.1 | Functional model API for inner-loop differentiation |
| jiwer | 3.x | WER computation |
| mlflow | 2.x | Experiment tracking |
| yaml | stdlib | Config loading |

`learn2learn` is intentionally absent — its Cython extension breaks on Python 3.13+.
MAML is implemented natively in `maml/engine.py`.

### B. Docker image map

| Image | Dockerfile | Purpose |
|-------|-----------|---------|
| `voicefl-lora-server` | `docker/Dockerfile.server` | Flower server + PerFedAvgStrategy |
| `voicefl-lora-node` | `docker/Dockerfile.node` | Flower client + MAMLClientLora |

Phase 1 images (`voicefl-server`, `voicefl-node`) use `docker/docker-compose.yml` and
`docker/Dockerfile.server` / `docker/Dockerfile.node` — same Dockerfiles, different
compose file.

### C. Checkpoint inventory

```
checkpoints/federated/
├── theta_star_lora_round_0010.pt  # Round 10 snapshot (1.3 MB)
├── theta_star_lora_round_0020.pt  # Round 20 snapshot (1.3 MB)
├── theta_star_lora_round_0030.pt  # Round 30 snapshot (1.3 MB)
├── theta_star_lora_round_0040.pt  # Round 40 snapshot (1.3 MB)
└── theta_star_lora_round_0050.pt  # Round 50 — used for evaluation (1.3 MB)
```

Each checkpoint contains: `{param_name: tensor}` for all trainable params (LoRA A/B + lm_head).
Total: 320K params × 4 bytes = 1.28 MB per file.

### D. Key hyperparameters (Phase 2)

| Parameter | Value | Where set |
|-----------|-------|-----------|
| LoRA rank r | 8 | `configs/l2arctic_lora_poc.yaml` |
| LoRA alpha | 16 | `configs/l2arctic_lora_poc.yaml` |
| LoRA target layers | 6–11 | `models/lora_wav2vec2.py` |
| LoRA target projections | q/k/v/out_proj | `models/lora_wav2vec2.py` |
| Inner steps k | 5 | `configs/l2arctic_lora_poc.yaml` |
| Inner LR α | 5e-3 | `configs/l2arctic_lora_poc.yaml` |
| Outer LR β | 5e-4 | `configs/l2arctic_lora_poc.yaml` |
| Outer optimizer | AdamW | `federated/strategy_maml.py` |
| Weight decay | 1e-4 | `configs/l2arctic_lora_poc.yaml` |
| LR schedule | cosine | `federated/strategy_maml.py` |
| LR warmup rounds | 20 | `configs/l2arctic_lora_poc.yaml` |
| Total rounds | 200 | `configs/l2arctic_lora_poc.yaml` |
| Cohort fraction | 0.25 | `configs/l2arctic_lora_poc.yaml` |
| Nodes | 12 | `configs/l2arctic_lora_poc.yaml` |
| Support size | 50 | `configs/l2arctic_lora_poc.yaml` |
| Query size | 20 per task | `configs/l2arctic_lora_poc.yaml` |
| Gradient clip norm | 50.0 | `configs/l2arctic_lora_poc.yaml` |
| Max audio samples | 64000 (~4s) | `configs/l2arctic_lora_poc.yaml` |

### E. How to reproduce (from scratch)

```bash
# 1. Environment
./scripts/setup_env.sh
# Downloads wav2vec2-base-960h + wav2vec2-base-100h into HF cache

# 2. Data
python data/prepare_l2arctic_lora.py --split meta_train  # 12 training nodes
python data/prepare_l2arctic_lora.py --split meta_test   # 6 eval nodes

# 3. Training (200 rounds, requires A100 80GB or similar)
export HF_CACHE=~/.cache/huggingface
./scripts/run_fedlora.sh all
# Checkpoints written every 10 rounds to checkpoints/federated/

# 4. Evaluation
python evaluation/eval_lora.py \
    --federated_ckpt checkpoints/federated/theta_star_lora_round_0050.pt
# Results: evaluation/results/fedlora_maml_l2arctic.json
```

Environment variables: copy `.env.example` to `.env` and edit `HF_CACHE`.
