# VoiceFL-MAML — Session Handoff

**Branch:** `master` · **Last commit:** `e334a20` · **Date:** 2026-04-22

---

## What this project is

**Goal:** Prove that a federated meta-initialization θ* lets a new, unseen speaker
adapt a speech recognition model faster than fine-tuning from the pretrained baseline —
with voice data never leaving the speaker's node.

**Stack:**

| Layer | Implementation |
|-------|----------------|
| Model | `facebook/wav2vec2-base-960h` (~95M params) |
| Algorithm (Phase 1) | FOMAML + ANIL |
| Algorithm (Phase 2) | True second-order MAML over LoRA (~320K params) |
| FL framework | Flower (`flwr`) |
| Dataset (Phase 1) | LibriSpeech test-clean + dev-clean (20 speakers) |
| Dataset (Phase 2) | VCTK corpus (20 British/Scottish/Irish speakers) |
| Tracking | MLflow (`maml/tracking.py`) |

---

## Repository layout

```
poc/
├── ctc/
│   └── differentiable_ctc.py   # Custom CTC — pure PyTorch, double-backward safe
├── data/
│   ├── task_sampler.py          # VoiceTaskSampler — K-shot task builder
│   ├── features.py              # Audio → features.pt + labels.txt
│   ├── prepare_vctk_lora.py     # VCTK download + node prep (Phase 2)
│   ├── prepare_librispeech.py   # LibriSpeech node prep (Phase 1)
│   ├── meta_split.py            # Lock meta-train/val/test speaker split
│   ├── diagnostics/             # Gradient norm analysis scripts + results
│   └── insights/                # Domain shift, adaptation curve data
├── models/
│   ├── wav2vec2_maml.py         # Wav2Vec2MAML — FOMAML / ANIL model (Phase 1)
│   └── lora_wav2vec2.py         # LoRAWav2Vec2 — Phase 2 model (~320K trainable)
├── maml/
│   ├── engine.py                # MAMLEngine — fomaml / second_order_ctc / lora_maml
│   ├── meta_train.py            # Centralized training gate (both phases)
│   ├── tracking.py              # MLflow Tracker wrapper
│   └── meta_eval.py             # Standalone eval utilities
├── federated/
│   ├── client_maml.py           # Flower client — sends meta-gradients, not weights
│   ├── server_maml.py           # Flower server — loads θ*, starts rounds
│   ├── strategy_maml.py         # PerFedAvgStrategy — gradient descent on server
│   └── run_node.py              # Entrypoint for each Docker node container
├── evaluation/
│   ├── eval_poc.py              # Phase 1 final evaluation (run ONCE)
│   ├── eval_lora.py             # Phase 2 final evaluation (run ONCE)
│   └── plot_results.py          # WER curves + CI plots
├── configs/
│   ├── poc.yaml                 # Phase 1 — LibriSpeech, FOMAML, 20 FL rounds
│   ├── lora_poc.yaml            # Phase 2 — L2-ARCTIC, lora_maml
│   └── vctk_lora_poc.yaml       # Phase 2 — VCTK, lora_maml, 200 rounds, AdamW
├── docker/
│   ├── Dockerfile.train         # Lightning AI single-GPU training image
│   ├── Dockerfile.node          # FL node container
│   ├── Dockerfile.server        # FL server container
│   ├── docker-compose.train.yml # Single-GPU training (centralized gate)
│   └── docker-compose.yml       # Full FL simulation (server + 5 nodes)
├── tests/
│   └── test_differentiable_ctc.py  # 5 CTC correctness tests (all pass)
└── scripts/
    ├── setup_env.sh             # Reproducible environment setup for new machines
    ├── verify_differentiable_ctc.py
    └── benchmark_ctc.py
```

---

## Key design decisions

### ANIL split (Phase 1)

```
wav2vec2 encoder (~94M)  →  outer loop  →  aggregated by FL server
lm_head (~25K)           →  inner loop  →  stays local, never transmitted
```

### FedLoRA-MAML (Phase 2) — NOT ANIL

```
LoRA A/B in layers 6–11 (~295K)  →  inner loop + FL aggregation
lm_head (~25K)                   →  inner loop + FL aggregation
frozen backbone (~94.5M)         →  never trained, never transmitted
```

LoRA targets layers 6–11 only (layers 0–5 encode speaker-independent acoustics).
Communication per round: ~1.3 MB (280× less than full encoder).

### Why a custom differentiable CTC

`nn.CTCLoss` calls `aten::_ctc_loss_backward`, a CUDA kernel with no registered
second derivative. `higher.innerloop_ctx(track_higher_grads=True)` raises
`RuntimeError: derivative for _ctc_loss_backward is not implemented`.

`ctc/differentiable_ctc.py` reimplements the CTC forward DP entirely in native
PyTorch ops (logsumexp, gather, cat, where) — all support `create_graph=True`.
Five verification tests confirm correctness (Hessian within 5% of finite diff).

### Why `attn_implementation="eager"` in LoRAWav2Vec2

`F.scaled_dot_product_attention` on CPU calls
`aten::_scaled_dot_product_flash_attention_for_cpu`, which has no second
derivative. Eager mode uses explicit bmm-based attention that autograd can
differentiate through twice.

### Why `model.eval()` inside `higher.innerloop_ctx`

Calling `model.train()` when using `higher`'s functional parameter patching
causes wav2vec2's dropout layers to produce NaN logits. Gradients flow correctly
in eval mode.

### Outer optimizer: AdamW + cosine LR with warmup

Plain SGD at fixed LR caused WER degradation at k=10 (over-adaptation at 5e-4)
or negligible movement (at 1e-4). AdamW normalizes gradient scale per-parameter.

Schedule in `meta_train.py`:
- Rounds 1–20: linear warmup from 0 → `outer_lr` (5e-4)
- Rounds 21–200: cosine decay from 5e-4 → 1e-6

### Audio clip truncation (`max_audio_samples: 48000`)

VCTK clips average 7.1 seconds (113,600 samples). `higher` retains k×S forward
graphs simultaneously — k=5 steps × S=8 support clips = 40 retained graphs.
At full 7s clips this requires ~25GB VRAM; at 3s clips (~48,000 samples) ~10GB.

**On Lightning AI (48GB):** set `max_audio_samples: null` and `support_size: 20`
in `configs/vctk_lora_poc.yaml` to use full clips and larger support sets.

---

## Bugs found and fixed

### 1. torch.load without `weights_only=True` (security)

**File:** `meta_train.py` (lines 292, 298), `evaluation/eval_lora.py` (line 256)

**Problem:** `torch.load()` with default settings can execute arbitrary Python
code via pickle. Constitutes an arbitrary code execution vector when loading
checkpoints from untrusted sources.

**Fix:** All `torch.load()` calls now pass `weights_only=True`.

### 2. `.env.nodes` committed to git (security)

**File:** `.env.nodes` at repo root

**Problem:** File contained node hash values (speaker identifiers). Committing
this to a public repo could reveal which speakers map to which nodes.

**Fix:** Deleted from repo, added to `.gitignore` alongside `.env`, `.env.*`.

### 3. WER identical at k=0 and k=5 (train/eval mismatch)

**Symptom:** Gate eval showed WER(k=0) ≈ WER(k=5) for every speaker (~0.98).
Adaptation steps had no visible effect.

**Root cause:** Training truncated clips to 3s (`max_audio_samples=48000`).
Evaluation used full 7.1s clips. The model adapted on 3s audio but was measured
on 7s audio — the adaptation signal did not transfer.

**Fix:** `_compute_wer_lora_at_k()` now takes a `max_audio_samples` parameter
and applies the same truncation used during training. Clips failing the CTC
constraint (T_frames < 2×S−1) are skipped rather than crashing.

### 4. `data.nodes_dir` not read from YAML

**Symptom:** `meta_train.py --config configs/vctk_lora_poc.yaml` always looked in
`data/nodes/` (the LibriSpeech default) regardless of the config file.

**Root cause:** `load_config()` read `meta_split` and `split_file` from the
`data:` section but did not read `nodes_dir`.

**Fix:** Added `if "nodes_dir" in data_cfg: defaults["nodes_dir"] = data_cfg["nodes_dir"]`
in `load_config()`.

### 5. `rounds` key not picked up from YAML

**Symptom:** Training ran 20 rounds regardless of `rounds: 200` in the config.

**Root cause:** `rounds: 200` was placed under `federated.num_rounds`, but
`load_config()` only merges from `maml:` into defaults.

**Fix:** Moved the `rounds` key under `maml:` in `configs/vctk_lora_poc.yaml`.

### 6. VCTK HuggingFace dataset path wrong

**Symptom:** `prepare_vctk_lora.py` failed to load the dataset.

**Root cause:** `speechbrain/vctk` does not exist on HuggingFace Hub.

**Fix:** `_try_load_vctk()` tries `speechbrain/vctk` first then falls back to
`vctk` (the correct path).

---

## Invariants (must never be violated)

| # | Rule |
|---|------|
| I1 | Raw audio (`.pkl`) deleted after `features.py`. No pkl under `data/nodes/`. |
| I2 | No `speaker_id` in node artifacts. |
| I3 | (Phase 1 only) `lm_head` never in `get_parameters()` / `set_parameters()`. |
| I4 | `fit()` returns meta-gradients, not weight updates. |
| I5 | Support ∩ query indices = ∅. Asserted in `VoiceTaskSampler.sample_task()`. |
| I6 | Encoder `requires_grad` stays True throughout. |

---

## Current training configuration (Phase 2 — VCTK)

`configs/vctk_lora_poc.yaml`:

```yaml
maml:
  mode: lora_maml
  rounds: 200
  k: 5
  inner_lr: 1.0e-4
  outer_lr: 5.0e-4
  outer_optimizer: adamw
  outer_weight_decay: 1.0e-4
  lr_schedule: cosine
  lr_warmup_rounds: 20
  tasks_per_node: 4         # 12 nodes × 4 tasks = 48 updates/round
  support_size: 8           # bump to 20 on 48GB GPU
  query_size: 30
  eval_support_size: 20
  eval_query_size: 50
  max_audio_samples: 48000  # set null on 48GB GPU
  lora_rank: 8
  lora_alpha: 16.0
```

---

## VCTK speaker split (locked in `data/vctk_split.json`)

| Split | Speakers | Purpose |
|-------|----------|---------|
| meta_train (12) | p225–p234, p236, p237 | FL training nodes |
| meta_val (4) | p238–p241 | Hyperparameter tuning only |
| meta_test (4) | p243–p246 | Final eval — evaluate ONCE at the end |

**meta_test speakers must never be seen during any training decision.**

---

## How to set up on a new machine (Lightning AI)

```bash
# 1. Clone repo
git clone <repo-url>
cd voice_fl/poc

# 2. Install dependencies
bash scripts/setup_env.sh

# 3. Download and prepare VCTK nodes
python data/prepare_vctk_lora.py                    # meta-train + meta-val
# python data/prepare_vctk_lora.py --splits meta_test  # ONLY at the very end

# 4. (On 48GB GPU) Update config for full clips
#    Edit configs/vctk_lora_poc.yaml:
#      max_audio_samples: null
#      support_size: 20

# 5. Run centralized gate
python -u maml/meta_train.py --config configs/vctk_lora_poc.yaml

# Or via Docker (recommended for Lightning AI)
docker compose -f docker/docker-compose.train.yml up --build

# 6. Monitor training
mlflow ui --backend-store-uri mlruns
```

---

## Build order and phase status

### Phase 1 — PoC (LibriSpeech)

| Step | Status | Command |
|------|--------|---------|
| Lock meta-test speakers | ✅ Done | `data/meta_split.json` committed |
| Gradient norm diagnostic | ✅ Done | Results in `data/diagnostics/` |
| Data pipeline | ✅ Done | `prepare_librispeech.py` + `features.py` |
| Hyperparameter tuning | ✅ Done | inner_lr=1e-4, k=3 locked |
| Centralized MAML gate | ✅ Done | `checkpoints/centralized/theta_star.pt` |
| Federated training (20 rounds) | Pending | `docker-compose up` |
| Matched-update baseline | Pending | `meta_train.py --max_updates N` |
| Prepare meta-test data | Pending | `prepare_librispeech.py --splits meta_test` |
| Final evaluation | Pending — run ONCE | `eval_poc.py` |

### Phase 2 — FedLoRA-MAML (VCTK)

| Step | Status | Command |
|------|--------|---------|
| Lock VCTK speaker split | ✅ Done | `data/vctk_split.json` committed |
| LoRA model + custom CTC | ✅ Done | `models/lora_wav2vec2.py`, `ctc/differentiable_ctc.py` |
| Data pipeline for VCTK | ✅ Code ready | `data/prepare_vctk_lora.py` |
| Centralized gate (200 rounds) | **NEXT STEP** | `meta_train.py --config configs/vctk_lora_poc.yaml` |
| Hyperparameter search | Pending | Tune on meta-val only |
| Federated training (200 rounds) | Pending | Docker FL stack |
| Meta-test data prep | Pending — once | `prepare_vctk_lora.py --splits meta_test` |
| Final evaluation | Pending — run ONCE | `eval_lora.py` |

---

## Immediate next step

**Run the Phase 2 centralized gate on Lightning AI.**

```bash
# On the 48GB machine, first update vctk_lora_poc.yaml:
#   max_audio_samples: null
#   support_size: 20

python data/prepare_vctk_lora.py
python -u maml/meta_train.py --config configs/vctk_lora_poc.yaml
```

Gate passes when: `WER(k=5) < WER(k=0)` on ≥ 3 of 4 meta-val speakers.

If gate fails, diagnose with the adaptation curve before touching hyperparameters:
check that WER actually decreases over inner steps (k=1,2,3,4,5) on at least
one speaker. A flat curve indicates a gradient flow problem; a U-curve (improves
then degrades) indicates inner_lr is too high for the number of steps.

---

## Known issues / things to watch

1. **`peft` in requirements but not used.** `LoRAWav2Vec2` uses manual LoRA
   (not peft) because `peft.LoraModel` is incompatible with `higher.innerloop_ctx`.
   The `peft` pin in `requirements.txt` can be removed if no other component uses it.

2. **`strategy_maml.py` stores θ* as float16.** After each round the aggregated
   theta_star is cast to float16 (`astype(np.float16)`). This halves server memory
   but accumulates quantization error over 200 rounds. Consider keeping float32 on
   the 48GB machine.

3. **`strategy_maml.py` ignores `fraction_fit`.** `configure_fit` always requests
   `min_available_clients` clients regardless of `fraction_fit`. This is intentional
   for Phase 1 (all 12 nodes participate every round), but if you want random
   cohort sampling in Phase 2 you need to wire up the fraction logic.

4. **`eval_lora.py` expects L2-ARCTIC data.** The eval script targets
   `data/l2arctic/` and `data/l2arctic_eval_clips.json`. If you evaluate on
   VCTK instead, pass `--l2arctic_dir data/vctk_nodes` or update the defaults.

5. **Differentiable CTC is 220–450× slower than `nn.CTCLoss`.** At k=5 steps
   with 8 support + 30 query clips at 3s each, expect ~45–90 minutes per
   200-round training run on a 48GB GPU. With full 7s clips and support_size=20,
   expect 3–5× longer.
