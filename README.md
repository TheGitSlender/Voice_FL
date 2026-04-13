# VoiceFL-MAML

> **Privacy-Preserving Federated Meta-Learning for Personalized Speech Recognition**

**Architecture:** Per-FedAvg MAML + Wav2Vec2-base-960h + manual DP + BAE  
**Dev hardware:** RTX 4070 Super (FOMAML) → A100 (full second-order, pending CTC fix)  
**Scale:** 20 simulated nodes (LibriSpeech speakers) → production deployment later

---

## Table of Contents

1. [Project Goal](#1-project-goal)
2. [Architecture Stack](#2-architecture-stack)
3. [Dataset](#3-dataset)
4. [Data Pipeline](#4-data-pipeline)
5. [Model: Wav2Vec2MAML](#5-model-wav2vec2maml)
6. [MAML Engine](#6-maml-engine)
7. [Task Sampler](#7-task-sampler)
8. [Federated Layer](#8-federated-layer)
9. [Privacy](#9-privacy)
10. [Security: BAE](#10-security-bae)
11. [Evaluation](#11-evaluation)
12. [Configuration](#12-configuration)
13. [Environment](#13-environment)
14. [Phase Status](#14-phase-status)
15. [Test Status](#15-test-status)
16. [Known Issues](#16-known-issues)
17. [Repository Structure](#17-repository-structure)
18. [References](#18-references)

---

## 1. Project Goal

Build a **federated meta-learning system** for personalized automatic speech recognition. Each FL client is a different speaker. The system learns a global speech model that adapts rapidly to any new speaker with only a few examples (K-shot learning via MAML).

Core problem: centralized voice AI requires uploading raw audio. VoiceFL never does — each speaker's data stays local. Only sanitized meta-gradients leave the device.

| | Centralized | VoiceFL-MAML |
|---|---|---|
| Raw voice data | Uploaded to servers | Never leaves the node |
| Personalization | One model for all | Per-speaker lm_head adaptation |
| Privacy guarantee | Policy document | (ε, δ)-DP on transmitted gradients |
| Node trust | Implicit | Zero-trust + BAE anomaly screening |

---

## 2. Architecture Stack

| Component | Choice | Reason |
|-----------|--------|--------|
| Model | `facebook/wav2vec2-base-960h` | CTC, raw waveform input, clean MAML inner loop |
| FL Algorithm | Per-FedAvg (Fallah et al. NeurIPS 2020) | Clients send meta-gradients; server applies outer lr β |
| MAML variant | FOMAML (dev) / Full MAML (A100) | First-order cheaper, second-order more accurate |
| ANIL split | encoder → FL global / lm_head → local | Fast inner loop, stable outer loop |
| Privacy | Manual DP (L2 clip + Gaussian noise) | Opacus incompatible with `higher`'s functional wrapper |
| Security | Behavioral Analysis Engine (BAE) | 4-layer anomaly detection on meta-gradients |
| FL Framework | Flower `flwr[simulation]` | Virtual Client Engine, single-machine simulation |
| Tracking | MLflow | Per-round metrics, WER curves, epsilon spend |

### Data flow

```
LibriSpeech cache
      │
      ▼
pii_masking.py  →  raw_clips.pkl (temp, per node)
      │
      ▼
features.py     →  features.pt (1D float32 waveforms) + labels.txt
                   raw_clips.pkl deleted (I1)
      │
      ▼
partition.py    →  partition_manifest.json (validated, maml_ready: true)
      │
      ▼
VoiceTaskSampler  →  (support_audio, support_labels, query_audio, query_labels)
      │
      ▼
MAMLEngine.compute_meta_gradient()
      │
      ▼
apply_dp_to_meta_gradient()  →  sanitized outer-loop grads
      │
      ▼ (over Flower VCE)
PerFedAvgStrategy.aggregate_fit()
  θ* ← θ* − β · weighted_avg(meta_grads)   [Per-FedAvg update]
```

**Critical invariants:**

| # | Invariant | Enforced in |
|---|-----------|-------------|
| I1 | Raw audio deleted after features.py | `assert not os.path.exists(pkl_path)` |
| I2 | No speaker_id in node artifacts | `data/partition.py` validation |
| I3 | `lm_head` never transmitted via FL | `Wav2Vec2MAML.get_outer_loop_params()` |
| I4 | `fit()` returns meta-gradients, not weights | `MAMLClient.fit()` |
| I5 | No Opacus — manual DP only | `privacy/dp_meta.py` |
| I6 | Support ∩ query = ∅ | `VoiceTaskSampler.sample_task()` |
| I7 | Encoder `requires_grad` stays True | `Wav2Vec2MAML.__init__()` |

---

## 3. Dataset

**LibriSpeech ASR** (train.100 split) via HuggingFace `datasets`

- 28,539 clips total — cached at `data/openslr___librispeech_asr/` (30 GB)
- 251 speakers available; **20 selected** for simulation
- Selection strategy: stratified by `duration_std` to maximize non-IID diversity
- Speaker → node mapping: `data/speaker_selection.json` (speaker IDs anonymized with one-way hash)

### Node statistics (current)

| Node | Clips | Duration | Vocab Richness |
|------|-------|----------|----------------|
| node_001 | 108 | 25.1 min | 0.339 |
| node_002 | 111 | 25.0 min | 0.313 |
| node_003 | 103 | 24.3 min | 0.299 |
| node_004 | 102 | 22.5 min | 0.313 |
| node_005 | 102 | 23.7 min | 0.257 |
| node_006 | 81 | 17.4 min | 0.283 |
| node_007 | 114 | 25.2 min | 0.323 |
| node_008 | 115 | 25.2 min | 0.296 |
| node_009 | 115 | 25.1 min | 0.328 |
| node_010 | 115 | 25.1 min | 0.332 |
| node_011 | 107 | 23.9 min | 0.256 |
| node_012 | 121 | 25.1 min | 0.320 |
| node_013 | 116 | 25.1 min | 0.273 |
| node_014 | 56 | 12.3 min | 0.370 |
| node_015 | 122 | 25.1 min | 0.345 |
| node_016 | 118 | 24.0 min | 0.290 |
| node_017 | 123 | 25.0 min | 0.275 |
| node_018 | 86 | 17.1 min | 0.258 |
| node_019 | 110 | 21.6 min | 0.353 |
| node_020 | 137 | 25.1 min | 0.359 |
| **Total** | **2,162** | **7.71 hours** | — |

Vocab richness = type-token ratio (unique words / total words) — proxy for non-IID heterogeneity.

---

## 4. Data Pipeline

Run in this order:

```bash
python data/pii_masking.py   # strips PII, saves raw_clips.pkl per node
python data/features.py      # converts to 1D waveforms, deletes pkl
python data/partition.py     # validates format, writes manifest
```

### `data/pii_masking.py`

- Reads `data/speaker_selection.json`
- Streams LibriSpeech from local HF cache (no re-download)
- Applies `data/cleaning_config.json` filters: min/max duration, silence fraction
  - Default `max_silence_fraction: 0.99` (relaxed from 0.5 — the 0.01 amplitude threshold is too aggressive for raw float32 PCM)
- Strips all PII fields: retains only `{audio array, text, duration_s}`
- Saves `data/nodes/node_XXX/raw_clips.pkl` (temporary)
- Saves `data/nodes/node_XXX/metadata.json` (no speaker_id)

### `data/features.py`

- Loads each `raw_clips.pkl`
- Per clip: cast to float32, resample to 16kHz if needed, peak-normalize to `[-1, 1]`
- Saves `features.pt` as `List[torch.Tensor]` with each tensor shape `(T_samples,)`
- Saves `labels.txt` as UPPERCASE transcriptions (one per line)
- **Deletes `raw_clips.pkl`** with assertion (invariant I1)

### Feature format

```
features.pt → list of torch.Tensor
  shape:  (T_samples,)           variable length, e.g. (92160,) for 5.76s
  dtype:  float32
  range:  [-1, 1]                peak normalized
  rate:   16 kHz
```

This is raw waveform — **not** log-mel spectrograms. Wav2Vec2 consumes raw audio directly.

### `data/partition.py`

- Validates 1D tensors, float32 dtype, values in `[-1.1, 1.1]`, ≥50 clips per node
- Computes per-node vocabulary richness
- Writes `data/partition_manifest.json` with `feature_format: raw_waveform_float32_1d` and `maml_ready: true`

---

## 5. Model: Wav2Vec2MAML

**File:** `models/wav2vec2_maml.py`  
**Base:** `Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")`

Wav2Vec2 is a CTC model. It takes raw waveform as input and outputs character-level logits in a single forward pass. There is no decoder autoregression — this makes the MAML inner loop clean and fast.

### ANIL Split

```
wav2vec2 encoder  →  ~94M params  →  outer loop  →  FL aggregated (global)
lm_head           →  ~25K params  →  inner loop  →  stays local (personalization)
```

```python
get_inner_loop_params() → list(model.lm_head.parameters())     # adapts per speaker
get_outer_loop_params() → [p for name,p in model.named_parameters()
                            if not name.startswith("lm_head")]  # transmitted via FL
forward(input_values, labels) → Wav2Vec2ForCTCOutput            # .loss (CTC), .logits
decode(logits) → List[str]                                      # greedy CTC argmax
```

**Why not freeze the encoder?** Invariant I7: `encoder.requires_grad = True` always. For second-order MAML (full mode), the meta-gradient must flow through `lm_head → encoder`. Freezing the encoder would break the outer loop update.

**Model sizes:**
- Total: ~94M + 25K ≈ 94M params
- Encoder: ~94M (wav2vec2.*)
- lm_head: ~25K (768 → 32 linear)

---

## 6. MAML Engine

**File:** `maml/engine.py`

```python
@dataclass
class MAMLConfig:
    mode: str = "fomaml"    # "full" | "fomaml" | "reptile"
    k: int = 3              # inner loop steps
    inner_lr: float = 1e-4  # α
    outer_lr: float = 2e-4  # β (centralized mode)
    adaptation_mode: str = "anil"
    use_bf16: bool = True

class MAMLEngine:
    compute_meta_gradient(sup_audio, sup_labels, qry_audio, qry_labels)
        → (meta_grads: List[Tensor|None], query_loss: float)
    adapt(audio, labels, k) → adapted_model_copy
```

### Three modes

**FOMAML** (`mode="fomaml"`) — RTX 4070 Super

```
1. save init_state = deepcopy(model.state_dict())
2. run k SGD steps on lm_head using support set
3. evaluate on query set at adapted lm_head
4. meta_grads = autograd.grad(query_loss, model.parameters(), create_graph=False)
5. model.load_state_dict(init_state)  ← restores original state
```

First-order approximation: gradient at `θ'` (adapted params), no Hessian term. Equivalent to `learn2learn.MAML(first_order=True)` but without Cython (which breaks on Python 3.13+).

**Full MAML** (`mode="full"`) — A100 required (currently broken, see §16)

```
higher.innerloop_ctx(track_higher_grads=True):
    k differentiable SGD steps on lm_head
    query forward pass
autograd.grad flows THROUGH the inner loop → Hessian term included
```

**Reptile** (`mode="reptile"`) — RTX 4070 Super

```
1. save init_params
2. k SGD steps on full model
3. meta_grad = (θ_init - θ_adapted) / k
4. restore init_params
```

### `maml/meta_train.py`

Centralized validation gate before federating:

```python
run_centralized_meta_training(model, engine, task_samplers, n_epochs=50)
# outer optimizer: AdamW(model.parameters(), lr=2e-4, wd=0.01)
# gate: WER(k=3) < WER(k=0) on ≥12/20 nodes
```

### `maml/meta_eval.py`

```python
evaluate_adaptation_at_k(k_values=[0, 1, 3, 5, 10])
# k=0: eval θ* directly (no adaptation)
# k>0: engine.adapt(support, k) → eval adapted model → compute WER via jiwer
```

---

## 7. Task Sampler

**File:** `data/task_sampler.py`

```python
class VoiceTaskSampler:
    def __init__(self, node_dir, processor, K=8, Q=8, device='cpu')
    def sample_task(seed=None)
        → (support_audio, support_labels, query_audio, query_labels)
    def sample_eval_batch(n=10) → (audio, labels)
```

- Samples K+Q non-overlapping clip indices (invariant I6: support ∩ query = ∅ always)
- Audio batching: `Wav2Vec2Processor` pads variable-length 1D arrays to `(K, T_max)`
- Label tokenization: `processor.tokenizer(texts, ...)` — note: `as_target_processor()` was removed in transformers 5.x
- Padding token → `-100` for CTC loss masking

---

## 8. Federated Layer

### `federated/client_maml.py` — `MAMLClient(NumPyClient)`

```python
get_parameters() → encoder params as numpy arrays     # lm_head excluded (I3)
set_parameters() → restores encoder from server arrays
fit():
    1. set_parameters() from server
    2. check accountant.is_exhausted() → skip if ε budget spent
    3. sample_task() → support + query
    4. compute_meta_gradient()
    5. _extract_outer_grads()  # align grads to encoder params by id(p)
    6. apply_dp_to_meta_gradient()
    7. accountant.step()
    8. return grad_numpy, n_samples, {query_loss, grad_norm, epsilon}
evaluate():
    → evaluate_adaptation_at_k([0, k])
    → returns (wer_k, n_clips, {wer, wer_0shot, adaptation_gain})
```

### `federated/strategy_maml.py` — `PerFedAvgStrategy(FedAvg)`

Per-FedAvg is gradient descent on the server, not weight averaging:

```
FedAvg:      θ_new = weighted_avg(θ_clients)
Per-FedAvg:  θ* ← θ* − β · weighted_avg(meta_grads)
```

```python
aggregate_fit():
    1. extract meta-gradients from client results
    2. BAE screening → adjusted_weights (optional)
    3. weighted_avg(meta_grads, adjusted_weights)
    4. θ* ← θ* − β · avg_gradient
    5. MLflow: query_loss, epsilon_avg, active_nodes

aggregate_evaluate():
    → aggregate per-node WER + adaptation_gain
    → MLflow: eval/mean_wer, eval/mean_adaptation_gain
```

### `federated/simulation_maml.py` — entry point

```bash
python federated/simulation_maml.py --config configs/dev.yaml
```

Loads all 20 `VoiceTaskSamplers`, pre-shares `Wav2Vec2Processor`, creates `client_fn(cid)` factory, runs `fl.simulation.start_simulation(client_fn, 20, strategy=PerFedAvgStrategy)`.

---

## 9. Privacy

**Files:** `privacy/dp_meta.py`, `privacy/rdp_accountant.py`

### Mechanism

Applied to the **outer-loop meta-gradient** (encoder params only) before transmission. Never applied to `lm_head` (never transmitted — I3).

```
Step 1 — L2 clip:   ΔW̄ = ΔW · min(1, C / ‖ΔW‖₂)
Step 2 — Noise:     ΔW_private = ΔW̄ + N(0, σ²C²I)

C = clipping threshold (default 1.0)
σ = noise multiplier (calibrated via autodp RDP)
```

```python
# privacy/dp_meta.py
DPConfig(epsilon, delta, C, sigma, enabled, sample_rate)
compute_sigma(epsilon, delta, C) → float
apply_dp_to_meta_gradient(meta_grads, C, sigma) → (sanitized_grads, original_norm)

# privacy/rdp_accountant.py
RDPAccountant(target_epsilon, target_delta)
    .step(noise_multiplier, sample_rate)   # compose RDP mechanism per round
    .get_epsilon()                          # current ε spend
    .is_exhausted()                         # True if ε >= target_epsilon
    .summary()                              # "Steps: N | ε spent: X / Y | Remaining: Z"
```

**No Opacus** (I5): `higher.innerloop_ctx` wraps the model in a stateless functional form. Opacus requires `nn.Module` hooks to track per-sample gradients — incompatible. DP is applied manually to the already-computed meta-gradient tensor.

### Privacy budget reference

| Use case | ε | WER impact |
|----------|---|------------|
| Dev (no DP) | ∞ | None |
| Dev (light DP) | 8 | < 5% |
| Production target | 4 | 5–15% |
| High-security | 2 | 15–30% |

---

## 10. Security: BAE

**Files:** `security/bae_maml.py`, `security/attacks/`

### 4-Layer Screening Pipeline

```
Round N meta-gradients → flatten to 1D vectors
    │
    ├─ Layer 1: cosine_anomaly(flat, round_median)
    │           score = 1 - max(cosine_sim, 0)
    │           detects: Byzantine, sign-flipping
    │
    ├─ Layer 2: norm_feature(node_id, flat)
    │           z-score vs node history, clip to [0,1]
    │           detects: gradient amplification, poisoning
    │
    ├─ Layer 3: temporal_feature(node_id, current_cosine)
    │           deviation from recent cosine trend
    │           detects: delayed poisoning, Sybil takeover
    │
    └─ Layer 4: IsolationForest(contamination=0.1)
                joint anomaly score over [f1, f2, f3]
                requires ≥4 nodes; falls back to cosine score
```

### Response tiers

```
score < 0.6:              weight = 1.0              (pass)
0.6 ≤ score < 0.8:        weight = 1.0 - score      (soft penalty)
0.8 ≤ score < 0.95:       weight = 0.0, quarantined
score ≥ 0.95:             weight = 0.0, permanently excluded
```

Audit log: `security/audit_log.jsonl`

### Attack simulators

```
security/attacks/byzantine_maml.py   — random gradient (fully adversarial)
security/attacks/freerider_maml.py   — zero gradient (parasitic)
security/attacks/poisoning_maml.py   — clean + scaled noise (stealth)
security/attacks/sybil_maml.py       — n_fake near-copies of real gradient
```

Integration: `PerFedAvgStrategy.aggregate_fit()` calls `bae.screen_updates(grad_dict, round_num)` **before** computing the weighted average.

---

## 11. Evaluation

```python
# evaluation/eval_maml.py
run_full_evaluation()
    → WER at k=0,1,3,5,10 per node
    → aggregate mean/std/min/max
    → saves evaluation/results/eval_results.json
    → logs to MLflow

# evaluation/ablation.py
run_ablation()
    → MAML mode (fomaml / reptile) × k (1,3,5) × epsilon (∞,8,4) × ANIL ablation
```

Primary gate: **WER(k=3) < WER(k=0) on ≥12/20 nodes** — confirms MAML actually improves over zero-shot.

---

## 12. Configuration

### `configs/dev.yaml` — RTX 4070 Super

```yaml
maml.mode: fomaml
maml.k: 3
maml.inner_lr: 1.0e-4
maml.outer_lr: 2.0e-4
privacy.enabled: false
bae.enabled: false
hardware.device: cuda
hardware.gpu_per_client: 0.5
fl.num_rounds: 50
fl.clients_per_round: 2
```

### `configs/experiment.yaml` — A100

```yaml
maml.mode: full          # NOTE: currently broken — see §16
maml.k: 3
privacy.enabled: true
privacy.epsilon: 4.0
privacy.delta: 1.0e-5
privacy.C: 1.0
hardware.device: cuda
hardware.gpu_per_client: 1.0
fl.num_rounds: 100
fl.clients_per_round: 5
```

---

## 13. Environment

```
Python 3.13.5
torch >= 2.0.0
torchaudio >= 2.0.0
transformers 5.5.0          (Wav2Vec2ForCTC)
higher 0.2.1                (second-order MAML — see §16 for CTC limitation)
autodp >= 0.2               (RDP accounting)
flwr[simulation] 1.28.0     (Flower + Virtual Client Engine)
mlflow 3.10.1
jiwer >= 3.0.0              (WER computation)
scikit-learn >= 1.2.0       (IsolationForest in BAE)
datasets >= 2.0.0
soundfile >= 0.12.0
librosa >= 0.10.0
numpy >= 1.24.0
tqdm >= 4.65.0
pyyaml >= 6.0
evaluate >= 0.4.0
```

**Intentionally absent:**

| Package | Reason |
|---------|--------|
| `opacus` | Incompatible with `higher.innerloop_ctx` functional model |
| `learn2learn` | Python 3.13 build failure — Cython extension missing `longintrepr.h` (removed in CPython 3.12). FOMAML is implemented natively via `torch.autograd.grad(create_graph=False)`. |

---

## 14. Phase Status

| Phase | Status | Entry point |
|-------|--------|-------------|
| 0 — Verification | COMPLETE | `scripts/verify_env.py` |
| 1 — Data Pipeline | **COMPLETE** | `data/pii_masking.py → features.py → partition.py` |
| 2 — MAML Engine | Code done, tests partial | `maml/engine.py`, `maml/meta_train.py` |
| 3 — Model | COMPLETE | `models/wav2vec2_maml.py` |
| 4 — Federated Layer | Code done, untested | `federated/simulation_maml.py --config configs/dev.yaml` |
| 5 — Privacy | COMPLETE | `privacy/dp_meta.py`, `privacy/rdp_accountant.py` |
| 6 — Security (BAE) | COMPLETE | `security/bae_maml.py` |
| 7 — Evaluation + Config | COMPLETE | `evaluation/eval_maml.py`, `evaluation/ablation.py` |

### Pending gates (in order)

| Step | Command | Gate |
|------|---------|------|
| 1 | Fix 4 failing MAML engine tests | 8/8 tests passing |
| 2 | `python maml/meta_train.py` | WER(k=3) < WER(k=0) on ≥12/20 nodes |
| 3 | `python federated/simulation_maml.py` (5 rounds) | Params change each round, metrics in MLflow |
| 4 | Full 50-round dev run (DP + BAE) | WER improves over rounds |
| 5 | A100: switch to `experiment.yaml` | Depends on CTC second-order fix |

---

## 15. Test Status

```
tests/test_dp_meta.py       17 tests   ALL PASS
tests/test_bae.py           12 tests   ALL PASS
tests/test_task_sampler.py  13 tests   ALL PASS
tests/test_maml_engine.py    8 tests   4 PASS / 4 FAIL
                            ─────────────────────
Total:                      50 tests   46 PASS / 4 FAIL
```

Run all:
```bash
pytest tests/ -v
```

---

## 16. Known Issues

### 1. Full MAML broken with CTC loss

`_full_maml()` in `maml/engine.py` uses `higher.innerloop_ctx(track_higher_grads=True)`. This computes second-order gradients by differentiating through the CTC loss backward pass. PyTorch does not implement `derivative for aten::_ctc_loss_backward`, so this always fails.

**Impact:** `configs/experiment.yaml` cannot use `maml.mode: full`. The A100 experiment path has no second-order MAML.

**Options:**
- Switch inner loop loss to token-level cross-entropy (drops CTC, requires label alignment changes)
- Accept FOMAML as the only viable mode and treat second-order MAML as out of scope
- Investigate GradScaler / detach patterns that might allow approximation

### 2. MAML engine test fixture contamination

4 tests in `tests/test_maml_engine.py` fail when the full file runs but pass in isolation. Root cause: `scope="module"` fixtures share one `Wav2Vec2MAML` instance across all 8 tests. A residual CTC computation graph leaks from one test into the next, causing `autograd.grad` to unexpectedly attempt a second-order derivative.

**Fix:** change fixture scope from `"module"` to `"function"`, or add explicit `model.zero_grad()` and graph detachment between tests.

---

## 17. Repository Structure

```
voice_fl/
├── README.md                         # This file
├── requirements.txt
├── data/
│   ├── download.py                   # Speaker selection from LibriSpeech
│   ├── pii_masking.py                # PII stripping, raw_clips.pkl per node
│   ├── features.py                   # 1D waveform extraction, pkl deletion
│   ├── partition.py                  # Validation, partition_manifest.json
│   ├── task_sampler.py               # VoiceTaskSampler — K-shot tasks
│   ├── generate_report.py            # Data visualization (standalone)
│   ├── cleaning_config.json          # Silence/duration filter settings
│   ├── speaker_selection.json        # 20 selected speakers (anonymized)
│   ├── partition_manifest.json       # Generated by partition.py
│   └── nodes/
│       └── node_001/ ... node_020/
│           ├── features.pt           # List[Tensor shape (T_samples,)]
│           ├── labels.txt            # UPPERCASE transcriptions
│           └── metadata.json         # clip count, duration stats (no speaker_id)
├── models/
│   ├── __init__.py
│   └── wav2vec2_maml.py              # Wav2Vec2MAML wrapper, ANIL split
├── maml/
│   ├── __init__.py
│   ├── engine.py                     # MAMLEngine: full / fomaml / reptile
│   ├── meta_train.py                 # Centralized validation gate
│   └── meta_eval.py                  # WER at k=0,1,3,5,10
├── federated/
│   ├── __init__.py
│   ├── client_maml.py                # MAMLClient (NumPyClient)
│   ├── strategy_maml.py              # PerFedAvgStrategy
│   └── simulation_maml.py            # Flower VCE entry point
├── privacy/
│   ├── __init__.py
│   ├── dp_meta.py                    # DPConfig, apply_dp_to_meta_gradient()
│   └── rdp_accountant.py             # RDPAccountant wrapping autodp
├── security/
│   ├── __init__.py
│   ├── bae_maml.py                   # BehavioralAnalysisEngine (4 layers)
│   ├── audit_log.jsonl               # Per-event anomaly log
│   └── attacks/
│       ├── __init__.py
│       ├── byzantine_maml.py
│       ├── freerider_maml.py
│       ├── poisoning_maml.py
│       └── sybil_maml.py
├── evaluation/
│   ├── __init__.py
│   ├── eval_maml.py                  # run_full_evaluation()
│   ├── ablation.py                   # run_ablation()
│   └── results/
├── configs/
│   ├── dev.yaml                      # RTX 4070 Super — FOMAML, no DP
│   └── experiment.yaml               # A100 — full MAML, DP ε=4.0
├── tests/
│   ├── __init__.py
│   ├── test_task_sampler.py          # 13 tests — all pass
│   ├── test_maml_engine.py           # 8 tests — 4 pass / 4 fail (see §16)
│   ├── test_dp_meta.py               # 17 tests — all pass
│   └── test_bae.py                   # 12 tests — all pass
├── scripts/
│   └── verify_env.py                 # Environment sanity check
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_maml_toy_verification.ipynb
└── docs/
    └── CODEMAPS/
        ├── architecture.md
        ├── data.md
        ├── dependencies.md
        ├── phases.md
        ├── maml_federated.md
        └── privacy_security.md
```

---

## 18. References

- Fallah et al. (2020) — [Per-FedAvg: Personalized Federated Learning](https://arxiv.org/abs/2002.07948) — the core FL algorithm
- Finn et al. (2017) — [MAML: Model-Agnostic Meta-Learning](https://arxiv.org/abs/1703.03400) — inner/outer loop framework
- Raghu et al. (2020) — [ANIL: Rapid Learning or Feature Reuse?](https://arxiv.org/abs/1909.09157) — ANIL split justification
- Baevski et al. (2020) — [wav2vec 2.0](https://arxiv.org/abs/2006.11477) — the base model
- Abadi et al. (2016) — [DP-SGD](https://arxiv.org/abs/1607.00133) — differential privacy for ML
- McMahan et al. (2017) — [FedAvg](https://arxiv.org/abs/1602.05629) — federated averaging baseline
- Blanchard et al. (2017) — [Krum](https://proceedings.neurips.cc/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html) — Byzantine-robust aggregation
- [Flower Framework](https://flower.ai) — `flwr[simulation]` for FL simulation
- [higher](https://github.com/facebookresearch/higher) — differentiable inner loop for second-order MAML

---

*VoiceFL-MAML · Per-FedAvg + Wav2Vec2 + manual DP + BAE · 20 nodes · 2,162 clips · 7.71 hours*
