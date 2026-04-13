# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Environment check
python scripts/verify_env.py

# Install dependencies
pip install -r requirements.txt

# Data pipeline (run in order)
python data/pii_masking.py
python data/features.py
python data/partition.py

# Tests
pytest tests/ -v
pytest tests/test_dp_meta.py -v          # single file
pytest tests/ --cov=. --cov-report=term-missing  # with coverage

# Federated simulation
python federated/simulation_maml.py --config configs/dev.yaml

# Centralized MAML gate (run before federating)
python maml/meta_train.py

# Full evaluation
python -c "from evaluation.eval_maml import run_full_evaluation; run_full_evaluation()"
```

## Architecture

**Goal:** Federated meta-learning for personalized speech recognition. 20 simulated FL nodes (LibriSpeech speakers). Each node adapts `lm_head` locally; only encoder meta-gradients are transmitted.

### Data flow

```
LibriSpeech HF cache
  → pii_masking.py     raw_clips.pkl per node (temp)
  → features.py        features.pt (List[Tensor shape (T_samples,)], float32, 16kHz, [-1,1]) + labels.txt; deletes pkl
  → partition.py       partition_manifest.json (maml_ready: true)
  → VoiceTaskSampler   K-shot tasks: (support_audio, support_labels, query_audio, query_labels)
  → MAMLEngine         meta-gradients
  → apply_dp_to_meta_gradient()  sanitized outer-loop grads
  → PerFedAvgStrategy  θ* ← θ* − β · weighted_avg(meta_grads)
```

### ANIL split (critical)

```
wav2vec2 encoder  (~94M params)  →  outer loop  →  aggregated by FL server
lm_head           (~25K params)  →  inner loop  →  stays local, never transmitted
```

The encoder must keep `requires_grad=True` (invariant I7). Second-order MAML needs gradients to flow through `lm_head → encoder`. Never freeze the encoder.

### MAML modes

- **FOMAML** (`mode="fomaml"`) — default dev mode, RTX 4070 Super. `create_graph=False`. No `higher` dependency at runtime.
- **Full MAML** (`mode="full"`) — **currently broken**: `higher.innerloop_ctx` + CTC loss fails because PyTorch has no `aten::_ctc_loss_backward` derivative. Do not attempt on `configs/experiment.yaml` until this is resolved.
- **Reptile** (`mode="reptile"`) — fallback, no second-order, runs on RTX 4070 Super.

`learn2learn` is intentionally absent — its Cython extension breaks on Python 3.13+. FOMAML is implemented natively in `maml/engine.py`.

### Key invariants

| # | Invariant | Where enforced |
|---|-----------|---------------|
| I1 | Raw audio (`raw_clips.pkl`) deleted after `features.py` | `assert not os.path.exists(pkl_path)` |
| I2 | No `speaker_id` in node artifacts | `data/partition.py` validation |
| I3 | `lm_head` never transmitted via FL | `Wav2Vec2MAML.get_outer_loop_params()` |
| I4 | `fit()` returns meta-gradients, not weights | `MAMLClient.fit()` |
| I5 | No Opacus — manual DP only | `privacy/dp_meta.py` |
| I6 | Support ∩ query = ∅ | `VoiceTaskSampler.sample_task()` |
| I7 | Encoder `requires_grad` stays True | `Wav2Vec2MAML.__init__()` |

### Privacy (DP)

Applied to encoder meta-gradients only (never `lm_head` — I3). Manual L2 clip + Gaussian noise (`privacy/dp_meta.py`). RDP accounting via `autodp` (`privacy/rdp_accountant.py`). Opacus is intentionally not used — it is incompatible with `higher.innerloop_ctx`.

### Security (BAE)

`security/bae_maml.py` runs a 4-layer screening pipeline on meta-gradients each round: cosine anomaly → norm z-score history → temporal trend → IsolationForest. Called inside `PerFedAvgStrategy.aggregate_fit()` before the weighted average. Scores ≥ 0.95 permanently exclude a node.

### FL protocol (Per-FedAvg vs FedAvg)

This is **not** weight averaging. Clients send meta-gradients; the server does gradient descent: `θ* ← θ* − β · weighted_avg(meta_grads)`. This matters when reading `strategy_maml.py` — never confuse it with standard FedAvg.

## Known issues

1. **Full MAML broken**: `higher.innerloop_ctx(track_higher_grads=True)` + CTC loss raises a missing-derivative error. `configs/experiment.yaml` (`maml.mode: full`) is non-functional.
2. **MAML engine test contamination**: 4/8 tests in `tests/test_maml_engine.py` fail when the full suite runs (pass in isolation). Root cause: `scope="module"` fixture shares one model across tests; a residual CTC graph leaks between tests. Fix: change fixture scope to `"function"` or add `model.zero_grad()` + graph detachment between tests.

## Configs

- `configs/dev.yaml` — RTX 4070 Super, FOMAML, DP disabled, 50 FL rounds, 2 clients/round
- `configs/experiment.yaml` — A100, full MAML (broken), DP ε=4.0, 100 rounds, 5 clients/round

## Phase status

| Phase | Status |
|-------|--------|
| 0 — Env verification | Complete |
| 1 — Data pipeline | Complete |
| 2 — MAML engine | Code done, 4/8 tests failing (see known issue 2) |
| 3 — Model | Complete |
| 4 — Federated layer | Code done, untested end-to-end |
| 5 — Privacy | Complete |
| 6 — Security (BAE) | Complete |
| 7 — Evaluation + config | Complete |

Next gate: fix `test_maml_engine.py` (fixture scope) → run `maml/meta_train.py` → gate is WER(k=3) < WER(k=0) on ≥12/20 nodes.
