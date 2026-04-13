# VoiceFL-MAML PoC

**Proves:** A shared meta-initialization θ* adapts to a new speaker's voice
faster than a randomly initialized model, with voice data never leaving its node.

---

## Phase

**Phase 1 — PoC (v0.1-poc)**. Tag `v0.1-poc` before adding Phase 2 features.

---

## What This PoC Does NOT Include

- No differential privacy
- No SecAgg or cryptographic masking
- No behavioral analysis engine (BAE)
- No attack simulation
- No mTLS or certificates (plain gRPC only)
- No cloud deployment
- No MLflow or experiment tracking (stdout logging only)
- No second-order MAML (FOMAML only, `first_order=True`)
- No more than 5 nodes
- No TTS, no LLM integration, no STS
- No Whisper (Wav2Vec2 only)
- No per-layer learning rate tuning
- No iMAML or MAML++

---

## Invariants (Non-Negotiable)

| # | Invariant | Checked By |
|---|-----------|-----------|
| I1 | Raw audio never persists after `features.py` runs. No `.pkl` files under `data/nodes/`. | `eval_poc.py` |
| I2 | Speaker identity never stored in node artifacts. No `speaker_id` in `data/nodes/`. | `eval_poc.py` |
| I3 | `lm_head` parameters never included in `get_parameters()` or `set_parameters()`. Flower client serializes encoder weights only. | `eval_poc.py` |
| I4 | `fit()` returns meta-gradients, not weight updates. Strategy aggregates gradients and applies β to update θ*. Standard FedAvg weight averaging is not used. | `eval_poc.py` |
| I5 | Support and query sets have zero index overlap. Asserted in `task_sampler.sample_task()`. | `eval_poc.py` |
| I6 | Encoder `requires_grad` stays `True` throughout training. Setting it to `False` is not permitted. | `eval_poc.py` |

---

## Success Criteria

**1. ADAPTATION**
WER at k=3 < WER at k=0 on held-out clips, for at least 4 out of 5 nodes.
Measured by `evaluation/eval_poc.py`.

**2. FEDERATION**
θ* from 20 federated rounds produces WER within 15% of θ* from centralized
MAML run on the same combined data.
Measured by `evaluation/eval_poc.py`.

**3. SOVEREIGNTY**
`grep -r "speaker_id" data/nodes/` returns nothing.
`find data/nodes -name "*.pkl"` returns nothing.
Checked by `evaluation/eval_poc.py`.

---

## Architecture

**Model:** `facebook/wav2vec2-base-960h` (~95M params)
**MAML:** FOMAML via `learn2learn`, `first_order=True`
**FL:** Flower (flwr), PerFedAvg strategy (gradient aggregation, not weight averaging)
**Dataset:** LibriSpeech `train-clean-100`, 5 speakers, low heterogeneity

**ANIL split:**
- Inner loop: `lm_head` only (~25K params), stays local, never transmitted
- Outer loop / FL: `wav2vec2` encoder only (~94.5M params)

**Data flow:**
```
download.py → pii_masking.py → features.py → task_sampler
    → MAMLEngine.compute_meta_gradient()
    → client_maml.fit() returns meta-gradients
    → strategy_maml.aggregate_fit() applies θ* ← θ* − β · avg(grads)
```

---

## Key Parameters

| Parameter | Value |
|-----------|-------|
| Rounds | 20 |
| Cohort | All 5 nodes (fraction_fit=1.0) |
| Inner steps (k) | 3 |
| Inner LR (α) | 1e-4 |
| Outer LR (β) | 2e-4 |
| Support size (K) | 8 |
| Query size (Q) | 8 |
| Batch size | 1 (process clips one at a time) |
| VRAM budget/node | 3.5 GB |
| Precision | BF16 |

---

## Build Order

- [ ] **STEP 1** — Environment check: `python -c "import learn2learn, flwr, transformers"` exits 0
- [ ] **STEP 2** — Toy verification: `notebooks/01_maml_toy_verify.ipynb` — query loss k=3 < k=0
- [ ] **STEP 3** — Data pipeline: 5 node dirs, no `.pkl`, no `speaker_id`, float32 tensors
- [ ] **STEP 4** — Task sampler: `sample_task()` returns 4 tensors, no index overlap
- [ ] **STEP 5** — Model wrapper: inner/outer params split correct, no overlap, no gap
- [ ] **STEP 6** — MAML engine: `compute_meta_gradient()` returns non-None, shapes match
- [ ] **STEP 7** — Centralized validation: WER(k=3) < WER(k=0) on ≥4/5 nodes
- [ ] **STEP 8** — Docker build: `docker-compose build` exits 0, both images visible
- [ ] **STEP 9** — Federated smoke test: 5-round run, all nodes send gradients, θ* changes
- [ ] **STEP 10** — Full PoC: `eval_poc.py` passes all 3 success criteria

---

## Commands

```bash
# Environment
pip install -r requirements.txt
python -c "import learn2learn, flwr, transformers; print('OK')"

# Data pipeline (run in order)
python data/download.py
python data/pii_masking.py
python data/features.py

# Centralized gate (Step 7)
python maml/meta_train.py --config configs/poc.yaml

# Docker
./scripts/build.sh
docker-compose -f docker/docker-compose.yml up

# Full PoC evaluation
./scripts/run_poc.sh
python evaluation/eval_poc.py
```

---

*VoiceFL-MAML PoC · 5 nodes · FOMAML · ANIL · Wav2Vec2 · Docker · No DP*
*Low heterogeneity · Same hardware · LibriSpeech train-clean-100 · 20 rounds*
