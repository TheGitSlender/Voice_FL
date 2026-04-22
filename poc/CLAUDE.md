# VoiceFL-MAML PoC

**Proves:** A federated meta-initialization θ* enables a NEW, UNSEEN speaker
to adapt a speech recognition model to their voice faster than the pretrained
baseline, with voice data never leaving its node.

The words NEW and UNSEEN are load-bearing. Meta-test speakers must never appear
in any training round, and the base model must not have been pretrained on them.

---

## Phase

**Phase 1 — PoC (v0.1-poc)**. Tag `v0.1-poc` before adding Phase 2 features.

**Phase 2 — FedLoRA-MAML**. See [FedLoRA-MAML section](#fedlora-maml-phase-2)
below. Phase 2 must not touch Phase 1 meta-test data or results. Tag `v0.2-lora`
after verifying `eval_lora.py` runs end-to-end on dummy data.

---

## DESIGN DECISION — MAML Modes

### FOMAML (default)
**Algorithm:** FOMAML (Nichol et al. 2018, first-order approximation).
Uses `copy.deepcopy` + manual SGD. Fast, runs on RTX 4070 Super.

### Second-Order MAML via Custom Differentiable CTC
**File:** `ctc/differentiable_ctc.py`
**Mode:** `second_order_ctc` in `MAMLEngine`

PyTorch's `nn.CTCLoss` calls `aten::_ctc_loss_backward`, a C++ kernel not
registered for second derivatives. `higher.innerloop_ctx(track_higher_grads=True)`
raises `RuntimeError: derivative for _ctc_loss_backward is not implemented`.

**Solution:** A custom CTC loss implemented entirely in native PyTorch ops
(logsumexp, gather, cat, where) — all support `create_graph=True`. The full
forward-backward DP is traceable through autograd.

**Verification:** 5 tests in `scripts/verify_differentiable_ctc.py` (all pass):
1. Loss matches `nn.CTCLoss` within 1e-4 relative error
2. Double-backward succeeds without RuntimeError
3. Second-order gradient differs from first-order (cosine sim < 0.99)
4. Hessian diagonal matches finite difference within 5% (definitive correctness)
5. Full MAML inner loop: 210/211 encoder params get non-None finite meta-gradients

**Speed:** ~220-450x slower than native CTC kernel (absolute: ~31ms for T=100,
~165ms for T=500). The overhead is the Python loop over T time steps vs a single
CUDA kernel. Acceptable for k≤5 inner steps with short clips.

**Limitation:** Python loop over T frames — not parallelized across time.
Precomputed emissions (`log_probs[:, targets_ext]`) avoid per-step gather calls.

---

## DATASET — LibriSpeech test-clean + dev-clean

**Dataset: LibriSpeech `test` and `validation` splits (openslr/librispeech_asr, clean config)**

The original PoC used `train-clean-100` speakers — those ARE in wav2vec2-base-960h's
fine-tuning data (train-960h = train-100 + train-360 + train-other-500). All WER
numbers were invalid.

VCTK was planned as the replacement (genuinely out-of-domain, ~11 GB download).
In practice, LibriSpeech test-clean + dev-clean are already cached and are the correct
held-out speakers: wav2vec2-base-960h was NEVER fine-tuned on these speakers.
The lm_head has not been adapted to their speech patterns. The k=0 WER gap is real.

**Scientific validity:**
- Fine-tuned on: train-clean-100 + train-clean-360 + train-other-500 (960h)
- NOT fine-tuned on: test-clean (40 speakers) + dev-clean (40 speakers)
- Self-supervised pretraining used train splits only (not test/dev)
- 80 candidate speakers, 32–108 clips each; 76 have ≥40 clips

**Speaker split (20 speakers, locked in `data/meta_split.json`):**

| Split | Count | Purpose |
|-------|-------|---------|
| meta-train | 12 | Federation nodes — θ* trained on these |
| meta-val | 4 | Hyperparameter tuning only |
| meta-test | 4 | Final evaluation only — fewest clips (hardest speakers) |

**meta-test speakers are locked before any training decisions.**
Commit `data/meta_split.json` to git before running any training.

Audio: already 16kHz — no resampling needed.
Data prep: `data/prepare_librispeech.py` (reads from HuggingFace cache, no download).

---

## EVALUATION INTEGRITY

1. **meta-test speakers are evaluated ONCE, at the end.**
   Running `eval_poc.py` on meta-test before training is complete invalidates results.

2. **Hyperparameters are tuned on meta-val only.**
   Never adjust hyperparameters after seeing meta-test results.

3. **All WER estimates include bootstrap 95% CI** (1000 resamples, 20 query clips).
   Do not report point estimates without confidence intervals.
   If CIs of MAML vs baseline overlap, the result is not statistically significant.

4. **Null results are documented, not hidden.**
   If WER_federated_k10 ≈ WER_pretrained_k10, this is a valid scientific finding:
   FOMAML over 20 rounds with 12 speakers does not improve adaptation on VCTK
   relative to direct fine-tuning. Document it accurately.

5. **No perturbation protocol.**
   The original lm_head Gaussian noise perturbation has been removed.
   VCTK speakers are genuinely unseen — the k=0 WER gap is real without
   artificial noise injection.

6. **Federation comparison controls for total gradient updates, not rounds.**
   Centralized matched-update baseline = same total gradient updates as federation.
   Do not compare 20 federated rounds against 20 centralized epochs (different computation).

---

## What This PoC Does NOT Include

- No differential privacy
- No SecAgg or cryptographic masking
- No behavioral analysis engine (BAE)
- No attack simulation
- No mTLS or certificates (plain gRPC only)
- No cloud deployment
- No MLflow or experiment tracking (stdout logging only)
- No more than 12 training nodes (Phase 1); 12 nodes in Phase 2 also
- No TTS, no LLM integration, no STS
- No Whisper (Wav2Vec2 only)
- No per-layer learning rate tuning
- No iMAML or MAML++

---

## FedLoRA-MAML (Phase 2)

### PHASE 2 — FedLoRA-MAML

**Model:** Wav2Vec2-base-960h with LoRA in transformer layers 6–11
**Trainable:** ~320K params (LoRA lora_A, lora_B + lm_head)
**Algorithm:** `lora_maml` mode — true second-order MAML via custom CTC
**Dataset:** L2-ARCTIC (24 non-native English speakers, 6 L1 groups)
**Meta-test:** SKA, YBAA (Arabic L1), PNV, THV (Vietnamese L1)
**Status:** [TODO — see Phase 2 Build Order]

### DESIGN DECISION — Second-Order Accuracy

FedLoRA-MAML achieves TRUE second-order MAML on CTC.
Enabled by `ctc/differentiable_ctc.py` — custom CTC in pure PyTorch.
The Hessian captures:
  - LoRA subspace curvature — how adaptation changes gradient landscape
  - CTC alignment curvature — how alignment paths shift during adaptation

Do NOT call this FOMAML or "partial second-order."
It is full second-order MAML, restricted to the LoRA parameter subspace.

**Why eager attention (`attn_implementation="eager"`):**
`F.scaled_dot_product_attention`'s CPU backend calls
`aten::_scaled_dot_product_flash_attention_for_cpu`, which has no registered
second derivative. Eager mode uses explicit bmm-based attention that autograd
can differentiate through twice. This is set at model load time in
`LoRAWav2Vec2.__init__()`.

**IMPORTANT — dropout/higher interaction:**
Do NOT call `model.train()` when using `higher.innerloop_ctx`. wav2vec2's dropout
layers produce NaN logits when combined with higher's functional parameter patching.
Gradients flow correctly in eval mode.

### DESIGN DECISION — Why LoRA Layers 6–11 Only

Probing studies show layers 0–5 encode speaker-independent acoustic-phonetic
features (formants, pitch structure). Layers 6–11 encode linguistic context
and speaker-specific prosodic patterns. Adapting upper layers captures
speaker variation without disrupting universal acoustic representations.

### Architecture

**Model:** `models/lora_wav2vec2.py` — `LoRAWav2Vec2`
**Dataset:** L2-ARCTIC corpus (24 non-native English speakers, 6 L1 groups, 22050→16kHz)
**Config:** `configs/lora_poc.yaml`
**Eval:** `evaluation/eval_lora.py`

**LoRA injection:**
- Layers: transformer layers 6–11 (upper encoder half)
- Projections: q_proj, k_proj, v_proj, out_proj
- r=8, alpha=16, scaling=2.0
- ~295K LoRA params (lora_A + lora_B matrices)
- Base encoder weights frozen (never trained, never transmitted)

**Parameter groups in FedLoRA-MAML (NOT ANIL):**
| Group | Params | Role |
|-------|--------|------|
| LoRA lora_A/lora_B | ~295K | Inner loop + FL aggregation. Meta-gradient target. |
| lm_head | ~25K | Inner loop + FL aggregation. |
| Base encoder | ~94.5M | Frozen. Not trained. Not transmitted. |

`get_outer_loop_params()` = all trainable = LoRA + lm_head.
Communication per round: ~1.3 MB (280x less than full Per-FedAvg encoder).

### L2-ARCTIC Speaker Split (locked in `data/l2arctic_split.json`)

```
meta_train (12): RRBI, HQTV, TNI, NCC (hi), YKWK, ERMS, EBVS, HJK (ko),
                 MBMPS, HKK, BWC, LXC (zh)
meta_val   (4):  SVBI, NJS, TXHC, ZHAA (es) — hyperparameter tuning only
meta_test  (4):  SKA, YBAA (ar), PNV, THV (vi) — evaluated ONCE at end
```

**Scientific validity:**
wav2vec2-base-960h was fine-tuned on LibriSpeech train-960h (native US English).
L2-ARCTIC speakers are non-native English — genuinely out-of-domain.
Arabic and Vietnamese L1 speakers are the hardest generalization challenge.

### Phase 2 Build Order

- [ ] **STEP 0** — Lock speaker split and eval clips:
  `python data/prepare_l2arctic_eval.py`
  `git commit data/l2arctic_split.json data/l2arctic_eval_clips.json`
  Gate: files exist with correct speakers

- [ ] **STEP 1** — LoRA model smoke-test:
  `python models/lora_wav2vec2.py`
  Gate: ~320K trainable params, second-order gradient flows, self-test passes

- [ ] **STEP 2** — Gradient norm diagnostic:
  `python data/diagnostics/run_lora_gradient_analysis.py`
  Gate: norms documented in `data/diagnostics/lora_gradient_norms.json`

- [ ] **STEP 3** — Verify engine integration:
  `python -c "from maml.engine import MAMLEngine, MAMLConfig; cfg = MAMLConfig(mode='lora_maml'); print(cfg)"`
  Gate: succeeds

- [ ] **STEP 4** — L2-ARCTIC data pipeline:
  `python data/download_l2arctic.py --data_dir /path/to/l2arctic`
  Gate: 16 node dirs, no pkl, no speaker_id, all at 16kHz

- [ ] **STEP 5** — Centralized validation (no Docker):
  `python maml/meta_train.py --config configs/lora_poc.yaml`
  Gate: meta-val WER at k=5 < k=0 on ≥ 3/4 val speakers; no NaN in meta-grads

- [ ] **STEP 6** — Federated training (12 nodes, 30 rounds):
  Gate: `checkpoints/lora_federated/theta_star_lora.pt` exists, no NaN

- [ ] **STEP 7** — Hyperparameter search on meta-val:
  Gate: `data/hparam_log.json` populated, best config in `configs/lora_poc.yaml`

- [ ] **STEP 8** — Retrain with best hyperparameters

- [ ] **STEP 9** — Final evaluation (meta-test, run ONCE):
  `python evaluation/eval_lora.py`
  Gate: `evaluation/results/fedlora_maml_l2arctic.json` exists with all 4 baselines

---

## Invariants (Non-Negotiable)

| # | Invariant | Checked By |
|---|-----------|-----------|
| I1 | Raw audio never persists after `features.py` runs. No `.pkl` files under `data/nodes/` or `data/test_nodes/`. | `eval_poc.py` |
| I2 | Speaker identity never stored in node artifacts. No `speaker_id` in node dirs. | `eval_poc.py` |
| I3 | `lm_head` parameters never included in `get_parameters()` or `set_parameters()`. Flower client serializes encoder weights only. | `eval_poc.py` |
| I4 | `fit()` returns meta-gradients, not weight updates. Strategy aggregates gradients and applies β to update θ*. | `eval_poc.py` |
| I5 | Support and query sets have zero index overlap. Asserted in `task_sampler.sample_task()`. | `eval_poc.py` |
| I6 | Encoder `requires_grad` stays `True` throughout training. | `eval_poc.py` |

---

## Success Criteria (Revised)

**A valid positive result requires:**
- `WER_federated_k10 < WER_pretrained_k10` (MAML helps over direct fine-tuning)
- Confidence intervals do not overlap
- Adaptation curve shows faster early convergence for federated θ*

**A valid null result:**
- `WER_federated_k10 ≈ WER_pretrained_k10`
- Conclusion: FOMAML over 20 rounds with 12 speakers does not improve
  adaptation on VCTK relative to direct fine-tuning. Document accurately.

**Required baselines (without these, results are uninterpretable):**
1. `WER_pretrained_k0`: pretrained model, no MAML, no adaptation
2. `WER_pretrained_k10`: pretrained model, no MAML, 10 adaptation steps
3. `WER_federated_k10`: federated θ*, 10 adaptation steps
4. `WER_centralized_k10`: centralized θ* (matched total updates), 10 adaptation steps

**1. SOVEREIGNTY** (unchanged)
`grep -r "speaker_id" data/nodes/` returns nothing.
`find data/nodes data/test_nodes -name "*.pkl"` returns nothing.

---

## Architecture

**Model:** `facebook/wav2vec2-base-960h` (~95M params)
**Algorithm:** FOMAML (first-order MAML, `create_graph=False`)
**FL:** Flower (flwr), PerFedAvg strategy (gradient aggregation, not weight averaging)
**Dataset:** VCTK Corpus, 20 speakers, 12/4/4 meta-train/val/test split

**ANIL split:**
- Inner loop: `lm_head` only (~25K params), stays local, never transmitted
- Outer loop / FL: `wav2vec2` encoder only (~94.5M params)

**Data flow:**
```
meta_split.py → prepare_librispeech.py → features.py → task_sampler
    → MAMLEngine.compute_meta_gradient()
    → client_maml.fit() returns meta-gradients
    → strategy_maml.aggregate_fit() applies θ* ← θ* − β · avg(grads)
```

---

## Key Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Rounds | 20 | Fixed |
| Meta-train nodes | 12 | meta_split.json |
| Inner steps (k) | 3 | Tuned on meta-val |
| Inner LR (α) | 1e-4 | After gradient norm diagnostic |
| Outer LR (β) | 2e-4 | Tuned on meta-val |
| Support size (K) | 8 | Fixed |
| Query size (Q) | 8 | Fixed |
| Eval clips | 20 | For bootstrap CI |
| Bootstrap samples | 1000 | For 95% CI |
| VRAM budget/node | 3.5 GB | RTX 4070 Super |

---

## Build Order

Run in strict sequence. Each step gates the next.

- [ ] **STEP 0** — Lock meta-test speakers:
  `python data/meta_split.py`
  Gate: `data/meta_split.json` exists, 4 meta-test hashes locked.
  **Commit immediately: `git add data/meta_split.json data/salt.txt && git commit`**

- [ ] **STEP 1** — Gradient norm diagnostic:
  `python data/diagnostics/run_gradient_analysis.py`
  Gate: `data/diagnostics/gradient_norms.json` exists.
  Set `inner_lr` in `configs/poc.yaml` based on recommendation.

- [ ] **STEP 2** — Data pipeline (meta-train + meta-val):
  `python data/prepare_librispeech.py && python data/features.py`
  Gate: 16 node dirs, no `.pkl`, features at 16kHz, no `speaker_id`.

- [ ] **STEP 3** — Data insights:
  `python data/insights/run_insights.py`
  Gate: 4 JSON files in `data/insights/`. Review `domain_shift.json`.
  **If VCTK k=0 WER < 5% with pretrained model, document and reassess.**

- [ ] **STEP 4** — Hyperparameter search on meta-val:
  Try `inner_lr` ∈ [1e-5, 1e-4] and `k` ∈ [1, 3, 5, 10].
  Lock hyperparameters in `configs/poc.yaml` before federated training.

- [ ] **STEP 5** — Centralized MAML (meta-train speakers):
  `python maml/meta_train.py --config configs/poc.yaml`
  Gate: WER(k=10) < WER(k=0) on ≥ 3/4 meta-val speakers.

- [ ] **STEP 6** — Federated training (20 rounds, 12 nodes):
  `./scripts/run_poc.sh`
  Gate: `checkpoints/federated/theta_star.pt` exists, no NaN values.

- [ ] **STEP 7** — Centralized matched-update baseline:
  Count updates from federated run, run:
  `python maml/meta_train.py --max_updates N --output_name theta_star_matched`
  Gate: `checkpoints/centralized/theta_star_matched.pt` exists.

- [ ] **STEP 8** — Prepare meta-test data (first time):
  `python data/prepare_librispeech.py --splits meta_test && python data/features.py --nodes_dir data/test_nodes`
  Gate: 4 test node dirs, no `.pkl`, no `speaker_id`.

- [ ] **STEP 9** — Final evaluation (run ONCE):
  `python evaluation/eval_poc.py --config configs/poc.yaml`
  `python evaluation/plot_results.py`
  Gate: `evaluation/results/final_eval.json` exists with bootstrap CIs.
  Document findings regardless of outcome.

---

## Commands

```bash
# Environment
pip install -r requirements.txt
python -c "import flwr, transformers; print('OK')"

# STEP 0: Lock meta-test speakers
python data/meta_split.py
git add data/meta_split.json data/salt.txt && git commit -m "chore: lock meta-test speakers"

# STEP 1: Gradient norm diagnostic
python data/diagnostics/run_gradient_analysis.py --device cuda

# STEP 2: Data pipeline
python data/prepare_librispeech.py
python data/features.py

# STEP 3: Data insights
python data/insights/run_insights.py --device cuda

# STEP 5: Centralized gate
python maml/meta_train.py --config configs/poc.yaml

# STEP 6: Federated
./scripts/build.sh
docker-compose -f docker/docker-compose.yml up

# STEP 7: Matched-update baseline
python maml/meta_train.py --max_updates N --output_name theta_star_matched

# STEP 8: Prepare test data
python data/prepare_librispeech.py --splits meta_test
python data/features.py --nodes_dir data/test_nodes

# STEP 9: Final evaluation
python evaluation/eval_poc.py --config configs/poc.yaml
python evaluation/plot_results.py
```

---

*VoiceFL-MAML PoC · FOMAML · ANIL · Wav2Vec2 · VCTK · 20 speakers (12/4/4)*
*No DP · No perturbation protocol · Bootstrap CI · Valid meta-test protocol*
