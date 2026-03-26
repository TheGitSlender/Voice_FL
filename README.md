# VoiceFL

> **Privacy-Preserving Federated Learning Platform for Personalized Voice AI Assistants**

**Version:** V1 — Simplified Initial Plan  
**Philosophy:** Utility-First Decentralization — voice data remains local; acoustic intelligence is global  
**Dev environment:** Local (RTX 4070 Super, 32GB RAM) → AWS deployment later  
**Starting scale:** 20 nodes → 50 → 100+ as pipeline stabilizes

---

## Table of Contents

1. [Vision](#1-vision)
2. [Architecture Overview](#2-architecture-overview)
3. [Node Design](#3-node-design)
4. [Privacy Stack](#4-privacy-stack)
5. [Behavioral Analysis Engine](#5-behavioral-analysis-engine)
6. [Pipeline — S0 through S7](#6-pipeline--s0-through-s7)
7. [Security](#7-security)
8. [Dataset Strategy](#8-dataset-strategy)
9. [Technology Stack](#9-technology-stack)
10. [Development Phases & Deliverables](#10-development-phases--deliverables)
11. [Evaluation Metrics](#11-evaluation-metrics)
12. [Repository Structure](#12-repository-structure)
13. [References](#13-references)

---

## 1. Vision

Voice AI assistants are permanent biometric interfaces. They hear everything. Every major platform's answer to making them smarter — collect voice data centrally, train on it — is architecturally wrong and increasingly legally untenable.

> *How do you build a voice assistant that gets better the more you use it — without ever knowing who you are?*

VoiceFL answers this with federated learning: the model goes to the data, not the other way around. Each user's voice never leaves their device. The collective intelligence of the fleet improves everyone's assistant through privacy-sanitized gradient updates only.

### The Problem in One Table

| | Centralized (Current Industry) | VoiceFL |
|---|---|---|
| **Raw voice data** | Uploaded to central servers | Never leaves the device |
| **Model quality** | One model, mediocre for everyone | Global backbone + per-user personalization layers |
| **Privacy guarantee** | Policy document | Mathematical (ε-DP + SecAgg) |
| **Node trust** | Implicit | Zero-trust via mutual TLS |
| **Regulatory posture** | GDPR exposure | Compliance by design |

---

## 2. Architecture Overview

Three layers. Data flows upward as sanitized gradients only. Intelligence flows downward as model weights.

```
┌──────────────────────────────────────────────────────────────┐
│  LAYER 3 — MLOps                                             │
│  SageMaker · MLflow · S3 · Model Registry · CodePipeline     │
└────────────────────────────┬─────────────────────────────────┘
                             │ model artifacts / metrics
┌────────────────────────────▼─────────────────────────────────┐
│  LAYER 2 — Central Aggregator (EKS)                          │
│  Flower Server (FedAvg/FedProx) · SecAgg+ · BAE              │
│  All comms over mTLS — mutual X.509 per node                 │
└──────────────┬──────────────────────┬────────────────────────┘
               │ ΔW only (DP-sanitized)│
┌──────────────▼──────────────────────▼────────────────────────┐
│  LAYER 1 — Sovereign Edge Nodes (Local Silo Boundary)        │
│                                                              │
│  [Node 1]   [Node 2]   [Node 3]  ...  [Node 20]             │
│  raw audio  raw audio  raw audio       raw audio             │
│  PII mask   PII mask   PII mask        PII mask              │
│  features   features   features        features              │
│  local train + DP clip+noise = ΔW                           │
│                                                              │
│  Raw audio NEVER crosses this boundary                       │
└──────────────────────────────────────────────────────────────┘
```

**In development:** all layers run locally. Layer 1 = Flower VCE simulated clients. Layer 2 = Flower server on localhost. Layer 3 = local MLflow.

**In production:** Layer 1 = ECS Fargate containers. Layer 2 = EKS. Layer 3 = managed SageMaker.

---

## 3. Node Design

### Simulation approach

One node = one speaker from LibriSpeech. Each speaker's ~100–150 clips provides enough data for meaningful local gradients. Non-IID heterogeneity is genuine — speakers differ in accent, vocabulary, pacing, recording quality.

In production framing, one node = one user device. The speaker-as-node simulation is an explicit proxy for single-user deployment. Cold-start (new user, few clips) is addressed by the global backbone warm-start.

### Starting at 20 nodes

- 20 speakers selected from LibriSpeech train-clean-100 (251 available)
- Selection criterion: maximize WER spread — WER as proxy for accent and clarity diversity
- Each node: all clips from one speaker, ~80–150 clips
- Cohort per round: 10% = 2 nodes at 20-node scale
- Scale path: 20 → 50 → 251 → 921+

### Node internals

| Component | Tool | Role |
|---|---|---|
| PII Masking | librosa + custom | Strip speaker metadata; waveform → features only |
| Feature Extractor | librosa / torchaudio | Log-mel (80 coeff, 25ms/10ms) locally |
| Local Trainer | PyTorch + HuggingFace | N local epochs on feature tensors |
| DP Engine | Opacus | Clip + Gaussian noise; Rényi budget tracking |
| Flower Client | `flwr.client.NumPyClient` | `get_parameters`, `fit`, `evaluate` — backbone only |
| Comm (prod) | gRPC over mTLS | Bidirectional X.509; ACM certificates |
| Personalization Store | Local disk (dev) / EBS (prod) | Top layers; never transmitted |

### FedPer split

**Global backbone** — lower + middle transformer layers. Aggregated every round. Travels as ΔW after DP.

**Personalization layers** — top task-specific layers. Never transmitted. Saved locally per node.

---

## 4. Privacy Stack

### Formal ε-Differential Privacy

$$P[\mathcal{A}(D) \in S] \leq e^{\varepsilon} \cdot P[\mathcal{A}(D') \in S]$$

$e^\varepsilon$ bounds how much any adversary can learn about any individual user's voice from global model updates.

**Step A — L2 Gradient Clipping**

$$\Delta\bar{W} = \Delta W \cdot \min\!\left(1,\ \frac{C}{\|\Delta W\|_2}\right)$$

**Step B — Gaussian Noise Injection**

$$\Delta W_{\text{private}} = \Delta\bar{W} + \mathcal{N}(0,\ \sigma^2 C^2 \mathbf{I})$$

**Step C — Privacy Accounting**

Rényi DP accountant tracks cumulative ε spend across rounds. Training halts when budget exhausted.

**Step D — Secure Aggregation**

Flower SecAgg+: server sees only $\sum_k \Delta W_k^{\text{private}}$, never individual updates.

### Budget reference

| Use | ε | WER impact |
|---|---|---|
| Dev / prototype | ≥ 10 | Negligible |
| Development baseline | 8 | < 5% |
| Production target | 4 | 5–15% |
| High-security | 2 | 15–30% |

Start: ε = 8, δ = 1e-5.

---

## 5. Behavioral Analysis Engine

Runs at S5 between update receipt and FedAvg. Protects model integrity. Orthogonal to DP.

### Threat model

| Threat | Voice-specific risk |
|---|---|
| Byzantine | Corrupt global acoustic representations |
| Gradient poisoning | Force misrecognition of phrases/accents |
| Free-riders | Zero-gradient nodes degrade convergence |
| Sybil | Multiple fake identities amplify aggregation weight |

### Detection stack

1. **Cosine screening** — each ΔW vs round median gradient. Flags directional outliers.
2. **Norm bounding** — L2 norm > 3σ historical = quarantine.
3. **Temporal fingerprint** — per-node statistical profile across rounds. Catches slow-burn attacks.
4. **Isolation Forest** — joint anomaly score over (cosine, norm, temporal). No labeled data needed.
5. **Robust aggregation fallback** — FLTrust or Krum replaces FedAvg on cluster-level anomaly.

### Response

| Level | Action |
|---|---|
| Soft | Update weight reduced |
| Medium | Node quarantined N rounds |
| Hard | Node excluded; cert revoked |
| Critical | Round invalidated; model rolled back |

---

## 6. Pipeline — S0 through S7

```
S0 Scoping → S1 Discovery → S2 Ingestion+PII → S3 Node Setup
                                                      ↓
S7 Monitor  ← S6 Validate  ← S5 BAE Filter  ← S4 FL Training
```

### S0 — Scoping & KPI Definition
Define ε targets (dev=8, prod=4), WER targets (≤12% global, ≤8% personalized), BAE thresholds (detection >95%, FP <3%), 20-node plan with scale path.

### S1 — Data Source Discovery
LibriSpeech train-clean-100 via `openslr/librispeech_asr` HuggingFace. Verify `speaker_id` field. Select 20 speakers by WER spread.

### S2 — Local Ingestion & Voice Data Governance
Strip speaker metadata → anonymous hashes. Extract log-mel (80 coeff, 25ms/10ms). Discard raw audio. Save `(T, 80)` tensors to `data/nodes/node_{id}/`. No PII in any stored artifact.

### S3 — Federated Node Setup
Dev: Flower VCE, all in-process. Prod: Docker on ECS Fargate, mTLS via ACM. Health check: 1 round with 2 toy nodes.

### S4 — Privacy-Preserving Training
Node: receive $W^{(t)}_g$ → local train → clip → noise → transmit $\Delta W_{\text{private}}$.  
Server: collect → BAE → SecAgg unmask → FedAvg → broadcast $W^{(t+1)}_g$.

### S5 — Behavioral Analysis
BAE runs synchronously pre-FedAvg. Anomaly scores logged to MLflow. See Section 5.

### S6 — Global Validation
WER on test.clean + test.other. Per-node personalization gain. Quality gate: promote only on improvement.

### S7 — Deployment & Monitoring
Dev: local inference script. Prod: SageMaker endpoint + Model Monitor + CloudWatch. Retraining on WER drift >5%.

---

## 7. Security

| Concern | Dev | Prod |
|---|---|---|
| Node auth | None (local) | mTLS via ACM X.509 |
| Communication | gRPC plain | gRPC over mTLS |
| Storage | Local disk | S3 + EBS (KMS encrypted) |
| BAE | Full | Full |
| DP + SecAgg | Full | Full |
| PII | No raw audio in any artifact | Same + IAM least-privilege |

mTLS is designed and documented in dev but not implemented until Phase 7 (AWS).

---

## 8. Dataset Strategy

### LibriSpeech (openslr/librispeech_asr)

| Subset | Rows | Speakers | Role |
|---|---|---|---|
| train.clean.100 | 28,539 | 251 | Simulation nodes + seed training |
| test.clean | 2,620 | 40 | Primary eval |
| test.other | 2,939 | 33 | Robustness eval |
| validation.clean | 2,703 | — | Per-round dev validation |

### Scale path

| Phase | Nodes | Dataset scope |
|---|---|---|
| Dev (Phase 1–5) | 20 | train-clean-100, 20 speakers |
| Phase 6 | 50 | train-clean-100, 50 speakers |
| Later | 251 | train-clean-100, all speakers |
| Production sim | 921 | + train-clean-360 |

### Future (not in scope yet)
Common Voice English (FL domain adaptation), VCTK (fairness audit), TED-LIUM 3 (domain shift).

---

## 9. Technology Stack

### Local dev

| Component | Tool |
|---|---|
| FL framework | Flower (flwr) with VCE |
| ML | PyTorch + HuggingFace Transformers |
| ASR base | Whisper-small or Wav2Vec2-base-960h |
| DP | Opacus |
| Audio | librosa / torchaudio |
| Tracking | MLflow (local) |
| Data | HuggingFace `datasets` |
| Anomaly detection | scikit-learn IsolationForest |

### AWS (Phase 7+)

| Component | Tool |
|---|---|
| FL server | Flower on Amazon EKS |
| Nodes | Amazon ECS Fargate |
| MLOps | Amazon SageMaker |
| Artifacts | Amazon S3 (SSE-KMS) |
| Monitoring | CloudWatch + Model Monitor |
| Node identity | AWS Certificate Manager |
| CI/CD | CodePipeline + CodeBuild |
| IaC | Terraform |

---

## 10. Development Phases & Deliverables

---

### Phase 0 — Environment `→ S0`

- [ ] Python 3.10+ with PyTorch, flwr, opacus, transformers, librosa, datasets
- [ ] CUDA verified (RTX 4070 Super)
- [ ] MLflow running locally; first experiment logged
- [ ] HuggingFace token configured; librispeech_asr streams successfully
- [ ] Repo structure initialized (Section 12)
- [ ] `requirements.txt` pinned
- [ ] KPI doc written

---

### Phase 1 — Data Pipeline `→ S1, S2`

- [ ] `data/download.py` — stream train-clean-100, group by speaker_id, save metadata
- [ ] 20 speakers selected; WER spread documented
- [ ] `data/pii_masking.py` — strip metadata; assign anonymous node hashes
- [ ] `data/features.py` — log-mel extraction; raw audio discarded after
- [ ] `data/partition.py` — 20 per-node tensor dirs in `data/nodes/`
- [ ] Per-node stats logged (clip count, duration, shape)
- [ ] No PII in any `data/nodes/` artifact

---

### Phase 2 — Seed Model `→ S1`

- [ ] `models/seed_training.py` — centralized fine-tune on full train-clean-100 features
- [ ] `models/backbone.py` + `models/personalization.py` — FedPer split defined
- [ ] Seed WER ≤ 10% on test.clean
- [ ] Per-speaker WER breakdown saved
- [ ] Seed model saved to `models/checkpoints/seed/`

---

### Phase 3 — Basic FL `→ S3, S4`

- [ ] `federated/client.py` — NumPyClient with backbone-only param exchange
- [ ] `federated/server.py` — Flower server + FedAvg
- [ ] `federated/simulation.py` — 20-node VCE, 10% cohort, N rounds
- [ ] Personalization layers never appear in `get_parameters` / `set_parameters`
- [ ] 20 rounds complete; per-round metrics in MLflow
- [ ] Federated gap vs seed < 30%
- [ ] Checkpoints saved every round

---

### Phase 4 — Differential Privacy `→ S4`

- [ ] `privacy/dp_engine.py` — Opacus wraps trainer; configurable C, σ, ε, δ
- [ ] Clipping + noise formulas verified
- [ ] Rényi accountant tracks budget; training halts on exhaustion
- [ ] `privacy/secagg.py` — SecAgg+ active
- [ ] ε ∈ {10, 8, 4, 2} ablation; WER per ε saved
- [ ] Privacy-utility curve saved

---

### Phase 5 — BAE `→ S5`

- [ ] `security/bae.py` — all 4 detection layers + response logic
- [ ] `security/isolation_forest.py` — joint IF scoring
- [ ] `security/attacks/` — 4 attack simulators
- [ ] Byzantine test: >90% detection
- [ ] Poisoning: detected within 3 rounds
- [ ] Free-rider: quarantined after 2 offenses
- [ ] FLTrust/Krum fallback fires correctly

---

### Phase 6 — Evaluation `→ S6`

- [ ] `evaluation/eval_asr.py` — WER/CER on test.clean + test.other
- [ ] `evaluation/benchmark.py` — 4-way ablation
- [ ] Per-node personalization gain quantified
- [ ] 50-round, 20-node run stable
- [ ] All targets met: global WER ≤12%, personalized ≤8%

---

### Phase 7 — AWS Deployment `→ S7`

- [ ] Flower server on EKS; nodes on ECS Fargate
- [ ] mTLS: ACM certs; connection refused without valid cert
- [ ] Terraform reproduces full environment from scratch
- [ ] SageMaker Pipeline: S2 → S4 → S5 → S6 → registry
- [ ] Model Monitor active; drift baseline established
- [ ] CloudWatch retraining alarm configured

---

## 11. Evaluation Metrics

| Metric | Minimum | Target |
|---|---|---|
| WER — Global (test.clean) | ≤ 15% | ≤ 12% |
| WER — Personalized | — | ≤ 8% |
| Federated gap vs seed | < 30% | < 15% |
| Privacy budget ε (prod) | ≤ 10 | ≤ 4 |
| Byzantine detection rate | > 80% | > 95% |
| BAE false positive rate | < 10% | < 3% |
| Rounds to convergence | ≤ 50 | ≤ 30 |
| MB per round per node | < 50 MB | < 20 MB |

---

## 12. Repository Structure

```
voicefl/
├── CLAUDE.md                     # Claude Code memory file
├── README.md                     # This file
│
├── data/
│   ├── download.py
│   ├── pii_masking.py
│   ├── features.py
│   ├── partition.py
│   └── nodes/                    # node_001/ ... node_020/
│
├── models/
│   ├── backbone.py
│   ├── personalization.py
│   ├── seed_training.py
│   └── checkpoints/
│
├── federated/
│   ├── client.py
│   ├── server.py
│   ├── strategy.py
│   └── simulation.py
│
├── privacy/
│   ├── dp_engine.py
│   └── secagg.py
│
├── security/
│   ├── bae.py
│   ├── isolation_forest.py
│   ├── attacks/
│   └── audit.py
│
├── evaluation/
│   ├── eval_asr.py
│   ├── benchmark.py
│   └── results/
│
├── mlops/                        # Phase 7 — AWS only
├── infra/                        # Phase 7 — Terraform
├── notebooks/
├── tests/
├── scripts/
├── Dockerfile
└── requirements.txt
```

---

## 13. References

- McMahan et al. (2017) — [FedAvg](https://arxiv.org/abs/1602.05629)
- Arivazhagan et al. (2019) — [FedPer](https://arxiv.org/abs/1912.00818)
- Pelikan et al. (2023) — [pfl4asr](https://arxiv.org/abs/2310.00098) — closest reference, open-sourced by Apple
- Li et al. (2020) — [FedProx](https://arxiv.org/abs/1812.06127)
- Blanchard et al. (2017) — [Krum](https://proceedings.neurips.cc/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html)
- Liu et al. (2020) — [FLTrust](https://arxiv.org/abs/2012.13995)
- Abadi et al. (2016) — [DP-SGD](https://arxiv.org/abs/1607.00133)
- [Apple ml-pfl4asr](https://github.com/apple/ml-pfl4asr)

---

*VoiceFL · V1 · 20 nodes · Local → AWS · S0–S7*
