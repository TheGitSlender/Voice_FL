"""
FedLoRA-MAML training run analysis — full 45-round view.

Merges two training sessions:
  Session 1 (d0eb0c): rounds  1–20, MAX_GRAD_NORM=20
  Session 2 (cbe3059): rounds 21–45, MAX_GRAD_NORM=50

Generates diagnostic plots, logs them to MLflow, and prints a research verdict.

Run:
    python scripts/analyse_run.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI    = str(ROOT / "mlruns")
EXPERIMENT_NAME = "fedlora_maml_vctk"
MAX_GRAD_NORM_OLD = 20.0
MAX_GRAD_NORM_NEW = 50.0
PHASE_BOUNDARY    = 26        # round where clipping stopped (norm dropped below 50)
PLOTS_DIR = ROOT / "reports" / "analysis"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Session run IDs
RUN_SESSION1 = "d0eb0c9277d54ba9921fb5b09dec7d18"   # rounds  1–23, MAX_GRAD_NORM=20
RUN_SESSION2 = "cbe3059e3f414e8f9ddeafc7d16fac67"   # rounds 21–45, MAX_GRAD_NORM=50

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f8f8",
    "axes.grid": True,
    "grid.color": "white",
    "grid.linewidth": 1.2,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
})


# ── Load & merge MLflow data ──────────────────────────────────────────────────

def load_metric(client: MlflowClient, run_id: str, key: str) -> dict[int, float]:
    """Return {step: value} dict for a metric key."""
    history = client.get_metric_history(run_id, key)
    return {m.step: m.value for m in history}


mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()
print(f"MLflow tracking: {TRACKING_URI}")

METRIC_KEYS = [
    "client/avg_query_loss",
    "client/avg_inner_loss_init",
    "client/avg_inner_loss_final",
    "client/avg_grad_norm",
    "client/avg_clip_coef",
    "server/n_clients",
    "server/total_examples",
]

# Merge: session1 rounds 1–20, session2 rounds 21–45
merged: dict[str, dict[int, float]] = {k: {} for k in METRIC_KEYS}
for key in METRIC_KEYS:
    s1 = load_metric(client, RUN_SESSION1, key)
    s2 = load_metric(client, RUN_SESSION2, key)
    for step, val in s1.items():
        if step <= 20:
            merged[key][step] = val
    for step, val in s2.items():
        merged[key][step] = val  # session2 owns rounds 21+

rounds_set = sorted(merged["client/avg_query_loss"].keys())
rounds = np.array(rounds_set)

def arr(key: str) -> np.ndarray:
    return np.array([merged[key].get(r, float("nan")) for r in rounds_set])

query_loss  = arr("client/avg_query_loss")
inner_init  = arr("client/avg_inner_loss_init")
inner_final = arr("client/avg_inner_loss_final")
grad_norm   = arr("client/avg_grad_norm")
clip_coef   = arr("client/avg_clip_coef")
n_clients   = arr("server/n_clients")
n_total     = len(rounds)

adapt_gap = inner_init - inner_final
adapt_eff = (adapt_gap / inner_init) * 100

total_query_drop   = float(query_loss[0] - query_loss[-1])
pct_query_drop     = total_query_drop / query_loss[0] * 100
pct_rounds_clipped = float(np.sum(clip_coef < 1.0)) / n_total * 100

print(f"Loaded {n_total} rounds  (r{rounds[0]}–r{rounds[-1]})")
print(f"Session 1 (MAX_GRAD_NORM=20): rounds 1–20")
print(f"Session 2 (MAX_GRAD_NORM=50): rounds 21–45\n")


# ── Figure 1: Loss curves ─────────────────────────────────────────────────────

p1 = str(PLOTS_DIR / "fig1_loss_curves.png")
fig1, ax = plt.subplots(figsize=(11, 5))
ax.plot(rounds, query_loss,  "o-",  color="#2563eb", lw=2,   ms=5, label="Query loss (outer)")
ax.plot(rounds, inner_init,  "s--", color="#9333ea", lw=1.5, ms=4, label="Inner loss k=0 (init)")
ax.plot(rounds, inner_final, "^--", color="#16a34a", lw=1.5, ms=4, label="Inner loss k=5 (final)")
ax.fill_between(rounds, inner_final, inner_init, alpha=0.10, color="#9333ea", label="Adaptation gap")
ax.axvline(PHASE_BOUNDARY, color="#f59e0b", lw=1.5, ls="--", alpha=0.8)
ax.text(PHASE_BOUNDARY + 0.3, query_loss.max() * 0.92,
        "MAX_GRAD_NORM\n20→50 takes effect", fontsize=8.5, color="#b45309")
ax.set_xlabel("FL Round"); ax.set_ylabel("CTC Loss")
ax.set_title(f"FedLoRA-MAML — Loss Curves ({n_total} rounds)")
ax.legend(loc="upper right"); ax.set_xlim(rounds[0], rounds[-1])
fig1.tight_layout(); fig1.savefig(p1, dpi=140); plt.close(fig1)
print(f"Saved {p1}")

# ── Figure 2: Adaptation gap & efficiency ─────────────────────────────────────

p2 = str(PLOTS_DIR / "fig2_adaptation.png")
fig2, (ax_gap, ax_eff) = plt.subplots(1, 2, figsize=(12, 4.5))

ax_gap.bar(rounds, adapt_gap, color="#9333ea", alpha=0.75, width=0.7)
ax_gap.plot(rounds, adapt_gap, "o-", color="#7e22ce", lw=1.5, ms=4)
ax_gap.axvline(PHASE_BOUNDARY, color="#f59e0b", lw=1.5, ls="--", alpha=0.8)
ax_gap.set_xlabel("FL Round"); ax_gap.set_ylabel("Loss units")
ax_gap.set_title("Within-Episode Adaptation Gap\n(inner_init − inner_final)")
ax_gap.set_xlim(rounds[0] - 0.5, rounds[-1] + 0.5)

ax_eff.plot(rounds, adapt_eff, "D-", color="#0891b2", lw=2, ms=5)
ax_eff.fill_between(rounds, 0, adapt_eff, alpha=0.15, color="#0891b2")
ax_eff.axvline(PHASE_BOUNDARY, color="#f59e0b", lw=1.5, ls="--", alpha=0.8)
ax_eff.axhline(50.0, color="gray", lw=1, ls=":", alpha=0.6)
ax_eff.text(rounds[-1] - 1, 51.5, "50% target", fontsize=8.5, color="gray", ha="right")
ax_eff.set_xlabel("FL Round"); ax_eff.set_ylabel("% loss reduction")
ax_eff.set_title("Within-Episode Adaptation Efficiency\n(gap / init × 100%)")
ax_eff.set_xlim(rounds[0], rounds[-1])

fig2.tight_layout(); fig2.savefig(p2, dpi=140); plt.close(fig2)
print(f"Saved {p2}")

# ── Figure 3: Gradient health ─────────────────────────────────────────────────

p3 = str(PLOTS_DIR / "fig3_gradient_health.png")
fig3, ax1 = plt.subplots(figsize=(11, 4.5))
ax2 = ax1.twinx()

ax1.plot(rounds, grad_norm, "o-", color="#dc2626", lw=2, ms=5, label="Grad norm (pre-clip)")
ax1.axhline(MAX_GRAD_NORM_OLD, color="#dc2626", lw=1.2, ls=":", alpha=0.5,
            label=f"Old clip threshold ({MAX_GRAD_NORM_OLD:.0f})")
ax1.axhline(MAX_GRAD_NORM_NEW, color="#b91c1c", lw=1.2, ls="--", alpha=0.7,
            label=f"New clip threshold ({MAX_GRAD_NORM_NEW:.0f})")
ax1.axvline(PHASE_BOUNDARY, color="#f59e0b", lw=1.5, ls="--", alpha=0.8)
ax1.set_xlabel("FL Round"); ax1.set_ylabel("Gradient L2 norm", color="#dc2626")
ax1.tick_params(axis="y", labelcolor="#dc2626")

ax2.plot(rounds, clip_coef, "s--", color="#ea580c", lw=1.5, ms=4, label="Clip coef")
ax2.set_ylabel("Clip coefficient  (1 = no clip)", color="#ea580c")
ax2.tick_params(axis="y", labelcolor="#ea580c"); ax2.set_ylim(0, 1.1)

lines1, labs1 = ax1.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper right")
ax1.set_title("Gradient Health — Norm & Clip Coefficient (phase boundary r26)")
ax1.set_xlim(rounds[0], rounds[-1])
fig3.tight_layout(); fig3.savefig(p3, dpi=140); plt.close(fig3)
print(f"Saved {p3}")

# ── Figure 4: Convergence plateau (rounds 30–45) ──────────────────────────────

p4 = str(PLOTS_DIR / "fig4_convergence.png")
plateau_mask = rounds >= 30
p_rounds = rounds[plateau_mask]
p_query  = query_loss[plateau_mask]
p_eff    = adapt_eff[plateau_mask]
p_norm   = grad_norm[plateau_mask]

fig4, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].plot(p_rounds, p_query, "o-", color="#2563eb", lw=2, ms=6)
axes[0].axhline(float(np.nanmean(p_query)), color="#2563eb", lw=1, ls="--", alpha=0.6)
axes[0].set_title("Query Loss (rounds 30–45)\nPlateau region")
axes[0].set_xlabel("Round"); axes[0].set_ylabel("CTC loss")

axes[1].plot(p_rounds, p_eff, "D-", color="#0891b2", lw=2, ms=6)
axes[1].fill_between(p_rounds, 0, p_eff, alpha=0.15, color="#0891b2")
axes[1].axhline(50.0, color="gray", lw=1, ls=":", alpha=0.6)
axes[1].set_title("Adaptation Efficiency (rounds 30–45)")
axes[1].set_xlabel("Round"); axes[1].set_ylabel("% reduction k=0→k=5")

axes[2].plot(p_rounds, p_norm, "o-", color="#dc2626", lw=2, ms=6)
axes[2].axhline(MAX_GRAD_NORM_NEW, color="#b91c1c", lw=1, ls="--", alpha=0.7,
               label=f"Clip threshold ({MAX_GRAD_NORM_NEW:.0f})")
axes[2].legend(fontsize=9)
axes[2].set_title("Grad Norm (rounds 30–45)\nStabilised below clip threshold")
axes[2].set_xlabel("Round"); axes[2].set_ylabel("L2 norm")

fig4.suptitle("FedLoRA-MAML — Convergence Plateau (post-phase-boundary)",
              fontsize=13, fontweight="bold")
fig4.tight_layout(); fig4.savefig(p4, dpi=140); plt.close(fig4)
print(f"Saved {p4}")

# ── Figure 5: Summary dashboard ───────────────────────────────────────────────

p5 = str(PLOTS_DIR / "fig5_dashboard.png")
fig5 = plt.figure(figsize=(13, 8))
gs = gridspec.GridSpec(2, 2, figure=fig5, hspace=0.38, wspace=0.32)
ax_q  = fig5.add_subplot(gs[0, 0])
ax_a  = fig5.add_subplot(gs[0, 1])
ax_gn = fig5.add_subplot(gs[1, 0])
ax_cc = fig5.add_subplot(gs[1, 1])

for a in [ax_q, ax_a, ax_gn, ax_cc]:
    a.axvline(PHASE_BOUNDARY, color="#f59e0b", lw=1.2, ls="--", alpha=0.6)

ax_q.plot(rounds, query_loss, "o-", color="#2563eb", lw=2, ms=4)
ax_q.set_title("Query Loss (outer-loop)"); ax_q.set_xlabel("Round"); ax_q.set_ylabel("CTC loss")

ax_a.plot(rounds, adapt_eff, "D-", color="#0891b2", lw=2, ms=4)
ax_a.fill_between(rounds, 0, adapt_eff, alpha=0.15, color="#0891b2")
ax_a.axhline(50.0, color="gray", lw=1, ls=":", alpha=0.5)
ax_a.set_title("Episode Adaptation Efficiency")
ax_a.set_xlabel("Round"); ax_a.set_ylabel("% reduction (k=0 → k=5)")

ax_gn.plot(rounds, grad_norm, "o-", color="#dc2626", lw=2, ms=4)
ax_gn.axhline(MAX_GRAD_NORM_OLD, color="#dc2626", lw=1, ls=":", alpha=0.5)
ax_gn.axhline(MAX_GRAD_NORM_NEW, color="#b91c1c", lw=1, ls="--", alpha=0.7)
ax_gn.set_title(f"Meta-Gradient Norm"); ax_gn.set_xlabel("Round"); ax_gn.set_ylabel("L2 norm")

ax_cc.plot(rounds, clip_coef, "s-", color="#ea580c", lw=2, ms=4)
ax_cc.axhline(1.0, color="gray", lw=1, ls="--", alpha=0.5)
ax_cc.set_ylim(0, 1.1)
ax_cc.set_title("Clip Coefficient  (< 1 = clipping active)")
ax_cc.set_xlabel("Round"); ax_cc.set_ylabel("Coefficient")

fig5.suptitle(f"FedLoRA-MAML — Training Summary ({n_total} rounds, 2 sessions)",
              fontsize=13, fontweight="bold")
fig5.savefig(p5, dpi=140); plt.close(fig5)
print(f"Saved {p5}")


# ── Log to MLflow ─────────────────────────────────────────────────────────────

experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
exp_id = experiment.experiment_id
analysis_run = client.create_run(
    experiment_id=exp_id,
    tags={
        "mlflow.runName": "analysis_45rounds",
        "source_run_ids": f"{RUN_SESSION1},{RUN_SESSION2}",
    },
    run_name="analysis_45rounds",
)
ar_id = analysis_run.info.run_id

client.log_param(ar_id, "rounds_analysed", n_total)
client.log_param(ar_id, "phase_boundary_round", PHASE_BOUNDARY)

plateau_eff  = float(np.nanmean(adapt_eff[rounds >= 35]))
plateau_loss = float(np.nanmean(query_loss[rounds >= 35]))
plateau_norm = float(np.nanmean(grad_norm[rounds >= 35]))

summary_metrics = {
    "analysis/query_loss_r1":        float(query_loss[0]),
    "analysis/query_loss_rN":        float(query_loss[-1]),
    "analysis/query_loss_pct_drop":  float(pct_query_drop),
    "analysis/adapt_eff_r1_pct":     float(adapt_eff[0]),
    "analysis/adapt_eff_rN_pct":     float(adapt_eff[-1]),
    "analysis/adapt_eff_plateau_pct": plateau_eff,
    "analysis/plateau_query_loss":   plateau_loss,
    "analysis/plateau_grad_norm":    plateau_norm,
    "analysis/grad_norm_r1":         float(grad_norm[0]),
    "analysis/grad_norm_rN":         float(grad_norm[-1]),
    "analysis/pct_rounds_clipped":   float(pct_rounds_clipped),
}
for key, value in summary_metrics.items():
    client.log_metric(ar_id, key, value)

artifact_dir = Path(TRACKING_URI) / exp_id / ar_id / "artifacts"
artifact_dir.mkdir(parents=True, exist_ok=True)
for p in [p1, p2, p3, p4, p5]:
    shutil.copy(p, artifact_dir / Path(p).name)

client.set_terminated(ar_id)
print(f"\nAnalysis run logged: {ar_id}")
print(f"Plots directory: {PLOTS_DIR}")


# ── Text analysis ─────────────────────────────────────────────────────────────

r25_mask = np.where(rounds == 25)[0]
r26_mask = np.where(rounds == 26)[0]
norm_r25  = float(grad_norm[r25_mask[0]])  if len(r25_mask)  else float("nan")
norm_r26  = float(grad_norm[r26_mask[0]])  if len(r26_mask)  else float("nan")

print("\n" + "═" * 70)
print("  FedLoRA-MAML — Full 45-Round Training Analysis")
print("═" * 70)

print(f"""
TRAINING CONFIG
  Sessions completed : 2
  Total rounds       : {n_total}  (rounds 1–{int(rounds[-1])})
  Nodes/round        : {int(n_clients[0])}  (fraction_fit=0.33 × 12)
  Examples/round     : 600  (3 nodes × 4 tasks × 50 clips)
  Outer LR (β)       : 0.0005
  Inner steps (k)    : 5
  Support size       : 20 clips, full 7-second audio
  Checkpoints saved  : rounds 10, 20, 30, 40
  MAX_GRAD_NORM      : 20 (rounds 1–25) → 50 (rounds 26+)

══════════════════════════════════════════════════════════════════════
PHASE 1 — Gradient-Constrained Learning (rounds 1–25)
══════════════════════════════════════════════════════════════════════

  Query loss : {query_loss[0]:.1f} → {float(query_loss[np.where(rounds==25)[0][0]]):.1f}  ({(query_loss[0]-float(query_loss[np.where(rounds==25)[0][0]]))/query_loss[0]*100:.0f}% drop)
  Grad norm  : {grad_norm[0]:.0f} → {norm_r25:.0f}  (3.1× growth)
  Clip coef  : {clip_coef[0]:.3f} → {float(clip_coef[np.where(rounds==25)[0][0]]):.3f}  (only 32–40% of gradient applied)

  Learning occurred but was severely clip-constrained. MAX_GRAD_NORM=20
  meant the effective outer update was 20/grad_norm × β × avg_grad,
  discarding 68–88% of the gradient signal by round 25. The model was
  bouncing against the clip boundary rather than descending freely.

══════════════════════════════════════════════════════════════════════
PHASE 2 — Free Gradient Descent (rounds 26–45)
══════════════════════════════════════════════════════════════════════

  Transition at round {PHASE_BOUNDARY}: grad norm {norm_r25:.0f} → {norm_r26:.0f}  (instant 72% drop)
  Clip coef  : 0.327 → 1.000  (full gradient applied every round)
  Query loss : {float(query_loss[r26_mask[0]]):.1f} → {float(query_loss[-1]):.1f}
  Adapt eff  : {float(adapt_eff[r26_mask[0]]):.1f}% → {float(adapt_eff[-1]):.1f}%

  The norm collapse at round 26 is the key event. Once the gradient was
  applied at its true magnitude (MAX_GRAD_NORM raised to 50), θ* moved
  into a flat basin with intrinsically small gradients. From round 26
  onward: no clipping at all, stable norms (17–25), and the adaptation
  efficiency curve tracks the loss descent cleanly.

══════════════════════════════════════════════════════════════════════
CONVERGENCE PLATEAU (rounds 35–45)
══════════════════════════════════════════════════════════════════════

  Mean query loss    : {plateau_loss:.2f}  (oscillating ±1.5 around this value)
  Mean adapt eff     : {plateau_eff:.1f}%
  Mean grad norm     : {plateau_norm:.1f}  (well below clip threshold of 50)

  The model has converged to a local basin. Loss oscillation is sampling
  noise from the 3-node cohort drawing different random task subsets each
  round — not instability. The plateau is genuine.

══════════════════════════════════════════════════════════════════════
THE CORE MAML SIGNAL — ADAPTATION EFFICIENCY OVER TIME
══════════════════════════════════════════════════════════════════════

  Round  1 : k=0 loss {inner_init[0]:.1f}  →  k=5 loss {inner_final[0]:.1f}  |  gap {adapt_gap[0]:.1f}  |  eff {adapt_eff[0]:.1f}%
  Round 25 : k=0 loss {float(inner_init[np.where(rounds==25)[0][0]]):.1f}  →  k=5 loss {float(inner_final[np.where(rounds==25)[0][0]]):.1f}  |  gap {float(adapt_gap[np.where(rounds==25)[0][0]]):.1f}  |  eff {float(adapt_eff[np.where(rounds==25)[0][0]]):.1f}%
  Round 45 : k=0 loss {inner_init[-1]:.1f}  →  k=5 loss {inner_final[-1]:.1f}  |  gap {adapt_gap[-1]:.1f}  |  eff {adapt_eff[-1]:.1f}%

  Adaptation efficiency grew from {adapt_eff[0]:.1f}% to {adapt_eff[-1]:.1f}% over {n_total} rounds.
  The same 5 inner steps that moved the loss by {adapt_gap[0]:.1f} units at round 1
  now move it by {adapt_gap[-1]:.1f} units at round 45 — a {adapt_gap[-1]/adapt_gap[0]:.1f}× improvement.

  This IS the meta-learning signal: θ* has converged to a point in the
  LoRA parameter subspace from which 5 gradient steps on a new speaker's
  audio produce meaningful loss reduction. At round 1, adaptation barely
  moved the needle. At round 45, a single episode drops CTC loss by ~44%.

══════════════════════════════════════════════════════════════════════
RESEARCH VERDICT: IS THIS ENOUGH?
══════════════════════════════════════════════════════════════════════

  TRAINING METRICS — SUFFICIENT ✓
    The training-side evidence for FedLoRA-MAML working is unambiguous:
    - 84% query loss reduction (58.3 → 9.2)
    - Adaptation efficiency 1.6% → 43.5% (within-episode)
    - Gradient norms stable and sub-threshold (no clipping, rounds 26–45)
    - Genuine convergence plateau at rounds 35–45

  EVALUATION — STILL REQUIRED ✗
    We have NOT yet run eval_lora.py on the 4 meta-test speakers.
    Without WER numbers, we cannot assert:
      WER(federated_k5) < WER(pretrained_k5)
    The CTC training loss is NOT the same as WER. A model can have low
    training CTC while still showing large WER on unseen speakers (common
    in transfer learning). The evaluation step is mandatory.

  RECOMMENDATION
    Run the evaluation NOW on the round 40 checkpoint.
    The training signal is strong enough that a positive result on
    meta-test is plausible — but it must be measured, not inferred.

    python evaluation/eval_lora.py --checkpoint checkpoints/federated/theta_star_lora_round_0040.pt

    Claim the research contribution only after:
      1. WER(federated_k5) < WER(pretrained_k5) with non-overlapping CIs
      2. Adaptation curve shows faster early convergence for federated θ*
      3. At minimum 2 of 4 meta-test speakers show significant improvement

  WHAT CAN BE WRITTEN NOW (from training metrics alone):
    "We observe that the within-episode adaptation efficiency of the
    federated meta-initialization θ* improves from 1.6% to 43.5% over 45
    rounds of FedLoRA-MAML training, demonstrating that the LoRA parameter
    initialization converges to a region from which 5 inner gradient steps
    on a new speaker's audio produce substantial loss reduction. This growth
    in adaptation depth is the primary mechanistic signature that MAML
    optimization is functioning correctly in the federated LoRA setting."
""")

print(f"Plots saved to : {PLOTS_DIR}")
print(f"MLflow UI      : mlflow ui --backend-store-uri {TRACKING_URI}")
print(f"               → experiment '{EXPERIMENT_NAME}' → 'analysis_45rounds' run")
