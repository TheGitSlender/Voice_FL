"""
Generate all figures for the VoiceFL-MAML PoC final report.
Saves PNGs to ~/Desktop/Obsidian/voicefl/report/figures/
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

ROOT = Path(__file__).parent.parent
OUT  = Path.home() / "Desktop" / "Obsidian" / "voicefl" / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── colour palette ──────────────────────────────────────────────────────────
C_BLUE    = "#4C78A8"
C_ORANGE  = "#F58518"
C_GREEN   = "#54A24B"
C_RED     = "#E45756"
C_PURPLE  = "#B279A2"
C_TEAL    = "#4CAFA0"
C_GREY    = "#9B9B9B"
BG        = "#FAFAFA"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   BG,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.color":       "#E0E0E0",
    "grid.linewidth":   0.6,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
})

NODE_LABELS = ["Node 1\nb15129ee", "Node 2\nc57fbbea", "Node 3\n8a12e006",
               "Node 4\nf14ed0f2", "Node 5\n6cea687f"]
NODE_SHORT  = ["Node 1", "Node 2", "Node 3", "Node 4", "Node 5"]

# ── load data ───────────────────────────────────────────────────────────────
metrics_path = ROOT / "checkpoints" / "centralized" / "training_metrics.json"
step7_path   = ROOT / "checkpoints" / "centralized" / "step7_results.json"

metrics = json.loads(metrics_path.read_text())
step7   = json.loads(step7_path.read_text())

rounds      = [m["round"]           for m in metrics]
losses      = [m["avg_query_loss"]  for m in metrics]
grad_norms  = [m["grad_norm"]       for m in metrics]

# centralized gate
nodes_order = sorted(step7.keys())  # sort by hash for consistency
cen_k0  = [step7[n]["wer_k0"] * 100 for n in nodes_order]
cen_k10 = [step7[n]["wer_k3"] * 100 for n in nodes_order]

# federated final eval (hardcoded from eval_poc.py 20-round run)
fed_k0  = [7.9, 8.1, 3.7, 1.6, 10.7]   # %
fed_k10 = [4.3, 4.3, 2.9, 1.0,  2.7]

# criterion 2 comparison
fed_raw = 5.4  # % mean WER (no-adapt)
cen_raw = 5.7  # % mean WER (no-adapt)

# ── FIG 1: Training Loss Curve ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(rounds, losses, color=C_BLUE, lw=2, marker="o", markersize=5,
        markerfacecolor="white", markeredgewidth=1.5, label="Avg query loss")

# trend line
z = np.polyfit(rounds, losses, 1)
p = np.poly1d(z)
ax.plot(rounds, p(rounds), "--", color=C_ORANGE, lw=1.5, alpha=0.8,
        label=f"Trend  (slope {z[0]:+.3f}/round)")

ax.axhspan(min(losses) - 1, min(losses) + 1, color=C_GREEN, alpha=0.08)
ax.set_xlabel("Federated Round")
ax.set_ylabel("CTC Query Loss")
ax.set_title("Fig 1 — Centralized FOMAML: Query Loss over 20 Rounds")
ax.set_xlim(0.5, 20.5)
ax.legend(framealpha=0.9)
fig.tight_layout()
fig.savefig(OUT / "fig1_training_loss.png", dpi=150)
plt.close()
print("fig1 done")

# ── FIG 2: Gradient Norm ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(rounds, grad_norms, color=C_PURPLE, alpha=0.75, width=0.7,
       label="Raw grad norm (before clip)")

# clip line
ax.axhline(1.0, color=C_RED, lw=1.8, ls="--", label="Clip threshold (1.0)")
ax.axhline(np.mean(grad_norms), color=C_ORANGE, lw=1.5, ls=":",
           label=f"Mean = {np.mean(grad_norms):.0f}")

ax.set_xlabel("Federated Round")
ax.set_ylabel("L2 Norm of avg meta-gradient")
ax.set_title("Fig 2 — Meta-Gradient Norm Before Clipping (max_norm=1.0)")
ax.set_xlim(0.5, 20.5)
ax.legend(framealpha=0.9)

# annotate clipping note
ax.text(10.5, 1020,
        "All raw norms 650–1035 → clipped to 1.0\n"
        "Effective step = β × 1.0 = 2×10⁻⁴  (tiny, intentional)",
        fontsize=9, color="#555555",
        bbox=dict(boxstyle="round,pad=0.4", fc="#FFFBE6", ec="#E0C060", alpha=0.9))
fig.tight_layout()
fig.savefig(OUT / "fig2_grad_norm.png", dpi=150)
plt.close()
print("fig2 done")

# ── FIG 3: Centralized Gate WER ─────────────────────────────────────────────
x = np.arange(5)
w = 0.35
fig, ax = plt.subplots(figsize=(10, 5))
b1 = ax.bar(x - w/2, cen_k0,  w, color=C_RED,   alpha=0.85, label="WER k=0 (perturbed, no adapt)")
b2 = ax.bar(x + w/2, cen_k10, w, color=C_GREEN,  alpha=0.85, label="WER k=10 (perturbed + adapt)")

# value labels
for bar in b1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
for bar in b2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9,
            color=C_GREEN, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(NODE_SHORT)
ax.set_ylabel("Word Error Rate (%)")
ax.set_title("Fig 3 — Centralized Gate (Step 7): WER Before vs After Adaptation\n"
             "ANIL protocol: lm_head perturbed std=0.3, same noise seed, k=10 @ lr=1e-3")
ax.legend(framealpha=0.9)
ax.set_ylim(0, max(cen_k0) * 1.25)

# improvement arrows
for i, (k0, k10) in enumerate(zip(cen_k0, cen_k10)):
    improvement = (k0 - k10) / k0 * 100
    ax.annotate("", xy=(i + w/2, k10 + 0.05), xytext=(i + w/2, k0 - 0.05),
                arrowprops=dict(arrowstyle="-|>", color=C_TEAL, lw=1.5))
    ax.text(i + w/2 + 0.2, (k0 + k10)/2,
            f"−{improvement:.0f}%", fontsize=8, color=C_TEAL, va="center")

fig.tight_layout()
fig.savefig(OUT / "fig3_centralized_gate.png", dpi=150)
plt.close()
print("fig3 done")

# ── FIG 4: WER Improvement % ────────────────────────────────────────────────
cen_imp  = [(k0 - k10) / k0 * 100 for k0, k10 in zip(cen_k0,  cen_k10)]
fed_imp  = [(k0 - k10) / k0 * 100 for k0, k10 in zip(fed_k0,  fed_k10)]

x = np.arange(5)
w = 0.38
fig, ax = plt.subplots(figsize=(10, 5))
b1 = ax.bar(x - w/2, cen_imp, w, color=C_BLUE,   alpha=0.85, label="Centralized θ* (20 rounds)")
b2 = ax.bar(x + w/2, fed_imp, w, color=C_ORANGE,  alpha=0.85, label="Federated θ* (20 FL rounds)")

for bar, v in zip(list(b1) + list(b2), cen_imp + fed_imp):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{v:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(NODE_SHORT)
ax.set_ylabel("WER Reduction  (k=0 → k=10)  %")
ax.set_title("Fig 4 — Adaptation Gain: WER Reduction After k=10 Steps\n"
             "Centralized vs Federated θ* on the same 5 nodes")
ax.legend(framealpha=0.9)
ax.set_ylim(0, 110)

fig.tight_layout()
fig.savefig(OUT / "fig4_improvement.png", dpi=150)
plt.close()
print("fig4 done")

# ── FIG 5: Federated Final WER ───────────────────────────────────────────────
x = np.arange(5)
w = 0.35
fig, ax = plt.subplots(figsize=(10, 5))
b1 = ax.bar(x - w/2, fed_k0,  w, color=C_RED,   alpha=0.85, label="WER k=0 (perturbed, no adapt)")
b2 = ax.bar(x + w/2, fed_k10, w, color=C_GREEN,  alpha=0.85, label="WER k=10 (perturbed + adapt)")

for bar in b1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
for bar in b2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9,
            color=C_GREEN, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(NODE_SHORT)
ax.set_ylabel("Word Error Rate (%)")
ax.set_title("Fig 5 — Federated Eval (Step 10): WER Before vs After Adaptation\n"
             "20-round federated θ* — all 5 nodes PASS")
ax.legend(framealpha=0.9)
ax.set_ylim(0, max(fed_k0) * 1.3)

for i, (k0, k10) in enumerate(zip(fed_k0, fed_k10)):
    improvement = (k0 - k10) / k0 * 100
    ax.annotate("", xy=(i + w/2, k10 + 0.05), xytext=(i + w/2, k0 - 0.05),
                arrowprops=dict(arrowstyle="-|>", color=C_TEAL, lw=1.5))
    ax.text(i + w/2 + 0.2, (k0 + k10)/2,
            f"−{improvement:.0f}%", fontsize=8, color=C_TEAL, va="center")

fig.tight_layout()
fig.savefig(OUT / "fig5_federated_eval.png", dpi=150)
plt.close()
print("fig5 done")

# ── FIG 6: Federated vs Centralized quality (Criterion 2) ───────────────────
fig, ax = plt.subplots(figsize=(6, 5))

categories = ["Centralized\nθ* (20 rounds)", "Federated\nθ* (20 FL rounds)"]
values     = [cen_raw, fed_raw]
colors     = [C_BLUE, C_ORANGE]

bars = ax.bar(categories, values, color=colors, width=0.45, alpha=0.85)
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f"{v:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")

# tolerance band: cen * 1.15
tolerance = cen_raw * 1.15
ax.axhline(tolerance, color=C_RED, lw=1.8, ls="--",
           label=f"+15% tolerance ceiling = {tolerance:.1f}%")
ax.fill_between([-0.5, 1.5], [cen_raw, cen_raw], [tolerance, tolerance],
                color=C_RED, alpha=0.06)

# difference annotation
diff_pct = (fed_raw - cen_raw) / cen_raw * 100
ax.annotate("", xy=(1, fed_raw), xytext=(0, cen_raw),
            arrowprops=dict(arrowstyle="-|>", color=C_GREEN, lw=2))
ax.text(0.5, (cen_raw + fed_raw)/2 + 0.1, f"{diff_pct:+.1f}%\n(federated is\nbetter)",
        ha="center", fontsize=9, color=C_GREEN, fontweight="bold")

ax.set_ylabel("Mean WER — no adaptation (perturbed head)")
ax.set_title("Fig 6 — Criterion 2: Federated vs Centralized θ* Quality\n"
             "Federated is 5.1% better → well within 15% tolerance")
ax.set_ylim(0, tolerance * 1.35)
ax.legend(framealpha=0.9)
fig.tight_layout()
fig.savefig(OUT / "fig6_criterion2.png", dpi=150)
plt.close()
print("fig6 done")

# ── FIG 7: ANIL Evaluation Protocol schematic ────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
ax.set_xlim(0, 12)
ax.set_ylim(-0.5, 4)
ax.axis("off")
ax.set_facecolor("#F8F9FF")
fig.patch.set_facecolor("#F8F9FF")

def box(ax, x, y, w, h, txt, color, fs=9):
    rect = mpatches.FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.15", fc=color, ec="white", lw=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, txt, ha="center", va="center",
            fontsize=fs, fontweight="bold", color="white", zorder=4,
            multialignment="center")

def arrow(ax, x1, y, x2, label="", color="black"):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8), zorder=5)
    if label:
        ax.text((x1+x2)/2, y + 0.18, label, ha="center", fontsize=8, color=color)

# shared encoder
box(ax, 0.2, 1.2, 2.0, 1.6, "Meta-trained\nEncoder θ*\n(shared)", C_BLUE, fs=9)

# perturbation
box(ax, 2.8, 2.0, 1.8, 0.9, "Perturb\nlm_head\nstd=0.3", C_PURPLE, fs=8)
arrow(ax, 2.2, 2.5, 2.8, color=C_PURPLE)

# fork into k=0 and k=k paths
ax.plot([4.6, 4.6], [2.45, 3.2], color=C_GREY, lw=1.5, zorder=2)
ax.plot([4.6, 4.6], [2.45, 1.7], color=C_GREY, lw=1.5, zorder=2)

# k=0 path (top)
arrow(ax, 4.6, 3.2, 5.0, color=C_RED)
box(ax, 5.0, 2.75, 1.9, 0.9, "k=0\nno adapt", C_RED, fs=9)
box(ax, 7.3, 2.75, 1.9, 0.9, "WER k=0\n(baseline)", "#D44", fs=9)
arrow(ax, 6.9, 3.2, 7.3, color=C_RED)

# k=K path (bottom)
arrow(ax, 4.6, 1.7, 5.0, color=C_GREEN)
box(ax, 5.0, 1.25, 1.9, 0.9, f"k=10 steps\nα=1e-3\n(support)", C_GREEN, fs=8)
box(ax, 7.3, 1.25, 1.9, 0.9, "WER k=10\n(after adapt)", "#2A7", fs=9)
arrow(ax, 6.9, 1.7, 7.3, color=C_GREEN)

# comparison
box(ax, 9.5, 1.8, 2.2, 1.4, "Compare:\nWER k=10\n< WER k=0?\n→ PASS", C_TEAL, fs=9)
arrow(ax, 9.2, 3.2, 9.5+0.5, color=C_RED)
arrow(ax, 9.2, 1.7, 9.5+0.5, color=C_GREEN)

# same seed annotation
ax.text(2.8, 1.0, "Same noise seed\nfor k=0 and k=10\n(fair comparison)",
        fontsize=8, ha="left", color=C_PURPLE,
        bbox=dict(boxstyle="round", fc="white", ec=C_PURPLE, alpha=0.8))

ax.set_title("Fig 7 — ANIL Evaluation Protocol: Why We Perturb the lm_head",
             fontsize=13, fontweight="bold", pad=10)
fig.tight_layout()
fig.savefig(OUT / "fig7_anil_protocol.png", dpi=150)
plt.close()
print("fig7 done")

# ── FIG 8: Summary scorecard ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
ax.axis("off")

rows = [
    ("Criterion 1",  "Adaptation",     "WER(k=10) < WER(k=0) on ≥4/5 nodes",  "5 / 5",  "✓"),
    ("Criterion 2",  "Federation",     "Federated WER ≤ Centralized × 1.15",   "5.1% better", "✓"),
    ("Criterion 3",  "Sovereignty",    "No raw audio, no speaker_id on disk",   "Clean",  "✓"),
    ("Invariant I1", "No raw audio",   "No .pkl after features.py",             "0 files","✓"),
    ("Invariant I2", "No speaker ID",  "No speaker_id in artifacts",            "0 hits", "✓"),
    ("Invariant I3", "lm_head private","lm_head not in get_outer_loop_params()", "0 overlap","✓"),
    ("Invariant I4", "Grad protocol",  "fit() returns meta-gradients not weights","Verified","✓"),
    ("Invariant I5", "No data leak",   "support ∩ query = ∅",                   "Asserted","✓"),
    ("Invariant I6", "Grad flow",      "encoder.requires_grad=True",            "All True","✓"),
]

col_labels = ["ID", "Name", "What it checks", "Result", ""]
col_widths  = [0.12, 0.13, 0.47, 0.18, 0.05]
col_x       = np.cumsum([0] + col_widths[:-1])

# header
for j, (lbl, cw, cx) in enumerate(zip(col_labels, col_widths, col_x)):
    ax.text(cx + cw/2, 0.97, lbl, ha="center", va="top",
            fontsize=10, fontweight="bold", color="white",
            transform=ax.transAxes)
ax.add_patch(plt.Rectangle((0, 0.90), 1, 0.08, transform=ax.transAxes,
                            color=C_BLUE, zorder=0))

# rows
row_h = 0.085
for i, row in enumerate(rows):
    y = 0.88 - i * row_h
    bg = "#F0FFF0" if i % 2 == 0 else "#FAFFFE"
    ax.add_patch(plt.Rectangle((0, y - row_h + 0.005), 1, row_h - 0.005,
                                transform=ax.transAxes, color=bg, zorder=0))
    for j, (val, cw, cx) in enumerate(zip(row, col_widths, col_x)):
        color = C_GREEN if j == 4 else "black"
        fw = "bold" if j in (0, 4) else "normal"
        ax.text(cx + cw/2, y - row_h/2, val, ha="center", va="center",
                fontsize=9, color=color, fontweight=fw,
                transform=ax.transAxes)

ax.set_title("Fig 8 — Final Scorecard: All 3 Criteria + All 6 Invariants",
             fontsize=13, fontweight="bold", pad=12)
fig.tight_layout()
fig.savefig(OUT / "fig8_scorecard.png", dpi=150)
plt.close()
print("fig8 done")

print(f"\nAll figures saved to {OUT}")
