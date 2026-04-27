"""
Generate evaluation figures from final_eval.json and adaptation_curves.json.

Figures:
  1. Adaptation curves — federated θ* vs pretrained baseline per meta-test speaker
  2. Mean WER with 95% CI bars — pretrained k=0, pretrained k=10, federated k=10
  3. Federation cost — federated vs centralized matched-update (if available)
  4. Support set size sensitivity (from data/insights/support_size_sensitivity.json)

Output: evaluation/figures/*.png

Usage:
    python evaluation/plot_results.py
    python evaluation/plot_results.py --results evaluation/results/final_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

FIGURES_DIR = ROOT / "evaluation" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = ROOT / "evaluation" / "results"

def load_json(path: Path) -> dict | None:
    if not path.exists():
        print(f"  [SKIP] Not found: {path}")
        return None
    return json.loads(path.read_text())

def figure_1_adaptation_curves(curves_data: dict, out_dir: Path) -> None:
    """Figure 1: Adaptation curves — federated θ* vs pretrained baseline."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    k_values = curves_data.get("k_values", [0, 1, 3, 5, 10])
    speakers = curves_data.get("speakers", {})

    if not speakers:
        print("  [SKIP] No speaker data in adaptation_curves.json")
        return

    n_speakers = len(speakers)
    ncols = min(2, n_speakers)
    nrows = (n_speakers + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)

    colors = {"pretrained": "#d62728", "federated": "#1f77b4"}
    labels = {"pretrained": "Pretrained (no MAML)", "federated": "Federated θ*"}

    for idx, (spk_name, curves) in enumerate(speakers.items()):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        spk_short = spk_name[:8]

        for model_key in ("pretrained", "federated"):
            if model_key not in curves:
                continue
            curve = curves[model_key]
            ks = sorted(int(k) for k in curve.keys())
            means = [curve[str(k)]["mean"] for k in ks]
            lowers = [curve[str(k)]["ci_lower"] for k in ks]
            uppers = [curve[str(k)]["ci_upper"] for k in ks]

            ax.plot(ks, means, "o-", color=colors[model_key],
                    label=labels[model_key], linewidth=2, markersize=6)
            ax.fill_between(ks, lowers, uppers, alpha=0.15, color=colors[model_key])

        ax.set_title(f"Speaker {spk_short}", fontsize=12)
        ax.set_xlabel("Adaptation steps (k)", fontsize=10)
        ax.set_ylabel("WER", fontsize=10)
        ax.set_xticks(k_values)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    for idx in range(n_speakers, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle("Figure 1: Adaptation Curves — Federated θ* vs Pretrained Baseline\n"
                 "Shaded region: 95% bootstrap CI", fontsize=13, y=1.02)
    fig.tight_layout()
    out = out_dir / "fig1_adaptation_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

def figure_2_wer_comparison(final_data: dict, out_dir: Path) -> None:
    """Figure 2: Mean WER with 95% CI bars across conditions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    per_speaker = final_data.get("per_speaker", [])
    k = final_data.get("eval_k", 10)

    conditions = [
        ("pretrained_k0", f"Pretrained\nk=0"),
        ("pretrained_adapted", f"Pretrained\nk={k}"),
        ("federated_adapted", f"Federated θ*\nk={k}"),
    ]
    if any("centralized_adapted" in r for r in per_speaker):
        conditions.append(("centralized_adapted", f"Centralized\n(matched)\nk={k}"))

    means = []
    lowers = []
    uppers = []
    tick_labels = []
    colors_bar = []
    color_map = {
        "pretrained_k0": "#aec7e8",
        "pretrained_adapted": "#ffbb78",
        "federated_adapted": "#1f77b4",
        "centralized_adapted": "#2ca02c",
    }

    for key, tick in conditions:
        vals = [r[key]["mean"] for r in per_speaker if key in r]
        los = [r[key]["ci_lower"] for r in per_speaker if key in r]
        his = [r[key]["ci_upper"] for r in per_speaker if key in r]
        if not vals:
            continue
        means.append(float(np.mean(vals)))
        lowers.append(float(np.mean(los)))
        uppers.append(float(np.mean(his)))
        tick_labels.append(tick)
        colors_bar.append(color_map[key])

    if not means:
        print("  [SKIP] No data for Figure 2")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(means))
    bars = ax.bar(x, means, color=colors_bar, alpha=0.85, width=0.6, edgecolor="black", linewidth=0.8)
    ax.errorbar(x, means,
                yerr=[np.array(means) - np.array(lowers),
                      np.array(uppers) - np.array(means)],
                fmt="none", color="black", capsize=6, linewidth=1.5)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=10)
    ax.set_ylabel("WER (lower is better)", fontsize=11)
    ax.set_title(
        "Figure 2: Mean WER Comparison (95% bootstrap CI)\n"
        f"Meta-test speakers (n={len(per_speaker)}), k={k} adaptation steps",
        fontsize=12,
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    out = out_dir / "fig2_wer_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

def figure_3_federation_cost(final_data: dict, out_dir: Path) -> None:
    """Figure 3: Per-speaker WER comparison federated vs centralized matched-update."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    per_speaker = final_data.get("per_speaker", [])
    k = final_data.get("eval_k", 10)

    speakers_with_cen = [r for r in per_speaker if "centralized_adapted" in r]
    if not speakers_with_cen:
        print("  [SKIP] No centralized matched-update checkpoint — Fig 3 skipped")
        return

    spk_names = [r["speaker"][:8] for r in speakers_with_cen]
    fed_means = [r["federated_adapted"]["mean"] for r in speakers_with_cen]
    cen_means = [r["centralized_adapted"]["mean"] for r in speakers_with_cen]
    fed_los = [r["federated_adapted"]["ci_lower"] for r in speakers_with_cen]
    fed_his = [r["federated_adapted"]["ci_upper"] for r in speakers_with_cen]
    cen_los = [r["centralized_adapted"]["ci_lower"] for r in speakers_with_cen]
    cen_his = [r["centralized_adapted"]["ci_upper"] for r in speakers_with_cen]

    x = np.arange(len(spk_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, fed_means, width, label=f"Federated θ* k={k}",
                   color="#1f77b4", alpha=0.85, edgecolor="black", linewidth=0.8)
    bars2 = ax.bar(x + width / 2, cen_means, width, label=f"Centralized (matched) k={k}",
                   color="#2ca02c", alpha=0.85, edgecolor="black", linewidth=0.8)

    ax.errorbar(x - width / 2, fed_means,
                yerr=[np.array(fed_means) - np.array(fed_los),
                      np.array(fed_his) - np.array(fed_means)],
                fmt="none", color="black", capsize=4)
    ax.errorbar(x + width / 2, cen_means,
                yerr=[np.array(cen_means) - np.array(cen_los),
                      np.array(cen_his) - np.array(cen_means)],
                fmt="none", color="black", capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels(spk_names, fontsize=10)
    ax.set_xlabel("Meta-test Speaker", fontsize=11)
    ax.set_ylabel("WER", fontsize=11)
    ax.set_title(
        "Figure 3: Federation Cost\nFederated θ* vs Centralized Matched-Update\n"
        "(Centralized trained for same total gradient updates as federation)",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    out = out_dir / "fig3_federation_cost.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

def figure_4_support_size_sensitivity(out_dir: Path) -> None:
    """Figure 4: Support set size sensitivity from data insights."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    insights_path = ROOT / "data" / "insights" / "support_size_sensitivity.json"
    data = load_json(insights_path)
    if not data:
        return

    per_speaker = data.get("per_speaker", {})
    k_sizes = data.get("k_values_tested", [2, 4, 8, 16])

    if not per_speaker:
        print("  [SKIP] No data in support_size_sensitivity.json")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.get_cmap("tab10", len(per_speaker))

    for i, (spk_name, size_results) in enumerate(per_speaker.items()):
        ks = sorted(int(k) for k in size_results.keys())
        wers = [size_results[str(k)] for k in ks]
        ax.plot(ks, wers, "o-", color=cmap(i), label=spk_name[:8], linewidth=1.8,
                markersize=6)

    ax.set_xlabel("Support set size K (clips)", fontsize=11)
    ax.set_ylabel("WER at k=5 adaptation steps", fontsize=11)
    ax.set_xticks(k_sizes)
    ax.set_title(
        "Figure 4: Support Set Size Sensitivity\n"
        f"(Inner steps = {data.get('inner_steps', 5)}, "
        f"inner_lr = {data.get('inner_lr_used', '1e-4')})",
        fontsize=12,
    )
    ax.legend(title="Speaker", fontsize=9, title_fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    out = out_dir / "fig4_support_size_sensitivity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(RESULTS_DIR / "final_eval.json"))
    parser.add_argument("--curves", default=str(RESULTS_DIR / "adaptation_curves.json"))
    parser.add_argument("--out_dir", default=str(FIGURES_DIR))
    args = parser.parse_args()

    try:
        import matplotlib              
    except ImportError:
        print("ERROR: matplotlib not installed. Run: pip install matplotlib")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating evaluation figures...")

    final_data = load_json(Path(args.results))
    curves_data = load_json(Path(args.curves))

    if curves_data:
        figure_1_adaptation_curves(curves_data, out_dir)
    else:
        print("  [SKIP] adaptation_curves.json not found — run eval_poc.py first")

    if final_data:
        figure_2_wer_comparison(final_data, out_dir)
        figure_3_federation_cost(final_data, out_dir)
    else:
        print("  [SKIP] final_eval.json not found — run eval_poc.py first")

    figure_4_support_size_sensitivity(out_dir)

    print(f"\nFigures saved to: {out_dir}")

if __name__ == "__main__":
    main()
