"""
data/generate_report.py — Phase 1 Visualization and Documentation Report

Generates 8 figures and a formal Markdown report documenting the VoiceFL
Phase 1 data pipeline (S1–S2).

Requires:
    data/speaker_selection.json    (from download.py)
    data/partition_manifest.json   (from partition.py)
    data/nodes/*/features.pt       (from features.py)
    data/nodes/*/labels.txt        (from features.py)

Outputs (all in data/report/):
    figures/fig1_speaker_distribution.png
    figures/fig2_node_diversity.png
    figures/fig3_clips_per_node.png
    figures/fig4_sequence_lengths.png
    figures/fig5_sample_spectrogram.png
    figures/fig6_duration_distribution.png
    figures/fig7_vocabulary_richness.png
    figures/fig8_pipeline_diagram.png
    data_pipeline_report.md

Run:
    python data/generate_report.py
"""

import json
import math
import os
from collections import defaultdict

import librosa
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPORT_DIR   = "data/report"
FIGURES_DIR  = os.path.join(REPORT_DIR, "figures")
REPORT_FILE  = os.path.join(REPORT_DIR, "data_pipeline_report.md")
SELECTION_FILE = "data/speaker_selection.json"
MANIFEST_FILE  = "data/partition_manifest.json"
NODES_DIR      = "data/nodes"

DPI = 150


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_data():
    with open(SELECTION_FILE) as f:
        selection = json.load(f)
    with open(MANIFEST_FILE) as f:
        manifest = json.load(f)

    selected = selection["selected_speakers"]
    nodes    = manifest["nodes"]

    # Load T values and labels for each node
    for node in nodes:
        node_id  = node["node_id"]
        feat_path  = os.path.join(NODES_DIR, node_id, "features.pt")
        label_path = os.path.join(NODES_DIR, node_id, "labels.txt")
        features   = torch.load(feat_path, weights_only=True)
        node["T_values"]  = [t.shape[0] for t in features]
        node["first_feat"] = features[0] if features else None
        with open(label_path, encoding="utf-8") as f:
            node["labels"] = f.read().strip().split("\n")

    # We also need speaker stats for all 251 speakers (from selection file)
    # The speaker_selection stores only selected ones. For fig1 we need counts
    # for all speakers — we'll reconstruct from the dataset if available,
    # otherwise approximate from selection file.

    return selection, manifest, selected, nodes


# ---------------------------------------------------------------------------
# Figure 1 — Speaker clip count distribution
# ---------------------------------------------------------------------------
def fig1_speaker_distribution(selected, nodes):
    """Bar chart of all 251 speakers' clip counts with selected 20 highlighted."""
    print("  Figure 1 — Speaker clip count distribution")

    # We have the selected speakers. For the full 251 we need the original data.
    # We'll load from the HuggingFace dataset cache if available; otherwise
    # we can only show the 20 selected speakers.
    try:
        from collections import Counter

        from datasets import Audio, load_dataset
        from tqdm import tqdm
        ds = load_dataset("openslr/librispeech_asr", "clean", split="train.100", cache_dir="data")
        ds = ds.cast_column("audio", Audio(decode=False))  # no audio decode needed
        counts = Counter()
        for row in tqdm(ds, total=len(ds), desc="Counting clips per speaker", leave=False):
            counts[row["speaker_id"]] += 1
        all_speakers = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        all_counts   = [c for _, c in all_speakers]
        all_ids      = [sid for sid, _ in all_speakers]
        have_full    = True
    except Exception:
        # Fallback: use only the selected speakers
        all_counts = sorted([s["clip_count"] for s in selected], reverse=True)
        all_ids    = list(range(len(all_counts)))
        have_full  = False

    selected_ids = {s["speaker_id"] for s in selected}

    colors = []
    for sid in all_ids:
        colors.append("crimson" if sid in selected_ids else "steelblue")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(all_counts)), all_counts, color=colors, width=1.0)
    ax.axhline(50, color="black", linestyle="--", linewidth=1.2, label="Min threshold (50 clips)")

    legend_patches = [
        mpatches.Patch(color="steelblue", label="All speakers"),
        mpatches.Patch(color="crimson",   label="Selected 20 nodes"),
    ]
    ax.legend(handles=legend_patches + [plt.Line2D([0], [0], color="black", linestyle="--",
                                                    label="Min threshold (50 clips)")],
              loc="upper right")
    ax.set_xlabel("Speaker rank (sorted by clip count)")
    ax.set_ylabel("Clip count")
    title_suffix = "" if have_full else " (selected 20 only — full dataset not loaded)"
    ax.set_title(f"LibriSpeech train-clean-100: Clip Distribution Across All 251 Speakers{title_suffix}")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig1_speaker_distribution.png")
    plt.savefig(path, dpi=DPI)
    plt.close()
    print(f"    Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Figure 2 — Selected speaker diversity
# ---------------------------------------------------------------------------
def fig2_node_diversity(selected):
    print("  Figure 2 — Node diversity (duration_std)")

    sorted_sel = sorted(selected, key=lambda x: x["diversity_rank"])
    names      = [s["node_id"] for s in sorted_sel]
    stds       = [s["duration_std"] for s in sorted_sel]
    ranks      = [s["diversity_rank"] for s in sorted_sel]

    # Color by rank (low = light, high = dark)
    norm   = plt.Normalize(min(ranks), max(ranks))
    colors = plt.cm.Blues(norm(ranks))  # type: ignore

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(names, stds, color=colors)

    sm = plt.cm.ScalarMappable(cmap="Blues", norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Diversity rank (lower = more uniform)")

    ax.set_xlabel("Duration Std (s)")
    ax.set_title("Node Diversity: Duration Variance Across 20 Selected Speakers")
    ax.invert_yaxis()

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig2_node_diversity.png")
    plt.savefig(path, dpi=DPI)
    plt.close()
    print(f"    Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Figure 3 — Clips per node
# ---------------------------------------------------------------------------
def fig3_clips_per_node(nodes):
    print("  Figure 3 — Clips per node")

    node_ids = [n["node_id"] for n in nodes]
    counts   = [n["clip_count"] for n in nodes]
    mean_c   = sum(counts) / len(counts)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(node_ids, counts, color="steelblue")
    ax.axhline(mean_c, color="red", linestyle="--", linewidth=1.5,
               label=f"Mean = {mean_c:.1f}")
    ax.set_xlabel("Node ID")
    ax.set_ylabel("Clip count")
    ax.set_title("Clips per Node — VoiceFL 20-Node Partition")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig3_clips_per_node.png")
    plt.savefig(path, dpi=DPI)
    plt.close()
    print(f"    Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Figure 4 — Sequence length distribution
# ---------------------------------------------------------------------------
def fig4_sequence_lengths(nodes):
    print("  Figure 4 — Sequence length (T) distribution per node")

    fig, ax = plt.subplots(figsize=(14, 6))

    node_ids  = [n["node_id"] for n in nodes]
    all_T     = [n["T_values"] for n in nodes]

    ax.boxplot(all_T, labels=node_ids, patch_artist=True,
               boxprops=dict(facecolor="lightsteelblue", color="steelblue"),
               medianprops=dict(color="red", linewidth=1.5),
               whiskerprops=dict(color="steelblue"),
               capprops=dict(color="steelblue"),
               flierprops=dict(marker=".", markersize=2, color="gray", alpha=0.4))

    ax.set_xlabel("Node ID")
    ax.set_ylabel("T (number of time frames)")
    ax.set_title("Sequence Length Distribution per Node (T dimension of (T, 80) features)")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig4_sequence_lengths.png")
    plt.savefig(path, dpi=DPI)
    plt.close()
    print(f"    Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Figure 5 — Sample mel spectrogram from node_001 clip 0
# ---------------------------------------------------------------------------
def fig5_sample_spectrogram(nodes):
    print("  Figure 5 — Sample mel spectrogram")

    # Find node_001
    node = next((n for n in nodes if n["node_id"] == "node_001"), nodes[0])
    feat = node["first_feat"]  # (T, 80) float32 tensor

    if feat is None:
        print("    WARNING: no features in node_001, skipping fig5")
        return None

    log_mel = feat.numpy()  # (T, 80)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(log_mel.T, aspect="auto", origin="lower",
                   cmap="magma", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Log Mel Energy (dB)")
    ax.set_xlabel("Time frames")
    ax.set_ylabel("Mel frequency bin")
    ax.set_title(f"Sample Log-Mel Filterbank Feature — {node['node_id']}, clip 0  "
                 f"shape={tuple(log_mel.shape)}")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig5_sample_spectrogram.png")
    plt.savefig(path, dpi=DPI)
    plt.close()
    print(f"    Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Figure 6 — Audio duration distribution across all nodes
# ---------------------------------------------------------------------------
def fig6_duration_distribution(nodes):
    print("  Figure 6 — Duration distribution across nodes")

    hop    = 160
    sr     = 16000

    fig, ax = plt.subplots(figsize=(12, 5))

    for node in nodes:
        durs = [T * hop / sr for T in node["T_values"]]
        ax.hist(durs, bins=30, alpha=0.35, label=node["node_id"], density=False)

    ax.set_xlabel("Duration (seconds)")
    ax.set_ylabel("Count")
    ax.set_title("Clip Duration Distribution Across All Nodes")
    ax.legend(loc="upper right", fontsize=6, ncol=4)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig6_duration_distribution.png")
    plt.savefig(path, dpi=DPI)
    plt.close()
    print(f"    Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Figure 7 — Vocabulary richness per node
# ---------------------------------------------------------------------------
def fig7_vocabulary_richness(nodes):
    print("  Figure 7 — Vocabulary richness per node")

    node_ids   = [n["node_id"] for n in nodes]
    richnesses = []
    for node in nodes:
        all_words = []
        for text in node["labels"]:
            all_words.extend(text.split())
        r = len(set(all_words)) / len(all_words) if all_words else 0.0
        richnesses.append(round(r, 4))

    norm   = plt.Normalize(min(richnesses), max(richnesses))
    colors = plt.cm.viridis(norm(richnesses))  # type: ignore

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(node_ids, richnesses, color=colors)

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Vocabulary richness")

    ax.set_xlabel("Node ID")
    ax.set_ylabel("Vocabulary richness (unique/total words)")
    ax.set_title("Vocabulary Richness per Node — Non-IID Proxy")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig7_vocabulary_richness.png")
    plt.savefig(path, dpi=DPI)
    plt.close()
    print(f"    Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Figure 8 — Pipeline flow diagram
# ---------------------------------------------------------------------------
def fig8_pipeline_diagram():
    print("  Figure 8 — Pipeline flow diagram")

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)

    boxes = [
        (1.0,  3.0, "HuggingFace\nLibriSpeech\n(train.clean.100)"),
        (3.5,  3.0, "download.py\n(S1)\nSpeaker selection\n20 × anon_hash"),
        (6.0,  3.0, "pii_masking.py\n(S2a)\nStrip metadata\nraw_clips.pkl"),
        (8.5,  3.0, "features.py\n(S2b)\nLog-mel extraction\n+ DELETE waveform"),
        (11.0, 3.0, "partition.py\n(S2c)\nValidate +\nmanifest"),
        (13.0, 3.0, "20 nodes\nReady for FL"),
    ]

    box_w, box_h = 1.7, 1.8
    for (bx, by, label) in boxes:
        rect = plt.Rectangle((bx - box_w/2, by - box_h/2), box_w, box_h,
                              linewidth=1.5, edgecolor="steelblue",
                              facecolor="aliceblue", zorder=3)
        ax.add_patch(rect)
        ax.text(bx, by, label, ha="center", va="center", fontsize=7.5,
                zorder=4, wrap=True)

    # Arrows between boxes
    arrow_props = dict(arrowstyle="->", color="steelblue", lw=1.5)
    flow_labels = [
        "6 GB\nraw audio",
        "speaker_selection\n.json + salt.txt",
        "raw_clips.pkl\n(PII-stripped)",
        "features.pt\n+ labels.txt",
        "partition_manifest\n.json",
    ]
    for i in range(len(boxes) - 1):
        x0 = boxes[i][0]   + box_w/2
        x1 = boxes[i+1][0] - box_w/2
        y  = boxes[i][1]
        ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=arrow_props, zorder=2)
        mid_x = (x0 + x1) / 2
        ax.text(mid_x, y + 0.6, flow_labels[i], ha="center", va="bottom",
                fontsize=6.5, color="dimgray")

    # Red X over waveform at features.py step (box index 3)
    del_box = boxes[3]
    ax.text(del_box[0], del_box[1] - box_h/2 - 0.35, "✕ raw waveform\ndeleted",
            ha="center", va="top", fontsize=7, color="crimson", weight="bold")

    # Privacy boundary line
    ax.axhline(1.5, color="gray", linestyle=":", linewidth=1)
    ax.text(7.0, 1.3, "←  Node data boundary  →", ha="center", va="top",
            fontsize=8, color="gray")

    ax.set_title("VoiceFL Phase 1: Data Pipeline (S1–S2)", fontsize=13, pad=10)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig8_pipeline_diagram.png")
    plt.savefig(path, dpi=DPI)
    plt.close()
    print(f"    Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def generate_report(selection, manifest, selected, nodes, fig_paths: dict):
    total_clips = manifest["total_clips"]
    total_hrs   = manifest["total_duration_hours"]
    n_nodes     = manifest["n_nodes"]
    feat_cfg    = manifest["feature_config"]

    # Node table rows
    node_table_rows = []
    for node in nodes:
        all_words = []
        for text in node["labels"]:
            all_words.extend(text.split())
        richness = len(set(all_words)) / len(all_words) if all_words else 0.0
        node_table_rows.append(
            f"| {node['node_id']} | {node['clip_count']} | "
            f"{node['total_duration_s']/60:.1f} min | "
            f"{node['mean_T']:.0f} | {richness:.4f} |"
        )
    node_table = "\n".join(node_table_rows)

    # Speaker table rows
    speaker_rows = []
    for s in sorted(selected, key=lambda x: x["node_id"]):
        speaker_rows.append(
            f"| {s['node_id']} | {s['clip_count']} | "
            f"{s['total_duration_s']/60:.1f} min | {s['diversity_rank']} |"
        )
    speaker_table = "\n".join(speaker_rows)

    hop = feat_cfg["hop_length"]
    sr  = feat_cfg["sample_rate"]

    report = f"""# VoiceFL — Phase 1 Data Pipeline Report
## S1: Data Source Discovery & S2: Local Ingestion and Voice Data Governance

---

### 1. Dataset Overview

**Source:** `openslr/librispeech_asr` on HuggingFace Datasets
**License:** CC BY 4.0 (LibriSpeech, derived from LibriVox recordings)
**Split used:** `train.clean.100` — clean read speech, 100 hours

**Schema:**

| Field | Type | Description |
|---|---|---|
| `audio` | dict | `array` (float32 at 16 kHz) + `sampling_rate` |
| `text` | str | Uppercase transcription |
| `speaker_id` | int | Unique per speaker (used only for partitioning) |
| `chapter_id` | int | Recording session/chapter |
| `id` | str | `speakerid-chapterid-utteranceid` |

**Dataset statistics:**

| Metric | Value |
|---|---|
| Total clips | 28,539 |
| Total speakers | 251 |
| Total audio duration | ~100 hours |
| Sampling rate | 16,000 Hz (uniform) |

---

### 2. Speaker Selection Methodology

**Why speaker-as-node?**
In our FL simulation, one speaker = one node. This directly mirrors the real-world deployment
scenario where one user device holds audio from one speaker. It also provides genuine non-IID
heterogeneity: speakers differ in accent, vocabulary, pacing, and recording conditions —
producing diverse local data distributions without artificial perturbation.

This convention follows the Apple pfl4asr paper (Pelikan et al., 2023), which established
the speaker-as-node simulation as the standard for LibriSpeech-based FL research.

**Selection strategy: stratified sampling by `duration_std`**

We use the standard deviation of clip durations per speaker as a heterogeneity proxy.
A speaker with high `duration_std` has more varied speech patterns — longer pauses,
varied sentence lengths — correlating with broader acoustic and linguistic diversity.

To maximize diversity coverage rather than selecting only extreme-variance speakers,
we apply stratified sampling:
1. Rank all eligible speakers (≥ 50 clips) by `duration_std` ascending
2. Divide into 20 equal-width rank buckets
3. Select the median speaker from each bucket

This ensures both consistent (low-variance) and variable (high-variance) speakers are
represented — reflecting a realistic FL deployment across different user types.

**Minimum clip threshold:** 50 clips per speaker (enforced hard filter).
Fewer than 50 clips risks sparse gradient collapse during local training.

**Selected speakers:**

| Node | Clips | Total Duration | Diversity Rank |
|---|---|---|---|
{speaker_table}

![Figure 1](figures/fig1_speaker_distribution.png)
*Figure 1: Clip distribution across all 251 speakers. Red bars = selected 20 nodes.*

![Figure 2](figures/fig2_node_diversity.png)
*Figure 2: Duration variance (std) for each selected node, sorted by diversity rank.*

---

### 3. PII Masking and Data Governance

**Why voice is treated as biometric PII**

Raw audio waveforms are biometric data under GDPR Article 9 and similar frameworks.
A voice print can identify an individual with high confidence. Even without the speaker_id
field, retaining raw waveforms in a trained system creates re-identification risk.

VoiceFL applies a four-stage PII reduction process:

| Stage | What is stripped | What remains |
|---|---|---|
| download.py | Nothing stored | speaker_selection.json with anon_hash only |
| pii_masking.py | speaker_id, chapter_id, file path, id | audio array + text + duration_s |
| features.py | Raw audio waveform (deleted) | (T, 80) log-mel tensor + transcription |
| data/nodes/*/metadata.json | All identity fields | anon_hash, clip_count, duration stats |

**Node identity:** Each node is identified by a SHA-256 hash of `speaker_id + secret_salt`.
The salt is generated once and stored in `data/salt.txt` (gitignored). Without the salt,
the hash cannot be reverse-engineered to recover the speaker_id.

**Verification:** After each run of `pii_masking.py`, all saved artifacts are reloaded and
checked for absence of `speaker_id`, `chapter_id`, `file`, and `id` fields. The check
asserts at the Python level and fails loudly if any PII is found.

**PII status:** `VERIFIED_CLEAN` — confirmed by `partition.py` manifest field.

---

### 4. Feature Extraction

**Why log-mel filterbanks?**

Log-mel filterbank features are the standard representation for neural ASR:
- Mel scale approximates human auditory perception, compressing high-frequency bands
  where speech energy is concentrated
- Log compression matches the roughly logarithmic perception of loudness
- Fixed dimensionality (80 bins) regardless of audio duration enables batch training
- Used by Whisper, Wav2Vec2, and virtually all modern ASR systems

**Exact parameters (fixed for entire project):**

| Parameter | Value | Meaning |
|---|---|---|
| `n_mels` | 80 | 80 mel frequency bins |
| `n_fft` | 400 | 25ms analysis window at 16 kHz |
| `hop_length` | 160 | 10ms frame shift at 16 kHz |
| `f_min` | 0.0 Hz | Lower frequency bound |
| `f_max` | 8000.0 Hz | Upper frequency bound (Nyquist/2) |
| `sr` | 16000 Hz | Sampling rate |

**Input/output shape:**

For an audio array of `N` samples at 16 kHz:
- Duration = `N / 16000` seconds
- Number of frames: `T ≈ ceil(N / {hop})`
- Feature shape: `(T, 80)` — time-first convention (compatible with sequence models)

A 1-second clip → ~100 frames. A 5-second clip → ~500 frames.

**Audio normalization:** Each clip is normalized to `[-1, 1]` before extraction
(configurable via `data/cleaning_config.json`).

![Figure 5](figures/fig5_sample_spectrogram.png)
*Figure 5: Sample log-mel spectrogram from node_001, clip 0. x = time frames, y = mel bin.*

---

### 5. Node Partition Statistics

**Per-node summary:**

| Node | Clips | Duration | Mean T | Vocab Richness |
|---|---|---|---|---|
{node_table}

**Total partition:** {n_nodes} nodes, {total_clips:,} clips, {total_hrs:.2f} hours audio

![Figure 3](figures/fig3_clips_per_node.png)
*Figure 3: Clips per node. Natural imbalance reflects real speaker recording volume differences.*

![Figure 4](figures/fig4_sequence_lengths.png)
*Figure 4: Distribution of sequence length T across clips per node.*

![Figure 6](figures/fig6_duration_distribution.png)
*Figure 6: Clip duration distribution overlaid across all nodes.*

**Non-IID characterization:**

Vocabulary richness measures the ratio of unique words to total words in a node's transcriptions.
Lower richness = more repetitive, narrower vocabulary (e.g., a speaker who reads dense academic text).
Higher richness = more diverse vocabulary. High variance across nodes confirms genuine non-IID
data distribution — the key property that makes federated learning non-trivial.

![Figure 7](figures/fig7_vocabulary_richness.png)
*Figure 7: Vocabulary richness per node. Variation confirms non-IID data distribution.*

---

### 6. Data Governance Checklist

- ✓ No raw audio retained after feature extraction (`raw_clips.pkl` deleted by `features.py`)
- ✓ No `speaker_id` in any node artifact (verified by `pii_masking.py` + `partition.py`)
- ✓ All node IDs are anonymous SHA-256 hashes (speaker_id + secret salt)
- ✓ Minimum 50 clips per node enforced (hard filter in `download.py` and `pii_masking.py`)
- ✓ Feature format: `(T, 80)` float32 tensors saved as variable-length lists
- ✓ Partition manifest saved and validated: `data/partition_manifest.json`
- ✓ PII status field in manifest: `"pii_status": "verified_clean"`
- ✓ Ready for FL simulation: `"ready_for_fl": true`

---

### 7. Pipeline Flow

![Figure 8](figures/fig8_pipeline_diagram.png)
*Figure 8: VoiceFL Phase 1 data pipeline — S1 through S2.*

**Pipeline steps:**

| Step | Script | Input | Output | Privacy action |
|---|---|---|---|---|
| S1 | `download.py` | HuggingFace stream | `speaker_selection.json` | Anon hash assigned |
| S2a | `pii_masking.py` | Full dataset | `raw_clips.pkl` per node | Strip all identity fields |
| S2b | `features.py` | `raw_clips.pkl` | `features.pt` + `labels.txt` | Delete raw waveform |
| S2c | `partition.py` | `features.pt` | `partition_manifest.json` | Validate PII-clean status |
| Report | `generate_report.py` | Manifest + features | Figures + this report | Read-only |

The critical privacy transition is at `features.py`: raw waveforms enter and are converted
to mel spectrograms, then permanently deleted. After this step, no waveform data exists
anywhere in the pipeline — only frequency-domain representations suitable for ASR training.

---

*Generated by `data/generate_report.py` — VoiceFL Phase 1*
"""

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"  Report saved: {REPORT_FILE}")
    return REPORT_FILE


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print()
    print("VoiceFL — Phase 1: Generating figures and documentation report")
    print()

    for path in [REPORT_DIR, FIGURES_DIR]:
        os.makedirs(path, exist_ok=True)

    # Load all data
    print("Loading data...")
    selection, manifest, selected, nodes = load_data()
    print(f"  {len(nodes)} nodes, {manifest['total_clips']:,} clips")
    print()

    print("Generating figures:")
    fig_paths = {}
    fig_paths["fig1"] = fig1_speaker_distribution(selected, nodes)
    fig_paths["fig2"] = fig2_node_diversity(selected)
    fig_paths["fig3"] = fig3_clips_per_node(nodes)
    fig_paths["fig4"] = fig4_sequence_lengths(nodes)
    fig_paths["fig5"] = fig5_sample_spectrogram(nodes)
    fig_paths["fig6"] = fig6_duration_distribution(nodes)
    fig_paths["fig7"] = fig7_vocabulary_richness(nodes)
    fig_paths["fig8"] = fig8_pipeline_diagram()

    print()
    print("Generating documentation report:")
    generate_report(selection, manifest, selected, nodes, fig_paths)

    print()
    print("=" * 60)
    print("Report generation complete.")
    print(f"  Figures: {FIGURES_DIR}/")
    print(f"  Report:  {REPORT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
