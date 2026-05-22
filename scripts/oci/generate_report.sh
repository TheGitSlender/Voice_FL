#!/usr/bin/env bash
# generate_report.sh — Generate the training metrics report (Table 9) after training.
#
# Reads MLflow data from completed training runs and generates:
#   1. Five diagnostic plots in reports/analysis/
#   2. A formatted training metrics table to stdout
#   3. JSON summary in checkpoints/federated/training_metrics.json
#
# Prerequisites:
#   - Training must have completed (MLflow data in mlruns/)
#   - Python with mlflow, matplotlib, numpy installed
#
# Usage:
#   bash scripts/oci/generate_report.sh
#   bash scripts/oci/generate_report.sh --rounds 1,25,45,50
#   bash scripts/oci/generate_report.sh --mlruns_dir /path/to/mlruns

set -euo pipefail
cd "$(dirname "$0")/../.."

# ── Parse arguments ───────────────────────────────────────────────────────────
CHECKPOINT_ROUNDS="1,25,45,50"
MLRUNS_DIR="mlruns"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rounds)       CHECKPOINT_ROUNDS="$2"; shift 2 ;;
        --mlruns_dir)   MLRUNS_DIR="$2"; shift 2 ;;
        -h|--help)
            grep '^#' "$0" | head -20 | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

log()  { printf '\033[36m[report]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[report] WARNING:\033[0m %s\n' "$*" >&2; }
err()  { printf '\033[31m[report] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# ── 1. Check prerequisites ───────────────────────────────────────────────────
[[ -d "$MLRUNS_DIR" ]] || err "MLflow directory not found: $MLRUNS_DIR"

python3 -c "import mlflow, matplotlib, numpy" 2>/dev/null \
    || err "Missing Python dependencies. Run: pip install mlflow matplotlib numpy"

mkdir -p reports/analysis

# ── 2. Run the analysis script (generates 5 diagnostic plots) ────────────────
log "Running analysis script..."
if [[ -f scripts/analyse_run.py ]]; then
    MLFLOW_TRACKING_URI="$MLRUNS_DIR" python3 scripts/analyse_run.py 2>/dev/null && \
        log "Diagnostic plots generated in reports/analysis/" || \
        warn "analyse_run.py encountered issues (may need 2 MLflow sessions to merge)"
fi

# ── 3. Extract training metrics at checkpoint rounds ─────────────────────────
log "Extracting metrics at rounds: $CHECKPOINT_ROUNDS"

python3 - "$MLRUNS_DIR" "$CHECKPOINT_ROUNDS" << 'PYEOF'
import sys
import json
from pathlib import Path

mlruns_dir = sys.argv[1]
checkpoint_rounds = [int(r) for r in sys.argv[2].split(",")]

# Try to load training_metrics.json from checkpoints
metrics_files = [
    Path("checkpoints/federated/training_metrics.json"),
    Path("checkpoints/centralized/training_metrics.json"),
]

metrics_data = None
for mf in metrics_files:
    if mf.exists():
        metrics_data = json.loads(mf.read_text())
        break

if metrics_data is None:
    # Fallback: try to read from MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri(mlruns_dir)
        # Find the latest experiment
        experiments = mlflow.search_experiments()
        if experiments:
            runs = mlflow.search_runs(
                experiment_ids=[experiments[0].experiment_id],
                order_by=["start_time DESC"],
                max_results=1,
            )
            if not runs.empty:
                print("Found MLflow run, but structured extraction requires training_metrics.json")
                print("Run the training with the latest code to generate this file.")
                sys.exit(0)
    except ImportError:
        pass
    print("No training metrics found. Run training first.")
    sys.exit(0)

# Build lookup: round -> metrics
by_round = {m["round"]: m for m in metrics_data}

# Print Table 9 format
print()
print("=" * 72)
print("  Table 9: Key training metrics at checkpoint rounds")
print("=" * 72)
print()
header = f"{'Metric':<22}"
for r in checkpoint_rounds:
    header += f"{'Round ' + str(r):>12}"
print(header)
print("-" * (22 + 12 * len(checkpoint_rounds)))

# Query CTC loss
row = f"{'Query CTC loss':<22}"
for r in checkpoint_rounds:
    m = by_round.get(r)
    row += f"{m['avg_query_loss']:>12.1f}" if m else f"{'—':>12}"
print(row)

# Grad norm
row = f"{'Grad norm':<22}"
for r in checkpoint_rounds:
    m = by_round.get(r)
    row += f"{m['grad_norm']:>12.1f}" if m else f"{'—':>12}"
print(row)

# Learning rate
row = f"{'Learning rate':<22}"
for r in checkpoint_rounds:
    m = by_round.get(r)
    row += f"{m.get('lr', 0):>12.2e}" if m else f"{'—':>12}"
print(row)

# Total updates
row = f"{'Total updates':<22}"
for r in checkpoint_rounds:
    m = by_round.get(r)
    row += f"{m.get('total_updates', 0):>12}" if m else f"{'—':>12}"
print(row)

print()
print(f"Total rounds completed: {max(m['round'] for m in metrics_data)}")
print(f"Final query loss: {metrics_data[-1]['avg_query_loss']:.2f}")
print(f"Final grad norm:  {metrics_data[-1]['grad_norm']:.2f}")
print()

PYEOF

# ── 4. List generated artifacts ──────────────────────────────────────────────
echo ""
log "═══════════════════════════════════════════════════════"
log "  Report generation complete"
log "═══════════════════════════════════════════════════════"
echo ""

if ls reports/analysis/*.png 2>/dev/null | head -1 > /dev/null; then
    log "Generated plots:"
    ls -1 reports/analysis/*.png | while read -r f; do
        log "  $(basename "$f")"
    done
fi

echo ""
log "View MLflow UI:  mlflow ui --backend-store-uri $MLRUNS_DIR --host 0.0.0.0 --port 5000"
echo ""
