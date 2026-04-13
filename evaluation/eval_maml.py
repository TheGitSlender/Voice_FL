"""
evaluation/eval_maml.py — Full evaluation protocol for Per-FedAvg MAML

Computes:
  1. Per-node WER at k=0,1,3,5,10 (adaptation curves)
  2. Aggregate stats (mean, std, min, max) across nodes at each k
  3. Adaptation gain per node: WER(k=0) - WER(k=3)

Results saved to evaluation/results/eval_results.json
All metrics logged to the active MLflow run when available.

Run:
    python evaluation/eval_maml.py
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlflow
import numpy as np

from data.task_sampler import VoiceTaskSampler
from maml.engine import MAMLEngine
from maml.meta_eval import evaluate_adaptation_at_k
from models.wav2vec2_maml import Wav2Vec2MAML

RESULTS_DIR = "evaluation/results"


def run_full_evaluation(
    model: Wav2Vec2MAML,
    engine: MAMLEngine,
    task_samplers: List[VoiceTaskSampler],
    k_values: Tuple[int, ...] = (0, 1, 3, 5, 10),
    save_dir: str = RESULTS_DIR,
    mlflow_run=None,
) -> Dict:
    """
    Full evaluation protocol across all nodes.

    For each node:
      - evaluate_adaptation_at_k() produces WER at each k
      - Result: adaptation curve showing personalization improvement

    Aggregates across nodes and logs to MLflow.

    Returns the full results dict (also saved to eval_results.json).
    """
    os.makedirs(save_dir, exist_ok=True)

    results: Dict = {
        "per_node": {},
        "aggregate": {},
        "metadata": {
            "k_values": list(k_values),
            "n_nodes": len(task_samplers),
            "maml_mode": engine.config.mode,
            "k_inner": engine.config.k,
        },
    }

    # Per-node adaptation curves
    for sampler in task_samplers:
        node_id = Path(sampler.node_dir).name
        wers = evaluate_adaptation_at_k(model, engine, sampler, list(k_values))
        results["per_node"][node_id] = wers
        gain = wers.get("k=0", 1.0) - wers.get(f"k={engine.config.k}", 1.0)
        results["per_node"][node_id]["adaptation_gain"] = round(gain, 4)
        print(f"  {node_id}: " + "  ".join(f"{k}: {v:.3f}" for k, v in wers.items()))

    # Aggregate across nodes
    for k in k_values:
        key = f"k={k}"
        vals = [results["per_node"][n][key] for n in results["per_node"]]
        results["aggregate"][key] = {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
            "min": round(float(np.min(vals)), 4),
            "max": round(float(np.max(vals)), 4),
        }

    # Adaptation gains
    gains = [results["per_node"][n]["adaptation_gain"] for n in results["per_node"]]
    results["aggregate"]["adaptation_gain"] = {
        "mean": round(float(np.mean(gains)), 4),
        "nodes_positive": int(sum(1 for g in gains if g > 0)),
        "nodes_total": len(gains),
    }

    # Save results
    output_path = os.path.join(save_dir, "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_path}")

    # Log to MLflow
    if mlflow_run is not None:
        for k in k_values:
            mlflow.log_metric(
                f"eval/mean_wer_k{k}",
                results["aggregate"][f"k={k}"]["mean"],
            )
        mlflow.log_metric(
            "eval/mean_adaptation_gain",
            results["aggregate"]["adaptation_gain"]["mean"],
        )
        mlflow.log_metric(
            "eval/nodes_with_positive_gain",
            results["aggregate"]["adaptation_gain"]["nodes_positive"],
        )

    return results
