"""
evaluation/ablation.py — Ablation study framework

Primary comparisons:
  1. full MAML vs FOMAML vs Reptile
     → validates second-order computation matters

  2. k=1 vs k=3 vs k=5
     → adaptation step sensitivity

  3. ε=2 vs ε=4 vs ε=8 vs no DP
     → privacy-utility tradeoff

  4. ANIL vs full model adaptation
     → confirms ANIL (lm_head only) is sufficient

Results saved to evaluation/results/ablation.json
"""

import json
import os
from typing import Any, Dict, List

ABLATION_MATRIX = {
    "modes": ["full", "fomaml", "reptile"],
    "k_values": [1, 3, 5],
    "epsilon": [2.0, 4.0, 8.0, None],  # None = no DP
    "adaptation": ["anil", "full"],
}

RESULTS_DIR = "evaluation/results"


def run_ablation(base_config_path: str = "configs/experiment.yaml") -> Dict[str, Any]:
    """
    Run key ablation comparisons from the ablation matrix.

    Each combination is run as a separate simulation.
    Results are collected and saved to ablation.json.

    For full ablation on A100:
      python evaluation/ablation.py

    For quick partial ablation (FOMAML, k=3, no DP):
      python evaluation/ablation.py --quick
    """
    import yaml

    with open(base_config_path) as f:
        base_cfg = yaml.safe_load(f)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results: Dict[str, Any] = {"runs": [], "matrix": ABLATION_MATRIX}

    # Ablation 1: MAML mode comparison (k fixed at 3, no DP)
    for mode in ABLATION_MATRIX["modes"]:
        cfg = _patch_config(base_cfg, {
            "maml.mode": mode,
            "maml.k": 3,
            "privacy.enabled": False,
        })
        result = _run_one(cfg, label=f"mode={mode}")
        results["runs"].append(result)

    # Ablation 2: Adaptation step sensitivity (FOMAML, no DP)
    for k in ABLATION_MATRIX["k_values"]:
        cfg = _patch_config(base_cfg, {
            "maml.mode": "fomaml",
            "maml.k": k,
            "privacy.enabled": False,
        })
        result = _run_one(cfg, label=f"k={k}")
        results["runs"].append(result)

    # Ablation 3: Privacy-utility tradeoff (FOMAML, k=3)
    for eps in ABLATION_MATRIX["epsilon"]:
        cfg = _patch_config(base_cfg, {
            "maml.mode": "fomaml",
            "maml.k": 3,
            "privacy.enabled": eps is not None,
            "privacy.epsilon": eps if eps is not None else 8.0,
        })
        label = f"epsilon={eps}" if eps is not None else "no_dp"
        result = _run_one(cfg, label=label)
        results["runs"].append(result)

    # Ablation 4: ANIL vs full adaptation (FOMAML, k=3, no DP)
    for adaptation in ABLATION_MATRIX["adaptation"]:
        cfg = _patch_config(base_cfg, {
            "maml.mode": "fomaml",
            "maml.k": 3,
            "maml.adaptation_mode": adaptation,
            "privacy.enabled": False,
        })
        result = _run_one(cfg, label=f"adaptation={adaptation}")
        results["runs"].append(result)

    output_path = os.path.join(RESULTS_DIR, "ablation.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Ablation results saved: {output_path}")
    return results


def _patch_config(base: Dict, patches: Dict[str, Any]) -> Dict:
    """Apply dot-notation patches to a config dict."""
    import copy
    cfg = copy.deepcopy(base)
    for key, val in patches.items():
        parts = key.split(".")
        d = cfg
        for part in parts[:-1]:
            d = d[part]
        d[parts[-1]] = val
    return cfg


def _run_one(cfg: Dict, label: str) -> Dict:
    """
    Run one simulation configuration and return summary metrics.
    Stub: replace with actual simulation call when running full ablation.
    """
    print(f"[ablation] {label} — stub (implement with simulation_maml.run_simulation)")
    return {
        "label": label,
        "config": {
            "mode": cfg["maml"]["mode"],
            "k": cfg["maml"]["k"],
            "dp": cfg["privacy"]["enabled"],
            "epsilon": cfg["privacy"].get("epsilon"),
        },
        "mean_wer_k3": None,
        "mean_adaptation_gain": None,
        "status": "stub",
    }
