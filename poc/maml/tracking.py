"""
MLflow experiment tracking for FedLoRA-MAML training runs.

Wraps mlflow calls so the rest of the codebase doesn't need to know whether
mlflow is installed. If mlflow is missing, all calls become no-ops.

Usage:
    from maml.tracking import Tracker

    tracker = Tracker(experiment_name="fedlora_maml_vctk", run_name="round20_k5")
    tracker.log_params({"mode": "lora_maml", "inner_steps": 5, ...})

    for rnd in ...:
        tracker.log_metrics({"train/loss": ..., "train/grad_norm": ...}, step=rnd)

    tracker.log_gate_results(gate_results, inner_steps=5)
    tracker.log_artifact(checkpoint_path)
    tracker.end()
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

try:
    import mlflow

    _MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None                            
    _MLFLOW_AVAILABLE = False

class Tracker:
    """MLflow experiment tracker. All methods are no-ops if mlflow is unavailable."""

    def __init__(
        self,
        experiment_name: str = "fedlora_maml",
        run_name: str | None = None,
        tracking_uri: str = "mlruns",
    ) -> None:
        self._active = False

        if not _MLFLOW_AVAILABLE:
            warnings.warn(
                "mlflow not installed — experiment tracking disabled. "
                "Install with: pip install mlflow",
                stacklevel=2,
            )
            return

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._run = mlflow.start_run(run_name=run_name)
        self._active = True
        print(f"MLflow tracking: experiment='{experiment_name}'  run_id={self._run.info.run_id}")
        print(f"  View at: mlflow ui --backend-store-uri {tracking_uri}")

    def log_params(self, params: dict[str, Any]) -> None:
        if not self._active:
            return
                                                                        
        flat = {k: str(v)[:500] for k, v in params.items()}
        mlflow.log_params(flat)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if not self._active:
            return
        mlflow.log_metrics(metrics, step=step)

    def log_gate_results(self, gate_results: dict, inner_steps: int) -> None:
        """Log per-speaker WER gate results as MLflow metrics."""
        if not self._active:
            return
        passed_count = sum(1 for v in gate_results.values() if v.get("passed"))
        total = len(gate_results)
        mlflow.log_metrics(
            {"gate/passed": passed_count, "gate/total": total, "gate/pass_rate": passed_count / max(total, 1)}
        )
        for spk_id, result in gate_results.items():
            tag = spk_id[:8]
            mlflow.log_metrics(
                {
                    f"gate/{tag}/wer_k0": result.get("wer_k0", float("nan")),
                    f"gate/{tag}/wer_k{inner_steps}": result.get("wer_k_adapted", float("nan")),
                    f"gate/{tag}/improved": float(result.get("passed", False)),
                }
            )

    def log_artifact(self, path: str | Path) -> None:
        if not self._active:
            return
        p = Path(path)
        if p.exists():
            mlflow.log_artifact(str(p))

    def log_training_summary(self, round_metrics: list[dict]) -> None:
        """Log final training summary stats from the round_metrics list."""
        if not self._active or not round_metrics:
            return
        last = round_metrics[-1]
        mlflow.log_metrics(
            {
                "summary/final_loss": last.get("avg_query_loss", float("nan")),
                "summary/final_grad_norm": last.get("grad_norm", float("nan")),
                "summary/total_updates": last.get("total_updates", 0),
                "summary/rounds_completed": len(round_metrics),
            }
        )

    def end(self) -> None:
        if not self._active:
            return
        mlflow.end_run()
        self._active = False
        print("MLflow run ended.")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.end()
