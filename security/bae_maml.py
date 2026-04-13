"""
security/bae_maml.py — Behavioral Analysis Engine for MAML meta-gradients

4-layer anomaly detection for meta-gradient screening in Per-FedAvg.

Runs synchronously in PerFedAvgStrategy.aggregate_fit() BEFORE
meta-gradients are averaged. Anomalous nodes get reduced or zero weight.

Layer 1: Cosine similarity to round median gradient
  — detects directional outliers (Byzantine, sign-flipping attacks)

Layer 2: L2 norm vs historical distribution (3σ bound)
  — detects magnitude outliers (gradient amplification, poisoning)

Layer 3: Temporal behavioral fingerprint per node
  — detects sudden behavioral shifts (delayed poisoning, Sybil takeover)

Layer 4: IsolationForest joint anomaly score
  — combines all features for holistic anomaly detection

Response tiers:
  score < soft_threshold:      weight = 1.0  (normal)
  soft ≤ score < quarantine:   weight = 1 - score  (soft penalty)
  quarantine ≤ score < hard:   weight = 0.0, node quarantined
  score ≥ hard_threshold:      weight = 0.0, node permanently excluded

Audit log: security/audit_log.jsonl  (one JSON line per anomalous event)
"""

import datetime as dt
import json
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from sklearn.ensemble import IsolationForest


@dataclass
class BAEConfig:
    cosine_threshold: float = 0.3
    norm_sigma: float = 3.0
    history_len: int = 10
    contamination: float = 0.1
    soft_threshold: float = 0.6
    quarantine_threshold: float = 0.8
    hard_threshold: float = 0.95


class BehavioralAnalysisEngine:
    """
    4-layer meta-gradient screening for Wav2Vec2 MAML federation.

    Call screen_updates() once per FL round in PerFedAvgStrategy.aggregate_fit()
    before computing the weighted average of meta-gradients.
    """

    def __init__(self, config: BAEConfig) -> None:
        self.config = config
        self.history: Dict[str, deque] = {}
        self.if_model = IsolationForest(
            contamination=config.contamination,
            random_state=42,
            n_estimators=100,
        )
        self.if_fitted = False
        self.quarantined: set = set()
        self.excluded: set = set()
        os.makedirs("security", exist_ok=True)

    def screen_updates(
        self,
        updates: Dict[str, List[np.ndarray]],
        round_num: int,
    ) -> Dict[str, float]:
        """
        Screen all meta-gradients. Return weight per node (0.0 = excluded).

        updates: node_id → list of numpy arrays (meta-gradient)
        Returns: node_id → float weight ∈ [0.0, 1.0]
        """
        flat_grads: Dict[str, np.ndarray] = {}
        for nid, grads in updates.items():
            if nid not in self.excluded:
                flat_grads[nid] = np.concatenate([g.flatten() for g in grads])

        if not flat_grads:
            return {nid: 0.0 for nid in updates}

        median = np.median(list(flat_grads.values()), axis=0)

        features: Dict[str, List[float]] = {}
        for nid, flat in flat_grads.items():
            f1 = self._cosine_anomaly(flat, median)
            f2 = self._norm_feature(nid, flat)
            f3 = self._temporal_feature(nid, f1)
            self._update_history(nid, flat, f1)
            features[nid] = [f1, f2, f3]

        scores: Dict[str, float] = (
            self._if_scores(features)
            if len(features) >= 4
            else {nid: f[0] for nid, f in features.items()}
        )

        weights: Dict[str, float] = {}
        for nid in updates:
            if nid in self.excluded:
                weights[nid] = 0.0
                continue
            if nid not in scores:
                weights[nid] = 1.0
                continue

            s = scores[nid]
            if s >= self.config.hard_threshold:
                self.excluded.add(nid)
                weights[nid] = 0.0
                self._log(round_num, nid, s, "EXCLUDED")
            elif s >= self.config.quarantine_threshold:
                self.quarantined.add(nid)
                weights[nid] = 0.0
                self._log(round_num, nid, s, "QUARANTINED")
            elif s >= self.config.soft_threshold:
                weights[nid] = 1.0 - s
                self._log(round_num, nid, s, "SOFT_PENALTY")
            else:
                weights[nid] = 1.0

        return weights

    def _cosine_anomaly(self, flat: np.ndarray, median: np.ndarray) -> float:
        """Layer 1: cosine distance from round median (0 = aligned, 1 = opposite)."""
        dot = np.dot(flat, median)
        norm_prod = np.linalg.norm(flat) * np.linalg.norm(median) + 1e-8
        cos = dot / norm_prod
        return float(1.0 - max(cos, 0))

    def _norm_feature(self, nid: str, flat: np.ndarray) -> float:
        """Layer 2: L2 norm z-score vs node history, normalized to [0, 1]."""
        norm = float(np.linalg.norm(flat))
        hist = self.history.get(nid, deque())
        if len(hist) < 3:
            return 0.0
        past_norms = [h["norm"] for h in hist]
        mean = np.mean(past_norms)
        std = np.std(past_norms) + 1e-8
        z = abs(norm - mean) / std
        return float(min(z / self.config.norm_sigma, 1.0))

    def _temporal_feature(self, nid: str, current_cosine: float) -> float:
        """Layer 3: deviation from recent cosine anomaly trend."""
        hist = self.history.get(nid, deque())
        if len(hist) < 3:
            return 0.0
        recent = [h["cosine"] for h in list(hist)[-3:]]
        return float(abs(current_cosine - np.mean(recent)))

    def _update_history(self, nid: str, flat: np.ndarray, cosine: float) -> None:
        if nid not in self.history:
            self.history[nid] = deque(maxlen=self.config.history_len)
        self.history[nid].append({
            "norm": float(np.linalg.norm(flat)),
            "cosine": cosine,
        })

    def _if_scores(self, features: Dict[str, List[float]]) -> Dict[str, float]:
        """Layer 4: IsolationForest joint anomaly score, normalized to [0, 1]."""
        X = np.array(list(features.values()))
        nids = list(features.keys())
        self.if_model.fit(X)
        raw = self.if_model.score_samples(X)  # more negative = more anomalous
        rng = raw.max() - raw.min() + 1e-8
        normalized = 1.0 - (raw - raw.min()) / rng  # flip: 1 = most anomalous
        return dict(zip(nids, normalized.tolist()))

    def _log(self, round_num: int, nid: str, score: float, action: str) -> None:
        event = {
            "round": round_num,
            "node": nid,
            "score": round(score, 4),
            "action": action,
            "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        with open("security/audit_log.jsonl", "a") as f:
            f.write(json.dumps(event) + "\n")
