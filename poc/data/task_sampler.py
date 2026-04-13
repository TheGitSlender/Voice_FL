"""
VoiceTaskSampler — draws K-shot tasks from a node's feature store.

A task is:
  support_audio:  list of K float32 tensors (1D, variable length)
  support_labels: list of K str
  query_audio:    list of Q float32 tensors
  query_labels:   list of Q str

Invariant I5: support ∩ query indices = ∅. Asserted on every call.
"""

import random
from pathlib import Path
from typing import NamedTuple

import torch


class Task(NamedTuple):
    support_audio: list  # list of (T_i,) float32 tensors
    support_labels: list  # list of str
    query_audio: list    # list of (T_j,) float32 tensors
    query_labels: list   # list of str


class VoiceTaskSampler:
    def __init__(
        self,
        node_dir: str | Path,
        support_size: int = 8,
        query_size: int = 8,
        seed: int | None = None,
    ):
        self.node_dir = Path(node_dir)
        self.support_size = support_size
        self.query_size = query_size
        self._rng = random.Random(seed)

        features_path = self.node_dir / "features.pt"
        labels_path = self.node_dir / "labels.txt"

        if not features_path.exists():
            raise FileNotFoundError(f"features.pt not found in {self.node_dir}")
        if not labels_path.exists():
            raise FileNotFoundError(f"labels.txt not found in {self.node_dir}")

        self._features: list = torch.load(features_path, weights_only=True)
        self._labels: list = labels_path.read_text().strip().splitlines()

        if len(self._features) != len(self._labels):
            raise ValueError(
                f"features/labels mismatch: {len(self._features)} vs {len(self._labels)}"
            )

        min_needed = support_size + query_size
        if len(self._features) < min_needed:
            raise ValueError(
                f"Node has only {len(self._features)} clips; "
                f"need at least {min_needed} (K={support_size}+Q={query_size})"
            )

    @property
    def num_clips(self) -> int:
        return len(self._features)

    def sample_task(self) -> Task:
        """Draw a fresh K-shot task. Support ∩ query = ∅ (asserted)."""
        total = len(self._features)
        indices = self._rng.sample(range(total), self.support_size + self.query_size)
        support_idx = indices[: self.support_size]
        query_idx = indices[self.support_size :]

        # Invariant I5
        overlap = set(support_idx) & set(query_idx)
        assert not overlap, f"INVARIANT VIOLATION I5: support∩query={overlap}"

        return Task(
            support_audio=[self._features[i] for i in support_idx],
            support_labels=[self._labels[i] for i in support_idx],
            query_audio=[self._features[i] for i in query_idx],
            query_labels=[self._labels[i] for i in query_idx],
        )


def _smoke_test(node_dir: str) -> None:
    """Quick smoke test — run as __main__."""
    sampler = VoiceTaskSampler(node_dir, support_size=8, query_size=8)
    print(f"Node clips: {sampler.num_clips}")
    for trial in range(3):
        task = sampler.sample_task()
        assert len(task.support_audio) == 8
        assert len(task.query_audio) == 8
        assert len(task.support_labels) == 8
        assert len(task.query_labels) == 8
        assert all(isinstance(t, torch.Tensor) for t in task.support_audio)
        assert all(isinstance(t, torch.Tensor) for t in task.query_audio)
        print(
            f"  trial {trial}: support[0].shape={task.support_audio[0].shape}, "
            f"label='{task.support_labels[0][:30]}'"
        )
    print("Smoke test PASSED")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    nodes_dir = Path(__file__).parent / "nodes"
    dirs = sorted(d for d in nodes_dir.iterdir() if d.is_dir())
    if not dirs:
        print("No node directories found. Run the data pipeline first.")
        sys.exit(1)
    _smoke_test(str(dirs[0]))
