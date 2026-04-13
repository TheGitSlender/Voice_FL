"""
data/task_sampler.py — K-shot task constructor for Wav2Vec2 MAML

Builds (support, query) task pairs from per-node feature files.
Used by MAMLEngine and MAMLClient to sample tasks each FL round.

Critical invariant (I6): support and query sets are ALWAYS different clips.
No clip ever appears in both support and query for a given task sample.

Feature format: list of 1D float32 tensors (T_samples,) at 16kHz.
Wav2Vec2Processor handles padding to a common length within each batch.
"""

import random
from typing import List, Optional, Tuple

import torch
from transformers import Wav2Vec2Processor


class VoiceTaskSampler:
    """
    K-shot task constructor for Wav2Vec2 MAML.

    Each call to sample_task() returns:
      support_audio:  Tensor (K, T_max) — padded batch of K clips
      support_labels: Tensor (K, N_tokens) — tokenized transcriptions
      query_audio:    Tensor (Q, T_max) — padded batch of Q clips
      query_labels:   Tensor (Q, N_tokens) — tokenized transcriptions

    Support and query are DIFFERENT clips drawn without replacement.
    No clip appears in both sets within a single sample_task() call.
    Clips are re-sampled each call — different sample each FL round.

    Labels use -100 for CTC padding positions (ignored by loss).
    """

    def __init__(
        self,
        node_dir: str,
        K: int = 8,
        Q: int = 8,
        processor: Optional[Wav2Vec2Processor] = None,
        device: str = "cuda",
    ) -> None:
        self.K = K
        self.Q = Q
        self.device = device
        self.node_dir = node_dir

        features_path = f"{node_dir}/features.pt"
        labels_path = f"{node_dir}/labels.txt"

        self.features: List[torch.Tensor] = torch.load(features_path, weights_only=True)

        with open(labels_path, encoding="utf-8") as f:
            self.labels: List[str] = [line.strip() for line in f if line.strip()]

        assert len(self.features) == len(self.labels), (
            f"Feature/label count mismatch in {node_dir}: "
            f"{len(self.features)} features vs {len(self.labels)} labels"
        )
        assert len(self.features) >= K + Q, (
            f"Not enough clips ({len(self.features)}) for K={K} + Q={Q} in {node_dir}"
        )

        self.total_clips = len(self.features)

        if processor is None:
            processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.processor = processor

    def sample_task(
        self,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample one K-shot task.

        Returns (support_audio, support_labels, query_audio, query_labels).

        support_audio:  (K, T_max) padded float32 on self.device
        support_labels: (K, N_tokens) int64 on self.device, -100 for padding
        query_audio:    (Q, T_max) padded float32 on self.device
        query_labels:   (Q, N_tokens) int64 on self.device, -100 for padding

        Invariant (I6): support and query clip indices never overlap.
        """
        if seed is not None:
            random.seed(seed)

        # Sample K+Q unique indices — guarantees no overlap between support and query
        indices = random.sample(range(self.total_clips), self.K + self.Q)
        support_indices = indices[: self.K]
        query_indices = indices[self.K :]

        support_audio, support_labels = self._batch(support_indices)
        query_audio, query_labels = self._batch(query_indices)

        return (
            support_audio.to(self.device),
            support_labels.to(self.device),
            query_audio.to(self.device),
            query_labels.to(self.device),
        )

    def sample_eval_batch(
        self,
        n: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample n clips for evaluation (no support/query split needed).
        Defaults to min(20, total_clips).
        """
        n = n or min(20, self.total_clips)
        indices = random.sample(range(self.total_clips), n)
        audio, labels = self._batch(indices)
        return audio.to(self.device), labels.to(self.device)

    def _batch(
        self,
        indices: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert a list of clip indices to padded batch tensors.

        Audio: Wav2Vec2Processor pads variable-length 1D arrays to a common length.
        Labels: Processor tokenizes text; padding positions set to -100 for CTC loss.
        """
        audio_arrays = [self.features[i].numpy() for i in indices]
        texts = [self.labels[i] for i in indices]

        audio_inputs = self.processor(
            audio_arrays,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        audio_batch: torch.Tensor = audio_inputs.input_values  # (n, T_max)

        # transformers >= 5.0 removed as_target_processor(); use tokenizer directly
        label_inputs = self.processor.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
        )
        label_batch: torch.Tensor = label_inputs.input_ids  # (n, N_tokens)

        # Replace pad_token_id with -100 — CTC loss ignores -100 positions
        pad_id = self.processor.tokenizer.pad_token_id
        label_batch = label_batch.masked_fill(label_batch == pad_id, -100)

        return audio_batch, label_batch


if __name__ == "__main__":
    """Quick self-test: verify sampler loads and produces non-overlapping batches."""
    import os

    node_dir = "data/nodes/node_001"
    if not os.path.exists(f"{node_dir}/features.pt"):
        print("Run: python data/features.py first")
    else:
        sampler = VoiceTaskSampler(node_dir, K=4, Q=4, device="cpu")
        print(f"Loaded {sampler.total_clips} clips from {node_dir}")

        # Run 100 samples and verify no index overlap
        for trial in range(100):
            indices = random.sample(range(sampler.total_clips), sampler.K + sampler.Q)
            support_idx = set(indices[: sampler.K])
            query_idx = set(indices[sampler.K :])
            assert len(support_idx & query_idx) == 0, f"Trial {trial}: overlap detected!"

        print("100 overlap tests passed  ✓")

        sup_a, sup_l, qry_a, qry_l = sampler.sample_task()
        print(f"support_audio:  {sup_a.shape}  dtype={sup_a.dtype}")
        print(f"support_labels: {sup_l.shape}  dtype={sup_l.dtype}")
        print(f"query_audio:    {qry_a.shape}  dtype={qry_a.dtype}")
        print(f"query_labels:   {qry_l.shape}  dtype={qry_l.dtype}")
        print("VoiceTaskSampler self-test passed  ✓")
