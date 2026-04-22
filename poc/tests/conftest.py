"""Shared fixtures for CTC + wav2vec2 MAML tests."""

from __future__ import annotations

import pytest
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

WAV2VEC2_MODEL = "facebook/wav2vec2-base-960h"
VOCAB_SIZE = 32
BLANK = 0
SAMPLE_RATE = 16000
                                                         
DOWNSAMPLE_RATIO = 326.5

@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture(scope="session")
def processor() -> Wav2Vec2Processor:
    return Wav2Vec2Processor.from_pretrained(WAV2VEC2_MODEL)

@pytest.fixture(scope="session")
def wav2vec2_model(device: torch.device) -> Wav2Vec2ForCTC:
    """Shared wav2vec2 model — DO NOT mutate weights in tests.

    Tests that need to modify weights must deepcopy first.
    """
    model = Wav2Vec2ForCTC.from_pretrained(WAV2VEC2_MODEL)
    return model.to(device=device, dtype=torch.float32)

def tokenize(text: str, processor: Wav2Vec2Processor) -> torch.Tensor:
    """Tokenize text to label ids (no batch dim)."""
    return processor.tokenizer(text, return_tensors="pt").input_ids[0]

def make_audio(duration_ms: int, device: torch.device) -> torch.Tensor:
    """Create random audio at 16kHz."""
    n_samples = int(SAMPLE_RATE * duration_ms / 1000)
    return torch.randn(1, n_samples, device=device, dtype=torch.float32)
