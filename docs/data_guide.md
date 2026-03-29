# VoiceFL — Data Guide
## What we're building, what we're looking for, and the format everything must arrive at

---

## 1. The Big Picture — One Phone, One Node

The core idea: **a phone is a node**. Each user's phone holds their own voice data and
trains locally. No audio ever leaves the device. The phones collectively improve a shared
global model by sending only gradient updates (ΔW) — never raw audio, never features.

```
Phone A (your voice)          Phone B (different voice)      Phone C
│                              │                              │
│  raw audio                   │  raw audio                   │  raw audio
│     ↓                        │     ↓                        │     ↓
│  feature extraction          │  feature extraction          │  feature extraction
│     ↓                        │     ↓                        │     ↓
│  local training              │  local training              │  local training
│     ↓                        │     ↓                        │     ↓
│  ΔW (gradient update)        │  ΔW                          │  ΔW
└──────────────────────────────┴──────────────────────────────┘
                               ↓
                     Flower aggregation server
                     FedAvg(ΔW_A, ΔW_B, ΔW_C)
                               ↓
                     Updated global model W^(t+1)
                               ↓
              Broadcast back to all phones (weights only)
```

**What each phone ends up with:**
- A global backbone (shared, gets better for everyone across rounds)
- Personalization layers (local only, never transmitted, adapts to your specific voice)

In simulation (Phase 1–6), one LibriSpeech speaker = one phone.
251 speakers available → we pick 20 for the initial simulation.

---

## 2. What We Are Looking For in the Dataset

Before touching the pipeline, the exploration notebook answers five questions:

### Q1 — Is the speaker distribution usable?
We need at least **50 clips per speaker** to compute meaningful gradients.
LibriSpeech has 251 speakers with varying clip counts. The exploration shows
how many fall below the threshold and whether our 20-node selection is realistic.

**What to check:**
- Min/max/mean clips per speaker
- How many speakers have < 50 clips (these are ineligible)
- Whether clip counts are heavily skewed (a few speakers with hundreds, many with ~50)

### Q2 — Are the speakers actually different from each other?
Federated learning is non-trivial because nodes have **non-IID data** — different
distributions. If all speakers had identical speech patterns, FL would behave like
centralized training and lose its research interest.

**What to check:**
- `duration_std` per speaker (our heterogeneity proxy): high std = variable speech patterns
- Vocabulary richness per speaker (unique/total words in transcriptions)
- We want high variance *across* speakers on both metrics

### Q3 — Is the audio clean enough to train on?
LibriSpeech is read speech from audiobooks, so it should be very clean.
But we verify rather than assume.

**What to check:**
- Silence fraction (> 50% silence = unusable clip)
- Clipping fraction (> 1% at max amplitude = distorted)
- Duration outliers (< 1s = too short for meaningful features; > 30s = uncommon)
- Sample rate uniformity (must be exactly 16000 Hz — LibriSpeech guarantees this)

### Q4 — Does the feature extraction look right?
The mel spectrogram is the "image" the ASR model sees. Before running the full pipeline
we preview it on one clip to confirm the shape, value range, and visual structure.

**What to check:**
- Shape is `(T, 80)` with T proportional to clip duration
- Values are in roughly `[-80, 0]` dB range (log energy)
- The spectrogram shows clear formant structure (dark/bright bands = phonemes)

### Q5 — Does the speaker selection strategy make sense?
We use stratified sampling on `duration_std` to pick 20 speakers spread across
the full diversity spectrum. The notebook previews which 20 would be chosen so
you can override the strategy before it gets locked in by `download.py`.

**What to check:**
- The 20 selected speakers appear spread across the scatter plot (not all clustered)
- No selected speaker has an obviously anomalous clip count
- The selection covers both low-std (consistent) and high-std (variable) speakers

---

## 3. TorchCodec — What It Is and How to Use It

### What it is

TorchCodec is a PyTorch-native audio/video decoder developed by Meta. It wraps
FFmpeg and exposes audio/video decoding directly as PyTorch tensors. The key
advantage over `librosa` or `soundfile` is that it returns `torch.Tensor`
natively — no numpy intermediary — and integrates cleanly into a PyTorch data
pipeline.

### Why it appears in this project

`datasets` (HuggingFace) versions ≥ 3.x switched their default audio backend
from `soundfile` to `torchcodec`. When you call `load_dataset(...)` and iterate
rows with audio, `datasets` internally calls TorchCodec to decode the FLAC files
in the LibriSpeech cache.

**The current issue:** TorchCodec requires `libnvrtc.so.13` to load its CUDA
extension, which is not present in this environment (PyTorch 2.6.0+cu124 ships
a different CUDA runtime version). The workaround is:

```python
import os
os.environ.setdefault("DATASETS_AUDIO_BACKEND", "soundfile")  # before importing datasets
```

This tells `datasets` to use `soundfile` instead. All pipeline scripts already
include this line.

### How to use TorchCodec directly (when it works)

TorchCodec is most useful when you want to load audio files **directly** from
disk — bypassing HuggingFace datasets entirely. This is relevant if you later
want to load audio from a phone's local storage in a production node container.

```python
from torchcodec.decoders import AudioDecoder

# Open a FLAC or MP3 or WAV file
decoder = AudioDecoder("path/to/clip.flac")

# Inspect metadata before decoding
meta = decoder.metadata
print(f"Sample rate:  {meta.sample_rate} Hz")
print(f"Duration:     {meta.duration_seconds:.2f}s")
print(f"Num channels: {meta.num_channels}")

# Decode the entire file into a tensor
samples = decoder.get_all_samples()
# samples.data        → torch.Tensor  shape: (num_channels, num_samples)
# samples.pts_seconds → torch.Tensor  timestamps for each sample

audio = samples.data  # (channels, samples)

# Collapse to mono if stereo
if audio.shape[0] > 1:
    audio = audio.mean(dim=0, keepdim=True)  # (1, samples)

audio = audio.squeeze(0)  # (num_samples,)  — 1D float32 tensor
```

### TorchCodec vs librosa — when to use which

| Situation | Tool | Reason |
|---|---|---|
| Loading audio files directly in a PyTorch DataLoader | TorchCodec | Returns tensors natively, integrates with pin_memory/DataLoader |
| Quick exploratory analysis (notebooks) | librosa | Mature API, easy to use, soundfile-backed |
| Feature extraction in the pipeline | librosa | `melspectrogram` + `power_to_db` is battle-tested for ASR |
| Production node on a phone (no FFmpeg) | torchaudio | Lighter dependency, mobile-compatible backends |

In **Phase 1** we use `librosa` for feature extraction and `soundfile` as the
HuggingFace audio backend. TorchCodec will be revisited when the CUDA environment
is resolved or when we move to a production container where FFmpeg is properly
installed.

### Fixing TorchCodec if you want to use it

The root cause is a mismatch between the TorchCodec `.so` files and the CUDA
runtime in the current environment. Options:

```bash
# Option A: Reinstall TorchCodec matching your PyTorch version
pip install torchcodec==0.1.0  # check compatibility table at github.com/pytorch/torchcodec

# Option B: Install the CPU-only TorchCodec (no CUDA needed)
pip install torchcodec --index-url https://download.pytorch.org/whl/cpu

# Option C: Keep using soundfile (what we do now — zero issues)
os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"
```

---

## 4. Target Data Format

This is the format everything must arrive at before FL training can begin.
Every step in the pipeline (pii_masking → features → partition) exists to
produce this and nothing else.

### Per-node directory layout

```
data/nodes/
└── node_001/
    ├── features.pt       ← the actual training data
    ├── labels.txt        ← transcriptions for WER evaluation
    └── metadata.json     ← node statistics, no PII
```

### features.pt — the core artifact

A Python `list` of `torch.float32` tensors saved with `torch.save()`.

```python
features = torch.load("data/nodes/node_001/features.pt", weights_only=True)

# features is a list — NOT a padded batch tensor
# Each element is one clip's log-mel filterbank

len(features)       # e.g. 113  (clip count for this node)
features[0].shape   # e.g. torch.Size([487, 80])
features[0].dtype   # torch.float32
```

**Shape of each tensor: `(T, 80)`**

| Dimension | Meaning | Value |
|---|---|---|
| `T` | Time frames | Varies — proportional to clip duration |
| `80` | Mel frequency bins | Fixed — always 80 |

**Why variable-length?** We do NOT pad or stack. Each clip has a different
duration, so T varies. Padding would introduce silence artifacts and waste
memory. The FL training loop handles variable-length sequences using masking
or packing.

**Duration → T relationship:**

```
T = ceil(num_samples / hop_length)
  = ceil(duration_seconds × sample_rate / hop_length)
  = ceil(duration_seconds × 16000 / 160)
  = ceil(duration_seconds × 100)

Example:
  1.0s clip  →  T ≈ 100 frames
  3.5s clip  →  T ≈ 350 frames
  8.2s clip  →  T ≈ 820 frames
```

**Value range:** `[-80, 0]` dB approximately. Log energy relative to the
clip's peak. The mel spectrogram is computed then converted to dB with
`librosa.power_to_db(mel, ref=np.max)`.

### Feature extraction parameters (fixed for the entire project)

```python
N_MELS      = 80      # mel frequency bins
N_FFT       = 400     # analysis window: 400 samples = 25ms at 16kHz
HOP_LENGTH  = 160     # frame shift:    160 samples = 10ms at 16kHz
SAMPLE_RATE = 16000   # Hz — LibriSpeech is always 16kHz
F_MIN       = 0.0     # Hz
F_MAX       = 8000.0  # Hz (Nyquist of 16kHz signal)
```

These match Whisper's preprocessing spec exactly. Do not change them —
if you change them, the seed model and FL model no longer share a feature
space and training will fail silently.

### How features.pt is produced

```python
import librosa
import numpy as np
import torch

def extract_log_mel(audio: np.ndarray) -> torch.Tensor:
    # 1. Normalize to [-1, 1]
    audio = audio / (np.max(np.abs(audio)) + 1e-8)

    # 2. Extract mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio, sr=16000, n_mels=80, n_fft=400,
        hop_length=160, fmin=0.0, fmax=8000.0
    )

    # 3. Convert to log scale (dB)
    log_mel = librosa.power_to_db(mel, ref=np.max)  # shape: (80, T)

    # 4. Transpose to time-first convention
    log_mel = log_mel.T                              # shape: (T, 80)

    return torch.tensor(log_mel, dtype=torch.float32)
```

### labels.txt — transcriptions

Plain text file, one line per clip, same order as `features.pt`.

```
AND HE FELT A GREAT TENDERNESS FOR THE GIRL
WHEN SHE LOOKED UP AND MET HIS EYES
THE ROAD WOUND THROUGH THE DARK PINES
...
```

Used for WER computation during evaluation. Uppercase, no punctuation
(LibriSpeech standard format).

### metadata.json — node statistics

```json
{
  "node_id": "node_001",
  "anon_hash": "a3f9c2d1e8b47...",
  "clip_count": 113,
  "total_duration_s": 504.3,
  "mean_duration_s": 4.46
}
```

**Critically absent:** `speaker_id`, `chapter_id`, `file`, `id`.
These are stripped in `pii_masking.py` and never written to disk.

### Loading data in the FL training loop (Phase 3+)

```python
import torch
from torch.utils.data import Dataset

class NodeDataset(Dataset):
    def __init__(self, node_id: str):
        self.features = torch.load(
            f"data/nodes/{node_id}/features.pt",
            weights_only=True
        )
        with open(f"data/nodes/{node_id}/labels.txt") as f:
            self.labels = f.read().strip().split("\n")
        assert len(self.features) == len(self.labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
        # Returns: (T, 80) float32 tensor, str transcription

# Usage
dataset = NodeDataset("node_001")
print(f"Clips: {len(dataset)}")
print(f"First clip shape: {dataset[0][0].shape}")   # e.g. torch.Size([487, 80])
print(f"First label:      {dataset[0][1]}")
```

For batching variable-length sequences, use `collate_fn` with padding:

```python
def collate_fn(batch):
    features, labels = zip(*batch)
    # Pad to longest sequence in batch
    lengths = torch.tensor([f.shape[0] for f in features])
    padded  = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
    # padded shape: (batch_size, T_max, 80)
    return padded, lengths, list(labels)
```

---

## 5. End-to-End Data Flow Summary

```
LibriSpeech (HuggingFace cache)
  │
  │  load_dataset(..., cache_dir="data")
  │  Backend: soundfile (not torchcodec)
  ↓
Raw clips: {audio: np.float32[], text: str, speaker_id: int, ...}
  │
  │  pii_masking.py
  │  — filter to 20 selected speakers
  │  — strip speaker_id, chapter_id, file, id
  │  — apply cleaning config (min/max duration, silence)
  ↓
Stripped clips per node: {audio: np.float32[], text: str, duration_s: float}
  │  (saved as raw_clips.pkl — TEMPORARY)
  │
  │  features.py
  │  — normalize audio to [-1, 1]
  │  — librosa.melspectrogram → power_to_db → transpose
  │  — DELETE raw_clips.pkl
  ↓
data/nodes/node_XXX/
  ├── features.pt   list of (T_i, 80) torch.float32 tensors
  ├── labels.txt    one transcription per line
  └── metadata.json node stats, no PII
  │
  │  partition.py
  │  — validate shapes, dtypes, label counts
  │  — verify no raw audio remains
  ↓
data/partition_manifest.json
  ready_for_fl: true
```

---

*docs/data_guide.md — VoiceFL Phase 1 reference*
