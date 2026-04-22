"""
Gradient norm diagnostic for FOMAML-ANIL with CTC loss on VCTK data.

Run BEFORE hyperparameter search to understand gradient magnitudes.
Checks whether CTC loss is summed (norms 650-1035) or averaged (norms 1-50).
Prints a recommended inner_lr based on findings.

Gate: If norms > 100 after CTC length normalization, add 'reduction=mean'
to the CTC loss call before proceeding.

Output: data/diagnostics/gradient_norms.json

Usage:
    python data/diagnostics/run_gradient_analysis.py
    python data/diagnostics/run_gradient_analysis.py --device cuda
    python data/diagnostics/run_gradient_analysis.py --node_dir data/nodes/<hash>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

DIAG_DIR = Path(__file__).parent
DIAG_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLE_CLIPS = 10
LR_VALUES = [1e-5, 1e-4, 1e-3]
INNER_STEPS_CHECK = 3

def encode_audio(audio_tensor, processor, device, dtype):
    arr = audio_tensor.float().numpy()
    inputs = processor(arr, sampling_rate=16000, return_tensors="pt", padding=False)
    return inputs.input_values.to(device=device, dtype=dtype)

def encode_labels(text: str, processor, device):
    return processor.tokenizer(text, return_tensors="pt").input_ids.to(device)

def compute_lm_head_grad_norm(model, processor, audio, text, device) -> tuple[float, float]:
    """Return (loss_value, lm_head_gradient_norm) for a single clip."""
    import torch

    model_copy = __import__("copy").deepcopy(model.model)
    model_copy.train()
    dtype = next(model_copy.parameters()).dtype

    input_values = encode_audio(audio, processor, device, dtype)
    labels = encode_labels(text, processor, device)

    out = model_copy(input_values=input_values, labels=labels)
    lm_params = list(model_copy.lm_head.parameters())
    grads = torch.autograd.grad(out.loss, lm_params, create_graph=False)

    total_norm = float(sum(g.norm() ** 2 for g in grads) ** 0.5)
    return float(out.loss.item()), total_norm

def check_loss_trajectory(
    model,
    processor,
    support_audio,
    support_texts,
    inner_lr: float,
    k: int,
    device,
) -> list[float]:
    """Run k inner steps and return the loss at each step."""
    import copy
    import torch

    model_copy = copy.deepcopy(model.model)
    model_copy.train()
    dtype = next(model_copy.parameters()).dtype
    lm_params = list(model_copy.lm_head.parameters())

    trajectory = []
    for step in range(k):
        step_loss = 0.0
        step_grads = [torch.zeros_like(p) for p in lm_params]
        for audio, text in zip(support_audio[:4], support_texts[:4]):
            iv = encode_audio(audio, processor, device, dtype)
            lb = encode_labels(text, processor, device)
            out = model_copy(input_values=iv, labels=lb)
            clip_grads = torch.autograd.grad(out.loss, lm_params, create_graph=False)
            step_loss += out.loss.item()
            for acc, g in zip(step_grads, clip_grads):
                acc.add_(g)
            del out, clip_grads
        avg_loss = step_loss / min(4, len(support_audio))
        trajectory.append(avg_loss)
        for p, g in zip(lm_params, step_grads):
            p.data = p.data - inner_lr * g

    return trajectory

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--node_dir", default=None,
                        help="Specific node dir to use (default: first available)")
    args = parser.parse_args()

    import torch
    from models.wav2vec2_maml import Wav2Vec2MAML, load_processor
    from data.task_sampler import VoiceTaskSampler

    device = torch.device(args.device)

    nodes_dir = ROOT / "data" / "nodes"
    if args.node_dir:
        node_dir = Path(args.node_dir)
    else:
        candidates = sorted(d for d in nodes_dir.iterdir() if d.is_dir())
        if not candidates:
            print(f"ERROR: No node directories found in {nodes_dir}")
            print("Run: python data/prepare_vctk.py && python data/features.py")
            sys.exit(1)
        node_dir = candidates[0]

    features_path = node_dir / "features.pt"
    labels_path = node_dir / "labels.txt"
    if not features_path.exists() or not labels_path.exists():
        print(f"ERROR: {node_dir} missing features.pt or labels.txt")
        print("Run: python data/features.py")
        sys.exit(1)

    print(f"Using node: {node_dir.name}")
    print(f"Device: {device}")
    print("Loading model...")

    model = Wav2Vec2MAML(device=device)
    processor = load_processor()

    tensors = torch.load(features_path, weights_only=True)
    texts = labels_path.read_text().splitlines()

    clips = list(zip(tensors[:N_SAMPLE_CLIPS], texts[:N_SAMPLE_CLIPS]))
    if len(clips) < 4:
        print(f"ERROR: Need at least 4 clips, found {len(clips)}")
        sys.exit(1)

    print(f"\n--- Step 1: lm_head gradient norms ({len(clips)} clips) ---")
    losses = []
    norms = []
    for i, (audio, text) in enumerate(clips):
        loss_val, norm = compute_lm_head_grad_norm(model, processor, audio, text, device)
        losses.append(loss_val)
        norms.append(norm)
        print(f"  clip {i+1:2d}: loss={loss_val:.4f}  lm_head_grad_norm={norm:.2f}")

    import numpy as np
    mean_loss = float(np.mean(losses))
    mean_norm = float(np.mean(norms))
    std_norm = float(np.std(norms))

    print(f"\n  mean loss={mean_loss:.4f}  mean_norm={mean_norm:.2f}  std={std_norm:.2f}")

    ctc_averaged = mean_norm < 100.0
    print(f"\n  CTC loss appears {'AVERAGED' if ctc_averaged else 'SUMMED'} over time steps")
    if not ctc_averaged:
        print("  RECOMMENDATION: Add length normalization to CTC loss.")
        print("  In engine.py _accumulate_grads_over_clips, after out.loss:")
        print("    loss = out.loss / input_values.shape[-1]  # normalize by T_frames")

    print(f"\n--- Step 2: Loss trajectory per inner_lr over {INNER_STEPS_CHECK} steps ---")
    support_audio = [c[0] for c in clips[:4]]
    support_texts = [c[1] for c in clips[:4]]

    lr_results = {}
    for lr in LR_VALUES:
        trajectory = check_loss_trajectory(
            model, processor, support_audio, support_texts, lr, INNER_STEPS_CHECK, device
        )
        direction = "DECREASING" if trajectory[-1] < trajectory[0] else "DIVERGING/FLAT"
        print(f"  lr={lr:.0e}: {[f'{v:.4f}' for v in trajectory]}  [{direction}]")
        lr_results[str(lr)] = {
            "trajectory": trajectory,
            "converges": trajectory[-1] < trajectory[0],
            "direction": direction,
        }

    converging_lrs = [lr for lr in LR_VALUES if lr_results[str(lr)]["converges"]]
    if converging_lrs:
        recommended_lr = max(converging_lrs)                                   
        print(f"\n  Recommended inner_lr: {recommended_lr:.0e}")
    else:
        recommended_lr = LR_VALUES[0]                     
        print(f"\n  No lr converged cleanly — use most conservative: {recommended_lr:.0e}")
        print("  Consider adding CTC length normalization first.")

    result = {
        "node_dir": str(node_dir),
        "device": str(device),
        "n_clips": len(clips),
        "mean_loss": mean_loss,
        "mean_lm_head_grad_norm": mean_norm,
        "std_lm_head_grad_norm": std_norm,
        "ctc_loss_appears_averaged": ctc_averaged,
        "per_clip": [{"loss": l, "norm": n} for l, n in zip(losses, norms)],
        "lr_trajectory_check": lr_results,
        "recommended_inner_lr": recommended_lr,
        "recommendation": (
            "inner_lr looks reasonable" if ctc_averaged
            else "Add CTC length normalization before setting inner_lr"
        ),
    }

    out_path = DIAG_DIR / "gradient_norms.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved: {out_path}")
    print(f"\nSUMMARY: recommended inner_lr = {recommended_lr:.0e}")
    print("Update configs/poc.yaml: maml.inner_lr: {:.0e}".format(recommended_lr))

if __name__ == "__main__":
    main()
