"""Custom differentiable CTC loss supporting double-backward (second-order MAML)."""

from ctc.differentiable_ctc import DifferentiableCTC, ctc_loss_differentiable
from ctc.differentiable_ctc_v2 import DifferentiableCTCFast, ctc_loss_differentiable_fast

__all__ = [
    "DifferentiableCTC", "ctc_loss_differentiable",
    "DifferentiableCTCFast", "ctc_loss_differentiable_fast",
]
