"""Custom differentiable CTC loss supporting double-backward (second-order MAML)."""

from ctc.differentiable_ctc import DifferentiableCTC, ctc_loss_differentiable

__all__ = ["DifferentiableCTC", "ctc_loss_differentiable"]
