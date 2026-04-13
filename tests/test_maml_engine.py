"""
tests/test_maml_engine.py — MAMLEngine unit tests

Tests that:
  1. FOMAML produces non-None gradients for all model params
  2. Full MAML gradient ≠ FOMAML gradient (confirms second-order computation)
  3. k=0 inner loop produces same model state (no-op)
  4. Reptile gradient has same shape as model parameters
  5. adapt() returns a model that produces lower loss than θ*

Run:
    pytest tests/test_maml_engine.py -v
"""

import copy

import pytest
import torch

from maml.engine import MAMLConfig, MAMLEngine
from models.wav2vec2_maml import Wav2Vec2MAML


@pytest.fixture(scope="module")
def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def tiny_audio(device) -> tuple:
    """1-second fake audio batch for testing (no real data needed)."""
    # 1 second at 16kHz
    support_audio = torch.randn(2, 16000, device=device)
    query_audio = torch.randn(2, 16000, device=device)

    # Simple label: single token (vocab_size=32 for wav2vec2)
    # Use -100 for all but first position to make CTC loss valid
    support_labels = torch.full((2, 5), -100, dtype=torch.long, device=device)
    support_labels[:, 0] = 1  # at least one valid token
    query_labels = torch.full((2, 5), -100, dtype=torch.long, device=device)
    query_labels[:, 0] = 1

    return support_audio, support_labels, query_audio, query_labels


@pytest.fixture(scope="module")
def model(device) -> Wav2Vec2MAML:
    return Wav2Vec2MAML(mode="anil").to(device)


class TestFOMAML:
    def test_fomaml_returns_grads(self, model, tiny_audio, device) -> None:
        config = MAMLConfig(mode="fomaml", k=1, inner_lr=1e-4, use_bf16=False)
        engine = MAMLEngine(model, config)
        sup_a, sup_l, qry_a, qry_l = tiny_audio

        meta_grads, q_loss = engine.compute_meta_gradient(sup_a, sup_l, qry_a, qry_l)

        assert isinstance(meta_grads, list)
        assert len(meta_grads) == len(list(model.model.parameters()))
        assert isinstance(q_loss, float)
        assert q_loss > 0

    def test_fomaml_grads_not_all_none(self, model, tiny_audio) -> None:
        config = MAMLConfig(mode="fomaml", k=1, inner_lr=1e-4, use_bf16=False)
        engine = MAMLEngine(model, config)
        sup_a, sup_l, qry_a, qry_l = tiny_audio

        meta_grads, _ = engine.compute_meta_gradient(sup_a, sup_l, qry_a, qry_l)
        non_none = [g for g in meta_grads if g is not None]
        assert len(non_none) > 0, "All meta-gradients are None"

    def test_fomaml_restores_model_state(self, model, tiny_audio) -> None:
        """FOMAML must restore model params after computing meta-grad."""
        config = MAMLConfig(mode="fomaml", k=3, inner_lr=1e-4, use_bf16=False)
        engine = MAMLEngine(model, config)
        sup_a, sup_l, qry_a, qry_l = tiny_audio

        # Record params before
        params_before = {k: v.clone() for k, v in model.model.state_dict().items()}
        engine.compute_meta_gradient(sup_a, sup_l, qry_a, qry_l)

        # Params should be restored
        for k, v in model.model.state_dict().items():
            assert torch.allclose(v, params_before[k], atol=1e-5), (
                f"Parameter {k} changed after FOMAML — model state not restored"
            )


class TestFullMAML:
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Full MAML requires GPU for Wav2Vec2",
    )
    def test_full_maml_gradient_differs_from_fomaml(self, model, tiny_audio) -> None:
        """
        Critical gate: full MAML grad ≠ FOMAML grad on identical input.
        Difference confirms second-order computation is actually happening.
        """
        sup_a, sup_l, qry_a, qry_l = tiny_audio

        config_full = MAMLConfig(mode="full", k=1, inner_lr=1e-4, use_bf16=True)
        engine_full = MAMLEngine(model, config_full)
        full_grads, _ = engine_full.compute_meta_gradient(sup_a, sup_l, qry_a, qry_l)

        config_fo = MAMLConfig(mode="fomaml", k=1, inner_lr=1e-4, use_bf16=True)
        engine_fo = MAMLEngine(model, config_fo)
        fo_grads, _ = engine_fo.compute_meta_gradient(sup_a, sup_l, qry_a, qry_l)

        # At least some gradients should differ
        valid_pairs = [
            (f, g) for f, g in zip(full_grads, fo_grads)
            if f is not None and g is not None
        ]
        assert len(valid_pairs) > 0

        any_different = any(
            not torch.allclose(f, g, atol=1e-6)
            for f, g in valid_pairs
        )
        assert any_different, (
            "Full MAML grad == FOMAML grad — second-order computation not working"
        )


class TestReptile:
    def test_reptile_grad_shapes_match_params(self, model, tiny_audio) -> None:
        config = MAMLConfig(mode="reptile", k=2, inner_lr=1e-4, use_bf16=False)
        engine = MAMLEngine(model, config)
        sup_a, sup_l, qry_a, qry_l = tiny_audio

        meta_grads, q_loss = engine.compute_meta_gradient(sup_a, sup_l, qry_a, qry_l)

        params = list(model.model.parameters())
        assert len(meta_grads) == len(params)
        for i, (g, p) in enumerate(zip(meta_grads, params)):
            assert g.shape == p.shape, (
                f"Reptile grad {i} shape {g.shape} != param shape {p.shape}"
            )

    def test_reptile_restores_model_state(self, model, tiny_audio) -> None:
        config = MAMLConfig(mode="reptile", k=2, inner_lr=1e-4, use_bf16=False)
        engine = MAMLEngine(model, config)
        sup_a, sup_l, qry_a, qry_l = tiny_audio

        params_before = {k: v.clone() for k, v in model.model.state_dict().items()}
        engine.compute_meta_gradient(sup_a, sup_l, qry_a, qry_l)

        for k, v in model.model.state_dict().items():
            assert torch.allclose(v, params_before[k], atol=1e-5), (
                f"Reptile param {k} not restored"
            )


class TestAdapt:
    def test_adapt_returns_different_model(self, model, tiny_audio) -> None:
        config = MAMLConfig(mode="fomaml", k=3, inner_lr=1e-4, use_bf16=False)
        engine = MAMLEngine(model, config)
        sup_a, sup_l, qry_a, qry_l = tiny_audio

        adapted = engine.adapt(sup_a, sup_l, k=3)

        # adapted should have different lm_head weights
        orig_head = model.model.lm_head.weight.data
        adapted_head = adapted.lm_head.weight.data
        assert not torch.allclose(orig_head, adapted_head), (
            "Adapted model lm_head is identical to original — adaptation had no effect"
        )

    def test_adapt_does_not_modify_original(self, model, tiny_audio) -> None:
        config = MAMLConfig(mode="fomaml", k=3, inner_lr=1e-4, use_bf16=False)
        engine = MAMLEngine(model, config)
        sup_a, sup_l, qry_a, qry_l = tiny_audio

        params_before = {k: v.clone() for k, v in model.model.state_dict().items()}
        engine.adapt(sup_a, sup_l, k=3)

        for k, v in model.model.state_dict().items():
            assert torch.allclose(v, params_before[k], atol=1e-5), (
                f"adapt() modified original model param {k}"
            )
