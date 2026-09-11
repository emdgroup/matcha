"""Tests for explainability input schemas."""

from matcha.utils.schemas.explainability import ExplainerInputModel


def _valid_params() -> dict:
    return {
        "positional_analogue_scanning_params": None,
        "nitrogen_walk_params": None,
        "lime_descriptor_set": None,
        "lime_fingerprint_params": None,
        "lime_scale_coeff": True,
        "lime_remove_noise": True,
    }


def test_reverse_defaults_on():
    model = ExplainerInputModel(**_valid_params())

    assert model.reverse_positional_analogue_scanning is True


def test_reverse_accepts_disabled_pas_vocabulary():
    model = ExplainerInputModel(
        **_valid_params(), reverse_positional_analogue_scanning=True
    )

    assert model.positional_analogue_scanning_params is None
    assert model.reverse_positional_analogue_scanning is True


def test_reverse_can_be_disabled():
    model = ExplainerInputModel(
        **_valid_params(), reverse_positional_analogue_scanning=False
    )

    assert model.reverse_positional_analogue_scanning is False
