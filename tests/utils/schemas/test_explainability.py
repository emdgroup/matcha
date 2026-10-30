"""Tests for explainability input schemas."""

import pytest
from pydantic import ValidationError

from matcha.utils.schemas.explainability import ExplainerInputModel


def _valid_params() -> dict:
    return {
        "positional_analogue_scanning_params": None,
        "nitrogen_walk_params": None,
        "lime_descriptor_set": None,
        "lime_fingerprint_params": None,
        "lime_remove_noise": True,
    }


def test_removed_lime_scale_coeff_field_is_absent():
    assert "lime_scale_coeff" not in ExplainerInputModel.model_fields


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


def test_generation_timeout_defaults_to_sixty_seconds():
    model = ExplainerInputModel(**_valid_params())

    assert model.generation_timeout == 60.0


def test_generation_timeout_accepts_zero():
    model = ExplainerInputModel(**_valid_params(), generation_timeout=0)

    assert model.generation_timeout == 0.0


@pytest.mark.parametrize("generation_timeout", [-1, float("inf"), float("nan")])
def test_generation_timeout_rejects_invalid_values(generation_timeout):
    with pytest.raises(ValidationError):
        ExplainerInputModel(**_valid_params(), generation_timeout=generation_timeout)


def test_num_sample_defaults_to_one_hundred():
    model = ExplainerInputModel(**_valid_params())

    assert model.num_sample == 100


def test_num_sample_accepts_zero():
    model = ExplainerInputModel(**_valid_params(), num_sample=0)

    assert model.num_sample == 0


@pytest.mark.parametrize("num_sample", [-1, 1.5, "100", True, False, None])
def test_num_sample_rejects_invalid_values(num_sample):
    with pytest.raises(ValidationError):
        ExplainerInputModel(**_valid_params(), num_sample=num_sample)


def test_random_seed_defaults_to_zero():
    model = ExplainerInputModel(**_valid_params())

    assert model.random_seed == 0


def test_random_seed_accepts_negative_integer():
    model = ExplainerInputModel(**_valid_params(), random_seed=-7)

    assert model.random_seed == -7


@pytest.mark.parametrize("random_seed", [1.5, "0", True, False, None])
def test_random_seed_rejects_non_int(random_seed):
    with pytest.raises(ValidationError):
        ExplainerInputModel(**_valid_params(), random_seed=random_seed)
