"""Tests for auto-computation of total_steps in TrainingManager.

Verifies that when `total_steps` is not explicitly set in `scheduler_args`,
it is automatically computed as `num_epochs * ceil(len(train_data) / batch_size)`.
When explicitly set, the user's value is preserved unchanged.
"""

import math

import pytest

from matcha.sklearn.graph import AttentiveFPRegressor


@pytest.fixture()
def model_kwargs():
    return dict(
        enc_num_layers=1,
        enc_atom_hidden_dim=32,
        pred_hidden_dims=[32],
        rwse_k=0,
        laplacian_k=0,
        elstatic_k=0,
        distmat_k=0,
        rrwp_k=0,
        num_virtual_nodes=0,
        num_epochs=5,
        batch_size=8,
        accelerator="cpu",
        devices=1,
        early_stopping=False,
        stochastic_weight_averaging=False,
    )


class TestTotalStepsAutoCompute:
    """Tests for auto-computation of total_steps in the training manager."""

    def test_total_steps_auto_computed_when_absent(
        self, mol_list, regression_y, model_kwargs
    ):
        """When total_steps is not in scheduler_args, it should be auto-computed."""
        model_kwargs["scheduler_args"] = {"min_lr": 1e-5}
        model = AttentiveFPRegressor(**model_kwargs)
        model.fit(mol_list, regression_y)

        # Expected: num_epochs * ceil(train_size / batch_size)
        # With early_stopping=False, all 30 samples are used for training
        train_size = 30
        batch_size = model_kwargs["batch_size"]
        num_epochs = model_kwargs["num_epochs"]
        expected = num_epochs * math.ceil(train_size / batch_size)

        assert model._model.hparams["scheduler_args"]["total_steps"] == expected

    def test_total_steps_preserved_when_explicitly_set(
        self, mol_list, regression_y, model_kwargs
    ):
        """When total_steps is explicitly provided, it must be used unchanged."""
        explicit_total_steps = 999
        model_kwargs["scheduler_args"] = {
            "min_lr": 1e-5,
            "total_steps": explicit_total_steps,
        }
        model = AttentiveFPRegressor(**model_kwargs)
        model.fit(mol_list, regression_y)

        assert (
            model._model.hparams["scheduler_args"]["total_steps"]
            == explicit_total_steps
        )

    def test_scheduler_recreated_with_correct_total_steps(
        self, mol_list, regression_y, model_kwargs
    ):
        """The scheduler object should reflect the auto-computed total_steps."""
        model_kwargs["scheduler_args"] = {"min_lr": 1e-5}
        model = AttentiveFPRegressor(**model_kwargs)
        model.fit(mol_list, regression_y)

        # AttentiveFPRegressor defaults to warmup_linear_decay which stores total_steps
        scheduler = model._model.scheduler
        train_size = 30
        batch_size = model_kwargs["batch_size"]
        num_epochs = model_kwargs["num_epochs"]
        expected = num_epochs * math.ceil(train_size / batch_size)

        assert scheduler.total_steps == expected

    def test_injection_for_cosine_annealing_cyclic(
        self, mol_list, regression_y, model_kwargs
    ):
        """cosine_annealing_cyclic now accepts total_steps and should get it injected."""
        model_kwargs["scheduler"] = "cosine_annealing_cyclic"
        model_kwargs["scheduler_args"] = {"min_lr": 1e-5, "num_cycles": 5}
        model = AttentiveFPRegressor(**model_kwargs)
        model.fit(mol_list, regression_y)

        train_size = 30
        batch_size = model_kwargs["batch_size"]
        num_epochs = model_kwargs["num_epochs"]
        expected = num_epochs * math.ceil(train_size / batch_size)

        assert model._model.hparams["scheduler_args"]["total_steps"] == expected

    def test_auto_compute_with_warmup_cosine_annealing(
        self, mol_list, regression_y, model_kwargs
    ):
        """Verify auto-computation works with warmup_cosine_annealing scheduler."""
        model_kwargs["scheduler"] = "warmup_cosine_annealing"
        model_kwargs["scheduler_args"] = {"min_lr": 1e-6, "warmup_ratio": 0.05}
        model = AttentiveFPRegressor(**model_kwargs)
        model.fit(mol_list, regression_y)

        train_size = 30
        batch_size = model_kwargs["batch_size"]
        num_epochs = model_kwargs["num_epochs"]
        expected = num_epochs * math.ceil(train_size / batch_size)

        assert model._model.hparams["scheduler_args"]["total_steps"] == expected
        assert model._model.scheduler.total_steps == expected
