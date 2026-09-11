"""Finetuner save/load integration and path-independence tests."""

import os
import shutil

import numpy as np
import pytest

from matcha.sklearn.finetuner import FinetuningRegressor

from .conftest import (
    FINETUNE_TRAIN,
    make_chemprop_regressor,
    make_gin_regressor,
    make_mlp_regressor,
)


# =========================================================================
# Self-contained loading integration tests (issue #373, stage 3)
# =========================================================================

_SELF_CONTAINED_REGRESSOR_FACTORIES = [
    pytest.param(make_mlp_regressor, id="MLP"),
    pytest.param(make_gin_regressor, id="GIN"),
]


class TestSelfContainedLoading:
    """Verify finetuned models are self-contained and path-independent.

    After saving, a finetuned model must be loadable without access to
    ancestor model directories.
    """

    @pytest.fixture(params=_SELF_CONTAINED_REGRESSOR_FACTORIES)
    def pretrained_path(self, request, mol_list, regression_y, tmp_path):
        """Fit a base regressor, save it, return its path."""
        factory = request.param
        model = factory()
        save_dir = str(tmp_path / f"pretrained_{factory.__name__}")
        model.fit(mol_list, regression_y)
        model.save_model(save_dir)
        return save_dir

    @pytest.fixture()
    def fitted_and_saved_finetuner(
        self, pretrained_path, mol_list, regression_y, tmp_path
    ):
        """Fit a finetuner, save it, return (save_path, expected_preds)."""
        finetuner = FinetuningRegressor(
            path_to_pretrained=pretrained_path,
            **FINETUNE_TRAIN,
        )
        finetuner.fit(mol_list, regression_y)
        preds_before = finetuner.predict(mol_list)

        save_dir = str(tmp_path / "finetuned_saved")
        finetuner.save_model(save_dir)
        return save_dir, preds_before, pretrained_path

    def test_load_after_source_deleted(self, fitted_and_saved_finetuner, mol_list):
        """Load a saved finetuner after deleting the pretrained source path."""
        save_dir, preds_before, pretrained_path = fitted_and_saved_finetuner

        # Delete the pretrained source
        shutil.rmtree(pretrained_path)
        assert not os.path.exists(pretrained_path)

        # Load from saved path — should not need the pretrained source
        loaded = FinetuningRegressor.from_folder(save_dir, accelerator="cpu")
        preds_after = loaded.predict(mol_list)

        np.testing.assert_allclose(preds_after, preds_before, rtol=1e-5, atol=1e-6)

    def test_load_from_moved_directory(
        self, fitted_and_saved_finetuner, mol_list, tmp_path
    ):
        """Load a saved finetuner after moving its directory to a new path."""
        save_dir, preds_before, _ = fitted_and_saved_finetuner

        # Move the saved model to a different location
        new_dir = str(tmp_path / "moved_model")
        shutil.move(save_dir, new_dir)
        assert not os.path.exists(save_dir)

        # Load from new path
        loaded = FinetuningRegressor.from_folder(new_dir, accelerator="cpu")
        preds_after = loaded.predict(mol_list)

        np.testing.assert_allclose(preds_after, preds_before, rtol=1e-5, atol=1e-6)

    def test_save_load_roundtrip_regression(self, fitted_and_saved_finetuner, mol_list):
        """Regression test: basic save/load round-trip produces same predictions."""
        save_dir, preds_before, _ = fitted_and_saved_finetuner

        loaded = FinetuningRegressor.from_folder(save_dir, accelerator="cpu")
        preds_after = loaded.predict(mol_list)

        np.testing.assert_allclose(preds_after, preds_before, rtol=1e-5, atol=1e-6)


_NESTED_FINETUNER_FACTORIES = [
    pytest.param(make_mlp_regressor, id="MLP"),
    pytest.param(make_gin_regressor, id="GIN"),
]


class TestNestedSelfContainedLoading:
    """Verify nested finetuning (A → B → C) produces self-contained models.

    After saving model C, removing A and B paths must not break loading.
    """

    @pytest.fixture(params=_NESTED_FINETUNER_FACTORIES)
    def nested_finetuner_artifacts(self, request, mol_list, regression_y, tmp_path):
        """Build a three-level finetuning chain: base → B → C, save C."""
        factory = request.param

        # Level A: base model
        base = factory()
        base_dir = str(tmp_path / "base_A")
        base.fit(mol_list, regression_y)
        base.save_model(base_dir)

        # Level B: finetuned from A
        finetuner_b = FinetuningRegressor(
            path_to_pretrained=base_dir,
            **FINETUNE_TRAIN,
        )
        finetuner_b.fit(mol_list, regression_y)
        b_dir = str(tmp_path / "finetuned_B")
        finetuner_b.save_model(b_dir)

        # Level C: finetuned from B
        finetuner_c = FinetuningRegressor(
            path_to_pretrained=b_dir,
            **FINETUNE_TRAIN,
        )
        finetuner_c.fit(mol_list, regression_y)
        preds_c = finetuner_c.predict(mol_list)
        c_dir = str(tmp_path / "finetuned_C")
        finetuner_c.save_model(c_dir)

        return c_dir, preds_c, base_dir, b_dir

    def test_nested_load_after_ancestors_deleted(
        self, nested_finetuner_artifacts, mol_list
    ):
        """Load model C after deleting both A and B source directories."""
        c_dir, preds_c, base_dir, b_dir = nested_finetuner_artifacts

        # Remove ancestor paths
        shutil.rmtree(base_dir)
        shutil.rmtree(b_dir)
        assert not os.path.exists(base_dir)
        assert not os.path.exists(b_dir)

        # Load C — must work without A or B
        loaded = FinetuningRegressor.from_folder(c_dir, accelerator="cpu")
        preds_loaded = loaded.predict(mol_list)

        np.testing.assert_allclose(preds_loaded, preds_c, rtol=1e-5, atol=1e-6)


# =========================================================================
# Self-contained loading for ChempropFinetuner (issue #396, stage 3)
# =========================================================================


class TestSelfContainedChempropLoading:
    """Verify ChempropFinetuner-backed models are self-contained after save.

    After saving, loading must work without access to the original pretrained
    Chemprop model directory.
    """

    @pytest.fixture()
    def chemprop_pretrained_path(self, mol_list, regression_y, tmp_path):
        """Fit a base ChempropRegressor, save it, return its path."""
        model = make_chemprop_regressor()
        save_dir = str(tmp_path / "pretrained_chemprop")
        model.fit(mol_list, regression_y)
        model.save_model(save_dir)
        return save_dir

    @pytest.fixture()
    def fitted_chemprop_finetuner(
        self, chemprop_pretrained_path, mol_list, regression_y, tmp_path
    ):
        """Fit a ChempropFinetuner, save it, return (save_path, preds, pretrained_path)."""
        finetuner = FinetuningRegressor(
            path_to_pretrained=chemprop_pretrained_path,
            **FINETUNE_TRAIN,
        )
        finetuner.fit(mol_list, regression_y)
        preds_before = finetuner.predict(mol_list)

        save_dir = str(tmp_path / "finetuned_chemprop")
        finetuner.save_model(save_dir)
        return save_dir, preds_before, chemprop_pretrained_path

    def test_model_yaml_has_sentinel(self, fitted_chemprop_finetuner):
        """model.yaml should contain path_to_pretrained: __self_contained__."""
        from matcha.utils.serialization import load_yaml

        save_dir, _, _ = fitted_chemprop_finetuner
        model_yaml = load_yaml(os.path.join(save_dir, "config", "model.yaml"))
        assert model_yaml["path_to_pretrained"] == "__self_contained__"

    def test_pretrain_config_yaml_exists(self, fitted_chemprop_finetuner):
        """pretrain_config.yaml should be written to the config directory."""
        from matcha.utils.serialization import load_yaml

        save_dir, _, _ = fitted_chemprop_finetuner
        config_path = os.path.join(save_dir, "config", "pretrain_config.yaml")
        assert os.path.exists(config_path)

        config = load_yaml(config_path)
        assert config["origin_type"] == "chemprop"
        assert "pretrain_params" in config

    def test_load_after_source_deleted(self, fitted_chemprop_finetuner, mol_list):
        """Load a saved ChempropFinetuner after deleting the pretrained source."""
        save_dir, preds_before, pretrained_path = fitted_chemprop_finetuner

        # Delete the pretrained source
        shutil.rmtree(pretrained_path)
        assert not os.path.exists(pretrained_path)

        # Load — should not need the pretrained source
        loaded = FinetuningRegressor.from_folder(save_dir, accelerator="cpu")
        preds_after = loaded.predict(mol_list)

        np.testing.assert_allclose(preds_after, preds_before, rtol=1e-5, atol=1e-6)

    def test_load_from_moved_directory(
        self, fitted_chemprop_finetuner, mol_list, tmp_path
    ):
        """Load a saved ChempropFinetuner after moving its directory."""
        save_dir, preds_before, _ = fitted_chemprop_finetuner

        new_dir = str(tmp_path / "moved_chemprop_model")
        shutil.move(save_dir, new_dir)
        assert not os.path.exists(save_dir)

        loaded = FinetuningRegressor.from_folder(new_dir, accelerator="cpu")
        preds_after = loaded.predict(mol_list)

        np.testing.assert_allclose(preds_after, preds_before, rtol=1e-5, atol=1e-6)
