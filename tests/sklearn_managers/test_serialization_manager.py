"""Test SerializationManager through the sklearn API.

Model: GINRegressor (Graph), ChempropRegressor (Chemprop)
Exercises: save, load_from_folder, export_to_yaml, get_input_args,
    round-trip save → load → predict, and calibrator round-trip.
"""

import os

import numpy as np
import pytest
import torch
from rdkit.Chem.rdchem import Mol

from matcha.sklearn.graph import ChempropRegressor, GINRegressor
from matcha.sklearn.finetuner import FinetuningRegressor
from matcha.torch.models.finetuning.finetuner import _SELF_CONTAINED_SENTINEL


# =========================================================================
# GIN kwargs / fixtures
# =========================================================================


@pytest.fixture()
def model_kwargs():
    return dict(
        enc_num_layers=2,
        enc_atom_hidden_dim=32,
        laplacian_k=0,
        pred_hidden_dims=[32],
        num_epochs=1,
        batch_size=32,
        accelerator="cpu",
        devices=1,
        early_stopping=False,
        stochastic_weight_averaging=False,
    )


@pytest.fixture()
def fitted_model(mol_list: list[Mol], regression_y, model_kwargs):
    model = GINRegressor(**model_kwargs)
    model.fit(mol_list, regression_y)
    return model


@pytest.fixture()
def calibrated_model(mol_list: list[Mol], regression_y, model_kwargs):
    """A fitted GINRegressor with an ICP calibrator attached."""
    model = GINRegressor(**model_kwargs)
    model.fit(mol_list, regression_y)
    model.calibrate_uncertainty(
        calibration_mols=mol_list,
        calibration_y=regression_y,
        num_iterations=3,
        algorithm="icp_regression",
    )
    return model


# =========================================================================
# Chemprop kwargs / fixtures
# =========================================================================


@pytest.fixture()
def chemprop_model_kwargs():
    return dict(
        enc_num_layers=1,
        enc_atom_hidden_dim=32,
        pred_hidden_dim=32,
        pred_num_layers=1,
        feature_list=None,
        num_epochs=1,
        batch_size=32,
        accelerator="cpu",
        devices=1,
        early_stopping=False,
        stochastic_weight_averaging=False,
    )


@pytest.fixture()
def chemprop_fitted_model(mol_list: list[Mol], regression_y, chemprop_model_kwargs):
    model = ChempropRegressor(**chemprop_model_kwargs)
    model.fit(mol_list, regression_y)
    return model


@pytest.fixture()
def chemprop_calibrated_model(mol_list: list[Mol], regression_y, chemprop_model_kwargs):
    """A fitted ChempropRegressor with an ICP calibrator attached."""
    model = ChempropRegressor(**chemprop_model_kwargs)
    model.fit(mol_list, regression_y)
    try:
        model.calibrate_uncertainty(
            calibration_mols=mol_list,
            calibration_y=regression_y,
            num_iterations=3,
            algorithm="icp_regression",
        )
    except ValueError as e:
        if "MC Dropout" in str(e):
            pytest.skip("Chemprop does not support MC Dropout")
        raise
    return model


class TestSerializationManagerSave:
    """Tests for save_model and the artifact layout."""

    def test_save_creates_checkpoint(self, fitted_model, tmp_path):
        save_dir = str(tmp_path / "saved_model")
        fitted_model.save_model(save_dir)
        assert os.path.exists(os.path.join(save_dir, "model.ckpt"))

    def test_save_creates_config_directory(self, fitted_model, tmp_path):
        save_dir = str(tmp_path / "saved_model")
        fitted_model.save_model(save_dir)
        config_dir = os.path.join(save_dir, "config")
        assert os.path.isdir(config_dir)

    def test_save_creates_yaml_configs(self, fitted_model, tmp_path):
        save_dir = str(tmp_path / "saved_model")
        fitted_model.save_model(save_dir)
        config_dir = os.path.join(save_dir, "config")
        for fname in [
            "model.yaml",
            "training.yaml",
            "datamodule.yaml",
            "metadata.yaml",
            "manifest.yaml",
        ]:
            assert os.path.exists(os.path.join(config_dir, fname)), (
                f"Missing config file: {fname}"
            )

    def test_save_creates_state_directory(self, fitted_model, tmp_path):
        save_dir = str(tmp_path / "saved_model")
        fitted_model.save_model(save_dir)
        state_dir = os.path.join(save_dir, "state")
        assert os.path.isdir(state_dir)

    def test_save_creates_datamodule_state(self, fitted_model, tmp_path):
        save_dir = str(tmp_path / "saved_model")
        fitted_model.save_model(save_dir)
        state_path = os.path.join(save_dir, "state", "datamodule_state.pkl")
        assert os.path.exists(state_path)


class TestSerializationManagerLoadRoundTrip:
    """Tests for save → load → predict round-trip."""

    def test_load_from_folder_produces_fitted_model(
        self, fitted_model, mol_list, tmp_path
    ):
        save_dir = str(tmp_path / "saved_model")
        fitted_model.save_model(save_dir)
        loaded = GINRegressor.from_folder(save_dir, accelerator="cpu")
        assert loaded.is_fitted is True

    def test_loaded_model_predicts_same_shape(self, fitted_model, mol_list, tmp_path):
        save_dir = str(tmp_path / "saved_model")
        fitted_model.save_model(save_dir)
        loaded = GINRegressor.from_folder(save_dir, accelerator="cpu")
        preds = loaded.predict(mol_list)
        assert preds.shape == (len(mol_list), 1)

    def test_loaded_model_predicts_similar_values(
        self, fitted_model, mol_list, tmp_path
    ):
        save_dir = str(tmp_path / "saved_model")
        preds_original = fitted_model.predict(mol_list)
        fitted_model.save_model(save_dir)
        loaded = GINRegressor.from_folder(save_dir, accelerator="cpu")
        preds_loaded = loaded.predict(mol_list)
        np.testing.assert_array_almost_equal(preds_original, preds_loaded, decimal=3)


class TestSerializationManagerExportYaml:
    """Tests for export_to_yaml (config only, no weights)."""

    def test_export_creates_config_directory(self, fitted_model, tmp_path):
        export_dir = str(tmp_path / "exported_config")
        fitted_model.export_to_yaml(export_dir)
        config_dir = os.path.join(export_dir, "config")
        assert os.path.isdir(config_dir)

    def test_export_does_not_create_checkpoint(self, fitted_model, tmp_path):
        export_dir = str(tmp_path / "exported_config")
        fitted_model.export_to_yaml(export_dir)
        assert not os.path.exists(os.path.join(export_dir, "model.ckpt"))

    def test_export_creates_yaml_files(self, fitted_model, tmp_path):
        export_dir = str(tmp_path / "exported_config")
        fitted_model.export_to_yaml(export_dir)
        config_dir = os.path.join(export_dir, "config")
        for fname in [
            "model.yaml",
            "training.yaml",
            "datamodule.yaml",
            "metadata.yaml",
            "manifest.yaml",
        ]:
            assert os.path.exists(os.path.join(config_dir, fname))


class TestSerializationManagerGetInputArgs:
    """Tests for get_input_args."""

    def test_get_input_args_returns_dict(self, fitted_model):
        args = fitted_model.get_input_args()
        assert isinstance(args, dict)

    def test_get_input_args_contains_constructor_params(self, fitted_model):
        args = fitted_model.get_input_args()
        # Should contain model-architecture params
        assert "enc_atom_hidden_dim" in args
        # Should contain training params
        assert "num_epochs" in args


# =========================================================================
# Chemprop variants – exercises the ChempropDataModule serialization path
# =========================================================================


class TestChempropSerializationManagerSave:
    """Tests for save_model and the artifact layout with ChempropRegressor."""

    def test_save_creates_checkpoint(self, chemprop_fitted_model, tmp_path):
        save_dir = str(tmp_path / "saved_model")
        chemprop_fitted_model.save_model(save_dir)
        assert os.path.exists(os.path.join(save_dir, "model.ckpt"))

    def test_save_creates_config_directory(self, chemprop_fitted_model, tmp_path):
        save_dir = str(tmp_path / "saved_model")
        chemprop_fitted_model.save_model(save_dir)
        config_dir = os.path.join(save_dir, "config")
        assert os.path.isdir(config_dir)

    def test_save_creates_yaml_configs(self, chemprop_fitted_model, tmp_path):
        save_dir = str(tmp_path / "saved_model")
        chemprop_fitted_model.save_model(save_dir)
        config_dir = os.path.join(save_dir, "config")
        for fname in [
            "model.yaml",
            "training.yaml",
            "datamodule.yaml",
            "metadata.yaml",
            "manifest.yaml",
        ]:
            assert os.path.exists(os.path.join(config_dir, fname)), (
                f"Missing config file: {fname}"
            )

    def test_save_creates_state_directory(self, chemprop_fitted_model, tmp_path):
        save_dir = str(tmp_path / "saved_model")
        chemprop_fitted_model.save_model(save_dir)
        state_dir = os.path.join(save_dir, "state")
        assert os.path.isdir(state_dir)

    def test_save_creates_datamodule_state(self, chemprop_fitted_model, tmp_path):
        save_dir = str(tmp_path / "saved_model")
        chemprop_fitted_model.save_model(save_dir)
        state_path = os.path.join(save_dir, "state", "datamodule_state.pkl")
        assert os.path.exists(state_path)


class TestChempropSerializationManagerLoadRoundTrip:
    """Tests for save → load → predict round-trip with ChempropRegressor."""

    def test_load_from_folder_produces_fitted_model(
        self, chemprop_fitted_model, mol_list, tmp_path
    ):
        save_dir = str(tmp_path / "saved_model")
        chemprop_fitted_model.save_model(save_dir)
        loaded = ChempropRegressor.from_folder(save_dir, accelerator="cpu")
        assert loaded.is_fitted is True

    def test_loaded_model_predicts_same_shape(
        self, chemprop_fitted_model, mol_list, tmp_path
    ):
        save_dir = str(tmp_path / "saved_model")
        chemprop_fitted_model.save_model(save_dir)
        loaded = ChempropRegressor.from_folder(save_dir, accelerator="cpu")
        preds = loaded.predict(mol_list)
        assert preds.shape == (len(mol_list), 1)

    def test_loaded_model_predicts_similar_values(
        self, chemprop_fitted_model, mol_list, tmp_path
    ):
        save_dir = str(tmp_path / "saved_model")
        preds_original = chemprop_fitted_model.predict(mol_list)
        chemprop_fitted_model.save_model(save_dir)
        loaded = ChempropRegressor.from_folder(save_dir, accelerator="cpu")
        preds_loaded = loaded.predict(mol_list)
        np.testing.assert_array_almost_equal(preds_original, preds_loaded, decimal=3)


class TestChempropSerializationManagerExportYaml:
    """Tests for export_to_yaml with ChempropRegressor."""

    def test_export_creates_config_directory(self, chemprop_fitted_model, tmp_path):
        export_dir = str(tmp_path / "exported_config")
        chemprop_fitted_model.export_to_yaml(export_dir)
        config_dir = os.path.join(export_dir, "config")
        assert os.path.isdir(config_dir)

    def test_export_does_not_create_checkpoint(self, chemprop_fitted_model, tmp_path):
        export_dir = str(tmp_path / "exported_config")
        chemprop_fitted_model.export_to_yaml(export_dir)
        assert not os.path.exists(os.path.join(export_dir, "model.ckpt"))

    def test_export_creates_yaml_files(self, chemprop_fitted_model, tmp_path):
        export_dir = str(tmp_path / "exported_config")
        chemprop_fitted_model.export_to_yaml(export_dir)
        config_dir = os.path.join(export_dir, "config")
        for fname in [
            "model.yaml",
            "training.yaml",
            "datamodule.yaml",
            "metadata.yaml",
            "manifest.yaml",
        ]:
            assert os.path.exists(os.path.join(config_dir, fname))


class TestChempropSerializationManagerGetInputArgs:
    """Tests for get_input_args with ChempropRegressor."""

    def test_get_input_args_returns_dict(self, chemprop_fitted_model):
        args = chemprop_fitted_model.get_input_args()
        assert isinstance(args, dict)

    def test_get_input_args_contains_constructor_params(self, chemprop_fitted_model):
        args = chemprop_fitted_model.get_input_args()
        # Should contain Chemprop model-architecture params
        assert "enc_atom_hidden_dim" in args
        assert "pred_hidden_dim" in args
        # Should contain training params
        assert "num_epochs" in args


# =========================================================================
# GIN with calibrator – exercises calibrator serialization path
# =========================================================================


class TestSerializationManagerCalibratorSave:
    """Tests for save_model artifact layout when a calibrator is present."""

    def test_save_creates_calibrator_pickle(self, calibrated_model, tmp_path):
        save_dir = str(tmp_path / "saved_model")
        calibrated_model.save_model(save_dir)
        calibrator_path = os.path.join(save_dir, "state", "calibrator.pkl")
        assert os.path.exists(calibrator_path)

    def test_save_creates_calibration_yaml(self, calibrated_model, tmp_path):
        save_dir = str(tmp_path / "saved_model")
        calibrated_model.save_model(save_dir)
        cal_yaml = os.path.join(save_dir, "config", "calibration.yaml")
        assert os.path.exists(cal_yaml)

    def test_save_still_creates_standard_artifacts(self, calibrated_model, tmp_path):
        save_dir = str(tmp_path / "saved_model")
        calibrated_model.save_model(save_dir)
        assert os.path.exists(os.path.join(save_dir, "model.ckpt"))
        assert os.path.isdir(os.path.join(save_dir, "config"))
        assert os.path.isdir(os.path.join(save_dir, "state"))


class TestSerializationManagerCalibratorLoadRoundTrip:
    """Tests for save → load → predict round-trip with calibrator."""

    def test_load_restores_calibrator(self, calibrated_model, mol_list, tmp_path):
        save_dir = str(tmp_path / "saved_model")
        calibrated_model.save_model(save_dir)
        loaded = GINRegressor.from_folder(save_dir, accelerator="cpu")
        assert loaded._uncertainty_manager.calibrator is not None

    def test_loaded_model_with_calibrator_is_fitted(
        self, calibrated_model, mol_list, tmp_path
    ):
        save_dir = str(tmp_path / "saved_model")
        calibrated_model.save_model(save_dir)
        loaded = GINRegressor.from_folder(save_dir, accelerator="cpu")
        assert loaded.is_fitted is True

    def test_loaded_model_with_calibrator_predicts_same_shape(
        self, calibrated_model, mol_list, tmp_path
    ):
        save_dir = str(tmp_path / "saved_model")
        calibrated_model.save_model(save_dir)
        loaded = GINRegressor.from_folder(save_dir, accelerator="cpu")
        preds = loaded.predict(mol_list)
        assert preds.shape == (len(mol_list), 1)

    def test_loaded_model_with_calibrator_predicts_similar_values(
        self, calibrated_model, mol_list, tmp_path
    ):
        save_dir = str(tmp_path / "saved_model")
        preds_original = calibrated_model.predict(mol_list)
        calibrated_model.save_model(save_dir)
        loaded = GINRegressor.from_folder(save_dir, accelerator="cpu")
        preds_loaded = loaded.predict(mol_list)
        np.testing.assert_array_almost_equal(preds_original, preds_loaded, decimal=3)

    def test_loaded_calibrated_uncertainty_is_finite(
        self, calibrated_model, mol_list, tmp_path
    ):
        save_dir = str(tmp_path / "saved_model")
        calibrated_model.save_model(save_dir)
        loaded = GINRegressor.from_folder(save_dir, accelerator="cpu")
        unc = loaded.compute_uncertainty(mol_list, num_iterations=3)
        assert isinstance(unc, np.ndarray)
        assert unc.shape[0] == len(mol_list)
        assert np.all(np.isfinite(unc))
        assert np.all(unc >= 0.0)


# =========================================================================
# Chemprop with calibrator – exercises calibrator serialization path
# =========================================================================


class TestChempropSerializationManagerCalibratorSave:
    """Tests for save_model artifact layout when a calibrator is present (Chemprop)."""

    def test_save_creates_calibrator_pickle(self, chemprop_calibrated_model, tmp_path):
        save_dir = str(tmp_path / "saved_model")
        chemprop_calibrated_model.save_model(save_dir)
        calibrator_path = os.path.join(save_dir, "state", "calibrator.pkl")
        assert os.path.exists(calibrator_path)

    def test_save_creates_calibration_yaml(self, chemprop_calibrated_model, tmp_path):
        save_dir = str(tmp_path / "saved_model")
        chemprop_calibrated_model.save_model(save_dir)
        cal_yaml = os.path.join(save_dir, "config", "calibration.yaml")
        assert os.path.exists(cal_yaml)


class TestChempropSerializationManagerCalibratorLoadRoundTrip:
    """Tests for save → load → predict round-trip with calibrator (Chemprop)."""

    def test_load_restores_calibrator(
        self, chemprop_calibrated_model, mol_list, tmp_path
    ):
        save_dir = str(tmp_path / "saved_model")
        chemprop_calibrated_model.save_model(save_dir)
        loaded = ChempropRegressor.from_folder(save_dir, accelerator="cpu")
        assert loaded._uncertainty_manager.calibrator is not None

    def test_loaded_model_with_calibrator_predicts_same_shape(
        self, chemprop_calibrated_model, mol_list, tmp_path
    ):
        save_dir = str(tmp_path / "saved_model")
        chemprop_calibrated_model.save_model(save_dir)
        loaded = ChempropRegressor.from_folder(save_dir, accelerator="cpu")
        preds = loaded.predict(mol_list)
        assert preds.shape == (len(mol_list), 1)

    def test_loaded_model_with_calibrator_predicts_similar_values(
        self, chemprop_calibrated_model, mol_list, tmp_path
    ):
        save_dir = str(tmp_path / "saved_model")
        preds_original = chemprop_calibrated_model.predict(mol_list)
        chemprop_calibrated_model.save_model(save_dir)
        loaded = ChempropRegressor.from_folder(save_dir, accelerator="cpu")
        preds_loaded = loaded.predict(mol_list)
        np.testing.assert_array_almost_equal(preds_original, preds_loaded, decimal=3)

    def test_loaded_calibrated_uncertainty_is_finite(
        self, chemprop_calibrated_model, mol_list, tmp_path
    ):
        save_dir = str(tmp_path / "saved_model")
        chemprop_calibrated_model.save_model(save_dir)
        loaded = ChempropRegressor.from_folder(save_dir, accelerator="cpu")
        unc = loaded.compute_uncertainty(mol_list, num_iterations=3)
        assert isinstance(unc, np.ndarray)
        assert unc.shape[0] == len(mol_list)
        assert np.all(np.isfinite(unc))
        assert np.all(unc >= 0.0)


# =========================================================================
# Finetuner self-contained checkpoint tests
# =========================================================================


@pytest.fixture()
def pretrained_gin_path(mol_list, regression_y, tmp_path):
    """Fit a GINRegressor, save it, return the path."""
    model = GINRegressor(
        enc_num_layers=1,
        enc_atom_hidden_dim=32,
        laplacian_k=0,
        pred_hidden_dims=[32],
        num_epochs=1,
        batch_size=32,
        accelerator="cpu",
        devices=1,
        early_stopping=False,
        stochastic_weight_averaging=False,
    )
    model.fit(mol_list, regression_y)
    save_dir = str(tmp_path / "pretrained")
    model.save_model(save_dir)
    return save_dir


@pytest.fixture()
def fitted_finetuner(mol_list, regression_y, pretrained_gin_path, tmp_path):
    """Fit a FinetuningRegressor from a pretrained GIN, save it, return (model, save_dir)."""
    ft = FinetuningRegressor(
        path_to_pretrained=pretrained_gin_path,
        pred_hidden_dims=[32],
        num_epochs=1,
        batch_size=32,
        accelerator="cpu",
        devices=1,
        early_stopping=False,
        stochastic_weight_averaging=False,
        finetuning_strategy="full",
    )
    ft.fit(mol_list, regression_y)
    save_dir = str(tmp_path / "finetuned")
    ft.save_model(save_dir)
    return ft, save_dir


class TestFinetunerSelfContainedCheckpoint:
    """Tests that SerializationManager.save() produces self-contained
    Finetuner checkpoints with sentinel and embedded config."""

    def test_checkpoint_contains_sentinel(self, fitted_finetuner):
        _, save_dir = fitted_finetuner
        ckpt = torch.load(
            os.path.join(save_dir, "model.ckpt"),
            weights_only=False,
            map_location="cpu",
        )
        hparams = ckpt["hyper_parameters"]
        assert hparams["path_to_pretrained"] == _SELF_CONTAINED_SENTINEL

    def test_checkpoint_contains_pretrain_config(self, fitted_finetuner):
        _, save_dir = fitted_finetuner
        ckpt = torch.load(
            os.path.join(save_dir, "model.ckpt"),
            weights_only=False,
            map_location="cpu",
        )
        hparams = ckpt["hyper_parameters"]
        assert "_pretrain_config" in hparams
        config = hparams["_pretrain_config"]
        assert "origin_type" in config
        assert "pretrain_params" in config

    def test_pretrain_config_yaml_written(self, fitted_finetuner):
        _, save_dir = fitted_finetuner
        config_path = os.path.join(save_dir, "config", "pretrain_config.yaml")
        assert os.path.exists(config_path)

    def test_pretrain_config_yaml_matches_checkpoint(self, fitted_finetuner):
        from matcha.utils import load_yaml

        _, save_dir = fitted_finetuner
        ckpt = torch.load(
            os.path.join(save_dir, "model.ckpt"),
            weights_only=False,
            map_location="cpu",
        )
        yaml_config = load_yaml(
            os.path.join(save_dir, "config", "pretrain_config.yaml")
        )
        ckpt_config = ckpt["hyper_parameters"]["_pretrain_config"]
        assert yaml_config["origin_type"] == ckpt_config["origin_type"]

    def test_classic_model_not_affected(self, fitted_model, tmp_path):
        """Non-finetuner models should not get sentinel treatment."""
        save_dir = str(tmp_path / "classic_model")
        fitted_model.save_model(save_dir)
        ckpt = torch.load(
            os.path.join(save_dir, "model.ckpt"),
            weights_only=False,
            map_location="cpu",
        )
        hparams = ckpt["hyper_parameters"]
        assert "_pretrain_config" not in hparams
        assert hparams.get("path_to_pretrained") != _SELF_CONTAINED_SENTINEL
