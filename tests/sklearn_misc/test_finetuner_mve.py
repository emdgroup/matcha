"""MVE finetuning tests for GIN and Chemprop models."""

import shutil

import numpy as np
import pytest

from matcha.sklearn.finetuner import FinetuningRegressor

from .conftest import (
    FINETUNE_TRAIN,
    make_chemprop_regressor,
    make_gin_regressor,
)


@pytest.fixture(scope="module")
def mve_gin_pretrained_path(mol_list, regression_y, tmp_path_factory):
    model = make_gin_regressor()
    save_dir = str(tmp_path_factory.mktemp("mve_gin_pretrained"))
    model.fit(mol_list, regression_y)
    model.save_model(save_dir)
    return save_dir


@pytest.fixture(scope="module")
def mve_chemprop_pretrained_path(mol_list, regression_y, tmp_path_factory):
    model = make_chemprop_regressor()
    save_dir = str(tmp_path_factory.mktemp("mve_chemprop_pretrained"))
    model.fit(mol_list, regression_y)
    model.save_model(save_dir)
    return save_dir


class TestFinetuningRegressorMVE:
    @pytest.mark.parametrize("strategy", ["full", "lora"])
    def test_standard_mve_full_and_lora(
        self, strategy, mve_gin_pretrained_path, mol_list, regression_y
    ):
        from matcha.torch.predictors.mve import MVEPredictor

        kwargs = {**FINETUNE_TRAIN, "finetuning_strategy": strategy}
        model = FinetuningRegressor(
            path_to_pretrained=mve_gin_pretrained_path,
            uncertainty="mve",
            loss_fn="beta-nll",
            **kwargs,
        )
        model.fit(mol_list, regression_y)

        predictions = model.predict(mol_list)
        std = model.compute_uncertainty(mol_list)
        assert isinstance(model._model.predictor, MVEPredictor)
        assert predictions.shape == std.shape == (len(mol_list), 1)
        assert np.all(np.isfinite(predictions))
        assert np.all(np.isfinite(std))
        assert np.all(std >= 0.0)

    def test_standard_mve_multitask_calibration_and_roundtrip(
        self,
        mve_gin_pretrained_path,
        mol_list,
        multitask_regression_y,
        tmp_path,
    ):
        model = FinetuningRegressor(
            path_to_pretrained=mve_gin_pretrained_path,
            uncertainty="mve",
            loss_fn="beta-nll",
            num_endpoints=2,
            **FINETUNE_TRAIN,
        )
        model.fit(mol_list, multitask_regression_y)
        model.calibrate_uncertainty(
            mol_list,
            multitask_regression_y,
            algorithm="icp_regression",
        )
        predictions = model.predict(mol_list)
        std = model.compute_uncertainty(mol_list)

        save_dir = str(tmp_path / "mve_finetuner")
        model.save_model(save_dir)
        loaded = FinetuningRegressor.from_folder(save_dir, accelerator="cpu")

        assert predictions.shape == std.shape == (len(mol_list), 2)
        np.testing.assert_allclose(loaded.predict(mol_list), predictions, rtol=1e-5)
        np.testing.assert_allclose(loaded.compute_uncertainty(mol_list), std, rtol=1e-5)

    def test_standard_mve_rejects_non_mve_loss(self, mve_gin_pretrained_path):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="requires loss_fn"):
            FinetuningRegressor(
                path_to_pretrained=mve_gin_pretrained_path,
                uncertainty="mve",
                loss_fn="mse",
                **FINETUNE_TRAIN,
            )


class TestChempropFinetuningRegressorMVE:
    def test_mve_multitask_calibration_and_roundtrip(
        self,
        mve_chemprop_pretrained_path,
        mol_list,
        multitask_regression_y,
        tmp_path,
    ):
        from chemprop.nn.metrics import MVELoss
        from chemprop.nn.predictors import MveFFN

        model = FinetuningRegressor(
            path_to_pretrained=mve_chemprop_pretrained_path,
            uncertainty="mve",
            loss_fn="mve",
            num_endpoints=2,
            **FINETUNE_TRAIN,
        )
        model.fit(mol_list, multitask_regression_y)
        model.calibrate_uncertainty(
            mol_list,
            multitask_regression_y,
            algorithm="icp_regression",
        )
        predictions = model.predict(mol_list)
        std = model.compute_uncertainty(mol_list)

        save_dir = str(tmp_path / "mve_chemprop_finetuner")
        model.save_model(save_dir)
        loaded = FinetuningRegressor.from_folder(save_dir, accelerator="cpu")

        assert isinstance(model._model.predictor, MveFFN)
        assert isinstance(model._model.criterion, MVELoss)
        assert predictions.shape == std.shape == (len(mol_list), 2)
        assert np.all(np.isfinite(std))
        np.testing.assert_allclose(loaded.predict(mol_list), predictions, rtol=1e-5)
        np.testing.assert_allclose(loaded.compute_uncertainty(mol_list), std, rtol=1e-5)

    def test_rejects_matcha_mve_loss_alias(self, mve_chemprop_pretrained_path):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="requires loss_fn"):
            FinetuningRegressor(
                path_to_pretrained=mve_chemprop_pretrained_path,
                uncertainty="mve",
                loss_fn="beta-nll",
                **FINETUNE_TRAIN,
            )

    @pytest.mark.parametrize(
        ("uncertainty", "loss_fn"),
        [("mc-dropout", "mse"), ("mve", "mve")],
    )
    def test_rejects_mve_pretrained_model(
        self,
        uncertainty,
        loss_fn,
        mve_chemprop_pretrained_path,
        tmp_path,
    ):
        from matcha.utils.serialization import load_yaml, save_yaml

        source = tmp_path / "mve_source"
        shutil.copytree(mve_chemprop_pretrained_path, source)
        model_yaml = source / "config" / "model.yaml"
        params = load_yaml(model_yaml)
        params["uncertainty"] = "mve"
        params["loss_fn"] = "mve"
        save_yaml(model_yaml, params)

        with pytest.raises(ValueError, match="MVE-pretrained Chemprop"):
            FinetuningRegressor(
                path_to_pretrained=str(source),
                uncertainty=uncertainty,
                loss_fn=loss_fn,
                **FINETUNE_TRAIN,
            )
