"""Finetuner label-encoder and predictor-forwarding tests."""

import numpy as np
import pytest

from matcha.sklearn.finetuner import FinetuningClassifier, FinetuningRegressor

from .conftest import (
    FINETUNE_TRAIN,
    make_chemprop_regressor,
    make_cnn_classifier,
    make_cnn_regressor,
    make_gin_classifier,
    make_gin_regressor,
    make_mlp_regressor,
)


# =========================================================================
# Label encoder preservation in FinetuningClassifier (issue #407, stage 1)
# =========================================================================

_ALL_REGRESSOR_FACTORIES = [
    pytest.param(make_mlp_regressor, id="MLP"),
    pytest.param(make_gin_regressor, id="GIN"),
    pytest.param(make_cnn_regressor, id="CNN"),
    pytest.param(make_chemprop_regressor, id="Chemprop"),
]


class TestFinetuningClassifierLabelEncoder:
    """Verify label encoder params are preserved through FinetuningClassifier construction.

    When finetuning regression → classification with user-provided class_thresholds
    and class_labels, the label encoder must retain those parameters so that
    predictions produce categorical outputs.
    """

    LABEL_ENCODER_PARAMS = {
        "encoder_type": "binary_classification",
        0: {
            "task_label": "activity",
            "class_thresholds": [0.5],
            "class_labels": ["inactive", "active"],
        },
    }

    @pytest.fixture(params=_ALL_REGRESSOR_FACTORIES)
    def pretrained_regressor_path(self, request, mol_list, regression_y, tmp_path):
        """Fit a base regressor, save it, and return the save path."""
        factory = request.param
        model = factory()
        save_dir = str(tmp_path / f"pretrained_{factory.__name__}")
        model.fit(mol_list, regression_y)
        model.save_model(save_dir)
        return save_dir

    def test_label_encoder_preserves_thresholds(self, pretrained_regressor_path):
        """User-provided class_thresholds are preserved after construction."""
        finetuner = FinetuningClassifier(
            path_to_pretrained=pretrained_regressor_path,
            label_encoder_params=self.LABEL_ENCODER_PARAMS.copy(),
            **FINETUNE_TRAIN,
        )
        encoder = finetuner.datamodule._label_encoder
        assert encoder.params.class_thresholds[0] == [0.5]

    def test_label_encoder_preserves_class_labels(self, pretrained_regressor_path):
        """User-provided class_labels are preserved after construction."""
        finetuner = FinetuningClassifier(
            path_to_pretrained=pretrained_regressor_path,
            label_encoder_params=self.LABEL_ENCODER_PARAMS.copy(),
            **FINETUNE_TRAIN,
        )
        encoder = finetuner.datamodule._label_encoder
        assert encoder.params.class_labels[0] == ["inactive", "active"]

    def test_label_encoder_is_set(self, pretrained_regressor_path):
        """label_encoder.is_set() returns True when thresholds are configured."""
        finetuner = FinetuningClassifier(
            path_to_pretrained=pretrained_regressor_path,
            label_encoder_params=self.LABEL_ENCODER_PARAMS.copy(),
            **FINETUNE_TRAIN,
        )
        assert finetuner.datamodule._label_encoder.is_set() is True

    def test_label_encoder_is_binary_classification_type(
        self, pretrained_regressor_path
    ):
        """Label encoder is a BinaryClassificationLabelEncoder."""
        from matcha.datamodules.classic.label_encoder import (
            BinaryClassificationLabelEncoder,
        )

        finetuner = FinetuningClassifier(
            path_to_pretrained=pretrained_regressor_path,
            label_encoder_params=self.LABEL_ENCODER_PARAMS.copy(),
            **FINETUNE_TRAIN,
        )
        assert isinstance(
            finetuner.datamodule._label_encoder, BinaryClassificationLabelEncoder
        )

    def test_is_classification_flag_set(self, pretrained_regressor_path):
        """datamodule.params.is_classification is True after construction."""
        finetuner = FinetuningClassifier(
            path_to_pretrained=pretrained_regressor_path,
            label_encoder_params=self.LABEL_ENCODER_PARAMS.copy(),
            **FINETUNE_TRAIN,
        )
        assert finetuner.datamodule.params.is_classification is True

    def test_predict_produces_categorical_output(
        self, pretrained_regressor_path, mol_list, regression_y
    ):
        """Predictions from a classifier with thresholds produce categorical (0/1) outputs."""
        finetuner = FinetuningClassifier(
            path_to_pretrained=pretrained_regressor_path,
            label_encoder_params=self.LABEL_ENCODER_PARAMS.copy(),
            **FINETUNE_TRAIN,
        )
        finetuner.fit(mol_list, regression_y)
        preds = finetuner.predict(mol_list)
        unique_vals = set(np.unique(preds))
        assert unique_vals.issubset({0.0, 1.0})

    def test_empty_label_encoder_params_still_creates_binary_encoder(
        self, pretrained_regressor_path
    ):
        """Even without user thresholds, the encoder type is binary_classification."""
        from matcha.datamodules.classic.label_encoder import (
            BinaryClassificationLabelEncoder,
        )

        finetuner = FinetuningClassifier(
            path_to_pretrained=pretrained_regressor_path,
            label_encoder_params={},
            **FINETUNE_TRAIN,
        )
        assert isinstance(
            finetuner.datamodule._label_encoder, BinaryClassificationLabelEncoder
        )

    def test_adapt_dicts_for_mixin_sets_encoder_type(self, pretrained_regressor_path):
        """_adapt_dicts_for_mixin sets encoder_type without dropping other keys."""
        finetuner = FinetuningClassifier(
            path_to_pretrained=pretrained_regressor_path,
            label_encoder_params=self.LABEL_ENCODER_PARAMS.copy(),
            **FINETUNE_TRAIN,
        )
        # Call _adapt_dicts_for_mixin with a dict that has extra keys
        dm_dict = {
            "label_encoder_params": {
                "encoder_type": "binary_classification",
                "extra_key": "should_persist",
            },
            "is_classification": False,
        }
        result_dm, _ = finetuner._adapt_dicts_for_mixin(dm_dict, {})
        assert result_dm["is_classification"] is True
        assert (
            result_dm["label_encoder_params"]["encoder_type"] == "binary_classification"
        )
        assert result_dm["label_encoder_params"]["extra_key"] == "should_persist"


# =========================================================================
# keep_existing_predictor forwarding through sklearn wrappers (issue #85)
# =========================================================================

# Factories used to exercise the flag being set to False. The tabular MLP
# encoder does not expose the ``fp_dim`` attribute that ``keep_existing_predictor=False``
# needs at the torch layer, so we restrict the False-value forwarding tests
# to GIN and CNN modalities. The True-value tests still cover all non-Chemprop
# modalities since that path is unchanged from the pre-#85 behavior.
_FLAG_FALSE_REGRESSOR_FACTORIES = [
    pytest.param(make_gin_regressor, id="GINRegressor"),
    pytest.param(make_cnn_regressor, id="CNNRegressor"),
]

_FLAG_FALSE_CLASSIFIER_FACTORIES = [
    pytest.param(make_gin_classifier, id="GINClassifier"),
    pytest.param(make_cnn_classifier, id="CNNClassifier"),
]

_FLAG_TRUE_REGRESSOR_FACTORIES = [
    pytest.param(make_mlp_regressor, id="MLPRegressor"),
    pytest.param(make_gin_regressor, id="GINRegressor"),
    pytest.param(make_cnn_regressor, id="CNNRegressor"),
]


class TestKeepExistingPredictorForwarding:
    """Verify keep_existing_predictor is threaded through the sklearn wrappers.

    The flag exists on ``Finetuner.__init__`` but was not accepted or forwarded
    by the sklearn wrappers before issue #85. These tests exercise the plumbing
    end-to-end: signature acceptance, hparams forwarding to the underlying
    torch module, a fit+predict smoke test, and the Chemprop guard.
    """

    @pytest.fixture(params=_FLAG_FALSE_REGRESSOR_FACTORIES)
    def flag_false_regressor_path(self, request, mol_list, regression_y, tmp_path):
        """Fit a regressor for which flag=False is supported, return its save path."""
        factory = request.param
        model = factory()
        save_dir = str(tmp_path / f"pretrained_kep_{factory.__name__}")
        model.fit(mol_list, regression_y)
        model.save_model(save_dir)
        return save_dir

    @pytest.fixture(params=_FLAG_FALSE_CLASSIFIER_FACTORIES)
    def flag_false_classifier_path(self, request, mol_list, classification_y, tmp_path):
        """Fit a classifier for which flag=False is supported, return its save path."""
        factory = request.param
        model = factory()
        save_dir = str(tmp_path / f"pretrained_kep_{factory.__name__}")
        model.fit(mol_list, classification_y)
        model.save_model(save_dir)
        return save_dir

    @pytest.fixture(params=_FLAG_TRUE_REGRESSOR_FACTORIES)
    def flag_true_regressor_path(self, request, mol_list, regression_y, tmp_path):
        """Fit any non-Chemprop regressor, return its save path."""
        factory = request.param
        model = factory()
        save_dir = str(tmp_path / f"pretrained_kep_{factory.__name__}")
        model.fit(mol_list, regression_y)
        model.save_model(save_dir)
        return save_dir

    @pytest.fixture()
    def chemprop_regressor_path(self, mol_list, regression_y, tmp_path):
        """Fit a Chemprop regressor, save it, return the save path."""
        model = make_chemprop_regressor()
        save_dir = str(tmp_path / "pretrained_kep_chemprop")
        model.fit(mol_list, regression_y)
        model.save_model(save_dir)
        return save_dir

    def test_regressor_construction_with_keep_existing_predictor_false(
        self, flag_false_regressor_path
    ):
        """FinetuningRegressor accepts keep_existing_predictor=False and forwards it."""
        finetuner = FinetuningRegressor(
            path_to_pretrained=flag_false_regressor_path,
            keep_existing_predictor=False,
            **FINETUNE_TRAIN,
        )
        assert finetuner._model.hparams["keep_existing_predictor"] is False

    def test_regressor_construction_with_keep_existing_predictor_true(
        self, flag_true_regressor_path
    ):
        """FinetuningRegressor forwards keep_existing_predictor=True (default)."""
        finetuner = FinetuningRegressor(
            path_to_pretrained=flag_true_regressor_path,
            keep_existing_predictor=True,
            **FINETUNE_TRAIN,
        )
        assert finetuner._model.hparams["keep_existing_predictor"] is True

    def test_classifier_construction_with_keep_existing_predictor_false(
        self, flag_false_classifier_path
    ):
        """FinetuningClassifier accepts keep_existing_predictor=False and forwards it."""
        finetuner = FinetuningClassifier(
            path_to_pretrained=flag_false_classifier_path,
            keep_existing_predictor=False,
            **FINETUNE_TRAIN,
        )
        assert finetuner._model.hparams["keep_existing_predictor"] is False

    def test_regressor_fit_predict_with_flag_false(
        self, mol_list, regression_y, tmp_path
    ):
        """End-to-end fit + predict works with keep_existing_predictor=False."""
        base = make_gin_regressor()
        save_dir = str(tmp_path / "pretrained_kep_smoke")
        base.fit(mol_list, regression_y)
        base.save_model(save_dir)

        finetuner = FinetuningRegressor(
            path_to_pretrained=save_dir,
            keep_existing_predictor=False,
            **FINETUNE_TRAIN,
        )
        finetuner.fit(mol_list, regression_y)
        preds = finetuner.predict(mol_list)

        assert preds.shape == (len(mol_list), 1)
        assert np.all(np.isfinite(preds))

    def test_chemprop_regressor_raises_on_flag_false(self, chemprop_regressor_path):
        """Chemprop pretrained models reject keep_existing_predictor=False."""
        with pytest.raises(ValueError, match="keep_existing_predictor=False"):
            FinetuningRegressor(
                path_to_pretrained=chemprop_regressor_path,
                keep_existing_predictor=False,
                **FINETUNE_TRAIN,
            )

    def test_chemprop_regressor_accepts_flag_true(self, chemprop_regressor_path):
        """Chemprop pretrained models accept the default keep_existing_predictor=True."""
        finetuner = FinetuningRegressor(
            path_to_pretrained=chemprop_regressor_path,
            keep_existing_predictor=True,
            **FINETUNE_TRAIN,
        )
        assert finetuner._model is not None
