"""Tests for Error Model calibration (regression and classification)."""

import numpy as np
import pytest

from matcha.calibration.error_model import (
    EMCalibration,
    EMRegressionCalibration,
    EMClassificationCalibration,
    EM_REGISTRY,
)


# ===================================================================
# EM_REGISTRY
# ===================================================================


class TestEMRegistry:
    def test_contains_gradient_boosting(self):
        assert "GradientBoostingClassifier" in EM_REGISTRY

    def test_contains_random_forest(self):
        assert "RandomForestClassifier" in EM_REGISTRY

    def test_contains_kneighbors(self):
        assert "KNeighborsClassifier" in EM_REGISTRY

    def test_contains_logistic_regression(self):
        assert "LogisticRegression" in EM_REGISTRY


# ===================================================================
# EMCalibration – abstract contract
# ===================================================================


class TestEMCalibrationIsAbstract:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            EMCalibration()


# ===================================================================
# EMRegressionCalibration – initialisation
# ===================================================================


class TestEMRegressionInit:
    def test_default_params(self):
        cal = EMRegressionCalibration()
        assert cal.params.algorithm == "GradientBoostingClassifier"
        assert cal.params.fold == pytest.approx(2.0)
        assert cal.params.log10 is False
        assert cal.params.use_ecfp is True
        assert cal.params.use_interaction is True
        assert cal.params.min_compounds == 50

    def test_custom_fold(self):
        cal = EMRegressionCalibration(fold=3.0)
        assert cal.params.fold == pytest.approx(3.0)

    def test_custom_algorithm(self):
        cal = EMRegressionCalibration(algorithm="RandomForestClassifier")
        assert cal.params.algorithm == "RandomForestClassifier"

    def test_log10_mode(self):
        cal = EMRegressionCalibration(log10=True)
        assert cal.params.log10 is True

    def test_not_fitted_on_init(self):
        cal = EMRegressionCalibration()
        assert cal.is_fitted is False

    def test_empty_model_box_on_init(self):
        cal = EMRegressionCalibration()
        assert cal.model_box == []

    def test_none_algorithm_params_defaults_to_empty(self):
        cal = EMRegressionCalibration(algorithm_params=None)
        assert cal.params.algorithm_params == {}

    def test_custom_algorithm_params(self):
        cal = EMRegressionCalibration(algorithm_params={"n_estimators": 50})
        assert cal.params.algorithm_params == {"n_estimators": 50}


# ===================================================================
# EMRegressionCalibration – _compute_error_mask (linear space)
# ===================================================================


class TestEMRegressionErrorMaskLinear:
    def test_within_fold(self):
        cal = EMRegressionCalibration(fold=2.0, log10=False)
        y_true = np.array([10.0])
        y_pred = np.array([15.0])  # ratio 1.5 ∈ [0.5, 2.0]
        mask = cal._compute_error_mask(y_true, y_pred)
        assert mask[0] is np.True_

    def test_outside_fold_high(self):
        cal = EMRegressionCalibration(fold=2.0, log10=False)
        y_true = np.array([10.0])
        y_pred = np.array([25.0])  # ratio 2.5 > 2.0
        mask = cal._compute_error_mask(y_true, y_pred)
        assert mask[0] is np.False_

    def test_outside_fold_low(self):
        cal = EMRegressionCalibration(fold=2.0, log10=False)
        y_true = np.array([10.0])
        y_pred = np.array([3.0])  # ratio 0.3 < 0.5
        mask = cal._compute_error_mask(y_true, y_pred)
        assert mask[0] is np.False_

    def test_exact_boundary_upper(self):
        cal = EMRegressionCalibration(fold=2.0, log10=False)
        y_true = np.array([10.0])
        y_pred = np.array([20.0])  # ratio exactly 2.0
        mask = cal._compute_error_mask(y_true, y_pred)
        assert mask[0] is np.True_

    def test_exact_boundary_lower(self):
        cal = EMRegressionCalibration(fold=2.0, log10=False)
        y_true = np.array([10.0])
        y_pred = np.array([5.0])  # ratio 0.5 == 1/2
        mask = cal._compute_error_mask(y_true, y_pred)
        assert mask[0] is np.True_

    def test_multiple_samples(self):
        cal = EMRegressionCalibration(fold=2.0, log10=False)
        y_true = np.array([10.0, 10.0, 10.0])
        y_pred = np.array([15.0, 25.0, 5.0])
        mask = cal._compute_error_mask(y_true, y_pred)
        expected = np.array([True, False, True])
        np.testing.assert_array_equal(mask, expected)

    def test_flattens_2d(self):
        cal = EMRegressionCalibration(fold=2.0, log10=False)
        y_true = np.array([[10.0], [10.0]])
        y_pred = np.array([[15.0], [25.0]])
        mask = cal._compute_error_mask(y_true, y_pred)
        assert mask.ndim == 1


# ===================================================================
# EMRegressionCalibration – _compute_error_mask (log10 space)
# ===================================================================


class TestEMRegressionErrorMaskLog10:
    def test_within_fold_log10(self):
        cal = EMRegressionCalibration(fold=2.0, log10=True)
        y_true = np.array([1.0])
        y_pred = np.array([1.2])  # diff 0.2 <= log10(2) ≈ 0.301
        mask = cal._compute_error_mask(y_true, y_pred)
        assert mask[0] is np.True_

    def test_outside_fold_log10(self):
        cal = EMRegressionCalibration(fold=2.0, log10=True)
        y_true = np.array([1.0])
        y_pred = np.array([1.5])  # diff 0.5 > log10(2) ≈ 0.301
        mask = cal._compute_error_mask(y_true, y_pred)
        assert mask[0] is np.False_

    def test_exact_boundary_log10(self):
        """At exactly the fold boundary, floating-point rounding may push
        the residual a tiny bit above log10(fold).  We therefore test a
        value that is clearly inside the boundary instead."""
        cal = EMRegressionCalibration(fold=2.0, log10=True)
        y_true = np.array([1.0])
        # Slightly inside the boundary to avoid fp rounding
        y_pred = np.array([1.0 + np.log10(2.0) - 1e-9])
        mask = cal._compute_error_mask(y_true, y_pred)
        assert mask[0] is np.True_


# ===================================================================
# EMClassificationCalibration – initialisation
# ===================================================================


class TestEMClassificationInit:
    def test_default_params(self):
        cal = EMClassificationCalibration()
        assert cal.params.error_threshold == pytest.approx(0.5)
        assert cal.params.algorithm == "GradientBoostingClassifier"

    def test_custom_threshold(self):
        cal = EMClassificationCalibration(error_threshold=0.3)
        assert cal.params.error_threshold == pytest.approx(0.3)

    def test_not_fitted_on_init(self):
        cal = EMClassificationCalibration()
        assert cal.is_fitted is False


# ===================================================================
# EMClassificationCalibration – _compute_error_mask
# ===================================================================


class TestEMClassificationErrorMask:
    def test_within_threshold(self):
        cal = EMClassificationCalibration(error_threshold=0.5)
        y_true = np.array([1.0])
        y_pred = np.array([0.6])  # gap 0.4 <= 0.5
        mask = cal._compute_error_mask(y_true, y_pred)
        assert mask[0] is np.True_

    def test_outside_threshold(self):
        cal = EMClassificationCalibration(error_threshold=0.3)
        y_true = np.array([1.0])
        y_pred = np.array([0.2])  # gap 0.8 > 0.3
        mask = cal._compute_error_mask(y_true, y_pred)
        assert mask[0] is np.False_

    def test_exact_boundary(self):
        cal = EMClassificationCalibration(error_threshold=0.5)
        y_true = np.array([1.0])
        y_pred = np.array([0.5])  # gap exactly 0.5
        mask = cal._compute_error_mask(y_true, y_pred)
        assert mask[0] is np.True_

    def test_class_0_predictions(self):
        cal = EMClassificationCalibration(error_threshold=0.5)
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([0.3, 0.8])
        mask = cal._compute_error_mask(y_true, y_pred)
        expected = np.array([True, False])
        np.testing.assert_array_equal(mask, expected)

    def test_flattens_2d(self):
        cal = EMClassificationCalibration(error_threshold=0.5)
        y_true = np.array([[1.0], [0.0]])
        y_pred = np.array([[0.8], [0.2]])
        mask = cal._compute_error_mask(y_true, y_pred)
        assert mask.ndim == 1


# ===================================================================
# EMRegressionCalibration – _feature_engineer
# ===================================================================


class TestEMFeatureEngineer:
    def test_no_ecfp_no_interaction(self, small_mol_list):
        cal = EMRegressionCalibration(use_ecfp=False, use_interaction=False)
        preds = np.random.default_rng(0).normal(size=(len(small_mol_list), 1))
        std = np.abs(preds) + 0.01
        features = cal._feature_engineer(small_mol_list, preds, std)
        # Should be concat of preds and std → 2 columns
        assert features.shape == (len(small_mol_list), 2)

    def test_with_interaction_no_ecfp(self, small_mol_list):
        cal = EMRegressionCalibration(use_ecfp=False, use_interaction=True)
        n = len(small_mol_list)
        preds = np.random.default_rng(0).normal(size=(n, 1))
        std = np.abs(preds) + 0.01
        features = cal._feature_engineer(small_mol_list, preds, std)
        # preds(1) + std(1) + interaction(1*1=1) = 3
        assert features.shape == (n, 3)

    def test_with_ecfp_no_interaction(self, small_mol_list):
        cal = EMRegressionCalibration(use_ecfp=True, use_interaction=False)
        n = len(small_mol_list)
        preds = np.random.default_rng(0).normal(size=(n, 1))
        std = np.abs(preds) + 0.01
        features = cal._feature_engineer(small_mol_list, preds, std)
        # preds(1) + std(1) + ecfp(2048) = 2050
        assert features.shape[0] == n
        assert features.shape[1] > 2  # ecfp adds many features

    def test_multitask_interaction_shape(self, small_mol_list):
        cal = EMRegressionCalibration(use_ecfp=False, use_interaction=True)
        n = len(small_mol_list)
        n_tasks = 3
        preds = np.random.default_rng(0).normal(size=(n, n_tasks))
        std = np.abs(preds) + 0.01
        features = cal._feature_engineer(small_mol_list, preds, std)
        # preds(3) + std(3) + interaction(3*3=9) = 15
        assert features.shape == (n, 15)


# ===================================================================
# EMRegressionCalibration – _check_is_fitted
# ===================================================================


class TestEMCheckIsFitted:
    def test_raises_when_not_fitted(self):
        cal = EMRegressionCalibration()
        with pytest.raises(RuntimeError, match="must be fitted"):
            cal._check_is_fitted()

    def test_no_raise_when_fitted(self, em_regression_data):
        mols, y_true, y_pred, y_error = em_regression_data
        cal = EMRegressionCalibration(
            use_ecfp=False, use_interaction=False, min_compounds=5
        )
        cal.fit(mols, y_true, y_pred, y_error)
        cal._check_is_fitted()  # should not raise


# ===================================================================
# EMRegressionCalibration – fit (end-to-end, no ECFP for speed)
# ===================================================================


class TestEMRegressionFit:
    def test_fit_sets_is_fitted(self, em_regression_data):
        mols, y_true, y_pred, y_error = em_regression_data
        cal = EMRegressionCalibration(
            use_ecfp=False, use_interaction=False, min_compounds=5
        )
        cal.fit(mols, y_true, y_pred, y_error)
        assert cal.is_fitted is True

    def test_fit_populates_model_box(self, em_regression_data):
        mols, y_true, y_pred, y_error = em_regression_data
        cal = EMRegressionCalibration(
            use_ecfp=False, use_interaction=False, min_compounds=5
        )
        cal.fit(mols, y_true, y_pred, y_error)
        assert len(cal.model_box) == 1  # 1 task

    def test_fit_populates_scaler_box(self, em_regression_data):
        mols, y_true, y_pred, y_error = em_regression_data
        cal = EMRegressionCalibration(
            use_ecfp=False, use_interaction=False, min_compounds=5
        )
        cal.fit(mols, y_true, y_pred, y_error)
        assert len(cal.scaler_box) == 1

    def test_fit_records_error_ratio(self, em_regression_data):
        mols, y_true, y_pred, y_error = em_regression_data
        cal = EMRegressionCalibration(
            use_ecfp=False, use_interaction=False, min_compounds=5
        )
        cal.fit(mols, y_true, y_pred, y_error)
        assert len(cal.error_ratio) == 1
        assert 0.0 <= cal.error_ratio[0] <= 1.0

    def test_fit_records_n_compounds(self, em_regression_data):
        mols, y_true, y_pred, y_error = em_regression_data
        cal = EMRegressionCalibration(
            use_ecfp=False, use_interaction=False, min_compounds=5
        )
        cal.fit(mols, y_true, y_pred, y_error)
        assert cal.n_compounds[0] == len(mols)

    def test_fit_1d_arrays(self, em_regression_data):
        mols, y_true, y_pred, y_error = em_regression_data
        cal = EMRegressionCalibration(
            use_ecfp=False, use_interaction=False, min_compounds=5
        )
        cal.fit(mols, y_true.ravel(), y_pred.ravel(), y_error.ravel())
        assert cal.is_fitted is True

    def test_fit_model_none_when_all_same_class(self, rng, small_mol_list):
        """If all error masks are True, model should be None."""
        n = len(small_mol_list)
        y_true = np.ones((n, 1)) * 10.0
        y_pred = np.ones((n, 1)) * 10.0  # perfect → all within fold
        y_error = np.ones((n, 1)) * 0.1
        cal = EMRegressionCalibration(
            use_ecfp=False, use_interaction=False, min_compounds=1, fold=2.0
        )
        cal.fit(small_mol_list, y_true, y_pred, y_error)
        assert cal.model_box[0] is None

    def test_fit_model_none_when_insufficient_compounds(self, rng, small_mol_list):
        """If fewer compounds than min_compounds, model can still be trained but
        compute_uncertainty will return -2."""
        n = len(small_mol_list)
        y_true = rng.normal(5, 2, size=(n, 1))
        y_pred = y_true + rng.normal(0, 1, size=(n, 1))
        y_error = np.abs(rng.normal(0, 0.5, size=(n, 1))) + 0.01
        cal = EMRegressionCalibration(
            use_ecfp=False, use_interaction=False, min_compounds=1000
        )
        cal.fit(small_mol_list, y_true, y_pred, y_error)
        # n < 1000, so even if mask is mixed, the condition
        # len(y_true_i) >= min_compounds fails → model is None
        assert cal.model_box[0] is None

    def test_fit_with_random_forest(self, em_regression_data):
        mols, y_true, y_pred, y_error = em_regression_data
        cal = EMRegressionCalibration(
            algorithm="RandomForestClassifier",
            use_ecfp=False,
            use_interaction=False,
            min_compounds=5,
        )
        cal.fit(mols, y_true, y_pred, y_error)
        assert cal.is_fitted is True

    def test_fit_with_kneighbors(self, em_regression_data):
        """KNeighborsClassifier has no random_state param – tests the except branch."""
        mols, y_true, y_pred, y_error = em_regression_data
        cal = EMRegressionCalibration(
            algorithm="KNeighborsClassifier",
            use_ecfp=False,
            use_interaction=False,
            min_compounds=5,
        )
        cal.fit(mols, y_true, y_pred, y_error)
        assert cal.is_fitted is True


# ===================================================================
# EMRegressionCalibration – compute_uncertainty
# ===================================================================


class TestEMRegressionComputeUncertainty:
    def test_raises_if_not_fitted(self, small_mol_list):
        cal = EMRegressionCalibration()
        with pytest.raises(RuntimeError, match="must be fitted"):
            n = len(small_mol_list)
            cal.compute_uncertainty(
                small_mol_list,
                np.zeros((n, 1)),
                np.zeros((n, 1)),
            )

    def test_output_shape(self, em_regression_data):
        mols, y_true, y_pred, y_error = em_regression_data
        cal = EMRegressionCalibration(
            use_ecfp=False, use_interaction=False, min_compounds=5
        )
        cal.fit(mols, y_true, y_pred, y_error)
        result = cal.compute_uncertainty(mols, y_pred, y_error)
        assert result.shape == (len(mols), 1)

    def test_output_probabilities_in_range(self, em_regression_data):
        mols, y_true, y_pred, y_error = em_regression_data
        cal = EMRegressionCalibration(
            use_ecfp=False, use_interaction=False, min_compounds=5
        )
        cal.fit(mols, y_true, y_pred, y_error)
        result = cal.compute_uncertainty(mols, y_pred, y_error)
        # Values should be probabilities (or special -1, -2)
        for v in result.ravel():
            assert (
                v == pytest.approx(-2.0)
                or v == pytest.approx(-1.0)
                or (0.0 <= v <= 1.0)
            )

    def test_returns_minus2_when_insufficient_data(self, rng, small_mol_list):
        n = len(small_mol_list)
        y_true = rng.normal(5, 2, size=(n, 1))
        y_pred = y_true + rng.normal(0, 1, size=(n, 1))
        y_error = np.abs(rng.normal(0, 0.5, size=(n, 1))) + 0.01
        cal = EMRegressionCalibration(
            use_ecfp=False, use_interaction=False, min_compounds=1000
        )
        cal.fit(small_mol_list, y_true, y_pred, y_error)
        result = cal.compute_uncertainty(small_mol_list, y_pred, y_error)
        np.testing.assert_array_equal(result, -2.0)

    def test_returns_minus1_when_all_same_class(self, small_mol_list):
        n = len(small_mol_list)
        y_true = np.ones((n, 1)) * 10.0
        y_pred = np.ones((n, 1)) * 10.0
        y_error = np.ones((n, 1)) * 0.1
        cal = EMRegressionCalibration(
            use_ecfp=False, use_interaction=False, min_compounds=1
        )
        cal.fit(small_mol_list, y_true, y_pred, y_error)
        result = cal.compute_uncertainty(small_mol_list, y_pred, y_error)
        np.testing.assert_array_equal(result, -1.0)

    def test_1d_input(self, em_regression_data):
        mols, y_true, y_pred, y_error = em_regression_data
        cal = EMRegressionCalibration(
            use_ecfp=False, use_interaction=False, min_compounds=5
        )
        cal.fit(mols, y_true, y_pred, y_error)
        result = cal.compute_uncertainty(mols, y_pred.ravel(), y_error.ravel())
        assert result.shape == (len(mols), 1)


# ===================================================================
# EMClassificationCalibration – fit (end-to-end, no ECFP for speed)
# ===================================================================


class TestEMClassificationFit:
    def test_fit_sets_is_fitted(self, em_classification_data):
        mols, y_true, y_pred, y_error = em_classification_data
        cal = EMClassificationCalibration(
            use_ecfp=False, use_interaction=False, min_compounds=5
        )
        cal.fit(mols, y_true, y_pred, y_error)
        assert cal.is_fitted is True

    def test_fit_populates_model_box(self, em_classification_data):
        mols, y_true, y_pred, y_error = em_classification_data
        cal = EMClassificationCalibration(
            use_ecfp=False, use_interaction=False, min_compounds=5
        )
        cal.fit(mols, y_true, y_pred, y_error)
        assert len(cal.model_box) == 1


# ===================================================================
# EMClassificationCalibration – compute_uncertainty
# ===================================================================


class TestEMClassificationComputeUncertainty:
    def test_output_shape(self, em_classification_data):
        mols, y_true, y_pred, y_error = em_classification_data
        cal = EMClassificationCalibration(
            use_ecfp=False, use_interaction=False, min_compounds=5
        )
        cal.fit(mols, y_true, y_pred, y_error)
        result = cal.compute_uncertainty(mols, y_pred, y_error)
        assert result.shape == (len(mols), 1)

    def test_output_values_valid(self, em_classification_data):
        mols, y_true, y_pred, y_error = em_classification_data
        cal = EMClassificationCalibration(
            use_ecfp=False, use_interaction=False, min_compounds=5
        )
        cal.fit(mols, y_true, y_pred, y_error)
        result = cal.compute_uncertainty(mols, y_pred, y_error)
        for v in result.ravel():
            assert (
                v == pytest.approx(-2.0)
                or v == pytest.approx(-1.0)
                or (0.0 <= v <= 1.0)
            )


# ===================================================================
# EMRegressionCalibration – fit with ECFP (integration test)
# ===================================================================


class TestEMRegressionWithECFP:
    def test_fit_with_ecfp(self, em_regression_data):
        mols, y_true, y_pred, y_error = em_regression_data
        cal = EMRegressionCalibration(
            use_ecfp=True, use_interaction=False, min_compounds=5
        )
        cal.fit(mols, y_true, y_pred, y_error)
        assert cal.is_fitted is True

    def test_predict_with_ecfp(self, em_regression_data):
        mols, y_true, y_pred, y_error = em_regression_data
        cal = EMRegressionCalibration(
            use_ecfp=True, use_interaction=False, min_compounds=5
        )
        cal.fit(mols, y_true, y_pred, y_error)
        result = cal.compute_uncertainty(mols, y_pred, y_error)
        assert result.shape == (len(mols), 1)


# ===================================================================
# EMRegressionCalibration – NaN handling in multitask
# ===================================================================


class TestEMRegressionNaNHandling:
    def test_nan_rows_excluded_per_task(self, rng, mol_list):
        n = len(mol_list)
        y_true = rng.normal(5, 2, size=(n, 2))
        y_pred = y_true + rng.normal(0, 1, size=(n, 2))
        y_error = np.abs(rng.normal(0, 0.5, size=(n, 2))) + 0.01
        # Put NaN in task 1 for first 10 samples
        y_true[:10, 1] = np.nan

        cal = EMRegressionCalibration(
            use_ecfp=False, use_interaction=False, min_compounds=5
        )
        cal.fit(mol_list, y_true, y_pred, y_error)

        assert cal.n_compounds[0] == n
        assert cal.n_compounds[1] == n - 10
