"""Tests for ICP regression and classification calibration."""

import numpy as np
import pytest

from matcha.calibration.inductive_conformal import (
    ICPRegressionCalibration,
    ICPClassificationCalibration,
)


# ===================================================================
# ICPRegressionCalibration – initialisation
# ===================================================================


class TestICPRegressionInit:
    def test_default_alpha(self):
        cal = ICPRegressionCalibration()
        assert cal.params.confidence_alpha == pytest.approx(0.2)

    def test_custom_alpha(self):
        cal = ICPRegressionCalibration(confidence_alpha=0.05)
        assert cal.params.confidence_alpha == pytest.approx(0.05)

    def test_initial_quantile_zero(self):
        cal = ICPRegressionCalibration()
        assert cal.params.quantile == pytest.approx(0.0)

    def test_not_fitted_on_init(self):
        cal = ICPRegressionCalibration()
        assert cal.is_fitted is False


# ===================================================================
# ICPRegressionCalibration – nonconformity scores
# ===================================================================


class TestICPRegressionNonconformityScores:
    def test_scores_without_error(self):
        cal = ICPRegressionCalibration()
        y_true = np.array([[1.0], [2.0], [3.0]])
        y_pred = np.array([[1.5], [1.8], [3.5]])
        scores = cal._compute_nonconformity_scores(y_true, y_pred, y_error=None)
        expected = np.abs(y_true - y_pred)
        np.testing.assert_array_almost_equal(scores, expected)

    def test_scores_with_error_normalises(self):
        cal = ICPRegressionCalibration()
        y_true = np.array([[1.0], [2.0]])
        y_pred = np.array([[1.5], [2.5]])
        y_error = np.array([[0.5], [1.0]])
        scores = cal._compute_nonconformity_scores(y_true, y_pred, y_error)
        # |1-1.5|/(0.5+1e-6) ≈ 1.0, |2-2.5|/(1.0+1e-6) ≈ 0.5
        assert scores[0, 0] == pytest.approx(1.0, abs=1e-4)
        assert scores[1, 0] == pytest.approx(0.5, abs=1e-4)

    def test_scores_shape_matches_input_2d(self, regression_calibration_data):
        y_true, y_pred, y_error = regression_calibration_data
        cal = ICPRegressionCalibration()
        scores = cal._compute_nonconformity_scores(y_true, y_pred, y_error)
        assert scores.shape == y_true.shape

    def test_nan_in_ytrue_propagates(self):
        cal = ICPRegressionCalibration()
        y_true = np.array([[1.0], [np.nan], [3.0]])
        y_pred = np.array([[1.5], [2.0], [3.5]])
        scores = cal._compute_nonconformity_scores(y_true, y_pred, y_error=None)
        assert np.isnan(scores[1, 0])
        assert not np.isnan(scores[0, 0])


# ===================================================================
# ICPRegressionCalibration – quantile computation
# ===================================================================


class TestICPRegressionQuantile:
    def test_quantile_single_returns_nan_for_empty(self):
        cal = ICPRegressionCalibration(confidence_alpha=0.2)
        result = cal._compute_quantile_single(np.array([np.nan, np.nan]))
        assert np.isnan(result)

    def test_quantile_single_finite_sample_correction(self):
        """With alpha=0.2 and n=10, adjusted level = ceil(11*0.8)/10 = 0.9."""
        cal = ICPRegressionCalibration(confidence_alpha=0.2)
        scores = np.arange(1.0, 11.0)  # 1..10
        q = cal._compute_quantile_single(scores)
        # adjusted_level = ceil(11*0.8)/10 = ceil(8.8)/10 = 9/10 = 0.9
        expected = np.nanquantile(scores, 0.9, method="higher")
        assert q == pytest.approx(expected)

    def test_quantile_per_task(self, multitask_regression_calibration_data):
        y_true, y_pred, y_error = multitask_regression_calibration_data
        cal = ICPRegressionCalibration(confidence_alpha=0.2)
        quantiles = cal._compute_quantile(y_true, y_pred, y_error)
        assert quantiles.shape == (3,)
        assert all(np.isfinite(quantiles))


# ===================================================================
# ICPRegressionCalibration – fit
# ===================================================================


class TestICPRegressionFit:
    def test_fit_sets_is_fitted(self, regression_calibration_data):
        y_true, y_pred, y_error = regression_calibration_data
        cal = ICPRegressionCalibration()
        cal.fit(y_true, y_pred, y_error)
        assert cal.is_fitted is True

    def test_fit_sets_quantile(self, regression_calibration_data):
        y_true, y_pred, y_error = regression_calibration_data
        cal = ICPRegressionCalibration()
        cal.fit(y_true, y_pred, y_error)
        assert np.all(np.isfinite(cal.params.quantile))

    def test_fit_without_error(self, regression_calibration_data):
        y_true, y_pred, _ = regression_calibration_data
        cal = ICPRegressionCalibration()
        cal.fit(y_true, y_pred, y_error=None)
        assert cal.is_fitted is True

    def test_fit_1d_arrays(self, regression_calibration_data_1d):
        y_true, y_pred, y_error = regression_calibration_data_1d
        cal = ICPRegressionCalibration()
        cal.fit(y_true, y_pred, y_error)
        assert cal.is_fitted is True

    def test_fit_multitask(self, multitask_regression_calibration_data):
        y_true, y_pred, y_error = multitask_regression_calibration_data
        cal = ICPRegressionCalibration(confidence_alpha=0.1)
        cal.fit(y_true, y_pred, y_error)
        assert cal.params.quantile.shape == (3,)


# ===================================================================
# ICPRegressionCalibration – compute_uncertainty
# ===================================================================


class TestICPRegressionComputeUncertainty:
    def test_raises_if_not_fitted(self):
        cal = ICPRegressionCalibration()
        with pytest.raises(RuntimeError, match="must be fitted"):
            cal.compute_uncertainty(np.array([0.1, 0.2]))

    def test_output_shape_2d(self, regression_calibration_data):
        y_true, y_pred, y_error = regression_calibration_data
        cal = ICPRegressionCalibration()
        cal.fit(y_true, y_pred, y_error)
        result = cal.compute_uncertainty(y_error)
        assert result.shape == y_error.shape

    def test_output_shape_1d(self, regression_calibration_data_1d):
        y_true, y_pred, y_error = regression_calibration_data_1d
        cal = ICPRegressionCalibration()
        cal.fit(y_true, y_pred, y_error)
        result = cal.compute_uncertainty(y_error)
        assert result.ndim == 1
        assert result.shape == y_error.shape

    def test_output_scales_by_quantile(self):
        """Calibrated = input * quantile."""
        cal = ICPRegressionCalibration()
        y_true = np.array([[1.0], [2.0], [3.0]])
        y_pred = np.array([[1.1], [2.2], [3.3]])
        y_error = np.array([[0.5], [0.5], [0.5]])
        cal.fit(y_true, y_pred, y_error)
        q = cal.params.quantile
        result = cal.compute_uncertainty(y_error)
        np.testing.assert_array_almost_equal(result, y_error * q)

    def test_multitask_output_shape(self, multitask_regression_calibration_data):
        y_true, y_pred, y_error = multitask_regression_calibration_data
        cal = ICPRegressionCalibration()
        cal.fit(y_true, y_pred, y_error)
        result = cal.compute_uncertainty(y_error)
        assert result.shape == y_error.shape

    def test_zero_error_gives_zero_uncertainty(self, regression_calibration_data):
        y_true, y_pred, y_error = regression_calibration_data
        cal = ICPRegressionCalibration()
        cal.fit(y_true, y_pred, y_error)
        zero_error = np.zeros_like(y_error)
        result = cal.compute_uncertainty(zero_error)
        np.testing.assert_array_almost_equal(result, 0.0)


# ===================================================================
# ICPRegressionCalibration – coverage property
# ===================================================================


class TestICPRegressionCoverage:
    def test_coverage_approximately_meets_target(self, rng):
        """On held-out data, the empirical coverage should be close to 1-alpha."""
        n_cal, n_test = 500, 200
        alpha = 0.1

        y_true_all = rng.normal(0, 1, size=(n_cal + n_test, 1))
        noise = rng.normal(0, 0.5, size=(n_cal + n_test, 1))
        y_pred_all = y_true_all + noise
        y_error_all = np.abs(noise) + 0.01

        cal = ICPRegressionCalibration(confidence_alpha=alpha)
        cal.fit(y_true_all[:n_cal], y_pred_all[:n_cal], y_error_all[:n_cal])

        unc = cal.compute_uncertainty(y_error_all[n_cal:])
        covered = np.abs(y_true_all[n_cal:] - y_pred_all[n_cal:]) <= unc
        empirical_coverage = np.mean(covered)
        # Allow some slack
        assert empirical_coverage >= (1 - alpha) - 0.1


# ===================================================================
# ICPClassificationCalibration – initialisation
# ===================================================================


class TestICPClassificationInit:
    def test_default_alpha(self):
        cal = ICPClassificationCalibration()
        assert cal.params.confidence_alpha == pytest.approx(0.2)

    def test_custom_alpha(self):
        cal = ICPClassificationCalibration(confidence_alpha=0.1)
        assert cal.params.confidence_alpha == pytest.approx(0.1)

    def test_not_fitted_on_init(self):
        cal = ICPClassificationCalibration()
        assert cal.is_fitted is False

    def test_calibration_scores_none_on_init(self):
        cal = ICPClassificationCalibration()
        assert cal._calibration_scores is None


# ===================================================================
# ICPClassificationCalibration – nonconformity scores
# ===================================================================


class TestICPClassificationNonconformityScores:
    def test_perfect_prediction_class_1(self):
        """If y_true=1 and y_pred=1.0, score = 1 - 1.0 = 0.0."""
        cal = ICPClassificationCalibration()
        y_true = np.array([[1.0]])
        y_pred = np.array([[1.0]])
        scores = cal._compute_nonconformity_scores(y_true, y_pred)
        assert scores[0, 0] == pytest.approx(0.0)

    def test_perfect_prediction_class_0(self):
        """If y_true=0 and y_pred=0.0, P(true)=1-0=1, score=0."""
        cal = ICPClassificationCalibration()
        y_true = np.array([[0.0]])
        y_pred = np.array([[0.0]])
        scores = cal._compute_nonconformity_scores(y_true, y_pred)
        assert scores[0, 0] == pytest.approx(0.0)

    def test_wrong_prediction_class_1(self):
        """If y_true=1 and y_pred=0.2, score = 1 - 0.2 = 0.8."""
        cal = ICPClassificationCalibration()
        y_true = np.array([[1.0]])
        y_pred = np.array([[0.2]])
        scores = cal._compute_nonconformity_scores(y_true, y_pred)
        assert scores[0, 0] == pytest.approx(0.8)

    def test_wrong_prediction_class_0(self):
        """If y_true=0 and y_pred=0.9, P(true)=1-0.9=0.1, score=0.9."""
        cal = ICPClassificationCalibration()
        y_true = np.array([[0.0]])
        y_pred = np.array([[0.9]])
        scores = cal._compute_nonconformity_scores(y_true, y_pred)
        assert scores[0, 0] == pytest.approx(0.9)

    def test_nan_propagation(self):
        cal = ICPClassificationCalibration()
        y_true = np.array([[1.0], [np.nan], [0.0]])
        y_pred = np.array([[0.8], [0.5], [0.2]])
        scores = cal._compute_nonconformity_scores(y_true, y_pred)
        assert not np.isnan(scores[0, 0])
        assert np.isnan(scores[1, 0])
        assert not np.isnan(scores[2, 0])

    def test_1d_input_made_2d(self):
        cal = ICPClassificationCalibration()
        y_true = np.array([1.0, 0.0])
        y_pred = np.array([0.8, 0.3])
        scores = cal._compute_nonconformity_scores(y_true, y_pred)
        assert scores.ndim == 2
        assert scores.shape == (2, 1)

    def test_scores_between_zero_and_one(self, classification_calibration_data):
        y_true, y_pred = classification_calibration_data
        cal = ICPClassificationCalibration()
        scores = cal._compute_nonconformity_scores(y_true, y_pred)
        valid = scores[~np.isnan(scores)]
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 1.0)


# ===================================================================
# ICPClassificationCalibration – fit
# ===================================================================


class TestICPClassificationFit:
    def test_fit_sets_is_fitted(self, classification_calibration_data):
        y_true, y_pred = classification_calibration_data
        cal = ICPClassificationCalibration()
        cal.fit(y_true, y_pred)
        assert cal.is_fitted is True

    def test_fit_stores_calibration_scores(self, classification_calibration_data):
        y_true, y_pred = classification_calibration_data
        cal = ICPClassificationCalibration()
        cal.fit(y_true, y_pred)
        assert cal._calibration_scores is not None
        assert len(cal._calibration_scores) == 1  # 1 task

    def test_fit_1d_input(self, classification_calibration_data_1d):
        y_true, y_pred = classification_calibration_data_1d
        cal = ICPClassificationCalibration()
        cal.fit(y_true, y_pred)
        assert cal.is_fitted is True

    def test_fit_multitask(self, multitask_classification_calibration_data):
        y_true, y_pred = multitask_classification_calibration_data
        cal = ICPClassificationCalibration()
        cal.fit(y_true, y_pred)
        assert len(cal._calibration_scores) == 2

    def test_fit_multitask_nan_excluded(
        self, multitask_classification_calibration_data
    ):
        y_true, y_pred = multitask_classification_calibration_data
        cal = ICPClassificationCalibration()
        cal.fit(y_true, y_pred)
        # Task 0 has 1 NaN (index 7), task 1 has 1 NaN (index 2)
        n_valid_0 = int(np.sum(~np.isnan(y_true[:, 0])))
        n_valid_1 = int(np.sum(~np.isnan(y_true[:, 1])))
        assert len(cal._calibration_scores[0]) == n_valid_0
        assert len(cal._calibration_scores[1]) == n_valid_1


# ===================================================================
# ICPClassificationCalibration – compute_uncertainty (p-values)
# ===================================================================


class TestICPClassificationComputeUncertainty:
    def test_raises_if_not_fitted(self):
        cal = ICPClassificationCalibration()
        with pytest.raises(RuntimeError, match="must be fitted"):
            cal.compute_uncertainty(np.array([0.5]))

    def test_output_shape(self, classification_calibration_data):
        y_true, y_pred = classification_calibration_data
        cal = ICPClassificationCalibration()
        cal.fit(y_true, y_pred)
        pv = cal.compute_uncertainty(y_pred)
        assert pv.shape == (y_pred.shape[0], 1)

    def test_p_values_between_zero_and_one(self, classification_calibration_data):
        y_true, y_pred = classification_calibration_data
        cal = ICPClassificationCalibration()
        cal.fit(y_true, y_pred)
        pv = cal.compute_uncertainty(y_pred)
        valid = pv[~np.isnan(pv)]
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 1.0)

    def test_high_prob_class1_gives_high_pvalue(self):
        """A prediction near 1.0 should have high p-value for class 1."""
        cal = ICPClassificationCalibration(confidence_alpha=0.2)
        # All calibration labels are class 1 with good predictions
        y_true = np.array([[1.0]] * 20)
        y_pred = np.linspace(0.6, 0.95, 20).reshape(-1, 1)
        cal.fit(y_true, y_pred)
        # Test with very high probability → low nonconformity for class 1
        test_pred = np.array([[0.99]])
        pv = cal.compute_uncertainty(test_pred)
        assert pv[0, 0] > 0.5

    def test_task_mismatch_raises(self, classification_calibration_data):
        y_true, y_pred = classification_calibration_data
        cal = ICPClassificationCalibration()
        cal.fit(y_true, y_pred)  # 1 task
        with pytest.raises(ValueError, match="Number of tasks"):
            cal.compute_uncertainty(np.array([[0.5, 0.5]]))  # 2 tasks


# ===================================================================
# ICPClassificationCalibration – p-values both classes
# ===================================================================


class TestICPClassificationPValuesBothClasses:
    def test_raises_if_not_fitted(self):
        cal = ICPClassificationCalibration()
        with pytest.raises(RuntimeError, match="must be fitted"):
            cal.compute_p_values_both_classes(np.array([0.5]))

    def test_returns_two_arrays(self, classification_calibration_data):
        y_true, y_pred = classification_calibration_data
        cal = ICPClassificationCalibration()
        cal.fit(y_true, y_pred)
        pv0, pv1 = cal.compute_p_values_both_classes(y_pred)
        assert pv0.shape == pv1.shape
        assert pv0.shape == (y_pred.shape[0], 1)

    def test_both_p_values_in_range(self, classification_calibration_data):
        y_true, y_pred = classification_calibration_data
        cal = ICPClassificationCalibration()
        cal.fit(y_true, y_pred)
        pv0, pv1 = cal.compute_p_values_both_classes(y_pred)
        for pv in [pv0, pv1]:
            valid = pv[~np.isnan(pv)]
            assert np.all(valid >= 0.0)
            assert np.all(valid <= 1.0)


# ===================================================================
# ICPClassificationCalibration – predict_sets
# ===================================================================


class TestICPClassificationPredictSets:
    def test_raises_if_not_fitted(self):
        cal = ICPClassificationCalibration()
        with pytest.raises(RuntimeError, match="must be fitted"):
            cal.predict_sets(np.array([0.5]))

    def test_output_is_ndarray(self, classification_calibration_data):
        y_true, y_pred = classification_calibration_data
        cal = ICPClassificationCalibration()
        cal.fit(y_true, y_pred)
        result = cal.predict_sets(y_pred)
        assert isinstance(result, np.ndarray)
        assert result.shape == (y_pred.shape[0], 1)

    def test_values_in_expected_set(self, classification_calibration_data):
        y_true, y_pred = classification_calibration_data
        cal = ICPClassificationCalibration()
        cal.fit(y_true, y_pred)
        result = cal.predict_sets(y_pred)
        assert set(np.unique(result)).issubset({-1, 0, 1})

    def test_custom_alpha_override(self, classification_calibration_data):
        y_true, y_pred = classification_calibration_data
        cal = ICPClassificationCalibration(confidence_alpha=0.2)
        cal.fit(y_true, y_pred)
        # Very small alpha → most predictions should be uncertain (0 = both classes)
        result = cal.predict_sets(y_pred, alpha=0.001)
        uncertain_count = np.sum(result == 0)
        assert uncertain_count > 0

    def test_very_high_alpha_gives_unusual(self, classification_calibration_data):
        y_true, y_pred = classification_calibration_data
        cal = ICPClassificationCalibration()
        cal.fit(y_true, y_pred)
        result = cal.predict_sets(y_pred, alpha=0.999)
        unusual_count = np.sum(result == -1)
        # At extreme alpha at least some predictions should be unusual
        assert unusual_count > 0

    def test_multitask_predict_sets(self, multitask_classification_calibration_data):
        y_true, y_pred = multitask_classification_calibration_data
        cal = ICPClassificationCalibration()
        cal.fit(y_true, y_pred)
        result = cal.predict_sets(y_pred)
        assert result.shape == (y_pred.shape[0], 2)  # 2 tasks
        assert set(np.unique(result)).issubset({-1, 0, 1})
