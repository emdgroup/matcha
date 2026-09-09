"""End-to-end tests for MVE regression through the sklearn API.

Covers the intrinsic-uncertainty pathway (``uncertainty="mve"``) across
one representative regressor per model family (MLP, GIN, E3GNN, RoFormer):
predict-shape parity with non-MVE, multitask, bounded-beta-nll on censored
labels, uncertainty non-negativity, save/load round-trip, calibration,
symmetric validation errors, and ensemble law-of-total-variance
aggregation.
"""

from pathlib import Path

import lightning as L
import numpy as np
import pytest
from pydantic import ValidationError

from matcha.sklearn import Ensemble, autoload
from matcha.sklearn.tabular import MLPClassifier, MLPRegressor, SNNRegressor
from matcha.sklearn.graph import ChempropRegressor

from .conftest import _ARCH_KWARGS, _MVE_KWARGS, _MVE_KWARGS_CHEMPROP

_MVE_BASELINE_PATH = Path(__file__).parent / "mve_baseline.npz"


# =========================================================================
# Local fixtures — multitask labels and bounded regression setup
# =========================================================================


@pytest.fixture(scope="session")
def multitask_regression_y_t3(regression_y) -> np.ndarray:
    """(N, 3) multitask regression labels — original + two noisy copies."""
    rng = np.random.default_rng(0)
    n1 = rng.normal(0, 0.1, size=regression_y.shape)
    n2 = rng.normal(0, 0.2, size=regression_y.shape)
    return np.hstack([regression_y, regression_y + n1, regression_y + n2])


@pytest.fixture()
def bound_mask_mixed(mol_list) -> list[str]:
    """Bound mask with a mix of exact and censored values, length matches mol_list."""
    masks = []
    for i in range(len(mol_list)):
        if i % 5 == 0:
            masks.append("<")
        elif i % 7 == 0:
            masks.append(">")
        else:
            masks.append("=")
    return masks


# =========================================================================
# Predict shape parity (single-task) — one per family
# =========================================================================


class TestMVEPredictShape:
    """Predict output shape must match a non-MVE model with the same num_endpoints."""

    def test_predict_returns_ndarray(self, fitted_mve_regressor, mol_list):
        preds = fitted_mve_regressor.predict(mol_list)
        assert isinstance(preds, np.ndarray)

    def test_predict_single_task_shape(self, fitted_mve_regressor, mol_list):
        preds = fitted_mve_regressor.predict(mol_list)
        assert preds.shape == (len(mol_list), 1)

    def test_predict_values_finite(self, fitted_mve_regressor, mol_list):
        preds = fitted_mve_regressor.predict(mol_list)
        assert np.all(np.isfinite(preds))


# =========================================================================
# Compute-uncertainty shape / values (single-task) — one per family
# =========================================================================


class TestMVEComputeUncertainty:
    """``compute_uncertainty`` returns non-negative, finite std with the same shape as predict."""

    def test_std_shape_matches_predict(self, fitted_mve_regressor, mol_list):
        std = fitted_mve_regressor.compute_uncertainty(mol_list)
        assert std.shape == (len(mol_list), 1)

    def test_std_non_negative(self, fitted_mve_regressor, mol_list):
        std = fitted_mve_regressor.compute_uncertainty(mol_list)
        assert np.all(std >= 0.0)

    def test_std_finite(self, fitted_mve_regressor, mol_list):
        std = fitted_mve_regressor.compute_uncertainty(mol_list)
        assert np.all(np.isfinite(std))


# =========================================================================
# Multitask (T=3) — MLP only (keeps CI runtime bounded)
# =========================================================================


class TestMVEMultitask:
    """Multitask MVE returns per-task means and per-task variances."""

    @pytest.fixture()
    def fitted_multitask_mve(self, mol_list, multitask_regression_y_t3):
        model = MLPRegressor(
            **_ARCH_KWARGS[MLPRegressor],
            **_MVE_KWARGS,
            num_endpoints=3,
        )
        model.fit(mol_list, multitask_regression_y_t3)
        return model

    def test_multitask_predict_shape(self, fitted_multitask_mve, mol_list):
        preds = fitted_multitask_mve.predict(mol_list)
        assert preds.shape == (len(mol_list), 3)

    def test_multitask_std_shape(self, fitted_multitask_mve, mol_list):
        std = fitted_multitask_mve.compute_uncertainty(mol_list)
        assert std.shape == (len(mol_list), 3)

    def test_multitask_std_non_negative(self, fitted_multitask_mve, mol_list):
        std = fitted_multitask_mve.compute_uncertainty(mol_list)
        assert np.all(std >= 0.0)


class TestMVEMultitaskChemprop:
    """Multitask MVE on the chemprop path: chemprop's MveFFN emits (N, T, 2)
    directly, so per-task means and per-task variances must survive routing
    through ChempropRegressor's sklearn wrapper unchanged."""

    @pytest.fixture()
    def fitted_multitask_chemprop_mve(self, mol_list, multitask_regression_y_t3):
        model = ChempropRegressor(
            **_ARCH_KWARGS[ChempropRegressor],
            **_MVE_KWARGS_CHEMPROP,
            num_endpoints=3,
        )
        model.fit(mol_list, multitask_regression_y_t3)
        return model

    def test_multitask_predict_shape(self, fitted_multitask_chemprop_mve, mol_list):
        preds = fitted_multitask_chemprop_mve.predict(mol_list)
        assert preds.shape == (len(mol_list), 3)

    def test_multitask_std_shape(self, fitted_multitask_chemprop_mve, mol_list):
        std = fitted_multitask_chemprop_mve.compute_uncertainty(mol_list)
        assert std.shape == (len(mol_list), 3)

    def test_multitask_std_non_negative(self, fitted_multitask_chemprop_mve, mol_list):
        std = fitted_multitask_chemprop_mve.compute_uncertainty(mol_list)
        assert np.all(std >= 0.0)


# =========================================================================
# bounded-beta-nll — censored labels via bound_mask (MLP only)
# =========================================================================


class TestMVEBoundedBetaNLL:
    """bounded-beta-nll accepts censored labels and produces well-formed output."""

    @pytest.fixture()
    def fitted_bounded_mve(self, mol_list, regression_y, bound_mask_mixed):
        model = MLPRegressor(
            **_ARCH_KWARGS[MLPRegressor],
            uncertainty="mve",
            loss_fn="bounded-beta-nll",
        )
        model.fit(mol_list, regression_y, bound_mask=bound_mask_mixed)
        return model

    def test_bounded_predict_shape(self, fitted_bounded_mve, mol_list):
        preds = fitted_bounded_mve.predict(mol_list)
        assert preds.shape == (len(mol_list), 1)

    def test_bounded_std_shape(self, fitted_bounded_mve, mol_list):
        std = fitted_bounded_mve.compute_uncertainty(mol_list)
        assert std.shape == (len(mol_list), 1)

    def test_bounded_std_non_negative(self, fitted_bounded_mve, mol_list):
        std = fitted_bounded_mve.compute_uncertainty(mol_list)
        assert np.all(std >= 0.0)


# =========================================================================
# Save / load round-trip — predict AND compute_uncertainty must match exactly
# =========================================================================


class TestMVESaveLoad:
    """Serialising and reloading an MVE model preserves predict and compute_uncertainty.

    CLM families use SMILES augmentation whose per-call RNG state resets on
    reload, so exact-equality is not achievable there — a small relative
    tolerance covers that without weakening the behavioural check.
    """

    def test_predict_matches_after_roundtrip(
        self, fitted_mve_regressor, mol_list, tmp_path
    ):
        save_dir = str(tmp_path / "mve_model")
        preds_before = fitted_mve_regressor.predict(mol_list)
        fitted_mve_regressor.save_model(save_dir)

        loaded = autoload(save_dir, accelerator="cpu")
        preds_after = loaded.predict(mol_list)

        np.testing.assert_allclose(preds_before, preds_after, rtol=1e-2)

    def test_uncertainty_matches_after_roundtrip(
        self, fitted_mve_regressor, mol_list, tmp_path
    ):
        save_dir = str(tmp_path / "mve_model")
        std_before = fitted_mve_regressor.compute_uncertainty(mol_list)
        fitted_mve_regressor.save_model(save_dir)

        loaded = autoload(save_dir, accelerator="cpu")
        std_after = loaded.compute_uncertainty(mol_list)

        np.testing.assert_allclose(std_before, std_after, rtol=1e-2)

    def test_loaded_model_reports_mve_method(self, fitted_mve_regressor, tmp_path):
        save_dir = str(tmp_path / "mve_model")
        fitted_mve_regressor.save_model(save_dir)
        loaded = autoload(save_dir, accelerator="cpu")
        assert loaded._model.uncertainty_method == "mve"


# =========================================================================
# Calibration — calibrate_uncertainty must fit and then be applied
# =========================================================================


class TestMVECalibration:
    """calibrate_uncertainty on an MVE model fits an ICP calibrator that
    then re-shapes compute_uncertainty output."""

    def test_calibrator_is_set_after_calibrate(self, mol_list, regression_y):
        model = MLPRegressor(**_ARCH_KWARGS[MLPRegressor], **_MVE_KWARGS)
        model.fit(mol_list, regression_y)
        assert model._uncertainty_manager.calibrator is None
        model.calibrate_uncertainty(
            calibration_mols=mol_list,
            calibration_y=regression_y,
            algorithm="icp_regression",
        )
        assert model._uncertainty_manager.calibrator is not None

    def test_calibrated_uncertainty_is_finite_and_non_negative(
        self, mol_list, regression_y
    ):
        model = MLPRegressor(**_ARCH_KWARGS[MLPRegressor], **_MVE_KWARGS)
        model.fit(mol_list, regression_y)
        model.calibrate_uncertainty(
            calibration_mols=mol_list,
            calibration_y=regression_y,
            algorithm="icp_regression",
        )
        std = model.compute_uncertainty(mol_list)
        assert std.shape == (len(mol_list), 1)
        assert np.all(np.isfinite(std))
        assert np.all(std >= 0.0)

    def test_chemprop_calibrator_is_set_and_produces_finite_std(
        self, mol_list, regression_y
    ):
        # Guards the chemprop MVE path through UncertaintyManager.calibrate:
        # softplus → log-variance adapter in ChempropModel.predict_variance_step
        # must survive the calibrator fit + apply round-trip.
        model = ChempropRegressor(
            **_ARCH_KWARGS[ChempropRegressor], **_MVE_KWARGS_CHEMPROP
        )
        model.fit(mol_list, regression_y)
        assert model._uncertainty_manager.calibrator is None
        model.calibrate_uncertainty(
            calibration_mols=mol_list,
            calibration_y=regression_y,
            algorithm="icp_regression",
        )
        assert model._uncertainty_manager.calibrator is not None
        std = model.compute_uncertainty(mol_list)
        assert std.shape == (len(mol_list), 1)
        assert np.all(np.isfinite(std))
        assert np.all(std >= 0.0)


# =========================================================================
# Symmetric validation — every rejection path listed in the plan
# =========================================================================


class TestMVEValidationErrors:
    """Every symmetric validation and sklearn-guard error is raised."""

    def test_mve_without_mve_loss_raises(self):
        with pytest.raises(ValidationError):
            MLPRegressor(
                **_ARCH_KWARGS[MLPRegressor],
                uncertainty="mve",
                loss_fn="mse",
            )

    def test_mve_loss_without_mve_uncertainty_raises(self):
        with pytest.raises(ValidationError):
            MLPRegressor(
                **_ARCH_KWARGS[MLPRegressor],
                loss_fn="beta-nll",
            )

    def test_bounded_beta_nll_without_mve_uncertainty_raises(self):
        with pytest.raises(ValidationError):
            MLPRegressor(
                **_ARCH_KWARGS[MLPRegressor],
                loss_fn="bounded-beta-nll",
            )

    def test_mve_with_non_standard_scaler_raises(self):
        with pytest.raises(ValidationError, match="standard"):
            MLPRegressor(
                **_ARCH_KWARGS[MLPRegressor],
                uncertainty="mve",
                loss_fn="beta-nll",
                scaler_type="quantile",
            )

    def test_mve_with_multitask_loss_raises(self):
        with pytest.raises(ValidationError):
            MLPRegressor(
                **_ARCH_KWARGS[MLPRegressor],
                uncertainty="mve",
                loss_fn="multitask",
            )

    def test_mve_with_gradnorm_loss_raises(self):
        with pytest.raises(ValidationError):
            MLPRegressor(
                **_ARCH_KWARGS[MLPRegressor],
                uncertainty="mve",
                loss_fn="gradnorm",
            )

    def test_snn_regressor_mve_raises(self):
        with pytest.raises(NotImplementedError, match="SNN"):
            SNNRegressor(
                hidden_dims=[32],
                feature_list=["ECFP"],
                num_epochs=1,
                batch_size=32,
                accelerator="cpu",
                devices=1,
                early_stopping=False,
                stochastic_weight_averaging=False,
                num_parallel=2,
                uncertainty="mve",
                loss_fn="beta-nll",
            )

    def test_chemprop_regressor_mve_beta_nll_raises(self):
        # Chemprop paths only accept chemprop's own MVELoss (alias "mve");
        # β-NLL / bounded-β-NLL stay matcha-only. See
        # ChempropInputModel._validate_mve_pairing.
        with pytest.raises(ValidationError):
            ChempropRegressor(
                **_ARCH_KWARGS[ChempropRegressor],
                uncertainty="mve",
                loss_fn="beta-nll",
            )

    def test_chemprop_regressor_mve_bounded_beta_nll_raises(self):
        with pytest.raises(ValidationError):
            ChempropRegressor(
                **_ARCH_KWARGS[ChempropRegressor],
                uncertainty="mve",
                loss_fn="bounded-beta-nll",
            )

    def test_classifier_rejects_mve(self):
        # Classifiers narrow the surface to Literal["mc-dropout"]; a caller
        # requesting MVE trips the Pydantic pairing rule (classifier losses
        # are not in the MVE-family loss set).
        with pytest.raises(ValidationError):
            MLPClassifier(
                **_ARCH_KWARGS[MLPClassifier],
                uncertainty="mve",
            )


# =========================================================================
# Backwards-compat regression — legacy uncertainty=None YAML configs
# =========================================================================


class TestMVEBackwardsCompat:
    """Legacy configs with ``uncertainty=None`` still validate and behave as MC dropout."""

    def test_legacy_none_normalizes_to_mc_dropout(self, mol_list, regression_y):
        model = MLPRegressor(**_ARCH_KWARGS[MLPRegressor], uncertainty=None)
        model.fit(mol_list, regression_y)
        assert model._model.uncertainty_method == "mc-dropout"


# =========================================================================
# Ensemble law of total variance — MLP K=2
# =========================================================================


class TestMVEEnsemble:
    """Ensemble aggregation applies the law of total variance for MVE members."""

    @pytest.fixture()
    def fitted_mve_ensemble(self, mol_list, regression_y):
        template = MLPRegressor(**_ARCH_KWARGS[MLPRegressor], **_MVE_KWARGS)
        ens = Ensemble(model=template, n_models=2)
        ens.fit(mol_list, regression_y)
        return ens

    def test_ensemble_predict_returns_tuple(self, fitted_mve_ensemble, mol_list):
        result = fitted_mve_ensemble.predict(mol_list)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_ensemble_predict_shapes(self, fitted_mve_ensemble, mol_list):
        mean, std = fitted_mve_ensemble.predict(mol_list)
        assert mean.shape == (len(mol_list), 1)
        assert std.shape == (len(mol_list), 1)

    def test_ensemble_std_non_negative_and_finite(self, fitted_mve_ensemble, mol_list):
        _, std = fitted_mve_ensemble.predict(mol_list)
        assert np.all(np.isfinite(std))
        assert np.all(std >= 0.0)

    def test_ensemble_matches_manual_law_of_total_variance(
        self, fitted_mve_ensemble, mol_list
    ):
        """Var_total = mean(var_i) + var(mean_i); mu_total = mean(mean_i)."""
        member_means = []
        member_stds = []
        for m in fitted_mve_ensemble.model_box:
            member_means.append(m.predict(mol_list))
            member_stds.append(m.compute_uncertainty(mol_list))
        member_means = np.stack(member_means, axis=2)
        member_stds = np.stack(member_stds, axis=2)

        expected_mu = np.mean(member_means, axis=2)
        expected_var = np.mean(member_stds**2, axis=2) + np.var(member_means, axis=2)
        expected_std = np.sqrt(expected_var)

        actual_mu, actual_std = fitted_mve_ensemble.predict(mol_list)
        np.testing.assert_allclose(actual_mu, expected_mu, rtol=1e-5)
        np.testing.assert_allclose(actual_std, expected_std, rtol=1e-5)


# =========================================================================
# Numerical-equivalence regression — locks in the (B, T, 2) shape migration
# =========================================================================


class TestMVEShapeMigrationBaseline:
    """Fixed-seed regression against a baseline captured pre shape migration.

    Guards issue #99 Stage 1: swapping ``MVEPredictor.forward`` from
    ``torch.cat([mean, log_var], -1)`` (``(B, 2·T)``) to
    ``torch.stack([mean, log_var], -1)`` (``(B, T, 2)``) is intended to be
    a pure refactor — training arithmetic and eval-time predictions must
    reproduce the pre-migration baseline stored in ``mve_baseline.npz``.

    The baseline was captured pre-migration using the fit sequence encoded
    below (one warm-up fit, then a re-seeded measured fit). The warm-up
    burns off first-call lazy initialisation (RDKit fingerprint generator
    caches, Lightning ``Trainer`` bootstrap) that consumes RNG in a way
    ``L.seed_everything`` cannot reset. Without it, the measured fit is
    only deterministic when this test runs first in the pytest session —
    which is order-dependent and unreliable. Both the capture and the test
    follow the same warm-up + measurement pattern so the baseline stays
    reproducible regardless of test ordering.
    """

    @staticmethod
    def _make_mve_regressor() -> MLPRegressor:
        return MLPRegressor(
            hidden_dims=[32],
            feature_list=["ECFP"],
            num_epochs=1,
            batch_size=32,
            accelerator="cpu",
            devices=1,
            early_stopping=False,
            stochastic_weight_averaging=False,
            uncertainty="mve",
            loss_fn="beta-nll",
            seed=0,
        )

    def test_mve_predictions_match_pre_migration_baseline(self, mol_list, regression_y):
        L.seed_everything(0, workers=True, verbose=False)
        self._make_mve_regressor().fit(mol_list, regression_y)

        L.seed_everything(0, workers=True, verbose=False)
        model = self._make_mve_regressor()
        model.fit(mol_list, regression_y)

        preds = model.predict(mol_list)
        std = model.compute_uncertainty(mol_list)

        baseline = np.load(_MVE_BASELINE_PATH)
        np.testing.assert_allclose(preds, baseline["preds"], atol=1e-6)
        np.testing.assert_allclose(std, baseline["std"], atol=1e-6)
