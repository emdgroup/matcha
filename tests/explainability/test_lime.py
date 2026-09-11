"""Tests for matcha.explainability.lime.LIME."""

import numpy as np
import pandas as pd
import pytest

import matcha.explainability.lime as lime_module
from matcha.explainability.lime import LIME, _default, _fp_default


# ===================================================================
# LIME – initialisation
# ===================================================================


class TestLIMEInit:
    """Tests for LIME.__init__."""

    def test_default_descriptor_set(self):
        lime = LIME()
        assert lime.descriptor_set == _default

    def test_custom_descriptor_set(self):
        custom = ["MolWt", "MolLogP"]
        lime = LIME(descriptor_set=custom)
        assert lime.descriptor_set == custom

    def test_default_fingerprint_params(self):
        lime = LIME()
        assert lime._fingerprint_params_set == _fp_default

    def test_custom_fingerprint_params(self):
        params = {"nBits": 2048, "radius": 2, "useFeatures": True}
        lime = LIME(fingerprint_params=params)
        assert lime._fingerprint_params_set == params

    def test_scale_coeff_default_true(self):
        lime = LIME()
        assert lime.scale_coeff is True

    def test_scale_coeff_false(self):
        lime = LIME(scale_coeff=False)
        assert lime.scale_coeff is False

    def test_use_fingerprints_default_false(self):
        lime = LIME()
        assert lime._use_fingerprints is False

    def test_use_fingerprints_true(self):
        lime = LIME(use_fingerprints=True)
        assert lime._use_fingerprints is True

    def test_initial_model_box_empty(self):
        lime = LIME()
        assert lime._model_box == []

    def test_initial_r2_box_empty(self):
        lime = LIME()
        assert lime.r2_box == []

    def test_initial_coeff_box_none(self):
        lime = LIME()
        assert lime._coeff_box is None


# ===================================================================
# LIME – feature extraction
# ===================================================================


class TestLIMEFeatureExtraction:
    """Tests for descriptor and fingerprint computation."""

    def test_get_features_returns_ndarray(self, small_mol_list):
        lime = LIME()
        feats = lime._get_features(small_mol_list, _default)
        assert isinstance(feats, np.ndarray)

    def test_get_features_shape(self, small_mol_list):
        lime = LIME()
        feats = lime._get_features(small_mol_list, _default)
        assert feats.shape[0] == len(small_mol_list)
        assert feats.shape[1] == len(_default)

    def test_get_features_custom_descriptors(self, small_mol_list):
        descs = ["MolWt", "MolLogP"]
        lime = LIME(descriptor_set=descs)
        feats = lime._get_features(small_mol_list, descs)
        assert feats.shape == (len(small_mol_list), 2)

    def test_get_ecfps_returns_ndarray(self, small_mol_list):
        lime = LIME(use_fingerprints=True)
        feats = lime._get_ecfps(small_mol_list, _fp_default)
        assert isinstance(feats, np.ndarray)

    def test_get_ecfps_shape_default(self, small_mol_list):
        lime = LIME(use_fingerprints=True)
        feats = lime._get_ecfps(small_mol_list, _fp_default)
        assert feats.shape == (len(small_mol_list), _fp_default["nBits"])

    def test_get_ecfps_custom_nbits(self, small_mol_list):
        params = {"nBits": 512, "radius": 2, "useFeatures": False}
        lime = LIME(fingerprint_params=params, use_fingerprints=True)
        feats = lime._get_ecfps(small_mol_list, params)
        assert feats.shape == (len(small_mol_list), 512)

    def test_features_no_nans(self, small_mol_list):
        lime = LIME()
        feats = lime._get_features(small_mol_list, _default)
        assert not np.any(np.isnan(feats))

    def test_ecfps_binary_values(self, small_mol_list):
        lime = LIME(use_fingerprints=True)
        feats = lime._get_ecfps(small_mol_list, _fp_default)
        unique_vals = np.unique(feats)
        assert all(v in [0, 1] for v in unique_vals)


# ===================================================================
# LIME – _fit
# ===================================================================


class TestLIMEFit:
    """Tests for LIME._fit (the bootstrap Ridge regression loop)."""

    def test_fit_returns_coeff_box(self, small_mol_list, small_regression_targets):
        lime = LIME()
        feats = lime._get_features(small_mol_list, _default)
        bootstrap_num = 3
        coeff_box = lime._fit(feats, small_regression_targets, bootstrap_num)
        assert isinstance(coeff_box, np.ndarray)

    def test_fit_coeff_box_shape(self, small_mol_list, small_regression_targets):
        lime = LIME()
        feats = lime._get_features(small_mol_list, _default)
        bootstrap_num = 3
        coeff_box = lime._fit(feats, small_regression_targets, bootstrap_num)
        assert coeff_box.shape == (bootstrap_num, feats.shape[1])

    def test_fit_populates_r2_box(self, small_mol_list, small_regression_targets):
        lime = LIME()
        feats = lime._get_features(small_mol_list, _default)
        bootstrap_num = 3
        lime._fit(feats, small_regression_targets, bootstrap_num)
        assert len(lime.r2_box) == bootstrap_num

    def test_fit_r2_values_finite(self, small_mol_list, small_regression_targets):
        lime = LIME()
        feats = lime._get_features(small_mol_list, _default)
        lime._fit(feats, small_regression_targets, 3)
        assert all(np.isfinite(r2) for r2 in lime.r2_box)

    def test_fit_populates_model_box(self, small_mol_list, small_regression_targets):
        lime = LIME()
        feats = lime._get_features(small_mol_list, _default)
        bootstrap_num = 3
        lime._fit(feats, small_regression_targets, bootstrap_num)
        assert len(lime._model_box) == bootstrap_num

    def test_fit_uses_aligned_sample_and_feature_subsets(self, monkeypatch):
        records = []

        class IdentityScaler:
            def fit_transform(self, values):
                return values

        class RecordingRidge:
            def fit(self, values, targets):
                records.append((values.copy(), targets.copy()))
                self.coef_ = values[0].copy()
                self._targets = targets.copy()
                return self

            def predict(self, values):
                return self._targets

        monkeypatch.setattr(lime_module, "StandardScaler", IdentityScaler)
        monkeypatch.setattr(lime_module, "Ridge", RecordingRidge)
        feats = np.tile(np.arange(4, dtype=float), (4, 1))
        targets = np.array([10.0, 20.0, 30.0, 40.0])

        coeff_box = LIME()._fit(feats, targets, bootstrap_num=2)

        np.testing.assert_array_equal(records[0][0], feats[np.ix_([2, 3], [2, 3])])
        np.testing.assert_array_equal(records[0][1], targets[[2, 3]])
        np.testing.assert_array_equal(records[1][0], feats[np.ix_([0, 1], [0, 1])])
        np.testing.assert_array_equal(records[1][1], targets[[0, 1]])
        np.testing.assert_array_equal(coeff_box[0, 2:], [2.0, 3.0])
        assert np.isnan(coeff_box[0, :2]).all()
        np.testing.assert_array_equal(coeff_box[1, :2], [0.0, 1.0])
        assert np.isnan(coeff_box[1, 2:]).all()

    def test_fit_uses_pinned_sample_fold_formula(self, monkeypatch):
        fitted_targets = []

        class IdentityScaler:
            def fit_transform(self, values):
                return values

        class RecordingRidge:
            def fit(self, values, targets):
                fitted_targets.append(targets.copy())
                self.coef_ = np.zeros(values.shape[1])
                self._targets = targets.copy()
                return self

            def predict(self, values):
                return self._targets

        monkeypatch.setattr(lime_module, "StandardScaler", IdentityScaler)
        monkeypatch.setattr(lime_module, "Ridge", RecordingRidge)

        LIME()._fit(
            np.arange(6, dtype=float).reshape(3, 2),
            np.array([10.0, 20.0, 30.0]),
            bootstrap_num=1,
        )

        assert len(fitted_targets) == 1
        np.testing.assert_array_equal(fitted_targets[0], [20.0, 30.0])

    def test_fit_resets_run_state(self):
        lime = LIME()
        feats = np.arange(16, dtype=float).reshape(4, 4)
        targets = np.arange(4, dtype=float)

        lime._fit(feats, targets, bootstrap_num=2)
        lime._fit(feats, targets, bootstrap_num=2)

        assert len(lime._model_box) == 2
        assert len(lime.r2_box) == 2


# ===================================================================
# LIME – explain (descriptor mode)
# ===================================================================


class TestLIMEExplainDescriptors:
    """Tests for LIME.explain with descriptor-based features."""

    def test_explain_returns_dataframe(self, small_mol_list, small_regression_targets):
        lime = LIME(use_fingerprints=False)
        df = lime.explain(small_mol_list, small_regression_targets, bootstrap_num=3)
        assert isinstance(df, pd.DataFrame)

    def test_explain_has_required_columns(
        self, small_mol_list, small_regression_targets
    ):
        lime = LIME(use_fingerprints=False)
        df = lime.explain(small_mol_list, small_regression_targets, bootstrap_num=3)
        required = {"Descriptor", "Coefficient", "Standard deviation"}
        assert required.issubset(set(df.columns))

    def test_explain_last_row_is_r2(self, small_mol_list, small_regression_targets):
        lime = LIME(use_fingerprints=False)
        df = lime.explain(small_mol_list, small_regression_targets, bootstrap_num=3)
        last_row = df.iloc[-1]
        assert last_row["Descriptor"] == "Local fit R2"

    def test_explain_r2_in_valid_range(self, small_mol_list, small_regression_targets):
        lime = LIME(use_fingerprints=False)
        df = lime.explain(small_mol_list, small_regression_targets, bootstrap_num=3)
        r2_val = df.iloc[-1]["Coefficient"]
        assert np.isfinite(r2_val)

    def test_explain_sorted_descending(self, small_mol_list, small_regression_targets):
        lime = LIME(use_fingerprints=False)
        df = lime.explain(small_mol_list, small_regression_targets, bootstrap_num=3)
        # Exclude the R2 row (last)
        coefficients = df.iloc[:-1]["Coefficient"].values
        assert all(
            coefficients[i] >= coefficients[i + 1] for i in range(len(coefficients) - 1)
        )

    def test_explain_descriptor_names_match_default(
        self, small_mol_list, small_regression_targets
    ):
        lime = LIME(use_fingerprints=False)
        df = lime.explain(small_mol_list, small_regression_targets, bootstrap_num=3)
        descriptors = df.iloc[:-1]["Descriptor"].tolist()
        assert set(descriptors) == set(_default)

    def test_explain_rejects_bootstrap_above_feature_count(
        self, small_mol_list, small_regression_targets
    ):
        custom = ["MolWt", "MolLogP", "TPSA"]
        lime = LIME(descriptor_set=custom, use_fingerprints=False)

        with pytest.raises(
            ValueError,
            match=r"bootstrap_num \(4\) cannot exceed available feature count \(3\)",
        ):
            lime.explain(small_mol_list, small_regression_targets, bootstrap_num=4)

    def test_explain_no_scale_coeff(self, small_mol_list, small_regression_targets):
        lime = LIME(scale_coeff=False, use_fingerprints=False)
        df = lime.explain(small_mol_list, small_regression_targets, bootstrap_num=3)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 1

    def test_explain_rejects_fewer_than_three_rows(
        self, small_mol_list, small_regression_targets
    ):
        lime = LIME(use_fingerprints=False)

        with pytest.raises(ValueError, match="at least 3 molecules"):
            lime.explain(
                small_mol_list[:2], small_regression_targets[:2], bootstrap_num=1
            )

    def test_explain_rejects_misaligned_lengths(
        self, small_mol_list, small_regression_targets
    ):
        with pytest.raises(ValueError, match="molecule and target counts must match"):
            LIME().explain(
                small_mol_list[:3], small_regression_targets[:2], bootstrap_num=1
            )

    def test_explain_rejects_non_one_dimensional_targets(
        self, small_mol_list, small_regression_targets
    ):
        with pytest.raises(ValueError, match="targets must be one-dimensional"):
            LIME().explain(
                small_mol_list[:3],
                small_regression_targets[:3, np.newaxis],
                bootstrap_num=1,
            )

    def test_explain_rejects_non_positive_bootstrap_num(
        self, small_mol_list, small_regression_targets
    ):
        with pytest.raises(ValueError, match="bootstrap_num must be at least 1"):
            LIME().explain(
                small_mol_list[:3], small_regression_targets[:3], bootstrap_num=0
            )

    def test_explain_bounds_effective_fits_by_rows(
        self, monkeypatch, small_mol_list, small_regression_targets
    ):
        requested_fits = []
        lime = LIME(descriptor_set=["a", "b", "c", "d", "e"])
        monkeypatch.setattr(
            lime,
            "_get_features",
            lambda mols, descriptors: np.ones((len(mols), len(descriptors))),
        )

        def fit(feats, targets, bootstrap_num):
            requested_fits.append(bootstrap_num)
            lime._r2_box = [1.0] * bootstrap_num
            return np.ones((bootstrap_num, feats.shape[1]))

        monkeypatch.setattr(lime, "_fit", fit)

        lime.explain(small_mol_list[:3], small_regression_targets[:3], bootstrap_num=5)

        assert requested_fits == [3]

    def test_explain_preserves_fitted_zero_when_aggregating(
        self, monkeypatch, small_mol_list, small_regression_targets
    ):
        lime = LIME(descriptor_set=["selected", "zero"], scale_coeff=False)
        monkeypatch.setattr(
            lime,
            "_get_features",
            lambda mols, descriptors: np.ones((len(mols), len(descriptors))),
        )

        def fit(feats, targets, bootstrap_num):
            lime._r2_box = [1.0, 1.0]
            return np.array([[np.nan, 0.0], [2.0, np.nan]])

        monkeypatch.setattr(lime, "_fit", fit)

        result = lime.explain(
            small_mol_list[:3], small_regression_targets[:3], bootstrap_num=2
        ).set_index("Descriptor")

        assert result.loc["zero", "Coefficient"] == 0.0
        assert np.isfinite(result.loc["zero", "Standard deviation"])

    def test_explain_uses_l1_coefficient_norm(
        self, monkeypatch, small_mol_list, small_regression_targets
    ):
        lime = LIME(descriptor_set=["positive", "negative", "zero"])
        monkeypatch.setattr(
            lime,
            "_get_features",
            lambda mols, descriptors: np.ones((len(mols), len(descriptors))),
        )

        def fit(feats, targets, bootstrap_num):
            lime._r2_box = [1.0, 1.0, 1.0]
            return np.array([[1.0, -1.0, 0.0], [2.0, -2.0, 0.0], [3.0, -3.0, 0.0]])

        monkeypatch.setattr(lime, "_fit", fit)

        result = lime.explain(
            small_mol_list[:3], small_regression_targets[:3], bootstrap_num=3
        ).set_index("Descriptor")

        assert result.loc[
            ["positive", "negative", "zero"], "Coefficient"
        ].abs().sum() == pytest.approx(1.0)
        assert result.loc["positive", "Coefficient"] == pytest.approx(0.5)
        assert result.loc["negative", "Coefficient"] == pytest.approx(-0.5)
        raw_std = np.std([1.0, 2.0, 3.0])
        assert result.loc["positive", "Standard deviation"] == pytest.approx(
            raw_std / 4.0
        )

    def test_explain_keeps_all_zero_statistics_finite(
        self, monkeypatch, small_mol_list, small_regression_targets
    ):
        lime = LIME(descriptor_set=["first", "second"])
        monkeypatch.setattr(
            lime,
            "_get_features",
            lambda mols, descriptors: np.ones((len(mols), len(descriptors))),
        )

        def fit(feats, targets, bootstrap_num):
            lime._r2_box = [1.0, 1.0]
            return np.zeros((2, 2))

        monkeypatch.setattr(lime, "_fit", fit)

        result = lime.explain(
            small_mol_list[:3], small_regression_targets[:3], bootstrap_num=2
        ).iloc[:-1]

        assert np.isfinite(result[["Coefficient", "Standard deviation"]]).all().all()
        assert (result[["Coefficient", "Standard deviation"]] == 0.0).all().all()


# ===================================================================
# LIME – explain (fingerprint mode)
# ===================================================================


class TestLIMEExplainFingerprints:
    """Tests for LIME.explain with ECFP fingerprints."""

    def test_explain_ecfp_returns_dataframe(
        self, small_mol_list, small_regression_targets
    ):
        lime = LIME(use_fingerprints=True)
        df = lime.explain(small_mol_list, small_regression_targets, bootstrap_num=3)
        assert isinstance(df, pd.DataFrame)

    def test_explain_ecfp_has_required_columns(
        self, small_mol_list, small_regression_targets
    ):
        lime = LIME(use_fingerprints=True)
        df = lime.explain(small_mol_list, small_regression_targets, bootstrap_num=3)
        required = {"Descriptor", "Coefficient", "Standard deviation"}
        assert required.issubset(set(df.columns))

    def test_explain_ecfp_last_row_is_r2(
        self, small_mol_list, small_regression_targets
    ):
        lime = LIME(use_fingerprints=True)
        df = lime.explain(small_mol_list, small_regression_targets, bootstrap_num=3)
        assert df.iloc[-1]["Descriptor"] == "Local fit R2"

    def test_explain_ecfp_descriptor_names_pattern(
        self, small_mol_list, small_regression_targets
    ):
        lime = LIME(use_fingerprints=True)
        df = lime.explain(small_mol_list, small_regression_targets, bootstrap_num=3)
        descriptors = df.iloc[:-1]["Descriptor"].tolist()
        # ECFP descriptors should be named like "F_0", "F_1", ...
        assert all(d.startswith("F_") for d in descriptors)


# ===================================================================
# LIME – ECFP environment extraction
# ===================================================================


class TestLIMEECFPEnvs:
    """Tests for _get_ECFP_envs and get_envs_and_weights."""

    def test_get_ecfp_envs_returns_dict(self, single_mol):
        lime = LIME()
        envs = lime._get_ECFP_envs(single_mol)
        assert isinstance(envs, dict)

    def test_get_ecfp_envs_keys_are_ints(self, single_mol):
        lime = LIME()
        envs = lime._get_ECFP_envs(single_mol)
        assert all(isinstance(k, int) for k in envs.keys())

    def test_get_ecfp_envs_values_are_sets(self, single_mol):
        lime = LIME()
        envs = lime._get_ECFP_envs(single_mol)
        assert all(isinstance(v, set) for v in envs.values())

    def test_get_ecfp_envs_atom_indices_valid(self, single_mol):
        lime = LIME()
        envs = lime._get_ECFP_envs(single_mol)
        n_atoms = single_mol.GetNumAtoms()
        for atoms in envs.values():
            assert all(0 <= a < n_atoms for a in atoms)

    def test_get_ecfp_envs_nonempty_for_nontrivial_mol(self, single_mol):
        lime = LIME()
        envs = lime._get_ECFP_envs(single_mol)
        assert len(envs) > 0

    def test_get_ecfp_envs_custom_params(self, single_mol):
        lime = LIME()
        envs = lime._get_ECFP_envs(single_mol, radius=1, nBits=512, useFeatures=True)
        assert isinstance(envs, dict)
        # All bit IDs should be within [0, 512)
        assert all(0 <= k < 512 for k in envs.keys())

    def test_get_envs_and_weights_returns_tuple(
        self, small_mol_list, small_regression_targets
    ):
        lime = LIME(use_fingerprints=True)
        df = lime.explain(small_mol_list, small_regression_targets, bootstrap_num=3)
        envs, weights = lime.get_envs_and_weights(small_mol_list[0], df)
        assert isinstance(envs, dict)
        assert isinstance(weights, dict)

    def test_get_envs_and_weights_keys_are_ints(
        self, small_mol_list, small_regression_targets
    ):
        lime = LIME(use_fingerprints=True)
        df = lime.explain(small_mol_list, small_regression_targets, bootstrap_num=3)
        envs, weights = lime.get_envs_and_weights(small_mol_list[0], df)
        assert all(isinstance(k, int) for k in weights.keys())

    def test_get_envs_and_weights_values_are_floats(
        self, small_mol_list, small_regression_targets
    ):
        lime = LIME(use_fingerprints=True)
        df = lime.explain(small_mol_list, small_regression_targets, bootstrap_num=3)
        _, weights = lime.get_envs_and_weights(small_mol_list[0], df)
        assert all(isinstance(v, float) for v in weights.values())
