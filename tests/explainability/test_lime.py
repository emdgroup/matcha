"""Tests for matcha.explainability.lime.LIME."""

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from sklearn.preprocessing import StandardScaler

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
        assert lime._fingerprint_params_set == {
            "nBits": 8192,
            "radius": 3,
            "useFeatures": False,
        }
        assert lime._fingerprint_params_set is not _fp_default

    def test_partial_fingerprint_params_merge_over_copied_defaults(self):
        params = {"radius": 2}
        lime = LIME(fingerprint_params=params)

        assert lime._fingerprint_params_set == {
            "nBits": 8192,
            "radius": 2,
            "useFeatures": False,
        }
        assert params == {"radius": 2}

    def test_full_fingerprint_params_are_copied(self):
        params = {"nBits": 2048, "radius": 2, "useFeatures": True}
        lime = LIME(fingerprint_params=params)

        assert lime._fingerprint_params_set == params
        assert lime._fingerprint_params_set is not params

    def test_fingerprint_params_do_not_mutate_defaults_or_caller_values(self):
        params = {"radius": 2}
        lime = LIME(fingerprint_params=params)
        lime._fingerprint_params_set["radius"] = 1

        assert params == {"radius": 2}
        assert _fp_default == {"nBits": 8192, "radius": 3, "useFeatures": False}

    def test_random_seed_defaults_to_zero(self):
        assert LIME()._random_seed == 0

    def test_random_seed_can_be_overridden(self):
        assert LIME(random_seed=-7)._random_seed == -7

    def test_removed_scale_coeff_argument_is_rejected(self):
        with pytest.raises(TypeError, match="scale_coeff"):
            LIME(scale_coeff=True)

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
        feats = lime._get_ecfps(small_mol_list)
        assert isinstance(feats, np.ndarray)

    def test_get_ecfps_shape_default(self, small_mol_list):
        lime = LIME(use_fingerprints=True)
        feats = lime._get_ecfps(small_mol_list)
        assert feats.shape == (len(small_mol_list), _fp_default["nBits"])

    def test_get_ecfps_custom_nbits(self, small_mol_list):
        params = {"nBits": 512, "radius": 2, "useFeatures": False}
        lime = LIME(fingerprint_params=params, use_fingerprints=True)
        feats = lime._get_ecfps(small_mol_list)
        assert feats.shape == (len(small_mol_list), 512)

    def test_features_no_nans(self, small_mol_list):
        lime = LIME()
        feats = lime._get_features(small_mol_list, _default)
        assert not np.any(np.isnan(feats))

    def test_ecfps_binary_values(self, small_mol_list):
        lime = LIME(use_fingerprints=True)
        feats = lime._get_ecfps(small_mol_list)
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

    @pytest.mark.parametrize("use_fingerprints", [False, True])
    def test_fit_bootstraps_full_rows_and_all_columns(
        self, monkeypatch, use_fingerprints
    ):
        samples = [np.array([0, 0, 2, 1]), np.array([3, 1, 3, 0])]
        choice_calls = []
        fitted = []

        class ScriptedGenerator:
            def choice(self, row_count, size, replace):
                choice_calls.append((row_count, size, replace))
                return samples[len(choice_calls) - 1]

        class IdentityScaler:
            def fit_transform(self, values):
                return values

        class RecordingRidge:
            def fit(self, values, targets):
                fitted.append((values.copy(), targets.copy()))
                self.coef_ = np.arange(values.shape[1], dtype=float)
                self._targets = targets.copy()
                return self

            def predict(self, values):
                return self._targets

        monkeypatch.setattr(
            lime_module.np.random, "default_rng", lambda seed: ScriptedGenerator()
        )
        monkeypatch.setattr(lime_module, "StandardScaler", IdentityScaler)
        monkeypatch.setattr(lime_module, "Ridge", RecordingRidge)
        feats = np.arange(12, dtype=float).reshape(4, 3)
        targets = np.array([10.0, 20.0, 30.0, 40.0])

        coeff_box = LIME(use_fingerprints=use_fingerprints)._fit(
            feats, targets, bootstrap_num=2
        )

        assert choice_calls == [(4, 4, True), (4, 4, True)]
        for fit_index, sample_rows in enumerate(samples):
            np.testing.assert_array_equal(fitted[fit_index][0], feats[sample_rows])
            np.testing.assert_array_equal(fitted[fit_index][1], targets[sample_rows])
        assert coeff_box.shape == (2, 3)
        assert not np.isnan(coeff_box).any()

    def test_fit_standardizes_each_descriptor_bootstrap_independently(
        self, monkeypatch
    ):
        samples = [np.array([0, 0, 2, 1]), np.array([3, 1, 3, 0])]
        fitted_features = []
        scaler_instances = []

        class ScriptedGenerator:
            def __init__(self):
                self.fit_index = 0

            def choice(self, row_count, size, replace):
                sample = samples[self.fit_index]
                self.fit_index += 1
                return sample

        class RecordingScaler:
            def __init__(self):
                scaler_instances.append(self)

            def fit_transform(self, values):
                self.values = values.copy()
                return StandardScaler().fit_transform(values)

        class RecordingRidge:
            def fit(self, values, targets):
                fitted_features.append(values.copy())
                self.coef_ = np.zeros(values.shape[1])
                self._targets = targets.copy()
                return self

            def predict(self, values):
                return self._targets

        monkeypatch.setattr(
            lime_module.np.random, "default_rng", lambda seed: ScriptedGenerator()
        )
        monkeypatch.setattr(lime_module, "StandardScaler", RecordingScaler)
        monkeypatch.setattr(lime_module, "Ridge", RecordingRidge)
        feats = np.arange(12, dtype=float).reshape(4, 3)

        LIME()._fit(feats, np.arange(4, dtype=float), bootstrap_num=2)

        assert len(scaler_instances) == 2
        for fit_index, sample_rows in enumerate(samples):
            sampled = feats[sample_rows]
            np.testing.assert_array_equal(scaler_instances[fit_index].values, sampled)
            np.testing.assert_allclose(
                fitted_features[fit_index], StandardScaler().fit_transform(sampled)
            )

    def test_fit_passes_raw_binary_ecfp_rows_to_ridge(self, monkeypatch):
        sample_rows = np.array([0, 2, 2, 1])
        fitted_features = []

        class ScriptedGenerator:
            def choice(self, row_count, size, replace):
                return sample_rows

        class UnexpectedScaler:
            def __init__(self):
                raise AssertionError("ECFP features must not be standardized")

        class RecordingRidge:
            def fit(self, values, targets):
                fitted_features.append(values.copy())
                self.coef_ = np.zeros(values.shape[1])
                self._targets = targets.copy()
                return self

            def predict(self, values):
                return self._targets

        monkeypatch.setattr(
            lime_module.np.random, "default_rng", lambda seed: ScriptedGenerator()
        )
        monkeypatch.setattr(lime_module, "StandardScaler", UnexpectedScaler)
        monkeypatch.setattr(lime_module, "Ridge", RecordingRidge)
        feats = np.array([[0, 1, 0], [1, 0, 1], [1, 1, 0], [0, 0, 1]], dtype=float)

        LIME(use_fingerprints=True)._fit(
            feats, np.arange(4, dtype=float), bootstrap_num=1
        )

        np.testing.assert_array_equal(fitted_features[0], feats[sample_rows])

    @pytest.mark.parametrize("use_fingerprints", [False, True])
    def test_fit_replays_negative_seed_independently_of_ambient_rng(
        self, use_fingerprints
    ):
        feats = np.arange(30, dtype=float).reshape(6, 5)
        targets = np.array([0.5, 1.0, 1.5, 2.5, 4.0, 6.5])
        lime = LIME(use_fingerprints=use_fingerprints, random_seed=-7)

        first = lime._fit(feats, targets, bootstrap_num=4).copy()
        first_r2 = lime.r2_box.copy()
        np.random.seed(123)
        np.random.random(1000)
        second = lime._fit(feats, targets, bootstrap_num=4).copy()

        np.testing.assert_allclose(second, first)
        np.testing.assert_allclose(lime.r2_box, first_r2)

    def test_descriptor_and_ecfp_modes_use_identical_row_sequences(self, monkeypatch):
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
        feats = np.arange(30, dtype=float).reshape(6, 5)
        targets = np.arange(6, dtype=float)

        LIME(random_seed=19)._fit(feats, targets, bootstrap_num=3)
        LIME(use_fingerprints=True, random_seed=19)._fit(
            feats, targets, bootstrap_num=3
        )

        for descriptor_targets, ecfp_targets in zip(
            fitted_targets[:3], fitted_targets[3:]
        ):
            np.testing.assert_array_equal(descriptor_targets, ecfp_targets)

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

    @pytest.mark.parametrize("use_fingerprints", [False, True])
    def test_explain_runs_exact_count_above_row_and_feature_counts(
        self, monkeypatch, small_mol_list, small_regression_targets, use_fingerprints
    ):
        lime = LIME(
            descriptor_set=["first", "second"],
            fingerprint_params={"nBits": 2},
            use_fingerprints=use_fingerprints,
        )
        requested_fits = []
        if use_fingerprints:
            monkeypatch.setattr(
                lime, "_get_ecfps", lambda mols: np.ones((len(mols), 2))
            )
        else:
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

        assert requested_fits == [5]

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

    def test_explain_returns_raw_descriptor_statistics(
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

        assert result.loc["positive", "Coefficient"] == pytest.approx(2.0)
        assert result.loc["negative", "Coefficient"] == pytest.approx(-2.0)
        raw_std = np.std([1.0, 2.0, 3.0])
        assert result.loc["positive", "Standard deviation"] == pytest.approx(raw_std)

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

    def test_production_shaped_ecfp_bootstrap_is_warning_free(self):
        smiles = ["CCO", "CCN", "c1ccccc1", "CC(=O)O", "CCCl"]
        mols = [Chem.MolFromSmiles(smiles[index % len(smiles)]) for index in range(200)]
        targets = np.linspace(-1.0, 1.0, len(mols))

        result = LIME(use_fingerprints=True).explain(mols, targets, bootstrap_num=1)

        assert len(result) == 8193
        assert np.isfinite(result[["Coefficient", "Standard deviation"]]).all().all()

    def test_explain_returns_raw_fingerprint_statistics(
        self, monkeypatch, small_mol_list, small_regression_targets
    ):
        lime = LIME(fingerprint_params={"nBits": 2}, use_fingerprints=True)
        monkeypatch.setattr(
            lime,
            "_get_ecfps",
            lambda mols: np.ones((len(mols), 2)),
        )

        def fit(feats, targets, bootstrap_num):
            lime._r2_box = [1.0, 1.0]
            return np.array([[2.0, -4.0], [6.0, -8.0]])

        monkeypatch.setattr(lime, "_fit", fit)

        result = lime.explain(
            small_mol_list[:3], small_regression_targets[:3], bootstrap_num=2
        ).set_index("Descriptor")

        assert result.loc["F_0", "Coefficient"] == pytest.approx(4.0)
        assert result.loc["F_1", "Coefficient"] == pytest.approx(-6.0)
        assert result.loc["F_0", "Standard deviation"] == pytest.approx(
            np.std([2.0, 6.0])
        )


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
