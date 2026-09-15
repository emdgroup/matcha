"""Tests for matcha.explainability.explainer (MatchaExplainer and MatchaExplanation)."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
import plotly.graph_objects as go

import matcha.explainability.explainer as explainer_module
from matcha.explainability.explainer import MatchaExplainer, MatchaExplanation


# ===================================================================
# MatchaExplainer – initialisation
# ===================================================================


class TestMatchaExplainerInit:
    """Tests for MatchaExplainer.__init__."""

    def test_default_init(self):
        exp = MatchaExplainer()
        assert exp._pos_params is not None
        assert exp._nitrogen_walk_params is not None

    def test_no_r_group_params_attribute(self):
        exp = MatchaExplainer()
        assert not hasattr(exp, "_r_group_params")

    def test_default_substituents_contain_fragments(self):
        exp = MatchaExplainer()
        substituents = exp._pos_params["substituents"]
        fragment_subs = [s for s in substituents if s.startswith("[*]")]
        assert len(fragment_subs) > 0

    def test_default_substituents_no_primary_amine(self):
        exp = MatchaExplainer()
        substituents = exp._pos_params["substituents"]
        assert "N" not in substituents

    def test_default_pos_params(self):
        exp = MatchaExplainer()
        assert "substituents" in exp._pos_params
        assert "anchors" in exp._pos_params
        assert "num_sub" not in exp._pos_params

    def test_default_nitrogen_walk_params(self):
        exp = MatchaExplainer()
        assert exp._nitrogen_walk_params == {}

    def test_custom_pos_params(self):
        custom = {"substituents": ["F"], "anchors": ["[cH]"]}
        exp = MatchaExplainer(positional_analogue_scanning_params=custom)
        assert exp._pos_params == custom

    def test_custom_nitrogen_walk_params(self):
        custom = {"timeout": 5.0}
        exp = MatchaExplainer(nitrogen_walk_params=custom)
        assert exp._nitrogen_walk_params == custom

    def test_disable_pos_params(self):
        exp = MatchaExplainer(positional_analogue_scanning_params=None)
        assert exp._pos_params is None

    def test_disable_nitrogen_walk(self):
        exp = MatchaExplainer(nitrogen_walk_params=None)
        assert exp._nitrogen_walk_params is None

    def test_default_descriptor_set_none(self):
        exp = MatchaExplainer()
        assert exp._descriptor_set is None

    def test_custom_descriptor_set(self):
        exp = MatchaExplainer(lime_descriptor_set=["MolWt", "MolLogP"])
        assert exp._descriptor_set == ["MolWt", "MolLogP"]

    def test_default_fingerprint_params_none(self):
        exp = MatchaExplainer()
        assert exp._fingerprint_params is None

    def test_custom_fingerprint_params(self):
        fp = {"nBits": 2048, "radius": 2, "useFeatures": True}
        exp = MatchaExplainer(lime_fingerprint_params=fp)
        assert exp._fingerprint_params == fp

    def test_removed_lime_scale_coeff_argument_is_rejected(self):
        with pytest.raises(TypeError, match="lime_scale_coeff"):
            MatchaExplainer(lime_scale_coeff=True)

    def test_default_remove_noise_true(self):
        exp = MatchaExplainer()
        assert exp._remove_noise is True

    def test_remove_noise_false(self):
        exp = MatchaExplainer(lime_remove_noise=False)
        assert exp._remove_noise is False

    def test_reverse_defaults_on_and_can_be_disabled(self):
        assert MatchaExplainer()._reverse_positional_analogue_scanning is True
        assert (
            MatchaExplainer(
                reverse_positional_analogue_scanning=False
            )._reverse_positional_analogue_scanning
            is False
        )

    def test_remove_noise_reverse_and_timeout_are_trailing_positional_arguments(self):
        positional = {"substituents": ["C"], "anchors": []}
        nitrogen = {"timeout": 3.0}
        exp = MatchaExplainer(
            positional,
            nitrogen,
            ["MolWt"],
            {"radius": 2},
            False,
            False,
            2.5,
        )

        assert exp._pos_params == positional
        assert exp._nitrogen_walk_params == nitrogen
        assert exp._descriptor_set == ["MolWt"]
        assert exp._fingerprint_params == {"radius": 2}
        assert exp._remove_noise is False
        assert exp._reverse_positional_analogue_scanning is False
        assert exp._generation_timeout == 2.5

    def test_generation_timeout_defaults_to_sixty_seconds(self):
        assert MatchaExplainer()._generation_timeout == 60.0

    def test_num_sample_defaults_to_one_hundred(self):
        assert MatchaExplainer()._num_sample == 100

    def test_num_sample_can_be_overridden(self):
        assert MatchaExplainer(num_sample=42)._num_sample == 42

    def test_random_seed_defaults_to_zero(self):
        assert MatchaExplainer()._random_seed == 0

    def test_random_seed_can_be_overridden(self):
        assert MatchaExplainer(random_seed=-7)._random_seed == -7

    @pytest.mark.parametrize("num_sample", [-1, 1.5, "100", True])
    def test_num_sample_rejects_invalid_values(self, num_sample):
        with pytest.raises(Exception):
            MatchaExplainer(num_sample=num_sample)

    @pytest.mark.parametrize("random_seed", [1.5, "0", True])
    def test_random_seed_rejects_non_int(self, random_seed):
        with pytest.raises(Exception):
            MatchaExplainer(random_seed=random_seed)

    def test_default_positional_parameters_are_copied(self, monkeypatch):
        defaults = {
            "substituents": ["F"],
            "anchors": ["[cH]"],
        }
        monkeypatch.setattr(explainer_module, "_pos_params", defaults)

        first = MatchaExplainer()
        second = MatchaExplainer()
        first._pos_params["substituents"].append("Cl")

        assert second._pos_params["substituents"] == ["F"]
        assert defaults["substituents"] == ["F"]


# ===================================================================
# MatchaExplainer – generate_analogues
# ===================================================================


class TestMatchaExplainerGenerateAnalogues:
    """Tests for MatchaExplainer.generate_analogues."""

    def test_returns_list(self, default_explainer, single_mol):
        result = default_explainer.generate_analogues(single_mol)
        assert isinstance(result, list)

    def test_returns_mol_objects(self, default_explainer, single_mol):
        result = default_explainer.generate_analogues(single_mol)
        assert all(isinstance(m, Mol) for m in result)

    def test_generates_analogues(self, default_explainer, single_mol):
        result = default_explainer.generate_analogues(single_mol)
        assert len(result) > 0

    def test_forwards_reverse_setting_and_generation_timeout(self, monkeypatch):
        calls = []

        def generate(
            cls,
            mol,
            pos_params,
            nitrogen_params,
            reverse,
            timeout,
            num_sample,
            random_seed,
        ):
            calls.append(
                (
                    mol,
                    pos_params,
                    nitrogen_params,
                    reverse,
                    timeout,
                    num_sample,
                    random_seed,
                )
            )
            return []

        monkeypatch.setattr(
            explainer_module.AnalogueGenerator,
            "generate_analogues",
            classmethod(generate),
        )
        mol = Chem.MolFromSmiles("Cc1ccccc1")
        exp = MatchaExplainer(
            reverse_positional_analogue_scanning=False,
            generation_timeout=2.5,
            num_sample=7,
            random_seed=13,
        )

        assert exp.generate_analogues(mol) == []
        assert calls == [
            (
                mol,
                exp._pos_params,
                exp._nitrogen_walk_params,
                False,
                2.5,
                7,
                13,
            )
        ]

    def test_generation_timeout_propagates(self):
        exp = MatchaExplainer(generation_timeout=0)

        with pytest.raises(TimeoutError, match="no partial results"):
            exp.generate_analogues(Chem.MolFromSmiles("c1ccccc1"))

    def test_reverse_noops_when_positional_generation_is_disabled(self):
        exp = MatchaExplainer(
            positional_analogue_scanning_params=None,
            nitrogen_walk_params=None,
        )

        assert exp.generate_analogues(Chem.MolFromSmiles("Cc1ccccc1")) == []


# ===================================================================
# MatchaExplainer – decompose
# ===================================================================


class TestMatchaExplainerDecompose:
    """Tests for MatchaExplainer.decompose."""

    def test_returns_list(self, default_explainer, single_mol):
        result = default_explainer.decompose(single_mol)
        assert isinstance(result, list)

    def test_returns_mol_objects(self, default_explainer, single_mol):
        result = default_explainer.decompose(single_mol)
        assert all(isinstance(m, Mol) for m in result)


# ===================================================================
# MatchaExplainer – _run_lime_desc
# ===================================================================


class TestMatchaExplainerLimeDesc:
    """Tests for MatchaExplainer._run_lime_desc."""

    def test_constructs_lime_with_keywords_and_seed(self, monkeypatch):
        calls = []

        class RecordingLIME:
            def __init__(self, **kwargs):
                calls.append(kwargs)

            def explain(self, mols, predictions, bootstrap_num):
                return pd.DataFrame()

        monkeypatch.setattr(explainer_module, "LIME", RecordingLIME)
        explainer = MatchaExplainer(
            lime_descriptor_set=["MolWt"],
            lime_fingerprint_params={"radius": 2},
            random_seed=17,
        )

        explainer._run_lime_desc([], np.array([]), bootstrap_num=3)

        assert calls == [
            {
                "descriptor_set": ["MolWt"],
                "fingerprint_params": {"radius": 2},
                "use_fingerprints": False,
                "random_seed": 17,
            }
        ]

    def test_returns_dataframe(
        self, default_explainer, small_mol_list, small_regression_targets
    ):
        df = default_explainer._run_lime_desc(
            small_mol_list, small_regression_targets, bootstrap_num=3
        )
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(
        self, default_explainer, small_mol_list, small_regression_targets
    ):
        df = default_explainer._run_lime_desc(
            small_mol_list, small_regression_targets, bootstrap_num=3
        )
        assert "Descriptor" in df.columns
        assert "Coefficient" in df.columns
        assert "Standard deviation" in df.columns


# ===================================================================
# MatchaExplainer – _run_lime_ecfp
# ===================================================================


class TestMatchaExplainerLimeEcfp:
    """Tests for MatchaExplainer._run_lime_ecfp."""

    def test_constructs_lime_with_keywords_and_seed(self, monkeypatch, single_mol):
        calls = []

        class RecordingLIME:
            def __init__(self, **kwargs):
                calls.append(kwargs)

            def explain(self, mols, predictions, bootstrap_num):
                return pd.DataFrame()

            def get_attributions(self, mol, result):
                return {}, {}, np.zeros(mol.GetNumAtoms())

        monkeypatch.setattr(explainer_module, "LIME", RecordingLIME)
        explainer = MatchaExplainer(
            lime_descriptor_set=["MolWt"],
            lime_fingerprint_params={"radius": 2},
            random_seed=17,
        )

        explainer._run_lime_ecfp([single_mol], np.array([1.0]), bootstrap_num=3)

        assert calls == [
            {
                "descriptor_set": ["MolWt"],
                "fingerprint_params": {"radius": 2},
                "use_fingerprints": True,
                "random_seed": 17,
            }
        ]

    def test_returns_tuple(
        self, default_explainer, small_mol_list, small_regression_targets
    ):
        result = default_explainer._run_lime_ecfp(
            small_mol_list, small_regression_targets, bootstrap_num=3
        )
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_returns_envs_and_weights(
        self, default_explainer, small_mol_list, small_regression_targets
    ):
        envs, weights, atom_weights = default_explainer._run_lime_ecfp(
            small_mol_list, small_regression_targets, bootstrap_num=3
        )
        assert isinstance(envs, dict)
        assert isinstance(weights, dict)
        assert isinstance(atom_weights, np.ndarray)


# ===================================================================
# MatchaExplainer – explain (integration)
# ===================================================================


class TestMatchaExplainerExplain:
    """Tests for MatchaExplainer.explain (end-to-end)."""

    def test_returns_explanation(
        self, default_explainer, small_mol_list, small_regression_targets
    ):
        result = default_explainer.explain(
            small_mol_list, small_regression_targets, bootstrap_num=3
        )
        assert isinstance(result, MatchaExplanation)

    def test_explanation_has_df_desc(
        self, default_explainer, small_mol_list, small_regression_targets
    ):
        result = default_explainer.explain(
            small_mol_list, small_regression_targets, bootstrap_num=3
        )
        assert isinstance(result.df_desc, pd.DataFrame)

    def test_explanation_has_envs(
        self, default_explainer, small_mol_list, small_regression_targets
    ):
        result = default_explainer.explain(
            small_mol_list, small_regression_targets, bootstrap_num=3
        )
        assert isinstance(result.envs, dict)

    def test_explanation_has_weights(
        self, default_explainer, small_mol_list, small_regression_targets
    ):
        result = default_explainer.explain(
            small_mol_list, small_regression_targets, bootstrap_num=3
        )
        assert isinstance(result.weights, dict)

    def test_explanation_has_atom_weights(
        self, default_explainer, small_mol_list, small_regression_targets
    ):
        result = default_explainer.explain(
            small_mol_list, small_regression_targets, bootstrap_num=3
        )
        assert isinstance(result.atom_weights, np.ndarray)
        assert result.atom_weights.shape == (small_mol_list[0].GetNumAtoms(),)

    def test_explanation_has_analogues(
        self, default_explainer, small_mol_list, small_regression_targets
    ):
        result = default_explainer.explain(
            small_mol_list, small_regression_targets, bootstrap_num=3
        )
        assert isinstance(result.analogues, list)

    def test_explanation_mol_set(
        self, default_explainer, small_mol_list, small_regression_targets
    ):
        result = default_explainer.explain(
            small_mol_list, small_regression_targets, bootstrap_num=3
        )
        assert result._mol is not None

    @pytest.mark.parametrize(
        ("molecule_count", "predictions", "bootstrap_num", "message"),
        [
            (3, np.array([1.0, 2.0]), 1, "molecule and prediction counts must match"),
            (
                3,
                np.array([[1.0], [2.0], [3.0]]),
                1,
                "predictions must be one-dimensional",
            ),
            (2, np.array([1.0, 2.0]), 1, "at least 3 molecules"),
            (3, np.array([1.0, 2.0, 3.0]), 0, "bootstrap_num must be at least 1"),
        ],
    )
    def test_rejects_invalid_lime_inputs_before_running_either_path(
        self,
        monkeypatch,
        small_mol_list,
        molecule_count,
        predictions,
        bootstrap_num,
        message,
    ):
        explainer = MatchaExplainer(lime_descriptor_set=["MolWt", "MolLogP"])
        descriptor_lime = Mock()
        fingerprint_lime = Mock()
        monkeypatch.setattr(explainer, "_run_lime_desc", descriptor_lime)
        monkeypatch.setattr(explainer, "_run_lime_ecfp", fingerprint_lime)

        with pytest.raises(ValueError, match=message):
            explainer.explain(
                small_mol_list[:molecule_count], predictions, bootstrap_num
            )

        descriptor_lime.assert_not_called()
        fingerprint_lime.assert_not_called()

    def test_forwards_exact_fit_count_above_row_and_feature_counts(
        self, monkeypatch, small_mol_list
    ):
        explainer = MatchaExplainer(lime_descriptor_set=["MolWt", "MolLogP"])
        df_desc = pd.DataFrame(
            {
                "Descriptor": ["Local fit R2"],
                "Coefficient": [1.0],
                "Standard deviation": [0.0],
            }
        )
        descriptor_lime = Mock(return_value=df_desc)
        fingerprint_lime = Mock(return_value=({}, {}, np.zeros(3)))
        monkeypatch.setattr(explainer, "_run_lime_desc", descriptor_lime)
        monkeypatch.setattr(explainer, "_run_lime_ecfp", fingerprint_lime)
        predictions = np.array([1.0, 2.0, 3.0])

        explainer.explain(small_mol_list[:3], predictions, bootstrap_num=5)

        descriptor_lime.assert_called_once_with(small_mol_list[:3], predictions, 5)
        fingerprint_lime.assert_called_once_with(small_mol_list[:3], predictions, 5)


# ===================================================================
# MatchaExplanation – initialisation
# ===================================================================


class TestMatchaExplanationInit:
    """Tests for MatchaExplanation.__init__."""

    def test_stores_df_desc(self):
        df = pd.DataFrame(
            {"Descriptor": ["a"], "Coefficient": [0.5], "Standard deviation": [0.1]}
        )
        expl = MatchaExplanation(df, {}, {}, Chem.MolFromSmiles("CCO"), [], np.zeros(3))
        assert expl.df_desc is df

    def test_stores_envs(self):
        envs = {0: [1, 2]}
        expl = MatchaExplanation(
            pd.DataFrame(), envs, {}, Chem.MolFromSmiles("CCO"), [], np.zeros(3)
        )
        assert expl.envs is envs

    def test_stores_weights(self):
        weights = {0: 0.5}
        expl = MatchaExplanation(
            pd.DataFrame(), {}, weights, Chem.MolFromSmiles("CCO"), [], np.zeros(3)
        )
        assert expl.weights is weights

    def test_stores_analogues(self):
        analogues = ["CCO", "CC(C)C"]
        expl = MatchaExplanation(
            pd.DataFrame(),
            {},
            {},
            Chem.MolFromSmiles("CCO"),
            analogues,
            np.zeros(3),
        )
        assert expl.analogues == analogues

    def test_stores_mol(self):
        mol = Chem.MolFromSmiles("CCO")
        expl = MatchaExplanation(pd.DataFrame(), {}, {}, mol, [], np.zeros(3))
        assert expl._mol is mol

    def test_stores_owned_raw_atom_weights(self):
        mol = Chem.MolFromSmiles("CCO")
        atom_weights = np.array([1.0, -0.5, 0.0])

        expl = MatchaExplanation(pd.DataFrame(), {}, {}, mol, [], atom_weights)
        atom_weights[0] = 99.0

        assert expl.atom_weights.dtype == np.float64
        assert expl.atom_weights == pytest.approx([1.0, -0.5, 0.0])


# ===================================================================
# MatchaExplanation – plot_coefficients
# ===================================================================


class TestMatchaExplanationPlotCoefficients:
    """Tests for MatchaExplanation.plot_coefficients."""

    @pytest.fixture()
    def explanation_for_plot(self):
        """Build a minimal MatchaExplanation with realistic data for plotting."""
        data = {
            "Descriptor": ["MolWt", "MolLogP", "TPSA", "NumHDonors", "Local fit R2"],
            "Coefficient": [0.5, -0.3, 0.2, -0.1, 0.85],
            "Standard deviation": [0.05, 0.02, 0.01, 0.15, 0.03],
        }
        df = pd.DataFrame(data)
        mol = Chem.MolFromSmiles("CCO")
        return MatchaExplanation(df, {}, {}, mol, [], np.zeros(mol.GetNumAtoms()))

    def test_returns_figure(self, explanation_for_plot):
        fig = explanation_for_plot.plot_coefficients()
        assert isinstance(fig, go.Figure)

    def test_remove_noise_filters(self, explanation_for_plot):
        fig = explanation_for_plot.plot_coefficients(remove_noise=True)
        assert isinstance(fig, go.Figure)

    def test_no_remove_noise(self, explanation_for_plot):
        fig = explanation_for_plot.plot_coefficients(remove_noise=False)
        assert isinstance(fig, go.Figure)

    def test_keep_k_limits_bars(self, explanation_for_plot):
        fig = explanation_for_plot.plot_coefficients(keep_k=2)
        assert isinstance(fig, go.Figure)

    def test_normalizes_coefficients_and_standard_deviations_for_display(self):
        df = pd.DataFrame(
            {
                "Descriptor": ["positive", "negative", "Local fit R2"],
                "Coefficient": [4.0, -2.0, 0.9],
                "Standard deviation": [0.8, 0.4, 0.02],
            }
        )
        expl = MatchaExplanation(df, {}, {}, Chem.MolFromSmiles("CCO"), [], np.zeros(3))

        fig = expl.plot_coefficients(remove_noise=False)

        assert list(fig.data[0].y) == ["negative", "positive"]
        assert list(fig.data[0].x) == pytest.approx([-2.0 / 6.0, 4.0 / 6.0])
        assert list(fig.data[0].error_x.array) == pytest.approx([0.4 / 6.0, 0.8 / 6.0])

    def test_raw_filter_matches_l1_normalized_equivalent(self):
        raw = pd.DataFrame(
            {
                "Descriptor": ["dominant", "medium", "small", "Local fit R2"],
                "Coefficient": [8.0, 2.0, 0.5, 0.9],
                "Standard deviation": [0.1, 0.1, 0.1, 0.02],
            }
        )
        normalized = raw.copy()
        normalized.loc[:2, ["Coefficient", "Standard deviation"]] /= 10.5
        mol = Chem.MolFromSmiles("CCO")

        atom_weights = np.zeros(mol.GetNumAtoms())
        raw_fig = MatchaExplanation(
            raw, {}, {}, mol, [], atom_weights
        ).plot_coefficients()
        normalized_fig = MatchaExplanation(
            normalized, {}, {}, mol, [], atom_weights
        ).plot_coefficients()

        assert list(raw_fig.data[0].y) == list(normalized_fig.data[0].y)
        assert list(raw_fig.data[0].x) == pytest.approx(normalized_fig.data[0].x)
        assert list(raw_fig.data[0].error_x.array) == pytest.approx(
            normalized_fig.data[0].error_x.array
        )

    def test_plot_coefficients_does_not_mutate_raw_dataframe(self):
        df = pd.DataFrame(
            {
                "Descriptor": ["first", "second", "Local fit R2"],
                "Coefficient": [3.0, -1.0, 0.9],
                "Standard deviation": [0.3, 0.1, 0.02],
            }
        )
        original = df.copy(deep=True)
        expl = MatchaExplanation(df, {}, {}, Chem.MolFromSmiles("CCO"), [], np.zeros(3))

        expl.plot_coefficients(remove_noise=False)

        pd.testing.assert_frame_equal(expl.df_desc, original)

    def test_low_r2_plot(self):
        """When R² < 0.75, colorbar ticks should not include 'R²' label."""
        data = {
            "Descriptor": ["MolWt", "MolLogP", "Local fit R2"],
            "Coefficient": [0.5, -0.3, 0.5],  # R² = 0.5 < 0.75
            "Standard deviation": [0.05, 0.02, 0.03],
        }
        df = pd.DataFrame(data)
        mol = Chem.MolFromSmiles("CCO")
        expl = MatchaExplanation(df, {}, {}, mol, [], np.zeros(mol.GetNumAtoms()))
        fig = expl.plot_coefficients(remove_noise=False)
        assert isinstance(fig, go.Figure)

    def test_empty_after_noise_removal(self):
        """When all descriptors are noisy, plot_coefficients should raise ValueError."""
        data = {
            "Descriptor": ["MolWt", "Local fit R2"],
            "Coefficient": [0.01, 0.9],
            "Standard deviation": [0.5, 0.01],  # std > |coeff| -> noisy
        }
        df = pd.DataFrame(data)
        mol = Chem.MolFromSmiles("CCO")
        expl = MatchaExplanation(df, {}, {}, mol, [], np.zeros(mol.GetNumAtoms()))
        with pytest.raises(ValueError, match="No reliable coefficients remain"):
            expl.plot_coefficients(remove_noise=True)


# ===================================================================
# MatchaExplanation – plot_heatmap
# ===================================================================


class TestMatchaExplanationPlotHeatmap:
    """Tests for MatchaExplanation.plot_heatmap."""

    @pytest.fixture()
    def explanation_for_heatmap(self):
        """Build a MatchaExplanation with envs/weights for heatmap plotting."""
        mol = Chem.MolFromSmiles("c1ccc(O)cc1")  # phenol
        n_atoms = mol.GetNumAtoms()
        # Create fake envs and weights mapping to atoms in the molecule
        envs = {0: list(range(n_atoms)), 1: [0, 1, 2]}
        weights = {0: 100.0, 1: -100.0}
        atom_weights = np.linspace(-2.0, 4.0, n_atoms)
        df = pd.DataFrame()
        return MatchaExplanation(df, envs, weights, mol, [], atom_weights)

    def test_returns_figure(self, explanation_for_heatmap):
        result = explanation_for_heatmap.plot_heatmap()
        assert isinstance(result, go.Figure)

    def test_custom_colormap(self, explanation_for_heatmap):
        result = explanation_for_heatmap.plot_heatmap(colormap="coolwarm")
        assert isinstance(result, go.Figure)

    def test_figure_contains_image(self, explanation_for_heatmap):
        result = explanation_for_heatmap.plot_heatmap()
        assert len(result.layout.images) > 0

    def test_uses_normalized_atom_weight_copy_only(
        self, monkeypatch, explanation_for_heatmap
    ):
        captured = []
        original_renderer = explainer_module.SimilarityMaps.GetSimilarityMapFromWeights
        original_atom_weights = explanation_for_heatmap.atom_weights.copy()

        def capture_weights(mol, weights, **kwargs):
            captured.extend(weights)
            return original_renderer(mol, weights, **kwargs)

        monkeypatch.setattr(
            explainer_module.SimilarityMaps,
            "GetSimilarityMapFromWeights",
            capture_weights,
        )

        explanation_for_heatmap.plot_heatmap()

        assert captured == pytest.approx(
            np.linspace(-1.0, 1.0, explanation_for_heatmap._mol.GetNumAtoms())
        )
        assert explanation_for_heatmap.atom_weights == pytest.approx(
            original_atom_weights
        )
