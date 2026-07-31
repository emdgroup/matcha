"""Tests for matcha.explainability.explainer (MatchaExplainer and MatchaExplanation)."""

import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
import plotly.graph_objects as go

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
        assert "num_sub" in exp._pos_params

    def test_default_nitrogen_walk_params(self):
        exp = MatchaExplainer()
        assert exp._nitrogen_walk_params == {"num_sub": 1}

    def test_custom_pos_params(self):
        custom = {"substituents": ["F"], "anchors": ["[cH]"], "num_sub": 2}
        exp = MatchaExplainer(positional_analogue_scanning_params=custom)
        assert exp._pos_params == custom

    def test_custom_nitrogen_walk_params(self):
        custom = {"num_sub": 3}
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

    def test_default_scale_coeff_true(self):
        exp = MatchaExplainer()
        assert exp._scale_coeff is True

    def test_scale_coeff_false(self):
        exp = MatchaExplainer(lime_scale_coeff=False)
        assert exp._scale_coeff is False

    def test_default_remove_noise_true(self):
        exp = MatchaExplainer()
        assert exp._remove_noise is True

    def test_remove_noise_false(self):
        exp = MatchaExplainer(lime_remove_noise=False)
        assert exp._remove_noise is False


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


@pytest.mark.filterwarnings("ignore:Degrees of freedom <= 0 for slice.:RuntimeWarning")
class TestMatchaExplainerLimeDesc:
    """Tests for MatchaExplainer._run_lime_desc."""

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


@pytest.mark.filterwarnings("ignore:Degrees of freedom <= 0 for slice.:RuntimeWarning")
class TestMatchaExplainerLimeEcfp:
    """Tests for MatchaExplainer._run_lime_ecfp."""

    def test_returns_tuple(
        self, default_explainer, small_mol_list, small_regression_targets
    ):
        result = default_explainer._run_lime_ecfp(
            small_mol_list, small_regression_targets, bootstrap_num=3
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_envs_and_weights(
        self, default_explainer, small_mol_list, small_regression_targets
    ):
        envs, weights = default_explainer._run_lime_ecfp(
            small_mol_list, small_regression_targets, bootstrap_num=3
        )
        assert isinstance(envs, dict)
        assert isinstance(weights, dict)


# ===================================================================
# MatchaExplainer – explain (integration)
# ===================================================================


@pytest.mark.filterwarnings("ignore:Degrees of freedom <= 0 for slice.:RuntimeWarning")
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


# ===================================================================
# MatchaExplanation – initialisation
# ===================================================================


class TestMatchaExplanationInit:
    """Tests for MatchaExplanation.__init__."""

    def test_stores_df_desc(self):
        df = pd.DataFrame(
            {"Descriptor": ["a"], "Coefficient": [0.5], "Standard deviation": [0.1]}
        )
        expl = MatchaExplanation(df, {}, {}, Chem.MolFromSmiles("CCO"), [])
        assert expl.df_desc is df

    def test_stores_envs(self):
        envs = {0: [1, 2]}
        expl = MatchaExplanation(
            pd.DataFrame(), envs, {}, Chem.MolFromSmiles("CCO"), []
        )
        assert expl.envs is envs

    def test_stores_weights(self):
        weights = {0: 0.5}
        expl = MatchaExplanation(
            pd.DataFrame(), {}, weights, Chem.MolFromSmiles("CCO"), []
        )
        assert expl.weights is weights

    def test_stores_analogues(self):
        analogues = ["CCO", "CC(C)C"]
        expl = MatchaExplanation(
            pd.DataFrame(), {}, {}, Chem.MolFromSmiles("CCO"), analogues
        )
        assert expl.analogues == analogues

    def test_stores_mol(self):
        mol = Chem.MolFromSmiles("CCO")
        expl = MatchaExplanation(pd.DataFrame(), {}, {}, mol, [])
        assert expl._mol is mol


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
        return MatchaExplanation(df, {}, {}, mol, [])

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

    def test_low_r2_plot(self):
        """When R² < 0.75, colorbar ticks should not include 'R²' label."""
        data = {
            "Descriptor": ["MolWt", "MolLogP", "Local fit R2"],
            "Coefficient": [0.5, -0.3, 0.5],  # R² = 0.5 < 0.75
            "Standard deviation": [0.05, 0.02, 0.03],
        }
        df = pd.DataFrame(data)
        mol = Chem.MolFromSmiles("CCO")
        expl = MatchaExplanation(df, {}, {}, mol, [])
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
        expl = MatchaExplanation(df, {}, {}, mol, [])
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
        weights = {0: 0.5, 1: -0.3}
        df = pd.DataFrame()
        return MatchaExplanation(df, envs, weights, mol, [])

    def test_returns_figure(self, explanation_for_heatmap):
        result = explanation_for_heatmap.plot_heatmap()
        assert isinstance(result, go.Figure)

    def test_custom_colormap(self, explanation_for_heatmap):
        result = explanation_for_heatmap.plot_heatmap(colormap="coolwarm")
        assert isinstance(result, go.Figure)

    def test_figure_contains_image(self, explanation_for_heatmap):
        result = explanation_for_heatmap.plot_heatmap()
        # The figure should contain a layout image
        assert len(result.layout.images) > 0
