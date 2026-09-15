"""Test ExplainabilityManager through the sklearn API.

Model: SNNRegressor (tabular)
Exercises: explain_prediction (LIME), explainer property, create_explainer.
"""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem.rdchem import Mol

import matcha.sklearn.managers.explainability_manager as manager_module
from matcha.sklearn.managers import ExplainabilityManager
from matcha.sklearn.tabular import SNNRegressor


@pytest.fixture()
def model_kwargs():
    return dict(
        hidden_dims=[32],
        num_parallel=4,
        feature_list=["ECFP"],
        num_epochs=1,
        batch_size=32,
        accelerator="cpu",
        devices=1,
        early_stopping=False,
        stochastic_weight_averaging=False,
    )


@pytest.fixture()
def fitted_model(mol_list: list[Mol], regression_y, model_kwargs):
    model = SNNRegressor(**model_kwargs)
    model.fit(mol_list, regression_y)
    return model


# Aromatic molecule with substituent — sufficient for the analogue generators
# to produce at least a handful of variants.
_EXPLAIN_MOL = Chem.MolFromSmiles("c1ccc(O)cc1")
_PAS_PARAMS = {
    "substituents": ["F", "[*]O"],
    "anchors": ["[cH]"],
}


class TestExplainabilityManagerInit:
    """Tests for initial state of ExplainabilityManager."""

    def test_explainer_is_none_by_default(self):
        mgr = ExplainabilityManager()
        assert mgr.explainer is None


class TestExplainabilityManagerExplain:
    """Tests for explain_prediction through the sklearn API."""

    def test_explain_returns_explanation_object(self, fitted_model):
        explanation = fitted_model.explain_prediction(
            input=_EXPLAIN_MOL,
            task_idx=0,
            lime_bootstrap_num=3,
        )
        assert explanation is not None

    def test_explainer_property_after_create(self):
        mgr = ExplainabilityManager()
        mgr.create_explainer(
            {
                "positional_analogue_scanning_params": None,
                "nitrogen_walk_params": None,
            }
        )
        assert mgr.explainer is not None
        assert mgr.explainer.generate_analogues(Chem.MolFromSmiles("CCO")) == []

    def test_reverse_default_reaches_fitted_estimator(self, fitted_model):
        fitted_model._explainability_manager.create_explainer(
            {
                "positional_analogue_scanning_params": _PAS_PARAMS,
                "nitrogen_walk_params": None,
            }
        )

        explanation = fitted_model.explain_prediction(
            input=_EXPLAIN_MOL,
            task_idx=0,
            lime_bootstrap_num=3,
        )

        assert "c1ccccc1" in explanation.analogues

    def test_caller_can_disable_reverse_for_fitted_estimator(self, fitted_model):
        fitted_model._explainability_manager.create_explainer(
            {
                "positional_analogue_scanning_params": _PAS_PARAMS,
                "nitrogen_walk_params": None,
                "reverse_positional_analogue_scanning": False,
            }
        )

        explanation = fitted_model.explain_prediction(
            input=_EXPLAIN_MOL,
            task_idx=0,
            lime_bootstrap_num=3,
        )

        assert "c1ccccc1" not in explanation.analogues

    def test_implicit_explainer_reuses_reverse_default(self, monkeypatch):
        implicit = Mock()
        implicit.generate_analogues.return_value = [
            Chem.MolFromSmiles("CC"),
            Chem.MolFromSmiles("CCC"),
        ]
        constructor = Mock(return_value=implicit)
        monkeypatch.setattr(manager_module, "MatchaExplainer", constructor)
        mgr = ExplainabilityManager()
        expected = object()
        monkeypatch.setattr(mgr, "_get_explanations", Mock(return_value=expected))

        result = mgr.explain(Mock(), _EXPLAIN_MOL)

        assert result is expected
        assert (
            "reverse_positional_analogue_scanning" not in constructor.call_args.kwargs
        )

    def test_configured_fingerprint_and_seed_overrides_are_honored(self):
        mgr = ExplainabilityManager()

        mgr.create_explainer(
            {
                "positional_analogue_scanning_params": None,
                "nitrogen_walk_params": None,
                "lime_fingerprint_params": {"nBits": 32, "radius": 1},
                "random_seed": -11,
            }
        )

        assert mgr.explainer._fingerprint_params == {"nBits": 32, "radius": 1}
        assert mgr.explainer._random_seed == -11

    def test_caller_configured_reverse_setting_is_honored(self, monkeypatch):
        mgr = ExplainabilityManager()
        mgr.create_explainer(
            {
                "positional_analogue_scanning_params": None,
                "nitrogen_walk_params": None,
                "reverse_positional_analogue_scanning": False,
            }
        )
        configured = mgr.explainer
        expected = object()
        monkeypatch.setattr(
            configured,
            "generate_analogues",
            Mock(
                return_value=[
                    Chem.MolFromSmiles("CC"),
                    Chem.MolFromSmiles("CCC"),
                ]
            ),
        )
        monkeypatch.setattr(mgr, "_get_explanations", Mock(return_value=expected))

        result = mgr.explain(Mock(), _EXPLAIN_MOL)

        assert result is expected
        assert configured._reverse_positional_analogue_scanning is False
        configured.generate_analogues.assert_called_once_with(_EXPLAIN_MOL)

    def test_forwards_uncapped_bootstrap_count_to_explainer(self):
        analogues = [Chem.MolFromSmiles("CC"), Chem.MolFromSmiles("CCC")]
        explainer = Mock()
        explainer.generate_analogues.return_value = analogues
        expected = object()
        explainer.explain.return_value = expected
        model = Mock()
        predictions = np.array([[1.0], [2.0], [3.0]])
        model._default_predict.return_value = predictions
        mgr = ExplainabilityManager()
        mgr._explainer = explainer

        result = mgr.explain(model, _EXPLAIN_MOL, lime_bootstrap_num=7, use_std=False)

        assert result is expected
        explainer.explain.assert_called_once()
        call = explainer.explain.call_args.kwargs
        assert call["mols"] == [_EXPLAIN_MOL, *analogues]
        assert call["predictions"] == pytest.approx(predictions[:, 0])
        assert call["bootstrap_num"] == 7

    def test_manager_matches_direct_explanation_for_identical_predictions(
        self, monkeypatch
    ):
        analogues = [Chem.MolFromSmiles("CC"), Chem.MolFromSmiles("CCC")]
        mols = [_EXPLAIN_MOL, *analogues]
        predictions = np.array([1.0, 2.0, 3.0])
        mgr = ExplainabilityManager()
        mgr.create_explainer(
            {
                "positional_analogue_scanning_params": None,
                "nitrogen_walk_params": None,
                "lime_descriptor_set": ["MolWt", "MolLogP"],
                "lime_fingerprint_params": {"nBits": 16, "radius": 1},
                "random_seed": 5,
            }
        )
        monkeypatch.setattr(
            mgr.explainer, "generate_analogues", Mock(return_value=analogues)
        )
        model = Mock()
        model._default_predict.return_value = predictions[:, np.newaxis]

        direct = mgr.explainer.explain(mols, predictions, bootstrap_num=4)
        managed = mgr.explain(model, _EXPLAIN_MOL, lime_bootstrap_num=4)

        pd.testing.assert_frame_equal(managed.df_desc, direct.df_desc)
        assert managed.envs == direct.envs
        assert managed.weights == direct.weights
        assert managed.atom_weights == pytest.approx(direct.atom_weights)

    def test_fallback_explainer_repeats_outputs_with_default_seed(self, monkeypatch):
        analogues = [Chem.MolFromSmiles("CC"), Chem.MolFromSmiles("CCC")]
        monkeypatch.setattr(
            manager_module.MatchaExplainer,
            "generate_analogues",
            lambda self, mol: analogues,
        )
        model = Mock()
        model._default_predict.return_value = np.array([[1.0], [2.0], [3.0]])
        mgr = ExplainabilityManager()

        first = mgr.explain(model, _EXPLAIN_MOL, lime_bootstrap_num=2)
        second = mgr.explain(model, _EXPLAIN_MOL, lime_bootstrap_num=2)

        pd.testing.assert_frame_equal(first.df_desc, second.df_desc)
        assert first.weights == second.weights
        assert first.atom_weights == pytest.approx(second.atom_weights)

    @pytest.mark.parametrize("use_std", [False, True])
    @pytest.mark.parametrize("analogue_smiles", [[], ["CC"]])
    def test_rejects_insufficient_neighborhood_before_model_work(
        self, use_std, analogue_smiles
    ):
        explainer = Mock()
        explainer.generate_analogues.return_value = [
            Chem.MolFromSmiles(smiles) for smiles in analogue_smiles
        ]
        model = Mock()
        molecule_count = len(analogue_smiles) + 1
        model._default_predict.return_value = np.zeros((molecule_count, 1))
        model.compute_uncertainty.return_value = np.zeros((molecule_count, 1))
        mgr = ExplainabilityManager()
        mgr._explainer = explainer

        with pytest.raises(
            ValueError,
            match=rf"requires at least 3 total molecules; received {molecule_count}",
        ):
            mgr.explain(model, _EXPLAIN_MOL, use_std=use_std)

        model._default_predict.assert_not_called()
        model.compute_uncertainty.assert_not_called()
        explainer.explain.assert_not_called()
