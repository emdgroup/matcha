"""Test ExplainabilityManager through the sklearn API.

Model: SNNRegressor (tabular)
Exercises: explain_prediction (LIME), explainer property, create_explainer.

LIME's internal k-fold cross-validation requires at least ``bootstrap_num``
molecules (analogues + the query molecule).  We keep ``lime_bootstrap_num``
small (3) and enable all analogue generators so there is enough data.
"""

from unittest.mock import Mock

import numpy as np
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
    "num_sub": 1,
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
