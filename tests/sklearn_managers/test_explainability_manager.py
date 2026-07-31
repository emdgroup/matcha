"""Test ExplainabilityManager through the sklearn API.

Model: SNNRegressor (tabular)
Exercises: explain_prediction (LIME), explainer property, create_explainer.

LIME's internal k-fold cross-validation requires at least ``bootstrap_num``
molecules (analogues + the query molecule).  We keep ``lime_bootstrap_num``
small (3) and enable all analogue generators so there is enough data.
"""

import pytest
from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from matcha.sklearn.tabular import SNNRegressor
from matcha.sklearn.managers import ExplainabilityManager


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


class TestExplainabilityManagerInit:
    """Tests for initial state of ExplainabilityManager."""

    def test_explainer_is_none_by_default(self):
        mgr = ExplainabilityManager()
        assert mgr.explainer is None


@pytest.mark.filterwarnings("ignore:Degrees of freedom <= 0 for slice.:RuntimeWarning")
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
