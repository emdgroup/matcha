"""Test feature_list=["estate"] with graph and CLM models.

Each modality is tested with a *different* model class:

* Graph:   AttentiveFPRegressor
* CLM:     CNNRegressor

Using ``feature_list=["estate"]`` triggers a CombinedDataModule that
concatenates the main representation (graph / CLM) with
EState tabular fingerprints.

Runtime optimisation: each (model, estate) combination is fitted once via
a parametrized fixture.
"""

import numpy as np
import pytest

from matcha.sklearn.clm import CNNRegressor
from matcha.sklearn.graph import AttentiveFPRegressor


# ── Shared ───────────────────────────────────────────────────────────────

_BASE = dict(
    num_epochs=1,
    batch_size=32,
    accelerator="cpu",
    devices=1,
    early_stopping=False,
    stochastic_weight_averaging=False,
)

_PE_OFF = dict(
    rwse_k=0,
    laplacian_k=0,
    elstatic_k=0,
    distmat_k=0,
    rrwp_k=0,
    num_virtual_nodes=0,
)


# ── Per-model kwargs ─────────────────────────────────────────────────────

_CASES = [
    pytest.param(
        (
            AttentiveFPRegressor,
            {
                **_BASE,
                **_PE_OFF,
                "enc_num_layers": 1,
                "enc_atom_hidden_dim": 32,
                "pred_hidden_dims": [32],
                "feature_list": ["estate"],
            },
        ),
        id="AttentiveFP-estate",
    ),
    pytest.param(
        (
            CNNRegressor,
            {
                **_BASE,
                "enc_hidden_dim": 32,
                "enc_kernel_dims": [3, 5],
                "enc_num_heads": 4,
                "pred_hidden_dims": [32],
                "num_augmentations": 1,
                "max_length": 100,
                "feature_list": ["estate"],
            },
        ),
        id="CNN-estate",
    ),
]


@pytest.fixture(params=_CASES)
def fitted_estate_model(request, mol_list, regression_y):
    """Fit each (model, estate) combo once; reuse for all assertions."""
    model_cls, kwargs = request.param
    model = model_cls(**kwargs)
    model.fit(mol_list, regression_y)
    return model


class TestFeatureListEstate:
    def test_predict_returns_finite(self, fitted_estate_model, mol_list):
        preds = fitted_estate_model.predict(mol_list)
        assert isinstance(preds, np.ndarray)
        assert preds.shape[0] == len(mol_list)
        assert np.all(np.isfinite(preds))
