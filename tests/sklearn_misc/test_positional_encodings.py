"""Test graph positional encodings with different model / PE combinations.

Each PE type is tested with a *different* graph model to maximise coverage:

* GINRegressor       + laplacian_k=4
* GatedGCNRegressor  + rwse_k=4
* GPSRegressor       + rrwp_k=4
* GTRegressor        + distmat_k=4
* AttentiveFPRegressor + elstatic_k=4

All remaining PE knobs are set to 0 so only the target encoding is active.

Runtime optimisation: each (model, PE) combination is fitted once via a
parametrized fixture.
"""

import numpy as np
import pytest

from matcha.sklearn.graph import (
    AttentiveFPRegressor,
    GatedGCNRegressor,
    GINRegressor,
    GPSRegressor,
    GTRegressor,
)

# ── Shared fragment ─────────────────────────────────────────────────────
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

_SMALL_ARCH = dict(
    enc_num_layers=1,
    enc_atom_hidden_dim=8,
    pred_hidden_dims=[8],
)


# ── fixtures parametrized over (model_cls, override_kwargs) ─────────────

_PE_CASES = [
    pytest.param(
        (GINRegressor, {**_BASE, **_PE_OFF, **_SMALL_ARCH, "laplacian_k": 4}),
        id="GIN-laplacian",
    ),
    pytest.param(
        (GatedGCNRegressor, {**_BASE, **_PE_OFF, **_SMALL_ARCH, "rwse_k": 4}),
        id="GatedGCN-rwse",
    ),
    pytest.param(
        (
            GPSRegressor,
            {
                **_BASE,
                **_PE_OFF,
                **_SMALL_ARCH,
                "enc_num_heads": 2,
                "enc_expansion_k": 1,
                "rrwp_k": 4,
            },
        ),
        id="GPS-rrwp",
    ),
    pytest.param(
        (
            GTRegressor,
            {
                **_BASE,
                **_PE_OFF,
                **_SMALL_ARCH,
                "enc_num_heads": 2,
                "enc_expansion_k": 1,
                "distmat_k": 4,
            },
        ),
        id="GT-distmat",
    ),
    pytest.param(
        (AttentiveFPRegressor, {**_BASE, **_PE_OFF, **_SMALL_ARCH, "elstatic_k": 4}),
        id="AttentiveFP-elstatic",
    ),
]


@pytest.fixture(params=_PE_CASES)
def fitted_pe_model(request, mol_list, regression_y):
    """Fit each (model, PE) combo once; reuse for all assertions."""
    model_cls, kwargs = request.param
    model = model_cls(**kwargs)
    model.fit(mol_list, regression_y)
    return model


class TestPositionalEncodings:
    def test_predict_returns_finite(self, fitted_pe_model, mol_list):
        preds = fitted_pe_model.predict(mol_list)
        assert isinstance(preds, np.ndarray)
        assert preds.shape[0] == len(mol_list)
        assert np.all(np.isfinite(preds))
