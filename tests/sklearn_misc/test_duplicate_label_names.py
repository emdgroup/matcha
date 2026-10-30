"""Regression test: duplicate task_label values must not crash validation_step.

When multiple endpoints share the same task_label (e.g. several auxiliary
'dummy' endpoints), ``validation_step`` previously logged the same metric
name (e.g. ``val_r2_dummy``) with different torchmetrics objects, causing
PyTorch Lightning to raise:

    MisconfigurationException: You called `self.log(val_r2_dummy, ...)` twice
    in `validation_step` with different arguments.

The fix must make metric tags unique when label names are duplicated.
"""

import numpy as np

from matcha.sklearn.graph import GINRegressor


GRAPH_PE_OFF = dict(
    rwse_k=0,
    laplacian_k=0,
    elstatic_k=0,
    distmat_k=0,
    rrwp_k=0,
    num_virtual_nodes=0,
)

BASE_TRAIN = dict(
    num_epochs=1,
    batch_size=32,
    accelerator="cpu",
    devices=1,
    stochastic_weight_averaging=False,
)


def test_fit_does_not_crash_with_duplicate_task_labels(mol_list):
    """Training must complete when multiple endpoints share the same task_label.

    Reproduces the bug where PretrainedEncoderWrapper (and any model using
    ModelMixin.validation_step) crashes during validation when label_names
    contains duplicates (e.g. seven 'dummy' auxiliary endpoints).
    """
    # 3 endpoints: one named 'primary', two named 'dummy'
    label_encoder_params = {
        0: {"task_label": "primary"},
        1: {"task_label": "dummy"},
        2: {"task_label": "dummy"},  # duplicate — triggers the bug
    }
    y = np.column_stack(
        [
            np.random.default_rng(0).normal(size=len(mol_list)),
            np.random.default_rng(1).normal(size=len(mol_list)),
            np.random.default_rng(2).normal(size=len(mol_list)),
        ]
    )

    model = GINRegressor(
        enc_num_layers=1,
        enc_atom_hidden_dim=8,
        pred_hidden_dims=[8],
        num_endpoints=3,
        label_encoder_params=label_encoder_params,
        **GRAPH_PE_OFF,
        **BASE_TRAIN,
        early_stopping=True,  # required to trigger validation_step
    )

    # Must not raise MisconfigurationException about logging the same metric twice
    model.fit(mol_list, y)
