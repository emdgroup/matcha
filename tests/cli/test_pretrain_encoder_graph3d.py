"""End-to-end integration tests for the ``pretrain_encoder`` graph3d branch.

Drives :func:`matcha.cli.pretrain_encoder.main` programmatically on a tiny
tmp_path fixture — a handful of SMILES with matched packed-npz node
labels, graph labels, and coords — and asserts the full artifact directory
layout is produced.

These tests are the acceptance-criterion regression for the ``graph3d``
CLI path added in Stage 3 of issue #29: an E3GNN pretraining job runs
end-to-end from a YAML config and yields the standard finetuner-consumable
artifact directory (``model.ckpt``, ``encoder.ckpt``, ``manifest.yaml``,
``datamodule_state.pkl``).
"""

import os
import pickle
import yaml

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem


pytest.importorskip("torch_geometric")
pytest.importorskip("torch")


# A tiny set of hand-picked, RDKit-parseable SMILES.  Kept small to keep
# the test fast; each contributes canonical-atom counts to the packed
# ``y_node`` / ``coords`` arrays below.
_SMILES = [
    "CCO",  # ethanol           (3 heavy atoms)
    "CCC",  # propane           (3)
    "CCN",  # ethylamine        (3)
    "CCCC",  # butane            (4)
    "CCCCC",  # pentane           (5)
    "c1ccccc1",  # benzene           (6)
    "CCOC",  # methoxyethane     (4)
    "CC(=O)C",  # acetone           (4)
]

_NUM_NODE_TARGETS = 2
_NUM_GRAPH_TARGETS = 3


def _canonical_atom_counts(smiles: list[str]) -> list[int]:
    """Number of heavy atoms in the canonical form of each SMILES."""
    counts = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        canonical = Chem.MolFromSmiles(Chem.MolToSmiles(mol, canonical=True))
        counts.append(canonical.GetNumAtoms())
    return counts


def _pack_flat_offsets(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate variable-length arrays into (flat, offsets) form."""
    flat = np.concatenate(arrays, axis=0)
    offsets = np.cumsum([0] + [a.shape[0] for a in arrays]).astype(np.int64)
    return flat, offsets


def _write_split(
    tmp_dir: str,
    split: str,
    smiles: list[str],
    rng: np.random.Generator,
) -> dict[str, str]:
    """Write parquet + three npz files for one split; return their paths."""
    atom_counts = _canonical_atom_counts(smiles)

    # Parquet with SMILES column
    smiles_path = os.path.join(tmp_dir, f"{split}_smiles.parquet")
    pd.DataFrame({"SMILES": smiles}).to_parquet(smiles_path)

    # Graph-level labels: (N, G)
    y_graph_path = os.path.join(tmp_dir, f"{split}_y_graph.npz")
    y_graph = rng.standard_normal((len(smiles), _NUM_GRAPH_TARGETS)).astype(np.float32)
    np.savez_compressed(y_graph_path, descriptors=y_graph)

    # Node-level labels: list of (A_i, T) → flat + offsets
    node_arrays = [
        rng.standard_normal((a, _NUM_NODE_TARGETS)).astype(np.float32)
        for a in atom_counts
    ]
    node_flat, node_offsets = _pack_flat_offsets(node_arrays)
    y_node_path = os.path.join(tmp_dir, f"{split}_y_node.npz")
    np.savez_compressed(y_node_path, flat=node_flat, offsets=node_offsets)

    # Coords: list of (A_i, 3) → flat + offsets
    coord_arrays = [rng.standard_normal((a, 3)).astype(np.float32) for a in atom_counts]
    coord_flat, coord_offsets = _pack_flat_offsets(coord_arrays)
    coords_path = os.path.join(tmp_dir, f"{split}_coords.npz")
    np.savez_compressed(coords_path, flat=coord_flat, offsets=coord_offsets)

    return {
        "smiles": smiles_path,
        "y_graph": y_graph_path,
        "y_node": y_node_path,
        "coords": coords_path,
    }


@pytest.fixture
def graph3d_dataset(tmp_path):
    """Write a tiny train/val fixture and return its file paths."""
    rng = np.random.default_rng(0)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    train = _write_split(str(data_dir), "train", _SMILES, rng)
    val = _write_split(str(data_dir), "val", _SMILES[:4], rng)
    return {"train": train, "val": val, "out": str(tmp_path / "out")}


def _build_cfg(paths: dict, out_dir: str) -> dict:
    """Build a minimal graph3d config dict for :func:`main`."""
    train, val = paths["train"], paths["val"]
    return {
        "dataset": {
            "task_type": "graph3d",
            "train_smiles": train["smiles"],
            "val_smiles": val["smiles"],
            "train_y_graph": train["y_graph"],
            "val_y_graph": val["y_graph"],
            "train_y_node": train["y_node"],
            "val_y_node": val["y_node"],
            "train_coords": train["coords"],
            "val_coords": val["coords"],
        },
        "model": {
            "architecture": "E3GNNPretraining",
            "params": {
                "num_node_targets": _NUM_NODE_TARGETS,
                "num_graph_targets": _NUM_GRAPH_TARGETS,
                "enc_num_layers": 2,
                "enc_atom_hidden_dim": 16,
                "enc_m_dim": 8,
                "enc_fourier_features": 2,
                "enc_norm_feats": False,
                "enc_norm_coors": False,
                "enc_jk": "last",
                "enc_readout": "sum",
                "enc_activation": "relu",
                "enc_dropout": 0.0,
                "node_head_dims": [8],
                "graph_head_dims": [8],
                "pred_activation": "relu",
                "pred_dropout": 0.0,
                "loss_fn": "mse",
                "loss_args": {},
                "optimizer": "adamw",
                "optimizer_args": {"lr": 1.0e-3},
                "scheduler": "warmup_linear_decay",
            },
            "datamodule": {
                "laplacian_k": 0,
                "rwse_k": 0,
                "elstatic_k": 0,
                "distmat_k": 0,
                "rrwp_k": 0,
                "compute_distances": True,
                "num_virtual_nodes": 0,
                "init_virtual_nodes": False,
                "batch_size": 4,
                "num_workers": 0,
                "augment_resonance": False,
                "scale_y_graph": False,
                "scale_y_node": False,
            },
            "training": {
                "num_epochs": 1,
                "early_stopping": False,
                "patience": 1,
                "accelerator": "cpu",
                "devices": 1,
                "seed": 0,
            },
        },
        "pipe": {
            "dataloader_num_workers": 0,
            "fit_datamodule_size": None,
            "gradient_accumulation_steps": 1,
            "gradient_clip_val": 0.0,
        },
        "output": {"serialization": out_dir},
    }


def test_main_produces_graph3d_artifact_directory(graph3d_dataset):
    """``pretrain_encoder`` graph3d branch runs 1 epoch and lays down all artifacts."""
    from matcha.cli.pretrain_encoder import main

    cfg = _build_cfg(graph3d_dataset, graph3d_dataset["out"])
    main(cfg=cfg)

    out = graph3d_dataset["out"]

    # Full lightning checkpoint + encoder-only checkpoint
    assert os.path.isfile(os.path.join(out, "model.ckpt"))
    assert os.path.isfile(os.path.join(out, "encoder.ckpt"))

    # Config directory
    manifest_path = os.path.join(out, "config", "manifest.yaml")
    assert os.path.isfile(manifest_path)
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)
    assert manifest["task_type"] == "graph3d"
    assert manifest["source_class"] == "E3GNNPretraining"
    assert manifest["origin_type"] == "pretraining"

    # Datamodule state pickle carries the 3D pretraining discriminator
    state_path = os.path.join(out, "state", "datamodule_state.pkl")
    assert os.path.isfile(state_path)
    with open(state_path, "rb") as f:
        dm_state = pickle.load(f)
    assert dm_state["ID"] == "graph3d_pretraining"


def test_example_config_validates(tmp_path):
    """The shipped ``pretrain_encoder_graph3d.yaml`` example passes schema validation.

    Guards against drift between the example config the docs point users at
    and the pydantic schema they will hit at CLI startup.
    """
    import matcha.cli as cli_pkg
    from matcha.cli.pretrain_encoder import CLIEncoderPretrainInputModel

    cfg_path = os.path.join(
        os.path.dirname(cli_pkg.__file__),
        "example_configs",
        "pretrain_encoder_graph3d.yaml",
    )
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    CLIEncoderPretrainInputModel.model_validate(raw)
