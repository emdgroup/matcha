"""Tests for ``pretrain_multitask`` coordinate auto-discovery (issue #29 stage 4).

Three cases via :func:`matcha.cli.pretrain_multitask.main`:

- (a) ``{dataset_dir}/{split}_coords.npz`` present and the base datamodule
  accepts ``coords=`` — the wrapper forwards them into
  ``base.generate_features(..., coords=...)`` at collate time.
- (b) coord files present but the base is 2-D-only — the wrapper logs a
  single warning and drops coords.
- (c) no coord files — behavior is byte-identical to the pre-issue-29 path
  (``dm._raw_train.coords is None``, probe cache untouched).

Full training is stubbed out (``L.Trainer.fit`` and ``_save_artifacts`` are
replaced with no-ops); the DM instance is captured on construction so the
tests can trigger a single collate step manually.
"""

import json
import logging
import os

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

pytest.importorskip("torch_geometric")
pytest.importorskip("torch")

import lightning as L  # noqa: E402

from matcha.datamodules.pretraining.on_the_fly_datamodule import (  # noqa: E402
    OnTheFlyDataModule,
)


# ---------------------------------------------------------------------------
# Fixture builder
# ---------------------------------------------------------------------------

_SMILES = [
    "CCO",
    "CCC",
    "CCN",
    "CCCC",
    "CCCCC",
    "c1ccccc1",
    "CCOC",
    "CC(=O)C",
]
_VAL_SMILES = _SMILES[:4]
_NUM_TASKS = 2


def _pack_coords(counts: list[int], rng: np.random.Generator) -> tuple:
    """Return (flat, offsets) for a per-molecule coord list."""
    arrays = [rng.standard_normal((n, 3)).astype(np.float32) for n in counts]
    flat = np.concatenate(arrays, axis=0)
    offsets = np.cumsum([0] + [a.shape[0] for a in arrays]).astype(np.int64)
    return flat, offsets


def _write_multitask_fixture(dataset_dir: str, *, include_coords: bool) -> None:
    """Create the minimum on-disk layout ``pretrain_multitask`` expects."""
    os.makedirs(dataset_dir, exist_ok=True)
    rng = np.random.default_rng(0)

    pd.DataFrame({"SMILES": _SMILES}).to_parquet(
        os.path.join(dataset_dir, "train_molecules.parquet")
    )
    pd.DataFrame({"SMILES": _VAL_SMILES}).to_parquet(
        os.path.join(dataset_dir, "val_molecules.parquet")
    )

    train_y = rng.integers(-1, 2, size=(len(_SMILES), _NUM_TASKS)).astype(np.float32)
    val_y = rng.integers(-1, 2, size=(len(_VAL_SMILES), _NUM_TASKS)).astype(np.float32)
    sp.save_npz(os.path.join(dataset_dir, "train_tasks.npz"), sp.csr_matrix(train_y))
    sp.save_npz(os.path.join(dataset_dir, "val_tasks.npz"), sp.csr_matrix(val_y))

    task_metadata = {
        "file_to_tasks": {"regression": list(range(_NUM_TASKS))},
        "task_to_file": {str(i): "regression" for i in range(_NUM_TASKS)},
    }
    with open(os.path.join(dataset_dir, "task_metadata.json"), "w") as f:
        json.dump(task_metadata, f)

    if include_coords:
        # Atom counts must match RDKit's canonical order — but the OnTheFly
        # wrapper does no reordering itself; the base does. In tests (a) and
        # (b) we short-circuit the base's featurizer or use a 2-D base that
        # never touches coords, so any per-molecule row count works for the
        # collate-level assertions.
        train_counts = [len(s) for s in _SMILES]  # approximate; not read by base
        val_counts = [len(s) for s in _VAL_SMILES]
        flat, offsets = _pack_coords(train_counts, rng)
        np.savez_compressed(
            os.path.join(dataset_dir, "train_coords.npz"), flat=flat, offsets=offsets
        )
        flat, offsets = _pack_coords(val_counts, rng)
        np.savez_compressed(
            os.path.join(dataset_dir, "val_coords.npz"), flat=flat, offsets=offsets
        )


def _build_cfg(dataset_dir: str, out_dir: str) -> dict:
    return {
        "dataset": {"dataset_dir": dataset_dir},
        "model": {
            "architecture": "GatedGCNRegressor",
            "params": {
                "enc_num_layers": 1,
                "enc_atom_hidden_dim": 16,
                "enc_norm": None,
                "enc_readout": "sum",
                "enc_activation": "relu",
                "enc_dropout": 0.0,
                "pred_hidden_dims": [16],
                "pred_activation": "relu",
                "pred_dropout": 0.0,
                "optimizer": "adamw",
                "optimizer_args": {"lr": 1.0e-3},
                "scheduler": "warmup_linear_decay",
            },
            "datamodule": {
                "batch_size": 4,
                "rwse_k": 0,
                "laplacian_k": 0,
                "elstatic_k": 0,
                "distmat_k": 0,
                "rrwp_k": 0,
                "num_virtual_nodes": 0,
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
        "loss": [
            {
                "dataset": "regression",
                "loss_fn": "mse",
                "loss_args": {},
                "init_w": 1.0,
                "final_w": 1.0,
                "T": 1.0,
                "warmup": 0.0,
            }
        ],
        "pipe": {
            "dataloader_num_workers": 0,
            "fit_datamodule_size": None,
            "gradient_accumulation_steps": 1,
            "gradient_clip_val": 0.0,
        },
        "output": {"serialization": out_dir},
    }


@pytest.fixture
def stubbed_main(monkeypatch):
    """Patch out training + artifact save, and capture the OnTheFlyDataModule."""
    captured: dict = {}

    original_init = OnTheFlyDataModule.__init__

    def spy_init(self, base, num_workers=0, **kwargs):
        original_init(self, base, num_workers=num_workers, **kwargs)
        captured["dm"] = self

    monkeypatch.setattr(OnTheFlyDataModule, "__init__", spy_init)
    monkeypatch.setattr(L.Trainer, "fit", lambda self, *a, **k: None)
    monkeypatch.setattr(
        "matcha.cli.pretrain_multitask._save_artifacts", lambda *a, **k: None
    )
    return captured


# ---------------------------------------------------------------------------
# (a) coord files present + base accepts coords → base receives coords=
# ---------------------------------------------------------------------------


def test_coords_forwarded_when_base_accepts(tmp_path, stubbed_main):
    """3-D-capable base + coord files → coords land in base.generate_features."""
    from matcha.cli.pretrain_multitask import main

    dataset_dir = tmp_path / "data"
    _write_multitask_fixture(str(dataset_dir), include_coords=True)

    cfg = _build_cfg(str(dataset_dir), str(tmp_path / "out"))
    main(cfg=cfg)

    dm = stubbed_main["dm"]

    # CLI wired coords into set_data.
    assert dm._raw_train.coords is not None
    assert dm._raw_val.coords is not None
    assert len(dm._raw_train.coords) == len(_SMILES)
    assert len(dm._raw_val.coords) == len(_VAL_SMILES)

    # Swap the base's generate_features for a coords-accepting spy that
    # falls back to the real 2-D featurizer (dropping the coords arg) so the
    # rest of the collate machinery still produces a batch. The signature
    # advertises a ``coords`` kwarg so ``inspect.signature`` sees it.
    base = dm.base
    calls: list[dict] = []
    real_generate = base.generate_features

    def coords_accepting(mol_list, y=None, coords=None, n_jobs=1):
        calls.append(
            {
                "n_mols": len(mol_list),
                "coords_received": coords is not None,
                "coords_len": len(coords) if coords is not None else None,
            }
        )
        return real_generate(mol_list, y, n_jobs=n_jobs)

    base.generate_features = coords_accepting
    dm._base_accepts_coords = None  # invalidate cached probe

    dm.setup(stage="fit")
    _ = next(iter(dm.train_dataloader()))

    assert calls, "collate never reached the base"
    assert calls[0]["coords_received"] is True
    assert calls[0]["coords_len"] == calls[0]["n_mols"]
    assert dm._base_accepts_coords is True


# ---------------------------------------------------------------------------
# (b) coord files present + 2-D-only base → single WARNING, coords dropped
# ---------------------------------------------------------------------------


def test_coords_dropped_with_warning_on_2d_base(tmp_path, stubbed_main, caplog):
    """2-D base + coord files → single warning and coords never reach the base."""
    from matcha.cli.pretrain_multitask import main

    dataset_dir = tmp_path / "data"
    _write_multitask_fixture(str(dataset_dir), include_coords=True)

    cfg = _build_cfg(str(dataset_dir), str(tmp_path / "out"))
    main(cfg=cfg)

    dm = stubbed_main["dm"]

    # set_data still received coords — the drop decision is made at collate.
    assert dm._raw_train.coords is not None

    caplog.set_level(
        logging.WARNING,
        logger="matcha.datamodules.pretraining.on_the_fly_datamodule",
    )
    dm.setup(stage="fit")
    _ = next(iter(dm.train_dataloader()))

    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "coords" in r.getMessage().lower()
    ]
    assert len(warnings) == 1
    assert dm._base_accepts_coords is False
    assert dm._coords_ignore_warned is True


# ---------------------------------------------------------------------------
# (c) no coord files → probe never runs, no warning, coords are None
# ---------------------------------------------------------------------------


def test_no_coord_files_leaves_probe_dormant(tmp_path, stubbed_main, caplog):
    """Absent coord files → set_data receives None; probe cache untouched."""
    from matcha.cli.pretrain_multitask import main

    dataset_dir = tmp_path / "data"
    _write_multitask_fixture(str(dataset_dir), include_coords=False)

    cfg = _build_cfg(str(dataset_dir), str(tmp_path / "out"))

    caplog.set_level(
        logging.WARNING,
        logger="matcha.datamodules.pretraining.on_the_fly_datamodule",
    )
    main(cfg=cfg)

    dm = stubbed_main["dm"]

    assert dm._raw_train.coords is None
    assert dm._raw_val.coords is None

    dm.setup(stage="fit")
    _ = next(iter(dm.train_dataloader()))

    coord_warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "coords" in r.getMessage().lower()
    ]
    assert coord_warnings == []
    # Probe cache stays untouched — no coords in the batch means the wrapper
    # never inspects ``base.generate_features``.
    assert dm._base_accepts_coords is None
    assert dm._coords_ignore_warned is False
