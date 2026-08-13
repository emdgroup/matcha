"""Tests for the ``prepare_dataset`` CLI command and its schema.

Stage 1 covers the behavior-preserving rename: the ``COMMANDS`` registry
exposes ``prepare_dataset`` (and no longer ``prepare_sparse_dataset``),
and :class:`PrepareDatasets` accepts a ``sparse`` toggle that defaults
to ``True`` so existing configs keep validating.

Stage 2 covers the dense-mode preparation helpers as pure functions,
before they are wired into ``main()`` in stage 3.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from matcha.cli import COMMANDS
from matcha.cli.prepare_dataset import (
    apply_dense_scaling,
    compute_dense_scaling_stats,
    create_validation_set_dense,
    merge_datasets_streaming_dense,
)
from matcha.utils.schemas.cli import PrepareDatasets


@pytest.fixture()
def stub_logger() -> logging.Logger:
    """Silent logger for helpers that log progress info."""
    logger = logging.getLogger("test_prepare_dataset_stage2")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


class TestCommandRegistry:
    """The COMMANDS mapping exposes prepare_dataset and drops the old name."""

    def test_prepare_dataset_registered(self):
        assert COMMANDS["prepare_dataset"] == "matcha.cli.prepare_dataset"

    def test_prepare_sparse_dataset_removed(self):
        assert "prepare_sparse_dataset" not in COMMANDS


class TestPrepareDatasetsSchema:
    """PrepareDatasets accepts an optional ``sparse`` field, default True."""

    def test_defaults_to_sparse_true(self):
        model = PrepareDatasets.model_validate(
            {"files": ["a.parquet"], "task_type": ["regression"]}
        )
        assert model.sparse is True

    def test_accepts_sparse_false(self):
        model = PrepareDatasets.model_validate(
            {
                "files": ["a.parquet"],
                "task_type": ["regression"],
                "sparse": False,
            }
        )
        assert model.sparse is False


class TestDenseScalingHelpers:
    """Pure-function scaling helpers for dense mode ignore NaN entries."""

    def test_compute_dense_scaling_stats_ignores_nans(self, stub_logger):
        matrix = np.array(
            [
                [1.0, np.nan, 0.0],
                [2.0, 4.0, np.nan],
                [3.0, np.nan, 1.0],
                [np.nan, 6.0, 1.0],
            ],
            dtype=np.float32,
        )
        stats = compute_dense_scaling_stats(matrix, [0, 1, 2], stub_logger)

        for task_idx in (0, 1, 2):
            col = matrix[:, task_idx]
            expected_mean = float(np.nanmean(col))
            expected_std = float(np.nanstd(col)) or 1.0
            assert stats[task_idx]["mean"] == pytest.approx(expected_mean)
            assert stats[task_idx]["std"] == pytest.approx(expected_std)

    def test_compute_dense_scaling_stats_handles_all_nan_column(self, stub_logger):
        matrix = np.array([[np.nan, 1.0], [np.nan, 2.0]], dtype=np.float32)
        stats = compute_dense_scaling_stats(matrix, [0, 1], stub_logger)
        assert stats[0] == {"mean": 0.0, "std": 1.0}
        assert stats[1]["std"] > 0.0

    def test_compute_dense_scaling_stats_guards_zero_std(self, stub_logger):
        matrix = np.array([[5.0], [5.0], [np.nan]], dtype=np.float32)
        stats = compute_dense_scaling_stats(matrix, [0], stub_logger)
        assert stats[0]["mean"] == pytest.approx(5.0)
        assert stats[0]["std"] == 1.0

    def test_apply_dense_scaling_preserves_nans(self, stub_logger):
        matrix = np.array(
            [[10.0, np.nan], [20.0, 4.0], [np.nan, 8.0]], dtype=np.float32
        )
        stats = {
            0: {"mean": 10.0, "std": 5.0},
            1: {"mean": 4.0, "std": 2.0},
        }
        scaled = apply_dense_scaling(matrix, stats, stub_logger)

        # NaNs remain NaN in the same positions.
        assert np.isnan(scaled[0, 1])
        assert np.isnan(scaled[2, 0])

        # Non-NaN entries reflect (x - mean) / std.
        assert scaled[0, 0] == pytest.approx(0.0)
        assert scaled[1, 0] == pytest.approx(2.0)
        assert scaled[1, 1] == pytest.approx(0.0)
        assert scaled[2, 1] == pytest.approx(2.0)

    def test_apply_dense_scaling_does_not_mutate_input(self, stub_logger):
        matrix = np.array([[10.0, 4.0]], dtype=np.float32)
        original = matrix.copy()
        stats = {0: {"mean": 10.0, "std": 5.0}}
        apply_dense_scaling(matrix, stats, stub_logger)
        assert np.array_equal(matrix, original)


class TestMergeDatasetsStreamingDense:
    """The dense builder mirrors the sparse builder's discovery/fill logic."""

    def _write_csv(self, path, df):
        df.to_csv(path, index=False)

    def test_merge_datasets_streaming_dense_shape_and_missing(
        self, tmp_path, stub_logger
    ):
        # File A has a regression task with a NaN entry.
        # File B has a regression task on a subset of compounds.
        file_a = tmp_path / "file_a.csv"
        file_b = tmp_path / "file_b.csv"
        self._write_csv(
            file_a,
            pd.DataFrame(
                {
                    "smiles": ["CCO", "CCC", "CCCC"],
                    "reg_a": [1.0, np.nan, 3.0],
                }
            ),
        )
        self._write_csv(
            file_b,
            pd.DataFrame(
                {
                    "smiles": ["CCO", "CCCCC"],
                    "reg_b": [10.0, 20.0],
                }
            ),
        )

        (
            mol_df,
            dense_matrix,
            task_cols,
            column_to_task_type,
            task_to_file,
            file_to_tasks,
        ) = merge_datasets_streaming_dense(
            files=[str(file_a), str(file_b)],
            merge_col="smiles",
            task_types=["regression", "regression"],
            tag_to_add="v1",
            logger=stub_logger,
        )

        # 4 unique compounds across both files, 2 task columns.
        assert dense_matrix.shape == (4, 2)
        assert dense_matrix.dtype == np.float32
        assert task_cols == ["reg_a_v1", "reg_b_v1"]
        assert column_to_task_type == {
            "reg_a_v1": "regression",
            "reg_b_v1": "regression",
        }
        # Every column must have at least one NaN gap since neither file
        # covers all compounds and file A has an explicit NaN.
        assert np.isnan(dense_matrix).any(axis=0).all()

        # File-to-task and task-to-file wiring stays intact.
        assert task_to_file["reg_a_v1"] == str(file_a)
        assert task_to_file["reg_b_v1"] == str(file_b)
        assert file_to_tasks[str(file_a)] == [0]
        assert file_to_tasks[str(file_b)] == [1]

        # Values land at the right (compound, task) coordinates.
        smiles_to_row = {s: i for i, s in enumerate(mol_df["smiles"].tolist())}
        assert dense_matrix[smiles_to_row["CCO"], 0] == pytest.approx(1.0)
        assert np.isnan(dense_matrix[smiles_to_row["CCC"], 0])
        assert dense_matrix[smiles_to_row["CCCC"], 0] == pytest.approx(3.0)
        assert dense_matrix[smiles_to_row["CCO"], 1] == pytest.approx(10.0)
        assert dense_matrix[smiles_to_row["CCCCC"], 1] == pytest.approx(20.0)

    def test_merge_datasets_streaming_dense_classification_no_remap(
        self, tmp_path, stub_logger
    ):
        file_a = tmp_path / "cls.csv"
        self._write_csv(
            file_a,
            pd.DataFrame(
                {
                    "smiles": ["CCO", "CCC", "CCCC", "CCCCC"],
                    "cls_a": [0, 1, np.nan, 0],
                }
            ),
        )

        _, dense_matrix, _, _, _, _ = merge_datasets_streaming_dense(
            files=[str(file_a)],
            merge_col="smiles",
            task_types=["classification"],
            tag_to_add="v1",
            logger=stub_logger,
        )

        col = dense_matrix[:, 0]
        finite = col[~np.isnan(col)]
        # Classification values pass through unchanged — only 0/1 remain.
        assert set(np.unique(finite).tolist()) <= {0.0, 1.0}
        # Sanity: no `-1` sentinel from the sparse convention leaked in.
        assert not np.any(finite == -1)


class TestCreateValidationSetDense:
    """Random per-task sampling identifies compounds via ~np.isnan masks."""

    def test_create_validation_set_dense_split_sizes(self, stub_logger):
        # 10 compounds; task 0 covers all, task 1 covers only the first 5.
        n_compounds = 10
        dense_matrix = np.full((n_compounds, 2), np.nan, dtype=np.float32)
        dense_matrix[:, 0] = np.arange(n_compounds, dtype=np.float32)
        dense_matrix[:5, 1] = np.arange(5, dtype=np.float32)

        mol_df = pd.DataFrame({"smiles": [f"C{i}" for i in range(n_compounds)]})

        train_mol_df, val_mol_df, train_dense, val_dense = create_validation_set_dense(
            mol_df=mol_df,
            dense_matrix=dense_matrix,
            task_cols=["task_0", "task_1"],
            min_compounds=2,
            sampling_rate=0.2,
            seed=42,
            logger=stub_logger,
        )

        # Partition is exact: train + val = full set with no overlap.
        assert len(train_mol_df) + len(val_mol_df) == n_compounds
        assert train_dense.shape[0] == len(train_mol_df)
        assert val_dense.shape[0] == len(val_mol_df)
        assert train_dense.shape[1] == 2

        # Validation set has at least ``min_compounds`` (2) per task's sampling
        # request; task 0 samples 2 (max(2, int(10*0.2))=2) and task 1 samples 2
        # (max(2, int(5*0.2))=2). Overlap between the two per-task sample sets
        # is possible, so we only assert an upper and lower bound.
        assert 2 <= len(val_mol_df) <= 4

        # Determinism: same seed → same split.
        _, val_mol_df_again, _, _ = create_validation_set_dense(
            mol_df=mol_df,
            dense_matrix=dense_matrix,
            task_cols=["task_0", "task_1"],
            min_compounds=2,
            sampling_rate=0.2,
            seed=42,
            logger=stub_logger,
        )
        assert val_mol_df["smiles"].tolist() == val_mol_df_again["smiles"].tolist()
