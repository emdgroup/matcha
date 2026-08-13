"""Tests for CLI pydantic schemas.

Focuses on the ``EncoderPretrainDataset`` task_type / coords contract that
gates the ``pretrain_encoder`` CLI. The rules under test:

- ``task_type`` must be one of ``"mlm"``, ``"graph"``, ``"graph3d"``.
- ``graph3d`` requires all four y_* label paths *and* both coords paths.
- ``train_coords`` / ``val_coords`` are only valid when
  ``task_type == "graph3d"``.
"""

import pytest
from pydantic import ValidationError

from matcha.utils.schemas.cli import EncoderPretrainDataset


class TestEncoderPretrainDatasetHappyPaths:
    """Every valid config shape must round-trip through validation."""

    def test_mlm_minimal(self):
        ds = EncoderPretrainDataset(
            task_type="mlm",
            train_smiles="train.parquet",
            val_smiles="val.parquet",
        )
        assert ds.task_type == "mlm"
        assert ds.train_coords is None and ds.val_coords is None

    def test_graph_with_all_y_fields(self):
        ds = EncoderPretrainDataset(
            task_type="graph",
            train_smiles="train.parquet",
            val_smiles="val.parquet",
            train_y_graph="train_yg.npz",
            val_y_graph="val_yg.npz",
            train_y_node="train_yn.npz",
            val_y_node="val_yn.npz",
        )
        assert ds.task_type == "graph"

    def test_graph_without_y_fields_still_ok(self):
        # Stage 1 does not tighten the 2D contract — existing configs that
        # rely on runtime label loading remain valid at schema time.
        ds = EncoderPretrainDataset(
            task_type="graph",
            train_smiles="train.parquet",
            val_smiles="val.parquet",
        )
        assert ds.task_type == "graph"

    def test_graph3d_full(self):
        ds = EncoderPretrainDataset(
            task_type="graph3d",
            train_smiles="train.parquet",
            val_smiles="val.parquet",
            train_y_graph="train_yg.npz",
            val_y_graph="val_yg.npz",
            train_y_node="train_yn.npz",
            val_y_node="val_yn.npz",
            train_coords="train_coords.npz",
            val_coords="val_coords.npz",
        )
        assert ds.task_type == "graph3d"
        assert ds.train_coords == "train_coords.npz"
        assert ds.val_coords == "val_coords.npz"


class TestEncoderPretrainDatasetRejections:
    """The schema is the only checkpoint that runs before the CLI opens the
    npz files — malformed configs must be rejected here, not later inside
    featurize().
    """

    def test_unknown_task_type(self):
        with pytest.raises(ValidationError):
            EncoderPretrainDataset(
                task_type="graph4d",
                train_smiles="train.parquet",
                val_smiles="val.parquet",
            )

    @pytest.mark.parametrize(
        "missing_field",
        [
            "train_y_graph",
            "val_y_graph",
            "train_y_node",
            "val_y_node",
            "train_coords",
            "val_coords",
        ],
    )
    def test_graph3d_missing_required_path(self, missing_field):
        kwargs = dict(
            task_type="graph3d",
            train_smiles="train.parquet",
            val_smiles="val.parquet",
            train_y_graph="train_yg.npz",
            val_y_graph="val_yg.npz",
            train_y_node="train_yn.npz",
            val_y_node="val_yn.npz",
            train_coords="train_coords.npz",
            val_coords="val_coords.npz",
        )
        kwargs.pop(missing_field)
        with pytest.raises(ValidationError, match=missing_field):
            EncoderPretrainDataset(**kwargs)

    def test_graph3d_missing_all_labels_and_coords(self):
        with pytest.raises(ValidationError, match="graph3d"):
            EncoderPretrainDataset(
                task_type="graph3d",
                train_smiles="train.parquet",
                val_smiles="val.parquet",
            )

    def test_coords_with_mlm_rejected(self):
        with pytest.raises(ValidationError, match="graph3d"):
            EncoderPretrainDataset(
                task_type="mlm",
                train_smiles="train.parquet",
                val_smiles="val.parquet",
                train_coords="train_coords.npz",
                val_coords="val_coords.npz",
            )

    def test_coords_with_graph_rejected(self):
        with pytest.raises(ValidationError, match="graph3d"):
            EncoderPretrainDataset(
                task_type="graph",
                train_smiles="train.parquet",
                val_smiles="val.parquet",
                train_y_graph="train_yg.npz",
                val_y_graph="val_yg.npz",
                train_y_node="train_yn.npz",
                val_y_node="val_yn.npz",
                train_coords="train_coords.npz",
                val_coords="val_coords.npz",
            )

    def test_single_coord_side_with_mlm_rejected(self):
        # Even a single coord path must be rejected when task_type != graph3d.
        with pytest.raises(ValidationError, match="graph3d"):
            EncoderPretrainDataset(
                task_type="mlm",
                train_smiles="train.parquet",
                val_smiles="val.parquet",
                train_coords="train_coords.npz",
            )
