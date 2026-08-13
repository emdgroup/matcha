"""Tests for the ``prepare_dataset`` CLI command and its schema.

Stage 1 covers the behavior-preserving rename: the ``COMMANDS`` registry
exposes ``prepare_dataset`` (and no longer ``prepare_sparse_dataset``),
and :class:`PrepareDatasets` accepts a ``sparse`` toggle that defaults
to ``True`` so existing configs keep validating.
"""

from matcha.cli import COMMANDS
from matcha.utils.schemas.cli import PrepareDatasets


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
