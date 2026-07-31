"""Tests for matcha.cli.utils.get_splits and bootstrap independence.

Covers the split-generation contract after the ``n_repeat`` removal
(issue #420): every split method produces a fixed number of train/test
splits with aligned list lengths, and bootstrapping is independent of the
split method because it operates only on a split's test arrays.
"""

import numpy as np
import pytest

from matcha.cli.utils import bootstrap_metrics, get_splits
from matcha.utils.metrics import process_regression
from matcha.utils.schemas.cli import Split


# =========================================================================
# cv split — produces exactly n_subset splits (no n_repeat)
# =========================================================================


class TestCVSplit:
    """Cross-validation split count and alignment."""

    @pytest.mark.parametrize("n_subset", [2, 5, 10])
    def test_cv_returns_n_subset_splits(self, molecule_df, n_subset):
        split = Split(method="cv", n_subset=n_subset)
        train_splits, val_splits = get_splits(molecule_df, split)
        assert len(train_splits) == n_subset
        assert len(val_splits) == n_subset

    def test_cv_val_splits_returns_pairs(self, molecule_df):
        split = Split(method="cv", n_subset=3)
        train_splits, val_splits = get_splits(
            molecule_df, split, return_val_splits=True
        )
        assert len(train_splits) == 3
        assert len(val_splits) == 3
        # Each entry is a [df_val, df_test] pair when return_val_splits=True.
        assert all(len(pair) == 2 for pair in val_splits)

    def test_cv_method_params_none_is_accepted(self, molecule_df):
        # CV no longer reads method_params (n_repeat removed); None must work.
        split = Split(method="cv", n_subset=4, method_params=None)
        train_splits, _ = get_splits(molecule_df, split)
        assert len(train_splits) == 4

    def test_cv_ignores_legacy_n_repeat(self, molecule_df):
        # Legacy configs may still set n_repeat; it must be silently ignored
        # and produce exactly n_subset splits (not n_subset * n_repeat).
        split = Split(method="cv", n_subset=5, method_params={"n_repeat": 3})
        train_splits, _ = get_splits(molecule_df, split)
        assert len(train_splits) == 5


# =========================================================================
# All split methods produce aligned train/val list lengths
# =========================================================================


class TestSplitAlignment:
    """Smoke tests: every method yields equal-length train/val lists."""

    def test_cv_aligned(self, molecule_df):
        split = Split(method="cv", n_subset=3)
        train_splits, val_splits = get_splits(molecule_df, split)
        assert len(train_splits) == len(val_splits) == 3

    def test_time_aligned(self, molecule_df):
        df = molecule_df.copy()
        # Provide a monotonically increasing time key to sort on.
        df["time_key"] = np.arange(len(df))
        split = Split(
            method="time",
            n_subset=3,
            method_params={"key": "time_key", "split_size": 0.2},
        )
        train_splits, val_splits = get_splits(df, split)
        assert len(train_splits) == len(val_splits) == 3

    def test_file_aligned(self, molecule_df, dataset_cfg):
        # Two external test files (reusing the same CSV) -> two splits.
        paths = [dataset_cfg.path, dataset_cfg.path]
        split = Split(method="file", n_subset=2, method_params={"path": paths})
        train_splits, val_splits = get_splits(
            molecule_df, split, return_val_splits=False, dataset_cfg=dataset_cfg
        )
        assert len(train_splits) == len(val_splits) == 2

    def test_cluster_aligned(self, molecule_df, monkeypatch):
        # The real cluster_split runs UMAP + DBSCAN (slow, environment
        # dependent — test_splitting.py also avoids it). Here we exercise
        # the cluster *branch* of get_splits, not UMAP, so cluster_split is
        # replaced with a deterministic 80/20 head/tail splitter.
        def fake_cluster_split(df, feature_set, metric, split_size, n_jobs):
            cut = int(len(df) * (1 - split_size))
            return df.iloc[:cut], df.iloc[cut:]

        monkeypatch.setattr("matcha.cli.utils.cluster_split", fake_cluster_split)
        split = Split(
            method="cluster",
            n_subset=2,
            method_params={
                "features": ["ecfp", "ecfp"],
                "metric": ["jaccard", "jaccard"],
                "split_size": 0.2,
                "n_jobs": 1,
            },
        )
        train_splits, val_splits = get_splits(molecule_df, split)
        assert len(train_splits) == len(val_splits) == 2


# =========================================================================
# bootstrap_metrics is split-method-independent
# =========================================================================


class TestBootstrapIndependence:
    """bootstrap_metrics operates on arrays only, never on split metadata."""

    def test_returns_n_bootstrap_dicts(self):
        rng = np.random.default_rng(0)
        y_true = rng.normal(5.0, 1.0, size=100)
        y_pred = y_true + rng.normal(0, 0.3, size=100)
        scores = bootstrap_metrics(
            process_regression, 50, 0.8, y_true, y_pred, log10=False
        )
        assert len(scores) == 50
        assert all(isinstance(s, dict) for s in scores)

    def test_deterministic_regardless_of_data_origin(self):
        # Same test arrays -> identical bootstrap results, irrespective of
        # which split method produced them. bootstrap_metrics seeds its own
        # RNG, so the output depends only on the arrays passed in.
        rng = np.random.default_rng(1)
        y_true = rng.normal(5.0, 1.0, size=120)
        y_pred = y_true + rng.normal(0, 0.3, size=120)

        first = bootstrap_metrics(process_regression, 10, 0.5, y_true, y_pred)
        second = bootstrap_metrics(process_regression, 10, 0.5, y_true, y_pred)
        assert first == second

    def test_bootstrap_independent_of_split_method(self, molecule_df):
        # Take the test set from two different split methods and confirm
        # bootstrap_metrics behaves identically on each (same contract:
        # n_bootstrap dicts), proving it is split-agnostic. A trivial metric
        # is used so the assertion targets bootstrap_metrics, not the
        # numerical quirks of any particular metric function.
        def mean_metric(labels, preds):
            return {"mean_label": float(np.mean(labels))}

        cv_split = Split(method="cv", n_subset=2)
        _, cv_val = get_splits(molecule_df, cv_split)

        df = molecule_df.copy()
        df["time_key"] = np.arange(len(df))
        time_split = Split(
            method="time",
            n_subset=2,
            method_params={"key": "time_key", "split_size": 0.2},
        )
        _, time_val = get_splits(df, time_split)

        for test_df in (cv_val[0], time_val[0]):
            y = test_df["Regression"].to_numpy()
            preds = y + 0.1
            scores = bootstrap_metrics(mean_metric, 5, 0.8, y, preds)
            assert len(scores) == 5
            assert all(isinstance(s, dict) for s in scores)
