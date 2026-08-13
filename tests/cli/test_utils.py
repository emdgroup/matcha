"""Tests for matcha.cli.utils.get_splits and bootstrap independence.

Covers the split-generation contract after the ``n_repeat`` removal
(issue #420): every split method produces a fixed number of train/test
splits with aligned list lengths, and bootstrapping is independent of the
split method because it operates only on a split's test arrays.
"""

import numpy as np
import pytest

from matcha.cli.utils import (
    _load_coords_npz,
    _load_npz_list,
    bootstrap_metrics,
    get_splits,
)
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


# =========================================================================
# _load_npz_list — flat + offsets round-trip
# =========================================================================


class TestLoadNpzList:
    """Legacy variable-length packing loader (used for node-level labels)."""

    def test_round_trip_1d(self, tmp_path):
        arrs = [
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0, 5.0]),
            np.array([6.0]),
        ]
        flat = np.concatenate(arrs)
        offsets = np.cumsum([0] + [len(a) for a in arrs])
        path = tmp_path / "y_node.npz"
        np.savez_compressed(path, flat=flat, offsets=offsets)

        loaded = _load_npz_list(str(path))
        assert len(loaded) == len(arrs)
        for got, want in zip(loaded, arrs):
            np.testing.assert_array_equal(got, want)


# =========================================================================
# _load_coords_npz — happy path + four startup guards
# =========================================================================


def _write_coords_npz(path, flat, offsets):
    np.savez_compressed(path, flat=flat, offsets=offsets)


class TestLoadCoordsNpz:
    """Guard rails on the coords npz on-disk layout.

    Uses the same ``flat + offsets`` packing as node-level labels, but the
    loader enforces stricter shape/monotonicity invariants so we fail fast
    at CLI startup instead of deep inside featurize().
    """

    def _make_valid(self, tmp_path, atom_counts=(3, 4, 2)):
        rng = np.random.default_rng(0)
        flat = rng.standard_normal((sum(atom_counts), 3)).astype(np.float64)
        offsets = np.cumsum([0] + list(atom_counts)).astype(np.int64)
        path = tmp_path / "coords.npz"
        _write_coords_npz(path, flat, offsets)
        return path, flat, offsets, atom_counts

    def test_happy_path_round_trip(self, tmp_path):
        path, flat, offsets, atom_counts = self._make_valid(tmp_path)
        loaded = _load_coords_npz(str(path))
        assert len(loaded) == len(atom_counts)
        for i, (arr, count) in enumerate(zip(loaded, atom_counts)):
            assert arr.shape == (count, 3)
            assert arr.dtype == np.float32
            np.testing.assert_allclose(
                arr, flat[offsets[i] : offsets[i + 1]].astype(np.float32)
            )

    def test_returns_float32(self, tmp_path):
        path, _, _, _ = self._make_valid(tmp_path)
        loaded = _load_coords_npz(str(path))
        assert all(a.dtype == np.float32 for a in loaded)

    def test_rejects_flat_1d(self, tmp_path):
        path = tmp_path / "bad_ndim.npz"
        _write_coords_npz(
            path,
            flat=np.zeros(9, dtype=np.float32),
            offsets=np.array([0, 3, 6, 9], dtype=np.int64),
        )
        with pytest.raises(AssertionError, match="flat.ndim"):
            _load_coords_npz(str(path))

    def test_rejects_wrong_last_dim(self, tmp_path):
        path = tmp_path / "bad_shape.npz"
        _write_coords_npz(
            path,
            flat=np.zeros((9, 2), dtype=np.float32),
            offsets=np.array([0, 3, 6, 9], dtype=np.int64),
        )
        with pytest.raises(AssertionError, match=r"flat\.shape\[1\]"):
            _load_coords_npz(str(path))

    def test_rejects_non_monotonic_offsets(self, tmp_path):
        path = tmp_path / "non_monotonic.npz"
        _write_coords_npz(
            path,
            flat=np.zeros((9, 3), dtype=np.float32),
            offsets=np.array([0, 6, 3, 9], dtype=np.int64),
        )
        with pytest.raises(AssertionError, match="monotonic"):
            _load_coords_npz(str(path))

    def test_rejects_offsets_last_mismatch(self, tmp_path):
        path = tmp_path / "offset_mismatch.npz"
        _write_coords_npz(
            path,
            flat=np.zeros((9, 3), dtype=np.float32),
            offsets=np.array([0, 3, 6, 8], dtype=np.int64),
        )
        with pytest.raises(AssertionError, match="offsets"):
            _load_coords_npz(str(path))

    def test_rejects_offsets_2d(self, tmp_path):
        path = tmp_path / "offsets_2d.npz"
        _write_coords_npz(
            path,
            flat=np.zeros((9, 3), dtype=np.float32),
            offsets=np.array([[0, 3], [6, 9]], dtype=np.int64),
        )
        with pytest.raises(AssertionError, match="offsets.ndim"):
            _load_coords_npz(str(path))
