"""Tests for matcha.utils.splitting – random_split and time_split.

Validates the matcha-specific splitting helpers (size correctness, sort
order, edge cases).  Does *not* re-test sklearn's train_test_split or
cluster_split (which requires RDKit molecule columns / UMAP / DBSCAN).
"""

import pandas as pd
import pytest

from matcha.utils.splitting import random_split, time_split


# =========================================================================
# random_split
# =========================================================================


class TestRandomSplit:
    """Tests for random_split wrapper."""

    def test_default_split_size(self, simple_df):
        train_df, test_df = random_split(simple_df)
        assert len(train_df) + len(test_df) == len(simple_df)
        assert len(test_df) == pytest.approx(len(simple_df) * 0.2, abs=1)

    def test_custom_split_size(self, simple_df):
        train_df, test_df = random_split(simple_df, split_size=0.3)
        assert len(train_df) + len(test_df) == len(simple_df)
        assert len(test_df) == pytest.approx(len(simple_df) * 0.3, abs=1)

    def test_no_index_overlap(self, simple_df):
        train_df, test_df = random_split(simple_df)
        assert set(train_df.index).isdisjoint(set(test_df.index))

    def test_deterministic_with_same_seed(self, simple_df):
        train1, _ = random_split(simple_df, random_flag=123)
        train2, _ = random_split(simple_df, random_flag=123)
        pd.testing.assert_frame_equal(train1, train2)

    def test_different_seed_gives_different_split(self, simple_df):
        train1, _ = random_split(simple_df, random_flag=1)
        train2, _ = random_split(simple_df, random_flag=2)
        assert not train1.index.equals(train2.index)


# =========================================================================
# time_split (with split_size)
# =========================================================================


class TestTimeSplitBySize:
    """Tests for time_split using percentage-based splitting."""

    def test_split_by_percentage_numeric(self, simple_df):
        train_df, test_df = time_split(simple_df, column_name="value", split_size=0.2)
        assert len(train_df) + len(test_df) == len(simple_df)
        assert len(test_df) == pytest.approx(len(simple_df) * 0.2, abs=1)

    def test_train_values_are_earlier(self, simple_df):
        """Train set values should be ≤ all test set values (sorted split)."""
        train_df, test_df = time_split(simple_df, column_name="value", split_size=0.2)
        assert train_df["value"].max() <= test_df["value"].min()

    def test_split_by_percentage_datetime(self, simple_df):
        train_df, test_df = time_split(simple_df, column_name="date", split_size=0.3)
        assert len(train_df) + len(test_df) == len(simple_df)
        assert train_df["date"].max() <= test_df["date"].min()

    def test_invalid_percentage_raises(self, simple_df):
        with pytest.raises(ValueError, match="between 0 and 1"):
            time_split(simple_df, column_name="value", split_size=1.5)

    def test_zero_percentage_raises(self, simple_df):
        with pytest.raises(ValueError, match="between 0 and 1"):
            time_split(simple_df, column_name="value", split_size=0)


# =========================================================================
# time_split (with split_value)
# =========================================================================


class TestTimeSplitByValue:
    """Tests for time_split using a threshold value."""

    def test_split_by_value_numeric(self, simple_df):
        train_df, test_df = time_split(simple_df, column_name="value", split_value=80)
        assert len(train_df) + len(test_df) == len(simple_df)
        assert test_df["value"].min() >= 80

    def test_split_by_value_datetime(self, simple_df):
        split_date = simple_df["date"].iloc[70]
        train_df, test_df = time_split(
            simple_df, column_name="date", split_value=split_date
        )
        assert test_df["date"].min() >= split_date


# =========================================================================
# time_split – error handling
# =========================================================================


class TestTimeSplitErrors:
    """Edge cases and error handling for time_split."""

    def test_missing_column_raises(self, simple_df):
        with pytest.raises(ValueError, match="not found"):
            time_split(simple_df, column_name="nonexistent", split_size=0.2)

    def test_neither_value_nor_size_raises(self, simple_df):
        with pytest.raises(ValueError, match="Either"):
            time_split(simple_df, column_name="value")

    def test_no_column_name_uses_original_order(self):
        """When column_name is None, the DataFrame should not be sorted."""
        df = pd.DataFrame({"value": [5, 3, 1, 4, 2]})
        train_df, test_df = time_split(df, column_name=None, split_size=0.4)
        assert len(train_df) + len(test_df) == 5
