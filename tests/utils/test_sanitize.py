"""Tests for matcha.utils.sanitize – ensure_1d_array.

Validates custom squeeze / flatten logic that is *not* provided by numpy
out of the box (e.g. intelligent handling of 2-D and higher-dimensional
arrays with a single non-singleton axis, and proper error messages for
ambiguous shapes).
"""

import numpy as np
import pytest

from matcha.utils.sanitize import ensure_1d_array


class TestEnsure1dArray:
    """Tests for ensure_1d_array."""

    def test_already_1d(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = ensure_1d_array(arr)
        assert result.ndim == 1
        np.testing.assert_array_equal(result, arr)

    def test_column_vector(self):
        arr = np.array([[1.0], [2.0], [3.0]])
        result = ensure_1d_array(arr)
        assert result.ndim == 1
        assert result.shape == (3,)

    def test_row_vector(self):
        arr = np.array([[1.0, 2.0, 3.0]])
        result = ensure_1d_array(arr)
        assert result.ndim == 1
        assert result.shape == (3,)

    def test_2d_non_singleton_raises(self):
        arr = np.ones((3, 4))
        with pytest.raises(ValueError, match="Cannot convert 2D array"):
            ensure_1d_array(arr)

    def test_3d_single_non_singleton(self):
        arr = np.arange(5).reshape(1, 5, 1)
        result = ensure_1d_array(arr)
        assert result.ndim == 1
        assert result.shape == (5,)

    def test_3d_multiple_non_singleton_raises(self):
        arr = np.ones((2, 3, 4))
        with pytest.raises(ValueError, match="Cannot convert"):
            ensure_1d_array(arr)

    def test_preserves_dtype(self):
        arr = np.array([[1, 2, 3]], dtype=np.int32)
        result = ensure_1d_array(arr)
        assert result.dtype == np.int32
