"""Array sanitization utilities for ensuring consistent array dimensions."""

import numpy as np


def ensure_1d_array(arr: np.ndarray) -> np.ndarray:
    """
    Ensure that a numpy array is 1-dimensional.

    Args:
        arr: Input numpy array

    Returns:
        1-dimensional numpy array

    Raises:
        ValueError: If array has more than 1 non-singleton dimension
    """
    if arr.ndim == 1:
        return arr
    elif arr.ndim == 2:
        # Check if one dimension is singleton (e.g., shape (10, 1) or (1, 10))
        if arr.shape[0] == 1:
            return arr.flatten()
        elif arr.shape[1] == 1:
            return arr.flatten()
        else:
            raise ValueError(
                f"Cannot convert 2D array with shape {arr.shape} to 1D. "
                "Array must have at least one singleton dimension."
            )
    else:
        # For arrays with more than 2 dimensions, check if only one dimension is non-singleton
        non_singleton_dims = [i for i, size in enumerate(arr.shape) if size > 1]
        if len(non_singleton_dims) == 1:
            return arr.flatten()
        else:
            raise ValueError(
                f"Cannot convert {arr.ndim}D array with shape {arr.shape} to 1D. "
                "Array must have at most one non-singleton dimension."
            )
