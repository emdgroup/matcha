"""Label transformation functions for scaling targets (log, ln, etc.)."""

import numpy as np
from matcha.utils.wrapper import deepcopy_input
from matcha.utils.schemas.label import LabelTransformInputModel

tolerance = 0.001


@deepcopy_input
def no_scale(x: np.ndarray, tol: float = tolerance):
    return x


@deepcopy_input
def log10p(x: np.ndarray):
    return np.log10(x + 1)


@deepcopy_input
def norm_log10(x: np.ndarray, tol: float = tolerance):
    x[x < tol] = tol
    return np.log10(x)


@deepcopy_input
def norm_log2(x: np.ndarray, tol: float = tolerance):
    x[x < tol] = tol
    return np.log2(x)


@deepcopy_input
def norm_log1p(x: np.ndarray, tol: float = tolerance):
    x[x < tol] = tol
    return np.log1p(x)


@deepcopy_input
def norm_ln(x: np.ndarray, tol: float = tolerance):
    x[x < tol] = tol
    return np.log(x)


@deepcopy_input
def norm_logk100(x: np.ndarray, tol: float = tolerance):
    x[x < tol] = tol
    return np.log10((100.0 - x) / x)


@deepcopy_input
def inv_log10(x: np.ndarray):
    return 10**x


@deepcopy_input
def inv_log2(x: np.ndarray):
    return 2**x


@deepcopy_input
def inv_log1p(x: np.ndarray):
    return np.exp(x) - 1


@deepcopy_input
def inv_ln(x: np.ndarray):
    return np.exp(x)


@deepcopy_input
def inv_logk100(x: np.ndarray):
    return 100 / (10**x + 1)


@deepcopy_input
def inv_log10p(x: np.ndarray):
    return 10**x - 1


class ForwardTransformRegistry:
    """Simple registry for label transformation functions in forward mode
    (e.g., from unscaled to scaled)
    """

    mapping = {
        "log10": norm_log10,
        "log2": norm_log2,
        "log1p": norm_log1p,
        "ln": norm_ln,
        "logk100": norm_logk100,
        "log10p": log10p,
        "none": no_scale,
    }

    @classmethod
    def scale(cls, x, method):
        return cls.mapping[method.lower()](x)


class BackwardTransformRegistry:
    """Simple registry for inverted label transformation functions
    (e.g., from scaled to original space)
    """

    mapping = {
        "log10": inv_log10,
        "log2": inv_log2,
        "log1p": inv_log1p,
        "ln": inv_ln,
        "logk100": inv_logk100,
        "log10p": inv_log10p,
        "none": no_scale,
    }

    @classmethod
    def scale(cls, x, method):
        return cls.mapping[method.lower()](x)


class LabelTransform:
    """Scales labels back and forth according to a specified
    function.

    If users pass transform_map=str, then that transform is
    used across all tasks.

    If users pass transform_map=list[str], then transform_map[0]
    is used on y[:,0] and so forth.

    If users pass transform_map=dict, the keys correspond to the
    task idx, the values correspond to the function. For convenience,
    this pattern can also be the opposite way when many tasks share
    the same scaling fn. Internally, it will be still stored with
    task idx as keys.

    Example usage:

    .. code-block:: python
        t = LabelTransform(["log10", "logk100])
        y = np.random.rand(100,2)
        y_scaled = t.process(y, forward=True)
        y_recon = t.process(y_scaled, forward=False)

    :param str | list[str] | dict | None transform_map: ruleset to
        follow when trnasforming inputs
    """

    def __init__(
        self,
        transform_map: str | list[str] | dict | None = None,
        y_clip: dict | None = None,
    ):
        if isinstance(transform_map, str):
            processed_map = transform_map

        elif isinstance(transform_map, list):
            idx = list(range(len(transform_map)))
            processed_map = dict(zip(idx, transform_map))

        elif isinstance(transform_map, dict):
            keys = list(transform_map.keys())
            if isinstance(keys[0], int):
                processed_map = transform_map
            elif isinstance(keys[0], str):
                processed_map = {}
                for key in transform_map:
                    for value in transform_map[key]:
                        processed_map[value] = key

        elif transform_map is None:
            processed_map = None

        self.params = LabelTransformInputModel(
            transform_map=processed_map, y_clip=y_clip
        )

    def set_clipping_bounds(self, y_clip: dict | None):
        """Set clipping bounds for the label transformer.

        :param dict | None y_clip: dictionary with ``"Min"`` and ``"Max"`` keys
            for clipping bounds, or None to disable clipping
        """
        self.params.y_clip = y_clip

    def process(self, x: np.ndarray, forward: bool):
        """Apply label transformation in forward or inverse direction.

        :param np.ndarray x: array to transform
        :param bool forward: if True, apply forward transform (e.g. log);
            if False, apply inverse transform (e.g. 10^x)
        :returns: transformed array (copy, not in-place)
        :rtype: np.ndarray
        """
        x = x.copy()

        if forward is True:
            fn_box = ForwardTransformRegistry
        else:
            fn_box = BackwardTransformRegistry

        if isinstance(self.params.transform_map, str):
            x = fn_box.scale(x, self.params.transform_map)

        elif self.params.transform_map is not None:
            for i in range(x.shape[1]):
                x[:, i] = fn_box.scale(x[:, i], self.params.transform_map[i])

        if (
            not forward
            and isinstance(self.params.y_clip, dict)
            and "Min" in self.params.y_clip
            and "Max" in self.params.y_clip
        ):
            x[x < self.params.y_clip["Min"]] = self.params.y_clip["Min"]
            x[x > self.params.y_clip["Max"]] = self.params.y_clip["Max"]

        return x

    def state_dict(self) -> dict:
        """Utility for MLFlow logging"""

        return self.params.model_dump()

    def load_state_dict(self, state_dict: dict):
        """Utility for MLFlow logging"""
        self.params = LabelTransformInputModel.model_validate(state_dict)
