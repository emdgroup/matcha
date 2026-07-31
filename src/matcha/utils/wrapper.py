import importlib
from functools import wraps
import numpy as np
from joblib import Parallel, delayed
from typing import Callable


class Wrapper:
    """Utility class to wrap a module so that it can be used in parallel via
    joblib. This is typically necessary with some RDKIT functions.
    """

    def __init__(self, method_name: str, module_name: str):
        """Initialize the Wrapper by dynamically importing the target module.

        :param str method_name: Name of the function/method to wrap.
        :param str module_name: Fully qualified module path to import.
        """
        self.method_name = method_name
        self.module = importlib.import_module(module_name)

    @property
    def method(self):
        """Resolve and return the wrapped callable from the imported module.

        :returns: The function or method referenced by :attr:`method_name`.
        :rtype: Callable
        """
        return getattr(self.module, self.method_name)

    def __call__(self, *args, **kwargs):
        """Invoke the wrapped method with the given arguments.

        :param args: Positional arguments forwarded to the wrapped method.
        :param kwargs: Keyword arguments forwarded to the wrapped method.
        :returns: The result of calling the wrapped method.
        """
        return self.method(*args, **kwargs)


def deepcopy_input(func):
    """Decorator that copies the first positional argument before passing it.

    Creates a numpy copy of the first argument to prevent in-place mutation
    of the original array.

    :param Callable func: The function to wrap.
    :returns: Wrapped function that operates on a copy of its first argument.
    :rtype: Callable
    """

    @wraps(func)
    def wrapper(x, *args, **kwargs):
        x_copy = np.copy(x)
        return func(x_copy, *args, **kwargs)

    return wrapper


def parallelize(func: Callable, input: list, n_jobs: int, **kwargs):
    """Simple parallelization wrapper for misc tasks"""
    batch_size = max(1, len(input) // (n_jobs * 4))
    batches = [input[i : i + batch_size] for i in range(0, len(input), batch_size)]
    batch_results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(func)(batch, **kwargs) for batch in batches
    )
    output = [x for batch in batch_results for x in batch]
    return output
