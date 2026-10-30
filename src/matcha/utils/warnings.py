"""Utilities for suppressing known harmless warnings from third-party libraries."""

import warnings


def silence_nuisance_warnings():
    """Suppress known non-actionable warnings from PyTorch Lightning, Optuna,
    atomInSmiles, Hugging Face, and RDKit.

    This function should be called early in the application lifecycle to
    prevent noisy but harmless warning messages from cluttering logs.
    """
    # Pytorch Lightning warnings that are safe to ignore
    # see discussion here: https://github.com/Lightning-AI/pytorch-lightning/issues/10644
    warnings.filterwarnings(
        "ignore",
        ".*does not have many workers.which may be a bottleneck. Consider increasing the value of.*",
    )
    warnings.filterwarnings(
        "ignore",
        ".*Starting from v1.9.0, `tensorboardX` has been removed as a dependency.*",
    )
    warnings.filterwarnings(
        "ignore", ".*ou defined a `validation_step` but have no `val_dataloader`.*"
    )
    warnings.filterwarnings(
        "ignore",
        ".*Using padding='same' with even kernel lengths and odd dilation may require a zero-padded copy of the input be created.*",
    )
    warnings.filterwarnings(
        "ignore", ".*GPU available but not used. You can set it by doing.*"
    )
    warnings.filterwarnings(
        "ignore", ".*is smaller than the logging interval Trainer.*"
    )
    warnings.filterwarnings(
        "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
    )
    warnings.filterwarnings(
        "ignore", ".*You are using `torch.load` with `weights_only=False`.*"
    )

    # Optuna warnings that are safe to ignore
    warnings.filterwarnings(
        "ignore",
        ".*Choices for a categorical distribution should be a tuple of None, bool, int, float and str for persistent storage but contains.*",
    )
    warnings.filterwarnings("ignore", ".*QMCSampler is experimental.*")

    # atomInSmiles warnings that are safe to ignore
    warnings.filterwarnings("ignore", ".*invalid escape sequence.*")

    # InfoAlign/huggingface warnings that are safe to ignore
    warnings.filterwarnings(
        "ignore",
        ".*`local_dir_use_symlinks` parameter is deprecated and will be ignored.*",
    )

    # RDKIT warnings that are safe to ignore
    warnings.filterwarnings("ignore", ".*please use MorganGenerator.*")
    warnings.filterwarnings(
        "ignore", ".*DEPRECATION WARNING: please use MorganGenerator.*"
    )
