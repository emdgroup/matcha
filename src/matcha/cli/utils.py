"""Shared utilities for MATCHA CLI commands.

Provides dataset loading, train/test splitting, score aggregation, bootstrap
metric computation, performance-dict writing, and configuration serialization
helpers used across multiple CLI subcommands.
"""

from rdkit import Chem
from rdkit.Chem import PandasTools
import pandas as pd
from sklearn.model_selection import KFold
from matcha.utils.splitting import cluster_split
import numpy as np
from typing import Callable


def load_dataset(cfg) -> pd.DataFrame:
    """Load a molecular dataset from CSV or SDF and return a DataFrame with ROMol column.

    Reads the file specified in *cfg*, ensures a SMILES column exists, converts
    SMILES to RDKit ``Mol`` objects, and drops rows with invalid molecules.

    :param cfg: Dataset configuration object with ``path``, ``smiles_key``,
        and optionally other column-key attributes.
    :returns: DataFrame containing at least a ``ROMol`` column and the
        original SMILES column.
    """
    if ".csv" in cfg.path:
        df = pd.read_csv(cfg.path)
        df[cfg.smiles_key] = df[cfg.smiles_key].replace("", None)
        df = df.dropna(subset=[cfg.smiles_key])
        df["ROMol"] = df[cfg.smiles_key].apply(Chem.MolFromSmiles)
    else:
        df = PandasTools.LoadSDF(cfg.path)
        if cfg.smiles_key not in df.columns:
            df[cfg.smiles_key] = df["ROMol"].apply(Chem.MolToSmiles)
            df[cfg.smiles_key] = df[cfg.smiles_key].replace("", None)
        df = df.dropna(subset=[cfg.smiles_key])
    return df.dropna(subset=["ROMol"])


def get_splits(
    df: pd.DataFrame,
    split_cfg,
    return_val_splits: bool = False,
    dataset_cfg: dict | None = None,
) -> tuple[list[pd.DataFrame]]:
    """Generate train/test (or train/val/test) splits from a DataFrame.

    Supports multiple splitting strategies: cross-validation (``cv``),
    temporal (``time``), cluster-based (``cluster``), and file-based
    (``file``). Every method produces ``n_subset`` splits (or, for the
    ``file`` method, one split per supplied test-set path).

    Bootstrapping (``n_bootstrap`` / ``frac_bootstrap``) is applied
    downstream of this function on each split's test predictions and is
    therefore independent of the split method chosen here.

    :param df: Input DataFrame containing molecular data.
    :param split_cfg: Split configuration object with ``method``,
        ``n_subset``, and ``method_params``.
    :param return_val_splits: If ``True``, return validation and test sets
        separately as a list of ``[df_val, df_test]`` pairs instead of
        concatenating them.
    :param dataset_cfg: Optional dataset configuration used by file-based
        splitting to load external test sets.
    :returns: Tuple of ``(train_splits, val_splits)`` where each is a list
        of DataFrames (one per split).
    """
    train_splits = []
    val_splits = []

    if split_cfg.method.lower() == "file":
        test_paths = split_cfg.method_params["path"]
        for test_path in test_paths:
            train_splits.append(df)
            dataset_ith_cfg = dataset_cfg.model_copy()
            dataset_ith_cfg.path = test_path
            df_test = load_dataset(dataset_ith_cfg)
            if return_val_splits:
                mid_point = int(len(df_test) / 2)
                val_splits.append([df_test.iloc[:mid_point], df_test.iloc[mid_point:]])
            else:
                val_splits.append(df_test)

    elif split_cfg.method.lower() == "cv":
        splitter = KFold(n_splits=split_cfg.n_subset, shuffle=True, random_state=0)
        for train_index, test_index in splitter.split(df):
            df_train = df.iloc[train_index]
            train_splits.append(df_train)

            val_index = test_index[: int(len(test_index) / 2)]
            test_index = test_index[int(len(test_index) / 2) :]
            df_test = df.iloc[test_index]
            df_val = df.iloc[val_index]

            if return_val_splits:
                val_splits.append([df_val, df_test])
            else:
                val_splits.append(pd.concat([df_val, df_test], axis=0))

    elif split_cfg.method.lower() == "time":
        # Sort dataframe by time column first
        df_sorted = df.sort_values(by=split_cfg.method_params["key"]).reset_index(
            drop=True
        )

        for i in range(split_cfg.n_subset):
            # Calculate split point: train on first (1 - split_size*(n_subset-i)) portion
            # This ensures expanding training sets: first iteration uses more data for training
            train_portion = 1 - split_cfg.method_params["split_size"] * (
                split_cfg.n_subset - i
            )

            # Ensure train_portion is valid (not negative)
            if train_portion <= 0:
                raise ValueError(
                    f"Invalid train portion {train_portion}. Consider reducing split_size or n_subset."
                )

            # Calculate split index
            split_index = int(len(df_sorted) * train_portion)

            # Create train and test splits
            df_train = df_sorted.iloc[:split_index].copy()
            df_test = df_sorted.iloc[
                split_index : split_index
                + int(len(df_sorted) * split_cfg.method_params["split_size"])
            ].copy()

            train_splits.append(df_train)

            # Split test set into validation and test portions
            mid_point = int(len(df_test) / 2)
            df_val = df_test.iloc[:mid_point]
            df_test = df_test.iloc[mid_point:]

            if return_val_splits:
                val_splits.append([df_val, df_test])
            else:
                val_splits.append(pd.concat([df_val, df_test], axis=0))

    elif split_cfg.method.lower() == "cluster":
        for i in range(split_cfg.n_subset):
            features = split_cfg.method_params["features"][i]
            metric = split_cfg.method_params["metric"][i]
            df_train, df_test = cluster_split(
                df,
                feature_set=features,
                metric=metric,
                split_size=split_cfg.method_params["split_size"],
                n_jobs=split_cfg.method_params["n_jobs"],
            )
            mid_point = int(len(df_test) / 2)
            df_val = df_test.iloc[:mid_point]
            df_test = df_test.iloc[mid_point:]
            train_splits.append(df_train)
            if return_val_splits:
                val_splits.append([df_val, df_test])
            else:
                val_splits.append(pd.concat([df_val, df_test], axis=0))

    return train_splits, val_splits


def aggregate_scores(perf_dict, mode: str = "mean"):
    """Aggregate per-split performance scores into a single summary.

    Collects metric values across all splits (excluding ``mean`` and ``std``
    keys) and reduces them with the specified aggregation function.

    :param perf_dict: Dictionary mapping split indices to
        ``{label: {metric: value}}`` nested dicts.
    :param mode: Aggregation mode — ``"mean"`` for average, ``"std"`` for
        standard deviation.
    :returns: Dictionary of aggregated scores with the same
        ``{label: {metric: value}}`` structure.
    """
    aggregated_scores = {}
    # Get all label keys
    label_keys = set()
    for split in perf_dict:
        if split not in ("mean", "std"):
            label_keys.update(perf_dict[split].keys())
    for label in label_keys:
        # Collect all values for this label across splits
        for split in perf_dict:
            if split not in ("mean", "std") and label in perf_dict[split]:
                score = perf_dict[split][label]
                # If score is a dict (multiple metrics), aggregate each metric
                if isinstance(score, dict):
                    for metric, val in score.items():
                        aggregated_scores.setdefault(label, {}).setdefault(
                            metric, []
                        ).append(val)
                else:
                    aggregated_scores.setdefault(label, []).append(score)
        # Aggregate
        for metric in aggregated_scores[label]:
            vals = aggregated_scores[label][metric]
            if mode == "std":
                aggregated_scores[label][metric] = float(np.std(vals))
            else:
                aggregated_scores[label][metric] = float(np.mean(vals))

    return aggregated_scores


def bootstrap_metrics(
    metric_fn: Callable,
    n_bootstrap: int,
    frac_bootstrap: float,
    *arrays,
    **metric_kwargs,
) -> list:
    """Compute metrics over bootstrap subsamples.

    Repeatedly draws random subsets (with replacement) from the provided
    arrays and evaluates *metric_fn* on each subsample to estimate metric
    variability.

    :param metric_fn: Callable accepting the arrays positionally followed by
        ``**metric_kwargs`` and returning a ``dict[str, float]`` of metric
        values.
    :param n_bootstrap: Number of bootstrap iterations.
    :param frac_bootstrap: Fraction of rows to sample in each iteration.
    :param arrays: Arrays to subsample (labels, predictions, probabilities,
        etc.). All must share the same length along axis 0.
    :param metric_kwargs: Extra keyword arguments forwarded to *metric_fn*.
    :returns: List of length *n_bootstrap*, where each element is a
        ``dict[str, float]`` of metric values for that subsample.
    """
    n = len(arrays[0])
    sample_size = max(1, int(n * frac_bootstrap))
    all_scores = []
    rng = np.random.default_rng(seed=0)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=sample_size, replace=True)
        sampled = [a[idx] for a in arrays]
        scores = metric_fn(*sampled, **metric_kwargs)
        all_scores.append(scores)
    return all_scores


def store_split_scores(
    perf_dict: dict,
    split_idx: int,
    label: str,
    metric_fn: Callable,
    n_bootstrap,
    frac_bootstrap,
    *arrays,
    **metric_kwargs,
) -> None:
    """Compute and store per-split metric scores, handling bootstrap and non-bootstrap paths.

    When bootstrapping is inactive (``n_bootstrap`` is ``None`` or ``1``),
    computes the metric once and stores it under the integer *split_idx* key.
    When bootstrapping is active, runs :func:`bootstrap_metrics` and stores
    each subsample result under a ``"{split_idx}_{b}"`` string key so that
    downstream bootstrap comparison tools can identify them.

    :param perf_dict: Mutable performance dictionary to write into.
    :param split_idx: Integer split index (e.g. fold number).
    :param label: Endpoint label key (e.g. ``"SOLUBILITY"``).
    :param metric_fn: Metric callable — e.g. ``process_regression`` or
        ``process_classification``.
    :param n_bootstrap: Number of bootstrap iterations, or ``None`` / ``1``
        to skip bootstrapping.
    :param frac_bootstrap: Fraction of test-set rows to sample per bootstrap
        iteration.  Only used when *n_bootstrap* > 1.
    :param arrays: Positional arrays forwarded to *metric_fn* (and to
        :func:`bootstrap_metrics` for subsampling).
    :param metric_kwargs: Keyword arguments forwarded to *metric_fn*.
    """
    if n_bootstrap is None or n_bootstrap == 1:
        perf_dict.setdefault(split_idx, {})[label] = metric_fn(*arrays, **metric_kwargs)
    else:
        for b, scores in enumerate(
            bootstrap_metrics(
                metric_fn, n_bootstrap, frac_bootstrap, *arrays, **metric_kwargs
            )
        ):
            perf_dict.setdefault(f"{split_idx}_{b}", {})[label] = scores


def save_config_as_yaml(pydantic_config, output_path: str) -> None:
    """Save a Pydantic model configuration as a YAML file.

    :param pydantic_config: Pydantic model instance to serialize.
    :param output_path: Path where the YAML file should be written.
    """
    import yaml

    cfg_dict = pydantic_config.model_dump()
    with open(output_path, "w") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False, sort_keys=False)


def _load_npz_list(path: str) -> list[np.ndarray]:
    """Load a list of variable-length arrays from a packed npz file.

    Expected on-disk layout (``flat`` + ``offsets``)::

        np.savez_compressed("node_y.npz", flat=<concatenated>, offsets=<cumulative>)

    where ``offsets`` has length ``N + 1`` and ``flat`` has shape
    ``(total_items,)`` or ``(total_items, D)``.

    :param path: Path to the packed ``.npz`` file.
    :returns: A list of ``N`` numpy arrays reconstructed from the flat buffer.
    """
    data = np.load(path)
    flat = data["flat"]
    offsets = data["offsets"]
    return [flat[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]


def _load_coords_npz(path: str) -> list[np.ndarray]:
    """Load per-molecule 3D coordinates from a packed npz file.

    Uses the same ``flat + offsets`` layout as :func:`_load_npz_list` but
    with two additional invariants: ``flat`` is a 2D array of shape
    ``(total_atoms, 3)`` and each molecule's slice becomes an
    ``(A_i, 3)`` ``float32`` array.

    Startup guards raise ``AssertionError`` on malformed inputs, so the CLI
    fails fast before featurization is ever attempted:

    - ``flat.ndim == 2``
    - ``flat.shape[1] == 3``
    - ``offsets.ndim == 1``
    - ``offsets`` is monotonic non-decreasing
    - ``offsets[-1] == flat.shape[0]``

    :param path: Path to the packed ``.npz`` file.
    :returns: A list of ``N`` ``float32`` arrays, each of shape ``(A_i, 3)``.
    """
    data = np.load(path)
    flat = data["flat"]
    offsets = data["offsets"]

    assert flat.ndim == 2, (
        f"coords npz '{path}': expected flat.ndim == 2, got {flat.ndim}"
    )
    assert flat.shape[1] == 3, (
        f"coords npz '{path}': expected flat.shape[1] == 3, got {flat.shape[1]}"
    )
    assert offsets.ndim == 1, (
        f"coords npz '{path}': expected offsets.ndim == 1, got {offsets.ndim}"
    )
    assert bool(np.all(np.diff(offsets) >= 0)), (
        f"coords npz '{path}': offsets must be monotonic non-decreasing"
    )
    assert int(offsets[-1]) == flat.shape[0], (
        f"coords npz '{path}': offsets[-1]={int(offsets[-1])} does not match "
        f"flat.shape[0]={flat.shape[0]}"
    )

    flat32 = flat.astype(np.float32, copy=False)
    return [flat32[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]
