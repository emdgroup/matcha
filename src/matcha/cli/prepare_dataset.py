"""CLI command for preparing multi-task datasets for pretraining.

Merges multiple parquet/CSV files into a single (compounds x tasks) label
matrix, creates train/validation splits, applies standard scaling to
regression tasks, and saves the result in a parquet + labels format consumed
by the ``pretrain_multitask`` command.

Two storage modes are supported and selected via ``datasets.sparse``:

* ``sparse=True`` (default) — labels are stored as ``scipy.sparse`` CSR
  matrices (``train_tasks_sparse.npz`` / ``val_tasks_sparse.npz``). Missing
  entries are represented by omission and classification zeros are remapped
  to ``-1`` so they survive the sparse round-trip.
* ``sparse=False`` — labels are stored as dense ``float32`` arrays
  (``train_tasks_dense.npy`` / ``val_tasks_dense.npy``). Missing entries are
  ``NaN`` and classification values pass through as ``0``/``1`` unchanged.

The storage mode is written into ``task_metadata.json`` under
``storage_mode`` so downstream commands can dispatch on the artifact layout
without a user-visible flag.
"""

import os
import argparse
import yaml
import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from typing import List, Dict, Tuple, Any
from rdkit import Chem
from matcha.utils.logging import get_default_logger
from matcha.utils.schemas.cli import CLIPrepareInputModel
from matcha.utils.serialization import save_json
from matcha.cli.utils import save_config_as_yaml


def compute_sparse_scaling_stats(
    sparse_matrix: csr_matrix, task_indices: List[int], logger: Any
) -> Dict[int, Dict[str, float]]:
    """Compute mean and std for specified task columns in sparse matrix without densifying.

    :param sparse_matrix: CSR sparse matrix.
    :param task_indices: List of column indices to compute stats for.
    :param logger: Logger instance.
    :returns: Dictionary mapping task index to
        ``{"mean": float, "std": float}``.
    """
    scaling_stats = {}

    for task_idx in task_indices:
        # Extract column data (non-zero values only)
        col_data = sparse_matrix[:, task_idx]
        non_zero_data = col_data.data  # Only non-zero values

        if len(non_zero_data) > 0:
            mean_val = float(np.mean(non_zero_data))
            std_val = float(np.std(non_zero_data))
            # Avoid division by zero
            if std_val == 0:
                std_val = 1.0
        else:
            mean_val = 0.0
            std_val = 1.0

        scaling_stats[task_idx] = {"mean": mean_val, "std": std_val}
        logger.info(f"Task {task_idx}: mean={mean_val:.4f}, std={std_val:.4f}")

    return scaling_stats


def apply_sparse_scaling(
    sparse_matrix: csr_matrix, scaling_stats: Dict[int, Dict[str, float]], logger: Any
) -> csr_matrix:
    """Apply scaling to sparse matrix columns without densifying the entire matrix.

    :param sparse_matrix: Input sparse matrix.
    :param scaling_stats: Dictionary with scaling statistics per column
        (from :func:`compute_sparse_scaling_stats`).
    :param logger: Logger instance.
    :returns: Scaled sparse matrix.
    """
    # Create a copy to avoid modifying the original
    scaled_matrix = sparse_matrix.copy()

    # Convert to COO format for easier column-wise operations
    scaled_matrix_coo = scaled_matrix.tocoo()

    for task_idx, stats in scaling_stats.items():
        mean_val = stats["mean"]
        std_val = stats["std"]

        # Find all entries for this column
        col_mask = scaled_matrix_coo.col == task_idx

        if np.any(col_mask):
            # Apply scaling to all values in this column
            scaled_matrix_coo.data[col_mask] = (
                scaled_matrix_coo.data[col_mask] - mean_val
            ) / std_val

    # Convert back to CSR format
    scaled_matrix = scaled_matrix_coo.tocsr()

    logger.info(f"Applied scaling to {len(scaling_stats)} regression tasks")
    return scaled_matrix


def compute_dense_scaling_stats(
    dense_matrix: np.ndarray, task_indices: List[int], logger: Any
) -> Dict[int, Dict[str, float]]:
    """Compute mean and std for specified task columns in a dense matrix, ignoring NaNs.

    Mirrors :func:`compute_sparse_scaling_stats` but operates on a 2D ``np.ndarray``
    where missing entries are represented as ``NaN`` rather than omission.

    :param dense_matrix: 2D dense matrix of shape ``(n_compounds, n_tasks)``.
    :param task_indices: List of column indices to compute stats for.
    :param logger: Logger instance.
    :returns: Dictionary mapping task index to
        ``{"mean": float, "std": float}``.
    """
    scaling_stats: Dict[int, Dict[str, float]] = {}

    for task_idx in task_indices:
        col = dense_matrix[:, task_idx]
        # Determine whether the column contains at least one non-NaN entry
        # without densifying anything extra — `np.nanmean` on an all-NaN slice
        # would emit a RuntimeWarning otherwise.
        finite_mask = ~np.isnan(col)
        if np.any(finite_mask):
            mean_val = float(np.nanmean(col))
            std_val = float(np.nanstd(col))
            if std_val == 0:
                std_val = 1.0
        else:
            mean_val = 0.0
            std_val = 1.0

        scaling_stats[task_idx] = {"mean": mean_val, "std": std_val}
        logger.info(f"Task {task_idx}: mean={mean_val:.4f}, std={std_val:.4f}")

    return scaling_stats


def apply_dense_scaling(
    dense_matrix: np.ndarray, scaling_stats: Dict[int, Dict[str, float]], logger: Any
) -> np.ndarray:
    """Apply column-wise standard scaling to a dense matrix, preserving NaN entries.

    :param dense_matrix: 2D dense matrix of shape ``(n_compounds, n_tasks)``.
    :param scaling_stats: Dictionary with scaling statistics per column
        (from :func:`compute_dense_scaling_stats`).
    :param logger: Logger instance.
    :returns: Scaled dense matrix (NaN positions preserved unchanged).
    """
    scaled_matrix = dense_matrix.copy()

    for task_idx, stats in scaling_stats.items():
        mean_val = stats["mean"]
        std_val = stats["std"]

        col = scaled_matrix[:, task_idx]
        finite_mask = ~np.isnan(col)
        if np.any(finite_mask):
            col[finite_mask] = (col[finite_mask] - mean_val) / std_val

    logger.info(f"Applied scaling to {len(scaling_stats)} regression tasks")
    return scaled_matrix


def merge_datasets_streaming_dense(
    files: List[str],
    merge_col: str,
    task_types: List[str],
    tag_to_add: str,
    logger: Any,
) -> Tuple[
    pd.DataFrame,
    np.ndarray,
    List[str],
    Dict[str, str],
    Dict[str, str],
    Dict[str, List[int]],
]:
    """Build a dense ``(n_compounds, n_tasks)`` label matrix sequentially.

    Mirrors :func:`merge_datasets_streaming_sparse`: the first pass (compound and
    task discovery) is identical; the second pass fills a
    ``np.full((n, T), np.nan, dtype=np.float32)`` array. Missing entries stay NaN
    and classification values are written as-is (no ``0 → -1`` remap — dense mode
    uses NaN to denote missing, so the remap is unnecessary).
    """

    logger.info("First pass: collecting compound IDs and task columns...")
    all_compounds = set()
    all_task_info = []  # [(file_idx, original_col, new_col, task_type), ...]

    for file_idx, (file, task_type) in enumerate(zip(files, task_types)):
        file_path = file
        logger.info(f"Scanning {file_path}")

        if file_path.endswith(".csv"):
            df_compounds = pd.read_csv(file_path, usecols=[merge_col])
        else:
            df_compounds = pd.read_parquet(file_path, columns=[merge_col])

        df_compounds["ROMol"] = df_compounds[merge_col].apply(Chem.MolFromSmiles)
        df_compounds = df_compounds.dropna(subset="ROMol", axis=0)
        df_compounds[merge_col] = df_compounds.ROMol.apply(Chem.MolToSmiles)

        compounds_in_file = set(df_compounds[merge_col].tolist())
        all_compounds.update(compounds_in_file)

        if file_path.endswith(".csv"):
            df_cols = pd.read_csv(file_path, nrows=0)
        else:
            df_cols = pd.read_parquet(file_path).head(0)
        task_cols = [col for col in df_cols.columns if col != merge_col]

        for col in task_cols:
            new_col = f"{col}_{tag_to_add}"
            all_task_info.append((file_idx, col, new_col, task_type))

        logger.info(
            f"File {file_idx}: {len(compounds_in_file)} compounds, {len(task_cols)} tasks"
        )
        del df_compounds, df_cols

    all_compounds = sorted(list(all_compounds))
    compound_to_idx = {comp: idx for idx, comp in enumerate(all_compounds)}
    task_cols_final = [info[2] for info in all_task_info]
    column_to_task_type = {info[2]: info[3] for info in all_task_info}

    task_to_file: Dict[str, str] = {}
    file_to_tasks: Dict[str, List[int]] = {}

    for task_col_idx, (file_idx, _orig, new_col, _task_type) in enumerate(
        all_task_info
    ):
        filename = files[file_idx]
        task_to_file[new_col] = filename
        if filename not in file_to_tasks:
            file_to_tasks[filename] = []
        file_to_tasks[filename].append(task_col_idx)

    logger.info(f"Total: {len(all_compounds)} compounds, {len(task_cols_final)} tasks")

    logger.info("Second pass: building dense matrix...")
    n_compounds = len(all_compounds)
    n_tasks = len(task_cols_final)

    dense_matrix = np.full((n_compounds, n_tasks), np.nan, dtype=np.float32)

    current_task_idx = 0
    for file_idx in range(len(files)):
        file_path = files[file_idx]
        logger.info(f"Processing {file_path} for dense matrix...")

        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_parquet(file_path)

        df["ROMol"] = df[merge_col].apply(Chem.MolFromSmiles)
        df = df.dropna(subset="ROMol", axis=0)
        df[merge_col] = df.ROMol.apply(Chem.MolToSmiles)
        df = df.drop("ROMol", axis=1)

        file_task_info = [info for info in all_task_info if info[0] == file_idx]

        for _, orig_col, _new_col, _task_type in file_task_info:
            if orig_col in df.columns:
                non_null_mask = df[orig_col].notna()
                if non_null_mask.any():
                    compounds_with_data = df.loc[non_null_mask, merge_col]
                    values = df.loc[non_null_mask, orig_col].to_numpy(dtype=np.float32)
                    compound_indices = np.fromiter(
                        (compound_to_idx[comp] for comp in compounds_with_data),
                        dtype=np.int64,
                        count=len(compounds_with_data),
                    )
                    dense_matrix[compound_indices, current_task_idx] = values

            current_task_idx += 1

        del df
        logger.info(f"Processed file {file_idx}")

    logger.info(
        f"Dense matrix created: {dense_matrix.shape}, "
        f"non-NaN fraction: {float(np.mean(~np.isnan(dense_matrix))):.6f}"
    )

    mol_df_clean = pd.DataFrame({merge_col: all_compounds})

    logger.info(f"Final molecules DataFrame: {mol_df_clean.shape}")
    logger.info(f"Final dense task matrix: {dense_matrix.shape}")

    return (
        mol_df_clean,
        dense_matrix,
        task_cols_final,
        column_to_task_type,
        task_to_file,
        file_to_tasks,
    )


def create_validation_set_dense(
    mol_df: pd.DataFrame,
    dense_matrix: np.ndarray,
    task_cols: List[str],
    min_compounds: int,
    sampling_rate: float,
    seed: int,
    logger: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Create validation split for dense data by sampling compounds globally.

    In dense mode essentially every compound has non-NaN entries for every
    task, so the sparse-mode per-task union heuristic would compound the
    per-compound inclusion probability toward ``1 - (1 - sampling_rate) ** n_tasks``
    and push most of the dataset into validation. This function instead draws
    a single global sample of compounds so that ``len(val) ≈ sampling_rate *
    n_compounds`` regardless of the number of tasks. ``min_compounds`` acts as
    a global floor on the validation size — contrast with the sparse variant,
    where it is a per-task floor. The exact size is::

        n_val = min(n_compounds, max(min_compounds, int(sampling_rate * n_compounds)))

    :param mol_df: DataFrame of compounds (one row per compound).
    :param dense_matrix: ``(n_compounds, n_tasks)`` label matrix; ``NaN``
        marks missing entries.
    :param task_cols: Task column names; used for logging only.
    :param min_compounds: Global floor on the validation-set size. Must be
        non-negative and no greater than ``n_compounds``.
    :param sampling_rate: Target fraction of compounds to place in
        validation. Must lie in ``[0.0, 1.0]``.
    :param seed: Seed for the numpy RNG driving the split.
    :param logger: Logger instance.
    :returns: ``(train_mol_df, val_mol_df, train_dense, val_dense)``.
    :raises ValueError: If ``n_compounds == 0``, ``min_compounds > n_compounds``,
        ``sampling_rate`` is outside ``[0.0, 1.0]``, or ``min_compounds < 0``.
    """

    n_compounds = len(mol_df)

    if n_compounds == 0:
        raise ValueError("Cannot create validation set from an empty dataset.")
    if min_compounds < 0:
        raise ValueError(f"min_compounds must be non-negative, got {min_compounds}.")
    if min_compounds > n_compounds:
        raise ValueError(
            f"min_compounds ({min_compounds}) exceeds n_compounds "
            f"({n_compounds}); floor cannot be honored."
        )
    if not 0.0 <= sampling_rate <= 1.0:
        raise ValueError(f"sampling_rate must lie in [0.0, 1.0], got {sampling_rate}.")

    rng = np.random.default_rng(seed)

    logger.info(f"Creating validation set for {len(task_cols)} tasks")

    n_val = min(n_compounds, max(min_compounds, int(sampling_rate * n_compounds)))
    chosen = rng.choice(n_compounds, size=n_val, replace=False)
    val_mask = np.zeros(n_compounds, dtype=bool)
    val_mask[chosen] = True

    train_mol_df = mol_df.iloc[~val_mask].reset_index(drop=True)
    val_mol_df = mol_df.iloc[val_mask].reset_index(drop=True)

    train_dense = dense_matrix[~val_mask]
    val_dense = dense_matrix[val_mask]

    logger.info(
        f"Train: {len(train_mol_df)} compounds, Val: {len(val_mol_df)} compounds"
    )

    return train_mol_df, val_mol_df, train_dense, val_dense


def generate_datacard_dense(
    train_mol_df: pd.DataFrame,
    val_mol_df: pd.DataFrame,
    train_dense: np.ndarray,
    val_dense: np.ndarray,
    task_cols: List[str],
    column_to_task_type: Dict[str, str],
    files: List[str],
    file_to_tasks: Dict[str, List[int]],
    logger: Any,
) -> Dict[str, Any]:
    """Generate datacard for dense-format data (NaN = missing)."""

    values_per_task = [
        int(np.sum(~np.isnan(train_dense[:, i]))) for i in range(len(task_cols))
    ]
    avg_values_per_task = float(np.mean(values_per_task)) if values_per_task else 0.0

    non_nan_mask = ~np.isnan(train_dense)
    measurements_per_compound = non_nan_mask.sum(axis=1)
    avg_measurements_per_compound = float(np.mean(measurements_per_compound))

    tasks_per_file = {}
    for filename, task_indices in file_to_tasks.items():
        task_types = [
            column_to_task_type.get(task_cols[idx], "unknown") for idx in task_indices
        ]

        tasks_per_file[filename] = {
            "count": len(task_indices),
            "task_types": task_types[0],
        }

    datacard = {
        "total_tasks": len(task_cols),
        "tasks_per_file": tasks_per_file,
        "compounds_in_train": len(train_mol_df),
        "compounds_in_val": len(val_mol_df),
        "avg_values_per_task": avg_values_per_task,
        "avg_measurements_per_compound": avg_measurements_per_compound,
        "non_nan_fraction": float(np.mean(non_nan_mask)),
    }
    return datacard


def merge_datasets_streaming_sparse(
    files: List[str],
    merge_col: str,
    task_types: List[str],
    tag_to_add: str,
    logger: Any,
) -> Tuple[
    pd.DataFrame,
    csr_matrix,
    List[str],
    Dict[str, str],
    Dict[str, str],
    Dict[str, List[int]],
]:
    """Build sparse matrix sequentially without loading all datasets into memory."""

    # First pass: collect all unique compounds and task column names
    logger.info("First pass: collecting compound IDs and task columns...")
    all_compounds = set()
    all_task_info = []  # [(file_idx, original_col, new_col, task_type), ...]

    for file_idx, (file, task_type) in enumerate(zip(files, task_types)):
        file_path = file
        logger.info(f"Scanning {file_path}")

        # Load just the merge column to get compounds
        if file_path.endswith(".csv"):
            df_compounds = pd.read_csv(file_path, usecols=[merge_col])
        else:
            df_compounds = pd.read_parquet(file_path, columns=[merge_col])

        # Standardize compound IDs
        df_compounds["ROMol"] = df_compounds[merge_col].apply(Chem.MolFromSmiles)
        df_compounds = df_compounds.dropna(subset="ROMol", axis=0)
        df_compounds[merge_col] = df_compounds.ROMol.apply(Chem.MolToSmiles)

        compounds_in_file = set(df_compounds[merge_col].tolist())
        all_compounds.update(compounds_in_file)

        # Get column info (we'll load the full file later)
        if file_path.endswith(".csv"):
            df_cols = pd.read_csv(file_path, nrows=0)
        else:
            df_cols = pd.read_parquet(file_path).head(0)
        task_cols = [col for col in df_cols.columns if col != merge_col]

        for col in task_cols:
            new_col = f"{col}_{tag_to_add}"
            all_task_info.append((file_idx, col, new_col, task_type))

        logger.info(
            f"File {file_idx}: {len(compounds_in_file)} compounds, {len(task_cols)} tasks"
        )
        del df_compounds, df_cols  # Free memory

    # Create mappings
    all_compounds = sorted(list(all_compounds))
    compound_to_idx = {comp: idx for idx, comp in enumerate(all_compounds)}
    task_cols_final = [
        info[2] for info in all_task_info
    ]  # new column names (excluding descriptors)
    column_to_task_type = {info[2]: info[3] for info in all_task_info}

    # Create task-to-file mapping for curriculum learning (excluding descriptors)
    task_to_file = {}
    file_to_tasks = {}

    for task_col_idx, (file_idx, original_col, new_col, task_type) in enumerate(
        all_task_info
    ):
        filename = files[file_idx]
        task_to_file[new_col] = filename

        # Build reverse mapping with column indices instead of task names
        if filename not in file_to_tasks:
            file_to_tasks[filename] = []
        file_to_tasks[filename].append(task_col_idx)

    logger.info(f"Total: {len(all_compounds)} compounds, {len(task_cols_final)} tasks")

    # Second pass: build sparse matrix incrementally and compute molecular descriptors
    logger.info("Second pass: building sparse matrix...")
    n_compounds = len(all_compounds)
    n_tasks = len(task_cols_final)

    # Collect sparse matrix data
    rows, cols, data = [], [], []

    # Process file-based tasks
    current_task_idx = 0
    for file_idx in range(len(files)):
        file_path = files[file_idx]
        logger.info(f"Processing {file_path} for sparse matrix...")

        # Load full dataset
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_parquet(file_path)

        # Standardize compounds (same as before)
        df["ROMol"] = df[merge_col].apply(Chem.MolFromSmiles)
        df = df.dropna(subset="ROMol", axis=0)
        df[merge_col] = df.ROMol.apply(Chem.MolToSmiles)
        df = df.drop("ROMol", axis=1)

        # Get task columns for this file
        file_task_info = [info for info in all_task_info if info[0] == file_idx]

        # Process each task column
        for _, orig_col, new_col, task_type in file_task_info:
            if orig_col in df.columns:
                # Get non-null values
                non_null_mask = df[orig_col].notna()
                if non_null_mask.any():
                    compounds_with_data = df.loc[non_null_mask, merge_col]
                    values = df.loc[non_null_mask, orig_col].tolist()

                    # For classification tasks, replace 0 with -1 to distinguish from missing values
                    if task_type == "classification":
                        values = [-1 if v == 0 else v for v in values]

                    # Convert to sparse matrix indices
                    compound_indices = [
                        compound_to_idx[comp] for comp in compounds_with_data
                    ]

                    rows.extend(compound_indices)
                    cols.extend([current_task_idx] * len(compound_indices))
                    data.extend(values)

            current_task_idx += 1

        del df  # Free memory immediately
        logger.info(f"Processed file {file_idx}, current sparse entries: {len(data)}")

    # Create sparse matrix
    logger.info("Creating final sparse matrix...")
    sparse_matrix = csr_matrix((data, (rows, cols)), shape=(n_compounds, n_tasks))
    logger.info(
        f"Sparse matrix created: {sparse_matrix.shape}, density: {sparse_matrix.nnz / np.prod(sparse_matrix.shape):.6f}"
    )

    mol_df_clean = pd.DataFrame({merge_col: all_compounds})

    logger.info(f"Final molecules DataFrame: {mol_df_clean.shape}")
    logger.info(f"Final sparse task matrix: {sparse_matrix.shape}")

    return (
        mol_df_clean,
        sparse_matrix,
        task_cols_final,
        column_to_task_type,
        task_to_file,
        file_to_tasks,
    )


def create_validation_set_sparse(
    mol_df: pd.DataFrame,
    sparse_matrix: csr_matrix,
    task_cols: List[str],
    min_compounds: int,
    sampling_rate: float,
    seed: int,
    logger: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame, csr_matrix, csr_matrix]:
    """Create validation split for sparse data."""

    rng = np.random.default_rng(seed)
    n_compounds = len(mol_df)

    logger.info(f"Creating validation set for {len(task_cols)} tasks")

    val_mask = np.zeros(n_compounds, dtype=bool)

    for task_idx in range(len(task_cols)):
        # Get compounds with data for this task
        compound_indices = sparse_matrix[:, task_idx].nonzero()[0]

        if len(compound_indices) > 0:
            n_samples = min(
                max(min_compounds, int(len(compound_indices) * sampling_rate)),
                len(compound_indices),
            )
            chosen = rng.choice(compound_indices, size=n_samples, replace=False)
            val_mask[chosen] = True

    # Split data
    train_mol_df = mol_df.iloc[~val_mask].reset_index(drop=True)
    val_mol_df = mol_df.iloc[val_mask].reset_index(drop=True)

    train_sparse = sparse_matrix[~val_mask]
    val_sparse = sparse_matrix[val_mask]

    logger.info(
        f"Train: {len(train_mol_df)} compounds, Val: {len(val_mol_df)} compounds"
    )

    return train_mol_df, val_mol_df, train_sparse, val_sparse


def generate_datacard_sparse(
    train_mol_df: pd.DataFrame,
    val_mol_df: pd.DataFrame,
    train_sparse: csr_matrix,
    val_sparse: csr_matrix,
    task_cols: List[str],
    column_to_task_type: Dict[str, str],
    files: List[str],
    file_to_tasks: Dict[str, List[int]],
    logger: Any,
) -> Dict[str, Any]:
    """Generate datacard for sparse format data."""

    # Average number of non-null values per task
    values_per_task = [train_sparse[:, i].nnz for i in range(len(task_cols))]
    avg_values_per_task = float(np.mean(values_per_task)) if values_per_task else 0.0

    # Average number of measurements per compound
    measurements_per_compound = np.array(train_sparse.sum(axis=1)).flatten()
    avg_measurements_per_compound = float(np.mean(measurements_per_compound))

    # Create detailed file-to-task mapping
    tasks_per_file = {}
    for filename, task_indices in file_to_tasks.items():
        task_types = [
            column_to_task_type.get(task_cols[idx], "unknown") for idx in task_indices
        ]

        tasks_per_file[filename] = {
            "count": len(task_indices),
            "task_types": task_types[0],
        }

    datacard = {
        "total_tasks": len(task_cols),
        "tasks_per_file": tasks_per_file,
        "compounds_in_train": len(train_mol_df),
        "compounds_in_val": len(val_mol_df),
        "avg_values_per_task": avg_values_per_task,
        "avg_measurements_per_compound": avg_measurements_per_compound,
        "sparse_matrix_density": train_sparse.nnz / np.prod(train_sparse.shape),
    }
    return datacard


def main(cfg=None) -> None:
    """Main function to process datasets according to config.

    Branches on ``cfg.datasets.sparse`` to emit either sparse ``.npz`` CSR
    label artifacts or dense ``.npy`` label arrays. ``task_metadata.json``
    records the choice under ``storage_mode`` so the pretraining command can
    dispatch without a user-visible flag.
    """

    if cfg is None:
        parser = argparse.ArgumentParser(description="Prepare datasets for pretraining")
        parser.add_argument(
            "--config", type=str, required=True, help="Path to the YAML config file"
        )
        args = parser.parse_args()
        with open(args.config, "r") as f:
            raw = yaml.safe_load(f)
        cfg = CLIPrepareInputModel.model_validate(raw)
    elif not isinstance(cfg, CLIPrepareInputModel):
        cfg = CLIPrepareInputModel.model_validate(cfg)
    log_path = f"{cfg.output}/prepare.log"
    logger = get_default_logger("PREPARE", logging_path=log_path)

    # Create output directory if it doesn't exist
    os.makedirs(cfg.output, exist_ok=True)

    # Extract task types from config
    task_types = [x for x in cfg.datasets.task_type]
    storage_mode = "sparse" if cfg.datasets.sparse else "dense"
    logger.info(f"Preparing dataset in {storage_mode} mode")

    if cfg.datasets.sparse:
        (
            mol_df,
            sparse_matrix,
            task_cols,
            column_to_task_type,
            task_to_file,
            file_to_tasks,
        ) = merge_datasets_streaming_sparse(
            cfg.datasets.files,
            cfg.metadata.merge_col,
            task_types,
            cfg.metadata.tag_to_add,
            logger,
        )

        (
            train_mol_df,
            val_mol_df,
            train_sparse,
            val_sparse,
        ) = create_validation_set_sparse(
            mol_df,
            sparse_matrix,
            task_cols,
            cfg.validation.min_compounds,
            cfg.validation.sampling_rate,
            cfg.validation.seed,
            logger,
        )

        # Apply standard scaling to regression tasks
        logger.info("Applying standard scaling to regression tasks...")
        regression_task_indices = [
            i
            for i, col in enumerate(task_cols)
            if column_to_task_type.get(col) == "regression"
        ]

        scaling_stats: Dict[int, Dict[str, float]] = {}
        if regression_task_indices:
            logger.info(
                f"Computing scaling statistics for {len(regression_task_indices)} regression tasks"
            )
            scaling_stats = compute_sparse_scaling_stats(
                train_sparse, regression_task_indices, logger
            )
            train_sparse = apply_sparse_scaling(train_sparse, scaling_stats, logger)
            val_sparse = apply_sparse_scaling(val_sparse, scaling_stats, logger)
        else:
            logger.info("No regression tasks found, skipping scaling")

        train_mol_path = os.path.join(cfg.output, "train_molecules.parquet")
        val_mol_path = os.path.join(cfg.output, "val_molecules.parquet")
        train_tasks_path = os.path.join(cfg.output, "train_tasks_sparse.npz")
        val_tasks_path = os.path.join(cfg.output, "val_tasks_sparse.npz")

        logger.info("Saving outputs...")
        train_mol_df.to_parquet(train_mol_path, index=False)
        val_mol_df.to_parquet(val_mol_path, index=False)

        # Convert sparse matrices to fp32 before saving for memory efficiency
        train_sparse.data = train_sparse.data.astype(np.float32)
        val_sparse.data = val_sparse.data.astype(np.float32)

        sp.save_npz(train_tasks_path, train_sparse)
        sp.save_npz(val_tasks_path, val_sparse)

        datacard = generate_datacard_sparse(
            train_mol_df,
            val_mol_df,
            train_sparse,
            val_sparse,
            task_cols,
            column_to_task_type,
            cfg.datasets.files,
            file_to_tasks,
            logger,
        )
    else:
        (
            mol_df,
            dense_matrix,
            task_cols,
            column_to_task_type,
            task_to_file,
            file_to_tasks,
        ) = merge_datasets_streaming_dense(
            cfg.datasets.files,
            cfg.metadata.merge_col,
            task_types,
            cfg.metadata.tag_to_add,
            logger,
        )

        (
            train_mol_df,
            val_mol_df,
            train_dense,
            val_dense,
        ) = create_validation_set_dense(
            mol_df,
            dense_matrix,
            task_cols,
            cfg.validation.min_compounds,
            cfg.validation.sampling_rate,
            cfg.validation.seed,
            logger,
        )

        logger.info("Applying standard scaling to regression tasks...")
        regression_task_indices = [
            i
            for i, col in enumerate(task_cols)
            if column_to_task_type.get(col) == "regression"
        ]

        scaling_stats = {}
        if regression_task_indices:
            logger.info(
                f"Computing scaling statistics for {len(regression_task_indices)} regression tasks"
            )
            scaling_stats = compute_dense_scaling_stats(
                train_dense, regression_task_indices, logger
            )
            train_dense = apply_dense_scaling(train_dense, scaling_stats, logger)
            val_dense = apply_dense_scaling(val_dense, scaling_stats, logger)
        else:
            logger.info("No regression tasks found, skipping scaling")

        train_mol_path = os.path.join(cfg.output, "train_molecules.parquet")
        val_mol_path = os.path.join(cfg.output, "val_molecules.parquet")
        train_tasks_path = os.path.join(cfg.output, "train_tasks_dense.npy")
        val_tasks_path = os.path.join(cfg.output, "val_tasks_dense.npy")

        logger.info("Saving outputs...")
        train_mol_df.to_parquet(train_mol_path, index=False)
        val_mol_df.to_parquet(val_mol_path, index=False)

        train_dense = train_dense.astype(np.float32, copy=False)
        val_dense = val_dense.astype(np.float32, copy=False)

        np.save(train_tasks_path, train_dense, allow_pickle=False)
        np.save(val_tasks_path, val_dense, allow_pickle=False)

        datacard = generate_datacard_dense(
            train_mol_df,
            val_mol_df,
            train_dense,
            val_dense,
            task_cols,
            column_to_task_type,
            cfg.datasets.files,
            file_to_tasks,
            logger,
        )

    # Create task name to index mapping for convenience
    task_name_to_index = {task_name: idx for idx, task_name in enumerate(task_cols)}

    # Remove file extensions from task_to_file and file_to_tasks mappings
    task_to_file_clean = {
        task: os.path.splitext(filename)[0] for task, filename in task_to_file.items()
    }
    file_to_tasks_clean = {
        os.path.splitext(filename)[0]: tasks
        for filename, tasks in file_to_tasks.items()
    }

    metadata = {
        "task_columns": task_cols,
        "column_to_task_type": column_to_task_type,
        "task_to_file": task_to_file_clean,
        "file_to_tasks": file_to_tasks_clean,
        "task_name_to_index": task_name_to_index,
        "merge_col": cfg.metadata.merge_col,
        "scaling_stats": scaling_stats,
        "storage_mode": storage_mode,
    }
    save_json(os.path.join(cfg.output, "task_metadata.json"), metadata)

    save_json(os.path.join(cfg.output, "datacard.json"), datacard)
    save_config_as_yaml(cfg, f"{cfg.output}/cfg.yaml")

    logger.info(f"{storage_mode.capitalize()} data preparation completed successfully")
    logger.info(f"Saved {len(task_cols)} tasks")


if __name__ == "__main__":
    main()
