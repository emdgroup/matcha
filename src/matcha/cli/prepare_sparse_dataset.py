"""CLI command for preparing sparse multi-task datasets for pretraining.

Merges multiple parquet files into a single sparse matrix (compounds x tasks),
creates train/validation splits, applies standard scaling to regression tasks,
and saves the result in an efficient npz + parquet format consumed by the
``pretrain_multitask`` command.
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
    """Main function to process datasets according to config."""

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

    # Build sparse matrix sequentially
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

    # Create validation set with sparse data
    train_mol_df, val_mol_df, train_sparse, val_sparse = create_validation_set_sparse(
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

    # Identify regression task indices
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

        # Compute scaling statistics from training set
        scaling_stats = compute_sparse_scaling_stats(
            train_sparse, regression_task_indices, logger
        )

        # Apply scaling to both train and validation sets
        train_sparse = apply_sparse_scaling(train_sparse, scaling_stats, logger)
        val_sparse = apply_sparse_scaling(val_sparse, scaling_stats, logger)
    else:
        logger.info("No regression tasks found, skipping scaling")

    # Save outputs in sparse format
    train_mol_path = os.path.join(cfg.output, "train_molecules.parquet")
    val_mol_path = os.path.join(cfg.output, "val_molecules.parquet")
    train_sparse_path = os.path.join(cfg.output, "train_tasks.npz")
    val_sparse_path = os.path.join(cfg.output, "val_tasks.npz")

    logger.info("Saving outputs...")
    train_mol_df.to_parquet(train_mol_path, index=False)
    val_mol_df.to_parquet(val_mol_path, index=False)

    # Convert sparse matrices to fp32 before saving for memory efficiency
    train_sparse.data = train_sparse.data.astype(np.float32)
    val_sparse.data = val_sparse.data.astype(np.float32)

    sp.save_npz(train_sparse_path, train_sparse)
    sp.save_npz(val_sparse_path, val_sparse)

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

    # Save metadata
    metadata = {
        "task_columns": task_cols,
        "column_to_task_type": column_to_task_type,
        "task_to_file": task_to_file_clean,
        "file_to_tasks": file_to_tasks_clean,
        "task_name_to_index": task_name_to_index,
        "merge_col": cfg.metadata.merge_col,
        "scaling_stats": scaling_stats,
    }
    save_json(os.path.join(cfg.output, "task_metadata.json"), metadata)

    # Generate datacard
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

    save_json(os.path.join(cfg.output, "datacard.json"), datacard)
    save_config_as_yaml(cfg, f"{cfg.output}/cfg.yaml")

    logger.info("Sparse data preparation completed successfully")
    logger.info(f"Saved {len(task_cols)} tasks")


if __name__ == "__main__":
    main()
