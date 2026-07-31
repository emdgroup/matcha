"""Multitask utilities: task affinity computation (TAG) and dataset stitching."""

import copy
from dataclasses import dataclass
from math import ceil
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from matcha.sklearn.base_sklearn_model import BaseScikitLearnModel
from matcha.nn.optimizers import OptimizerRegistry
from matcha.nn.schedulers import SchedulerRegistry


@dataclass
class TaskAffinityResult:
    """Container for Task Affinity Grouping (TAG) results.

    Based on: https://arxiv.org/pdf/2109.04617

    :param affinity_matrix: ``(num_tasks, num_tasks)`` matrix where entry ``[i, j]``
        represents the affinity of task *i* to task *j*.
    :type affinity_matrix: numpy.ndarray
    :param task_names: Task names corresponding to matrix indices.
    :type task_names: list[str]
    :param int num_epochs: Number of epochs used to compute the affinity.
    """

    affinity_matrix: np.ndarray
    task_names: List[str]
    num_epochs: int

    def plot_affinity_matrix(
        self,
        title: str = "Task Affinity Matrix",
        width: int = 700,
        height: int = 600,
        show_values: bool = True,
    ):
        """Create an interactive heatmap of the task affinity matrix using Plotly.

        :param str title: Title of the plot.
        :param int width: Width of the figure in pixels.
        :param int height: Height of the figure in pixels.
        :param bool show_values: Whether to annotate cells with affinity values.
        :returns: Interactive heatmap figure.
        :rtype: plotly.graph_objects.Figure
        """
        import plotly.graph_objects as go

        # Create text annotations if requested
        text = None
        if show_values:
            text = np.round(self.affinity_matrix, 3).astype(str)

        # Custom colorscale: low affinity (#2dbecd) to high affinity (#149b5f)
        colorscale = [
            [0.0, "#2dbecd"],  # Low affinity - cyan/teal
            [0.5, "#FFFFFF"],  # Neutral - white
            [1.0, "#149b5f"],  # High affinity - green
        ]

        fig = go.Figure(
            data=go.Heatmap(
                z=self.affinity_matrix,
                x=self.task_names,
                y=self.task_names,
                colorscale=colorscale,
                zmid=0,
                text=text,
                texttemplate="%{text}" if show_values else None,
                textfont={"size": 10},
                hovertemplate=(
                    "Source Task: %{y}<br>"
                    "Target Task: %{x}<br>"
                    "Affinity: %{z:.4f}<extra></extra>"
                ),
            )
        )

        fig.update_layout(
            title={
                "text": title,
                "x": 0.5,
                "xanchor": "center",
            },
            xaxis_title="Target Task",
            yaxis_title="Source Task",
            width=width,
            height=height,
            yaxis=dict(autorange="reversed"),  # To match matrix convention
        )

        return fig

    def get_top_k(
        self,
        source_task: int,
        k: int,
        include_self: bool = False,
    ) -> List[int]:
        """Get the top-k most affine tasks for a given source task.

        :param int source_task: Index of the source task.
        :param int k: Number of top affine tasks to return.
        :param bool include_self: Whether to include the source task in results.
        :returns: Task indices sorted by affinity (highest first).
        :rtype: list[int]
        :raises ValueError: If ``source_task`` is out of bounds or ``k`` is invalid.
        """
        num_tasks = self.affinity_matrix.shape[0]

        if source_task < 0 or source_task >= num_tasks:
            raise ValueError(
                f"source_task must be between 0 and {num_tasks - 1}, got {source_task}"
            )

        max_k = num_tasks if include_self else num_tasks - 1
        if k < 1 or k > max_k:
            raise ValueError(f"k must be between 1 and {max_k}, got {k}")

        # Get affinity scores for the source task
        affinities = self.affinity_matrix[source_task, :]

        # Sort indices by affinity (descending)
        sorted_indices = np.argsort(affinities)[::-1]

        # Filter out self if requested
        if not include_self:
            sorted_indices = sorted_indices[sorted_indices != source_task]

        return sorted_indices[:k].tolist()


def compute_task_affinity(
    sklearn_model: BaseScikitLearnModel,
    molecules: list,
    labels: np.ndarray,
    affinity_every_n_steps: int = 10,
    device: Optional[str] = None,
) -> TaskAffinityResult:
    """Compute task affinity matrix using the Task Affinity Grouping (TAG) algorithm.

    Trains a single multitask model, periodically measuring how each task's
    gradient step affects all other tasks' losses. The collected affinity
    snapshots are averaged to produce the final matrix.

    The affinity ``Z[i, j]`` measures how much a gradient update on task *i*
    affects task *j*'s loss. A positive affinity indicates that training on
    task *i* reduces the loss of task *j*.

    Based on: https://arxiv.org/pdf/2109.04617

    :param sklearn_model: A fitted MATCHA sklearn model instance.
    :type sklearn_model: matcha.sklearn.base_sklearn_model.BaseScikitLearnModel
    :param list molecules: Training molecules (RDKit Mol objects or SMILES strings).
    :param numpy.ndarray labels: Training labels of shape ``(n_samples, n_tasks)``.
    :param int affinity_every_n_steps: Collect affinity every N training steps.
    :param device: Device for computation (``'cuda'``, ``'cpu'``, or ``None`` for auto).
    :type device: str or None
    :returns: Object containing the affinity matrix and helper methods.
    :rtype: TaskAffinityResult

    Example::

        from matcha.sklearn.graph import GraphRegressor
        from matcha.nn.multitask import compute_task_affinity

        model = GraphRegressor(num_endpoints=4)
        model.fit(train_mols, train_labels)

        result = compute_task_affinity(model, train_mols, train_labels)
        fig = result.plot_affinity_matrix()
        fig.show()

        # Get top 2 most affine tasks for task 0
        top_tasks = result.get_top_k(source_task=0, k=2)
    """
    # Extract information from the sklearn model
    datamodule = sklearn_model.datamodule
    pytorch_model = sklearn_model.model
    batch_size = sklearn_model.params.training.batch_size
    num_epochs = sklearn_model.params.training.num_epochs
    is_classification = sklearn_model.params.datamodule.is_classification
    logger = sklearn_model.logger

    # Get task names from the label encoder
    if hasattr(datamodule, "_label_encoder") and hasattr(
        datamodule._label_encoder, "label_names"
    ):
        task_names = list(datamodule._label_encoder.label_names)
    else:
        task_names = [f"Task_{i}" for i in range(labels.shape[1])]

    num_tasks = labels.shape[1]

    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create loss function based on task type
    if is_classification:
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    else:
        loss_fn = nn.MSELoss(reduction="none")

    # Transform molecules to dataset format
    train_dataset = sklearn_model.transform(molecules, labels, is_training=False)

    # Create train dataloader
    datamodule.dataset_train = train_dataset
    datamodule.params.batch_size = batch_size
    datamodule.setup(stage="fit")
    train_dataloader = datamodule.train_dataloader()

    # Clone the model for multitask training
    model = copy.deepcopy(pytorch_model)
    model = model.to(device)
    model.train()

    # Use the same optimizer and settings as the original model
    optimizer_name = sklearn_model.params.model.optimizer
    optimizer_args = sklearn_model.params.model.optimizer_args.copy()
    optimizer_cls = OptimizerRegistry.get(optimizer_name)
    optimizer = optimizer_cls(model.parameters(), **optimizer_args)

    # Check if the model has a scheduler configured (Finetuner does not have one)
    # The scheduler field is present in standard models (GIN, SNN, etc.) but not in Finetuner
    model_params = sklearn_model.params.model
    has_scheduler = (
        hasattr(model_params, "scheduler") and model_params.scheduler is not None
    )
    scheduler = None

    # Schedulers that do not accept total_steps (must match training_manager.py blocklist)
    _SCHEDULERS_WITHOUT_TOTAL_STEPS = frozenset({"chemprop"})

    if has_scheduler:
        scheduler_name = model_params.scheduler
        scheduler_args = (
            model_params.scheduler_args.copy()
            if hasattr(model_params, "scheduler_args")
            else {}
        )

        # Auto-compute total_steps as num_epochs * batches_per_epoch when not explicitly set,
        # consistent with the sklearn TrainingManager behavior. The scheduler is stepped per
        # batch (per optimizer step), matching the classic Lightning model interval: "step".
        if scheduler_name not in _SCHEDULERS_WITHOUT_TOTAL_STEPS:
            if "total_steps" not in scheduler_args:
                num_batches_per_epoch = ceil(len(train_dataset) / batch_size)
                scheduler_args["total_steps"] = num_epochs * num_batches_per_epoch
                logger.info(
                    f"TAG: Auto-computed total_steps={scheduler_args['total_steps']} "
                    f"({num_epochs} epochs * {num_batches_per_epoch} batches/epoch)"
                )

        scheduler_cls = SchedulerRegistry.get(scheduler_name)
        scheduler = scheduler_cls(optimizer, **scheduler_args)
        logger.info(
            f"TAG: Using scheduler '{scheduler_name}' with total_steps={scheduler_args.get('total_steps', 'N/A')} "
            f"(stepping per batch)"
        )
    else:
        logger.info("TAG: No scheduler configured, using constant learning rate")

    # Accumulator for affinity matrices collected during training
    affinity_matrices_accumulated = []

    logger.info(f"TAG: Starting task affinity computation with {num_epochs} epochs")
    logger.info(f"TAG: Collecting affinities every {affinity_every_n_steps} steps")

    global_step = 0

    # Train the multitask model for num_epochs
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_dataloader:
            # Get current learning rate (from scheduler if available, otherwise from optimizer)
            if scheduler is not None:
                current_lr = scheduler.get_last_lr()[0]
            else:
                current_lr = optimizer.param_groups[0]["lr"]

            # Move batch to device and keep an unmodified copy for affinity computation
            # (model.forward() may mutate the batch by adding/modifying 'mol_features')
            batch = _move_batch_to_device(batch, device)
            original_batch = _copy_batch(batch)  # Save before forward modifies it
            y = batch["y"]

            optimizer.zero_grad()

            # Forward pass (this may modify batch["mol_features"])
            y_pred = model.forward(batch)

            # Compute multitask loss (over all tasks with valid labels)
            total_loss = torch.tensor(0.0, device=device)
            for task_idx in range(num_tasks):
                mask = ~torch.isnan(y[:, task_idx])
                if mask.sum() == 0:
                    continue
                task_pred = y_pred[mask, task_idx]
                task_target = y[mask, task_idx]
                task_loss = loss_fn(task_pred, task_target).mean()
                total_loss = total_loss + task_loss

            total_loss.backward()
            optimizer.step()

            # Step the scheduler per batch (per optimizer step), consistent with
            # the classic Lightning model behavior (interval: "step")
            if scheduler is not None:
                scheduler.step()

            epoch_loss += total_loss.item()
            num_batches += 1
            global_step += 1

            # Collect affinity every N steps using the ORIGINAL (unmodified) batch
            # This follows TAG paper: use single minibatch gradient, not full-pass
            # Use the current learning rate from the scheduler for affinity computation
            if global_step % affinity_every_n_steps == 0:
                affinity_matrix = _compute_affinity_matrix_single_step(
                    model=model,
                    train_batch=original_batch,
                    num_tasks=num_tasks,
                    loss_fn=loss_fn,
                    lr=current_lr,
                )
                affinity_matrices_accumulated.append(affinity_matrix)

        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        current_lr = (
            scheduler.get_last_lr()[0]
            if scheduler is not None
            else optimizer.param_groups[0]["lr"]
        )
        logger.info(
            f"TAG: Epoch {epoch + 1}/{num_epochs} completed - Avg Loss: {avg_epoch_loss:.4f} - LR: {current_lr:.2e} - Affinities collected: {len(affinity_matrices_accumulated)}"
        )

    # Mean all collected affinity matrices
    final_affinity_matrix = np.mean(affinity_matrices_accumulated, axis=0)

    # Normalize each row by its diagonal element (self-affinity)
    # This makes each task's influence relative to its self-influence
    diagonal = np.diag(final_affinity_matrix)
    safe_diagonal = np.where(np.abs(diagonal) > 1e-8, diagonal, 1.0)
    final_affinity_matrix = final_affinity_matrix / safe_diagonal[:, np.newaxis]

    logger.info(
        f"TAG: Computation complete - Mean {len(affinity_matrices_accumulated)} affinity measurements (row-normalized by diagonal)"
    )

    return TaskAffinityResult(
        affinity_matrix=final_affinity_matrix,
        task_names=task_names,
        num_epochs=num_epochs,
    )


def _compute_affinity_matrix_single_step(
    model: nn.Module,
    train_batch: dict,
    num_tasks: int,
    loss_fn: nn.Module,
    lr: float,
) -> np.ndarray:
    """Compute the task affinity matrix at the current training state.

    For each source task, simulates a single SGD gradient step using only that
    task's loss, then measures the effect on all tasks' losses on the same batch.

    Follows the TAG formulation:
    ``Z[i,j] = (L_j(θ) - L_j(θ - η∇L_i(θ))) / L_j(θ)``

    :param torch.nn.Module model: Current model (parameters are restored after).
    :param dict train_batch: Single batch already on device.
    :param int num_tasks: Number of tasks.
    :param torch.nn.Module loss_fn: Loss function with ``reduction='none'``.
    :param float lr: Learning rate for the simulated gradient step.
    :returns: Affinity matrix of shape ``(num_tasks, num_tasks)``.
    :rtype: numpy.ndarray
    """
    affinity_matrix = np.zeros((num_tasks, num_tasks))

    # Store original parameter values FIRST (stays on same device as model)
    original_params = [p.detach().clone() for p in model.parameters()]

    # Get labels tensor
    y = train_batch["y"]

    # Compute baseline losses L_j(θ) at current parameter point
    # Use eval mode to disable dropout and use running stats for BatchNorm
    model.eval()
    baseline_losses = np.zeros(num_tasks)
    with torch.no_grad():
        batch_for_baseline = _copy_batch(train_batch)
        y_pred_baseline = model.forward(batch_for_baseline)
        for task_idx in range(num_tasks):
            mask = ~torch.isnan(y[:, task_idx])
            if mask.sum() == 0:
                continue
            task_pred = y_pred_baseline[mask, task_idx]
            task_target = y[mask, task_idx]
            baseline_losses[task_idx] = loss_fn(task_pred, task_target).mean().item()

    # For each source task, compute affinity to all target tasks
    for source_task in range(num_tasks):
        mask_source = ~torch.isnan(y[:, source_task])
        if mask_source.sum() == 0:
            continue

        # Restore original parameters before computing gradients
        with torch.no_grad():
            for p, p_orig in zip(model.parameters(), original_params):
                p.copy_(p_orig)

        model.zero_grad()

        # CRITICAL: Use eval mode for gradient computation too, to ensure consistency
        # with how baseline and post-step losses are computed. This prevents BatchNorm
        # and Dropout from causing inconsistent behavior between forward passes.
        model.eval()

        # Enable gradients for the forward pass (eval mode doesn't disable gradients)
        # We need torch.enable_grad() since we're inside no_grad context from earlier
        with torch.enable_grad():
            # Forward pass to compute gradient for source task
            batch_copy = _copy_batch(train_batch)
            y_pred = model.forward(batch_copy)
            task_pred = y_pred[mask_source, source_task]
            task_target = y[mask_source, source_task]

            loss = loss_fn(task_pred, task_target).mean()
            loss.backward()

        # Manual SGD step: θ' = θ - η∇L_i(θ)
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    param.data.sub_(lr * param.grad)

        # Compute losses after simulated step: L_j(θ') for all j
        # Model is already in eval mode from gradient computation
        post_step_losses = np.zeros(num_tasks)
        with torch.no_grad():
            batch_for_eval = _copy_batch(train_batch)
            y_pred_post = model.forward(batch_for_eval)
            for task_idx in range(num_tasks):
                mask = ~torch.isnan(y[:, task_idx])
                if mask.sum() == 0:
                    continue
                task_pred = y_pred_post[mask, task_idx]
                task_target = y[mask, task_idx]
                post_step_losses[task_idx] = (
                    loss_fn(task_pred, task_target).mean().item()
                )

        # Compute affinity using TAG formula:
        # Z[i,j] = (L_j(θ) - L_j(θ')) / L_j(θ)
        for target_task in range(num_tasks):
            if baseline_losses[target_task] > 1e-8:
                affinity = (
                    baseline_losses[target_task] - post_step_losses[target_task]
                ) / baseline_losses[target_task]
                affinity_matrix[source_task, target_task] = affinity

    # Restore original parameters and model state after all tasks are done
    with torch.no_grad():
        for p, p_orig in zip(model.parameters(), original_params):
            p.copy_(p_orig)
    model.train()

    return affinity_matrix


def _move_batch_to_device(batch: dict, device: str) -> dict:
    """Move all tensors in a batch dictionary to the specified device.

    Handles both plain :class:`torch.Tensor` values and objects with a ``.to()``
    method (e.g., PyG Batch). Creates a shallow copy of the dictionary.

    :param dict batch: Batch dictionary.
    :param str device: Target device string.
    :returns: New dictionary with values transferred to *device*.
    :rtype: dict
    """
    moved_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved_batch[key] = value.to(device)
        elif hasattr(value, "to") and callable(value.to):
            # Handle PyTorch Geometric Data/Batch objects and similar
            moved_batch[key] = value.to(device)
        else:
            moved_batch[key] = value
    return moved_batch


def _copy_batch(batch: dict) -> dict:
    """Create a deep copy of a batch dictionary.

    Needed because ``model.forward()`` may mutate the batch (e.g., adding
    or modifying ``'mol_features'``).

    :param dict batch: Batch dictionary to copy.
    :returns: Deep-copied batch.
    :rtype: dict
    """
    copied_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            # Clone tensors to avoid shared state between forward passes
            copied_batch[key] = value.clone()
        elif hasattr(value, "clone") and callable(value.clone):
            # Clone PyTorch Geometric Data/Batch objects
            copied_batch[key] = value.clone()
        else:
            # For non-tensor values, use copy.deepcopy as fallback
            copied_batch[key] = copy.deepcopy(value)
    return copied_batch


def stitch_datasets(
    df_list: list[pd.DataFrame],
    property_list: list[str],
    smiles_key: str = "SMILES",
    bound_key: str | None = "OPERATOR",
    index_key: str | None = None,
    tag: str = "endpoint",
) -> pd.DataFrame:
    """Stitch multiple single-endpoint DataFrames into one multitask DataFrame.

    Pivots each DataFrame's property column into a wide-format table keyed by
    SMILES, with NaN for missing task labels.

    :param list[pd.DataFrame] df_list: DataFrames to stitch.
    :param list[str] property_list: Property column name for each DataFrame.
    :param str smiles_key: Column name for SMILES identifiers.
    :param bound_key: Column name for bound/operator information, or ``None``.
    :type bound_key: str or None
    :param index_key: Optional column to preserve as a shared index.
    :type index_key: str or None
    :param str tag: Suffix appended to property column names in the output.
    :returns: Wide-format DataFrame with one row per unique SMILES.
    :rtype: pandas.DataFrame
    """

    if not df_list:
        return tuple()

    prepared_dfs = []
    for i, df in enumerate(df_list):
        df_prep = df.copy()

        df_prep["dataset_id"] = i
        df_prep["property_name"] = property_list[i]

        essential_cols = [smiles_key, property_list[i], "dataset_id", "property_name"]
        if bound_key in df.columns:
            essential_cols.append(bound_key)
        if index_key is not None:
            essential_cols.append(index_key)

        available_cols = [col for col in essential_cols if col in df_prep.columns]
        df_prep = df_prep[available_cols]

        prepared_dfs.append(df_prep)

    df_combined = pd.concat(prepared_dfs, ignore_index=True)

    unique_properties = df_combined["property_name"].unique()
    if len(unique_properties) == 1:
        pivot_values = unique_properties[0]
    else:
        df_combined["property_value"] = np.nan
        for prop in unique_properties:
            mask = df_combined["property_name"] == prop
            if prop in df_combined.columns:
                df_combined.loc[mask, "property_value"] = df_combined.loc[mask, prop]
        pivot_values = "property_value"

    df_final = df_combined.pivot_table(
        index=smiles_key, columns="dataset_id", values=pivot_values, aggfunc="first"
    ).reset_index()

    if hasattr(df_final.columns, "levels"):
        df_final.columns = [
            f"{property_list[col]}_{col}"
            if col != "" and col != smiles_key
            else smiles_key
            if col == smiles_key or str(col) == smiles_key
            else str(col)
            for col in df_final.columns.get_level_values(-1)
        ]
    else:
        rename_dict = {}
        for col in df_final.columns:
            if col != smiles_key and str(col).isdigit():
                idx = int(col)
                rename_dict[col] = f"{property_list[idx]}_{idx}"
        df_final = df_final.rename(columns=rename_dict)

    bound_key_list = []
    if bound_key is not None:
        bound_df = df_combined.pivot_table(
            index=smiles_key, columns="dataset_id", values=bound_key, aggfunc="first"
        ).reset_index()

        bound_rename_dict = {}
        for col in bound_df.columns:
            if col != smiles_key and str(col).isdigit():
                idx = int(col)
                bound_rename_dict[col] = f"{bound_key}_{idx}"
                bound_key_list.append(f"{bound_key}_{idx}")

        bound_df = bound_df.rename(columns=bound_rename_dict)
        df_final = pd.merge(df_final, bound_df, on=smiles_key, how="left")

    if index_key is not None:
        idx_df = df_combined.pivot_table(
            index=smiles_key, columns="dataset_id", values=index_key, aggfunc="first"
        ).reset_index()
        idx_columns = [
            col for col in idx_df.columns if col != smiles_key and str(col).isdigit()
        ]
        idx_df[index_key] = idx_df[idx_columns].bfill(axis=1).iloc[:, 0]
        idx_df = idx_df[[smiles_key, index_key]]
        df_final = pd.merge(df_final, idx_df, on=smiles_key, how="left")

    df_final = df_final.drop(
        columns=[col for col in df_final.columns if "ROMol_" in col]
    )

    prop_key_list = [f"{property_list[i]}_{i}" for i in range(len(property_list))]

    property_cols_in_df = [col for col in prop_key_list if col in df_final.columns]
    df_final = df_final.dropna(subset=property_cols_in_df, how="all")

    for col in prop_key_list:
        if col in df_final.columns:
            df_final.rename(columns={col: f"{col}_{tag}"}, inplace=True)
    return df_final
