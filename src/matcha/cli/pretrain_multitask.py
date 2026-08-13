"""CLI command for multi-task pretraining of neural network models.

Trains a MATCHA neural-network model (e.g. GatedGCN) on a sparse
multi-task dataset produced by the ``prepare_sparse_dataset`` command.
Supports multiple loss functions with independent weighting schedules,
distributed training via DDP, and produces serialized artifacts
compatible with downstream finetuning.
"""

import os
import datetime
import argparse
import json

import yaml
import pandas as pd
import numpy as np
import scipy.sparse as sp
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from rdkit import Chem

from matcha.sklearn.clm.base_sklearn_clm import BaseScikitLearnCLM
from matcha.utils.logging import get_default_logger, MatchaLogger
from matcha.utils.serialization import save_pickle, save_yaml, load_json
from matcha.sklearn.base_sklearn_model import ScikitLearnModelRegistry
from matcha.utils.schemas.cli import (
    CLIPretrainMultitaskInputModel,
    PretrainTraining,
)
from matcha.cli.utils import _load_coords_npz, save_config_as_yaml
from matcha.datamodules.pretraining.on_the_fly_datamodule import OnTheFlyDataModule
from matcha import __version__

torch.set_float32_matmul_precision("high")


# ──────────────────────────────────────────────────────────────────────
# Save artifacts
# ──────────────────────────────────────────────────────────────────────


def _save_artifacts(
    cfg: CLIPretrainMultitaskInputModel,
    model,
    trainer: L.Trainer,
    base_dm,
    params,
    logger,
):
    """Save pretraining artifacts (checkpoint, configs, manifest) to disk.

    Produces the standard artifact layout expected by the MATCHA finetuning
    pipeline: a full Lightning checkpoint, datamodule state, config YAMLs,
    metadata, and a manifest file indicating ``origin_type: "pretraining"``.

    :param cfg: The full CLI configuration.
    :param model: Trained Lightning model.
    :param trainer: Lightning trainer instance.
    :param base_dm: Base datamodule (for state serialization).
    :param params: Container params holding model and datamodule settings.
    :param logger: Logger instance.
    """
    out_dir = cfg.output.serialization
    config_dir = os.path.join(out_dir, "config")
    state_dir = os.path.join(out_dir, "state")
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(state_dir, exist_ok=True)

    # Full checkpoint
    ckpt_path = os.path.join(out_dir, "model.ckpt")
    trainer.save_checkpoint(ckpt_path)
    logger.info(f"Full checkpoint saved: {ckpt_path}")

    # Datamodule state
    dm_state = base_dm.state_dict()
    save_pickle(os.path.join(state_dir, "datamodule_state.pkl"), dm_state)

    # Config YAMLs
    save_yaml(os.path.join(config_dir, "model.yaml"), params.model.model_dump())

    training_cfg = cfg.model.training or PretrainTraining()
    training_yaml = {
        "num_epochs": training_cfg.num_epochs,
        "batch_size": params.training.batch_size,
        "accelerator": str(trainer.accelerator.__class__.__name__)
        .lower()
        .replace("accelerator", ""),
        "devices": trainer.num_devices,
        "early_stopping": any(
            isinstance(cb, EarlyStopping) for cb in (trainer.callbacks or [])
        ),
        "patience": next(
            (
                cb.patience
                for cb in (trainer.callbacks or [])
                if isinstance(cb, EarlyStopping)
            ),
            training_cfg.patience,
        ),
        "stochastic_weight_averaging": False,
        "seed": training_cfg.seed,
    }
    save_yaml(os.path.join(config_dir, "training.yaml"), training_yaml)
    save_yaml(
        os.path.join(config_dir, "datamodule.yaml"), params.datamodule.model_dump()
    )

    metadata = {
        "model_type": cfg.model.architecture,
        "model_name": "multitask pretraining",
        "model_version": 1,
        "model_scope": "pretraining",
        "model_owner": "matcha",
        "matcha_version": __version__,
        "date": datetime.datetime.now().isoformat(),
        "description": f"Multitask pretrained model ({cfg.model.architecture})",
        "extra": {},
    }
    save_yaml(os.path.join(config_dir, "metadata.yaml"), metadata)

    # Manifest — this is what the Finetuner reads
    save_yaml(
        os.path.join(config_dir, "manifest.yaml"),
        {
            "matcha_version": __version__,
            "serialization_version": 2,
            "origin_type": "classic",
            "source_class": cfg.model.architecture,
            "task_type": "multitask",
            "saved_at": datetime.datetime.now().isoformat(),
        },
    )

    # Copy of input config
    save_config_as_yaml(cfg, os.path.join(out_dir, "cfg.yaml"))

    logger.info(f"All artifacts saved to {out_dir}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def main(cfg=None) -> None:
    """Run multi-task pretraining from a YAML configuration.

    Loads sparse training/validation data, builds a neural-network model via
    the scikit-learn registry interface, wraps the datamodule in an on-the-fly
    featurizer for memory efficiency, trains with Lightning (optionally
    distributed via DDP), and saves artifacts for downstream finetuning.

    :param cfg: Pre-parsed configuration object or ``None`` to parse from
        CLI ``--config`` argument. Accepts a
        :class:`~matcha.utils.schemas.cli.CLIPretrainMultitaskInputModel`
        instance or a raw dict that will be validated.
    """

    if cfg is None:
        parser = argparse.ArgumentParser(description="Pretrain a model")
        parser.add_argument(
            "--config", type=str, required=True, help="Path to the YAML config file"
        )
        args = parser.parse_args()
        with open(args.config, "r") as f:
            raw = yaml.safe_load(f)
        cfg = CLIPretrainMultitaskInputModel.model_validate(raw)
    elif not isinstance(cfg, CLIPretrainMultitaskInputModel):
        cfg = CLIPretrainMultitaskInputModel.model_validate(cfg)

    os.makedirs(cfg.output.serialization, exist_ok=True)
    log_path = os.path.join(cfg.output.serialization, "pretrain.log")
    logger = get_default_logger("PRETRAIN", logging_path=log_path)

    # Environment (optional)
    if cfg.pipe.visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.pipe.visible_devices)
        logger.info(f"Set CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    # Resolve training config (use typed defaults when omitted)
    training_cfg = cfg.model.training or PretrainTraining()
    seed = training_cfg.seed
    L.seed_everything(seed, workers=True, verbose=False)

    # Resolve dataset paths: use override if set, else construct from dataset_dir
    ds = cfg.dataset
    train_smiles_path = ds.train_smiles or os.path.join(
        ds.dataset_dir, "train_molecules.parquet"
    )
    val_smiles_path = ds.val_smiles or os.path.join(
        ds.dataset_dir, "val_molecules.parquet"
    )
    train_tasks_path = ds.train_tasks or os.path.join(ds.dataset_dir, "train_tasks.npz")
    val_tasks_path = ds.val_tasks or os.path.join(ds.dataset_dir, "val_tasks.npz")
    task_metadata_path = ds.task_metadata or os.path.join(
        ds.dataset_dir, "task_metadata.json"
    )

    # Auto-discover per-molecule 3D coords packed as flat + offsets under the
    # dataset directory. When present, coords are threaded to the on-the-fly
    # wrapper, which forwards them to base.generate_features(..., coords=...)
    # if the base datamodule accepts a coords kwarg. Otherwise they are
    # silently dropped with a one-shot logger warning (see OnTheFlyDataModule).
    train_coords_path = os.path.join(ds.dataset_dir, "train_coords.npz")
    val_coords_path = os.path.join(ds.dataset_dir, "val_coords.npz")
    train_coords = (
        _load_coords_npz(train_coords_path)
        if os.path.isfile(train_coords_path)
        else None
    )
    val_coords = (
        _load_coords_npz(val_coords_path) if os.path.isfile(val_coords_path) else None
    )

    # load data
    logger.info(msg="Loading train data")
    train_mols = pd.read_parquet(train_smiles_path).SMILES.tolist()
    train_y = sp.load_npz(train_tasks_path)
    logger.info(f"Training set size: {len(train_mols)}")
    if train_coords is not None:
        logger.info(f"Loaded train coords: {train_coords_path}")

    logger.info(msg="Loading validation data")
    val_mols = pd.read_parquet(val_smiles_path).SMILES.tolist()
    val_y = sp.load_npz(val_tasks_path)
    logger.info(f"Validation set size: {len(val_mols)}")
    if val_coords is not None:
        logger.info(f"Loaded val coords: {val_coords_path}")

    metadata = load_json(task_metadata_path)

    logger.info(msg=f"Found {train_y.shape[1]} tasks for training")
    logger.info(msg=f"Found {val_y.shape[1]} tasks for validation")

    # parse cfg for losses
    loss_box = []
    for loss in cfg.loss:
        loss_cfg = {
            "task_map": metadata["file_to_tasks"][loss.dataset],
            "loss_fn": loss.loss_fn,
            "loss_args": loss.loss_args,
            "init_w": loss.init_w,
            "final_w": loss.final_w,
            "T": loss.T,
            "warmup": loss.warmup,
            "name": loss.dataset,
        }
        loss_box.append(loss_cfg)

    # ── Build model via scikit-learn interface ─────────────────────────
    logger.info(msg="Creating model and datamodule instance")

    if cfg.pipe.visible_devices:
        devices = len(cfg.pipe.visible_devices.split(","))
    else:
        devices = training_cfg.devices

    if "Model" in cfg.model.architecture:
        architecture = cfg.model.architecture.replace("Model", "Regressor")
    else:
        architecture = cfg.model.architecture

    # Build the flat kwargs expected by the ScikitLearnModelRegistry.
    # Neural-network params come from cfg.model.params; training and
    # datamodule overrides are merged in from their own config sections
    # so that the container sees the full picture.
    container_kwargs = dict(cfg.model.params)
    container_kwargs.update(
        {
            "num_epochs": training_cfg.num_epochs,
            "early_stopping": training_cfg.early_stopping,
            "patience": training_cfg.patience,
            "accelerator": training_cfg.accelerator,
            "seed": training_cfg.seed,
        }
    )
    if cfg.model.datamodule:
        container_kwargs.update(cfg.model.datamodule)

    container = ScikitLearnModelRegistry[architecture](
        num_endpoints=train_y.shape[1],
        devices=devices,
        loss_fn="multiloss",
        loss_args={"loss_configs": loss_box},
        **container_kwargs,
    )

    model = container.model
    datamodule = container.datamodule
    params = container.params

    logger.info(
        f"Datamodule params: {json.dumps(datamodule.params.model_dump(), indent=2)}"
    )
    logger.info(f"Model params: {json.dumps(model.params.model_dump(), indent=2)}")

    # fit datamodules on a sample of the training set
    if cfg.pipe.fit_datamodule_size is not None:
        logger.info("Fitting base datamodule stateful components")
        sample_size = cfg.pipe.fit_datamodule_size
        sample_indices = np.random.choice(len(train_mols), sample_size, replace=False)
        sample_smiles = [train_mols[i] for i in sample_indices]
        sample_mols = [Chem.MolFromSmiles(x) for x in sample_smiles]
        sample_y = train_y[sample_indices]
        train_set = datamodule.generate_features(sample_mols, sample_y.toarray())
        datamodule.fit(train_set)

    # have to adjust num_tokens and remake model accordingly
    if isinstance(container, BaseScikitLearnCLM):
        model_params = params.model.model_dump()
        model_params["enc_num_characters"] = datamodule.params.num_tokens
        model_params.pop("torch_type")
        model = container._architecture(**model_params)

    del container

    # ── On-the-fly datamodule wrapper ─────────────────────────────────
    logger.info("Creating OnTheFly wrapper for delayed feature generation")
    on_the_fly_datamodule = OnTheFlyDataModule(
        base=datamodule, num_workers=cfg.pipe.dataloader_num_workers
    )

    # Set raw data instead of pre-computing features. Coords are passed
    # unconditionally — the wrapper handles ``None`` and the signature probe
    # inside collate_fn decides whether to forward them to the base.
    on_the_fly_datamodule.set_data(
        train_smiles=train_mols,
        train_y=train_y,
        val_smiles=val_mols,
        val_y=val_y,
        train_coords=train_coords,
        val_coords=val_coords,
    )

    # Use the OnTheFly datamodule for training
    base_dm = datamodule  # keep reference for artifact saving
    datamodule = on_the_fly_datamodule
    datamodule.setup(stage="fit")
    datamodule.params.batch_size = int(params.training.batch_size)

    # ── Trainer ───────────────────────────────────────────────────────
    logger.info("Setting up Trainer...")
    num_epochs = training_cfg.num_epochs
    accelerator = training_cfg.accelerator
    patience = training_cfg.patience
    early_stopping = training_cfg.early_stopping

    strategy = "auto"
    if accelerator == "gpu" and devices > 1:
        logger.info(f"Using DDP strategy for distributed training on {devices} GPUs")
        strategy = DDPStrategy(
            timeout=datetime.timedelta(
                seconds=int(cfg.pipe.strategy.get("timeout_seconds", 3600))
            ),
            find_unused_parameters=bool(
                cfg.pipe.strategy.get("find_unused_parameters", False)
            ),
        )

    callbacks = []
    if early_stopping:
        callbacks.append(
            EarlyStopping(monitor="val_loss", mode="min", patience=patience)
        )
        callbacks.append(ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min"))

    mlflow_logger = True
    if cfg.output.mlflow is not None:
        logger.info(
            f"Setting up MLFlow logging | experiment: '{cfg.output.mlflow.experiment_name}' "
            f"| log_dir: '{cfg.output.mlflow.log_dir}'"
        )

        mlflow_logger = MatchaLogger(
            experiment_name=cfg.output.mlflow.experiment_name,
            run_name=cfg.output.mlflow.run_name,
            save_dir=cfg.output.mlflow.log_dir,
            tracking_uri=cfg.output.mlflow.server_uri,
            log_model=True,
        )
        if cfg.output.mlflow.tags:
            for k, v in cfg.output.mlflow.tags.items():
                mlflow_logger.experiment.set_tag(mlflow_logger.run_id, k, v)

    trainer = L.Trainer(
        max_epochs=num_epochs,
        enable_checkpointing=bool(early_stopping),
        devices=devices,
        accelerator=accelerator,
        strategy=strategy,
        callbacks=callbacks,
        logger=mlflow_logger,
        accumulate_grad_batches=cfg.pipe.gradient_accumulation_steps,
        gradient_clip_val=cfg.pipe.gradient_clip_val,
        num_sanity_val_steps=0,
        deterministic=True,
    )

    # ── Train ─────────────────────────────────────────────────────────
    logger.info("Beginning training...")
    trainer.fit(model=model, datamodule=datamodule)

    # If early stopping was used, reload best weights
    if early_stopping and trainer.checkpoint_callback is not None:
        best = trainer.checkpoint_callback.best_model_path
        if best:
            logger.info(f"Loading best checkpoint: {best}")
            ckpt = torch.load(best, weights_only=False)
            model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)

    # ── Save ──────────────────────────────────────────────────────────
    _save_artifacts(cfg, model, trainer, base_dm, params, logger)
    logger.info("Done.")


if __name__ == "__main__":
    main()
