"""CLI entrypoint for encoder-only pretraining (MLM or graph multi-task).

Produces an artifact directory that the :class:`Finetuner` can consume
directly via ``origin_type: "pretraining"`` in the manifest.

Artifact layout::

    <output>/
    ├── model.ckpt               # full Lightning checkpoint (for resumption)
    ├── encoder.ckpt             # encoder-only state_dict (for finetuning)
    ├── config/
    │   ├── manifest.yaml        # origin_type: "pretraining", source_class, …
    │   ├── model.yaml           # pretraining model hparams
    │   ├── training.yaml        # training hparams
    │   ├── datamodule.yaml      # datamodule params
    │   └── metadata.yaml        # metadata
    ├── state/
    │   └── datamodule_state.pkl # fitted scalers / encoders / dictionary
    └── cfg.yaml                 # copy of the input config
"""

import os
import datetime
import argparse
import yaml

import numpy as np
import pandas as pd
import torch
import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.strategies import DDPStrategy
from rdkit import Chem

from matcha.utils.logging import get_default_logger, MatchaLogger
from matcha.utils.serialization import save_pickle, save_yaml
from matcha.utils.schemas.cli import (
    CLIEncoderPretrainInputModel,
    EncoderPretrainGraphDatamodule,
    EncoderPretrainMLMDatamodule,
    PretrainTraining,
)
from matcha.cli.utils import _load_coords_npz, _load_npz_list, save_config_as_yaml
from matcha.torch.models.pretraining import PretrainingModelRegistry
from matcha import __version__

torch.set_float32_matmul_precision("high")


# Keys from the datamodule config that must also be forwarded to graph
# model constructors (as ``enc_<key>``).
_GRAPH_DM_KEYS_BROADCAST_TO_MODEL = {
    "laplacian_k",
    "rwse_k",
    "elstatic_k",
    "distmat_k",
    "rrwp_k",
}


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _load_smiles(path: str) -> list[str]:
    """Load SMILES from a parquet file (expects a ``SMILES`` column)."""
    return pd.read_parquet(path).SMILES.tolist()


def _load_npz_array(path: str, key: str = "descriptors") -> np.ndarray:
    """Load a single array from an npz file.

    :param path: Path to the ``.npz`` file.
    :param key: Array key within the archive (default ``"descriptors"``).
    :returns: The loaded NumPy array.
    """
    return np.load(path)[key]


def _filter_node_label_mismatches(
    smiles: list[str],
    y_graph: np.ndarray,
    y_node: list[np.ndarray],
    logger,
    split_name: str = "data",
    coords: list[np.ndarray] | None = None,
) -> tuple[list[str], np.ndarray, list[np.ndarray], list[np.ndarray] | None]:
    """Drop molecules whose atom-level labels don't match the canonical atom count.

    Compares the number of rows in each ``y_node[i]`` against the number of
    heavy atoms in the canonical form of the SMILES.  Entries that fail the
    check (wrong row count) are removed from all supplied arrays so downstream
    datamodules never see them.

    :param smiles: list of SMILES strings
    :param y_graph: array ``(N, G)`` of molecule-level targets
    :param y_node: list of N arrays, each ``(A_i, T)``
    :param logger: logger instance
    :param split_name: label used in the log message (e.g. "train", "val")
    :param coords: optional list of N arrays of shape ``(A_i, 3)`` filtered by
        the same ``keep`` mask.  When ``None``, the fourth tuple element is
        ``None``.
    :return: filtered (smiles, y_graph, y_node, coords)
    """
    keep: list[int] = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        canonical_mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, canonical=True))
        expected_atoms = canonical_mol.GetNumAtoms()
        yn = np.asarray(y_node[i])
        if yn.ndim == 1:
            yn = yn.reshape(-1, 1)
        if yn.shape[0] != expected_atoms:
            continue
        keep.append(i)

    n_dropped = len(smiles) - len(keep)
    if n_dropped > 0:
        logger.warning(
            "Dropped %d / %d %s molecules due to atom-level label row count mismatch.",
            n_dropped,
            len(smiles),
            split_name,
        )

    filtered_smiles = [smiles[i] for i in keep]
    filtered_y_graph = y_graph[keep]
    filtered_y_node = [y_node[i] for i in keep]
    filtered_coords = [coords[i] for i in keep] if coords is not None else None
    return filtered_smiles, filtered_y_graph, filtered_y_node, filtered_coords


# ──────────────────────────────────────────────────────────────────────
# Graph pretraining path
# ──────────────────────────────────────────────────────────────────────


def _run_graph_pretraining_generic(
    cfg: CLIEncoderPretrainInputModel,
    logger,
    *,
    dm_cls,
    coords_loader=None,
):
    """Shared build path for 2D and 3D graph encoder pretraining.

    Loads SMILES + graph/node labels (and optionally coords), filters out
    mismatched atom counts, instantiates ``dm_cls`` from the typed
    :class:`EncoderPretrainGraphDatamodule` config, and wraps it in the
    :class:`OnTheFlyGraphPretrainingDataModule` streaming wrapper.

    :param cfg: Validated CLI configuration.
    :param logger: Logger instance for progress messages.
    :param dm_cls: Concrete base datamodule class — e.g.
        :class:`GraphPretrainingDataModule` for 2D or
        :class:`Graph3DPretrainingDataModule` for 3D.
    :param coords_loader: Optional callable ``(path) -> list[np.ndarray]``
        used to load per-split coordinate npz files.  When supplied,
        ``cfg.dataset.train_coords`` and ``cfg.dataset.val_coords`` are
        loaded and threaded through featurization; the schema guarantees
        both paths are present when this branch is entered.
    :returns: ``(base_dm, otf_dm)`` ready for training.
    """
    from matcha.datamodules.pretraining.on_the_fly_graph_pretraining_datamodule import (
        OnTheFlyGraphPretrainingDataModule,
    )

    ds = cfg.dataset

    # Load data
    logger.info("Loading training data…")
    train_smiles = _load_smiles(ds.train_smiles)
    train_y_graph = _load_npz_array(ds.train_y_graph)
    train_y_node = _load_npz_list(ds.train_y_node)
    train_coords = coords_loader(ds.train_coords) if coords_loader is not None else None
    logger.info(f"  train: {len(train_smiles)} molecules")

    logger.info("Loading validation data…")
    val_smiles = _load_smiles(ds.val_smiles)
    val_y_graph = _load_npz_array(ds.val_y_graph)
    val_y_node = _load_npz_list(ds.val_y_node)
    val_coords = coords_loader(ds.val_coords) if coords_loader is not None else None
    logger.info(f"  val:   {len(val_smiles)} molecules")

    # Resolve datamodule config (use typed defaults when omitted)
    dm_cfg = EncoderPretrainGraphDatamodule(**(cfg.model.datamodule or {}))

    # Filter out molecules with mismatched atom-level labels upfront
    train_smiles, train_y_graph, train_y_node, train_coords = (
        _filter_node_label_mismatches(
            train_smiles,
            train_y_graph,
            train_y_node,
            logger,
            split_name="train",
            coords=train_coords,
        )
    )
    val_smiles, val_y_graph, val_y_node, val_coords = _filter_node_label_mismatches(
        val_smiles,
        val_y_graph,
        val_y_node,
        logger,
        split_name="val",
        coords=val_coords,
    )
    logger.info(f"  after filtering: train={len(train_smiles)}, val={len(val_smiles)}")

    # Build base datamodule from typed config
    dm_kwargs = dm_cfg.model_dump()
    base_dm = dm_cls(**dm_kwargs)

    # Optional: fit datamodule on a sample for any stateful components
    if cfg.pipe.fit_datamodule_size is not None:
        sample_size = min(cfg.pipe.fit_datamodule_size, len(train_smiles))
        idx = np.random.choice(len(train_smiles), sample_size, replace=False)
        sample_mols = [Chem.MolFromSmiles(train_smiles[i]) for i in idx]
        sample_y_graph = train_y_graph[idx]
        sample_y_node = [train_y_node[i] for i in idx]
        if train_coords is not None:
            sample_coords = [train_coords[i] for i in idx]
            base_dm.featurize(
                sample_mols,
                sample_y_graph,
                sample_y_node,
                coords=sample_coords,
                is_training=True,
            )
        else:
            base_dm.featurize(
                sample_mols, sample_y_graph, sample_y_node, is_training=True
            )

    # On-the-fly wrapper
    otf_dm = OnTheFlyGraphPretrainingDataModule(
        base=base_dm,
        num_workers=cfg.pipe.dataloader_num_workers,
    )
    set_data_kwargs = dict(
        train_smiles=train_smiles,
        train_y_graph=train_y_graph,
        train_y_node=train_y_node,
        val_smiles=val_smiles,
        val_y_graph=val_y_graph,
        val_y_node=val_y_node,
    )
    if train_coords is not None:
        set_data_kwargs["train_coords"] = train_coords
    if val_coords is not None:
        set_data_kwargs["val_coords"] = val_coords
    otf_dm.set_data(**set_data_kwargs)
    otf_dm.setup(stage="fit")
    otf_dm.params.batch_size = dm_cfg.batch_size

    return base_dm, otf_dm


def _run_graph_pretraining(cfg: CLIEncoderPretrainInputModel, logger):
    """Train a 2D graph pretraining model (node + graph multi-task)."""
    from matcha.datamodules.pretraining.graph_pretraining_datamodule import (
        GraphPretrainingDataModule,
    )

    return _run_graph_pretraining_generic(
        cfg, logger, dm_cls=GraphPretrainingDataModule
    )


def _run_graph3d_pretraining(cfg: CLIEncoderPretrainInputModel, logger):
    """Train a 3D graph pretraining model with user-supplied atomic coordinates."""
    from matcha.datamodules.pretraining.graph_3d_pretraining_datamodule import (
        Graph3DPretrainingDataModule,
    )

    return _run_graph_pretraining_generic(
        cfg,
        logger,
        dm_cls=Graph3DPretrainingDataModule,
        coords_loader=_load_coords_npz,
    )


def _run_mlm_pretraining(cfg: CLIEncoderPretrainInputModel, logger):
    """Train an MLM pretraining model."""
    from matcha.datamodules.pretraining.clm_mlm_datamodule import CLMMLMDataModule
    from matcha.datamodules.pretraining.on_the_fly_mlm_datamodule import (
        OnTheFlyMLMDataModule,
    )

    ds = cfg.dataset

    logger.info("Loading training data…")
    train_smiles = _load_smiles(ds.train_smiles)
    logger.info(f"  train: {len(train_smiles)} molecules")

    logger.info("Loading validation data…")
    val_smiles = _load_smiles(ds.val_smiles)
    logger.info(f"  val:   {len(val_smiles)} molecules")

    # Resolve datamodule config (use typed defaults when omitted)
    dm_cfg = EncoderPretrainMLMDatamodule(**(cfg.model.datamodule or {}))

    # Build base MLM datamodule from typed config
    dm_kwargs = dm_cfg.model_dump()
    base_dm = CLMMLMDataModule(**dm_kwargs)

    # Fit dictionary on a sample (or all) of training SMILES
    fit_size = cfg.pipe.fit_datamodule_size or len(train_smiles)
    fit_size = min(fit_size, len(train_smiles))
    sample_smiles = np.random.choice(train_smiles, fit_size, replace=False).tolist()
    logger.info(f"Fitting dictionary on {fit_size} SMILES…")
    mols = [Chem.MolFromSmiles(s) for s in sample_smiles]
    mols = [m for m in mols if m is not None]
    base_dm.featurize(mols, y=None, is_training=True, augment=False)
    logger.info(f"Dictionary size: {base_dm.params.num_tokens}")

    # On-the-fly wrapper
    otf_dm = OnTheFlyMLMDataModule(
        base=base_dm,
        num_workers=cfg.pipe.dataloader_num_workers,
    )
    otf_dm.set_data(train_smiles=train_smiles, val_smiles=val_smiles)
    otf_dm.setup(stage="fit")
    otf_dm.params.batch_size = dm_cfg.batch_size

    return base_dm, otf_dm


# ──────────────────────────────────────────────────────────────────────
# Save artifacts
# ──────────────────────────────────────────────────────────────────────


def _save_artifacts(
    cfg: CLIEncoderPretrainInputModel,
    model,
    trainer: L.Trainer,
    base_dm,
    logger,
):
    """Save encoder pretraining artifacts (checkpoints, configs, manifest) to disk.

    Produces the standard artifact layout: full Lightning checkpoint,
    encoder-only checkpoint (for finetuning), datamodule state, config
    YAMLs, metadata, and a manifest file.

    :param cfg: The full CLI configuration.
    :param model: Trained Lightning model with ``export_encoder_checkpoint``.
    :param trainer: Lightning trainer instance.
    :param base_dm: Base datamodule (for state serialization).
    :param logger: Logger instance.
    """
    out_dir = cfg.output.serialization
    config_dir = os.path.join(out_dir, "config")
    state_dir = os.path.join(out_dir, "state")
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(state_dir, exist_ok=True)

    # Full checkpoint (for resumption)
    ckpt_path = os.path.join(out_dir, "model.ckpt")
    trainer.save_checkpoint(ckpt_path)
    logger.info(f"Full checkpoint saved: {ckpt_path}")

    # Encoder-only checkpoint (for finetuning)
    model.export_encoder_checkpoint(out_dir)
    logger.info(f"Encoder checkpoint saved: {os.path.join(out_dir, 'encoder.ckpt')}")

    # Datamodule state
    dm_state = base_dm.state_dict()
    save_pickle(os.path.join(state_dir, "datamodule_state.pkl"), dm_state)

    # Config YAMLs
    model_params = (
        model.hparams.copy() if hasattr(model.hparams, "copy") else dict(model.hparams)
    )
    model_params["torch_type"] = cfg.model.architecture.lower()
    save_yaml(os.path.join(config_dir, "model.yaml"), model_params)

    training_cfg = cfg.model.training or PretrainTraining()
    training_yaml = {
        "num_epochs": training_cfg.num_epochs,
        "batch_size": base_dm.params.batch_size,
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
    save_yaml(os.path.join(config_dir, "datamodule.yaml"), base_dm.params.model_dump())

    metadata = {
        "model_type": cfg.model.architecture,
        "model_name": "encoder pretraining",
        "model_version": 1,
        "model_scope": "pretraining",
        "model_owner": "matcha",
        "matcha_version": __version__,
        "date": datetime.datetime.now().isoformat(),
        "description": f"Encoder pretrained with {cfg.dataset.task_type} objective",
        "extra": {},
    }
    save_yaml(os.path.join(config_dir, "metadata.yaml"), metadata)

    # Manifest — this is what the Finetuner reads
    save_yaml(
        os.path.join(config_dir, "manifest.yaml"),
        {
            "matcha_version": __version__,
            "serialization_version": 2,
            "origin_type": "pretraining",
            "source_class": cfg.model.architecture,
            "task_type": cfg.dataset.task_type,
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
    """Run encoder pretraining (MLM or graph multi-task) from a YAML configuration.

    Supports three pretraining modes controlled by ``dataset.task_type``:

    - **mlm**: Masked language modelling over SMILES strings.
    - **graph**: Supervised node-level and graph-level multi-task objectives
      on 2D molecular graphs.
    - **graph3d**: Same as ``graph`` but with user-supplied precomputed 3D
      atomic coordinates (``train_coords`` / ``val_coords`` npz files) routed
      through to a 3D-capable base datamodule such as
      :class:`Graph3DPretrainingDataModule` for architectures like
      :class:`E3GNNPretraining`.

    Trains the encoder with Lightning, optionally using DDP for multi-GPU
    training, and saves artifacts (encoder checkpoint, configs, manifest)
    compatible with downstream finetuning via the ``Finetuner``.

    :param cfg: Pre-parsed configuration object or ``None`` to parse from
        CLI ``--config`` argument. Accepts a
        :class:`~matcha.utils.schemas.cli.CLIEncoderPretrainInputModel`
        instance or a raw dict that will be validated.
    """
    if cfg is None:
        parser = argparse.ArgumentParser(
            description="Pretrain an encoder (MLM, graph, or graph3d)"
        )
        parser.add_argument(
            "--config",
            type=str,
            required=True,
            help="Path to the YAML config file",
        )
        args = parser.parse_args()
        with open(args.config, "r") as f:
            raw = yaml.safe_load(f)
        cfg = CLIEncoderPretrainInputModel.model_validate(raw)
    elif not isinstance(cfg, CLIEncoderPretrainInputModel):
        cfg = CLIEncoderPretrainInputModel.model_validate(cfg)

    log_path = os.path.join(cfg.output.serialization, "pretrain_encoder.log")
    os.makedirs(cfg.output.serialization, exist_ok=True)
    logger = get_default_logger("PRETRAIN_ENCODER", logging_path=log_path)

    # GPU visibility
    if cfg.pipe.visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.pipe.visible_devices)
        logger.info(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    # Seed
    training_cfg = cfg.model.training or PretrainTraining()
    seed = training_cfg.seed
    L.seed_everything(seed, workers=True, verbose=False)

    # ── Build datamodule ──────────────────────────────────────────────
    task_type = cfg.dataset.task_type.lower()
    if task_type == "graph":
        base_dm, otf_dm = _run_graph_pretraining(cfg, logger)
    elif task_type == "graph3d":
        base_dm, otf_dm = _run_graph3d_pretraining(cfg, logger)
    elif task_type == "mlm":
        base_dm, otf_dm = _run_mlm_pretraining(cfg, logger)
    else:
        raise ValueError(
            f"Unknown dataset.task_type '{task_type}'. "
            "Expected 'mlm', 'graph', or 'graph3d'."
        )

    # ── Build model ───────────────────────────────────────────────────
    logger.info(f"Creating pretraining model: {cfg.model.architecture}")
    model_init_params = cfg.model.params.copy()

    # For MLM models, inject num_characters from the fitted dictionary
    if task_type == "mlm" and "enc_num_characters" not in model_init_params:
        model_init_params["enc_num_characters"] = base_dm.params.num_tokens

    # For graph models, broadcast positional-encoding keys from the
    # datamodule config into the model constructor (as ``enc_<key>``).
    if task_type in ("graph", "graph3d"):
        dm_cfg = EncoderPretrainGraphDatamodule(**(cfg.model.datamodule or {}))
        for key in _GRAPH_DM_KEYS_BROADCAST_TO_MODEL:
            enc_key = f"enc_{key}"
            if enc_key not in model_init_params:
                model_init_params[enc_key] = getattr(dm_cfg, key)

    model = PretrainingModelRegistry[cfg.model.architecture](**model_init_params)
    logger.info(
        f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # ── Trainer ───────────────────────────────────────────────────────
    if cfg.pipe.visible_devices:
        devices = len(cfg.pipe.visible_devices.split(","))
    else:
        devices = training_cfg.devices

    accelerator = training_cfg.accelerator
    num_epochs = training_cfg.num_epochs
    patience = training_cfg.patience
    early_stopping = training_cfg.early_stopping

    strategy = "auto"
    if accelerator == "gpu" and devices > 1:
        logger.info(f"Using DDP on {devices} GPUs")
        strategy = DDPStrategy(
            timeout=datetime.timedelta(
                seconds=int(cfg.pipe.strategy.get("timeout_seconds", 3600))
            ),
            find_unused_parameters=bool(
                cfg.pipe.strategy.get("find_unused_parameters", False)
            ),
        )

    callbacks = [LearningRateMonitor(logging_interval="step")]
    if early_stopping:
        callbacks.append(
            EarlyStopping(monitor="val_loss", mode="min", patience=patience)
        )
        callbacks.append(ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min"))

    mlflow_logger = True
    if cfg.output.mlflow is not None:
        mlf = cfg.output.mlflow
        logger.info(
            f"MLFlow logging: experiment='{mlf.experiment_name}' "
            f"log_dir='{mlf.log_dir}'"
        )
        mlflow_logger = MatchaLogger(
            experiment_name=mlf.experiment_name,
            run_name=mlf.run_name,
            save_dir=mlf.log_dir,
            tracking_uri=mlf.server_uri,
            log_model=True,
        )
        if mlf.tags:
            for k, v in mlf.tags.items():
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
    logger.info("Starting training…")
    trainer.fit(model=model, datamodule=otf_dm)

    # If early stopping was used, reload best weights
    if early_stopping and trainer.checkpoint_callback is not None:
        best = trainer.checkpoint_callback.best_model_path
        if best:
            logger.info(f"Loading best checkpoint: {best}")
            ckpt = torch.load(best, weights_only=False)
            model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)

    # ── Save ──────────────────────────────────────────────────────────
    _save_artifacts(cfg, model, trainer, base_dm, logger)
    logger.info("Done.")


if __name__ == "__main__":
    main()
