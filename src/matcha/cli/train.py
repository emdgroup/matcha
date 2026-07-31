"""CLI command for training a MATCHA sklearn model.

Loads a dataset, constructs an ensemble (or single) model from the
specified architecture, fits it, optionally calibrates uncertainty, and
serializes the trained artifact.
"""

from matcha.cli.utils import load_dataset, save_config_as_yaml
from matcha.utils.serialization import parse_df, load_yaml
from matcha.utils.logging import get_default_logger
from matcha.sklearn import Ensemble
from matcha.sklearn.base_sklearn_model import ScikitLearnModelRegistry
from matcha.utils.schemas.cli import CLITrainInputModel
import argparse
import json
import yaml


def main(cfg=None):
    """Train a MATCHA sklearn model from a YAML configuration.

    Orchestrates the full training pipeline: dataset loading, optional
    calibration set splitting, model instantiation (single or ensemble),
    fitting, uncertainty calibration, MLflow logging, and serialization.

    :param cfg: Pre-parsed configuration object or ``None`` to parse from
        CLI ``--config`` argument. Accepts a
        :class:`~matcha.utils.schemas.cli.CLITrainInputModel` instance or a
        raw dict that will be validated.
    """

    if cfg is None:
        parser = argparse.ArgumentParser(description="Train a model")
        parser.add_argument(
            "--config", type=str, required=True, help="Path to the YAML config file"
        )
        args = parser.parse_args()
        with open(args.config, "r") as f:
            raw = yaml.safe_load(f)
        cfg = CLITrainInputModel.model_validate(raw)
    elif not isinstance(cfg, CLITrainInputModel):
        cfg = CLITrainInputModel.model_validate(cfg)
    log_path = (
        f"{cfg.output.serialization.path}/train.log"
        if cfg.output.serialization is not None
        else None
    )
    logger = get_default_logger("TRAIN", logging_path=log_path)

    # Load data
    logger.info(
        f"Loading dataset | path: {cfg.dataset.path} | label: {cfg.dataset.label_key}"
    )
    input_df = load_dataset(cfg.dataset)
    logger.info("Dataset loaded")

    # Parse calibration settings
    if cfg.dataset.calibration is not None:
        logger.info("Generating calibration set from training data")
        input_df = input_df.sort_values(
            cfg.dataset.calibration.split_column, axis=0, ascending=True
        )
        k = int(len(input_df) * cfg.dataset.calibration.split_ratio)
        calibration_df = input_df.iloc[k:]
        input_df = input_df.iloc[:k]
        logger.info("Calibration setup finished")
    else:
        logger.info("No calibration options specified in data, skipping")

    # Parse dataframe into MATCHA-ready input
    logger.info("Parsing dataset")
    mols, y, operator = parse_df(
        input_df, cfg.dataset.label_key, cfg.dataset.operator_key
    )
    logger.info("Dataset parsed")

    # Parse model params / reload from HPO output
    if cfg.model.config_path is not None:
        logger.info(f"Loading model config | path: {cfg.model.config_path}")
        optimum_config = load_yaml(cfg.model.config_path)
        optimum_config["datamodule"]["label_encoder_params"] = cfg.model.params[
            "label_encoder_params"
        ]
        cfg.model.params = optimum_config

    # Create model / ensemble instance
    arch_cls = ScikitLearnModelRegistry[cfg.model.architecture]
    if cfg.model.ensemble is not None:
        logger.info("Generating ensemble model:")
        logger.info(f"  Architecture: '{cfg.model.architecture}'")
        logger.info(f"  Ensemble size: {cfg.model.ensemble} models")
        if cfg.model.config_path is None:
            logger.info(f"  Model parameters: {json.dumps(cfg.model.params, indent=2)}")
        template = (
            arch_cls.from_config(cfg.model.params)
            if cfg.model.config_path is not None
            else arch_cls(**cfg.model.params)
        )
        model = Ensemble(model=template, n_models=cfg.model.ensemble)
    else:
        logger.info("Generating single model:")
        logger.info(f"  Architecture: '{cfg.model.architecture}'")
        if cfg.model.config_path is None:
            logger.info(f"  Model parameters: {json.dumps(cfg.model.params, indent=2)}")
        model = (
            arch_cls.from_config(cfg.model.params)
            if cfg.model.config_path is not None
            else arch_cls(**cfg.model.params)
        )

    # Store additional info in model metadata
    model.annotate("dataset_metadata", cfg.dataset.model_dump())
    model.params.metadata.model_name = cfg.model.metadata.model_name
    model.params.metadata.model_version = cfg.model.metadata.model_version
    model.params.metadata.model_scope = cfg.model.metadata.model_scope
    model.params.metadata.model_owner = cfg.model.metadata.model_owner
    model.params.metadata.description = cfg.model.metadata.description

    # Set up MLFlow
    if cfg.output.mlflow is not None:
        logger.info(
            f"Setting up MLFlow logging | experiment: '{cfg.output.mlflow.experiment_name}' | log_dir: '{cfg.output.mlflow.log_dir}'"
        )
        model.set_mlflow_experiment(
            experiment_name=cfg.output.mlflow.experiment_name,
            tag=cfg.output.mlflow.tags,
            log_dir=cfg.output.mlflow.log_dir,
        )

    # Fit model
    logger.info("Beginning model fit")
    model.fit(mols, y, operator)
    logger.info("Model successfully fit")

    # Run calibration
    if cfg.model.calibration is not None:
        logger.info(
            f"Running fit on calibration set | algorithm: '{cfg.model.calibration.algorithm}'"
        )
        mols_cal, y_cal, _ = parse_df(calibration_df, cfg.dataset.label_key)
        model.calibrate_uncertainty(
            mols_cal,
            y_cal,
            algorithm=cfg.model.calibration.algorithm,
            algorithm_args=cfg.model.calibration.params,
        )
    else:
        logger.info("Skipping fit on calibration set")

    # Serialize model in target path
    if cfg.output.serialization is not None:
        logger.info(
            f"Saving model | path: '{cfg.output.serialization.path}' | quantize: {cfg.output.serialization.quantize}"
        )
        model.save_model(
            cfg.output.serialization.path, cfg.output.serialization.quantize
        )
        save_config_as_yaml(cfg, f"{cfg.output.serialization.path}/cfg.yaml")
    else:
        logger.info("Skipping serialization in specific folder")

    logger.info("Finished")


if __name__ == "__main__":
    main()
