"""CLI command for automated hyperparameter optimization.

Performs architecture and optimizer search using MATCHA's built-in tuning
infrastructure, saves the optimal parameter set as YAML, and optionally
logs results to MLflow.
"""

from matcha.cli.utils import load_dataset, get_splits, save_config_as_yaml
from matcha.utils.schemas.cli import CLIAutotuneOutput
from matcha.utils.serialization import parse_df, save_yaml
from matcha.sklearn.base_sklearn_model import ScikitLearnModelRegistry
from matcha.utils.logging import get_default_logger
import argparse
import json
import yaml


def main(cfg=None):
    """Run hyperparameter optimization from a YAML configuration.

    Loads training data, generates train/validation splits, creates a model
    instance, computes features, and runs a two-stage search (architecture
    then optimizer) to find optimal hyperparameters. The best configuration
    is saved as YAML.

    :param cfg: Pre-parsed configuration object or ``None`` to parse from
        CLI ``--config`` argument. Accepts a
        :class:`~matcha.utils.schemas.cli.CLIAutotuneOutput` instance or a
        raw dict that will be validated.
    """
    if cfg is None:
        parser = argparse.ArgumentParser(description="Autotune hyperparameters")
        parser.add_argument(
            "--config", type=str, required=True, help="Path to the YAML config file"
        )
        args = parser.parse_args()
        with open(args.config, "r") as f:
            raw = yaml.safe_load(f)
        cfg = CLIAutotuneOutput.model_validate(raw)
    elif not isinstance(cfg, CLIAutotuneOutput):
        cfg = CLIAutotuneOutput.model_validate(cfg)
    log_path = f"{cfg.output.optimum.path}/{cfg.output.optimum.filename}_autotune.log"
    logger = get_default_logger("HPO", logging_path=log_path)

    logger.info(f"Loading training dataset | settings: {cfg.dataset}")
    input_df = load_dataset(cfg.dataset)

    logger.info("Creating splits")
    train_splits, val_splits = get_splits(input_df, cfg.split, return_val_splits=True)

    logger.info(f"Generated {len(train_splits)} splits")

    logger.info("Creating model instance:")
    logger.info(f"    Fixed params: {json.dumps(cfg.model.params, indent=2)}")
    model = ScikitLearnModelRegistry[cfg.model.architecture](**cfg.model.params)

    if cfg.output.mlflow is not None:
        logger.info("Setting up MLFlow")
        model.set_mlflow_experiment(
            experiment_name=cfg.output.mlflow.experiment_name,
            tag=cfg.output.mlflow.tags,
            log_dir=cfg.output.mlflow.log_dir,
        )

    logger.info("Generating features")
    train_feats = []
    val_feats = []
    for i in range(len(train_splits)):
        logger.info(f"Processing split {i}")
        t_mols, t_y, t_operator = parse_df(
            train_splits[i], cfg.dataset.label_key, cfg.dataset.operator_key
        )
        v1_mols, v1_y, v1_operator = parse_df(
            val_splits[i][0], cfg.dataset.label_key, cfg.dataset.operator_key
        )
        v2_mols, v2_y, v2_operator = parse_df(
            val_splits[i][1], cfg.dataset.label_key, cfg.dataset.operator_key
        )

        train_feats.append(
            model.transform(
                t_mols,
                t_y,
                t_operator,
                is_training=True,
            )
        )
        val_feats.append(
            [
                model.transform(v1_mols, v1_y, v1_operator, is_training=False),
                model.transform(v2_mols, v2_y, v2_operator, is_training=False),
            ]
        )

    logger.info("Beginning HPO run")
    _ = model.tune(
        train_feats,
        val_feats,
        architecture_search_budget=cfg.tuning.architecture_search.budget,
        optimizer_search_budget=cfg.tuning.optimizer_search.budget,
    )
    optimum = model.params.model_dump()
    logger.info("HPO finished")
    logger.info(f"Best params: {json.dumps(optimum, indent=2)}")

    save_yaml(f"{cfg.output.optimum.path}/{cfg.output.optimum.filename}.yaml", optimum)
    save_config_as_yaml(
        cfg, f"{cfg.output.optimum.path}/{cfg.output.optimum.filename}_cfg.yaml"
    )


if __name__ == "__main__":
    main()
