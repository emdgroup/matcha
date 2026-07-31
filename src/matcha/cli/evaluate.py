"""CLI command for cross-validated model evaluation.

Trains a single (non-ensemble) MATCHA model on each split, computes
regression or classification metrics, generates plots, and logs results
to MLflow.
"""

from matcha.cli.utils import (
    load_dataset,
    get_splits,
    aggregate_scores,
    save_config_as_yaml,
    store_split_scores,
)

from matcha.utils.metrics import (
    process_censor,
    process_regression,
    process_classification,
)

from matcha.utils.plotting import plot_regression, plot_classification, save_plot

from matcha.utils.schemas.cli import CLIEvaluationInputModel
from matcha.utils.serialization import parse_df, save_json, load_yaml
from matcha.sklearn.base_sklearn_model import ScikitLearnModelRegistry
from matcha.utils.logging import MatchaLogger
from rdkit import Chem
from matcha.utils.logging import get_default_logger
import argparse
import json
import yaml
import numpy as np
import pandas as pd
import os


def flatten_performance_dict(perf_dict: dict, tag: str):
    """Flatten a nested performance dictionary into a single-level dict for MLflow logging.

    Converts the hierarchical ``{split: {endpoint: {metric: value}}}``
    structure into flat keys of the form ``{tag}_{split}_{endpoint}_{metric}``.

    :param perf_dict: Nested performance dictionary with split, endpoint,
        and metric levels.
    :param tag: Prefix string to prepend to flattened keys (e.g. ``"log10"``).
    :returns: Flat dictionary mapping composite key strings to float values.
    """
    flat = {}
    for split_key, split_val in perf_dict.items():
        if split_key in ["mean", "std"]:
            prefix = tag + "_" + split_key if tag else split_key
        else:
            prefix = f"{tag}_split_{split_key}" if tag else f"split_{split_key}"
        for endpoint, metrics in split_val.items():
            for metric, value in metrics.items():
                metric_name = f"{prefix}_{endpoint}_{metric}"
                flat[metric_name] = float(value)
    return flat


def main(cfg=None):
    """Run cross-validated model evaluation from a YAML configuration.

    For each split, trains a single model instance, computes predictions,
    calculates regression or classification metrics (with optional
    bootstrapping), generates plots, and aggregates results across splits.
    Final metrics and artifacts are optionally logged to MLflow.

    :param cfg: Pre-parsed configuration object or ``None`` to parse from
        CLI ``--config`` argument. Accepts a
        :class:`~matcha.utils.schemas.cli.CLIEvaluationInputModel` instance
        or a raw dict that will be validated.
    """
    if cfg is None:
        parser = argparse.ArgumentParser(description="Evaluate a model")
        parser.add_argument(
            "--config", type=str, required=True, help="Path to the YAML config file"
        )
        args = parser.parse_args()
        with open(args.config, "r") as f:
            raw = yaml.safe_load(f)
        cfg = CLIEvaluationInputModel.model_validate(raw)
    elif not isinstance(cfg, CLIEvaluationInputModel):
        cfg = CLIEvaluationInputModel.model_validate(cfg)
    log_path = (
        f"{cfg.output.serialization.path}/evaluate.log"
        if cfg.output.serialization is not None
        else None
    )
    logger = get_default_logger("EVAL", logging_path=log_path)

    performance_dict = {}
    performance_dict_log10 = {}
    n_bootstrap = cfg.split.n_bootstrap
    frac_bootstrap = cfg.split.frac_bootstrap

    logger.info(f"Loading training dataset | settings: {cfg.dataset}")
    input_df = load_dataset(cfg.dataset)

    logger.info("Creating splits")
    train_splits, test_splits = get_splits(
        input_df, cfg.split, return_val_splits=False, dataset_cfg=cfg.dataset
    )

    logger.info(f"Generated {len(train_splits)} splits")

    # Parse model params / reload from HPO output
    if cfg.model.config_path is not None:
        logger.info(f"Loading model config | path: {cfg.model.config_path}")
        optimum_config = load_yaml(cfg.model.config_path)
        optimum_config["datamodule"]["label_encoder_params"] = cfg.model.params[
            "label_encoder_params"
        ]
        cfg.model.params = optimum_config

    for i in range(len(train_splits)):
        logger.info(f"Processing split {i}")

        logger.info("Logging splits")
        train_splits[i].to_csv(
            f"{cfg.output.serialization.path}/train_split_{i}.csv", index=False
        )
        test_splits[i].to_csv(
            f"{cfg.output.serialization.path}/test_split_{i}.csv", index=False
        )

        train_mols, train_y, train_operator = parse_df(
            train_splits[i], cfg.dataset.label_key, cfg.dataset.operator_key
        )
        test_mols, test_y, test_operator = parse_df(
            test_splits[i], cfg.dataset.label_key, cfg.dataset.operator_key
        )
        test_smi = [Chem.MolToSmiles(x) for x in test_mols]

        if isinstance(cfg.dataset.label_key, str):
            label_key = [
                x for x in train_splits[i].columns if cfg.dataset.label_key in x
            ]
        else:
            label_key = cfg.dataset.label_key

        # Create model / ensemble instance
        if cfg.model.ensemble is not None:
            raise ValueError(
                "Invalid model configuration, the evaluate command cannot be used with ensembles! You can fix this by removing the `ensemble` argument from the config."
            )
            # logger.info("Generating ensemble model:")
            # logger.info(f"  Architecture: '{cfg.model.architecture}'")
            # logger.info(f"  Ensemble size: {cfg.model.ensemble} models")
            # logger.info(f"  Model parameters: {json.dumps(cfg.model.params, indent=2)}")
            # template = ScikitLearnModelRegistry[cfg.model.architecture](**cfg.model.params)
            # model = Ensemble(model=template, n_models=cfg.model.ensemble)
        else:
            logger.info("Generating single model:")
            logger.info(f"  Architecture: '{cfg.model.architecture}'")
            arch_cls = ScikitLearnModelRegistry[cfg.model.architecture]
            if cfg.model.config_path is not None:
                cfg.model.params["training"]["seed"] = i
                model = arch_cls.from_config(cfg.model.params)
            else:
                cfg.model.params["seed"] = i
                logger.info(
                    f"  Model parameters: {json.dumps(cfg.model.params, indent=2)}"
                )
                model = arch_cls(**cfg.model.params)

        if cfg.output.mlflow is not None:
            logger.info(
                f"Setting up MLFlow logging | experiment: '{cfg.output.mlflow.experiment_name}' | log_dir: '{cfg.output.mlflow.log_dir}'"
            )

            if cfg.output.mlflow.run_name is None:
                run_name = cfg.model.architecture
            else:
                run_name = cfg.output.mlflow.run_name

            model.set_mlflow_experiment(
                experiment_name=cfg.output.mlflow.experiment_name,
                run_name=f"{run_name}_split_{i}",
                tag=cfg.output.mlflow.tags,
                log_dir=cfg.output.mlflow.log_dir,
            )

        logger.info("Beginning model fit")
        model.fit(train_mols, train_y, train_operator)
        logger.info("Model successfully fit")

        preds = model.predict(test_mols)
        logger.info("Predictions calculated")

        if "Classifier" in model.params.metadata.model_type:
            probs = model.predict_proba(test_mols)

            if len(np.unique(test_y)) != 2:
                test_y = model._datamodule._label_encoder._all_to_categorical(test_y)

        if not isinstance(preds, np.ndarray):
            preds = preds[0]

        for j in range(preds.shape[1]):
            logger.info(f"Computing metrics for {label_key[j]}")

            # skip this index j if all test labels are NaN
            if np.all(np.isnan(test_y[:, j])):
                logger.info(f"Skipping {label_key[j]} - all test labels are NaN")
                continue

            if "Regressor" in model.params.metadata.model_type:
                if test_operator is not None:
                    operator = (
                        test_operator
                        if isinstance(test_operator[0], str)
                        else test_operator[j]
                    )
                    preds[:, j] = process_censor(test_y[:, j], preds[:, j], operator)

                store_split_scores(
                    performance_dict,
                    i,
                    label_key[j],
                    process_regression,
                    n_bootstrap,
                    frac_bootstrap,
                    test_y[:, j],
                    preds[:, j],
                    log10=False,
                )

                df_preds = pd.DataFrame(
                    {"smiles": test_smi, "true": test_y[:, j], "pred": preds[:, j]}
                )

                try:
                    store_split_scores(
                        performance_dict_log10,
                        i,
                        label_key[j],
                        process_regression,
                        n_bootstrap,
                        frac_bootstrap,
                        test_y[:, j],
                        preds[:, j],
                        log10=True,
                    )

                    df_scaled = pd.DataFrame(
                        {
                            "true_log10": np.log10(test_y[:, j]),
                            "pred_log10": np.log10(preds[:, j]),
                        }
                    )
                    df_preds = pd.concat([df_preds, df_scaled], axis=1)
                except Exception as e:
                    logger.error(f"Error processing log10 for {label_key[j]}: {e}")

                df_preds.to_csv(
                    f"{cfg.output.serialization.path}/{label_key[j]}_{i}_preds.csv",
                    index=False,
                )

                # Create plots directory
                plots_dir = os.path.join(cfg.output.serialization.path, "plots")
                os.makedirs(plots_dir, exist_ok=True)

                # Plot unscaled true vs pred
                fig = plot_regression(
                    test_y[:, j],
                    preds[:, j],
                    plot_title=f"{label_key[j]} - Split {i} (Unscaled)",
                    labels=test_smi,
                    is_log10=False,
                )
                save_plot(
                    fig, os.path.join(plots_dir, f"{label_key[j]}_{i}_unscaled.html")
                )
                logger.info("Saved unscaled plot")

                # Plot log10 scaled true vs pred (if log10 data is available)
                if (
                    "true_log10" in df_preds.columns
                    and "pred_log10" in df_preds.columns
                ):
                    fig = plot_regression(
                        df_preds["true_log10"].values,
                        df_preds["pred_log10"].values,
                        plot_title=f"{label_key[j]} - Split {i} (Log10)",
                        labels=test_smi,
                        is_log10=True,
                    )
                    save_plot(
                        fig, os.path.join(plots_dir, f"{label_key[j]}_{i}_log10.html")
                    )
                    logger.info("Saved log10 plot")

            if "Classifier" in model.params.metadata.model_type:
                store_split_scores(
                    performance_dict,
                    i,
                    label_key[j],
                    process_classification,
                    n_bootstrap,
                    frac_bootstrap,
                    test_y[:, j],
                    preds[:, j],
                    probs[:, j],
                )

                df_preds = pd.DataFrame(
                    {
                        "smiles": test_smi,
                        "true": test_y[:, j],
                        "pred": preds[:, j],
                        "probs": probs[:, j],
                    }
                )

                df_preds.to_csv(
                    f"{cfg.output.serialization.path}/{label_key[j]}_{i}_preds.csv",
                    index=False,
                )

                plots_dir = os.path.join(cfg.output.serialization.path, "plots")
                os.makedirs(plots_dir, exist_ok=True)

                fig = plot_classification(
                    df_preds["true"].values,
                    df_preds["pred"].values,
                    df_preds["probs"].values,
                    plot_title=f"{label_key[j]} - Split {i} (Classification)",
                )
                save_plot(fig, os.path.join(plots_dir, f"{label_key[j]}_{i}_clf.html"))

    performance_dict["mean"] = aggregate_scores(performance_dict, "mean")
    performance_dict_log10["mean"] = aggregate_scores(performance_dict_log10, "mean")

    performance_dict["std"] = aggregate_scores(performance_dict, "std")
    performance_dict_log10["std"] = aggregate_scores(performance_dict_log10, "std")

    logger.info("Evaluation finished")
    save_json(f"{cfg.output.serialization.path}/performance.json", performance_dict)
    save_json(
        f"{cfg.output.serialization.path}/performance_log10.json",
        performance_dict_log10,
    )
    save_config_as_yaml(cfg, f"{cfg.output.serialization.path}/cfg.yaml")

    if cfg.output.mlflow is not None:
        flat_metrics = flatten_performance_dict(performance_dict, "")
        flat_metrics_log10 = flatten_performance_dict(performance_dict_log10, "log10")

        mlflow_logger = MatchaLogger(
            experiment_name=cfg.output.mlflow.experiment_name,
            run_name=f"{run_name}_metrics",
            save_dir=cfg.output.mlflow.log_dir,
            tracking_uri=None,
            log_model=True,
        )
        mlflow_logger.experiment.log_artifact(
            mlflow_logger._run_id, f"{cfg.output.serialization.path}/performance.json"
        )
        mlflow_logger.experiment.log_artifact(
            mlflow_logger._run_id,
            f"{cfg.output.serialization.path}/performance_log10.json",
        )

        for fname in os.listdir(cfg.output.serialization.path):
            if fname.endswith(".csv"):
                mlflow_logger.experiment.log_artifact(
                    mlflow_logger._run_id,
                    os.path.join(cfg.output.serialization.path, fname),
                )

        # Log plot artifacts
        plots_dir = os.path.join(cfg.output.serialization.path, "plots")
        if os.path.exists(plots_dir):
            for fname in os.listdir(plots_dir):
                if fname.endswith(".html"):
                    mlflow_logger.experiment.log_artifact(
                        mlflow_logger._run_id, os.path.join(plots_dir, fname)
                    )

        for key, value in {**flat_metrics, **flat_metrics_log10}.items():
            mlflow_logger.experiment.log_metric(mlflow_logger._run_id, key, value)

        mlflow_logger.conclude_experiment()


if __name__ == "__main__":
    main()
