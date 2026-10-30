"""CLI command for scikit-learn baseline evaluation.

Evaluates simple scikit-learn models (Random Forest, SVM, KNN, etc.) with
RDKit molecular descriptors as features, using the same splitting and
metrics infrastructure as the MATCHA evaluate command.
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
from matcha.cli.evaluate import flatten_performance_dict
from matcha.utils.schemas.cli import CLIBaselineInputModel
from matcha.utils.serialization import parse_df, save_json
from matcha.utils.logging import get_default_logger, MatchaLogger
from matcha.datamodules.classic.label_transform import (
    ForwardTransformRegistry,
    BackwardTransformRegistry,
)
from matcha.datamodules.classic.label_encoder import BinaryClassificationLabelEncoder
from matcha.datamodules.classic.rdkit_engine import Engine
import argparse
import yaml
from rdkit import Chem
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.base import is_classifier
import json
import numpy as np
import pandas as pd
import os

model_registry = {
    "RandomForestRegressor": RandomForestRegressor,
    "RandomForestClassifier": RandomForestClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "SVC": SVC,
    "SVR": SVR,
    "KNeighborsClassifier": KNeighborsClassifier,
    "KNeighborsRegressor": KNeighborsRegressor,
    "LinearRegression": LinearRegression,
    "LogisticRegression": LogisticRegression,
}


def main(cfg=None):
    """Run baseline evaluation using scikit-learn models with molecular descriptors.

    For each cross-validation split, computes RDKit features, fits a
    scikit-learn model (e.g. Random Forest), evaluates predictions, and
    aggregates metrics. Supports both regression and classification
    endpoints with optional label transforms and bootstrapped confidence
    intervals.

    :param cfg: Pre-parsed configuration object or ``None`` to parse from
        CLI ``--config`` argument. Accepts a
        :class:`~matcha.utils.schemas.cli.CLIBaselineInputModel` instance
        or a raw dict that will be validated.
    """
    if cfg is None:
        parser = argparse.ArgumentParser(description="Run baseline evaluation")
        parser.add_argument(
            "--config", type=str, required=True, help="Path to the YAML config file"
        )
        args = parser.parse_args()
        with open(args.config, "r") as f:
            raw = yaml.safe_load(f)
        cfg = CLIBaselineInputModel.model_validate(raw)
    elif not isinstance(cfg, CLIBaselineInputModel):
        cfg = CLIBaselineInputModel.model_validate(cfg)
    log_path = (
        f"{cfg.output.serialization.path}/baseline.log"
        if cfg.output.serialization is not None
        else None
    )
    logger = get_default_logger("BASELINE", logging_path=log_path)

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

    for i in range(len(train_splits)):
        logger.info(f"Processing split {i}")

        logger.info("Logging splits")
        train_splits[i].to_csv(
            f"{cfg.output.serialization.path}/train_split_{i}.csv", index=False
        )
        test_splits[i].to_csv(
            f"{cfg.output.serialization.path}/test_split_{i}.csv", index=False
        )

        train_mols, train_y, _ = parse_df(
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

        logger.info(f"Generating features: {cfg.model.feature_list}")
        engine = Engine(n_jobs=cfg.model.n_jobs)
        train_x = engine.get_features(train_mols, cfg.model.feature_list)
        test_x = engine.get_features(test_mols, cfg.model.feature_list)

        for j in range(train_y.shape[1]):
            # skip this index j if all test labels are NaN
            if np.all(np.isnan(test_y[:, j])):
                logger.info(f"Skipping {label_key[j]} - all test labels are NaN")
                continue

            logger.info(f"Processing endpoint {j}")

            logger.info(f"Generating {cfg.model.algorithm}:")
            logger.info(f"  Model parameters: {json.dumps(cfg.model.params, indent=2)}")
            try:
                model = model_registry[cfg.model.algorithm](
                    random_state=i, **cfg.model.params
                )
            except Exception:
                logger.warning(
                    "Could not set seed, using deterministic training. Performance variance might be underestimated"
                )
                model = model_registry[cfg.model.algorithm](**cfg.model.params)

            logger.info(f"Beginning {cfg.model.algorithm} model fit")
            valid_mask = ~np.isnan(train_y[:, j])
            train_x_clean = train_x[valid_mask]
            train_y_clean = train_y[valid_mask, j]

            if not is_classifier(model):
                if cfg.model.label_transform is not None:
                    train_y_clean = ForwardTransformRegistry.scale(
                        train_y_clean, cfg.model.label_transform
                    )

                model.fit(train_x_clean, train_y_clean)
                logger.info(f"{cfg.model.algorithm} model successfully fit")
                preds = model.predict(test_x)
                logger.info("Predictions calculated")

                if cfg.model.label_transform is not None:
                    preds = BackwardTransformRegistry.scale(
                        preds, cfg.model.label_transform
                    )

                logger.info(f"Computing metrics for {label_key[j]}")
                if test_operator is not None:
                    operator = (
                        test_operator
                        if isinstance(test_operator[0], str)
                        else test_operator[j]
                    )
                    preds = process_censor(test_y[:, j], preds, operator)

                store_split_scores(
                    performance_dict,
                    i,
                    label_key[j],
                    process_regression,
                    n_bootstrap,
                    frac_bootstrap,
                    test_y[:, j],
                    preds,
                    log10=False,
                )

                df_preds = pd.DataFrame(
                    {"smiles": test_smi, "true": test_y[:, j], "pred": preds}
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
                        preds,
                        log10=True,
                    )

                    df_scaled = pd.DataFrame(
                        {
                            "true_log10": np.log10(test_y[:, j]),
                            "pred_log10": np.log10(preds),
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
                    preds,
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

            if is_classifier(model):
                if len(np.unique(train_y_clean)) != 2:
                    label_encoder = BinaryClassificationLabelEncoder(
                        cfg.model.label_encoder_params
                    )
                    train_y_clean = label_encoder._continuous_to_categorical(
                        train_y_clean, j
                    )
                    test_y[:, j] = label_encoder._continuous_to_categorical(
                        test_y[:, j], j
                    )[:, 0]

                model.fit(train_x_clean, train_y_clean)
                logger.info(f"{cfg.model.algorithm} model successfully fit")
                preds = model.predict(test_x)
                probs = model.predict_proba(test_x)[:, 1]
                logger.info("Predictions calculated")

                store_split_scores(
                    performance_dict,
                    i,
                    label_key[j],
                    process_classification,
                    n_bootstrap,
                    frac_bootstrap,
                    test_y[:, j],
                    preds,
                    probs,
                )

                df_preds = pd.DataFrame(
                    {
                        "smiles": test_smi,
                        "true": test_y[:, j],
                        "pred": preds,
                        "probs": probs,
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

    logger.info("Baseline evaluation finished")
    logger.info(f"Output will be logged at {cfg.output.serialization.path}")
    save_json(f"{cfg.output.serialization.path}/performance.json", performance_dict)
    save_json(
        f"{cfg.output.serialization.path}/performance_log10.json",
        performance_dict_log10,
    )
    save_config_as_yaml(cfg, f"{cfg.output.serialization.path}/cfg.yaml")

    if cfg.output.mlflow is not None:
        flat_metrics = flatten_performance_dict(performance_dict, "")
        flat_metrics_log10 = flatten_performance_dict(performance_dict_log10, "log10")

        if cfg.output.mlflow.run_name is None:
            run_name = "baseline"
        else:
            run_name = cfg.output.mlflow.run_name

        logger.info(
            f"Output will be logged in MLFlow under {cfg.output.mlflow.experiment_name} as {run_name}"
        )
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
