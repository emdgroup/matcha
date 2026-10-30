"""CLI command for running inference with a trained MATCHA model.

Loads a serialized model, reads an input dataset (CSV or SDF), runs
predictions and uncertainty estimation, and writes output CSV files.
"""

from matcha.sklearn.autoload import autoload
from rdkit.Chem import PandasTools, MolFromSmiles, MolToSmiles
import argparse
import yaml
import pandas as pd
from matcha.sklearn import Ensemble
import os
import json
from matcha.utils.schemas.cli import CLIPredictInputModel
from matcha.utils.logging import get_default_logger
from matcha.cli.utils import save_config_as_yaml


def main(cfg=None):
    """Run predictions on a dataset using a serialized MATCHA model.

    Loads the model from disk, converts input molecules to RDKit ``Mol``
    objects, runs batch inference (predictions and uncertainty), parses the
    raw output into labeled columns, and saves results as CSV.

    :param cfg: Pre-parsed configuration object or ``None`` to parse from
        CLI ``--config`` argument. Accepts a
        :class:`~matcha.utils.schemas.cli.CLIPredictInputModel` instance or
        a raw dict that will be validated.
    """

    if cfg is None:
        parser = argparse.ArgumentParser(description="Run predictions")
        parser.add_argument(
            "--config", type=str, required=True, help="Path to the YAML config file"
        )
        args = parser.parse_args()
        with open(args.config, "r") as f:
            raw = yaml.safe_load(f)
        cfg = CLIPredictInputModel.model_validate(raw)
    elif not isinstance(cfg, CLIPredictInputModel):
        cfg = CLIPredictInputModel.model_validate(cfg)
    log_path = f"{cfg.output}/predict.log"
    logger = get_default_logger("PREDICT", logging_path=log_path)

    logger.info(f"Loading data | path: {cfg.dataset.path}")
    if ".csv" in cfg.dataset.path:
        df = pd.read_csv(cfg.dataset.path)
        df["ROMol"] = df[cfg.dataset.smiles_key].apply(MolFromSmiles)

    elif ".sdf" in cfg.dataset.path:
        df = PandasTools.LoadSDF(cfg.dataset.path)

    if cfg.dataset.keep_cols:
        pass
    else:
        if cfg.dataset.smiles_key is not None:
            df = df[["ROMol", cfg.dataset.smiles_key]]
        else:
            df = df[["ROMol"]]
            df["SMILES"] = df.ROMol.apply(MolToSmiles)

    failed = df[df.ROMol.isnull()]
    df = df.dropna(inplace=False, subset="ROMol")
    mols = df.ROMol.tolist()

    logger.info(f"Loading model | path: {cfg.model.path}")
    model = autoload(cfg.model.path)

    logger.info(f"Model config: {json.dumps(model.params.model_dump(), indent=2)}")

    logger.info("Beginning inference:")
    logger.info(f"    accelerator: {cfg.model.inference.accelerator}")
    logger.info(f"    devices: {cfg.model.inference.devices}")
    logger.info(f"    batch size: {cfg.model.inference.batch_size}")
    if isinstance(model, Ensemble):
        preds, std = model.predict(
            mols,
            accelerator=cfg.model.inference.accelerator,
            devices=cfg.model.inference.devices,
            batch_size=cfg.model.inference.batch_size,
        )
    else:
        preds = model.predict(
            mols,
            accelerator=cfg.model.inference.accelerator,
            devices=cfg.model.inference.devices,
            batch_size=cfg.model.inference.batch_size,
        )
        std = model.compute_uncertainty(
            mols,
            accelerator=cfg.model.inference.accelerator,
            devices=cfg.model.inference.devices,
            batch_size=cfg.model.inference.batch_size,
        )

    logger.info("Parsing outputs")
    df_preds_raw = model.parse_output(preds, tag="prediction", convert_to_labels=False)
    df_std = model.parse_output(std, tag="std", convert_to_labels=False)

    if model.has_class_labels():
        df_preds_label = model.parse_output(preds, tag="label", convert_to_labels=True)
        df_out = pd.concat([df, df_preds_label, df_preds_raw, df_std], axis=1)
    else:
        df_out = pd.concat([df, df_preds_raw, df_std], axis=1)

    df_out = df_out.drop("ROMol", axis=1)
    failed = failed.drop("ROMol", axis=1)

    logger.info("Storing predictions")
    os.makedirs(cfg.output, exist_ok=True)
    df_out.to_csv(f"{cfg.output}/output.csv")
    failed.to_csv(f"{cfg.output}/failed.csv")
    save_config_as_yaml(cfg, f"{cfg.output}/cfg.yaml")
    logger.info("Finished")


if __name__ == "__main__":
    main()
