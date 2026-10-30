"""CLI command for stitching multiple molecular datasets into one.

Loads individual datasets (CSV/SDF), merges them on a shared SMILES
column into a single wide-format DataFrame with one column per endpoint,
and saves the result as CSV.
"""

from matcha.cli.utils import load_dataset, save_config_as_yaml
from matcha.nn.multitask import stitch_datasets
import argparse
import yaml
import os
from matcha.utils.logging import get_default_logger
from matcha.utils.schemas.cli import CLIStitcherInputModel, Dataset


def main(cfg=None):
    """Stitch multiple molecular datasets into a single multi-task CSV.

    Loads each input dataset, selects the relevant columns (SMILES, label,
    optionally operator and index), then merges them via
    :func:`~matcha.nn.multitask.stitch_datasets` into a wide-format
    DataFrame with a shared SMILES column and one label column per endpoint.

    :param cfg: Pre-parsed configuration object or ``None`` to parse from
        CLI ``--config`` argument. Accepts a
        :class:`~matcha.utils.schemas.cli.CLIStitcherInputModel` instance
        or a raw dict that will be validated.
    """

    if cfg is None:
        parser = argparse.ArgumentParser(description="Stitch datasets")
        parser.add_argument(
            "--config", type=str, required=True, help="Path to the YAML config file"
        )
        args = parser.parse_args()
        with open(args.config, "r") as f:
            raw = yaml.safe_load(f)
        cfg = CLIStitcherInputModel.model_validate(raw)
    elif not isinstance(cfg, CLIStitcherInputModel):
        cfg = CLIStitcherInputModel.model_validate(cfg)
    log_prefix = os.path.splitext(cfg.output.filename)[0]
    logging_path = f"{cfg.output.folder_path}/{log_prefix}_stitcher.log"
    logger = get_default_logger("STITCHER", logging_path=logging_path)

    if cfg.input.dataset_names == "auto":
        dataset_names = os.listdir(cfg.input.folder_path)
        paths = [f"{cfg.input.folder_path}/{x}" for x in dataset_names]
    else:
        paths = [f"{cfg.input.folder_path}/{x}" for x in cfg.input.dataset_names]

    if isinstance(cfg.input.label_keys, str):
        label_keys = [cfg.input.label_keys] * len(paths)
    else:
        label_keys = cfg.input.label_keys

    dfs = []
    for i, path in enumerate(paths):
        dataset_config = Dataset(
            path=path, smiles_key=cfg.input.smiles_key, label_key=""
        )
        logger.info(f"Loading dataset: {dataset_config.path}")
        df = load_dataset(dataset_config)

        keys = [cfg.input.smiles_key, label_keys[i]]

        if cfg.input.operator_key is not None:
            keys.append(cfg.input.operator_key)

        if cfg.input.index_key is not None:
            keys.append(cfg.input.index_key)

        df = df[keys]
        df[label_keys[i]] = df[label_keys[i]].astype(float)
        dfs.append(df)

    logger.info("Stitching datasets")
    df_final = stitch_datasets(
        df_list=dfs,
        property_list=label_keys,
        smiles_key=cfg.input.smiles_key,
        bound_key=cfg.input.operator_key,
        index_key=cfg.input.index_key,
        tag=cfg.input.tag,
    )

    logger.info(f"Saving output at {cfg.output.folder_path}/{cfg.output.filename}")
    os.makedirs(cfg.output.folder_path, exist_ok=True)
    df_final.to_csv(f"{cfg.output.folder_path}/{cfg.output.filename}", index=False)
    save_config_as_yaml(cfg, f"{cfg.output.folder_path}/{cfg.output.filename}_cfg.yaml")
    logger.info("Finished")


if __name__ == "__main__":
    main()
