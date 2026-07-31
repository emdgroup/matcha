# `cli/` — command-line entry points

Thin drivers over the `../sklearn/` API, wired up as `matcha <command> --config <yaml>`. Every command is one file; every command has a matching pydantic schema in `../utils/schemas/cli.py` and a runnable example config in `example_configs/`.

## Layout

```text
cli/
├── __init__.py                   # COMMANDS dict + dispatcher (main())
├── utils.py                      # shared: load_dataset, get_splits, save_config_as_yaml, ...
├── train.py                      # fit a single or ensemble model
├── predict.py                    # inference from a saved model folder
├── evaluate.py                   # cross-validated performance metrics
├── autotune.py                   # Optuna HPO (arch + optimizer phases)
├── baseline.py                   # quick RF/XGB baselines for comparison
├── stitch.py                     # combine multiple training runs
├── summarize.py                  # collate metrics across runs
├── statistical_tests.py          # significance tests between models
├── prepare_sparse_dataset.py     # convert a wide multi-endpoint table to sparse form
├── pretrain_encoder.py           # single-task self-supervised pretraining
├── pretrain_multitask.py         # multi-task supervised pretraining
└── example_configs/              # one *.yaml per command — runnable examples
```

## Command dispatch

`__init__.py` defines a `COMMANDS` dict mapping command names to module paths and imports lazily. Adding a new command means: create `cli/<name>.py` with a `main(cfg=None)` entry point, add a line to `COMMANDS`, add a schema to `../utils/schemas/cli.py`, add an example config to `example_configs/`.

## Command shape

Every command follows the same skeleton:

1. Parse `--config` (or accept a pre-parsed cfg for programmatic use).
2. Validate against a `CLI*InputModel` schema from `../utils/schemas/cli.py`.
3. Load data via `cli.utils.load_dataset` and `../utils/serialization.parse_df`.
4. Instantiate an estimator through `ScikitLearnModelRegistry[class_name]` (or `Ensemble`).
5. Call one sklearn method (`fit`, `predict`, `autotune`, ...) — the CLI does no ML logic itself.
6. Persist via the estimator's serialization manager and echo the config via `save_config_as_yaml`.

`main(cfg=None)` is the convention — pre-parsed configs let tests drive commands without shelling out.

## Conventions

- **No ML logic in the CLI.** If you're writing a loop over epochs or computing metrics inline, it belongs in `../sklearn/` or `../sklearn/managers/`.
- **`example_configs/` is contract.** Users copy these; keep them minimal and current. If you add a required field, update every affected example.
- **`load_dataset` handles both CSV and SDF** — don't reinvent the reader per command.
- Repo-wide pydantic-boundary rule ([`PATTERNS.md`](../../PATTERNS.md) §4) applies: every CLI config is validated by a `CLI*InputModel` at the top of `main()`.
