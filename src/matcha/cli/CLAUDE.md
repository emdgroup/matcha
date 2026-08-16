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
├── prepare_dataset.py            # convert a wide multi-endpoint table to sparse or dense form
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

## Multitask pretraining data storage modes

`prepare_dataset` (the sole command name — the previous `prepare_sparse_dataset` name has been dropped, without an alias) emits either a **sparse** CSR artifact or a **dense** `float32` array, controlled by `datasets.sparse` in the input config (default `True` preserves prior behavior):

- **Sparse mode** — writes `train_tasks_sparse.npz` / `val_tasks_sparse.npz`. Classification labels are remapped `0 → -1` at prep time so zero-omission in the CSR encodes "missing" unambiguously; the collate side re-inverts (`0 → NaN`, `-1 → 0`) at batch construction.
- **Dense mode** — writes `train_tasks_dense.npy` / `val_tasks_dense.npy`. `NaN` marks missing entries directly; classification passes through as `0`/`1` with no remap. Choose this when the underlying label matrix is largely populated and CSR overhead outweighs the storage savings.

Auto-detection contract: `task_metadata.json` carries a `storage_mode: "sparse" | "dense"` field. `pretrain_multitask` reads it up front and dispatches to the right loader and artifact filenames — users do **not** set any additional flag on the pretraining side. Both modes converge on the same `(B, T)` `float32` tensor with `NaN` marking missing at the collate boundary, so loss modules and model heads see an identical batch shape either way.

Validation-split sampling **intentionally diverges** between the two modes, and the divergence is load-bearing — do not unify:

- **Sparse mode** — per-task sampling with per-task `min_compounds` floor and OR aggregation across tasks. With most cells missing the per-task samplers rarely overlap, so the union stays close to `sampling_rate * n_compounds` while guaranteeing every task sees at least `min_compounds` validation rows.
- **Dense mode** — single global compound sample: `n_val = min(n_compounds, max(min_compounds, int(sampling_rate * n_compounds)))`, `min_compounds` acting as a global floor rather than a per-task guarantee. The sparse OR-aggregation strategy would drive per-compound inclusion probability toward `1 - (1 - sampling_rate)^n_tasks` on near-fully-populated matrices and blow up the val split, so dense uses the direct compound fraction instead.

Both paths remain seeded reproducibly through `validation.seed`.
