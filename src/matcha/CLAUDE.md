# Package guide — `src/matcha`

Orientation for the MATCHA package internals. Each subpackage has its own `CLAUDE.md` with the details — this file is just the map.

## What lives here

Users interact with a scikit-learn-style API (`fit` / `predict` / `predict_proba`) that wraps PyTorch Lightning models over descriptor, SMILES, and graph inputs. A thin CLI layers on top for training, prediction, tuning, and pretraining.

## Layout

```text
src/matcha/
├── sklearn/          # user-facing sklearn-style API — the main entry point
├── torch/            # Lightning modules, encoders, predictors, tuning grids
├── datamodules/      # Lightning DataModules (classic + pretraining)
├── nn/               # reusable layers, activations, losses, optimizers, schedulers, readouts
├── calibration/      # inductive conformal + error models
├── explainability/   # LIME, analogue generation (PAS + nitrogen walk)
├── utils/            # logging, metrics, plotting, registry, sanitize, serialization, splitting, schemas (pydantic)
├── cli/              # command-line entry points + example configs
├── __init__.py       # exposes `__version__` only
└── py.typed          # PEP 561 marker — package ships typed
```

## How the pieces fit

- **`sklearn/` is the front door.** Estimator classes per model family live in `sklearn/tabular/`, `sklearn/graph/`, `sklearn/graph3d/`, `sklearn/clm/`. Each family has a `base_sklearn_*.py` that concrete estimators subclass. Cross-cutting behavior — training, serialization, MLflow, HPO, uncertainty, explainability, ensembles — is factored into `sklearn/managers/` and composed onto the base, not inherited.
- **`torch/` is the model layer.** `encoders/` build representations, `predictors/` are heads (MLP / SNN), `models/` glues encoder + predictor into `LightningModule`s (`classic/` for supervised, `finetuning/` for adapters + LoRA, `pretraining/` for MLM / masked-graph objectives), `tuning/` holds HPO search grids as JSON.
- **`datamodules/`** wraps input data as Lightning DataModules — `classic/` for supervised, `pretraining/` for self-supervised. Featurization helpers (RDKit engine, label encoder, positional encoders) live alongside.
- **`nn/`, `utils/`, `calibration/`, `explainability/`** are building blocks pulled in by the layers above. `utils/schemas/` holds the pydantic contracts crossing sklearn ↔ torch ↔ CLI; `utils/registry.py` provides the `ClassRegistry` used for string-alias lookup everywhere.
- **`cli/`** is the outermost layer — every command is a thin driver over a `sklearn/` estimator, configured from YAML files matching the shapes in `cli/example_configs/`.

Dependency direction (rough): `cli → sklearn → torch + datamodules → nn + utils`. `calibration/` and `explainability/` plug into `sklearn/managers/` and are exposed transparently through the estimator API.

## Where to look next

- **Repo-wide patterns** (base + concrete, registries, serialization aliases, pydantic boundaries, `HyperparametersMixin`, `locals()` capture, managers, prefix discipline, tests mirror source): see [`../../PATTERNS.md`](../../PATTERNS.md). Every subpackage assumes these.
- **Subpackage internals:** each subfolder has its own `CLAUDE.md`.
- **Cross-cutting workflows** (adding a model / manager / datamodule / calibrator): see `docs/source/contributing/`.
- **Python style:** [`PATTERNS.md`](../../PATTERNS.md) §10 for project rules not caught by pre-commit; `.pre-commit-config.yaml` for what Ruff / Pyright enforce automatically.
