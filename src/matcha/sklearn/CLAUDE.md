# `sklearn/` — the user-facing API

Scikit-learn-style wrappers (`fit`, `predict`, `predict_proba`) over the Lightning models in `../torch/`. This is the layer users import and the layer the CLI drives.

## Layout

```text
sklearn/
├── base_sklearn_model.py   # BaseScikitLearnModel + Classifier/Regressor mixins + ScikitLearnModelRegistry
├── ensemble.py             # Ensemble — wraps N estimators with KFold
├── finetuner.py            # FinetuningClassifier / FinetuningRegressor over pretrained encoders
├── autoload.py             # autoload(path) — reads config/manifest.yaml → registry → model
├── tabular/                # MLP, SNN (base_sklearn_tabular.py + one file per model)
├── graph/                  # 2-D GNNs (GIN, GT, GPS, GatedGCN, AttentiveFP, Chemprop)
├── graph3d/                # 3-D GNNs (E3GNN, GT3D, GPS3D) — inherits BaseScikitLearnGNN
├── clm/                    # chemical language models (CNN, RNN, RoFormer)
├── managers/               # cross-cutting behavior (see below)
└── utils.py
```

## The three inheritance dimensions

Every concrete estimator picks one from each axis:

1. **Family base** (data shape) — `BaseScikitLearnTabular`, `BaseScikitLearnGNN`, `BaseScikitLearnGNN3D`, `BaseScikitLearnCLM`. Owns the datamodule choice and featurizer.
2. **Task mixin** — `ScikitLearnClassifierMixin` or `ScikitLearnRegressorMixin`. Owns `predict_proba`, decision thresholds, metric defaults. Every model ships **both** halves.
3. **Concrete class** — pairs a `torch/models/classic/*Model` with hyperparameter defaults tuned for that architecture.

Concrete classes are typically ~150 lines: a docstring, an `__init__` using the `locals()`-capture idiom (see [`PATTERNS.md`](../../PATTERNS.md) §6), and `super().__init__(params)`. All logic lives in the base + managers.

## Managers (`managers/`)

`BaseScikitLearnModel` composes cross-cutting concerns via managers rather than deep inheritance. Each manager owns one responsibility:

| Manager | Responsibility |
| --- | --- |
| `DataModuleManager` | Build the right Lightning DataModule from user input |
| `TrainingManager` (+ `CLM`, `Finetuner` variants) | Configure Trainer, callbacks, fit/predict loops |
| `SerializationManager` | Write / read `config/`, `weights/`, `manifest.yaml` |
| `MLFlowManager` | Optional MLflow logging |
| `UncertaintyManager` | Conformal / error-model calibration hookup |
| `ExplainabilityManager` | LIME + analogue-generation entry points |
| `HPOManager` | Optuna-based tuning driver |
| `Ensemble{MLFlow, Serialization, Calibration}Manager` | Ensemble-aware variants of the above |

Ensemble managers shadow their single-model counterparts because ensembles serialize N children under one folder — adding a feature to the non-ensemble manager usually requires touching the ensemble manager too.

## Serialization contract

`SerializationManager` writes a folder with `config/manifest.yaml` recording `class_name`, `matcha_version`, and metadata. `autoload(path)` reads the manifest and dispatches through `ScikitLearnModelRegistry[class_name].from_folder(...)`. See [`PATTERNS.md`](../../PATTERNS.md) §3 on why the alias must stay stable.

## Adding a new estimator

Walkthrough in `docs/source/contributing/adding-a-model.md`. Ship both `Classifier` and `Regressor` and export from `sklearn/__init__.py`.
