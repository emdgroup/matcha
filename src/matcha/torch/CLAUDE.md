# `torch/` — Lightning model layer

Encoders, prediction heads, and the `LightningModule`s that combine them. The sklearn wrappers in `../sklearn/` are drivers over this layer; the CLI never touches it directly.

## Layout

```text
torch/
├── encoders/         # molecular representation → embedding (one file per architecture)
├── predictors/       # embedding → prediction (MLP, SNN heads)
├── models/
│   ├── classic/      # supervised end-to-end: encoder + predictor + BaseClassicModel
│   ├── finetuning/   # transfer learning: pretrained encoder + head, with LoRA support
│   ├── pretraining/  # self-supervised objectives (masked-graph, MLM)
│   └── mixin.py      # shared Lightning helpers (loss parsing, MC-dropout, metrics)
└── tuning/           # Optuna HPO routine + architecture / optimizer / scheduler grids (JSON)
```

## The encoder / predictor / model split

Three abstractions, cleanly separated:

- **Encoder** (`encoders/`, base: `BaseEncoder` / `BaseGraphEncoder`): takes a batch of molecular inputs (SMILES tokens, PyG graph, descriptor vector) and returns an embedding. Registered on `EncoderRegistry`. Inherits `HyperparametersMixin` and calls `save_hyperparameters()` — the serializer relies on it.
- **Predictor** (`predictors/`, base: `BasePredictor`): takes an embedding and returns predictions. Registered on `PredictorRegistry`. Currently MLP and SNN.
- **Model** (`models/classic/`, base: `BaseClassicModel`): a `LightningModule` that owns an encoder + predictor and drives training. Registered on `ClassicModelRegistry`. Argument prefixes matter: `enc_*` forwards to the encoder, `pred_*` to the predictor, bare names (`loss_fn`, `optimizer`, `scheduler`) resolve through registries in `../nn/`.

Pretraining and finetuning have their own base classes (`BasePretrainingModel`, `Finetuner`, `ChempropFinetuner`) and registries. LoRA lives in `models/finetuning/lora.py`.

## Registries defined here

`EncoderRegistry`, `PredictorRegistry`, `ClassicModelRegistry`, `PretrainingModelRegistry`. Optimizer / scheduler / loss / activation / readout / layer registries live in `../nn/` and are consumed here — new ones must be registered there before they can be referenced by string. See [`PATTERNS.md`](../../PATTERNS.md) §2.

## Tuning (`tuning/`)

`routine.py` runs a two-phase Optuna search: architecture first (grid in `architecture_grid.json`), then optimizer + scheduler (`optimizer_grid.json`, `scheduler_grid.json`). Callers come from `sklearn/managers/hpo_manager.py`, not from `torch/` directly. Grid JSONs are the extension point — no code change needed to add a new search dimension for an already-registered component.

## See also

- [`PATTERNS.md`](../../PATTERNS.md) §5 (`HyperparametersMixin` requirement) and §8 (prefix discipline) apply throughout this subpackage.
- Model-addition workflow: `docs/source/contributing/adding-a-model.md`.
