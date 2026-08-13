# pretrain-multitask

**CLI command:** `matcha pretrain_multitask --config pretrain_multitask.yaml`

Trains a MATCHA neural-network (typically a graph model such as `GatedGCNRegressor`) on a multi-task dataset produced by `matcha prepare_dataset`. Handles both sparse (CSR `.npz`) and dense (`.npy` with `NaN` for missing) label artifacts — the storage mode is auto-detected from `task_metadata.json` (`storage_mode: "sparse" | "dense"`), and users do not need to configure it here. Supports multiple heterogeneous loss functions with independent curriculum weighting schedules, distributed training via DDP, and optional MLflow logging. If `train_coords.npz` / `val_coords.npz` are dropped alongside the other artifacts in `dataset_dir`, they are auto-loaded and forwarded to 3D-aware base datamodules — see [Coords Auto-Discovery](#coords-auto-discovery). The resulting checkpoint is consumable by `FinetuningRegressor` / `FinetuningClassifier` in downstream `matcha train` runs.

---

## YAML Schema (`CLIPretrainMultitaskInputModel`)

```yaml
dataset:
  dataset_dir: <str>         # required — output directory from prepare_dataset
  # Optional per-file overrides (omit to use defaults resolved from dataset_dir):
  train_smiles: <str>        # default: dataset_dir/train_molecules.parquet
  val_smiles: <str>          # default: dataset_dir/val_molecules.parquet
  train_tasks: <str>         # default: dataset_dir/train_tasks_{sparse.npz|dense.npy} per task_metadata.storage_mode
  val_tasks: <str>           # default: dataset_dir/val_tasks_{sparse.npz|dense.npy} per task_metadata.storage_mode
  task_metadata: <str>       # default: dataset_dir/task_metadata.json

model:
  architecture: <str>        # required — graph model, e.g. "GatedGCNRegressor"
  params:                    # neural-network constructor args
    enc_num_layers: <int>
    enc_atom_hidden_dim: <int>
    enc_dropout: <float>
    pred_dropout: <float>
    pred_hidden_dims: <list[int]>
    pred_activation: <str>
    enc_activation: <str>
  datamodule:                # optional
    batch_size: <int>        # default: 256
  training:                  # optional
    num_epochs: <int>        # default: 50
    early_stopping: <bool>   # default: true
    patience: <int>          # default: 10
    accelerator: <str>       # "gpu" or "cpu" (default: "gpu")
    devices: <int>           # default: 1
    seed: <int>              # default: 0

loss:                        # one entry per source dataset (key in task_metadata.json file_to_tasks)
  - dataset: <str>           # dataset key matching a stem from file_to_tasks (strip extension)
    loss_fn: <str>           # any LossRegistry alias — see "Loss configuration" below
    loss_args: <dict>        # constructor args for the loss (e.g. {} for mse, {"gamma": 2.0} for focal-bce)
    init_w: <float>          # initial curriculum weight
    final_w: <float>         # final weight at end of schedule
    T: <float>               # schedule length in epochs (1.0 = constant)
    warmup: <float>          # fraction of T used for linear warm-up (0.0 = no warm-up)

pipe:
  visible_devices: <str>               # optional — e.g. "0" or "0,1,2,3"
  strategy:
    timeout_seconds: <int>             # DDP timeout (default: 3600)
    find_unused_parameters: <bool>     # default: false
  dataloader_num_workers: <int>        # default: 8
  fit_datamodule_size: <int>           # optional
  gradient_accumulation_steps: <int>   # default: 1
  gradient_clip_val: <float>           # default: 0.0; 1.0 is a safe starting value

output:
  serialization: <str>       # required — output model directory
  mlflow:                    # optional
    experiment_name: <str>
    run_name: <str>
    log_dir: <str>
    server_uri: <str>
    tags: <dict>
```

---

## Step-by-Step Config Generation

### 1. Dataset directory
Ask: "What is the output directory from your `prepare_dataset` run?"

Set `dataset.dataset_dir`. Individual file paths (including the split task label artifact for the correct storage mode) are resolved automatically unless you need to override them.

### 2. Architecture and model params

Recommended architectures for multitask pretraining:

| Architecture | Best for |
|---|---|
| `GatedGCNRegressor` | Large-scale multi-task regression (default recommendation) |
| `GINRegressor` | Simpler graph model, faster training |

Typical starting params:
```yaml
params:
  enc_num_layers: 3
  enc_atom_hidden_dim: 256
  enc_dropout: 0.05
  pred_dropout: 0.19
  pred_hidden_dims: [256, 256]
  pred_activation: relu
  enc_activation: relu
```

### 3. Loss configuration
Ask: "How many distinct dataset groups are in your `task_metadata.json`? Look at `file_to_tasks` keys."

Each key in `task_metadata.json → file_to_tasks` (file path without extension) needs one `loss` entry.

**Loss options.** `loss_fn` accepts any alias registered in `LossRegistry` — see [`docs/source/contributing/adding-a-loss.md`](../../../docs/source/contributing/adding-a-loss.md) for the authoritative alias list. Common families:

- **Base regression:** `mse`, `mae`, `huber`, `smoothl1`.
- **Base classification:** `bce`, `focal-bce`, `poly1-bce`, `weighted-bce`.
- **`bounded-*` family** — respects `<` / `>` bound info on the target so predictions inside the allowed halfspace are not penalized. Aliases: `bounded-mse`, `bounded-mae`, `bounded-huber` (+ `bounded-smoothl1`).
- **`dropout-*` family** — wraps a per-element loss and resamples a random mask of loss entries every training step, as a regularizer for wide multi-endpoint pretraining. Aliases: `dropout-mse`, `dropout-bce`, `dropout-focal-bce` (+ `dropout-mae`, `dropout-huber`, `dropout-smoothl1`, `dropout-poly1-bce`, `dropout-weighted-bce`). Pass `loss_args: {dropout: <fraction in [0, 1)>}` to control the mask rate.

Typical starting points:

- Regression tasks → `loss_fn: mse`, `loss_args: {}`. Switch to `bounded-mse` when the source has `<` / `>` operator columns; switch to `dropout-mse` for wide multi-endpoint regression pretraining.
- Classification tasks → `loss_fn: focal-bce`, `loss_args: {gamma: 2.0}`.

### 4. Training settings
- `training.num_epochs: 20` for exploration; 50–100 for production.
- `pipe.gradient_clip_val: 1.0` is recommended.

### 5. Generate YAML, confirm, and run

```
matcha pretrain_multitask --config pretrain_multitask.yaml
```

After training, remind the user:

> "Pretraining is done. To finetune on a downstream task, use `model.architecture: FinetuningRegressor` (or `FinetuningClassifier`) and set `model.path_to_pretrained: <output.serialization>` in your `matcha train` config."

---

## Example Config

```yaml
dataset:
  dataset_dir: ./data/prepared_pretraining

model:
  architecture: GatedGCNRegressor
  params:
    enc_num_layers: 3
    enc_atom_hidden_dim: 256
    enc_dropout: 0.05
    pred_dropout: 0.19
    pred_hidden_dims: [256, 256]
    pred_activation: relu
    enc_activation: relu
  datamodule:
    batch_size: 256
  training:
    num_epochs: 20
    early_stopping: true
    patience: 10
    accelerator: gpu
    devices: 1
    seed: 42

loss:
  - dataset: adme_regression
    loss_fn: mse
    loss_args: {}
    init_w: 1.0
    final_w: 1.0
    T: 1.0
    warmup: 0.0
  - dataset: tox_classification
    loss_fn: focal-bce
    loss_args:
      gamma: 2.0
    init_w: 1.0
    final_w: 1.0
    T: 1.0
    warmup: 0.0

pipe:
  visible_devices: "0"
  strategy:
    timeout_seconds: 3600
    find_unused_parameters: false
  dataloader_num_workers: 8
  gradient_clip_val: 1.0

output:
  serialization: ./models/pretrained_multitask
```

---

## Coords Auto-Discovery

`pretrain_multitask` auto-loads per-molecule 3D coordinates from a filesystem-only convention. There is **no schema flag** — the switch is purely presence-of-file:

- If `{dataset_dir}/train_coords.npz` and `{dataset_dir}/val_coords.npz` both exist, they are read via `_load_coords_npz` (see [`src/matcha/cli/utils.py`](../../../src/matcha/cli/utils.py)) and threaded through `OnTheFlyDataModule.set_data(..., train_coords=..., val_coords=...)`.
- The datamodule probes whether the wrapped base datamodule (e.g. `Graph3DDataModule`) accepts a `coords` kwarg on `generate_features`. If it does, coords are forwarded per batch. If it does not (a 2D-only base such as `GraphDataModule`), coords are **silently dropped** with a one-shot `logging.warning` — no error, and pretraining continues with 2D features only.
- If either coords file is missing, both are treated as absent (no partial coverage — you cannot train with coords for one split and none for the other).

> **Coords are user-supplied.** `matcha prepare_dataset` does **not** emit `train_coords.npz` / `val_coords.npz`. Users must produce them externally (e.g. RDKit ETKDG, OpenFF, xTB, ORCA) and drop them into `dataset_dir` alongside the other artifacts. If they do not have coordinates ready, they can still run `pretrain_multitask` on a 2D base and coords will simply not be loaded.

**Coords npz contract** (enforced by `_load_coords_npz` at CLI startup — failures are `AssertionError`s that fail fast before featurization):

- Keys: `flat` (2D `float32`, shape `(N_atoms, 3)`) and `offsets` (1D int, length `N + 1`).
- `offsets` must be monotonic non-decreasing.
- `offsets[-1] == flat.shape[0]`.
- Each molecule's slice `flat[offsets[i]:offsets[i+1]]` is an `(A_i, 3)` `float32` array.

Coords must index the same molecules in the same order as `train_smiles` / `val_smiles`.

To take advantage of coords, pair a 3D-aware architecture (e.g. `E3GNNRegressor`, `GPS3DRegressor`, `GT3DRegressor`) with a `Graph3DDataModule`-family base. Choosing a 2D architecture will trigger the silent-drop warning.

---

## Key Behaviors

- **`dataset` key in `loss` block:** Must match keys from `task_metadata.json → file_to_tasks` (file paths without extension).
- **DDP multi-GPU:** Set `pipe.visible_devices: "0,1"` — device count is inferred from the comma-separated list.
- **Coords auto-discovery:** `train_coords.npz` / `val_coords.npz` under `dataset_dir` are auto-loaded and forwarded to 3D-aware base datamodules; 2D bases silently drop them with a one-shot `logging.warning`. Coords are user-supplied — `prepare_dataset` does not emit them.
- **Finetuning chain:** After pretraining, pass `output.serialization` as `model.path_to_pretrained` in a `train` config with `model.architecture: FinetuningRegressor` or `FinetuningClassifier`.
