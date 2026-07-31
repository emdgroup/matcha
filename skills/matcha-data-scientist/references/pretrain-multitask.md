# pretrain-multitask

**CLI command:** `matcha pretrain_multitask --config pretrain_multitask.yaml`

Trains a MATCHA neural-network (typically a graph model such as `GatedGCNRegressor`) on a sparse multi-task dataset produced by `matcha prepare_sparse_dataset`. Supports multiple heterogeneous loss functions with independent curriculum weighting schedules, distributed training via DDP, and optional MLflow logging. The resulting checkpoint is consumable by `FinetuningRegressor` / `FinetuningClassifier` in downstream `matcha train` runs.

---

## YAML Schema (`CLIPretrainMultitaskInputModel`)

```yaml
dataset:
  dataset_dir: <str>         # required — output directory from prepare_sparse_dataset
  # Optional per-file overrides (omit to use defaults resolved from dataset_dir):
  train_smiles: <str>        # default: dataset_dir/train_molecules.parquet
  val_smiles: <str>          # default: dataset_dir/val_molecules.parquet
  train_tasks: <str>         # default: dataset_dir/train_tasks.npz
  val_tasks: <str>           # default: dataset_dir/val_tasks.npz
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
    loss_fn: <str>           # "mse" (regression), "focal" / "bce" (classification)
    loss_args: <dict>        # {} for mse; {"gamma": 2.0} for focal
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
Ask: "What is the output directory from your `prepare_sparse_dataset` run?"

Set `dataset.dataset_dir`. Individual file paths are resolved automatically unless you need to override them.

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

- Regression tasks → `loss_fn: mse`, `loss_args: {}`
- Classification tasks → `loss_fn: focal`, `loss_args: {gamma: 2.0}`

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
    loss_fn: focal
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

## Key Behaviors

- **`dataset` key in `loss` block:** Must match keys from `task_metadata.json → file_to_tasks` (file paths without extension).
- **DDP multi-GPU:** Set `pipe.visible_devices: "0,1"` — device count is inferred from the comma-separated list.
- **Finetuning chain:** After pretraining, pass `output.serialization` as `model.path_to_pretrained` in a `train` config with `model.architecture: FinetuningRegressor` or `FinetuningClassifier`.
