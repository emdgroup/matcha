# pretrain-encoder

**CLI command:** `matcha pretrain_encoder --config pretrain_encoder.yaml`

Pretrains a MATCHA encoder using one of two self-supervised objectives:

- **MLM (masked language modelling):** Trains a `RoFormerMLM` sequence model on SMILES. No task labels are needed — the model learns by reconstructing masked tokens.
- **Graph pretraining:** Trains a graph encoder (`GINPretraining`, `GatedGCNPretraining`, etc.) on node-level and graph-level targets. Requires pre-computed node and graph label arrays.

Both modes produce an `encoder.ckpt` artifact that can be loaded by `FinetuningRegressor` / `FinetuningClassifier` in downstream `matcha train` runs.

---

## Task-Type Branching

Choose the mode based on the target architecture:

| Goal | Mode | Architecture |
|---|---|---|
| Pretrain a sequence/transformer encoder | `mlm` | `RoFormerMLM` |
| Pretrain a graph encoder with QM/physical labels | `graph` | `GINPretraining`, `GatedGCNPretraining` |

Ask the user: "Do you want to pretrain a sequence model (MLM) or a graph model with node/graph labels (graph)?"

---

## YAML Schema (`CLIEncoderPretrainInputModel`)

### Shared skeleton

```yaml
dataset:
  task_type: <str>           # required — "mlm" or "graph"
  train_smiles: <str>        # required — path to parquet file with SMILES column
  val_smiles: <str>          # required — path to parquet file with SMILES column
  # graph mode only:
  train_y_graph: <str>       # path to npz file (key: "descriptors") — graph-level targets
  val_y_graph: <str>
  train_y_node: <str>        # path to npz file (keys: "flat", "offsets") — node-level targets
  val_y_node: <str>

model:
  architecture: <str>        # required — see tables below
  params: <dict>             # neural-network constructor args (mode-specific, see below)
  datamodule: <dict>         # optional — datamodule overrides (mode-specific, see below)
  training:                  # optional
    num_epochs: <int>        # default: 50
    early_stopping: <bool>   # default: true
    patience: <int>          # default: 10
    accelerator: <str>       # "gpu" or "cpu" (default: "gpu")
    devices: <int>           # default: 1
    seed: <int>              # default: 0

pipe:
  visible_devices: <str>
  strategy:
    timeout_seconds: <int>
    find_unused_parameters: <bool>
  dataloader_num_workers: <int>
  fit_datamodule_size: <int>
  gradient_accumulation_steps: <int>
  gradient_clip_val: <float>

output:
  serialization: <str>       # required — output directory
  mlflow:                    # optional
    experiment_name: <str>
    run_name: <str>
    log_dir: <str>
    server_uri: <str>
    tags: <dict>
```

---

## MLM Mode

### Dataset
```yaml
dataset:
  task_type: mlm
  train_smiles: ./data/train_smiles.parquet
  val_smiles: ./data/val_smiles.parquet
```

### Typical `model.params` for `RoFormerMLM`

```yaml
params:
  enc_hidden_dim: 256
  enc_expansion_dim: 1024
  enc_num_heads: 4
  enc_num_layers: 4
  enc_attention_dropout: 0.1
  enc_hidden_dropout: 0.1
  pred_hidden_dims: [512]
  pred_activation: gelu
  pred_dropout: 0.1
  loss_fn: cross_entropy
  loss_args: {}
  optimizer: adamw
  optimizer_args:
    lr: 1.0e-4
  scheduler: warmup_linear_decay
```

> `enc_num_characters` is injected automatically — do **not** set it manually.

### `model.datamodule` for MLM

```yaml
datamodule:
  mask_rate: 0.15
  max_length: 128
  num_augmentations: 2
  num_test_augmentations: 4
  include_canonical: true
  batch_size: 256
  num_workers: 0
```

---

## Graph Mode

### Dataset
```yaml
dataset:
  task_type: graph
  train_smiles: ./data/train_smiles.parquet
  val_smiles: ./data/val_smiles.parquet
  train_y_graph: ./data/train_graph_y.npz
  val_y_graph: ./data/val_graph_y.npz
  train_y_node: ./data/train_node_y.npz
  val_y_node: ./data/val_node_y.npz
```

### Model architectures for graph pretraining

| Architecture | Notes |
|---|---|
| `GINPretraining` | Simpler GIN encoder — fast, good baseline |
| `GatedGCNPretraining` | Gated GCN with positional encodings — stronger |

### Typical `model.params` for `GINPretraining`

```yaml
params:
  num_node_targets: <int>
  num_graph_targets: <int>
  enc_num_layers: 3
  enc_atom_hidden_dim: 300
  enc_jk: last
  enc_activation: swish
  enc_dropout: 0.2
  node_head_dims: [256]
  graph_head_dims: [256, 256]
  pred_activation: swish
  pred_dropout: 0.2
  loss_fn: mse
  node_loss_weight: 0.5
  graph_loss_weight: 0.5
  optimizer: adamw
  optimizer_args:
    lr: 1.0e-4
  scheduler: warmup_linear_decay
```

### `model.datamodule` for graph mode

```yaml
datamodule:
  laplacian_k: 10
  rwse_k: 20
  elstatic_k: 0
  distmat_k: 0
  rrwp_k: 20
  compute_distances: true
  batch_size: 256
  num_workers: 0
```

> `laplacian_k`, `rwse_k`, etc. are automatically forwarded to the model constructor as `enc_<key>`. Do not set them twice in `model.params`.

---

## Step-by-Step Config Generation

### 1. Select mode
Ask: "Do you want to pretrain a sequence model (MLM on SMILES) or a graph model (node + graph targets)?"

### 2. Dataset paths
Ask for SMILES parquet paths (both modes). Ask for graph/node label npz paths (graph mode only).

### 3. Architecture and params
Select architecture from the tables above. Use the provided example params as a starting point.

### 4. Training settings
- `training.num_epochs: 100` for MLM; `50` for graph pretraining.
- `pipe.fit_datamodule_size: 50000` — how many SMILES to use for fitting. Larger is better.

### 5. Generate YAML, confirm, and run

```
matcha pretrain_encoder --config pretrain_encoder.yaml
```

After training, remind the user:

> "Encoder pretraining is done. Use `model.architecture: FinetuningRegressor` (or `FinetuningClassifier`) and `model.path_to_pretrained: <output.serialization>` in your `train` config to finetune on a downstream task."

---

## Key Behaviors

- **MLM dictionary:** Built automatically by fitting on `pipe.fit_datamodule_size` SMILES. `enc_num_characters` is injected into the model automatically — do not set it in `params`.
- **Graph PE broadcasting:** Keys in `model.datamodule` are automatically broadcast to the model constructor as `enc_<key>`. Set them only in `datamodule`.
- **Finetuning chain:** After pretraining, pass `output.serialization` as `model.path_to_pretrained` in a `train` config with `model.architecture: FinetuningRegressor` or `FinetuningClassifier`.
