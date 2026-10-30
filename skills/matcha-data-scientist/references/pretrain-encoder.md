# pretrain-encoder

**CLI command:** `matcha pretrain_encoder --config pretrain_encoder.yaml`

Pretrains a MATCHA encoder using one of three self-supervised objectives:

- **MLM (masked language modelling):** Trains a `RoFormerMLM` sequence model on SMILES. No task labels are needed — the model learns by reconstructing masked tokens.
- **Graph pretraining (2D):** Trains a 2D graph encoder (`GINPretraining`, `GatedGCNPretraining`, `GPSPretraining`, `GTPretraining`, `AttentiveFPPretraining`) on node-level and graph-level targets. Requires pre-computed node and graph label arrays.
- **Graph3D pretraining:** Trains a 3D-aware graph encoder (`E3GNNPretraining`, `GPS3DPretraining`, `GT3DPretraining`) on node- and graph-level targets plus per-molecule atomic coordinates. Same label requirements as `graph`, plus `train_coords` / `val_coords` npz files.

All three modes produce an `encoder.ckpt` artifact that can be loaded by `FinetuningRegressor` / `FinetuningClassifier` in downstream `matcha train` runs.

---

## Task-Type Branching

Choose the mode based on the target architecture:

| Goal | Mode | Architecture |
|---|---|---|
| Pretrain a sequence/transformer encoder | `mlm` | `RoFormerMLM` |
| Pretrain a 2D graph encoder with QM/physical labels | `graph` | `GINPretraining`, `GatedGCNPretraining`, `GPSPretraining`, `GTPretraining`, `AttentiveFPPretraining` |
| Pretrain a 3D-aware graph encoder on precomputed coordinates | `graph3d` | `E3GNNPretraining`, `GPS3DPretraining`, `GT3DPretraining` |

Ask the user: "Do you want to pretrain a sequence model (MLM), a 2D graph model with node/graph labels (graph), or a 3D-aware graph model with precomputed atomic coordinates (graph3d)?" If they mention E3GNN, 3D pretraining, or coordinate-aware pretraining, route to graph3d and confirm that they can produce the `train_coords.npz` / `val_coords.npz` artifacts — MATCHA does not run conformer generation for them.

---

## YAML Schema (`CLIEncoderPretrainInputModel`)

### Shared skeleton

```yaml
dataset:
  task_type: <str>           # required — "mlm" | "graph" | "graph3d"
  train_smiles: <str>        # required — path to parquet file with SMILES column
  val_smiles: <str>          # required — path to parquet file with SMILES column
  # graph and graph3d modes:
  train_y_graph: <str>       # path to npz file (key: "descriptors") — graph-level targets
  val_y_graph: <str>
  train_y_node: <str>        # path to npz file (keys: "flat", "offsets") — node-level targets
  val_y_node: <str>
  # graph3d only:
  train_coords: <str>        # path to npz file (keys: "flat", "offsets") — atomic coordinates
  val_coords: <str>

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

The CLI graph mode uses the 2D `GraphPretrainingDataModule` and covers the 2D graph encoders. For 3D-aware architectures (`E3GNNPretraining`, `GPS3DPretraining`, `GT3DPretraining`), use `task_type: graph3d` instead — see the Graph3D Mode section below.

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
| `GPSPretraining`, `GTPretraining`, `AttentiveFPPretraining` | Additional 2D encoders — same YAML shape, swap architecture and the matching `enc_*` hyperparameters |

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

## Graph3D Mode

The CLI graph3d mode uses `Graph3DPretrainingDataModule` and adds precomputed per-molecule atomic coordinates on top of the same node + graph targets used in graph mode. All three registered 3D architectures (`E3GNNPretraining`, `GPS3DPretraining`, `GT3DPretraining`) consume the same batches — they only differ in the encoder `enc_*` surface.

> **Coords are user-supplied.** `matcha prepare_dataset` does **not** emit `train_coords.npz` / `val_coords.npz`. The user must produce these files externally (e.g. RDKit ETKDG, OpenFF, xTB, ORCA) before invoking `pretrain_encoder`. If they don't have coordinates ready, route them to `graph` mode instead.

### Dataset
```yaml
dataset:
  task_type: graph3d
  train_smiles: ./data/train_smiles.parquet
  val_smiles: ./data/val_smiles.parquet
  train_y_graph: ./data/train_graph_y.npz
  val_y_graph: ./data/val_graph_y.npz
  train_y_node: ./data/train_node_y.npz
  val_y_node: ./data/val_node_y.npz
  train_coords: ./data/train_coords.npz
  val_coords: ./data/val_coords.npz
```

> **Coords npz contract** (enforced by `_load_coords_npz` at CLI startup — failures are `AssertionError`s that fail fast before featurization):
> - Keys: `flat` (2D `float32`, shape `(N_atoms, 3)`) and `offsets` (1D int, length `N + 1`).
> - `offsets` must be monotonic non-decreasing.
> - `offsets[-1] == flat.shape[0]`.
> - Each molecule's slice `flat[offsets[i]:offsets[i+1]]` is an `(A_i, 3)` `float32` array.
>
> The same `flat + offsets` packing is used for `train_y_node` / `val_y_node`, so users who already have node targets in this layout can reuse the same packing helper for coords. Node targets and coords must both index the same molecules in the same order.

### Model architectures for graph3d pretraining

| Architecture | Notes |
|---|---|
| `E3GNNPretraining` | E(n)-equivariant message passing with optional coordinate updates. Extra encoder keys: `enc_m_dim`, `enc_fourier_features`, `enc_soft_edge`, `enc_norm_feats`, `enc_norm_coors`, `enc_update_coors`, `enc_coor_weights_clamp_value`, `enc_norm_coors_scale_init`. |
| `GPS3DPretraining` | GPS transformer with 3D Gaussian-basis-kernel spatial encoding. Extra encoder keys: `enc_num_heads`, `enc_num_kernels`, `enc_expansion_k`, `enc_norm`. |
| `GT3DPretraining` | Graph Transformer with 3D Gaussian-basis-kernel spatial encoding. Extra encoder keys: `enc_num_heads`, `enc_num_kernels`, `enc_expansion_k` (no `enc_norm`). |

### Typical `model.params` for `E3GNNPretraining`

```yaml
params:
  num_node_targets: <int>
  num_graph_targets: <int>
  enc_num_layers: 3
  enc_atom_hidden_dim: 128
  # ── E3GNNPretraining-specific ──
  enc_m_dim: 16
  enc_fourier_features: 4
  enc_update_coors: true
  enc_norm_coors: true
  # ────
  enc_jk: concat
  enc_readout: vpa
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

For `GPS3DPretraining` / `GT3DPretraining`, swap the E3GNN-specific block for:

```yaml
  # ── GPS3DPretraining / GT3DPretraining ──
  enc_num_heads: 8
  enc_num_kernels: 128
  enc_expansion_k: 2
  # enc_norm: layer         # GPS3DPretraining only
```

### `model.datamodule` for graph3d mode

Same shape as graph mode — same positional-encoding keys, same broadcast-to-model rule (`enc_<key>`):

```yaml
datamodule:
  laplacian_k: 10
  rwse_k: 0
  elstatic_k: 0
  distmat_k: 0
  rrwp_k: 0
  compute_distances: true
  num_virtual_nodes: 0
  init_virtual_nodes: false
  batch_size: 256
  num_workers: 0
  augment_resonance: false
  scale_y_graph: false
  scale_y_node: false
```

> A complete runnable example lives at `src/matcha/cli/example_configs/pretrain_encoder_graph3d.yaml` — use it as a starting template.

---

## Step-by-Step Config Generation

### 1. Select mode
Ask: "Do you want to pretrain a sequence model (MLM on SMILES), a 2D graph model (node + graph targets), or a 3D-aware graph model (node + graph targets + precomputed atomic coordinates)?"

### 2. Dataset paths
Ask for SMILES parquet paths (all modes). Ask for graph/node label npz paths (graph and graph3d modes). Ask for coords npz paths (graph3d only) and confirm they follow the `flat + offsets` contract described in the Graph3D Mode section.

### 3. Architecture and params
Select architecture from the tables above. Use the provided example params as a starting point.

### 4. Training settings
- `training.num_epochs: 100` for MLM; `50` for graph / graph3d pretraining.
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
- **Graph3D coords contract:** `train_coords` / `val_coords` npz files must use the `flat + offsets` packing (`flat` shape `(N_atoms, 3)` `float32`, `offsets` length `N + 1`, monotonic non-decreasing, `offsets[-1] == flat.shape[0]`). Coords must index the same molecules in the same order as `train_y_node` / `val_y_node`. MATCHA never runs conformer generation — coords come from the user.
- **Finetuning chain:** After pretraining, pass `output.serialization` as `model.path_to_pretrained` in a `train` config with `model.architecture: FinetuningRegressor` or `FinetuningClassifier`.
