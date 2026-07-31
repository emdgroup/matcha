# train

**CLI command:** `matcha train --config train.yaml`

Trains a single or ensemble MATCHA model on a labeled molecular dataset and serializes the result.

---

## YAML Schema (`CLITrainInputModel`)

```yaml
dataset:
  path: <str>                        # required — path to CSV or SDF file
  label_key: <str>                   # required — column name containing labels (e.g. "endpoint")
  smiles_key: <str>                  # optional — SMILES column (auto-detected for SDF)
  operator_key: <str>                # optional — column with censoring operators (">", "<", "=")
  calibration:                       # optional — split calibration set from training data
    split_column: <str>              # column to sort by (default: "compound_id")
    split_ratio: <float>             # fraction to keep as training (default: 0.99)
  statistics: <dict>                 # optional — dataset statistics metadata (rarely needed)

model:
  architecture: <str>                # required — e.g. "ChempropRegressor"
  params: <dict>                     # required — model hyperparameters (see below)
  metadata:                          # required
    model_name: <str>                # unique identifier for this model
    model_version: <int>             # version number (integer)
    model_scope: <str>               # e.g. "global" or "local"
    model_owner: <str>               # full name of the model owner
    description: <str>              # optional — human-readable description
  ensemble: <int>                    # optional — number of ensemble members
  config_path: <str>                 # optional — path to autotune output YAML
  calibration:                       # optional — uncertainty calibration
    algorithm: <str>                 # e.g. "error_model" or "inductive_conformal"
    params: <dict>                   # optional — algorithm-specific params

output:
  mlflow:                            # optional — MLflow experiment tracking
    experiment_name: <str>
    log_dir: <str>
    tags: <dict>                     # optional
    run_name: <str>                  # optional
    server_uri: <str>                # optional
  serialization:                     # optional but recommended
    path: <str>                      # directory to save the model
    quantize: <bool>                 # default: false
```

---

## Key `model.params` Fields

These go inside `model.params` and are passed to the architecture constructor:

| Field | Type | Description |
|---|---|---|
| `loss_fn` | str | Loss function: `"mse"`, `"bounded_mse"`, `"multitask"`, `"cross_entropy"` |
| `loss_args` | dict | Arguments for the loss (e.g. `{"loss_fn": "bounded_mse"}` for multitask) |
| `num_endpoints` | int | Number of prediction targets |
| `num_epochs` | int | Training epochs (default varies by architecture) |
| `batch_size` | int | Batch size |
| `label_encoder_params` | dict | Maps endpoint index (0-based int) to `{"task_label": "<name>"}` |
| `feature_list` | list[str] | Descriptor features for tabular models (e.g. `["rdkit_all_descriptors", "ecfp"]`) |
| `path_to_pretrained` | str | Path to pretrained encoder (for `FinetuningRegressor`/`FinetuningClassifier`) |

---

## Step-by-Step Config Generation

### 1. Dataset
Ask: "What is the path to your dataset? What column holds the labels? What column holds the SMILES?"

- For multi-task datasets (e.g., produced by `stitch`), `label_key` is typically `"endpoint"`.
- `operator_key` is needed only if the dataset contains censored measurements (e.g., `"> 10 µM"`).

### 2. Architecture
Ask: "Which architecture would you like to use? Do you need a regressor or a classifier?"

- Default recommendation: `ChempropRegressor` (regression) or `ChempropClassifier` (classification).
- For multi-task: `GatedGCNRegressor` or `ChempropRegressor` with `loss_fn: multitask`.

### 3. Metadata
Ask: "What should the model be named? Who is the owner? What is the scope (global/local)?"

### 4. Hyperparameters
Ask: "Do you have HPO results from `autotune`? If yes, provide the path."

- If `config_path` is set, the model params are loaded from the autotune output. You still need to set `label_encoder_params` manually.
- If no HPO results, fill in `loss_fn`, `num_endpoints`, and `num_epochs` at minimum.

### 5. Label encoder params
Required for multi-task models. One entry per endpoint:

```yaml
label_encoder_params:
  0: {task_label: "Endpoint_A"}
  1: {task_label: "Endpoint_B"}
```

### 6. Calibration (optional)
Ask: "Do you need uncertainty calibration?"

- `dataset.calibration` reserves a portion of training data as calibration set.
- `model.calibration.algorithm` applies the calibration fit (e.g. `"error_model"`).

### 7. MLflow (optional)
Ask: "Should training metrics be tracked in MLflow?"

- Requires `experiment_name` and `log_dir`. MLflow must be accessible from the training environment.

### 8. Output path
Ask: "Where should the trained model be saved?"

- The serialized model directory will contain `config/manifest.yaml`, model weights, `train.log`, and `cfg.yaml`.
- This path is used as `model.path` in a subsequent `predict` config.

### 9. Generate YAML, confirm, and run

Show the complete YAML, confirm with the user, then run:

```
matcha train --config train.yaml
```

---

## Minimal Config Example (single model, no MLflow)

```yaml
dataset:
  path: ./data/my_dataset.csv
  smiles_key: SMILES
  label_key: ACTIVITY

model:
  architecture: ChempropRegressor
  params:
    loss_fn: mse
    num_endpoints: 1
    num_epochs: 50
  metadata:
    model_name: chemprop-activity-v1
    model_version: 1
    model_scope: global
    model_owner: Your Name

output:
  serialization:
    path: ./models/activity_model
```

## Production Config Example (ensemble, MLflow, multi-task)

```yaml
dataset:
  path: ./data/dataset.csv
  smiles_key: SMILES
  label_key: endpoint
  operator_key: OPERATOR

model:
  architecture: GatedGCNRegressor
  ensemble: 5
  params:
    loss_fn: multitask
    loss_args: {loss_fn: bounded_mse}
    num_endpoints: 4
    num_epochs: 50
    label_encoder_params:
      0: {task_label: Fub_human}
      1: {task_label: Fub_mouse}
      2: {task_label: Fub_rat}
      3: {task_label: Fu_mic}
  metadata:
    model_name: gatedgcn-fu-v1
    model_version: 1
    model_scope: global
    model_owner: Your Name

output:
  mlflow:
    experiment_name: training-fu-gatedgcn
    tags:
      version: 1
    log_dir: ./mlruns
  serialization:
    path: ./models/fu_gatedgcn
```

---

## Key Behaviors

- **Finetuning:** Set `architecture: FinetuningRegressor` and include `path_to_pretrained: <encoder_path>` in `model.params`. The encoder must be a checkpoint produced by `pretrain_multitask` or `pretrain_encoder`.
- **Censored data:** If your dataset has operator columns (e.g., `">"`, `"<"`), set `operator_key` and use `loss_fn: bounded_mse` or `loss_fn: multitask` with `bounded_mse` as the inner loss.
- **Calibration:** Uncertainty calibration requires a held-out calibration set. Set `dataset.calibration.split_ratio: 0.99` to reserve the most-recent 1% of data (sorted by `split_column`).
- **Quantization:** Set `output.serialization.quantize: true` to reduce model size (may reduce precision slightly).
