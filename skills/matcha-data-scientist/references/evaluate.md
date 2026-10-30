# evaluate

**CLI command:** `matcha evaluate --config evaluate.yaml`

Runs cross-validated model evaluation: for each split, trains a single model, computes predictions, calculates regression or classification metrics, generates plots, and aggregates results. Optionally logs to MLflow.

> **Important:** The `evaluate` command trains **single models only** — do not set `model.ensemble`. Setting `ensemble` raises a `ValueError` and aborts the run.

---

## YAML Schema (`CLIEvaluationInputModel`)

```yaml
dataset:
  path: <str>                        # required — CSV or SDF path
  label_key: <str>                   # required — label column name
  smiles_key: <str>                  # optional
  operator_key: <str>                # optional — censoring operator column
  calibration: <dict>                # optional (rarely used in evaluate)
  statistics: <dict>                 # optional

split:
  method: <str>                      # required — "cv", "time", "cluster", or "file"
  n_subset: <int>                    # required — number of splits / folds
  n_bootstrap: <int>                 # optional — bootstrap resampling iterations
  frac_bootstrap: <float>            # optional — fraction of test set to sample per bootstrap
  method_params: <dict>              # required for "time", "cluster", "file" (see below)

model:
  architecture: <str>                # required — no ensemble allowed
  params: <dict>                     # required — model hyperparameters
  metadata: <dict>                   # optional (not required for evaluate)
  ensemble: null                     # MUST be null / absent — ensembles are forbidden
  config_path: <str>                 # optional — load params from autotune output

output:
  mlflow:                            # optional
    experiment_name: <str>
    log_dir: <str>
    tags: <dict>
    run_name: <str>
    server_uri: <str>
  serialization:                     # recommended
    path: <str>                      # directory for performance.json, plots, CSVs
    quantize: <bool>                 # default: false

quantize: <bool>                     # optional top-level flag (default: false)
```

---

## Split Methods

### `cv` — K-Fold Cross-Validation
```yaml
split:
  method: cv
  n_subset: 10         # number of folds
```
No `method_params` needed.

### `time` — Temporal Split
Sorts data by a time/ID column and evaluates on progressively more-recent slices.
```yaml
split:
  method: time
  n_subset: 3
  method_params:
    key: compound_id           # column to sort by (ascending)
    split_size: 0.1      # fraction of data used as test per split
```

### `cluster` — Cluster-Based Split
Splits by chemical similarity clusters. Requires one entry per split.
```yaml
split:
  method: cluster
  n_subset: 3
  method_params:
    features: ["ecfp", "ecfp", "rdkit_all_descriptors"]   # one per split
    metric: ["tanimoto", "tanimoto", "euclidean"]          # one per split
    split_size: 0.2
    n_jobs: 8
```

### `file` — External Test Set
Uses pre-defined train/test file pairs.
```yaml
split:
  method: file
  n_subset: 2           # number of test files
  method_params:
    path:
      - ./test_set_1.csv
      - ./test_set_2.csv
```

---

## Step-by-Step Config Generation

### 1. Dataset
Ask: "What is the path to your evaluation dataset? What column holds the labels?"

### 2. Split strategy
Ask: "Which split method should we use?"
- General benchmarking → `cv` with `n_subset: 10`
- Temporal generalization → `time` with `key: compound_id`, `split_size: 0.1`, `n_subset: 3`
- Chemical diversity → `cluster`
- Held-out test set → `file`

Ask: "Do you need bootstrap confidence intervals on the metrics?"
- If yes → set `n_bootstrap: 1000` and `frac_bootstrap: 1.0`

### 3. Model architecture
Ask: "Which architecture do you want to evaluate?"

**Raise an error if the user requests `ensemble`.** Say:
> "The `evaluate` command does not support ensemble models. Please remove `ensemble` from the config. To train a production ensemble, use `matcha train` instead."

### 4. Model params
Same as `train`: `loss_fn`, `num_endpoints`, `label_encoder_params`, etc.

### 5. MLflow (optional)
Ask: "Should evaluation metrics be tracked in MLflow?"

Set `run_name` to the architecture name for easy comparison across runs.

### 6. Output path
Ask: "Where should evaluation results be written?"

The directory will contain:
- `performance.json` — mean/std metrics per endpoint per split
- `performance_log10.json` — same metrics in log10 space
- `plots/` — HTML scatter plots per endpoint per split
- `<endpoint>_<split>_preds.csv` — raw predictions
- `cfg.yaml` — saved config

### 7. Generate YAML, confirm, and run

```
matcha evaluate --config evaluate.yaml
```

---

## Minimal Config Example

```yaml
dataset:
  path: ./data/my_dataset.csv
  smiles_key: SMILES
  label_key: ACTIVITY

split:
  method: cv
  n_subset: 5

model:
  architecture: ChempropRegressor
  params:
    loss_fn: mse
    num_endpoints: 1
    num_epochs: 50

output:
  serialization:
    path: ./results/chemprop_eval
```

## Production Config Example (temporal split, MLflow, multi-task)

```yaml
dataset:
  path: ./data/dataset.csv
  smiles_key: SMILES
  label_key: endpoint
  operator_key: OPERATOR

split:
  method: time
  n_subset: 3
  method_params:
    key: compound_id
    split_size: 0.1

model:
  architecture: ChempropRegressor
  params:
    label_encoder_params:
      0: {task_label: Fub_human}
      1: {task_label: Fub_mouse}
      2: {task_label: Fub_rat}
      3: {task_label: Fu_mic}
    num_endpoints: 4
    loss_fn: bounded-mse

output:
  mlflow:
    experiment_name: evaluating-fu
    run_name: chemprop_eval
    tags:
      version: 1.0
    log_dir: ./mlruns
  serialization:
    path: ./results/chemprop_eval
```

---

## Key Behaviors

- **No ensembles:** The evaluate command always trains single models. If you need ensemble evaluation, use `matcha train` with an ensemble and separate held-out test data.
- **Metric types:** For regressors: R², MAE, RMSE, Pearson r (raw and log10 scales). For classifiers: accuracy, AUC, F1.
- **Plots:** HTML scatter plots (true vs. predicted) are saved to `output.serialization.path/plots/`.
- **MLflow run naming:** Each split creates an MLflow run named `<run_name>_split_<i>`. A final summary run `<run_name>_metrics` logs the aggregated mean/std metrics.
- **Bootstrap:** Bootstrapping resamples the test set predictions — it does not affect the train/test assignment. Use it to estimate metric confidence intervals.
