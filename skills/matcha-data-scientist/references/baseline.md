# baseline

**CLI command:** `matcha baseline --config baseline.yaml`

Evaluates a scikit-learn model (e.g., Random Forest, SVM, Gradient Boosting) using RDKit molecular descriptors as features. Uses the same splitting and metrics infrastructure as `matcha evaluate`, making results directly comparable when the same split config is used.

> **Split matching:** For a fair comparison against a MATCHA model, use the **exact same `split` block** as in your `evaluate` config.

---

## YAML Schema (`CLIBaselineInputModel`)

```yaml
dataset:
  path: <str>                        # required — CSV or SDF path
  label_key: <str>                   # required — label column name
  smiles_key: <str>                  # optional
  operator_key: <str>                # optional — censoring operator column

split:
  method: <str>                      # required — "cv", "time", "cluster", or "file"
  n_subset: <int>                    # required — number of splits/folds
  n_bootstrap: <int>                 # optional
  frac_bootstrap: <float>            # optional
  method_params: <dict>              # required for "time", "cluster", "file"

model:
  algorithm: <str>                   # required — scikit-learn class name (see table below)
  n_jobs: <int>                      # optional — parallel workers (default: 16)
  feature_list: <list[str]>          # optional — molecular feature sets (default: ["rdkit_all_descriptors", "ecfp"])
  params: <dict>                     # optional — algorithm hyperparameters (default: {"n_estimators": 100})
  label_transform: <str>             # optional — "log10", "sqrt", or null
  label_encoder_params: <dict>       # optional — for classification: maps endpoint index to threshold

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
```

---

## Available Algorithms

| `algorithm` string | Type | Notes |
|---|---|---|
| `RandomForestRegressor` | Regression | Default; robust, interpretable |
| `RandomForestClassifier` | Classification | |
| `GradientBoostingRegressor` | Regression | Slower but often more accurate than RF |
| `GradientBoostingClassifier` | Classification | |
| `SVR` | Regression | Good for small datasets |
| `SVC` | Classification | |
| `KNeighborsRegressor` | Regression | Distance-based; slow on large datasets |
| `KNeighborsClassifier` | Classification | |
| `LinearRegression` | Regression | Fastest; useful as a lower bound |
| `LogisticRegression` | Classification | |

**Recommendation:** Start with `RandomForestRegressor` (regression) or `RandomForestClassifier` (classification). They are fast, require minimal tuning, and provide strong baselines.

---

## Available Feature Sets (`feature_list`)

| Feature string | Description |
|---|---|
| `rdkit_all_descriptors` | RDKit 2D physicochemical descriptors (~200 features) |
| `ecfp` | Extended Connectivity Fingerprint (binary, radius 2) |

Mix feature sets as a list: `["rdkit_all_descriptors", "ecfp"]` concatenates them.

---

## Step-by-Step Config Generation

### 1. Dataset
Ask: "What is the path to your dataset? What column holds the labels?"

### 2. Split strategy
Ask: "What split config did you use (or plan to use) for `evaluate`?"

- **Copy the exact same `split` block from your `evaluate` config.** This is critical for a fair comparison — different splits produce incomparable performance metrics.

### 3. Algorithm choice
Ask: "Which scikit-learn algorithm would you like to use?"

- Default: `RandomForestRegressor` for regression, `RandomForestClassifier` for classification.

### 4. Feature list
Ask: "Which molecular features should be used?"

- Default: `["rdkit_all_descriptors", "ecfp"]` — good all-around coverage.

### 5. Hyperparameters

Common examples:
```yaml
# Random Forest
params:
  n_estimators: 200
  max_depth: 20
  min_samples_leaf: 2

# Gradient Boosting
params:
  n_estimators: 100
  learning_rate: 0.1
  max_depth: 4
```

Default: `{"n_estimators": 100}`.

### 6. Label transform (regression only)
Ask: "Should labels be log-transformed before fitting?"

- `label_transform: log10` — fit on log10 values, back-transform predictions before metrics.
- Use `log10` when your labels span multiple orders of magnitude (e.g., IC50 in nM).

### 7. Output path

For `matcha summarize` in directory mode, this path should be a subdirectory of the `root_dir` you plan to use.

### 8. Generate YAML, confirm, and run

```
matcha baseline --config baseline.yaml
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
  algorithm: RandomForestRegressor
  feature_list: ["rdkit_all_descriptors"]
  params:
    n_estimators: 100

output:
  serialization:
    path: ./results/baseline_rf
```

## Production Config Example (temporal split, MLflow, log transform)

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
  algorithm: RandomForestRegressor
  n_jobs: 16
  feature_list: ["rdkit_all_descriptors"]
  label_transform: log10
  params:
    n_estimators: 100

output:
  mlflow:
    experiment_name: evaluating-fu
    run_name: rdkit_rf
    tags:
      version: 1.0
    log_dir: ./mlruns
  serialization:
    path: ./results/baseline_rf
```

---

## Key Behaviors

- **Multi-task support:** For multi-task datasets, baseline runs one model per endpoint independently.
- **Censored data:** Baseline handles censored measurements via the same `operator_key` mechanism as MATCHA.
- **Label transform back-transformation:** When `label_transform` is set, predictions are automatically back-transformed before metric computation.
- **Comparison with evaluate:** The metrics in `performance.json` use the same computation as `matcha evaluate`, so results are directly comparable when the split config is identical.
