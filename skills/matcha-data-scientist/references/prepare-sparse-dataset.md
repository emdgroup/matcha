# prepare-sparse-dataset

**CLI command:** `matcha prepare_sparse_dataset --config prepare.yaml`

Merges multiple parquet files into a single sparse compounds × tasks matrix. Each parquet file contributes one or more task columns. Regression tasks are standard-scaled; classification task labels have 0 replaced with −1 to distinguish from missing values. The output is a directory of files consumed directly by `matcha pretrain_multitask`.

---

## YAML Schema (`CLIPrepareInputModel`)

```yaml
datasets:
  files: <list[str]>         # required — parquet file paths; each contributes tasks
  task_type: <list[str]>     # required — one per file: "regression" or "classification"

metadata:
  merge_col: <str>           # optional — SMILES column name shared across all files (default: "SMILES")
  tag_to_add: <str>          # optional — suffix appended to every task column (default: "PRETRAIN")

validation:
  min_compounds: <int>       # optional — minimum compounds per task in val set (default: 10)
  sampling_rate: <float>     # optional — fraction of task compounds assigned to val (default: 0.001)
  seed: <int>                # optional — random seed for reproducibility (default: 42)

output: <str>                # required — output directory path (created if absent)
```

> **Constraint:** `datasets.files` and `datasets.task_type` must have the same length.

---

## Step-by-Step Config Generation

### 1. Dataset files and task types
Ask: "What parquet files should be merged? Is each a regression or classification dataset?"

- All files must have a SMILES column (default: `"SMILES"`) and one or more numeric task columns.
- Regression — values are standardised (mean 0, std 1).
- Classification — 0 labels are remapped to −1 (so 0 in the sparse matrix means "missing").

### 2. Validation split
Defaults are usually fine: `min_compounds: 10`, `sampling_rate: 0.001`, `seed: 42`.

### 3. Output directory
Ask: "Where should the prepared dataset be saved?"

### 4. Generate YAML, confirm, and run

```
matcha prepare_sparse_dataset --config prepare.yaml
```

After preparation completes, remind the user:

> "Data preparation is done. Use `dataset.dataset_dir: <output>` in your `pretrain_multitask` config. The directory contains `train_molecules.parquet`, `val_molecules.parquet`, `train_tasks.npz`, `val_tasks.npz`, and `task_metadata.json`."

---

## Example Config

```yaml
datasets:
  files:
    - ./data/adme_regression.parquet
    - ./data/tox_classification.parquet
  task_type:
    - regression
    - classification

metadata:
  merge_col: SMILES
  tag_to_add: PRETRAIN

validation:
  min_compounds: 10
  sampling_rate: 0.001
  seed: 42

output: ./data/prepared_pretraining
```

---

## Output Files

| File | Description |
|---|---|
| `train_molecules.parquet` | SMILES of training compounds |
| `val_molecules.parquet` | SMILES of validation compounds |
| `train_tasks.npz` | Sparse matrix of training task labels (fp32 CSR) |
| `val_tasks.npz` | Sparse matrix of validation task labels (fp32 CSR) |
| `task_metadata.json` | Task column names, file→task mapping, scaling stats |
| `datacard.json` | Dataset statistics (sparsity, avg measurements, etc.) |

---

## Key Behaviors

- **Compound deduplication:** Compounds are matched by RDKit canonical SMILES. Duplicates across files are merged into a single row.
- **Regression scaling:** Mean and std are computed from training compounds only (no data leakage).
- **Classification encoding:** 0 → −1 so that "missing" and "inactive" are distinguishable.
- **Downstream chaining:** The `output` directory feeds into `pretrain_multitask` as `dataset.dataset_dir`.
