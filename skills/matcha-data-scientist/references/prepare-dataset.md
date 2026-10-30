# prepare-dataset

**CLI command:** `matcha prepare_dataset --config prepare.yaml`

Merges multiple parquet/CSV files into a single compounds × tasks label matrix consumed directly by `matcha pretrain_multitask`. Two storage modes are supported, selected by `datasets.sparse` in the input config:

- **Sparse mode** (`sparse: true`, the default) — emits a `scipy.sparse.csr_matrix` per split. Classification labels are remapped `0 → -1` internally so zero-omission in the CSR unambiguously encodes "missing". Best when the label matrix is largely empty.
- **Dense mode** (`sparse: false`) — emits a `float32` `numpy` array per split. Missing entries are `NaN`; classification labels pass through as `0`/`1` with no remap. Best when the label matrix is largely populated (e.g. ≥90%), where CSR overhead outweighs the storage savings.

Regression tasks are standard-scaled in both modes (`np.nanmean` / `np.nanstd` in dense mode; non-zero-only stats in sparse mode) — the two modes produce equivalent statistics on the same underlying data.

The mode is stamped into `task_metadata.json` as `storage_mode: "sparse" | "dense"`. `pretrain_multitask` reads this field and auto-detects the loader and artifact filenames — no additional flag is set on the pretraining side.

---

## YAML Schema (`CLIPrepareInputModel`)

```yaml
datasets:
  files: <list[str]>         # required — parquet/CSV file paths; each contributes tasks
  task_type: <list[str]>     # required — one per file: "regression" or "classification"
  sparse: <bool>             # optional — true for CSR .npz output (default), false for dense .npy

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
Ask: "What parquet/CSV files should be merged? Is each a regression or classification dataset?"

- All files must have a SMILES column (default: `"SMILES"`) and one or more numeric task columns.
- Regression — values are standardised (mean 0, std 1) over non-missing entries.
- Classification — in sparse mode, `0` labels are remapped to `-1`; in dense mode, `0`/`1` pass through unchanged.

### 2. Storage mode
Ask: "How populated is the resulting label matrix likely to be? If it is largely empty, choose sparse (default). If it is largely populated (e.g. ≥90%), choose dense."

Set `datasets.sparse: false` for dense output; omit or set `true` for sparse.

### 3. Validation split
Defaults are usually fine: `min_compounds: 10`, `sampling_rate: 0.001`, `seed: 42`.

### 4. Output directory
Ask: "Where should the prepared dataset be saved?"

### 5. Generate YAML, confirm, and run

```
matcha prepare_dataset --config prepare.yaml
```

After preparation completes, remind the user:

> "Data preparation is done. Use `dataset.dataset_dir: <output>` in your `pretrain_multitask` config. The directory contains `train_molecules.parquet`, `val_molecules.parquet`, the split task label artifacts (sparse: `train_tasks_sparse.npz` / `val_tasks_sparse.npz`; dense: `train_tasks_dense.npy` / `val_tasks_dense.npy`), and `task_metadata.json` (with a `storage_mode` field the pretraining loader auto-detects)."

---

## Example Configs

### Sparse mode (default)

```yaml
datasets:
  files:
    - ./data/adme_regression.parquet
    - ./data/tox_classification.parquet
  task_type:
    - regression
    - classification
  # sparse: true  # optional, default

metadata:
  merge_col: SMILES
  tag_to_add: PRETRAIN

validation:
  min_compounds: 10
  sampling_rate: 0.001
  seed: 42

output: ./data/prepared_pretraining
```

### Dense mode

```yaml
datasets:
  files:
    - ./data/adme_regression.parquet
    - ./data/tox_classification.parquet
  task_type:
    - regression
    - classification
  sparse: false

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
| `train_tasks_sparse.npz` / `val_tasks_sparse.npz` | Sparse task labels (fp32 CSR) — sparse mode only |
| `train_tasks_dense.npy` / `val_tasks_dense.npy` | Dense task labels (fp32 ndarray with NaN for missing) — dense mode only |
| `task_metadata.json` | Task column names, file→task mapping, scaling stats, `storage_mode` |
| `datacard.json` | Dataset statistics (sparsity / non-NaN fraction, avg measurements, etc.) |

---

## Key Behaviors

- **Compound deduplication:** Compounds are matched by RDKit canonical SMILES. Duplicates across files are merged into a single row.
- **Regression scaling:** Mean and std are computed from training compounds only (no data leakage). Dense mode uses `np.nanmean` / `np.nanstd`; sparse mode uses non-zero-only stats. Both are numerically equivalent on the same underlying data.
- **Classification encoding:** In sparse mode, `0 → -1` remap so "missing" and "inactive" are distinguishable. In dense mode, `NaN` alone marks missing and classification labels pass through as `0`/`1`.
- **Storage mode is stamped into `task_metadata.json`:** the `storage_mode` field is read by `pretrain_multitask` to pick the right loader and artifact filenames — no configuration is required on the pretraining side.
- **Downstream chaining:** The `output` directory feeds into `pretrain_multitask` as `dataset.dataset_dir`.
