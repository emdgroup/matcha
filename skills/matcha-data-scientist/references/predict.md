# predict

**CLI command:** `matcha predict --config predict.yaml`

Loads a serialized MATCHA ensemble model and runs inference on a new molecular dataset. Produces a predictions CSV alongside a log of failed molecules.

> **Prerequisite:** A trained and serialized model directory produced by `matcha train`. The directory must contain `config/manifest.yaml`.

---

## YAML Schema (`CLIPredictInputModel`)

```yaml
dataset:
  path: <str>                        # required — CSV or SDF of molecules to score
  smiles_key: <str>                  # optional — SMILES column name (default auto-detected)
  label_key: <str>                   # optional — ignored during prediction
  keep_cols: <bool>                  # optional — carry all input columns through to output (default: true)
  operator_key: <str>                # optional — ignored during prediction

model:
  path: <str>                        # required — path to serialized model directory
  inference:                         # optional
    accelerator: <str>               # "cpu" or "gpu" (default: "cpu")
    devices: <int>                   # number of devices (default: 1)
    batch_size: <int>                # molecules per batch (default: 256)

output: <str>                        # required — directory to write output files
```

---

## Step-by-Step Config Generation

### 1. Trained model path
Ask: "What is the path to your serialized model directory?"

- The directory must contain `config/manifest.yaml`. This is created automatically by `matcha train` when `output.serialization.path` is set.

### 2. Input dataset
Ask: "What is the path to the dataset of molecules you want to score?"

- Accepts CSV or SDF. The file does not need a label column — only SMILES are required.

### 3. keep_cols
Ask: "Should the output file include all columns from the input file alongside predictions?"

- `true` (default) — output CSV contains all original columns plus prediction columns.
- `false` — output CSV contains only SMILES and prediction columns.

### 4. Inference config
Ask: "Do you have a GPU available? How large is your dataset?"

- For small datasets (< 50k molecules) on CPU: use defaults.
- For large datasets or GPU: set `accelerator: gpu`, `devices: 1`, and increase `batch_size` to 512 or 1024.

### 5. Output path

The output directory will contain:
- `output.csv` — all molecules with successful predictions.
- `failed.csv` — molecules that could not be featurized (invalid SMILES, etc.).
- `cfg.yaml` — saved config.

### 6. Generate YAML, confirm, and run

```
matcha predict --config predict.yaml
```

---

## Example Config

```yaml
dataset:
  path: ./data/new_molecules.csv
  smiles_key: SMILES
  keep_cols: true

model:
  path: ./models/fu_gatedgcn
  inference:
    accelerator: cpu
    devices: 1
    batch_size: 256

output: ./predictions/fu_gatedgcn
```

---

## Key Behaviors

- **Ensemble averaging:** If the model is an ensemble, predictions are the mean across all ensemble members. Uncertainty estimates (standard deviation) are also written to `output.csv` when the model was calibrated.
- **Failed molecules:** Molecules with invalid SMILES or featurization errors are written to `failed.csv` and excluded from `output.csv`. Always check this file after a run.
- **Column naming:** Output prediction columns are named after the `task_label` values set in `label_encoder_params` during training.
- **GPU inference:** Setting `accelerator: gpu` can significantly speed up inference for large datasets (> 100k molecules). Ensure the environment has CUDA available.
