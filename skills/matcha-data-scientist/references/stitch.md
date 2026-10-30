# stitch

**CLI command:** `matcha stitch --config stitch.yaml`

Merges multiple single-endpoint molecular files (SDF or CSV) into a single multi-task CSV dataset. Each input file contributes one endpoint column. The output is a wide-format CSV ready for `matcha train`, `matcha evaluate`, or `matcha baseline`.

---

## YAML Schema (`CLIStitcherInputModel`)

```yaml
input:
  folder_path: <str>                 # required — directory containing the input files
  dataset_names: <list[str]>         # required — filenames within folder_path (SDF or CSV)
  label_keys: <list[str]>            # required — one label column name per file
  tag: <str>                         # optional — name of the endpoint column in output (default: "endpoint")
  operator_key: <str>                # optional — censoring operator column name (default: "OPERATOR")
  smiles_key: <str>                  # optional — SMILES column name (default: "SMILES")
  index_key: <str>                   # optional — compound ID column for merging (default: None)

output:
  folder_path: <str>                 # optional — output directory (default: "./outputs")
  filename: <str>                    # optional — output filename (default: "df.csv")
```

> **Constraint:** `label_keys` length must equal `dataset_names` length. Each entry maps to the corresponding file.

---

## Step-by-Step Config Generation

### 1. Input folder and files
Ask: "What directory contains your input files? What are the filenames and the label column in each file?"

- All input files must be in the same directory. Both SDF and CSV are supported.
- `dataset_names` and `label_keys` must have the same length.

### 2. Tag (endpoint column name)
Ask: "What should the endpoint column in the output be called?"

- Default: `"endpoint"`. This becomes the `label_key` in downstream `train`/`evaluate` configs.

### 3. Output path
Ask: "Where should the merged dataset be saved and what should it be named?"

### 4. Generate YAML, confirm, and run

```
matcha stitch --config stitch.yaml
```

After stitch completes, remind the user:

> "Stitching is done. The merged dataset is at `<output.folder_path>/<output.filename>`. Use this path as `dataset.path` in your `train`, `evaluate`, or `baseline` config. Set `dataset.label_key` to `'<tag>'` (default: `'endpoint'`)."

---

## Example Config

```yaml
input:
  folder_path: ./data/raw
  dataset_names: ['Fub_human.sdf', 'Fub_mouse.sdf', 'Fub_rat.sdf', 'Fu_mic.sdf']
  label_keys: ["Fub", "Fub", "Fub", "Fu_mic"]
  tag: "endpoint"
  operator_key: "OPERATOR"
  smiles_key: "SMILES"

output:
  folder_path: ./data/merged
  filename: "dataset.csv"
```

---

## Key Behaviors

- **Output format:** Wide format — each unique compound gets one row; each endpoint gets a column named `<tag>_<i>`. Compounds present in only some files have `NaN` for missing endpoints.
- **Compound matching:** By default, compounds are matched by canonical SMILES. Set `index_key` to match by a compound ID column instead.
- **Downstream label_key:** The `tag` field (default `"endpoint"`) becomes the `dataset.label_key` value in all downstream `train`, `evaluate`, and `baseline` configs.
