# Benchmarking uncertainty with UNIQUE

[UNIQUE](https://github.com/Novartis/UNIQUE) benchmarks uncertainty
quantification methods from a table of labels, predictions and uncertainty
inputs. This page shows how to turn `matcha predict` outputs into that table
for a single-task regression model.

## 1. Split, train, predict

Split your data into `train.csv`, `calibration.csv` and `test.csv`, each with
a SMILES column and a label column. Train on `train.csv` with `matcha train`
(see `src/matcha/cli/example_configs/train.yaml`). Use an ensemble
(`model.ensemble: 5`) so the model reports a standard deviation.

Run `matcha predict` once per split, keeping the input columns so the labels
end up in the output:

```yaml
# predict_test.yaml — repeat for train and calibration
dataset:
  path: /path/to/test.csv
  smiles_key: SMILES
  keep_cols: True

output: /path/to/predictions/test

model:
  path: /path/to/models/my_model
  inference:
    accelerator: cpu
    devices: 1
    batch_size: 256
```

Each run writes `output.csv` with your input columns plus
`<task_label>_prediction` and `<task_label>_std`.

## 2. Build the UNIQUE input table

UNIQUE needs a subset column with the values `TRAIN`, `CALIBRATION` and
`TEST`, and expects the ensemble *variance* rather than the standard
deviation:

```python
import pandas as pd

task = "my_task"  # the task_label used at training time

subsets = {"TRAIN": "train", "CALIBRATION": "calibration", "TEST": "test"}
df = pd.concat(
    [
        pd.read_csv(f"/path/to/predictions/{folder}/output.csv", index_col=0)
        .assign(subset=subset)
        for subset, folder in subsets.items()
    ],
    ignore_index=True,
)
df["variance"] = df[f"{task}_std"] ** 2
df.to_csv("/path/to/unique_input.csv", index_label="ID")
```

## 3. Run UNIQUE

```yaml
# unique_config.yaml
data_path: "/path/to/unique_input.csv"
output_path: "/path/to/unique_output"
id_column_name: "ID"
labels_column_name: "label"            # your label column
predictions_column_name: "my_task_prediction"
which_set_column_name: "subset"
model_name: "my_model"
problem_type: "regression"
mode: "compact"
inputs_list:
  - ModelInputType:
      column_name: "variance"
```

```python
from unique import Pipeline

pipeline = Pipeline.from_config("unique_config.yaml")
uq_methods_outputs, uq_evaluation_outputs = pipeline.fit()
```

See the [UNIQUE docs](https://opensource.nibr.com/UNIQUE/) for data-based
inputs (e.g. fingerprints for distance-to-training methods), error models and
the full list of configuration options.
