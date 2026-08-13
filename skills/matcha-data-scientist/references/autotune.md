# autotune

**CLI command:** `matcha autotune --config autotune.yaml`

Runs automated hyperparameter optimization (HPO) using Optuna. Searches over architecture hyperparameters and optimizer settings, then writes the best configuration to a YAML file.

> **Critical:** `autotune` produces optimized hyperparameters **only** — it does **not** train a production model. After autotune, always chain with `matcha train` and set `model.config_path: <path to hpo_output.yaml>`. You will still need to set `label_encoder_params` manually in the train config.

---

## YAML Schema (`CLIAutotuneOutput`)

```yaml
dataset:
  path: <str>                        # required — CSV or SDF path
  label_key: <str>                   # required — label column name
  smiles_key: <str>                  # optional
  operator_key: <str>                # optional

split:
  method: <str>                      # required — "cv", "time", "cluster", or "file"
  n_subset: <int>                    # required — number of splits used for HPO evaluation
  n_bootstrap: <int>                 # optional
  frac_bootstrap: <float>            # optional
  method_params: <dict>              # required for "time", "cluster", "file"

model:
  architecture: <str>                # required — architecture to optimize
  params: <dict>                     # required — fixed params that are NOT searched
                                     # (e.g. loss_fn, num_endpoints, label_encoder_params)
  metadata: <dict>                   # optional

tuning:
  architecture_search:
    config: <str | dict>             # "default" or custom search space dict (default: "default")
    budget: <int>                    # number of Optuna trials (default: 70)
  optimizer_search:
    config: <str | dict>             # "default" or custom search space dict (default: "default")
    budget: <int>                    # number of Optuna trials (default: 70)

output:
  mlflow:                            # optional
    experiment_name: <str>
    log_dir: <str>
    tags: <dict>
    run_name: <str>
  serialization:                     # optional
    path: <str>
    quantize: <bool>
  optimum:                           # required — where to write the best params
    path: <str>                      # directory for the output file
    filename: <str>                  # filename without extension (e.g. "hpo_output")
```

---

## Step-by-Step Config Generation

### 1. Dataset
Ask: "What is the path to your dataset? What column holds the labels? Is this classification or regression?"

### 2. Split strategy
Ask: "Which split method should autotune use to evaluate each trial?"

- Use the **same split method** you plan to use in subsequent `evaluate` or `train` runs for consistency.
- Temporal splits (`time`) are recommended for production workflows.
- Use a smaller `n_subset` (e.g., 3) to keep autotune runtime manageable.

### 3. Architecture
Ask: "Which architecture do you want to optimize?"

- Only one architecture can be searched per autotune run.
- The default search space (`config: default`) covers learning rate, dropout, hidden dimensions, and number of layers for the chosen architecture.

### 4. Fixed params
Ask: "What parameters should stay fixed (not searched)?"

Fixed params go in `model.params`. At minimum, set:
- `loss_fn` — must match your training objective (see loss options below).
- `num_endpoints` — number of prediction targets.
- `label_encoder_params` — endpoint name mapping (if multi-task).
- `num_epochs` — training epochs per trial (keep lower than full training to save time, e.g. 50).

**Loss options.** `loss_fn` accepts any alias registered in `LossRegistry` — see [`docs/source/contributing/adding-a-loss.md`](../../../docs/source/contributing/adding-a-loss.md) for the authoritative alias list. Common families:

- **Base regression:** `mse`, `mae`, `huber`, `smoothl1`.
- **Base classification:** `bce`, `focal-bce`, `poly1-bce`, `weighted-bce`.
- **`bounded-*` family** — respects `<` / `>` bound info on the target so predictions inside the allowed halfspace are not penalized. Aliases: `bounded-mse`, `bounded-mae`, `bounded-huber` (+ `bounded-smoothl1`).
- **`dropout-*` family** — wraps a per-element loss and resamples a random mask of loss entries every training step, as a regularizer for wide multi-endpoint pretraining. Aliases: `dropout-mse`, `dropout-bce`, `dropout-focal-bce` (+ `dropout-mae`, `dropout-huber`, `dropout-smoothl1`, `dropout-poly1-bce`, `dropout-weighted-bce`). Pass `loss_args: {dropout: <fraction in [0, 1)>}` to control the mask rate.

The chosen alias is held fixed across all Optuna trials — the search does not explore the loss surface. If you want to compare losses, run separate `autotune` jobs.

### 5. Search budgets
Ask: "How much compute time can you allocate to HPO?"

- `architecture_search.budget` — number of trials for architecture params (layer sizes, dropout).
- `optimizer_search.budget` — number of trials for optimizer params (learning rate, weight decay).
- **Recommended defaults:** `architecture_search.budget: 100`, `optimizer_search.budget: 30`.
- Reduce to `budget: 30` / `budget: 30` for a quick exploratory run.
- Extensive search would be e.g. `budget: 150` / `budget: 60`

### 6. MLflow (optional)
Ask: "Should HPO trials be tracked in MLflow?"

### 7. Output path
Ask: "Where should the best hyperparameter config be written?"

The `output.optimum.path` + `output.optimum.filename` determine where the YAML output lands:
- Example: `path: ./optima`, `filename: chemprop_fu` → writes `./optima/chemprop_fu.yaml`
- This file path is used as `model.config_path` in the subsequent `train` config.

### 8. Generate YAML, confirm, and run

Show the complete YAML, confirm with the user, then run:

```
matcha autotune --config autotune.yaml
```

After autotune completes, remind the user:

> "Autotune is done. The optimized hyperparameters are at `<output.optimum.path>/<output.optimum.filename>.yaml`. To train a model with these settings, set `model.config_path` to that path in the `train` config. Remember to also set `model.label_encoder_params` there."

---

## Example Config

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
    split_size: 0.05

tuning:
  architecture_search:
    config: default
    budget: 100
  optimizer_search:
    config: default
    budget: 30

model:
  architecture: ChempropRegressor
  params:
    loss_fn: bounded-mse
    num_endpoints: 4
    num_epochs: 50
    batch_size: 64
    label_encoder_params:
      0: {task_label: Fub_human}
      1: {task_label: Fub_mouse}
      2: {task_label: Fub_rat}
      3: {task_label: Fu_mic}

output:
  mlflow:
    experiment_name: tuning-chemprop-fu
    tags:
      version: 1.0
    log_dir: ./mlruns
  optimum:
    path: ./optima
    filename: chemprop_fu
```

---

## Key Behaviors

- **Output is params only:** The `.yaml` written to `output.optimum` contains the best model parameters found. It does **not** contain a trained model. Pass it to `train` via `model.config_path`.
- **`label_encoder_params` is not searched:** Always specify it in `model.params` (fixed), and re-specify it in the subsequent `train` config even when using `config_path`.
- **Custom search space:** Replace `config: default` with a nested dict to define a custom Optuna search space. Only do this if you have deep familiarity with the architecture's parameter space.
- **Parallelism:** Autotune runs trials sequentially by default. For faster HPO, run multiple autotune jobs with different architectures.
