# matcha — Workflow Guide

## Decision Tree: Which Commands Do I Need?

```
Do I need to merge multiple datasets?
├── Yes → start with: stitch
└── No  → skip stitch

Do I need to train a production model?
├── Yes, with optimized hyperparameters → autotune → train
├── Yes, with default hyperparameters   → train
└── No → skip train

Do I need to benchmark before training?
├── Yes, with a MATCHA model      → evaluate
├── Yes, with a classical model   → baseline
├── Yes, with both and compare    → evaluate + baseline → summarize
└── No → skip evaluate / baseline

Do I need to score new molecules?
├── Yes → predict (requires a trained model from train)
└── No → skip predict

Do I need to compare multiple runs statistically?
├── Yes → summarize
└── No → skip summarize

Do I want to pretrain a foundation model first?
├── Yes, on a large multi-task activity dataset → prepare_dataset → pretrain_multitask → train (FinetuningRegressor)
├── Yes, self-supervised on SMILES (MLM)        → pretrain_encoder (mlm) → train (FinetuningRegressor)
├── Yes, self-supervised with QM labels (2D)    → pretrain_encoder (graph) → train (FinetuningRegressor)
├── Yes, self-supervised with 3D coords         → pretrain_encoder (graph3d) → train (FinetuningRegressor)
└── No → skip pretraining
```

---

## Pattern 1 — Basic Benchmarking

**Goal:** Estimate model performance on your dataset before committing to full training.

**Commands:** `evaluate` (optionally + `baseline` → `summarize`)

1. Run `evaluate` — configure and run cross-validation for your MATCHA model.
2. (Optional) Run `baseline` — run a RandomForest baseline with the **same split** for fair comparison.
3. (Optional) Run `summarize` — aggregate results and run a statistical test.

---

## Pattern 2 — Production Model (autotune → train → predict)

**Goal:** Train an ensemble model with optimized hyperparameters, then score new molecules.

**Commands:** `autotune` → `train` → `predict`

1. Run `autotune` — run HPO to find optimal architecture and optimizer settings. This produces a YAML file but does **not** train a model.
2. Run `train` — train an ensemble using the HPO output via `model.config_path`. Set `model.ensemble: 5` (or higher) for a more robust final model.
3. Run `predict` — point `model.path` at the serialized model directory from step 2.

**Artifact chain:**
```
autotune → hpo_output.yaml
               ↓ (model.config_path)
           train → <model_dir>/config/manifest.yaml + weights
                       ↓ (model.path)
                   predict → output.csv
```

---

## Pattern 3 — Multi-Task Full Pipeline (stitch → evaluate → summarize)

**Goal:** Merge multiple single-endpoint files, benchmark a multi-task model, report results.

1. Run `stitch` — merge individual SDF/CSV files into a single multi-task CSV.
2. Run `evaluate` — cross-validate a multi-task MATCHA model on the stitched dataset.
3. (Optional) Run `baseline` — run a RandomForest baseline on the same stitched dataset with the **identical** split config.
4. (Optional) Run `summarize` — compare model performances and run a statistical test.

---

## Pattern 4 — Model Comparison (evaluate + baseline → summarize)

**Goal:** Quantitatively compare a MATCHA model against a scikit-learn baseline.

1. Run `evaluate` — set `output.serialization.path: ./results/matcha_model`.
2. Run `baseline` — use the **exact same** `split` block as step 1. Set `output.serialization.path: ./results/baseline`.
3. Run `summarize` — set `root_dir: ./results`.

**Split matching is critical:** Mismatched splits produce invalid comparisons.

---

## Pattern 5 — Large-Scale Model Comparison (autotune for N Models)

1. Run `autotune` — find optimal params for each architecture.
2. Run `evaluate` — run evals using the HPO output via `model.config_path`.
3. Run `summarize` — compare all results.

---

## Pattern 6 — Baseline Only (Quick Sanity Check)

1. Run `baseline` — configure a RandomForestRegressor or RandomForestClassifier with `rdkit_all_descriptors` features and 10-fold CV.

---

## Pattern 7a — Foundation Model via Multitask Pretraining

**Commands:** `prepare_dataset` → `pretrain_multitask` → `train` (FinetuningRegressor/Classifier)

1. Run `prepare_dataset` — merge parquet/CSV files into a label matrix. Set `datasets.sparse: true` (default) for a `scipy.sparse.csr_matrix` per split, or `datasets.sparse: false` for a dense `float32` array per split with `NaN` marking missing entries. Choose dense when the underlying matrix is largely populated (e.g. ≥90%).
2. Run `pretrain_multitask` — train a graph model on the prepared labels. The command auto-detects the storage mode from `task_metadata.json` (`storage_mode: "sparse" | "dense"`); no additional flag is set on the pretraining side. If `{dataset_dir}/train_coords.npz` and `{dataset_dir}/val_coords.npz` are present, they are auto-loaded and threaded into the datamodule for 3D-aware bases (`E3GNNPretraining`, `GPS3DPretraining`, `GT3DPretraining`); with a 2D-only base the coords are silently dropped with a one-shot `logging.warning`. `prepare_dataset` does **not** emit coords — produce the npz files externally. See `references/pretrain-multitask.md § Coords Auto-Discovery` for the file convention and packing layout.
3. Run `train` — set `model.architecture: FinetuningRegressor` and `model.path_to_pretrained: <pretrain output dir>`.

**Artifact chain:**
```
prepare_dataset → train_tasks_{sparse.npz|dense.npy} + val_tasks_{sparse.npz|dense.npy} + task_metadata.json
                              ↓ (dataset.dataset_dir)
                     pretrain_multitask → model.ckpt + config/manifest.yaml
                                              ↓ (model.path_to_pretrained)
                                          train → manifest.yaml + weights
```

---

## Pattern 7b — Foundation Model via Encoder Pretraining

**Commands:** `pretrain_encoder` → `train` (FinetuningRegressor/Classifier)

1. Run `pretrain_encoder` — choose mode:
   - **MLM mode** (`task_type: mlm`): pretrain `RoFormerMLM` on SMILES strings.
   - **Graph mode** (`task_type: graph`): pretrain a 2D graph encoder (`GINPretraining`, `GatedGCNPretraining`, `GPSPretraining`, `GTPretraining`, `AttentiveFPPretraining`) on node/graph targets.
   - **Graph3D mode** (`task_type: graph3d`): pretrain a 3D-coordinate-aware encoder (`E3GNNPretraining`, `GPS3DPretraining`, `GT3DPretraining`) on node/graph targets with per-atom coordinates. Requires `train_coords` / `val_coords` npz files alongside the four `y_graph` / `y_node` files. See `references/pretrain-encoder.md § Graph3D Mode` for the schema, packing contract, and starter `model.params` per architecture; a runnable example lives at `src/matcha/cli/example_configs/pretrain_encoder_graph3d.yaml`.
2. Run `train` — set `model.architecture: FinetuningRegressor` and `model.path_to_pretrained: <pretrain output dir>`.

---

## Output Artifact Reference

| Command | Output path | Key files |
|---|---|---|
| `stitch` | `output.folder_path/output.filename` | Merged CSV |
| `train` | `output.serialization.path/` | `config/manifest.yaml`, model weights, `train.log`, `cfg.yaml` |
| `evaluate` | `output.serialization.path/` | `performance.json`, `plots/`, `cfg.yaml` |
| `baseline` | `output.serialization.path/` | `performance.json`, `plots/`, `cfg.yaml` |
| `autotune` | `output.optimum.path/output.optimum.filename` | `hpo_output.yaml` |
| `predict` | `output` (flat path) | `output.csv`, `failed.csv`, `cfg.yaml` |
| `summarize` | `output_path` | `summary_analysis.json`, HTML plots |
| `prepare_dataset` | `output/` | Sparse mode: `train_tasks_sparse.npz`, `val_tasks_sparse.npz`. Dense mode: `train_tasks_dense.npy`, `val_tasks_dense.npy`. Both: `train_molecules.parquet`, `val_molecules.parquet`, `task_metadata.json` (with `storage_mode`), `datacard.json` |
| `pretrain_multitask` | `output.serialization/` | `model.ckpt`, `config/manifest.yaml` |
| `pretrain_encoder` | `output.serialization/` | `encoder.ckpt`, `model.ckpt`, `config/manifest.yaml` |

---

## Tips

- **Ensemble size:** Use `model.ensemble: 5` (or higher) when you want a more robust final model.
- **Autotune → train chaining:** After autotune, set `model.config_path: <path to hpo_output.yaml>` in your train config. You still need to set `model.label_encoder_params` manually even when using `config_path`.
- **Split consistency:** When running `evaluate` and `baseline` for comparison, always use the identical `split` block.
- **Statistical tests:** `summarize` defaults to non-parametric (Friedman + Wilcoxon + Benjamini-Hochberg). Use parametric only when you have ≥10 splits and metrics are approximately normally distributed.
- **Label scaling:** For regression scenarios, always ask if labels need scaling due to skew using `model.label_transform_map` (e.g., `log10`, `log10p`).
- **Label extraction:** When loading datasets with multiple labels or operators, don't use `*` or similar regex in the string, just put the common substring. Do: `label_key: endpoint`, don't: `label_key: *_endpoint`.
