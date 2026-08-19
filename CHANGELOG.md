## v0.0.22 (2026-08-19)

### Feat

- **finetuning**: add keep_existing_predictor flag to Finetuner

## v0.0.21 (2026-08-18)

### Feat

- **datamodules**: wire multi-conformer ETKDG+MMFF pipeline into Graph3DDataModule._calculate_coords
- **datamodules**: add multi-conformer ETKDG+MMFF helpers for 3D coords

### Refactor

- **datamodules**: generalize embed timeout helper and bump default budget

### Perf

- **datamodules**: reduce conformer defaults to numConfs=5, maxIters=500

## v0.0.20 (2026-08-18)

### Feat

- **datamodules**: thread optional coords through Graph3DDataModule to skip ETKDG re-embedding
- **datamodules**: add shared coords_utils helpers for 3D atomic coordinates

### Refactor

- **datamodules**: delegate 3D pretraining coord helpers to coords_utils

## v0.0.19 (2026-08-17)

### Fix

- **torch**: resolve leaf encoder when finetuning nested Finetuner artifacts

## v0.0.18 (2026-08-16)

### Fix

- **cli**: sample dense-mode validation compounds globally so val size tracks sampling_rate

## v0.0.17 (2026-08-16)

### Fix

- **losses**: honor reduction in DropoutLoss so it returns per-element output when wrapped by MultiLoss/MultitaskLoss

## v0.0.16 (2026-08-14)

### Fix

- **losses**: drop caller-supplied reduction kwarg in DropoutLoss so it can be wrapped by MultiLoss/MultitaskLoss
- **finetuner**: advance global_step in full manual-opt path so MultiLoss weight curriculum interpolates

## v0.0.15 (2026-08-14)

### Fix

- **finetuner**: unpack MultiLoss tuple in training_step

## v0.0.14 (2026-08-14)

### Fix

- **finetuner**: route wrapper MLM path through forward_tokens

## v0.0.13 (2026-08-14)

### Fix

- **losses**: revert MultiLoss to always-tuple return + align callers (stage 1/1)

## v0.0.12 (2026-08-13)

### Feat

- **pretraining**: add GPS3DPretraining and GT3DPretraining (stage 1/1)
- **pretraining**: sparsity-agnostic multitask loader + collate (stage 4/5)
- **cli**: wire dense branch into prepare_dataset (stage 3/5)
- **cli**: dense-mode prep helpers (stage 2/5)
- **cli**: rename prepare command + sparse schema toggle (stage 1/5)
- **losses**: register 8 dropout-* concrete aliases (stage 2/3)
- **losses**: add DropoutLoss wrapper for per-label random masking (stage 1/3)
- **cli**: auto-discover multitask coords + docs update (stage 4/4)
- **cli**: wire graph3d branch into pretrain_encoder (stage 3/4)
- **pretraining**: thread coords through on-the-fly wrappers (stage 2/4)
- **cli**: schema + shared coords loader for graph3d pretraining (stage 1/4)
- **pretraining**: add E3GNNPretraining model + schema (stage 3/4)
- **pretraining**: add Graph3DPretrainingDataModule (stage 2/4)

### Refactor

- **encoders**: unify 3D encoders on graph.pos contract (stage 1/4)
- **pretraining**: unify canonical + MLM RoFormer (stage 4/5)
- **pretraining**: delete PretrainingEncoder duplicates for graph models (stage 3/5)
- **encoders**: hoist forward to base for gatedgcn/gps/gt/attentivefp (stage 2/5)
- **encoders**: unify canonical + pretraining GIN (stage 1/5)

## v0.0.11 (2026-08-12)

### Feat

- **predictors**: add BatchEnsembleLinear primitive for SNN (stage 1/2)

### Fix

- **predictors**: rewire SNN with BatchEnsembleLinear (stage 2/2)
- **encoders**: reconcile E3GNN with reference implementation (stage 1)
- **layers**: reconcile SpatialEncoder / SpatialEncoder3d internals (stage 3/3)
- **encoders**: reconcile GPS3D and GT3D encoders (stage 2/3)
- **encoders**: reconcile GPS and GT 2D transformer encoders (stage 1/3)
- **encoders**: reconcile GIN, AttentiveFP, GatedGCN with reference implementations

### Refactor

- **encoders**: drop PyG private-API dependency in E3GNN and land cleanups (stage 2)

## v0.0.10 (2026-08-03)

### Feat

- first commit
