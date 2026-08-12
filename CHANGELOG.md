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
