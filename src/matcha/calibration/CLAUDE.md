# `calibration/` — uncertainty calibration

Post-hoc calibration methods invoked from `../sklearn/managers/uncertainty_manager.py` (single models) and `ensemble_calibration_manager.py` (ensembles). Not called directly by users.

## Layout

```text
calibration/
├── base_calibration.py   # BaseCalibration + CalibrationRegistry
├── inductive_conformal.py  # ICP — split conformal prediction (regression + classification)
└── error_model.py        # RF/GBM/kNN/LogReg error models over descriptors (regression + classification)
```

## Contract

Every calibrator subclasses `BaseCalibration` and registers with `CalibrationRegistry` (see [`PATTERNS.md`](../../PATTERNS.md) §1–2). It exposes `fit(cal_predictions, cal_targets, ...)` and `calibrate(predictions, ...)`, and pickles via `../utils/serialization.py`. Input configs: `ICPRegressionInputModel`, `ICPClassificationInputModel`, `EMRegressionInputModel`, `EMClassificationInputModel` in `../utils/schemas/calibration.py`.

## Notes

- ICP requires a held-out calibration set — the CLI (`train.py`) splits this off explicitly before fit.
- Error models call into `../datamodules/classic/rdkit_engine.Engine` for their descriptor features; they don't reuse the model's featurizer.
- The pickled calibrator lives inside the saved-model folder next to the weights, keyed by class name.
