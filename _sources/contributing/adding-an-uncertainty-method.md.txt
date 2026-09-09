# Adding a new uncertainty method

```{note}
This page is written for LLM agents contributing to MATCHA. It trades prose for a step-by-step recipe with concrete file paths and code snippets.
```

Uncertainty methods are dispatched by the `UncertaintyMethod` literal enum defined in `src/matcha/utils/schemas/generic_models.py`. Currently the enum covers `"mc-dropout"` (default) and `"mve"`. Adding a new method — e.g. deep ensembles or evidential regression — is a three-edit change: extend the literal, add a manager branch, and declare any pairing rules.

---

## 1. Extend the literal — `src/matcha/utils/schemas/generic_models.py`

Widen the `UncertaintyMethod` type alias with the new kebab-case value:

```python
# src/matcha/utils/schemas/generic_models.py
UncertaintyMethod: TypeAlias = Literal["mc-dropout", "mve", "evidential"]
```

If the new method has pairing constraints with `loss_fn` or `scaler_type`, declare them alongside the existing `_validate_mve_pairing` (`ClassicMatchaModel`) and `_validate_mve_scaler_pairing` (`ScikitLearnInputModel` in `sklearn_api.py`). Every uncertainty-related validation lives in the Pydantic layer — do not add checks at the sklearn constructor site.

## 2. Add a manager branch — `src/matcha/sklearn/managers/uncertainty_manager.py`

`UncertaintyManager.compute` dispatches by explicit value. Add a new `elif` branch and a `_compute_<method>` helper (mirror `_compute_mc_dropout` / `_compute_mve`):

```python
# src/matcha/sklearn/managers/uncertainty_manager.py
method = getattr(model_instance._model, "uncertainty_method", "mc-dropout")

if method == "mc-dropout":
    std = self._compute_mc_dropout(...)
elif method == "mve":
    std = self._compute_mve(...)
elif method == "evidential":
    std = self._compute_evidential(...)
else:
    raise NotImplementedError(f"Unknown uncertainty method: {method!r}")
```

The `else` branch is the safety net — any literal value without a registered manager branch fails fast.

## 3. Propagate the signature — Lightning models + sklearn estimators

Every classic Lightning model in `src/matcha/torch/models/classic/*_model.py` accepts `uncertainty: UncertaintyMethod = "mc-dropout"`. Extending the literal is enough for those signatures. Every sklearn **regressor** under `sklearn/{tabular,graph,graph3d,clm}/*.py` uses the same `UncertaintyMethod` alias — again nothing to touch. Sklearn **classifiers** narrow their type to `Literal["mc-dropout"]` (MVE is regression-only). If the new method is classifier-compatible, widen the classifier signature site-by-site.

## 4. Tests

- `tests/utils/schemas/test_generic_models.py`: parametrize the "invalid uncertainty literal" test to confirm the new value is accepted; add pairing-rejection cases if applicable.
- `tests/sklearn_managers/test_uncertainty_manager.py`: extend `TestUncertaintyManagerDispatch::test_dispatch_selects_correct_branch` with the new method.
- End-to-end integration test under `tests/sklearn_models/` mirroring `test_mve.py` — a fit / predict round-trip that exercises the new branch.

---

## Checklist

- [ ] Kebab-case literal value added to `UncertaintyMethod` in `src/matcha/utils/schemas/generic_models.py`.
- [ ] `_compute_<method>` helper + `elif` branch in `src/matcha/sklearn/managers/uncertainty_manager.py`.
- [ ] Pairing rules (loss / scaler compatibility) declared as Pydantic validators, not at the sklearn constructor site.
- [ ] Classifier signature widened only if the method is classifier-compatible.
- [ ] Dispatch + pairing tests added.
- [ ] `uv run pytest -k 'not gpu'` passes locally.
- [ ] Codecov PR check stays green (see `CONTRIBUTING.md` → Testing).
