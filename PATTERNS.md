# MATCHA architectural patterns

Repository-wide patterns and Python conventions. Each subpackage's `CLAUDE.md` describes what's specific to that folder; this file describes the shared shapes those files assume.

Pre-commit (Ruff + Pyright, see `.pre-commit-config.yaml`) enforces formatting, unused-import removal, and type correctness where annotations exist. The **architectural patterns** (§1–9) and **Python conventions** (§10) below are project rules that pre-commit does *not* enforce — follow them by hand.

---

## 1. Base + concrete

Every model family in MATCHA has a `base_*.py` (`BaseScikitLearnGNN`, `BaseClassicModel`, `BaseDataModule`, `BaseCalibration`, ...) with the shared logic, and one file per concrete implementation. New models slot into the existing shape.

**Rule:** don't invent parallel hierarchies. If a new model doesn't fit an existing base, either extend the base or flag it in the PR — don't fork.

**Example:** `src/matcha/sklearn/graph/base_sklearn_gnn.py` → `gin.py`, `gt.py`, `gps.py`, ...

---

## 2. Registry-driven composition

Cross-cutting components (encoders, predictors, losses, optimizers, schedulers, activations, readouts, layers, datamodules, label encoders, calibrators, sklearn estimators) are wired by **string alias**, not by import. `ClassRegistry` in `src/matcha/utils/registry.py` backs every named registry:

| Registry | Defined in | Consumed by |
| --- | --- | --- |
| `ScikitLearnModelRegistry` | `sklearn/base_sklearn_model.py` | CLI, `autoload` |
| `EncoderRegistry`, `PredictorRegistry`, `ClassicModelRegistry`, `PretrainingModelRegistry` | `torch/` | `torch/models/`, sklearn wrappers |
| `DataModuleRegistry` | `datamodules/base_datamodule.py` | `sklearn/managers/datamodule_manager.py` |
| `LabelEncoderRegistry` | `datamodules/classic/label_encoder.py` | `datamodules/base_datamodule.py` |
| `OptimizerRegistry`, `SchedulerRegistry`, `LossRegistry`, `ActivationRegistry`, `ReadoutRegistry`, `LayerRegistry` | `nn/` | `torch/models/` |
| `CalibrationRegistry` | `calibration/base_calibration.py` | `sklearn/managers/uncertainty_manager.py` |

**Rule:** register with `@Registry.register("alias")`. Import the registry from the file that defines it, not from a subpackage `__init__.py`. New components must be registered before anything downstream can reach them.

---

## 3. Registered aliases are load-bearing

`autoload(path)` reads `config/manifest.yaml` and dispatches through `ScikitLearnModelRegistry[class_name]`. Datamodules, calibrators, and label encoders serialize the same way. **Renaming a class or its registry alias breaks every saved artifact with that name.**

**Rule:** treat the alias as a public interface. Adding new aliases (`@Registry.register(["gin", "gin_v2"])`) is safe; renaming or removing existing ones needs a migration.

---

## 4. Pydantic schemas at every subpackage boundary

Data crossing a subpackage boundary (except for numpy arrays and tensors) is validated by a pydantic v2 `BaseModel` in `src/matcha/utils/schemas/`. Each estimator, datamodule, calibrator, explainer, and CLI command has a matching `*InputModel`. Composed via mixins in `generic_models.py`.

**Rule:** extend an existing schema or mixin before adding an ad-hoc `dict`. If the data has no schema, it doesn't cross a boundary — it stays inside one subpackage.

Full boundary table lives in `src/matcha/utils/CLAUDE.md`.

---

## 5. `HyperparametersMixin` on every Lightning module

Encoders, `LightningModule`s, and finetuners inherit `lightning.pytorch.core.mixins.HyperparametersMixin` and call `self.save_hyperparameters()` in `__init__`. `SerializationManager` reads `self.hparams` — skipping this silently breaks save/load.

**Rule:** if it's a `LightningModule` or an encoder, it inherits `HyperparametersMixin` and calls `save_hyperparameters()`. No exceptions.

---

## 6. `locals()` capture in sklearn `__init__`

Sklearn-wrapper `__init__` methods end with:

```python
params = {k: v for k, v in locals().items() if k not in ["self", "__class__"]}
super().__init__(params)
```

This is how `BaseScikitLearnModel` captures the estimator's hyperparameters into its serialized config without a manual dict.

**Rule:** copy the idiom verbatim in new sklearn wrappers. Don't build the dict manually — the pattern is what makes the base class's introspection work.

---

## 7. Managers — composition over inheritance

Cross-cutting behavior on sklearn estimators (training, serialization, MLflow, HPO, uncertainty, explainability) is factored into `src/matcha/sklearn/managers/` and composed onto `BaseScikitLearnModel`, not inherited. Ensemble variants shadow single-model managers.

**Rule:** new cross-cutting features go into a new or existing manager. Don't sprinkle logic across every estimator class.

---

## 8. Argument prefix discipline

In `torch/models/`, argument prefixes route to the sub-component:

| Prefix | Forwards to |
| --- | --- |
| `enc_*` | encoder constructor |
| `pred_*` | predictor constructor |
| `loss_*`, `optimizer_*`, `scheduler_*` | resolved through `nn/` registries |
| bare (no prefix) | model-level (e.g. `num_endpoints`, `additional_mol_features_dim`) |

**Rule:** keep the prefix map above stable. Don't invent new prefixes; if a genuinely new sub-component appears, add its prefix here first.

---

## 9. Tests mirror source layout

`src/matcha/foo/bar.py` has tests at `tests/foo/test_bar.py`. The sklearn model test suite is parametrized over registered estimators — new estimators plug in via `_ARCH_KWARGS`, `_CLASSIFIER_CLASSES`, `_REGRESSOR_CLASSES` in `tests/sklearn_models/conftest.py`.

**Rule:** every source file gets a test file at the mirrored path. Warnings are errors in pytest (`filterwarnings = ["error", ...]`) — route third-party noise through `matcha.utils.warnings.silence_nuisance_warnings`, don't filter locally.

---

## 10. Python conventions

Not enforced by Ruff's current ruleset — apply by hand and in review.

### Type annotations

Annotate **every** parameter and return value on every method, including `-> None` for procedures. Use modern union syntax (Python ≥ 3.12 is the target, pinned in `.mise.toml` and `pyproject.toml`):

| Use | Not |
| --- | --- |
| `X \| Y` | `Union[X, Y]` |
| `X \| None` | `Optional[X]` |
| `list[str]`, `dict[str, int]`, `tuple[int, ...]` | `List`, `Dict`, `Tuple` |

Pyright runs in basic mode: it checks annotations that exist but doesn't require them. Missing return types will silently pass — add them.

### Docstrings — reST / Sphinx style

Every module, class, and public method. One-line summary, blank line, then params. Types in `:param` blocks must match the annotation. Add a short `Example::` block only when the call pattern isn't obvious from the signature.

```python
"""Fit a molecular property model.

:param list[Mol] mols: RDKit molecules
:param np.ndarray | None y: targets, or None for unsupervised
:return None:
:raises ValueError: if `mols` and `y` differ in length

Example::

    model = GINClassifier()
    model.fit(train_mols, train_y)
"""
```

Keep it terse — no multi-paragraph prose. Link to a subpackage `CLAUDE.md` or `docs/source/contributing/` for detail.

### Import ordering — four groups, blank-line separated

Ruff's current config does **not** sort these. Do it by hand:

```python
# 1. stdlib
import datetime
from pathlib import Path

# 2. scientific / cheminformatics
import numpy as np
from rdkit.Chem.rdchem import Mol

# 3. ML / deep-learning frameworks
import lightning as L
import torch

# 4. internal (matcha.*)
from matcha.utils.registry import ClassRegistry
```

### Naming

| Kind | Convention | Example |
| --- | --- | --- |
| Classes | `PascalCase` | `DataModuleManager` |
| Public functions / methods | `snake_case` | `compute_uncertainty` |
| Private attrs / methods | `_snake_case` | `_create_model` |
| Factory methods | `_create_*` / `_make_*` | `_create_datamodule` |
| Module-level constants | `UPPER_SNAKE_CASE` | `MAX_BATCH_SIZE` |
| Files / modules | `snake_case` | `base_sklearn_model.py` |
| Test files | `test_*.py` | `test_registry.py` |
| Test classes | `Test<Subject>` | `TestClassRegistry` |

### f-strings only

```python
# do
logger.info(f"Loaded {n} molecules from {path}")
# don't
logger.info("Loaded %d molecules from %s" % (n, path))
logger.info("Loaded {} molecules from {}".format(n, path))
```

### No mutable defaults

Use `None` and guard inside the body. Shared mutable defaults leak state across calls.

```python
def setup(self, tags: dict | None = None) -> None:
    if tags is None:
        tags = {}
```

### Error handling

- Validate inputs at the boundary — pydantic for structured data (see §4), `isinstance` in `__init__` for invariants.
- Never bare `except:` — always name the exception type.
- Error messages must include the valid options:

```python
valid = ", ".join(repr(k) for k in self.keys())
raise ValueError(f"'{key}' is not valid. Valid options: {valid}")
```

### Logging

Create a per-component logger at init:

```python
self.logger = get_default_logger("SKLEARN_MODEL")
```

Log at the **start and end** of major operations (`fit`, `predict`, `save`, `load`) — never inside tight inner loops (destroys performance). Use `logger.info` for normal flow, `logger.warning` for recoverable anomalies, `logger.error` for failures.

### `pathlib.Path` for filesystem

No `os.path.join`, no `os.makedirs`, no string concatenation for paths.

```python
# do
output_dir = Path(folder)
output_dir.mkdir(parents=True, exist_ok=True)
config = output_dir / "config.yaml"

# don't
os.makedirs(folder, exist_ok=True)
config = os.path.join(folder, "config.yaml")
```

### `__all__` in every public module

Explicit export surface — prevents namespace leaks under `from module import *` and makes the intended API self-documenting. See any `sklearn/__init__.py` for a canonical shape.

### Pydantic v2 `BaseModel` for structured data

Never `@dataclass`. Use `@field_validator` for field checks, `@model_validator(mode="after")` for cross-field constraints. §4 explains where these go (`src/matcha/utils/schemas/`).

---

## When to reach for a new pattern

Add a new top-level pattern here (and update the referring CLAUDE.md files) only when the pattern shows up in **three or more** subpackages. Two-subpackage overlaps stay in the individual CLAUDE.md files. This keeps PATTERNS.md from becoming a catch-all.

## Related docs

- **Repo layout:** `CLAUDE.md`
- **Package layout + dependency direction:** `src/matcha/CLAUDE.md`
- **Subpackage internals:** `src/matcha/<pkg>/CLAUDE.md`
- **Adding a model, manager, datamodule, calibrator:** `docs/source/contributing/`
- **Python style:** §10 above (project rules); `.pre-commit-config.yaml` for what Ruff / Pyright enforce automatically
