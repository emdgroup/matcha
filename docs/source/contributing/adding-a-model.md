# Adding a new model

```{note}
This page is written for LLM agents contributing to MATCHA. It trades prose for a step-by-step recipe with concrete file paths and code snippets.
```

This recipe walks through everything that has to line up when adding a new model architecture to MATCHA. The reference implementation used throughout is **GIN** — every step below points at the corresponding file in the existing GIN pipeline so you can copy the shape.

A new model touches five layers, in this order:

```text
torch/encoders/        →  torch/models/classic/   →  utils/schemas/    →  sklearn/<family>/   →  tests/
(encoder)                 (Lightning module)         (pydantic schema)    (Classifier + Regressor)
```

Skipping a layer will break something downstream: unregistered encoders won't be resolvable by string alias, missing schemas will fail input validation, missing sklearn wrappers can't be `autoload`ed, and the parametrized test suite won't cover the new class.

---

## 1. Encoder — `src/matcha/torch/encoders/<name>.py`

Subclass the appropriate base (`BaseGraphEncoder`, `BaseGraphEncoder3D`, or the CLM / tabular equivalents) and register with `EncoderRegistry`:

```python
# see src/matcha/torch/encoders/gin.py
from matcha.torch.encoders.base_encoder import EncoderRegistry
from matcha.torch.encoders.base_graph_encoder import BaseGraphEncoder
from lightning.pytorch.core.mixins import HyperparametersMixin

@EncoderRegistry.register()
class GIN(BaseGraphEncoder, HyperparametersMixin):
    def __init__(self, num_layers, atom_input_dim, atom_hidden_dim, ...):
        super().__init__(...)
        self.save_hyperparameters()
        ...
```

- Inherit `HyperparametersMixin` and call `save_hyperparameters()` — the serializer relies on it.
- Encoder arguments should be prefixed nothing (they're internal); the `enc_*` prefix appears at the Lightning-module level in step 2.

## 2. Lightning module — `src/matcha/torch/models/classic/<name>_model.py`

Wrap the encoder + a predictor head into a `BaseClassicModel` subclass and register with `ClassicModelRegistry`:

```python
# see src/matcha/torch/models/classic/gin_model.py
from matcha.torch.models.classic.base_classic_model import (
    BaseClassicModel, ClassicModelRegistry,
)
from matcha.torch.encoders.gin import GIN

@ClassicModelRegistry.register()
class GINModel(BaseClassicModel, HyperparametersMixin):
    def __init__(self, enc_num_layers=6, enc_atom_hidden_dim=300, ...,
                 pred_hidden_dims=[512, 256], loss_fn="mse", optimizer="adam", ...):
        ...
```

- Encoder-facing args use the `enc_*` prefix; predictor-facing args use `pred_*`. Loss / optimizer / scheduler are string aliases resolved through the `nn/` registries.
- Add the class to `torch/models/classic/__init__.py` (import + `__all__`).

For finetuning or pretraining variants, use `BaseFinetuningModel` / `BasePretrainingModel` under the corresponding subfolder instead.

## 3. Schema — `src/matcha/utils/schemas/`

Two touches:

1. **Add an encoder mixin** in `generic_models.py` for the fields specific to this architecture:

   ```python
   # see src/matcha/utils/schemas/generic_models.py
   class GINMixin(BaseDataModel):
       enc_aggregation: str
       enc_norm: str | None
       enc_jk: str
   ```

2. **Compose the top-level input model** in `torch_api.py`:

   ```python
   class GINInputModel(ClassicMatchaModel, GraphMixin, GINMixin):
       """Schema for the GIN model configuration."""
   ```

- Reuse existing mixins (`ClassicMatchaModel`, `GraphMixin`, `TabularMixin`, `CLMMixin`, ...) wherever possible — only put genuinely new fields into the new mixin.
- Update the `__init__.py` / `__all__` of `utils/schemas/` and `utils/schemas/torch_api.py`.

## 4. Sklearn wrapper — `src/matcha/sklearn/<family>/<name>.py`

Ship a **`Classifier` + `Regressor` pair** subclassing the family base + the appropriate mixin, and register both:

```python
# see src/matcha/sklearn/graph/gin.py
from matcha.torch.models.classic import GINModel
from matcha.sklearn.base_sklearn_model import (
    ScikitLearnModelRegistry,
    ScikitLearnClassifierMixin,
    ScikitLearnRegressorMixin,
)
from matcha.sklearn.graph.base_sklearn_gnn import BaseScikitLearnGNN

@ScikitLearnModelRegistry.register()
class GINClassifier(BaseScikitLearnGNN, ScikitLearnClassifierMixin):
    def __init__(self, enc_num_layers=3, enc_atom_hidden_dim=256, ..., label_encoder_params={}):
        params = {k: v for k, v in locals().items() if k not in ["self", "__class__"]}
        super().__init__(params)

@ScikitLearnModelRegistry.register()
class GINRegressor(BaseScikitLearnGNN, ScikitLearnRegressorMixin):
    ...
```

- The `params = {k: v for k, v in locals().items() ...}` idiom is how the base captures the hyperparameters — keep it exactly.
- The registered alias (default: lowercase class name) is what `autoload` looks up from `manifest.yaml`. Do not rename it later without a migration.
- Update `sklearn/<family>/__init__.py` and `sklearn/__init__.py` (both `import` and `__all__`).

## 5. Tests — `tests/sklearn_models/conftest.py` + friends

The suite is parametrized over every `Classifier` / `Regressor` — you only need to plug the new class in:

```python
# see tests/sklearn_models/conftest.py
_ARCH_KWARGS[GINClassifier] = {
    **_GRAPH_SHARED,
    "enc_num_layers": 1,          # keep tiny — this runs on CPU in CI
    "enc_atom_hidden_dim": 32,
    "pred_hidden_dims": [32],
}
_ARCH_KWARGS[GINRegressor] = {**_ARCH_KWARGS[GINClassifier]}

_CLASSIFIER_CLASSES = [..., GINClassifier, ...]
_REGRESSOR_CLASSES = [..., GINRegressor, ...]
```

- Match kwargs to the **smallest** configuration that exercises the model — no big hidden dims, no expensive positional encoders.
- If the model needs a distinct datamodule or feature setup, mirror an existing entry in `tests/datamodules/` first.

---

## Checklist

- [ ] Encoder in `torch/encoders/`, registered with `EncoderRegistry`, calls `save_hyperparameters()`.
- [ ] Lightning module in `torch/models/classic/` (or `finetuning/` / `pretraining/`), registered with the family registry, `enc_*` / `pred_*` prefixes.
- [ ] Mixin in `utils/schemas/generic_models.py` + input model in `utils/schemas/torch_api.py`, both re-exported.
- [ ] `Classifier` + `Regressor` in `sklearn/<family>/`, both registered with `ScikitLearnModelRegistry`, both exported from `sklearn/__init__.py`.
- [ ] Test kwargs added to `tests/sklearn_models/conftest.py` and the class added to `_CLASSIFIER_CLASSES` / `_REGRESSOR_CLASSES`.
- [ ] `uv run pytest -k 'not gpu'` passes locally.
- [ ] Codecov PR check stays green — new lines are covered and total coverage doesn't drop (see `CONTRIBUTING.md` → Testing).

If any of these steps feels awkward — e.g. the new model doesn't cleanly fit an existing family — flag it in the PR rather than inventing a parallel hierarchy.
