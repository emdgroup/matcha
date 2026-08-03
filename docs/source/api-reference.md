# API reference

The MATCHA API is organised into a handful of subpackages, each covering a
different layer of the toolkit. The full reference is generated automatically
from the source docstrings — use the links below as entry points, and follow
the `matcha.*` tree from there.

## Subpackages

- [`matcha.sklearn`](api/matcha/sklearn/index) — scikit-learn-style estimators
  (single models, ensembles, finetuners) that are the main user-facing API.
- [`matcha.torch`](api/matcha/torch/index) — PyTorch `Module` and
  `LightningModule` implementations backing the estimators.
- [`matcha.nn`](api/matcha/nn/index) — reusable neural network building blocks
  (layers, heads, losses, activations).
- [`matcha.datamodules`](api/matcha/datamodules/index) — Lightning
  `DataModule`s and dataset wrappers for molecular data.
- [`matcha.calibration`](api/matcha/calibration/index) — post-hoc calibration
  and uncertainty estimation utilities.
- [`matcha.explainability`](api/matcha/explainability/index) — attribution and
  interpretability methods.
- [`matcha.cli`](api/matcha/cli/index) — command-line entry points.
- [`matcha.utils`](api/matcha/utils/index) — shared helpers used across the
  package.

## Full module index

For the complete auto-generated reference, see the
[top-level `matcha` package](api/matcha/index).
