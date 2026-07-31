# Adding a new loss function

```{note}
This page is written for LLM agents contributing to MATCHA. It trades prose for a step-by-step recipe with concrete file paths and code snippets.
```

All loss functions live in `src/matcha/nn/losses.py` and register with `LossRegistry`. Lightning modules resolve `loss_fn="<alias>"` strings through this registry — no other file needs to change to make a new loss available to every model.

A new loss touches two locations:

1. A `nn.Module` subclass in `src/matcha/nn/losses.py`, decorated with `@LossRegistry.register(alias=...)`.
2. Coverage in `tests/nn/test_losses.py` (registry key + at least one forward-pass check).

---

## 1. Add the loss class

Subclass `nn.Module`, register with an alias, and implement `forward(inputs, targets) -> torch.Tensor`. Accept `reduction` if the loss can be broadcast per-element — several composed losses (`MultitaskLoss`, `MultiLoss`, `GradNormLoss`, `BoundedLoss`) instantiate their inner loss with `reduction="none"`, so this is required if you want your loss to compose.

```python
# src/matcha/nn/losses.py
@LossRegistry.register(alias="dummy-mse")
class DummyScaledMSELoss(nn.Module):
    """Dummy example: MSE scaled by a constant. Not useful — shape only."""

    def __init__(self, scale: float = 1.0, reduction: str = "mean"):
        """
        :param float scale: Multiplicative factor on the squared error.
        :param str reduction: Reduction mode: ``'mean'``, ``'sum'``, or ``'none'``.
        """
        super().__init__()
        self.scale = scale
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = self.scale * (inputs - targets) ** 2
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
```

- The **alias is load-bearing** — it's what `loss_fn="dummy-mse"` resolves to from YAML configs, pydantic schemas, and Lightning modules. Lowercase, kebab-case, no renaming later without a migration.
- If your loss simply wraps a `torch.nn` class, you can subclass it directly (see `MSELoss`, `BCELoss`, `HuberLoss` — one-liners).
- If your loss needs multitask semantics (per-column NaN masking, per-task weights, epoch scheduling), follow `MultitaskLoss` / `MultiLoss` — do not reimplement the masking logic.

## 2. Tests — `tests/nn/test_losses.py`

Two touches:

1. Add the alias to `TestLossRegistryKeys.EXPECTED_KEYS`:

   ```python
   # tests/nn/test_losses.py
   class TestLossRegistryKeys:
       EXPECTED_KEYS = [
           ...,
           "dummy-mse",
       ]
   ```

2. Add a `TestDummyScaledMSELoss` class with at least a scalar-output check and a value check:

   ```python
   class TestDummyScaledMSELoss:
       def test_output_scalar_mean(self):
           loss_fn = LossRegistry["dummy-mse"](scale=2.0, reduction="mean")
           preds = torch.randn(8, 1)
           targets = torch.randn(8, 1)
           loss = loss_fn(preds, targets)
           assert loss.dim() == 0

       def test_matches_manual(self):
           loss_fn = LossRegistry["dummy-mse"](scale=3.0, reduction="mean")
           preds = torch.tensor([[1.0], [2.0]])
           targets = torch.tensor([[0.0], [0.0]])
           expected = 3.0 * ((1.0**2 + 2.0**2) / 2)
           assert torch.isclose(loss_fn(preds, targets), torch.tensor(expected))
   ```

- Resolve through `LossRegistry["dummy-mse"](...)` in tests, not by importing the class — this exercises the alias, which is what real callers use.

---

## Checklist

- [ ] `nn.Module` subclass in `src/matcha/nn/losses.py`, registered via `@LossRegistry.register(alias="...")`.
- [ ] Supports `reduction="none"` if the loss should compose inside `MultitaskLoss` / `MultiLoss` / `GradNormLoss` / `BoundedLoss`.
- [ ] Alias added to `TestLossRegistryKeys.EXPECTED_KEYS` in `tests/nn/test_losses.py`.
- [ ] Forward-pass test class covering shape + at least one numeric check, resolving through `LossRegistry[alias]`.
- [ ] `uv run pytest tests/nn/test_losses.py` passes.
- [ ] Codecov PR check stays green — new lines are covered and total coverage doesn't drop (see `CONTRIBUTING.md` → Testing).
