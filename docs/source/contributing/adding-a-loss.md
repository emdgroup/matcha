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

---

## Wrapper losses

Some losses don't stand alone — they take an existing per-element loss and transform it (clip predictions to a valid range, randomly mask a fraction of entries, etc.). Use this pattern when the new loss is fundamentally *composed* of another registered loss rather than a distinct formula. Examples in the codebase: `BoundedLoss` and its `bounded-*` aliases, `DropoutLoss` and its `dropout-*` aliases.

### Two-tier structure

Wrappers ship as a generic base plus one concrete subclass per supported inner loss:

- A **base class** (e.g. `BoundedLoss`, `DropoutLoss`) accepts a `loss_fn="<alias>"` argument and resolves the inner loss through `LossRegistry`.
- **Concrete subclasses** (`BoundedMSELoss`, `DropoutMSELoss`, ...) hard-wire the inner alias and register their own kebab-case alias (`bounded-mse`, `dropout-mse`, ...). Each is three lines.

The concrete aliases are what YAML configs and pydantic schemas normally reference — `loss_fn: dropout-mse` is more discoverable than `loss_fn: dropout` + `loss_args: {loss_fn: mse}`.

### The `reduction="none"` requirement

The wrapper needs per-element access to mask entries before reduction, so it instantiates the inner loss with `reduction="none"` internally. Any loss you want to wrap must therefore accept `reduction="none"` and return a tensor shaped like `targets`. This is the same requirement `MultitaskLoss` and `MultiLoss` impose.

### The wrapper-nesting rule

Wrappers must reject other wrappers as their inner loss. Nested wrappers combine masks in ways that are semantically ambiguous (whose mask wins? do fractions compose multiplicatively?) and would explode the alias surface combinatorially (`dropout-bounded-mse`?). The check is class-based against the wrapper family:

```python
# src/matcha/nn/losses.py — inside DropoutLoss.__init__
inner_cls = LossRegistry[loss_fn]
if issubclass(
    inner_cls,
    (MultitaskLoss, MultiLoss, BoundedLoss, GradNormLoss, DropoutLoss),
):
    raise ValueError(
        f"DropoutLoss cannot wrap another wrapper loss (got {inner_cls.__name__})."
    )
```

Use `issubclass` on the resolved class (not `isinstance` on the constructed instance), so the check fires before the inner loss's `__init__` is called with the wrapper's `reduction="none"` kwarg — most wrappers don't accept that kwarg and would raise `TypeError` first, hiding the real reason for the rejection.

### `DropoutLoss` — seed semantics

`DropoutLoss` accepts an optional `seed: int | None`. When set, the wrapper builds a private `torch.Generator` (created lazily on the loss tensor's device on the first training forward) and draws its masks from that generator. The consequences:

- **Reproducibility.** Two `DropoutLoss(..., seed=42)` instances produce identical mask trajectories over identical inputs.
- **RNG isolation.** The private generator does not perturb the ambient torch RNG — weight initialization, data loader shuffling, and any other `torch.rand*` call outside the wrapper are unaffected.
- **`seed=None` (default).** Falls back to the ambient torch RNG, matching `torch.nn.Dropout` semantics.
- **Not persisted in `state_dict`.** The generator state resets on checkpoint reload, so the mask trajectory restarts. This matches `nn.Dropout` and is intentional — mask trajectories are training-time nuisance state, not model parameters.
- **DDP.** All workers seeded identically will draw identical masks. If you want per-worker mask diversity, offset by rank at construction time (e.g. `seed=42 + rank`).

### Reference

The random-dropout-MSE technique in descriptor pretraining is described in Jackson Burns's "how to train your chemeleon" repo: <https://github.com/JacksonBurns/how-to-train-your-chemeleon/blob/main/pretraining/random_dropout_mse.py>.

### Code snippet — concrete alias registration

```python
# src/matcha/nn/losses.py
@LossRegistry.register(alias="dropout-mse")
class DropoutMSELoss(DropoutLoss):
    """:class:`DropoutLoss` with MSE as the inner loss."""

    def __init__(self, **kwargs):
        super().__init__(loss_fn="mse", **kwargs)
```

Config usage:

```yaml
loss_fn: dropout-mse
loss_args:
  dropout: 0.2
  seed: 42
```
