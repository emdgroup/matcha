# `nn/` — reusable neural-network primitives

Registered building blocks consumed by `../torch/` models. Every file exposes a `*Registry` and populates it with `@Registry.register("alias")`-decorated classes; downstream code looks them up by string.

## Layout

```text
nn/
├── activations.py    # ActivationRegistry — relu, swish, gelu, geglu, mish, ...
├── layers.py         # LayerRegistry — LnBnDr, norm/dropout blocks
├── losses.py         # LossRegistry — mse, bce, focal, MultiLoss, MultitaskLoss, GradNormLoss
├── optimizers.py     # OptimizerRegistry — adam, adamw, sgd, ...
├── schedulers.py     # SchedulerRegistry — cosine_annealing, step, ...
├── readouts.py       # ReadoutRegistry — sum, mean, attentive, set2set, deepsets, ...
├── multitask.py      # Multi-endpoint loss aggregation utilities (GradNorm, uncertainty weighting)
└── deep_lasso.py     # Sparse regularization for tabular models
```

## Adding a primitive

Register with the appropriate registry — e.g. `@LossRegistry.register("focal") class FocalLoss(nn.Module): ...`. Lightning modules in `../torch/` then resolve `loss_fn="focal"` through the registry, no code change needed elsewhere.

Import registries from the file that defines them (`from matcha.nn.losses import LossRegistry`), not from `nn/__init__.py`. See [`PATTERNS.md`](../../PATTERNS.md) §2–3 for the alias-stability rule.
