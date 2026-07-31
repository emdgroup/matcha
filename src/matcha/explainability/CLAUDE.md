# `explainability/` — post-hoc model interpretation

Model-agnostic explainers invoked from `../sklearn/managers/explainability_manager.py`. Users interact via `estimator.explain(...)`; this folder holds the underlying implementations.

## Layout

```text
explainability/
├── explainer.py            # Explainer — top-level dispatcher + atom-highlight rendering (RDKit + PIL + plotly)
├── lime.py                 # LIME — local linear surrogate over molecular fragments
└── analogue_generator.py   # Structural analogue enumeration (positional analogue scanning + nitrogen walk, over molecule and Murcko scaffold)
```

## Contract

`MatchaExplainer` is the public surface. It composes `LIME` and `AnalogueGenerator` and produces per-atom attribution scores + a rendered molecule PNG, plus a ranked list of structural analogues (from positional analogue scanning and nitrogen walk) with predicted properties. Inputs validated against `ExplainerInputModel` in `../utils/schemas/explainability.py` (see [`PATTERNS.md`](../../PATTERNS.md) §4).

## Notes

- LIME builds a surrogate `sklearn.linear_model.Ridge` over fragment presence/absence, using `../datamodules/classic/rdkit_engine.Engine` for descriptors.
- The analogue generator runs positional analogue scanning and nitrogen walk on both the input molecule and its Murcko scaffold, then a second-pass PAS/nitrogen walk over the first-pass PAS results. `AnalogueGenerator.decompose` provides BRICS-fragment enumeration as a separate helper.
- Explanations are computed on demand (no serialized state); the manager just wires the estimator's `predict` into the explainer.
