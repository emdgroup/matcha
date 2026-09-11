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
- The analogue generator runs forward positional analogue scanning and nitrogen walk on both the input molecule and its Murcko scaffold, then a second pass over the forward PAS results. Reverse PAS is enabled by default for standalone and sklearn explanations, operates only on the original query, and never recurses or runs on the scaffold. Pass `reverse_positional_analogue_scanning=False` to opt out; reverse silently contributes no candidates when the forward PAS vocabulary is absent.
- Candidate deduplication preserves first-discovery order and is invariant to Python hash randomization.
- One aggregate generation deadline covers every strategy. Expiry raises `TimeoutError` without returning partial results.
- Explanations are computed on demand (no serialized state); the manager wires the estimator's prediction methods into the explainer and rejects neighborhoods with fewer than three total molecules before prediction or LIME work.
- `AnalogueGenerator.decompose` provides BRICS-fragment enumeration as a separate helper.
