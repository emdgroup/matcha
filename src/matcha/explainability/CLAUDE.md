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
- The analogue generator runs five systematic first-pass sources — query PAS, query reverse PAS, query nitrogen walk, Murcko-scaffold PAS, Murcko-scaffold nitrogen walk — then a bounded deterministic multi-step sample from two independent parent pools (query pool: query PAS + query reverse PAS + query nitrogen walk; scaffold pool: scaffold PAS + scaffold nitrogen walk). Each branch selects one enabled strategy per attempt uniformly among PAS, reverse PAS, and nitrogen walk, accepting up to `num_sample` valid, branch-local canonical-unique candidates and stopping after `2 * num_sample` attempts. Reverse PAS is enabled by default for standalone and sklearn explanations, operates systematically only on the original query (it never runs on the scaffold as a first-pass source but may be selected as a sampled second-step strategy for either branch), and reverse participation requires a nonempty PAS vocabulary. Pass `reverse_positional_analogue_scanning=False` to opt out; the two branch RNGs are seeded independently from `random_seed` so identical inputs, configuration, and seed produce identical sampled candidates in identical order. The aggregate `generation_timeout` covers first-pass, both samplers, and finalization; expiry raises `TimeoutError` without partial results.
- Explanations are computed on demand (no serialized state); the manager wires the estimator's prediction methods into the explainer and rejects neighborhoods with fewer than three total molecules before prediction or LIME work.
- `AnalogueGenerator.decompose` provides BRICS-fragment enumeration as a separate helper.
