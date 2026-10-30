# `utils/` — shared helpers + pydantic boundary contracts

Small building blocks used everywhere else. The important content is `registry.py` (backs every string-alias lookup) and `schemas/` (validates every subpackage boundary).

## Layout

```text
utils/
├── registry.py         # ClassRegistry — decorator + dict-of-classes
├── schemas/            # pydantic contracts, one file per boundary (see below)
├── serialization.py    # load/save json, yaml, pickle, dataframe helpers
├── logging.py          # MatchaLogger + get_default_logger
├── warnings.py         # silence_nuisance_warnings — call early in the process
├── metrics.py          # regression + classification metric functions
├── plotting.py         # standard plots (parity, ROC, calibration, ...)
├── splitting.py        # train/test/calibration split strategies (random, scaffold, ...)
├── sanitize.py         # RDKit molecule sanitization helpers
├── wrapper.py          # Wrapper + parallelize — joblib-safe RDKit wrapping
└── __init__.py         # re-exports the shortlist (see below)
```

## `schemas/` — the boundary layer

Every subpackage boundary is validated by a pydantic model. When a schema fails, the origin is almost always a mismatched user config or a stale test fixture, not a runtime bug.

| File | Guards |
| --- | --- |
| `torch_api.py` | `{Model}InputModel` — one per architecture (GIN, GT, MLP, ...) |
| `generic_models.py` | Mixins composed into `torch_api` input models (`GraphMixin`, `TabularMixin`, `GINMixin`, ...) |
| `datamodules.py` | `{Family}DataModuleInputModel` |
| `sklearn_api.py` | `ScikitLearnInputModel`, `ScikitLearnEnsembleInputModel`, `TrainingInputModel`, `MetadataInputModel` |
| `cli.py` | `CLI{Command}InputModel` — one per CLI command |
| `data.py` | `MolDataset`, `MolReadout` — molecule + label containers |
| `calibration.py` | ICP + error-model configs |
| `explainability.py` | `ExplainerInputModel` |
| `label.py` | Label-encoder / label-transform configs |
| `base.py` | Shared base (`BaseDataModel`) with strict-mode config |

**Rule of thumb:** if data crosses a subpackage boundary and isn't a numpy array or tensor, it should be a schema instance. Extend an existing mixin rather than adding an ad-hoc dict.

## `registry.py` — the composition backbone

`ClassRegistry` is a `dict[str, Type[T]]` with a `@register("alias")` decorator. The default alias is `cls.__name__.lower()`. Full list of registries in the codebase: [`PATTERNS.md`](../../PATTERNS.md) §2.

## Conventions

- **`utils/__init__.py` re-exports a shortlist.** `load_json`, `save_yaml`, `Wrapper`, `silence_nuisance_warnings`, `MatchaLogger` are stable public helpers. Everything else, import from its file directly.
- **Warnings suppression happens in `warnings.py`.** Don't `warnings.filterwarnings(...)` at module scope elsewhere — add to `silence_nuisance_warnings` so it stays discoverable. Pytest runs with warnings-as-errors, so unsilenced third-party noise fails the suite.
- **Serialization uses only three formats:** JSON for structured metadata, YAML for configs, pickle for arbitrary objects (label encoders, calibrators). Prefer JSON/YAML where readable.
