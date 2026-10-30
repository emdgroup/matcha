# Adding a new fingerprint or descriptor set

```{note}
This page is written for LLM agents contributing to MATCHA. It trades prose for a step-by-step recipe with concrete file paths and code snippets.
```

All descriptor / fingerprint calculators live in `src/matcha/datamodules/classic/rdkit_engine.py` inside the `Engine` class. Adding a new one is a **single-file change** plus tests — do not scatter feature code across datamodules.

A new feature set touches four locations in `Engine`, plus the test file:

1. A `get_<name>` method on `Engine`.
2. An entry in `Engine._mapping` (alias → method).
3. An entry in `Engine._defaults` (alias → parameter dict), if the calculator has tunable knobs.
4. An entry in `Engine._dims` (alias → output dimensionality).
5. Coverage in `tests/datamodules/test_rdkit_engine.py`.

---

## 1. Add the calculator method

Follow the shape of `get_ECFP` — parallelize over the mol list with `parallelize`, read tunables from `self.defaults[<alias>]`, return a `np.ndarray` of shape `(N, dim)` with `dtype=np.float32`.

```python
# src/matcha/datamodules/classic/rdkit_engine.py
from rdkit.Chem import AllChem

def get_dummy_embedding(
    self, mols: list[Mol], n_jobs: int | None = None
) -> np.ndarray:
    """Dummy example: mean of Morgan bit vector per molecule, then tiled to `dim`.

    Purpose here is to show the shape, not to be a useful descriptor.
    """

    def batch_wrapper(batch):
        params = self.defaults["dummy_embedding"]
        out = []
        for mol in batch:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=params["radius"], nBits=params["n_source_bits"]
            )
            arr = np.zeros(params["n_source_bits"], dtype=np.float32)
            from rdkit import DataStructs
            DataStructs.ConvertToNumpyArray(fp, arr)
            out.append(np.tile(arr.mean(keepdims=True), params["dim"]))
        return out

    feats = parallelize(
        batch_wrapper, mols, n_jobs=n_jobs if n_jobs is not None else self.n_jobs
    )
    return np.array(feats, dtype=np.float32)
```

- Use `parallelize(batch_wrapper, mols, n_jobs=...)` — RDKit objects are non-pickleable at module scope, so the batch closure pattern is load-bearing.
- If the underlying RDKit function is not picklable at all (e.g. `GetErGFingerprint`), wrap it with `Wrapper(name, module)` at module top-level — see `_wrapped_GetErGFingerprint`.
- Return `float32` for consistency with the rest of `Engine`.

## 2. Register the alias in `_mapping`, `_defaults`, `_dims`

All three dicts are populated in `Engine.__init__`. Add matching entries:

```python
# src/matcha/datamodules/classic/rdkit_engine.py, inside Engine.__init__
self._mapping = {
    ...,
    "dummy_embedding": self.get_dummy_embedding,
}

self._defaults = {
    ...,
    "dummy_embedding": {"radius": 2, "n_source_bits": 1024, "dim": 64},
}

self._dims = {
    ...,
    "dummy_embedding": 64,  # must match self._defaults["dummy_embedding"]["dim"]
}
```

- The **alias is load-bearing** — it's what `TabularFeaturizer` / `get_features` / `calculate_feature_dim` look up. Lowercase, snake_case, no renaming later without a migration.
- If `dim` depends on a default (e.g. `nBits`), reference it: `self._dims["dummy_embedding"] = self._defaults["dummy_embedding"]["dim"]`.
- If the feature has no tunables, still register the empty dict in `_defaults` only if `get_features` needs a lookup — otherwise skip it (see `pubchem_fp` and `estate` for the mixed pattern).

## 3. Tests — `tests/datamodules/test_rdkit_engine.py`

The suite has one `TestGet<Name>` class per feature. Mirror the smallest existing one:

```python
# tests/datamodules/test_rdkit_engine.py
class TestGetDummyEmbedding:
    def test_shape(self, engine, mols):
        out = engine.get_dummy_embedding(mols)
        assert out.shape == (len(mols), engine.defaults["dummy_embedding"]["dim"])
        assert out.dtype == np.float32

    def test_dim_matches_registry(self, engine):
        assert engine._dims["dummy_embedding"] == engine.defaults["dummy_embedding"]["dim"]
```

- Also add `"dummy_embedding"` to any parametrized `get_features` / `calculate_feature_dim` test that iterates over `Engine._mapping.keys()`.

---

## Checklist

- [ ] `get_<name>` method on `Engine`, uses `parallelize`, returns `np.ndarray` of `float32`.
- [ ] Alias added to `Engine._mapping`, `Engine._defaults` (if tunable), `Engine._dims`.
- [ ] `_dims[alias]` matches actual output width.
- [ ] Test class in `tests/datamodules/test_rdkit_engine.py` covering shape + dtype.
- [ ] `uv run pytest tests/datamodules/test_rdkit_engine.py` passes.
- [ ] Codecov PR check stays green — new lines are covered and total coverage doesn't drop (see `CONTRIBUTING.md` → Testing).
