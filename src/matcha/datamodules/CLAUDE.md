# `datamodules/` — Lightning DataModules

Turns raw molecular data (RDKit `Mol` objects, SMILES, labels) into batched tensors for the models in `../torch/`. Every family in `../sklearn/` picks its datamodule from here.

## Layout

```text
datamodules/
├── base_datamodule.py         # BaseDataModule + DataModuleRegistry
├── utils.py                   # CombinedStackDataset, collate helpers
├── classic/                   # supervised datamodules + featurization
│   ├── tabular_datamodule.py    # descriptor / fingerprint vectors
│   ├── graph_datamodule.py      # 2-D and 3-D PyG graphs (atom/bond features)
│   ├── clm_datamodule.py        # tokenized SMILES for language models
│   ├── chemprop_datamodule.py   # Chemprop-native MoleculeDataset
│   ├── combined_datamodule.py   # stacks datamodules for hybrid inputs
│   ├── rdkit_engine.py          # descriptor / fingerprint calculators (RDKit + skfp)
│   ├── graph_positional_encoder.py  # laplacian, RWSE, RRWP, ...
│   ├── label_encoder.py         # LabelEncoderRegistry — classification/regression
│   └── label_transform.py       # optional user-supplied label mapping
└── pretraining/               # self-supervised datamodules
    ├── graph_pretraining_datamodule.py
    ├── clm_mlm_datamodule.py
    └── on_the_fly_*            # in-memory variants for streaming pretraining
```

## How it fits together

Each `BaseScikitLearn*` family in `../sklearn/` instantiates one of these via `DataModuleManager`:

| Model family | Datamodule |
| --- | --- |
| `BaseScikitLearnTabular` | `TabularDataModule` (uses `rdkit_engine.Engine`) |
| `BaseScikitLearnGNN` | `GraphDataModule` |
| `BaseScikitLearnGNN3D` | `Graph3DDataModule` |
| `BaseScikitLearnCLM` | `CLMDataModule` |
| Chemprop wrapper | `ChempropDataModule` |
| Anything with extra descriptor features | `CombinedDataModule` (stacks two datamodules) |

`ATOM_FEAT_DIM` / `BOND_FEAT_DIM` constants exposed by `graph_datamodule` are the input dimensions graph encoders default to.

## Registries defined here

`DataModuleRegistry` (in `base_datamodule.py`) and `LabelEncoderRegistry` (in `classic/label_encoder.py`, keys: `"regression"`, `"binary_classification"`, ...). See [`PATTERNS.md`](../../PATTERNS.md) §2.

## Featurization notes

- `rdkit_engine.Engine` is the single entry point for descriptor / fingerprint calculation (Morgan, MACCS, MAP, MHFP, Mordred, ...). New descriptor types go here, not in individual datamodules.
- `graph_positional_encoder.GraphPE` handles the `enc_laplacian_k`, `enc_rwse_k`, `enc_rrwp_k`, `enc_distmat_k`, `enc_elstatic_k` fields you see on graph models. Set the corresponding dim to 0 to disable.
- Featurization inside datamodules is parallelized via `../utils/wrapper.py` (`Wrapper` + `parallelize`), which handles RDKit's non-pickleable state.

## Lazy imports

`datamodules/__init__.py` lazy-imports most classic and pretraining datamodules to break a circular chain (`base_datamodule → label_encoder → classic/__init__ → tabular_datamodule → base_datamodule`). If you add a new datamodule, wire it into `_LAZY_IMPORTS` rather than importing it eagerly at the top.

## Conventions

- Repo-wide rules from [`PATTERNS.md`](../../PATTERNS.md) apply: pydantic input models in `../utils/schemas/datamodules.py` (§4), registered aliases are load-bearing (§3).
- **Datamodule-specific:** featurization is CPU-bound — keep it parallel-safe (no globals, no mutable class state); `Wrapper` in `../utils/wrapper.py` handles the RDKit joblib gotcha.
