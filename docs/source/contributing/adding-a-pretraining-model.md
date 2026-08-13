# Adding a new pretraining model

```{note}
This page is written for LLM agents contributing to MATCHA. It trades prose for a step-by-step recipe with concrete file paths and code snippets.
```

Pretraining in MATCHA is a self-supervised, CLI-only path. Unlike classic supervised models, pretraining models are **not** wrapped by an sklearn estimator — they are trained by the `pretrain_encoder` CLI command via the `PretrainingModelRegistry` and reused later by feeding their encoder weights into a matching classic `*Model` for finetuning.

Two reference implementations to copy from:

- **2D case** — `GINPretraining` (`src/matcha/torch/models/pretraining/gin_pretraining.py`) + `GraphPretrainingDataModule` (`src/matcha/datamodules/pretraining/graph_pretraining_datamodule.py`).
- **3D case** — `E3GNNPretraining` (`src/matcha/torch/models/pretraining/e3gnn_pretraining.py`) + `Graph3DPretrainingDataModule` (`src/matcha/datamodules/pretraining/graph_3d_pretraining_datamodule.py`). Coordinates ride on `graph.pos` — no separate collate key.

A new pretraining model touches four layers, in this order:

```text
torch/encoders/   →  datamodules/pretraining/   →  utils/schemas/   →  torch/models/pretraining/   →  tests/
(canonical enc)      (batch producer)              (pydantic)          (Lightning module)             (parity + fit)
```

The encoder itself is shared with the classic path — do **not** create a pretraining-specific encoder twin. Issue #24 deleted every one of those; the invariant "one canonical encoder per architecture" is load-bearing.

---

## 1. Encoder — reuse the canonical class

Pretraining models consume the same encoder as the classic model of the same family. Add a new encoder only when a genuinely new architecture is being introduced (in which case follow `adding-a-model.md` first, then come back here).

The pretraining base class calls `encoder.forward_nodes_per_layer(graph)` to obtain one node-feature tensor per layer, then feeds the list through the shared jumping-knowledge merge and the two prediction heads. Any encoder exposing that method Just Works — see `BaseGraphPretrainingModel._get_per_layer_embeddings` in `src/matcha/torch/models/pretraining/base_graph_pretraining.py`.

For 3D encoders the input contract is the PyG convention: coordinates are read from `graph.pos`. Missing `pos` must raise a `ValueError` with a clear message pointing at the datamodule — that error is the tripwire for misconfigured pipelines.

## 2. Datamodule — `src/matcha/datamodules/pretraining/<name>_datamodule.py`

Pretraining datamodules produce batches with a `graph` field, per-atom targets `y_node`, and per-molecule targets `y_graph`. Add positional/structural fields (like 3D coords on `graph.pos`) on top of the parent class rather than parallel to it.

Inherit from `GraphPretrainingDataModule` (never from a classic `Graph3DDataModule` — that would carry the ETKDG conformer path, which is out of scope for pretraining) and register on `DataModuleRegistry`:

```python
# see src/matcha/datamodules/pretraining/graph_3d_pretraining_datamodule.py
from matcha.datamodules.base_datamodule import DataModuleRegistry
from matcha.datamodules.pretraining.graph_pretraining_datamodule import (
    GraphPretrainingDataModule,
)

@DataModuleRegistry.register("graph3d_pretraining")
class Graph3DPretrainingDataModule(GraphPretrainingDataModule):
    def featurize(self, mol_list, y_graph, y_node, coords, is_training=True, n_jobs=None):
        # validate + reorder coords, attach to Data.pos, parent collate handles the rest
        ...
    def export_to_classic(self) -> Graph3DDataModule:
        # mirror pretraining PE settings into the classic 3D datamodule
        ...
```

Key rules:

- **Wire coords through `Data.pos`, not a parallel collate key.** `Batch.from_data_list` auto-concatenates `pos`, so the inherited collate function needs no changes. Any 3D encoder reads `graph.pos` inside its per-layer hook.
- **Validate against the canonical-SMILES atom count.** The parent's `_validate_node_labels` uses this convention for `y_node`; new node-aligned fields (coords, per-atom features) must mirror it so shape errors fail fast with a clear message.
- **Reorder user-supplied per-atom data to canonical order.** `GraphDataModule._calculate_graph` reparses each molecule from its canonical SMILES; user rows come back misaligned otherwise. Use `mol.GetSubstructMatch(canonical_mol)` to remap.
- **Zero-pad virtual nodes for coordinate-like fields**, never NaN — NaN would poison distance-based features (E3GNN's Fourier distance) on real neighbours of a virtual node. `y_node`'s NaN padding is safe only because `MultitaskLoss` masks it.
- **Override `export_to_classic()`.** It returns a classic datamodule that mirrors the pretraining PE settings so downstream finetuning inherits the same featurization.
- **Set a stable `state_dict` `"ID"` string** so serialization survives round-trips.

## 3. Schema — `src/matcha/utils/schemas/`

Three touches:

1. **Datamodule schema** in `datamodules.py` — subclass the parent pretraining schema, override `datamodule_type`, and add to the union at the bottom:

   ```python
   # see src/matcha/utils/schemas/datamodules.py
   class Graph3DPretrainingDataModuleInputModel(GraphPretrainingDataModuleInputModel):
       datamodule_type: Literal["graph3d_pretraining"] = "graph3d_pretraining"

   DataModuleModel = (
       ...
       | Graph3DPretrainingDataModuleInputModel
   )
   ```

2. **Model schema** in `torch_api.py` — compose `PretrainingMatchaModel` + `GraphMixin` + `GraphPretrainingMixin` + the architecture's `*Mixin`. `PretrainingMatchaModel` and `GraphPretrainingMixin` live in `generic_models.py` and cover loss/optimizer/scheduler and the joint head fields (`num_node_targets`, `num_graph_targets`, `node_head_dims`, `graph_head_dims`, `node_loss_weight`, ...) that every graph pretraining model shares.

   ```python
   # see src/matcha/utils/schemas/torch_api.py
   class E3GNNPretrainingInputModel(
       PretrainingMatchaModel, GraphMixin, GraphPretrainingMixin, E3GNNMixin
   ):
       torch_type: Literal["e3gnn_pretraining"] = "e3gnn_pretraining"
       pred_hidden_dims: list[int] | None = None
       pred_task_head_dims: list[int] | None = None

   TorchModel = (
       ...
       | E3GNNPretrainingInputModel
   )
   ```

   `pred_hidden_dims` / `pred_task_head_dims` come from `GraphMixin` but are unused in pretraining — default them to `None` so callers don't need to pass them.

3. **Export both new schemas** from `utils/schemas/__init__.py` (`import` + `__all__`).

## 4. Lightning module — `src/matcha/torch/models/pretraining/<name>_pretraining.py`

Subclass `BaseGraphPretrainingModel`, register on `PretrainingModelRegistry`, and implement `_build_encoder()`. The base class handles the per-layer hook, JK merge, both heads, both losses, `training_step`, `validation_step`, and per-task logging.

```python
# see src/matcha/torch/models/pretraining/e3gnn_pretraining.py
from matcha.torch.encoders.e3gnn import E3GNN
from matcha.torch.models.pretraining.base_graph_pretraining import (
    BaseGraphPretrainingModel,
)
from matcha.torch.models.pretraining.base_pretraining_model import (
    PretrainingModelRegistry,
)

@PretrainingModelRegistry.register()
class E3GNNPretraining(BaseGraphPretrainingModel):
    def __init__(self, num_node_targets=1, num_graph_targets=1,
                 enc_num_layers=3, enc_atom_hidden_dim=128, ...,
                 node_head_dims=None, graph_head_dims=None, ...):
        super().__init__(num_node_targets=..., num_graph_targets=..., ...)
        self.save_hyperparameters()
        self._build_encoder()
        head_input_dim = enc_atom_hidden_dim * enc_num_layers if enc_jk == "concat" else enc_atom_hidden_dim
        self.node_head  = self._build_prediction_head(head_input_dim, node_head_dims, num_node_targets, ...)
        self.graph_head = self._build_prediction_head(head_input_dim, graph_head_dims, num_graph_targets, ...)
        self._parse_train_config()

    def _build_encoder(self):
        self.encoder = E3GNN(
            num_layers=self.hparams["enc_num_layers"],
            atom_input_dim=self.hparams["enc_atom_input_dim"] + self.hparams["enc_laplacian_k"] + ...,
            ...
        )
```

Rules:

- **Never override `forward` / `training_step` / `_get_per_layer_embeddings`.** For 3D encoders, coordinates flow through `graph.pos` — the base class reads them via `self.encoder.forward_nodes_per_layer(batch["graph"])`. Adding an override reintroduces the drift issues #24 and #26 were designed to eliminate.
- **`enc_*`-prefix every encoder-facing hyperparameter.** The classic `*Model` uses the same prefixes, which is what makes encoder-weight transfer at finetuning time a strict `load_state_dict`.
- **Add the class to `torch/models/pretraining/__init__.py`** (import + `__all__`).

## 5. Tests — parity, fit, and encoder transfer

The pretraining test suite lives under `tests/pretraining/`. Add a new file `test_<name>_pretraining.py` covering:

- **Registry lookup:** `"<name>pretraining" in PretrainingModelRegistry`.
- **Canonical encoder wiring:** `isinstance(model.encoder, <CanonicalEncoder>)`.
- **Forward shapes:** `model(batch)["node"].shape == (N_atoms, num_node_targets)` and `["graph"].shape == (batch_size, num_graph_targets)`.
- **Per-layer hook length:** `len(model._get_per_layer_embeddings(batch)[0]) == num_layers`.
- **One-step fit:** `model.training_step(batch, 0)` returns a finite scalar that `requires_grad`.
- **Optimizer step actually moves weights:** compare `parameters_to_vector(model.encoder.parameters())` before and after.
- **`save_hyperparameters` round-trip** through `state_dict`/`load_state_dict`.
- **Encoder weight transfer to the classic `*Model`** — the headline acceptance criterion. Strict `load_state_dict` from the pretraining encoder into the classic encoder, then `parameters_to_vector` equality:

  ```python
  missing, unexpected = classic.encoder.load_state_dict(
      pretrain.encoder.state_dict(), strict=True
  )
  assert missing == []
  assert unexpected == []
  ```

Then extend the parametrized parity suites so the new model is exercised alongside the existing ones:

- `tests/pretraining/test_encoder_parity.py` — add a `pytest.param(<ClassicModel>, <PretrainingModel>, dict(enc_*=...), id="<name>")` entry. The three checks (parameter-key equality, module-tree string equality, `allclose` on encoder output after a weight-sync `load_state_dict`) then run automatically.
- `tests/pretraining/test_graph_pretraining_encoders.py` — add `pytest.param(<PretrainingModel>, <CanonicalEncoder>, dict(enc_*=...), id="<name>")` for the `is_canonical` / per-layer-length / forward-shape checks.

For 3D architectures also add a **classic ↔ pretraining datamodule parity test**: build a small mol list, featurize once through `Graph3DDataModule` (ETKDG coords), reuse the resulting `graph.pos` as user-supplied coords in the pretraining datamodule, and assert that `E3GNN.forward_nodes_per_layer` produces bit-identical per-layer outputs on both batches. See `tests/pretraining/test_e3gnn_pretraining.py::test_classic_and_pretraining_datamodules_produce_identical_e3gnn_features` for the shape.

## 6. Wire the pretraining path

- `torch/models/pretraining/__init__.py` — import + `__all__`.
- `datamodules/pretraining/__init__.py` — import + `__all__`.
- `datamodules/__init__.py` — add a lazy-import entry (`_LAZY_IMPORTS`); a plain top-level import will reintroduce the circular chain.
- `utils/schemas/__init__.py` — export both schemas.

No sklearn / CLI wiring is required. `pretrain_encoder` dispatches through the two registries.

---

## Checklist

- [ ] Reuse the canonical encoder from `torch/encoders/` (no pretraining twin).
- [ ] Datamodule inherits from `GraphPretrainingDataModule`, registered with `DataModuleRegistry`, validates + reorders per-atom user data to canonical order, ships an `export_to_classic()` override.
- [ ] 3D specifics: coordinates on `graph.pos`, zero-padded for virtual nodes; encoder raises `ValueError` on missing `pos`.
- [ ] Datamodule schema in `utils/schemas/datamodules.py` + model schema in `torch_api.py` (composed from `PretrainingMatchaModel`, `GraphMixin`, `GraphPretrainingMixin`, the architecture mixin), both re-exported.
- [ ] Lightning module in `torch/models/pretraining/`, registered with `PretrainingModelRegistry`, `enc_*` / `pred_*` prefixes, `_build_encoder` only — no `forward` / `_get_per_layer_embeddings` override.
- [ ] Tests: per-model `test_<name>_pretraining.py` (registry, forward shapes, fit, encoder transfer), plus entries in `test_encoder_parity.py` and `test_graph_pretraining_encoders.py`.
- [ ] 3D only: classic ↔ pretraining datamodule parity test.
- [ ] `uv run pytest -k 'not gpu'` passes locally.

If any of these steps feels awkward — e.g. the new pretraining path requires an override on `BaseGraphPretrainingModel` — flag it in the PR rather than adding it silently. The one-canonical-encoder + coords-on-`pos` invariants exist because parallel pretraining twins drifted the last three times.
