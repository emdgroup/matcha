"""Tests for matcha.nn.multitask – TaskAffinityResult and stitch_datasets."""

import numpy as np
import pandas as pd
import pytest

from matcha.nn.multitask import (
    TaskAffinityResult,
    compute_task_affinity,
    stitch_datasets,
    _move_batch_to_device,
    _copy_batch,
)


# ===================================================================
# TaskAffinityResult
# ===================================================================


class TestTaskAffinityResultInit:
    @pytest.fixture()
    def affinity_result(self):
        matrix = np.array(
            [
                [1.0, 0.5, 0.1],
                [0.3, 1.0, 0.7],
                [0.2, 0.4, 1.0],
            ]
        )
        return TaskAffinityResult(
            affinity_matrix=matrix,
            task_names=["A", "B", "C"],
            num_epochs=10,
        )

    def test_attributes(self, affinity_result):
        assert affinity_result.affinity_matrix.shape == (3, 3)
        assert affinity_result.task_names == ["A", "B", "C"]
        assert affinity_result.num_epochs == 10


class TestTaskAffinityGetTopK:
    @pytest.fixture()
    def affinity_result(self):
        matrix = np.array(
            [
                [1.0, 0.8, 0.1, 0.5],
                [0.3, 1.0, 0.7, 0.2],
                [0.2, 0.4, 1.0, 0.9],
                [0.6, 0.1, 0.3, 1.0],
            ]
        )
        return TaskAffinityResult(
            affinity_matrix=matrix,
            task_names=["A", "B", "C", "D"],
            num_epochs=5,
        )

    def test_top_k_exclude_self(self, affinity_result):
        top = affinity_result.get_top_k(source_task=0, k=2, include_self=False)
        assert len(top) == 2
        assert 0 not in top  # self excluded

    def test_top_k_include_self(self, affinity_result):
        top = affinity_result.get_top_k(source_task=0, k=2, include_self=True)
        assert len(top) == 2

    def test_top_k_returns_highest_affinity(self, affinity_result):
        # Row 0: [1.0, 0.8, 0.1, 0.5] → excluding self → top is [1 (0.8), 3 (0.5)]
        top = affinity_result.get_top_k(source_task=0, k=2, include_self=False)
        assert top[0] == 1  # highest affinity to task B
        assert top[1] == 3  # second highest to task D

    def test_invalid_source_task_raises(self, affinity_result):
        with pytest.raises(ValueError):
            affinity_result.get_top_k(source_task=10, k=1)

    def test_invalid_source_task_negative_raises(self, affinity_result):
        with pytest.raises(ValueError):
            affinity_result.get_top_k(source_task=-1, k=1)

    def test_invalid_k_raises(self, affinity_result):
        with pytest.raises(ValueError):
            affinity_result.get_top_k(source_task=0, k=0)

    def test_k_too_large_raises(self, affinity_result):
        # 4 tasks, exclude self → max k = 3
        with pytest.raises(ValueError):
            affinity_result.get_top_k(source_task=0, k=4, include_self=False)


# ===================================================================
# stitch_datasets
# ===================================================================


class TestStitchDatasets:
    @pytest.fixture()
    def df_a(self):
        return pd.DataFrame(
            {
                "SMILES": ["CCO", "CC(C)C", "CC(=O)O"],
                "pIC50": [5.0, 6.0, 7.0],
            }
        )

    @pytest.fixture()
    def df_b(self):
        return pd.DataFrame(
            {
                "SMILES": ["CCO", "CC(C)C", "c1ccccc1"],
                "logP": [1.0, 2.0, 3.0],
            }
        )

    def test_stitched_has_correct_columns(self, df_a, df_b):
        result = stitch_datasets(
            df_list=[df_a, df_b],
            property_list=["pIC50", "logP"],
            smiles_key="SMILES",
            bound_key=None,
        )
        assert "SMILES" in result.columns

    def test_stitched_union_of_smiles(self, df_a, df_b):
        result = stitch_datasets(
            df_list=[df_a, df_b],
            property_list=["pIC50", "logP"],
            smiles_key="SMILES",
            bound_key=None,
        )
        # Should contain all unique SMILES from both frames
        smiles_set = set(result["SMILES"])
        assert "CCO" in smiles_set
        assert "CC(C)C" in smiles_set
        assert "CC(=O)O" in smiles_set
        assert "c1ccccc1" in smiles_set

    def test_nan_for_missing_values(self, df_a, df_b):
        result = stitch_datasets(
            df_list=[df_a, df_b],
            property_list=["pIC50", "logP"],
            smiles_key="SMILES",
            bound_key=None,
        )
        # CC(=O)O only in df_a → logP column should be NaN
        acetic_row = result[result["SMILES"] == "CC(=O)O"]
        # Find the logP column
        logp_cols = [c for c in result.columns if "logP" in c]
        if logp_cols:
            assert acetic_row[logp_cols[0]].isna().values[0]

    def test_empty_list_returns_empty(self):
        result = stitch_datasets(
            df_list=[],
            property_list=[],
            smiles_key="SMILES",
            bound_key=None,
        )
        assert result == ()

    def test_single_dataframe(self):
        df = pd.DataFrame(
            {
                "SMILES": ["CCO", "CC(C)C"],
                "activity": [1.0, 2.0],
            }
        )
        result = stitch_datasets(
            df_list=[df],
            property_list=["activity"],
            smiles_key="SMILES",
            bound_key=None,
        )
        assert len(result) == 2

    def test_with_bound_key(self):
        df_a = pd.DataFrame(
            {
                "SMILES": ["CCO", "CC(C)C"],
                "pIC50": [5.0, 6.0],
                "OPERATOR": ["=", "<"],
            }
        )
        df_b = pd.DataFrame(
            {
                "SMILES": ["CCO", "CC(=O)O"],
                "logP": [1.0, 2.0],
                "OPERATOR": [">", "="],
            }
        )
        result = stitch_datasets(
            df_list=[df_a, df_b],
            property_list=["pIC50", "logP"],
            smiles_key="SMILES",
            bound_key="OPERATOR",
        )
        # Should have operator columns
        operator_cols = [c for c in result.columns if "OPERATOR" in c]
        assert len(operator_cols) > 0


# ===================================================================
# _move_batch_to_device / _copy_batch
# ===================================================================


class TestMoveBatchToDevice:
    def test_tensors_moved(self):
        import torch

        batch = {"x": torch.randn(4, 8), "y": torch.randn(4, 1), "name": "test"}
        moved = _move_batch_to_device(batch, "cpu")
        assert moved["x"].device.type == "cpu"
        assert moved["y"].device.type == "cpu"
        assert moved["name"] == "test"

    def test_original_unchanged(self):
        import torch

        batch = {"x": torch.randn(4, 8)}
        original_data = batch["x"].clone()
        _ = _move_batch_to_device(batch, "cpu")
        assert torch.allclose(batch["x"], original_data)


class TestCopyBatch:
    def test_tensors_cloned(self):
        import torch

        batch = {"x": torch.randn(4, 8), "y": torch.randn(4)}
        copied = _copy_batch(batch)
        # Modify copy should not affect original
        copied["x"].fill_(0)
        assert batch["x"].abs().sum() > 0

    def test_non_tensor_deepcopied(self):
        batch = {"meta": {"key": "value"}, "tags": [1, 2, 3]}
        copied = _copy_batch(batch)
        copied["tags"].append(4)
        assert len(batch["tags"]) == 3


# ===================================================================
# compute_task_affinity (integration)
# ===================================================================


class TestComputeTaskAffinityIntegration:
    """Integration tests for compute_task_affinity using minimal GIN models."""

    NUM_TASKS = 3
    NUM_SAMPLES = 30  # keep small for speed

    @pytest.fixture()
    def smiles_and_mols(self, testing_df):
        """Return a small list of (SMILES, Mol) tuples from the testing CSV."""
        from rdkit import Chem

        smiles = testing_df["SMILES"].iloc[: self.NUM_SAMPLES].tolist()
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        # Filter out molecules that failed to parse
        valid = [(s, m) for s, m in zip(smiles, mols) if m is not None]
        smiles, mols = zip(*valid) if valid else ([], [])
        return list(smiles), list(mols)

    TASK_NAMES = ["TaskA", "TaskB", "TaskC"]

    # -- regressor ---------------------------------------------------------

    def test_task_affinity_regressor(self, smiles_and_mols):
        from matcha.sklearn.graph.gin import GINRegressor

        _smiles, mols = smiles_and_mols
        rng = np.random.RandomState(0)
        labels = rng.randn(len(mols), self.NUM_TASKS).astype(np.float32)

        model = GINRegressor(
            enc_num_layers=1,
            enc_atom_hidden_dim=16,
            pred_hidden_dims=[16],
            num_endpoints=self.NUM_TASKS,
            num_epochs=1,
            batch_size=16,
            accelerator="cpu",
            devices=1,
            early_stopping=False,
            stochastic_weight_averaging=False,
            scheduler_args={"min_lr": 1e-5, "total_steps": 1},
        )

        # Register task names so that the label encoder exposes them
        for idx, name in enumerate(self.TASK_NAMES):
            model.configure_label_encoder_task(task_idx=idx, task_label=name)

        model.fit(mols, labels)

        result = compute_task_affinity(
            sklearn_model=model,
            molecules=mols,
            labels=labels,
            affinity_every_n_steps=1,  # collect at every step for such a short run
            device="cpu",
        )

        assert isinstance(result, TaskAffinityResult)
        assert result.affinity_matrix.shape == (self.NUM_TASKS, self.NUM_TASKS)
        assert result.num_epochs == 1
        assert result.task_names == self.TASK_NAMES
        # Diagonal (self-affinity) should be ~1.0 after row normalisation
        np.testing.assert_allclose(
            np.diag(result.affinity_matrix),
            1.0,
            atol=1e-5,
        )
        # Matrix should contain only finite values
        assert np.all(np.isfinite(result.affinity_matrix))

    # -- classifier --------------------------------------------------------

    def test_task_affinity_classifier(self, smiles_and_mols):
        from matcha.sklearn.graph.gin import GINClassifier

        _smiles, mols = smiles_and_mols
        rng = np.random.RandomState(1)
        labels = rng.randint(0, 2, size=(len(mols), self.NUM_TASKS)).astype(np.float32)

        model = GINClassifier(
            enc_num_layers=1,
            enc_atom_hidden_dim=16,
            pred_hidden_dims=[16],
            num_endpoints=self.NUM_TASKS,
            num_epochs=1,
            batch_size=16,
            accelerator="cpu",
            devices=1,
            early_stopping=False,
            stochastic_weight_averaging=False,
            scheduler_args={"min_lr": 1e-5, "total_steps": 1},
        )

        # Register task names with class thresholds/labels required by the
        # BinaryClassificationLabelEncoder to avoid TypeError on threshold
        # comparison during label processing.
        for idx, name in enumerate(self.TASK_NAMES):
            model.configure_label_encoder_task(
                task_idx=idx,
                task_label=name,
                class_thresholds=[0.5],
                class_labels=["inactive", "active"],
            )

        model.fit(mols, labels)

        result = compute_task_affinity(
            sklearn_model=model,
            molecules=mols,
            labels=labels,
            affinity_every_n_steps=1,
            device="cpu",
        )

        assert isinstance(result, TaskAffinityResult)
        assert result.affinity_matrix.shape == (self.NUM_TASKS, self.NUM_TASKS)
        assert result.num_epochs == 1
        assert result.task_names == self.TASK_NAMES
        np.testing.assert_allclose(
            np.diag(result.affinity_matrix),
            1.0,
            atol=1e-5,
        )
        assert np.all(np.isfinite(result.affinity_matrix))


# ===================================================================
# compute_task_affinity scheduler stepping (Stage 3 tests)
# ===================================================================


class TestComputeTaskAffinityScheduler:
    """Tests verifying scheduler steps per batch and total_steps auto-computation."""

    NUM_TASKS = 2
    NUM_SAMPLES = 20

    @pytest.fixture()
    def smiles_and_mols(self, testing_df):
        from rdkit import Chem

        smiles = testing_df["SMILES"].iloc[: self.NUM_SAMPLES].tolist()
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        valid = [(s, m) for s, m in zip(smiles, mols) if m is not None]
        smiles, mols = zip(*valid) if valid else ([], [])
        return list(smiles), list(mols)

    def _make_model(self, mols, labels, batch_size, num_epochs, scheduler_args):
        from matcha.sklearn.graph.gin import GINRegressor

        model = GINRegressor(
            enc_num_layers=1,
            enc_atom_hidden_dim=16,
            pred_hidden_dims=[16],
            num_endpoints=self.NUM_TASKS,
            num_epochs=num_epochs,
            batch_size=batch_size,
            accelerator="cpu",
            devices=1,
            early_stopping=False,
            stochastic_weight_averaging=False,
            scheduler_args=scheduler_args,
        )
        model.fit(mols, labels)
        return model

    def test_total_steps_auto_computed_when_absent(self, smiles_and_mols):
        """When total_steps is not in scheduler_args, it should be auto-computed
        as num_epochs * ceil(len(train_data) / batch_size)."""
        from math import ceil
        from unittest.mock import patch

        _smiles, mols = smiles_and_mols
        rng = np.random.RandomState(42)
        labels = rng.randn(len(mols), self.NUM_TASKS).astype(np.float32)

        batch_size = 8
        num_epochs = 3
        expected_total_steps = num_epochs * ceil(len(mols) / batch_size)

        model = self._make_model(
            mols, labels, batch_size, num_epochs, scheduler_args={"min_lr": 1e-5}
        )

        # Intercept the SchedulerRegistry lookup in the multitask module
        import matcha.nn.multitask as mt

        original_registry = mt.SchedulerRegistry
        captured_args = {}

        class CapturingRegistry(dict):
            """Wraps SchedulerRegistry to capture args passed to scheduler __init__."""

            def get(self, name, default=None):
                cls = original_registry.get(name, default)
                if cls is None:
                    return default

                class Wrapper(cls):
                    def __init__(self, optimizer, **kwargs):
                        captured_args.update(kwargs)
                        super().__init__(optimizer, **kwargs)

                return Wrapper

        with patch.object(
            mt, "SchedulerRegistry", CapturingRegistry(original_registry)
        ):
            compute_task_affinity(
                sklearn_model=model,
                molecules=mols,
                labels=labels,
                affinity_every_n_steps=1,
                device="cpu",
            )

        assert "total_steps" in captured_args
        assert captured_args["total_steps"] == expected_total_steps

    def test_total_steps_respected_when_explicitly_set(self, smiles_and_mols):
        """When total_steps is explicitly provided, it should be used unchanged."""
        from unittest.mock import patch

        _smiles, mols = smiles_and_mols
        rng = np.random.RandomState(42)
        labels = rng.randn(len(mols), self.NUM_TASKS).astype(np.float32)

        explicit_total_steps = 999

        model = self._make_model(
            mols,
            labels,
            batch_size=8,
            num_epochs=2,
            scheduler_args={"min_lr": 1e-5, "total_steps": explicit_total_steps},
        )

        import matcha.nn.multitask as mt

        original_registry = mt.SchedulerRegistry
        captured_args = {}

        class CapturingRegistry(dict):
            def get(self, name, default=None):
                cls = original_registry.get(name, default)
                if cls is None:
                    return default

                class Wrapper(cls):
                    def __init__(self, optimizer, **kwargs):
                        captured_args.update(kwargs)
                        super().__init__(optimizer, **kwargs)

                return Wrapper

        with patch.object(
            mt, "SchedulerRegistry", CapturingRegistry(original_registry)
        ):
            compute_task_affinity(
                sklearn_model=model,
                molecules=mols,
                labels=labels,
                affinity_every_n_steps=1,
                device="cpu",
            )

        assert captured_args["total_steps"] == explicit_total_steps

    def test_scheduler_steps_per_batch_not_per_epoch(self, smiles_and_mols):
        """The scheduler should be stepped once per optimizer step (per batch),
        not once per epoch. Verify by checking total number of scheduler.step() calls
        made during the training loop (excluding the initial step from __init__)."""
        from math import ceil
        from unittest.mock import patch

        _smiles, mols = smiles_and_mols
        rng = np.random.RandomState(42)
        labels = rng.randn(len(mols), self.NUM_TASKS).astype(np.float32)

        batch_size = 8
        num_epochs = 2
        num_batches_per_epoch = ceil(len(mols) / batch_size)
        expected_total_calls = num_epochs * num_batches_per_epoch

        model = self._make_model(
            mols, labels, batch_size, num_epochs, scheduler_args={"min_lr": 1e-5}
        )

        import matcha.nn.multitask as mt

        original_registry = mt.SchedulerRegistry
        step_counter = {"count": 0, "initialized": False}

        class CountingRegistry(dict):
            def get(self, name, default=None):
                cls = original_registry.get(name, default)
                if cls is None:
                    return default

                class Wrapper(cls):
                    def __init__(self, optimizer, **kwargs):
                        super().__init__(optimizer, **kwargs)
                        # Mark initialization complete — don't count the step from __init__
                        step_counter["initialized"] = True

                    def step(self):
                        if step_counter["initialized"]:
                            step_counter["count"] += 1
                        return super().step()

                return Wrapper

        with patch.object(mt, "SchedulerRegistry", CountingRegistry(original_registry)):
            compute_task_affinity(
                sklearn_model=model,
                molecules=mols,
                labels=labels,
                affinity_every_n_steps=1,
                device="cpu",
            )

        assert step_counter["count"] == expected_total_calls
