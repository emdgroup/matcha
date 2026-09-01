"""Tests for the `_call_predict_step` signature dispatcher."""

import lightning as L
import torch

from matcha.sklearn.base_sklearn_model import _call_predict_step


class _BatchOnly(L.LightningModule):
    def predict_step(self, batch):
        return 2 * batch["x"]


class _FullSignature(L.LightningModule):
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return batch["x"] + batch_idx + dataloader_idx


class _BatchAndBatchIdx(L.LightningModule):
    def predict_step(self, batch, batch_idx):
        return batch["x"] * (batch_idx + 1)


class TestCallPredictStep:
    def test_batch_only_signature(self):
        model = _BatchOnly()
        batch = {"x": torch.tensor([1.0, 2.0, 3.0])}
        out = _call_predict_step(model, batch, batch_idx=7, dataloader_idx=3)
        assert torch.equal(out, torch.tensor([2.0, 4.0, 6.0]))

    def test_full_signature_receives_indices(self):
        model = _FullSignature()
        batch = {"x": torch.tensor([10.0])}
        out = _call_predict_step(model, batch, batch_idx=5, dataloader_idx=2)
        assert torch.equal(out, torch.tensor([17.0]))

    def test_batch_and_batch_idx_signature(self):
        model = _BatchAndBatchIdx()
        batch = {"x": torch.tensor([4.0])}
        out = _call_predict_step(model, batch, batch_idx=2, dataloader_idx=0)
        assert torch.equal(out, torch.tensor([12.0]))
