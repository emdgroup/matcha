"""Tests for mean-variance-estimation losses."""

import torch

from matcha.nn.losses import BetaNLLLoss, BoundedBetaNLLLoss, LossRegistry


# ===================================================================
# BetaNLLLoss
# ===================================================================


def _reference_beta_nll(
    mean: torch.Tensor,
    log_var: torch.Tensor,
    targets: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Hand-computed β-NLL reference for a fully-observed batch.

    Averages per-task then across tasks (matches ``BetaNLLLoss.forward``).
    """
    var = log_var.exp()
    per_element = 0.5 * log_var + 0.5 * (targets - mean) ** 2 / var
    weight = var.detach() ** beta
    losses = weight * per_element
    per_task = losses.mean(dim=0)
    return per_task.sum() / per_task.numel()


class TestBetaNLLLoss:
    def test_registry_lookup_alias_stability(self):
        assert LossRegistry["beta-nll"] is BetaNLLLoss
        assert LossRegistry["mve"] is BetaNLLLoss

    def test_requires_mve_head_marker(self):
        assert BetaNLLLoss._requires_mve_head is True

    def test_output_scalar_single_task(self):
        loss_fn = BetaNLLLoss(beta=0.5)
        preds = torch.randn(8, 1, 2)  # 1 task: (mean, log_var) along last axis
        targets = torch.randn(8, 1)
        loss = loss_fn(preds, targets)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_output_scalar_multitask(self, multitask_targets):
        num_tasks = multitask_targets.shape[1]
        loss_fn = BetaNLLLoss(beta=0.5)
        preds = torch.randn(multitask_targets.shape[0], num_tasks, 2)
        loss = loss_fn(preds, multitask_targets.clone())
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_matches_reference_no_nan(self):
        torch.manual_seed(0)
        num_tasks = 3
        mean = torch.randn(16, num_tasks)
        log_var = torch.randn(16, num_tasks) * 0.5
        targets = torch.randn(16, num_tasks)
        preds = torch.stack([mean, log_var], dim=-1)

        loss_fn = BetaNLLLoss(beta=0.5)
        loss = loss_fn(preds, targets)
        expected = _reference_beta_nll(mean, log_var, targets, beta=0.5)
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_beta_zero_recovers_vanilla_nll(self):
        torch.manual_seed(0)
        mean = torch.randn(16, 2)
        log_var = torch.randn(16, 2) * 0.3
        targets = torch.randn(16, 2)
        preds = torch.stack([mean, log_var], dim=-1)

        vanilla_per_element = (
            0.5 * log_var + 0.5 * (targets - mean) ** 2 / log_var.exp()
        )
        expected = vanilla_per_element.mean()

        loss_fn = BetaNLLLoss(beta=0.0)
        loss = loss_fn(preds, targets)
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_beta_one_matches_reference(self):
        """β = 1 gives MSE-like weighting (σ² times per-element NLL)."""
        torch.manual_seed(0)
        mean = torch.randn(16, 2)
        log_var = torch.randn(16, 2) * 0.3
        targets = torch.randn(16, 2)
        preds = torch.stack([mean, log_var], dim=-1)

        loss_fn = BetaNLLLoss(beta=1.0)
        loss = loss_fn(preds, targets)
        expected = _reference_beta_nll(mean, log_var, targets, beta=1.0)
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_gradient_flows_to_mean_and_log_var(self):
        loss_fn = BetaNLLLoss(beta=0.5)
        preds = torch.randn(8, 2, 2, requires_grad=True)
        targets = torch.randn(8, 2)
        loss = loss_fn(preds, targets)
        loss.backward()
        assert preds.grad is not None
        # Grad on both channels must be non-zero.
        assert torch.any(preds.grad[..., 0] != 0)
        assert torch.any(preds.grad[..., 1] != 0)

    def test_sigma_reweighting_factor_is_detached(self):
        """The σ^(2β) weight must not contribute a gradient to log_var beyond
        the standard per-element NLL term."""
        torch.manual_seed(0)
        # β = 0 → weight = 1 (no reweighting). Grad of log_var must equal the
        # vanilla-NLL grad. β > 0 must give the SAME grad shape on log_var as
        # if the weight were a constant, i.e. detach() worked.
        mean = torch.randn(16, 2)
        log_var = torch.zeros(16, 2, requires_grad=True)
        targets = torch.randn(16, 2)

        # Manually compute grad WITHOUT detach — would flow through σ^(2β).
        var = log_var.exp()
        per_element = 0.5 * log_var + 0.5 * (targets - mean) ** 2 / var
        weight_no_detach = var**0.5
        loss_no_detach = (weight_no_detach * per_element).mean()
        grad_no_detach = torch.autograd.grad(loss_no_detach, log_var)[0]

        # BetaNLLLoss with detach.
        log_var2 = torch.zeros(16, 2, requires_grad=True)
        preds = torch.stack([mean, log_var2], dim=-1)
        loss = BetaNLLLoss(beta=0.5)(preds, targets)
        grad_detach = torch.autograd.grad(loss, log_var2)[0]

        # The two gradients must differ — proves the detach short-circuits
        # the flow through σ^(2β).
        assert not torch.allclose(grad_no_detach, grad_detach, atol=1e-6)

    def test_extreme_log_var_stays_finite(self):
        """log_var outside the clamp range must not produce NaN/inf loss."""
        preds = torch.zeros(4, 2, 2)
        preds[..., 1] = 1e6  # extreme log_var
        targets = torch.zeros(4, 2)
        loss = BetaNLLLoss(beta=0.5)(preds, targets)
        assert torch.isfinite(loss)

        preds[..., 1] = -1e6
        loss = BetaNLLLoss(beta=0.5)(preds, targets)
        assert torch.isfinite(loss)

    def test_nan_targets_masked_per_task(self):
        """Entries with NaN targets should not contribute to the loss."""
        num_tasks = 3
        preds = torch.randn(8, num_tasks, 2)
        targets = torch.randn(8, num_tasks)
        targets[0, 1] = float("nan")
        targets[3, 0] = float("nan")
        loss = BetaNLLLoss(beta=0.5)(preds, targets)
        assert torch.isfinite(loss)

    def test_all_nan_column_finite(self):
        num_tasks = 3
        preds = torch.randn(8, num_tasks, 2)
        targets = torch.full((8, num_tasks), float("nan"))
        targets[:, 0] = torch.randn(8)
        loss = BetaNLLLoss(beta=0.5)(preds, targets)
        assert torch.isfinite(loss)

    def test_per_task_losses_attribute(self, multitask_targets):
        num_tasks = multitask_targets.shape[1]
        loss_fn = BetaNLLLoss(beta=0.5)
        preds = torch.randn(multitask_targets.shape[0], num_tasks, 2)
        loss_fn(preds, multitask_targets.clone())
        assert hasattr(loss_fn, "_per_task_losses")
        assert loss_fn._per_task_losses.shape == (num_tasks,)
        assert not loss_fn._per_task_losses.requires_grad


# ===================================================================
# BoundedBetaNLLLoss
# ===================================================================


class TestBoundedBetaNLLLoss:
    def test_registry_lookup_alias_stability(self):
        assert LossRegistry["bounded-beta-nll"] is BoundedBetaNLLLoss

    def test_requires_mve_head_marker(self):
        assert BoundedBetaNLLLoss._requires_mve_head is True

    def test_output_scalar_single_task(self):
        loss_fn = BoundedBetaNLLLoss(beta=0.5)
        preds = torch.randn(8, 1, 2)  # 1 task: (mean, log_var) along last axis
        targets = torch.zeros(8, 1, 2)
        targets[..., 0] = torch.randn(8, 1)
        loss = loss_fn(preds, targets)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_output_scalar_multitask(self):
        loss_fn = BoundedBetaNLLLoss(beta=0.5)
        preds = torch.randn(8, 2, 2)  # 2 tasks
        targets = torch.zeros(8, 2, 2)
        targets[..., 0] = torch.randn(8, 2)
        targets[:, 0, 1] = -1
        targets[:, 1, 1] = 1
        loss = loss_fn(preds, targets)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_exact_bound_matches_reference_beta_nll(self):
        """With ``bound_code == 0`` everywhere, the loss must equal the
        standard β-NLL on those same values."""
        torch.manual_seed(0)
        num_tasks = 3
        mean = torch.randn(16, num_tasks)
        log_var = torch.randn(16, num_tasks) * 0.3
        values = torch.randn(16, num_tasks)
        preds = torch.stack([mean, log_var], dim=-1)
        targets = torch.stack([values, torch.zeros_like(values)], dim=-1)

        loss = BoundedBetaNLLLoss(beta=0.5)(preds, targets)
        expected = _reference_beta_nll(mean, log_var, values, beta=0.5)
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_lt_bound_uses_log_ndtr(self):
        """For a ``bound_code == -1`` row, the per-element contribution must
        equal ``-log_ndtr((y - μ) / σ)``."""
        mean = torch.tensor([[0.0]])
        log_var = torch.tensor([[0.0]])  # σ = 1
        preds = torch.stack([mean, log_var], dim=-1)
        targets = torch.tensor([[[1.5, -1.0]]])

        loss = BoundedBetaNLLLoss(beta=0.5)(preds, targets)
        expected = -torch.special.log_ndtr(torch.tensor(1.5))
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_gt_bound_uses_negated_log_ndtr(self):
        """For ``bound_code == +1``, contribution must equal
        ``-log_ndtr(-(y - μ) / σ)``."""
        mean = torch.tensor([[0.0]])
        log_var = torch.tensor([[0.0]])
        preds = torch.stack([mean, log_var], dim=-1)
        targets = torch.tensor([[[-0.5, 1.0]]])

        loss = BoundedBetaNLLLoss(beta=0.5)(preds, targets)
        expected = -torch.special.log_ndtr(torch.tensor(0.5))
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_gradient_flows(self):
        preds = torch.randn(8, 2, 2, requires_grad=True)
        targets = torch.zeros(8, 2, 2)
        targets[..., 0] = torch.randn(8, 2)
        targets[:, 0, 1] = -1
        targets[:, 1, 1] = 1
        loss = BoundedBetaNLLLoss(beta=0.5)(preds, targets)
        loss.backward()
        assert preds.grad is not None
        assert torch.any(preds.grad != 0)

    def test_nan_values_masked(self):
        preds = torch.randn(8, 2, 2)
        targets = torch.zeros(8, 2, 2)
        targets[..., 0] = torch.randn(8, 2)
        targets[0, 0, 0] = float("nan")
        targets[3, 1, 0] = float("nan")
        loss = BoundedBetaNLLLoss(beta=0.5)(preds, targets)
        assert torch.isfinite(loss)

    def test_extreme_log_var_stays_finite(self):
        preds = torch.zeros(4, 2, 2)
        preds[..., 1] = 1e6
        targets = torch.zeros(4, 2, 2)
        targets[..., 0] = torch.zeros(4, 2)
        loss = BoundedBetaNLLLoss(beta=0.5)(preds, targets)
        assert torch.isfinite(loss)
