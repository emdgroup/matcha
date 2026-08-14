"""Loss functions for regression, classification, and multitask learning."""

import torch
import torch.nn.functional as F
from torch import nn

from matcha.utils.registry import ClassRegistry

LossRegistry = ClassRegistry()


@LossRegistry.register(alias="focal-bce")
class BCEFocalLoss(nn.Module):
    """Focal loss implementation using binary cross entropy with logits.
    Suitable for binary classification with class imbalance.
    Reference: https://arxiv.org/abs/1708.02002
    """

    def __init__(self, gamma=2, alpha=None, reduction="mean"):
        """
        :param float gamma: Focusing parameter that down-weights easy examples.
        :param alpha: Balancing factor for the positive class, or ``None``.
        :type alpha: float or None
        :param str reduction: Reduction mode: ``'mean'``, ``'sum'``, or ``'none'``.
        """
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.eps = 1e-7

    def forward(self, inputs, targets) -> torch.Tensor:
        """
        :param torch.Tensor inputs: Raw logits.
        :param torch.Tensor targets: Binary target labels.
        :returns: Focal loss value.
        :rtype: torch.Tensor
        """

        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p = torch.sigmoid(inputs)
        p = torch.clamp(p, self.eps, 1 - self.eps)
        pt = p * targets + (1 - p) * (1 - targets)
        pt = torch.clamp(pt, self.eps, 1 - self.eps)
        focal_loss = ((1 - pt) ** self.gamma) * bce_loss

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


@LossRegistry.register(alias="poly1-bce")
class Poly1BCELoss(nn.Module):
    """Polynomial expansion of the binary cross entropy loss, which can lead to better
    classification performance if epsilon is tuned.
    Suitable for binary classification.
    Reference: https://arxiv.org/abs/2204.12511
    """

    def __init__(self, epsilon: float = 1.0, reduction: str = "mean"):
        """
        :param float epsilon: Polynomial coefficient controlling the additional term.
        :param str reduction: Reduction mode: ``'mean'``, ``'sum'``, or ``'none'``.
        """
        super(Poly1BCELoss, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.eps = 1e-7

    def forward(self, inputs, targets) -> torch.Tensor:
        """
        :param torch.Tensor inputs: Raw logits.
        :param torch.Tensor targets: Binary target labels.
        :returns: Poly1 BCE loss value.
        :rtype: torch.Tensor
        """
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p = torch.sigmoid(inputs)
        p = torch.clamp(p, self.eps, 1 - self.eps)
        pt = p * targets + (1 - p) * (1 - targets)
        poly1 = bce_loss + self.epsilon * (1 - pt)
        if self.reduction == "mean":
            return poly1.mean()
        elif self.reduction == "sum":
            return poly1.sum()
        else:
            return poly1


@LossRegistry.register(alias="multitask")
class MultitaskLoss(nn.Module):
    """Implementation of a customizable multitask loss. The same loss is
    used for all tasks.

    loss_fn is the name of the loss to broadcast, loss_args are its arguments.
    """

    def __init__(self, loss_fn: str = "mse", loss_args: dict = {}):
        """
        :param str loss_fn: Name of the loss function to broadcast across tasks.
        :param dict loss_args: Additional arguments passed to the loss constructor.
        """
        super().__init__()
        self.loss = LossRegistry[loss_fn](reduction="none", **loss_args)

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        :param torch.Tensor outputs: Predictions of shape ``(batch, num_tasks)``.
        :param torch.Tensor targets: Targets of shape ``(batch, num_tasks)``; NaN marks missing.
        :returns: Scalar loss averaged across valid entries and tasks.
        :rtype: torch.Tensor
        """
        valid_mask = torch.isnan(targets)
        targets = targets.clone()
        targets[valid_mask] = 0.0

        losses = self.loss(outputs, targets)
        if valid_mask.dim() == 2:
            losses[valid_mask] = 0.0
            valid_counts = (~valid_mask).sum(dim=0)
        else:
            losses[valid_mask[:, :, 0]] = 0.0
            valid_counts = (~valid_mask[:, :, 0]).sum(dim=0)

        total_loss = losses.sum(dim=0) / (
            valid_counts + 1e-8
        )  # Prevent division by zero
        self._per_task_losses = total_loss.detach().clone()
        return total_loss.sum() / total_loss.numel()  # Average across tasks


@LossRegistry.register(alias="multiloss")
class MultiLoss(nn.Module):
    """Implementation of a customizable multi-loss function that supports
    dynamic weight scheduling during training.

    Each loss configuration is a dictionary with:
    - loss_fn: name of the loss function
    - loss_args: arguments for the loss function
    - task_map: tuple/list indicating which columns this loss applies to (start, end)
    - init_w: initial weight at T=0
    - final_w: final weight at T=end
    - T: total epochs to transition from init_w to final_w
    - warmup: epochs to keep init_w fixed before starting transition
    """

    def __init__(self, loss_configs: list):
        """
        :param list loss_configs: List of dicts, each with keys ``loss_fn``, ``loss_args``,
            ``task_map``, ``init_w``, ``final_w``, ``T``, and ``warmup``.
        """
        super().__init__()
        self.loss_configs = loss_configs
        self.losses = nn.ModuleList()

        # Initialize loss functions
        for config in loss_configs:
            loss_fn = config["loss_fn"]
            loss_args = config.get("loss_args", {})
            loss = LossRegistry[loss_fn](reduction="none", **loss_args)
            self.losses.append(loss)

    def _calculate_weight(
        self, init_w: float, final_w: float, T: int, warmup: int, T_current: int
    ) -> float:
        """Calculate the current weight based on the scheduling parameters.

        :param float init_w: Initial weight at T=0.
        :param float final_w: Final weight at T=end.
        :param int T: Total epochs for the transition.
        :param int warmup: Epochs to keep ``init_w`` fixed.
        :param int T_current: Current epoch.
        :returns: Interpolated weight value.
        :rtype: float
        """
        if T_current < warmup:
            return init_w
        elif T_current >= warmup + T:
            return final_w
        else:
            # Linear interpolation between init_w and final_w
            progress = (T_current - warmup) / T
            return init_w + progress * (final_w - init_w)

    def forward(
        self, outputs: torch.Tensor, targets: torch.Tensor, T_current: int = 0
    ) -> tuple[torch.Tensor, dict]:
        """Forward pass with dynamic weight scheduling.

        Always returns ``(total_loss, loss_log)`` so training and validation share
        one contract. See issue #41: the earlier training-mode / eval-mode split
        was a hotfix with no in-repo reproducer and left ``BaseClassicModel`` /
        ``MLPModel`` callers unpacking a 0-d tensor. Callers that do not need the
        per-task log can ignore the second element.

        :param torch.Tensor outputs: Model predictions.
        :param torch.Tensor targets: Ground truth targets; NaN marks missing entries.
        :param int T_current: Current training epoch for weight scheduling.
        :returns: Tuple of ``(total_loss, loss_log)`` where ``loss_log`` maps each
            task name to a ``{"loss": float, "weight": float}`` record.
        :rtype: tuple[torch.Tensor, dict]
        """
        total_loss = 0.0
        loss_log = {}

        for i, config in enumerate(self.loss_configs):
            # Extract task mapping
            task_map = config["task_map"]
            task_outputs = outputs[:, task_map]
            task_targets = targets[:, task_map]

            # Handle NaN values (similar to MultitaskLoss)
            valid_mask = torch.isnan(task_targets)
            task_targets = task_targets.clone()
            task_targets[valid_mask] = 0.0

            # Compute loss for this task
            task_loss = self.losses[i](task_outputs, task_targets)

            # Mask out invalid entries (assuming 2D tensors only)
            task_loss[valid_mask] = 0.0
            valid_counts = (~valid_mask).sum(dim=0)

            # Average loss for this task
            task_loss_avg = task_loss.sum(dim=0) / (valid_counts + 1e-8)
            task_loss_final = task_loss_avg.sum() / task_loss_avg.numel()

            # Calculate dynamic weight
            current_weight = self._calculate_weight(
                init_w=config["init_w"],
                final_w=config["final_w"],
                T=config["T"],
                warmup=config["warmup"],
                T_current=T_current,
            )

            # Add weighted loss to total
            total_loss += current_weight * task_loss_final

            # log i-th loss
            name = f"task_{i}" if "name" not in config else config["name"]
            loss_log[name] = {"loss": task_loss_final.item(), "weight": current_weight}

        return total_loss, loss_log


@LossRegistry.register(alias="bounded")
class BoundedLoss(nn.Module):
    """Loss function for handling bounded regression

    Allows the use of bound information on the readout (e.g. IC50 < x),
    so that the model is not penalized when it predicts e.g. y_pred < x.
    Implementation is based on: https://chemprop.readthedocs.io/en/latest/_modules/chemprop/nn/metrics.html#BoundedMixin
    """

    def __init__(self, loss_fn: str = "mse", **kwargs):
        """
        :param str loss_fn: Name of the inner loss function.
        :param kwargs: Additional arguments passed to the inner loss.
        """
        super().__init__()
        self.loss = LossRegistry[loss_fn](**kwargs)

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        :param torch.Tensor outputs: Predictions.
        :param torch.Tensor targets: Targets with bound info in the last dimension.
            Shape ``(batch, [num_tasks,] 2)`` where ``[..., 0]`` is the value and
            ``[..., 1]`` encodes the bound type (``-1`` = less-than, ``1`` = greater-than).
        :returns: Loss computed only on non-masked predictions.
        :rtype: torch.Tensor
        """
        if outputs.dim() == 1:
            mask_values = targets[:, 0, 1]
            target_values = targets[:, 0, 0]
        else:
            mask_values = targets[:, :, 1]
            target_values = targets[:, :, 0]
        lt_mask = mask_values == -1
        gt_mask = mask_values == 1
        outputs = torch.where(
            (outputs < target_values) & lt_mask, target_values, outputs
        )
        outputs = torch.where(
            (outputs > target_values) & gt_mask, target_values, outputs
        )
        return self.loss(outputs, target_values)


@LossRegistry.register("mse")
class MSELoss(nn.MSELoss):
    """Mean squared error loss (wraps :class:`torch.nn.MSELoss`)."""

    pass


@LossRegistry.register("mae")
class L1Loss(nn.L1Loss):
    """Mean absolute error loss (wraps :class:`torch.nn.L1Loss`)."""

    pass


@LossRegistry.register("huber")
class HuberLoss(nn.HuberLoss):
    """Huber loss (wraps :class:`torch.nn.HuberLoss`)."""

    pass


@LossRegistry.register("smoothl1")
class SmoothL1Loss(nn.SmoothL1Loss):
    """Smooth L1 loss (wraps :class:`torch.nn.SmoothL1Loss`)."""

    pass


@LossRegistry.register(alias="bounded-mse")
class BoundedMSELoss(BoundedLoss):
    """:class:`BoundedLoss` with MSE as the inner loss."""

    def __init__(self, **kwargs):
        super().__init__(loss_fn="mse", **kwargs)


@LossRegistry.register(alias="bounded-mae")
class BoundedMAELoss(BoundedLoss):
    """:class:`BoundedLoss` with MAE as the inner loss."""

    def __init__(self, **kwargs):
        super().__init__(loss_fn="mae", **kwargs)


@LossRegistry.register(alias="bounded-huber")
class BoundedHuberLoss(BoundedLoss):
    """:class:`BoundedLoss` with Huber as the inner loss."""

    def __init__(self, **kwargs):
        super().__init__(loss_fn="huber", **kwargs)


@LossRegistry.register(alias="bounded-smoothl1")
class BoundedSmoothL1Loss(BoundedLoss):
    """:class:`BoundedLoss` with Smooth L1 as the inner loss."""

    def __init__(self, **kwargs):
        super().__init__(loss_fn="smoothl1", **kwargs)


@LossRegistry.register("bce")
class BCELoss(nn.BCEWithLogitsLoss):
    """Binary cross-entropy with logits (wraps :class:`torch.nn.BCEWithLogitsLoss`)."""

    pass


@LossRegistry.register("cross_entropy")
class CrossEntropyLoss(nn.CrossEntropyLoss):
    """Cross-entropy loss (wraps :class:`torch.nn.CrossEntropyLoss`)."""

    pass


@LossRegistry.register(alias="weighted-bce")
class WeightedBCELoss(nn.Module):
    """Weighted binary cross entropy loss with logits.

    Applies per-class weights to handle class imbalance in binary classification.
    The user specifies the weight for the positive (minority) class; the weight
    for the negative class is computed so that ``w0 + w1 = 1``.

    :param float w1: Weight for class 1 (positive / minority class). Must be in (0, 1).
    :param str reduction: Reduction mode: ``'mean'``, ``'sum'``, or ``'none'``.
    """

    def __init__(self, w1: float = 0.5, reduction: str = "mean"):
        super().__init__()
        if not 0 < w1 < 1:
            raise ValueError(f"w1 must be in (0, 1), got {w1}")
        self.w1 = w1
        self.w0 = 1.0 - w1
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        :param torch.Tensor inputs: Raw logits.
        :param torch.Tensor targets: Binary target labels.
        :returns: Weighted BCE loss value.
        :rtype: torch.Tensor
        """
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        weights = targets * self.w1 + (1 - targets) * self.w0
        weighted_loss = weights * bce_loss

        if self.reduction == "mean":
            return weighted_loss.mean()
        elif self.reduction == "sum":
            return weighted_loss.sum()
        else:
            return weighted_loss


@LossRegistry.register(alias="gradnorm")
class GradNormLoss(nn.Module):
    """Implementation of GradNorm for adaptive loss balancing in multitask learning.

    GradNorm automatically balances training by dynamically tuning gradient magnitudes.
    It adjusts task weights to ensure that all tasks train at similar rates.
    The weight updates are handled internally — no separate optimizer needed.

    Reference: https://arxiv.org/abs/1711.02257

    :param str loss_fn: Name of the loss function to use for all tasks (e.g., ``"mse"``).
    :param dict loss_args: Arguments to pass to the loss function constructor.
    :param int num_endpoints: Number of tasks.
    :param float weight_lr: Learning rate for updating task weights internally.

    Example::

        loss_fn = GradNormLoss(loss_fn="mse", num_endpoints=3)

        # Training loop (no changes needed):
        loss = loss_fn(outputs, targets, shared_layer=model.backbone[-1])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    """

    ALPHA = 1.5  # Asymmetry hyperparameter (hardcoded as per paper recommendation)

    def __init__(
        self,
        loss_fn: str = "mse",
        loss_args: dict = {},
        num_endpoints: int = 1,
        weight_lr: float = 0.025,
    ):
        super().__init__()
        self.loss = LossRegistry[loss_fn](reduction="none", **loss_args)
        self.num_endpoints = num_endpoints
        self.weight_lr = weight_lr

        # Task weights (not nn.Parameter - updated manually via GradNorm)
        self.register_buffer("weights", torch.ones(num_endpoints))

        # Store initial losses for computing relative inverse training rate
        # Not saved in state_dict - recomputed on first forward pass after loading
        self.initial_losses = None

    def _compute_task_losses(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute individual task losses, handling NaN values."""
        valid_mask = torch.isnan(targets)
        targets = targets.clone()
        targets[valid_mask] = 0.0

        losses = self.loss(outputs, targets)

        if valid_mask.dim() == 2:
            losses[valid_mask] = 0.0
            valid_counts = (~valid_mask).sum(dim=0)
        else:
            losses[valid_mask[:, :, 0]] = 0.0
            valid_counts = (~valid_mask[:, :, 0]).sum(dim=0)

        # Average loss per task
        task_losses = losses.sum(dim=0) / (valid_counts + 1e-8)
        return task_losses

    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        shared_layer: nn.Module = None,
    ) -> torch.Tensor:
        """Forward pass computing weighted multitask loss.

        :param torch.Tensor outputs: Predictions of shape ``(batch_size, num_endpoints)``.
        :param torch.Tensor targets: Targets of shape ``(batch_size, num_endpoints)``.
        :param shared_layer: The last shared layer of the network. Required during
            training for GradNorm weight updates.
        :type shared_layer: torch.nn.Module or None
        :returns: Weighted sum of task losses.
        :rtype: torch.Tensor
        """
        task_losses = self._compute_task_losses(outputs, targets)
        self._per_task_losses = task_losses.detach().clone()

        # Initialize initial losses on first forward pass
        if self.initial_losses is None:
            self.initial_losses = task_losses.detach().clone()

        # Normalize weights to sum to num_endpoints
        normalized_weights = self.weights * (self.num_endpoints / self.weights.sum())

        # Compute weighted loss
        weighted_losses = normalized_weights * task_losses
        total_loss = weighted_losses.sum()

        # During training with shared_layer, update weights using GradNorm
        if self.training and shared_layer is not None:
            self._update_weights(task_losses, normalized_weights, shared_layer)

        return total_loss

    def _update_weights(
        self,
        task_losses: torch.Tensor,
        normalized_weights: torch.Tensor,
        shared_layer: nn.Module,
    ) -> None:
        """Update task weights internally using GradNorm algorithm."""
        # Get the weight parameter of the shared layer
        if hasattr(shared_layer, "weight"):
            shared_weight = shared_layer.weight
        else:
            shared_weight = next(shared_layer.parameters())

        # Compute gradient norms for each task
        grad_norms = []
        for i in range(self.num_endpoints):
            grad = torch.autograd.grad(
                task_losses[i],
                shared_weight,
                retain_graph=True,
                create_graph=False,  # No need to track higher-order gradients
            )[0]
            # Weighted gradient norm: w_i * ||grad_i||
            grad_norm = (normalized_weights[i] * grad).norm(p=2)
            grad_norms.append(grad_norm)

        grad_norms = torch.stack(grad_norms)

        # Compute average gradient norm (target baseline)
        avg_grad_norm = grad_norms.mean()

        # Compute relative inverse training rate
        loss_ratios = task_losses.detach() / (self.initial_losses + 1e-8)
        relative_inv_rates = loss_ratios / (loss_ratios.mean() + 1e-8)

        # Compute target gradient norms
        target_grad_norms = avg_grad_norm * (relative_inv_rates**self.ALPHA)

        # Compute weight updates: move weights toward targets
        # If grad_norm > target, decrease weight; if grad_norm < target, increase weight
        with torch.no_grad():
            # Gradient of L1 loss w.r.t. weights (simplified direct update)
            # weights should change to make grad_norms closer to target_grad_norms
            grad_norm_diff = grad_norms - target_grad_norms

            # Update weights: decrease weight if gradient is too large, increase if too small
            # Since grad_norm = w_i * ||grad_i||, we adjust w_i proportionally
            weight_grad = grad_norm_diff / (grad_norms + 1e-8) * self.weights
            self.weights -= self.weight_lr * weight_grad

            # Clamp weights to be positive
            self.weights.clamp_(min=1e-8)

            # Renormalize weights to sum to num_endpoints
            self.weights *= self.num_endpoints / self.weights.sum()

    def reset_initial_losses(self) -> None:
        """Reset the initial losses for a fresh start of GradNorm tracking."""
        self.initial_losses = None


@LossRegistry.register(alias="dropout")
class DropoutLoss(nn.Module):
    """Wrap a per-element loss and randomly mask a fraction of entries each step.

    Intended as a regularizer for multi-endpoint pretraining (e.g. predicting many
    molecular descriptors at once): randomly dropping a fraction of labels from the
    loss on every forward pass discourages overfitting to any single endpoint.

    Reference: https://github.com/JacksonBurns/how-to-train-your-chemeleon/blob/main/pretraining/random_dropout_mse.py

    The inner loss is instantiated with ``reduction="none"`` so masking happens
    before reduction. The dropout mask is resampled every forward and composes
    with the existing NaN mask (NaN targets are always excluded, as in
    :class:`MultitaskLoss`). In ``eval()`` mode the wrapper reduces to a plain
    NaN-masked mean of the inner loss, regardless of ``dropout``.

    :param str loss_fn: Alias of the inner per-element loss (resolved via
        :class:`LossRegistry`). Defaults to ``"mse"``.
    :param float dropout: Fraction of non-NaN entries to drop from the loss on
        each training forward pass. Must satisfy ``0.0 <= dropout < 1.0``.
    :param seed: Optional integer seed for a private :class:`torch.Generator`.
        When set, the mask trajectory is reproducible across runs and does not
        perturb the ambient torch RNG. When ``None`` (default), masks are drawn
        from the ambient RNG (like :class:`torch.nn.Dropout`).
    :type seed: int or None
    :param kwargs: Extra keyword arguments forwarded to the inner loss constructor.
    """

    def __init__(
        self,
        loss_fn: str = "mse",
        dropout: float = 0.0,
        seed: int | None = None,
        **kwargs,
    ):
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must lie in [0.0, 1.0), got {dropout}")
        super().__init__()
        self.dropout = float(dropout)
        self._seed = seed
        self._generator: torch.Generator | None = None
        inner_cls = LossRegistry[loss_fn]
        if issubclass(
            inner_cls,
            (MultitaskLoss, MultiLoss, BoundedLoss, GradNormLoss, DropoutLoss),
        ):
            raise ValueError(
                f"DropoutLoss cannot wrap another wrapper loss "
                f"(got {inner_cls.__name__}). Wrap a per-element loss "
                f"(e.g. 'mse', 'mae', 'bce', 'focal-bce') instead."
            )
        kwargs.pop("reduction", None)
        self.loss = inner_cls(reduction="none", **kwargs)

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        :param torch.Tensor outputs: Predictions of shape ``(batch, num_tasks)``.
        :param torch.Tensor targets: Targets of shape ``(batch, num_tasks)``; NaN
            marks missing entries and is always excluded from the loss.
        :returns: Scalar loss averaged across kept (non-NaN, non-dropped) entries.
        :rtype: torch.Tensor
        """
        nan_mask = torch.isnan(targets)
        targets = targets.clone()
        targets[nan_mask] = 0.0

        losses = self.loss(outputs, targets)
        keep_mask = ~nan_mask

        if self.training and self.dropout > 0.0:
            if self._seed is not None and self._generator is None:
                self._generator = torch.Generator(device=losses.device)
                self._generator.manual_seed(self._seed)
            keep_from_dropout = (
                torch.rand(
                    losses.shape,
                    device=losses.device,
                    generator=self._generator,
                )
                >= self.dropout
            )
            keep_mask = keep_mask & keep_from_dropout

        losses = losses * keep_mask
        return losses.sum() / (keep_mask.sum() + 1e-8)


@LossRegistry.register(alias="dropout-mse")
class DropoutMSELoss(DropoutLoss):
    """:class:`DropoutLoss` with MSE as the inner loss."""

    def __init__(self, **kwargs):
        super().__init__(loss_fn="mse", **kwargs)


@LossRegistry.register(alias="dropout-mae")
class DropoutMAELoss(DropoutLoss):
    """:class:`DropoutLoss` with MAE as the inner loss."""

    def __init__(self, **kwargs):
        super().__init__(loss_fn="mae", **kwargs)


@LossRegistry.register(alias="dropout-huber")
class DropoutHuberLoss(DropoutLoss):
    """:class:`DropoutLoss` with Huber as the inner loss."""

    def __init__(self, **kwargs):
        super().__init__(loss_fn="huber", **kwargs)


@LossRegistry.register(alias="dropout-smoothl1")
class DropoutSmoothL1Loss(DropoutLoss):
    """:class:`DropoutLoss` with Smooth L1 as the inner loss."""

    def __init__(self, **kwargs):
        super().__init__(loss_fn="smoothl1", **kwargs)


@LossRegistry.register(alias="dropout-bce")
class DropoutBCELoss(DropoutLoss):
    """:class:`DropoutLoss` with BCE-with-logits as the inner loss."""

    def __init__(self, **kwargs):
        super().__init__(loss_fn="bce", **kwargs)


@LossRegistry.register(alias="dropout-focal-bce")
class DropoutFocalBCELoss(DropoutLoss):
    """:class:`DropoutLoss` with focal BCE as the inner loss."""

    def __init__(self, **kwargs):
        super().__init__(loss_fn="focal-bce", **kwargs)


@LossRegistry.register(alias="dropout-poly1-bce")
class DropoutPoly1BCELoss(DropoutLoss):
    """:class:`DropoutLoss` with Poly1 BCE as the inner loss."""

    def __init__(self, **kwargs):
        super().__init__(loss_fn="poly1-bce", **kwargs)


@LossRegistry.register(alias="dropout-weighted-bce")
class DropoutWeightedBCELoss(DropoutLoss):
    """:class:`DropoutLoss` with weighted BCE as the inner loss."""

    def __init__(self, **kwargs):
        super().__init__(loss_fn="weighted-bce", **kwargs)
