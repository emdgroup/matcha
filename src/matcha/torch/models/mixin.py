"""Shared Lightning module mixin providing loss parsing, validation, and MC-dropout."""

import warnings
import torch
import torchmetrics
from typing import Any
import lightning as L
from matcha.nn.losses import LossRegistry, MultiLoss, MultitaskLoss, GradNormLoss
from matcha.nn.schedulers import SchedulerRegistry

# Schedulers that require total_steps to instantiate. A placeholder of 1 is
# injected at __init__ time; the real value is set by
# TrainingManager._maybe_inject_total_steps before training begins.
_SCHEDULERS_WITH_TOTAL_STEPS = frozenset(
    {
        "cosine_annealing",
        "cosine_annealing_cyclic",
        "warmup_cosine_annealing",
        "warmup_linear_decay",
        "one_cycle",
        "linear",
        "constant",
        "sequential",
    }
)


class ModelMixin(L.LightningModule):
    """Mixin providing common training infrastructure for all MATCHA Lightning models.

    Includes loss function parsing, per-task validation metric tracking, and
    MC-dropout toggling. Subclasses must define ``forward`` and ``training_step``.
    """

    @property
    def mc_dropout_flag(self) -> bool:
        """Whether MC-dropout is active during prediction for uncertainty estimation."""
        return self._mc_dropout_flag

    def _parse_loss_fn(self, loss_fn, loss_args, num_endpoints):
        """Instantiate the appropriate loss function and assign to ``self.loss_fn``.

        Handles single-endpoint, multitask, multiloss, and GradNorm configurations.

        :param str loss_fn: loss function name (or ``"multitask"``/``"multiloss"``/``"gradnorm"``).
        :param dict loss_args: keyword arguments forwarded to the loss constructor.
        :param int num_endpoints: number of prediction endpoints.
        """
        if num_endpoints == 1:
            self.loss_fn = LossRegistry[loss_fn](**loss_args)
        elif num_endpoints > 1 and loss_fn not in (
            "multitask",
            "multiloss",
            "gradnorm",
        ):
            self.loss_fn = MultitaskLoss(loss_fn=loss_fn, loss_args=loss_args)
        elif loss_fn == "gradnorm":
            loss_fn = loss_args["loss_fn"]
            loss_args = (
                loss_args[loss_args["loss_args"]] if "loss_args" in loss_args else {}
            )
            self.loss_fn = GradNormLoss(
                loss_fn=loss_fn, loss_args=loss_args, num_endpoints=num_endpoints
            )
        elif loss_fn == "multitask":
            loss_fn = loss_args["loss_fn"]
            loss_args = (
                loss_args[loss_args["loss_args"]] if "loss_args" in loss_args else {}
            )
            self.loss_fn = MultitaskLoss(
                loss_fn=loss_fn,
                loss_args=loss_args,
            )
        elif loss_fn == "multiloss":
            loss_configs = loss_args["loss_configs"]
            self.loss_fn = MultiLoss(loss_configs)

    def _make_scheduler(self, scheduler_name: str, optimizer, scheduler_args: dict):
        """Instantiate a scheduler, injecting a placeholder ``total_steps=1`` when the
        scheduler requires it but no value was provided.

        The placeholder is replaced with the real step count by
        ``TrainingManager._maybe_inject_total_steps`` before training begins.

        :param str scheduler_name: key in :data:`SchedulerRegistry`
        :param optimizer: the optimizer to attach the scheduler to
        :param dict scheduler_args: keyword arguments forwarded to the scheduler
        :returns: instantiated scheduler
        """
        scheduler_args = dict(scheduler_args)
        if (
            scheduler_name in _SCHEDULERS_WITH_TOTAL_STEPS
            and "total_steps" not in scheduler_args
        ):
            scheduler_args["total_steps"] = 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return SchedulerRegistry[scheduler_name](optimizer, **scheduler_args)

    def _init_metric_containers(self):
        """Initialize per-task torchmetrics containers for validation logging."""
        num_tasks = self.hparams["num_endpoints"]

        if num_tasks < self._max_task_tracking_n:
            self.val_precision = torch.nn.ModuleList(
                [torchmetrics.Precision(task="binary") for _ in range(num_tasks)]
            )
            self.val_recall = torch.nn.ModuleList(
                [torchmetrics.Recall(task="binary") for _ in range(num_tasks)]
            )
            self.val_mcc = torch.nn.ModuleList(
                [torchmetrics.MatthewsCorrCoef(task="binary") for _ in range(num_tasks)]
            )
            self.val_auroc = torch.nn.ModuleList(
                [torchmetrics.AUROC(task="binary") for _ in range(num_tasks)]
            )

            self.val_mse = torch.nn.ModuleList(
                [torchmetrics.MeanSquaredError() for _ in range(num_tasks)]
            )
            self.val_mae = torch.nn.ModuleList(
                [torchmetrics.MeanAbsoluteError() for _ in range(num_tasks)]
            )
            self.val_r2 = torch.nn.ModuleList(
                [torchmetrics.R2Score() for _ in range(num_tasks)]
            )
            self.val_pearson = torch.nn.ModuleList(
                [torchmetrics.PearsonCorrCoef() for _ in range(num_tasks)]
            )

    def switch_mc_dropout(self):
        """Enables switching of the mc_dropout flag, which defines whether dropout
        is kept on or off during the prediction step for uncertainty measurement
        """
        if self.mc_dropout_flag is True:
            self._mc_dropout_flag = False
        elif self.mc_dropout_flag is False:
            self._mc_dropout_flag = True

    def validation_step(self, batch: dict[str, Any], batch_idx) -> dict:
        """Validation step logic for classic models. Analogous as the training step.

        :param dict[str, Any] batch: batch of inputs to process
        :param int batch_idx: leftover from lightning tutorial which I am too scared
            to remove (TODO)
        :return torch.Tensor: batch loss
        """
        y = batch["y"]
        y_pred = self.forward(batch)
        if not isinstance(self.loss_fn, MultiLoss):
            val_loss = self.loss_fn(y_pred, y)
            self.log("val_loss", val_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        else:
            val_loss, loss_log = self.loss_fn(y_pred, y, self.global_step)
            self.log("val_loss", val_loss, prog_bar=True, on_epoch=True, sync_dist=True)
            for name, log in loss_log.items():
                self.log(
                    f"val_{name}_loss",
                    log["loss"],
                    prog_bar=True,
                    on_epoch=True,
                    sync_dist=True,
                )

        # very hacky, but it does the job
        # we will only track task-level metrics if the number of tasks is reasonable (e.g. under 100)
        # can be overridden at least for very custom jobs
        if y.shape[1] < self._max_task_tracking_n:
            seen_tags: set = set()
            for i in range(y.shape[1]):
                if self._label_names != []:
                    base = self._label_names[i]
                    tag = base if base not in seen_tags else f"{base}_{i}"
                    seen_tags.add(tag)
                else:
                    tag = i

                if y.dim() <= 2:
                    valid_mask = (
                        ~torch.isnan(y[:, i])
                        if y.dtype.is_floating_point
                        else y[:, i] != -1
                    )
                    y_task = y[valid_mask, i]
                else:
                    valid_mask = (
                        ~torch.isnan(y[:, i, 0])
                        if y.dtype.is_floating_point
                        else y[:, i] != -1
                    )
                    y_task = y[valid_mask, i, 0]

                y_pred_task = y_pred[valid_mask, i]

                if valid_mask.sum() == 0:
                    continue

                is_classification = torch.all((y_task == 0) | (y_task == 1))

                if is_classification:
                    y_pred_probs = torch.sigmoid(y_pred_task)
                    y_pred_binary = (y_pred_probs > 0.5).float()

                    if hasattr(self, "val_accuracy") and len(self.val_accuracy) > i:
                        self.val_precision[i].update(y_pred_binary, y_task.long())
                        self.val_recall[i].update(y_pred_binary, y_task.long())
                        self.val_mcc[i].update(y_pred_binary, y_task.long())
                        self.val_auroc[i].update(y_pred_probs, y_task.long())

                        self.log(
                            f"val_precision_{tag}",
                            self.val_precision[i],
                            prog_bar=False,
                            on_epoch=True,
                        )
                        self.log(
                            f"val_recall_{tag}",
                            self.val_recall[i],
                            prog_bar=False,
                            on_epoch=True,
                        )
                        self.log(
                            f"val_mcc_{tag}",
                            self.val_mcc[i],
                            prog_bar=False,
                            on_epoch=True,
                        )
                        self.log(
                            f"val_auroc_{tag}",
                            self.val_auroc[i],
                            prog_bar=False,
                            on_epoch=True,
                        )
                else:
                    if hasattr(self, "val_r2") and len(self.val_r2) > i:
                        self.val_r2[i].update(y_pred_task, y_task)
                        self.val_mae[i].update(y_pred_task, y_task)
                        self.val_mse[i].update(y_pred_task, y_task)
                        self.val_pearson[i].update(y_pred_task, y_task)

                        self.log(
                            f"val_r2_{tag}",
                            self.val_r2[i],
                            prog_bar=False,
                            on_epoch=True,
                        )
                        self.log(
                            f"val_mae_{tag}",
                            self.val_mae[i],
                            prog_bar=False,
                            on_epoch=True,
                        )
                        self.log(
                            f"val_mse_{tag}",
                            self.val_mse[i],
                            prog_bar=False,
                            on_epoch=True,
                        )
                        self.log(
                            f"val_pearsonr_{tag}",
                            self.val_pearson[i],
                            prog_bar=False,
                            on_epoch=True,
                        )

        return {"val_loss": val_loss}

    def set_label_names(self, label_names: list[str]):
        """Set human-readable label names used for per-task metric logging.

        :param label_names: list of endpoint names matching the order of prediction columns.
        """
        self._label_names = label_names
