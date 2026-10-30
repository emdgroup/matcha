"""Learning-rate schedulers registered in the :data:`SchedulerRegistry`."""

from torch.optim import lr_scheduler
from torch.optim.optimizer import Optimizer
from matcha.utils.registry import ClassRegistry
import math

SchedulerRegistry = ClassRegistry()


@SchedulerRegistry.register("chemprop")
class ChempropSchedulerConfig:
    """Thin configuration holder for Chemprop's built-in NoamLR scheduler.

    Chemprop's MPNN base class manages its own learning-rate schedule internally
    (via ``NoamLR``).  This class simply stores the configuration so that the
    rest of the Matcha scheduler interface can treat ``"chemprop"`` uniformly
    as a scheduler name with associated ``scheduler_args``.

    Unlike real PyTorch schedulers this is **never stepped** — the MPNN
    ``configure_optimizers`` takes care of that.

    Expected ``scheduler_args`` keys:
        * ``warmup_epochs`` (int)
        * ``max_lr`` (float)
        * ``final_lr`` (float)
    """

    def __init__(self, optimizer=None, **kwargs):
        self.warmup_epochs = kwargs.get("warmup_epochs", 2)
        self.max_lr = kwargs.get("max_lr", 1e-3)
        self.final_lr = kwargs.get("final_lr", 1e-5)

    def step(self, *args, **kwargs):
        """No-op: MPNN handles its own scheduling."""
        pass

    def state_dict(self):
        return {
            "warmup_epochs": self.warmup_epochs,
            "max_lr": self.max_lr,
            "final_lr": self.final_lr,
        }

    def load_state_dict(self, state_dict):
        self.warmup_epochs = state_dict.get("warmup_epochs", self.warmup_epochs)
        self.max_lr = state_dict.get("max_lr", self.max_lr)
        self.final_lr = state_dict.get("final_lr", self.final_lr)


@SchedulerRegistry.register("one_cycle")
class OneCycleLR(lr_scheduler.OneCycleLR):
    """One-cycle learning rate policy (wraps :class:`torch.optim.lr_scheduler.OneCycleLR`)."""

    pass


@SchedulerRegistry.register("cosine_annealing")
class CosineAnnealing(lr_scheduler.CosineAnnealingLR):
    """Cosine annealing scheduler with configurable minimum learning rate.

    :param optimizer: Wrapped optimizer.
    :param float min_lr: Minimum learning rate.
    :param int total_steps: Total number of steps (mapped to ``T_max``).
    """

    def __init__(
        self, optimizer, *, min_lr: float = 1e-5, total_steps: int = 1, **kwargs
    ):
        super().__init__(optimizer, T_max=total_steps, eta_min=min_lr, **kwargs)


@SchedulerRegistry.register("cosine_annealing_cyclic")
class CosineAnnealingCyclic(lr_scheduler.CosineAnnealingWarmRestarts):
    """Cosine annealing with warm restarts, dividing total steps into equal cycles.

    :param optimizer: Wrapped optimizer.
    :param int total_steps: Total number of steps.
    :param int num_cycles: Number of restart cycles.
    :param float min_lr: Minimum learning rate.
    """

    def __init__(
        self,
        optimizer,
        *,
        total_steps: int,
        num_cycles: int = 5,
        min_lr: float = 0,
        **kwargs,
    ):
        T_0 = max(1, total_steps // num_cycles)
        super().__init__(optimizer, T_0=T_0, eta_min=min_lr, **kwargs)


@SchedulerRegistry.register("step")
class Step(lr_scheduler.StepLR):
    """Step decay scheduler (wraps :class:`torch.optim.lr_scheduler.StepLR`).

    Accepts and discards a ``total_steps`` kwarg for interface compatibility.
    """

    def __init__(self, optimizer, **kwargs):
        kwargs.pop("total_steps", None)
        super().__init__(optimizer, **kwargs)


@SchedulerRegistry.register("warmup_cosine_annealing")
class WarmupCosineAnnealingLR(lr_scheduler._LRScheduler):
    """Linear warmup followed by cosine annealing decay.

    :param optimizer: Wrapped optimizer.
    :param int total_steps: Total number of steps (warmup + annealing).
    :param float warmup_ratio: Fraction of ``total_steps`` used for warmup.
    :param float peak_lr_factor: Multiplier applied to base LR to compute peak LR.
    :param float min_lr: Minimum learning rate after annealing.
    :param int last_epoch: Index of last epoch (for resume).
    """

    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_ratio: float = 0.05,
        peak_lr_factor: float = 10.0,
        min_lr: float = 1e-5,
        last_epoch: int = -1,
    ):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.last_epoch = last_epoch
        self.start_lr = [group["lr"] for group in optimizer.param_groups]
        self._warmup_steps = int(total_steps * warmup_ratio)
        self.cosine_steps = total_steps - self._warmup_steps
        self.current_step = 0
        self.peak_lr_factor = peak_lr_factor
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self._warmup_steps:
            # Linear warmup
            return [
                base_lr
                + (base_lr * self.peak_lr_factor - base_lr)
                * (self.last_epoch / self._warmup_steps)
                for base_lr in self.start_lr
            ]
        else:
            # Cosine annealing
            if self.last_epoch < self.total_steps:
                decay_step = self.last_epoch - self._warmup_steps
                return [
                    self.min_lr
                    + (base_lr * self.peak_lr_factor - self.min_lr)
                    * (0.5 * (1 + math.cos(math.pi * decay_step / self.cosine_steps)))
                    for base_lr in self.start_lr
                ]
            else:
                return [self.min_lr for base_lr in self.start_lr]


@SchedulerRegistry.register("linear")
class Linear(lr_scheduler.LinearLR):
    """Linear learning rate schedule (wraps :class:`torch.optim.lr_scheduler.LinearLR`).

    Maps ``total_steps`` to ``total_iters`` for interface compatibility.
    """

    def __init__(self, optimizer, **kwargs):
        if "total_steps" in kwargs:
            kwargs["total_iters"] = kwargs.pop("total_steps")
        super().__init__(optimizer, **kwargs)


@SchedulerRegistry.register("constant")
class Constant(lr_scheduler.ConstantLR):
    """Constant learning rate schedule (wraps :class:`torch.optim.lr_scheduler.ConstantLR`).

    Maps ``total_steps`` to ``total_iters`` for interface compatibility.
    """

    def __init__(self, optimizer, **kwargs):
        if "total_steps" in kwargs:
            kwargs["total_iters"] = kwargs.pop("total_steps")
        super().__init__(optimizer, **kwargs)


@SchedulerRegistry.register("warmup_linear_decay")
class WarmupLinearDecayLR(lr_scheduler._LRScheduler):
    """Linear warmup followed by linear decay.

    :param optimizer: Wrapped optimizer.
    :param int total_steps: Total number of steps (warmup + decay).
    :param float warmup_ratio: Fraction of ``total_steps`` used for warmup.
    :param float peak_lr_factor: Multiplier applied to base LR to compute peak LR.
    :param float min_lr: Minimum learning rate after decay.
    :param int last_epoch: Index of last epoch (for resume).
    """

    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_ratio: float = 0.05,
        peak_lr_factor: float = 10.0,
        min_lr: float = 1e-5,
        last_epoch: int = -1,
    ):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.last_epoch = last_epoch
        self.start_lr = [group["lr"] for group in optimizer.param_groups]
        self._warmup_steps = int(total_steps * warmup_ratio)
        self.decay_steps = total_steps - self._warmup_steps
        self.current_step = 0
        self.peak_lr_factor = peak_lr_factor
        super(WarmupLinearDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self._warmup_steps:
            # Linear warmup
            return [
                base_lr
                + (base_lr * self.peak_lr_factor - base_lr)
                * (self.last_epoch / self._warmup_steps)
                for base_lr in self.start_lr
            ]
        else:
            # Linear decay
            if self.last_epoch < self.total_steps:
                decay_step = self.last_epoch - self._warmup_steps
                return [
                    self.min_lr
                    + (base_lr * self.peak_lr_factor - self.min_lr)
                    * (1 - decay_step / self.decay_steps)
                    for base_lr in self.start_lr
                ]
            else:
                return [self.min_lr for base_lr in self.start_lr]


@SchedulerRegistry.register("sequential")
class Sequential:
    """Wrapper that composes multiple schedulers sequentially via ``SequentialLR``.

    Uses ``__new__`` to return a real ``torch.optim.lr_scheduler.SequentialLR``
    instance, so all standard scheduler methods (``step``, ``state_dict``, etc.)
    work without delegation.
    """

    _PHASE_LENGTH_PARAM: dict[str, str] = {
        "cosine_annealing": "total_steps",
        "cosine_annealing_cyclic": "total_steps",
        "warmup_cosine_annealing": "total_steps",
        "warmup_linear_decay": "total_steps",
        "one_cycle": "total_steps",
        "linear": "total_steps",
        "constant": "total_steps",
    }

    def __new__(
        cls,
        optimizer: Optimizer,
        *,
        schedulers: dict[str, dict],
        total_steps: int,
    ) -> lr_scheduler.SequentialLR:
        if "chemprop" in schedulers:
            msg = (
                "The 'chemprop' scheduler cannot be used as a phase in a "
                "sequential schedule because it is a no-op config holder that "
                "manages its own schedule internally."
            )
            raise ValueError(msg)

        n_phases = len(schedulers)
        phase_length = total_steps // n_phases
        milestones = [phase_length * (i + 1) for i in range(n_phases - 1)]

        sub_schedulers = []
        for name, kwargs in schedulers.items():
            kwargs = dict(kwargs)
            if name in cls._PHASE_LENGTH_PARAM:
                kwargs[cls._PHASE_LENGTH_PARAM[name]] = phase_length
            sub_schedulers.append(SchedulerRegistry[name](optimizer, **kwargs))

        return lr_scheduler.SequentialLR(
            optimizer, schedulers=sub_schedulers, milestones=milestones
        )
