"""Tests for matcha.nn.schedulers – SchedulerRegistry and custom schedulers."""

import pytest
import torch

from matcha.nn.schedulers import (
    CosineAnnealing,
    SchedulerRegistry,
    WarmupCosineAnnealingLR,
    WarmupLinearDecayLR,
    Sequential,
)
from unittest.mock import MagicMock

from matcha.torch.models.classic.base_classic_model import BaseClassicModel
from matcha.torch.models.pretraining.base_pretraining_model import BasePretrainingModel
from matcha.torch.models.finetuning.finetuner import Finetuner


# ===================================================================
# Registry completeness
# ===================================================================


class TestSchedulerRegistryKeys:
    EXPECTED_KEYS = [
        "one_cycle",
        "cosine_annealing",
        "cosine_annealing_cyclic",
        "step",
        "warmup_cosine_annealing",
        "warmup_linear_decay",
        "linear",
        "constant",
        "sequential",
    ]

    @pytest.mark.parametrize("key", EXPECTED_KEYS)
    def test_key_registered(self, key):
        assert key in SchedulerRegistry, f"'{key}' not found in SchedulerRegistry"


# ===================================================================
# Helper
# ===================================================================


@pytest.fixture()
def optimizer():
    """Simple optimizer to attach schedulers to."""
    model = torch.nn.Linear(10, 2)
    return torch.optim.SGD(model.parameters(), lr=0.1)


def _step(optimizer, scheduler):
    """Perform optimizer.step() then scheduler.step() in the correct order."""
    optimizer.step()
    scheduler.step()


# ===================================================================
# WarmupCosineAnnealingLR
# ===================================================================


class TestWarmupCosineAnnealingLR:
    def test_lr_at_start(self, optimizer):
        sched = WarmupCosineAnnealingLR(
            optimizer,
            total_steps=100,
            warmup_ratio=0.1,
            peak_lr_factor=10.0,
            min_lr=0.0,
        )
        # At epoch 0, LR should equal start_lr
        lrs = sched.get_lr()
        assert len(lrs) == 1
        assert abs(lrs[0] - 0.1) < 1e-6

    def test_lr_increases_during_warmup(self, optimizer):
        sched = WarmupCosineAnnealingLR(
            optimizer,
            total_steps=100,
            warmup_ratio=0.1,
            peak_lr_factor=10.0,
            min_lr=0.0,
        )
        lrs = []
        for _ in range(10):
            lrs.append(sched.get_last_lr()[0])
            _step(optimizer, sched)
        # LR should be monotonically increasing during warmup
        for i in range(1, len(lrs)):
            assert lrs[i] >= lrs[i - 1]

    def test_lr_decreases_after_warmup(self, optimizer):
        sched = WarmupCosineAnnealingLR(
            optimizer,
            total_steps=100,
            warmup_ratio=0.1,
            peak_lr_factor=10.0,
            min_lr=0.0,
        )
        # Skip through warmup
        for _ in range(10):
            _step(optimizer, sched)
        peak_lr = sched.get_last_lr()[0]
        # A few more steps should decrease
        for _ in range(10):
            _step(optimizer, sched)
        assert sched.get_last_lr()[0] < peak_lr

    def test_lr_reaches_min_lr(self, optimizer):
        sched = WarmupCosineAnnealingLR(
            optimizer,
            total_steps=50,
            warmup_ratio=0.1,
            peak_lr_factor=10.0,
            min_lr=0.001,
        )
        for _ in range(100):
            _step(optimizer, sched)
        assert abs(sched.get_last_lr()[0] - 0.001) < 1e-6

    def test_warmup_ratio(self, optimizer):
        """warmup_ratio should be interpreted as fraction of total_steps."""
        sched = WarmupCosineAnnealingLR(
            optimizer, total_steps=100, warmup_ratio=0.2, peak_lr_factor=5.0
        )
        assert sched._warmup_steps == 20

    def test_default_min_lr(self, optimizer):
        """Default min_lr should be 1e-5."""
        sched = WarmupCosineAnnealingLR(
            optimizer, total_steps=100, warmup_ratio=0.1, peak_lr_factor=5.0
        )
        assert sched.min_lr == 1e-5


# ===================================================================
# WarmupLinearDecayLR
# ===================================================================


class TestWarmupLinearDecayLR:
    def test_lr_at_start(self, optimizer):
        sched = WarmupLinearDecayLR(
            optimizer,
            total_steps=100,
            warmup_ratio=0.1,
            peak_lr_factor=10.0,
            min_lr=0.0,
        )
        lrs = sched.get_lr()
        assert abs(lrs[0] - 0.1) < 1e-6

    def test_lr_increases_during_warmup(self, optimizer):
        sched = WarmupLinearDecayLR(
            optimizer,
            total_steps=100,
            warmup_ratio=0.1,
            peak_lr_factor=10.0,
            min_lr=0.0,
        )
        lrs = []
        for _ in range(10):
            lrs.append(sched.get_last_lr()[0])
            _step(optimizer, sched)
        for i in range(1, len(lrs)):
            assert lrs[i] >= lrs[i - 1]

    def test_lr_decreases_linearly_after_warmup(self, optimizer):
        sched = WarmupLinearDecayLR(
            optimizer,
            total_steps=100,
            warmup_ratio=0.1,
            peak_lr_factor=10.0,
            min_lr=0.0,
        )
        for _ in range(10):
            _step(optimizer, sched)
        lrs_after = []
        for _ in range(20):
            lrs_after.append(sched.get_last_lr()[0])
            _step(optimizer, sched)
        # Should be monotonically decreasing
        for i in range(1, len(lrs_after)):
            assert lrs_after[i] <= lrs_after[i - 1] + 1e-8

    def test_lr_reaches_min_lr(self, optimizer):
        sched = WarmupLinearDecayLR(
            optimizer,
            total_steps=50,
            warmup_ratio=0.1,
            peak_lr_factor=10.0,
            min_lr=0.001,
        )
        for _ in range(100):
            _step(optimizer, sched)
        assert abs(sched.get_last_lr()[0] - 0.001) < 1e-6

    def test_warmup_ratio(self, optimizer):
        sched = WarmupLinearDecayLR(
            optimizer, total_steps=100, warmup_ratio=0.2, peak_lr_factor=5.0
        )
        assert sched._warmup_steps == 20

    def test_default_min_lr(self, optimizer):
        """Default min_lr should be 1e-5."""
        sched = WarmupLinearDecayLR(
            optimizer, total_steps=100, warmup_ratio=0.1, peak_lr_factor=5.0
        )
        assert sched.min_lr == 1e-5


# ===================================================================
# Built-in scheduler aliases – verify correct parent class
# ===================================================================


class TestBuiltinSchedulerAliases:
    @pytest.mark.parametrize(
        "key,expected_parent",
        [
            ("one_cycle", torch.optim.lr_scheduler.OneCycleLR),
            ("cosine_annealing", torch.optim.lr_scheduler.CosineAnnealingLR),
            (
                "cosine_annealing_cyclic",
                torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
            ),
            ("step", torch.optim.lr_scheduler.StepLR),
        ],
    )
    def test_is_subclass(self, key, expected_parent):
        sched_cls = SchedulerRegistry[key]
        assert issubclass(sched_cls, expected_parent)


# ===================================================================
# CosineAnnealing
# ===================================================================


class TestCosineAnnealing:
    def test_default_min_lr(self, optimizer):
        """Default min_lr should be 1e-5."""
        sched = CosineAnnealing(optimizer, total_steps=100)
        assert sched.eta_min == 1e-5

    def test_custom_min_lr(self, optimizer):
        """User-supplied min_lr is forwarded as eta_min."""
        sched = CosineAnnealing(optimizer, total_steps=100, min_lr=1e-7)
        assert sched.eta_min == 1e-7

    def test_total_steps_maps_to_t_max(self, optimizer):
        """total_steps should be forwarded as T_max."""
        sched = CosineAnnealing(optimizer, total_steps=200, min_lr=0.0)
        assert sched.T_max == 200

    def test_no_silent_override(self, optimizer):
        """min_lr=0 should not be silently replaced with a non-zero default."""
        sched = CosineAnnealing(optimizer, total_steps=50, min_lr=0.0)
        assert sched.eta_min == 0.0

    def test_stepping_reaches_min_lr(self, optimizer):
        """After T_max steps, LR should reach min_lr."""
        sched = CosineAnnealing(optimizer, total_steps=50, min_lr=1e-6)
        for _ in range(50):
            _step(optimizer, sched)
        assert abs(optimizer.param_groups[0]["lr"] - 1e-6) < 1e-8


# ===================================================================
# CosineAnnealingCyclic
# ===================================================================


class TestCosineAnnealingCyclic:
    def test_t0_from_total_steps_and_num_cycles(self, optimizer):
        sched = SchedulerRegistry["cosine_annealing_cyclic"](
            optimizer, total_steps=100, num_cycles=5
        )
        assert sched.T_0 == 20

    def test_default_num_cycles(self, optimizer):
        sched = SchedulerRegistry["cosine_annealing_cyclic"](optimizer, total_steps=100)
        # default num_cycles=5 → T_0=20
        assert sched.T_0 == 20

    def test_min_lr_maps_to_eta_min(self, optimizer):
        sched = SchedulerRegistry["cosine_annealing_cyclic"](
            optimizer, total_steps=100, num_cycles=4, min_lr=1e-5
        )
        assert sched.eta_min == 1e-5

    def test_stepping_runs_without_error(self, optimizer):
        sched = SchedulerRegistry["cosine_annealing_cyclic"](
            optimizer, total_steps=100, num_cycles=5, min_lr=0.0
        )
        for _ in range(100):
            _step(optimizer, sched)


# ===================================================================
# Step (pops total_steps)
# ===================================================================


class TestStepScheduler:
    def test_ignores_total_steps(self, optimizer):
        """Step scheduler should silently ignore total_steps."""
        sched = SchedulerRegistry["step"](optimizer, step_size=10, total_steps=999)
        assert isinstance(sched, torch.optim.lr_scheduler.StepLR)

    def test_works_without_total_steps(self, optimizer):
        sched = SchedulerRegistry["step"](optimizer, step_size=10)
        for _ in range(20):
            _step(optimizer, sched)


# ===================================================================
# LinearLR
# ===================================================================


class TestLinearLR:
    def test_is_subclass(self):
        assert issubclass(
            SchedulerRegistry["linear"], torch.optim.lr_scheduler.LinearLR
        )

    def test_instantiation_and_step(self, optimizer):
        sched = SchedulerRegistry["linear"](
            optimizer, start_factor=0.1, end_factor=1.0, total_steps=10
        )
        assert isinstance(sched, torch.optim.lr_scheduler.LinearLR)
        for _ in range(10):
            _step(optimizer, sched)


# ===================================================================
# ConstantLR
# ===================================================================


class TestConstantLR:
    def test_is_subclass(self):
        assert issubclass(
            SchedulerRegistry["constant"], torch.optim.lr_scheduler.ConstantLR
        )

    def test_instantiation_and_step(self, optimizer):
        sched = SchedulerRegistry["constant"](optimizer, factor=0.5, total_steps=10)
        assert isinstance(sched, torch.optim.lr_scheduler.ConstantLR)
        for _ in range(10):
            _step(optimizer, sched)


# ===================================================================
# SequentialLR
# ===================================================================


class TestSequentialLR:
    def test_basic_creation(self, optimizer):
        sched = SchedulerRegistry["sequential"](
            optimizer,
            schedulers={
                "linear": {"start_factor": 0.1, "end_factor": 1.0},
                "cosine_annealing": {"min_lr": 1e-7},
            },
            total_steps=200,
        )
        assert isinstance(sched, torch.optim.lr_scheduler.SequentialLR)

    def test_milestones_even_split(self, optimizer):
        sched = SchedulerRegistry["sequential"](
            optimizer,
            schedulers={
                "linear": {"start_factor": 0.1, "end_factor": 1.0},
                "cosine_annealing": {"min_lr": 1e-7},
                "constant": {"factor": 0.1},
            },
            total_steps=300,
        )
        assert sched._milestones == [100, 200]

    def test_auto_injects_total_steps(self, optimizer):
        """cosine_annealing sub-scheduler should get total_steps (T_max) set to phase length."""
        sched = SchedulerRegistry["sequential"](
            optimizer,
            schedulers={
                "linear": {"start_factor": 0.1, "end_factor": 1.0},
                "cosine_annealing": {"min_lr": 1e-7},
            },
            total_steps=200,
        )
        # The cosine_annealing scheduler is the second sub-scheduler
        cosine_sched = sched._schedulers[1]
        assert cosine_sched.T_max == 100

    def test_auto_injects_total_steps_linear_constant(self, optimizer):
        """linear and constant sub-schedulers should get total_steps (total_iters) set to phase length."""
        sched = SchedulerRegistry["sequential"](
            optimizer,
            schedulers={
                "linear": {"start_factor": 0.1, "end_factor": 1.0},
                "constant": {"factor": 0.5},
            },
            total_steps=200,
        )
        assert sched._schedulers[0].total_iters == 100
        assert sched._schedulers[1].total_iters == 100

    def test_auto_overrides_explicit_param(self, optimizer):
        """User-provided phase-length param should be overridden."""
        sched = SchedulerRegistry["sequential"](
            optimizer,
            schedulers={
                "cosine_annealing": {"total_steps": 999, "min_lr": 1e-7},
                "linear": {"start_factor": 0.1, "end_factor": 1.0},
            },
            total_steps=200,
        )
        assert sched._schedulers[0].T_max == 100

    def test_chemprop_raises_value_error(self, optimizer):
        with pytest.raises(ValueError, match="chemprop"):
            SchedulerRegistry["sequential"](
                optimizer,
                schedulers={
                    "chemprop": {"warmup_epochs": 2},
                    "linear": {"start_factor": 0.1, "end_factor": 1.0},
                },
                total_steps=200,
            )

    def test_stepping_runs_without_error(self, optimizer):
        sched = SchedulerRegistry["sequential"](
            optimizer,
            schedulers={
                "linear": {"start_factor": 0.1, "end_factor": 1.0},
                "cosine_annealing": {"min_lr": 1e-7},
            },
            total_steps=200,
        )
        for _ in range(200):
            _step(optimizer, sched)

    def test_lr_progression_through_phases(self, optimizer):
        """LR should change character at phase boundaries."""
        sched = SchedulerRegistry["sequential"](
            optimizer,
            schedulers={
                "linear": {"start_factor": 0.1, "end_factor": 1.0},
                "constant": {"factor": 1.0},
            },
            total_steps=200,
        )
        # During linear phase, LR should increase
        lrs_phase1 = []
        for _ in range(100):
            lrs_phase1.append(optimizer.param_groups[0]["lr"])
            _step(optimizer, sched)
        assert lrs_phase1[-1] > lrs_phase1[0]

        # During constant phase, LR should stay the same
        lrs_phase2 = []
        for _ in range(100):
            lrs_phase2.append(optimizer.param_groups[0]["lr"])
            _step(optimizer, sched)
        for i in range(1, len(lrs_phase2)):
            assert abs(lrs_phase2[i] - lrs_phase2[0]) < 1e-6

    def test_single_scheduler(self, optimizer):
        """Edge case: single scheduler means milestones=[]."""
        sched = SchedulerRegistry["sequential"](
            optimizer,
            schedulers={"linear": {"start_factor": 0.1, "end_factor": 1.0}},
            total_steps=100,
        )
        assert sched._milestones == []
        for _ in range(100):
            _step(optimizer, sched)

    def test_two_schedulers(self, optimizer):
        """Edge case: two schedulers means one milestone."""
        sched = SchedulerRegistry["sequential"](
            optimizer,
            schedulers={
                "linear": {"start_factor": 0.1, "end_factor": 1.0},
                "cosine_annealing": {"min_lr": 0.0},
            },
            total_steps=100,
        )
        assert sched._milestones == [50]
        for _ in range(100):
            _step(optimizer, sched)

    def test_phase_length_param_uses_total_steps_uniformly(self):
        """All entries in _PHASE_LENGTH_PARAM should map to 'total_steps'."""
        for name, param in Sequential._PHASE_LENGTH_PARAM.items():
            assert param == "total_steps", (
                f"Scheduler '{name}' maps to '{param}' instead of 'total_steps'"
            )


# ===================================================================
# Integration: configure_optimizers returns step-level scheduling
# ===================================================================


class TestConfigureOptimizersStepInterval:
    """Verify configure_optimizers returns interval='step' for all model bases."""

    def _make_optimizer_and_scheduler(self):
        model = torch.nn.Linear(10, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = SchedulerRegistry["cosine_annealing"](
            optimizer, total_steps=100, min_lr=1e-6
        )
        return optimizer, scheduler

    def test_base_classic_model(self):
        """BaseClassicModel.configure_optimizers returns step-level scheduling."""
        optimizer, scheduler = self._make_optimizer_and_scheduler()
        # Directly call the unbound method with a mock instance
        instance = MagicMock()
        instance.optimizer = optimizer
        instance.scheduler = scheduler
        result = BaseClassicModel.configure_optimizers(instance)
        assert result["optimizer"] is optimizer
        assert result["lr_scheduler"]["scheduler"] is scheduler
        assert result["lr_scheduler"]["interval"] == "step"
        assert result["lr_scheduler"]["frequency"] == 1

    def test_base_pretraining_model(self):
        """BasePretrainingModel.configure_optimizers returns step-level scheduling."""
        optimizer, scheduler = self._make_optimizer_and_scheduler()
        instance = MagicMock()
        instance.optimizer = optimizer
        instance.scheduler = scheduler
        result = BasePretrainingModel.configure_optimizers(instance)
        assert result["optimizer"] is optimizer
        assert result["lr_scheduler"]["scheduler"] is scheduler
        assert result["lr_scheduler"]["interval"] == "step"
        assert result["lr_scheduler"]["frequency"] == 1

    def test_finetuner_lora_mode(self):
        """Finetuner.configure_optimizers returns step-level scheduling in LoRA mode."""
        optimizer, scheduler = self._make_optimizer_and_scheduler()
        instance = MagicMock()
        instance.predictor_optimizer = optimizer
        instance.predictor_scheduler = scheduler
        instance.pretrain_optimizer = None  # LoRA mode (no pretrain optimizer)
        result = Finetuner.configure_optimizers(instance)
        assert result["optimizer"] is optimizer
        assert result["lr_scheduler"]["scheduler"] is scheduler
        assert result["lr_scheduler"]["interval"] == "step"
        assert result["lr_scheduler"]["frequency"] == 1
