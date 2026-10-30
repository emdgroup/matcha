"""Tests for matcha.torch.tuning.routine — scheduler param tuning."""

import pytest
from optuna import create_study
from optuna.trial import Trial

from matcha.torch.tuning.routine import (
    load_default_scheduler_grid,
    parse_config,
)


# ===================================================================
# load_default_scheduler_grid
# ===================================================================


class TestLoadSchedulerGrid:
    """Tests for load_default_scheduler_grid loading behavior."""

    @pytest.mark.parametrize(
        "scheduler_name",
        [
            "cosine_annealing",
            "cosine_annealing_cyclic",
            "warmup_cosine_annealing",
            "warmup_linear_decay",
            "chemprop",
        ],
    )
    def test_tunable_scheduler_returns_non_empty(self, scheduler_name):
        grid = load_default_scheduler_grid(scheduler_name)
        assert len(grid) > 0

    @pytest.mark.parametrize(
        "scheduler_name",
        ["constant", "step", "linear", "one_cycle"],
    )
    def test_non_tunable_scheduler_returns_empty(self, scheduler_name):
        grid = load_default_scheduler_grid(scheduler_name)
        assert grid == {}

    def test_none_returns_empty(self):
        assert load_default_scheduler_grid(None) == {}

    def test_unknown_scheduler_returns_empty(self):
        assert load_default_scheduler_grid("nonexistent_scheduler") == {}


# ===================================================================
# Scheduler param suggestion via parse_config
# ===================================================================


class TestSchedulerParamSuggestion:
    """Tests that parse_config correctly suggests scheduler params."""

    def _make_trial(self) -> Trial:
        study = create_study()
        return study.ask()

    def test_cosine_annealing_suggests_min_lr(self):
        grid = load_default_scheduler_grid("cosine_annealing")
        trial = self._make_trial()
        result = parse_config(trial, grid)
        assert "min_lr" in result
        assert 1e-7 <= result["min_lr"] <= 1e-4

    def test_cosine_annealing_cyclic_suggests_min_lr_and_num_cycles(self):
        grid = load_default_scheduler_grid("cosine_annealing_cyclic")
        trial = self._make_trial()
        result = parse_config(trial, grid)
        assert "min_lr" in result
        assert "num_cycles" in result
        assert 1e-7 <= result["min_lr"] <= 1e-4
        assert 2 <= result["num_cycles"] <= 10

    def test_warmup_cosine_annealing_suggests_min_lr_and_peak_lr_factor(self):
        grid = load_default_scheduler_grid("warmup_cosine_annealing")
        trial = self._make_trial()
        result = parse_config(trial, grid)
        assert "min_lr" in result
        assert "peak_lr_factor" in result
        assert 1e-7 <= result["min_lr"] <= 1e-4
        assert 2.0 <= result["peak_lr_factor"] <= 20.0

    def test_warmup_linear_decay_suggests_min_lr_and_peak_lr_factor(self):
        grid = load_default_scheduler_grid("warmup_linear_decay")
        trial = self._make_trial()
        result = parse_config(trial, grid)
        assert "min_lr" in result
        assert "peak_lr_factor" in result
        assert 1e-7 <= result["min_lr"] <= 1e-4
        assert 2.0 <= result["peak_lr_factor"] <= 20.0

    def test_chemprop_suggests_max_lr_and_final_lr(self):
        grid = load_default_scheduler_grid("chemprop")
        trial = self._make_trial()
        result = parse_config(trial, grid)
        assert "max_lr" in result
        assert "final_lr" in result
        assert 5e-4 <= result["max_lr"] <= 5e-3
        assert 1e-6 <= result["final_lr"] <= 1e-5

    def test_all_suggested_keys_match_grid_keys(self):
        for scheduler in [
            "cosine_annealing",
            "cosine_annealing_cyclic",
            "warmup_cosine_annealing",
            "warmup_linear_decay",
            "chemprop",
        ]:
            grid = load_default_scheduler_grid(scheduler)
            trial = self._make_trial()
            result = parse_config(trial, grid)
            assert all(k in grid for k in result)


# ===================================================================
# Post-Phase-2 param splitting
# ===================================================================


class TestParamSplitting:
    """Tests that scheduler keys are routed to scheduler_args after Phase 2."""

    def test_scheduler_keys_split_from_optimizer_keys(self):
        """Simulates Phase 2 best_params containing both optimizer and scheduler keys."""
        scheduler_grid = load_default_scheduler_grid("warmup_cosine_annealing")
        scheduler_param_keys = frozenset(scheduler_grid.keys())
        optimum = {
            "lr": 1e-4,
            "eps": 1e-7,
            "min_lr": 5e-6,
            "peak_lr_factor": 8.0,
        }
        sched_params = {k: v for k, v in optimum.items() if k in scheduler_param_keys}
        opt_params = {k: v for k, v in optimum.items() if k not in scheduler_param_keys}

        assert sched_params == {"min_lr": 5e-6, "peak_lr_factor": 8.0}
        assert opt_params == {"lr": 1e-4, "eps": 1e-7}

    def test_no_scheduler_keys_produces_empty_sched_params(self):
        """When no scheduler keys present, sched_params should be empty."""
        scheduler_grid = load_default_scheduler_grid("warmup_cosine_annealing")
        scheduler_param_keys = frozenset(scheduler_grid.keys())
        optimum = {"lr": 1e-4, "betas": (0.9, 0.999)}
        sched_params = {k: v for k, v in optimum.items() if k in scheduler_param_keys}
        assert sched_params == {}

    def test_num_cycles_routed_to_scheduler_args(self):
        """num_cycles should be recognized as a scheduler param."""
        scheduler_grid = load_default_scheduler_grid("cosine_annealing_cyclic")
        scheduler_param_keys = frozenset(scheduler_grid.keys())
        optimum = {"lr": 1e-4, "num_cycles": 3, "eps": 1e-7}
        sched_params = {k: v for k, v in optimum.items() if k in scheduler_param_keys}
        assert "num_cycles" in sched_params
        assert "lr" not in sched_params

    def test_chemprop_max_lr_final_lr_routed_to_scheduler_args(self):
        """max_lr and final_lr should be recognized as scheduler params."""
        scheduler_grid = load_default_scheduler_grid("chemprop")
        scheduler_param_keys = frozenset(scheduler_grid.keys())
        optimum = {"lr": 1e-4, "max_lr": 1e-3, "final_lr": 1e-5}
        sched_params = {k: v for k, v in optimum.items() if k in scheduler_param_keys}
        opt_params = {k: v for k, v in optimum.items() if k not in scheduler_param_keys}
        assert sched_params == {"max_lr": 1e-3, "final_lr": 1e-5}
        assert opt_params == {"lr": 1e-4}
