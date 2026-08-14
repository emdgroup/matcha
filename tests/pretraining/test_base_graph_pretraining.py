"""Tests for constant loss weighting in BaseGraphPretrainingModel."""

from unittest.mock import MagicMock

import pytest
import torch
from torch import nn
from torch_geometric.data import Batch, Data

from matcha.nn.layers import MultiMLP
from matcha.torch.models.pretraining.base_graph_pretraining import (
    BaseGraphPretrainingModel,
)


@pytest.fixture()
def mock_model():
    """Create a mock BaseGraphPretrainingModel with controlled forward() output.

    Uses MagicMock to avoid instantiating a full concrete subclass with encoder,
    while preserving the real training_step/validation_step logic.
    """
    model = MagicMock(spec=BaseGraphPretrainingModel)
    # Bind the real methods to the mock so we test actual logic
    model.training_step = BaseGraphPretrainingModel.training_step.__get__(model)
    model.validation_step = BaseGraphPretrainingModel.validation_step.__get__(model)
    model._log_per_task_losses = BaseGraphPretrainingModel._log_per_task_losses.__get__(
        model
    )
    model.node_loss_fn = MagicMock(return_value=torch.tensor(2.0))
    model.loss_fn = MagicMock(return_value=torch.tensor(3.0))

    # Controlled forward output
    model.forward = MagicMock(
        return_value={"node": torch.randn(5, 1), "graph": torch.randn(2, 1)}
    )

    # Default hparams with 0.5/0.5 weights
    hparams = MagicMock()
    hparams.node_loss_weight = 0.5
    hparams.graph_loss_weight = 0.5
    model.hparams = hparams

    # Track self.log() and self.log_dict() calls
    logged = {}

    def fake_log(name, value, **kwargs):
        logged[name] = value.item() if isinstance(value, torch.Tensor) else value

    def fake_log_dict(metrics, **kwargs):
        for name, value in metrics.items():
            logged[name] = value.item() if isinstance(value, torch.Tensor) else value

    model.log = MagicMock(side_effect=fake_log)
    model.log_dict = MagicMock(side_effect=fake_log_dict)
    model._logged = logged
    model._per_task_log_every_n_steps = 1
    model._per_task_log_step_counter = 0
    return model


def _make_batch(num_nodes: int = 5, batch_size: int = 2):
    """Create a minimal batch dict matching training_step expectations."""
    graphs = [Data(x=torch.randn(3, 4)) for _ in range(batch_size)]
    graphs[0].x = torch.randn(3, 4)
    graphs[1].x = torch.randn(2, 4)
    pyg_batch = Batch.from_data_list(graphs)
    return {
        "graph": pyg_batch,
        "y_node": torch.randn(num_nodes, 1),
        "y_graph": torch.randn(batch_size, 1),
    }


class TestTrainingStepConstantWeights:
    """Verify constant weights are applied in training_step."""

    def test_default_weights(self, mock_model) -> None:
        batch = _make_batch()

        result = mock_model.training_step(batch, 0)

        # Default: 0.5 * 2.0 + 0.5 * 3.0 = 2.5
        assert result.item() == pytest.approx(2.5)

    def test_custom_weights(self, mock_model) -> None:
        mock_model.hparams.node_loss_weight = 0.7
        mock_model.hparams.graph_loss_weight = 0.3
        batch = _make_batch()

        result = mock_model.training_step(batch, 0)

        # 0.7 * 2.0 + 0.3 * 3.0 = 2.3
        assert result.item() == pytest.approx(2.3)

    def test_logs_train_loss(self, mock_model) -> None:
        batch = _make_batch()

        mock_model.training_step(batch, 0)

        assert mock_model._logged["train_loss"] == pytest.approx(2.5)

    def test_logs_component_losses(self, mock_model) -> None:
        batch = _make_batch()

        mock_model.training_step(batch, 0)

        assert mock_model._logged["train_node_loss"] == pytest.approx(2.0)
        assert mock_model._logged["train_graph_loss"] == pytest.approx(3.0)

    def test_no_curriculum_weight_logging(self, mock_model) -> None:
        batch = _make_batch()

        mock_model.training_step(batch, 0)

        assert "node_loss_weight" not in mock_model._logged
        assert "graph_loss_weight" not in mock_model._logged


class TestValidationStepConstantWeights:
    """Verify validation_step logs both weighted and unweighted losses."""

    def test_unweighted_total_loss(self, mock_model) -> None:
        batch = _make_batch()

        result = mock_model.validation_step(batch, 0)

        # Unweighted: node_loss + graph_loss = 2.0 + 3.0 = 5.0
        assert result.item() == pytest.approx(5.0)

    def test_weighted_loss_logged(self, mock_model) -> None:
        batch = _make_batch()

        mock_model.validation_step(batch, 0)

        # Weighted: 0.5 * 2.0 + 0.5 * 3.0 = 2.5
        assert mock_model._logged["val_loss_weighted"] == pytest.approx(2.5)

    def test_weighted_loss_with_custom_weights(self, mock_model) -> None:
        mock_model.hparams.node_loss_weight = 0.8
        mock_model.hparams.graph_loss_weight = 0.2
        batch = _make_batch()

        mock_model.validation_step(batch, 0)

        # 0.8 * 2.0 + 0.2 * 3.0 = 2.2
        assert mock_model._logged["val_loss_weighted"] == pytest.approx(2.2)

    def test_logs_val_loss(self, mock_model) -> None:
        batch = _make_batch()

        mock_model.validation_step(batch, 0)

        assert mock_model._logged["val_loss"] == pytest.approx(5.0)

    def test_logs_component_losses(self, mock_model) -> None:
        batch = _make_batch()

        mock_model.validation_step(batch, 0)

        assert mock_model._logged["val_node_loss"] == pytest.approx(2.0)
        assert mock_model._logged["val_graph_loss"] == pytest.approx(3.0)


class TestPerTaskLossLogging:
    """Verify per-task losses are logged when loss functions expose _per_task_losses."""

    @pytest.fixture()
    def mock_model_with_per_task(self, mock_model):
        """Attach _per_task_losses to the mock loss functions."""
        mock_model.node_loss_fn._per_task_losses = torch.tensor([1.0, 2.0, 3.0])
        mock_model.loss_fn._per_task_losses = torch.tensor([0.5, 1.5])
        return mock_model

    def test_training_step_logs_per_task_node_losses(
        self, mock_model_with_per_task
    ) -> None:
        batch = _make_batch()

        mock_model_with_per_task.training_step(batch, 0)

        assert mock_model_with_per_task._logged[
            "train_node_loss_col_0"
        ] == pytest.approx(1.0)
        assert mock_model_with_per_task._logged[
            "train_node_loss_col_1"
        ] == pytest.approx(2.0)
        assert mock_model_with_per_task._logged[
            "train_node_loss_col_2"
        ] == pytest.approx(3.0)

    def test_training_step_logs_per_task_graph_losses(
        self, mock_model_with_per_task
    ) -> None:
        batch = _make_batch()

        mock_model_with_per_task.training_step(batch, 0)

        assert mock_model_with_per_task._logged[
            "train_graph_loss_col_0"
        ] == pytest.approx(0.5)
        assert mock_model_with_per_task._logged[
            "train_graph_loss_col_1"
        ] == pytest.approx(1.5)

    def test_validation_step_logs_per_task_node_losses(
        self, mock_model_with_per_task
    ) -> None:
        batch = _make_batch()

        mock_model_with_per_task.validation_step(batch, 0)

        assert mock_model_with_per_task._logged["val_node_loss_col_0"] == pytest.approx(
            1.0
        )
        assert mock_model_with_per_task._logged["val_node_loss_col_1"] == pytest.approx(
            2.0
        )
        assert mock_model_with_per_task._logged["val_node_loss_col_2"] == pytest.approx(
            3.0
        )

    def test_validation_step_logs_per_task_graph_losses(
        self, mock_model_with_per_task
    ) -> None:
        batch = _make_batch()

        mock_model_with_per_task.validation_step(batch, 0)

        assert mock_model_with_per_task._logged[
            "val_graph_loss_col_0"
        ] == pytest.approx(0.5)
        assert mock_model_with_per_task._logged[
            "val_graph_loss_col_1"
        ] == pytest.approx(1.5)

    def test_aggregated_losses_unaffected(self, mock_model_with_per_task) -> None:
        batch = _make_batch()

        mock_model_with_per_task.training_step(batch, 0)

        assert mock_model_with_per_task._logged["train_loss"] == pytest.approx(2.5)
        assert mock_model_with_per_task._logged["train_node_loss"] == pytest.approx(2.0)
        assert mock_model_with_per_task._logged["train_graph_loss"] == pytest.approx(
            3.0
        )

    def test_no_per_task_attribute_no_log_dict(self, mock_model) -> None:
        """When loss functions lack _per_task_losses, log_dict should not be called."""
        # Configure mocks to not have _per_task_losses via spec
        mock_model.node_loss_fn = MagicMock(spec=["__call__"])
        mock_model.node_loss_fn.return_value = torch.tensor(2.0)
        mock_model.loss_fn = MagicMock(spec=["__call__"])
        mock_model.loss_fn.return_value = torch.tensor(3.0)
        batch = _make_batch()

        mock_model.training_step(batch, 0)

        mock_model.log_dict.assert_not_called()


class TestPerTaskLoggingGradNormLoss:
    """Verify per-task logging works when graph loss is GradNormLoss."""

    @pytest.fixture()
    def mock_model_gradnorm(self, mock_model):
        """Attach _per_task_losses to loss_fn simulating GradNormLoss behavior."""
        mock_model.node_loss_fn._per_task_losses = torch.tensor([0.8, 1.2, 0.6])
        mock_model.loss_fn._per_task_losses = torch.tensor([0.3, 0.7, 1.1, 0.9])
        return mock_model

    def test_training_step_logs_per_task_graph_losses_gradnorm(
        self, mock_model_gradnorm
    ) -> None:
        """GradNormLoss with 4 endpoints should log 4 per-task graph metrics."""
        batch = _make_batch()
        mock_model_gradnorm.training_step(batch, 0)

        assert mock_model_gradnorm._logged["train_graph_loss_col_0"] == pytest.approx(
            0.3
        )
        assert mock_model_gradnorm._logged["train_graph_loss_col_1"] == pytest.approx(
            0.7
        )
        assert mock_model_gradnorm._logged["train_graph_loss_col_2"] == pytest.approx(
            1.1
        )
        assert mock_model_gradnorm._logged["train_graph_loss_col_3"] == pytest.approx(
            0.9
        )

    def test_validation_step_logs_per_task_graph_losses_gradnorm(
        self, mock_model_gradnorm
    ) -> None:
        """GradNormLoss per-task logging should also work in validation."""
        batch = _make_batch()
        mock_model_gradnorm.validation_step(batch, 0)

        assert mock_model_gradnorm._logged["val_graph_loss_col_0"] == pytest.approx(0.3)
        assert mock_model_gradnorm._logged["val_graph_loss_col_1"] == pytest.approx(0.7)
        assert mock_model_gradnorm._logged["val_graph_loss_col_2"] == pytest.approx(1.1)
        assert mock_model_gradnorm._logged["val_graph_loss_col_3"] == pytest.approx(0.9)

    def test_training_step_logs_per_task_node_losses_with_gradnorm_graph(
        self, mock_model_gradnorm
    ) -> None:
        """Node per-task logging should still work alongside GradNormLoss graph loss."""
        batch = _make_batch()
        mock_model_gradnorm.training_step(batch, 0)

        assert mock_model_gradnorm._logged["train_node_loss_col_0"] == pytest.approx(
            0.8
        )
        assert mock_model_gradnorm._logged["train_node_loss_col_1"] == pytest.approx(
            1.2
        )
        assert mock_model_gradnorm._logged["train_node_loss_col_2"] == pytest.approx(
            0.6
        )


class TestPerTaskLogEveryNSteps:
    """Verify per_task_log_every_n_steps gates logging frequency."""

    @pytest.fixture()
    def mock_model_every_2(self, mock_model):
        """Model that only logs per-task metrics every 2 steps."""
        mock_model._per_task_log_every_n_steps = 2
        mock_model._per_task_log_step_counter = 0
        mock_model.node_loss_fn._per_task_losses = torch.tensor([1.0, 2.0])
        mock_model.loss_fn._per_task_losses = torch.tensor([0.5])
        return mock_model

    def test_skips_on_non_matching_step(self, mock_model_every_2) -> None:
        """Per-task metrics should not be logged on step 1 (counter % 2 != 0)."""
        batch = _make_batch()
        mock_model_every_2.training_step(batch, 0)

        assert "train_node_loss_col_0" not in mock_model_every_2._logged

    def test_logs_on_matching_step(self, mock_model_every_2) -> None:
        """Per-task metrics should be logged on step 2 (counter % 2 == 0)."""
        batch = _make_batch()
        mock_model_every_2.training_step(batch, 0)  # step 1: skip
        mock_model_every_2.training_step(batch, 1)  # step 2: log

        assert mock_model_every_2._logged["train_node_loss_col_0"] == pytest.approx(1.0)
        assert mock_model_every_2._logged["train_node_loss_col_1"] == pytest.approx(2.0)
        assert mock_model_every_2._logged["train_graph_loss_col_0"] == pytest.approx(
            0.5
        )


class TestBuildPredictionHeadTaskHeadDims:
    """Verify _build_prediction_head supports per-task branches via task_head_dims."""

    @pytest.fixture()
    def model_instance(self):
        """Create a minimal BaseGraphPretrainingModel instance for testing head builder.

        We bind only the _build_prediction_head method to a mock to avoid needing
        a full concrete subclass.
        """
        model = MagicMock(spec=BaseGraphPretrainingModel)
        model._build_prediction_head = (
            BaseGraphPretrainingModel._build_prediction_head.__get__(model)
        )
        return model

    def test_shared_only_returns_sequential_with_linear(self, model_instance) -> None:
        """Without task_head_dims, output layer should be nn.Linear."""
        head = model_instance._build_prediction_head(
            input_dim=128,
            hidden_dims=[64],
            num_targets=4,
            dropout=0.1,
            activation="swish",
            task_head_dims=None,
        )
        assert isinstance(head, nn.Sequential)
        assert isinstance(head[-1], nn.Linear)
        assert head[-1].out_features == 4

    def test_task_head_dims_returns_sequential_with_multimlp(
        self, model_instance
    ) -> None:
        """With task_head_dims, output layer should be a MultiMLP."""
        head = model_instance._build_prediction_head(
            input_dim=128,
            hidden_dims=[64],
            num_targets=4,
            dropout=0.1,
            activation="swish",
            task_head_dims=[32],
        )
        assert isinstance(head, nn.Sequential)
        assert isinstance(head[-1], MultiMLP)

    def test_task_head_dims_without_shared_layers(self, model_instance) -> None:
        """task_head_dims should work even when hidden_dims is None."""
        head = model_instance._build_prediction_head(
            input_dim=128,
            hidden_dims=None,
            num_targets=4,
            dropout=0.1,
            activation="swish",
            task_head_dims=[32],
        )
        assert isinstance(head, nn.Sequential)
        assert len(head) == 1
        assert isinstance(head[0], MultiMLP)

    def test_output_shape_with_task_heads(self, model_instance) -> None:
        """Forward pass should produce (batch, num_targets) output."""
        head = model_instance._build_prediction_head(
            input_dim=128,
            hidden_dims=[64],
            num_targets=4,
            dropout=0.1,
            activation="swish",
            task_head_dims=[32],
        )
        head.eval()
        x = torch.randn(16, 128)
        out = head(x)
        assert out.shape == (16, 4)

    def test_output_shape_without_task_heads(self, model_instance) -> None:
        """Forward pass without task_head_dims should also produce correct shape."""
        head = model_instance._build_prediction_head(
            input_dim=128,
            hidden_dims=[64],
            num_targets=4,
            dropout=0.1,
            activation="swish",
            task_head_dims=None,
        )
        head.eval()
        x = torch.randn(16, 128)
        out = head(x)
        assert out.shape == (16, 4)

    def test_backward_compatibility_default_none(self, model_instance) -> None:
        """Calling without task_head_dims kwarg should default to None (old behavior)."""
        head = model_instance._build_prediction_head(
            input_dim=128,
            hidden_dims=[64],
            num_targets=4,
            dropout=0.1,
            activation="swish",
        )
        assert isinstance(head[-1], nn.Linear)


class TestParseTrainConfigRejectsMultiloss:
    """Regression for issue #41: graph pretraining must reject loss_fn='multiloss'
    at construction time so MultiLoss's (loss, log) tuple never reaches the
    two-head training/validation flow.
    """

    def test_parse_train_config_rejects_multiloss(self) -> None:
        model = MagicMock(spec=BaseGraphPretrainingModel)
        model._parse_train_config = (
            BaseGraphPretrainingModel._parse_train_config.__get__(model)
        )
        model.hparams = {"loss_fn": "multiloss", "loss_args": {"loss_configs": []}}

        with pytest.raises(ValueError, match="multiloss"):
            model._parse_train_config()
