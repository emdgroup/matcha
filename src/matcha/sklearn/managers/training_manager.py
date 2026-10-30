import math
import warnings

import torch
import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)

from matcha.nn.schedulers import SchedulerRegistry
from matcha.utils.logging import get_default_logger
from matcha.utils.schemas.sklearn_api import TrainingInputModel

# Schedulers that do NOT accept or use the `total_steps` parameter.
_SCHEDULERS_WITHOUT_TOTAL_STEPS = frozenset({"chemprop"})


class TrainingManager:
    """Manages Lightning Trainer creation, callbacks, and training loop."""

    def __init__(self):
        self._trainer: L.Trainer | None = None
        self._params: TrainingInputModel | None = None
        self._is_fitted: bool = False
        self.logger = get_default_logger("TRAINING")

    @property
    def trainer(self) -> L.Trainer | None:
        """The underlying Lightning Trainer, available after training."""
        return self._trainer

    @property
    def params(self) -> TrainingInputModel | None:
        """The training parameters owned by this manager."""
        return self._params

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been successfully trained via run()."""
        return self._is_fitted

    def configure(self, train_dict: dict) -> None:
        """Set training parameters. Called once during init.

        :param dict train_dict: dictionary of training parameters
        """
        self._params = TrainingInputModel(**train_dict)

    def build_callbacks(self, training_params: TrainingInputModel) -> list:
        """Create training callbacks based on training parameters.

        :param TrainingInputModel training_params: training configuration
        :return list: list of Lightning callbacks
        """
        callbacks = []
        callbacks.append(LearningRateMonitor(logging_interval="step"))

        if training_params.early_stopping:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    mode="min",
                    patience=int(training_params.patience),
                )
            )
            callbacks.append(
                ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min")
            )

        if training_params.stochastic_weight_averaging:
            callbacks.append(
                StochasticWeightAveraging(swa_lrs=1e-4, swa_epoch_start=0.5)
            )

        return callbacks

    def run(
        self,
        model,
        datamodule,
        architecture_cls,
        mlflow_manager=None,
        model_instance=None,
    ):
        """Execute the training loop.

        :param model: the Lightning model to train
        :param datamodule: the Lightning datamodule
        :param architecture_cls: the model class (for checkpoint loading)
        :param mlflow_manager: optional MLFlowManager for logging
        :param model_instance: the sklearn model instance (for mlflow logging)
        :return: the trained model (potentially loaded from best checkpoint)
        """
        training_params = self._params
        callbacks = self.build_callbacks(training_params)

        # Configure logger
        if mlflow_manager is not None and mlflow_manager.is_active:
            self.logger.info("Fit: setting up MLFlow")
            mlflow_logger = mlflow_manager.create_logger()
            logger = mlflow_logger
        else:
            mlflow_logger = None
            logger = True

        self._trainer = L.Trainer(
            max_epochs=training_params.num_epochs,
            devices=training_params.devices,
            accelerator=training_params.accelerator,
            callbacks=callbacks,
            enable_checkpointing=True,
            deterministic=True,
            logger=logger,
            num_sanity_val_steps=0,
        )

        model.set_label_names(list(datamodule._label_encoder.label_names))

        # Auto-compute total_steps when not explicitly provided
        self._maybe_inject_total_steps(model, datamodule, training_params)

        self.logger.info("Fit: starting model training")
        self._trainer.fit(model=model, datamodule=datamodule)

        # Load best checkpoint weights if early stopping was used
        trained_model = model
        if training_params.early_stopping:
            best_ckpt_path = self._trainer.checkpoint_callback.best_model_path
            if best_ckpt_path:
                try:
                    trained_model = architecture_cls.load_from_checkpoint(
                        best_ckpt_path
                    )
                except (TypeError, Exception):
                    self.logger.warning(
                        "load_from_checkpoint failed; "
                        "falling back to manual state_dict loading."
                    )
                    ckpt = torch.load(best_ckpt_path, weights_only=False)
                    state_dict = ckpt.get("state_dict", ckpt)
                    model.load_state_dict(state_dict, strict=False)
                    trained_model = model

        # MLflow logging
        if mlflow_logger is not None and model_instance is not None:
            self._mlflow_log(model_instance, mlflow_manager, mlflow_logger)

        self._is_fitted = True
        self.logger.info("Fit: model trained")
        return trained_model

    def _maybe_inject_total_steps(self, model, datamodule, training_params):
        """Auto-compute and inject total_steps into the scheduler when not explicitly set.

        If the model's scheduler accepts ``total_steps`` and it is not already present
        in ``scheduler_args``, computes it as::

            total_steps = num_epochs * ceil(len(train_dataset) / batch_size)

        Then updates ``scheduler_args`` in the model's hparams and recreates the scheduler.

        :param model: the Lightning model whose scheduler may be recreated
        :param datamodule: the Lightning datamodule (must have dataset_train populated)
        :param training_params: the TrainingInputModel with num_epochs and batch_size
        """
        scheduler_name = model.hparams.get("scheduler")
        scheduler_args = model.hparams.get("scheduler_args", {})

        if scheduler_name in _SCHEDULERS_WITHOUT_TOTAL_STEPS:
            return

        if "total_steps" in scheduler_args:
            return

        train_size = len(datamodule.dataset_train)
        batch_size = training_params.batch_size
        num_epochs = training_params.num_epochs
        total_steps = num_epochs * math.ceil(train_size / batch_size)

        # Update hparams in-place and recreate the scheduler(s)
        scheduler_args["total_steps"] = total_steps
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            if hasattr(model, "predictor_optimizer"):
                # Finetuner uses separate optimizers/schedulers
                model.predictor_scheduler = SchedulerRegistry[scheduler_name](
                    model.predictor_optimizer, **scheduler_args
                )
            else:
                model.scheduler = SchedulerRegistry[scheduler_name](
                    model.optimizer, **scheduler_args
                )

        self.logger.debug(
            "Auto-computed total_steps=%d "
            "(num_epochs=%d, train_size=%d, batch_size=%d)",
            total_steps,
            num_epochs,
            train_size,
            batch_size,
        )

    def _mlflow_log(self, model_instance, mlflow_manager, mlflow_logger):
        """Default MLflow logging implementation. Override in subclasses for
        custom logging behavior (e.g. CLM, Finetuner).

        :param model_instance: the sklearn model instance
        :param mlflow_manager: the MLFlowManager
        :param mlflow_logger: the active MatchaLogger
        """
        mlflow_manager.log_training(model_instance, mlflow_logger)

    def save_checkpoint(self, path: str):
        """Delegate to trainer.save_checkpoint.

        :param str path: file path for the checkpoint
        """
        if self._trainer is None:
            raise RuntimeError("No trainer available. Call run() first.")
        self._trainer.save_checkpoint(path)


class CLMTrainingManager(TrainingManager):
    """Training manager specialized for CLM (Chemical Language Model) training.

    Overrides the training loop to recreate the model with the correct num_tokens
    parameter before training, and customizes MLflow logging to exclude
    the vocabulary dictionary.
    """

    def run(
        self,
        model,
        datamodule,
        architecture_cls,
        mlflow_manager=None,
        model_instance=None,
    ):
        """Execute CLM training loop.

        Recreates the model with the correct num_tokens parameter discovered
        during featurization, then delegates to the parent training loop.

        :param model: the Lightning model to train
        :param datamodule: the Lightning datamodule
        :param architecture_cls: the model class (for checkpoint loading)
        :param mlflow_manager: optional MLFlowManager for logging
        :param model_instance: the sklearn model instance (for mlflow logging and model recreation)
        :return: the trained model
        """
        # Recreate model with the correct num_tokens parameter
        if model_instance is not None:
            model_dict = model_instance._model.params.model_dump()
            model_instance._create_model(model_dict)
            model = model_instance._model

        return super().run(
            model, datamodule, architecture_cls, mlflow_manager, model_instance
        )

    def _mlflow_log(self, model_instance, mlflow_manager, mlflow_logger):
        """Override MLflow logging to exclude the vocabulary dictionary.

        :param model_instance: the sklearn model instance
        :param mlflow_manager: the MLFlowManager
        :param mlflow_logger: the active MatchaLogger
        """
        import shutil

        mlflow_logger.store = True
        params_to_log = model_instance.params.model_dump().copy()
        params_to_log["datamodule"].pop("dictionary", None)
        mlflow_logger.log_hyperparams(params_to_log)
        model_instance.save_model("./.matcha_temp")
        mlflow_logger.log_matcha_artifacts("./.matcha_temp")
        mlflow_logger.conclude_experiment()
        shutil.rmtree("./.matcha_temp")


class FinetunerTrainingManager(TrainingManager):
    """Training manager specialized for fine-tuning models.

    Customizes MLflow logging to exclude the vocabulary dictionary
    when present in the datamodule params.
    """

    def _mlflow_log(self, model_instance, mlflow_manager, mlflow_logger):
        """Override MLflow logging to exclude the vocabulary dictionary if present.

        :param model_instance: the sklearn model instance
        :param mlflow_manager: the MLFlowManager
        :param mlflow_logger: the active MatchaLogger
        """
        import shutil

        mlflow_logger.store = True
        params_to_log = model_instance.params.model_dump().copy()

        # Pop dictionary if present
        if "dictionary" in params_to_log["datamodule"]:
            params_to_log["datamodule"].pop("dictionary")

        mlflow_logger.log_hyperparams(params_to_log)
        model_instance.save_model("./.matcha_temp")
        mlflow_logger.log_matcha_artifacts("./.matcha_temp")
        mlflow_logger.conclude_experiment()
        shutil.rmtree("./.matcha_temp")
