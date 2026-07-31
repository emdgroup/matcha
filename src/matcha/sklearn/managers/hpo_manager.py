from torch.utils.data import StackDataset

from matcha.datamodules import CLMDataModule, CombinedDataModule
from matcha.torch.tuning.routine import (
    run_hparam_tuning,
    load_default_architecture_grid,
    load_default_optimizer_grid,
    load_default_scheduler_grid,
)
from matcha.utils.logging import get_default_logger
from matcha.utils.schemas.sklearn_api import TuningInputModel


def _patch_clm_init_params(init_params: dict, datamodule) -> None:
    """Patch enc_num_characters with the fitted vocabulary size before HPO.

    Mirrors BaseScikitLearnCLM._create_model() so that every HPO trial
    model is built with the real vocabulary size rather than the placeholder
    value of 4 assigned during CLM model __init__. No-op for non-CLM models.
    """
    if isinstance(datamodule, CLMDataModule):
        init_params["enc_num_characters"] = datamodule.params.num_tokens
    elif isinstance(datamodule, CombinedDataModule):
        clm_datamodules = [
            f for f in datamodule.datamodules if isinstance(f, CLMDataModule)
        ]
        if len(clm_datamodules) > 1:
            raise ValueError("Multiple CLMDataModules found in the CombinedDataModule")
        elif len(clm_datamodules) == 1:
            init_params["enc_num_characters"] = clm_datamodules[0].params.num_tokens


class HPOManager:
    """Manages hyperparameter optimization via Optuna."""

    def __init__(self):
        self._params: TuningInputModel | None = None
        self.logger = get_default_logger("TUNING")

    @property
    def params(self) -> TuningInputModel | None:
        """The tuning parameters, populated after tune() is called."""
        return self._params

    def tune(
        self,
        model_instance,
        train_set: list[StackDataset] | StackDataset,
        val_set: list[StackDataset] | StackDataset,
        architecture_search_budget: int = 30,
        architecture_grid: dict | None = None,
        optimizer_search_budget: int = 30,
        optimizer_grid: dict | None = None,
        scheduler_grid: dict | None = None,
    ) -> tuple:
        """Hyperparameter tuning routine, adapted from:
        https://github.com/google-research/tuning_playbook

        :param model_instance: the sklearn model instance
        :param StackDataset train_set: dataset to use for training
        :param StackDataset val_set: target to tune parameters against
        :param int architecture_search_budget: iterations for architecture search
        :param dict | None architecture_grid: architecture parameters to tune
        :param int optimizer_search_budget: iterations for optimizer search
        :param dict | None optimizer_grid: optimizer parameters to tune
        :param dict | None scheduler_grid: scheduler parameters to tune
        :return tuple: study objects from Optuna for architecture and optimizer
        """
        self.logger.info("Tuning: beginning process")
        if not isinstance(train_set, list):
            train_set = [train_set]
            val_set = [val_set]
        else:
            assert len(train_set) == len(val_set)

        for i in range(len(train_set)):
            model_instance.datamodule.dataset_train = train_set[i]
            model_instance.datamodule.dataset_val = val_set[i][0]
            model_instance.datamodule.dataset_test = val_set[i][1]
            model_instance.datamodule.setup(stage="fit")
            model_instance.datamodule.setup(stage="test")

            train_set[i] = model_instance.datamodule.train_dataloader()
            val_set[i][0] = model_instance.datamodule.val_dataloader()
            val_set[i][1] = model_instance.datamodule.test_dataloader()

        if model_instance._mlflow_manager.is_active:
            logging_uri = model_instance._mlflow_manager.params.log_dir
            study_name_arc = f"{model_instance._mlflow_manager.params.experiment}_architecture_search"
            study_name_opt = (
                f"{model_instance._mlflow_manager.params.experiment}_optimizer_search"
            )
        else:
            study_name_arc = "architecture_search"
            study_name_opt = "optimizer_search"
            logging_uri = None

        if architecture_grid is None:
            architecture_grid = load_default_architecture_grid(
                model_instance._architecture.__name__
            )
        if optimizer_grid is None:
            optimizer_grid = load_default_optimizer_grid(
                model_instance._model.params.optimizer
            )
        self.logger.info("Tuning: beginning HPO")
        init_params = model_instance._model.params.model_dump()
        _patch_clm_init_params(init_params, model_instance.datamodule)
        init_params.pop("torch_type")
        if scheduler_grid is None:
            scheduler_grid = load_default_scheduler_grid(init_params.get("scheduler"))
        params = run_hparam_tuning(
            train_set=train_set,
            val_set=val_set,
            model_architecture=model_instance._architecture.__name__,
            model_params=init_params,
            optimizer=model_instance._model.params.optimizer,
            num_epochs=model_instance._training_manager.params.num_epochs,
            patience=model_instance._training_manager.params.patience,
            stochastic_weight_averaging=model_instance._training_manager.params.stochastic_weight_averaging,
            accelerator=model_instance._training_manager.params.accelerator,
            devices=model_instance._training_manager.params.devices,
            architecture_search_budget=architecture_search_budget,
            architecture_grid=architecture_grid,
            optimizer_search_budget=optimizer_search_budget,
            optimizer_grid=optimizer_grid,
            scheduler_grid=scheduler_grid,
            logging_uri=logging_uri,
            study_name_arc=study_name_arc,
            study_name_opt=study_name_opt,
        )

        optimized_params = params["optimum"]

        model_instance._create_model(optimized_params)

        # Store tuning state
        self._params = TuningInputModel(
            architecture_search_budget=architecture_search_budget,
            architecture_grid=architecture_grid,
            optimizer_search_budget=optimizer_search_budget,
            optimizer_grid=optimizer_grid,
            scheduler_grid=scheduler_grid,
        )

        self.logger.info("Tuning: HPO finished")
        return params["architecture"], params["optimizer"]
