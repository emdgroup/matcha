"""Optuna-based hyperparameter tuning with architecture and optimizer search phases."""

from typing import Any
import logging

from matcha.utils import load_json, silence_nuisance_warnings
from optuna.samplers import TPESampler
from optuna.pruners import NopPruner
from optuna.integration import MLflowCallback
from optuna import create_study, trial
from matcha.torch.models.classic.base_classic_model import (
    ClassicModelRegistry,
    BaseClassicModel,
)
import os
from lightning.pytorch.callbacks import (
    EarlyStopping,
    StochasticWeightAveraging,
)
import lightning as L
from torch.utils.data import DataLoader
import numpy as np

logger = logging.getLogger(__name__)


def parse_betas(dictionary: dict) -> dict:
    """Convert ``beta_1`` key into a ``betas`` tuple expected by PyTorch optimizers.

    :param dictionary: parameter dictionary containing a ``beta_1`` key.
    :returns: updated dictionary with ``betas`` tuple and ``beta_1`` removed.
    :rtype: dict
    """
    dictionary["betas"] = (dictionary["beta_1"], 0.999)
    dictionary.pop("beta_1")
    return dictionary


def parse_num_heads(dictionary: dict) -> dict:
    """Adjust ``enc_num_heads`` so it evenly divides the hidden dimension.

    :param dictionary: parameter dictionary potentially containing ``enc_num_heads``
        and a hidden dimension key.
    :returns: dictionary with ``enc_num_heads`` corrected if necessary.
    :rtype: dict
    """
    if "enc_num_heads" in dictionary:
        # Check for enc_atom_hidden_dim first, then enc_hidden_dim
        hidden_dim = None
        if "enc_atom_hidden_dim" in dictionary:
            hidden_dim = dictionary["enc_atom_hidden_dim"]
        elif "enc_hidden_dim" in dictionary:
            hidden_dim = dictionary["enc_hidden_dim"]

        if hidden_dim is not None:
            num_heads = dictionary["enc_num_heads"]
            if hidden_dim % num_heads != 0:
                # Walk down to the largest divisor of hidden_dim that is <= num_heads.
                while num_heads > 1 and hidden_dim % num_heads != 0:
                    num_heads -= 1
                dictionary["enc_num_heads"] = num_heads

    return dictionary


def parse_config(trial: trial.Trial, dictionary: dict) -> dict:
    """Helper function to update the model params with the new
    suggested ones from the optimizer

    :param Trial trial: Optuna class for handling hparam process

    :param dict dictionary: params of the model

    :return dict: new params of the model to eval in next iteration
    """
    keys = list(dictionary.keys())
    trial_type = [x[0] for x in list(dictionary.values())]
    trial_params = [x[1] for x in list(dictionary.values())]

    update = {}

    for i, key in enumerate(keys):
        if trial_type[i] == "float":
            update[key] = trial.suggest_float(key, **trial_params[i])
        elif trial_type[i] == "int":
            update[key] = trial.suggest_int(key, **trial_params[i])
        elif trial_type[i] == "choice":
            trial_params[i] = [x if x != "none" else None for x in trial_params[i]]
            update[key] = trial.suggest_categorical(key, trial_params[i])
        elif trial_type[i] == "constant":
            update[key] = trial_params[i]

    if "beta_1" in update:
        update = parse_betas(update)

    update = parse_num_heads(update)
    return update


def load_default_architecture_grid(architecture) -> dict:
    """Load the default architecture search space from a bundled JSON file.

    :param str architecture: name of the architecture to look up.
    :returns: search space dictionary for the given architecture.
    :rtype: dict
    """
    return load_json(
        f"{os.path.dirname(os.path.realpath(__file__))}/architecture_grid.json"
    )[architecture]


def load_default_optimizer_grid(optimizer) -> dict:
    """Load the default optimizer search space from a bundled JSON file.

    :param str optimizer: name of the optimizer to look up.
    :returns: search space dictionary for the given optimizer.
    :rtype: dict
    """
    return load_json(
        f"{os.path.dirname(os.path.realpath(__file__))}/optimizer_grid.json"
    )[optimizer]


def load_default_scheduler_grid(scheduler_name: str | None) -> dict:
    """Load the default scheduler search space for a given scheduler.

    :param scheduler_name: name of the active scheduler, or None.
    :returns: search space dictionary for the scheduler, or empty dict if the
        scheduler has no tunable parameters.
    :rtype: dict
    """
    if scheduler_name is None:
        return {}
    grid = load_json(
        f"{os.path.dirname(os.path.realpath(__file__))}/scheduler_grid.json"
    )
    return grid.get(scheduler_name, {})


def run_hparam_tuning(
    train_set: list[DataLoader] | DataLoader,
    val_set: list[DataLoader],
    model_architecture: str | BaseClassicModel,
    model_params: dict,
    optimizer: str = "adam",
    num_epochs: int = 200,
    patience: int = 20,
    stochastic_weight_averaging: bool = True,
    accelerator: str = "gpu",
    devices: int = 1,
    architecture_search_budget: int = 70,
    architecture_grid: dict | None = None,
    optimizer_search_budget: int = 30,
    optimizer_grid: dict | None = None,
    scheduler_grid: dict | None = None,
    logging_uri: str | None = None,
    study_name_arc: str = "architecture_search",
    study_name_opt: str = "optimizer_search",
) -> dict[str, Any]:
    """Hyperparameter optimization function for MATCHA models.

    It first runs a QMC scan across different architecture configurations. Then,
    it runs bayesian optimization via TPE on the parameters of the learning rate.
    Each trial runs with early stopping to terminate underperforming runs.

    The procedure was adapted from here:
    https://github.com/google-research/tuning_playbook

    :param tuple train_set: training set to use for hparam scan, should be the
        output of the correct featurizer

    :param tuple val_set: validation set to improve on (ideally time split),
        should be the output of the correct featurizer using is_training=False

    :param str model_architecture: name of the model to optimize, or model to
        use for tuning (useful for pretrained ones)

    :param dict model_params: custom params to keep fixed during the process,
        a typical example would be in case extra features are used one has
        to specify additional_mol_features_dim

    :param str optimizer: which optimizer to use for model training and for
        tuning its hparams, defaults to "adam"

    :param int num_epochs: max number of epochs to use, defaults to 200

    :param int patience: how many epochs to wait before early stopping, defaults
        to 20

    :param bool stochastic_weight_averaging: whether to use SWA

    :param str accelerator: whether to train on cpu or gpu, defaults to "gpu"

    :param int architecture_search_budget: how many QMC sampling iterations
        to do for the architecture search step, defaults to 70

    :param dict | None architecture_grid: which params to change for the
        given architecture, defaults to None, meaning it will be loaded
        from architecture_grid.json

    :param int optimizer_search_budget: How many TPE steps to do when
        tuning the learning rate parameters, defaults to 30

    :param dict | None optimizer_grid: which params to tune for the
        optimizer, defaults to None, meaning it will be loaded
        from optimizer_grid.json

    :param dict | None scheduler_grid: search space for scheduler parameters
        to tune in Phase 2, defaults to None, meaning it will be loaded
        from scheduler_grid.json based on the active scheduler

    :param str | None logging_uri: MLflow tracking URI for logging trials.
        If None, logging is disabled.

    :param str study_name_arc: Optuna study name for the architecture search phase.

    :param str study_name_opt: Optuna study name for the optimizer search phase.

    :returns: dictionary with keys ``"optimum"`` (best params), ``"architecture"``
        (Optuna study or ``"Not executed"``), and ``"optimizer"`` (Optuna study
        or ``"Not executed"``).
    :rtype: dict[str, Any]
    """

    silence_nuisance_warnings()

    params = model_params.copy()
    if architecture_grid is None:
        architecture_grid = load_default_architecture_grid(model_architecture)
    if optimizer_grid is None:
        optimizer_grid = load_default_optimizer_grid(optimizer)
    if scheduler_grid is None:
        scheduler_grid = load_default_scheduler_grid(params.get("scheduler"))
    params["optimizer"] = optimizer
    target_grid = None

    if not isinstance(train_set, list):
        train_set = [train_set]
    else:
        assert len(train_set) == len(val_set), (
            "Mismatched list size between train and val splits"
        )
    if not isinstance(val_set[0], list):
        val_set = [val_set]

    def objective(trial):
        update = parse_config(trial, target_grid)
        if "lr" in update:
            new_params = params
            new_params["optimizer_args"] = update
            # Suggest scheduler params alongside optimizer params in Phase 2
            if scheduler_grid:
                sched_update = parse_config(trial, scheduler_grid)
                new_params["scheduler_args"] = {
                    **params.get("scheduler_args", {}),
                    **sched_update,
                }
        else:
            new_params = params | update

        out_loss = []

        for i in range(len(train_set)):
            callbacks = [
                EarlyStopping("val_loss", mode="min", patience=patience),
                # ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min"),
            ]

            if stochastic_weight_averaging:
                callbacks.append(
                    StochasticWeightAveraging(swa_lrs=1e-4, swa_epoch_start=0.5)
                )

            model = ClassicModelRegistry[model_architecture](**new_params)

            trainer = L.Trainer(
                callbacks=callbacks,
                max_epochs=num_epochs,
                accelerator=accelerator,
                devices=devices,
                enable_checkpointing=False,
                logger=False,
            )

            trainer.fit(model, train_set[i], val_set[i][0])

            # try:
            #     best_model = ClassicModelRegistry[
            #         model_architecture
            #     ].load_from_checkpoint(callbacks[2].best_model_path)
            # except Exception:
            #     # this whole block is caused by using hybrid torch/huggingface models
            #     # and requires a lot of obscure/unsafe stuff to work...
            #     ckpt = torch.load(callbacks[2].best_model_path, weights_only=False)
            #     state_dict = ckpt.get("state_dict", ckpt)
            #     best_model = ClassicModelRegistry[model_architecture](**new_params)
            #     best_model.load_state_dict(state_dict, strict=False)

            val = trainer.validate(model=model, dataloaders=val_set[i][1])
            out_loss.append(val[0]["val_loss"])

        return np.mean(out_loss)

    if architecture_search_budget > 0:
        target_grid = architecture_grid
        callbacks = []
        if logging_uri is not None:
            optuna_callback = MLflowCallback(
                tracking_uri=logging_uri, metric_name="val_loss"
            )
            callbacks.append(optuna_callback)
        study_architecture = create_study(
            direction="minimize",
            pruner=NopPruner(),
            sampler=TPESampler(
                n_startup_trials=max(10, int(architecture_search_budget * 0.25)),
                multivariate=True,
            ),
            study_name=study_name_arc,
        )
        study_architecture.optimize(
            objective,
            n_trials=architecture_search_budget,
            gc_after_trial=True,
            callbacks=callbacks,
        )
        params = params | study_architecture.best_params
        # gotta correct the optuna output for num_heads
        params = parse_num_heads(params)
    else:
        study_architecture = "Not executed"

    if optimizer_search_budget > 0:
        target_grid = optimizer_grid
        callbacks = []
        if logging_uri is not None:
            optuna_callback = MLflowCallback(
                tracking_uri=logging_uri, metric_name="val_loss"
            )
            callbacks.append(optuna_callback)
        study_optimizer = create_study(
            direction="minimize",
            sampler=TPESampler(
                multivariate=True,
                n_startup_trials=max(10, int(optimizer_search_budget * 0.25)),
            ),
            pruner=NopPruner(),
            study_name=study_name_opt,
        )
        study_optimizer.optimize(
            objective,
            n_trials=optimizer_search_budget,
            gc_after_trial=True,
            callbacks=callbacks,
        )
        optimum = study_optimizer.best_params
        if "beta_1" in optimum:
            optimum = parse_betas(optimum)
        # Split scheduler params from optimizer params using the grid keys
        scheduler_param_keys = frozenset(scheduler_grid.keys())
        sched_params = {k: v for k, v in optimum.items() if k in scheduler_param_keys}
        opt_params = {k: v for k, v in optimum.items() if k not in scheduler_param_keys}
        params["optimizer_args"] = opt_params
        if sched_params:
            params["scheduler_args"] = {
                **params.get("scheduler_args", {}),
                **sched_params,
            }
    else:
        study_optimizer = "Not executed"

    return {
        "optimum": params,
        "architecture": study_architecture,
        "optimizer": study_optimizer,
    }
