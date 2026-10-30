from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from typing_extensions import override
import os
import logging


class MatchaLogger(MLFlowLogger):
    """Custom MLflow logger for matcha that defers hyperparameter logging
    and prevents premature experiment finalization.
    """

    def __init__(self, experiment_name=None, run_name=None, **kwargs):
        """Initialize the MatchaLogger.

        :param str experiment_name: Name of the MLflow experiment.
        :param str run_name: Name of the MLflow run.
        :param kwargs: Additional keyword arguments passed to
            :class:`~lightning.pytorch.loggers.MLFlowLogger`.
        """
        super().__init__(experiment_name=experiment_name, run_name=run_name, **kwargs)
        self._store = False

    @property
    def store(self) -> bool:
        """Whether hyperparameter logging is enabled.

        :returns: ``True`` if logging is active, ``False`` otherwise.
        :rtype: bool
        """
        return self._store

    @store.setter
    def store(self, value: bool):
        if isinstance(value, bool):
            self._store = value
        else:
            raise ValueError(f"MatchaLogger.store must be set as bool, found {value}")

    @override
    @rank_zero_only
    def finalize(self, status: str = "success"):
        """Override original finalize method to prevent closing the experiment
        once training is finished, and logging of raw pytorch artifacts"""
        pass

    @rank_zero_only
    def conclude_experiment(self):
        """Manually conclude the experiment."""
        self.experiment.set_terminated(self.run_id, "FINISHED")

    def log_hyperparams(self, params: dict):
        """Simple wrapper to prevent params to be saved until the method
        is called manually"""
        if self.store:
            super().log_hyperparams(params)

    def log_matcha_artifacts(self, folder_path: list[str]):
        """Add default artifacts after training.

        Handles both files and subdirectories. Files are logged at the
        artifact root; subdirectories are logged with their relative path
        preserved.
        """
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                self.experiment.log_artifacts(self._run_id, item_path, item)
            else:
                self.experiment.log_artifact(self._run_id, item_path)


def get_default_logger(name: str, logging_path: str | None = None):
    """Create or retrieve a standard Python logger for matcha components.

    Configures a logger with console output and, optionally, file output.
    If the logger already has handlers, it is returned as-is to avoid
    duplicate handler registration.

    :param str name: Identifier appended to the ``MATCHA`` logger namespace.
    :param logging_path: Optional file path for log output. Directories are
        created if they do not exist.
    :type logging_path: str | None
    :returns: Configured logger instance.
    :rtype: logging.Logger
    """
    logger = logging.getLogger(f"MATCHA {name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

        if logging_path is not None:
            log_dir = os.path.dirname(logging_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            file_handler = logging.FileHandler(logging_path)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
