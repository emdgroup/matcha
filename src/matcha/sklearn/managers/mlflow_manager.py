import os
import shutil

from matcha.utils.logging import MatchaLogger, get_default_logger
from matcha.utils.schemas.sklearn_api import MLFlowInputModel


class MLFlowManager:
    """Manages MLflow experiment setup, logging, and artifact tracking."""

    def __init__(self):
        self.params: MLFlowInputModel | None = None
        self.logger = get_default_logger("MLFLOW")

    @property
    def is_active(self) -> bool:
        """Whether an MLflow experiment has been configured."""
        return self.params is not None

    def setup_experiment(
        self,
        experiment_name: str,
        run_name: str = "run",
        tag: dict | None = None,
        log_dir: str | None = "./matcha_log",
        server_uri: str | None = None,
    ) -> MLFlowInputModel:
        """Sets the experiment name and run name for the model.

        :param str experiment_name: name of the experiment to set
        :param str run_name: name of the run to associate with the experiment
        :param dict | None tag: tags to associate with the run
        :param str | None log_dir: directory for logging
        :param str | None server_uri: MLflow server URI
        :return MLFlowInputModel: the configured MLflow params
        """
        if tag is None:
            tag = {}
        # MLflow 4.9+ requires opt-in for filesystem backends; set the flag
        # when no remote server URI is configured so existing workflows keep working.
        if server_uri is None:
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        self.logger.info("MLFlow: setting experiment")
        mlflow_params = {
            "experiment": experiment_name,
            "run": run_name,
            "tag": tag,
            "log_dir": log_dir,
            "server_uri": server_uri,
        }
        self.params = MLFlowInputModel(**mlflow_params)
        self.logger.info("MLFlow: setup done")
        return self.params

    def create_logger(self) -> MatchaLogger:
        """Creates a MatchaLogger instance from the current MLflow params.

        :return MatchaLogger: configured MLflow logger
        """
        mlflow_logger = MatchaLogger(
            experiment_name=self.params.experiment,
            run_name=self.params.run,
            save_dir=self.params.log_dir,
            tracking_uri=self.params.server_uri,
            log_model=True,
        )
        if self.params.tag != {}:
            for i in range(len(self.params.tag)):
                mlflow_logger.experiment.set_tag(
                    mlflow_logger.run_id,
                    list(self.params.tag.keys())[i],
                    list(self.params.tag.values())[i],
                )
        return mlflow_logger

    def log_training(self, model_instance, mlflow_logger: MatchaLogger):
        """Log training artifacts and params to MLflow.

        :param model_instance: the sklearn model instance to log
        :param MatchaLogger mlflow_logger: the MLflow logger to use
        """
        mlflow_logger.store = True
        mlflow_logger.log_hyperparams(model_instance.params.model_dump())
        model_instance.save_model("./.matcha_temp")
        mlflow_logger.log_matcha_artifacts("./.matcha_temp")
        mlflow_logger.conclude_experiment()
        shutil.rmtree("./.matcha_temp")

    def log_calibrator(self, calibrator):
        """Log calibrator artifact to a new MLflow run.

        :param calibrator: the calibrator object to log
        """
        logger = self.create_logger()
        logger.experiment.set_tag(logger.run_id, "artifact_type", "calibrator")
        calibrator.save_calibrator("./.matcha_temp")
        logger.experiment.log_artifact(logger._run_id, "./.matcha_temp/calibrator.pkl")
        logger.conclude_experiment()
        shutil.rmtree("./.matcha_temp")
