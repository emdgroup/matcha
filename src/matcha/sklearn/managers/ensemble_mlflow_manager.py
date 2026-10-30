import os
import shutil

from matcha.utils.logging import MatchaLogger, get_default_logger
from matcha.utils.schemas.sklearn_api import MLFlowInputModel
from matcha.utils.serialization import save_yaml


class EnsembleMLFlowManager:
    """Manages MLflow experiment setup and artifact logging for ensembles.

    This mirrors :class:`MLFlowManager` but is tailored to ensemble-specific
    concerns such as per-member run naming and ensemble-level artifact logging.
    """

    def __init__(self):
        self.params: MLFlowInputModel | None = None
        self.logger = get_default_logger("ENSEMBLE_MLFLOW")

    @property
    def is_active(self) -> bool:
        """Whether an MLflow experiment has been configured."""
        return self.params is not None

    def setup_experiment(
        self,
        experiment_name: str,
        run_name: str = "model",
        tag: dict | None = None,
        log_dir: str | None = "./matcha_log",
        server_uri: str | None = None,
    ) -> MLFlowInputModel:
        """Configure the MLflow experiment for the ensemble.

        :param str experiment_name: name of the experiment to set
        :param str run_name: name of the run to associate with the experiment
        :param dict | None tag: optional tags to associate with the run
        :param str | None log_dir: optional directory for logging
        :param str | None server_uri: optional MLflow server URI
        :return MLFlowInputModel: the configured MLflow params
        """
        if tag is None:
            tag = {}
        tag.update({"model type": "ensemble"})
        self.params = MLFlowInputModel(
            experiment=experiment_name,
            run=run_name,
            tag=tag,
            log_dir=log_dir,
            server_uri=server_uri,
        )
        self.logger.info("MLFlow: ensemble experiment setup done")
        return self.params

    def setup_member_mlflow(self, member_model, member_idx: int) -> None:
        """Configure MLflow on an individual ensemble member for its training run.

        :param member_model: the BaseScikitLearnModel member
        :param int member_idx: index of the member in the ensemble
        """
        tag = self.params.tag.copy()
        tag["model ID"] = str(member_idx)
        member_model.set_mlflow_experiment(
            experiment_name=self.params.experiment,
            run_name=f"{self.params.run}_{member_idx}",
            log_dir=self.params.log_dir,
            server_uri=self.params.server_uri,
            tag=tag,
        )

    def log_artifact(self, artifact_path: str, tag: dict) -> None:
        """Log an artifact to a dedicated ensemble-level MLflow run.

        :param str artifact_path: path to the artifact file to log
        :param dict tag: tags specific to this artifact
        """
        logger = MatchaLogger(
            experiment_name=self.params.experiment,
            run_name=f"{self.params.run}_ensemble_artifacts",
            save_dir=self.params.model_dump().get("log_dir", None),
            tracking_uri=self.params.model_dump().get("server_uri", None),
        )
        default_tag = self.params.tag.copy()
        default_tag.update(tag)

        for key, value in default_tag.items():
            logger.experiment.set_tag(logger.run_id, key, value)

        logger.experiment.log_artifact(logger._run_id, artifact_path)
        logger.conclude_experiment()

    def log_ensemble_params(self, params_dump: dict) -> None:
        """Log ensemble parameters as a YAML artifact.

        :param dict params_dump: the ensemble params dict to serialize
        """
        os.makedirs("./.matcha_temp", exist_ok=True)
        save_yaml("./.matcha_temp/ensemble_params.yaml", params_dump)
        self.log_artifact(
            "./.matcha_temp/ensemble_params.yaml",
            {"Type": "Ensemble artifact"},
        )
        shutil.rmtree("./.matcha_temp")

    def log_calibrator(self, calibrator) -> None:
        """Log a fitted calibrator artifact to MLflow.

        :param calibrator: the calibrator object to log
        """
        os.makedirs("./.matcha_temp", exist_ok=True)
        calibrator.save_calibrator("./.matcha_temp")
        self.log_artifact(
            "./.matcha_temp/calibrator.pkl",
            {"Type": "Ensemble calibrator", "Status": "Trained"},
        )
        shutil.rmtree("./.matcha_temp")
