from pydantic import Field, field_validator
from matcha.utils.schemas.base import BaseDataModel
from typing import Annotated, Literal, Sequence, Union
import numpy as np


class ICPBaseInputModel(BaseDataModel):
    """Base input model for Inductive Conformal Prediction calibrators."""

    confidence_alpha: float
    quantile: Union[float, list, np.ndarray] = 0.0

    @field_validator("quantile", mode="before")
    @classmethod
    def _coerce_quantile(cls, v):
        """Accept lists (e.g. from YAML round-trip) and convert to np.ndarray."""
        if isinstance(v, list):
            return np.array(v)
        return v


class ICPRegressionInputModel(ICPBaseInputModel):
    """Input model for ICP Regression calibrator."""

    calibrator_type: Literal["icp_regression"] = "icp_regression"


class ICPClassificationInputModel(ICPBaseInputModel):
    """Input model for ICP Classification calibrator (multitask binary)."""

    calibrator_type: Literal["icp_classification"] = "icp_classification"


class EMBaseInputModel(BaseDataModel):
    """Base input model for Error Model calibrators."""

    algorithm: str = "GradientBoostingClassifier"
    algorithm_params: dict = {}
    use_ecfp: bool = True
    use_interaction: bool = True
    min_compounds: int = 50


class EMRegressionInputModel(EMBaseInputModel):
    """Input model for Error Model Regression calibrator."""

    calibrator_type: Literal["error_model_regression"] = "error_model_regression"
    fold: float = 2.0
    log10: bool = False


class EMClassificationInputModel(EMBaseInputModel):
    """Input model for Error Model Classification calibrator."""

    calibrator_type: Literal["error_model_classification"] = (
        "error_model_classification"
    )
    error_threshold: float = 0.5


CalibratorModel = Union[
    ICPRegressionInputModel,
    ICPClassificationInputModel,
    EMRegressionInputModel,
    EMClassificationInputModel,
]


class CalibratorModels(BaseDataModel):
    """Container holding a sequence of calibrator input models discriminated by type."""

    inputmodels: Sequence[
        Annotated[CalibratorModel, Field(discriminator="calibrator_type")]
    ]
