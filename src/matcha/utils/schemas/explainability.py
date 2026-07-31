from matcha.utils.schemas.base import BaseDataModel


class ExplainerInputModel(BaseDataModel):
    """Schema for explainability method parameters including PAS, nitrogen walk, and LIME."""

    positional_analogue_scanning_params: dict | None
    nitrogen_walk_params: dict | None
    lime_descriptor_set: list[str] | None
    lime_fingerprint_params: dict | None
    lime_scale_coeff: bool
    lime_remove_noise: bool
