"""Base schema module providing the foundational Pydantic model for all matcha schemas."""

from pydantic import BaseModel, ConfigDict


class BaseDataModel(BaseModel):
    """Base Pydantic model for all matcha data schemas.

    Configures population by field name and allows arbitrary types such as
    NumPy arrays and RDKit molecules.
    """

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)
