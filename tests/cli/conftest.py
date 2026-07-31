"""Shared pytest fixtures for CLI utility tests."""

import os

import pytest

from matcha.cli.utils import load_dataset
from matcha.utils.schemas.cli import Dataset

# Path to the shared regression/classification test dataset.
TESTING_DATA = os.path.join(os.path.dirname(__file__), os.pardir, "testing_data.csv")


@pytest.fixture(scope="module")
def dataset_cfg() -> Dataset:
    """Dataset config pointing at the shared testing CSV."""
    return Dataset(
        path=os.path.abspath(TESTING_DATA),
        label_key="Regression",
        smiles_key="SMILES",
    )


@pytest.fixture(scope="module")
def molecule_df(dataset_cfg):
    """DataFrame of RDKit molecules loaded from the shared testing CSV."""
    return load_dataset(dataset_cfg)
