"""Tests for rdkit_engine.Engine."""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from matcha.datamodules.classic.rdkit_engine import Engine


@pytest.fixture(scope="module")
def engine() -> Engine:
    return Engine(n_jobs=1)


@pytest.fixture(scope="module")
def mols() -> list[Mol]:
    smiles = ["c1ccccc1", "CC(=O)O", "CCO", "c1ccc(O)cc1", "CC(C)C"]
    return [Chem.MolFromSmiles(s) for s in smiles]


# ===================================================================
# Constructor / properties
# ===================================================================


class TestEngineInit:
    def test_default_n_jobs(self):
        e = Engine()
        assert e.n_jobs == 32

    def test_custom_n_jobs(self):
        e = Engine(n_jobs=4)
        assert e.n_jobs == 4

    def test_set_n_jobs(self):
        e = Engine(n_jobs=4)
        e.n_jobs = 8
        assert e.n_jobs == 8

    def test_negative_n_jobs_raises(self):
        e = Engine()
        with pytest.raises(ValueError):
            e.n_jobs = -1

    def test_non_int_n_jobs_raises(self):
        e = Engine()
        with pytest.raises(ValueError):
            e.n_jobs = "abc"


class TestEngineDefaults:
    def test_defaults_contain_expected_keys(self):
        e = Engine()
        expected = {
            "ecfp",
            "ecfp_count",
            "erg",
            "avalon",
            "estate",
            "rdkit_all_descriptors",
        }
        assert expected.issubset(set(e.defaults.keys()))

    def test_set_defaults(self):
        e = Engine()
        new_params = {"nBits": 512, "radius": 2, "useFeatures": True}
        e.set_defaults("ecfp", new_params)
        assert e.defaults["ecfp"]["nBits"] == 512

    def test_set_defaults_invalid_key(self):
        e = Engine()
        with pytest.raises(ValueError):
            e.set_defaults("nonexistent", {})


# ===================================================================
# Individual featurizers
# ===================================================================


class TestECFP:
    def test_shape(self, engine, mols):
        result = engine.get_ECFP(mols, n_jobs=1)
        assert result.shape == (len(mols), engine.defaults["ecfp"]["nBits"])

    def test_dtype(self, engine, mols):
        result = engine.get_ECFP(mols, n_jobs=1)
        assert result.dtype == np.float32

    def test_binary_values(self, engine, mols):
        result = engine.get_ECFP(mols, n_jobs=1)
        assert set(np.unique(result)).issubset({0.0, 1.0})


class TestECFPCount:
    def test_shape(self, engine, mols):
        result = engine.get_ECFP_count(mols, n_jobs=1)
        assert result.shape == (len(mols), engine.defaults["ecfp_count"]["nBits"])

    def test_non_negative(self, engine, mols):
        result = engine.get_ECFP_count(mols, n_jobs=1)
        assert (result >= 0).all()


class TestERG:
    def test_shape(self, engine, mols):
        result = engine.get_ERG(mols, n_jobs=1)
        assert result.shape[0] == len(mols)
        assert result.shape[1] == 315


class TestAvalon:
    def test_shape(self, engine, mols):
        result = engine.get_Avalon(mols, n_jobs=1)
        assert result.shape == (len(mols), engine.defaults["avalon"]["nBits"])


class TestESTATE:
    def test_shape(self, engine, mols):
        result = engine.get_ESTATE(mols, n_jobs=1)
        assert result.shape[0] == len(mols)
        assert result.shape[1] == 79


class TestRDKitAllDescriptors:
    def test_shape(self, engine, mols):
        result = engine.get_rdkit_all_descriptors(mols, n_jobs=1)
        assert result.shape[0] == len(mols)

    def test_no_infs(self, engine, mols):
        result = engine.get_rdkit_all_descriptors(mols, n_jobs=1)
        assert np.isfinite(result).all()


# ===================================================================
# get_features (multiple combined)
# ===================================================================


class TestGetFeatures:
    def test_single_feature(self, engine, mols):
        result = engine.get_features(mols, ["ecfp"])
        assert result.shape[0] == len(mols)

    def test_multiple_features_concatenated(self, engine, mols):
        result = engine.get_features(mols, ["ecfp", "erg"])
        expected_dim = engine.defaults["ecfp"]["nBits"] + 315
        assert result.shape == (len(mols), expected_dim)

    def test_calculate_feature_dim(self):
        e = Engine()
        dim = e.calculate_feature_dim(["ecfp", "erg"])
        expected = e.defaults["ecfp"]["nBits"] + 315
        assert dim == expected
