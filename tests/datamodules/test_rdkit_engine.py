"""Tests for rdkit_engine.Engine."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from matcha.datamodules.classic import rdkit_engine
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

    @pytest.mark.parametrize(
        "feature_name,expected_dim",
        [
            ("ecfp", 1024),
            ("ecfp_count", 2048),
            ("erg", 315),
            ("avalon", 4096),
            ("estate", 79),
            ("rdkit_all_descriptors", None),
            ("map4", 2048),
            ("mhfp", 2048),
            ("pubchem_fp", 881),
            ("rdkit_fp", 2048),
            ("mordred", 1613),
        ],
    )
    def test_feature_inventory_and_dimensions(self, feature_name, expected_dim):
        e = Engine()
        if expected_dim is None:
            expected_dim = len(e.rdkit_all_descriptors)

        assert feature_name in e._mapping
        assert e.calculate_feature_dim([feature_name.upper()]) == expected_dim


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


class TestECFPCount:
    def test_shape(self, engine, mols):
        result = engine.get_ECFP_count(mols, n_jobs=1)
        assert result.shape == (len(mols), engine.defaults["ecfp_count"]["nBits"])


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

    @pytest.mark.parametrize(
        "use_bits,expected",
        [
            (True, [[1.0, 2.0]]),
            (False, [[3.0, 4.0]]),
        ],
    )
    def test_selects_configured_tuple_side(self, monkeypatch, use_bits, expected):
        e = Engine(n_jobs=3)
        e.defaults["estate"]["use_bits"] = use_bits
        parallelize = MagicMock(
            return_value=[(np.array([1.0, 2.0]), np.array([3.0, 4.0]))]
        )
        monkeypatch.setattr(rdkit_engine, "parallelize", parallelize)

        result = e.get_ESTATE([object()])

        np.testing.assert_array_equal(result, expected)
        assert parallelize.call_args.kwargs["n_jobs"] == 3


class TestRDKitAllDescriptors:
    def test_shape(self, engine, mols):
        result = engine.get_rdkit_all_descriptors(mols, n_jobs=1)
        assert result.shape[0] == len(mols)

    def test_clips_and_converts_non_finite_values(self, monkeypatch, engine, mols):
        values = np.array([np.nan, np.inf, -np.inf, 20000.0, -20000.0])
        monkeypatch.setattr(
            rdkit_engine, "parallelize", MagicMock(return_value=[values])
        )

        result = engine.get_arbitrary_rdkit_descriptors([mols[0]], ["MolWt"], n_jobs=1)

        np.testing.assert_array_equal(
            result, [[0.0, 10000.0, -10000.0, 10000.0, -10000.0]]
        )


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

    def test_normalizes_names_forwards_jobs_and_preserves_order(
        self, monkeypatch, engine, mols
    ):
        first = np.full((len(mols), 2), 1.0)
        second = np.full((len(mols), 3), 2.0)
        ecfp = MagicMock(return_value=first)
        erg = MagicMock(return_value=second)
        monkeypatch.setitem(engine._mapping, "ecfp", ecfp)
        monkeypatch.setitem(engine._mapping, "erg", erg)

        result = engine.get_features(mols, ["ECFP", "ErG"], n_jobs=7)

        np.testing.assert_array_equal(result, np.concatenate([first, second], axis=1))
        ecfp.assert_called_once_with(mols, n_jobs=7)
        erg.assert_called_once_with(mols, n_jobs=7)

    def test_calculate_feature_dim(self):
        e = Engine()
        dim = e.calculate_feature_dim(["ecfp", "erg"])
        expected = e.defaults["ecfp"]["nBits"] + 315
        assert dim == expected


class TestScikitFingerprintAdapters:
    @pytest.mark.parametrize(
        "class_name,method_name,defaults_key,expected_kwargs",
        [
            (
                "MAPFingerprint",
                "get_MAP4",
                "map4",
                {"fp_size": 17, "n_jobs": 7},
            ),
            (
                "MHFPFingerprint",
                "get_MHFP",
                "mhfp",
                {"fp_size": 19, "n_jobs": 7},
            ),
            (
                "RDKitFingerprint",
                "get_rdkit_fp",
                "rdkit_fp",
                {"fp_size": 23, "n_jobs": 7},
            ),
            ("PubChemFingerprint", "get_pubchem_fp", None, {"n_jobs": 7}),
            (
                "MordredFingerprint",
                "get_mordred",
                "mordred",
                {"use_3D": True, "n_jobs": 7},
            ),
        ],
    )
    def test_constructor_and_transform_forwarding(
        self,
        monkeypatch,
        mols,
        class_name,
        method_name,
        defaults_key,
        expected_kwargs,
    ):
        fingerprint = MagicMock()
        fingerprint.transform.return_value = np.zeros((len(mols), 3))
        constructor = MagicMock(return_value=fingerprint)
        monkeypatch.setattr(rdkit_engine, class_name, constructor)
        e = Engine()
        if defaults_key is not None:
            configured_values = {
                key: value for key, value in expected_kwargs.items() if key != "n_jobs"
            }
            e.defaults[defaults_key].update(configured_values)

        result = getattr(e, method_name)(mols, n_jobs=7)

        constructor.assert_called_once_with(**expected_kwargs)
        fingerprint.transform.assert_called_once_with(mols)
        assert result.shape == (len(mols), 3)

    def test_mordred_clips_and_converts_non_finite_values(self, monkeypatch, mols):
        fingerprint = MagicMock()
        fingerprint.transform.return_value = np.array(
            [[np.nan, np.inf, -np.inf, 20.0, -20.0]]
        )
        monkeypatch.setattr(
            rdkit_engine, "MordredFingerprint", MagicMock(return_value=fingerprint)
        )
        e = Engine()
        e.defaults["mordred"].update({"posinf": 9.0, "neginf": -7.0})

        result = e.get_mordred([mols[0]], n_jobs=1)

        np.testing.assert_array_equal(result, [[0.0, 9.0, -7.0, 9.0, -7.0]])
