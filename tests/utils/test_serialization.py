"""Tests for matcha.utils.serialization – JSON, YAML, and pickle helpers.

Validates the round-trip (save → load) fidelity and directory-creation
behaviour that wraps the standard library serializers.
"""

import os

import numpy as np

from matcha.utils.serialization import (
    load_json,
    load_pickle,
    load_yaml,
    save_json,
    save_pickle,
    save_yaml,
)


class TestJsonRoundTrip:
    """save_json / load_json round-trip tests."""

    def test_dict_round_trip(self, tmp_path):
        data = {"a": 1, "b": [2, 3], "c": {"nested": True}}
        path = str(tmp_path / "data.json")
        save_json(path, data)
        loaded = load_json(path)
        assert loaded == data

    def test_creates_intermediate_dirs(self, tmp_path):
        path = str(tmp_path / "sub" / "dir" / "data.json")
        save_json(path, {"key": "value"})
        assert os.path.isfile(path)

    def test_indent_formatting(self, tmp_path):
        path = str(tmp_path / "pretty.json")
        save_json(path, {"x": 1})
        with open(path) as f:
            raw = f.read()
        # indent=4 should produce multi-line output
        assert "\n" in raw


class TestYamlRoundTrip:
    """save_yaml / load_yaml round-trip tests."""

    def test_dict_round_trip(self, tmp_path):
        data = {"alpha": 0.1, "layers": [64, 32], "flag": False}
        path = str(tmp_path / "config.yaml")
        save_yaml(path, data)
        loaded = load_yaml(path)
        assert loaded == data

    def test_creates_intermediate_dirs(self, tmp_path):
        path = str(tmp_path / "deep" / "nested" / "config.yaml")
        save_yaml(path, {"k": "v"})
        assert os.path.isfile(path)

    def test_preserves_key_order(self, tmp_path):
        """sort_keys=False should preserve insertion order."""
        data = {"z_key": 1, "a_key": 2, "m_key": 3}
        path = str(tmp_path / "order.yaml")
        save_yaml(path, data)
        with open(path) as f:
            lines = [line.strip() for line in f if line.strip()]
        keys = [line.split(":")[0] for line in lines]
        assert keys == ["z_key", "a_key", "m_key"]


class TestPickleRoundTrip:
    """save_pickle / load_pickle round-trip tests."""

    def test_numpy_array_round_trip(self, tmp_path):
        arr = np.array([1.0, 2.0, 3.0])
        path = str(tmp_path / "arr.pkl")
        save_pickle(path, arr)
        loaded = load_pickle(path)
        np.testing.assert_array_equal(loaded, arr)

    def test_complex_object_round_trip(self, tmp_path):
        data = {"list": [1, 2, 3], "set": {4, 5}, "tuple": (6, 7)}
        path = str(tmp_path / "obj.pkl")
        save_pickle(path, data)
        loaded = load_pickle(path)
        assert loaded == data

    def test_creates_intermediate_dirs(self, tmp_path):
        path = str(tmp_path / "a" / "b" / "data.pkl")
        save_pickle(path, 42)
        assert os.path.isfile(path)
