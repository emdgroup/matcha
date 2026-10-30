"""Regression tests for evaluate.main() honouring model.config_path (issue #455).

Also covers bootstrap key-guard behaviour (issue #460, stage 1) and autotune YAML
format handling (issue #470, stage 2).
"""

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from matcha.cli.evaluate import main
from matcha.cli.utils import aggregate_scores
from matcha.utils.schemas.cli import CLIEvaluationInputModel

TESTING_DATA: Path = Path(__file__).parent.parent / "testing_data.csv"

_YAML_PARAMS: dict = {
    "num_epochs": 1,
    "batch_size": 32,
    "accelerator": "cpu",
    "devices": 1,
    "early_stopping": False,
    "stochastic_weight_averaging": False,
    "enc_num_layers": 1,
    "enc_atom_hidden_dim": 32,
    "pred_hidden_dims": [32],
    "rwse_k": 0,
    "laplacian_k": 0,
    "elstatic_k": 0,
    "distmat_k": 0,
    "rrwp_k": 0,
    "num_virtual_nodes": 0,
    "label_encoder_params": {"encoder_type": "regression"},
}

# Simulates the nested ScikitLearnInputModel dict that autotune saves via model_dump().
# enc_num_layers/enc_atom_hidden_dim differ from _YAML_PARAMS to verify override.
# datamodule.label_encoder_params differs from _YAML_PARAMS to verify base-YAML wins.
_AUTOTUNE_YAML: dict = {
    "training": {
        "num_epochs": 1,
        "batch_size": 32,
        "accelerator": "cpu",
        "devices": 1,
        "early_stopping": False,
        "stochastic_weight_averaging": False,
    },
    "datamodule": {
        "label_encoder_params": {"encoder_type": "classification"},
    },
    "model": {
        "enc_num_layers": 5,
        "enc_atom_hidden_dim": 256,
        "pred_hidden_dims": [64, 64],
        "rwse_k": 0,
        "laplacian_k": 0,
        "elstatic_k": 0,
        "distmat_k": 0,
        "rrwp_k": 0,
        "num_virtual_nodes": 0,
        "torch_type": "GINRegressor",
    },
    "metadata": None,
    "task_type": "regression",
    "calibration": None,
    "mlflow": None,
    "tuning": None,
}


class TestEvaluateNoConfigPath:
    """evaluate.main() uses base YAML params when config_path is not set."""

    def _make_cfg(self, tmp_path: Path) -> CLIEvaluationInputModel:
        return CLIEvaluationInputModel.model_validate(
            {
                "dataset": {
                    "path": str(TESTING_DATA),
                    "label_key": "Regression",
                    "smiles_key": "SMILES",
                },
                "model": {
                    "architecture": "GINRegressor",
                    "params": dict(_YAML_PARAMS),
                    "metadata": {
                        "model_name": "test",
                        "model_version": 1,
                        "model_scope": "test",
                        "model_owner": "test",
                    },
                },
                "output": {
                    "serialization": {"path": str(tmp_path)},
                },
                "split": {"method": "cv", "n_subset": 2},
            }
        )

    def _run_main(self, cfg: CLIEvaluationInputModel) -> list[dict]:
        captured_kwargs: list[dict] = []

        def fake_model_cls(**kwargs: object) -> MagicMock:
            captured_kwargs.append(dict(kwargs))
            mock = MagicMock()
            mock.params.metadata.model_type = "GINRegressor"
            mock.predict.return_value = np.ones((10, 1))
            return mock

        mock_registry = MagicMock()
        mock_registry.__getitem__ = MagicMock(return_value=fake_model_cls)

        fake_df = pd.DataFrame({"SMILES": ["c1ccccc1"] * 10, "Regression": np.ones(10)})
        fake_mols: list[Mol] = [Chem.MolFromSmiles("c1ccccc1")] * 10
        fake_y = np.ones((10, 1))

        with (
            patch("matcha.cli.evaluate.ScikitLearnModelRegistry", mock_registry),
            patch(
                "matcha.cli.evaluate.get_splits",
                return_value=([fake_df, fake_df], [fake_df, fake_df]),
            ),
            patch(
                "matcha.cli.evaluate.parse_df",
                return_value=(fake_mols, fake_y, None),
            ),
            patch(
                "matcha.cli.evaluate.process_regression",
                return_value={"r2": 0.9},
            ),
            patch("matcha.cli.evaluate.plot_regression", return_value=MagicMock()),
            patch("matcha.cli.evaluate.save_plot"),
            patch("matcha.cli.evaluate.aggregate_scores", return_value={}),
            patch("matcha.cli.evaluate.save_json"),
            patch("matcha.cli.evaluate.save_config_as_yaml"),
        ):
            main(cfg)

        return captured_kwargs

    def test_uses_yaml_params(self, tmp_path: Path) -> None:
        """Without config_path, model is instantiated with the original YAML params."""
        cfg = self._make_cfg(tmp_path)
        kwargs_list = self._run_main(cfg)

        for kwargs in kwargs_list:
            assert kwargs["enc_num_layers"] == _YAML_PARAMS["enc_num_layers"]
            assert kwargs["enc_atom_hidden_dim"] == _YAML_PARAMS["enc_atom_hidden_dim"]

    def test_seed_injected_per_split(self, tmp_path: Path) -> None:
        """seed is set to the split index for each split."""
        cfg = self._make_cfg(tmp_path)
        kwargs_list = self._run_main(cfg)

        assert len(kwargs_list) == 2
        assert kwargs_list[0]["seed"] == 0
        assert kwargs_list[1]["seed"] == 1


class TestBootstrapKeyGuard:
    """evaluate.main() guards integer split keys based on n_bootstrap (issue #460, stage 1)."""

    def _make_cfg(
        self, tmp_path: Path, n_bootstrap: int | None = None
    ) -> CLIEvaluationInputModel:
        """Build a minimal two-split config, optionally with bootstrapping enabled."""
        split_cfg: dict = {"method": "cv", "n_subset": 2}
        if n_bootstrap is not None:
            split_cfg["n_bootstrap"] = n_bootstrap
            split_cfg["frac_bootstrap"] = 0.8
        return CLIEvaluationInputModel.model_validate(
            {
                "dataset": {
                    "path": str(TESTING_DATA),
                    "label_key": "Regression",
                    "smiles_key": "SMILES",
                },
                "model": {
                    "architecture": "GINRegressor",
                    "params": dict(_YAML_PARAMS),
                    "metadata": {
                        "model_name": "test",
                        "model_version": 1,
                        "model_scope": "test",
                        "model_owner": "test",
                    },
                },
                "output": {
                    "serialization": {"path": str(tmp_path)},
                },
                "split": split_cfg,
            }
        )

    def _run_and_capture(self, cfg: CLIEvaluationInputModel) -> dict:
        """Run evaluate.main() and return the performance_dict passed to save_json."""
        captured: list[dict] = []

        def fake_model_cls(**kwargs: object) -> MagicMock:
            mock = MagicMock()
            mock.params.metadata.model_type = "GINRegressor"
            mock.predict.return_value = np.ones((10, 1))
            return mock

        mock_registry = MagicMock()
        mock_registry.__getitem__ = MagicMock(return_value=fake_model_cls)

        fake_df = pd.DataFrame({"SMILES": ["c1ccccc1"] * 10, "Regression": np.ones(10)})
        fake_mols: list[Mol] = [Chem.MolFromSmiles("c1ccccc1")] * 10
        fake_y = np.ones((10, 1))

        def capture_save_json(path: str, data: dict) -> None:
            if "performance.json" in path:
                captured.append(dict(data))

        with (
            patch("matcha.cli.evaluate.ScikitLearnModelRegistry", mock_registry),
            patch(
                "matcha.cli.evaluate.get_splits",
                return_value=([fake_df, fake_df], [fake_df, fake_df]),
            ),
            patch(
                "matcha.cli.evaluate.parse_df", return_value=(fake_mols, fake_y, None)
            ),
            patch(
                "matcha.cli.evaluate.process_regression",
                return_value={"r2": 0.9},
            ),
            patch("matcha.cli.evaluate.plot_regression", return_value=MagicMock()),
            patch("matcha.cli.evaluate.save_plot"),
            patch("matcha.cli.evaluate.save_json", side_effect=capture_save_json),
            patch("matcha.cli.evaluate.save_config_as_yaml"),
        ):
            main(cfg)

        return captured[0] if captured else {}

    def test_bootstrap_mode_has_no_integer_keys(self, tmp_path: Path) -> None:
        """When n_bootstrap > 1, performance_dict contains no integer split keys."""
        cfg = self._make_cfg(tmp_path, n_bootstrap=3)
        perf = self._run_and_capture(cfg)

        integer_keys = [k for k in perf if isinstance(k, int)]
        assert integer_keys == [], f"Expected no integer keys, found: {integer_keys}"

    def test_bootstrap_mode_has_string_keys(self, tmp_path: Path) -> None:
        """When n_bootstrap > 1, performance_dict contains bootstrap string keys."""
        cfg = self._make_cfg(tmp_path, n_bootstrap=3)
        perf = self._run_and_capture(cfg)

        bootstrap_keys = [k for k in perf if re.match(r"^\d+_\d+$", str(k))]
        assert len(bootstrap_keys) > 0, (
            "Expected bootstrap string keys in performance_dict"
        )

    def test_non_bootstrap_mode_has_integer_keys(self, tmp_path: Path) -> None:
        """When n_bootstrap is None, performance_dict has integer split keys and no bootstrap keys."""
        cfg = self._make_cfg(tmp_path, n_bootstrap=None)
        perf = self._run_and_capture(cfg)

        integer_keys = [k for k in perf if isinstance(k, int)]
        assert len(integer_keys) == 2, (
            f"Expected 2 integer split keys, found: {integer_keys}"
        )

        bootstrap_keys = [
            k for k in perf if isinstance(k, str) and re.match(r"^\d+_\d+$", k)
        ]
        assert bootstrap_keys == [], (
            f"Expected no bootstrap keys, found: {bootstrap_keys}"
        )

    def test_aggregate_scores_from_bootstrap_only_keys(self) -> None:
        """aggregate_scores correctly computes mean/std from bootstrap-only keys."""
        perf_dict: dict = {
            "0_0": {"endpoint": {"r2": 0.8}},
            "0_1": {"endpoint": {"r2": 0.9}},
            "1_0": {"endpoint": {"r2": 0.7}},
            "1_1": {"endpoint": {"r2": 0.6}},
        }

        mean_result = aggregate_scores(perf_dict, "mean")
        assert abs(mean_result["endpoint"]["r2"] - 0.75) < 1e-9

        std_result = aggregate_scores(perf_dict, "std")
        expected_std = float(np.std([0.8, 0.9, 0.7, 0.6]))
        assert abs(std_result["endpoint"]["r2"] - expected_std) < 1e-9


class TestEvaluateConfigPathAutotuneFormat:
    """evaluate.main() uses from_config() for nested autotune YAML format (issue #470)."""

    def _make_cfg(
        self, tmp_path: Path, config_path: Path | None = None
    ) -> CLIEvaluationInputModel:
        return CLIEvaluationInputModel.model_validate(
            {
                "dataset": {
                    "path": str(TESTING_DATA),
                    "label_key": "Regression",
                    "smiles_key": "SMILES",
                },
                "model": {
                    "architecture": "GINRegressor",
                    "params": dict(_YAML_PARAMS),
                    "metadata": {
                        "model_name": "test",
                        "model_version": 1,
                        "model_scope": "test",
                        "model_owner": "test",
                    },
                    "config_path": str(config_path)
                    if config_path is not None
                    else None,
                },
                "output": {
                    "serialization": {"path": str(tmp_path)},
                },
                "split": {"method": "cv", "n_subset": 2},
            }
        )

    def _run_autotune_main(self, cfg: CLIEvaluationInputModel) -> list[dict]:
        """Run evaluate.main() with autotune config; return configs passed to from_config."""
        captured_configs: list[dict] = []

        def fake_from_config(config: dict) -> MagicMock:
            captured_configs.append(
                {
                    "model": dict(config.get("model") or {}),
                    "datamodule": dict(config.get("datamodule") or {}),
                    "training": dict(config.get("training") or {}),
                }
            )
            mock = MagicMock()
            mock.params.metadata.model_type = "GINRegressor"
            mock.predict.return_value = np.ones((10, 1))
            return mock

        fake_arch_cls = MagicMock()
        fake_arch_cls.from_config = fake_from_config

        mock_registry = MagicMock()
        mock_registry.__getitem__ = MagicMock(return_value=fake_arch_cls)

        fake_df = pd.DataFrame({"SMILES": ["c1ccccc1"] * 10, "Regression": np.ones(10)})
        fake_mols: list[Mol] = [Chem.MolFromSmiles("c1ccccc1")] * 10
        fake_y = np.ones((10, 1))

        with (
            patch("matcha.cli.evaluate.ScikitLearnModelRegistry", mock_registry),
            patch(
                "matcha.cli.evaluate.get_splits",
                return_value=([fake_df, fake_df], [fake_df, fake_df]),
            ),
            patch(
                "matcha.cli.evaluate.parse_df",
                return_value=(fake_mols, fake_y, None),
            ),
            patch(
                "matcha.cli.evaluate.process_regression",
                return_value={"r2": 0.9},
            ),
            patch("matcha.cli.evaluate.plot_regression", return_value=MagicMock()),
            patch("matcha.cli.evaluate.save_plot"),
            patch("matcha.cli.evaluate.aggregate_scores", return_value={}),
            patch("matcha.cli.evaluate.save_json"),
            patch("matcha.cli.evaluate.save_config_as_yaml"),
        ):
            main(cfg)

        return captured_configs

    def test_autotune_yaml_format_succeeds(self, tmp_path: Path) -> None:
        """Nested autotune YAML triggers from_config(); model section keys reach the config."""
        yaml_file = tmp_path / "autotune_output.yaml"
        yaml_file.write_text(yaml.dump(_AUTOTUNE_YAML))

        cfg = self._make_cfg(tmp_path, config_path=yaml_file)
        configs = self._run_autotune_main(cfg)

        assert len(configs) == 2
        for config in configs:
            assert (
                config["model"]["enc_num_layers"]
                == _AUTOTUNE_YAML["model"]["enc_num_layers"]
            )
            assert (
                config["model"]["enc_atom_hidden_dim"]
                == _AUTOTUNE_YAML["model"]["enc_atom_hidden_dim"]
            )

    def test_autotune_yaml_label_encoder_params_preserved(self, tmp_path: Path) -> None:
        """label_encoder_params from the base YAML config is set in datamodule before from_config."""
        yaml_file = tmp_path / "autotune_output.yaml"
        yaml_file.write_text(yaml.dump(_AUTOTUNE_YAML))

        cfg = self._make_cfg(tmp_path, config_path=yaml_file)
        configs = self._run_autotune_main(cfg)

        expected_lep = _YAML_PARAMS["label_encoder_params"]
        for config in configs:
            assert config["datamodule"]["label_encoder_params"] == expected_lep

    def test_seed_injected_per_split(self, tmp_path: Path) -> None:
        """training.seed is set to the split index when loading from autotune config."""
        yaml_file = tmp_path / "autotune_output.yaml"
        yaml_file.write_text(yaml.dump(_AUTOTUNE_YAML))

        cfg = self._make_cfg(tmp_path, config_path=yaml_file)
        configs = self._run_autotune_main(cfg)

        assert len(configs) == 2
        assert configs[0]["training"]["seed"] == 0
        assert configs[1]["training"]["seed"] == 1
