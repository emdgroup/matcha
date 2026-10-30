"""Regression tests for train.main() honouring model.config_path (issue #470, stage 2)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import yaml
from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from matcha.cli.train import main
from matcha.utils.schemas.cli import CLITrainInputModel

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


class TestTrainNoConfigPath:
    """train.main() uses base YAML params when config_path is not set."""

    def _make_cfg(self, tmp_path: Path) -> CLITrainInputModel:
        return CLITrainInputModel.model_validate(
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
            }
        )

    def _run_main(self, cfg: CLITrainInputModel) -> dict:
        captured_kwargs: list[dict] = []

        def fake_model_cls(**kwargs: object) -> MagicMock:
            captured_kwargs.append(dict(kwargs))
            return MagicMock()

        mock_registry = MagicMock()
        mock_registry.__getitem__ = MagicMock(return_value=fake_model_cls)

        fake_mols: list[Mol] = [Chem.MolFromSmiles("c1ccccc1")] * 10
        fake_y = np.ones((10, 1))

        with (
            patch("matcha.cli.train.ScikitLearnModelRegistry", mock_registry),
            patch("matcha.cli.train.load_dataset", return_value=MagicMock()),
            patch("matcha.cli.train.parse_df", return_value=(fake_mols, fake_y, None)),
            patch("matcha.cli.train.save_config_as_yaml"),
        ):
            main(cfg)

        return captured_kwargs[0] if captured_kwargs else {}

    def test_uses_yaml_params(self, tmp_path: Path) -> None:
        """Without config_path, model is instantiated with the original YAML params."""
        kwargs = self._run_main(self._make_cfg(tmp_path))

        assert kwargs["enc_num_layers"] == _YAML_PARAMS["enc_num_layers"]
        assert kwargs["enc_atom_hidden_dim"] == _YAML_PARAMS["enc_atom_hidden_dim"]


class TestTrainConfigPathAutotuneFormat:
    """train.main() uses from_config() for nested autotune YAML format (issue #470)."""

    def _make_cfg(
        self, tmp_path: Path, config_path: Path | None = None
    ) -> CLITrainInputModel:
        return CLITrainInputModel.model_validate(
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
            }
        )

    def _run_autotune_main(self, cfg: CLITrainInputModel) -> dict:
        """Run train.main() with autotune config; return the config passed to from_config."""
        captured_configs: list[dict] = []

        def fake_from_config(config: dict) -> MagicMock:
            captured_configs.append(
                {
                    "model": dict(config.get("model") or {}),
                    "datamodule": dict(config.get("datamodule") or {}),
                    "training": dict(config.get("training") or {}),
                }
            )
            return MagicMock()

        fake_arch_cls = MagicMock()
        fake_arch_cls.from_config = fake_from_config

        mock_registry = MagicMock()
        mock_registry.__getitem__ = MagicMock(return_value=fake_arch_cls)

        fake_mols: list[Mol] = [Chem.MolFromSmiles("c1ccccc1")] * 10
        fake_y = np.ones((10, 1))

        with (
            patch("matcha.cli.train.ScikitLearnModelRegistry", mock_registry),
            patch("matcha.cli.train.load_dataset", return_value=MagicMock()),
            patch(
                "matcha.cli.train.parse_df",
                return_value=(fake_mols, fake_y, None),
            ),
            patch("matcha.cli.train.save_config_as_yaml"),
        ):
            main(cfg)

        return captured_configs[0] if captured_configs else {}

    def test_autotune_yaml_format_succeeds(self, tmp_path: Path) -> None:
        """Nested autotune YAML triggers from_config(); model section keys reach the config."""
        yaml_file = tmp_path / "autotune_output.yaml"
        yaml_file.write_text(yaml.dump(_AUTOTUNE_YAML))

        cfg = self._make_cfg(tmp_path, config_path=yaml_file)
        config = self._run_autotune_main(cfg)

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
        config = self._run_autotune_main(cfg)

        expected_lep = _YAML_PARAMS["label_encoder_params"]
        assert config["datamodule"]["label_encoder_params"] == expected_lep
