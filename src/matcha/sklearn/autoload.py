"""Auto-loading utility for saved matcha models."""

from matcha.sklearn.base_sklearn_model import ScikitLearnModelRegistry
from matcha.sklearn import Ensemble
from typing import Any
from matcha.utils import load_yaml
import os


def autoload(path: str, accelerator: str = "cuda") -> Any:
    """Load a saved matcha model from a folder.

    :param str path: path to the saved model folder
    :param str accelerator: device to load the model onto
    :return: the loaded model instance
    """
    files = os.listdir(path)

    if "config" not in files:
        raise FileNotFoundError(
            f"No recognized serialization format found in '{path}'. "
            f"Expected a 'config/' directory."
        )

    manifest = load_yaml(os.path.join(path, "config", "manifest.yaml"))
    class_name = manifest["class_name"]

    if class_name == "Ensemble":
        model = Ensemble.from_folder(path, accelerator)
    else:
        model = ScikitLearnModelRegistry[class_name].from_folder(path, accelerator)

    return model
