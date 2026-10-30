import numpy as np
from rdkit.Chem.rdchem import Mol


class ClassifierContract:
    def test_fit(
        self,
        classifier_cls,
        mol_list: list[Mol],
        classification_y: np.ndarray,
        arch_kwargs,
    ):
        model = classifier_cls(**arch_kwargs[classifier_cls])

        assert model.is_fitted is False
        model.fit(mol_list, classification_y)
        assert model.is_fitted is True

    def test_predict(self, fitted_classifier, mol_list: list[Mol]):
        predictions = fitted_classifier.predict(mol_list)

        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (len(mol_list), 1)
        assert set(np.unique(predictions)) <= {0.0, 1.0}

    def test_predict_proba(self, fitted_classifier, mol_list: list[Mol]):
        probabilities = fitted_classifier.predict_proba(mol_list)

        assert isinstance(probabilities, np.ndarray)
        assert probabilities.shape == (len(mol_list), 1)
        assert np.all(np.isfinite(probabilities))
        assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


class RegressorContract:
    def test_fit(
        self,
        regressor_cls,
        mol_list: list[Mol],
        regression_y: np.ndarray,
        arch_kwargs,
    ):
        model = regressor_cls(**arch_kwargs[regressor_cls])

        assert model.is_fitted is False
        model.fit(mol_list, regression_y)
        assert model.is_fitted is True

    def test_predict(self, fitted_regressor, mol_list: list[Mol]):
        predictions = fitted_regressor.predict(mol_list)

        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (len(mol_list), 1)
        assert np.all(np.isfinite(predictions))
