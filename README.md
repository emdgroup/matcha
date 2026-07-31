<!-- markdownlint-disable MD033 MD041 -->
<p align="center">
  <img src="docs/source/_static/matcha_logo.png" alt="MATCHA Logo" width="75%">
</p>

# MATCHA - Modelling and AI Toolkit for Chemistry in Healthcare Applications

![Python](https://img.shields.io/badge/python-3.12-blue)
[![Release](https://github.com/emdgroup/matcha/actions/workflows/release.yml/badge.svg)](https://github.com/emdgroup/matcha/actions/workflows/release.yml)
[![Unit tests](https://github.com/emdgroup/matcha/actions/workflows/tests.yml/badge.svg)](https://github.com/emdgroup/matcha/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/emdgroup/matcha/branch/main/graph/badge.svg)](https://codecov.io/gh/emdgroup/matcha)
[![Docs](https://github.com/emdgroup/matcha/actions/workflows/docs.yml/badge.svg)](https://emdgroup.github.io/matcha/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](./LICENSE)
[![ChemRxiv](https://img.shields.io/badge/ChemRxiv-preprint-b31b1b)](https://chemrxiv.org/doi/abs/10.26434/chemrxiv.15006766/v1)

> ⚠️ **Work in progress.** MATCHA is under active development and no stable release is available yet. Public APIs, CLI commands, and packaging may change without notice. Pin exact versions if you depend on it, and expect breaking changes between releases.

MATCHA is a general toolkit for molecular property prediction with deep learning models.

Its core design philosophy is to wrap a wide variety of algorithms (e.g. descriptor/fingerprints, SMILES-based, graph-based, conformer-based) into a consistent Python APIs and CLI. Additionally, the package aims to be as agent-friendly as possible, with extensive docs to help LLMs use it and contribute to it.

MATCHA can help solve many typical issues found when developing QSAR models, such as:

- ⚖️ Quickly comparing many different algorithms to check for statistically meaningful differences
- 🎯 Pretraining and finetuning models, both supervised and self-supervised
- 🧩 Multitask learning
- 🔍 Explainability
- 📊 Uncertainty estimation

Making models, whether single predictors, ensembles or finetuned foundation models, can be done with two lines of code, starting from RDKIT molecules and numpy arrays:

```python
# training and predicting with a generic model
from matcha.sklearn.graph import ChempropRegressor

model = ChempropRegressor()
model.fit(mols, y)
predictions = model.predict(mols)

# training an ensemble
from matcha.sklearn import Ensemble
from matcha.sklearn.clm import CNNRegressor

ensemble = Ensemble(
  model=CNNRegressor(),
  n_models=10
)
ensemble.fit(mols, y)

# Finetuning a foundation model
from matcha.sklearn.finetuning import FinetuningRegressor

finetuner = FinetuningRegressor(
  path_to_pretrained = "path/to/your/matcha/foundation/model"
)
finetuner.fit(mols, y)
```

Head over to the [documentation](https://emdgroup.github.io/matcha/) to get started!

## 🛠️ How to contribute

Please check out the [contribution guide](./CONTRIBUTING.md). We strongly recommend using
the amazing [Mach10](https://github.com/LeanAndMean/mach10) coding workflows to help contributing
to the codebase with agents.

## 👨🏻‍🔧 Contacts

- Davide Boldini, (Merck KGaA, Darmstadt, Germany), [Contact](mailto:davide.boldini@merckgroup.com)
- Lukas Friedrich (Merck KGaA, Darmstadt, Germany), [Contact](mailto:lukas.friedrich@merckgroup.com)
- Jakub Gunera (Merck KGaA, Darmstadt, Germany), [Contact](mailto:jakub.gunera@merckgroup.com)
- Christina Schindler (Merck KGaA, Darmstadt, Germany), [Contact](mailto:christina.schindler@merckgroup.com)
- Daniel Kuhn (Merck KGaA, Darmstadt, Germany), [Contact](mailto:daniel.kuhn@merckgroup.com)

## 📖 Citation

If you use MATCHA, please consider citing [our preprint](https://chemrxiv.org/doi/abs/10.26434/chemrxiv.15006766/v1):

```bibtex
@article{matcha_2026,
  author = "Boldini, Davide and Friedrich, Lukas and
            Gunera, Jakub and Schindler, Christina and Kuhn, Daniel",
  title  = "{MATCHA}: a toolkit for streamlining molecular property prediction
            for drug discovery applications",
  journal = "ChemRxiv",
  year = "2026",
  doi = "10.26434/chemrxiv.15006766/v1",
  url = "https://chemrxiv.org/doi/abs/10.26434/chemrxiv.15006766/v1",
}
```

## 📄 License

Copyright 2026 Merck KGaA, Darmstadt, Germany
and/or its affiliates. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

<http://www.apache.org/licenses/LICENSE-2.0>

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
