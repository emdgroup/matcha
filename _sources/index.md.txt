<!-- markdownlint-disable MD033 MD041 -->
<p align="center">
  <img src="_static/matcha_logo.png" alt="MATCHA Logo" width="75%">
</p>

# MATCHA — Modelling and AI Toolkit for Chemistry in Healthcare Applications

![Python](https://img.shields.io/badge/python-3.12-blue)
[![Release](https://github.com/emdgroup/matcha/actions/workflows/release.yml/badge.svg)](https://github.com/emdgroup/matcha/actions/workflows/release.yml)
[![Unit tests](https://github.com/emdgroup/matcha/actions/workflows/tests.yml/badge.svg)](https://github.com/emdgroup/matcha/actions/workflows/tests.yml)
[![Docs](https://github.com/emdgroup/matcha/actions/workflows/docs.yml/badge.svg)](https://emdgroup.github.io/matcha/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](./LICENSE)

MATCHA is a general toolkit for molecular property prediction with deep learning models.

Its core design philosophy is to wrap a wide variety of algorithms (e.g. descriptor/fingerprints, SMILES-based, graph-based, conformer-based) into a consistent Python APIs and CLI. Additionally, the package aims to be as agent-friendly as possible, with extensive docs to help LLMs use it and contribute to it.

MATCHA can help solve many typical issues found when developing QSAR models, such as:

- ⚖️ Quickly comparing many different algorithms to check for statistically meaningful differences
- 🎯 Pretraining and finetuning models, both supervised and self-supervised
- 🧩 Multitask learning
- 🔍 Explainability
- 📊 Uncertainty estimation

```{toctree}
:maxdepth: 1
:caption: Contents

installation
getting-started
contributing/adding-a-model
contributing/adding-a-fingerprint
contributing/adding-a-loss
api/index
```
