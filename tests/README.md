# Tests

## `testing_data.csv` — provenance

The molecule fixture shared across the test suite is the `CHEMBL1862_Ki.csv`
benchmark from [MoleculeACE](https://github.com/molML/MoleculeACE) (MIT
licensed), 794 SMILES for tyrosine-protein kinase ABL (ChEMBL target
`CHEMBL1862`) with Ki measurements curated from ChEMBL.

Column mapping:

| `testing_data.csv` | MoleculeACE source | Meaning                                                    |
| ------------------ | ------------------ | ---------------------------------------------------------- |
| `SMILES`           | `smiles`           | Canonical SMILES                                           |
| `Regression`       | `y`                | Log-transformed Ki (regression target)                     |
| `Classification`   | `cliff_mol`        | Activity-cliff membership flag (binary classification)     |

Reference: van Tilborg, Alenicheva & Grisoni, *Exposing the Limitations of
Molecular Machine Learning with Activity Cliffs*, J. Chem. Inf. Model. **62**
(2022) 5938–5951. [doi:10.1021/acs.jcim.2c01073](https://doi.org/10.1021/acs.jcim.2c01073).

The dataset is public and license-compatible; no internal or proprietary
compound data is used anywhere in the test suite.
