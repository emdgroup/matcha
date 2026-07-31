# Getting started

Models in MATCHA — whether single predictors, ensembles, or finetuned
foundation models — can be built with two lines of code, starting from RDKit
molecules and NumPy arrays.

## Train and predict with a single model

```python
from matcha.sklearn.graph import ChempropRegressor

model = ChempropRegressor()
model.fit(mols, y)
predictions = model.predict(mols)
```

## Train an ensemble

```python
from matcha.sklearn import Ensemble
from matcha.sklearn.clm import CNNRegressor

ensemble = Ensemble(
    model=CNNRegressor(),
    n_models=10,
)
ensemble.fit(mols, y)
```

## Finetune a foundation model

```python
from matcha.sklearn.finetuning import FinetuningRegressor

finetuner = FinetuningRegressor(
    path_to_pretrained="path/to/your/matcha/foundation/model",
)
finetuner.fit(mols, y)
```
