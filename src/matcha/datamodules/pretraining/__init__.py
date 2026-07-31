"""Pretraining datamodules for self-supervised learning."""

from matcha.datamodules.pretraining.clm_mlm_datamodule import CLMMLMDataModule
from matcha.datamodules.pretraining.on_the_fly_datamodule import (
    OnTheFlyDataModule,
    OnTheFlyDataset,
)
from matcha.datamodules.pretraining.on_the_fly_mlm_datamodule import (
    OnTheFlyMLMDataModule,
    OnTheFlyMLMDataset,
)
from matcha.datamodules.pretraining.graph_pretraining_datamodule import (
    GraphPretrainingDataModule,
)
from matcha.datamodules.pretraining.on_the_fly_graph_pretraining_datamodule import (
    OnTheFlyGraphPretrainingDataModule,
    OnTheFlyGraphPretrainingDataset,
)

__all__ = [
    "CLMMLMDataModule",
    "OnTheFlyDataModule",
    "OnTheFlyDataset",
    "OnTheFlyMLMDataModule",
    "OnTheFlyMLMDataset",
    "GraphPretrainingDataModule",
    "OnTheFlyGraphPretrainingDataModule",
    "OnTheFlyGraphPretrainingDataset",
]
