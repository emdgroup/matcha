"""PyTorch Lightning-based model training, tuning, and inference infrastructure."""

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "ClassicModelRegistry":
        from matcha.torch.models.classic.base_classic_model import ClassicModelRegistry

        return ClassicModelRegistry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ClassicModelRegistry"]
