"""Explainability tools for molecular property predictions.

This module provides methods to interpret and explain model predictions:

- **LIME analysis:** Identifies which molecular descriptors or fingerprint bits
  most influence a prediction via local linear surrogate models.
- **Analogue generation:** Produces structural analogues of a query molecule using
  positional analogue scanning and nitrogen walking.
- **MatchaExplainer:** High-level interface that combines LIME analysis with
  analogue generation and provides visualization via :class:`MatchaExplanation`.
"""

from matcha.explainability.analogue_generator import AnalogueGenerator
from matcha.explainability.lime import LIME
from matcha.explainability.explainer import MatchaExplainer, MatchaExplanation

__all__ = [
    "AnalogueGenerator",
    "LIME",
    "MatchaExplainer",
    "MatchaExplanation",
]
