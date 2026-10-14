"""SpecRule interpretable spectral rule discovery."""

from .features import ndsi, ratio
from .search import ThresholdRule, discover_class_rules, find_best_threshold

__all__ = ["ThresholdRule", "discover_class_rules", "find_best_threshold", "ndsi", "ratio"]
__version__ = "0.1.0"
