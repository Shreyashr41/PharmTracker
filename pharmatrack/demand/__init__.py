"""
Demand analysis and rule-based replenishment suggestions.
"""

from .forecasting import DemandForecaster, compute_moving_average
from .replenishment import ReplenishmentEngine, ReplenishmentSuggestion

__all__ = [
    "DemandForecaster",
    "compute_moving_average",
    "ReplenishmentEngine",
    "ReplenishmentSuggestion",
]