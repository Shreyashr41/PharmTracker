"""
Rule-based replenishment suggestions for PharmaTrack.

Generates reorder recommendations based on inventory levels,
demand forecasts, and expiry risk.
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np

from ..inventory.manager import InventoryManager
from ..inventory.models import InventorySnapshot
from .forecasting import DemandForecaster


@dataclass
class ReplenishmentSuggestion:
    """Replenishment recommendation for a medicine."""
    medicine_id: str
    name: str
    category: str
    current_saleable_qty: int
    reorder_level: int
    avg_daily_demand: float
    forecast_daily_demand: float
    days_of_stock_remaining: float
    suggested_order_qty: int
    priority: str  # "URGENT", "HIGH", "MEDIUM", "LOW", "OK"
    reason: str
    estimated_stockout_date: Optional[str] = None
    near_expiry_qty: int = 0
    expired_qty: int = 0

    def to_dict(self) -> dict:
        return {
            "medicine_id": self.medicine_id,
            "name": self.name,
            "category": self.category,
            "current_saleable_qty": self.current_saleable_qty,
            "reorder_level": self.reorder_level,
            "avg_daily_demand": round(self.avg_daily_demand, 2),
            "forecast_daily_demand": round(self.forecast_daily_demand, 2),
            "days_of_stock_remaining": round(self.days_of_stock_remaining, 1),
            "suggested_order_qty": self.suggested_order_qty,
            "priority": self.priority,
            "reason": self.reason,
            "estimated_stockout_date": self.estimated_stockout_date,
            "near_expiry_qty": self.near_expiry_qty,
            "expired_qty": self.expired_qty,
        }


class ReplenishmentEngine:
    """
    Generates replenishment suggestions using rule-based logic.

    Rules:
    - If saleable_qty <= reorder_level: URGENT/HIGH priority
    - If days_of_stock < 7: HIGH priority
    - If days_of_stock < 14: MEDIUM priority
    - If near_expiry_qty > 0: factor in potential loss
    - Suggested order = max(reorder_qty, forecast_demand * lead_time * safety_factor)
    """

    def __init__(
        self,
        inventory_manager: InventoryManager,
        demand_forecaster: DemandForecaster,
        lead_time_days: int = 7,
        safety_factor: float = 1.5,
        forecast_horizon_days: int = 30,
    ):
        """
        Initialize replenishment engine.

        Args:
            inventory_manager: InventoryManager instance.
            demand_forecaster: DemandForecaster instance.
            lead_time_days: Supplier lead time in days.
            safety_factor: Safety stock multiplier.
            forecast_horizon_days: Days to forecast for planning.
        """
        self.inv = inventory_manager
        self.demand = demand_forecaster
        self.lead_time_days = lead_time_days
        self.safety_factor = safety_factor
        self.forecast_horizon_days = forecast_horizon_days

    def _calculate_priority(
        self,
        snapshot: InventorySnapshot,
        days_of_stock: float,
        near_expiry_qty: int,
    ) -> tuple[str, str]:
        """
        Determine priority and reason.

        Returns:
            (priority, reason)
        """
        if snapshot.total_saleable == 0:
            return "URGENT", "Out of stock (no saleable inventory)"
        if snapshot.total_saleable <= snapshot.reorder_level:
            return "URGENT", f"Below reorder level ({snapshot.reorder_level})"
        if days_of_stock <= 7:
            return "HIGH", f"Less than 7 days of stock ({days_of_stock:.1f} days)"
        if days_of_stock <= 14:
            return "MEDIUM", f"Less than 14 days of stock ({days_of_stock:.1f} days)"
        if near_expiry_qty > 0:
            return "MEDIUM", f"Has {near_expiry_qty} units near expiry"
        return "OK", "Adequate stock levels"

    def _estimate_stockout_date(
        self,
        current_qty: int,
        daily_demand: float,
    ) -> Optional[str]:
        """Estimate stockout date based on current stock and demand."""
        if daily_demand <= 0 or current_qty <= 0:
            return None
        days_left = current_qty / daily_demand
        stockout = pd.Timestamp.now().normalize() + pd.Timedelta(days=days_left)
        return stockout.strftime("%Y-%m-%d")

    def generate_suggestion(self, snapshot: InventorySnapshot) -> ReplenishmentSuggestion:
        """Generate replenishment suggestion for one medicine."""
        # Get demand forecast
        demand_summary = self.demand.get_demand_summary(
            snapshot.medicine_id, days=self.forecast_horizon_days
        )
        avg_daily = demand_summary["avg_daily"]
        cv = demand_summary["coefficient_of_variation"]

        # Use moving average forecast
        forecast_result = self.demand.forecast_moving_average(
            snapshot.medicine_id, window=7, forecast_days=7
        )
        forecast_daily = forecast_result.get("last_ma", avg_daily)

        # Days of stock remaining
        if forecast_daily > 0:
            days_of_stock = snapshot.total_saleable / forecast_daily
        else:
            days_of_stock = float("inf")

        # Near expiry and expired quantities
        near_expiry_qty = snapshot.near_expiry_quantity
        expired_qty = snapshot.expired_quantity

        # Calculate suggested order quantity
        # Base: cover lead time + safety stock
        lead_time_demand = forecast_daily * self.lead_time_days
        safety_stock = forecast_daily * self.lead_time_days * (self.safety_factor - 1)
        # Adjust for demand variability
        safety_stock *= (1 + cv)

        suggested_qty = int(np.ceil(lead_time_demand + safety_stock - snapshot.total_saleable))
        suggested_qty = max(suggested_qty, 0)

        # Round to reasonable pack size (e.g., multiples of 10)
        suggested_qty = int(np.ceil(suggested_qty / 10) * 10)

        priority, reason = self._calculate_priority(
            snapshot, days_of_stock, near_expiry_qty
        )

        stockout_date = self._estimate_stockout_date(
            snapshot.total_saleable, forecast_daily
        )

        return ReplenishmentSuggestion(
            medicine_id=snapshot.medicine_id,
            name=snapshot.name,
            category=snapshot.category,
            current_saleable_qty=snapshot.total_saleable,
            reorder_level=snapshot.reorder_level,
            avg_daily_demand=avg_daily,
            forecast_daily_demand=forecast_daily,
            days_of_stock_remaining=days_of_stock if days_of_stock != float("inf") else 999,
            suggested_order_qty=suggested_qty,
            priority=priority,
            reason=reason,
            estimated_stockout_date=stockout_date,
            near_expiry_qty=near_expiry_qty,
            expired_qty=expired_qty,
        )

    def generate_all_suggestions(self) -> list[ReplenishmentSuggestion]:
        """Generate suggestions for all medicines."""
        snapshots = self.inv.get_all_inventory_snapshots()
        suggestions = [self.generate_suggestion(s) for s in snapshots]

        # Sort by priority then by days of stock
        priority_order = {"URGENT": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "OK": 4}
        suggestions.sort(key=lambda s: (priority_order.get(s.priority, 5), s.days_of_stock_remaining))

        return suggestions

    def get_replenishment_summary(self) -> dict:
        """Get summary of replenishment needs."""
        suggestions = self.generate_all_suggestions()

        counts = {}
        for s in suggestions:
            counts[s.priority] = counts.get(s.priority, 0) + 1

        total_order_value = sum(
            s.suggested_order_qty * self.inv.get_medicine(s.medicine_id).unit_selling_price
            for s in suggestions if s.suggested_order_qty > 0
        )

        return {
            "total_medicines": len(suggestions),
            "by_priority": counts,
            "urgent_count": counts.get("URGENT", 0),
            "high_count": counts.get("HIGH", 0),
            "total_suggested_order_qty": sum(s.suggested_order_qty for s in suggestions),
            "estimated_order_value": total_order_value,
            "medicines_at_stockout_risk": sum(1 for s in suggestions if s.days_of_stock_remaining < 7),
        }