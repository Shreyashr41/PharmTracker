"""
Data models for PharmaTrack inventory management.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Optional
import pandas as pd


@dataclass
class Medicine:
    """Medicine master data."""
    medicine_id: str
    name: str
    category: str
    manufacturer: str
    unit_selling_price: float
    reorder_level: int
    unit_of_measure: str

    @classmethod
    def from_series(cls, series: pd.Series) -> "Medicine":
        return cls(
            medicine_id=series["medicine_id"],
            name=series["name"],
            category=series["category"],
            manufacturer=series["manufacturer"],
            unit_selling_price=float(series["unit_selling_price"]),
            reorder_level=int(series["reorder_level"]),
            unit_of_measure=series["unit_of_measure"],
        )


@dataclass
class Batch:
    """Batch-level inventory record."""
    batch_id: str
    medicine_id: str
    quantity_received: int
    quantity_remaining: int
    purchase_cost_per_unit: float
    manufacture_date: pd.Timestamp
    expiry_date: pd.Timestamp
    supplier_id: str
    supplier_name: str
    # Computed fields (populated by InventoryManager)
    is_expired: bool = False
    days_to_expiry: int = 0
    saleable_quantity: int = 0
    batch_value_at_cost: float = 0.0

    @classmethod
    def from_series(cls, series: pd.Series) -> "Batch":
        return cls(
            batch_id=series["batch_id"],
            medicine_id=series["medicine_id"],
            quantity_received=int(series["quantity_received"]),
            quantity_remaining=int(series["quantity_remaining"]),
            purchase_cost_per_unit=float(series["purchase_cost_per_unit"]),
            manufacture_date=series["manufacture_date"],
            expiry_date=series["expiry_date"],
            supplier_id=series["supplier_id"],
            supplier_name=series["supplier_name"],
        )

    def update_computed_fields(self) -> None:
        """Update computed fields based on current date."""
        from pharmatrack.utils.dates import is_expired, days_until_expiry

        self.is_expired = is_expired(self.expiry_date)
        self.days_to_expiry = days_until_expiry(self.expiry_date)
        self.saleable_quantity = 0 if self.is_expired else self.quantity_remaining
        self.batch_value_at_cost = self.saleable_quantity * self.purchase_cost_per_unit


@dataclass
class InventorySnapshot:
    """Aggregated inventory view for a medicine."""
    medicine_id: str
    name: str
    category: str
    total_received: int
    total_remaining: int
    total_saleable: int
    expired_quantity: int
    near_expiry_quantity: int
    batches_count: int
    expired_batches_count: int
    near_expiry_batches_count: int
    total_value_at_cost: float
    total_value_at_price: float
    weighted_avg_purchase_cost: float
    unit_selling_price: float
    reorder_level: int
    is_low_stock: bool = False

    @property
    def potential_revenue(self) -> float:
        """Potential revenue from saleable stock at selling price."""
        return self.total_saleable * self.unit_selling_price

    @property
    def gross_profit_potential(self) -> float:
        """Potential gross profit from saleable stock."""
        return self.potential_revenue - self.total_value_at_cost


def create_inventory_snapshot(
    medicine: Medicine,
    batches: list[Batch],
    near_expiry_threshold: int = 30,
) -> InventorySnapshot:
    """
    Create an aggregated inventory snapshot for a medicine from its batches.
    """
    total_received = sum(b.quantity_received for b in batches)
    total_remaining = sum(b.quantity_remaining for b in batches)
    total_saleable = sum(b.saleable_quantity for b in batches)
    expired_qty = sum(b.quantity_remaining for b in batches if b.is_expired)
    near_expiry_qty = sum(
        b.quantity_remaining for b in batches
        if not b.is_expired and b.days_to_expiry <= near_expiry_threshold
    )
    batches_count = len(batches)
    expired_batches = sum(1 for b in batches if b.is_expired)
    near_expiry_batches = sum(
        1 for b in batches if not b.is_expired and b.days_to_expiry <= near_expiry_threshold
    )
    total_value_at_cost = sum(b.batch_value_at_cost for b in batches)

    weighted_avg_cost = 0.0
    if total_saleable > 0:
        weighted_avg_cost = total_value_at_cost / total_saleable

    total_value_at_price = total_saleable * medicine.unit_selling_price
    is_low_stock = total_saleable <= medicine.reorder_level

    return InventorySnapshot(
        medicine_id=medicine.medicine_id,
        name=medicine.name,
        category=medicine.category,
        total_received=total_received,
        total_remaining=total_remaining,
        total_saleable=total_saleable,
        expired_quantity=expired_qty,
        near_expiry_quantity=near_expiry_qty,
        batches_count=batches_count,
        expired_batches_count=expired_batches,
        near_expiry_batches_count=near_expiry_batches,
        total_value_at_cost=total_value_at_cost,
        total_value_at_price=total_value_at_price,
        weighted_avg_purchase_cost=weighted_avg_cost,
        unit_selling_price=medicine.unit_selling_price,
        reorder_level=medicine.reorder_level,
        is_low_stock=is_low_stock,
    )