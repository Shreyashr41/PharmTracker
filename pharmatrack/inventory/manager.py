"""
Inventory manager for batch-level operations and analytics.
"""

from typing import Optional
import pandas as pd

from .models import Medicine, Batch, InventorySnapshot, create_inventory_snapshot
from ..utils.dates import is_expired, is_near_expiry


class InventoryManager:
    """
    Manages batch-level inventory and provides analytical queries.

    Separates expired stock from saleable stock.
    All quantities and values distinguish between expired/unusable and available.
    """

    def __init__(
        self,
        medicines: pd.DataFrame,
        batches: pd.DataFrame,
        near_expiry_threshold: int = 30,
    ):
        """
        Initialize with validated DataFrames.

        Args:
            medicines: Medicine master data.
            batches: Batch inventory data (with parsed dates).
            near_expiry_threshold: Days ahead to flag as near-expiry.
        """
        self.medicines = medicines.copy()
        self.batches = batches.copy()
        self.near_expiry_threshold = near_expiry_threshold

        # Ensure date columns are datetime
        for col in ["manufacture_date", "expiry_date"]:
            if col in self.batches.columns:
                self.batches[col] = pd.to_datetime(self.batches[col])

        # Add computed fields to batches
        self.batches["is_expired"] = self.batches["expiry_date"].apply(is_expired)
        self.batches["days_to_expiry"] = self.batches["expiry_date"].apply(
            lambda d: (d - pd.Timestamp.now().normalize()).days
        )
        self.batches["saleable_quantity"] = self.batches.apply(
            lambda r: 0 if r["is_expired"] else r["quantity_remaining"], axis=1
        )
        self.batches["batch_value_at_cost"] = (
            self.batches["saleable_quantity"] * self.batches["purchase_cost_per_unit"]
        )

    def get_medicine(self, medicine_id: str) -> Optional[Medicine]:
        """Get medicine by ID."""
        row = self.medicines[self.medicines["medicine_id"] == medicine_id]
        if row.empty:
            return None
        return Medicine.from_series(row.iloc[0])

    def get_batches_for_medicine(self, medicine_id: str) -> list[Batch]:
        """Get all batches for a medicine as Batch objects."""
        batch_rows = self.batches[self.batches["medicine_id"] == medicine_id]
        batches = []
        for _, row in batch_rows.iterrows():
            batch = Batch.from_series(row)
            # Copy computed fields from the DataFrame
            batch.is_expired = bool(row.get("is_expired", False))
            batch.days_to_expiry = int(row.get("days_to_expiry", 0))
            batch.saleable_quantity = int(row.get("saleable_quantity", 0))
            batch.batch_value_at_cost = float(row.get("batch_value_at_cost", 0.0))
            batches.append(batch)
        return batches

    def get_saleable_batches(self, medicine_id: Optional[str] = None) -> pd.DataFrame:
        """Get only non-expired batches with remaining quantity."""
        df = self.batches[
            (self.batches["saleable_quantity"] > 0) & (~self.batches["is_expired"])
        ]
        if medicine_id:
            df = df[df["medicine_id"] == medicine_id]
        return df.copy()

    def get_expired_batches(self, medicine_id: Optional[str] = None) -> pd.DataFrame:
        """Get expired batches."""
        df = self.batches[self.batches["is_expired"]]
        if medicine_id:
            df = df[df["medicine_id"] == medicine_id]
        return df.copy()

    def get_near_expiry_batches(
        self, medicine_id: Optional[str] = None
    ) -> pd.DataFrame:
        """Get batches near expiry (not expired, within threshold)."""
        df = self.batches[
            (~self.batches["is_expired"])
            & (self.batches["days_to_expiry"] <= self.near_expiry_threshold)
            & (self.batches["days_to_expiry"] >= 0)
        ]
        if medicine_id:
            df = df[df["medicine_id"] == medicine_id]
        return df.copy()

    def get_inventory_snapshot(self, medicine_id: str) -> Optional[InventorySnapshot]:
        """Get aggregated inventory snapshot for a medicine."""
        medicine = self.get_medicine(medicine_id)
        if not medicine:
            return None
        batches = self.get_batches_for_medicine(medicine_id)
        return create_inventory_snapshot(
            medicine, batches, self.near_expiry_threshold
        )

    def get_all_inventory_snapshots(self) -> list[InventorySnapshot]:
        """Get inventory snapshots for all medicines."""
        snapshots = []
        for _, med_row in self.medicines.iterrows():
            med = Medicine.from_series(med_row)
            batches = self.get_batches_for_medicine(med.medicine_id)
            snap = create_inventory_snapshot(med, batches, self.near_expiry_threshold)
            snapshots.append(snap)
        return snapshots

    def get_low_stock_medicines(self) -> list[InventorySnapshot]:
        """Get medicines where saleable quantity <= reorder level."""
        return [s for s in self.get_all_inventory_snapshots() if s.is_low_stock]

    def get_total_inventory_value(self) -> dict:
        """Get total inventory value across all medicines."""
        saleable = self.batches[self.batches["saleable_quantity"] > 0]
        expired = self.batches[self.batches["is_expired"]]

        return {
            "saleable_value_at_cost": saleable["batch_value_at_cost"].sum(),
            "saleable_quantity": saleable["saleable_quantity"].sum(),
            "expired_value_at_cost": (
                expired["quantity_remaining"] * expired["purchase_cost_per_unit"]
            ).sum(),
            "expired_quantity": expired["quantity_remaining"].sum(),
            "total_batches": len(self.batches),
            "saleable_batches": len(saleable),
            "expired_batches": len(expired),
        }

    def get_category_summary(self) -> pd.DataFrame:
        """Get inventory summary by category."""
        snapshots = self.get_all_inventory_snapshots()
        if not snapshots:
            return pd.DataFrame()

        rows = []
        for s in snapshots:
            rows.append({
                "category": s.category,
                "medicines_count": 1,
                "total_saleable_qty": s.total_saleable,
                "total_expired_qty": s.expired_quantity,
                "total_near_expiry_qty": s.near_expiry_quantity,
                "value_at_cost": s.total_value_at_cost,
                "potential_revenue": s.potential_revenue,
                "gross_profit_potential": s.gross_profit_potential,
                "low_stock_count": 1 if s.is_low_stock else 0,
            })
        df = pd.DataFrame(rows)
        return df.groupby("category").sum(numeric_only=True).reset_index()

    def get_expiry_timeline(self, days_ahead: int = 90) -> pd.DataFrame:
        """
        Get expiry timeline showing quantity expiring in each period.

        Returns DataFrame with columns: period_start, period_end, expiring_qty, expiring_value.
        """
        today = pd.Timestamp.now().normalize()
        periods = []

        for i in range(0, days_ahead, 30):
            period_start = today + pd.Timedelta(days=i)
            period_end = today + pd.Timedelta(days=i + 30)

            mask = (
                (~self.batches["is_expired"])
                & (self.batches["expiry_date"] >= period_start)
                & (self.batches["expiry_date"] < period_end)
                & (self.batches["quantity_remaining"] > 0)
            )
            expiring = self.batches[mask]

            periods.append({
                "period_start": period_start,
                "period_end": period_end,
                "expiring_batches": len(expiring),
                "expiring_qty": expiring["quantity_remaining"].sum(),
                "expiring_value_at_cost": (
                    expiring["quantity_remaining"] * expiring["purchase_cost_per_unit"]
                ).sum(),
            })

        # Also add "already expired"
        expired = self.batches[self.batches["is_expired"] & (self.batches["quantity_remaining"] > 0)]
        periods.insert(0, {
            "period_start": pd.NaT,
            "period_end": today,
            "expiring_batches": len(expired),
            "expiring_qty": expired["quantity_remaining"].sum(),
            "expiring_value_at_cost": (
                expired["quantity_remaining"] * expired["purchase_cost_per_unit"]
            ).sum(),
        })

        return pd.DataFrame(periods)