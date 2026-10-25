"""
Expiry and wastage risk analysis for PharmaTrack.

Identifies expired and near-expiry batches, computes financial exposure.
"""

from enum import Enum
from typing import Optional
import pandas as pd
import numpy as np

from ..utils.dates import is_expired, is_near_expiry, days_until_expiry, get_expiry_status


class RiskLevel(Enum):
    """Risk level for batch expiry."""
    EXPIRED = "Expired"
    CRITICAL = "Critical (0-7 days)"
    HIGH = "High (8-30 days)"
    MEDIUM = "Medium (31-60 days)"
    LOW = "Low (61-90 days)"
    SAFE = "Safe (90+ days)"


def assess_batch_risk(
    expiry_date: pd.Timestamp,
    near_threshold: int = 30,
) -> RiskLevel:
    """
    Assess expiry risk level for a batch.

    Args:
        expiry_date: Batch expiry date.
        near_threshold: Days for near-expiry threshold.

    Returns:
        RiskLevel enum.
    """
    if is_expired(expiry_date):
        return RiskLevel.EXPIRED

    days = days_until_expiry(expiry_date)
    if days <= 7:
        return RiskLevel.CRITICAL
    elif days <= 30:
        return RiskLevel.HIGH
    elif days <= 60:
        return RiskLevel.MEDIUM
    elif days <= 90:
        return RiskLevel.LOW
    else:
        return RiskLevel.SAFE


def compute_financial_exposure(
    batches: pd.DataFrame,
    near_expiry_threshold: int = 30,
) -> dict:
    """
    Compute financial exposure from expired and near-expiry stock.

    Args:
        batches: Batch DataFrame with quantity_remaining, purchase_cost_per_unit,
                 unit_selling_price (optional), expiry_date, is_expired.
        near_expiry_threshold: Days ahead to consider near-expiry.

    Returns:
        Dictionary with exposure breakdown.
    """
    df = batches.copy()
    if "is_expired" not in df.columns:
        df["is_expired"] = df["expiry_date"].apply(is_expired)
    if "days_to_expiry" not in df.columns:
        df["days_to_expiry"] = df["expiry_date"].apply(days_until_expiry)

    # Only consider batches with remaining quantity
    df = df[df["quantity_remaining"] > 0]

    expired = df[df["is_expired"]]
    near_expiry = df[
        (~df["is_expired"])
        & (df["days_to_expiry"] <= near_expiry_threshold)
        & (df["days_to_expiry"] >= 0)
    ]
    healthy = df[
        (~df["is_expired"])
        & (df["days_to_expiry"] > near_expiry_threshold)
    ]

    def calc_value(subdf: pd.DataFrame, cost_col: str = "purchase_cost_per_unit") -> float:
        return (subdf["quantity_remaining"] * subdf[cost_col]).sum()

    def calc_revenue(subdf: pd.DataFrame) -> float:
        if "unit_selling_price" in subdf.columns:
            return (subdf["quantity_remaining"] * subdf["unit_selling_price"]).sum()
        return 0.0

    return {
        "expired": {
            "batches": len(expired),
            "quantity": expired["quantity_remaining"].sum(),
            "value_at_cost": calc_value(expired),
            "value_at_price": calc_revenue(expired),
        },
        "near_expiry": {
            "batches": len(near_expiry),
            "quantity": near_expiry["quantity_remaining"].sum(),
            "value_at_cost": calc_value(near_expiry),
            "value_at_price": calc_revenue(near_expiry),
        },
        "healthy": {
            "batches": len(healthy),
            "quantity": healthy["quantity_remaining"].sum(),
            "value_at_cost": calc_value(healthy),
            "value_at_price": calc_revenue(healthy),
        },
        "total_exposure_at_cost": calc_value(expired) + calc_value(near_expiry),
        "total_exposure_at_price": calc_revenue(expired) + calc_revenue(near_expiry),
    }


class ExpiryAnalyzer:
    """
    Comprehensive expiry analysis and reporting.
    """

    def __init__(
        self,
        batches: pd.DataFrame,
        medicines: Optional[pd.DataFrame] = None,
        near_expiry_threshold: int = 30,
    ):
        """
        Initialize with batch data.

        Args:
            batches: Batch inventory DataFrame.
            medicines: Optional medicine master data for enrichment.
            near_expiry_threshold: Days ahead for near-expiry flag.
        """
        self.batches = batches.copy()
        self.medicines = medicines.copy() if medicines is not None else None
        self.near_expiry_threshold = near_expiry_threshold

        # Ensure date columns
        for col in ["manufacture_date", "expiry_date"]:
            if col in self.batches.columns:
                self.batches[col] = pd.to_datetime(self.batches[col])

        # Compute risk fields
        self.batches["is_expired"] = self.batches["expiry_date"].apply(is_expired)
        self.batches["days_to_expiry"] = self.batches["expiry_date"].apply(days_until_expiry)
        self.batches["risk_level"] = self.batches["expiry_date"].apply(
            lambda d: assess_batch_risk(d, near_expiry_threshold).value
        )
        self.batches["expiry_status"] = self.batches["expiry_date"].apply(
            lambda d: get_expiry_status(d, near_expiry_threshold)
        )

        # Enrich with medicine info if available
        if self.medicines is not None:
            self.batches = self.batches.merge(
                self.medicines[["medicine_id", "name", "category", "unit_selling_price"]],
                on="medicine_id",
                how="left",
            )

    def get_expired_batches(self) -> pd.DataFrame:
        """Get all expired batches with remaining quantity."""
        return self.batches[
            (self.batches["is_expired"]) & (self.batches["quantity_remaining"] > 0)
        ].copy()

    def get_near_expiry_batches(self) -> pd.DataFrame:
        """Get near-expiry batches (not expired, within threshold)."""
        return self.batches[
            (~self.batches["is_expired"])
            & (self.batches["days_to_expiry"] <= self.near_expiry_threshold)
            & (self.batches["days_to_expiry"] >= 0)
            & (self.batches["quantity_remaining"] > 0)
        ].copy()

    def get_risk_breakdown(self) -> pd.DataFrame:
        """Get count and value by risk level."""
        df = self.batches[self.batches["quantity_remaining"] > 0]
        return df.groupby("risk_level").agg(
            batches=("batch_id", "count"),
            total_qty=("quantity_remaining", "sum"),
            value_at_cost=("quantity_remaining", lambda x: (x * df.loc[x.index, "purchase_cost_per_unit"]).sum()),
        ).reset_index()

    def get_expiry_by_category(self) -> pd.DataFrame:
        """Get expiry summary by medicine category."""
        if self.medicines is None:
            return pd.DataFrame()

        df = self.batches[self.batches["quantity_remaining"] > 0]
        return df.groupby("category").agg(
            total_batches=("batch_id", "count"),
            expired_batches=("is_expired", "sum"),
            near_expiry_batches=("days_to_expiry", lambda x: ((x <= self.near_expiry_threshold) & (x >= 0)).sum()),
            expired_qty=("quantity_remaining", lambda x: x[df.loc[x.index, "is_expired"]].sum()),
            near_expiry_qty=("quantity_remaining", lambda x: x[(~df.loc[x.index, "is_expired"]) & (x.index.isin(df[(df["days_to_expiry"] <= self.near_expiry_threshold) & (df["days_to_expiry"] >= 0)].index))].sum()),
            expired_value=("quantity_remaining", lambda x: (x[df.loc[x.index, "is_expired"]] * df.loc[x.index[df.loc[x.index, "is_expired"]], "purchase_cost_per_unit"]).sum()),
        ).reset_index()

    def get_expiry_timeline(self, days_ahead: int = 180) -> pd.DataFrame:
        """
        Get day-by-day expiry timeline.

        Returns DataFrame with date, expiring_batches, expiring_qty, expiring_value_at_cost.
        """
        today = pd.Timestamp.now().normalize()
        dates = pd.date_range(today, today + pd.Timedelta(days=days_ahead), freq="D")

        # Only non-expired batches with remaining qty
        df = self.batches[
            (~self.batches["is_expired"])
            & (self.batches["quantity_remaining"] > 0)
        ]

        timeline = []
        for d in dates:
            expiring = df[df["expiry_date"].dt.normalize() == d]
            timeline.append({
                "date": d,
                "expiring_batches": len(expiring),
                "expiring_qty": expiring["quantity_remaining"].sum(),
                "expiring_value_at_cost": (
                    expiring["quantity_remaining"] * expiring["purchase_cost_per_unit"]
                ).sum(),
            })

        return pd.DataFrame(timeline)

    def get_wastage_report(self) -> dict:
        """Generate comprehensive wastage risk report."""
        exposure = compute_financial_exposure(
            self.batches, self.near_expiry_threshold
        )

        expired_batches = self.get_expired_batches()
        near_expiry_batches = self.get_near_expiry_batches()

        return {
            "summary": exposure,
            "expired_batches_detail": expired_batches[
                ["batch_id", "medicine_id", "name", "quantity_remaining",
                 "purchase_cost_per_unit", "expiry_date", "days_to_expiry"]
            ].to_dict("records") if not expired_batches.empty else [],
            "near_expiry_batches_detail": near_expiry_batches[
                ["batch_id", "medicine_id", "name", "quantity_remaining",
                 "purchase_cost_per_unit", "expiry_date", "days_to_expiry"]
            ].to_dict("records") if not near_expiry_batches.empty else [],
            "risk_distribution": self.get_risk_breakdown().to_dict("records"),
            "category_breakdown": (
                self.get_expiry_by_category().to_dict("records")
                if self.medicines is not None else []
            ),
        }