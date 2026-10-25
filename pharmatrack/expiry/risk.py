"""
Expiry risk assessment utilities.

Provides risk scoring and prioritization for batch management.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional
import pandas as pd

from .analysis import RiskLevel, assess_batch_risk


@dataclass
class BatchRiskAssessment:
    """Risk assessment for a single batch."""
    batch_id: str
    medicine_id: str
    medicine_name: str
    quantity_remaining: int
    purchase_cost_per_unit: float
    expiry_date: pd.Timestamp
    days_to_expiry: int
    risk_level: RiskLevel
    value_at_risk: float
    recommended_action: str

    def to_dict(self) -> dict:
        return {
            "batch_id": self.batch_id,
            "medicine_id": self.medicine_id,
            "medicine_name": self.medicine_name,
            "quantity_remaining": self.quantity_remaining,
            "purchase_cost_per_unit": self.purchase_cost_per_unit,
            "expiry_date": self.expiry_date.strftime("%Y-%m-%d"),
            "days_to_expiry": self.days_to_expiry,
            "risk_level": self.risk_level.value,
            "value_at_risk": self.value_at_risk,
            "recommended_action": self.recommended_action,
        }


def assess_batch(
    batch_row: pd.Series,
    near_threshold: int = 30,
) -> BatchRiskAssessment:
    """
    Create a risk assessment for a batch row.

    Args:
        batch_row: DataFrame row with batch data.
        near_threshold: Near expiry threshold in days.

    Returns:
        BatchRiskAssessment object.
    """
    risk = assess_batch_risk(batch_row["expiry_date"], near_threshold)
    value_at_risk = batch_row["quantity_remaining"] * batch_row["purchase_cost_per_unit"]

    # Determine recommended action
    if risk == RiskLevel.EXPIRED:
        action = "Remove from saleable stock; process for disposal/return"
    elif risk == RiskLevel.CRITICAL:
        action = "Urgent: Promote/discount for immediate sale; contact supplier for return"
    elif risk == RiskLevel.HIGH:
        action = "Prioritize for sale; consider promotional pricing"
    elif risk == RiskLevel.MEDIUM:
        action = "Monitor closely; plan promotional activity"
    elif risk == RiskLevel.LOW:
        action = "Normal monitoring; include in regular rotation"
    else:
        action = "No immediate action needed"

    return BatchRiskAssessment(
        batch_id=batch_row["batch_id"],
        medicine_id=batch_row["medicine_id"],
        medicine_name=batch_row.get("name", "Unknown"),
        quantity_remaining=int(batch_row["quantity_remaining"]),
        purchase_cost_per_unit=float(batch_row["purchase_cost_per_unit"]),
        expiry_date=batch_row["expiry_date"],
        days_to_expiry=int(batch_row["days_to_expiry"]),
        risk_level=risk,
        value_at_risk=value_at_risk,
        recommended_action=action,
    )


def get_priority_action_list(
    batches: pd.DataFrame,
    near_threshold: int = 30,
    top_n: int = 20,
) -> list[BatchRiskAssessment]:
    """
    Get prioritized action list for batches at risk.

    Sorts by risk level (expired first) then by value at risk descending.

    Args:
        batches: Batch DataFrame.
        near_threshold: Near expiry threshold.
        top_n: Number of top items to return.

    Returns:
        List of BatchRiskAssessment sorted by priority.
    """
    df = batches[batches["quantity_remaining"] > 0].copy()
    if df.empty:
        return []

    df["is_expired"] = df["expiry_date"].apply(lambda d: d < pd.Timestamp.now().normalize())
    df["days_to_expiry"] = df["expiry_date"].apply(
        lambda d: (d - pd.Timestamp.now().normalize()).days
    )

    assessments = [assess_batch(row, near_threshold) for _, row in df.iterrows()]

    # Sort: expired first, then by risk severity, then by value at risk
    risk_order = {
        RiskLevel.EXPIRED: 0,
        RiskLevel.CRITICAL: 1,
        RiskLevel.HIGH: 2,
        RiskLevel.MEDIUM: 3,
        RiskLevel.LOW: 4,
        RiskLevel.SAFE: 5,
    }

    assessments.sort(key=lambda a: (risk_order[a.risk_level], -a.value_at_risk))

    return assessments[:top_n]