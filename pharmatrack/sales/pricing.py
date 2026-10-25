"""
Pricing engine for PharmaTrack.

Provides pricing analysis and margin calculations.
"""

from typing import Optional
import pandas as pd
import numpy as np


class PricingEngine:
    """
    Analyzes pricing effectiveness and margin performance.
    """

    def __init__(
        self,
        medicines: pd.DataFrame,
        sales: pd.DataFrame,
        batches: pd.DataFrame,
    ):
        self.medicines = medicines.copy()
        self.sales = sales.copy()
        self.batches = batches.copy()

    def get_price_vs_cost_analysis(self) -> pd.DataFrame:
        """
        Compare selling prices vs purchase costs across medicines.
        """
        # Get latest purchase cost per medicine (weighted by remaining qty)
        saleable = self.batches[~self.batches["is_expired"] if "is_expired" in self.batches.columns else True]
        if "saleable_quantity" not in saleable.columns:
            saleable["saleable_quantity"] = saleable["quantity_remaining"]

        cost_by_med = saleable.groupby("medicine_id").apply(
            lambda g: np.average(
                g["purchase_cost_per_unit"],
                weights=g["saleable_quantity"]
            ) if g["saleable_quantity"].sum() > 0 else g["purchase_cost_per_unit"].mean()
        ).reset_index(name="avg_purchase_cost")

        df = self.medicines[["medicine_id", "name", "category", "unit_selling_price"]].merge(
            cost_by_med, on="medicine_id", how="left"
        )

        df["markup_pct"] = np.where(
            df["avg_purchase_cost"] > 0,
            ((df["unit_selling_price"] - df["avg_purchase_cost"]) / df["avg_purchase_cost"]) * 100,
            0
        )
        df["margin_pct"] = np.where(
            df["unit_selling_price"] > 0,
            ((df["unit_selling_price"] - df["avg_purchase_cost"]) / df["unit_selling_price"]) * 100,
            0
        )
        return df

    def get_actual_vs_list_price(self, sales_analytics) -> pd.DataFrame:
        """
        Compare actual selling prices in transactions vs list prices.
        """
        enriched = sales_analytics.enriched_sales
        actual = enriched.groupby("medicine_id").agg(
            avg_actual_price=("selling_price_per_unit", "mean"),
            min_actual_price=("selling_price_per_unit", "min"),
            max_actual_price=("selling_price_per_unit", "max"),
            transactions=("sale_id", "count"),
        ).reset_index()

        df = self.medicines[["medicine_id", "name", "unit_selling_price"]].merge(
            actual, on="medicine_id", how="left"
        )
        df["price_variance_pct"] = np.where(
            df["unit_selling_price"] > 0,
            ((df["avg_actual_price"] - df["unit_selling_price"]) / df["unit_selling_price"]) * 100,
            0
        )
        return df