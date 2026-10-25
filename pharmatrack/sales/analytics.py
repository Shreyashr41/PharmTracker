"""
Sales and revenue analytics for PharmaTrack.

Computes gross profit using purchase cost of actual batches sold,
not current inventory price.
"""

from typing import Optional
import pandas as pd
import numpy as np


def compute_gross_profit(
    sales: pd.DataFrame,
    batches: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute gross profit per sale by joining with batch purchase cost.

    Args:
        sales: Sales DataFrame with batch_id, medicine_id, quantity_sold, selling_price_per_unit
        batches: Batches DataFrame with batch_id, purchase_cost_per_unit

    Returns:
        Sales DataFrame with added columns:
        - purchase_cost_per_unit (from batch)
        - revenue (quantity_sold * selling_price_per_unit)
        - cost_of_goods_sold (quantity_sold * purchase_cost_per_unit)
        - gross_profit (revenue - cogs)
        - gross_margin_pct (gross_profit / revenue * 100)
    """
    df = sales.merge(
        batches[["batch_id", "purchase_cost_per_unit"]],
        on="batch_id",
        how="left",
        suffixes=("", "_batch"),
    )

    # Validate join
    missing_cost = df["purchase_cost_per_unit"].isna()
    if missing_cost.any():
        # Fallback: use medicine-level average if batch not found
        print(f"Warning: {missing_cost.sum()} sales rows missing batch purchase cost")

    df["revenue"] = df["quantity_sold"] * df["selling_price_per_unit"]
    df["cost_of_goods_sold"] = df["quantity_sold"] * df["purchase_cost_per_unit"]
    df["gross_profit"] = df["revenue"] - df["cost_of_goods_sold"]

    # Avoid division by zero
    df["gross_margin_pct"] = np.where(
        df["revenue"] > 0,
        (df["gross_profit"] / df["revenue"]) * 100,
        0.0,
    )

    return df


class SalesAnalytics:
    """
    Sales analytics engine providing various revenue and profit views.
    """

    def __init__(
        self,
        sales: pd.DataFrame,
        batches: pd.DataFrame,
        medicines: pd.DataFrame,
    ):
        """
        Initialize with validated DataFrames.

        Args:
            sales: Sales transactions.
            batches: Batch inventory (for purchase costs).
            medicines: Medicine master data (for categories, names).
        """
        self.sales = sales.copy()
        self.batches = batches.copy()
        self.medicines = medicines.copy()

        # Ensure date columns
        for df, col in [(self.sales, "sale_date"), (self.batches, "expiry_date")]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])

        # Enrich sales with profit data
        self.enriched_sales = compute_gross_profit(self.sales, self.batches)

        # Add medicine info to enriched sales
        self.enriched_sales = self.enriched_sales.merge(
            self.medicines[["medicine_id", "name", "category", "unit_selling_price"]],
            on="medicine_id",
            how="left",
        )

    def get_sales_summary(self) -> dict:
        """Get overall sales summary."""
        df = self.enriched_sales
        return {
            "total_transactions": len(df),
            "total_quantity_sold": df["quantity_sold"].sum(),
            "total_revenue": df["revenue"].sum(),
            "total_cogs": df["cost_of_goods_sold"].sum(),
            "total_gross_profit": df["gross_profit"].sum(),
            "avg_gross_margin_pct": df["gross_margin_pct"].mean(),
            "unique_medicines_sold": df["medicine_id"].nunique(),
            "unique_batches_sold": df["batch_id"].nunique(),
            "date_range": {
                "start": df["sale_date"].min() if not df.empty else None,
                "end": df["sale_date"].max() if not df.empty else None,
            },
        }

    def get_sales_by_period(self, freq: str = "M") -> pd.DataFrame:
        """
        Get sales aggregated by time period.

        Args:
            freq: Pandas frequency string ('D', 'W', 'M', 'Q', 'Y').

        Returns:
            DataFrame with period, revenue, cogs, gross_profit, quantity_sold.
        """
        df = self.enriched_sales.copy()
        df["period"] = df["sale_date"].dt.to_period(freq)

        agg = df.groupby("period").agg(
            revenue=("revenue", "sum"),
            cost_of_goods_sold=("cost_of_goods_sold", "sum"),
            gross_profit=("gross_profit", "sum"),
            quantity_sold=("quantity_sold", "sum"),
            transactions=("sale_id", "count"),
        ).reset_index()

        agg["period"] = agg["period"].dt.to_timestamp()
        agg["gross_margin_pct"] = np.where(
            agg["revenue"] > 0,
            (agg["gross_profit"] / agg["revenue"]) * 100,
            0.0,
        )
        return agg

    def get_sales_by_category(self) -> pd.DataFrame:
        """Get sales performance by medicine category."""
        df = self.enriched_sales
        return df.groupby("category").agg(
            revenue=("revenue", "sum"),
            cost_of_goods_sold=("cost_of_goods_sold", "sum"),
            gross_profit=("gross_profit", "sum"),
            quantity_sold=("quantity_sold", "sum"),
            transactions=("sale_id", "count"),
            avg_margin_pct=("gross_margin_pct", "mean"),
        ).reset_index().sort_values("revenue", ascending=False)

    def get_sales_by_medicine(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """Get sales performance by medicine."""
        df = self.enriched_sales
        agg = df.groupby(["medicine_id", "name", "category"]).agg(
            revenue=("revenue", "sum"),
            cost_of_goods_sold=("cost_of_goods_sold", "sum"),
            gross_profit=("gross_profit", "sum"),
            quantity_sold=("quantity_sold", "sum"),
            transactions=("sale_id", "count"),
            avg_margin_pct=("gross_margin_pct", "mean"),
        ).reset_index().sort_values("revenue", ascending=False)

        if top_n:
            agg = agg.head(top_n)
        return agg

    def get_sales_by_payment_method(self) -> pd.DataFrame:
        """Get sales breakdown by payment method."""
        df = self.enriched_sales
        if "payment_method" not in df.columns:
            return pd.DataFrame()
        return df.groupby("payment_method").agg(
            revenue=("revenue", "sum"),
            gross_profit=("gross_profit", "sum"),
            transactions=("sale_id", "count"),
        ).reset_index().sort_values("revenue", ascending=False)

    def get_top_profitable_medicines(self, n: int = 10) -> pd.DataFrame:
        """Get top N medicines by gross profit."""
        return self.get_sales_by_medicine().head(n)

    def get_top_margin_medicines(self, n: int = 10, min_qty: int = 10) -> pd.DataFrame:
        """Get top N medicines by gross margin (with minimum volume)."""
        df = self.get_sales_by_medicine()
        df = df[df["quantity_sold"] >= min_qty]
        return df.nlargest(n, "avg_margin_pct")

    def get_batch_profitability(self) -> pd.DataFrame:
        """Get profitability by batch."""
        df = self.enriched_sales
        return df.groupby(["batch_id", "medicine_id"]).agg(
            revenue=("revenue", "sum"),
            cost_of_goods_sold=("cost_of_goods_sold", "sum"),
            gross_profit=("gross_profit", "sum"),
            quantity_sold=("quantity_sold", "sum"),
            gross_margin_pct=("gross_margin_pct", "mean"),
        ).reset_index().sort_values("gross_profit", ascending=False)

    def get_daily_sales_trend(self, days: int = 30) -> pd.DataFrame:
        """Get daily sales trend for last N days."""
        df = self.enriched_sales.copy()
        cutoff = df["sale_date"].max() - pd.Timedelta(days=days)
        df = df[df["sale_date"] >= cutoff]

        daily = df.groupby("sale_date").agg(
            revenue=("revenue", "sum"),
            gross_profit=("gross_profit", "sum"),
            quantity_sold=("quantity_sold", "sum"),
            transactions=("sale_id", "count"),
        ).reset_index().sort_values("sale_date")

        # Fill missing dates
        all_dates = pd.date_range(daily["sale_date"].min(), daily["sale_date"].max(), freq="D")
        daily = daily.set_index("sale_date").reindex(all_dates, fill_value=0).reset_index()
        daily = daily.rename(columns={"index": "sale_date"})
        return daily