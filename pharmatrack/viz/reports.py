"""
Report generation module for PharmaTrack.

Creates text/HTML summary reports from analysis results.
"""

from typing import List, Optional
from datetime import datetime
import pandas as pd

from ..inventory.models import InventorySnapshot
from ..demand.replenishment import ReplenishmentSuggestion


def generate_inventory_report(
    snapshots: List[InventorySnapshot],
    total_value: dict,
    category_summary: Optional[pd.DataFrame] = None,
) -> str:
    """
    Generate text inventory report.

    Args:
        snapshots: List of inventory snapshots.
        total_value: Dict from InventoryManager.get_total_inventory_value().
        category_summary: Optional category summary DataFrame.

    Returns:
        Formatted text report.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("PHARMATRACK - INVENTORY REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)

    # Overall summary
    lines.append("\n📦 OVERALL INVENTORY SUMMARY")
    lines.append("-" * 40)
    lines.append(f"Total Saleable Batches:     {total_value.get('saleable_batches', 0)}")
    lines.append(f"Total Saleable Quantity:    {total_value.get('saleable_quantity', 0):,}")
    lines.append(f"Saleable Value (at cost):   ₹{total_value.get('saleable_value_at_cost', 0):,.2f}")
    lines.append(f"Expired Batches:            {total_value.get('expired_batches', 0)}")
    lines.append(f"Expired Quantity:           {total_value.get('expired_quantity', 0):,}")
    lines.append(f"Expired Value (at cost):    ₹{total_value.get('expired_value_at_cost', 0):,.2f}")

    # Category breakdown
    if category_summary is not None and not category_summary.empty:
        lines.append("\n📊 BY CATEGORY")
        lines.append("-" * 40)
        for _, row in category_summary.iterrows():
            lines.append(
                f"  {row['category']:20s} | "
                f"Saleable: {row['total_saleable_qty']:>6,} | "
                f"Expired: {row['total_expired_qty']:>5,} | "
                f"Near Expiry: {row['total_near_expiry_qty']:>5,} | "
                f"Value: ₹{row['value_at_cost']:>10,.0f} | "
                f"Profit Pot.: ₹{row['gross_profit_potential']:>10,.0f}"
            )

    # Low stock alerts
    low_stock = [s for s in snapshots if s.is_low_stock]
    if low_stock:
        lines.append("\n⚠️  LOW STOCK ALERTS")
        lines.append("-" * 40)
        for s in low_stock:
            lines.append(
                f"  {s.name:30s} | Saleable: {s.total_saleable:>5} | "
                f"Reorder: {s.reorder_level:>5} | Deficit: {s.reorder_level - s.total_saleable:>5}"
            )
    else:
        lines.append("\n✅ No low stock items")

    # Expired items detail
    expired_items = [s for s in snapshots if s.expired_quantity > 0]
    if expired_items:
        lines.append("\n🚫 EXPIRED STOCK DETAIL")
        lines.append("-" * 40)
        for s in expired_items:
            lines.append(
                f"  {s.name:30s} | Expired Qty: {s.expired_quantity:>5} | "
                f"Value at Cost: ₹{s.total_value_at_cost:>10,.2f}"
            )

    lines.append("\n" + "=" * 60)
    lines.append("END OF REPORT")
    lines.append("=" * 60)

    return "\n".join(lines)


def generate_expiry_report(
    wastage_report: dict,
    risk_breakdown: Optional[pd.DataFrame] = None,
) -> str:
    """
    Generate expiry/wastage risk report.

    Args:
        wastage_report: Dict from ExpiryAnalyzer.get_wastage_report().
        risk_breakdown: Optional risk breakdown DataFrame.

    Returns:
        Formatted text report.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("PHARMATRACK - EXPIRY & WASTAGE RISK REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)

    summary = wastage_report.get("summary", {})

    lines.append("\n📋 EXPOSURE SUMMARY")
    lines.append("-" * 40)
    exp = summary.get("expired", {})
    lines.append(f"EXPIRED:     {exp.get('batches', 0)} batches | {exp.get('quantity', 0):,} units | ₹{exp.get('value_at_cost', 0):,.2f} (cost) | ₹{exp.get('value_at_price', 0):,.2f} (price)")
    near = summary.get("near_expiry", {})
    lines.append(f"NEAR EXPIRY: {near.get('batches', 0)} batches | {near.get('quantity', 0):,} units | ₹{near.get('value_at_cost', 0):,.2f} (cost) | ₹{near.get('value_at_price', 0):,.2f} (price)")
    healthy = summary.get("healthy", {})
    lines.append(f"HEALTHY:     {healthy.get('batches', 0)} batches | {healthy.get('quantity', 0):,} units | ₹{healthy.get('value_at_cost', 0):,.2f} (cost)")

    lines.append(f"\nTOTAL EXPOSURE (Cost):  ₹{summary.get('total_exposure_at_cost', 0):,.2f}")
    lines.append(f"TOTAL EXPOSURE (Price): ₹{summary.get('total_exposure_at_price', 0):,.2f}")

    # Risk distribution
    if risk_breakdown is not None and not risk_breakdown.empty:
        lines.append("\n⚠️  RISK LEVEL DISTRIBUTION")
        lines.append("-" * 40)
        for _, row in risk_breakdown.iterrows():
            lines.append(
                f"  {row['risk_level']:25s} | "
                f"Batches: {row['batches']:>3} | "
                f"Qty: {row['total_qty']:>6,} | "
                f"Value at Risk: ₹{row['value_at_cost']:>10,.2f}"
            )

    # Expired batches detail
    expired_detail = wastage_report.get("expired_batches_detail", [])
    if expired_detail:
        lines.append("\n🚫 EXPIRED BATCHES (Top 20)")
        lines.append("-" * 40)
        for b in expired_detail[:20]:
            lines.append(
                f"  {b.get('batch_id', 'N/A'):15s} | "
                f"{b.get('medicine_name', 'N/A')[:25]:25s} | "
                f"Qty: {b.get('quantity_remaining', 0):>5} | "
                f"Cost: ₹{b.get('purchase_cost_per_unit', 0):>7.2f} | "
                f"Expired: {b.get('days_to_expiry', 0):>4} days ago"
            )

    # Near expiry batches detail
    near_detail = wastage_report.get("near_expiry_batches_detail", [])
    if near_detail:
        lines.append("\n⏰ NEAR EXPIRY BATCHES (Top 20)")
        lines.append("-" * 40)
        for b in near_detail[:20]:
            lines.append(
                f"  {b.get('batch_id', 'N/A'):15s} | "
                f"{b.get('medicine_name', 'N/A')[:25]:25s} | "
                f"Qty: {b.get('quantity_remaining', 0):>5} | "
                f"Cost: ₹{b.get('purchase_cost_per_unit', 0):>7.2f} | "
                f"Expires in: {b.get('days_to_expiry', 0):>3} days"
            )

    lines.append("\n" + "=" * 60)
    lines.append("END OF REPORT")
    lines.append("=" * 60)

    return "\n".join(lines)


def generate_sales_report(
    sales_summary: dict,
    category_sales: Optional[pd.DataFrame] = None,
    top_medicines: Optional[pd.DataFrame] = None,
    payment_breakdown: Optional[pd.DataFrame] = None,
) -> str:
    """
    Generate sales performance report.

    Args:
        sales_summary: Dict from SalesAnalytics.get_sales_summary().
        category_sales: Optional category sales DataFrame.
        top_medicines: Optional top medicines DataFrame.
        payment_breakdown: Optional payment method DataFrame.

    Returns:
        Formatted text report.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("PHARMATRACK - SALES PERFORMANCE REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)

    lines.append("\n📈 OVERALL SALES SUMMARY")
    lines.append("-" * 40)
    lines.append(f"Period: {sales_summary.get('date_range', {}).get('start', 'N/A')} to {sales_summary.get('date_range', {}).get('end', 'N/A')}")
    lines.append(f"Total Transactions:       {sales_summary.get('total_transactions', 0):,}")
    lines.append(f"Total Quantity Sold:      {sales_summary.get('total_quantity_sold', 0):,}")
    lines.append(f"Total Revenue:            ₹{sales_summary.get('total_revenue', 0):,.2f}")
    lines.append(f"Total COGS:               ₹{sales_summary.get('total_cogs', 0):,.2f}")
    lines.append(f"Total Gross Profit:       ₹{sales_summary.get('total_gross_profit', 0):,.2f}")
    lines.append(f"Average Gross Margin:     {sales_summary.get('avg_gross_margin_pct', 0):.1f}%")
    lines.append(f"Unique Medicines Sold:    {sales_summary.get('unique_medicines_sold', 0)}")
    lines.append(f"Unique Batches Sold:      {sales_summary.get('unique_batches_sold', 0)}")

    if category_sales is not None and not category_sales.empty:
        lines.append("\n📊 SALES BY CATEGORY")
        lines.append("-" * 40)
        for _, row in category_sales.iterrows():
            margin = row.get('avg_margin_pct', 0)
            lines.append(
                f"  {row['category']:20s} | "
                f"Revenue: ₹{row['revenue']:>10,.0f} | "
                f"Profit: ₹{row['gross_profit']:>10,.0f} | "
                f"Margin: {margin:>5.1f}% | "
                f"Units: {row['quantity_sold']:>6,}"
            )

    if top_medicines is not None and not top_medicines.empty:
        lines.append("\n🏆 TOP 10 MEDICINES BY REVENUE")
        lines.append("-" * 40)
        for _, row in top_medicines.head(10).iterrows():
            lines.append(
                f"  {row['name'][:30]:30s} | "
                f"Revenue: ₹{row['revenue']:>10,.0f} | "
                f"Profit: ₹{row['gross_profit']:>10,.0f} | "
                f"Margin: {row['avg_margin_pct']:>5.1f}%"
            )

    if payment_breakdown is not None and not payment_breakdown.empty:
        lines.append("\n💳 SALES BY PAYMENT METHOD")
        lines.append("-" * 40)
        for _, row in payment_breakdown.iterrows():
            pct = (row['revenue'] / sales_summary.get('total_revenue', 1)) * 100
            lines.append(
                f"  {row['payment_method']:15s} | "
                f"Revenue: ₹{row['revenue']:>10,.0f} | "
                f"Profit: ₹{row['gross_profit']:>10,.0f} | "
                f"{pct:>5.1f}%"
            )

    lines.append("\n" + "=" * 60)
    lines.append("END OF REPORT")
    lines.append("=" * 60)

    return "\n".join(lines)


def generate_replenishment_report(
    suggestions: List[ReplenishmentSuggestion],
    summary: dict,
) -> str:
    """
    Generate replenishment suggestions report.

    Args:
        suggestions: List of ReplenishmentSuggestion.
        summary: Dict from ReplenishmentEngine.get_replenishment_summary().

    Returns:
        Formatted text report.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("PHARMATRACK - REPLENISHMENT RECOMMENDATIONS")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)

    lines.append("\n📋 SUMMARY")
    lines.append("-" * 40)
    lines.append(f"Total Medicines Analyzed:  {summary.get('total_medicines', 0)}")
    lines.append(f"Urgent (Out/Reorder):      {summary.get('urgent_count', 0)}")
    lines.append(f"High (< 7 days stock):     {summary.get('high_count', 0)}")
    lines.append(f"At Stockout Risk (<7d):    {summary.get('medicines_at_stockout_risk', 0)}")
    lines.append(f"Total Suggested Order Qty: {summary.get('total_suggested_order_qty', 0):,}")
    lines.append(f"Estimated Order Value:     ₹{summary.get('estimated_order_value', 0):,.2f}")

    lines.append("\n📋 DETAILED RECOMMENDATIONS")
    lines.append("-" * 40)

    for s in suggestions:
        if s.priority == "OK":
            continue  # Skip OK items in detailed view

        stockout = s.estimated_stockout_date or "N/A"
        lines.append(
            f"\n  [{s.priority}] {s.name} ({s.medicine_id})"
        )
        lines.append(
            f"    Category: {s.category} | Current: {s.current_saleable_qty} | "
            f"Reorder: {s.reorder_level} | Days of Stock: {s.days_of_stock_remaining:.1f}"
        )
        lines.append(
            f"    Avg Daily Demand: {s.avg_daily_demand:.1f} | "
            f"Forecast: {s.forecast_daily_demand:.1f} | "
            f"Est. Stockout: {stockout}"
        )
        lines.append(
            f"    Suggested Order: {s.suggested_order_qty} units | "
            f"Near Expiry: {s.near_expiry_qty} | Expired: {s.expired_qty}"
        )
        lines.append(f"    Reason: {s.reason}")

    lines.append("\n" + "=" * 60)
    lines.append("END OF REPORT")
    lines.append("=" * 60)

    return "\n".join(lines)