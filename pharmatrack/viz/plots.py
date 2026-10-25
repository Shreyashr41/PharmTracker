"""
Visualization module for PharmaTrack.

Creates charts using Matplotlib and Seaborn for dashboard and reports.
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List
import io
import base64


# Set style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 10


def fig_to_base64(fig: plt.Figure) -> str:
    """Convert matplotlib figure to base64 string for embedding in HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_str


def plot_inventory_heatmap(
    inventory_snapshots: List,
    title: str = "Inventory Heatmap: Saleable vs Expired by Medicine",
) -> str:
    """
    Create heatmap showing saleable, near-expiry, expired quantities by medicine.

    Args:
        inventory_snapshots: List of InventorySnapshot objects.
        title: Chart title.

    Returns:
        Base64 encoded PNG.
    """
    if not inventory_snapshots:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No inventory data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return fig_to_base64(fig)

    data = []
    for s in inventory_snapshots:
        data.append({
            "Medicine": s.name[:25],
            "Saleable": s.total_saleable,
            "Near Expiry": s.near_expiry_quantity,
            "Expired": s.expired_quantity,
        })
    df = pd.DataFrame(data).set_index("Medicine")

    fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.35)))
    sns.heatmap(
        df.T,
        annot=True,
        fmt="d",
        cmap="RdYlGn_r",
        center=df.values.max() / 2,
        cbar_kws={"label": "Quantity"},
        ax=ax,
    )
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_sales_trend(
    daily_sales: pd.DataFrame,
    title: str = "Daily Sales Trend (Last 30 Days)",
) -> str:
    """
    Plot daily sales trend with revenue and gross profit.

    Args:
        daily_sales: DataFrame with columns sale_date, revenue, gross_profit, quantity_sold.
        title: Chart title.

    Returns:
        Base64 encoded PNG.
    """
    if daily_sales.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No sales data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return fig_to_base64(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Revenue and profit
    axes[0].plot(daily_sales["sale_date"], daily_sales["revenue"], label="Revenue", color="#2E86AB", linewidth=2)
    axes[0].plot(daily_sales["sale_date"], daily_sales["gross_profit"], label="Gross Profit", color="#A23B72", linewidth=2)
    axes[0].fill_between(daily_sales["sale_date"], 0, daily_sales["revenue"], alpha=0.1, color="#2E86AB")
    axes[0].fill_between(daily_sales["sale_date"], 0, daily_sales["gross_profit"], alpha=0.1, color="#A23B72")
    axes[0].set_ylabel("Amount (₹)")
    axes[0].set_title(title, fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Quantity sold
    axes[1].bar(daily_sales["sale_date"], daily_sales["quantity_sold"], color="#F18F01", alpha=0.7, width=0.8)
    axes[1].set_ylabel("Units Sold")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig_to_base64(fig)


def plot_expiry_distribution(
    batches: pd.DataFrame,
    title: str = "Batch Expiry Distribution",
) -> str:
    """
    Plot histogram of days to expiry for all batches.

    Args:
        batches: DataFrame with days_to_expiry column.
        title: Chart title.

    Returns:
        Base64 encoded PNG.
    """
    if batches.empty or "days_to_expiry" not in batches.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No expiry data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return fig_to_base64(fig)

    df = batches[batches["quantity_remaining"] > 0].copy()

    fig, ax = plt.subplots(figsize=(12, 6))

    # Separate expired and non-expired
    expired = df[df["days_to_expiry"] < 0]
    valid = df[df["days_to_expiry"] >= 0]

    bins = np.arange(-180, 365, 15)

    ax.hist(expired["days_to_expiry"], bins=bins[bins < 0], alpha=0.7, color="#E74C3C", label=f"Expired ({len(expired)} batches)", edgecolor="black", linewidth=0.5)
    ax.hist(valid["days_to_expiry"], bins=bins[bins >= 0], alpha=0.7, color="#2ECC71", label=f"Valid ({len(valid)} batches)", edgecolor="black", linewidth=0.5)

    # Add threshold lines
    ax.axvline(x=0, color="black", linestyle="--", linewidth=2, label="Today")
    ax.axvline(x=30, color="#F39C12", linestyle="--", linewidth=1.5, label="30-day threshold")
    ax.axvline(x=60, color="#E67E22", linestyle="--", linewidth=1.5, label="60-day threshold")
    ax.axvline(x=90, color="#3498DB", linestyle="--", linewidth=1.5, label="90-day threshold")

    ax.set_xlabel("Days to Expiry (negative = expired)")
    ax.set_ylabel("Number of Batches")
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig_to_base64(fig)


def plot_profit_by_category(
    category_sales: pd.DataFrame,
    title: str = "Gross Profit by Category",
) -> str:
    """
    Horizontal bar chart of gross profit by category.

    Args:
        category_sales: DataFrame with category, gross_profit, revenue columns.
        title: Chart title.

    Returns:
        Base64 encoded PNG.
    """
    if category_sales.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No category data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return fig_to_base64(fig)

    df = category_sales.sort_values("gross_profit", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.5)))

    bars = ax.barh(
        df["category"],
        df["gross_profit"],
        color="#2E86AB",
        alpha=0.8,
        edgecolor="#1B4F72",
        height=0.6,
    )

    # Add revenue as secondary bar
    ax.barh(
        df["category"],
        df["revenue"],
        color="#A23B72",
        alpha=0.4,
        edgecolor="#6B1E4A",
        height=0.6,
    )

    # Add value labels
    for i, (profit, rev) in enumerate(zip(df["gross_profit"], df["revenue"])):
        ax.text(profit + max(df["gross_profit"]) * 0.01, i, f"₹{profit:,.0f}", va="center", fontsize=9)

    ax.set_xlabel("Amount (₹)")
    ax.set_title(title, fontsize=14)
    ax.legend(["Gross Profit", "Revenue"])
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    return fig_to_base64(fig)


def plot_low_stock_alert(
    low_stock_snapshots: List,
    title: str = "Low Stock Alert",
) -> str:
    """
    Bar chart showing medicines below reorder level.

    Args:
        low_stock_snapshots: List of InventorySnapshot with is_low_stock=True.
        title: Chart title.

    Returns:
        Base64 encoded PNG.
    """
    if not low_stock_snapshots:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No low stock items", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title + " - All OK", fontsize=14, color="green")
        return fig_to_base64(fig)

    data = []
    for s in low_stock_snapshots:
        data.append({
            "Medicine": s.name[:30],
            "Saleable": s.total_saleable,
            "Reorder Level": s.reorder_level,
            "Deficit": s.reorder_level - s.total_saleable,
        })
    df = pd.DataFrame(data).sort_values("Deficit", ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(5, len(df) * 0.5)))

    bars = ax.barh(
        df["Medicine"],
        df["Saleable"],
        color="#E74C3C",
        alpha=0.8,
        label="Current Saleable",
        height=0.6,
    )
    ax.barh(
        df["Medicine"],
        df["Reorder Level"],
        color="#F39C12",
        alpha=0.5,
        label="Reorder Level",
        height=0.6,
    )

    # Add deficit labels
    for i, (saleable, reorder, deficit) in enumerate(zip(df["Saleable"], df["Reorder Level"], df["Deficit"])):
        ax.text(max(saleable, reorder) + max(df["Reorder Level"]) * 0.01, i, f"Short: {deficit}", va="center", fontsize=9, color="#E74C3C")

    ax.set_xlabel("Quantity")
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    return fig_to_base64(fig)


def plot_correlation_heatmap(
    df: pd.DataFrame,
    title: str = "Feature Correlation Heatmap",
) -> str:
    """
    Seaborn correlation heatmap for numeric columns.

    Args:
        df: DataFrame with numeric columns.
        title: Chart title.

    Returns:
        Base64 encoded PNG.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty or numeric_df.shape[1] < 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "Insufficient numeric data for correlation", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return fig_to_base64(fig)

    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        cbar_kws={"label": "Correlation"},
        ax=ax,
    )
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_expiry_timeline(
    timeline: pd.DataFrame,
    title: str = "Expiry Timeline (Next 90 Days)",
) -> str:
    """
    Plot area chart of expiring quantity over time.

    Args:
        timeline: DataFrame with date, expiring_qty, expiring_value_at_cost.
        title: Chart title.

    Returns:
        Base64 encoded PNG.
    """
    if timeline.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No expiry timeline data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return fig_to_base64(fig)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.fill_between(timeline["date"], 0, timeline["expiring_qty"], alpha=0.4, color="#E74C3C", label="Expiring Qty")
    ax.plot(timeline["date"], timeline["expiring_qty"], color="#C0392B", linewidth=2)

    ax.set_xlabel("Date")
    ax.set_ylabel("Expiring Quantity")
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Format x-axis
    fig.autofmt_xdate(rotation=45)

    plt.tight_layout()
    return fig_to_base64(fig)


def plot_payment_method_breakdown(
    payment_sales: pd.DataFrame,
    title: str = "Sales by Payment Method",
) -> str:
    """
    Pie chart of revenue by payment method.

    Args:
        payment_sales: DataFrame with payment_method, revenue columns.
        title: Chart title.

    Returns:
        Base64 encoded PNG.
    """
    if payment_sales.empty or "payment_method" not in payment_sales.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No payment method data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return fig_to_base64(fig)

    fig, ax = plt.subplots(figsize=(8, 8))

    colors = sns.color_palette("Set2", n_colors=len(payment_sales))
    wedges, texts, autotexts = ax.pie(
        payment_sales["revenue"],
        labels=payment_sales["payment_method"],
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        textprops={"fontsize": 11},
    )

    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")

    ax.set_title(title, fontsize=14, pad=20)
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_top_medicines(
    medicine_sales: pd.DataFrame,
    metric: str = "gross_profit",
    top_n: int = 10,
    title: str = "Top Medicines by Gross Profit",
) -> str:
    """
    Horizontal bar chart of top medicines by metric.

    Args:
        medicine_sales: DataFrame with medicine_id, name, metric columns.
        metric: Column to plot.
        top_n: Number of top items.
        title: Chart title.

    Returns:
        Base64 encoded PNG.
    """
    if medicine_sales.empty or metric not in medicine_sales.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, f"No data for {metric}", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return fig_to_base64(fig)

    df = medicine_sales.nlargest(top_n, metric).sort_values(metric, ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.5)))

    bars = ax.barh(
        df["name"],
        df[metric],
        color="#2E86AB",
        alpha=0.8,
        edgecolor="#1B4F72",
        height=0.6,
    )

    for i, val in enumerate(df[metric]):
        ax.text(val + max(df[metric]) * 0.01, i, f"₹{val:,.0f}", va="center", fontsize=9)

    ax.set_xlabel(metric.replace("_", " ").title() + " (₹)")
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    return fig_to_base64(fig)