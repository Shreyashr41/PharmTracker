"""
PharmaTrack Streamlit Dashboard.

Interactive dashboard for pharmacy inventory, sales, and expiry analytics.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pharmatrack.data.ingestion import load_all, load_medicines, load_batches, load_sales
from pharmatrack.data.sample_gen import generate_sample_data
from pharmatrack.data.schemas import ValidationError
from pharmatrack.inventory.manager import InventoryManager
from pharmatrack.sales.analytics import SalesAnalytics
from pharmatrack.expiry.analysis import ExpiryAnalyzer
from pharmatrack.demand.forecasting import DemandForecaster
from pharmatrack.demand.replenishment import ReplenishmentEngine
from pharmatrack.viz.plots import (
    plot_inventory_heatmap,
    plot_sales_trend,
    plot_expiry_distribution,
    plot_profit_by_category,
    plot_low_stock_alert,
    plot_correlation_heatmap,
    plot_expiry_timeline,
    plot_payment_method_breakdown,
    plot_top_medicines,
)
from pharmatrack.viz.reports import (
    generate_inventory_report,
    generate_expiry_report,
    generate_sales_report,
    generate_replenishment_report,
)


# Page config
st.set_page_config(
    page_title="PharmaTrack — Pharmacy Analytics",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner="Loading data...")
def load_data(data_dir: Path):
    """Load and validate all datasets."""
    try:
        medicines, batches, sales = load_all(data_dir)
        return medicines, batches, sales, None
    except ValidationError as e:
        return None, None, None, e.errors
    except FileNotFoundError as e:
        return None, None, None, [str(e)]


def initialize_analyzers(medicines, batches, sales):
    """Create analyzer instances."""
    inv = InventoryManager(medicines, batches, near_expiry_threshold=30)
    sales_an = SalesAnalytics(sales, batches, medicines)
    expiry_an = ExpiryAnalyzer(batches, medicines, near_expiry_threshold=30)
    demand = DemandForecaster(sales)
    replenish = ReplenishmentEngine(inv, demand, lead_time_days=7, safety_factor=1.5)
    return inv, sales_an, expiry_an, demand, replenish


def main():
    st.title("💊 PharmaTrack — Smart Pharmacy Analytics")
    st.caption("Inventory • Sales • Expiry • Replenishment")

    # Sidebar: Data source selection
    with st.sidebar:
        st.header("📁 Data Source")
        data_option = st.radio(
            "Select data source:",
            ["Sample Data (Generated)", "Custom CSV Files"],
            index=0,
        )

        data_dir = None
        if data_option == "Sample Data (Generated)":
            sample_dir = PROJECT_ROOT / "data" / "sample"
            if not (sample_dir / "medicines.csv").exists():
                with st.spinner("Generating sample data..."):
                    generate_sample_data(sample_dir)
            data_dir = sample_dir
            st.success("Using generated sample data")
        else:
            custom_dir = st.text_input(
                "Path to data directory (containing medicines.csv, batches.csv, sales.csv):",
                value=str(PROJECT_ROOT / "data" / "raw"),
            )
            data_dir = Path(custom_dir)

        # Load data
        medicines, batches, sales, errors = load_data(data_dir)

        if errors:
            st.error("Data validation failed:")
            for err in errors:
                st.error(f"• {err}")
            st.stop()

        st.success(f"Loaded: {len(medicines)} medicines, {len(batches)} batches, {len(sales)} sales")

        # Configuration
        st.header("⚙️ Settings")
        near_expiry_days = st.slider(
            "Near-expiry threshold (days):",
            min_value=7,
            max_value=90,
            value=30,
            step=7,
        )

        # Re-initialize with custom threshold
        inv = InventoryManager(medicines, batches, near_expiry_threshold=near_expiry_days)
        sales_an = SalesAnalytics(sales, batches, medicines)
        expiry_an = ExpiryAnalyzer(batches, medicines, near_expiry_threshold=near_expiry_days)
        demand = DemandForecaster(sales)
        replenish = ReplenishmentEngine(inv, demand, lead_time_days=7, safety_factor=1.5)

    # Main tabs
    tabs = st.tabs([
        "📦 Inventory",
        "📈 Sales Analytics",
        "⏰ Expiry Risk",
        "🔮 Demand & Replenishment",
        "📊 Reports",
    ])

    # ==================== TAB 1: INVENTORY ====================
    with tabs[0]:
        st.header("Inventory Overview")

        # KPIs
        total_val = inv.get_total_inventory_value()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Saleable Batches", total_val["saleable_batches"])
        col2.metric("Saleable Qty", f"{total_val['saleable_quantity']:,}")
        col3.metric("Value at Cost", f"₹{total_val['saleable_value_at_cost']:,.0f}")
        col4.metric("Expired Qty", f"{total_val['expired_quantity']:,}", delta_color="inverse")

        # Inventory heatmap
        st.subheader("Inventory Composition by Medicine")
        snapshots = inv.get_all_inventory_snapshots()
        heatmap_img = plot_inventory_heatmap(snapshots)
        st.image(f"data:image/png;base64,{heatmap_img}")

        # Low stock alert
        st.subheader("⚠️ Low Stock Alerts")
        low_stock = inv.get_low_stock_medicines()
        if low_stock:
            low_stock_img = plot_low_stock_alert(low_stock)
            st.image(f"data:image/png;base64,{low_stock_img}")

            # Table
            ls_df = pd.DataFrame([{
                "Medicine": s.name,
                "Category": s.category,
                "Saleable Qty": s.total_saleable,
                "Reorder Level": s.reorder_level,
                "Deficit": s.reorder_level - s.total_saleable,
                "Expired Qty": s.expired_quantity,
                "Near Expiry Qty": s.near_expiry_quantity,
            } for s in low_stock])
            st.dataframe(ls_df, use_container_width=True, hide_index=True)
        else:
            st.success("✅ All medicines above reorder level")

        # Category summary
        st.subheader("Inventory by Category")
        cat_summary = inv.get_category_summary()
        if not cat_summary.empty:
            st.dataframe(cat_summary, use_container_width=True, hide_index=True)

        # Detailed inventory table
        with st.expander("📋 Detailed Inventory Table"):
            inv_df = pd.DataFrame([{
                "Medicine ID": s.medicine_id,
                "Name": s.name,
                "Category": s.category,
                "Total Received": s.total_received,
                "Total Remaining": s.total_remaining,
                "Saleable": s.total_saleable,
                "Expired": s.expired_quantity,
                "Near Expiry": s.near_expiry_quantity,
                "Value at Cost": f"₹{s.total_value_at_cost:,.2f}",
                "Potential Revenue": f"₹{s.potential_revenue:,.2f}",
                "Gross Profit Potential": f"₹{s.gross_profit_potential:,.2f}",
                "Low Stock": "⚠️" if s.is_low_stock else "✅",
            } for s in snapshots])
            st.dataframe(inv_df, use_container_width=True, hide_index=True)

    # ==================== TAB 2: SALES ANALYTICS ====================
    with tabs[1]:
        st.header("Sales Analytics")

        # KPIs
        sales_summary = sales_an.get_sales_summary()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Revenue", f"₹{sales_summary['total_revenue']:,.0f}")
        col2.metric("Gross Profit", f"₹{sales_summary['total_gross_profit']:,.0f}")
        col3.metric("Avg Margin", f"{sales_summary['avg_gross_margin_pct']:.1f}%")
        col4.metric("Transactions", f"{sales_summary['total_transactions']:,}")

        # Sales trend
        st.subheader("Daily Sales Trend (Last 30 Days)")
        daily_sales = sales_an.get_daily_sales_trend(30)
        trend_img = plot_sales_trend(daily_sales)
        st.image(f"data:image/png;base64,{trend_img}")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Profit by Category")
            cat_sales = sales_an.get_sales_by_category()
            if not cat_sales.empty:
                cat_img = plot_profit_by_category(cat_sales)
                st.image(f"data:image/png;base64,{cat_img}")

        with col2:
            st.subheader("Top 10 Medicines by Gross Profit")
            top_meds = sales_an.get_top_profitable_medicines(10)
            if not top_meds.empty:
                top_img = plot_top_medicines(top_meds, "gross_profit", 10, "Top 10 by Gross Profit")
                st.image(f"data:image/png;base64,{top_img}")

        # Payment method breakdown
        st.subheader("Sales by Payment Method")
        payment_sales = sales_an.get_sales_by_payment_method()
        if not payment_sales.empty:
            payment_img = plot_payment_method_breakdown(payment_sales)
            st.image(f"data:image/png;base64,{payment_img}")

        # Detailed tables
        col1, col2 = st.columns(2)
        with col1:
            with st.expander("Sales by Category"):
                st.dataframe(cat_sales, use_container_width=True, hide_index=True)
        with col2:
            with st.expander("Top Medicines by Revenue"):
                top_rev = sales_an.get_sales_by_medicine(15)
                st.dataframe(top_rev, use_container_width=True, hide_index=True)

    # ==================== TAB 3: EXPIRY RISK ====================
    with tabs[2]:
        st.header("Expiry & Wastage Risk Analysis")

        # KPIs
        exposure = expiry_an.get_wastage_report()["summary"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Expired Batches", exposure["expired"]["batches"], delta_color="inverse")
        col2.metric("Expired Qty", f"{exposure['expired']['quantity']:,}", delta_color="inverse")
        col3.metric("Near Expiry Batches", exposure["near_expiry"]["batches"], delta_color="inverse")
        col4.metric("Exposure at Cost", f"₹{exposure['total_exposure_at_cost']:,.0f}", delta_color="inverse")

        # Expiry distribution
        st.subheader("Batch Expiry Distribution")
        expiry_img = plot_expiry_distribution(expiry_an.batches)
        st.image(f"data:image/png;base64,{expiry_img}")

        # Expiry timeline
        st.subheader("Expiry Timeline (Next 90 Days)")
        timeline = expiry_an.get_expiry_timeline(90)
        timeline_img = plot_expiry_timeline(timeline)
        st.image(f"data:image/png;base64,{timeline_img}")

        # Risk breakdown
        st.subheader("Risk Level Breakdown")
        risk_df = expiry_an.get_risk_breakdown()
        if not risk_df.empty:
            st.dataframe(risk_df, use_container_width=True, hide_index=True)

        # Expired batches
        st.subheader("🚫 Expired Batches (Action Required)")
        expired_batches = expiry_an.get_expired_batches()
        if not expired_batches.empty:
            exp_display = expired_batches[["batch_id", "medicine_id", "name", "quantity_remaining",
                                           "purchase_cost_per_unit", "expiry_date", "days_to_expiry"]].copy()
            exp_display["Value at Risk"] = exp_display["quantity_remaining"] * exp_display["purchase_cost_per_unit"]
            st.dataframe(exp_display, use_container_width=True, hide_index=True)
        else:
            st.success("✅ No expired batches with remaining stock")

        # Near expiry batches
        st.subheader("⏰ Near Expiry Batches (Monitor Closely)")
        near_batches = expiry_an.get_near_expiry_batches()
        if not near_batches.empty:
            near_display = near_batches[["batch_id", "medicine_id", "name", "quantity_remaining",
                                         "purchase_cost_per_unit", "expiry_date", "days_to_expiry"]].copy()
            near_display["Value at Risk"] = near_display["quantity_remaining"] * near_display["purchase_cost_per_unit"]
            st.dataframe(near_display, use_container_width=True, hide_index=True)
        else:
            st.success("✅ No batches near expiry")

    # ==================== TAB 4: DEMAND & REPLENISHMENT ====================
    with tabs[3]:
        st.header("Demand Forecasting & Replenishment")

        # Replenishment summary
        repl_summary = replenish.get_replenishment_summary()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Urgent Items", repl_summary["urgent_count"], delta_color="inverse")
        col2.metric("High Priority", repl_summary["high_count"], delta_color="inverse")
        col3.metric("Total Suggested Qty", f"{repl_summary['total_suggested_order_qty']:,}")
        col4.metric("Est. Order Value", f"₹{repl_summary['estimated_order_value']:,.0f}")

        # Suggestions table
        st.subheader("Replenishment Recommendations")
        suggestions = replenish.generate_all_suggestions()

        # Filter by priority
        priority_filter = st.multiselect(
            "Filter by priority:",
            ["URGENT", "HIGH", "MEDIUM", "LOW", "OK"],
            default=["URGENT", "HIGH", "MEDIUM"],
        )
        filtered = [s for s in suggestions if s.priority in priority_filter]

        if filtered:
            sug_df = pd.DataFrame([s.to_dict() for s in filtered])
            st.dataframe(sug_df, use_container_width=True, hide_index=True)

            # Show demand forecast for selected medicine
            if len(filtered) > 0:
                st.subheader("Demand Forecast Detail")
                selected_med = st.selectbox(
                    "Select medicine for forecast detail:",
                    options=[f"{s.medicine_id} - {s.name}" for s in filtered],
                    index=0,
                )
                if selected_med:
                    med_id = selected_med.split(" - ")[0]
                    demand_summary = demand.get_demand_summary(med_id, days=30)
                    forecast = demand.forecast_moving_average(med_id, window=7, forecast_days=14)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**30-Day Demand Summary**")
                        st.json(demand_summary)
                    with col2:
                        st.write("**14-Day Forecast (Moving Avg)**")
                        fc_df = pd.DataFrame(forecast["forecast"], columns=["Date", "Forecast Qty"])
                        st.dataframe(fc_df, use_container_width=True, hide_index=True)
        else:
            st.info("No items matching filter")

    # ==================== TAB 5: REPORTS ====================
    with tabs[5] if len(tabs) > 4 else tabs[4]:
        st.header("📄 Generated Reports")

        report_type = st.selectbox(
            "Select report type:",
            ["Inventory Report", "Expiry Risk Report", "Sales Report", "Replenishment Report"],
        )

        if report_type == "Inventory Report":
            report = generate_inventory_report(snapshots, total_val, cat_summary)
        elif report_type == "Expiry Risk Report":
            report = generate_expiry_report(expiry_an.get_wastage_report(), risk_df)
        elif report_type == "Sales Report":
            report = generate_sales_report(
                sales_summary,
                cat_sales,
                sales_an.get_sales_by_medicine(10),
                payment_sales,
            )
        else:
            report = generate_replenishment_report(suggestions, repl_summary)

        st.text_area("Report", report, height=500)

        # Download button
        st.download_button(
            "Download Report",
            data=report,
            file_name=f"pharmatrack_{report_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
        )


if __name__ == "__main__":
    main()