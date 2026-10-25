"""
Sales Analytics Dashboard
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import ensure_directories
from src.data import DataLoader, generate_sample_data
from src.analytics import SalesAnalytics
from src.visualization import PharmaCharts

st.set_page_config(page_title="Sales Analytics - PharmaTrack", page_icon="💰", layout="wide")

st.title("💰 Sales Analytics")
st.caption("*Demo dashboard - not production-ready or connected to a live pharmacy system*")

ensure_directories()


@st.cache_data
def load_data(use_sample=True, med_file=None, inv_file=None, sales_file=None):
    """Load data with caching."""
    try:
        if use_sample:
            generate_sample_data(seed=42)
            loader = DataLoader()
            med, inv, sales, reports = loader.load_and_validate()
            return med, inv, sales
        else:
            med = pd.read_csv(med_file) if med_file else None
            inv = pd.read_csv(inv_file) if inv_file else None
            sales = pd.read_csv(sales_file) if sales_file else pd.DataFrame()
            return med, inv, sales
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None


# Sidebar
with st.sidebar:
    st.header("Data Source")
    data_source = st.radio("Choose data source:", ["Use Synthetic Sample Data", "Upload CSV Files"], key="sales_data")
    
    med_df, inv_df, sales_df = None, None, None
    
    if data_source == "Use Synthetic Sample Data":
        med_df, inv_df, sales_df = load_data(use_sample=True)
        st.success("✅ Using synthetic sample data (seed=42)")
    else:
        med_file = st.file_uploader("Upload medicines.csv", type=['csv'])
        inv_file = st.file_uploader("Upload inventory.csv", type=['csv'])
        sales_file = st.file_uploader("Upload sales.csv", type=['csv'])
        
        if med_file and inv_file and sales_file:
            med_df, inv_df, sales_df = load_data(use_sample=False, med_file=med_file, 
                                                    inv_file=inv_file, sales_file=sales_file)
            st.success("✅ Files uploaded successfully")
        else:
            st.warning("⚠️ Please upload all three CSV files")


if med_df is None or inv_df is None or sales_df is None:
    st.warning("⚠️ No data loaded. Please select a data source in the sidebar.")
    st.stop()

# Initialize Sales Analytics
try:
    sa = SalesAnalytics(med_df, inv_df, sales_df)
except Exception as e:
    st.error(f"Error initializing sales analytics: {e}")
    st.stop()

# Date Filters
st.sidebar.markdown("---")
st.sidebar.header("Date Filters")
if not sa.enriched_sales.empty:
    min_date = sa.enriched_sales['date'].min().date()
    max_date = sa.enriched_sales['date'].max().date()
    start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
else:
    start_str, end_str = None, None
    st.sidebar.info("No sales data available")

# Overview Metrics
st.header("📊 Sales Overview")
stats = sa.get_sales_statistics()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Transactions", stats['total_transactions'])
with col2:
    st.metric("Total Revenue", f"${stats['total_revenue']:,.2f}")
with col3:
    st.metric("Total Gross Profit", f"${stats['total_gross_profit']:,.2f}")
with col4:
    avg_margin = (stats['total_gross_profit'] / stats['total_revenue'] * 100) if stats['total_revenue'] > 0 else 0
    st.metric("Avg Gross Margin", f"{avg_margin:.1f}%")

st.markdown("---")

# Revenue Trends
st.header("📈 Revenue Trends")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Daily Revenue Trend")
    daily = sa.get_sales_summary_daily(start_str, end_str)
    if not daily.empty:
        fig = PharmaCharts.line_chart_time_series(daily, 'date_only', 'revenue',
                                                   'Daily Revenue Trend', 'Date', 'Revenue ($)')
        st.pyplot(fig)
        plt.close()
    else:
        st.info("No sales data available for selected period.")

with col2:
    st.subheader("Monthly Revenue Trend")
    monthly = sa.get_sales_summary_monthly(start_str, end_str)
    if not monthly.empty:
        fig = PharmaCharts.line_chart_time_series(monthly, 'month_name', 'revenue',
                                                   'Monthly Revenue Trend', 'Month', 'Revenue ($)')
        st.pyplot(fig)
        plt.close()
    else:
        st.info("No sales data available for selected period.")

st.markdown("---")

# Top Selling Medicines
st.header("🏆 Top Selling Medicines")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Top 10 by Quantity Sold")
    top_qty = sa.get_top_selling_by_quantity(10, start_str, end_str)
    if not top_qty.empty:
        fig = PharmaCharts.bar_chart(top_qty, 'medicine_name', 'quantity_sold',
                                     'Top 10 by Quantity', 'Medicine', 'Quantity Sold')
        st.pyplot(fig)
        plt.close()
        st.dataframe(top_qty, use_container_width=True)
    else:
        st.info("No sales data available.")

with col2:
    st.subheader("Top 10 by Revenue")
    top_rev = sa.get_top_selling_by_revenue(10, start_str, end_str)
    if not top_rev.empty:
        fig = PharmaCharts.bar_chart(top_rev, 'medicine_name', 'revenue',
                                     'Top 10 by Revenue', 'Medicine', 'Revenue ($)')
        st.pyplot(fig)
        plt.close()
        st.dataframe(top_rev, use_container_width=True)
    else:
        st.info("No sales data available.")

st.markdown("---")

# Category Comparisons
st.header("📊 Category Comparisons")
by_category = sa.get_revenue_by_category(start_str, end_str)
if not by_category.empty:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Revenue by Category")
        fig = PharmaCharts.bar_chart(by_category, 'category', 'revenue',
                                     'Revenue by Category', 'Category', 'Revenue ($)')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("Quantity Sold by Category")
        fig = PharmaCharts.bar_chart(by_category, 'category', 'quantity_sold',
                                     'Quantity Sold by Category', 'Category', 'Quantity')
        st.pyplot(fig)
        plt.close()
    
    st.dataframe(by_category.round(2), use_container_width=True)
else:
    st.info("No sales data available for category analysis.")

st.markdown("---")

# Revenue by Medicine Table
st.header("💊 Revenue by Medicine (Detailed)")
by_medicine = sa.get_revenue_by_medicine(start_str, end_str)
if not by_medicine.empty:
    st.dataframe(by_medicine.round(2), use_container_width=True, height=400)
    
    csv = by_medicine.to_csv(index=False)
    st.download_button(
        label="📥 Download Revenue by Medicine (CSV)",
        data=csv,
        file_name=f"revenue_by_medicine_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
else:
    st.info("No sales data available.")

st.markdown("---")

# Statistical Analysis
st.header("📈 Statistical Analysis")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Unit Price Distribution")
    if not sa.enriched_sales.empty:
        fig = PharmaCharts.distribution_chart(sa.enriched_sales, 'unit_price',
                                               'Unit Price Distribution', bins=15)
        st.pyplot(fig)
        plt.close()
    else:
        st.info("No data available.")

with col2:
    st.subheader("Quantity Sold Distribution")
    if not sa.enriched_sales.empty:
        fig = PharmaCharts.distribution_chart(sa.enriched_sales, 'quantity_sold',
                                               'Quantity Sold Distribution', bins=15)
        st.pyplot(fig)
        plt.close()
    else:
        st.info("No data available.")

# Correlation heatmap
st.subheader("Correlation Analysis")
if not sa.enriched_sales.empty:
    numeric_cols = ['quantity_sold', 'unit_price', 'revenue', 'purchase_price', 'gross_profit']
    corr_df = sa.enriched_sales[[c for c in numeric_cols if c in sa.enriched_sales.columns]]
    fig = PharmaCharts.correlation_heatmap(corr_df, 'Sales Metrics Correlation')
    st.pyplot(fig)
    plt.close()
else:
    st.info("No data available for correlation analysis.")

st.markdown("---")

# Trend Analysis
st.header("🔮 Trend Analysis & Demand Estimates")
trend = sa.get_trend_analysis(days=30)
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Trend Summary:**")
    st.write(f"- Trend: **{trend['trend'].upper()}**")
    st.write(f"- Daily Average: **{trend['daily_avg']} units**")
    st.write(f"- Weekly Average: **{trend['weekly_avg']} units**")
    st.write(f"- Monthly Average: **{trend['monthly_avg']} units**")
    st.write(f"- Demand Estimate (Next 30 Days): **{trend['demand_estimate_next_30_days']} units**")

with col2:
    st.markdown("**Limitations:**")
    for lim in trend.get('limitations', []):
        st.caption(f"• {lim}")

st.markdown("---")

# Export Reports
st.header("📥 Export Reports")
col1, col2, col3 = st.columns(3)

with col1:
    if not monthly.empty:
        csv = monthly.to_csv(index=False)
        st.download_button(
            label="Download Monthly Summary",
            data=csv,
            file_name=f"sales_monthly_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

with col2:
    if not by_category.empty:
        csv = by_category.to_csv(index=False)
        st.download_button(
            label="Download Category Summary",
            data=csv,
            file_name=f"sales_category_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

with col3:
    if not top_rev.empty:
        csv = top_rev.to_csv(index=False)
        st.download_button(
            label="Download Top Sellers",
            data=csv,
            file_name=f"top_sellers_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
"""
