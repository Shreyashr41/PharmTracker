"""
Reports Dashboard
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import ensure_directories, REPORTS_DIR
from src.data import DataLoader, generate_sample_data
from src.inventory import InventoryManager
from src.analytics import SalesAnalytics
from src.expiry import ExpiryAnalyzer

st.set_page_config(page_title="Reports - PharmaTrack", page_icon="📄", layout="wide")

st.title("📄 Reports & Downloads")
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
    data_source = st.radio("Choose data source:", ["Use Synthetic Sample Data", "Upload CSV Files"], key="reports_data")
    
    med_df, inv_df, sales_df = None, None, None
    
    if data_source == "Use Synthetic Sample Data":
        med_df, inv_df, sales_df = load_data(use_sample=True)
        st.success("✅ Using synthetic sample data (seed=42)")
    else:
        med_file = st.file_uploader("Upload medicines.csv", type=['csv'])
        inv_file = st.file_uploader("Upload inventory.csv", type=['csv'])
        sales_file = st.file_uploader("Upload sales.csv (optional)", type=['csv'])
        
        if med_file and inv_file:
            med_df, inv_df, sales_df = load_data(use_sample=False, med_file=med_file, 
                                                    inv_file=inv_file, sales_file=sales_file)
            st.success("✅ Files uploaded successfully")
        else:
            st.warning("⚠️ Please upload medicines.csv and inventory.csv")


if med_df is None or inv_df is None:
    st.warning("⚠️ No data loaded. Please select a data source in the sidebar.")
    st.stop()

st.markdown("""
All analytical summaries are available for download as CSV files. 
Reports are generated from the current loaded data with all applied filters.
""")

st.markdown("---")

# Inventory Reports
st.header("📦 Inventory Reports")
try:
    im = InventoryManager(med_df, inv_df, sales_df if sales_df is not None else pd.DataFrame())
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        catalogue = im.get_medicine_catalogue()
        csv = catalogue.to_csv(index=False)
        st.download_button(
            label="📥 Medicine Catalogue",
            data=csv,
            file_name=f"medicine_catalogue_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        batch_inv = im.get_batch_inventory()
        csv = batch_inv.to_csv(index=False)
        st.download_button(
            label="📥 Batch Inventory",
            data=csv,
            file_name=f"batch_inventory_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col3:
        low_stock = im.get_low_stock_alerts()
        csv = low_stock.to_csv(index=False)
        st.download_button(
            label="📥 Low Stock Alerts",
            data=csv,
            file_name=f"low_stock_alerts_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col1:
        suggestions = im.get_replenishment_suggestions()
        csv = suggestions.to_csv(index=False)
        st.download_button(
            label="📥 Replenishment Suggestions",
            data=csv,
            file_name=f"replenishment_suggestions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        valuation = im.calculate_inventory_valuation()
        val_df = pd.DataFrame([valuation])
        csv = val_df.to_csv(index=False)
        st.download_button(
            label="📥 Inventory Valuation",
            data=csv,
            file_name=f"inventory_valuation_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
except Exception as e:
    st.error(f"Error generating inventory reports: {e}")

st.markdown("---")

# Sales Reports
st.header("💰 Sales Reports")
if sales_df is None or sales_df.empty:
    st.info("No sales data available")
else:
    try:
        sa = SalesAnalytics(med_df, inv_df, sales_df)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            by_medicine = sa.get_revenue_by_medicine()
            csv = by_medicine.to_csv(index=False)
            st.download_button(
                label="📥 Revenue by Medicine",
                data=csv,
                file_name=f"revenue_by_medicine_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            by_category = sa.get_revenue_by_category()
            csv = by_category.to_csv(index=False)
            st.download_button(
                label="📥 Revenue by Category",
                data=csv,
                file_name=f"revenue_by_category_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col3:
            monthly = sa.get_sales_summary_monthly()
            csv = monthly.to_csv(index=False)
            st.download_button(
                label="📥 Monthly Sales Summary",
                data=csv,
                file_name=f"sales_monthly_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            top_qty = sa.get_top_selling_by_quantity(20)
            csv = top_qty.to_csv(index=False)
            st.download_button(
                label="📥 Top Sellers by Quantity",
                data=csv,
                file_name=f"top_sellers_qty_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            top_rev = sa.get_top_selling_by_revenue(20)
            csv = top_rev.to_csv(index=False)
            st.download_button(
                label="📥 Top Sellers by Revenue",
                data=csv,
                file_name=f"top_sellers_rev_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col3:
            profit = sa.get_gross_profit_analysis()
            csv = profit.to_csv(index=False)
            st.download_button(
                label="📥 Gross Profit Analysis",
                data=csv,
                file_name=f"gross_profit_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        col1, col2 = st.columns(2)
        
        with col1:
            daily = sa.get_sales_summary_daily()
            csv = daily.to_csv(index=False)
            st.download_button(
                label="📥 Daily Sales Summary",
                data=csv,
                file_name=f"sales_daily_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            slow = sa.get_slow_moving_medicines()
            csv = slow.to_csv(index=False)
            st.download_button(
                label="📥 Slow Moving Medicines",
                data=csv,
                file_name=f"slow_moving_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
    except Exception as e:
        st.error(f"Error generating sales reports: {e}")

st.markdown("---")

# Expiry Reports
st.header("⚠️ Expiry Reports")
try:
    ea = ExpiryAnalyzer(med_df, inv_df)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        expiry_report = ea.generate_batch_expiry_report()
        csv = expiry_report.to_csv(index=False)
        st.download_button(
            label="📥 Batch Expiry Report",
            data=csv,
            file_name=f"batch_expiry_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        expired = ea.get_expired_batches()
        csv = expired.to_csv(index=False)
        st.download_button(
            label="📥 Expired Batches",
            data=csv,
            file_name=f"expired_batches_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col3:
        near = ea.get_near_expiry_batches()
        csv = near.to_csv(index=False)
        st.download_button(
            label="📥 Near-Expiry Batches",
            data=csv,
            file_name=f"near_expiry_batches_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fefo = ea.get_fefo_recommendations()
        csv = fefo.to_csv(index=False)
        st.download_button(
            label="📥 FEFO Recommendations",
            data=csv,
            file_name=f"fefo_recommendations_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        high_value = ea.get_high_value_batches_at_risk()
        csv = high_value.to_csv(index=False)
        st.download_button(
            label="📥 High-Value at Risk",
            data=csv,
            file_name=f"high_value_at_risk_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col3:
        value_risk = ea.calculate_value_at_risk()
        val_df = pd.DataFrame([value_risk])
        csv = val_df.to_csv(index=False)
        st.download_button(
            label="📥 Value at Risk Summary",
            data=csv,
            file_name=f"value_at_risk_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
except Exception as e:
    st.error(f"Error generating expiry reports: {e}")

st.markdown("---")

# All Reports Export
st.header("🎯 Export All Reports")
st.markdown("Generate and export all analytical summaries at once using the analytics modules.")

if st.button("Generate All Reports to CSV"):
    try:
        if sales_df is not None and not sales_df.empty:
            sa = SalesAnalytics(med_df, inv_df, sales_df)
            exports = sa.export_summary_csv(REPORTS_DIR)
            st.success(f"✅ Exported {len(exports)} sales reports to {REPORTS_DIR}")
            for name, path in exports.items():
                st.text(f"• {name}: {path}")
        else:
            st.warning("No sales data for full export")
    except Exception as e:
        st.error(f"Error exporting reports: {e}")

st.markdown("---")
st.caption("Reports are saved to the reports/ directory and can also be downloaded directly from the dashboard.")
"""
