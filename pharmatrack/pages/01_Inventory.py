"""
Inventory Management Dashboard
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import ensure_directories, RAW_DATA_DIR
from src.data import DataLoader, generate_sample_data
from src.inventory import InventoryManager
from src.visualization import PharmaCharts

st.set_page_config(page_title="Inventory - PharmaTrack", page_icon="📦", layout="wide")

st.title("📦 Inventory Management")
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
            # Load from uploaded files
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
    data_source = st.radio("Choose data source:", ["Use Synthetic Sample Data", "Upload CSV Files"])
    
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

# Initialize Inventory Manager
try:
    im = InventoryManager(med_df, inv_df, sales_df if sales_df is not None else pd.DataFrame())
except Exception as e:
    st.error(f"Error initializing inventory manager: {e}")
    st.stop()

# Filters
st.sidebar.markdown("---")
st.sidebar.header("Filters")
categories = ['All'] + sorted(med_df['category'].unique().tolist())
selected_category = st.sidebar.selectbox("Filter by Category", categories)
suppliers = ['All'] + sorted(inv_df['supplier'].unique().tolist())
selected_supplier = st.sidebar.selectbox("Filter by Supplier", suppliers)

# Apply filters
catalogue = im.get_medicine_catalogue()
if selected_category != 'All':
    catalogue = catalogue[catalogue['category'] == selected_category]
if selected_supplier != 'All':
    supplier_meds = inv_df[inv_df['supplier'] == selected_supplier]['medicine_id'].unique()
    catalogue = catalogue[catalogue['medicine_id'].isin(supplier_meds)]

# Overview Metrics
st.header("📊 Inventory Overview")
col1, col2, col3, col4 = st.columns(4)

total_batches = len(im.get_batch_inventory())
usable_stock = catalogue['usable_quantity'].sum() if not catalogue.empty else 0
total_value = catalogue['usable_purchase_value'].sum() if not catalogue.empty else 0
low_stock_count = len(im.get_low_stock_alerts())

with col1:
    st.metric("Total Batches", total_batches)
with col2:
    st.metric("Total Usable Stock", f"{int(usable_stock)} units")
with col3:
    st.metric("Usable Value (Purchase)", f"${total_value:,.2f}")
with col4:
    st.metric("Low Stock Items", low_stock_count, delta=f"-{low_stock_count}" if low_stock_count > 0 else None)

st.markdown("---")

# Searchable Medicine Catalogue
st.header("🔍 Medicine Catalogue")
search_term = st.text_input("Search medicines by name or ID:", "")
if search_term:
    catalogue_display = catalogue[
        catalogue['medicine_name'].str.contains(search_term, case=False) | 
        catalogue['medicine_id'].str.contains(search_term, case=False)
    ]
else:
    catalogue_display = catalogue

if catalogue_display.empty:
    st.info("No medicines match your filters.")
else:
    display_cols = ['medicine_id', 'medicine_name', 'category', 'usable_quantity', 
                    'total_batches', 'reorder_level', 'usable_purchase_value']
    st.dataframe(catalogue_display[display_cols].round(2), use_container_width=True, height=300)

st.markdown("---")

# Batch-Level Inventory Table
st.header("📋 Batch-Level Inventory (Searchable)")
batch_inv = im.get_batch_inventory(
    category=selected_category if selected_category != 'All' else None,
    supplier=selected_supplier if selected_supplier != 'All' else None
)
batch_search = st.text_input("Search batches by batch ID or medicine:", key="batch_search")
if batch_search:
    batch_display = batch_inv[
        batch_inv['batch_id'].str.contains(batch_search, case=False) |
        batch_inv['medicine_id'].str.contains(batch_search, case=False) |
        batch_inv.get('medicine_name', pd.Series()).str.contains(batch_search, case=False)
    ]
else:
    batch_display = batch_inv

if batch_display.empty:
    st.info("No batches match your filters.")
else:
    display_batch_cols = ['batch_id', 'medicine_id', 'medicine_name', 'supplier', 
                          'quantity', 'purchase_price', 'expiry_date', 'is_expired']
    st.dataframe(batch_display[display_batch_cols], use_container_width=True, height=300)

st.markdown("---")

# Charts
st.header("📈 Inventory Visualizations")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Stock Distribution by Category")
    if not catalogue.empty:
        cat_stock = catalogue.groupby('category')['usable_quantity'].sum().reset_index()
        fig = PharmaCharts.bar_chart(cat_stock, 'category', 'usable_quantity',
                                     'Usable Stock by Category', 'Category', 'Quantity (units)')
        st.pyplot(fig)
        plt.close()
    else:
        st.info("No data available for chart.")

with col2:
    st.subheader("Stock Value by Medicine (Top 10)")
    if not catalogue.empty:
        top_value = catalogue.nlargest(10, 'usable_purchase_value')[['medicine_name', 'usable_purchase_value']]
        fig = PharmaCharts.bar_chart(top_value, 'medicine_name', 'usable_purchase_value',
                                     'Top 10 Medicines by Stock Value', 'Medicine', 'Value ($)')
        st.pyplot(fig)
        plt.close()
    else:
        st.info("No data available for chart.")

st.markdown("---")

# Low Stock Reports
st.header("⚠️ Low Stock Alerts")
low_stock = im.get_low_stock_alerts()
if low_stock.empty:
    st.success("✅ No low stock items!")
else:
    st.warning(f"⚠️ {len(low_stock)} items below reorder level")
    st.dataframe(low_stock, use_container_width=True)
    
    # Download
    csv = low_stock.to_csv(index=False)
    st.download_button(
        label="📥 Download Low Stock Report (CSV)",
        data=csv,
        file_name=f"low_stock_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

st.markdown("---")

# Inventory Summary Report
st.header("📄 Inventory Summary Report")
if st.button("Generate Inventory Summary Report"):
    report = im.export_inventory_report()
    st.text_area("Report", report, height=300)
    
    csv_report = catalogue.to_csv(index=False)
    st.download_button(
        label="📥 Download Inventory Summary (CSV)",
        data=csv_report,
        file_name=f"inventory_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
"""
