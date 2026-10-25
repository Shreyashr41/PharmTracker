"""
PharmaTrack - Smart Pharmacy Inventory, Sales & Expiry Analytics System
Main Application Entry Point
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    PAGE_TITLE,
    PAGE_ICON,
    LAYOUT,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    ensure_directories,
)

# Page configuration
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded"
)


def main():
    """Main application entry point."""
    st.title("💊 PharmaTrack")
    st.subheader("Smart Pharmacy Inventory, Sales & Expiry Analytics System")
    st.caption("*Demo dashboard - not production-ready or connected to a live pharmacy system*")
    
    st.markdown("---")
    
    st.markdown("""
    ### Welcome to PharmaTrack
    
    This interactive dashboard helps you analyze pharmacy medicine inventory, sales transactions, 
    and batch expiry data to:
    
    - 📦 **Track Inventory** - Monitor stock levels across batches with different expiry dates
    - 💰 **Analyze Sales** - Calculate revenue and gross profit using batch-level costs
    - ⚠️ **Manage Expiry** - Identify expired and near-expiry batches to minimize wastage
    - 📊 **Generate Insights** - Make data-driven decisions for inventory management
    """)
    
    st.markdown("---")
    
    st.markdown("### 📋 Dashboard Sections")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **1. Overview** - Total usable stock, revenue, gross profit, low-stock & near-expiry alerts  
        **2. Inventory** - Searchable tables, stock distribution, low-stock reports  
        **3. Sales Analytics** - Trends, top sellers, category comparisons with date filters
        """)
    
    with col2:
        st.markdown("""
        **4. Expiry Analysis** - Expired/near-expiry batches, value at risk, FEFO recommendations  
        **5. Reports** - Downloadable CSV summaries of all analytics
        """)
    
    st.markdown("---")
    
    # Display project structure status
    with st.expander("📁 Data & Configuration Status"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Data Directories:**")
            st.write(f"- Raw Data: {RAW_DATA_DIR}")
            st.write(f"- Processed Data: {PROCESSED_DATA_DIR}")
            st.write(f"- Reports: {REPORTS_DIR}")
        with col2:
            st.markdown("**Features:**")
            st.write("✅ Matplotlib & Seaborn visualizations")
            st.write("✅ Inventory, Sales & Expiry modules")
            st.write("✅ Synthetic data generation")
            st.write("✅ CSV file upload support")
    
    st.success("👈 **Use the sidebar to navigate between dashboard sections**")
    st.info("💡 **Tip:** The dashboard uses synthetic sample data by default. You can upload your own compatible CSV files in the sidebar.")


if __name__ == "__main__":
    ensure_directories()
    main()
"""
