"""
Expiry Analysis Dashboard
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import ensure_directories
from src.data import DataLoader, generate_sample_data
from src.expiry import ExpiryAnalyzer
from src.visualization import PharmaCharts

st.set_page_config(page_title="Expiry Analysis - PharmaTrack", page_icon="⚠️", layout="wide")

st.title("⚠️ Expiry & Wastage Risk Analysis")
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
    data_source = st.radio("Choose data source:", ["Use Synthetic Sample Data", "Upload CSV Files"], key="expiry_data")
    
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
    
    st.markdown("---")
    st.header("Analysis Settings")
    near_expiry_threshold = st.slider("Near-Expiry Threshold (days)", min_value=7, max_value=90, value=30, step=1)
    use_custom_date = st.checkbox("Use custom reference date")
    if use_custom_date:
        ref_date = st.date_input("Reference Date", datetime.now())
    else:
        ref_date = None


if med_df is None or inv_df is None:
    st.warning("⚠️ No data loaded. Please select a data source in the sidebar.")
    st.stop()

# Initialize Expiry Analyzer
try:
    ea = ExpiryAnalyzer(med_df, inv_df, reference_date=ref_date, 
                        near_expiry_threshold=near_expiry_threshold)
except Exception as e:
    st.error(f"Error initializing expiry analyzer: {e}")
    st.stop()

# Overview Metrics
st.header("📊 Expiry Risk Overview")
summary = ea.get_expiry_summary()
value_risk = summary['value_at_risk']

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Batches", summary['total_batches'])
with col2:
    st.metric("Expired Batches", summary['expired_batches'], 
              delta=f"-{summary['expired_batches']}" if summary['expired_batches'] > 0 else None)
with col3:
    st.metric("Near Expiry Batches", summary['near_expiry_batches'],
              delta=f"⚠️ {summary['near_expiry_batches']}" if summary['near_expiry_batches'] > 0 else None)
with col4:
    st.metric("Value at Risk (Est.)", f"${value_risk['near_expiry_value_at_risk']:,.2f}")

st.caption(f"📅 Reference Date: {summary['reference_date']} | Threshold: {summary['near_expiry_threshold_days']} days")
st.caption(f"ℹ️ {value_risk['disclaimer']}")

st.markdown("---")

# Expired Batches
st.header("🚨 Expired Batches")
expired = ea.get_expired_batches()
if expired.empty:
    st.success("✅ No expired batches!")
else:
    st.error(f"⚠️ {len(expired)} expired batches found")
    display_cols = ['batch_id', 'medicine_id', 'medicine_name', 'quantity', 
                    'purchase_value_total', 'expiry_date', 'days_remaining']
    display_expired = expired[[c for c in display_cols if c in expired.columns]]
    st.dataframe(display_expired, use_container_width=True)
    
    csv = display_expired.to_csv(index=False)
    st.download_button(
        label="📥 Download Expired Batches (CSV)",
        data=csv,
        file_name=f"expired_batches_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

st.markdown("---")

# Near Expiry Batches
st.header(f"⏰ Near Expiry Batches (Within {near_expiry_threshold} Days)")
near_expiry = ea.get_near_expiry_batches()
if near_expiry.empty:
    st.success(f"✅ No batches expiring within {near_expiry_threshold} days!")
else:
    st.warning(f"⚠️ {len(near_expiry)} batches approaching expiry")
    display_cols = ['batch_id', 'medicine_id', 'medicine_name', 'quantity', 
                    'purchase_price', 'risk_value', 'expiry_date', 'days_remaining']
    display_near = near_expiry[[c for c in display_cols if c in near_expiry.columns]]
    st.dataframe(display_near.round(2), use_container_width=True)
    
    csv = display_near.to_csv(index=False)
    st.download_button(
        label="📥 Download Near-Expiry Batches (CSV)",
        data=csv,
        file_name=f"near_expiry_batches_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

st.markdown("---")

# High Value Batches at Risk
st.header("💎 High-Value Batches at Risk")
min_risk_value = st.number_input("Minimum risk value ($)", min_value=0.0, value=100.0, step=50.0)
high_value = ea.get_high_value_batches_at_risk(min_value=min_risk_value)
if high_value.empty:
    st.info(f"No batches with risk value >= ${min_risk_value}")
else:
    display_cols = ['batch_id', 'medicine_id', 'medicine_name', 'quantity', 
                    'purchase_price', 'risk_value', 'expiry_date', 'days_remaining']
    display_high = high_value[[c for c in display_cols if c in high_value.columns]]
    st.dataframe(display_high.round(2), use_container_width=True)
    
    csv = display_high.to_csv(index=False)
    st.download_button(
        label="📥 Download High-Value at Risk (CSV)",
        data=csv,
        file_name=f"high_value_at_risk_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

st.markdown("---")

# FEFO Recommendations
st.header("🔄 FEFO (First-Expiry-First-Out) Recommendations")
st.caption("Prioritize using batches expiring earliest first")
fefo = ea.get_fefo_recommendations()
if fefo.empty:
    st.info("No active batches for rotation")
else:
    st.dataframe(fefo, use_container_width=True, height=400)
    
    csv = fefo.to_csv(index=False)
    st.download_button(
        label="📥 Download FEFO Recommendations (CSV)",
        data=csv,
        file_name=f"fefo_recommendations_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

st.markdown("---")

# Visualizations
st.header("📈 Expiry Visualizations")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Expiry Status Distribution")
    status_counts = ea.expiry_df['expiry_status'].value_counts().reset_index()
    status_counts.columns = ['status', 'count']
    fig = PharmaCharts.pie_chart(status_counts, 'status', 'count', 'Batch Status Distribution')
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("Days Remaining Distribution")
    if not ea.expiry_df.empty:
        fig = PharmaCharts.distribution_chart(ea.expiry_df, 'days_remaining',
                                               'Days Remaining Distribution', bins=20)
        st.pyplot(fig)
        plt.close()
    else:
        st.info("No data available.")

st.subheader("Risk Value by Medicine (Near Expiry)")
if not near_expiry.empty and 'medicine_name' in near_expiry.columns:
    risk_by_med = near_expiry.groupby('medicine_name')['risk_value'].sum().reset_index()
    risk_by_med = risk_by_med.sort_values('risk_value', ascending=False).head(10)
    fig = PharmaCharts.bar_chart(risk_by_med, 'medicine_name', 'risk_value',
                                 'Risk Value by Medicine', 'Medicine', 'Risk Value ($)')
    st.pyplot(fig)
    plt.close()
else:
    st.info("No near-expiry batches to analyze.")

st.markdown("---")

# Batch-Level Expiry Report
st.header("📋 Complete Batch-Level Expiry Report")
report_df = ea.generate_batch_expiry_report()
if report_df.empty:
    st.info("No batches to report")
else:
    st.dataframe(report_df.round(2), use_container_width=True, height=400)
    
    csv = report_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Complete Expiry Report (CSV)",
        data=csv,
        file_name=f"batch_expiry_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

st.markdown("---")

# Wastage Analysis
st.header("🗑️ Wastage Analysis")
wastage = ea.analyze_wastage()
if not wastage['wastage_records_found']:
    st.info(f"ℹ️ {wastage['message']}")
    st.caption(wastage['disclaimer'])
else:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Wastage Records", wastage['total_records'])
    with col2:
        st.metric("Total Quantity Wasted", wastage.get('total_wastage_quantity', 0))
    with col3:
        st.metric("Total Wastage Value", f"${wastage.get('total_wastage_value', 0):,.2f}")
    
    st.caption(wastage['disclaimer'])
    
    if 'wastage_by_medicine' in wastage:
        st.subheader("Wastage by Medicine")
        wastage_med_df = pd.DataFrame(wastage['wastage_by_medicine'])
        st.dataframe(wastage_med_df, use_container_width=True)
    
    if 'wastage_by_reason' in wastage:
        st.subheader("Wastage by Reason")
        wastage_reason_df = pd.DataFrame(wastage['wastage_by_reason'])
        st.dataframe(wastage_reason_df, use_container_width=True)

st.markdown("---")

# Export Full Report
st.header("📄 Full Expiry Analysis Report")
if st.button("Generate Full Expiry Report"):
    report_text = ea.export_expiry_report()
    st.text_area("Report", report_text, height=400)
    
    csv_full = report_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Full Report Data (CSV)",
        data=csv_full,
        file_name=f"full_expiry_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
"""
