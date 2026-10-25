"""
Test suite for PharmaTrack sales analytics.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta

from pharmatrack.sales.analytics import compute_gross_profit, SalesAnalytics
from pharmatrack.sales.pricing import PricingEngine


class TestGrossProfit:
    """Test gross profit computation."""

    def test_compute_gross_profit_basic(self):
        sales = pd.DataFrame([
            {"sale_id": "S1", "batch_id": "B1", "medicine_id": "M1",
             "quantity_sold": 10, "selling_price_per_unit": 20.0,
             "sale_date": pd.Timestamp("2024-06-01"), "payment_method": "Cash"},
            {"sale_id": "S2", "batch_id": "B2", "medicine_id": "M1",
             "quantity_sold": 5, "selling_price_per_unit": 22.0,
             "sale_date": pd.Timestamp("2024-06-02"), "payment_method": "Card"},
        ])
        batches = pd.DataFrame([
            {"batch_id": "B1", "purchase_cost_per_unit": 10.0},
            {"batch_id": "B2", "purchase_cost_per_unit": 12.0},
        ])

        result = compute_gross_profit(sales, batches)
        assert len(result) == 2
        assert "purchase_cost_per_unit" in result.columns
        assert "revenue" in result.columns
        assert "cost_of_goods_sold" in result.columns
        assert "gross_profit" in result.columns
        assert "gross_margin_pct" in result.columns

        # Check calculations
        row1 = result[result["sale_id"] == "S1"].iloc[0]
        assert row1["revenue"] == 200.0  # 10 * 20
        assert row1["cost_of_goods_sold"] == 100.0  # 10 * 10
        assert row1["gross_profit"] == 100.0
        assert row1["gross_margin_pct"] == 50.0

        row2 = result[result["sale_id"] == "S2"].iloc[0]
        assert row2["revenue"] == 110.0  # 5 * 22
        assert row2["cost_of_goods_sold"] == 60.0  # 5 * 12
        assert row2["gross_profit"] == 50.0
        assert row2["gross_margin_pct"] == pytest.approx(45.45, rel=0.01)

    def test_compute_gross_profit_missing_batch(self):
        """Test handling of sales with missing batch cost."""
        sales = pd.DataFrame([
            {"sale_id": "S1", "batch_id": "B1", "medicine_id": "M1",
             "quantity_sold": 10, "selling_price_per_unit": 20.0,
             "sale_date": pd.Timestamp("2024-06-01"), "payment_method": "Cash"},
        ])
        batches = pd.DataFrame([
            {"batch_id": "B2", "purchase_cost_per_unit": 10.0},  # Different batch
        ])

        result = compute_gross_profit(sales, batches)
        assert result["purchase_cost_per_unit"].isna().all()
        assert result["cost_of_goods_sold"].isna().all()
        assert result["gross_profit"].isna().all()


class TestSalesAnalytics:
    """Test SalesAnalytics class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        medicines = pd.DataFrame([
            {"medicine_id": "M1", "name": "Drug A", "category": "Antibiotic",
             "manufacturer": "Mfr1", "unit_selling_price": 20.0, "reorder_level": 30, "unit_of_measure": "tablet"},
            {"medicine_id": "M2", "name": "Drug B", "category": "Analgesic",
             "manufacturer": "Mfr2", "unit_selling_price": 10.0, "reorder_level": 50, "unit_of_measure": "tablet"},
        ])

        batches = pd.DataFrame([
            {"batch_id": "B1", "medicine_id": "M1", "purchase_cost_per_unit": 10.0,
             "expiry_date": pd.Timestamp("2025-01-01")},
            {"batch_id": "B2", "medicine_id": "M1", "purchase_cost_per_unit": 11.0,
             "expiry_date": pd.Timestamp("2025-01-01")},
            {"batch_id": "B3", "medicine_id": "M2", "purchase_cost_per_unit": 5.0,
             "expiry_date": pd.Timestamp("2025-01-01")},
        ])

        sales = pd.DataFrame([
            {"sale_id": "S1", "batch_id": "B1", "medicine_id": "M1",
             "quantity_sold": 10, "selling_price_per_unit": 20.0,
             "sale_date": pd.Timestamp("2024-06-01"), "payment_method": "Cash"},
            {"sale_id": "S2", "batch_id": "B1", "medicine_id": "M1",
             "quantity_sold": 5, "selling_price_per_unit": 21.0,
             "sale_date": pd.Timestamp("2024-06-02"), "payment_method": "Card"},
            {"sale_id": "S3", "batch_id": "B2", "medicine_id": "M1",
             "quantity_sold": 8, "selling_price_per_unit": 19.0,
             "sale_date": pd.Timestamp("2024-06-03"), "payment_method": "UPI"},
            {"sale_id": "S4", "batch_id": "B3", "medicine_id": "M2",
             "quantity_sold": 20, "selling_price_per_unit": 10.0,
             "sale_date": pd.Timestamp("2024-06-01"), "payment_method": "Cash"},
        ])

        return medicines, batches, sales

    def test_sales_summary(self, sample_data):
        meds, batches, sales = sample_data
        sa = SalesAnalytics(sales, batches, meds)
        summary = sa.get_sales_summary()

        assert summary["total_transactions"] == 4
        assert summary["total_quantity_sold"] == 43
        assert summary["total_revenue"] == pytest.approx(20*10 + 21*5 + 19*8 + 10*20)
        assert summary["total_gross_profit"] > 0
        assert summary["unique_medicines_sold"] == 2
        assert summary["unique_batches_sold"] == 3

    def test_sales_by_period(self, sample_data):
        meds, batches, sales = sample_data
        sa = SalesAnalytics(sales, batches, meds)
        monthly = sa.get_sales_by_period("M")
        assert len(monthly) == 1  # All in June 2024
        assert monthly.iloc[0]["transactions"] == 4

    def test_sales_by_category(self, sample_data):
        meds, batches, sales = sample_data
        sa = SalesAnalytics(sales, batches, meds)
        cat = sa.get_sales_by_category()
        assert len(cat) == 2
        assert "Antibiotic" in cat["category"].values
        assert "Analgesic" in cat["category"].values

    def test_sales_by_medicine(self, sample_data):
        meds, batches, sales = sample_data
        sa = SalesAnalytics(sales, batches, meds)
        med_sales = sa.get_sales_by_medicine()
        assert len(med_sales) == 2
        # M1 should have higher revenue
        m1 = med_sales[med_sales["medicine_id"] == "M1"].iloc[0]
        assert m1["quantity_sold"] == 23

    def test_top_profitable_medicines(self, sample_data):
        meds, batches, sales = sample_data
        sa = SalesAnalytics(sales, batches, meds)
        top = sa.get_top_profitable_medicines(5)
        assert len(top) <= 2

    def test_batch_profitability(self, sample_data):
        meds, batches, sales = sample_data
        sa = SalesAnalytics(sales, batches, meds)
        batch_prof = sa.get_batch_profitability()
        assert len(batch_prof) == 3
        assert batch_prof.iloc[0]["gross_profit"] >= batch_prof.iloc[1]["gross_profit"]

    def test_daily_sales_trend(self, sample_data):
        meds, batches, sales = sample_data
        sa = SalesAnalytics(sales, batches, meds)
        daily = sa.get_daily_sales_trend(30)
        assert len(daily) >= 3  # At least 3 days of sales
        assert "sale_date" in daily.columns
        assert "revenue" in daily.columns


class TestPricingEngine:
    """Test PricingEngine class."""

    def test_price_vs_cost_analysis(self):
        medicines = pd.DataFrame([
            {"medicine_id": "M1", "name": "Drug A", "category": "Antibiotic", "unit_selling_price": 20.0},
            {"medicine_id": "M2", "name": "Drug B", "category": "Analgesic", "unit_selling_price": 10.0},
        ])
        batches = pd.DataFrame([
            {"batch_id": "B1", "medicine_id": "M1", "purchase_cost_per_unit": 10.0,
             "quantity_remaining": 50, "is_expired": False, "saleable_quantity": 50},
            {"batch_id": "B2", "medicine_id": "M2", "purchase_cost_per_unit": 5.0,
             "quantity_remaining": 30, "is_expired": False, "saleable_quantity": 30},
        ])
        sales = pd.DataFrame()

        pe = PricingEngine(medicines, sales, batches)
        analysis = pe.get_price_vs_cost_analysis()

        assert len(analysis) == 2
        m1 = analysis[analysis["medicine_id"] == "M1"].iloc[0]
        assert m1["avg_purchase_cost"] == 10.0
        assert m1["markup_pct"] == 100.0  # (20-10)/10 * 100
        assert m1["margin_pct"] == 50.0   # (20-10)/20 * 100