"""
Phase 3 Tests for Sales and Revenue Analytics
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from src.analytics import SalesAnalytics
from src.data import PharmaDataGenerator


class TestSalesAnalytics:
    """Test sales analytics functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        gen = PharmaDataGenerator(seed=789)
        med, inv, sales = gen.generate_all_data(10, 3, 150)
        return med, inv, sales
    
    def test_sales_analytics_initialization(self, sample_data):
        """Test sales analytics initialization."""
        med, inv, sales = sample_data
        sa = SalesAnalytics(med, inv, sales)
        assert sa is not None
        assert not sa.enriched_sales.empty
    
    def test_daily_sales_summary(self, sample_data):
        """Test daily sales summary."""
        med, inv, sales = sample_data
        sa = SalesAnalytics(med, inv, sales)
        daily = sa.get_sales_summary_daily()
        
        assert isinstance(daily, pd.DataFrame)
        assert 'revenue' in daily.columns
        assert 'quantity_sold' in daily.columns
        assert 'gross_profit' in daily.columns
        assert (daily['revenue'] >= 0).all()
    
    def test_weekly_monthly_summaries(self, sample_data):
        """Test weekly and monthly sales summaries."""
        med, inv, sales = sample_data
        sa = SalesAnalytics(med, inv, sales)
        
        weekly = sa.get_sales_summary_weekly()
        monthly = sa.get_sales_summary_monthly()
        
        assert isinstance(weekly, pd.DataFrame)
        assert isinstance(monthly, pd.DataFrame)
        assert 'revenue' in weekly.columns
        assert 'revenue' in monthly.columns
    
    def test_revenue_by_medicine_category(self, sample_data):
        """Test revenue breakdown by medicine and category."""
        med, inv, sales = sample_data
        sa = SalesAnalytics(med, inv, sales)
        
        by_medicine = sa.get_revenue_by_medicine()
        by_category = sa.get_revenue_by_category()
        
        assert isinstance(by_medicine, pd.DataFrame)
        assert isinstance(by_category, pd.DataFrame)
        assert len(by_medicine) == len(med)
        assert 'revenue' in by_medicine.columns
        assert 'gross_profit' in by_medicine.columns
    
    def test_top_selling_ranking(self, sample_data):
        """Test top-selling medicines ranking."""
        med, inv, sales = sample_data
        sa = SalesAnalytics(med, inv, sales)
        
        top_qty = sa.get_top_selling_by_quantity(5)
        top_rev = sa.get_top_selling_by_revenue(5)
        
        assert len(top_qty) <= 5
        assert len(top_rev) <= 5
        assert 'quantity_sold' in top_qty.columns
        assert 'revenue' in top_rev.columns
        # Should be sorted
        if len(top_qty) > 1:
            assert (top_qty['quantity_sold'].values[0] >= top_qty['quantity_sold'].values[1])
    
    def test_slow_moving_analysis(self, sample_data):
        """Test slow-moving medicine analysis."""
        med, inv, sales = sample_data
        sa = SalesAnalytics(med, inv, sales)
        
        slow = sa.get_slow_moving_medicines(days_threshold=30)
        assert isinstance(slow, pd.DataFrame)
        if not slow.empty:
            assert 'status' in slow.columns
            assert 'reason' in slow.columns
    
    def test_gross_profit_using_batch_costs(self, sample_data):
        """Test gross profit uses actual batch purchase costs."""
        med, inv, sales = sample_data
        sa = SalesAnalytics(med, inv, sales)
        
        profit_analysis = sa.get_gross_profit_analysis()
        assert isinstance(profit_analysis, pd.DataFrame)
        if not profit_analysis.empty:
            assert 'purchase_cost_per_unit' in profit_analysis.columns
            assert 'gross_profit' in profit_analysis.columns
            # Gross profit should not equal revenue
            assert not (profit_analysis['gross_profit'] == profit_analysis['revenue']).all()
            # Verify calculation uses batch cost
            for _, row in profit_analysis.head(1).iterrows():
                expected_profit = row['revenue'] - row['total_cost']
                assert abs(row['gross_profit'] - expected_profit) < 0.01
    
    def test_sales_statistics(self, sample_data):
        """Test sales quantity and price statistics."""
        med, inv, sales = sample_data
        sa = SalesAnalytics(med, inv, sales)
        
        stats = sa.get_sales_statistics()
        assert isinstance(stats, dict)
        assert 'avg_unit_price' in stats
        assert 'avg_transaction_quantity' in stats
        assert 'avg_transaction_value' in stats
        assert stats['avg_unit_price'] > 0
        assert stats['total_transactions'] == len(sales)
    
    def test_trend_analysis(self, sample_data):
        """Test trend analysis and demand estimates."""
        med, inv, sales = sample_data
        sa = SalesAnalytics(med, inv, sales)
        
        trend = sa.get_trend_analysis(days=30)
        assert isinstance(trend, dict)
        assert 'trend' in trend
        assert 'demand_estimate_next_30_days' in trend
        assert 'limitations' in trend
        assert len(trend['limitations']) > 0
        assert isinstance(trend['limitations'], list)
    
    def test_date_filtering(self, sample_data):
        """Test date filtering functionality."""
        med, inv, sales = sample_data
        sa = SalesAnalytics(med, inv, sales)
        
        # Filter by date range
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        filtered = sa.get_revenue_by_medicine(start_date=start_date, end_date=end_date)
        assert isinstance(filtered, pd.DataFrame)
    
    def test_aggregation_accuracy(self, sample_data):
        """Test aggregation accuracy."""
        med, inv, sales = sample_data
        sa = SalesAnalytics(med, inv, sales)
        
        daily = sa.get_sales_summary_daily()
        monthly = sa.get_sales_summary_monthly()
        
        if not daily.empty and not monthly.empty:
            # Total from daily should relate to monthly
            total_daily_qty = daily['quantity_sold'].sum()
            total_monthly_qty = monthly['quantity_sold'].sum()
            # Should be same or very close
            assert abs(total_daily_qty - total_monthly_qty) < 1
    
    def test_csv_export(self, sample_data):
        """Test CSV export functionality."""
        med, inv, sales = sample_data
        sa = SalesAnalytics(med, inv, sales)
        
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            exports = sa.export_summary_csv(Path(tmpdir))
            assert isinstance(exports, dict)
            if exports:
                for name, path in exports.items():
                    assert Path(path).exists()
                    # Verify it's CSV
                    df = pd.read_csv(path)
                    assert len(df) >= 0
    
    def test_empty_dataset_handling(self):
        """Test handling of empty datasets."""
        med = pd.DataFrame({'medicine_id': ['MED001'], 'medicine_name': ['Test'], 'category': ['Test'], 'reorder_level': [50]})
        inv = pd.DataFrame({'batch_id': ['BATCH001'], 'medicine_id': ['MED001'], 'quantity': [100], 'purchase_price': [10], 'selling_price': [15]})
        sales_empty = pd.DataFrame(columns=['sale_id', 'date', 'batch_id', 'quantity_sold', 'unit_price'])
        
        sa = SalesAnalytics(med, inv, sales_empty)
        assert sa.get_sales_summary_daily().empty
        assert sa.get_revenue_by_medicine().empty
        stats = sa.get_sales_statistics()
        assert stats['total_transactions'] == 0
    
    def test_profit_not_revenue(self, sample_data):
        """Test that profit is not treated as revenue."""
        med, inv, sales = sample_data
        sa = SalesAnalytics(med, inv, sales)
        
        by_medicine = sa.get_revenue_by_medicine()
        if not by_medicine.empty:
            # Gross profit should generally be less than revenue
            for _, row in by_medicine.iterrows():
                if row['revenue'] > 0 and pd.notna(row['gross_profit']):
                    assert row['gross_profit'] <= row['revenue']  # Can't have more profit than revenue
    
    def test_no_refunds_handling(self, sample_data):
        """Test handling when no refunds/returns in schema."""
        med, inv, sales = sample_data
        sa = SalesAnalytics(med, inv, sales)
        # Should work fine without refund columns
        assert 'refund' not in sa.enriched_sales.columns or sa.enriched_sales.empty


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
