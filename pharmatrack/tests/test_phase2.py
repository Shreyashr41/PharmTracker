"""
Phase 2 Tests for Inventory Management Module
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from src.inventory import InventoryManager
from src.data import PharmaDataGenerator, DataLoader


class TestInventoryManager:
    """Test inventory management functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        gen = PharmaDataGenerator(seed=123)
        med, inv, sales = gen.generate_all_data(10, 3, 100)
        return med, inv, sales
    
    def test_inventory_initialization(self, sample_data):
        """Test inventory manager initialization."""
        med, inv, sales = sample_data
        im = InventoryManager(med, inv, sales)
        assert im is not None
        assert len(im.medicines_df) == 10
        assert len(im.inventory_df) == len(inv)
    
    def test_medicine_catalogue(self, sample_data):
        """Test medicine catalogue generation."""
        med, inv, sales = sample_data
        im = InventoryManager(med, inv, sales)
        catalogue = im.get_medicine_catalogue()
        
        assert len(catalogue) == 10
        assert 'total_batches' in catalogue.columns
        assert 'usable_quantity' in catalogue.columns
        assert 'total_purchase_value' in catalogue.columns
        assert (catalogue['total_batches'] >= 0).all()
    
    def test_batch_level_tracking_preserved(self, sample_data):
        """Test that individual batch expiry dates and purchase costs are preserved."""
        med, inv, sales = sample_data
        im = InventoryManager(med, inv, sales)
        batch_inv = im.get_batch_inventory()
        
        # Check that original batch details are preserved
        assert 'batch_id' in batch_inv.columns
        assert 'expiry_date' in batch_inv.columns
        assert 'purchase_price' in batch_inv.columns
        assert batch_inv['batch_id'].nunique() == len(inv)  # No double counting
        # Each batch should have its own expiry date
        assert batch_inv.groupby('batch_id')['expiry_date'].nunique().max() == 1
    
    def test_expired_stock_exclusion(self, sample_data):
        """Test that expired stock is excluded from usable calculations."""
        med, inv, sales = sample_data
        im = InventoryManager(med, inv, sales)
        
        # Create a batch with expired date
        expired_batch = inv.iloc[0].copy()
        expired_batch['batch_id'] = 'EXPIRED001'
        expired_batch['expiry_date'] = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
        expired_batch['quantity'] = 100
        
        inv_with_expired = pd.concat([inv, pd.DataFrame([expired_batch])], ignore_index=True)
        im2 = InventoryManager(med, inv_with_expired, sales)
        
        catalogue = im2.get_medicine_catalogue()
        expired_stock = im2.get_expired_stock()
        
        # Expired batch should be in expired_stock
        assert len(expired_stock[expired_stock['batch_id'] == 'EXPIRED001']) == 1
        # Usable quantity should not include expired batch
        med_id = expired_batch['medicine_id']
        med_row = catalogue[catalogue['medicine_id'] == med_id].iloc[0]
        # usable_quantity should be less than total_quantity by expired amount
        assert med_row['usable_quantity'] < med_row['total_quantity']
    
    def test_low_stock_identification(self, sample_data):
        """Test low-stock alert identification."""
        med, inv, sales = sample_data
        im = InventoryManager(med, inv, sales)
        
        # Create low stock scenario
        low_med = med.iloc[0].copy()
        low_med['medicine_id'] = 'LOWMED'
        low_med['medicine_name'] = 'Low Stock Medicine'
        low_med['reorder_level'] = 100
        
        low_batch = inv.iloc[0].copy()
        low_batch['batch_id'] = 'LOWBATCH'
        low_batch['medicine_id'] = 'LOWMED'
        low_batch['quantity'] = 30  # Below reorder level
        low_batch['expiry_date'] = (datetime.now() + timedelta(days=100)).strftime('%Y-%m-%d')
        
        med_low = pd.concat([med, pd.DataFrame([low_med])], ignore_index=True)
        inv_low = pd.concat([inv, pd.DataFrame([low_batch])], ignore_index=True)
        
        im_low = InventoryManager(med_low, inv_low, sales)
        alerts = im_low.get_low_stock_alerts()
        
        assert len(alerts) > 0
        low_alert = alerts[alerts['medicine_id'] == 'LOWMED'].iloc[0]
        assert low_alert['usable_quantity'] == 30
        assert low_alert['reorder_level'] == 100
        assert low_alert['status'] in ['LOW', 'CRITICAL']
    
    def test_inventory_valuation(self, sample_data):
        """Test inventory valuation calculations."""
        med, inv, sales = sample_data
        im = InventoryManager(med, inv, sales)
        valuation = im.calculate_inventory_valuation()
        
        assert 'total_purchase_value_all' in valuation
        assert 'usable_purchase_value' in valuation
        assert 'expired_purchase_value' in valuation
        assert valuation['total_purchase_value_all'] >= valuation['usable_purchase_value']
        assert valuation['total_quantity_all'] >= valuation['usable_quantity']
        assert isinstance(valuation['total_purchase_value_all'], float)
        assert isinstance(valuation['usable_quantity'], int)
    
    def test_filtering_by_medicine_category_supplier(self, sample_data):
        """Test filtering functionality."""
        med, inv, sales = sample_data
        im = InventoryManager(med, inv, sales)
        
        # Filter by medicine
        med_id = med['medicine_id'].iloc[0]
        filtered = im.get_batch_inventory(medicine_id=med_id)
        assert all(filtered['medicine_id'] == med_id)
        
        # Filter by category
        category = med['category'].iloc[0]
        filtered_cat = im.get_batch_inventory(category=category)
        assert all(filtered_cat['category'] == category)
        
        # Filter by supplier
        supplier = inv['supplier'].iloc[0]
        filtered_sup = im.get_batch_inventory(supplier=supplier)
        assert all(filtered_sup['supplier'] == supplier)
    
    def test_multiple_batches_per_medicine_no_double_counting(self, sample_data):
        """Test multiple batches per medicine without double counting."""
        med, inv, sales = sample_data
        im = InventoryManager(med, inv, sales)
        
        # Find medicine with multiple batches
        batch_counts = inv.groupby('medicine_id')['batch_id'].count()
        multi_med = batch_counts[batch_counts >= 2].index[0]
        
        catalogue = im.get_medicine_catalogue()
        med_row = catalogue[catalogue['medicine_id'] == multi_med].iloc[0]
        
        # Check batch count matches
        assert med_row['total_batches'] == batch_counts[multi_med]
        # Check quantities are summed correctly
        total_qty = inv[inv['medicine_id'] == multi_med]['quantity'].sum()
        # Account for expired batches
        expired_qty = inv[(inv['medicine_id'] == multi_med) & (pd.to_datetime(inv['expiry_date']) < pd.Timestamp.now())]['quantity'].sum()
        expected_usable = total_qty - expired_qty
        assert med_row['usable_quantity'] == expected_usable
    
    def test_medicines_with_no_sales_history(self):
        """Test handling of medicines with no historical sales."""
        gen = PharmaDataGenerator(seed=456)
        med, inv, _ = gen.generate_all_data(5, 2, 0)  # No sales
        sales_empty = pd.DataFrame(columns=['sale_id', 'date', 'batch_id', 'quantity_sold', 'unit_price'])
        
        im = InventoryManager(med, inv, sales_empty)
        suggestions = im.get_replenishment_suggestions()
        
        # Should still work, use fallback rule
        assert isinstance(suggestions, pd.DataFrame)
        for _, sug in suggestions.iterrows():
            if sug['method'] == 'fallback_reorder_rule':
                assert sug['confidence'] == 'low'
            assert 'ESTIMATE - NOT AN AUTOMATIC PURCHASE ORDER' in sug['note']
    
    def test_replenishment_with_historical_demand(self, sample_data):
        """Test replenishment suggestions using historical demand."""
        med, inv, sales = sample_data
        im = InventoryManager(med, inv, sales)
        suggestions = im.get_replenishment_suggestions()
        
        assert isinstance(suggestions, pd.DataFrame)
        if not suggestions.empty:
            for _, sug in suggestions.iterrows():
                assert 'suggested_order_qty' in sug
                assert 'method' in sug
                assert sug['note'] == 'ESTIMATE - NOT AN AUTOMATIC PURCHASE ORDER'
                assert sug['method'] in ['historical_demand', 'fallback_reorder_rule']
    
    def test_report_generation(self, sample_data):
        """Test exportable reports."""
        med, inv, sales = sample_data
        im = InventoryManager(med, inv, sales)
        
        # Test inventory report
        report = im.export_inventory_report()
        assert isinstance(report, str)
        assert 'PHARMATRACK' in report
        assert 'INVENTORY SUMMARY REPORT' in report
        
        # Test low stock report
        low_report = im.export_low_stock_report()
        assert isinstance(low_report, str)
        assert 'LOW STOCK REPORT' in low_report
    
    def test_current_stock_calculations(self, sample_data):
        """Test current quantity and purchase-value calculations."""
        med, inv, sales = sample_data
        im = InventoryManager(med, inv, sales)
        
        current_stock = im.get_current_stock(include_expired=False)
        current_stock_all = im.get_current_stock(include_expired=True)
        
        assert len(current_stock) > 0
        assert 'usable_quantity' in current_stock.columns
        assert 'total_purchase_value' in current_stock.columns
        
        # Stock with expired included should be >= without
        for med_id in current_stock['medicine_id']:
            cs = current_stock[current_stock['medicine_id'] == med_id]['usable_quantity'].iloc[0]
            csa = current_stock_all[current_stock_all['medicine_id'] == med_id]['usable_quantity'].iloc[0]
            assert csa >= cs
    
    def test_pandas_numpy_usage(self, sample_data):
        """Test that Pandas and NumPy are used appropriately."""
        med, inv, sales = sample_data
        im = InventoryManager(med, inv, sales)
        
        # Pandas operations
        catalogue = im.get_medicine_catalogue()
        assert isinstance(catalogue, pd.DataFrame)
        
        # NumPy calculations in replenishment
        suggestions = im.get_replenishment_suggestions()
        if not suggestions.empty:
            # Check numeric types
            assert all(isinstance(x, (int, np.integer)) for x in suggestions['suggested_order_qty'] if pd.notna(x))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
