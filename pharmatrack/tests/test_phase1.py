"""
Phase 1 Tests for Data Ingestion, Generation, and Validation
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

from src.data import (
    PharmaDataGenerator, 
    generate_sample_data, 
    DataLoader,
    validate_medicines,
    validate_inventory,
    validate_sales,
    validate_all_data,
)


class TestDataGeneration:
    """Test synthetic data generation."""
    
    def test_generator_reproducible(self):
        """Test that generator produces same data with same seed."""
        gen1 = PharmaDataGenerator(seed=123)
        med1, inv1, sales1 = gen1.generate_all_data(5, 2, 10)
        
        gen2 = PharmaDataGenerator(seed=123)
        med2, inv2, sales2 = gen2.generate_all_data(5, 2, 10)
        
        pd.testing.assert_frame_equal(med1, med2)
        pd.testing.assert_frame_equal(inv1, inv2)
        pd.testing.assert_frame_equal(sales1, sales2)
    
    def test_medicines_structure(self):
        """Test medicines.csv structure."""
        gen = PharmaDataGenerator(seed=42)
        med = gen.generate_medicines(10)
        
        required_cols = ['medicine_id', 'medicine_name', 'category', 'reorder_level']
        assert list(med.columns) == required_cols
        assert len(med) == 10
        assert med['medicine_id'].nunique() == 10
        assert (med['reorder_level'] >= 0).all()
    
    def test_inventory_structure(self):
        """Test inventory.csv structure."""
        gen = PharmaDataGenerator(seed=42)
        med = gen.generate_medicines(5)
        inv = gen.generate_inventory(med, 2)
        
        required_cols = ['batch_id', 'medicine_id', 'quantity', 'purchase_price', 
                        'selling_price', 'expiry_date', 'supplier']
        assert list(inv.columns) == required_cols
        assert inv['batch_id'].nunique() == len(inv)
        assert (inv['quantity'] >= 0).all()
        assert (inv['purchase_price'] > 0).all()
        assert (inv['selling_price'] > 0).all()
        assert set(inv['medicine_id']).issubset(set(med['medicine_id']))
    
    def test_sales_structure(self):
        """Test sales.csv structure."""
        gen = PharmaDataGenerator(seed=42)
        med = gen.generate_medicines(5)
        inv = gen.generate_inventory(med, 2)
        sales = gen.generate_sales(inv, 20)
        
        required_cols = ['sale_id', 'date', 'batch_id', 'quantity_sold', 'unit_price']
        assert list(sales.columns) == required_cols
        assert sales['sale_id'].nunique() == len(sales)
        assert (sales['quantity_sold'] > 0).all()
        assert (sales['unit_price'] > 0).all()
        assert set(sales['batch_id']).issubset(set(inv['batch_id']))


class TestDataValidation:
    """Test data validation functionality."""
    
    def test_valid_medicines(self):
        """Test validation of valid medicines data."""
        med_data = pd.DataFrame({
            'medicine_id': ['MED001', 'MED002'],
            'medicine_name': ['Paracetamol', 'Amoxicillin'],
            'category': ['Analgesic', 'Antibiotic'],
            'reorder_level': [100, 50]
        })
        report = validate_medicines(med_data)
        assert report.valid == True
        assert len(report.errors) == 0
    
    def test_missing_values_medicines(self):
        """Test validation with missing values."""
        med_data = pd.DataFrame({
            'medicine_id': ['MED001', None],
            'medicine_name': ['Paracetamol', 'Amoxicillin'],
            'category': ['Analgesic', 'Antibiotic'],
            'reorder_level': [100, 50]
        })
        report = validate_medicines(med_data)
        assert report.valid == False
        assert len(report.errors) > 0
    
    def test_duplicate_medicine_ids(self):
        """Test validation with duplicate IDs."""
        med_data = pd.DataFrame({
            'medicine_id': ['MED001', 'MED001'],
            'medicine_name': ['Paracetamol', 'Amoxicillin'],
            'category': ['Analgesic', 'Antibiotic'],
            'reorder_level': [100, 50]
        })
        report = validate_medicines(med_data)
        assert report.valid == False
        assert any('Duplicate' in err for err in report.errors)
    
    def test_invalid_dates_inventory(self):
        """Test validation with invalid dates."""
        inv_data = pd.DataFrame({
            'batch_id': ['BATCH001'],
            'medicine_id': ['MED001'],
            'quantity': [100],
            'purchase_price': [10.0],
            'selling_price': [15.0],
            'expiry_date': ['invalid-date'],
            'supplier': ['Test Supplier']
        })
        report = validate_inventory(inv_data)
        assert report.valid == False
    
    def test_negative_quantities(self):
        """Test validation with negative quantities."""
        inv_data = pd.DataFrame({
            'batch_id': ['BATCH001'],
            'medicine_id': ['MED001'],
            'quantity': [-10],
            'purchase_price': [10.0],
            'selling_price': [15.0],
            'expiry_date': ['2025-12-31'],
            'supplier': ['Test Supplier']
        })
        report = validate_inventory(inv_data)
        assert report.valid == False
        assert any('negative' in err.lower() for err in report.errors)
    
    def test_broken_foreign_key_sales(self):
        """Test validation with broken foreign key references."""
        med_data = pd.DataFrame({
            'medicine_id': ['MED001'],
            'medicine_name': ['Paracetamol'],
            'category': ['Analgesic'],
            'reorder_level': [100]
        })
        inv_data = pd.DataFrame({
            'batch_id': ['BATCH001'],
            'medicine_id': ['MED001'],
            'quantity': [100],
            'purchase_price': [10.0],
            'selling_price': [15.0],
            'expiry_date': ['2025-12-31'],
            'supplier': ['Test Supplier']
        })
        sales_data = pd.DataFrame({
            'sale_id': ['SALE001'],
            'date': ['2024-01-01'],
            'batch_id': ['BATCH999'],  # Invalid batch
            'quantity_sold': [5],
            'unit_price': [15.0]
        })
        reports = validate_all_data(med_data, inv_data, sales_data)
        assert reports['sales'].valid == False
        assert any('invalid batch' in err.lower() for err in reports['sales'].errors)
    
    def test_missing_fields(self):
        """Test validation with missing fields."""
        med_data = pd.DataFrame({
            'medicine_id': ['MED001'],
            'medicine_name': ['Paracetamol'],
            # Missing category and reorder_level
        })
        report = validate_medicines(med_data)
        assert report.valid == False
        assert any('Missing required columns' in err for err in report.errors)
    
    def test_zero_or_negative_prices(self):
        """Test validation with invalid prices."""
        inv_data = pd.DataFrame({
            'batch_id': ['BATCH001'],
            'medicine_id': ['MED001'],
            'quantity': [100],
            'purchase_price': [0],  # Invalid
            'selling_price': [15.0],
            'expiry_date': ['2025-12-31'],
            'supplier': ['Test Supplier']
        })
        report = validate_inventory(inv_data)
        assert report.valid == False
    
    def test_duplicate_records_detection(self):
        """Test detection of duplicate records."""
        sales_data = pd.DataFrame({
            'sale_id': ['SALE001', 'SALE001'],  # Duplicate
            'date': ['2024-01-01', '2024-01-02'],
            'batch_id': ['BATCH001', 'BATCH001'],
            'quantity_sold': [5, 3],
            'unit_price': [15.0, 15.0]
        })
        inv_data = pd.DataFrame({
            'batch_id': ['BATCH001'],
            'medicine_id': ['MED001'],
            'quantity': [100],
            'purchase_price': [10.0],
            'selling_price': [15.0],
            'expiry_date': ['2025-12-31'],
            'supplier': ['Test Supplier']
        })
        report = validate_sales(sales_data, inv_data)
        assert report.valid == False
        assert any('Duplicate' in err for err in report.errors)


class TestDataLoader:
    """Test data loading functionality."""
    
    def test_load_and_validate_integration(self):
        """Test full load and validate pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = PharmaDataGenerator(seed=999)
            med, inv, sales = gen.generate_all_data(5, 2, 20)
            gen.save_data(med, inv, sales, Path(tmpdir))
            
            loader = DataLoader(Path(tmpdir))
            med_loaded, inv_loaded, sales_loaded, reports = loader.load_and_validate()
            
            assert len(med_loaded) == 5
            assert len(inv_loaded) > 0
            assert len(sales_loaded) > 0
            assert all(r.valid for r in reports.values())
    
    def test_summary_generation(self):
        """Test summary statistics generation."""
        gen = PharmaDataGenerator(seed=42)
        med, inv, sales = gen.generate_all_data(5, 2, 20)
        loader = DataLoader()
        summary = loader.generate_summary(med, inv, sales)
        
        assert 'medicines' in summary
        assert 'inventory' in summary
        assert 'sales' in summary
        assert summary['medicines']['count'] == 5
        assert summary['inventory']['total_batches'] == len(inv)
        assert summary['sales']['total_transactions'] == len(sales)
    
    def test_numpy_usage_in_validation(self):
        """Test that NumPy is used for numerical calculations."""
        gen = PharmaDataGenerator(seed=42)
        med, inv, sales = gen.generate_all_data(5, 2, 20)
        loader = DataLoader()
        summary = loader.generate_summary(med, inv, sales)
        
        assert isinstance(summary['medicines']['avg_reorder_level'], float)
        assert isinstance(summary['inventory']['total_quantity'], int)
