"""
Phase 4 Tests for Expiry and Wastage Risk Analysis
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from src.expiry import ExpiryAnalyzer
from src.data import PharmaDataGenerator


class TestExpiryAnalyzer:
    """Test expiry and wastage analysis functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        gen = PharmaDataGenerator(seed=999)
        med, inv, sales = gen.generate_all_data(10, 3, 100)
        return med, inv, sales
    
    @pytest.fixture
    def reference_date(self):
        """Fixed reference date for deterministic tests."""
        return datetime(2026, 1, 1)
    
    def test_expiry_analyzer_initialization(self, sample_data, reference_date):
        """Test expiry analyzer initialization."""
        med, inv, sales = sample_data
        ea = ExpiryAnalyzer(med, inv, reference_date=reference_date)
        assert ea is not None
        assert ea.reference_date == pd.Timestamp(reference_date)
    
    def test_days_remaining_calculation(self, sample_data, reference_date):
        """Test calculation of days remaining until expiry."""
        med, inv, sales = sample_data
        ea = ExpiryAnalyzer(med, inv, reference_date=reference_date)
        days_df = ea.get_days_remaining()
        
        assert isinstance(days_df, pd.DataFrame)
        assert 'days_remaining' in days_df.columns
        assert 'expiry_status' in days_df.columns
    
    def test_expired_batches_classification(self, sample_data, reference_date):
        """Test expired batches are classified correctly."""
        med, inv, sales = sample_data
        # Create expired batch
        inv_expired = inv.copy()
        expired_batch = inv_expired.iloc[0].copy()
        expired_batch['batch_id'] = 'EXPIRED_TEST'
        expired_batch['expiry_date'] = (reference_date - timedelta(days=10)).strftime('%Y-%m-%d')
        inv_expired = pd.concat([inv_expired, pd.DataFrame([expired_batch])], ignore_index=True)
        
        ea = ExpiryAnalyzer(med, inv_expired, reference_date=reference_date)
        expired = ea.get_expired_batches()
        
        assert len(expired[expired['batch_id'] == 'EXPIRED_TEST']) == 1
        # Expired batch must never be classified as near expiry
        near = ea.get_near_expiry_batches()
        assert 'EXPIRED_TEST' not in near['batch_id'].values
    
    def test_near_expiry_classification(self, sample_data, reference_date):
        """Test near-expiry classification with configurable threshold."""
        med, inv, sales = sample_data
        inv_test = inv.copy()
        
        # Create batch expiring in 15 days
        near_batch = inv_test.iloc[0].copy()
        near_batch['batch_id'] = 'NEAR_TEST'
        near_batch['expiry_date'] = (reference_date + timedelta(days=15)).strftime('%Y-%m-%d')
        near_batch['quantity'] = 50
        near_batch['purchase_price'] = 10.0
        inv_test = pd.concat([inv_test, pd.DataFrame([near_batch])], ignore_index=True)
        
        ea = ExpiryAnalyzer(med, inv_test, reference_date=reference_date, near_expiry_threshold=30)
        near = ea.get_near_expiry_batches()
        
        assert len(near[near['batch_id'] == 'NEAR_TEST']) == 1
        assert near[near['batch_id'] == 'NEAR_TEST']['days_remaining'].iloc[0] == 15
    
    def test_batch_expiring_today_classification(self, sample_data, reference_date):
        """Test batch expiring today is classified consistently."""
        med, inv, sales = sample_data
        inv_test = inv.copy()
        
        # Create batch expiring today
        today_batch = inv_test.iloc[0].copy()
        today_batch['batch_id'] = 'TODAY_TEST'
        today_batch['expiry_date'] = reference_date.strftime('%Y-%m-%d')
        inv_test = pd.concat([inv_test, pd.DataFrame([today_batch])], ignore_index=True)
        
        ea = ExpiryAnalyzer(med, inv_test, reference_date=reference_date, near_expiry_threshold=30)
        today_row = ea.expiry_df[ea.expiry_df['batch_id'] == 'TODAY_TEST'].iloc[0]
        
        # Batch expiring today must be classified as near expiry (days_remaining == 0 <= threshold)
        assert today_row['days_remaining'] == 0
        assert today_row['expiry_status'] == 'near_expiry'
        # Must not be expired
        expired = ea.get_expired_batches()
        assert 'TODAY_TEST' not in expired['batch_id'].values
    
    def test_threshold_configuration(self, sample_data, reference_date):
        """Test near-expiry threshold is configurable."""
        med, inv, sales = sample_data
        inv_test = inv.copy()
        
        batch = inv_test.iloc[0].copy()
        batch['batch_id'] = 'THRESH_TEST'
        batch['expiry_date'] = (reference_date + timedelta(days=25)).strftime('%Y-%m-%d')
        inv_test = pd.concat([inv_test, pd.DataFrame([batch])], ignore_index=True)
        
        # With threshold 30, should be near expiry
        ea30 = ExpiryAnalyzer(med, inv_test, reference_date=reference_date, near_expiry_threshold=30)
        near30 = ea30.get_near_expiry_batches()
        assert len(near30[near30['batch_id'] == 'THRESH_TEST']) == 1
        
        # With threshold 20, should not be near expiry
        ea20 = ExpiryAnalyzer(med, inv_test, reference_date=reference_date, near_expiry_threshold=20)
        near20 = ea20.get_near_expiry_batches()
        assert len(near20[near20['batch_id'] == 'THRESH_TEST']) == 0
    
    def test_threshold_boundary(self, sample_data, reference_date):
        """Test classification at warning threshold boundary."""
        med, inv, sales = sample_data
        inv_test = inv.copy()
        
        # Batch expiring exactly at threshold
        boundary_batch = inv_test.iloc[0].copy()
        boundary_batch['batch_id'] = 'BOUNDARY_TEST'
        boundary_batch['expiry_date'] = (reference_date + timedelta(days=30)).strftime('%Y-%m-%d')
        inv_test = pd.concat([inv_test, pd.DataFrame([boundary_batch])], ignore_index=True)
        
        ea = ExpiryAnalyzer(med, inv_test, reference_date=reference_date, near_expiry_threshold=30)
        boundary_row = ea.expiry_df[ea.expiry_df['batch_id'] == 'BOUNDARY_TEST'].iloc[0]
        assert boundary_row['days_remaining'] == 30
        assert boundary_row['expiry_status'] == 'near_expiry'
    
    def test_high_value_batches_at_risk(self, sample_data, reference_date):
        """Test identification of high-value batches approaching expiry."""
        med, inv, sales = sample_data
        inv_test = inv.copy()
        
        # Create high value near-expiry batch
        high_batch = inv_test.iloc[0].copy()
        high_batch['batch_id'] = 'HIGHVAL_TEST'
        high_batch['expiry_date'] = (reference_date + timedelta(days=10)).strftime('%Y-%m-%d')
        high_batch['quantity'] = 100
        high_batch['purchase_price'] = 50.0  # $5000 value
        inv_test = pd.concat([inv_test, pd.DataFrame([high_batch])], ignore_index=True)
        
        ea = ExpiryAnalyzer(med, inv_test, reference_date=reference_date)
        high_value = ea.get_high_value_batches_at_risk(min_value=100.0)
        
        assert len(high_value[high_value['batch_id'] == 'HIGHVAL_TEST']) == 1
    
    def test_value_at_risk_calculation(self, sample_data, reference_date):
        """Test purchase value of remaining stock at risk."""
        med, inv, sales = sample_data
        inv_test = inv.copy()
        
        near_batch = inv_test.iloc[0].copy()
        near_batch['batch_id'] = 'RISK_TEST'
        near_batch['expiry_date'] = (reference_date + timedelta(days=5)).strftime('%Y-%m-%d')
        near_batch['quantity'] = 20
        near_batch['purchase_price'] = 15.0  # $300
        inv_test = pd.concat([inv_test, pd.DataFrame([near_batch])], ignore_index=True)
        
        ea = ExpiryAnalyzer(med, inv_test, reference_date=reference_date)
        value_risk = ea.calculate_value_at_risk()
        
        assert value_risk['near_expiry_value_at_risk'] >= 300.0
        assert 'disclaimer' in value_risk
        assert 'estimate' in value_risk['disclaimer'].lower() or 'loss' not in value_risk['disclaimer'].lower()
    
    def test_batch_level_expiry_report(self, sample_data, reference_date):
        """Test batch-level expiry report generation."""
        med, inv, sales = sample_data
        ea = ExpiryAnalyzer(med, inv, reference_date=reference_date)
        report = ea.generate_batch_expiry_report()
        
        assert isinstance(report, pd.DataFrame)
        if not report.empty:
            assert 'batch_id' in report.columns
            assert 'days_remaining' in report.columns
            assert 'expiry_status' in report.columns
    
    def test_wastage_analysis_no_records(self, sample_data, reference_date):
        """Test wastage analysis when no wastage records available."""
        med, inv, sales = sample_data
        ea = ExpiryAnalyzer(med, inv, reference_date=reference_date)
        wastage = ea.analyze_wastage()
        
        assert wastage['wastage_records_found'] == False
        assert 'No wastage records found' in wastage['message']
        assert 'disclaimer' in wastage
    
    def test_wastage_analysis_with_records(self, sample_data, reference_date):
        """Test wastage analysis with recorded wastage transactions."""
        med, inv, sales = sample_data
        wastage_df = pd.DataFrame([
            {'wastage_id': 'W001', 'date': '2025-12-01', 'medicine_id': med.iloc[0]['medicine_id'], 
             'quantity': 10, 'cost': 5.0, 'reason': 'expired'},
            {'wastage_id': 'W002', 'date': '2025-12-15', 'medicine_id': med.iloc[1]['medicine_id'], 
             'quantity': 5, 'cost': 8.0, 'reason': 'damaged'}
        ])
        
        ea = ExpiryAnalyzer(med, inv, wastage_df=wastage_df, reference_date=reference_date)
        wastage = ea.analyze_wastage()
        
        assert wastage['wastage_records_found'] == True
        assert wastage['total_wastage_quantity'] == 15
        assert wastage['total_wastage_value'] == 90.0
    
    def test_fefo_recommendations(self, sample_data, reference_date):
        """Test FEFO stock rotation recommendations."""
        med, inv, sales = sample_data
        inv_test = inv.copy()
        
        # Add batches with different expiry dates
        batch1 = inv_test.iloc[0].copy()
        batch1['batch_id'] = 'FEFO1'
        batch1['expiry_date'] = (reference_date + timedelta(days=60)).strftime('%Y-%m-%d')
        
        batch2 = inv_test.iloc[0].copy()
        batch2['batch_id'] = 'FEFO2'
        batch2['expiry_date'] = (reference_date + timedelta(days=20)).strftime('%Y-%m-%d')
        
        batch3 = inv_test.iloc[0].copy()
        batch3['batch_id'] = 'FEFO3'
        batch3['expiry_date'] = (reference_date + timedelta(days=40)).strftime('%Y-%m-%d')
        
        inv_test = pd.concat([inv_test, pd.DataFrame([batch1, batch2, batch3])], ignore_index=True)
        
        ea = ExpiryAnalyzer(med, inv_test, reference_date=reference_date)
        fefo = ea.get_fefo_recommendations()
        
        assert isinstance(fefo, pd.DataFrame)
        if len(fefo) >= 3:
            # Should be sorted by expiry date (earliest first)
            fefo_list = fefo[fefo['batch_id'].isin(['FEFO1', 'FEFO2', 'FEFO3'])]['batch_id'].tolist()
            assert fefo_list == ['FEFO2', 'FEFO3', 'FEFO1']
    
    def test_zero_stock_handling(self, sample_data, reference_date):
        """Test handling of zero stock batches."""
        med, inv, sales = sample_data
        inv_test = inv.copy()
        
        zero_batch = inv_test.iloc[0].copy()
        zero_batch['batch_id'] = 'ZERO_TEST'
        zero_batch['quantity'] = 0
        zero_batch['expiry_date'] = (reference_date + timedelta(days=10)).strftime('%Y-%m-%d')
        inv_test = pd.concat([inv_test, pd.DataFrame([zero_batch])], ignore_index=True)
        
        ea = ExpiryAnalyzer(med, inv_test, reference_date=reference_date)
        report = ea.generate_batch_expiry_report()
        assert len(report[report['batch_id'] == 'ZERO_TEST']) == 1
    
    def test_multiple_batches_different_status(self, sample_data, reference_date):
        """Test multiple batches with different expiry statuses."""
        med, inv, sales = sample_data
        inv_test = inv.copy()
        
        expired = inv_test.iloc[0].copy()
        expired['batch_id'] = 'MULT_EXP'
        expired['expiry_date'] = (reference_date - timedelta(days=5)).strftime('%Y-%m-%d')
        
        near = inv_test.iloc[0].copy()
        near['batch_id'] = 'MULT_NEAR'
        near['expiry_date'] = (reference_date + timedelta(days=10)).strftime('%Y-%m-%d')
        
        ok = inv_test.iloc[0].copy()
        ok['batch_id'] = 'MULT_OK'
        ok['expiry_date'] = (reference_date + timedelta(days=100)).strftime('%Y-%m-%d')
        
        inv_test = pd.concat([inv_test, pd.DataFrame([expired, near, ok])], ignore_index=True)
        
        ea = ExpiryAnalyzer(med, inv_test, reference_date=reference_date)
        assert len(ea.get_expired_batches()[ea.get_expired_batches()['batch_id'] == 'MULT_EXP']) == 1
        assert len(ea.get_near_expiry_batches()[ea.get_near_expiry_batches()['batch_id'] == 'MULT_NEAR']) == 1
        ok_status = ea.expiry_df[ea.expiry_df['batch_id'] == 'MULT_OK']['expiry_status'].iloc[0]
        assert ok_status == 'ok'
    
    def test_fefo_excludes_expired(self, sample_data, reference_date):
        """Test FEFO recommendations exclude expired batches."""
        med, inv, sales = sample_data
        inv_test = inv.copy()
        
        expired = inv_test.iloc[0].copy()
        expired['batch_id'] = 'FEFO_EXP'
        expired['expiry_date'] = (reference_date - timedelta(days=5)).strftime('%Y-%m-%d')
        
        near = inv_test.iloc[0].copy()
        near['batch_id'] = 'FEFO_NEAR'
        near['expiry_date'] = (reference_date + timedelta(days=10)).strftime('%Y-%m-%d')
        
        inv_test = pd.concat([inv_test, pd.DataFrame([expired, near])], ignore_index=True)
        
        ea = ExpiryAnalyzer(med, inv_test, reference_date=reference_date)
        fefo = ea.get_fefo_recommendations()
        assert 'FEFO_EXP' not in fefo['batch_id'].values
        assert 'FEFO_NEAR' in fefo['batch_id'].values
    
    def test_expiry_report_export(self, sample_data, reference_date):
        """Test expiry report export."""
        med, inv, sales = sample_data
        ea = ExpiryAnalyzer(med, inv, reference_date=reference_date)
        
        report = ea.export_expiry_report()
        assert isinstance(report, str)
        assert 'PHARMATRACK' in report
        assert 'EXPIRY' in report
        assert 'FEFO' in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
